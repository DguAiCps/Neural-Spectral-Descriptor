#!/usr/bin/env python3
"""시간 전용 비교를 검증 시드 1에서 시작하고 고정 기준에 따라 확장한다."""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_temporal_focus as exp
from _pdv2_no_context_ablation import sha256, write_json

LOG = REPO / "logs/temporal_focus_20261018"


def call(args: list, name: str) -> None:
    """각 실행의 로그와 실패를 보존한다."""
    path = LOG / (name + ".log")
    with path.open("a") as handle:
        p = subprocess.run([sys.executable, "-u", *args], cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"{name}: {path.read_text()[-4500:]}")


def run_seed(seed: int, arms: tuple) -> dict:
    """지정된 조건만 같은 예산으로 학습하고 최종 검증 지표를 선택한다."""
    import torch
    for arm in arms:
        if arm in exp.REUSED:
            continue
        dest = exp.checkpoint_dir(seed, arm)
        marker = exp.arm_dir(seed, arm) / "training_complete.json"
        if marker.exists():
            for name, h in exp.read(marker)["checkpoints"].items():
                assert sha256(dest / name) == h
            continue
        assert not dest.exists() or not any(dest.iterdir()), dest
        write_json(exp.OUT / "status.json", {"status": "running", "stage": "training", "seed": seed, "arm": arm})
        tick = time.monotonic()
        call(["scripts/_train_temporal_focus.py", "--arm", arm, "--seed", str(seed),
              "--config", str(exp.training_config(arm)), "--checkpoint-dir", str(dest),
              "--data-root", "/workspace/data", "--encoder-preset", "no_interdiff", "--use-gated-context",
              "--gate-initial-alpha", "0.0625"], f"train_s{seed}_{arm}")
        hashes = {}
        for epoch in exp.EPOCHS:
            path = dest / f"checkpoint_epoch_{epoch}.pth"
            blob = torch.load(path, map_location="cpu", weights_only=False)
            assert blob["epoch"] == epoch - 1
            assert all(torch.isfinite(v).all() for v in blob["model_state_dict"].values())
            steps = [int(v["step"].item()) for v in blob["optimizer_state_dict"]["state"].values()]
            assert min(steps) == max(steps) == epoch * 45
            hashes[path.name] = sha256(path)
        write_json(marker, {"checkpoints": hashes, "runtime_s": time.monotonic() - tick,
                           "protocol_sha256": sha256(exp.OUT / "protocol.json")})
        print("TRAIN_DONE", seed, arm, time.monotonic() - tick, flush=True)
    jobs = [(a, e, s + "/" + q) for a in arms if a not in exp.REUSED for e in exp.EPOCHS for s, q in exp.base.VAL]

    def evaluate(action: str, job: tuple) -> None:
        arm, epoch, sequence = job
        call(["scripts/_pdv2_temporal_focus.py", action, "--seed", str(seed), "--arm", arm,
              "--epoch", str(epoch), "--sequence", sequence], f"{action}_s{seed}_{arm}_{epoch}_{sequence.replace('/', '_')}")

    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as gpu, concurrent.futures.ThreadPoolExecutor(max_workers=4) as cpu:
        pending = {gpu.submit(evaluate, "keys", job): job for job in jobs}
        scores = set()
        while pending or scores:
            ready, _ = concurrent.futures.wait(set(pending) | scores, return_when=concurrent.futures.FIRST_COMPLETED)
            for f in ready:
                f.result()
                if f in pending:
                    scores.add(cpu.submit(evaluate, "score", pending.pop(f)))
                else:
                    scores.remove(f)
                    count += 1
                    write_json(exp.OUT / "status.json", {"status": "running", "stage": "validation", "seed": seed, "done": count, "total": len(jobs)})
                    print("SCORED", seed, count, len(jobs), flush=True)
    return exp.select(seed, arms)


def main() -> None:
    """선택 기준을 먼저 고정하고 예비·조건부 확증·검사를 완료한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONFAULTHANDLER="1")
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    call(["scripts/_pdv2_temporal_focus.py", "prepare"], "prepare")
    chosen = run_seed(1, exp.ARMS)
    effects = exp.read(exp.OUT / "s1/effects.json")
    eligible = [a for a in exp.ARMS if a in effects and all(effects[a][op]["AP"] > 0 for op in exp.base.OPS)]
    winner = max(eligible, key=lambda a: chosen[a]["joint_AP"]) if eligible else None
    expansion = {"eligible": eligible, "winner": winner, "rule": exp.protocol()["expansion"], "test_used": False}
    write_json(exp.OUT / "expansion.json", expansion)
    per_seed = {"1": chosen}
    if winner:
        point = "point_" + winner.split("_")[1]
        for seed in (2, 3):
            per_seed[str(seed)] = run_seed(seed, (winner, point))
    write_json(exp.OUT / "summary.json", {"per_seed": per_seed, "expansion": expansion, "test_used": False,
                                         "protocol_sha256": sha256(exp.OUT / "protocol.json")})
    call(["scripts/_pdv2_temporal_focus_verify.py"], "verify")
    write_json(exp.OUT / "status.json", {"status": "complete", "runtime_s": time.time() - started, "test_used": False})
    print("COMPLETE", expansion, time.time() - started, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(exp.OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
