#!/usr/bin/env python3
"""네 조건 학습·검증과 조건부3시드확장·고정시험을 실행한다."""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_matching_eval import (
    OUT, CONFIG, ARMS, EPOCHS, VAL, TEST, checkpoint_dir, read_json, protocol,
)
from _pdv2_no_context_ablation import sha256, write_json

LOG = REPO / "logs/matching_mining_20261014"


def call(command: list, label: str) -> None:
    """단계별 로그를 분리하고 실패를 상위 실행으로 전달한다."""
    with (LOG / f"{label}.log").open("a") as handle:
        result = subprocess.run([sys.executable, "-u", *command], cwd=REPO,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label}: {(LOG / (label + '.log')).read_text()[-5000:]}")


def train(seed: int) -> None:
    """같은 예산으로 네 조건을 순차 학습한다."""
    import torch
    for arm in ARMS:
        output = OUT / f"s{seed}" / arm
        output.mkdir(parents=True, exist_ok=True)
        marker = output / "training_complete.json"
        if marker.exists():
            for name, digest in read_json(marker)["checkpoints"].items():
                assert sha256(checkpoint_dir(seed, arm) / name) == digest
            continue
        dest = checkpoint_dir(seed, arm)
        assert not dest.exists() or not any(dest.iterdir()), dest
        write_json(OUT / "status.json", {"stage": "training", "status": "running", "seed": seed, "arm": arm})
        started = time.time()
        call(["scripts/_train_matching_context.py", "--arm", arm, "--seed", str(seed),
              "--config", str(CONFIG), "--checkpoint-dir", str(dest), "--data-root", "/workspace/data",
              "--encoder-preset", "no_interdiff", "--use-gated-context", "--gate-initial-alpha", "0.0625"],
             f"train_s{seed}_{arm}")
        checkpoints = {}
        for epoch in EPOCHS:
            path = dest / f"checkpoint_epoch_{epoch}.pth"
            blob = torch.load(path, map_location="cpu", weights_only=False)
            assert blob["epoch"] == epoch - 1
            assert all(torch.isfinite(t).all() for t in blob["model_state_dict"].values())
            steps = [int(v["step"].item()) for v in blob["optimizer_state_dict"]["state"].values()]
            assert min(steps) == max(steps) == epoch * 45, (min(steps), max(steps))
            checkpoints[path.name] = sha256(path)
        write_json(marker, {"seed": seed, "arm": arm, "checkpoints": checkpoints,
                   "runtime_s": time.time() - started, "protocol_sha256": sha256(OUT / "protocol.json")})
        print("TRAIN_DONE", seed, arm, flush=True)
    for model in ("gnn", "point"):
        a, b = [OUT / f"s{seed}/{model}_{policy}" for policy in ("original", "matching")]
        assert read_json(a / "initial_model.json") == read_json(b / "initial_model.json")
        assert read_json(a / "training_graph.json") == read_json(b / "training_graph.json")
    print("MATCHED_INITIALIZATION_AND_GRAPH", seed, flush=True)


def evaluate(jobs: list, role: str) -> None:
    """GPU두작업과CPU네작업으로 최종 판정까지 계산한다."""
    def worker(action, job):
        seed, arm, epoch, sequence = job
        call(["scripts/_pdv2_matching_eval.py", action, "--seed", str(seed), "--arm", arm,
              "--epoch", str(epoch), "--role", role, "--sequence", sequence],
             f"{action}_s{seed}_{arm}_{epoch}_{role}_{sequence.replace('/', '_')}")
    done = 0
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as gpu, \
            concurrent.futures.ThreadPoolExecutor(max_workers=4) as cpu:
        keys = {gpu.submit(worker, "keys", job): job for job in jobs}
        scores = set()
        while keys or scores:
            ready, _ = concurrent.futures.wait(set(keys) | scores, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in ready:
                future.result()
                if future in keys:
                    scores.add(cpu.submit(worker, "score", keys.pop(future)))
                else:
                    scores.remove(future)
                    done += 1
                    write_json(OUT / "status.json", {"stage": role, "status": "running", "done": done,
                               "total": len(jobs), "started_unix": started})
                    print("SCORED", role, done, len(jobs), flush=True)


def seed_run(seed: int) -> None:
    """한 시드의 네 조건을 학습하고 검증으로만 선택한다."""
    if not (OUT / f"s{seed}/samples/audit.json").exists():
        call(["scripts/_pdv2_matching_prepare.py", "samples", "--seed", str(seed)], f"samples_s{seed}")
    train(seed)
    jobs = [(seed, arm, epoch, sensor + "/" + seq) for arm in ARMS for epoch in EPOCHS for sensor, seq in VAL]
    evaluate(jobs, "val")
    call(["scripts/_pdv2_matching_eval.py", "select", "--seed", str(seed)], f"select_s{seed}")


def main() -> None:
    """학습 전 고정→1시드 검증→조건부확장 순서를 유지한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1")
    LOG.mkdir(parents=True, exist_ok=True)
    call(["scripts/_pdv2_matching_eval.py", "prepare"], "protocol")
    protocol()
    seed_run(1)
    call(["scripts/_pdv2_matching_eval.py", "pilot"], "pilot")
    if not read_json(OUT / "pilot_summary.json")["expand"]:
        write_json(OUT / "status.json", {"stage": "complete", "status": "complete",
                   "scope": "pilot_validation_only", "test_used": False, "finished_unix": time.time()})
        print("PILOT_COMPLETE_NO_EXPANSION", flush=True)
        return
    print("EXPAND_TO_THREE_SEEDS", flush=True)
    for seed in (2, 3):
        seed_run(seed)
    call(["scripts/_pdv2_matching_eval.py", "freeze"], "freeze_test")
    frozen = read_json(OUT / "freeze_test.json")
    jobs = [(seed, arm, frozen["selected"][str(seed)][arm]["epoch"], sensor + "/" + seq)
            for seed in (1, 2, 3) for arm in ARMS for sensor, seq in TEST]
    evaluate(jobs, "test")
    call(["scripts/_pdv2_matching_eval.py", "report-test"], "report_test")
    write_json(OUT / "status.json", {"stage": "complete", "status": "complete",
               "scope": "three_seeds_test", "test_used": True, "finished_unix": time.time()})
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
