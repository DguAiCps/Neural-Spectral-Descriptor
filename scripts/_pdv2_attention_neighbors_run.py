#!/usr/bin/env python3
"""ring_baseline 안에서 네 조건을 학습하고 검증9의 최종 판정을 비교한다."""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_attention_neighbors import (
    OUT, ARMS, EPOCHS, base, read, protocol, checkpoint_dir, training_config,
)
from _pdv2_no_context_ablation import sha256, write_json

LOG = REPO / "logs/attention_neighbors_20261015"


def call(command: list, label: str) -> None:
    """실패와 단계별 로그를 보존한다."""
    path = LOG / (label + ".log")
    with path.open("a") as handle:
        result = subprocess.run([sys.executable, "-u", *command], cwd=REPO,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label}: {path.read_text()[-5000:]}")


def main() -> None:
    """고정·학습·최종 평가·수치 감사 순서로 끝까지 실행한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1")
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    call(["scripts/_pdv2_attention_neighbors.py", "prepare"], "prepare")
    protocol()
    import torch
    for arm in ARMS:
        folder = OUT / "s1" / arm
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / "training_complete.json"
        if marker.exists():
            for name, digest in read(marker)["checkpoints"].items():
                assert sha256(checkpoint_dir(1, arm) / name) == digest
            continue
        dest = checkpoint_dir(1, arm)
        assert not dest.exists() or not any(dest.iterdir()), dest
        write_json(OUT / "status.json", {"stage": "training", "status": "running", "arm": arm})
        tick = time.time()
        call(["scripts/_train_attention_neighbors.py", "--arm", arm, "--seed", "1",
              "--config", str(training_config(arm)), "--checkpoint-dir", str(dest),
              "--data-root", "/workspace/data", "--encoder-preset", "no_interdiff",
              "--use-gated-context", "--gate-initial-alpha", "0.0625"], "train_" + arm)
        hashes = {}
        for epoch in EPOCHS:
            path = dest / f"checkpoint_epoch_{epoch}.pth"
            blob = torch.load(path, map_location="cpu", weights_only=False)
            assert blob["epoch"] == epoch - 1
            assert all(torch.isfinite(v).all() for v in blob["model_state_dict"].values())
            steps = [int(v["step"].item()) for v in blob["optimizer_state_dict"]["state"].values()]
            assert min(steps) == max(steps) == epoch * 45
            hashes[path.name] = sha256(path)
        write_json(marker, {"checkpoints": hashes, "runtime_s": time.time() - tick,
                   "protocol_sha256": sha256(OUT / "protocol.json")})
        print("TRAIN_DONE", arm, time.time() - tick, flush=True)
    reference = read(OUT / "s1/diff_full/initial_model.json")
    assert all(read(OUT / "s1" / a / "initial_model.json") == reference for a in ARMS)
    jobs = [(a, e, s + "/" + q) for a in ARMS for e in EPOCHS for s, q in base.VAL]

    def evaluate(action: str, job: tuple) -> None:
        arm, epoch, sequence = job
        call(["scripts/_pdv2_attention_neighbors.py", action, "--arm", arm,
              "--epoch", str(epoch), "--sequence", sequence],
             f"{action}_{arm}_{epoch}_{sequence.replace('/', '_')}")

    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as gpu, \
            concurrent.futures.ThreadPoolExecutor(max_workers=4) as cpu:
        pending = {gpu.submit(evaluate, "keys", job): job for job in jobs}
        scores = set()
        while pending or scores:
            ready, _ = concurrent.futures.wait(set(pending) | scores,
                                               return_when=concurrent.futures.FIRST_COMPLETED)
            for future in ready:
                future.result()
                if future in pending:
                    scores.add(cpu.submit(evaluate, "score", pending.pop(future)))
                else:
                    scores.remove(future)
                    count += 1
                    write_json(OUT / "status.json", {"stage": "validation", "status": "running",
                               "done": count, "total": len(jobs)})
                    print("SCORED", count, len(jobs), flush=True)
    call(["scripts/_pdv2_attention_neighbors.py", "report"], "report")
    write_json(OUT / "status.json", {"stage": "audit", "status": "running"})
    call(["scripts/_pdv2_attention_neighbors_verify.py"], "verify")
    write_json(OUT / "status.json", {"stage": "complete", "status": "complete", "test_used": False,
               "runtime_s": time.time() - started, "finished_unix": time.time()})
    print("COMPLETE", time.time() - started, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
