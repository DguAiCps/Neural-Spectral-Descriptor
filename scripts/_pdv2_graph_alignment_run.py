#!/usr/bin/env python3
"""도커 안에서 정합 학습·최종 검증·감사를 끝까지 실행한다."""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_graph_alignment as exp
from _pdv2_no_context_ablation import sha256, write_json

LOG = REPO / "logs/graph_alignment_20261015"


def call(command: list, label: str) -> None:
    """단계별 로그와 실패 원인을 남긴다."""
    path = LOG / (label + ".log")
    with path.open("a") as f:
        result = subprocess.run([sys.executable, "-u", *command], cwd=REPO,
                                stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label}: {path.read_text()[-5000:]}")


def main() -> None:
    """시험 자료를 쓰지 않는 전체 실험을 실행한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1")
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    call(["scripts/_pdv2_graph_alignment.py", "prepare"], "prepare")
    exp.protocol(True)
    import torch
    root = exp.arm_root(exp.NEW)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "training_complete.json"
    if not marker.exists():
        dest = exp.checkpoint_dir(1, exp.NEW)
        assert not dest.exists() or not any(dest.iterdir()), dest
        write_json(exp.OUT / "status.json", {"status": "running", "stage": "training",
                   "started_unix": started, "optimizer_updates_target": 900,
                   "refresh_every_update": True, "test_used": False})
        tick = time.time()
        call(["scripts/_train_graph_alignment.py", "--arm", exp.NEW, "--seed", "1",
              "--config", str(exp.training_config(exp.NEW)), "--checkpoint-dir", str(dest),
              "--data-root", "/workspace/data", "--encoder-preset", "no_interdiff",
              "--use-gated-context", "--gate-initial-alpha", "0.0625"], "train_diff_online")
        hashes = {}
        for epoch in exp.EPOCHS:
            path = dest / f"checkpoint_epoch_{epoch}.pth"
            blob = torch.load(path, map_location="cpu", weights_only=False)
            assert blob["epoch"] == epoch - 1
            assert all(torch.isfinite(v).all() for v in blob["model_state_dict"].values())
            steps = [int(v["step"].item()) for v in blob["optimizer_state_dict"]["state"].values()]
            assert min(steps) == max(steps) == epoch * 45
            hashes[path.name] = sha256(path)
        write_json(marker, {"checkpoints": hashes, "runtime_s": time.time() - tick,
                   "protocol_sha256": sha256(exp.OUT / "protocol.json")})
        print("TRAIN_DONE", time.time() - tick, flush=True)
    else:
        for name, digest in exp.read(marker)["checkpoints"].items():
            assert sha256(exp.checkpoint_dir(1, exp.NEW) / name) == digest
    assert exp.read(root / "initial_model.json") == exp.read(exp.arm_root("diff_full") / "initial_model.json")
    jobs = [(epoch, sensor + "/" + seq) for epoch in exp.EPOCHS for sensor, seq in exp.base.VAL]

    def evaluate(action: str, job: tuple) -> None:
        epoch, sequence = job
        call(["scripts/_pdv2_graph_alignment.py", action, "--epoch", str(epoch), "--sequence", sequence],
             f"{action}_{epoch}_{sequence.replace('/', '_')}")

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
                    write_json(exp.OUT / "status.json", {"status": "running", "stage": "validation",
                               "done": count, "total": len(jobs), "test_used": False})
                    print("SCORED", count, len(jobs), flush=True)
    call(["scripts/_pdv2_graph_alignment.py", "report"], "report")
    write_json(exp.OUT / "status.json", {"status": "running", "stage": "audit", "test_used": False})
    call(["scripts/_pdv2_graph_alignment_verify.py"], "verify")
    validation = exp.read(exp.OUT / "completion_validation.json")
    write_json(exp.OUT / "status.json", {"status": "complete", "stage": "complete", "test_used": False,
               "runtime_s": time.time() - started, "finished_unix": time.time(),
               "verification_status": validation["status"], "summary_sha256": sha256(exp.OUT / "summary.json")})
    print("COMPLETE", time.time() - started, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(exp.OUT / "status.json", {"status": "failed", "error": repr(error), "test_used": False})
        raise
