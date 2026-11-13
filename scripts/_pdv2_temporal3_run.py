#!/usr/bin/env python3
"""ring_baseline에서 여섯 모델 학습·최종 검증·감사를 완료한다."""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_temporal3 import OUT, ARMS, SEEDS, EPOCHS, base, read, protocol, checkpoint_dir, training_config
from _pdv2_no_context_ablation import sha256, write_json

LOG = REPO / "logs/temporal3_20261017"


def call(command: list, label: str) -> None:
    """각 단계의 로그와 실패를 보존한다."""
    path = LOG / (label + ".log")
    with path.open("a") as handle:
        p = subprocess.run([sys.executable, "-u", *command], cwd=REPO,
                           stdout=handle, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"{label}: {path.read_text()[-5000:]}")


def main() -> None:
    """같은 예산의 학습과 검증 전용 평가를 순서대로 실행한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1", PYTHONFAULTHANDLER="1")
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    call(["scripts/_pdv2_temporal3.py", "prepare"], "prepare")
    import torch
    for seed in SEEDS:
        for arm in ARMS:
            folder = OUT / f"s{seed}/{arm}"
            folder.mkdir(parents=True, exist_ok=True)
            marker = folder / "training_complete.json"
            dest = checkpoint_dir(seed, arm)
            if marker.exists():
                for name, digest in read(marker)["checkpoints"].items():
                    assert sha256(dest / name) == digest
                continue
            assert not dest.exists() or not any(dest.iterdir()), dest
            write_json(OUT / "status.json", {"status": "running", "stage": "training", "seed": seed, "arm": arm})
            tick = time.time()
            call(["scripts/_train_temporal3.py", "--arm", arm, "--seed", str(seed),
                  "--config", str(training_config()), "--checkpoint-dir", str(dest),
                  "--data-root", "/workspace/data", "--encoder-preset", "no_interdiff",
                  "--use-gated-context", "--gate-initial-alpha", "0.0625"], f"train_s{seed}_{arm}")
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
            print("TRAIN_DONE", seed, arm, time.time() - tick, flush=True)
        assert read(OUT / f"s{seed}/temporal/batch_order.json") == read(OUT / f"s{seed}/point/batch_order.json")
    jobs = [(s, a, e, sensor + "/" + seq) for s in SEEDS for a in ARMS
            for e in EPOCHS for sensor, seq in base.VAL]

    def evaluate(action: str, job: tuple) -> None:
        seed, arm, epoch, sequence = job
        call(["scripts/_pdv2_temporal3.py", action, "--seed", str(seed), "--arm", arm,
              "--epoch", str(epoch), "--sequence", sequence],
             f"{action}_s{seed}_{arm}_{epoch}_{sequence.replace('/', '_')}")

    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as gpu, \
            concurrent.futures.ThreadPoolExecutor(max_workers=4) as cpu:
        pending = {gpu.submit(evaluate, "keys", job): job for job in jobs}
        scores = set()
        while pending or scores:
            ready, _ = concurrent.futures.wait(set(pending) | scores,
                                               return_when=concurrent.futures.FIRST_COMPLETED)
            for f in ready:
                f.result()
                if f in pending:
                    scores.add(cpu.submit(evaluate, "score", pending.pop(f)))
                else:
                    scores.remove(f)
                    count += 1
                    write_json(OUT / "status.json", {"status": "running", "stage": "validation", "done": count, "total": len(jobs)})
                    print("SCORED", count, len(jobs), flush=True)
    call(["scripts/_pdv2_temporal3.py", "report"], "report")
    write_json(OUT / "status.json", {"status": "running", "stage": "audit"})
    call(["scripts/_pdv2_temporal3_verify.py"], "verify")
    write_json(OUT / "status.json", {"status": "complete", "stage": "complete", "test_used": False,
               "runtime_s": time.time() - started, "finished_unix": time.time()})
    print("COMPLETE", time.time() - started, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
