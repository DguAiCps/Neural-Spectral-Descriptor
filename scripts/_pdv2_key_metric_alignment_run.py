#!/usr/bin/env python3
"""거리 정합 진단·3시드 학습·최종 검증을 순서대로 실행한다."""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_key_metric_alignment_20261026"
LOG = REPO / "logs/key_metric_alignment_20261026"


def call(label: str, arguments: list) -> None:
    """작업 로그를 보존하며 실패하면 즉시 중단한다."""
    path = LOG / (label + ".log")
    assert not path.exists(), path
    with path.open("w") as handle:
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                str(REPO / "scripts/_pdv2_key_metric_alignment.py"),
                *arguments,
            ],
            cwd=REPO,
            env=dict(
                os.environ,
                PYTHONDONTWRITEBYTECODE="1",
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            ),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(label + "\n" + path.read_text()[-5000:])


def main() -> None:
    """먼저 진단하고 같은 예산의 세 학습과 검증을 완료한다."""
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    call("prepare", ["prepare"])
    call("diagnose", ["diagnose"])
    print("DIAGNOSIS_DONE", flush=True)
    for seed in (1, 2, 3):
        call(f"train_s{seed}", ["train", "--seed", str(seed)])
        print("TRAIN_DONE", seed, round(time.time() - started, 1), flush=True)
    spec = json.loads((OUT / "protocol.json").read_text())
    jobs = [(s, e, q) for s in (1, 2, 3) for e in (10, 20) for q in spec["val"]]

    def evaluate(job):
        seed, epoch, sequence = job
        call(
            f"val_s{seed}_e{epoch}_{sequence.replace('/', '_')}",
            ["val", "--seed", str(seed), "--epoch", str(epoch), "--sequence", sequence],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate, j) for j in jobs]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(
                "VAL_DONE",
                count,
                len(jobs),
                round(time.time() - started, 1),
                flush=True,
            )
    call("finish", ["finish"])
    print("ALL_COMPLETE", round(time.time() - started, 1), flush=True)


if __name__ == "__main__":
    main()
