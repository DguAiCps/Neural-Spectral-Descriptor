#!/usr/bin/env python3
"""방향 균형·간선 제거 학습과 최종 평가를 큐로 실행한다."""
from __future__ import annotations

import concurrent.futures as futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_context_repair_train as exp
from _pdv2_no_context_ablation import write_json

LOG = REPO / "logs/context_repair_training_20261016"


def call(args: list, label: str) -> None:
    """로그와 실패 상태를 보존한다."""
    path = LOG / (label + ".log")
    with path.open("a") as f:
        result = subprocess.run([sys.executable, "-u", "scripts/_pdv2_context_repair_train.py", *args], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(label + ": " + path.read_text()[-6000:])


def main() -> None:
    """학습은 하나씩 수행하고 평가 집계는 CPU 세 작업으로 겹친다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    LOG.mkdir(parents=True, exist_ok=True)
    exp.OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    write_json(exp.OUT / "status.json", {"status": "running", "stage": "prepare", "test_used": False})
    call(["prepare"], "prepare")
    for arm in exp.ARMS:
        write_json(exp.OUT / "status.json", {"status": "running", "stage": "train", "arm": arm, "started_unix": start, "test_used": False})
        call(["train", "--arm", arm], "train_" + arm)
    extracted = scored = 0
    with futures.ThreadPoolExecutor(max_workers=1) as gpu, futures.ThreadPoolExecutor(max_workers=3) as cpu:
        pending = {gpu.submit(call, ["keys", "--arm", arm, "--sequence", s + "/" + q], f"keys_{arm}_{s}_{q}"): (arm, s, q)
                   for arm in exp.ARMS for s, q in exp.VAL}
        scores = set()
        while pending or scores:
            ready, _ = futures.wait(set(pending) | scores, return_when=futures.FIRST_COMPLETED)
            for job in ready:
                job.result()
                if job in pending:
                    arm, s, q = pending.pop(job)
                    extracted += 1
                    for mode in ("all", "temporal"):
                        label = arm + "_" + mode
                        scores.add(cpu.submit(call, ["score", "--arm", label, "--sequence", s + "/" + q], f"score_{label}_{s}_{q}"))
                else:
                    scores.remove(job)
                    scored += 1
                write_json(exp.OUT / "status.json", {"status": "running", "stage": "validation", "keys_done": extracted,
                    "keys_total": 36, "scores_done": scored, "scores_total": 72, "elapsed_s": time.time() - start, "test_used": False})
                print("PROGRESS", extracted, scored, flush=True)
    call(["report"], "report")
    write_json(exp.OUT / "status.json", {"status": "evaluated", "runtime_s": time.time() - start,
        "remaining": "추가 학습 모델의 지연 의존성 및 비교 보고서 검사", "test_used": False})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(exp.OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
