#!/usr/bin/env python3
"""학습 내부 하한 고정 후 연결 거절의 최종 평가와 검사를 실행한다."""
from __future__ import annotations

import concurrent.futures as futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_context_abstention as exp
from _pdv2_no_context_ablation import write_json

LOG = REPO / "logs/context_abstention_20261017"


def call(args: list, label: str, script: str = "_pdv2_context_abstention.py") -> None:
    """작업 로그와 실패 원인을 보존한다."""
    path = LOG / (label + ".log")
    with path.open("a") as f:
        result = subprocess.run([sys.executable, "-u", "scripts/" + script, *args], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(label + ": " + path.read_text()[-5000:])


def main() -> None:
    """GPU 하나·CPU 세 작업으로 전 조건을 실행하고 완료까지 검사한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    LOG.mkdir(parents=True, exist_ok=True)
    exp.OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    write_json(exp.OUT / "status.json", {"status": "running", "stage": "prepare", "test_used": False})
    call(["prepare"], "prepare")
    for model in exp.MODELS:
        write_json(exp.OUT / "status.json", {"status": "running", "stage": "calibration", "model": model, "test_used": False})
        call(["calibrate", "--model", model], "calibrate_" + model)
    keys_done = scores_done = 0
    with futures.ThreadPoolExecutor(max_workers=1) as gpu, futures.ThreadPoolExecutor(max_workers=3) as cpu:
        pending = {gpu.submit(call, ["keys", "--model", model, "--sequence", s + "/" + q], f"keys_{model}_{s}_{q}"): (model, s, q)
                   for model in exp.MODELS for s, q in exp.VAL}
        scoring = set()
        while pending or scoring:
            ready, _ = futures.wait(set(pending) | scoring, return_when=futures.FIRST_COMPLETED)
            for job in ready:
                job.result()
                if job in pending:
                    model, s, q = pending.pop(job)
                    keys_done += 1
                    for policy in exp.POLICIES:
                        scoring.add(cpu.submit(call, ["score", "--model", model, "--policy", policy, "--sequence", s + "/" + q], f"score_{model}_{policy}_{s}_{q}"))
                else:
                    scoring.remove(job)
                    scores_done += 1
                write_json(exp.OUT / "status.json", {"status": "running", "stage": "validation",
                    "keys_done": keys_done, "keys_total": 18, "scores_done": scores_done,
                    "scores_total": 108, "elapsed_s": time.time() - start, "test_used": False})
                print("PROGRESS", keys_done, scores_done, flush=True)
    write_json(exp.OUT / "status.json", {"status": "running", "stage": "audit", "test_used": False})
    call([], "audit", "_pdv2_context_abstention_audit.py")
    write_json(exp.OUT / "status.json", {"status": "complete", "runtime_s": time.time() - start,
        "verification_status": "passed_with_inherited_gpu_reproducibility_limit", "test_used": False})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(exp.OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
