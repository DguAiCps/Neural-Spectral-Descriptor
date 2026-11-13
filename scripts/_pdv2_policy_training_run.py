#!/usr/bin/env python3
"""정책별 학습과 모든 추론 대조를 실행하고 완료 검사를 수행한다."""
from __future__ import annotations

import concurrent.futures as futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_policy_training as exp
from _pdv2_no_context_ablation import write_json

LOG = REPO / "logs/policy_training_20261017"


def call(args: list, label: str, script: str = "_pdv2_policy_training.py") -> None:
    """단계 로그와 실패 원인을 보존한다."""
    path = LOG / (label + ".log")
    with path.open("a") as f:
        r = subprocess.run([sys.executable, "-u", "scripts/" + script, *args], cwd=REPO,
                           stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(label + ": " + path.read_text()[-6000:])


def main() -> None:
    """GPU 한 작업·CPU 세 작업으로 동결 조건을 완료한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    LOG.mkdir(parents=True, exist_ok=True)
    exp.OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    write_json(exp.OUT / "status.json", {"status": "running", "stage": "prepare", "test_used": False})
    call(["prepare"], "prepare")
    for arm in exp.ARMS:
        write_json(exp.OUT / "status.json", {"status": "running", "stage": "training", "arm": arm,
                                           "started_unix": start, "test_used": False})
        call(["train", "--arm", arm], "train_" + arm)
    keys_done = scores_done = 0
    with futures.ThreadPoolExecutor(max_workers=1) as gpu, futures.ThreadPoolExecutor(max_workers=3) as cpu:
        pending = {gpu.submit(call, ["keys", "--arm", arm, "--sequence", s + "/" + q],
                              f"keys_{arm}_{s}_{q}"): (arm, s, q)
                   for arm in exp.ARMS for s, q in exp.VAL}
        scoring = set()
        while pending or scoring:
            ready, _ = futures.wait(set(pending) | scoring, return_when=futures.FIRST_COMPLETED)
            for job in ready:
                job.result()
                if job in pending:
                    arm, s, q = pending.pop(job)
                    keys_done += 1
                    for policy in exp.POLICIES:
                        scoring.add(cpu.submit(call, ["score", "--arm", arm, "--policy", policy,
                            "--sequence", s + "/" + q], f"score_{arm}_{policy}_{s}_{q}"))
                else:
                    scoring.remove(job)
                    scores_done += 1
                write_json(exp.OUT / "status.json", {"status": "running", "stage": "validation",
                    "keys_done": keys_done, "keys_total": len(exp.ARMS) * len(exp.VAL),
                    "scores_done": scores_done, "scores_total": len(exp.ARMS) * len(exp.VAL) * len(exp.POLICIES),
                    "elapsed_s": time.time() - start, "test_used": False})
                print("PROGRESS", keys_done, scores_done, flush=True)
    write_json(exp.OUT / "status.json", {"status": "running", "stage": "audit", "test_used": False})
    call([], "audit", "_pdv2_policy_training_audit.py")
    write_json(exp.OUT / "status.json", {"status": "complete", "runtime_s": time.time() - start,
        "verification_status": "passed_with_inherited_gpu_reproducibility_limit", "test_used": False})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(exp.OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
