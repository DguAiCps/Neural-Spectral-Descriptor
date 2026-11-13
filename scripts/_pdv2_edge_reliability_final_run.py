#!/usr/bin/env python3
"""도커에서 최종 간선 비교·원장 재계산·지연 검사를 순서대로 실행한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_pdv2_edge_reliability_final_run.py
"""
from __future__ import annotations

import concurrent.futures as futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_edge_reliability_final as exp
from _pdv2_no_context_ablation import write_json, sha256

LOG = REPO / "logs/edge_reliability_final_20261016"
SCRIPT = "scripts/_pdv2_edge_reliability_final.py"


def call(args: list, label: str, script: str = SCRIPT) -> None:
    """작업별 로그를 보존하고 실패를 전달한다."""
    path = LOG / (label + ".log")
    with path.open("a") as log:
        p = subprocess.run([sys.executable, "-u", script, *args], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"{label}: {path.read_text()[-6000:]}")


def main() -> None:
    """GPU2·CPU3 작업 한도에서 모든 최종 조건을 완료한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    LOG.mkdir(parents=True, exist_ok=True)
    exp.OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    call(["prepare"], "prepare")
    extracted = scored = 0
    with futures.ThreadPoolExecutor(max_workers=2) as gpu, futures.ThreadPoolExecutor(max_workers=3) as cpu:
        pending = {gpu.submit(call, ["keys", "--sequence", s + "/" + q], f"keys_{s}_{q}"): (s, q) for s, q in exp.VAL}
        scores = set()
        while pending or scores:
            ready, _ = futures.wait(set(pending) | scores, return_when=futures.FIRST_COMPLETED)
            for job in ready:
                job.result()
                if job in pending:
                    s, q = pending.pop(job)
                    extracted += 1
                    for arm in exp.ARMS:
                        scores.add(cpu.submit(call, ["score", "--arm", arm, "--sequence", s + "/" + q], f"score_{arm}_{s}_{q}"))
                else:
                    scores.remove(job)
                    scored += 1
                write_json(exp.OUT / "status.json", {"status": "running", "keys": extracted, "keys_total": 9,
                    "scores": scored, "scores_total": 45, "elapsed_s": time.time() - started,
                    "started_unix": started, "test_used": False})
                print("PROGRESS", extracted, scored, flush=True)
    call(["report"], "report")
    write_json(exp.OUT / "status.json", {"status": "running", "stage": "verification", "test_used": False})
    call([], "verify", "scripts/_pdv2_edge_reliability_final_verify.py")
    write_json(exp.OUT / "status.json", {"status": "complete", "runtime_s": time.time() - started,
        "finished_unix": time.time(), "verification_status": exp.base.read(exp.OUT / "completion_validation.json")["status"],
        "test_used": False, "summary_sha256": sha256(exp.OUT / "summary.json")})
    print("COMPLETE", time.time() - started, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(exp.OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
