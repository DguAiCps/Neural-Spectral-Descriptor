#!/usr/bin/env python3
"""검증 전용 유사 간선 진단을 실행하고 최종 원장 검사를 마친다."""
from __future__ import annotations

import concurrent.futures as futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_similarity_causes as exp
from _pdv2_no_context_ablation import write_json, sha256

LOG = REPO / "logs/similarity_causes_20261016"
SCRIPT = "scripts/_pdv2_similarity_causes.py"


def call(args: list, label: str) -> None:
    """실패를 숨기지 않고 작업별 로그를 남긴다."""
    path = LOG / (label + ".log")
    with path.open("a") as log:
        p = subprocess.run([sys.executable, "-u", SCRIPT, *args], cwd=REPO,
                           stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"{label}: {path.read_text()[-4000:]}")


def main() -> None:
    """GPU 두 작업과 CPU 네 작업의 자원 한도에서 실험을 완료한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    call(["prepare"], "prepare")
    call(["ledger"], "ledger")
    extracted = scored = crossed = 0
    with futures.ThreadPoolExecutor(max_workers=2) as gpu, futures.ThreadPoolExecutor(max_workers=4) as cpu:
        pending = {gpu.submit(call, ["keys", "--sequence", s + "/" + q], f"keys_{s}_{q}"): (s, q) for s, q in exp.VAL}
        scores = {cpu.submit(call, ["crossed", "--sequence", s + "/" + q], f"cross_{s}_{q}"): "cross" for s, q in exp.VAL}
        while pending or scores:
            ready, _ = futures.wait(set(pending) | set(scores), return_when=futures.FIRST_COMPLETED)
            for job in ready:
                job.result()
                if job in pending:
                    s, q = pending.pop(job)
                    extracted += 1
                    for arm in exp.ARMS:
                        scores[cpu.submit(call, ["score", "--arm", arm, "--sequence", s + "/" + q], f"score_{arm}_{s}_{q}")] = "score"
                elif scores.pop(job) == "cross":
                    crossed += 1
                else:
                    scored += 1
                write_json(exp.OUT / "status.json", {"status": "running", "keys": extracted, "keys_total": 9,
                    "scores": scored, "scores_total": 54, "crossed": crossed, "crossed_total": 9,
                    "started_unix": started, "elapsed_s": time.time() - started, "test_used": False})
                print("PROGRESS", extracted, scored, crossed, flush=True)
    call(["report"], "report")
    write_json(exp.OUT / "status.json", {"status": "complete", "runtime_s": time.time() - started,
        "finished_unix": time.time(), "test_used": False,
        "verification_status": exp.base.read(exp.OUT / "completion_validation.json")["status"],
        "summary_sha256": sha256(exp.OUT / "summary.json")})
    print("COMPLETE", time.time() - started, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(exp.OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
