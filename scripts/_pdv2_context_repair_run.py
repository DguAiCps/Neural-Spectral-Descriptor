#!/usr/bin/env python3
"""고정 모델의 다방면 문맥 비교를 실행하고 작업 상태를 남긴다."""
from __future__ import annotations

import concurrent.futures as futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_context_repair as exp
from _pdv2_no_context_ablation import write_json

LOG = REPO / "logs/context_repair_20261016"


def call(args: list, label: str) -> None:
    """작업별 로그를 보존한다."""
    with (LOG / (label + ".log")).open("a") as f:
        p = subprocess.run([sys.executable, "-u", "scripts/_pdv2_context_repair.py", *args],
                           cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(label + ": " + (LOG / (label + ".log")).read_text()[-5000:])


def main() -> None:
    """GPU 한 작업과 CPU 세 작업으로 검증을 완료한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    LOG.mkdir(parents=True, exist_ok=True)
    exp.OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    call(["prepare"], "prepare")
    done_keys = done_scores = 0
    with futures.ThreadPoolExecutor(max_workers=1) as gpu, futures.ThreadPoolExecutor(max_workers=3) as cpu:
        pending = {gpu.submit(call, ["keys", "--sequence", s + "/" + q], "keys_" + s + "_" + q): (s, q) for s, q in exp.VAL}
        scores = {cpu.submit(call, ["score", "--arm", "fusion", "--sequence", s + "/" + q], "fusion_" + s + "_" + q) for s, q in exp.VAL}
        while pending or scores:
            ready, _ = futures.wait(set(pending) | scores, return_when=futures.FIRST_COMPLETED)
            for job in ready:
                job.result()
                if job in pending:
                    s, q = pending.pop(job)
                    done_keys += 1
                    for arm in exp.GRAPH_ARMS:
                        scores.add(cpu.submit(call, ["score", "--arm", arm, "--sequence", s + "/" + q], arm + "_" + s + "_" + q))
                else:
                    scores.remove(job)
                    done_scores += 1
                write_json(exp.OUT / "status.json", {"status": "running", "keys_done": done_keys, "keys_total": 9,
                    "score_jobs_done": done_scores, "score_jobs_total": 45, "started_unix": start,
                    "elapsed_s": time.time() - start, "test_used": False})
                print("PROGRESS", done_keys, done_scores, flush=True)
    call(["report"], "report")
    write_json(exp.OUT / "status.json", {"status": "evaluated", "runtime_s": time.time() - start,
        "remaining": "메시지·간선 선택 및 지연 의존성 보완 검사", "test_used": False})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(exp.OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
