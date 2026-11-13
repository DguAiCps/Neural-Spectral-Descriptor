#!/usr/bin/env python3
"""도커에서 간선 품질·고정 모델 제거 평가·검증을 실행한다."""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_edge_quality as exp
from _pdv2_no_context_ablation import write_json, sha256

LOG = REPO / "logs/edge_quality_20261016"


def call(args: list, label: str) -> None:
    """단계별 로그를 남기고 실패 시 중단한다."""
    path = LOG / (label + ".log")
    with path.open("a") as f:
        result = subprocess.run([sys.executable, "-u", *args], cwd=REPO,
                                stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label}: {path.read_text()[-4500:]}")


def main() -> None:
    """진단과 최종 판정을 완료하고 감사를 수행한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1")
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    call(["scripts/_pdv2_edge_quality.py", "prepare"], "prepare")
    jobs = [(name, role, s + "/" + q) for name in exp.MODELS
            for role, roster in (("val", exp.VAL), ("train", exp.TRAIN)) for s, q in roster]

    def extract(job: tuple) -> None:
        name, role, sequence = job
        call(["scripts/_pdv2_edge_quality.py", "extract", "--model", name, "--role", role,
              "--sequence", sequence], f"extract_{name}_{role}_{sequence.replace('/', '_')}")

    def score(name: str, policy: str, sequence: str) -> None:
        call(["scripts/_pdv2_edge_quality.py", "score", "--model", name, "--policy", policy,
              "--sequence", sequence], f"score_{name}_{policy}_{sequence.replace('/', '_')}")

    extracted = scored = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as gpu, \
            concurrent.futures.ThreadPoolExecutor(max_workers=4) as cpu:
        pending = {gpu.submit(extract, job): job for job in jobs}
        scores = set()
        while pending or scores:
            ready, _ = concurrent.futures.wait(set(pending) | scores,
                                               return_when=concurrent.futures.FIRST_COMPLETED)
            for f in ready:
                f.result()
                if f in pending:
                    name, role, sequence = pending.pop(f)
                    extracted += 1
                    if role == "val":
                        scores.update(cpu.submit(score, name, p, sequence) for p in exp.POLICIES)
                else:
                    scores.remove(f)
                    scored += 1
                write_json(exp.OUT / "status.json", {"status": "running", "stage": "diagnosis_and_validation",
                           "extract_done": extracted, "extract_total": len(jobs), "score_done": scored,
                           "score_total": len(exp.MODELS) * len(exp.POLICIES) * len(exp.VAL), "test_used": False})
                print("PROGRESS", extracted, scored, flush=True)
    call(["scripts/_pdv2_edge_quality.py", "report"], "report")
    write_json(exp.OUT / "status.json", {"status": "running", "stage": "audit", "test_used": False})
    call(["scripts/_pdv2_edge_quality_verify.py"], "verify")
    call(["scripts/_pdv2_edge_quality_plot.py"], "plot")
    v = exp.read(exp.OUT / "completion_validation.json")
    write_json(exp.OUT / "status.json", {"status": "complete", "stage": "complete", "test_used": False,
               "runtime_s": time.time() - started, "finished_unix": time.time(),
               "verification_status": v["status"], "summary_sha256": sha256(exp.OUT / "summary.json")})
    print("COMPLETE", time.time() - started, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(exp.OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
