#!/usr/bin/env python3
"""공유 특징·기존 신호 검사·B/C 학습·공통 판별 평가를 순서대로 실행한다."""
from __future__ import annotations

import concurrent.futures as futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_edge_reliability as exp
from _pdv2_no_context_ablation import write_json, sha256

LOG = REPO / "logs/edge_reliability_20261016"
SCRIPT = "scripts/_pdv2_edge_reliability.py"


def call(args: list, label: str, script: str = SCRIPT) -> None:
    """작업별 로그를 보존하며 실패를 숨기지 않는다."""
    path = LOG / (label + ".log")
    with path.open("a") as handle:
        p = subprocess.run([sys.executable, "-u", script, *args], cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"{label}: {path.read_text()[-5000:]}")


def stage(name: str, tasks: list, workers: int, started: float) -> None:
    """같은 단계 안의 독립 작업만 동시에 실행한다."""
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(call, args, label) for args, label in tasks]
        for i, job in enumerate(futures.as_completed(jobs), 1):
            job.result()
            write_json(exp.OUT / "status.json", {"status": "running", "stage": name, "done": i,
                "total": len(jobs), "elapsed_s": time.time() - started, "started_unix": started, "test_used": False})
            print(name, i, len(jobs), flush=True)


def main() -> None:
    """요청한 판별 진단만 완료하고 GNN이나 최종 파이프라인은 바꾸지 않는다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    call(["prepare"], "prepare")
    roster = [(role, s + "/" + q) for role, seqs in (("val", exp.VAL), ("train", exp.TRAIN)) for s, q in seqs]
    stage("features", [(["features", "--role", r, "--sequence", seq], "features_" + r + "_" + seq.replace("/", "_")) for r, seq in roster], 2, started)
    stage("stage1", [(["baseline", "--sequence", s + "/" + q], f"baseline_{s}_{q}") for s, q in exp.VAL], 2, started)
    for arm in exp.ARMS[1:]:
        write_json(exp.OUT / "status.json", {"status": "running", "stage": "classifier_training", "arm": arm, "elapsed_s": time.time() - started, "test_used": False})
        call(["train", "--arm", arm], "train_" + arm)
        print("TRAINED", arm, flush=True)
    records = [exp.base.read(exp.OUT / "classifiers" / (a + ".json")) for a in exp.ARMS[1:]]
    for r in records[1:]:
        for k in ("initial14_sha256", "initial16_sha256", "order_sha256", "rows", "updates", "positive_weight", "features_sha256"):
            assert r[k] == records[0][k], k
        assert r["mu"][:14] == records[0]["mu"][:14] and r["sd"][:14] == records[0]["sd"][:14]
    stage("predictions", [(["predict", "--role", r, "--sequence", seq], "predict_" + r + "_" + seq.replace("/", "_")) for r, seq in roster], 2, started)
    stage("diagnostics", [(["analyze_sequence", "--role", r, "--sequence", seq], "diagnose_" + r + "_" + seq.replace("/", "_")) for r, seq in roster], 3, started)
    call(["report"], "report")
    write_json(exp.OUT / "status.json", {"status": "running", "stage": "verification", "test_used": False})
    call([], "audit", "scripts/_pdv2_edge_reliability_audit.py")
    write_json(exp.OUT / "status.json", {"status": "complete", "finished_unix": time.time(),
        "runtime_s": time.time() - started, "verification_status": exp.base.read(exp.OUT / "completion_validation.json")["status"],
        "summary_sha256": sha256(exp.OUT / "summary.json"), "test_used": False})
    print("COMPLETE", time.time() - started, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(exp.OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
