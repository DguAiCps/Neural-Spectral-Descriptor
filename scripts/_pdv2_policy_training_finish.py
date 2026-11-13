#!/usr/bin/env python3
"""완료된 원장부터 순위 진단을 실행하고 최종 보고서를 검증한다."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_policy_training as exp
from _pdv2_no_context_ablation import write_json


def call(script: str, args: list) -> None:
    """후속 진단의 실패를 완료 상태로 숨기지 않는다."""
    subprocess.run([sys.executable, "-u", "scripts/" + script, *args], cwd=REPO, check=True)


def main() -> None:
    """검증 원장 도착을 기다리며 시퀀스별 진단을 순차 수행한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    done = set()
    while len(done) < len(exp.VAL):
        status = exp.base.read(exp.OUT / "status.json")
        if status["status"] == "failed":
            raise RuntimeError(status)
        for s, q in exp.VAL:
            seq = s + "/" + q
            if seq in done:
                continue
            required = [exp.folder(arm, policy) / f"val_{s}_{q}.json"
                        for arm in exp.ARMS for policy in ("all", "top1", "classifier_gate")]
            if all(p.exists() for p in required):
                call("_pdv2_policy_training_diagnose.py", ["--sequence", seq])
                done.add(seq)
                write_json(exp.OUT / "analysis_status.json", {"status": "running", "sequences_done": sorted(done),
                    "sequences_total": len(exp.VAL), "test_used": False})
        if len(done) < len(exp.VAL):
            time.sleep(10)
    call("_pdv2_policy_training_diagnose.py", [])
    while not (exp.OUT / "completion_validation.json").exists():
        status = exp.base.read(exp.OUT / "status.json")
        if status["status"] == "failed":
            raise RuntimeError(status)
        time.sleep(10)
    call("_pdv2_policy_training_report.py", [])
    write_json(exp.OUT / "analysis_status.json", {"status": "complete", "sequences_done": sorted(done),
        "report_validation": exp.base.read(exp.OUT / "report_validation.json")["status"], "test_used": False})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(exp.OUT / "analysis_status.json", {"status": "failed", "error": repr(error)})
        raise
