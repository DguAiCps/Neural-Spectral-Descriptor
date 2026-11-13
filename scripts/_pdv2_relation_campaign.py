#!/usr/bin/env python3
"""값 전달 평가 뒤 검산·묶음 attention·고정 모델 진단을 이어서 실행한다."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_relation_campaign_20261021"
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_no_context_ablation import write_json


def main() -> None:
    """독립 실험별 실패를 전파하고 완료 전 진단을 시작하지 않는다."""
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    try:
        while True:
            path = REPO / "results/pdv2_value_relation_20261021/status.json"
            status = json.loads(path.read_text())
            if status["status"] == "failed":
                raise RuntimeError(status["reason"])
            if status["status"] in ("evaluation_complete", "complete"):
                break
            write_json(OUT / "status.json", {"status": "running", "stage": "waiting_value_evaluation",
                       "elapsed_s": time.time() - started, "test_used": False})
            time.sleep(15)
        steps = ("value_relation_finish", "grouped_relation_run", "grouped_relation_finish",
                 "value_relation_diagnose", "grouped_relation_diagnose")
        for step in steps:
            write_json(OUT / "status.json", {"status": "running", "stage": step,
                       "elapsed_s": time.time() - started, "test_used": False})
            print("START", step, flush=True)
            with (OUT / (step + ".log")).open("a") as handle:
                result = subprocess.run([sys.executable, "-u", str(REPO / "scripts" / ("_pdv2_" + step + ".py"))],
                                        cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(step + "\n" + (OUT / (step + ".log")).read_text()[-4000:])
            print("DONE", step, flush=True)
        write_json(OUT / "status.json", {"status": "complete", "elapsed_s": time.time() - started,
                   "test_used": False})
    except Exception as error:
        write_json(OUT / "status.json", {"status": "failed", "reason": str(error),
                   "elapsed_s": time.time() - started, "test_used": False})
        raise


if __name__ == "__main__":
    main()
