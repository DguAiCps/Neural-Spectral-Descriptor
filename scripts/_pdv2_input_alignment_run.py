#!/usr/bin/env python3
"""GPU 학습·추론을 차례로 실행하고 현재 단계를 저장한다."""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_no_context_ablation import write_json

OUT = REPO / "results/pdv2_input_alignment_20261019"


def main() -> None:
    """고정된 네 학습 조건과 자세 진단을 모두 완료한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    jobs = [("prepare", None), ("val", "oracle_gt")]
    jobs += [("train", arm) for arm in ("gg", "go", "og", "oo")]
    jobs += [("val", arm) for arm in ("gg", "go", "og", "oo")]
    jobs += [("report", None)]
    started = time.time()
    for action, arm in jobs:
        tag = action + ("_" + arm if arm else "")
        command = [sys.executable, str(REPO / "scripts/_pdv2_input_alignment.py"), action]
        if arm: command += ["--arm", arm]
        write_json(OUT / "status.json", {"status": "running", "stage": tag, "elapsed_s": time.time() - started})
        print("START", tag, flush=True)
        with (OUT / (tag + ".log")).open("w") as stream:
            result = subprocess.run(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(OUT / "status.json", {"status": "failed", "stage": tag, "returncode": result.returncode})
            raise RuntimeError(tag + " 단계 실패. 로그를 확인한다.")
        print("DONE", tag, round(time.time() - started, 1), flush=True)
    write_json(OUT / "status.json", {"status": "evaluation_complete", "elapsed_s": time.time() - started, "test_used": False})


if __name__ == "__main__":
    main()
