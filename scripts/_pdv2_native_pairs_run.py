#!/usr/bin/env python3
"""고유 표본·두 예산의 학습과 실제 최종 검증을 차례로 끝낸다."""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_no_context_ablation import write_json

OUT = REPO / "results/pdv2_native_pairs_20261020"


def main() -> None:
    """네 조건을 같은 실행 환경에서 수행한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    arms = ("gt_small", "kiss_small", "gt_large", "kiss_large")
    jobs = [("prepare", None)]+[("train", a) for a in arms]+[("val", a) for a in arms]+[("report", None)]
    started = time.time()
    for action, arm in jobs:
        tag = action+("_"+arm if arm else "")
        command = [sys.executable, str(REPO / "scripts/_pdv2_native_pairs.py"), action]
        if arm: command += ["--arm", arm]
        write_json(OUT / "status.json", {"status": "running", "stage": tag, "elapsed_s": time.time()-started})
        print("START", tag, flush=True)
        with (OUT / (tag+".log")).open("w") as stream:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(OUT / "status.json", {"status": "failed", "stage": tag, "returncode": result.returncode})
            raise RuntimeError(tag+" 단계 실패")
        print("DONE", tag, round(time.time()-started, 1), flush=True)
    write_json(OUT / "status.json", {"status": "evaluation_complete", "elapsed_s": time.time()-started, "test_used": False})


if __name__ == "__main__":
    main()
