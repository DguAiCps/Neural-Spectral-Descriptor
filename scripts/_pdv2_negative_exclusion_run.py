#!/usr/bin/env python3
"""오답 제외 네 조건을 학습하고 모든 최종 판정을 평가한다."""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"scripts"))
from _pdv2_no_context_ablation import write_json

OUT = REPO/"results/pdv2_negative_exclusion_20261021"
LOG = REPO/"logs/negative_exclusion_20261021"
SCRIPT = REPO/"scripts/_pdv2_negative_exclusion.py"


def call(label: str, arguments: list) -> None:
    """독립 프로세스와 전용 로그로 실행하며 실패를 전파한다."""
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    with (LOG/(label+".log")).open("a") as handle:
        result = subprocess.run([sys.executable, "-u", str(SCRIPT), *arguments], cwd=REPO,
                                env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(label+"\n"+(LOG/(label+".log")).read_text()[-4000:])


def main() -> None:
    """네 학습 뒤 72시퀀스 평가와 조건별 최종 선택을 수행한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        call("prepare", ["prepare"])
        spec = json.loads((OUT/"protocol.json").read_text())
        call("diagnose", ["diagnose"])
        for arm in spec["config"]["arms"]:
            write_json(OUT/"status.json", {"status": "running", "stage": "train", "arm": arm,
                "elapsed_s": time.time()-started, "test_used": False})
            print("TRAIN", arm, flush=True)
            call("train_"+arm, ["train", "--arm", arm])
        jobs = [(arm, epoch, seq) for arm in spec["config"]["arms"] for epoch in (10, 20) for seq in spec["val"]]

        def evaluate(job):
            arm, epoch, sequence = job
            call(f"val_{arm}_{epoch}_{sequence.replace('/', '_')}",
                 ["val", "--arm", arm, "--epoch", str(epoch), "--sequence", sequence])

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(evaluate, j): j for j in jobs}
            for count, future in enumerate(concurrent.futures.as_completed(pending), 1):
                future.result()
                write_json(OUT/"status.json", {"status": "running", "stage": "val", "done": count,
                    "total": len(jobs), "elapsed_s": time.time()-started, "test_used": False})
                print("VAL_DONE", count, len(jobs), pending[future], flush=True)
        call("report", ["report"])
        write_json(OUT/"status.json", {"status": "evaluation_complete", "elapsed_s": time.time()-started, "test_used": False})
    except Exception as error:
        write_json(OUT/"status.json", {"status": "failed", "reason": str(error), "elapsed_s": time.time()-started, "test_used": False})
        raise


if __name__ == "__main__":
    main()
