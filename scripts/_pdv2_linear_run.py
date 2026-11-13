#!/usr/bin/env python3
"""가중합 네 조건의 학습·검증·선택·감사를 끝까지 실행한다."""
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_pilot import VAL
from _pdv2_no_context_ablation import write_json

OUT = REPO / "results/pdv2_linear_fusion_20261015"
LOG = REPO / "logs/linear_fusion_20261015"
CFG = yaml.safe_load((REPO / "configs/pdv2_linear_fusion.yaml").read_text())


def call(arguments: list, name: str) -> None:
    """각 단계의 로그를 보존하고 실패를 조용히 건너뛰지 않는다."""
    with (LOG / (name + ".log")).open("a") as handle:
        status = subprocess.run([sys.executable, "-u", *arguments], cwd=REPO,
                                stdout=handle, stderr=subprocess.STDOUT)
    if status.returncode:
        raise RuntimeError(f"{name}: {(LOG / (name + '.log')).read_text()[-3000:]}")


def main() -> None:
    """학습 전에 규칙을 고정하고 검증 전용 실험을 수행한다."""
    started = time.time()
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    call(["-m", "pytest", "-q", "tests/test_linear_fusion.py", "tests/test_candidate_learning.py"], "tests")
    call(["scripts/_pdv2_linear_fusion.py", "prepare"], "prepare")
    call(["scripts/_pdv2_linear_fusion.py", "smoke"], "smoke")
    call(["scripts/_pdv2_linear_fusion.py", "protocol"], "protocol")
    for arm in CFG["arms"]:
        write_json(OUT / "status.json", {"status": "running", "stage": "training", "arm": arm})
        call(["scripts/_pdv2_linear_fusion.py", "train", "--arm", arm], "train_" + arm)
        print("TRAIN_DONE", arm, flush=True)
    jobs = [(a, e, s + "/" + q) for a in CFG["arms"] for e in CFG["checkpoint_epochs"] for s, q in VAL]

    def evaluate(job):
        arm, epoch, sequence = job
        call(["scripts/_pdv2_linear_fusion.py", "evaluate", "--arm", arm,
              "--epoch", str(epoch), "--sequence", sequence],
             f"eval_{arm}_{epoch}_{sequence.replace('/', '_')}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate, job) for job in jobs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            write_json(OUT / "status.json", {"status": "running", "stage": "validation", "done": done, "total": len(jobs)})
            print("EVAL", done, len(jobs), flush=True)
    write_json(OUT / "status.json", {"status": "running", "stage": "selection_and_verification"})
    call(["scripts/_pdv2_linear_fusion.py", "report"], "report")
    call(["scripts/_pdv2_linear_verify.py"], "verify")
    write_json(OUT / "status.json", {"status": "complete", "stage": "complete", "test_used": False,
               "runtime_s": time.time() - started, "finished_unix": time.time()})
    print("LINEAR_PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
