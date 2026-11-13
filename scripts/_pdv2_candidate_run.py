#!/usr/bin/env python3
"""자료 준비 완료 후 네 조건의 학습·최종 검증·감사를 이어 실행한다."""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_candidate_data import OUT, train_path
from _pdv2_pilot import TRAIN, VAL
from _pdv2_no_context_ablation import write_json
import yaml

LOG = REPO / "logs/candidate_learning_20261014"
CFG = yaml.safe_load((REPO / "configs/pdv2_candidate_learning.yaml").read_text())


def call(arguments: list, name: str) -> None:
    """로그를 분리하고 실패하면 실행 흐름을 중단한다."""
    with (LOG / (name + ".log")).open("a") as log:
        result = subprocess.run([sys.executable, "-u", *arguments], cwd=REPO,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name}: {(LOG / (name + '.log')).read_text()[-3000:]}")


def main() -> None:
    """시험 자료를 읽지 않고 정한 검증 실험을 끝까지 실행한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    LOG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while True:
        pending = [s + "/" + q for s, q in TRAIN if not train_path(s + "/" + q).with_suffix(".json").exists()]
        if not pending:
            break
        write_json(OUT / "status.json", {"status": "running", "stage": "training_data",
                   "pending": pending, "started_unix": started})
        time.sleep(15)
    call(["-m", "pytest", "-q", "tests/test_candidate_learning.py"], "tests")
    call(["scripts/_pdv2_candidate_learning.py", "protocol"], "protocol")
    write_json(OUT / "status.json", {"status": "running", "stage": "initial_candidate_pools"})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call, ["scripts/_pdv2_candidate_learning.py", "pools", "--kind", kind], "pools_" + kind)
                   for kind in ("gnn", "point")]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    for arm in CFG["arms"]:
        write_json(OUT / "status.json", {"status": "running", "stage": "training", "arm": arm})
        call(["scripts/_pdv2_candidate_learning.py", "train", "--arm", arm], "train_" + arm)
        print("TRAIN_DONE", arm, flush=True)
    jobs = [(a, e, s + "/" + q) for a in CFG["arms"] for e in CFG["checkpoint_epochs"] for s, q in VAL]

    def evaluate(job):
        arm, epoch, sequence = job
        call(["scripts/_pdv2_candidate_learning.py", "evaluate", "--arm", arm,
              "--epoch", str(epoch), "--sequence", sequence],
             f"eval_{arm}_{epoch}_{sequence.replace('/', '_')}")

    write_json(OUT / "status.json", {"status": "running", "stage": "validation", "done": 0, "total": len(jobs)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate, job) for job in jobs]
        for done, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            write_json(OUT / "status.json", {"status": "running", "stage": "validation", "done": done, "total": len(jobs)})
            print("EVAL", done, len(jobs), flush=True)
    call(["scripts/_pdv2_candidate_learning.py", "report"], "report")
    call(["scripts/_pdv2_candidate_verify.py"], "verify")
    write_json(OUT / "status.json", {"status": "complete", "stage": "complete", "test_used": False,
               "runtime_s": time.time() - started, "finished_unix": time.time()})
    print("CANDIDATE_PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
