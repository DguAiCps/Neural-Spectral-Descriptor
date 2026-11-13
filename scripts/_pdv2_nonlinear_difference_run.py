#!/usr/bin/env python3
"""비선형 차이 키의 시드 1 다섯 조건을 고정 예산으로 실행한다."""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_no_context_ablation import write_json

OUT = REPO / "results/pdv2_nonlinear_difference_20261021"
LOG = REPO / "logs/nonlinear_difference_20261021"
SCRIPT = REPO / "scripts/_pdv2_nonlinear_difference.py"


def call(label: str, arguments: list) -> None:
    """실패 로그를 보존하면서 독립 프로세스에서 실행한다."""
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    with (LOG / (label + ".log")).open("a") as handle:
        result = subprocess.run([sys.executable, "-u", str(SCRIPT), *arguments],
                                cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(label + "\n" + (LOG / (label + ".log")).read_text()[-4000:])


def main() -> None:
    """동결 규칙대로 학습을 순차 실행하고 시퀀스 평가는 두 작업씩 실행한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    call("prepare", ["prepare"])
    spec = json.loads((OUT / "protocol.json").read_text())
    try:
        for seed in (1,):
            if seed > 1 and not json.loads((OUT / "summary.json").read_text())["pilot_eligible"]:
                break
            arms = spec["config"]["arms"] if seed == 1 else spec["config"]["expansion_arms"]
            for arm in arms:
                write_json(OUT / "status.json", {"status": "running", "stage": "train",
                           "seed": seed, "arm": arm, "elapsed_s": time.time()-started})
                print("TRAIN", seed, arm, flush=True)
                call(f"train_s{seed}_{arm}", ["train", "--seed", str(seed), "--arm", arm])
            jobs = [(arm, epoch, sequence) for arm in arms for epoch in (10, 20)
                    for sequence in spec["val"]]

            def evaluate(job):
                arm, epoch, sequence = job
                call(f"val_s{seed}_{arm}_{epoch}_{sequence.replace('/', '_')}",
                     ["val", "--seed", str(seed), "--arm", arm,
                      "--epoch", str(epoch), "--sequence", sequence])

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                pending = {pool.submit(evaluate, j): j for j in jobs}
                for done, future in enumerate(concurrent.futures.as_completed(pending), 1):
                    future.result()
                    write_json(OUT / "status.json", {"status": "running", "stage": "val",
                               "seed": seed, "done": done, "total": len(jobs),
                               "elapsed_s": time.time()-started})
                    print("VAL_DONE", seed, done, len(jobs), pending[future], flush=True)
            call(f"report_s{seed}", ["report"])
        write_json(OUT / "status.json", {"status": "evaluation_complete",
                   "elapsed_s": time.time()-started, "test_used": False})
    except Exception as error:
        write_json(OUT / "status.json", {"status": "failed", "reason": str(error),
                   "elapsed_s": time.time()-started, "test_used": False})
        raise


if __name__ == "__main__":
    main()
