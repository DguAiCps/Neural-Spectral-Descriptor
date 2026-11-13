#!/usr/bin/env python3
"""고정 혼합 값과 여섯 대조를 평가하고 통과 후보만 확장한다."""
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

OUT = REPO / "results/pdv2_mixed_value_20261023"
LOG = REPO / "logs/mixed_value_20261023"
SCRIPT = REPO / "scripts/_pdv2_mixed_value.py"


def call(label: str, arguments: list) -> None:
    """각 작업을 독립 프로세스로 실행하고 로그를 보존한다."""
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    with (LOG / (label + ".log")).open("a") as handle:
        result = subprocess.run([sys.executable, "-u", str(SCRIPT), *arguments],
                                cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(label + "\n" + (LOG / (label + ".log")).read_text()[-4000:])


def main() -> None:
    """학습은 순차, 검증은 두 작업씩 수행하며 사전 확장 규칙을 따른다."""
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        call("prepare", ["prepare"])
        spec = json.loads((OUT / "protocol.json").read_text())
        for seed in (1, 2, 3):
            if seed == 1:
                arms = [*spec["config"]["arms"], spec["config"]["reference"]]
                train_arms = spec["config"]["train_arms"]
            else:
                selection = json.loads((OUT / "expansion_selection.json").read_text())
                if seed not in selection["seeds"]:
                    break
                arms = train_arms = selection["arms"]
            for arm in train_arms:
                write_json(OUT / "status.json", {"status": "running", "stage": "train",
                    "seed": seed, "arm": arm, "elapsed_s": time.time() - started, "test_used": False})
                print("TRAIN", seed, arm, flush=True)
                call(f"train_s{seed}_{arm}", ["train", "--seed", str(seed), "--arm", arm])
            jobs = [(arm, epoch, seq) for arm in arms for epoch in (10, 20) for seq in spec["val"]]

            def evaluate(job):
                arm, epoch, sequence = job
                call(f"val_s{seed}_{arm}_{epoch}_{sequence.replace('/', '_')}",
                     ["val", "--seed", str(seed), "--arm", arm, "--epoch", str(epoch),
                      "--sequence", sequence])

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                pending = {pool.submit(evaluate, j): j for j in jobs}
                for count, future in enumerate(concurrent.futures.as_completed(pending), 1):
                    future.result()
                    write_json(OUT / "status.json", {"status": "running", "stage": "val",
                        "seed": seed, "done": count, "total": len(jobs),
                        "elapsed_s": time.time() - started, "test_used": False})
                    print("VAL_DONE", seed, count, len(jobs), pending[future], flush=True)
            call(f"report_s{seed}", ["report"])
        write_json(OUT / "status.json", {"status": "evaluation_complete",
                   "elapsed_s": time.time() - started, "test_used": False})
    except Exception as error:
        write_json(OUT / "status.json", {"status": "failed", "reason": str(error),
                   "elapsed_s": time.time() - started, "test_used": False})
        raise


if __name__ == "__main__":
    main()
