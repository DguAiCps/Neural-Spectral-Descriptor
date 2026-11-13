#!/usr/bin/env python3
"""계열 기울기 진단·같은 배치 재학습·네 최종 검증을 수행한다."""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
from _pdv2_no_context_ablation import write_json

OUT = REPO / "results/pdv2_family_balance_20261020"


def main() -> None:
    """실패 단계와 로그를 보존하며 동결한 순서대로 실행한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    script = str(REPO / "scripts/_pdv2_family_balance.py")
    diag = str(REPO / "scripts/_pdv2_family_balance_diagnose.py")
    jobs = [("prepare", [script, "prepare"])]
    jobs += [("diagnose_"+s, [diag, s]) for s in ("initial", "reference")]
    jobs += [("train_"+a, [script, "train", "--arm", a]) for a in ("natural", "balanced")]
    arms = ("natural_720", "natural_900", "balanced_720", "balanced_900")
    jobs += [("diagnose_"+a, [diag, a]) for a in arms]
    jobs += [("val_"+a, [script, "val", "--arm", a]) for a in arms]
    jobs += [("report", [script, "report"])]
    started = time.time()
    for tag, args in jobs:
        write_json(OUT / "status.json", {"status": "running", "stage": tag, "elapsed_s": time.time()-started})
        print("START", tag, flush=True)
        with (OUT / (tag+".log")).open("w") as stream:
            result = subprocess.run([sys.executable, *args], cwd=REPO, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(OUT / "status.json", {"status": "failed", "stage": tag, "returncode": result.returncode})
            raise RuntimeError(tag+" 실패")
        print("DONE", tag, round(time.time()-started, 1), flush=True)
    write_json(OUT / "status.json", {"status": "evaluation_complete", "elapsed_s": time.time()-started, "test_used": False})


if __name__ == "__main__":
    main()
