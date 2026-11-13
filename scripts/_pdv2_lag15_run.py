#!/usr/bin/env python3
"""컨테이너에서 분류기→검증 선택·고정→시험을 순서대로 실행한다."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_pilot import TRAIN, VAL, TEST
from _pdv2_lag15_campaign import classifier_path

OUT = REPO / "results/pdv2_symmetric3_20261013"
LOG = REPO / "logs/symmetric3_20261013"


def run(action: str, seed: int = 1, role: str = "val", seq: str | None = None,
        mode: str = "fixed", train_only: bool = False) -> None:
    """한 작업을 별도 프로세스로 격리하고 실패하면 전체 진행을 멈춘다."""
    command = [sys.executable, "scripts/_pdv2_lag15_campaign.py", action,
               "--seed", str(seed), "--role", role, "--mode", mode]
    if seq:
        command += ["--seq", seq]
    if train_only:
        command += ["--train-only"]
    name = "_".join((action, f"s{seed}", role, mode, (seq or "all").replace("/", "_"),
                     "fit" if train_only else "work"))
    print("START", name, flush=True)
    with (LOG / f"{name}.log").open("a") as handle:
        subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)
    print("DONE", name, flush=True)


def cpu_stage(action: str, seed: int, role: str = "val", mode: str = "fixed") -> None:
    """서로 다른 시퀀스의 CPU 검색을 네 프로세스로 실행한다."""
    roster = VAL if role == "val" else TEST
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, action, seed, role, s + "/" + q, mode) for s, q in roster]
        for future in futures:
            future.result()


def main() -> None:
    """모델이 준비된 범위만 선택해 실행하되 시험은 항상 전역 동결 뒤에 둔다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("seed1-val", "remaining", "test"))
    args = parser.parse_args()
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1")
    LOG.mkdir(parents=True, exist_ok=True)
    if args.stage != "test":
        seeds = (1,) if args.stage == "seed1-val" else (2, 3)
        for seed in seeds:
            if seed != 1:
                marker = OUT / f"training_s{seed}_complete.json"
                while not marker.exists():
                    print("WAIT_TRAINING", seed, flush=True)
                    time.sleep(30)
                if not classifier_path(seed).exists():
                    for sensor, seq in TRAIN:
                        run("clf", seed, seq=sensor + "/" + seq)
                    run("clf", seed, train_only=True)
            for sensor, seq in VAL:
                run("keys", seed, seq=sensor + "/" + seq)
            cpu_stage("grid", seed)
            run("select-tuple", seed)
            cpu_stage("score", seed, mode="selected")
            run("select-confidence", seed)
            cpu_stage("score", seed, mode="fixed")
        if args.stage == "seed1-val":
            return
    run("freeze")
    for seed in (1, 2, 3):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for sensor, seq in TEST:
                name = sensor + "/" + seq
                run("keys", seed, role="test", seq=name)
                futures.append(pool.submit(run, "score", seed, "test", name, "selected"))
            for future in futures:
                future.result()
    (OUT / "execution_complete.json").write_text(json.dumps({"created_unix": time.time(),
                                                              "seeds": [1, 2, 3]}))
    print("CAMPAIGN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
