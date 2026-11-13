#!/usr/bin/env python3
"""현재 표4 대조의 준비·검증·동결·시험·검산을 순서대로 실행한다."""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_table4_current_20261030"
LOG = REPO / "logs/table4_current_20261030"
STARTED = time.time()


def read(path: Path) -> dict:
    """작업 기록을 읽는다."""
    return json.loads(path.read_text())


def status(stage: str, **values) -> None:
    """진행 상태를 원자적으로 저장한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    value = {"state": "running", "stage": stage, "updated_unix": time.time(),
             "started_unix": STARTED, "elapsed_s": time.time() - STARTED, **values}
    temp = OUT / "status.tmp.json"
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(OUT / "status.json")


def invoke(args: list, name: str) -> None:
    """독립 로그로 실행하고 실패 시 후속 시험을 차단한다."""
    LOG.mkdir(parents=True, exist_ok=True)
    path = LOG / (name + ".log")
    started = time.time()
    with path.open("a") as stream:
        child = subprocess.run([sys.executable, "-u", "scripts/_pdv2_table4_current.py",
                                *map(str, args)], cwd=REPO, env=os.environ.copy(),
                               stdout=stream, stderr=subprocess.STDOUT)
    if child.returncode:
        raise RuntimeError(name + "\n" + path.read_text()[-7000:])
    print("COMPLETE", name, round(time.time() - started, 1), flush=True)


def jobs(tasks: list, stage: str) -> None:
    """메모리 경합을 제한한 네 작업으로 원장을 생성한다."""
    status(stage, completed=0, total=len(tasks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(invoke, args, name) for args, name in tasks]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            status(stage, completed=count, total=len(tasks))


def main() -> None:
    """재학습 없는 세 행을 먼저 끝내고 두 학습 대조를 이어간다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1")
    status("prepare")
    invoke(["prepare"], "prepare")
    spec = read(OUT / "protocol.json")
    for seed in (1, 2, 3):
        invoke(["smoke", "--seed", seed, "--sequence", "kitti360/0004"], f"smoke_s{seed}")
    tasks = []
    for seed in (1, 2, 3):
        for sequence in spec["val"]:
            args = ["eval", "--seed", seed, "--role", "val", "--arm", "retrieval", "--sequence", sequence]
            tasks.append((args, f"retrieval_val_s{seed}_{sequence.replace('/', '_')}"))
    jobs(tasks, "retrieval_validation")
    invoke(["freeze", "--arm", "retrieval"], "freeze_retrieval")
    tasks = []
    for seed in (1, 2, 3):
        for sequence in spec["test"]:
            args = ["eval", "--seed", seed, "--role", "test", "--arm", "retrieval", "--sequence", sequence]
            tasks.append((args, f"retrieval_test_s{seed}_{sequence.replace('/', '_')}"))
    jobs(tasks, "retrieval_test")
    invoke(["finish", "--arm", "retrieval"], "finish_retrieval")
    for arm in ("graph_free", "standard"):
        for seed in (1, 2, 3):
            status("training", arm=arm, seed=seed)
            invoke(["train", "--arm", arm, "--seed", seed], f"train_{arm}_s{seed}")
        tasks = []
        for seed in (1, 2, 3):
            for epoch in (10, 20):
                for sequence in spec["val"]:
                    args = ["eval", "--seed", seed, "--role", "val", "--arm", arm,
                            "--epoch", epoch, "--sequence", sequence]
                    tasks.append((args, f"{arm}_val_s{seed}_ep{epoch}_{sequence.replace('/', '_')}"))
        jobs(tasks, arm + "_validation")
        invoke(["freeze", "--arm", arm], "freeze_" + arm)
        selected = read(OUT / f"freeze_{arm}.json")["models"]
        tasks = []
        for seed in (1, 2, 3):
            for sequence in spec["test"]:
                args = ["eval", "--seed", seed, "--role", "test", "--arm", arm,
                        "--epoch", selected[str(seed)]["epoch"], "--sequence", sequence]
                tasks.append((args, f"{arm}_test_s{seed}_{sequence.replace('/', '_')}"))
        jobs(tasks, arm + "_test")
        invoke(["finish", "--arm", arm], "finish_" + arm)
    status("complete", state="complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", state="failed", error=repr(error))
        raise
