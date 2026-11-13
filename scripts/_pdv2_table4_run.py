#!/usr/bin/env python3
"""표 4의 검증·동결·시험·집계를 컨테이너에서 이어 실행한다."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_pilot import VAL, TEST
from _pdv2_no_context_ablation import write_json

OUT = REPO / "results/pdv2_table4_20261014"
LOG = REPO / "logs/table4_20261014"


def call(action: str, group: str, seed: int = 1, role: str = "val", seq: str = "") -> None:
    """한 작업을 별도 프로세스로 실행하고 실패를 상위 작업에 전달한다."""
    command = [sys.executable, "-u", "scripts/_pdv2_table4.py", action, "--group", group,
               "--seed", str(seed), "--role", role]
    if seq:
        command += ["--seq", seq]
    name = f"{group}_{action}_s{seed}_{role}_{seq.replace('/', '_') or 'all'}"
    print("START", name, flush=True)
    with (LOG / f"{name}.log").open("a") as handle:
        subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)
    print("DONE", name, flush=True)


def stage(group: str, role: str, workers: int) -> None:
    """고정 목록 전체를 실행하며 결과가 좋은 시퀀스만 고르지 않는다."""
    rows = VAL if role == "val" else TEST
    write_json(OUT / f"status_{group}.json", {"status": "running", "stage": role,
                                             "started_unix": time.time(), "jobs": len(rows) * 3})
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(call, "worker", group, seed, role, sensor + "/" + seq)
                   for seed in (1, 2, 3) for sensor, seq in rows]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main(group: str, workers: int) -> None:
    """시험을 열기 전에 해당 대조군의 선택을 세 시드 모두 고정한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1")
    call("protocol", group)
    if group == "retrieval":
        stage(group, "val", workers)
        for seed in (1, 2, 3):
            call("select", group, seed)
    else:
        write_json(OUT / "status_graph_free.json", {"status": "waiting_training", "seeds": [1, 2, 3]})
        while not all((OUT / f"training_s{s}_complete.json").exists() for s in (1, 2, 3)):
            path = OUT / "training_status.json"
            if path.exists():
                status = json.loads(path.read_text())
                if status["status"] == "failed":
                    raise RuntimeError(status)
            time.sleep(30)
    call("freeze", group)
    stage(group, "test", workers)
    call("report", group)
    write_json(OUT / f"status_{group}.json", {"status": "complete", "finished_unix": time.time(),
                                             "summary": f"summary_{group}.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("retrieval", "graph_free"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    try:
        main(args.group, args.workers)
    except Exception as error:
        write_json(OUT / f"status_{args.group}.json", {"status": "failed", "error": repr(error)})
        raise
