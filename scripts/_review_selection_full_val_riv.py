#!/usr/bin/env python3
"""승인된 인계 뒤 ImLPR 검증 입력을 서로 겹치지 않는 CPU 작업으로 복구한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]
from _review_selection_bev import read, write, sha
from _review_selection_full_inputs import OUT, INPUT, cache, configure


def worker(key: str) -> None:
    """비누적 누락 스캔은 기존 경로, HeLiPR 전수는 독립 CPU3개로 처리한다."""
    assert not os.environ.get("CUDA_VISIBLE_DEVICES")
    sensor, seq = key.split("/")
    if sensor != "helipr":
        from _review_selection_full_relay import cpu_worker
        cpu_worker("imlpr", key)
        return
    p = configure("imlpr")
    p.ROLES["test_odom"] = ([(sensor, seq)], p.ODOM, "val_", "test_odom")
    p.cmd_index("test_odom")
    ids = np.load(cache(sensor, seq))["scan_ids"]
    home = p.role_dir("test_odom", sensor, seq)
    from _review_selection_full_riv_parallel import generate
    tick = time.time()
    generate(sensor, seq, home, ids, processes=3)
    assert all((home / f"{i:06d}.npy").exists() for i in range(len(ids)))
    write(INPUT / "relay_status" / f"imlpr_{sensor}_{seq}.json",
          dict(status="complete",seconds=time.time()-tick,finished_unix=time.time(),
               cache_sha256=sha(cache(sensor,seq)),execution="independent CPU3 / original RIV / bounded raw cache"))


def run() -> None:
    """기존 worker 인계를 확인한 뒤 최대6개 CPU로 영속 작업열을 실행한다."""
    approval = read(OUT / "val_riv_handoff.json")
    assert approval["approved"] and approval["previous_workers_stopped"]
    proof = read(OUT / "riv_parallel_audit/validated.json")
    assert proof["all_arrays_exact"]
    assert proof["implementation_sha256"] == sha(ROOT / "scripts/_review_selection_full_riv_parallel.py")
    lock = (OUT / "val_riv_queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(OUT / "input_relay_plan.json")
    status = OUT / "val_riv_queue_state.json"
    completed = []
    failures = []
    def execute(key):
        marker = INPUT / "relay_status" / ("imlpr_" + key.replace("/", "_") + ".json")
        if marker.exists() and read(marker).get("status") == "complete":
            return key
        with (OUT / "logs" / ("parallel_val_imlpr_" + key.replace("/", "_") + ".log")).open("a") as log:
            code = subprocess.call(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "OMP_NUM_THREADS=1",
                                    "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-w", "/workspace/Neural-Spectral-Codec",
                                    "ring_baseline", "python3", "-u", "scripts/_review_selection_full_val_riv.py", "worker",
                                    "--sequence", key], stdout=log, stderr=subprocess.STDOUT)
        assert code == 0, (key, code)
        return key
    for accumulated, workers in ((False, 3), (True, 2)):
        keys = [r["sensor"] + "/" + r["seq"] for r in plan["rows"] if (r["sensor"] == "helipr") == accumulated]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute, key): key for key in keys}
            write(status, dict(status="running",pid=os.getpid(),stage="accumulated" if accumulated else "missing_single",
                               active=keys,completed=completed,failures=failures,updated_unix=time.time()))
            for future in concurrent.futures.as_completed(futures):
                try:
                    completed.append(future.result())
                except Exception as exc:
                    failures.append(dict(sequence=futures[future],error=str(exc)))
                write(status, dict(status="running",pid=os.getpid(),active=[v for f,v in futures.items() if not f.done()],
                                   completed=completed,failures=failures,updated_unix=time.time()))
    write(status, dict(status="failed" if failures else "complete",pid=os.getpid(),completed=completed,
                       failures=failures,updated_unix=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker"))
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.sequence)
    else:
        run()
