#!/usr/bin/env python3
"""고정 BEV 시간 대조를 두 GPU에 분산하고 실패·진척을 보존한다."""
import argparse
import concurrent.futures
import os
import queue
from pathlib import Path
import subprocess
import sys
import time

import _review_response_bev as b


def run(gpus: list, seed1_val_local: bool = False, seeds: list = None) -> None:
    """GPU마다 한 프로세스씩만 실행하고 검증을 먼저 완료한다."""
    spec = b.prepare()
    log = b.OUT/"logs"
    log.mkdir(exist_ok=True)
    start = time.time()
    for role in ("val", "test"):
        jobs = [j for j in spec["jobs"] if j["role"] == role and (seeds is None or j["seed"] in seeds) and
                not (seed1_val_local and role == "val" and j["seed"] == 1)]
        pending = queue.Queue()
        for job in jobs: pending.put(job)
        def lane(gpu: str) -> None:
            done = 0
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            while True:
                try: job = pending.get_nowait()
                except queue.Empty: break
                name = job["sensor"]+"/"+job["sequence"]
                logfile = log/f"{role}_s{job['seed']}_{name.replace('/', '_')}.log"
                try:
                    with logfile.open("w") as f:
                        subprocess.run([sys.executable, str(Path(b.__file__)), "worker", "--role", role,
                                        "--seed", str(job["seed"]), "--sequence", name], env=env,
                                       stdout=f, stderr=subprocess.STDOUT, check=True)
                except Exception as exc:
                    b.write(b.OUT/f"status_gpu{gpu}.json", dict(status="failed", job=job, error=str(exc), log=str(logfile)))
                    raise
                done += 1
                b.write(b.OUT/f"status_gpu{gpu}.json", dict(status="running", role=role, done=done,
                           total=len(jobs), remaining=pending.qsize(), last=job, elapsed_s=time.time()-start))
                print("PROGRESS", gpu, role, done, "remaining", pending.qsize(), name, flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(lane, gpu) for gpu in gpus]
            for f in concurrent.futures.as_completed(futures):
                f.result()
    b.write(b.OUT/"server_complete.json", dict(status="complete", elapsed_s=time.time()-start,
             seeds=seeds, seed1_validation_external=seed1_val_local, protocol_sha256=b.digest(b.OUT/"protocol.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--seed1-val-local", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int)
    a = parser.parse_args()
    run(a.gpus, a.seed1_val_local, a.seeds)
