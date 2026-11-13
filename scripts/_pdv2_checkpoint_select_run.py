#!/usr/bin/env python3
"""검증 체크포인트 비교→선택 동결→고정 시험을 컨테이너에서 실행한다."""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts")]
from _pdv2_checkpoint_select import OUT, read_json
from _pdv2_no_context_ablation import write_json

LOG = REPO / "logs/checkpoint_select_20261014"


def call(action: str, job: tuple | None = None, audit: bool = False) -> None:
    """실행 로그를 남기고 실패를 상위 실행에 알린다."""
    cmd = [sys.executable, "-u", "scripts/_pdv2_checkpoint_select.py", action]
    label = action
    if job:
        seed, cid, role, sequence = job
        cmd += ["--seed", str(seed), "--checkpoint", cid, "--role", role, "--sequence", sequence]
        label += f"_s{seed}_{cid}_{role}_{sequence.replace('/', '_')}"
    if audit:
        cmd.append("--audit")
        label += "_audit"
    with (LOG / f"{label}.log").open("a") as handle:
        process = subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"{label}: {(LOG / (label + '.log')).read_text()[-4000:]}")


def run_stage(jobs: list, role: str) -> None:
    """GPU 키 생성 두 작업과 CPU 평가 네 작업을 겹쳐 실행한다."""
    done = 0
    total = len(jobs)
    started = time.time()
    write_json(OUT / "status.json", {"stage": role, "status": "running", "done": 0,
                                     "total": total, "stage_started_unix": started})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as gpu, \
            concurrent.futures.ThreadPoolExecutor(max_workers=4) as cpu:
        key_futures = {gpu.submit(call, "keys", job): job for job in jobs}
        score_futures = set()
        while key_futures or score_futures:
            ready, _ = concurrent.futures.wait(
                set(key_futures) | score_futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in ready:
                future.result()
                if future in key_futures:
                    job = key_futures.pop(future)
                    score_futures.add(cpu.submit(call, "score", job))
                    print("KEY_READY", *job, flush=True)
                else:
                    score_futures.remove(future)
                    done += 1
                    write_json(OUT / "status.json", {"stage": role, "status": "running", "done": done,
                                                     "total": total, "stage_started_unix": started})
                    print("SCORED", role, done, total, flush=True)


def main() -> None:
    """기존 모델을 재현한 뒤 검증 선택과 고정 시험을 순서대로 실행한다."""
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1")
    LOG.mkdir(parents=True, exist_ok=True)
    call("prepare")
    spec = read_json(OUT / "protocol.json")
    write_json(OUT / "status.json", {"stage": "audit", "status": "running"})
    if not (OUT / "audit_complete.json").exists():
        for seed in (1, 2, 3):
            job = (seed, "best_model", "val", "kitti360/0006")
            call("keys", job, audit=True)
            call("score", job, audit=True)
            print("AUDIT_DONE", seed, flush=True)
        write_json(OUT / "audit_complete.json", {"seeds": [1, 2, 3], "status": "complete"})
    if not (OUT / "freeze.json").exists():
        jobs = [(int(seed), row["id"], "val", sequence)
                for seed, rows in spec["candidates"].items() for row in rows
                for sequence in spec["roster"]["val"]]
        run_stage(jobs, "val")
        call("select")
        print("SELECTION_FROZEN", read_json(OUT / "freeze.json")["selected"], flush=True)
    frozen = read_json(OUT / "freeze.json")
    jobs = [(int(seed), row["id"], "test", sequence)
            for seed, row in frozen["selected"].items() for sequence in spec["roster"]["test"]]
    run_stage(jobs, "test")
    call("report")
    write_json(OUT / "status.json", {"stage": "complete", "status": "complete", "finished_unix": time.time()})
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
