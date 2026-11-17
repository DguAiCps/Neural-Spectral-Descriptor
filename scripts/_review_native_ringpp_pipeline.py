#!/usr/bin/env python3
"""로컬 입력 준비·전송과 서버 GPU1 재평가를 재개 가능하게 이어 돌린다."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_ringpp_maxfix_20261105"
REMOTE = "joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/results/pdv2_ringpp_maxfix_20261105"
SCRIPT = "scripts/_review_native_pdv2_ringpp.py"


def write_json(path: Path, value: dict) -> None:
    """상태를 원자적으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def command(stage: str, sequence: str, gpu: str) -> list:
    """지정 GPU 밖에는 접근하지 않는 컨테이너 명령을 만든다."""
    argv = ["python3", "-u", SCRIPT, stage, "--sequence", sequence]
    return ["docker", "exec", "-e", f"CUDA_VISIBLE_DEVICES={gpu}",
            "-e", "OMP_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2",
            "-e", "OPENBLAS_NUM_THREADS=2", "-e", "LOKY_MAX_CPU_COUNT=2",
            "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "bash", "-lc",
            "ulimit -s 81920; exec nice -n 10 " + shlex.join(argv)]


def sequences() -> list:
    """작은 입력부터 실행하되 평가 목록 자체는 바꾸지 않는다."""
    p = json.loads((OUT / "protocol.json").read_text())
    return [f"{r['sensor']}/{r['sequence']}" for r in sorted(p["sequences"], key=lambda r: r["frames"])]


def produce_one(sequence: str) -> dict:
    """한 시퀀스의 손실 없는 정규화 입력을 준비하고 서버에 보낸다."""
    name = sequence.replace("/", "_")
    directory = OUT / "points" / name
    log_path = OUT / "logs" / f"prepare_{name}.log"
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("a") as log:
        result = subprocess.run(command("points", sequence, ""), cwd=REPO, stdout=log, stderr=log)
        if result.returncode:
            return {"sequence": sequence, "stage": "prepare_failed", "returncode": result.returncode}
        result = subprocess.run(["rsync", "-a", "--partial", "--exclude=*.tmp*", "--exclude=transferred.json", str(directory), REMOTE + "/points/"],
                                cwd=REPO, stdout=log, stderr=log)
        row = {"sequence": sequence, "stage": "transferred" if result.returncode == 0 else "transfer_failed",
               "returncode": result.returncode, "done_time": time.time()}
        if result.returncode == 0:
            write_json(directory / "transferred.json", row)
            marker = subprocess.run(["rsync", "-a", str(directory / "transferred.json"), REMOTE + f"/points/{name}/"],
                                    cwd=REPO, stdout=log, stderr=log)
            if marker.returncode:
                row.update(stage="marker_failed", returncode=marker.returncode)
        return row


def producer(workers: int) -> None:
    """입력 준비를 제한된 CPU 작업 수로 병렬 실행한다."""
    roster = sequences()
    # 완료 마커는 호스트가 쓴다. 컨테이너가 시퀀스 디렉터리를 만들기 전에
    # 호스트 소유로 준비해 root 소유 폴더의 쓰기 권한 문제를 방지한다.
    for sequence in roster:
        (OUT / "points" / sequence.replace("/", "_")).mkdir(parents=True, exist_ok=True)
    start = time.time()
    completed = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(produce_one, seq): seq for seq in roster}
        while len(completed) != len(roster):
            for future, seq in futures.items():
                if seq not in completed and future.done():
                    try:
                        completed[seq] = future.result()
                    except Exception as exc:
                        completed[seq] = {"sequence": seq, "stage": "exception", "error": repr(exc)}
                    print(json.dumps(completed[seq]), flush=True)
            write_json(OUT / "producer_status.json", {
                "pid": os.getpid(), "host": socket.gethostname(), "workers": workers,
                "elapsed_s": time.time() - start, "total": len(roster), "completed": completed,
                "finished": len(completed) == len(roster),
            })
            if len(completed) != len(roster):
                time.sleep(20)


def server(workers: int) -> None:
    """GPU1에서 특징 생성과 한 시퀀스씩 검색을 실행한다."""
    roster = sequences()
    start = time.time()
    processes = {}
    errors = {}
    handles = {}
    (OUT / "logs").mkdir(exist_ok=True)
    while True:
        for key, proc in list(processes.items()):
            result = proc.poll()
            if result is None:
                continue
            handles.pop(key).close()
            if result:
                errors[key] = {"returncode": result, "time": time.time()}
            processes.pop(key)
        completed = [seq for seq in roster if (OUT / "ledger_status" / seq.replace("/", "_") / "done.json").exists()]
        if len(completed) == len(roster):
            write_json(OUT / "server_status.json", {"finished": True, "completed": completed,
                       "elapsed_s": time.time() - start, "errors": errors, "pid": os.getpid()})
            break
        generating = sum(k.startswith("generate:") for k in processes)
        retrieving = any(k.startswith(("audit:", "dump:")) for k in processes)
        for seq in roster:
            name = seq.replace("/", "_")
            if seq in completed:
                continue
            bev = OUT / "bev" / name
            if (bev / "done.json").exists():
                stage = "dump" if (bev / "input_verification.json").exists() else "audit"
                key = f"{stage}:{seq}"
                if not retrieving and key not in errors:
                    log = (OUT / "logs" / f"{stage}_{name}.log").open("a")
                    processes[key] = subprocess.Popen(command(stage, seq, "1" if stage == "dump" else ""), cwd=REPO, stdout=log, stderr=log)
                    handles[key] = log
                    retrieving = True
                continue
            key = f"generate:{seq}"
            if generating >= workers or key in processes or key in errors:
                continue
            if not (OUT / "points" / name / "transferred.json").exists():
                continue
            log = (OUT / "logs" / f"generate_{name}.log").open("a")
            processes[key] = subprocess.Popen(command("generate", seq, "1"), cwd=REPO, stdout=log, stderr=log)
            handles[key] = log
            generating += 1
        status = {"finished": False, "pid": os.getpid(), "host": socket.gethostname(),
                  "gpu": "CUDA_VISIBLE_DEVICES=1 only", "generation_workers": workers,
                  "elapsed_s": time.time() - start, "completed": completed, "total": len(roster),
                  "active": {key: proc.pid for key, proc in processes.items()}, "errors": errors}
        write_json(OUT / "server_status.json", status)
        time.sleep(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["producer", "server"])
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    pidfile = OUT / f"{args.mode}.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
        except (ProcessLookupError, ValueError):
            pass
        else:
            raise SystemExit(f"이미 실행 중: {pidfile}")
    pidfile.write_text(str(os.getpid()) + "\n")
    if args.mode == "producer":
        producer(args.workers or 3)
    else:
        server(args.workers or 6)
