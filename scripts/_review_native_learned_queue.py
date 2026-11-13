#!/usr/bin/env python3
"""서버 GPU1에서 native 학습형 진단을 하나씩 재개 가능하게 실행한다."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105"


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def jobs() -> list:
    """고정 모델들을 저자→v2시드 순서로 생성한다."""
    tasks = []
    for model in ["author", "v2_s1", "v2_s2", "v2_s3"]:
        for role in ["database", "query"]:
            tasks.append({"id": f"seqot_{model}_{role}", "argv": ["scripts/_review_native_seqot.py", "infer", "--model", model, "--role", role, "--batch", "1"],
                "ready": [f"seqot/inputs/{role}.npy", f"seqot/inputs/{role}.json"], "done": f"seqot/{model}/sub/{role}/done.json", "gpu": "1"})
        tasks.append({"id": f"seqot_{model}_score", "argv": ["scripts/_review_native_seqot.py", "score", "--model", model],
            "ready": [f"seqot/{model}/sub/{r}/done.json" for r in ["database", "query"]], "done": f"seqot/{model}/native/report.json", "gpu": "1"})
        for n in (1, 2, 3):
            for sensor in ["Aeva", "Avia", "Ouster", "Velodyne"]:
                name = f"Roundabout{n:02d}-{sensor}"
                tasks.append({"id": f"helios_{model}_{name}", "argv": ["scripts/_review_native_helios.py", "infer", "--model", model, "--sequence", name],
                    "ready": ["helios/transfer_complete.json", f"helios/data/validation/{name}/extracted.json"],
                    "done": f"helios/{model}/desc/{name}/done.json", "gpu": "1"})
        for k in (15, 25):
            for seq in ["DCC01", "DCC02", "DCC03"]:
                tasks.append({"id": f"imlpr_batch1_{model}_k{k}_{seq}", "argv": ["scripts/_review_native_imlpr.py", "infer", "--model", model, "--k", str(k), "--sequence", seq],
                    "ready": [f"imlpr/inputs/{seq}/transfer_complete.json"], "done": f"imlpr/k{k}/{model}/desc_batch1/{seq}/done.json", "gpu": "1"})
                tasks.append({"id": f"imlpr_batch1_{model}_k{k}_{seq}_score", "argv": ["scripts/_review_native_imlpr.py", "score", "--model", model, "--k", str(k), "--sequence", seq],
                    "ready": [f"imlpr/k{k}/{model}/desc_batch1/{seq}/done.json"], "done": f"imlpr/k{k}/{model}/native_batch1_{seq}.json", "gpu": ""})
    return tasks


def run() -> None:
    """다른 작업에 손대지 않고 전용 native 실행 한 개만 유지한다."""
    p = OUT / "learned_queue.pid"
    if p.exists():
        try:
            os.kill(int(p.read_text()), 0)
        except (ValueError, ProcessLookupError):
            pass
        else:
            raise SystemExit(f"이미 실행 중: {p.read_text().strip()}")
    p.write_text(str(os.getpid()) + "\n")
    queue = jobs()
    write(OUT / "learned_queue_plan.json", {"jobs": queue, "gpu": "1 only", "cost": "정확도 진단; 독립속도측정 아님"})
    error_file = OUT / "learned_queue_errors.json"
    errors = json.loads(error_file.read_text()) if error_file.exists() else {}
    active, child, handle = None, None, None
    start = time.time()
    while True:
        if child is not None and child.poll() is not None:
            handle.close()
            if child.returncode:
                errors[active["id"]] = {"returncode": child.returncode, "time": time.time()}
                write(error_file, errors)
            child, active = None, None
        complete = [j["id"] for j in queue if (OUT / j["done"]).exists()]
        if child is None:
            for job in queue:
                if job["id"] in complete or job["id"] in errors:
                    continue
                if not all((OUT / p).exists() for p in job["ready"]):
                    continue
                log = OUT / "learned_logs" / f"{job['id']}.log"
                log.parent.mkdir(exist_ok=True)
                handle = log.open("a")
                cmd = ["docker", "exec", "-e", f"CUDA_VISIBLE_DEVICES={job['gpu']}", "-e", "OMP_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=2",
                       "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "bash", "-lc",
                       "ulimit -s 81920; exec nice -n 10 " + shlex.join(["python3", "-u"] + job["argv"])]
                child = subprocess.Popen(cmd, cwd=REPO, stdout=handle, stderr=handle)
                active = job
                break
        write(OUT / "learned_queue_status.json", {"pid": os.getpid(), "elapsed_s": time.time() - start,
             "active": None if active is None else {"id": active["id"], "pid": child.pid},
             "completed": complete, "total": len(queue), "errors": errors,
             "waiting_input": [j["id"] for j in queue if j["id"] not in complete and not all((OUT / p).exists() for p in j["ready"])]})
        if len(complete) == len(queue):
            break
        time.sleep(15)


if __name__ == "__main__":
    run()
