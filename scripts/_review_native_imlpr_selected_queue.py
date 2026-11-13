#!/usr/bin/env python3
"""동결된 ImLPR 선택 대조를 서버 GPU1에서 재개 가능하게 실행한다."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
NATIVE = REPO / "results/pdv2_review_native_20261105"
OUT = NATIVE / "imlpr/common_selection_native"


def write(path: Path, value: dict) -> None:
    """진행 파일을 원자적으로 저장한다."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main() -> None:
    """기존 원장 alias는 건드리지 않고 새 가중치만 순차 평가한다."""
    manifest = json.loads((OUT / "selection_manifest.json").read_text())
    pidfile = OUT / "queue.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit("선택 native 실행기가 이미 살아 있다.")
    pidfile.write_text(str(os.getpid()))
    jobs = []
    for candidate in manifest["candidates"]:
        if candidate["original_alias"]:
            continue
        name = candidate["id"]
        jobs.append({"id": name + "_smoke", "gpu": "1", "args": ["infer", "--model", name, "--sequence", "DCC01", "--k", "15", "--smoke"],
                     "done": f"k15/{name}/smoke_batch1/DCC01/done.json"})
        for k in manifest["ks"]:
            for sequence in manifest["sequences"]:
                common = ["--model", name, "--sequence", sequence, "--k", str(k)]
                jobs.append({"id": f"{name}_{k}_{sequence}_infer", "gpu": "1", "args": ["infer"] + common,
                             "done": f"k{k}/{name}/desc_batch1/{sequence}/done.json"})
                jobs.append({"id": f"{name}_{k}_{sequence}_score", "gpu": "", "args": ["score"] + common,
                             "done": f"k{k}/{name}/native_batch1_{sequence}.json"})
    jobs.append({"id": "independent_verify", "gpu": "", "args": ["verify"], "done": "independent_verification.json"})
    write(OUT / "queue_plan.json", {"jobs": jobs, "gpu": "1 only", "primary_inference_wait": True,
        "cost_claim": "정확도 진단이며 Table3 시간/메모리 측정이 아니다."})
    while True:
        state = {"pid": os.getpid(), "unix_time": time.time(), "active": None,
                 "completed": [j["id"] for j in jobs if (OUT / j["done"]).exists()],
                 "total": len(jobs), "complete": False}
        if len(state["completed"]) == len(jobs):
            state["complete"] = True
            write(OUT / "queue_status.json", state)
            return
        primary = json.loads((NATIVE / "learned_queue_status.json").read_text())
        if primary["active"] or primary["errors"] or len(primary["completed"]) != primary["total"]:
            state["waiting"] = "기존 저자/v2 native 전체 추론을 먼저 완료한다."
            write(OUT / "queue_status.json", state)
            time.sleep(30)
            continue
        job = next(j for j in jobs if j["id"] not in state["completed"])
        log = OUT / "logs" / (job["id"] + ".log")
        log.parent.mkdir(exist_ok=True)
        command = ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=" + job["gpu"],
                   "-e", "OMP_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=2",
                   "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "bash", "-lc",
                   "ulimit -s 81920; exec nice -n 10 " + shlex.join(["python3", "-u", "scripts/_review_native_imlpr_selected.py"] + job["args"])]
        with log.open("a") as stream:
            child = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=stream)
            state["active"] = {"job": job["id"], "pid": child.pid, "command": command}
            write(OUT / "queue_status.json", state)
            returncode = child.wait()
        if returncode or not (OUT / job["done"]).exists():
            state["error"] = {"job": job["id"], "returncode": returncode, "unix_time": time.time()}
            write(OUT / "queue_status.json", state)
            with (OUT / "failure_history.jsonl").open("a") as stream:
                stream.write(json.dumps(state["error"]) + "\n")
            raise SystemExit("실패 원장을 보존하고 재개 전 원인을 확인한다.")


if __name__ == "__main__":
    main()
