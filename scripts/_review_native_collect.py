#!/usr/bin/env python3
"""완료 원장을 주기적으로 회수하고 로컬 독립 검산을 이어간다."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
REMOTE = "joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/"
NATIVE = "results/pdv2_review_native_20261105"
RING = "results/pdv2_ringpp_maxfix_20261105"


def sync(relative: str, patterns: list, optional: bool = False) -> None:
    if optional:
        exists = subprocess.run(["ssh", "joms-gpu", "test", "-d", "/home/qkrrlans00/Neural-Spectral-Codec/" + relative]).returncode == 0
        if not exists:
            return
    destination = REPO / relative
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-az", "--omit-dir-times", "--include=*/"] +
                   ["--include=" + p for p in patterns] + ["--exclude=*", REMOTE + relative + "/", str(destination) + "/"], check=True)


def main() -> None:
    while True:
        start = time.time()
        try:
            sync(RING + "/odom", ["*.npz", "*.json"])
            sync(RING + "/ledger_status", ["*.json"])
            sync(RING + "/bev", ["status.json", "done.json", "input_verification.json"])
            subprocess.run(["rsync", "-az", REMOTE + RING + "/server_status.json", str(REPO / RING) + "/"], check=True)
            for model in ["author", "v2_s1", "v2_s2", "v2_s3"]:
                # 완료되지 않은 remote 폴더도 부모 descriptor 경로는 이미 존재한다.
                sync(NATIVE + "/seqot/" + model, ["report.json", "predicted_L2_dis.npz", "done.json", "status.json"])
                sync(NATIVE + "/helios/" + model, ["native*.json", "score_progress.json", "native_pairs/*.npz", "done.json", "status.json"], optional=True)
                for k in (15, 25):
                    # 서버 쪽 디렉터리는 첫 추론 전에 없을 수 있어 존재 여부를 확인한다.
                    relative = NATIVE + f"/imlpr/k{k}/" + model
                    exists = subprocess.run(["ssh", "joms-gpu", "test", "-d", "/home/qkrrlans00/Neural-Spectral-Codec/" + relative]).returncode == 0
                    if exists:
                        sync(relative, ["native_*.json", "native_*.npz", "done.json", "status.json"])
            subprocess.run(["python3", "scripts/_review_native_ringpp_aggregate.py"], cwd=REPO, check=True)
            subprocess.run(["python3", "scripts/_review_native_learned_verify.py"], cwd=REPO, check=True)
            state = {"unix_time": time.time(), "elapsed_s": time.time() - start, "passed": True}
        except Exception as exc:
            state = {"unix_time": time.time(), "error": repr(exc), "passed": False}
        with (REPO / NATIVE / "collection_history.jsonl").open("a") as stream:
            stream.write(json.dumps(state) + "\n")
        print(json.dumps(state), flush=True)
        time.sleep(300)


if __name__ == "__main__":
    main()
