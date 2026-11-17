#!/usr/bin/env python3
"""장기 실행의 비용 추정·별도 수식 진단·재생성 상태를 회수한다."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from _review_native_collect import NATIVE, REPO, RING, REMOTE, sync


def main() -> None:
    """기존 GPU 실행은 변경하지 않고 작은 상태와 완료 원장을 읽는다."""
    root = REPO / NATIVE
    pid = root / "aux_collection.pid"
    if pid.exists():
        try:
            os.kill(int(pid.read_text()), 0)
        except (ValueError, ProcessLookupError):
            pass
        else:
            raise SystemExit("보조 수집기가 이미 실행 중이다")
    pid.write_text(str(os.getpid()))
    while True:
        start = time.time()
        try:
            subprocess.run(["ssh", "joms-gpu", "cd /home/qkrrlans00/Neural-Spectral-Codec && python3 scripts/_review_native_ringpp_progress.py"], check=True)
            subprocess.run(["rsync", "-az", REMOTE + RING + "/measured_progress.json", str(REPO / RING) + "/"], check=True)
            sync(NATIVE + "/ring_entropy_formula", ["*.json", "pair_dists_*.npy", "*.log"], optional=True)
            entropy = root / "ring_entropy_formula"
            if (entropy / "comparison.json").exists():
                import _review_native_ring_verify as verify
                verify.OUT = entropy
                verify.main()
            sync(NATIVE + "/helios_regenerated_velodyne", ["runner_status.json", "runner.pid", "runner.log", "infer_*.log", "score_*.log"], optional=True)
            result = {"passed": True, "pid": os.getpid(), "unix_time": time.time(), "elapsed_s": time.time() - start}
        except Exception as exc:
            result = {"passed": False, "pid": os.getpid(), "unix_time": time.time(), "error": repr(exc)}
        with (root / "aux_collection_history.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        (root / "aux_collection_status.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        time.sleep(300)


if __name__ == "__main__":
    main()
