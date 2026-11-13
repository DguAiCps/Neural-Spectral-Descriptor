#!/usr/bin/env python3
"""원본 전송이 끝나면 서버에서 RING++ 캐시·측정·검산을 연속 실행한다."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pdv2_seqot_ring_repair_20261106/ringpp_server_cost_v2"


def status(stage: str, **fields) -> None:
    """재접속해 읽을 수 있는 상태를 원자적으로 저장한다."""
    value = {"stage": stage, "pid": os.getpid(), "updated_unix": time.time(), **fields}
    temp = OUT / "status.tmp.json"
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(OUT / "status.json")


def main() -> None:
    """동시에 같은 작업을 실행하지 않고 완료 또는 오류까지 상태를 남긴다."""
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = ROOT / "data/table3_server_input/mulran/Sejong01"
        mapping = json.loads((root / "scan_mapping.json").read_text())
        expected = [root / name for name in mapping["scan_files"].values()]
        while True:
            present = sum(p.is_file() and p.stat().st_size == 1048576 for p in expected)
            status("waiting_inputs", present=present, expected=len(expected),
                   scope="D: 원본 전송 완료 대기; 로컬 GPU 실행 없음")
            if present == len(expected):
                break
            time.sleep(15)
        for stage in ("cache", "measure"):
            status(stage, hardware="cps2 RTX PRO 6000; 표3 RTX5080과 분리")
            with (OUT / (stage + ".log")).open("w") as log:
                result = subprocess.run([sys.executable, "-u", "scripts/_review_ringpp_server_cost.py", stage],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                status("failed", failed_stage=stage, returncode=result.returncode,
                       log=str(OUT / (stage + ".log")))
                raise SystemExit(result.returncode)
        measured = json.loads((OUT / "measurement.json").read_text())
        assert measured["status"] == "passed" and measured["queries"] == 288
        measured["repeat_variation"]["interpretation"] = (
            "Exact grid maximum is installed. Repeated features, winners and scores are checked for equality; "
            "this server cost measurement is separate from the frozen 46-sequence evaluation.")
        measured["verification"]["scope"] = (
            "Fresh exact-maximum encodes and official correlation checks on server hardware.")
        (OUT / "measurement.json").write_text(json.dumps(measured, ensure_ascii=False, indent=2) + "\n")
        status("complete", measurement="measurement.json", timings_ms=measured["timings_ms"],
               local_table3_replacement=False)


if __name__ == "__main__":
    main()
