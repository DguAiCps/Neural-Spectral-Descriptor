#!/usr/bin/env python3
"""뒤늦게 확보되는 공개 HeLiOS archive를 전수 검증 후 실행 큐에 전달한다."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import _review_native_helios as native

REPO = native.REPO
ROOT = native.OUT
SOURCE = Path("/mnt/d/NSD_datasets/_review_native_20261105/helios/data_validation")
REMOTE = "/home/qkrrlans00/Neural-Spectral-Codec/"


def main() -> None:
    """기존 입력을 보존하고 새 공개 archive 완료 시에만 한 번 준비한다."""
    status = ROOT / "source_relay_status.json"
    history = ROOT / "source_relay_history.jsonl"
    receipt = ROOT / "source_relay_receipt.json"
    if not receipt.exists():
        initial = json.loads((ROOT / "input_manifest.json").read_text())
        native.write(receipt, {"sequences": [name for name, row in initial["sequences"].items() if row["available"]], "basis": "처음6묶음의기존전송완료마커·해시검사"})
    while True:
        before = json.loads((ROOT / "input_manifest.json").read_text())
        acknowledged = json.loads(receipt.read_text())["sequences"]
        available = [name for name in native.NAMES if (SOURCE / name / "LiDAR.tar.gz.complete").exists()]
        fresh = [name for name in available if name not in acknowledged]
        if fresh:
            stamp = str(int(time.time()))
            snapshots = ROOT / "input_manifest_history"
            snapshots.mkdir(exist_ok=True)
            shutil.copy2(ROOT / "input_manifest.json", snapshots / f"{stamp}_before.json")
            native.write(status, {"preparing": fresh, "pid": os.getpid(), "unix_time": time.time()})
            with (ROOT / f"prepare_new_{stamp}.log").open("a") as stream:
                subprocess.run(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "HELIOS_NATIVE_SOURCE=/workspace/data/_review_native_20261105/helios", "-w", "/workspace/Neural-Spectral-Codec", "ring_baseline", "python3", "-u", "scripts/_review_native_helios.py", "prepare"], stdout=stream, stderr=stream, check=True)
            after = json.loads((ROOT / "input_manifest.json").read_text())
            assert before["database_pickle_sha256"] == after["database_pickle_sha256"]
            assert before["query_pickle_sha256"] == after["query_pickle_sha256"]
            for name in fresh:
                folder = ROOT / "data/validation" / name
                # 로컬 D: 전용 입력 링크를 서버의 실제 점군으로 복사한다.
                subprocess.run(["rsync", "-azL", "--omit-dir-times", "--exclude=extracted.json", str(folder) + "/", "joms-gpu:" + REMOTE + str(folder.relative_to(REPO)) + "/"], check=True)
            transfer = ROOT / f"source_transfer_{stamp}.json"
            files = {str(p.relative_to(REPO)): native.sha(p) for name in fresh for p in (ROOT / "data/validation" / name / "LiDAR").glob("*.bin")}
            native.write(transfer, {"new_sequences": fresh, "sha256": files})
            subprocess.run(["rsync", "-az", str(transfer), "joms-gpu:" + REMOTE + str(ROOT.relative_to(REPO)) + "/"], check=True)
            code = f'''import hashlib,json
from pathlib import Path
r=Path({REMOTE!r})
d=json.loads((r/{str(transfer.relative_to(REPO))!r}).read_text())
for name,expected in d["sha256"].items():
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected,name
print("verified",len(d["sha256"]))
'''
            subprocess.run(["ssh", "joms-gpu", "python3", "-"], input=code, text=True, check=True)
            # 새 추론은 이 마커가 있어야 시작한다. 입력과 해시를 먼저 보낸다.
            for name in fresh:
                p = ROOT / "data/validation" / name / "extracted.json"
                subprocess.run(["rsync", "-az", str(p), "joms-gpu:" + REMOTE + str(p.parent.relative_to(REPO)) + "/"], check=True)
            subprocess.run(["rsync", "-az", str(ROOT / "input_manifest.json"), "joms-gpu:" + REMOTE + str(ROOT.relative_to(REPO)) + "/"], check=True)
            with history.open("a") as stream:
                stream.write(json.dumps({"transferred": fresh, "verified_files": len(files), "unix_time": time.time()}) + "\n")
            native.write(receipt, {"sequences": acknowledged + fresh, "unix_time": time.time()})
        native.write(status, {"available_archives": len(available), "required": 12, "pid": os.getpid(), "unix_time": time.time()})
        if len(available) == 12:
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
