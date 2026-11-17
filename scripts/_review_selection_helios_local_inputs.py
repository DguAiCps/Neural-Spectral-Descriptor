#!/usr/bin/env python3
"""HeLiOS 로컬 병행 검산을 위해 동일 검증 입력을 D:로 준비한다."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "src")]
from _review_selection_bev import VAL, TEST, read, write, sha
from _review_selection_full_inputs import cache

OUT = ROOT / "results/pdv2_review_selection_full_20261105"
DEST = OUT / "local_helios_inputs"
ROLE = "val"
ROSTER = VAL


def inventory() -> None:
    """기존 서버 검증 입력을 변경하지 않고 전수 해시를 기록한다."""
    DEST.mkdir(exist_ok=True)
    if ROLE == "test":
        assert (OUT / "retained/selection_helios.json").exists()
    for sensor, seq in ROSTER:
        key = f"{sensor}_{seq}"
        result = DEST / "source" / f"{key}.json"
        if result.exists():
            continue
        source = ((OUT / "inputs/helios/data/test_odom") if ROLE == "val" else
                  (ROOT / "_external/helios_official_pdv2/data/test_odom")) / key
        ids = np.load(cache(sensor, seq, ROLE))["scan_ids"]
        files = {f"LiDAR/{int(sid):06d}.bin": sha(source / f"LiDAR/{int(sid):06d}.bin") for sid in ids}
        files["index.npz"] = sha(source / "index.npz")
        write(result, dict(sequence=f"{sensor}/{seq}", frames=len(ids), files=files,
            cache_sha256=sha(cache(sensor, seq, ROLE)), source_bytes=sum((source / name).stat().st_size for name in files),
            source_path=str(source.relative_to(ROOT)), role=ROLE, completed_unix=time.time()))


def transfer() -> None:
    """원본과 모델을 바꾸지 않고 입력별 전송·해시 검산을 재개한다."""
    DEST.mkdir(exist_ok=True)
    plan = read(DEST / "plan.json")
    if ROLE == "test":
        assert sha(OUT / "retained/selection_helios.json") == plan["selection_sha256"]
    target_root = Path(plan["target_host_root"])
    assert shutil.disk_usage(target_root).free > 100_000_000_000
    remote_root = "joms-gpu:/home/qkrrlans00/Neural-Spectral-Codec/"
    while True:
        subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms",
            remote_root + str(DEST.relative_to(ROOT)) + "/source/", str(DEST) + "/source/"], check=True)
        progressed = False
        for sensor, seq in ROSTER:
            key = f"{sensor}_{seq}"
            manifest_path = DEST / "source" / f"{key}.json"
            proof_path = DEST / "verified" / f"{key}.json"
            if not manifest_path.exists() or proof_path.exists():
                continue
            manifest = read(manifest_path)
            assert manifest["cache_sha256"] == sha(cache(sensor, seq, ROLE))
            target = target_root / "test_odom" / key
            target.mkdir(parents=True, exist_ok=True)
            write(DEST / "state.json", dict(status="copying", sequence=f"{sensor}/{seq}",
                pid=os.getpid(), updated_unix=time.time()))
            subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms", "--partial",
                "--compress", "--compress-choice=zstd", "--compress-level=1",
                remote_root + manifest["source_path"] + "/", str(target) + "/"], check=True)
            write(DEST / "state.json", dict(status="verifying", sequence=f"{sensor}/{seq}",
                pid=os.getpid(), updated_unix=time.time()))
            for filename, digest in manifest["files"].items():
                assert sha(target / filename) == digest, (key, filename)
            write(proof_path, dict(all_files_SHA_exact=True, frames=manifest["frames"],
                source_manifest_sha256=sha(manifest_path), cache_sha256=manifest["cache_sha256"],
                target=str(target), completed_unix=time.time()))
            progressed = True
        if len(list((DEST / "verified").glob("*.json"))) == len(ROSTER):
            write(DEST / "complete.json", dict(sequences=len(ROSTER), all_files_SHA_exact=True,
                candidate_assignment_changed=False, plan_sha256=sha(DEST / "plan.json"), completed_unix=time.time()))
            write(DEST / "state.json", dict(status="complete", completed_unix=time.time()))
            return
        if not progressed:
            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "transfer"))
    parser.add_argument("--role", choices=("val", "test"), default="val")
    args = parser.parse_args()
    ROLE = args.role
    if ROLE == "test":
        DEST = OUT / "local_helios_test_inputs"
        ROSTER = TEST
    inventory() if args.action == "inventory" else transfer()
