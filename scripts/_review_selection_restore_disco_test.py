#!/usr/bin/env python3
"""DiSCO 선택 동결 뒤 보존 시험 polar와 원래 원장을 무손실 복구한다."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REL = Path("results/pdv2_review_selection_full_20261105")
OUT = ROOT / REL
REMOTE = "/home/qkrrlans00/Neural-Spectral-Codec"


def read(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(Path(path).read_text())


def write(path: Path, value: dict) -> None:
    """완성한 JSON만 원자적으로 공개한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def sha(path: Path) -> str:
    """파일 SHA256을 반환한다."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest() -> None:
    """서버의 보존 입력·원장 해시를 기록한다."""
    plan = read(OUT / "disco_test_input_plan.json")
    rows = []
    for row in plan["files"]:
        source = ROOT / row["source"]
        rows.append(dict(row, sha256=sha(source), size=source.stat().st_size))
    write(OUT / "disco_test_input_manifest.json", dict(plan_sha256=sha(OUT / "disco_test_input_plan.json"),
                                                       files=rows, total_bytes=sum(r["size"] for r in rows)))


def restore() -> None:
    """선택 동결 뒤 입력과 원래 원장을 복구하고 해시를 검산한다."""
    lock = (OUT / "disco_test_input_restore.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = OUT / "disco_test_input_restore_state.json"
    if state.exists() and read(state).get("status") == "complete":
        return
    selection = OUT / "retained/selection_disco.json"
    while not selection.exists():
        write(state, dict(status="waiting_selection_freeze", pid=os.getpid(), updated_unix=time.time()))
        time.sleep(20)
    frozen = read(selection)
    assert not frozen["test_used_for_selection"]
    sys.path.insert(0, str(ROOT / "scripts"))
    from _pdv2_pilot import TEST
    target = Path("/mnt/d/NSD_datasets/review_selection_cache_20261105/inputs/test_disco")
    target.mkdir(parents=True, exist_ok=True)
    alias = OUT / "inputs/test_disco"
    if not alias.exists():
        alias.symlink_to(target, target_is_directory=True)
    assert alias.resolve() == target
    files = []
    for sensor, seq in TEST:
        for name in ("bits.npy", "nhits.npy", "index.npz", "meta.json"):
            source = Path("_external/disco_official_pdv2/polar/test_odom") / f"{sensor}_{seq}" / name
            dest = REL / "inputs/test_disco/polar/test_odom" / f"{sensor}_{seq}" / name
            files.append(dict(source=str(source), destination=str(dest), kind="polar"))
    for info in frozen["original_choices"].values():
        parent = Path(info["selection_source"]).parent
        for sensor, seq in TEST:
            source = parent / "odom" / f"dump_disco_{sensor}_{seq}.npz"
            files.append(dict(source=str(source), destination=str(source), kind="historical_ledger"))
    plan = OUT / "disco_test_input_plan.json"
    write(plan, dict(created_unix=time.time(), selection_sha256=sha(selection), test_used_for_selection=False, files=files))
    write(state, dict(status="server_manifest", pid=os.getpid(), updated_unix=time.time()))
    subprocess.run(["rsync", "-a", str(Path(__file__)), "joms-gpu:" + REMOTE + "/scripts/"], check=True)
    subprocess.run(["rsync", "-a", str(plan), "joms-gpu:" + REMOTE + "/" + str(REL) + "/"], check=True)
    subprocess.run(["ssh", "joms-gpu", "python3", REMOTE + "/scripts/_review_selection_restore_disco_test.py", "manifest"], check=True)
    subprocess.run(["rsync", "-a", "joms-gpu:" + REMOTE + "/" + str(REL / "disco_test_input_manifest.json"), str(OUT) + "/"], check=True)
    source_manifest = read(OUT / "disco_test_input_manifest.json")
    assert source_manifest["plan_sha256"] == sha(plan)
    for row in source_manifest["files"]:
        dest = ROOT / row["destination"]
        if row["kind"] == "historical_ledger" and dest.exists():
            assert sha(dest) == row["sha256"], (str(dest), "existing historical ledger differs; preserved")
    write(state, dict(status="downloading", pid=os.getpid(), total_bytes=source_manifest["total_bytes"], updated_unix=time.time()))
    # 파일 목록으로 복사 범위를 한정한다. 원본의 위치·내용은 바꾸지 않는다.
    for kind in ("polar", "historical_ledger"):
        listing = OUT / f"disco_test_transfer_{kind}.txt"
        prefix = Path("_external/disco_official_pdv2/polar") if kind == "polar" else Path(".")
        names = [str(Path(row["source"]).relative_to(prefix)) if kind == "polar" else row["source"]
                 for row in files if row["kind"] == kind]
        listing.write_text("\n".join(names) + "\n")
        dest = target / "polar" if kind == "polar" else ROOT
        dest.mkdir(parents=True, exist_ok=True)
        source = "joms-gpu:" + REMOTE + "/" + (str(prefix) + "/" if kind == "polar" else "")
        subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms", "--checksum", "--bwlimit=10000",
                        "--files-from=" + str(listing), source, str(dest) + "/"], check=True)
    for row in source_manifest["files"]:
        dest = ROOT / row["destination"]
        assert sha(dest) == row["sha256"]
        with dest.open("rb") as stream:
            os.fsync(stream.fileno())
    write(state, dict(status="complete", all_files_SHA_exact=True, files=len(files),
                     selection_sha256=sha(selection), manifest_sha256=sha(OUT / "disco_test_input_manifest.json"),
                     total_bytes=source_manifest["total_bytes"], completed_unix=time.time(), GPU_used=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("restore", "manifest"))
    action = parser.parse_args().action
    restore() if action == "restore" else manifest()
