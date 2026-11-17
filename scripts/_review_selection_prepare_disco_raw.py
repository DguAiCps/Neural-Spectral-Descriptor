#!/usr/bin/env python3
"""보존 DiSCO polar의 누락 행에 필요한 원시 입력만 D:에 무손실 복구한다."""
from __future__ import annotations

import argparse
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
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    """JSON 기록을 원자적으로 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def sha(path: Path) -> str:
    """파일 SHA256을 반환한다."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest() -> None:
    """서버의 보존 무손실 파일을 읽기 전용으로 해시한다."""
    plan = read(OUT / "disco_raw_plan.json")
    rows = []
    for name in plan["files"]:
        path = OUT / "inputs/raw" / name
        rows.append(dict(path=name, sha256=sha(path), size=path.stat().st_size))
    write(OUT / "disco_raw_server_manifest.json", dict(plan_sha256=sha(OUT / "disco_raw_plan.json"),
                                                       files=rows, total_bytes=sum(r["size"] for r in rows)))


def source_audit() -> None:
    """원래 로더와 각 시퀀스의 초반·중간·후반 원시 배열을 대조한다."""
    assert not os.environ.get("CUDA_VISIBLE_DEVICES")
    import numpy as np
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]
    from _pd_build_test_caches import _make_loader
    plan = read(OUT / "disco_raw_plan.json")
    cases = []
    for key, names in plan["by_sequence"].items():
        if not names:
            continue
        sensor, seq = key.split("/")
        loader = _make_loader(sensor, seq)
        for index in sorted({0, len(names) // 4, len(names) // 2, len(names) - 1}):
            name = names[index]
            sid = int(Path(name).stem)
            original = loader[sid]
            with np.load(OUT / "inputs/disco_raw" / name) as z:
                np.testing.assert_array_equal(z["points"], original["points"])
                np.testing.assert_array_equal(z["pose"], original["pose"])
            cases.append(dict(path=name, raw_id=sid))
    write(OUT / "disco_raw_source_audit.json", dict(all_arrays_exact=True, cases=cases,
                                                  plan_sha256=sha(OUT / "disco_raw_plan.json")))


def prepare() -> None:
    """누락 polar 행만 복사하고 전수 파일 SHA와 원래 로더 표본을 검산한다."""
    import numpy as np
    state = OUT / "disco_raw_restore_state.json"
    target = Path("/mnt/d/NSD_datasets/review_selection_cache_20261105/inputs/disco_raw")
    target.mkdir(parents=True, exist_ok=True)
    alias = OUT / "inputs/disco_raw"
    if not alias.exists():
        alias.symlink_to(target, target_is_directory=True)
    assert alias.resolve() == target
    by_sequence = {}
    for row in read(OUT / "input_relay_plan.json")["rows"]:
        sensor, seq = row["sensor"], row["seq"]
        polar = OUT / "inputs/disco/polar/test_odom" / f"{sensor}_{seq}"
        meta = read(polar / "meta.json")
        todo = np.array([], np.int64) if meta.get("complete") else np.flatnonzero(~np.load(polar / "done.npy"))
        by_sequence[sensor + "/" + seq] = [f"{sensor}_{seq}/{row['scan_ids'][int(i)]:06d}.npz" for i in todo]
    plan_path = OUT / "disco_raw_plan.json"
    if not plan_path.exists():
        write(plan_path, dict(created_unix=time.time(), GPU_started=False, by_sequence=by_sequence,
                             files=[name for names in by_sequence.values() for name in names]))
    plan = read(plan_path)
    write(state, dict(status="server_manifest", pid=os.getpid(), files=len(plan["files"]), updated_unix=time.time()))
    subprocess.run(["rsync", "-a", str(Path(__file__)), "joms-gpu:" + REMOTE + "/scripts/"], check=True)
    subprocess.run(["rsync", "-a", str(plan_path), "joms-gpu:" + REMOTE + "/" + str(REL) + "/"], check=True)
    subprocess.run(["ssh", "joms-gpu", "python3", REMOTE + "/scripts/_review_selection_prepare_disco_raw.py", "manifest"], check=True)
    subprocess.run(["rsync", "-a", "joms-gpu:" + REMOTE + "/" + str(REL / "disco_raw_server_manifest.json"), str(OUT) + "/"], check=True)
    manifest = read(OUT / "disco_raw_server_manifest.json")
    assert manifest["plan_sha256"] == sha(plan_path)
    listing = OUT / "disco_raw_transfer_files.txt"
    listing.write_text("\n".join(plan["files"]) + "\n")
    write(state, dict(status="downloading", pid=os.getpid(), files=len(plan["files"]),
                     total_bytes=manifest["total_bytes"], target=str(target), updated_unix=time.time()))
    subprocess.run(["rsync", "-aO", "--no-owner", "--no-group", "--no-perms", "--checksum", "--bwlimit=50000",
                    "--files-from=" + str(listing), "joms-gpu:" + REMOTE + "/" + str(REL / "inputs/raw") + "/", str(target) + "/"], check=True)
    for i, row in enumerate(manifest["files"]):
        path = alias / row["path"]
        assert sha(path) == row["sha256"]
        with np.load(path) as z:
            assert z["points"].dtype == np.float32 and z["points"].shape[1] == 4
            assert np.isfinite(z["points"]).all() and np.isfinite(z["pose"]).all()
        if i % 256 == 0:
            write(state, dict(status="verifying", pid=os.getpid(), completed=i + 1,
                             total=len(manifest["files"]), updated_unix=time.time()))
    subprocess.run(["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "OMP_NUM_THREADS=1",
                    "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-w", "/workspace/Neural-Spectral-Codec",
                    "ring_baseline", "python3", "-u", "scripts/_review_selection_prepare_disco_raw.py", "source-audit"], check=True)
    proof = read(OUT / "disco_raw_source_audit.json")
    assert proof["all_arrays_exact"]
    write(state, dict(status="complete", all_files_exact=True, source_samples_exact=True,
                     files=len(manifest["files"]), total_bytes=manifest["total_bytes"], target=str(target),
                     manifest_sha256=sha(OUT / "disco_raw_server_manifest.json"),
                     source_audit_sha256=sha(OUT / "disco_raw_source_audit.json"), GPU_started=False,
                     completed_unix=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "manifest", "source-audit"))
    action = parser.parse_args().action
    {"prepare": prepare, "manifest": manifest, "source-audit": source_audit}[action]()
