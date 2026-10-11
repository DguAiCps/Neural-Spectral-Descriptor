#!/usr/bin/env python3
"""Export split-v2 test keyframe clouds for the OFFICIAL Scan Context++ MATLAB
code (scripts/scpp_official/*.m, run on the host -- MATLAB is not in the
container, the repo loaders are not on the host).

Runs inside docker ring_baseline. Points are written exactly as the repo
loader returns them (xyz only, float32, no filtering / height offset /
downsampling -- those happen on the MATLAB side in the official code path),
so the MATLAB reader sees what tro2022/reader/readBin.m would see.

  --info            {work}/seq_info.json: operating cache, sha256, n_frames
  --chunk S E       keyframes [S, E) ->
                      {work}/chunk_{S:06d}.xyz  float32 row-major (sum n_i, 3)
                      {work}/chunk_{S:06d}.cnt  int64 (E-S,) points per keyframe
                    written under .part names and renamed (MATLAB never sees a
                    half-written chunk).

Keyframes = scan_ids of the split-v2 operating cache of --cond (odom: KISS-ICP
keyframes, data/preprocessed_v2_odom), the same source as
scripts/_pdv2_c3.py cmd_scpp_dump.

  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 \
    scripts/scpp_official/export_keyframes.py --seq kitti/07 --work W --info
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seq", required=True, metavar="SENSOR/SEQ")
    ap.add_argument("--cond", default="odom", choices=("odom", "gt"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--chunk", type=int, nargs=2, metavar=("START", "END"))
    a = ap.parse_args()
    from _pdv2_c3 import TEST, _test_cache  # split-v2 roster + canonical cache
    sensor, q = a.seq.split("/")
    assert (sensor, q) in TEST, a.seq
    cache = _test_cache(a.cond, sensor, q)
    with np.load(cache) as c:
        scan_ids = c["scan_ids"].astype(np.int64)
    n = len(scan_ids)
    work = Path(a.work)
    assert work.is_dir(), f"create {work} on the host first"

    if a.info:
        info = {"sensor": sensor, "sequence": q, "cond": a.cond,
                "n_frames": n,
                "operating_cache": str(cache.relative_to(REPO)),
                "operating_cache_sha256": _sha256(cache)}
        (work / "seq_info.json").write_text(json.dumps(info, indent=1))
        print(f"INFO {a.seq} n_frames={n}", flush=True)
    if a.chunk:
        from _pdb_scpp_eval import _make_loader  # repo loader wiring, 6 sensors
        s, e = a.chunk
        e = min(e, n)
        assert 0 <= s < e, (s, e, n)
        loader = _make_loader(sensor, q)
        base = work / f"chunk_{s:06d}"
        counts = np.zeros(e - s, dtype=np.int64)
        with open(f"{base}.xyz.part", "wb") as f:
            for k, i in enumerate(range(s, e)):
                pts = loader[int(scan_ids[i])]["points"]
                xyz = np.ascontiguousarray(pts[:, :3], dtype=np.float32)
                counts[k] = len(xyz)
                f.write(xyz.tobytes())
        counts.tofile(f"{base}.cnt.part")
        os.replace(f"{base}.xyz.part", f"{base}.xyz")
        os.replace(f"{base}.cnt.part", f"{base}.cnt")
        print(f"CHUNK {a.seq} [{s},{e}) points={int(counts.sum())}", flush=True)


if __name__ == "__main__":
    main()
