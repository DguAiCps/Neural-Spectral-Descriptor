#!/usr/bin/env python3
"""SC++ Recall@{1,5,10} on the NCLT evaluation sequences with z-up loading.

Same loading and protocol as scripts/_verify_nclt_zfix_baselines.py (release
keyframe scan_ids, 5 m / 30-frame protocol, SC++'s own prefilter+rerank
pipeline), with k_values=[1,5,10] for the tab:rankk row. Run inside the pyg
container (mounts: repo at /workspace/Neural-Spectral-Codec, data at /data).
Output: results/_alias_source/scpp_rankk_nclt_zup.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
NCLT_ROOT = Path("/data/nclt")
OUT = REPO / "results/_alias_source/scpp_rankk_nclt_zup.json"
NCLT_DT = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"),
                    ("intensity", "u1"), ("padding", "u1"), ("extra", "<u4")])


def load_scan(path):
    raw = np.fromfile(path, dtype=NCLT_DT)
    x = raw["x"].astype(np.float32) * 0.005 - 100.0
    y = raw["y"].astype(np.float32) * 0.005 - 100.0
    z = -(raw["z"].astype(np.float32) * 0.005 - 100.0)  # z-up
    pts = np.column_stack([x, y, z, raw["intensity"].astype(np.float32) / 255.0])
    m = np.isfinite(pts[:, :3]).all(axis=1) & (np.abs(pts[:, :3]) < 200.0).all(axis=1)
    return pts[m]


def main():
    from baselines.scan_context import ScanContextPP
    out = json.load(open(OUT)) if OUT.exists() else {}
    for seq in (sys.argv[1:] or ["2013-01-10", "2012-01-08"]):
        if seq in out:
            continue
        z = np.load(CACHE / f"nclt_operating_{seq}_layout60_stride1.npz")
        poses = z["poses"]
        sel = z["scan_ids"].astype(int)
        files = sorted((NCLT_ROOT / seq / "velodyne_sync").glob("*.bin"))
        scans = [load_scan(files[s]) for s in sel]
        recalls, nq = ScanContextPP().compute_recalls(
            scans, poses, k_values=[1, 5, 10],
            distance_threshold=5.0, skip_frames=30)
        out[seq] = {"n_queries": int(nq),
                    **{f"r{k}": float(v) for k, v in recalls.items()}}
        print(seq, out[seq], flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
