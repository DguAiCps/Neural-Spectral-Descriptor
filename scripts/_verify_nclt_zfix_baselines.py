#!/usr/bin/env python3
"""Re-evaluate the analytic baselines on NCLT with the z-axis corrected.

The NCLT loader keeps body-frame z DOWN, which inverts height semantics for
every height-based analytic baseline (SC++ 13-01: 0.105 release -> 0.928 with
z flipped). This script re-runs SC++, M2DP, FreSCo, and LiDAR-Iris on the
exact evaluation keyframes with z flipped up, reusing each baseline's own
release pipeline (compute_recalls: prefilter + rerank, 5 m / 30-frame
protocol), and records release-convention runs for validation.

Run: docker run --rm -v $REPO:/ws -v /mnt/d/NSD_datasets:/data -w /ws \
       nvcr.io/nvidia/pyg:26.01-py3 python scripts/_verify_nclt_zfix_baselines.py \
       2013-01-10 [2012-01-08]
Output: results/_alias_source/nclt_zfix_baselines.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
NCLT_ROOT = Path("/data/nclt")
OUT = REPO / "results/_alias_source"
NCLT_DT = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"),
                    ("intensity", "u1"), ("padding", "u1"), ("extra", "<u4")])


def load_scan(path, flip):
    raw = np.fromfile(path, dtype=NCLT_DT)
    x = raw["x"].astype(np.float32) * 0.005 - 100.0
    y = raw["y"].astype(np.float32) * 0.005 - 100.0
    z = raw["z"].astype(np.float32) * 0.005 - 100.0
    if flip:
        z = -z
    pts = np.column_stack([x, y, z, raw["intensity"].astype(np.float32) / 255.0])
    m = np.isfinite(pts[:, :3]).all(axis=1) & (np.abs(pts[:, :3]) < 200.0).all(axis=1)
    return pts[m]


def make_baselines():
    from baselines.scan_context import ScanContextPP
    from baselines.m2dp import M2DP
    from baselines.fresco import FreSCo
    from baselines.lidar_iris import LiDARIris
    return {"sc++": ScanContextPP(), "m2dp": M2DP(), "fresco": FreSCo(),
            "lidar_iris": LiDARIris()}


def main():
    seqs = sys.argv[1:] or ["2013-01-10"]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "nclt_zfix_baselines.json"
    out = json.load(open(out_path)) if out_path.exists() else {}
    for seq in seqs:
        z = np.load(CACHE / f"nclt_operating_{seq}_layout60_stride1.npz")
        poses = z["poses"]
        sel = z["scan_ids"].astype(int)
        files = sorted((NCLT_ROOT / seq / "velodyne_sync").glob("*.bin"))
        out.setdefault(seq, {})
        for flip in (False, True):
            scans = [load_scan(files[s], flip) for s in sel]
            for name, enc in make_baselines().items():
                key = f"{name}_{'zfix' if flip else 'release'}"
                if key in out[seq]:
                    continue
                try:
                    recalls, _nq = enc.compute_recalls(scans, poses, k_values=[1],
                                                       distance_threshold=5.0, skip_frames=30)
                    r1 = float(recalls[1])
                except Exception as e:
                    r1 = None
                    print(f"  {key}: FAILED {e}", flush=True)
                out[seq][key] = r1
                print(f"  [{seq}] {key:22} R@1 = {r1}", flush=True)
                json.dump(out, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
