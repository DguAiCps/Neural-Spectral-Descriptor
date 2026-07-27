#!/usr/bin/env python3
"""SC++ parameter-sensitivity sweep on NCLT (baseline-fairness check).

The paper's aggregate margin over SC++ is carried by NCLT, so this script asks
whether SC++'s NCLT collapse (13-01: 0.105, 12-01: 0.331) is a parameter
artifact. It re-runs SC++ end-to-end on the exact evaluation keyframes (matched
by velodyne timestamp to the operating cache) under a grid of build parameters:

  max_range in {80 (paper default), 40}  -- 40 doubles radial resolution where
                                            NCLT's campus content actually is
  n_rings   in {20 (default), 40}
  z_min     in {-3.0 (default), -1.0}

Retrieval protocol identical to the release baseline harness: ring-key cosine
prefilter (top-200) -> full SC column-shift rerank; queries/candidates as in
Table 1 (<5 m, >=30-frame gap, |i-j|>=30).

Run: docker run --rm -v $REPO:/ws -v /mnt/d/NSD_datasets:/data -w /ws \
       nvcr.io/nvidia/pyg:26.01-py3 python scripts/_verify_scpp_nclt_sweep.py [seq]
Output: results/_alias_source/scpp_nclt_sweep.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from baselines.scan_context import build_scan_context, _ring_key, _distance_sc_columnwise  # noqa: E402

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
OUT = REPO / "results/_alias_source"
NCLT_ROOT = Path("/data/nclt")
POS_M, SKIP, N_COARSE = 5.0, 30, 200
# flip=False replicates the release convention (NCLT body frame, z DOWN --
# structures at negative z); flip=True corrects height semantics (z up).
GRID = [
    {"flip": False, "n_rings": 20, "n_sectors": 60, "max_range": 80.0, "z_min": -3.0},  # release
    {"flip": True,  "n_rings": 20, "n_sectors": 60, "max_range": 80.0, "z_min": -3.0},
    {"flip": True,  "n_rings": 20, "n_sectors": 60, "max_range": 40.0, "z_min": -3.0},
    {"flip": True,  "n_rings": 40, "n_sectors": 60, "max_range": 80.0, "z_min": -3.0},
    {"flip": False, "n_rings": 20, "n_sectors": 60, "max_range": 40.0, "z_min": -3.0},
    {"flip": True,  "n_rings": 20, "n_sectors": 60, "max_range": 80.0, "z_min": -1.0},
]


def load_nclt_scan(path):
    """Release-convention NCLT parsing (12 bytes/point, no z flip):
    matches data/nclt_loader.py exactly."""
    nclt_dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('intensity', 'u1'), ('laser', 'u1')])  # official 8-byte NCLT hit
    raw = np.fromfile(path, dtype=nclt_dtype)
    x = raw["x"].astype(np.float32) * 0.005 - 100.0
    y = raw["y"].astype(np.float32) * 0.005 - 100.0
    z = raw["z"].astype(np.float32) * 0.005 - 100.0
    pts = np.column_stack([x, y, z])
    m = np.isfinite(pts).all(axis=1) & (np.abs(pts) < 200.0).all(axis=1)
    return pts[m]


def find_queries(pos):
    q = []
    for j in range(SKIP, len(pos)):
        d = np.linalg.norm(pos[: j - SKIP + 1] - pos[j], axis=1)
        if (d < POS_M).any():
            q.append(j)
    return np.asarray(q, dtype=int)


def eval_config(scs, rks, pos, queries):
    idx = np.arange(len(pos))
    miss = 0
    for j in queries:
        cand = np.where(np.abs(idx - j) >= SKIP)[0]
        sims = rks[cand] @ rks[j]
        coarse = cand[np.argsort(-sims)[:N_COARSE]]
        d = np.array([_distance_sc_columnwise(scs[j], scs[c]) for c in coarse])
        t1 = coarse[int(np.argmin(d))]
        if np.linalg.norm(pos[t1] - pos[j]) >= POS_M:
            miss += 1
    return 1.0 - miss / max(len(queries), 1)


def main():
    seqs = sys.argv[1:] or ["2013-01-10"]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "scpp_nclt_sweep.json"
    out = json.load(open(out_path)) if out_path.exists() else {}
    for seq in seqs:
        z = np.load(CACHE / f"nclt_operating_{seq}_layout60_stride1.npz")
        pos = z["poses"][:, :3, 3]
        # cache scan_ids index the loader's sorted velodyne_sync file list
        # (cache timestamps are sequence-relative seconds, unusable for matching)
        sel = z["scan_ids"].astype(int)
        files = sorted((NCLT_ROOT / seq / "velodyne_sync").glob("*.bin"))
        assert sel.max() < len(files), f"scan_ids exceed file count ({sel.max()} vs {len(files)})"
        print(f"[{seq}] {len(sel)} keyframes indexed into {len(files)} velodyne files", flush=True)

        scans = [load_nclt_scan(files[s]) for s in sel]
        queries = find_queries(pos)
        out.setdefault(seq, {"n": int(len(sel)), "n_q": int(len(queries)), "grid": {}})
        for cfg in GRID:
            key = (f"{'zflip' if cfg['flip'] else 'release'}_r{cfg['n_rings']}"
                   f"_m{int(cfg['max_range'])}_z{cfg['z_min']}")
            pts_iter = ([p * np.array([1, 1, -1], dtype=np.float32) for p in scans]
                        if cfg["flip"] else scans)
            scs = np.stack([build_scan_context(p, cfg["n_rings"], cfg["n_sectors"],
                                               cfg["max_range"], cfg["z_min"]) for p in pts_iter])
            rk = np.stack([_ring_key(s) for s in scs])
            rk = rk / np.clip(np.linalg.norm(rk, axis=1, keepdims=True), 1e-8, None)
            r1 = eval_config(scs, rk, pos, queries)
            out[seq]["grid"][key] = r1
            print(f"  {key:24} R@1 = {r1:.4f}", flush=True)
        json.dump(out, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
