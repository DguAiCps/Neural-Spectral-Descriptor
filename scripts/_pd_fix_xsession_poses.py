#!/usr/bin/env python3
"""Rewrite cross-session key dumps with session-consistent GLOBAL positions.

The pd2026 operating caches re-center every session's poses on its own first
scan, so cross-session ground-truth distances computed from them are invalid
(e.g. MulRan KAIST01 vs KAIST03 frames are offset by ~245 m -> R@1 0.000).
This script replaces poses[:, :3, 3] in results/pd2026/keys_* npz files with
positions interpolated from the datasets' global ground truth:

  mulran: <root>/<seq>/global_pose.csv  (stamp ns; UTM tx,ty,tz at cols 4,8,12)
          matched via cache timestamps (epoch seconds * 1e9)
  nclt:   <root>/<date>/groundtruth_<date>.csv (utime us; campus-frame x,y,z)
          matched via scan_ids -> sorted velodyne_sync filenames (utime us)

Outputs keys_*_gp.npz next to the originals. HeLiPR Town is NOT rewritten:
its per-sequence GT has no shared global frame in this checkout (sessions
start at the same spot, so the local frames approximately coincide; the
helipr_town numbers carry that caveat instead).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/d/NSD_datasets")
KEYS = REPO / "results/pd2026"
PD_CACHE = REPO / "data/preprocessed_pd2026"

MULRAN_SEQS = [f"{site}{i:02d}" for site in ("DCC", "KAIST", "Riverside") for i in (1, 2, 3)]
NCLT_SEQS = ["2012-02-04", "2012-03-17"]


def cache_for(sensor: str, seq: str) -> np.lib.npyio.NpzFile:
    hits = sorted(glob.glob(str(PD_CACHE / f"cache_*_{sensor}_test_{seq}.npz")))
    assert len(hits) == 1, (sensor, seq, hits)
    return np.load(hits[0])


def mulran_positions(seq: str, cache) -> np.ndarray:
    gt = np.loadtxt(DATA / "mulran" / seq / "global_pose.csv", delimiter=",")
    t_gt = gt[:, 0]  # ns
    pos_gt = gt[:, [4, 8, 12]]
    t = cache["timestamps"] * 1e9
    assert t_gt[0] - 5e9 <= t[0] and t[-1] <= t_gt[-1] + 5e9, (seq, t[0], t_gt[0])
    return np.stack([np.interp(t, t_gt, pos_gt[:, i]) for i in range(3)], axis=1)


def nclt_positions(seq: str, cache) -> np.ndarray:
    files = sorted((DATA / "nclt" / seq / "velodyne_sync").glob("*.bin"))
    utimes = np.array([int(f.stem) for f in files], dtype=np.float64)
    scan_utimes = utimes[cache["scan_ids"].astype(int)]
    gt = np.loadtxt(DATA / "nclt" / seq / f"groundtruth_{seq}.csv", delimiter=",")
    gt = gt[np.isfinite(gt[:, 1])]
    return np.stack(
        [np.interp(scan_utimes, gt[:, 0], gt[:, 1 + i]) for i in range(3)], axis=1
    )


def main() -> None:
    jobs = [("mulran", s, mulran_positions) for s in MULRAN_SEQS] + [
        ("nclt", s, nclt_positions) for s in NCLT_SEQS
    ]
    for sensor, seq, fn in jobs:
        cache = cache_for(sensor, seq)
        pos = fn(seq, cache)
        n_fixed = 0
        for kf in sorted(KEYS.glob(f"keys_*_{sensor}_{seq}.npz")):
            if kf.stem.endswith("_gp"):
                continue
            d = dict(np.load(kf, allow_pickle=True))
            assert len(d["poses"]) == len(pos), (kf.name, len(d["poses"]), len(pos))
            poses = d["poses"].copy()
            poses[:, :3, 3] = pos
            d["poses"] = poses
            np.savez(kf.with_name(kf.stem + "_gp.npz"), **d)
            n_fixed += 1
        span = pos.max(0) - pos.min(0)
        print(f"{sensor}/{seq}: {n_fixed} keys files -> _gp, "
              f"span x={span[0]:.0f} y={span[1]:.0f}, first={pos[0].round(1)}")


if __name__ == "__main__":
    main()
