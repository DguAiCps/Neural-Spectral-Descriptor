#!/usr/bin/env python3
"""Sensor height above the ground, measured from our sensor-frame scans.

Used ONLY to transfer the official RING/RING++ per-dataset height band
(_external/RING/datasets/*Dataset.py) to datasets the official code has no
setting for (KITTI-360, HeLiPR VLP-16, MCD handheld OS1-64), and to check
that the official KITTI / MulRan / NCLT constants are consistent with our
loaders' frames.  CPU only, numpy only (pandas for the NCLT loader).

Method, per scan (points as returned by our src/data loaders: sensor frame,
z up; NCLT already z-flipped by the loader):
  1. ring of points 3 m <= hypot(x, y) <= 30 m with -4 m < z < 0;
  2. coarse ground z0, two estimators:
       A "mode":   mode of a 5 cm z-histogram of the ring (r <= 20 m);
       B "lowest": median over 72 azimuth sectors (5 deg) of the per-sector
                   minimum z (the lowest surface seen in each direction).
     A fails when few beams look down (NCLT flipped HDL-32E: lowest beam
     about -10.7 deg; VLP-16: -15 deg): the near-horizontal beams then
     dominate the histogram.  B is the primary estimate;
  3. least-squares plane z = a x + b y + c through ring points within
     +-0.15 m of z0; height = |c| / sqrt(1 + a^2 + b^2) (perpendicular
     distance of the sensor origin from the plane), tilt = atan(hypot(a, b)).
Scans whose plane uses fewer than 100 points are skipped.  Per sequence:
median / p10 / p90 over the sampled scans (evenly spaced loader indices).

Output: results/pdv2_ring_official/sensor_height.json

Run inside docker ring_baseline (one CPU process):
  python3 scripts/_ring_official_sensor_height.py --n-scans 20
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", "/workspace/data"))
OUT = REPO / "results/pdv2_ring_official/sensor_height.json"

# Train / validation sequences of split v2 only (scripts/_pdv2_pilot.py), so
# no test scan feeds a setting.  MCD has no train/val sequence: its four
# rows are test scans, geometry only (no poses, labels or scores read).
# NCLT is not measured: the official NCLT constants apply to our frame
# directly, and the flipped HDL-32E sees the ground with almost no beams.
SEQS = [("kitti", "01"), ("kitti", "05"), ("kitti", "09"),
        ("kitti360", "0004"), ("kitti360", "0006"),
        ("mulran", "Sejong01"), ("mulran", "Sejong02"), ("mulran", "Sejong03"),
        ("helipr", "Bridge01"), ("helipr", "Bridge03"),
        ("helipr", "Roundabout01"), ("helipr", "Roundabout02"),
        ("mcd", "kth_day_06"), ("mcd", "kth_night_01"),
        ("mcd", "tuhh_day_02"), ("mcd", "tuhh_night_08")]


def make_loader(sensor, seq):
    """Same constructors as scripts/_pd_build_test_caches._make_loader."""
    if sensor == "kitti":
        from data.kitti_loader import KITTILoader
        return KITTILoader(str(DATA_ROOT / "kitti/dataset"), seq, lazy_load=True)
    if sensor == "nclt":
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(str(DATA_ROOT / "nclt"), seq, lazy_load=True)
    if sensor == "helipr":
        from data.helipr_loader import HeLiPRLoader
        p = DATA_ROOT / "helipr" / seq
        if not (p / "LiDAR").is_dir():
            p = p / seq  # legacy double-nested layout
        return HeLiPRLoader(str(p), lazy_load=True)
    if sensor == "mulran":
        from data.mulran_loader import MulRanLoader
        return MulRanLoader(str(DATA_ROOT / "mulran"), seq, lazy_load=True)
    if sensor == "kitti360":
        from data.kitti360_loader import Kitti360Loader
        return Kitti360Loader(str(DATA_ROOT / "kitti360"), seq, lazy_load=True)
    if sensor == "mcd":
        from data.mcd_loader import MCDLoader
        return MCDLoader(str(DATA_ROOT / "mcd_extracted"), seq, lazy_load=True)
    raise ValueError(sensor)


def _plane(ring, z0):
    sel = ring[np.abs(ring[:, 2] - z0) <= 0.15]
    if len(sel) < 100:
        return None
    A = np.c_[sel[:, 0], sel[:, 1], np.ones(len(sel))]
    (a, b, c), *_ = np.linalg.lstsq(A, sel[:, 2], rcond=None)
    h = abs(c) / np.sqrt(1.0 + a * a + b * b)
    return float(h), float(np.degrees(np.arctan(np.hypot(a, b))))


def ground_height(pts: np.ndarray):
    """{'lowest': (h, tilt) | None, 'mode': (h, tilt) | None} (docstring)."""
    p = pts[:, :3].astype(np.float64)
    p = p[np.isfinite(p).all(axis=1)]
    r = np.hypot(p[:, 0], p[:, 1])
    ring = p[(r >= 3.0) & (r <= 30.0) & (p[:, 2] > -4.0) & (p[:, 2] < 0.0)]
    out = {"lowest": None, "mode": None}
    if len(ring) < 100:
        return out
    # B: lowest surface per azimuth sector, median over sectors
    sec = ((np.arctan2(ring[:, 1], ring[:, 0]) + np.pi) / (2 * np.pi) * 72
           ).astype(int) % 72
    mins = np.full(72, np.nan)
    np.fmin.at(mins, sec, ring[:, 2])
    if np.isfinite(mins).sum() >= 18:
        out["lowest"] = _plane(ring, float(np.nanmedian(mins)))
    # A: histogram mode (r <= 20 m)
    rr = np.hypot(ring[:, 0], ring[:, 1])
    near = ring[rr <= 20.0]
    if len(near) >= 100:
        hist, edges = np.histogram(near[:, 2],
                                   bins=np.arange(-4.0, 0.0001, 0.05))
        k = int(np.argmax(hist))
        out["mode"] = _plane(near, 0.5 * (edges[k] + edges[k + 1]))
    return out


def _stats(v):
    v = np.asarray(v, dtype=np.float64)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-scans", type=int, default=20)
    ap.add_argument("--seqs", nargs="*", default=None, metavar="SENSOR/SEQ")
    args = ap.parse_args()
    seqs = SEQS if not args.seqs else [tuple(s.split("/")) for s in args.seqs]
    out = {"_meta": {
        "method": ("per scan: ring 3-30 m, -4 < z < 0; coarse ground = "
                   "median over 72 azimuth sectors of the per-sector min z "
                   "('lowest', primary) or 5 cm z-histogram mode within "
                   "20 m ('mode'); LSQ plane through ring points within "
                   "0.15 m of it; height = perpendicular distance of the "
                   "sensor origin; scans with < 100 plane points skipped"),
        "frame": "our src/data loaders (sensor frame, z up; NCLT z flipped "
                 "by the loader)",
        "n_scans_per_sequence": args.n_scans}}
    per_sensor = {}
    for sensor, seq in seqs:
        t0 = time.time()
        loader = make_loader(sensor, seq)
        idx = np.unique(np.linspace(0, len(loader) - 1, args.n_scans).astype(int))
        acc = {"lowest": ([], []), "mode": ([], [])}
        for i in idx:
            g = ground_height(np.asarray(loader[int(i)]["points"]))
            for key in acc:
                if g[key] is not None:
                    acc[key][0].append(g[key][0])
                    acc[key][1].append(g[key][1])
        rec = {"n_sampled": int(len(idx))}
        for key, (hs, tilts) in acc.items():
            rec[key] = {"height": _stats(hs), "tilt_deg": _stats(tilts),
                        "heights": [round(x, 3) for x in hs]}
        out[f"{sensor}/{seq}"] = rec
        per_sensor.setdefault(sensor, {"lowest": [], "mode": []})
        for key in acc:
            per_sensor[sensor][key].extend(acc[key][0])
        L, M = rec["lowest"]["height"], rec["mode"]["height"]
        print(f"{sensor}/{seq:14s} lowest: n {L['n']:2d} median "
              f"{L.get('median', float('nan')):.3f} [p10 "
              f"{L.get('p10', float('nan')):.3f}, p90 "
              f"{L.get('p90', float('nan')):.3f}] tilt "
              f"{rec['lowest']['tilt_deg'].get('median', float('nan')):.2f} | "
              f"mode: median {M.get('median', float('nan')):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del loader
    out["_per_sensor"] = {s: {k: _stats(v[k]) for k in v}
                          for s, v in per_sensor.items()}
    for s, v in out["_per_sensor"].items():
        L, M = v["lowest"], v["mode"]
        print(f"== {s:9s} lowest n={L['n']:3d} median {L['median']:.3f} m "
              f"[p10 {L['p10']:.3f}, p90 {L['p90']:.3f}] | mode median "
              f"{M['median']:.3f} m", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
