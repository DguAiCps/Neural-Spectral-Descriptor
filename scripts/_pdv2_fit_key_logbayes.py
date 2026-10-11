#!/usr/bin/env python3
"""Refit the key re-metrization on split v2 train (2026-10-10).

Key block x_key = w * log1p(d) (then L2-normalized) instead of r * d, with
  w_k = sqrt(max(1/W_k - 1/B_k, 0))   ("Bayes diagonal")
  W_k = 0.5 E[(x_q - x_m)^2] over same-place pairs (q >= 30, nearest prior
        m <= q - 31 within 5 m: the _pd_fit_key_remetrize rule)
  B_k = 0.5 E[(x_i - x_j)^2] over random different-place pairs
        (i - j > 30, 3D distance > 25 m, same sequence; 40,000 per sequence
        at most, numpy default_rng(0))
  x = log1p(d); every train sequence weighted equally (per-sequence means,
  then the mean over sequences; KITTI 01 has no revisit and enters B only).
The 32 sigma coordinates that pool a single frequency are constant and get
weight 0. Chosen on the nine val sequences among the variants of
logs/check2_omega_20261009 (docs/RAW/실험일지/2026-10-09_fp_root_cause_checks.md).

Output: artifacts/key_remetrize_r_pdv2_logbayes.npy (288,) + .json sidecar.
Host numpy only.
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TRAIN_CACHE = REPO / "data/preprocessed_v2"
OUT = REPO / "artifacts/key_remetrize_r_pdv2_logbayes.npy"
TRAIN = [("kitti", s) for s in ("01", "02", "05", "06")] + \
        [("helipr", f"Bridge0{i}") for i in (1, 2, 3, 4)]
CONST = np.sort(np.r_[144 + np.arange(16) * 9, 145 + np.arange(16) * 9])
SAME_M, DIFF_M, GAP = 5.0, 25.0, 30


def train_path(s, q):
    p = [f for f in glob.glob(str(TRAIN_CACHE / f"cache_*_{s}_{q}.npz"))
         if "_val_" not in f and "_test_" not in f]
    assert len(p) == 1, p
    return p[0]


def harvest_same(pos):
    qs, ms = [], []
    for q in range(GAP, len(pos)):
        if q - GAP <= 0:
            continue
        geo = np.linalg.norm(pos[:q - GAP] - pos[q], axis=1)
        m = int(np.argmin(geo))
        if geo[m] < SAME_M:
            qs.append(q)
            ms.append(m)
    return np.array(qs, dtype=np.int64), np.array(ms, dtype=np.int64)


def sample_diff(pos, n_target, rng):
    n = len(pos)
    out_i, out_j, have = [], [], 0
    for _ in range(50):
        i = rng.integers(GAP + 1, n, size=2 * n_target)
        j = np.floor(rng.random(2 * n_target) * (i - GAP)).astype(np.int64)
        ok = np.linalg.norm(pos[i] - pos[j], axis=1) > DIFF_M
        out_i.append(i[ok])
        out_j.append(j[ok])
        have += int(ok.sum())
        if have >= n_target:
            break
    return np.concatenate(out_i)[:n_target], np.concatenate(out_j)[:n_target]


def main():
    rng = np.random.default_rng(0)
    Ws, Bs, counts = [], [], {}
    for s, q in TRAIN:
        z = np.load(train_path(s, q))
        x = np.log1p(z["descriptors"].astype(np.float64))
        pos = z["poses"][:, :3, 3].astype(np.float64)
        qs, ms = harvest_same(pos)
        i, j = sample_diff(pos, min(3 * len(pos), 40000), rng)
        if len(qs):
            Ws.append(0.5 * ((x[qs] - x[ms]) ** 2).mean(0))
        Bs.append(0.5 * ((x[i] - x[j]) ** 2).mean(0))
        counts[f"{s}/{q}"] = {"same_place_pairs": int(len(qs)), "diff_place_pairs": int(len(i))}
        print(f"{s}/{q}: same {len(qs)}  diff {len(i)}", flush=True)
    keep = np.setdiff1d(np.arange(288), CONST)
    W = np.mean(Ws, axis=0)[keep]
    B = np.mean(Bs, axis=0)[keep]
    w = np.zeros(288, dtype=np.float64)
    w[keep] = np.sqrt(np.maximum(1.0 / W - 1.0 / B, 0.0))
    np.save(OUT, w.astype(np.float32))
    OUT.with_suffix(".json").write_text(json.dumps({
        "generator": "scripts/_pdv2_fit_key_logbayes.py",
        "key_transform": "log1p",
        "rule": "w = sqrt(max(1/W - 1/B, 0)) on log1p(d); W, B per-sequence "
                "means averaged over the 8 v2 train sequences; constant sigma "
                "coordinates weight 0",
        "zero_weight_coords": int((w == 0).sum()),
        "pairs": counts}, indent=2))
    print(f"wrote {OUT} (zero weights {(w == 0).sum()}, max {w.max():.3f})", flush=True)


if __name__ == "__main__":
    main()
