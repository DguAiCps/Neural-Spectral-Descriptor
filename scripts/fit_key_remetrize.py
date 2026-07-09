#!/usr/bin/env python3
"""Fit the paper-main key re-metrization vector r (artifacts/key_remetrize_r.npy).

Closed-form diagonal within-place whitening (diag_wccn) of the frozen 288D
magnitude key: r_k = (Var_within[k] + eps)^(-1/2), mean-normalized. Fit on
TRAIN revisit pairs (<5 m, >=30-frame gap; ~38k pairs) harvested from the
operating caches, with the 9 validation sequences excluded. Zero training;
yaw invariance is preserved (diagonal reweight of the invariant magnitude).

Default is verify mode (compare against the committed artifact); pass
--write to overwrite it. See PAPER_MAIN_B3.md for provenance.
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data/preprocessed_cross_sensor_operating"
OUT = REPO / "artifacts/key_remetrize_r.npy"
VAL = {
    "kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
    "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"],
}
DTH, SKIP = 5.0, 30


def harvest_train_deltas():
    deltas = []
    valset = {(s, q) for s, qs in VAL.items() for q in qs}
    for f in sorted(glob.glob(str(CACHE / "*_operating_*_layout60_stride1.npz"))):
        base = Path(f).name.replace("_operating_", "|").replace("_layout60_stride1.npz", "")
        sensor, seq = base.split("|")
        if (sensor, seq) in valset:
            continue
        blob = np.load(f)
        d = blob["descriptors"].astype(np.float64)
        pos = blob["poses"][:, :3, 3].astype(np.float64)
        n = len(d)
        for q in range(SKIP, n):
            prior = np.arange(0, q - SKIP)
            if prior.size == 0:
                continue
            geo = np.linalg.norm(pos[prior] - pos[q], axis=1)
            positives = prior[geo < DTH]
            if positives.size == 0:
                continue
            m = positives[np.argmin(np.linalg.norm(pos[positives] - pos[q], axis=1))]
            deltas.append(d[q] - d[m])
    return np.array(deltas)


def main():
    deltas = harvest_train_deltas()
    print(f"harvested {len(deltas)} train revisit pairs")
    Wdiag = 0.5 * np.mean(deltas ** 2, axis=0)
    eps = 1e-6 * Wdiag.mean()
    r = 1.0 / np.sqrt(Wdiag + eps)
    r /= r.mean()

    if "--write" in sys.argv:
        np.save(OUT, r)
        print(f"wrote {OUT} ({r.shape[0]}D)")
    elif OUT.exists():
        ref = np.load(OUT)
        diff = float(np.abs(r - ref).max())
        print(f"verify vs committed artifact: max|diff| = {diff:.3e} "
              f"({'MATCH' if diff < 1e-6 else 'MISMATCH — cache set changed?'})")
    else:
        print(f"{OUT} missing; rerun with --write to create it")


if __name__ == "__main__":
    main()
