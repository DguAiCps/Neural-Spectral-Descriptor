#!/usr/bin/env python3
"""Fit key re-metrization omega for the 2026-08 place-disjoint (pd) retrain protocol.

Same closed form as scripts/fit_key_remetrize.py — diagonal within-place whitening
r_k = (Var_within[k] + eps)^(-1/2), mean-normalized — on TRAIN revisit pairs
(<5 m, >=30-frame gap), but harvested from the fresh data/preprocessed_pd2026
training caches (cache_<hash>_<dataset>_<seq>.npz with 288D `descriptors` +
`poses`). The fit is checkpoint-free: descriptors are the deterministic
spectral-encoder output, mirroring the paper-main fit.

Variants:
  pd    -> artifacts/key_remetrize_r_pd2026.npy   (kitti+helipr+mulran+nclt train)
  pdnn  -> artifacts/key_remetrize_r_pdnn2026.npy (same minus nclt)

Val caches (*_val_*) are never read. The frozen artifacts/key_remetrize_r.npy is
loaded read-only for a cosine-similarity sanity check and never written.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data/preprocessed_pd2026"
FROZEN = REPO / "artifacts/key_remetrize_r.npy"
OUT = {
    "pd": REPO / "artifacts/key_remetrize_r_pd2026.npy",
    "pdnn": REPO / "artifacts/key_remetrize_r_pdnn2026.npy",
}
DTH, SKIP = 5.0, 30


def harvest_train_deltas(variant: str):
    deltas = []
    for f in sorted(glob.glob(str(CACHE / "cache_*.npz"))):
        name = Path(f).name
        if "_val_" in name or "_test_" in name:
            # protocol: fit on TRAIN sequences only. The _test_ skip was added
            # 2026-08-11: the original 2026-08-04 fit predated the test caches,
            # so the bare glob was train-only then; without this skip a re-run
            # today would silently harvest the 42 test caches (verified: the
            # committed artifact is reproduced bit-identically WITH this skip,
            # and diverges without it).
            continue
        if variant == "pdnn" and "_nclt_" in name:
            continue  # pdnn config excludes NCLT from train
        blob = np.load(f)
        d = blob["descriptors"].astype(np.float64)
        pos = blob["poses"][:, :3, 3].astype(np.float64)
        n = len(d)
        n_seq = 0
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
            n_seq += 1
        print(f"  {name}: {n} frames, {n_seq} revisit pairs")
    return np.array(deltas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["pd", "pdnn"], required=True)
    args = ap.parse_args()

    print(f"[{args.variant}] harvesting from {CACHE}")
    deltas = harvest_train_deltas(args.variant)
    print(f"harvested {len(deltas)} train revisit pairs total")
    Wdiag = 0.5 * np.mean(deltas ** 2, axis=0)
    eps = 1e-6 * Wdiag.mean()
    r = 1.0 / np.sqrt(Wdiag + eps)
    r /= r.mean()

    out = OUT[args.variant]
    np.save(out, r)
    print(f"wrote {out} shape={r.shape} min={r.min():.4f} max={r.max():.4f} mean={r.mean():.4f}")

    if FROZEN.exists():
        ref = np.load(FROZEN)  # read-only sanity reference
        cos = float(np.dot(r, ref) / (np.linalg.norm(r) * np.linalg.norm(ref)))
        print(f"cosine vs frozen {FROZEN.name}: {cos:.4f}")


if __name__ == "__main__":
    main()
