#!/usr/bin/env python3
"""Height-spectral NSD-H: store p_H = low-frequency complex DFT of the 20x120
max-height polar grid (20 rings x F freqs, Re/Im), and derive BOTH roles from
it -- |p_H| (20F D) for yaw-invariant candidate retrieval and the complex
coefficients for band-limited cyclic alignment. The raw grid is NOT stored.
Stored state = 416 + 128 (cyl block kept) + 40F floats.

Same frozen-NSD union-and-fusion protocol as _height_candidate_fusion.py;
reuses its cached 20x120 raw_height grids (no point-cloud reload).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _height_candidate_fusion import (
    JOBS, WEIGHTS, CACHE_OUT, RESULT_OUT, bin288, l2,
    make_model, nsd_embedding, column_align_bank, evaluate,
)

VAL_TAGS = ["K00", "K05", "K08", "N12", "N13", "TOWN", "DCC", "KAI", "RIV"]
TRAIN_TAGS = ["K01_TR", "K02_TR", "K06_TR", "K07_TR", "N0511_TR", "N0804_TR"]


def hspec_inputs(raw_height: np.ndarray, n_freqs: int):
    """Retrieval key, alignment bank, and band-limited column bank from the
    first n_freqs complex coefficients of the height grid's azimuthal DFT."""
    coeffs = np.fft.rfft(raw_height.astype(np.float32), axis=-1, norm="ortho")
    sk = coeffs[:, :, 1:n_freqs + 1].astype(np.complex64)      # (n, 20, F)
    hkey = l2(np.abs(sk).reshape(len(sk), -1))                  # 20F D
    n_bins = raw_height.shape[-1] // 2 + 1                      # 61
    hfft = np.zeros((len(sk), raw_height.shape[1], n_bins), np.complex64)
    hfft[:, :, 1:n_freqs + 1] = sk
    hnorm = np.clip(np.linalg.norm(np.abs(sk).reshape(len(sk), -1), axis=1),
                    1e-8, None)
    recon = np.fft.irfft(hfft, n=raw_height.shape[-1], axis=-1).astype(np.float32)
    return hkey, hfft, hnorm, column_align_bank(recon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=VAL_TAGS + TRAIN_TAGS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-k", type=int, default=100)
    ap.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--freqs", nargs="*", type=int, default=[12])
    args = ap.parse_args()
    RESULT_OUT.mkdir(parents=True, exist_ok=True)
    cfg, model, clf, clf_blob = make_model(args.device, args.seed)
    suffix = "" if args.seed == 1 else f"_seed{args.seed}"
    result_path = RESULT_OUT / f"hspec_branch_fusion{suffix}.json"
    report = json.loads(result_path.read_text()) if result_path.exists() else {}

    for tag in args.tags:
        want = {f"hspec{f}" for f in args.freqs}
        if tag in report and want <= set(report[tag].get("variants", {})):
            print(f"[{tag}] already done, skipping", flush=True)
            continue
        sensor, sequence, cache_path = JOBS[tag]
        print(f"\n=== {tag} ===", flush=True)
        cache = np.load(REPO / cache_path)
        d288 = bin288(cache["fft_magnitudes"].astype(np.float32))
        hc = np.load(CACHE_OUT / f"{tag}_height120.npz")
        raw_height = hc["raw_height"].astype(np.float32)
        layouts = hc["nsd_layout"].astype(np.float32)
        nsd = nsd_embedding(cache, sensor, d288, layouts,
                            cfg, model, clf, clf_blob, args.device)
        variants = report.get(tag, {}).get("variants", {})
        for F in args.freqs:
            name = f"hspec{F}"
            if name in variants:
                continue
            hkey, hfft, hnorm, column_bank = hspec_inputs(raw_height, F)
            vals = evaluate(cache["poses"], nsd, hkey, hfft, hnorm,
                            column_bank, args.pool_k)
            variants[name] = vals
            best = max(WEIGHTS, key=lambda w: vals[str(w)])
            print(f"  {name}: nsd={vals['nsd']:.4f} h={vals['height']:.4f} "
                  f"best={vals[str(best)]:.4f} w={best} "
                  f"oracle={vals['oracle_union']:.4f} q={vals['n_queries']}",
                  flush=True)
        report[tag] = {"sensor": sensor, "sequence": sequence,
                       "pool_k": args.pool_k, "variants": variants}
        result_path.write_text(json.dumps(report, indent=2))
    print("HSPEC_BRANCH_FUSION_DONE", flush=True)


if __name__ == "__main__":
    main()
