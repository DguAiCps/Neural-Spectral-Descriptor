#!/usr/bin/env python3
"""Branch-protocol substitution with the BEV phase-sketch magnitude.

Two variants, per the audited experiment spec:
  pbev128_main    -- BEV 8-freq raw-complex block exactly as deployed
                     (derivable from the stored 384D p payload unchanged).
  pbev192_realloc -- BEV-only 12-freq reallocation of the same 384D budget.

Retrieval key = |z| of the raw complex coefficients (true yaw-invariant, no
log compression). Cyclic alignment = raw complex cross-correlation from the
SAME sketch. The 20x120 raw-height column grid is NOT used; the optional
column channel is a band-limited reconstruction from the same coefficients
(w3=0 rows give the strict sketch-only numbers). Frozen-NSD embedding,
candidate-union protocol, and weight sweep come verbatim from
scripts/_height_candidate_fusion.py. Actual query counts are recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _height_candidate_fusion import (
    JOBS, WEIGHTS, CACHE_OUT, RESULT_OUT, bin288, l2, loader_for,
    make_model, nsd_embedding, column_align_bank, evaluate,
)
from run_kitti_operating_point import _make_encoder, _project_nsd_layout
from encoding.bev_image import BEVProjector
from encoding.phase_features import _pool_rows, _phase_sketch

BP = BEVProjector(n_sectors=60, max_range=80.0, min_range=1.0, z_min=-3.0,
                  height_encoding='max', n_height_layers=8, z_max=5.0)

VAL_TAGS = ["K00", "K05", "K08", "N12", "N13", "TOWN", "DCC", "KAI", "RIV"]
TRAIN_TAGS = ["K01_TR", "K02_TR", "K06_TR", "K07_TR", "N0511_TR", "N0804_TR"]


def pbev_cache(tag: str, sensor: str, sequence: str, cache, encoder):
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    path = CACHE_OUT / f"{tag}_pbev.npz"
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["scan_ids"], cache["scan_ids"]) and "sk_re" in z.files:
            return z
        print(f"[{tag}] stale pbev cache; rebuilding", flush=True)
    hpath = CACHE_OUT / f"{tag}_height120.npz"
    have_layout = hpath.exists() and "nsd_layout" in np.load(hpath).files
    loader = loader_for(sensor, sequence)
    res, ims, layouts = [], [], []
    for i, scan_id in enumerate(cache["scan_ids"]):
        if i % 500 == 0:
            print(f"[{tag}] pbev {i}/{len(cache['scan_ids'])}", flush=True)
        points = loader[int(scan_id)]["points"]
        bev, _ = BP.project(points, keep_intensity=False)
        sk = _phase_sketch(_pool_rows(bev, 16), 12)      # (16,12) complex64
        res.append(sk.real.astype(np.float32))
        ims.append(sk.imag.astype(np.float32))
        if not have_layout:
            layouts.append(_project_nsd_layout(encoder, points, 60).astype(np.float16))
    out = dict(scan_ids=cache["scan_ids"],
               sk_re=np.asarray(res), sk_im=np.asarray(ims))
    if not have_layout:
        out["nsd_layout"] = np.asarray(layouts)
    np.savez_compressed(path, **out)
    return np.load(path)


def variant_inputs(sk: np.ndarray, n_freqs: int):
    """Retrieval key, alignment bank, and reconstructed column bank from the
    first n_freqs coefficients of the (n,16,12) sketch."""
    skf = sk[:, :, :n_freqs]
    hkey = l2(np.abs(skf).reshape(len(skf), -1).astype(np.float32))
    hfft = np.zeros((len(skf), 16, 31), np.complex64)
    hfft[:, :, 1:n_freqs + 1] = skf
    hnorm = np.clip(np.linalg.norm(np.abs(skf).reshape(len(skf), -1), axis=1),
                    1e-8, None)
    recon = np.fft.irfft(hfft, n=60, axis=-1).astype(np.float32)
    return hkey, hfft, hnorm, column_align_bank(recon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=VAL_TAGS + TRAIN_TAGS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-k", type=int, default=100)
    ap.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    args = ap.parse_args()
    RESULT_OUT.mkdir(parents=True, exist_ok=True)
    cfg, model, clf, clf_blob = make_model(args.device, args.seed)
    encoder = _make_encoder(cfg, args.device)
    suffix = "" if args.seed == 1 else f"_seed{args.seed}"
    result_path = RESULT_OUT / f"pbev_branch_fusion{suffix}.json"
    report = json.loads(result_path.read_text()) if result_path.exists() else {}

    for tag in args.tags:
        if tag in report and {"pbev128_main", "pbev192_realloc"} <= set(
                report[tag].get("variants", {})):
            print(f"[{tag}] already done, skipping", flush=True)
            continue
        sensor, sequence, cache_path = JOBS[tag]
        print(f"\n=== {tag} ===", flush=True)
        cache = np.load(REPO / cache_path)
        d288 = bin288(cache["fft_magnitudes"].astype(np.float32))
        pc = pbev_cache(tag, sensor, sequence, cache, encoder)
        sk = pc["sk_re"].astype(np.float32) + 1j * pc["sk_im"].astype(np.float32)

        if "nsd_layout" in pc.files:
            layouts = pc["nsd_layout"].astype(np.float32)
        else:
            layouts = np.load(CACHE_OUT / f"{tag}_height120.npz")["nsd_layout"].astype(np.float32)
        nsd = nsd_embedding(cache, sensor, d288, layouts,
                            cfg, model, clf, clf_blob, args.device)

        variants = {}
        for name, nf in (("pbev128_main", 8), ("pbev192_realloc", 12)):
            print(f"[{tag}] variant={name}", flush=True)
            hkey, hfft, hnorm, column_bank = variant_inputs(sk, nf)
            vals = evaluate(cache["poses"], nsd, hkey, hfft, hnorm,
                            column_bank, args.pool_k)
            variants[name] = vals
            best = max(WEIGHTS, key=lambda w: vals[str(w)])
            print(f"  nsd={vals['nsd']:.4f} pbev={vals['height']:.4f} "
                  f"w(0.25,1,1,0)={vals[str((0.25, 1.0, 1.0, 0.0))]:.4f} "
                  f"best={vals[str(best)]:.4f} w={best} "
                  f"oracle={vals['oracle_union']:.4f} q={vals['n_queries']}",
                  flush=True)
        report[tag] = {"sensor": sensor, "sequence": sequence,
                       "pool_k": args.pool_k, "variants": variants}
        result_path.write_text(json.dumps(report, indent=2))
    print("PBEV_BRANCH_FUSION_DONE", flush=True)


if __name__ == "__main__":
    main()
