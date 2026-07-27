#!/usr/bin/env python3
"""Public-evaluator recomputation of the |p_BEV| fusion row: same protocol as
_pbev_branch_fusion.py but on evaluator-native operating caches (the caches the
release evaluator builds itself), using their own nsd_layouts and query sets.
Fusion weights stay frozen at the train-calibrated values."""
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
    WEIGHTS, CACHE_OUT, RESULT_OUT, bin288, make_model, nsd_embedding, evaluate,
    loader_for,
)
from _pbev_branch_fusion import BP, variant_inputs
from encoding.phase_features import _pool_rows, _phase_sketch

OPJOBS = {
    "K00": ("kitti", "00", "data/preprocessed_kitti_operating/kitti_operating_00_layout60.npz"),
    "K05": ("kitti", "05", "data/preprocessed_kitti_operating/kitti_operating_05_layout60.npz"),
    "K08": ("kitti", "08", "data/preprocessed_kitti_operating/kitti_operating_08_layout60.npz"),
    "N12": ("nclt", "2012-01-08", "data/preprocessed_nclt_zup_recal/nclt_operating_2012-01-08_layout60_stride1.npz"),
    "N13": ("nclt", "2013-01-10", "data/preprocessed_nclt_zup_frozen_s1/nclt_operating_2013-01-10_layout60_stride1.npz"),
    "TOWN": ("helipr", "Town01", "data/preprocessed_cross_sensor_operating/helipr_operating_Town01_layout60_stride1.npz"),
    "DCC": ("mulran", "DCC03", "data/preprocessed_mulran_operating/kitti_operating_DCC03_layout60.npz"),
    "KAI": ("mulran", "KAIST03", "data/preprocessed_mulran_operating/kitti_operating_KAIST03_layout60.npz"),
    "RIV": ("mulran", "Riverside03", "data/preprocessed_mulran_operating/kitti_operating_Riverside03_layout60.npz"),
}
FROZEN = {"pbev128_main": "(0.25, 1.0, 0.0, 0.5)",
          "pbev192_realloc": "(0.25, 1.0, 0.0, 1.0)"}


def op_pbev_sketch(tag, sensor, sequence, cache):
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    path = CACHE_OUT / f"{tag}_op_pbev.npz"
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["scan_ids"], cache["scan_ids"]):
            return z["sk_re"].astype(np.float32) + 1j * z["sk_im"].astype(np.float32)
        print(f"[{tag}] stale op pbev cache; rebuilding", flush=True)
    loader = loader_for(sensor, sequence)
    res, ims = [], []
    for i, scan_id in enumerate(cache["scan_ids"]):
        if i % 500 == 0:
            print(f"[{tag}] op pbev {i}/{len(cache['scan_ids'])}", flush=True)
        bev, _ = BP.project(loader[int(scan_id)]["points"], keep_intensity=False)
        sk = _phase_sketch(_pool_rows(bev, 16), 12)
        res.append(sk.real.astype(np.float32))
        ims.append(sk.imag.astype(np.float32))
    np.savez_compressed(path, scan_ids=cache["scan_ids"],
                        sk_re=np.asarray(res), sk_im=np.asarray(ims))
    return np.asarray(res) + 1j * np.asarray(ims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=list(OPJOBS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-k", type=int, default=100)
    ap.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    args = ap.parse_args()
    cfg, model, clf, clf_blob = make_model(args.device, args.seed)
    suffix = "" if args.seed == 1 else f"_seed{args.seed}"
    result_path = RESULT_OUT / f"pbev_operating_eval{suffix}.json"
    report = json.loads(result_path.read_text()) if result_path.exists() else {}

    for tag in args.tags:
        if tag in report:
            print(f"[{tag}] already done, skipping", flush=True)
            continue
        sensor, sequence, cache_path = OPJOBS[tag]
        print(f"\n=== {tag} ===", flush=True)
        cache = np.load(REPO / cache_path)
        d288 = bin288(cache["fft_magnitudes"].astype(np.float32))
        sk = op_pbev_sketch(tag, sensor, sequence, cache)
        layouts = cache["nsd_layouts"].astype(np.float32)
        nsd = nsd_embedding(cache, sensor, d288, layouts,
                            cfg, model, clf, clf_blob, args.device)
        variants = {}
        for name, nf in (("pbev128_main", 8), ("pbev192_realloc", 12)):
            hkey, hfft, hnorm, column_bank = variant_inputs(sk, nf)
            vals = evaluate(cache["poses"], nsd, hkey, hfft, hnorm,
                            column_bank, args.pool_k)
            variants[name] = vals
            print(f"  {name}: nsd={vals['nsd']:.4f} pbev={vals['height']:.4f} "
                  f"frozen={vals[FROZEN[name]]:.4f} "
                  f"oracle={vals['oracle_union']:.4f} q={vals['n_queries']}",
                  flush=True)
        report[tag] = {"sensor": sensor, "sequence": sequence,
                       "pool_k": args.pool_k, "variants": variants}
        result_path.write_text(json.dumps(report, indent=2))

    # aggregate with frozen weights and actual query counts
    for name in FROZEN:
        num = den = 0.0
        for tag in OPJOBS:
            if tag not in report:
                continue
            v = report[tag]["variants"][name]
            num += v["n_queries"] * v[FROZEN[name]]
            den += v["n_queries"]
        if den:
            print(f"AGG {name} frozen={FROZEN[name]}: Rbar_q={num/den:.5f} "
                  f"(n={int(den)})", flush=True)
    print("PBEV_OPERATING_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
