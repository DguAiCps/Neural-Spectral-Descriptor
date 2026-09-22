#!/usr/bin/env python3
"""Dump per-query success@1 hit vectors at the DEPLOYED headline operating point.

Reproduces the exact two-index fusion pipelines of _pbev_branch_fusion.py (NSD,
pbev128_main, tuple (0.5,1,0,0.5)) and _hspec_branch_fusion.py (NSD-H, hspec12,
tuple (1,1,1,2)), but records per-query [query_idx, success_at_k1] for the
deployed weight tuple so a paired significance test can be run against the
per-query baseline dumps (SC++, BEVPlace++).

Output: results/per_query_headline/{nsd,nsd_h}/<DATASET>.json
Schema matches results/per_query_baselines/*: {records:[{query_idx, success_at_k1}]}.
Aggregate R@1 is printed and compared against the committed branch-fusion numbers.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _height_candidate_fusion import (
    JOBS, bin288, make_model, nsd_embedding, evaluate, height_cache, DTH, SKIP,
)
from _pbev_branch_fusion import pbev_cache, variant_inputs
from _hspec_branch_fusion import hspec_inputs
from run_kitti_operating_point import _make_encoder

VAL_TAGS = ["K00", "K05", "K08", "N12", "N13", "TOWN", "DCC", "KAI", "RIV"]
TAG2DS = {"K00": "KITTI_00", "K05": "KITTI_05", "K08": "KITTI_08",
          "N12": "NCLT_2012-01-08", "N13": "NCLT_2013-01-10",
          "TOWN": "HeLiPR_Town01", "DCC": "MulRan_DCC03",
          "KAI": "MulRan_KAIST03", "RIV": "MulRan_Riverside03"}
NSD_TUPLE = (0.5, 1.0, 0.0, 0.5)     # pbev128_main deployed
NSDH_TUPLE = (1.0, 1.0, 1.0, 2.0)    # hspec12 deployed
OUT = REPO / "results/per_query_headline"


def write_dump(method: str, tag: str, vals: dict, output_root: Path):
    perq = vals["_perquery"]
    ds = TAG2DS[tag]
    d = output_root / method
    d.mkdir(parents=True, exist_ok=True)
    rec = [{"query_idx": q, "success_at_k1": h} for q, h in perq]
    r1 = sum(h for _, h in perq) / max(len(perq), 1)
    (d / f"{ds}.json").write_text(json.dumps(
        {"method": method, "dataset": ds, "distance_threshold_m": DTH,
         "skip_frames": SKIP, "n_queries": len(perq),
         "deployed_tuple": list(NSD_TUPLE if method == "nsd" else NSDH_TUPLE),
         "recall_at_1": r1, "records": rec}, indent=1))
    return r1


def parse_cache_overrides(values: list[str]) -> dict[str, Path]:
    """Parse explicit TAG=PATH cache substitutions for lineage validation."""
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected TAG=PATH cache override, got {value!r}")
        tag, path = value.split("=", 1)
        if tag not in JOBS:
            raise ValueError(f"unknown cache tag {tag!r}")
        overrides[tag] = Path(path)
    return overrides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=VAL_TAGS)
    ap.add_argument("--methods", nargs="*", default=["nsd", "nsd_h"])
    ap.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-root", type=Path, default=OUT,
                    help="Root directory for method-specific per-query dumps")
    ap.add_argument("--cache-override", action="append", default=[], metavar="TAG=PATH",
                    help="Use a lineage-validation cache instead of the default JOBS entry.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cache_overrides = parse_cache_overrides(args.cache_override)

    cfg, model, clf, clf_blob = make_model(args.device, args.seed)
    encoder = _make_encoder(cfg, args.device)

    for tag in args.tags:
        sensor, sequence, cache_path = JOBS[tag]
        cache_path = cache_overrides.get(tag, REPO / cache_path)
        ds = TAG2DS[tag]
        need = [m for m in args.methods
                if args.force or not (args.output_root / m / f"{ds}.json").exists()]
        if not need:
            print(f"[{tag}] all methods done, skipping", flush=True)
            continue
        print(f"\n=== {tag} ({ds}) seed{args.seed} ===", flush=True)
        cache = np.load(cache_path)
        d288 = bin288(cache["fft_magnitudes"].astype(np.float32))
        poses = cache["poses"]

        if "nsd" in need:
            pc = pbev_cache(tag, sensor, sequence, cache, encoder)
            sk = pc["sk_re"].astype(np.float32) + 1j * pc["sk_im"].astype(np.float32)
            layouts = pc["nsd_layout"].astype(np.float32)
            nsd = nsd_embedding(cache, sensor, d288, layouts,
                                cfg, model, clf, clf_blob, args.device)
            hkey, hfft, hnorm, cb = variant_inputs(sk, 8)
            vals = evaluate(poses, nsd, hkey, hfft, hnorm, cb, 100,
                            dump_weight=NSD_TUPLE)
            r1 = write_dump("nsd", tag, vals, args.output_root)
            print(f"  [nsd]   R@1={r1:.4f} nsd_only={vals['nsd']:.4f} "
                  f"h_only={vals['height']:.4f} oracle={vals['oracle_union']:.4f} "
                  f"q={vals['n_queries']}", flush=True)

        if "nsd_h" in need:
            hc = height_cache(tag, sensor, sequence, cache, encoder)
            raw_height = hc["raw_height"].astype(np.float32)
            layouts_h = hc["nsd_layout"].astype(np.float32)
            nsd_h = nsd_embedding(cache, sensor, d288, layouts_h,
                                  cfg, model, clf, clf_blob, args.device)
            hkey, hfft, hnorm, cb = hspec_inputs(raw_height, 12)
            vals = evaluate(poses, nsd_h, hkey, hfft, hnorm, cb, 100,
                            dump_weight=NSDH_TUPLE)
            r1 = write_dump("nsd_h", tag, vals, args.output_root)
            print(f"  [nsd_h] R@1={r1:.4f} nsd_only={vals['nsd']:.4f} "
                  f"h_only={vals['height']:.4f} oracle={vals['oracle_union']:.4f} "
                  f"q={vals['n_queries']}", flush=True)

    print("HEADLINE_PERQUERY_DONE", flush=True)


if __name__ == "__main__":
    main()
