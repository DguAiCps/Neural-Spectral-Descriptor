#!/usr/bin/env python3
"""NSD / NSD-H cross-session evaluation (host, torch-free).

Protocol + hard rules: see scripts/_xs_eval_lib.py. Inputs are the frozen
global-pose key dumps results/pd2026/keys_{variant}_s{seed}_{sensor}_{seq}_gp.npz
(emb416 + magnitude keys) and the pd2026 spectrum caches
data/preprocessed_pd2026/testspec_{sensor}_{seq}.npz (alignment banks built
per session AFTER spacing slicing — per-frame builders, identical result).
Fusion tuples are the frozen val-selected ones
(results/pd2026/tuple_selection.json); never re-selected.

Method rows emitted per (protocol, seed):
  nsd         pd   / pbev128_main   (body: NSD)
  nsdh        pd   / hspec12        (body: NSD-H)
  nsd_pdnn    pdnn / pbev128_main   (appendix)
  nsdh_pdnn   pdnn / hspec12        (auxiliary NCLT zero-shot row)
  nsd_keyonly pd   / emb416 cosine  (appendix diagnostic)

Outputs: results/pd2026_xsession/xs_{method}_{protocol}_s{seed}.json
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path

import numpy as np

import _xs_eval_lib as lib

REPO = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO / "data/preprocessed_pd2026"
TUPLES = REPO / "results/pd2026/tuple_selection.json"

METHODS = {
    "nsd": ("pd", "pbev128_main"),
    "nsdh": ("pd", "hspec12"),
    "nsd_pdnn": ("pdnn", "pbev128_main"),
    "nsdh_pdnn": ("pdnn", "hspec12"),
}


def frozen_tuple(variant: str, op: str):
    sel = json.loads(TUPLES.read_text())[f"{variant}/{op}"]["selected_tuple"]
    w = ast.literal_eval(sel)
    assert isinstance(w, tuple) and len(w) == 4, sel
    return tuple(float(x) for x in w), sel


def load_keys(variant: str, seed: int, sensor: str, seq: str) -> dict:
    p = lib.KEYS_DIR / f"keys_{variant}_s{seed}_{sensor}_{seq}_gp.npz"
    with np.load(p, allow_pickle=True) as z:
        return {"emb416": z["emb416"].astype(np.float32),
                "pbev_mag": z["pbev_mag"].astype(np.float32),
                "hspec_mag": z["hspec_mag"].astype(np.float32),
                "scan_ids": z["scan_ids"].copy(),
                "positions": z["poses"][:, :3, 3].astype(np.float64),
                "path": str(p)}


def load_spec(sensor: str, seq: str) -> dict:
    p = SPEC_DIR / f"testspec_{sensor}_{seq}.npz"
    with np.load(p) as z:
        return {"sk": (z["sk_re"].astype(np.float32)
                       + 1j * z["sk_im"].astype(np.float32)),
                "raw_height": z["raw_height"].copy(),
                "scan_ids": z["scan_ids"].copy()}


def session_side(pair: dict, side: str, variant: str, seed: int):
    """Sliced keys + spectrum banks for one side of the pair, with the
    scan_id / pose-consistency asserts (single canonical pose source)."""
    sensor = pair["sensor"]
    seq = pair["db_seq"] if side == "db" else pair["query_seq"]
    idx = pair["db_idx"] if side == "db" else pair["q_idx"]
    session_sids = (pair["db_session_scan_ids"] if side == "db"
                    else pair["q_session_scan_ids"])
    keys = load_keys(variant, seed, sensor, seq)
    assert np.array_equal(keys["scan_ids"], session_sids), (seq, "keys scan_ids")
    canon_pos = pair["db_pos"] if side == "db" else pair["q_pos"]
    kp = keys["positions"][idx].copy()
    if lib.CONST["planar"]:
        kp[:, 2] = 0.0
    assert np.allclose(kp, canon_pos, atol=1e-6), (seq, "pose mismatch vs canonical")
    spec = load_spec(sensor, seq)
    assert np.array_equal(spec["scan_ids"], session_sids), (seq, "spec scan_ids")
    sliced = lib.take({"emb416": keys["emb416"], "pbev_mag": keys["pbev_mag"],
                       "hspec_mag": keys["hspec_mag"], "sk": spec["sk"],
                       "raw_height": spec["raw_height"]}, idx)
    banks = {"pbev128_main": lib.variant_inputs(sliced["sk"], 8),
             "hspec12": lib.hspec_inputs(sliced["raw_height"], 12)}
    # Bank hkey must reproduce the dump's magnitude keys (per-frame builders).
    assert np.allclose(banks["pbev128_main"][0], sliced["pbev_mag"], atol=1e-5)
    assert np.allclose(banks["hspec12"][0], sliced["hspec_mag"], atol=1e-5)
    return sliced, banks, keys["path"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocols", nargs="*", default=list(lib.PROTOCOLS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--methods", nargs="*", default=list(METHODS) + ["nsd_keyonly"])
    ap.add_argument("--pool-k", type=int, default=int(lib.CONST["pool_k"]))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for protocol in args.protocols:
        pair = lib.make_pair(protocol)
        print(f"=== {protocol}: n_db={len(pair['db_pos'])} "
              f"n_spaced_q={len(pair['q_pos'])} n_roster={len(pair['roster'])} "
              f"roster_sha={pair['roster_sha256'][:12]}", flush=True)
        for seed in args.seeds:
            sides = {}   # variant -> (db_sliced, db_banks, q_sliced, q_banks, paths)

            def get_sides(variant):
                if variant not in sides:
                    d, db_banks, dp = session_side(pair, "db", variant, seed)
                    q, q_banks, qp = session_side(pair, "query", variant, seed)
                    sides[variant] = (d, db_banks, q, q_banks, [dp, qp])
                return sides[variant]

            for method in args.methods:
                name = f"xs_{method}_{protocol}_s{seed}"
                out_p = lib.OUT_DIR / f"{name}.json"
                if out_p.exists() and not args.overwrite:
                    print(f"  {name}: exists, skip", flush=True)
                    continue
                if method == "nsd_keyonly":
                    d, _, q, _, paths = get_sides("pd")
                    ranked = lib.rank_cross_session(
                        d["emb416"], q["emb416"], pair["roster"],
                        n_coarse=int(lib.CONST["n_coarse_baseline"]))
                    scored = lib.score_rankings(ranked, pair["db_pos"],
                                                pair["q_pos"], pair["roster"])
                    payload = lib.result_payload(
                        method, pair, scored, variant="pd", seed=seed,
                        extra={"key_field": "emb416",
                               "n_coarse": int(lib.CONST["n_coarse_baseline"]),
                               "key_sources": paths})
                else:
                    variant, op = METHODS[method]
                    d, db_banks, q, q_banks, paths = get_sides(variant)
                    w, sel = frozen_tuple(variant, op)
                    ranked = lib.nsd_rank_fused(
                        d["emb416"], q["emb416"], db_banks[op], q_banks[op],
                        pair["roster"], w, args.pool_k)
                    scored = lib.score_rankings(ranked, pair["db_pos"],
                                                pair["q_pos"], pair["roster"])
                    payload = lib.result_payload(
                        method, pair, scored, variant=variant, seed=seed, op=op,
                        extra={"fusion": {"tuple": list(w), "tuple_str": sel,
                                          "tuple_source": str(TUPLES),
                                          "pool_k": args.pool_k},
                               "key_sources": paths})
                p = lib.write_result(payload, name)
                m = payload["metrics"]
                print(f"  {name}: R@1@5m={m['R@1@5m']:.4f} "
                      f"R@1@20m={m['R@1@20m']:.4f} "
                      f"R1pct@20m={m['R1pct@20m']:.4f} -> {p.name}", flush=True)
    print("XS_NSD_DONE", flush=True)


if __name__ == "__main__":
    main()
