#!/usr/bin/env python3
"""ImLPR cross-session evaluation (Table 2; torch-free, host or docker).

Implements the "RING-style cross-session protocol under a common keyframe
roster" (scripts/_xs_eval_lib.py — NOT the official RING roster): 20 m map /
5 m query spacing, fixed 20 m roster denominator at every threshold.
Structure mirrors scripts/_xs_bevplace_eval.py stage 2; there is NO stage 1
here — descriptors come from the FROZEN interface produced by the ImLPR
RIV/descriptor stage (_external/imlpr_data_pd2026/desc_{sensor}_{seq}.npz,
desc (n, 8448) float16 L2-normalized before the fp16 cast, scan_ids ==
pd2026 operating-cache scan_ids). This module imports NO torch.

Reused modules (no existing file is modified):
  * scripts/_pdb_imlpr_eval — load_desc (frozen-interface reader),
    TRAIN_OVERLAP and the METHODS registry (per-method desc prefix, default
    weights, encoder meta, zero-shot disclosure).
  * scripts/_pdb_bevplace_eval — sha256_head (checkpoint sha-logging
    convention).
  * scripts/_xs_eval_lib — pairing, DB-only cosine ranking, fixed-roster
    scoring, payload/writing. lib.rank_cross_session L2-normalizes both
    sides internally, so the fp16 round-trip is score-neutral.

Pose single source: positions come ONLY from lib.make_pair (canonical
global-GT results/pd2026/keys_pd_s1_{sensor}_{seq}_gp.npz). Poses are never
read from pd2026 operating caches, method artifacts, or descriptor caches.

R4 scan_id identity: for each side the frozen descriptor scan_ids are
asserted equal to the canonical session scan_ids before the lib.take spacing
slice (the operating-cache and gp-keys scan_id rosters are identical, so the
frozen interface satisfies this by construction).

Matching is plain cosine (DB-only, n_coarse=200, no rerank) — the intra
driver applies no query-time trick either; rotation robustness is baked into
the RIV/ViT descriptor. Output: results/pd2026_xsession/
xs_imlpr_{protocol}.json. None of the 7 protocol sessions (nclt 2012-02-04/
2012-03-17, mulran DCC/KAIST/Riverside 01-03) is in ImLPR's training split,
so train_overlap is False throughout; the flags are still carried per the
aed302d disclosure convention.

--method helios serves the HeLiOS (ICRA'25) descriptors on the SAME frozen
interface (desc_helios_* from scripts/_helios_desc.py, also 8448D fp16),
writing xs_helios_{protocol}.json; default imlpr is unchanged. The 7
protocol sessions are outside both methods' training splits (shared
TRAIN_OVERLAP set).

Usage:
    python3 scripts/_xs_imlpr_eval.py [--protocols dcc_01_02 ...] [--overwrite]
    python3 scripts/_xs_imlpr_eval.py --method helios
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

import _xs_eval_lib as lib  # noqa: E402  (torch-free)
from _pdb_bevplace_eval import sha256_head  # noqa: E402  (torch-free)
from _pdb_imlpr_eval import (  # noqa: E402  (torch-free)
    METHODS, TRAIN_OVERLAP, load_desc, rel_to_repo,
)


def load_side(pair: dict, side: str, prefix: str = "desc_"):
    """Frozen-interface load + the mandated scan_id identity asserts (R4),
    then the lib.take spacing slice."""
    seq = pair["db_seq"] if side == "db" else pair["query_seq"]
    idx = pair["db_idx"] if side == "db" else pair["q_idx"]
    session_sids = (pair["db_session_scan_ids"] if side == "db"
                    else pair["q_session_scan_ids"])
    desc, sids, p = load_desc(pair["sensor"], seq, prefix=prefix)
    assert np.array_equal(sids, session_sids), (seq, "desc scan_ids")
    sliced = lib.take({"desc": desc}, idx)
    return sliced["desc"], rel_to_repo(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", choices=sorted(METHODS), default="imlpr",
                    help="frozen descriptor stack to evaluate")
    ap.add_argument("--protocols", nargs="*", default=list(lib.PROTOCOLS))
    ap.add_argument("--weights", type=Path, default=None,
                    help="official checkpoint (sha-logged only; descriptors "
                         "are consumed precomputed; default: the method's "
                         "release checkpoint)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run protocols whose output JSON exists")
    args = ap.parse_args()

    spec = METHODS[args.method]
    if args.weights is None:
        args.weights = spec["weights"]
    unknown = [p for p in args.protocols if p not in lib.PROTOCOLS]
    if unknown:
        raise SystemExit(f"unknown protocols: {unknown} "
                         f"(known: {sorted(lib.PROTOCOLS)})")
    if not args.weights.is_file():
        raise SystemExit(f"missing {spec['display']} checkpoint: {args.weights}")
    weights_id = sha256_head(args.weights)

    n_coarse = int(lib.CONST["n_coarse_baseline"])
    for protocol in args.protocols:
        name = f"xs_{args.method}_{protocol}"
        out_p = lib.OUT_DIR / f"{name}.json"
        if out_p.exists() and not args.overwrite:
            print(f"[skip] {out_p.name} exists", flush=True)
            continue
        pair = lib.make_pair(protocol)
        print(f"=== {protocol}: n_db={len(pair['db_pos'])} "
              f"n_spaced_q={len(pair['q_pos'])} n_roster={len(pair['roster'])} "
              f"roster_sha={pair['roster_sha256'][:12]}", flush=True)
        db_desc, db_cache = load_side(pair, "db", prefix=spec["prefix"])
        q_desc, q_cache = load_side(pair, "query", prefix=spec["prefix"])
        ranked = lib.rank_cross_session(db_desc, q_desc, pair["roster"],
                                        n_coarse=n_coarse)
        scored = lib.score_rankings(ranked, pair["db_pos"], pair["q_pos"],
                                    pair["roster"])
        overlap = {
            "db": (pair["sensor"], pair["db_seq"])
                in METHODS[args.method].get("overlap", TRAIN_OVERLAP),
            "query": (pair["sensor"], pair["query_seq"])
                in METHODS[args.method].get("overlap", TRAIN_OVERLAP),
        }
        payload = lib.result_payload(
            args.method, pair, scored,
            extra={"weights": str(args.weights),
                   "weights_sha256_1mb": weights_id,
                   "encoder": dict(spec["encoder"]),
                   "coarse": {"metric": "cosine", "n_coarse": n_coarse,
                              "rerank": None},
                   "desc_caches": [db_cache, q_cache],
                   "weight_convention": spec["convention"],
                   "train_overlap": overlap["db"] or overlap["query"],
                   "train_overlap_sessions": overlap,
                   "zero_shot_wrt_pd_train": True})
        p = lib.write_result(payload, name)
        m = payload["metrics"]
        print(f"  {name}: R@1@5m={m['R@1@5m']:.4f} "
              f"R@1@20m={m['R@1@20m']:.4f} "
              f"R1pct@20m={m['R1pct@20m']:.4f} -> {p.name}", flush=True)
    print(f"XS_{args.method.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main()
