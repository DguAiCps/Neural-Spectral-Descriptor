#!/usr/bin/env python3
"""BEVPlace++ cross-session evaluation (docker ring_baseline, GPU stage 1).

Implements the "RING-style cross-session protocol under a common keyframe
roster" (scripts/_xs_eval_lib.py — NOT the official RING roster): 20 m map /
5 m query spacing, fixed 20 m roster denominator at every threshold.

Reused modules (no existing file is modified):
  * scripts/_pdb_bevplace_eval — make_loader (raw-scan loader factory) and
    sha256_head (checkpoint sha-logging convention).
  * baselines/bevplace.BEVPlaceBaseline — per-frame encoder with the
    pd-retrained checkpoint baselines/weights/bevplace_pd2026.pth (selected
    via NSD_BEVPLACE_WEIGHTS before import, as the intra driver does).
    Rotation invariance is baked into encode() (REM 8-rotation max + NetVLAD);
    the intra driver applies no additional query-time trick, so cross-session
    matching is plain cosine over the same descriptors.
  * scripts/_xs_eval_lib — pairing, DB-only cosine ranking, fixed-roster
    scoring, payload/writing.

Pose single source: positions come ONLY from lib.make_pair (canonical
global-GT results/pd2026/keys_pd_s1_{sensor}_{seq}_gp.npz). Poses are never
read from pd2026 operating caches, method artifacts, or BEV cache metadata.

Stage 1 (per session, cached, roster-independent): descriptors for the FULL
session keyframe roster -> results/pd2026_xsession/
desc_bevplace_{sensor}_{seq}.npz (desc + scan_ids + weights sha). Skip if the
npz exists (delete it to re-encode).
Stage 2 (per protocol): scan_id identity asserts vs the canonical session,
lib.take spacing slice, plain cosine lib.rank_cross_session (DB-only,
n_coarse=200, no rerank), lib.score_rankings ->
results/pd2026_xsession/xs_bevplace_{protocol}.json.

Usage (inside docker ring_baseline, repo at /workspace/Neural-Spectral-Codec):
    python3 scripts/_xs_bevplace_eval.py [--protocols dcc_01_02 ...] [--overwrite]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import _xs_eval_lib as lib
import _pdb_bevplace_eval as pdbev  # make_loader / sha256_head reuse only

DEFAULT_WEIGHTS = REPO / "baselines/weights/bevplace_pd2026.pth"
# BEVPlaceBaseline defaults, identical to the intra driver's encoder config.
ENCODER_PARAMS = {"img_size": 200, "max_range_m": 50.0, "descriptor_dim": 8192}


def desc_path(sensor: str, seq: str) -> Path:
    return lib.OUT_DIR / f"desc_bevplace_{sensor}_{seq}.npz"


def ensure_session_descs(sensor: str, seq: str, get_encoder,
                         weights_id: str) -> Path:
    """Stage 1: FULL-session descriptor dump (roster-independent cache)."""
    p = desc_path(sensor, seq)
    if p.exists():
        return p
    scan_ids = lib.load_global_session(sensor, seq)["scan_ids"].astype(np.int64)
    encoder = get_encoder()
    loader = pdbev.make_loader(sensor, seq)
    desc = []
    encode_s = []
    print(f"[{sensor}/{seq}] encoding {len(scan_ids)} keyframes", flush=True)
    for pos, sid in enumerate(scan_ids):
        pts = loader[int(sid)]["points"]
        t0 = time.perf_counter()
        desc.append(encoder.encode(pts))
        encode_s.append(time.perf_counter() - t0)
        if (pos + 1) % 500 == 0:
            print(f"  {pos + 1}/{len(scan_ids)} "
                  f"({np.mean(encode_s[-500:]) * 1e3:.1f} ms/kf)", flush=True)
    lib.OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f, desc=np.asarray(desc, dtype=np.float32), scan_ids=scan_ids,
            weights_sha256_1mb=weights_id,
            encode_ms_per_kf=float(np.mean(encode_s) * 1e3))
    os.replace(tmp, p)
    print(f"  -> {p} ({np.mean(encode_s) * 1e3:.1f} ms/kf)", flush=True)
    return p


def load_side(pair: dict, side: str, weights_id: str):
    """Stage 2 slicing with the mandated scan_id identity asserts (R4)."""
    seq = pair["db_seq"] if side == "db" else pair["query_seq"]
    idx = pair["db_idx"] if side == "db" else pair["q_idx"]
    session_sids = (pair["db_session_scan_ids"] if side == "db"
                    else pair["q_session_scan_ids"])
    p = desc_path(pair["sensor"], seq)
    with np.load(p) as z:
        desc = z["desc"].astype(np.float32)
        sids = z["scan_ids"].copy()
        cached_id = str(z["weights_sha256_1mb"])
        encode_ms = float(z["encode_ms_per_kf"])
    assert cached_id == weights_id, \
        (p.name, "stale descriptor cache: weights sha mismatch", cached_id)
    assert np.array_equal(sids, session_sids), (seq, "desc scan_ids")
    sliced = lib.take({"desc": desc}, idx)
    return sliced["desc"], str(p), encode_ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocols", nargs="*", default=list(lib.PROTOCOLS))
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run protocols whose output JSON exists "
                         "(descriptor caches are reused; delete the npz to "
                         "re-encode)")
    args = ap.parse_args()

    unknown = [p for p in args.protocols if p not in lib.PROTOCOLS]
    if unknown:
        raise SystemExit(f"unknown protocols: {unknown} "
                         f"(known: {sorted(lib.PROTOCOLS)})")
    if not args.weights.is_file():
        raise SystemExit(f"missing BEVPlace++ checkpoint: {args.weights}")
    weights_id = pdbev.sha256_head(args.weights)

    encoder = None

    def get_encoder():
        nonlocal encoder
        if encoder is None:
            os.environ["NSD_BEVPLACE_WEIGHTS"] = str(args.weights)
            from baselines.bevplace import BEVPlaceBaseline
            encoder = BEVPlaceBaseline()
            if not encoder.is_available():
                raise SystemExit("BEVPlace++ dependencies or weights unavailable")
        return encoder

    n_coarse = int(lib.CONST["n_coarse_baseline"])
    for protocol in args.protocols:
        name = f"xs_bevplace_{protocol}"
        out_p = lib.OUT_DIR / f"{name}.json"
        if out_p.exists() and not args.overwrite:
            print(f"[skip] {out_p.name} exists", flush=True)
            continue
        pair = lib.make_pair(protocol)
        print(f"=== {protocol}: n_db={len(pair['db_pos'])} "
              f"n_spaced_q={len(pair['q_pos'])} n_roster={len(pair['roster'])} "
              f"roster_sha={pair['roster_sha256'][:12]}", flush=True)
        for seq in (pair["db_seq"], pair["query_seq"]):
            ensure_session_descs(pair["sensor"], seq, get_encoder, weights_id)
        db_desc, db_cache, db_ms = load_side(pair, "db", weights_id)
        q_desc, q_cache, q_ms = load_side(pair, "query", weights_id)
        ranked = lib.rank_cross_session(db_desc, q_desc, pair["roster"],
                                        n_coarse=n_coarse)
        scored = lib.score_rankings(ranked, pair["db_pos"], pair["q_pos"],
                                    pair["roster"])
        payload = lib.result_payload(
            "bevplace", pair, scored,
            extra={"weights": str(args.weights),
                   "weights_sha256_1mb": weights_id,
                   "encoder": dict(ENCODER_PARAMS),
                   "coarse": {"metric": "cosine", "n_coarse": n_coarse,
                              "rerank": None},
                   "desc_caches": [db_cache, q_cache],
                   "encode_ms_per_kf": {"db": db_ms, "query": q_ms}})
        p = lib.write_result(payload, name)
        m = payload["metrics"]
        print(f"  {name}: R@1@5m={m['R@1@5m']:.4f} "
              f"R@1@20m={m['R@1@20m']:.4f} "
              f"R1pct@20m={m['R1pct@20m']:.4f} -> {p.name}", flush=True)
    print("XS_BEVPLACE_DONE", flush=True)


if __name__ == "__main__":
    main()
