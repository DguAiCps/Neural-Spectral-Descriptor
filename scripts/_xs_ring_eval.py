#!/usr/bin/env python3
"""RING / RING++ cross-session evaluation (docker ring_baseline, GPU).

Implements the "RING-style cross-session protocol under a common keyframe
roster" (scripts/_xs_eval_lib.py; NOT the official RING roster): map session
at 20 m spacing, query session at 5 m, one fixed query roster reused at every
threshold. Reuses, without modifying, the official-pipeline pieces of
scripts/_pdb_ring_eval.py (compute_specs TIRING spectrum, GRID/BOUNDS, the
per-session BEV cache layout) and glnet.models.utils.estimate_yaw (official
circular-correlation scorer). Poses come ONLY from the canonical global-GT
source inside lib.make_pair (results/pd2026/keys_pd_s1_{sensor}_{seq}_gp.npz);
BEV-cache / operating-cache poses are never read.

Per protocol and per variant (ring = 1-ch occupancy BEV, ringpp = 6-ch LPD
feature BEV): TIRING specs for the spacing-sliced frames only, exhaustive
DB-only coarse (cosine over the angle-summed spectrum, n_coarse = n_db, so
the coarse order is irrelevant), rerank = official circular correlation over
the concatenated [db; query] spec array (query row = n_db + roster index,
candidates always < n_db).

BEV caches (_external/ringsharp_data/pdb_ring_cache/bev_{sensor}_{seq}.npz)
are opened READ-ONLY and must match the full-session scan_ids; this driver
NEVER builds or rewrites them (_pdb_ring_eval.compute_bevs overwrites on
scan_id mismatch and must not be called). Sliced specs are cached per session
as results/pd2026_xsession/xs_ringspec_{sensor}_{seq}.npz so the two
protocols per site reuse the shared db-session work.

Outputs: results/pd2026_xsession/xs_{ring,ringpp}_{protocol}.json

Run inside docker ring_baseline (repo at /workspace/Neural-Spectral-Codec):
  python3 scripts/_xs_ring_eval.py [--protocols p1 p2 ...] [--overwrite]
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import torch

import _xs_eval_lib as lib
import _pdb_ring_eval as ringmod  # reuse compute_specs/CACHE_DIR/GRID/BOUNDS
                                  # ONLY — never compute_bevs (cache clobber)

VARIANTS = ("ring", "ringpp")
BEV_KEY = {"ring": "bevs_ring", "ringpp": "bevs_ringpp"}


def load_sliced_bevs(sensor: str, seq: str, session_scan_ids: np.ndarray,
                     idx: np.ndarray) -> dict:
    """Read-only load of the shared per-session BEV cache, sliced to the
    spaced frames. R4: full-session scan_ids must match before slicing."""
    p = ringmod.CACHE_DIR / f"bev_{sensor}_{seq}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"missing BEV cache {p} — build it separately inside docker "
            f"ring_baseline (python3 scripts/_pdb_ring_eval.py {sensor}/{seq} "
            f"--bev-only); this driver never builds BEVs")
    with np.load(p) as c:
        if not np.array_equal(c["scan_ids"], session_scan_ids):
            raise AssertionError(
                f"{p.name}: cached scan_ids != full-session scan_ids from the "
                f"canonical pose source; refusing (this driver never rebuilds "
                f"the BEV cache)")
        sliced = lib.take({"scan_ids": c["scan_ids"],
                           "bevs_ring": c["bevs_ring"],
                           "bevs_ringpp": c["bevs_ringpp"]}, idx)
    return sliced


def session_specs(pair: dict, side: str, device: str):
    """TIRING specs (both variants) for one side's spaced frames, cached per
    session (keyed by the spacing-derived scan_ids, which depend only on the
    full session + frozen constants — roster-independent, so the two
    protocols per site share the db-session cache)."""
    sensor = pair["sensor"]
    seq = pair["db_seq"] if side == "db" else pair["query_seq"]
    idx = pair["db_idx"] if side == "db" else pair["q_idx"]
    session_sids = (pair["db_session_scan_ids"] if side == "db"
                    else pair["q_session_scan_ids"])
    spaced_sids = pair["db_scan_ids"] if side == "db" else pair["q_scan_ids"]
    spacing = lib.CONST["db_spacing_m" if side == "db" else "query_spacing_m"]

    cpath = lib.OUT_DIR / f"xs_ringspec_{sensor}_{seq}.npz"
    if cpath.exists():
        with np.load(cpath) as z:
            if np.array_equal(z["scan_ids"], spaced_sids):
                print(f"  spec cache hit: {cpath.name}", flush=True)
                return ({"ring": z["spec_ring"].copy(),
                         "ringpp": z["spec_ringpp"].copy()}, str(cpath))
        print(f"  spec cache {cpath.name} stale (scan_ids mismatch), "
              f"rebuilding", flush=True)

    bevs = load_sliced_bevs(sensor, seq, session_sids, idx)
    t0 = time.time()
    specs = {v: ringmod.compute_specs(bevs[BEV_KEY[v]], device).numpy()
             for v in VARIANTS}
    print(f"  [{sensor}_{seq}] specs for {len(spaced_sids)} spaced frames "
          f"({time.time() - t0:.0f}s)", flush=True)
    lib.OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = lib.OUT_DIR / f".tmp_xs_ringspec_{sensor}_{seq}.npz"
    np.savez_compressed(tmp, scan_ids=spaced_sids, spacing_m=float(spacing),
                        spec_ring=specs["ring"], spec_ringpp=specs["ringpp"])
    os.replace(tmp, cpath)
    return specs, str(cpath)


def coarse_desc(spec: np.ndarray) -> np.ndarray:
    """Replicates _pdb_ring_eval.eval_variant's coarse descriptor: spectrum
    summed over the angle axis, flattened (rank_cross_session normalizes)."""
    return spec.astype(np.float32).sum(axis=-2).reshape(len(spec), -1)


def make_rerank(specs_cat: np.ndarray, device: str, rr: dict):
    """Official circular-correlation rerank over [db; query] spec rows;
    mirrors _pdb_ring_eval.eval_variant.rerank_fn."""
    from glnet.models.utils import estimate_yaw  # official correlation

    def rerank_fn(qi, cand):
        tq = time.time()
        cand = np.asarray(cand)
        qspec = torch.from_numpy(specs_cat[qi].astype(np.float32)).to(device)
        dists = np.empty(len(cand), dtype=np.float64)
        bs = 256
        with torch.no_grad():
            for j0 in range(0, len(cand), bs):
                js = cand[j0:j0 + bs]
                cspec = torch.from_numpy(
                    specs_cat[js].astype(np.float32)).to(device)
                _, scores, _ = estimate_yaw(
                    qspec.unsqueeze(0).expand(len(js), -1, -1, -1), cspec)
                dists[j0:j0 + bs] = (1 - scores).cpu().numpy().reshape(-1)
        rr["s"] += time.time() - tq
        rr["n"] += 1
        return cand[np.argsort(dists)]

    return rerank_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocols", nargs="*", default=list(lib.PROTOCOLS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for protocol in args.protocols:
        out_paths = {v: lib.OUT_DIR / f"xs_{v}_{protocol}.json"
                     for v in VARIANTS}
        if not args.overwrite and all(p.exists() for p in out_paths.values()):
            print(f"=== {protocol}: both results exist, skip", flush=True)
            continue
        pair = lib.make_pair(protocol)
        n_db = len(pair["db_pos"])
        print(f"=== {protocol}: n_db={n_db} n_spaced_q={len(pair['q_pos'])} "
              f"n_roster={len(pair['roster'])} "
              f"roster_sha={pair['roster_sha256'][:12]}", flush=True)
        db_specs, db_cache = session_specs(pair, "db", device)
        q_specs, q_cache = session_specs(pair, "query", device)

        for variant in VARIANTS:
            name = f"xs_{variant}_{protocol}"
            if out_paths[variant].exists() and not args.overwrite:
                print(f"  {name}: exists, skip", flush=True)
                continue
            specs_cat = np.concatenate([db_specs[variant], q_specs[variant]])
            rr = {"s": 0.0, "n": 0}
            t0 = time.time()
            ranked = lib.rank_cross_session(
                coarse_desc(db_specs[variant]), coarse_desc(q_specs[variant]),
                pair["roster"], rerank_fn=make_rerank(specs_cat, device, rr),
                n_coarse=n_db)
            scored = lib.score_rankings(ranked, pair["db_pos"],
                                        pair["q_pos"], pair["roster"])
            payload = lib.result_payload(
                variant, pair, scored,
                extra={
                    "grid": ringmod.GRID, "bounds": list(ringmod.BOUNDS),
                    "bev_channels": 1 if variant == "ring" else 6,
                    "coarse": {"metric": "cosine over angle-summed TIRING "
                                         "spectrum (_pdb_ring_eval glob)",
                               "n_coarse": n_db, "exhaustive": True},
                    "rerank": "official RINGSharp estimate_yaw circular "
                              "correlation, dist = 1 - score",
                    "bev_cache": {
                        "db": str(ringmod.CACHE_DIR
                                  / f"bev_{pair['sensor']}_{pair['db_seq']}.npz"),
                        "query": str(ringmod.CACHE_DIR
                                     / f"bev_{pair['sensor']}_{pair['query_seq']}.npz")},
                    "spec_cache": {"db": db_cache, "query": q_cache},
                    "timing": {"rank_s": round(time.time() - t0, 1),
                               "rerank_ms_per_query":
                                   round(1e3 * rr["s"] / max(rr["n"], 1), 1)},
                })
            p = lib.write_result(payload, name)
            m = payload["metrics"]
            print(f"  {name}: R@1@5m={m['R@1@5m']:.4f} "
                  f"R@1@20m={m['R@1@20m']:.4f} "
                  f"R1pct@20m={m['R1pct@20m']:.4f} -> {p.name}", flush=True)
            del specs_cat, ranked
        del db_specs, q_specs
        if device == "cuda":
            torch.cuda.empty_cache()
    print("XS_RING_DONE", flush=True)


if __name__ == "__main__":
    main()
