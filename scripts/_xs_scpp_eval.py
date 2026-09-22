#!/usr/bin/env python3
"""SC++ (Scan Context++) cross-session evaluation (CPU, multiprocessing).

Implements the "RING-style cross-session protocol under a common keyframe
roster" (our roster construction from scripts/_xs_eval_lib.py — NOT the
official RING evaluation roster): map session spaced at 20 m, query session
at 5 m, fixed radius-20 m roster, R@k / R@1% via lib.score_rankings.

Method = the paper SC++ path, reused verbatim from scripts/_pdb_scpp_eval.py
(SC_PARAMS, loader wiring via _enc_init/_enc_one, cyclic-shift rerank workers
_rr_init/_rr_one, operating-cache resolver _resolve_cache), which itself wraps
baselines/scan_context.build_scan_context and
utils.cyclic_shift_distance.cyclic_column_cosine_distance. Coarse metric is
the same ring-key cosine as the intra-session driver (sc.mean(axis=2),
L2-normalized inner product) but computed DB-only through
lib.rank_cross_session (n_coarse=200); no skip window / self-exclusion —
DB and query are different sessions.

Pose single-source rule: cross-session positions come ONLY from
lib.make_pair (canonical global-GT keys_pd_s1_*_gp.npz). The pd2026 test
operating caches (scripts/_pd_test_cache_manifest.json) are read for their
scan_ids ONLY — their poses are per-session re-centred and are never touched.

Stage 1 (per FULL session, roster-independent, cached):
  results/pd2026_xsession/sc_{sensor}_{seq}.npz  (scan_ids + SC matrices)
Stage 2 (per protocol): R4 scan_id asserts -> lib.take slicing -> DB-only
ring-key coarse -> parallel cyclic column-shift rerank over the concatenated
[db; query] SC array (query row = n_db + roster index; candidates stay DB
rows) -> results/pd2026_xsession/xs_scpp_{protocol}.json.

Run inside docker ring_baseline:
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_xs_scpp_eval.py --workers 8
"""
from __future__ import annotations
import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

import _pdb_scpp_eval as pdb_scpp  # reused SC encoding / rerank machinery
import _xs_eval_lib as lib

REPO = Path(__file__).resolve().parents[1]
METHOD = "scpp"


def ensure_sc_cache(sensor: str, seq: str, workers: int) -> Path:
    """Stage 1: SC matrices for the FULL session keyframe roster (cached)."""
    out = lib.OUT_DIR / f"sc_{sensor}_{seq}.npz"
    if out.exists():
        return out
    import multiprocessing as mp

    cache_path = pdb_scpp._resolve_cache(sensor, seq)
    with np.load(cache_path) as c:
        # scan_ids ONLY — the cache poses are per-session re-centred (R3).
        scan_ids = c["scan_ids"].astype(np.int64)
    n = len(scan_ids)
    print(f"[sc {sensor}/{seq}] encoding {n} keyframes "
          f"({cache_path.name})", flush=True)

    scs = np.zeros((n, pdb_scpp.SC_PARAMS["n_rings"],
                    pdb_scpp.SC_PARAMS["n_sectors"]), dtype=np.float32)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=pdb_scpp._enc_init,
                                      initargs=(sensor, seq)) as pool:
        args = [(i, scan_ids[i]) for i in range(n)]
        for m, (idx, sc, _lm, _em) in enumerate(
                pool.imap_unordered(pdb_scpp._enc_one, args, chunksize=8)):
            scs[idx] = sc
            if (m + 1) % 1000 == 0:
                rate = (m + 1) / (time.time() - t0)
                print(f"  [sc {sensor}/{seq}] {m+1}/{n} ({rate:.1f}/s, "
                      f"ETA {(n-m-1)/rate/60:.1f} min)", flush=True)
    print(f"[sc {sensor}/{seq}] done in {time.time()-t0:.1f}s", flush=True)

    lib.OUT_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(lib.OUT_DIR), suffix=".npz.tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, scan_ids=scan_ids, sc=scs)
        os.replace(tmp, out)  # atomic: partial writes never look like a cache
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return out


def load_side(npz_path: Path, session_sids: np.ndarray, spaced_sids: np.ndarray,
              idx: np.ndarray) -> np.ndarray:
    """R4: assert full-session scan_id identity, then slice via lib.take."""
    with np.load(npz_path) as z:
        arrs = {"scan_ids": z["scan_ids"].copy(), "sc": z["sc"].copy()}
    assert np.array_equal(arrs["scan_ids"], session_sids), (
        npz_path.name, "scan_ids mismatch vs canonical session roster")
    sliced = lib.take(arrs, idx)
    assert np.array_equal(sliced["scan_ids"], spaced_sids), npz_path.name
    return sliced["sc"]


def run_protocol(protocol: str, workers: int, overwrite: bool):
    import multiprocessing as mp

    name = f"xs_{METHOD}_{protocol}"
    out_p = lib.OUT_DIR / f"{name}.json"
    if out_p.exists() and not overwrite:
        print(f"{name}: exists, skip", flush=True)
        return

    pair = lib.make_pair(protocol)
    sensor = pair["sensor"]
    print(f"=== {protocol}: n_db={len(pair['db_pos'])} "
          f"n_spaced_q={len(pair['q_pos'])} n_roster={len(pair['roster'])} "
          f"roster_sha={pair['roster_sha256'][:12]}", flush=True)

    db_npz = ensure_sc_cache(sensor, pair["db_seq"], workers)
    q_npz = ensure_sc_cache(sensor, pair["query_seq"], workers)
    db_sc = load_side(db_npz, pair["db_session_scan_ids"],
                      pair["db_scan_ids"], pair["db_idx"])
    q_sc = load_side(q_npz, pair["q_session_scan_ids"],
                     pair["q_scan_ids"], pair["q_idx"])
    n_db = len(db_sc)

    # Coarse: ring-key cosine (paper Eq.5, _pdb_scpp_eval), DB-only.
    n_coarse = int(lib.CONST["n_coarse_baseline"])
    assert n_coarse == pdb_scpp.N_COARSE  # same constant as the intra driver
    db_rk = db_sc.mean(axis=2).astype(np.float32)
    q_rk = q_sc.mean(axis=2).astype(np.float32)
    coarse = lib.rank_cross_session(db_rk, q_rk, pair["roster"],
                                    n_coarse=n_coarse)

    # Rerank: reused cyclic column-shift workers over concatenated [db; q]
    # SC matrices; query row = n_db + roster index, candidates stay DB rows.
    tasks = [(n_db + int(ri), cand)
             for ri, cand in zip(pair["roster"], coarse)]
    ranked_by_q = {}
    t0 = time.time()
    fd, sc_path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    try:
        np.save(sc_path, np.concatenate([db_sc, q_sc], axis=0))
        with mp.get_context("spawn").Pool(workers,
                                          initializer=pdb_scpp._rr_init,
                                          initargs=(sc_path,)) as pool:
            for m, (qi, ranked, _ms) in enumerate(
                    pool.imap_unordered(pdb_scpp._rr_one, tasks, chunksize=8)):
                ranked_by_q[qi] = ranked
                if (m + 1) % 500 == 0:
                    print(f"  [{protocol}] rerank {m+1}/{len(tasks)}",
                          flush=True)
    finally:
        os.unlink(sc_path)
    rerank_wall = time.time() - t0

    ranked_lists = []
    for ri, _cand in tasks:
        r = np.asarray(ranked_by_q[ri])
        assert np.all(r < n_db)
        ranked_lists.append(r)

    scored = lib.score_rankings(ranked_lists, pair["db_pos"], pair["q_pos"],
                                pair["roster"])
    payload = lib.result_payload(METHOD, pair, scored, extra={
        "coarse": {"metric": "ring-key cosine (sc.mean(axis=2), "
                             "L2-normalized inner product, DB-only)",
                   "n_coarse": n_coarse},
        "rerank": "utils.cyclic_shift_distance.cyclic_column_cosine_distance "
                  "(via _pdb_scpp_eval._rr_one)",
        "sc_params": pdb_scpp.SC_PARAMS,
        "sc_caches": {"db": str(db_npz.relative_to(REPO)),
                      "query": str(q_npz.relative_to(REPO))},
        "scan_id_source": "pd2026 test operating caches "
                          "(scripts/_pd_test_cache_manifest.json); "
                          "scan_ids only, cache poses never read",
        "timing": {"workers": workers,
                   "rerank_wall_s": round(rerank_wall, 2)},
    })
    p = lib.write_result(payload, name)
    m = payload["metrics"]
    print(f"  {name}: R@1@5m={m['R@1@5m']:.4f} "
          f"R@1@20m={m['R@1@20m']:.4f} "
          f"R1pct@20m={m['R1pct@20m']:.4f} -> {p.name}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocols", nargs="*", default=list(lib.PROTOCOLS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for protocol in args.protocols:
        run_protocol(protocol, args.workers, args.overwrite)
    print("XS_SCPP_DONE", flush=True)


if __name__ == "__main__":
    main()
