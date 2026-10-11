#!/usr/bin/env python3
"""SC++ ALL-FRAMES per-query ledger for the pre-registered causal PR eval.

Track C of docs/CAUSAL_PR_PREREGISTRATION.md (baselines under the causal
protocol): per sensor/seq dump one row per query frame t >= 30 with what
scripts/_pr_eval_lib.py needs downstream — no PR accounting happens here.
Output schema and worker discipline mirror scripts/_causal_pr_dump.py.

Pipeline (native SC++ score per prereg §1.5, causal candidates per §1.1):
  * encoding: Scan Context matrices (20x60, max_range 80 m, z_min -3 m) for
    the WHOLE sequence up front via the imported _pdb_scpp_eval encode Pool
    workers — causal-legit: each frame's SC depends only on that frame's
    points, no cross-frame statistic.
  * coarse (PAST-ONLY, never global-then-filter): ring keys = per-ring mean
    of the SC matrix (row-local), L2-normalized in numpy (row-local; the
    faiss.normalize_L2 math of the intra path), then per query t a
    _causal_pr_dump.prefix_topk gemv over prefix rows j <= t - 30 only,
    k = min(200, |prefix|).  Rows > t - 30 are never touched, per-row math is
    row-local, so candidates are bit-stable under sequence truncation.
  * rerank: the imported _pdb_scpp_eval._rr_one cyclic column-cosine worker
    over the full-sequence SC bank, restricted to the coarse candidate list —
    pairwise (t, j) math only, causal-safe.
  * score_scpp = NEGATED cyclic_column_cosine_distance of the emitted top-1
    (higher = accept; negation recorded in meta).  The top-1 distance is
    recomputed in the parent with the same imported function on the same
    matrices (_rr_one returns ranked order only) — deterministic, exact.

Per-query row: j_star (emitted top-1), score_scpp, top1_geo_dist (3D pose
distance of the returned match), min_past_gt_dist (min 3D distance from t to
ANY frame <= t - 30 — per-query prefix scan, the _causal_pr_dump loop),
n_candidates.  With prefix j <= t - 30 the pool is non-empty for every
t >= 30 (frame 0 always eligible), so NaN rows are not expected; the arrays
still initialize NaN defensively.

Keyframes: scan_ids + poses from the frozen pd2026 TEST operating caches via
the imported _pdb_scpp_eval._resolve_cache (same source as the _pdb_*
drivers).  Raw clouds re-loaded at those scan_ids through the repo loaders.

Output, one npz per sequence (schema "pr-dump-v1"):
  results/pd2026_pr/dump_scpp_{sensor}_{seq}.npz
  arrays query_idx / j_star / score_scpp / top1_geo_dist / min_past_gt_dist /
  n_candidates + json-string meta.

CPU-only (numpy; no faiss, no torch).  One heavy sequence per invocation via
the chain (scripts/_causal_baseline_chain.sh); resumable per sequence.
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_causal_scpp_pr.py --seqs mulran/DCC03 --workers 8

Self-test (synthetic SC matrices, no data / no loaders):
  python3 scripts/_causal_scpp_pr.py --self-test
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _causal_pr_dump import prefix_topk  # noqa: E402  (torch-free)
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402  (torch-free)
from _pdb_scpp_eval import (  # noqa: E402  (torch-free at module level)
    N_COARSE, SC_PARAMS, SKIP, _enc_init, _enc_one, _resolve_cache,
    _rr_init, _rr_one,
)

OUT = REPO / "results/pd2026_pr"
assert SKIP == MIN_INDEX_GAP == 30, (SKIP, MIN_INDEX_GAP)


def dump_ledger(scs: np.ndarray, positions: np.ndarray, workers: int,
                queries=None, n_coarse: int = N_COARSE) -> dict:
    """Per-query ledger arrays (see module docstring).  `queries` defaults to
    all t >= 30; the self-test passes a subset for truncation checks."""
    from utils.cyclic_shift_distance import cyclic_column_cosine_distance
    import multiprocessing as mp

    n = len(scs)
    assert len(positions) == n, (n, len(positions))
    # Ring keys (paper Eq.5, the _pdb_scpp_eval line): row-local, so each
    # frame's key is invariant to sequence truncation.
    rks = scs.mean(axis=2).astype(np.float32)
    rks /= np.maximum(np.linalg.norm(rks, axis=1, keepdims=True), 1e-12)

    if queries is None:
        queries = np.arange(MIN_INDEX_GAP, n, dtype=np.int64)
    queries = np.asarray(queries, dtype=np.int64)
    assert queries.size, "no queries (sequence shorter than MIN_INDEX_GAP?)"
    assert queries.min() >= MIN_INDEX_GAP and queries.max() < n, \
        (int(queries.min()), int(queries.max()), n)

    # Coarse: per-query gemv prefix search (R2: rows beyond t - 30 never
    # touched — no global-search-then-filter).
    n_cand = np.zeros(len(queries), dtype=np.int64)
    tasks = []
    for qi, t in enumerate(queries):
        t = int(t)
        n_prefix = t - MIN_INDEX_GAP + 1  # prefix rows [0, t - 30]
        cand = prefix_topk(rks, t, n_prefix, n_coarse)
        assert cand.size and cand.max() <= t - MIN_INDEX_GAP, \
            "candidate beyond the causal prefix"
        n_cand[qi] = cand.size
        tasks.append((t, cand.astype(np.int64)))

    # Rerank: imported _rr_one over a temp .npy of the SC bank (the
    # _pdb_scpp_eval stage-2 pattern), pairwise (t, j) only.
    ranked_by_t, rr_ms = {}, []
    t1 = time.time()
    fd, sc_path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(sc_path, scs)
    try:
        with mp.get_context("spawn").Pool(workers, initializer=_rr_init,
                                          initargs=(sc_path,)) as pool:
            for m, (t, ranked, ms) in enumerate(
                    pool.imap_unordered(_rr_one, tasks, chunksize=8)):
                ranked_by_t[t] = ranked
                rr_ms.append(ms)
                if (m + 1) % 500 == 0:
                    print(f"  rerank {m+1}/{len(tasks)}", flush=True)
    finally:
        os.unlink(sc_path)
    rerank_wall = time.time() - t1

    j_star = np.full(len(queries), -1, dtype=np.int64)
    score_scpp = np.full(len(queries), np.nan)
    top1_geo = np.full(len(queries), np.nan)
    min_past = np.full(len(queries), np.nan)
    for qi, t in enumerate(queries):
        t = int(t)
        j = int(ranked_by_t[t][0])
        j_star[qi] = j
        # SCORE DIRECTION: similarity convention — negated native distance.
        score_scpp[qi] = -cyclic_column_cosine_distance(scs[t], scs[j])
        top1_geo[qi] = float(np.linalg.norm(positions[j] - positions[t]))
        # min 3D distance from t to ANY frame <= t - 30 (query classification
        # input; per-query prefix scan — never a full-sequence statistic).
        d = np.linalg.norm(positions[:t - MIN_INDEX_GAP + 1] - positions[t],
                           axis=1)
        min_past[qi] = float(d.min())
    return dict(arrays=dict(query_idx=queries, j_star=j_star,
                            score_scpp=score_scpp, top1_geo_dist=top1_geo,
                            min_past_gt_dist=min_past, n_candidates=n_cand),
                rerank_wall_s=rerank_wall,
                worker_rerank_ms=float(np.mean(rr_ms)) if rr_ms else 0.0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def run_sequence(sensor: str, seq: str, workers: int, out_dir: Path,
                 overwrite: bool, limit: int = 0):
    import multiprocessing as mp

    out_name = (f"dump_scpp_{sensor}_{seq}.npz" if not limit
                else f"smoke_limit{limit}_dump_scpp_{sensor}_{seq}.npz")
    out_path = out_dir / out_name
    if out_path.exists() and not overwrite:
        print(f"[{sensor}/{seq}] {out_path.name} exists, skipping", flush=True)
        return

    cache_path = _resolve_cache(sensor, seq)
    with np.load(cache_path) as c:
        scan_ids = c["scan_ids"].astype(np.int64)
        # Scoring uses ground-truth poses (poses_gt on odometry-keyframe caches).
        poses = (c["poses_gt"] if "poses_gt" in c.files
                 else c["poses"]).astype(np.float64)
    if limit:
        scan_ids, poses = scan_ids[:limit], poses[:limit]
    n = len(scan_ids)
    positions = poses[:, :3, 3]
    print(f"[{sensor}/{seq}] {n} keyframes ({cache_path.name})", flush=True)

    # Stage 1: whole-sequence SC encoding (imported _pdb_scpp_eval workers;
    # per-frame only, causal-legit).
    scs = np.zeros((n, SC_PARAMS["n_rings"], SC_PARAMS["n_sectors"]),
                   dtype=np.float32)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_enc_init,
                                      initargs=(sensor, seq)) as pool:
        args = [(i, scan_ids[i]) for i in range(n)]
        for m, (idx, sc, _lm, _em) in enumerate(
                pool.imap_unordered(_enc_one, args, chunksize=8)):
            scs[idx] = sc
            if (m + 1) % 1000 == 0:
                rate = (m + 1) / (time.time() - t0)
                print(f"  [{sensor}/{seq}] sc {m+1}/{n} ({rate:.1f}/s)",
                      flush=True)
    encode_wall = time.time() - t0

    # Stage 2+3: causal coarse + rerank + ledger.
    led = dump_ledger(scs, positions, workers)
    arrays = led["arrays"]
    n_q = len(arrays["query_idx"])

    meta = {
        "schema": "pr-dump-v1",
        "method": "scpp",
        "sensor": sensor, "sequence": seq,
        "n_frames": n, "n_queries": n_q,
        "min_index_gap": MIN_INDEX_GAP,
        "n_coarse": N_COARSE, "sc_params": SC_PARAMS, "limit": limit,
        "score_negation": (
            "score_scpp = -cyclic_column_cosine_distance(SC_t, SC_j*) of the "
            "reranked top-1 (native SC++ distance, prereg §1.5, stored "
            "NEGATED: higher = accept)"),
        "artifacts": {
            "operating_cache": str(cache_path.relative_to(REPO)),
            "operating_cache_sha256": _sha256(cache_path),
            "weights": None,  # SC++ is training-free
        },
        "code": {
            "encode_worker": "scripts/_pdb_scpp_eval.py:_enc_one (imported)",
            "rerank_worker": "scripts/_pdb_scpp_eval.py:_rr_one (imported)",
            "coarse": "scripts/_causal_pr_dump.py:prefix_topk (imported)",
            "top1_distance": "src/utils/cyclic_shift_distance.py:"
                             "cyclic_column_cosine_distance (parent recompute "
                             "of the emitted top-1 pair)",
        },
        "protocol": (
            "causal all-frames SC++ PR ledger: whole-sequence per-frame SC "
            "matrices (frame-local), ring-key cosine top-min(200,|prefix|) "
            "searched WITHIN the prefix j <= t - 30 (per-query gemv, never "
            "global-then-filter), cyclic column-cosine rerank of the "
            "candidates, score = negated distance of the emitted top-1; "
            "queries = all t >= 30 incl. no-GT frames; "
            "docs/CAUSAL_PR_PREREGISTRATION.md §1.1/§1.2/§1.5"),
        "deviation_from_intra": (
            "vs scripts/_pdb_scpp_eval.py: coarse search is a per-query "
            "numpy gemv over prefix ring keys (L2-normalized in numpy) "
            "instead of global FAISS search + |i-j|>30 filter; SC encoding "
            "and cyclic rerank are the identical imported workers; no "
            "candidate budget beyond the prereg n_coarse=200"),
        "timing": {
            "workers": workers,
            "encode_wall_s": round(encode_wall, 2),
            "rerank_wall_s": round(led["rerank_wall_s"], 2),
            "rerank_wall_ms_per_query": round(
                led["rerank_wall_s"] * 1e3 / max(n_q, 1), 3),
            "worker_rerank_ms_per_query": round(led["worker_rerank_ms"], 3),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    fin = int(np.isfinite(arrays["score_scpp"]).sum())
    print(f"[{sensor}/{seq}] n_queries={n_q} finite={fin} -> {out_path.name}",
          flush=True)


# ------------------------------------------------------------- self-test

def self_test():
    """CPU-safe synthetic check: schema, causal bound, negation, planted
    loops, min_past_gt_dist, and prefix invariance under truncation."""
    from utils.cyclic_shift_distance import cyclic_column_cosine_distance

    rng = np.random.default_rng(0)
    n = 240
    scs = rng.random((n, SC_PARAMS["n_rings"], SC_PARAMS["n_sectors"]),
                     dtype=np.float32)
    scs[200] = scs[50]   # planted loop, prefix fully within the top-200 pool
    scs[235] = scs[10]   # planted loop with |prefix| = 206 > 200 (pool cap on)
    positions = np.cumsum(rng.normal(0, 2.0, (n, 3)), axis=0)

    led = dump_ledger(scs, positions, workers=2)
    a = led["arrays"]
    q = a["query_idx"]
    assert (q == np.arange(MIN_INDEX_GAP, n)).all(), "query set != all t >= 30"
    assert np.isfinite(a["score_scpp"]).all(), "unexpected NaN score"
    assert (a["j_star"] <= q - MIN_INDEX_GAP).all(), "causality violated"
    assert (a["n_candidates"] ==
            np.minimum(N_COARSE, q - MIN_INDEX_GAP + 1)).all(), "pool sizes"
    for t, j_true in ((200, 50), (235, 10)):
        qi = int(t - MIN_INDEX_GAP)
        assert a["j_star"][qi] == j_true, (t, int(a["j_star"][qi]))
        assert a["score_scpp"][qi] > -1e-6, float(a["score_scpp"][qi])
    for qi in (0, 100, 205):  # negation + geo/min_past spot checks
        t, j = int(q[qi]), int(a["j_star"][qi])
        assert a["score_scpp"][qi] == -cyclic_column_cosine_distance(
            scs[t], scs[j]), "score is not the negated native distance"
        assert a["top1_geo_dist"][qi] == float(
            np.linalg.norm(positions[j] - positions[t]))
        d = np.linalg.norm(positions[:t - MIN_INDEX_GAP + 1] - positions[t],
                           axis=1)
        assert a["min_past_gt_dist"][qi] == float(d.min())
    # Prefix invariance (prereg §4.1): truncating the sequence at t leaves
    # query t's candidates, top-1 and score bit-identical.
    for t in (60, 150, 235):
        qi = int(t - MIN_INDEX_GAP)
        tr = dump_ledger(scs[:t + 1], positions[:t + 1], workers=2,
                         queries=np.array([t]))["arrays"]
        assert tr["j_star"][0] == a["j_star"][qi], f"top-1 changed at t={t}"
        assert tr["score_scpp"][0] == a["score_scpp"][qi], \
            f"score changed at t={t}"
        assert tr["n_candidates"][0] == a["n_candidates"][qi]
    print("SELF_TEST_OK: schema, causal bound, negation, planted loops, "
          "min_past, prefix invariance", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seqs", nargs="+", metavar="SENSOR/SEQ",
                    help="e.g. mulran/DCC03 (one heavy sequence per "
                         "invocation via the chain)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N keyframes; writes a smoke_-"
                         "prefixed npz, never the canonical ledger name")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic CPU-safe check, no data needed")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.seqs:
        ap.error("--seqs is required (or --self-test)")
    for s in args.seqs:
        assert s.count("/") == 1, f"--seqs entries must be sensor/seq: {s!r}"
    for s in args.seqs:
        sensor, seq = s.split("/")
        run_sequence(sensor, seq, args.workers, Path(args.out_dir),
                     args.overwrite, limit=args.limit)
    print("CAUSAL_SCPP_PR_DONE", flush=True)


if __name__ == "__main__":
    main()
