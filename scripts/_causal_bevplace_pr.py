#!/usr/bin/env python3
"""All-frames causal PR ledger for BEVPlace++ (Track C baseline dump).

Track C of docs/CAUSAL_PR_PREREGISTRATION.md: per sensor/seq of the Strict-v2
37-sequence roster (scripts/_pd_test_cache_manifest.json minus kitti/00,
kitti360/0004, helipr/Bridge04, nclt/2012-02-04, nclt/2012-03-17), dump one
row per query frame t >= 30 with everything scripts/_pr_eval_lib.py needs
downstream — no PR accounting happens here.

Two stages per sequence (worker subprocess):
  1. GPU encode — per-frame 8192D BEVPlace++ descriptors at the cached
     scan_ids (baselines/bevplace.BEVPlaceBaseline, pd2026 retrain weights),
     computed once for the whole sequence up front.  CAUSAL-SAFE: each
     descriptor depends only on its own scan (prereg Track C: "per-frame
     methods reuse descriptors; only the search is re-run").
  2. Torch-free ledger — per-query gemv cosine search restricted to prefix
     rows j <= t - 30 (scripts/_causal_pr_dump.prefix_topk, imported: one
     matrix-vector product over rows [0, n_prefix), rows beyond the prefix
     never read, fixed accumulation order -> bit-identical under sequence
     truncation).  Exhaustive top-1 over the prefix — never
     global-search-then-filter.  O(N^2) dots per sequence; CPU-BLAS minutes
     per large sequence, no caps applied.

Native score (prereg §1.5, BEVPlace++ row): 8192D cosine of the prefix
top-1.  Already a similarity (higher = accept) — NO negation applied;
recorded in meta.

Output, one npz per sequence (schema pr-dump-v1):
  results/pd2026_pr/dump_bevplace_{sensor}_{seq}.npz
  arrays query_idx / j_star / score_cos / top1_geo_dist / min_past_gt_dist /
  n_candidates + json-string meta.  NaN score/dist on an empty candidate
  pool (with prefix j <= t - 30 the pool is provably non-empty for every
  t >= 30 — frame 0 is always eligible — so NaN rows are not expected).

MEMORY DISCIPLINE (scripts/_causal_pr_dump.py convention): the CLI accepts a
sequence list, but the driver runs each sequence in its own --worker
subprocess.  Resumable: existing dumps are skipped unless --overwrite.

Usage (inside docker ring_baseline; GPU may be busy — see --device cpu):
    python3 scripts/_causal_bevplace_pr.py --seqs mulran/DCC03
    python3 scripts/_causal_bevplace_pr.py            # all 37, resumable
    python3 scripts/_causal_bevplace_pr.py --selftest # CPU-only, no torch
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from _causal_pr_dump import prefix_topk  # noqa: E402  (torch-free)
from _pdb_bevplace_eval import (  # noqa: E402  (torch-free at import)
    MANIFEST, make_loader, sha256_head,
)
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402  (torch-free)

OUT = REPO / "results/pd2026_pr"
METHOD = "bevplace"
DEFAULT_WEIGHTS = REPO / "baselines/weights/bevplace_pd2026.pth"
STRICT_V2_EXCLUDED = ("kitti/00", "kitti360/0004", "helipr/Bridge04",
                      "nclt/2012-02-04", "nclt/2012-03-17")


def strict_v2_roster() -> dict:
    """{sensor/seq: operating-cache relpath} for the Strict-v2 37 sequences,
    manifest order (same cache source the _pdb_* drivers use)."""
    rows = json.load(open(MANIFEST))["sequences"]
    roster = {f"{s['sensor']}/{s['sequence']}": s["expected"]["operating_cache"]
              for s in rows
              if f"{s['sensor']}/{s['sequence']}" not in STRICT_V2_EXCLUDED}
    assert len(roster) == 37, f"Strict-v2 roster size {len(roster)} != 37"
    return roster


def ledger(desc: np.ndarray, positions: np.ndarray) -> dict:
    """Torch-free per-query prefix-cosine ledger (stage 2).

    desc: (n, D) float32 L2-normalized descriptors; positions: (n, 3).
    Per query t >= 30: exhaustive cosine top-1 over prefix rows [0, t - 30]
    via _causal_pr_dump.prefix_topk (gemv touches only prefix rows — R2).
    """
    n = len(desc)
    assert len(positions) == n, "descriptor/pose length mismatch"

    # min 3D distance from t to ANY frame <= t - 30 (query classification
    # input; per-query prefix scan — replicates _causal_pr_dump.run_worker).
    min_past = np.full(n, np.nan)
    for t in range(MIN_INDEX_GAP, n):
        d = np.linalg.norm(positions[:t - MIN_INDEX_GAP + 1] - positions[t],
                           axis=1)
        min_past[t] = d.min()

    queries = np.arange(MIN_INDEX_GAP, n, dtype=np.int64)
    j_star = np.full(len(queries), -1, dtype=np.int64)
    score_cos = np.full(len(queries), np.nan)
    top1_geo = np.full(len(queries), np.nan)
    n_cand = np.zeros(len(queries), dtype=np.int64)
    for qi, t in enumerate(queries):
        t = int(t)
        n_prefix = t - MIN_INDEX_GAP + 1  # prefix rows [0, t - 30]
        if n_prefix <= 0:
            continue  # empty candidate pool -> NaN row (not expected, t >= 30)
        j = int(prefix_topk(desc, t, n_prefix, 1)[0])
        assert j <= t - MIN_INDEX_GAP, "candidate beyond the causal prefix"
        j_star[qi] = j
        score_cos[qi] = float(desc[j] @ desc[t])  # cosine (rows L2-normed)
        top1_geo[qi] = float(np.linalg.norm(positions[j] - positions[t]))
        n_cand[qi] = n_prefix  # exhaustive pool = the whole prefix
        if qi and qi % 1000 == 0:
            print(f"  ledger queries {qi}/{len(queries)}", flush=True)
    return dict(query_idx=queries, j_star=j_star, score_cos=score_cos,
                top1_geo_dist=top1_geo, min_past_gt_dist=min_past[queries],
                n_candidates=n_cand)


def encode_sequence(sensor: str, seq: str, cache_rel: str, weights: Path):
    """Stage 1: BEVPlace++ descriptors at the cached scan_ids (mirrors
    scripts/_pdb_bevplace_eval.eval_sequence loading/encode exactly)."""
    cache_path = REPO / cache_rel
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        scan_ids = cache["scan_ids"].astype(np.int64)
        # Scoring uses ground-truth poses (poses_gt on odometry-keyframe caches).
        poses = (cache["poses_gt"] if "poses_gt" in cache.files
                 else cache["poses"]).astype(np.float64)
    assert len(scan_ids) == len(poses), (sensor, seq, "scan_ids/poses length")

    os.environ["NSD_BEVPLACE_WEIGHTS"] = str(weights)  # before import
    from baselines.bevplace import BEVPlaceBaseline
    encoder = BEVPlaceBaseline()
    if not encoder.is_available():
        raise SystemExit("BEVPlace++ dependencies or weights unavailable")

    loader = make_loader(sensor, seq)
    desc, encode_s = [], []
    print(f"[{sensor}/{seq}] encoding {len(scan_ids)} keyframes "
          f"from {cache_rel}", flush=True)
    for pos, sid in enumerate(scan_ids):
        pts = loader[int(sid)]["points"]
        t0 = time.perf_counter()
        desc.append(encoder.encode(pts))
        encode_s.append(time.perf_counter() - t0)
        if (pos + 1) % 500 == 0:
            print(f"  {pos + 1}/{len(scan_ids)} "
                  f"({np.mean(encode_s[-500:]) * 1e3:.1f} ms/kf)", flush=True)
    return (np.asarray(desc, dtype=np.float32), poses[:, :3, 3],
            float(np.mean(encode_s) * 1e3))


def run_worker(args, sensor: str, seq: str, cache_rel: str):
    out_path = OUT / f"dump_{METHOD}_{sensor}_{seq}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"[{sensor}/{seq}] {out_path.name} exists, skipping", flush=True)
        return
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # mask before torch import

    desc, positions, ms_per_kf = encode_sequence(
        sensor, seq, cache_rel, args.weights)
    arrays = ledger(desc, positions)

    meta = {
        "schema": "pr-dump-v1",
        "method": METHOD,
        "sensor": sensor, "sequence": seq,
        "min_index_gap": MIN_INDEX_GAP,
        "n_frames": int(len(desc)),
        "n_queries": int(len(arrays["query_idx"])),
        "encode_ms_per_kf": ms_per_kf,
        "device_requested": args.device,
        "artifacts": {
            "weights": str(args.weights),
            "weights_sha256_1mb": sha256_head(args.weights),
        },
        "caches": {"operating_cache": cache_rel},
        "encoder": {"model": "BEVPlace++ REIN (vendored official, pd2026 "
                             "retrain weights)",
                    "img_size": 200, "max_range_m": 50.0,
                    "descriptor_dim": 8192, "l2_normalized": True},
        "primary_score": "score_cos",
        "scores": {"score_cos": "8192D descriptor cosine of the exhaustive "
                                "prefix top-1; native similarity, higher = "
                                "accept; NOT negated (prereg §1.5 BEVPlace++ "
                                "score is already a similarity)"},
        "protocol": ("causal all-frames PR ledger (Track C, bevplace): "
                     "per-frame 8192D descriptors computed once per sequence "
                     "(each depends only on its own scan — causal-safe "
                     "precompute), then per-query gemv cosine search "
                     "restricted to prefix rows j <= t - 30 (exhaustive "
                     "top-1; rows beyond the prefix never read, no budget "
                     "caps); query set / accounting inputs per "
                     "docs/CAUSAL_PR_PREREGISTRATION.md §1.2/§1.3/§1.5"),
    }
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    fin = np.isfinite(arrays["score_cos"])
    print(f"[{sensor}/{seq}] n_queries={len(arrays['query_idx'])} "
          f"finite={int(fin.sum())} ({ms_per_kf:.1f} ms/kf) "
          f"-> {out_path.name}", flush=True)


def selftest():
    """CPU-only synthetic check of the torch-free ledger stage (no torch, no
    data): causality, exhaustive-prefix reference match, prefix invariance."""
    rng = np.random.default_rng(0)
    n = 120
    desc = rng.standard_normal((n, 16)).astype(np.float32)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True)
    positions = rng.uniform(0.0, 100.0, size=(n, 3))
    arrays = ledger(desc, positions)

    q = arrays["query_idx"]
    assert (q == np.arange(MIN_INDEX_GAP, n)).all(), "query set != all t >= 30"
    assert (arrays["j_star"] <= q - MIN_INDEX_GAP).all(), "causality violated"
    assert np.isfinite(arrays["score_cos"]).all(), "unexpected NaN score"
    assert (arrays["n_candidates"] == q - MIN_INDEX_GAP + 1).all()

    # Independent reference: exhaustive cosine over the prefix, argmax.
    for qi, t in enumerate(q):
        t = int(t)
        sims = desc[:t - MIN_INDEX_GAP + 1] @ desc[t]
        j_ref = int(np.argmax(sims))
        assert arrays["j_star"][qi] == j_ref, (t, arrays["j_star"][qi], j_ref)
        assert arrays["score_cos"][qi] == float(desc[j_ref] @ desc[t]), t
        d = np.linalg.norm(positions[:t - MIN_INDEX_GAP + 1] - positions[t],
                           axis=1)
        assert arrays["min_past_gt_dist"][qi] == d.min(), t

    # Prefix invariance (prereg §4.1 flavor): truncating the sequence at t
    # leaves query t's row bit-identical.
    for t in (30, 57, 90, n - 1):
        trunc = ledger(desc[:t + 1], positions[:t + 1])
        qi = t - MIN_INDEX_GAP
        for k in ("j_star", "score_cos", "top1_geo_dist",
                  "min_past_gt_dist", "n_candidates"):
            a, b = arrays[k][qi], trunc[k][qi]
            assert a == b or (np.isnan(a) and np.isnan(b)), (t, k, a, b)
    print("SELFTEST_OK: ledger is causal, matches the exhaustive-prefix "
          "reference, and is prefix-invariant", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                    help="BEVPlace++ pd2026 retrain checkpoint")
    ap.add_argument("--seqs", nargs="+", default=None, metavar="SENSOR/SEQ",
                    help="subset of the Strict-v2 37, e.g. mulran/DCC03 "
                         "(default: all 37)")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                    help="cpu masks CUDA before torch import (encode stage "
                         "only; the ledger stage is torch-free)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--worker", action="store_true",
                    help="internal: run a single sequence in-process")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only synthetic ledger check (no torch, no data)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    roster = strict_v2_roster()
    if args.seqs:
        unknown = sorted(set(args.seqs) - set(roster))
        if unknown:
            raise SystemExit(
                f"not in the Strict-v2 37-seq roster: {unknown} "
                f"(excluded by prereg R5: {list(STRICT_V2_EXCLUDED)})")
        todo = [(s, roster[s]) for s in args.seqs]
    else:
        todo = list(roster.items())
    if not args.weights.is_file():
        raise SystemExit(f"missing BEVPlace++ checkpoint: {args.weights}")
    OUT.mkdir(parents=True, exist_ok=True)

    if args.worker:
        assert len(todo) == 1, "--worker takes exactly one sequence"
        (key, cache_rel), = todo
        sensor, seq = key.split("/")
        run_worker(args, sensor, seq, cache_rel)
        return

    # Driver: one heavy sequence per subprocess (memory discipline).
    for key, cache_rel in todo:
        sensor, seq = key.split("/")
        out_path = OUT / f"dump_{METHOD}_{sensor}_{seq}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path.name} exists", flush=True)
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--weights", str(args.weights), "--device", args.device,
               "--seqs", key, "--worker"]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"=== spawning worker: {key} ===", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"WORKER_FAILED {key} rc={rc}", flush=True)
            sys.exit(rc)
    print("CAUSAL_BEVPLACE_PR_DONE", flush=True)


if __name__ == "__main__":
    main()
