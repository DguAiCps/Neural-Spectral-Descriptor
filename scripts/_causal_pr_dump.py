#!/usr/bin/env python3
"""All-frames per-query score ledger for the pre-registered causal PR eval.

Track B of docs/CAUSAL_PR_PREREGISTRATION.md (§1.1 causal candidates, §1.2
all-frames query set, §1.5 confidence candidates): per (variant, seed,
sensor/seq, op) dump one row per query frame t >= 30 with everything
scripts/_pr_eval_lib.py needs downstream — no PR accounting happens here.

Retrieval per query t (PAST-ONLY, never global-then-filter):
  * candidates: two top-100 cosine pools over the prefix j <= t - 30
    (keys = causal two-pass 416D embedding from
    scripts/_causal_branch_fusion.nsd_embedding_causal; magnitude key per op
    via scripts/_xs_eval_lib.variant_inputs / hspec_inputs on the testspec
    caches), each pool a per-query gemv prefix search (the
    _causal_prefix_candidates pattern: one matrix-vector product over prefix
    rows, argpartition + stable descending sort — deterministic, bit-stable
    under truncation), then union; asserted <= t - 30.
  * scoring: per-query min-max 4-channel fusion at the --tuple weights using
    the torch-free _xs_eval_lib helpers (minmax / align_scores /
    column_align_scores index candidate subsets of the full banks — pairwise
    math, causal-safe).  Default tuple = the FROZEN bidirectional selection
    results/pd2026/tuple_selection.json ["pd/<op>"], a placeholder until the
    causal re-selection; tuple + source are echoed into the output meta.

Per-query row: j_star (fused argmax), score_margin (fused S1 - S2; 0.0 if a
single candidate), score_abs (raw 416D key cosine at the fused argmax — the
two §1.5 confidence candidates; the val stage picks one), top1_geo_dist (3D
pose distance of the returned match, the single 5 m correctness criterion),
min_past_gt_dist (min 3D distance to ANY frame <= t - 30, feeds
_pr_eval_lib.classify_queries), n_candidates.  An empty candidate pool dumps
NaN score/dist (with prefix j <= t - 30 the pool is provably non-empty for
every t >= 30 — frame 0 is always eligible — so NaN rows are not expected).

Output, one npz per (variant, seed, seq, op):
  results/pd2026_pr/dump_{variant}_{op}_s{seed}_{sensor}_{seq}.npz
  arrays query_idx / j_star / score_margin / score_abs / top1_geo_dist /
  min_past_gt_dist / n_candidates + json-string meta.

MEMORY DISCIPLINE (scripts/_pd_apply_frozen.py convention): the CLI accepts a
sequence list, but the driver runs each sequence in its own --worker
subprocess.  Caches resolve test_ > val_ via _pd_apply_frozen.resolve_caches.
"""
from __future__ import annotations
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _pd_apply_frozen import resolve_caches  # noqa: E402  (torch-free)
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402  (torch-free)
from _xs_eval_lib import (  # noqa: E402  (torch-free)
    align_scores, column_align_scores, hspec_inputs, minmax, variant_inputs,
)

OUT = REPO / "results/pd2026_pr"
TUPLES = REPO / "results/pd2026/tuple_selection.json"
OPS = ("pbev128_main", "hspec12")


def default_tuple(op: str):
    """Frozen bidirectional tuple as the pre-causal-reselection placeholder."""
    key = f"pd/{op}"
    sel = json.loads(TUPLES.read_text())[key]["selected_tuple"]
    w = ast.literal_eval(sel)
    assert isinstance(w, tuple) and len(w) == 4, sel
    return (tuple(float(x) for x in w),
            f"{TUPLES} [{key}] (frozen bidirectional placeholder)")


def prefix_topk(bank: np.ndarray, t: int, n_prefix: int, k: int) -> np.ndarray:
    """Per-query gemv prefix search (the _causal_prefix_candidates pattern):
    one matrix-vector product over rows [0, n_prefix), argpartition + stable
    descending sort.  Rows >= n_prefix are never touched (R2: no
    global-matrix-then-filter), and the per-row dot has a fixed accumulation
    order, so results are bit-identical under sequence truncation."""
    col = bank[:n_prefix] @ bank[t]
    k = min(k, n_prefix)
    if k < n_prefix:
        top = np.argpartition(-col, k - 1)[:k]
    else:
        top = np.arange(n_prefix)
    return top[np.argsort(-col[top], kind="stable")]


def dump_op(emb: np.ndarray, hkey: np.ndarray, hfft, hnorm, column_bank,
            positions: np.ndarray, min_past: np.ndarray, w, pool_k: int):
    """Per-query ledger arrays for one operating point (see module docstring)."""
    n = len(emb)
    queries = np.arange(MIN_INDEX_GAP, n, dtype=np.int64)
    j_star = np.full(len(queries), -1, dtype=np.int64)
    score_margin = np.full(len(queries), np.nan)
    score_abs = np.full(len(queries), np.nan)
    top1_geo = np.full(len(queries), np.nan)
    n_cand = np.zeros(len(queries), dtype=np.int64)
    for qi, t in enumerate(queries):
        t = int(t)
        n_prefix = t - MIN_INDEX_GAP + 1  # prefix rows [0, t - 30]
        if n_prefix <= 0:
            continue  # empty candidate pool -> NaN row (not expected, t >= 30)
        cn = prefix_topk(emb, t, n_prefix, pool_k)
        ch = prefix_topk(hkey, t, n_prefix, pool_k)
        cand = np.unique(np.concatenate([cn, ch]))
        assert cand.size and cand.max() <= t - MIN_INDEX_GAP, \
            "candidate beyond the causal prefix"
        sn = minmax(emb[cand] @ emb[t])
        sh = minmax(hkey[cand] @ hkey[t])
        sa = minmax(align_scores(hfft, hnorm, t, cand))
        sc = minmax(column_align_scores(column_bank, t, cand))
        score = w[0] * sn + w[1] * sh + w[2] * sa + w[3] * sc
        order = np.argsort(-score, kind="stable")
        j = int(cand[order[0]])
        j_star[qi] = j
        score_margin[qi] = (float(score[order[0]] - score[order[1]])
                            if cand.size >= 2 else 0.0)
        score_abs[qi] = float(emb[t] @ emb[j])  # raw key cosine (rows L2-normed)
        top1_geo[qi] = float(np.linalg.norm(positions[j] - positions[t]))
        n_cand[qi] = int(cand.size)
        if qi and qi % 500 == 0:
            print(f"  dump queries {qi}/{len(queries)}", flush=True)
    return dict(query_idx=queries, j_star=j_star, score_margin=score_margin,
                score_abs=score_abs, top1_geo_dist=top1_geo,
                min_past_gt_dist=min_past[queries], n_candidates=n_cand)


def run_worker(args, sensor: str, seq: str):
    sys.path.insert(0, str(REPO / "src"))
    from _causal_branch_fusion import (
        VARIANTS, make_model_causal, nsd_embedding_causal,
    )

    cache_p, lay_p, spec_p, marker = resolve_caches(sensor, seq)
    print(f"[{sensor}/{seq}] cache={cache_p.name} layouts={lay_p.name} "
          f"spec={spec_p.name} marker={marker}", flush=True)
    cache = np.load(cache_p)
    lay = np.load(lay_p)
    spec = np.load(spec_p)
    assert (lay["scan_ids"] == cache["scan_ids"]).all(), (sensor, seq, "layouts")
    assert (spec["scan_ids"] == cache["scan_ids"]).all(), (sensor, seq, "spec")
    d288 = cache["descriptors"].astype(np.float32)
    layouts = lay["nsd_layouts"].astype(np.float32)
    sk = spec["sk_re"].astype(np.float32) + 1j * spec["sk_im"].astype(np.float32)
    raw_height = spec["raw_height"].astype(np.float32)

    V = VARIANTS[args.variant]
    cfg, model, clf, clf_blob, r, ckpt = make_model_causal(
        args.variant, args.seed, args.device)
    emb = nsd_embedding_causal(cache, sensor, d288, layouts,
                               cfg, model, clf, clf_blob, r, args.device)
    n = len(emb)
    assert n == len(cache["poses"]), (sensor, seq, "embedding/pose length")

    # Scoring uses ground-truth poses; caches built on odometry keyframes
    # carry them separately as poses_gt (cache["poses"] = what the system sees).
    gt_poses = cache["poses_gt"] if "poses_gt" in cache.files else cache["poses"]
    positions = gt_poses[:, :3, 3]
    # min 3D distance from t to ANY frame <= t - 30 (query classification
    # input; per-query prefix scan — never a full-sequence statistic).
    min_past = np.full(n, np.nan)
    for t in range(MIN_INDEX_GAP, n):
        d = np.linalg.norm(positions[:t - MIN_INDEX_GAP + 1] - positions[t], axis=1)
        min_past[t] = d.min()

    for op in args.ops:
        out_path = OUT / (f"dump_{args.variant}_{op}_s{args.seed}_"
                          f"{sensor}_{seq}.npz")
        if out_path.exists() and not args.overwrite:
            print(f"[{sensor}/{seq}] {op}: {out_path.name} exists, skipping",
                  flush=True)
            continue
        per_op = {"pbev128_main": args.tuple_pbev, "hspec12": args.tuple_hspec}[op]
        if per_op is not None:
            w, tuple_source = (tuple(float(x) for x in per_op),
                               f"cli --tuple-{'pbev' if op == 'pbev128_main' else 'hspec'}")
        elif args.tuple is not None:
            w, tuple_source = tuple(float(x) for x in args.tuple), "cli --tuple"
        else:
            w, tuple_source = default_tuple(op)
        print(f"[{sensor}/{seq}] op={op} tuple={w} source={tuple_source}",
              flush=True)
        hkey, hfft, hnorm, column_bank = (
            variant_inputs(sk, 8) if op == "pbev128_main"
            else hspec_inputs(raw_height, 12))
        assert len(hkey) == n, (sensor, seq, op, "magnitude bank length")
        arrays = dump_op(emb, hkey, hfft, hnorm, column_bank,
                         positions, min_past, w, args.pool_k)
        meta = {
            "schema": "pr-dump-v1",
            "variant": args.variant, "op": op, "seed": args.seed,
            "sensor": sensor, "sequence": seq,
            "tuple": list(w), "tuple_source": tuple_source,
            "pool_k": args.pool_k, "min_index_gap": MIN_INDEX_GAP,
            "n_frames": n, "n_queries": int(len(arrays["query_idx"])),
            "device": args.device,
            "artifacts": {
                "checkpoint": str(ckpt), "rvec": str(V["rvec"]),
                "classifier": str(V["cls"]), "config": str(V["config"]),
            },
            "caches": {"cache": str(cache_p), "layouts": str(lay_p),
                       "spec": str(spec_p), "marker": marker},
            "protocol": ("causal all-frames PR ledger: causal two-pass "
                         "classifier-selected-edge embedding, per-query gemv "
                         "prefix candidates (two top-100 cosine pools over "
                         "j <= t - 30, union), per-query min-max 4-channel "
                         "fusion at the echoed tuple; scores/dists per "
                         "docs/CAUSAL_PR_PREREGISTRATION.md §1.2/§1.5"),
        }
        np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
        fin = np.isfinite(arrays["score_abs"])
        print(f"[{sensor}/{seq}] {op}: n_queries={len(arrays['query_idx'])} "
              f"finite={int(fin.sum())} -> {out_path.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("pd_causal", "pd"), required=True,
                    help="pd_causal = causal-retrained; pd = Frozen-GNN "
                         "causal adaptation")
    ap.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--seqs", nargs="+", required=True,
                    metavar="SENSOR/SEQ", help="e.g. helipr/Roundabout03")
    ap.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
    ap.add_argument("--tuple-pbev", nargs=4, type=float, default=None,
                    help="fusion tuple for op pbev128_main (overrides --tuple)")
    ap.add_argument("--tuple-hspec", nargs=4, type=float, default=None,
                    help="fusion tuple for op hspec12 (overrides --tuple)")
    ap.add_argument("--tuple", nargs=4, type=float, default=None,
                    metavar="W", help="fusion tuple override; requires "
                    "exactly one --ops entry (default: frozen bidirectional "
                    "tuple_selection.json placeholder)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-k", type=int, default=100)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--worker", action="store_true",
                    help="internal: run a single sequence in-process")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in args.seqs:
        assert s.count("/") == 1, f"--seqs entries must be sensor/seq, got {s!r}"
    if args.tuple is not None:
        assert len(args.ops) == 1, "--tuple applies to exactly one --ops entry"

    if args.worker:
        assert len(args.seqs) == 1, "--worker takes exactly one sequence"
        sensor, seq = args.seqs[0].split("/")
        run_worker(args, sensor, seq)
        return

    # Driver: one heavy sequence per subprocess (memory discipline).
    for s in args.seqs:
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--variant", args.variant, "--seed", str(args.seed),
               "--seqs", s, "--device", args.device,
               "--pool-k", str(args.pool_k), "--ops", *args.ops, "--worker"]
        if args.tuple is not None:
            cmd += ["--tuple", *[repr(x) for x in args.tuple]]
        if args.tuple_pbev is not None:
            cmd += ["--tuple-pbev", *[repr(x) for x in args.tuple_pbev]]
        if args.tuple_hspec is not None:
            cmd += ["--tuple-hspec", *[repr(x) for x in args.tuple_hspec]]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"=== spawning worker: {s} ===", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"WORKER_FAILED {s} rc={rc}", flush=True)
            sys.exit(rc)
    print("CAUSAL_PR_DUMP_DONE", flush=True)


if __name__ == "__main__":
    main()
