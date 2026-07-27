#!/usr/bin/env python3
"""DAC-strengthened height-spectral fusion: give the learned trajectory-context
code its own train-calibrated channel instead of leaving it buried inside f.

Channels: s_f (full 416D refined key) / s_H (|p_H| height-spectral magnitudes)
/ s_c (128D DAC context block alone) / s_col (band-limited column alignment
from the same p_H coefficients). The global-align slot of the standard grid is
repurposed for s_c -- train selection always assigned it weight 0, so nothing
measured is lost. Stored state unchanged (c is already part of f)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _height_candidate_fusion import (
    JOBS, WEIGHTS, CACHE_OUT, RESULT_OUT, bin288, l2, minmax,
    make_model, nsd_embedding, column_align_bank, column_align_scores,
)
from _hspec_branch_fusion import hspec_inputs, VAL_TAGS, TRAIN_TAGS
from run_kitti_operating_point import _find_queries, _topk_cosine

DTH, SKIP = 5.0, 30


def evaluate_ctx(poses, nsd, ctx, hkey, column_bank, pool_k):
    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    correct = {str(w): 0 for w in WEIGHTS}
    correct.update({"nsd": 0, "height": 0, "ctx": 0, "oracle_union": 0})
    for qi, (q, _) in enumerate(queries):
        cn = _topk_cosine(nsd, q, pool_k + 2 * SKIP, SKIP)[:pool_k]
        ch = _topk_cosine(hkey, q, pool_k + 2 * SKIP, SKIP)[:pool_k]
        candidates = np.unique(np.concatenate([cn, ch]))
        good = np.linalg.norm(positions[candidates] - positions[q], axis=1) < DTH
        correct["oracle_union"] += bool(good.any())
        sn = minmax(nsd[candidates] @ nsd[q])
        sh = minmax(hkey[candidates] @ hkey[q])
        sc = minmax(ctx[candidates] @ ctx[q])
        sd = minmax(column_align_scores(column_bank, q, candidates))
        for w in WEIGHTS:
            score = w[0] * sn + w[1] * sh + w[2] * sc + w[3] * sd
            correct[str(w)] += bool(good[int(np.argmax(score))])
        correct["nsd"] += np.linalg.norm(
            positions[_topk_cosine(nsd, q, 1 + 2 * SKIP, SKIP)[0]] - positions[q]) < DTH
        correct["height"] += np.linalg.norm(
            positions[_topk_cosine(hkey, q, 1 + 2 * SKIP, SKIP)[0]] - positions[q]) < DTH
        correct["ctx"] += np.linalg.norm(
            positions[_topk_cosine(ctx, q, 1 + 2 * SKIP, SKIP)[0]] - positions[q]) < DTH
        if qi and qi % 1000 == 0:
            print(f"  ctx-fusion queries {qi}/{len(queries)}", flush=True)
    n = max(len(queries), 1)
    return {k: float(v / n) for k, v in correct.items()} | {"n_queries": len(queries)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=VAL_TAGS + TRAIN_TAGS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-k", type=int, default=100)
    ap.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--freqs", type=int, default=12)
    args = ap.parse_args()
    cfg, model, clf, clf_blob = make_model(args.device, args.seed)
    suffix = "" if args.seed == 1 else f"_seed{args.seed}"
    result_path = RESULT_OUT / f"hspec_ctx_fusion{suffix}.json"
    report = json.loads(result_path.read_text()) if result_path.exists() else {}

    for tag in args.tags:
        name = f"hspec{args.freqs}_ctx"
        if tag in report and name in report[tag].get("variants", {}):
            print(f"[{tag}] already done, skipping", flush=True)
            continue
        sensor, sequence, cache_path = JOBS[tag]
        print(f"\n=== {tag} ===", flush=True)
        cache = np.load(REPO / cache_path)
        d288 = bin288(cache["fft_magnitudes"].astype(np.float32))
        hc = np.load(CACHE_OUT / f"{tag}_height120.npz")
        raw_height = hc["raw_height"].astype(np.float32)
        layouts = hc["nsd_layout"].astype(np.float32)
        nsd = nsd_embedding(cache, sensor, d288, layouts,
                            cfg, model, clf, clf_blob, args.device)
        ctx = l2(nsd[:, 288:])                    # DAC context block alone
        hkey, _, _, column_bank = hspec_inputs(raw_height, args.freqs)
        vals = evaluate_ctx(cache["poses"], nsd, ctx, hkey, column_bank,
                            args.pool_k)
        best = max(WEIGHTS, key=lambda w: vals[str(w)])
        print(f"  {name}: nsd={vals['nsd']:.4f} h={vals['height']:.4f} "
              f"ctx={vals['ctx']:.4f} best={vals[str(best)]:.4f} w={best} "
              f"oracle={vals['oracle_union']:.4f} q={vals['n_queries']}",
              flush=True)
        report[tag] = {"sensor": sensor, "sequence": sequence,
                       "pool_k": args.pool_k,
                       "variants": {**report.get(tag, {}).get("variants", {}),
                                    name: vals}}
        result_path.write_text(json.dumps(report, indent=2))
    print("HSPEC_CTX_FUSION_DONE", flush=True)


if __name__ == "__main__":
    main()
