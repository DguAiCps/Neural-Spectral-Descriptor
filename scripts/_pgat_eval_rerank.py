#!/usr/bin/env python3
"""
P-GAT native pairwise-rerank evaluation on the 9 val sequences.

P-GAT's published use is subgraph-pair matching: node descriptors are
contextualized by self+cross attention between two subgraphs and retrieval
reads node-pair similarities off the cross-score matrix (training/test.py
accumulates per-node-pair scores and ranks database nodes). We mirror that as
a reranker over a coarse search, which is how the paper deploys it on top of
a retrieval backbone:

  coarse: 544D raw spectral key cosine, top-K candidates under the harness
          protocol (5 m / |i-j| > 30 frames);
  rerank: for query q and candidate c, build the q-centered and c-centered
          windows (<= --window-m travel span, <= --max-nodes nodes, same
          conventions as training), run the trained PoseGAT on the pair, and
          score the candidate by scores[pos(q), pos(c)] — the exact node-pair
          entry their evaluate() ranks by. Candidates re-sorted by this score.

Both the coarse control and the reranked R@{1,5,10} are reported.

Usage (inside the pyg container):
  python scripts/_pgat_eval_rerank.py \
      --checkpoint _external/pgat_runs/full/attentional_graph.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pgat_common import REPO, build_model, pgat_forward  # noqa: E402
from _pgat_make_tensors import (CACHE_KEY, GROUND_DIMS,  # noqa: E402
                                normalize_window_poses)
from _pgat_eval import VAL_SEQS  # noqa: E402


def centered_window(xy, i, window_m, max_nodes):
    """Grow a window around node i until travel span >= window_m or max_nodes."""
    n = len(xy)
    lo = hi = i
    span = 0.0
    while (hi - lo + 1) < max_nodes and span < window_m and (lo > 0 or hi < n - 1):
        grow_lo = lo > 0
        grow_hi = hi < n - 1
        if grow_lo and grow_hi:
            d_lo = float(np.linalg.norm(xy[lo] - xy[lo - 1]))
            d_hi = float(np.linalg.norm(xy[hi + 1] - xy[hi]))
            if d_lo <= d_hi:
                lo -= 1; span += d_lo
            else:
                hi += 1; span += d_hi
        elif grow_lo:
            span += float(np.linalg.norm(xy[lo] - xy[lo - 1])); lo -= 1
        else:
            span += float(np.linalg.norm(xy[hi + 1] - xy[hi])); hi += 1
    return np.arange(lo, hi + 1, dtype=np.int64)


def find_queries(positions, dth, skip):
    """(query_idx, first prior <dth match) pairs, mirroring the harness."""
    out = []
    for j in range(skip, len(positions)):
        prior = positions[: j - skip + 1]
        d = np.linalg.norm(prior - positions[j], axis=1)
        hit = np.where(d < dth)[0]
        if hit.size:
            out.append((j, int(hit[0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-candidates", type=int, default=25)
    ap.add_argument("--window-m", type=float, default=50.0)
    ap.add_argument("--max-nodes", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cache-dir", default=os.path.join(REPO, "data", "preprocessed"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "pgat_eval_rerank.json"))
    args = ap.parse_args()

    from yacs.config import CfgNode
    cfg_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)), "config.yml")
    with open(cfg_path) as f:
        cfg = CfgNode.load_cfg(f)
    ag = cfg.MODEL.ATTENTION_GRAPH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ag.POSE_DIM, ag.FEATURE_DIM, ag.KEYPOINT_HIDDEN_DIM,
                        ag.INCLUDE_POSE, ag.NUM_HEADS, ag.NUM_LAYERS,
                        ag.DROPOUT).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"loaded {args.checkpoint} on {device}")

    KS = [1, 5, 10]
    DTH, SKIP = 5.0, 30
    per_seq = {}
    print(f"\n{'sequence':<20}{'mode':>8}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'queries':>9}")
    for dtype, seq in VAL_SEQS:
        name = f"{dtype}_{seq}"
        cache = os.path.join(args.cache_dir,
                             f"cache_{CACHE_KEY[dtype]}_{dtype}_val_{seq}.npz")
        data = np.load(cache)
        desc = data["descriptors"].astype(np.float32)
        normed = desc / np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-12)
        d0, d1 = GROUND_DIMS[dtype]
        xy = data["poses"][:, (d0, d1), 3].astype(np.float64)
        positions = data["poses"][:, :3, 3]
        n = len(desc)

        queries = find_queries(positions, DTH, SKIP)
        windows = {}

        def window_of(i):
            if i not in windows:
                nodes = centered_window(xy, i, args.window_m, args.max_nodes)
                pn = normalize_window_poses(xy, nodes)
                if np.isnan(pn).any():
                    pn = np.zeros_like(pn)
                windows[i] = (nodes, pn.astype(np.float32))
            return windows[i]

        # coarse candidates per query (cosine, |i-j| > SKIP both directions)
        sims = normed @ normed.T
        pairs = []          # (query, cand) flat list
        cand_lists = []     # per-query candidate arrays (coarse order)
        for qi, _ in queries:
            s = sims[qi].copy()
            lo, hi = max(0, qi - SKIP), min(n, qi + SKIP + 1)
            s[lo:hi] = -np.inf
            cands = np.argsort(-s)[: args.n_candidates]
            cand_lists.append(cands)
            pairs.extend((qi, int(c)) for c in cands)

        # batched pair forwards
        pad = args.max_nodes
        pair_scores = np.zeros(len(pairs), dtype=np.float32)
        with torch.no_grad():
            for s0 in range(0, len(pairs), args.batch_size):
                chunk = pairs[s0:s0 + args.batch_size]
                B = len(chunk)
                f = np.zeros((B, 2, pad, desc.shape[1]), dtype=np.float32)
                p = np.zeros((B, 2, pad, 2), dtype=np.float32)
                m = np.ones((B, 2, pad), dtype=bool)
                qpos = np.zeros(B, dtype=np.int64)
                cpos = np.zeros(B, dtype=np.int64)
                for k, (qi, ci) in enumerate(chunk):
                    for st, node in ((0, qi), (1, ci)):
                        nodes, pn = window_of(node)
                        ln = len(nodes)
                        f[k, st, :ln] = normed[nodes]
                        p[k, st, :ln] = pn
                        m[k, st, :ln] = False
                        if st == 0:
                            qpos[k] = int(np.where(nodes == qi)[0][0])
                        else:
                            cpos[k] = int(np.where(nodes == ci)[0][0])
                scores, _, _ = pgat_forward(
                    model,
                    torch.from_numpy(f).to(device),
                    torch.from_numpy(p).to(device),
                    torch.from_numpy(m).to(device))
                sc = scores.cpu().numpy()
                pair_scores[s0:s0 + B] = sc[np.arange(B), qpos, cpos]

        # score both modes
        res = {}
        for mode in ("coarse", "rerank"):
            correct = {k: 0 for k in KS}
            off = 0
            for (qi, _), cands in zip(queries, cand_lists):
                if mode == "rerank":
                    sc = pair_scores[off:off + len(cands)]
                    order = cands[np.argsort(-sc)]
                else:
                    order = cands
                dists = np.linalg.norm(positions[order[: max(KS)]] - positions[qi], axis=1)
                for k in KS:
                    if np.any(dists[:k] < DTH):
                        correct[k] += 1
                off += len(cands)
            res[mode] = {f"R@{k}": correct[k] / max(len(queries), 1) for k in KS}
            print(f"{name:<20}{mode:>8}"
                  + "".join(f"{res[mode][f'R@{k}']:8.3f}" for k in KS)
                  + f"{len(queries):9d}", flush=True)
        res["n_queries"] = len(queries)
        per_seq[name] = res

    total_q = sum(v["n_queries"] for v in per_seq.values())
    mean = {}
    for mode in ("coarse", "rerank"):
        for k in KS:
            mean[f"{mode}_R^q@{k}"] = sum(
                v[mode][f"R@{k}"] * v["n_queries"] for v in per_seq.values()) / total_q
    print("\nquery-weighted:  "
          + "  ".join(f"{m}={v:.4f}" for m, v in mean.items()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"checkpoint": os.path.abspath(args.checkpoint),
               "n_candidates": args.n_candidates,
               "scoring": "scores[pos(q), pos(c)] of (q-window, c-window) pair",
               "per_sequence": per_seq, "mean": mean},
              open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
