#!/usr/bin/env python3
"""
Evaluate a trained P-GAT baseline checkpoint on our 9 val sequences with the
harness protocol (intra-sequence R@K; query = revisit within 5 m of a frame
>= 30 frames earlier; cosine retrieval via baselines.eval_utils).

Per-node embedding reinterpretation (DOCUMENTED CHOICE):
  P-GAT natively scores subgraph PAIRS: node descriptors are contextualized by
  alternating self- and cross-attention between the two subgraphs, then paired
  node similarities are read off (attentional_graph/layers/attentional_graph.py).
  For single-vector retrieval we run the model on (g, g) pairs - each eval
  subgraph paired with ITSELF - so cross-attention degenerates to a second
  attention pass over the same subgraph, and take the stream-0 node features
  after output_fc + L2 normalization (PoseGAT eq. 6 head) as the per-keyframe
  embedding. Rationale: this exercises every trained weight on the trained
  compute path; extracting only the self-attention branch would skip the
  trained cross-attention/MLP blocks, and pairing with a different subgraph
  would make each node's embedding depend on an arbitrary partner. Note the
  two streams are not symmetric (stream 1 is cross-updated using the already
  cross-updated stream 0 - see _pgat_common.pgat_streams); stream 0 is used.

Eval subgraphs: the sequence is tiled into NON-overlapping windows (cut at
50 m travel or --max-nodes nodes, whichever first), so each keyframe belongs to
exactly one subgraph and gets exactly one embedding. This deviates from the
training windows (trailing 50 m sliding, 10 m stride) because per-node
embeddings must be unique; window statistics (span <= 50 m, <= max-nodes nodes)
match training. The final partial tile is kept. Per-window pose normalization
follows their generator (relative to first node, mean/scalar-unbiased-std);
degenerate windows (std == 0) fall back to zero poses instead of their NaN
(training skips NaN batches; eval needs an embedding for every node).

Usage (inside the pyg container):
  python scripts/_pgat_eval.py --checkpoint _external/pgat_runs/full/attentional_graph.pt
  python scripts/_pgat_eval.py --checkpoint _external/pgat_runs/smoke/attentional_graph.pt \
      --sequences nclt_2013-01-10 --out results/pgat_eval_smoke.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pgat_common import REPO, build_model, pgat_forward  # noqa: E402

sys.path.insert(0, REPO)
from baselines.eval_utils import compute_recall_multi_k  # noqa: E402

from _pgat_make_tensors import (CACHE_KEY, GROUND_DIMS,  # noqa: E402
                                normalize_window_poses)

# The 9 val sequences (configs/training_multi_dataset.yaml data.datasets.val);
# cache files carry a val_ prefix on the sequence name.
VAL_SEQS = [
    ("kitti", "00"), ("kitti", "05"), ("kitti", "08"),
    ("nclt", "2012-01-08"), ("nclt", "2013-01-10"),
    ("helipr", "Town01"),
    ("mulran", "DCC03"), ("mulran", "KAIST03"), ("mulran", "Riverside03"),
]


def tile_windows(xy, window_m, max_nodes):
    """Non-overlapping tiles: cut at window_m travel or max_nodes nodes."""
    n = len(xy)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    tiles = []
    start = 0
    total = 0.0
    for i in range(1, n):
        total += float(step[i - 1])
        length = i - start + 1
        if total >= window_m or length >= max_nodes:
            tiles.append(np.arange(start, i + 1, dtype=np.int64))
            start = i + 1
            total = 0.0
    if start < n:
        tiles.append(np.arange(start, n, dtype=np.int64))
    return tiles


def embed_sequence(model, desc, xy, window_m, max_nodes, batch_size, device):
    """Per-node P-GAT embeddings for one sequence. Returns [N, D] numpy."""
    n, dim = desc.shape
    tiles = tile_windows(xy, window_m, max_nodes)
    pad = max(len(t) for t in tiles)
    feats = np.zeros((len(tiles), pad, dim), dtype=np.float32)
    poses = np.zeros((len(tiles), pad, 2), dtype=np.float32)
    masks = np.ones((len(tiles), pad), dtype=bool)
    for k, nodes in enumerate(tiles):
        ln = len(nodes)
        feats[k, :ln] = desc[nodes]
        pn = normalize_window_poses(xy, nodes)
        if np.isnan(pn).any():
            pn = np.zeros_like(pn)  # degenerate window guard (eval only)
        poses[k, :ln] = pn
        masks[k, :ln] = False

    out = np.zeros((n, dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(tiles), batch_size):
            f = torch.from_numpy(feats[s:s + batch_size]).to(device)
            p = torch.from_numpy(poses[s:s + batch_size]).to(device)
            m = torch.from_numpy(masks[s:s + batch_size]).to(device)
            # (g, g) pair: both streams are the same subgraph
            f2 = torch.stack([f, f], dim=1)
            p2 = torch.stack([p, p], dim=1)
            m2 = torch.stack([m, m], dim=1)
            _, e0, _ = pgat_forward(model, f2, p2, m2)
            e0 = e0.cpu().numpy()
            for k in range(e0.shape[0]):
                nodes = tiles[s + k]
                out[nodes] = e0[k, :len(nodes)]
    return out


def main():
    ap = argparse.ArgumentParser(description="P-GAT baseline eval on val split")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None,
                    help="config.yml of the run (default: alongside checkpoint)")
    ap.add_argument("--sequences", default=None,
                    help="comma-separated <dtype>_<seq> filter")
    ap.add_argument("--window-m", type=float, default=50.0)
    ap.add_argument("--max-nodes", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache-dir",
                    default=os.path.join(REPO, "data", "preprocessed"))
    ap.add_argument("--out", default=os.path.join(REPO, "results",
                                                  "pgat_eval.json"))
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
    print(f"loaded {args.checkpoint} on {device}")

    seq_filter = set(args.sequences.split(",")) if args.sequences else None
    k_values = [1, 5, 10]
    per_seq = {}
    print(f"\n{'sequence':<20}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'queries':>9}")
    for dtype, seq in VAL_SEQS:
        name = f"{dtype}_{seq}"
        if seq_filter is not None and name not in seq_filter:
            continue
        cache = os.path.join(args.cache_dir,
                             f"cache_{CACHE_KEY[dtype]}_{dtype}_val_{seq}.npz")
        data = np.load(cache)
        desc = data["descriptors"].astype(np.float32)
        desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-12)
        d0, d1 = GROUND_DIMS[dtype]
        xy = data["poses"][:, (d0, d1), 3].astype(np.float64)
        emb = embed_sequence(model, desc, xy, args.window_m, args.max_nodes,
                             args.batch_size, device)
        recalls, n_queries = compute_recall_multi_k(
            emb, data["poses"], k_values=k_values,
            distance_threshold=5.0, skip_frames=30)
        per_seq[name] = {f"R@{k}": recalls[k] for k in k_values}
        per_seq[name]["n_queries"] = n_queries
        print(f"{name:<20}" + "".join(f"{recalls[k]:8.3f}" for k in k_values)
              + f"{n_queries:9d}")

    if not per_seq:
        print("no sequences evaluated")
        return
    total_q = sum(v["n_queries"] for v in per_seq.values())
    mean = {}
    for k in k_values:
        rq = sum(v[f"R@{k}"] * v["n_queries"] for v in per_seq.values()) / max(total_q, 1)
        rs = float(np.mean([v[f"R@{k}"] for v in per_seq.values()]))
        mean[f"R^q@{k}"] = rq
        mean[f"R^s@{k}"] = rs
    print(f"\nquery-weighted mean: "
          + "  ".join(f"R^q@{k}={mean[f'R^q@{k}']:.4f}" for k in k_values)
          + f"   (total {total_q} queries)")
    print("per-seq mean:        "
          + "  ".join(f"R^s@{k}={mean[f'R^s@{k}']:.4f}" for k in k_values))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "checkpoint": os.path.abspath(args.checkpoint),
            "protocol": {"distance_threshold": 5.0, "skip_frames": 30,
                         "window_m": args.window_m,
                         "max_nodes": args.max_nodes,
                         "embedding": "PoseGAT (g,g) stream-0 post output_fc "
                                      "+ L2, non-overlapping tiles"},
            "per_sequence": per_seq,
            "mean": mean,
        }, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
