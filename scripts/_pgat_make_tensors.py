#!/usr/bin/env python3
"""
Build P-GAT training tensors from our per-sequence descriptor caches.

Reproduces the data semantics of _external/P-GAT/datasets/dataset_generator_oxford.py
(commit a66a701) on OUR train split, with the deviations listed below.

Their generator, per run:
  - sliding window over consecutive nodes; a subgraph is emitted every time the
    accumulated travel distance crosses DISTANCE_THRESHOLD (50 m), then the window
    front is popped until the total drops back under 50 m  -> one subgraph per node.
  - dense global N x N adjacency (dist < THRESH_TRAIN=10 m, 2D) and switch matrix
    (dist < 10 m OR dist >= NEG_THRESH_TRAIN=50 m; the 10-50 m band is ignored in
    the loss).
  - per-subgraph poses: relative to the window's first node, then normalized by the
    per-subgraph mean (per axis) and scalar std (torch.std over all elements,
    unbiased). Descriptors unit-normalized (descriptor_normalize=True). Note their
    train.py never applies datasets.build.data_normalize, so no global feature
    mean/std normalization is applied here either.
  - pairing dict: subgraph A lists subgraph B as a positive iff any node pair
    (a in A, b in B) has dist < 10 m (temporal neighbors included, matching them).

Deviations (documented for the baseline write-up):
  1. PER-SEQUENCE, not global: each of our train sequences is its own "dataset"
     (P-GAT dataset_id). Adjacency/switch blocks are NOT materialized as dense
     N x N tensors; the training driver recomputes the exact same {0,1} blocks
     on the fly from the stored 2D node positions (mathematically identical,
     avoids ~4 GB of dense matrices). Positives therefore never cross sequences
     (their Oxford setup had cross-run positives; our sequences are mostly
     disjoint sites, NCLT/MulRan cross-session positives are given up).
  2. Emission stride: our keyframes are ~1 m apart (Oxford submaps: ~10-20 m), so
     per-node emission would give ~155K near-duplicate subgraphs. We emit a
     subgraph every --stride-m (default 10 m) of travel, matching their effective
     subgraph density on Oxford. The window itself is still the trailing <=50 m
     of travel (their shrink rule, applied every node instead of only at
     emission - equivalent at emission points).
  3. Node cap: windows are capped at --max-nodes (default 80) via uniform
     subsampling of the window's nodes (spatial extent preserved). Without the
     cap, slow/stationary stretches (KITTI 06) give windows of up to 705 nodes.
  4. Window includes the node that crossed the 50 m threshold (their code has an
     off-by-one that drops the newest node: subgraph = start .. i-1; ours is
     start .. i).
  5. Their paired_train.json nests the positive lists per run; train.py flattens
     them over the training runs (use_all=True). We store the flattened list
     directly (one run per sequence).
  6. NaN poses: a window whose per-subgraph pose std is 0 yields NaN poses (as in
     their generator); their trainer skips any batch containing one. We keep such
     subgraphs (faithful) and report the count.

Output layout (--out-dir, default _external/pgat_data/):
  meta.json                     - global params, per-split sequence lists, counts
  {train,val}/<dtype>_<seq>.npz - features   [N, 544] float32, unit-normalized
                                  xy         [N, 2]   float32 ground-plane position
                                             (KITTI camera frame: dims (0, 2);
                                              others: dims (0, 1) - see GROUND_DIMS)
                                  sub_nodes  [S, max_nodes] int32, -1 padded
                                  sub_len    [S] int32
                                  sub_poses  [S, max_nodes, 2] float32 (normalized)
                                  sub_masks  [S, max_nodes] bool (True = padding,
                                             torch key_padding_mask convention)
  {train,val}/paired.json       - {seq_name: {str(subgraph_idx): [positive idxs]}}

The "val" split here is a held-out fraction of TRAIN sequences (loss monitoring
only). The 9 eval sequences are never touched by this script; scripts/_pgat_eval.py
builds its subgraphs directly from the eval caches.

Usage:
  python3 scripts/_pgat_make_tensors.py                       # full build (host, numpy-only)
  python3 scripts/_pgat_make_tensors.py \
      --sequences kitti_06,kitti_07 --val-holdout kitti_06 \
      --out-dir _external/pgat_data_smoke                     # smoke build
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO, "data", "preprocessed")

# Cache keys per dataset type. NOTE: mulran 33919e6e holds the 288D
# no_interdiff key, not the 544D raw descriptor; c6f48230 holds 544D on
# IDENTICAL keyframes (poses and scan_ids verified equal), so P-GAT uses it to
# keep FEATURE_DIM consistent across all four datasets.
CACHE_KEY = {"helipr": "056e0a02", "kitti": "056e0a02", "nclt": "056e0a02",
             "mulran": "c6f48230"}

# Ground-plane translation dims per dataset. KITTI caches store camera-frame
# poses (x right, y DOWN, z forward -> ground plane is (x, z)); NCLT / MulRan /
# HeLiPR are world-frame (ground plane (x, y)). Verified against cache extents.
GROUND_DIMS = {"kitti": (0, 2), "nclt": (0, 1), "mulran": (0, 1),
               "helipr": (0, 1)}

# Train split from configs/training_multi_dataset.yaml (data.datasets.train).
# NOTE: kitti_01, nclt_2012-11-16, nclt_2013-02-23 have no cache under the
# designated cache keys in this checkout; they are skipped with a warning.
TRAIN_SEQS = [
    ("helipr", "Town02"), ("helipr", "Roundabout01"), ("helipr", "Bridge01"),
    ("helipr", "KAIST04"), ("helipr", "DCC04"), ("helipr", "Riverside04"),
    ("kitti", "01"), ("kitti", "02"), ("kitti", "06"), ("kitti", "07"),
    ("nclt", "2012-05-11"), ("nclt", "2012-08-04"), ("nclt", "2012-11-04"),
    ("nclt", "2012-11-16"), ("nclt", "2013-02-23"),
    ("mulran", "DCC01"), ("mulran", "DCC02"), ("mulran", "KAIST01"),
    ("mulran", "KAIST02"), ("mulran", "Riverside01"), ("mulran", "Riverside02"),
    ("mulran", "Sejong01"), ("mulran", "Sejong02"), ("mulran", "Sejong03"),
]

# Held-out TRAIN sequences used as the P-GAT val split (loss monitoring only).
DEFAULT_VAL_HOLDOUT = ["kitti_07", "nclt_2012-11-04", "helipr_DCC04",
                       "mulran_Riverside02"]


def build_windows(xy, window_m, stride_m):
    """Trailing-<=window_m sliding window, emitted every stride_m of travel.

    Returns list of (start, end) node index pairs, both inclusive.
    """
    n = len(xy)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    windows = []
    start = 0
    total = 0.0
    dist_since_emit = None  # None until the first emission
    for i in range(1, n):
        d = float(step[i - 1])
        total += d
        if dist_since_emit is not None:
            dist_since_emit += d
        if total >= window_m:
            if dist_since_emit is None or dist_since_emit >= stride_m:
                windows.append((start, i))
                dist_since_emit = 0.0
            while total > window_m and start < i:
                total -= float(step[start])
                start += 1
    return windows


def subsample(start, end, max_nodes):
    """Node ids of window [start, end] inclusive, uniformly capped at max_nodes."""
    length = end - start + 1
    if length <= max_nodes:
        return np.arange(start, end + 1, dtype=np.int64)
    idx = np.round(np.linspace(0, length - 1, max_nodes)).astype(np.int64)
    return start + idx


def normalize_window_poses(xy, nodes):
    """Their segmentation(): relative to first node, per-axis mean, scalar
    unbiased std over all elements. Returns [len, 2] float64 (may contain NaN
    when std == 0, matching their generator)."""
    rel = xy[nodes] - xy[nodes[0]]
    mean = rel.mean(axis=0)
    n_el = rel.size
    std = rel.std(ddof=1) if n_el > 1 else 0.0
    return (rel - mean) / std if std > 0 else np.full_like(rel, np.nan)


# Compatible 544D fallback caches for the three train sequences absent under
# the primary keys (same keyframe-selector density as the primary caches).
CACHE_KEY_FALLBACK = {("kitti", "01"): "de201724",
                      ("nclt", "2012-11-16"): "76c42cd7",
                      ("nclt", "2013-02-23"): "76c42cd7"}


def build_sequence(dtype, seq, args):
    cache = os.path.join(args.cache_dir,
                         f"cache_{CACHE_KEY[dtype]}_{dtype}_{seq}.npz")
    if not os.path.exists(cache) and (dtype, str(seq)) in CACHE_KEY_FALLBACK:
        cache = os.path.join(args.cache_dir,
                             f"cache_{CACHE_KEY_FALLBACK[(dtype, str(seq))]}_{dtype}_{seq}.npz")
    if not os.path.exists(cache):
        return None
    data = np.load(cache)
    desc = data["descriptors"].astype(np.float32)
    d0, d1 = GROUND_DIMS[dtype]
    xy = data["poses"][:, (d0, d1), 3].astype(np.float64)

    # descriptor_normalize=True (unit vectors)
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    desc = desc / np.maximum(norms, 1e-12)

    windows = build_windows(xy, args.window_m, args.stride_m)
    if not windows:
        print(f"  WARNING: {dtype}_{seq}: no 50 m window emitted, skipping")
        return None

    m = args.max_nodes
    s_count = len(windows)
    sub_nodes = np.full((s_count, m), -1, dtype=np.int32)
    sub_len = np.zeros(s_count, dtype=np.int32)
    sub_poses = np.zeros((s_count, m, 2), dtype=np.float32)
    sub_masks = np.ones((s_count, m), dtype=bool)  # True = padding
    nan_count = 0
    for k, (st, en) in enumerate(windows):
        nodes = subsample(st, en, m)
        ln = len(nodes)
        sub_nodes[k, :ln] = nodes
        sub_len[k] = ln
        pn = normalize_window_poses(xy, nodes)
        if np.isnan(pn).any():
            nan_count += 1
        sub_poses[k, :ln] = pn.astype(np.float32)
        sub_masks[k, :ln] = False

    # Pairing: A lists B as positive iff any stored node of B is within
    # pos_thresh (2D) of any stored node of A (self excluded).
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    neighbours = tree.query_ball_point(xy, r=args.pos_thresh, workers=-1)
    node_mat = np.maximum(sub_nodes.astype(np.int64), 0)
    valid = sub_nodes >= 0
    ind = np.zeros(len(xy), dtype=bool)
    paired = {}
    for a in range(s_count):
        ind[:] = False
        for nid in sub_nodes[a, : sub_len[a]]:
            nb = neighbours[nid]
            if nb:
                ind[nb] = True
        hits = ind[node_mat] & valid
        overlap = hits.any(axis=1)
        overlap[a] = False
        paired[str(a)] = np.nonzero(overlap)[0].tolist()

    n_pos = np.array([len(v) for v in paired.values()])
    print(f"  {dtype}_{seq}: N={len(xy)} subgraphs={s_count} "
          f"len med={int(np.median(sub_len))} max={int(sub_len.max())} "
          f"positives/subgraph med={int(np.median(n_pos))} "
          f"nan_pose_subgraphs={nan_count}")
    return {
        "features": desc, "xy": xy.astype(np.float32), "sub_nodes": sub_nodes,
        "sub_len": sub_len, "sub_poses": sub_poses, "sub_masks": sub_masks,
    }, paired, nan_count


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out-dir", default=os.path.join(REPO, "_external", "pgat_data"))
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--window-m", type=float, default=50.0,
                    help="DISTANCE_THRESHOLD: travel distance per subgraph")
    ap.add_argument("--stride-m", type=float, default=10.0,
                    help="travel distance between emitted subgraphs (deviation 2)")
    ap.add_argument("--max-nodes", type=int, default=80,
                    help="cap on nodes per subgraph, uniform subsample (deviation 3)")
    ap.add_argument("--pos-thresh", type=float, default=10.0,
                    help="THRESH_TRAIN: positive node-pair distance")
    ap.add_argument("--neg-thresh", type=float, default=50.0,
                    help="NEG_THRESH_TRAIN: pairs in [pos, neg) are loss-ignored")
    ap.add_argument("--sequences", default=None,
                    help="comma-separated <dtype>_<seq> filter (smoke builds)")
    ap.add_argument("--val-holdout", default=",".join(DEFAULT_VAL_HOLDOUT),
                    help="comma-separated train sequences held out as val split")
    args = ap.parse_args()

    seq_filter = set(args.sequences.split(",")) if args.sequences else None
    holdout = set(args.val_holdout.split(","))

    splits = {"train": [], "val": []}
    for dtype, seq in TRAIN_SEQS:
        name = f"{dtype}_{seq}"
        if seq_filter is not None and name not in seq_filter:
            continue
        splits["val" if name in holdout else "train"].append((dtype, seq))

    meta = {
        "window_m": args.window_m, "stride_m": args.stride_m,
        "max_nodes": args.max_nodes, "pos_thresh": args.pos_thresh,
        "neg_thresh": args.neg_thresh, "feature_dim": None,
        "sequences": {}, "skipped_missing_cache": [],
    }
    for split, seq_list in splits.items():
        out_split = os.path.join(args.out_dir, split)
        os.makedirs(out_split, exist_ok=True)
        paired_all = {}
        names = []
        print(f"[{split}]")
        for dtype, seq in seq_list:
            name = f"{dtype}_{seq}"
            result = build_sequence(dtype, seq, args)
            if result is None:
                print(f"  WARNING: no cache for {name} "
                      f"(key {CACHE_KEY[dtype]}), skipped")
                meta["skipped_missing_cache"].append(name)
                continue
            arrays, paired, nan_count = result
            np.savez_compressed(os.path.join(out_split, f"{name}.npz"), **arrays)
            paired_all[name] = paired
            names.append(name)
            meta["feature_dim"] = int(arrays["features"].shape[1])
            meta["sequences"][name] = {
                "split": split, "n_nodes": int(len(arrays["xy"])),
                "n_subgraphs": int(len(arrays["sub_len"])),
                "nan_pose_subgraphs": int(nan_count),
            }
        with open(os.path.join(out_split, "paired.json"), "w") as f:
            json.dump(paired_all, f)
        meta[f"{split}_sequences"] = names

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    total_sub = sum(v["n_subgraphs"] for v in meta["sequences"].values())
    print(f"done: {len(meta['sequences'])} sequences, {total_sub} subgraphs "
          f"-> {args.out_dir}")
    if meta["skipped_missing_cache"]:
        print(f"skipped (no cache): {meta['skipped_missing_cache']}")


if __name__ == "__main__":
    sys.exit(main())
