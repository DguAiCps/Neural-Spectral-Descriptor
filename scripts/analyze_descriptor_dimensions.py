#!/usr/bin/env python3
"""
Descriptor Dimension Importance Analysis

각 descriptor 차원이 loop closure 판별에 얼마나 기여하는지 측정.

분석 방법:
  1. Fisher Discriminant Ratio (FDR): (μ_pos - μ_neg)² / (σ²_pos + σ²_neg)
  2. AUC per dimension: 단일 차원만으로 pos/neg 분류 시 AUC
  3. Ring/bin/stat 그룹별 집계

Usage:
  python scripts/analyze_descriptor_dimensions.py
  python scripts/analyze_descriptor_dimensions.py --cache-hash 056e0a02 --sequence kitti_val_00
  python scripts/analyze_descriptor_dimensions.py --top 30
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict


def load_cached_data(cache_dir: str, cache_hash: str, sequence: str):
    """Load descriptors and poses from cache."""
    path = os.path.join(cache_dir, f'cache_{cache_hash}_{sequence}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data['descriptors'], data['poses']


def find_pairs(poses, pos_dist=5.0, neg_dist=10.0, temporal_gap=30, max_pairs=50000):
    """Find positive (loop closure) and negative pairs."""
    positions = poses[:, :3, 3]
    n = len(positions)

    tree = KDTree(positions)

    positives = []  # (i, j) pairs: same place, temporal gap
    negatives = []  # (i, j) pairs: different place

    # Find positives: close in space, far in time
    for j in range(temporal_gap, n):
        nearby = tree.query_ball_point(positions[j], pos_dist)
        for i in nearby:
            if i <= j - temporal_gap:
                positives.append((j, i))

    # Subsample if too many
    rng = np.random.RandomState(42)
    if len(positives) > max_pairs:
        idx = rng.choice(len(positives), max_pairs, replace=False)
        positives = [positives[i] for i in idx]

    # Find negatives: far in space (sample randomly)
    n_neg_target = len(positives) * 2
    neg_attempts = 0
    while len(negatives) < n_neg_target and neg_attempts < n_neg_target * 10:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if abs(i - j) < temporal_gap:
            neg_attempts += 1
            continue
        dist = np.linalg.norm(positions[i] - positions[j])
        if dist >= neg_dist:
            negatives.append((i, j))
        neg_attempts += 1

    return positives, negatives


def compute_pair_diffs(descriptors, pairs):
    """Compute |d_i - d_j| for each pair, per dimension."""
    idx_a = np.array([p[0] for p in pairs])
    idx_b = np.array([p[1] for p in pairs])
    return np.abs(descriptors[idx_a] - descriptors[idx_b])


def compute_fisher_scores(pos_diffs, neg_diffs):
    """Fisher Discriminant Ratio per dimension."""
    mu_pos = pos_diffs.mean(axis=0)
    mu_neg = neg_diffs.mean(axis=0)
    var_pos = pos_diffs.var(axis=0) + 1e-10
    var_neg = neg_diffs.var(axis=0) + 1e-10
    fdr = (mu_pos - mu_neg) ** 2 / (var_pos + var_neg)
    return fdr


def compute_auc_per_dim(pos_diffs, neg_diffs):
    """Fast AUC per dimension using Mann-Whitney U statistic."""
    D = pos_diffs.shape[1]
    aucs = np.zeros(D)
    n_pos = len(pos_diffs)
    n_neg = len(neg_diffs)

    for d in range(D):
        # For |diff|: smaller diff → positive (same place), larger → negative
        # AUC = P(neg_diff > pos_diff) = good separation
        pos_vals = pos_diffs[:, d]
        neg_vals = neg_diffs[:, d]

        # Approximate AUC via sorted rank method
        all_vals = np.concatenate([pos_vals, neg_vals])
        labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        order = np.argsort(all_vals)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(all_vals) + 1)
        pos_rank_sum = ranks[:n_pos].sum()
        u = pos_rank_sum - n_pos * (n_pos + 1) / 2
        auc = u / (n_pos * n_neg)
        # We want AUC where pos has SMALLER diffs → flip
        aucs[d] = 1.0 - auc

    return aucs


def map_dim_to_semantic(dim_idx, n_rows, n_bins, stats, inter_stats):
    """Map flat dimension index to (channel, ring, bin/diff, stat_type)."""
    n_stats = len(stats)
    n_inter = len(inter_stats)

    base_dim = n_rows * n_bins
    inter_base_dim = n_rows * (n_bins - 1)

    # Intra-bin channels
    for s_idx, stat_name in enumerate(stats):
        start = s_idx * base_dim
        end = start + base_dim
        if start <= dim_idx < end:
            offset = dim_idx - start
            ring = offset // n_bins
            bin_idx = offset % n_bins
            return {
                'channel': f'intra_{stat_name}',
                'ring': ring,
                'bin': bin_idx,
                'type': 'intra',
                'stat': stat_name,
            }

    # Inter-bin channels
    inter_start = n_stats * base_dim
    for inter_idx, inter_name in enumerate(inter_stats):
        for s_idx, stat_name in enumerate(stats):
            ch_idx = inter_idx * n_stats + s_idx
            start = inter_start + ch_idx * inter_base_dim
            end = start + inter_base_dim
            if start <= dim_idx < end:
                offset = dim_idx - start
                ring = offset // (n_bins - 1)
                diff_idx = offset % (n_bins - 1)
                return {
                    'channel': f'inter_{inter_name}_{stat_name}',
                    'ring': ring,
                    'bin': diff_idx,
                    'type': 'inter',
                    'stat': f'{inter_name}_{stat_name}',
                }

    return {'channel': 'unknown', 'ring': -1, 'bin': -1, 'type': '?', 'stat': '?'}


def main():
    parser = argparse.ArgumentParser(description='Descriptor Dimension Importance Analysis')
    parser.add_argument('--cache-dir', default='data/preprocessed')
    parser.add_argument('--cache-hash', default='056e0a02')
    parser.add_argument('--sequences', nargs='+',
                        default=['kitti_val_00', 'kitti_val_05'])
    parser.add_argument('--n-rows', type=int, default=16,
                        help='Number of rows (range_image=16, bev=79)')
    parser.add_argument('--n-bins', type=int, default=16)
    parser.add_argument('--stats', nargs='+', default=['mean', 'std'])
    parser.add_argument('--inter-stats', nargs='+', default=['diff'])
    parser.add_argument('--top', type=int, default=20,
                        help='Show top N dimensions')
    parser.add_argument('--pos-dist', type=float, default=5.0)
    parser.add_argument('--neg-dist', type=float, default=10.0)
    parser.add_argument('--temporal-gap', type=int, default=30)
    args = parser.parse_args()

    print("=" * 90)
    print("DESCRIPTOR DIMENSION IMPORTANCE ANALYSIS")
    print(f"Sequences: {args.sequences}")
    print(f"Layout: {args.n_rows} rows × {args.n_bins} bins, stats={args.stats}, inter={args.inter_stats}")
    print("=" * 90)

    # Aggregate across sequences
    all_pos_diffs = []
    all_neg_diffs = []

    for seq in args.sequences:
        print(f"\n[{seq}] Loading data...")
        try:
            descriptors, poses = load_cached_data(args.cache_dir, args.cache_hash, seq)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        D = descriptors.shape[1]
        print(f"  {len(descriptors)} keyframes, {D}D descriptors")

        print(f"  Finding pairs (pos<{args.pos_dist}m, neg>{args.neg_dist}m, gap>{args.temporal_gap})...")
        positives, negatives = find_pairs(
            poses, args.pos_dist, args.neg_dist, args.temporal_gap)
        print(f"  Found {len(positives)} positive pairs, {len(negatives)} negative pairs")

        if len(positives) == 0:
            print("  SKIP: no loop closures found")
            continue

        pos_diffs = compute_pair_diffs(descriptors, positives)
        neg_diffs = compute_pair_diffs(descriptors, negatives)
        all_pos_diffs.append(pos_diffs)
        all_neg_diffs.append(neg_diffs)

    if not all_pos_diffs:
        print("\nNo valid data found. Check cache-hash and sequences.")
        return 1

    pos_diffs = np.concatenate(all_pos_diffs, axis=0)
    neg_diffs = np.concatenate(all_neg_diffs, axis=0)
    D = pos_diffs.shape[1]

    print(f"\n{'='*90}")
    print(f"TOTAL: {len(pos_diffs)} positive pairs, {len(neg_diffs)} negative pairs, {D}D")
    print(f"{'='*90}")

    # 1. Fisher Discriminant Ratio
    print("\nComputing Fisher Discriminant Ratio...")
    fdr = compute_fisher_scores(pos_diffs, neg_diffs)

    # 2. AUC
    print("Computing per-dimension AUC...")
    aucs = compute_auc_per_dim(pos_diffs, neg_diffs)

    # Map dimensions to semantics
    dim_info = []
    for d in range(D):
        sem = map_dim_to_semantic(d, args.n_rows, args.n_bins, args.stats, args.inter_stats)
        dim_info.append({
            'dim': d,
            'fdr': fdr[d],
            'auc': aucs[d],
            **sem,
        })

    # =====================================================================
    # Top dimensions by FDR
    # =====================================================================
    print(f"\n{'='*90}")
    print(f"TOP {args.top} DIMENSIONS BY FISHER DISCRIMINANT RATIO")
    print(f"{'='*90}")
    print(f"{'Rank':>4} {'Dim':>5} {'FDR':>8} {'AUC':>6} {'Channel':<22} {'Ring':>4} {'Bin':>4}")
    print("-" * 60)

    sorted_by_fdr = sorted(dim_info, key=lambda x: x['fdr'], reverse=True)
    for rank, info in enumerate(sorted_by_fdr[:args.top], 1):
        print(f"{rank:>4} {info['dim']:>5} {info['fdr']:>8.4f} {info['auc']:>6.3f} "
              f"{info['channel']:<22} {info['ring']:>4} {info['bin']:>4}")

    # =====================================================================
    # Top dimensions by AUC
    # =====================================================================
    print(f"\n{'='*90}")
    print(f"TOP {args.top} DIMENSIONS BY AUC")
    print(f"{'='*90}")
    print(f"{'Rank':>4} {'Dim':>5} {'AUC':>6} {'FDR':>8} {'Channel':<22} {'Ring':>4} {'Bin':>4}")
    print("-" * 60)

    sorted_by_auc = sorted(dim_info, key=lambda x: x['auc'], reverse=True)
    for rank, info in enumerate(sorted_by_auc[:args.top], 1):
        print(f"{rank:>4} {info['dim']:>5} {info['auc']:>6.3f} {info['fdr']:>8.4f} "
              f"{info['channel']:<22} {info['ring']:>4} {info['bin']:>4}")

    # =====================================================================
    # Aggregate by ring
    # =====================================================================
    print(f"\n{'='*90}")
    print("AGGREGATE BY RING (elevation row)")
    print(f"{'='*90}")

    ring_fdr = defaultdict(list)
    ring_auc = defaultdict(list)
    for info in dim_info:
        r = info['ring']
        ring_fdr[r].append(info['fdr'])
        ring_auc[r].append(info['auc'])

    print(f"{'Ring':>4} {'Mean FDR':>10} {'Max FDR':>10} {'Mean AUC':>10} {'Max AUC':>10} {'N dims':>7}")
    print("-" * 55)
    for r in sorted(ring_fdr.keys()):
        fdrs = ring_fdr[r]
        au = ring_auc[r]
        print(f"{r:>4} {np.mean(fdrs):>10.4f} {np.max(fdrs):>10.4f} "
              f"{np.mean(au):>10.3f} {np.max(au):>10.3f} {len(fdrs):>7}")

    # =====================================================================
    # Aggregate by bin
    # =====================================================================
    print(f"\n{'='*90}")
    print("AGGREGATE BY FREQUENCY BIN")
    print(f"{'='*90}")

    bin_fdr = defaultdict(list)
    bin_auc = defaultdict(list)
    for info in dim_info:
        b = info['bin']
        bin_fdr[b].append(info['fdr'])
        bin_auc[b].append(info['auc'])

    print(f"{'Bin':>4} {'Mean FDR':>10} {'Max FDR':>10} {'Mean AUC':>10} {'Max AUC':>10} {'N dims':>7}")
    print("-" * 55)
    for b in sorted(bin_fdr.keys()):
        fdrs = bin_fdr[b]
        au = bin_auc[b]
        print(f"{b:>4} {np.mean(fdrs):>10.4f} {np.max(fdrs):>10.4f} "
              f"{np.mean(au):>10.3f} {np.max(au):>10.3f} {len(fdrs):>7}")

    # =====================================================================
    # Aggregate by channel (stat type)
    # =====================================================================
    print(f"\n{'='*90}")
    print("AGGREGATE BY CHANNEL (stat type)")
    print(f"{'='*90}")

    ch_fdr = defaultdict(list)
    ch_auc = defaultdict(list)
    for info in dim_info:
        ch = info['channel']
        ch_fdr[ch].append(info['fdr'])
        ch_auc[ch].append(info['auc'])

    print(f"{'Channel':<22} {'Mean FDR':>10} {'Max FDR':>10} {'Mean AUC':>10} {'Max AUC':>10} {'N dims':>7}")
    print("-" * 70)
    for ch in sorted(ch_fdr.keys()):
        fdrs = ch_fdr[ch]
        au = ch_auc[ch]
        print(f"{ch:<22} {np.mean(fdrs):>10.4f} {np.max(fdrs):>10.4f} "
              f"{np.mean(au):>10.3f} {np.max(au):>10.3f} {len(fdrs):>7}")

    # =====================================================================
    # Ring × Bin heatmap (text)
    # =====================================================================
    print(f"\n{'='*90}")
    print("RING × BIN FDR HEATMAP (intra_mean channel)")
    print(f"{'='*90}")

    # Build FDR matrix for intra_mean channel
    n_rows = args.n_rows
    n_bins = args.n_bins
    fdr_matrix = np.zeros((n_rows, n_bins))
    for info in dim_info:
        if info['channel'] == 'intra_mean' and info['ring'] >= 0:
            fdr_matrix[info['ring'], info['bin']] = info['fdr']

    # Print header
    header = "     " + "".join(f"B{b:>2}" for b in range(n_bins))
    print(header)
    for r in range(n_rows):
        row_str = f"R{r:>2}: "
        for b in range(n_bins):
            val = fdr_matrix[r, b]
            if val > 0.1:
                row_str += f" ██"
            elif val > 0.05:
                row_str += f" ▓▓"
            elif val > 0.01:
                row_str += f" ▒▒"
            elif val > 0.001:
                row_str += f" ░░"
            else:
                row_str += f"   "
        row_str += f"  (avg={np.mean(fdr_matrix[r]):.4f})"
        print(row_str)

    print(f"\nLegend: ██ >0.1  ▓▓ >0.05  ▒▒ >0.01  ░░ >0.001")

    # =====================================================================
    # Bottom dimensions (least useful)
    # =====================================================================
    print(f"\n{'='*90}")
    print(f"BOTTOM {args.top} DIMENSIONS (least discriminative)")
    print(f"{'='*90}")
    print(f"{'Rank':>4} {'Dim':>5} {'FDR':>8} {'AUC':>6} {'Channel':<22} {'Ring':>4} {'Bin':>4}")
    print("-" * 60)

    for rank, info in enumerate(sorted_by_fdr[-args.top:], 1):
        print(f"{rank:>4} {info['dim']:>5} {info['fdr']:>8.6f} {info['auc']:>6.3f} "
              f"{info['channel']:<22} {info['ring']:>4} {info['bin']:>4}")

    # =====================================================================
    # Summary statistics
    # =====================================================================
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"Total dimensions: {D}")
    print(f"FDR > 0.1 (highly discriminative): {sum(1 for x in fdr if x > 0.1)} dims")
    print(f"FDR > 0.01 (moderately discriminative): {sum(1 for x in fdr if x > 0.01)} dims")
    print(f"FDR < 0.001 (nearly useless): {sum(1 for x in fdr if x < 0.001)} dims")
    print(f"AUC > 0.7 (good separation): {sum(1 for x in aucs if x > 0.7)} dims")
    print(f"AUC > 0.6 (moderate separation): {sum(1 for x in aucs if x > 0.6)} dims")
    print(f"AUC < 0.55 (near random): {sum(1 for x in aucs if x < 0.55)} dims")

    # Effective dimension estimate (cumulative FDR)
    sorted_fdr = np.sort(fdr)[::-1]
    cumsum = np.cumsum(sorted_fdr) / sorted_fdr.sum()
    for pct in [0.5, 0.8, 0.9, 0.95]:
        n_dims = np.searchsorted(cumsum, pct) + 1
        print(f"  {pct*100:.0f}% of total FDR captured by top {n_dims} dims ({n_dims/D*100:.1f}%)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
