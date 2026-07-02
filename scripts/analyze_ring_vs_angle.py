#!/usr/bin/env python3
"""
Ring Number vs Elevation Angle — 센서별 FDR 비교

각 데이터셋(KITTI/NCLT/HeLiPR)에 대해:
  1. 캐시에서 descriptor + pose 로드
  2. Positive/negative pair 기반 FDR 계산 (ring별)
  3. Ring 번호 → 실제 elevation angle 매핑
  4. 상관 분석: FDR ~ ring_number vs FDR ~ elevation_angle

결론: ring 번호 자체가 중요한지, 아니면 특정 각도 범위가 중요한지.

Usage:
  python scripts/analyze_ring_vs_angle.py
  python scripts/analyze_ring_vs_angle.py --cache-hash 056e0a02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict


# Sensor elevation ranges (from config)
SENSOR_ELEVATION = {
    'kitti': {'min': -24.8, 'max': 2.0, 'sensor': 'HDL-64E'},
    'nclt': {'min': -30.67, 'max': 10.67, 'sensor': 'HDL-32E'},
    'helipr': {'min': -15.0, 'max': 15.0, 'sensor': 'VLP-16'},
}

DATASET_SEQUENCES = {
    'kitti': ['kitti_val_00', 'kitti_val_05', 'kitti_val_08'],
    'nclt': ['nclt_val_2012-01-08', 'nclt_val_2013-01-10'],
    'helipr': ['helipr_val_Town01'],
}

N_ROWS = 16  # range_image target_elevation_bins
N_BINS = 9   # octave binning actual bins


def ring_to_angle(ring_idx, dataset, n_rows=N_ROWS):
    """Map ring index to elevation angle in degrees.

    Ring 0 = max elevation (top), Ring n_rows-1 = min elevation (bottom).
    Linear interpolation between min and max.
    """
    elev = SENSOR_ELEVATION[dataset]
    # Ring 0 → max angle, Ring (n_rows-1) → min angle
    return elev['max'] - ring_idx * (elev['max'] - elev['min']) / (n_rows - 1)


def load_cached_data(cache_dir, cache_hash, sequence):
    path = os.path.join(cache_dir, f'cache_{cache_hash}_{sequence}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data['descriptors'], data['poses']


def find_pairs(poses, pos_dist=5.0, neg_dist=10.0, temporal_gap=30, max_pairs=50000):
    positions = poses[:, :3, 3]
    n = len(positions)
    tree = KDTree(positions)

    positives = []
    for j in range(temporal_gap, n):
        nearby = tree.query_ball_point(positions[j], pos_dist)
        for i in nearby:
            if i <= j - temporal_gap:
                positives.append((j, i))

    rng = np.random.RandomState(42)
    if len(positives) > max_pairs:
        idx = rng.choice(len(positives), max_pairs, replace=False)
        positives = [positives[i] for i in idx]

    negatives = []
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


def compute_ring_fdr(descriptors, positives, negatives, n_rows, n_bins, stats, inter_stats):
    """Compute per-ring aggregate FDR.

    Returns dict: ring_idx → mean FDR across all dims in that ring.
    Also returns per-ring, per-channel FDR for detailed analysis.
    """
    idx_a_pos = np.array([p[0] for p in positives])
    idx_b_pos = np.array([p[1] for p in positives])
    pos_diffs = np.abs(descriptors[idx_a_pos] - descriptors[idx_b_pos])

    idx_a_neg = np.array([p[0] for p in negatives])
    idx_b_neg = np.array([p[1] for p in negatives])
    neg_diffs = np.abs(descriptors[idx_a_neg] - descriptors[idx_b_neg])

    D = descriptors.shape[1]
    mu_pos = pos_diffs.mean(axis=0)
    mu_neg = neg_diffs.mean(axis=0)
    var_pos = pos_diffs.var(axis=0) + 1e-10
    var_neg = neg_diffs.var(axis=0) + 1e-10
    fdr = (mu_pos - mu_neg) ** 2 / (var_pos + var_neg)

    # Map dims to rings (same logic as analyze_descriptor_dimensions.py)
    n_stats = len(stats)
    n_inter = len(inter_stats)
    base_dim = n_rows * n_bins
    inter_base_dim = n_rows * (n_bins - 1)

    ring_fdrs = defaultdict(list)  # ring → [fdr values]
    ring_channel_fdrs = defaultdict(lambda: defaultdict(list))  # ring → channel → [fdr values]

    for d in range(D):
        ring = -1
        channel = 'unknown'

        # Intra-bin channels
        for s_idx, stat_name in enumerate(stats):
            start = s_idx * base_dim
            end = start + base_dim
            if start <= d < end:
                offset = d - start
                ring = offset // n_bins
                channel = f'intra_{stat_name}'
                break

        if ring == -1:
            # Inter-bin channels
            inter_start = n_stats * base_dim
            for inter_idx, inter_name in enumerate(inter_stats):
                for s_idx, stat_name in enumerate(stats):
                    ch_idx = inter_idx * n_stats + s_idx
                    start = inter_start + ch_idx * inter_base_dim
                    end = start + inter_base_dim
                    if start <= d < end:
                        offset = d - start
                        ring = offset // (n_bins - 1)
                        channel = f'inter_{inter_name}_{stat_name}'
                        break
                if ring != -1:
                    break

        if ring >= 0:
            ring_fdrs[ring].append(fdr[d])
            ring_channel_fdrs[ring][channel].append(fdr[d])

    # Aggregate: mean FDR per ring
    ring_mean_fdr = {}
    for r in range(n_rows):
        if r in ring_fdrs:
            ring_mean_fdr[r] = np.mean(ring_fdrs[r])
        else:
            ring_mean_fdr[r] = 0.0

    # Also compute intra_mean-only FDR per ring (most reliable channel)
    ring_intra_mean_fdr = {}
    for r in range(n_rows):
        vals = ring_channel_fdrs[r].get('intra_mean', [])
        ring_intra_mean_fdr[r] = np.mean(vals) if vals else 0.0

    return ring_mean_fdr, ring_intra_mean_fdr, ring_channel_fdrs


def analyze_dataset(dataset, cache_dir, cache_hash, n_rows, n_bins, stats, inter_stats):
    """Run FDR analysis for one dataset (all its val sequences)."""
    sequences = DATASET_SEQUENCES[dataset]
    elev = SENSOR_ELEVATION[dataset]

    all_pos_diffs_desc = []
    all_neg_diffs_desc = []
    all_descriptors = []
    all_positives = []
    all_negatives = []
    offset = 0

    print(f"\n{'─'*70}")
    print(f"  {dataset.upper()} ({elev['sensor']}): [{elev['min']}°, {elev['max']}°]")
    print(f"{'─'*70}")

    total_kf = 0
    total_pos = 0
    total_neg = 0

    for seq in sequences:
        try:
            descriptors, poses = load_cached_data(cache_dir, cache_hash, seq)
        except FileNotFoundError as e:
            print(f"  SKIP {seq}: {e}")
            continue

        positives, negatives = find_pairs(poses)
        n_kf = len(descriptors)
        print(f"  {seq}: {n_kf} kf, {len(positives)} pos, {len(negatives)} neg pairs")

        if len(positives) == 0:
            print(f"  SKIP {seq}: no loop closures")
            continue

        # Offset indices for concatenation
        adj_pos = [(a + offset, b + offset) for a, b in positives]
        adj_neg = [(a + offset, b + offset) for a, b in negatives]

        all_descriptors.append(descriptors)
        all_positives.extend(adj_pos)
        all_negatives.extend(adj_neg)
        offset += n_kf
        total_kf += n_kf
        total_pos += len(positives)
        total_neg += len(negatives)

    if not all_descriptors:
        return None, None

    print(f"  TOTAL: {total_kf} kf, {total_pos} pos, {total_neg} neg")

    all_desc = np.concatenate(all_descriptors, axis=0)
    ring_mean_fdr, ring_intra_mean_fdr, ring_ch_fdrs = compute_ring_fdr(
        all_desc, all_positives, all_negatives, n_rows, n_bins, stats, inter_stats
    )

    return ring_mean_fdr, ring_intra_mean_fdr


def main():
    parser = argparse.ArgumentParser(description='Ring vs Angle FDR Analysis')
    parser.add_argument('--cache-dir', default='data/preprocessed')
    parser.add_argument('--cache-hash', default='056e0a02')
    parser.add_argument('--n-rows', type=int, default=N_ROWS)
    parser.add_argument('--n-bins', type=int, default=N_BINS)
    parser.add_argument('--stats', nargs='+', default=['mean', 'std'])
    parser.add_argument('--inter-stats', nargs='+', default=['diff'])
    args = parser.parse_args()

    print("=" * 70)
    print("RING NUMBER vs ELEVATION ANGLE — 센서별 FDR 비교")
    print(f"Range image: {args.n_rows} rows × {args.n_bins} bins (octave)")
    print("=" * 70)

    datasets = ['kitti', 'nclt', 'helipr']
    results = {}  # dataset → (ring_mean_fdr, ring_intra_mean_fdr)

    for ds in datasets:
        ring_mean, ring_intra = analyze_dataset(
            ds, args.cache_dir, args.cache_hash,
            args.n_rows, args.n_bins, args.stats, args.inter_stats,
        )
        if ring_mean is not None:
            results[ds] = (ring_mean, ring_intra)

    if not results:
        print("\nNo data found!")
        return 1

    # =====================================================================
    # 1. Per-ring FDR comparison (all channels)
    # =====================================================================
    print(f"\n{'='*70}")
    print("1. PER-RING MEAN FDR (all channels)")
    print(f"{'='*70}")

    header = f"{'Ring':>4}"
    for ds in datasets:
        if ds in results:
            elev = SENSOR_ELEVATION[ds]
            header += f"  {'FDR':>8} {'angle':>7}"
    print(header)
    print("-" * (4 + 17 * len(results)))

    for r in range(args.n_rows):
        row = f"{r:>4}"
        for ds in datasets:
            if ds in results:
                fdr_val = results[ds][0][r]
                angle = ring_to_angle(r, ds, args.n_rows)
                row += f"  {fdr_val:>8.4f} {angle:>6.1f}°"
        print(row)

    # =====================================================================
    # 2. Per-ring FDR comparison (intra_mean only — most reliable)
    # =====================================================================
    print(f"\n{'='*70}")
    print("2. PER-RING INTRA_MEAN FDR (dead-dim-free)")
    print(f"{'='*70}")

    header = f"{'Ring':>4}"
    for ds in datasets:
        if ds in results:
            elev = SENSOR_ELEVATION[ds]
            header += f"  {ds:>8} {'angle':>7}"
    print(header)
    print("-" * (4 + 17 * len(results)))

    for r in range(args.n_rows):
        row = f"{r:>4}"
        for ds in datasets:
            if ds in results:
                fdr_val = results[ds][1][r]
                angle = ring_to_angle(r, ds, args.n_rows)
                row += f"  {fdr_val:>8.4f} {angle:>6.1f}°"
        print(row)

    # =====================================================================
    # 3. Correlation analysis
    # =====================================================================
    print(f"\n{'='*70}")
    print("3. CORRELATION ANALYSIS: FDR ~ ring_number vs FDR ~ elevation_angle")
    print(f"{'='*70}")

    for ds in datasets:
        if ds not in results:
            continue

        ring_fdr_vals = np.array([results[ds][1][r] for r in range(args.n_rows)])
        ring_numbers = np.arange(args.n_rows)
        angles = np.array([ring_to_angle(r, ds, args.n_rows) for r in range(args.n_rows)])

        # Spearman correlation (monotonic relationship)
        corr_ring, p_ring = spearmanr(ring_numbers, ring_fdr_vals)
        corr_angle, p_angle = spearmanr(angles, ring_fdr_vals)

        # Pearson correlation (linear relationship)
        pcorr_ring, pp_ring = pearsonr(ring_numbers, ring_fdr_vals)
        pcorr_angle, pp_angle = pearsonr(angles, ring_fdr_vals)

        elev = SENSOR_ELEVATION[ds]
        print(f"\n  {ds.upper()} ({elev['sensor']}, [{elev['min']}°, {elev['max']}°]):")
        print(f"    Spearman(ring_num,  FDR) = {corr_ring:+.3f}  (p={p_ring:.4f})")
        print(f"    Spearman(elev_angle, FDR) = {corr_angle:+.3f}  (p={p_angle:.4f})")
        print(f"    Pearson (ring_num,  FDR) = {pcorr_ring:+.3f}  (p={pp_ring:.4f})")
        print(f"    Pearson (elev_angle, FDR) = {pcorr_angle:+.3f}  (p={pp_angle:.4f})")

        # Which rings are top-3?
        top3 = np.argsort(ring_fdr_vals)[-3:][::-1]
        print(f"    Top-3 rings: {', '.join(f'R{r}({ring_to_angle(r, ds):.1f}°)' for r in top3)}")
        bot3 = np.argsort(ring_fdr_vals)[:3]
        print(f"    Bot-3 rings: {', '.join(f'R{r}({ring_to_angle(r, ds):.1f}°)' for r in bot3)}")

    # =====================================================================
    # 4. Cross-sensor angle alignment — do same angles have similar FDR?
    # =====================================================================
    print(f"\n{'='*70}")
    print("4. CROSS-SENSOR ANGLE ALIGNMENT")
    print("   같은 각도에서 FDR이 비슷한가?")
    print(f"{'='*70}")

    # Collect all (angle, fdr, dataset) points
    all_points = []
    for ds in results:
        for r in range(args.n_rows):
            angle = ring_to_angle(r, ds, args.n_rows)
            fdr_val = results[ds][1][r]
            all_points.append((angle, fdr_val, ds))

    # Find overlapping angle ranges
    angle_ranges = {}
    for ds in results:
        elev = SENSOR_ELEVATION[ds]
        angle_ranges[ds] = (elev['min'], elev['max'])

    # Overlapping range across all datasets
    all_min = max(ar[0] for ar in angle_ranges.values())
    all_max = min(ar[1] for ar in angle_ranges.values())
    print(f"\n  공통 각도 범위: [{all_min:.1f}°, {all_max:.1f}°]")

    # For pairs of datasets, compute correlation on overlapping angle bins
    ds_list = list(results.keys())
    for i in range(len(ds_list)):
        for j in range(i + 1, len(ds_list)):
            ds_a, ds_b = ds_list[i], ds_list[j]
            # Interpolate FDR at common angle grid
            angles_a = np.array([ring_to_angle(r, ds_a) for r in range(args.n_rows)])
            fdr_a = np.array([results[ds_a][1][r] for r in range(args.n_rows)])
            angles_b = np.array([ring_to_angle(r, ds_b) for r in range(args.n_rows)])
            fdr_b = np.array([results[ds_b][1][r] for r in range(args.n_rows)])

            # Common angle range
            overlap_min = max(angles_a.min(), angles_b.min())
            overlap_max = min(angles_a.max(), angles_b.max())

            if overlap_min >= overlap_max:
                print(f"  {ds_a} vs {ds_b}: 겹치는 각도 없음")
                continue

            # Interpolate both to common grid
            n_grid = 50
            grid = np.linspace(overlap_min, overlap_max, n_grid)
            interp_a = np.interp(grid, np.sort(angles_a), fdr_a[np.argsort(angles_a)])
            interp_b = np.interp(grid, np.sort(angles_b), fdr_b[np.argsort(angles_b)])

            corr, p = spearmanr(interp_a, interp_b)
            pcorr, pp = pearsonr(interp_a, interp_b)

            print(f"\n  {ds_a.upper()} vs {ds_b.upper()} (overlap [{overlap_min:.1f}°, {overlap_max:.1f}°]):")
            print(f"    Spearman = {corr:+.3f} (p={p:.4f})")
            print(f"    Pearson  = {pcorr:+.3f} (p={pp:.4f})")

    # =====================================================================
    # 5. Normalized ring position analysis
    # =====================================================================
    print(f"\n{'='*70}")
    print("5. NORMALIZED POSITION: ring/n_rows (0=top, 1=bottom)")
    print("   Ring 번호 자체가 중요하다면 normalized position과 FDR이 일관되어야 함")
    print(f"{'='*70}")

    norm_positions = np.linspace(0, 1, args.n_rows)

    header = f"{'Norm':>5}"
    for ds in results:
        header += f"  {ds:>8}"
    print(header)
    print("-" * (5 + 10 * len(results)))

    for r in range(args.n_rows):
        row = f"{norm_positions[r]:>5.2f}"
        for ds in results:
            row += f"  {results[ds][1][r]:>8.4f}"
        print(row)

    # Cross-dataset correlation on normalized position
    print(f"\n  센서 간 normalized position FDR 상관:")
    for i in range(len(ds_list)):
        for j in range(i + 1, len(ds_list)):
            ds_a, ds_b = ds_list[i], ds_list[j]
            fdr_a = np.array([results[ds_a][1][r] for r in range(args.n_rows)])
            fdr_b = np.array([results[ds_b][1][r] for r in range(args.n_rows)])
            corr, p = spearmanr(fdr_a, fdr_b)
            print(f"    {ds_a} vs {ds_b}: Spearman = {corr:+.3f} (p={p:.4f})")

    # =====================================================================
    # 6. Summary
    # =====================================================================
    print(f"\n{'='*70}")
    print("6. SUMMARY")
    print(f"{'='*70}")

    # Compare: are sensors consistent by ring number or by angle?
    # If by ring number: all sensors have similar FDR at same ring index
    # If by angle: sensors with overlapping angles have similar FDR at same angle
    print("""
  판단 기준:
  - Ring 번호가 중요: 센서 간 같은 ring에서 FDR 상관 높음 (Section 5)
  - 실제 각도가 중요: 같은 각도에서 FDR 상관 높음 (Section 4)
  - 둘 다 아님: 환경/장면 구조에 따라 다름 (sequence-specific)
""")

    return 0


if __name__ == '__main__':
    sys.exit(main())
