"""
Find optimal cosine similarity threshold for loop closure detection.

Sweeps cosine similarity thresholds on spectral descriptors and evaluates
precision/recall/F1 against GT pose distance for true loop closure detection.

Uses the same data pipeline as visualize_spectral_graph.py.

Usage (run inside container):
    python scripts/find_optimal_similarity_threshold.py --dataset both
    python scripts/find_optimal_similarity_threshold.py --dataset nclt --lc-distance 5.0
    python scripts/find_optimal_similarity_threshold.py --dataset town01 --n-steps 300
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import faiss
from pathlib import Path
from scipy.spatial.distance import cdist
import csv

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from encoding.spectral_encoder import SpectralEncoder

OUTPUT_DIR = Path("/workspace/Neural-Spectral-Codec/outputs")

DISTANCE_THRESHOLD = 3.0      # Keyframe selection distance (m)
TEMPORAL_NEIGHBORS = 15        # Matching training config

# Sensor configs matching training_multi_dataset.yaml
SENSOR_CONFIGS = {
    'nclt': {
        'elevation_range': (-30.67, 10.67),
        'n_elevation': 32,  # HDL-32E
    },
    'helipr': {
        'elevation_range': (-15.0, 15.0),
        'n_elevation': 16,  # VLP-16
    },
}


# ── Point cloud loaders ────────────────────────────────────
def load_nclt_bin(filepath):
    """Load NCLT bin → (N, 4) [x, y, z, intensity]"""
    nclt_dtype = np.dtype([
        ('x', '<u2'), ('y', '<u2'), ('z', '<u2'),
        ('intensity', 'u1'), ('padding', 'u1'), ('extra', '<u4')
    ])
    raw = np.fromfile(filepath, dtype=nclt_dtype)
    x = raw['x'].astype(np.float64) * 0.005 - 100.0
    y = raw['y'].astype(np.float64) * 0.005 - 100.0
    z = raw['z'].astype(np.float64) * 0.005 - 100.0
    intensity = raw['intensity'].astype(np.float32) / 255.0
    pts = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts[:, :3]) < 200).all(axis=1)
    return pts[valid]


def load_helipr_bin(filepath):
    """Load HeLiPR Velodyne bin → (N, 4) [x, y, z, intensity]"""
    dt = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('intensity', np.float32), ('ring', np.uint16), ('time', np.float32)
    ])
    data = np.fromfile(filepath, dtype=dt)
    pts = np.stack([data['x'], data['y'], data['z'], data['intensity']], axis=-1)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts[:, :3]) < 200).all(axis=1)
    return pts[valid]


# ── Pose loaders ───────────────────────────────────────────
def load_nclt_poses(gt_file):
    """→ (N, 7) [ts, x, y, z, roll, pitch, yaw]"""
    poses = []
    with open(gt_file) as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            try:
                vals = list(map(float, row))
            except ValueError:
                continue
            if np.isnan(vals[1]):
                continue
            poses.append(vals)
    return np.array(poses)


def load_helipr_poses(gt_file):
    """→ (N, 4) [ts, x, y, z]"""
    poses = []
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                vals = list(map(float, parts))
            except ValueError:
                continue
            poses.append([vals[0], vals[1], vals[2], vals[3]])
    return np.array(poses)


def match_timestamps(scan_ts, gt_ts):
    """Find nearest GT index for each scan timestamp."""
    idx = np.searchsorted(gt_ts, scan_ts)
    idx = np.clip(idx, 1, len(gt_ts) - 1)
    left = np.abs(gt_ts[idx - 1] - scan_ts)
    right = np.abs(gt_ts[idx] - scan_ts)
    return np.where(left < right, idx - 1, idx)


# ── Keyframe selection ─────────────────────────────────────
def select_keyframes_by_distance(positions, dist_thresh):
    kf_idx = [0]
    last = positions[0]
    for i in range(1, len(positions)):
        if np.linalg.norm(positions[i] - last) >= dist_thresh:
            kf_idx.append(i)
            last = positions[i]
    return np.array(kf_idx)


# ── Dataset preparation ────────────────────────────────────
def prepare_dataset(name):
    """Load poses, scan files, compute descriptors, select keyframes."""
    print(f"\n{'='*60}")
    print(f"Preparing {name}...")

    if name == 'nclt':
        gt_file = Path("/workspace/data/nclt/ground_truth/2012-01-08.csv")
        scan_dir = Path("/workspace/data/nclt/2012-01-08/velodyne_sync")
        cfg = SENSOR_CONFIGS['nclt']
        load_bin = load_nclt_bin

        raw_gt = load_nclt_poses(gt_file)
        gt_ts = raw_gt[:, 0]
        gt_xyz = raw_gt[:, 1:4]

        scan_files = sorted(scan_dir.glob("*.bin"))
        scan_ts = np.array([int(f.stem) for f in scan_files], dtype=np.float64)

        matched = match_timestamps(scan_ts, gt_ts)
        positions = gt_xyz[matched]

    elif name == 'town01':
        gt_file = Path("/workspace/data/helipr/Town01/Town01/LiDAR_GT/Velodyne_gt.txt")
        scan_dir = Path("/workspace/data/helipr/Town01/Town01/LiDAR/Velodyne")
        cfg = SENSOR_CONFIGS['helipr']
        load_bin = load_helipr_bin

        raw_gt = load_helipr_poses(gt_file)
        gt_ts = raw_gt[:, 0]
        gt_xyz = raw_gt[:, 1:4]

        scan_files = sorted(scan_dir.glob("*.bin"))
        scan_ts = np.array([int(f.stem) for f in scan_files], dtype=np.float64)

        matched = match_timestamps(scan_ts, gt_ts)
        positions = gt_xyz[matched]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    print(f"  Scans: {len(scan_files):,}, GT: {len(gt_ts):,}")

    # Select keyframes
    kf_scan_idx = select_keyframes_by_distance(positions, DISTANCE_THRESHOLD)
    n_kf = len(kf_scan_idx)
    kf_positions_3d = positions[kf_scan_idx]  # (n_kf, 3) XYZ
    print(f"  Keyframes: {n_kf} (Δd={DISTANCE_THRESHOLD}m)")

    # Compute spectral descriptors
    encoder = SpectralEncoder(
        n_elevation=cfg['n_elevation'],
        n_azimuth=360,
        n_bins=16,
        alpha=2.0,
        learnable_alpha=False,
        target_elevation_bins=16,
        elevation_range=cfg['elevation_range'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    encoder.eval()

    descriptors = []
    device = encoder.alpha.device
    print(f"  Computing descriptors on {device}...", flush=True)

    for i, si in enumerate(kf_scan_idx):
        pts = load_bin(scan_files[si])
        with torch.no_grad():
            desc = encoder.encode_points(pts)
        descriptors.append(desc.cpu().numpy())

        if (i + 1) % 200 == 0 or i == n_kf - 1:
            print(f"    [{i+1}/{n_kf}]", flush=True)

    descriptors = np.array(descriptors)
    print(f"  Descriptor shape: {descriptors.shape}")

    return {
        'name': name,
        'kf_positions_3d': kf_positions_3d,
        'descriptors': descriptors,
        'n_keyframes': n_kf,
        'n_total_scans': len(scan_files),
    }


# ── Analysis functions ─────────────────────────────────────
def build_temporal_sets(n_kf, temporal_neighbors):
    """Build temporal neighbor sets for exclusion."""
    hw = temporal_neighbors // 2
    temporal_sets = {}
    for i in range(n_kf):
        s = set()
        for off in range(-hw, hw + 1):
            if off == 0:
                continue
            j = i + off
            if 0 <= j < n_kf:
                s.add(j)
        temporal_sets[i] = s
    return temporal_sets, hw


def compute_all_candidate_edges(descriptors, temporal_sets, fetch_k=200, floor_sim=0.90):
    """
    Get all candidate similarity edges above floor threshold.

    Returns:
        List of (i, j, cos_sim) tuples, sorted by cos_sim descending
    """
    n, d = descriptors.shape
    desc_f32 = descriptors.astype(np.float32).copy()
    norms = np.linalg.norm(desc_f32, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    desc_norm = desc_f32 / norms

    fetch_k = min(fetch_k, n)

    index = faiss.IndexFlatIP(d)
    index.add(desc_norm)
    sims, indices = index.search(desc_norm, fetch_k)

    edges = []
    seen = set()

    for i in range(n):
        ts = temporal_sets.get(i, set())
        for jp in range(fetch_k):
            j = int(indices[i, jp])
            if j == i or j in ts:
                continue
            cos_sim = float(sims[i, jp])
            if cos_sim < floor_sim:
                break

            # Deduplicate: store each undirected pair once
            pair = (min(i, j), max(i, j))
            if pair not in seen:
                seen.add(pair)
                edges.append((i, j, cos_sim))

    # Sort by cos_sim descending
    edges.sort(key=lambda x: x[2], reverse=True)
    print(f"  Candidate edges (floor={floor_sim}): {len(edges):,}")
    return edges


def sweep_thresholds(candidate_edges, gt_dist_matrix, thresholds,
                     lc_distance, min_temporal_gap):
    """
    Sweep cosine thresholds and compute precision/recall/F1.

    Returns:
        List of dicts: [{threshold, n_edges, n_true_lc, precision, recall, f1}, ...]
    """
    n = gt_dist_matrix.shape[0]

    # Compute total true LC pairs (recall denominator)
    total_true_lc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) > min_temporal_gap and gt_dist_matrix[i, j] < lc_distance:
                total_true_lc += 1

    print(f"  Total true LC pairs (GT < {lc_distance}m, gap > {min_temporal_gap}): {total_true_lc:,}")

    # Precompute: for each candidate edge, whether it's a true LC
    edge_data = []
    for i, j, cos_sim in candidate_edges:
        is_true_lc = (abs(i - j) > min_temporal_gap and
                      gt_dist_matrix[i, j] < lc_distance)
        edge_data.append((cos_sim, is_true_lc))

    # Edges are sorted by cos_sim descending, so we can compute cumulatively
    results = []
    for thresh in sorted(thresholds, reverse=True):
        n_edges = 0
        n_true_lc = 0
        for cos_sim, is_true in edge_data:
            if cos_sim < thresh:
                break
            n_edges += 1
            if is_true:
                n_true_lc += 1

        precision = n_true_lc / n_edges if n_edges > 0 else 0.0
        recall = n_true_lc / total_true_lc if total_true_lc > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            'threshold': thresh,
            'n_edges': n_edges,
            'n_true_lc': n_true_lc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

    # Sort by threshold ascending for output
    results.sort(key=lambda x: x['threshold'])
    return results, total_true_lc


def plot_results(results, dataset_name, total_true_lc, output_path):
    """Plot precision/recall/F1 curves and save."""
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    n_edges_list = [r['n_edges'] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor('white')

    # Top: P/R/F1
    ax1.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
    ax1.plot(thresholds, recalls, 'g-', linewidth=2, label='Recall')
    ax1.plot(thresholds, f1s, 'r-', linewidth=2, label='F1')

    # Mark best points
    best_p_idx = max(range(len(results)), key=lambda i: results[i]['precision']
                     if results[i]['n_edges'] >= 5 else -1)
    best_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1'])

    ax1.axvline(results[best_p_idx]['threshold'], color='blue', linestyle='--',
                alpha=0.5, linewidth=1)
    ax1.axvline(results[best_f1_idx]['threshold'], color='red', linestyle='--',
                alpha=0.5, linewidth=1)

    # Current threshold line
    ax1.axvline(0.993, color='gray', linestyle=':', alpha=0.7, linewidth=1.5,
                label='Current (0.993)')

    ax1.annotate(
        f"Best P: {results[best_p_idx]['threshold']:.4f}\n"
        f"P={results[best_p_idx]['precision']:.1%}",
        xy=(results[best_p_idx]['threshold'], results[best_p_idx]['precision']),
        xytext=(15, 15), textcoords='offset points',
        fontsize=9, color='blue',
        arrowprops=dict(arrowstyle='->', color='blue', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    ax1.annotate(
        f"Best F1: {results[best_f1_idx]['threshold']:.4f}\n"
        f"F1={results[best_f1_idx]['f1']:.1%}",
        xy=(results[best_f1_idx]['threshold'], results[best_f1_idx]['f1']),
        xytext=(15, -25), textcoords='offset points',
        fontsize=9, color='red',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    ax1.set_ylabel('Score', fontsize=13)
    ax1.set_title(f'{dataset_name} — Similarity Threshold Analysis\n'
                  f'(True LC: GT < 5m, total true LC pairs: {total_true_lc:,})',
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.02, 1.02)

    # Bottom: Edge count
    ax2.plot(thresholds, n_edges_list, 'k-', linewidth=1.5)
    ax2.fill_between(thresholds, n_edges_list, alpha=0.1, color='gray')

    # Mark true LC count
    n_true_lc_list = [r['n_true_lc'] for r in results]
    ax2.plot(thresholds, n_true_lc_list, 'g-', linewidth=1.5, label='True LC edges')
    ax2.fill_between(thresholds, n_true_lc_list, alpha=0.15, color='green')

    ax2.axvline(0.993, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Cosine Similarity Threshold', fontsize=13)
    ax2.set_ylabel('Edge Count', fontsize=13)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def print_results_table(results, dataset_name, total_true_lc):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"  {dataset_name}")
    print(f"  Total true LC pairs: {total_true_lc:,}")
    print(f"{'='*80}")
    print(f"{'Threshold':>10} | {'Edges':>8} | {'True LC':>8} | {'Precision':>10} | "
          f"{'Recall':>8} | {'F1':>8}")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    # Print subset of results (every Nth + key points)
    step = max(1, len(results) // 40)
    printed_thresholds = set()

    for i, r in enumerate(results):
        # Always print if threshold is near current (0.993) or a round number
        is_key = (abs(r['threshold'] - 0.993) < 0.0005 or
                  r['threshold'] * 1000 % 5 < 0.5)
        if i % step == 0 or is_key:
            if r['threshold'] not in printed_thresholds:
                printed_thresholds.add(r['threshold'])
                print(f"{r['threshold']:>10.4f} | {r['n_edges']:>8,} | {r['n_true_lc']:>8,} | "
                      f"{r['precision']:>9.1%} | {r['recall']:>7.1%} | {r['f1']:>7.1%}")

    # Best results (min 5 edges for precision to avoid trivial cases)
    valid_results = [r for r in results if r['n_edges'] >= 5]
    if valid_results:
        best_p = max(valid_results, key=lambda r: r['precision'])
        print(f"\n  ★ Best Precision (≥5 edges): threshold={best_p['threshold']:.4f}, "
              f"P={best_p['precision']:.1%} ({best_p['n_true_lc']}/{best_p['n_edges']} edges)")

    best_f1 = max(results, key=lambda r: r['f1'])
    print(f"  ★ Best F1:                   threshold={best_f1['threshold']:.4f}, "
          f"F1={best_f1['f1']:.1%} (P={best_f1['precision']:.1%}, R={best_f1['recall']:.1%})")

    # Show current threshold performance
    current = min(results, key=lambda r: abs(r['threshold'] - 0.993))
    print(f"  ● Current (0.993):           "
          f"P={current['precision']:.1%}, R={current['recall']:.1%}, "
          f"F1={current['f1']:.1%} ({current['n_true_lc']}/{current['n_edges']} edges)")


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal cosine similarity threshold for loop closure')
    parser.add_argument('--dataset', default='both',
                        choices=['nclt', 'town01', 'both'])
    parser.add_argument('--lc-distance', type=float, default=5.0,
                        help='GT distance threshold for true loop closure (m)')
    parser.add_argument('--min-threshold', type=float, default=0.90,
                        help='Minimum cosine threshold to sweep')
    parser.add_argument('--max-threshold', type=float, default=0.999,
                        help='Maximum cosine threshold to sweep')
    parser.add_argument('--n-steps', type=int, default=200,
                        help='Number of threshold steps')
    parser.add_argument('--fetch-k', type=int, default=200,
                        help='FAISS top-K per node for candidate edges')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    names = []
    if args.dataset in ('nclt', 'both'):
        names.append('nclt')
    if args.dataset in ('town01', 'both'):
        names.append('town01')

    # Build threshold sweep: coarse + fine near high end
    thresholds_coarse = np.linspace(args.min_threshold, args.max_threshold, args.n_steps)
    thresholds_fine = np.linspace(0.990, 0.999, 50)
    thresholds = np.unique(np.concatenate([thresholds_coarse, thresholds_fine]))
    thresholds.sort()
    print(f"Sweeping {len(thresholds)} thresholds: [{thresholds[0]:.4f}, {thresholds[-1]:.4f}]")

    hw = TEMPORAL_NEIGHBORS // 2
    min_temporal_gap = 2 * hw  # Match visualization script's revisit definition

    all_results = {}

    for name in names:
        data = prepare_dataset(name)
        n_kf = data['n_keyframes']
        descriptors = data['descriptors']
        kf_pos = data['kf_positions_3d']

        # Build temporal neighbor sets
        temporal_sets, _ = build_temporal_sets(n_kf, TEMPORAL_NEIGHBORS)

        # Get all candidate edges
        print(f"\n  Computing candidate similarity edges...")
        candidate_edges = compute_all_candidate_edges(
            descriptors, temporal_sets, fetch_k=args.fetch_k, floor_sim=args.min_threshold)

        # GT pairwise distances
        print(f"  Computing GT pairwise distances ({n_kf}×{n_kf})...")
        gt_dist_matrix = cdist(kf_pos, kf_pos, metric='euclidean')

        # Sweep thresholds
        print(f"  Sweeping {len(thresholds)} thresholds...")
        results, total_true_lc = sweep_thresholds(
            candidate_edges, gt_dist_matrix, thresholds,
            lc_distance=args.lc_distance, min_temporal_gap=min_temporal_gap)

        all_results[name] = (results, total_true_lc)

        # Print table
        title_map = {'nclt': 'NCLT 2012-01-08', 'town01': 'HeLiPR Town01'}
        print_results_table(results, title_map.get(name, name), total_true_lc)

        # Plot
        out_path = OUTPUT_DIR / f"threshold_analysis_{name}.png"
        plot_results(results, title_map.get(name, name), total_true_lc, out_path)

    # Combined summary if both datasets
    if len(names) > 1:
        print(f"\n{'='*80}")
        print(f"  COMBINED SUMMARY")
        print(f"{'='*80}")

        # Merge results across datasets per threshold
        combined = {}
        for name, (results, _) in all_results.items():
            for r in results:
                t = r['threshold']
                if t not in combined:
                    combined[t] = {'n_edges': 0, 'n_true_lc': 0}
                combined[t]['n_edges'] += r['n_edges']
                combined[t]['n_true_lc'] += r['n_true_lc']

        total_lc_all = sum(tlc for _, (_, tlc) in all_results.items())

        combined_results = []
        for t in sorted(combined.keys()):
            c = combined[t]
            p = c['n_true_lc'] / c['n_edges'] if c['n_edges'] > 0 else 0
            rec = c['n_true_lc'] / total_lc_all if total_lc_all > 0 else 0
            f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
            combined_results.append({
                'threshold': t, 'n_edges': c['n_edges'],
                'n_true_lc': c['n_true_lc'],
                'precision': p, 'recall': rec, 'f1': f1,
            })

        print_results_table(combined_results, "Combined (NCLT + Town01)", total_lc_all)

        out_path = OUTPUT_DIR / "threshold_analysis_combined.png"
        plot_results(combined_results, "Combined (NCLT + Town01)", total_lc_all, out_path)


if __name__ == "__main__":
    main()
