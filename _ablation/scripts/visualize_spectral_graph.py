"""
Visualize trajectory with REAL spectral descriptor-based similarity edges.

Computes actual spectral histograms for each keyframe using SpectralEncoder,
then builds similarity edges via cosine threshold on descriptor space.

Usage:
    python visualize_spectral_graph.py --dataset nclt
    python visualize_spectral_graph.py --dataset town01
    python visualize_spectral_graph.py --dataset both
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.collections as mcoll
from pathlib import Path
from scipy.spatial.transform import Rotation
import csv
import faiss

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from encoding.spectral_encoder import SpectralEncoder

OUTPUT_DIR = Path("/workspace/Neural-Spectral-Codec/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISTANCE_THRESHOLD = 3.0
TEMPORAL_NEIGHBORS = 15
SIMILARITY_THRESHOLD = 0.993
SIMILARITY_MAX_K = 10

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


# ── Graph construction ─────────────────────────────────────
def select_keyframes_by_distance(positions, dist_thresh):
    kf_idx = [0]
    last = positions[0]
    for i in range(1, len(positions)):
        if np.linalg.norm(positions[i] - last) >= dist_thresh:
            kf_idx.append(i)
            last = positions[i]
    return np.array(kf_idx)


def build_temporal_edges(n, M):
    edges = []
    hw = M // 2
    for i in range(n):
        for off in range(-hw, hw + 1):
            if off == 0:
                continue
            j = i + off
            if 0 <= j < n:
                edges.append((i, j))
    return edges


def build_similarity_edges_cosine(descriptors, threshold, max_k, temporal_sets):
    """FAISS cosine threshold on actual spectral descriptors."""
    n, d = descriptors.shape
    desc_f32 = descriptors.astype(np.float32).copy()
    norms = np.linalg.norm(desc_f32, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    desc_norm = desc_f32 / norms

    max_t = max((len(s) for s in temporal_sets.values()), default=0)
    fetch_k = min(max_k + max_t + 1, n)

    index = faiss.IndexFlatIP(d)
    index.add(desc_norm)
    sims, indices = index.search(desc_norm, fetch_k)

    edges = []
    for i in range(n):
        count = 0
        ts = temporal_sets.get(i, set())
        for jp in range(fetch_k):
            j = int(indices[i, jp])
            if j == i or j in ts:
                continue
            cos_sim = float(sims[i, jp])
            if cos_sim < threshold:
                break  # sorted descending; rest will be lower
            edges.append((i, j, cos_sim))
            count += 1
            if count >= max_k:
                break
    return edges


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
    kf_xy = positions[kf_scan_idx, :2]
    print(f"  Keyframes: {n_kf} (Δd={DISTANCE_THRESHOLD}m)")

    # Compute spectral descriptors for keyframes
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
        'positions': positions,
        'kf_scan_idx': kf_scan_idx,
        'kf_xy': kf_xy,
        'descriptors': descriptors,
        'n_total': len(scan_files),
    }


# ── Plotting ───────────────────────────────────────────────
def plot_graph(ax, data):
    name = data['name']
    kf_xy = data['kf_xy']
    descriptors = data['descriptors']
    n_kf = len(kf_xy)
    hw = TEMPORAL_NEIGHBORS // 2

    # Temporal
    temporal_edges = build_temporal_edges(n_kf, TEMPORAL_NEIGHBORS)
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

    # Similarity (REAL descriptors, cosine threshold)
    sim_edges_raw = build_similarity_edges_cosine(
        descriptors, SIMILARITY_THRESHOLD, SIMILARITY_MAX_K, temporal_sets)
    sim_edges = [(i, j) for i, j, _ in sim_edges_raw]
    revisit = [(i, j) for i, j in sim_edges if abs(i - j) > 2 * hw]
    revisit_set = set((min(i,j), max(i,j)) for i, j in revisit)
    revisit_unique = sorted(revisit_set)

    # Compute mean cosine sim
    cos_sims = [s for _, _, s in sim_edges_raw]
    mean_cos = np.mean(cos_sims) if cos_sims else 0

    print(f"  [{name}] Temporal: {len(temporal_edges):,}, Similarity: {len(sim_edges):,}, "
          f"Revisit: {len(revisit_unique)}, Mean cos: {mean_cos:.3f}")

    # Full trajectory (subsampled)
    all_xy = data['positions'][::5, :2]
    ax.plot(all_xy[:, 0], all_xy[:, 1],
            color='#D0D0D0', linewidth=0.8, alpha=0.5, zorder=1)

    # Temporal chain
    chain = [[kf_xy[i], kf_xy[i+1]] for i in range(n_kf - 1)]
    ax.add_collection(mcoll.LineCollection(
        chain, colors='#3377BB', linewidths=1.0, alpha=0.5, zorder=2))

    # Extended temporal
    ext = []
    for i in range(n_kf):
        for off in [hw, -hw]:
            j = i + off
            if 0 <= j < n_kf:
                ext.append([kf_xy[i], kf_xy[j]])
    if ext:
        ax.add_collection(mcoll.LineCollection(
            ext, colors='#3377BB', linewidths=0.3, alpha=0.12, zorder=2))

    # Similarity edges (ALL revisits, real descriptor-based)
    # Split by GT pose distance: <=20m = close (solid), >20m = far (faded)
    DIST_SPLIT = 20.0
    close_seg, far_seg = [], []
    for i, j in revisit_unique:
        gt_dist = np.linalg.norm(kf_xy[i] - kf_xy[j])
        if gt_dist <= DIST_SPLIT:
            close_seg.append([kf_xy[i], kf_xy[j]])
        else:
            far_seg.append([kf_xy[i], kf_xy[j]])
    if far_seg:
        ax.add_collection(mcoll.LineCollection(
            far_seg, colors='#DD3333', linewidths=0.3, alpha=0.12,
            linestyles='dashed', zorder=6))
    if close_seg:
        ax.add_collection(mcoll.LineCollection(
            close_seg, colors='#DD3333', linewidths=0.8, alpha=0.6,
            zorder=7))

    # Keyframe nodes
    scatter = ax.scatter(
        kf_xy[:, 0], kf_xy[:, 1],
        c=np.arange(n_kf), cmap='viridis', s=12, zorder=5,
        edgecolors='white', linewidths=0.2, alpha=0.85)

    # Start / End
    ax.scatter(kf_xy[0, 0], kf_xy[0, 1], c='#22CC44', s=350, zorder=7,
               marker='^', edgecolors='black', linewidths=1.5)
    ax.scatter(kf_xy[-1, 0], kf_xy[-1, 1], c='#DD3333', s=350, zorder=7,
               marker='s', edgecolors='black', linewidths=1.5)

    # Legend
    handles = [
        mlines.Line2D([], [], color='#3377BB', linewidth=2, alpha=0.6,
                       label=f'Temporal edge (±{hw})'),
        mlines.Line2D([], [], color='#DD3333', linewidth=2, alpha=0.65,
                       linestyle='--',
                       label=f'Similarity (cos≥{SIMILARITY_THRESHOLD})'),
        mlines.Line2D([], [], color='#666666', marker='o', linestyle='None',
                       markersize=6, label=f'Keyframes (n={n_kf})'),
        mlines.Line2D([], [], color='#22CC44', marker='^', linestyle='None',
                       markersize=12, markeredgecolor='black', label='Start'),
        mlines.Line2D([], [], color='#DD3333', marker='s', linestyle='None',
                       markersize=12, markeredgecolor='black', label='End'),
    ]
    ax.legend(handles=handles, fontsize=11, loc='upper left',
              framealpha=0.92, edgecolor='#CCCCCC', fancybox=True)

    # Stats
    title_map = {'nclt': 'NCLT 2012-01-08', 'town01': 'HeLiPR Town01'}
    stats = (
        f"Scans: {data['n_total']:,}\n"
        f"KF: {n_kf:,} (Δd={DISTANCE_THRESHOLD}m)\n"
        f"Temporal: {len(temporal_edges):,}\n"
        f"Similarity: {len(sim_edges):,}\n"
        f" └revisit: {len(revisit_unique):,}\n"
        f"Mean cos: {mean_cos:.3f}"
    )
    ax.text(0.98, 0.02, stats, transform=ax.transAxes,
            fontsize=10, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92),
            fontfamily='monospace')

    ax.set_title(title_map.get(name, name), fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=11)

    return scatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='both', choices=['nclt', 'town01', 'both'])
    args = parser.parse_args()

    names = []
    if args.dataset in ('nclt', 'both'):
        names.append('nclt')
    if args.dataset in ('town01', 'both'):
        names.append('town01')

    datasets = [prepare_dataset(n) for n in names]

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(22 * n, 20))
    fig.patch.set_facecolor('white')
    if n == 1:
        axes = [axes]

    for ax, data in zip(axes, datasets):
        scatter = plot_graph(ax, data)

    cbar = fig.colorbar(scatter, ax=axes, shrink=0.4, pad=0.03,
                         location='right', aspect=30)
    cbar.set_label('Keyframe index (temporal order)', fontsize=13)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle('Dual-Edge Pose Graph — Temporal (blue) + Spectral Similarity (red)',
                 fontsize=22, fontweight='bold', y=1.01)

    suffix = args.dataset if args.dataset != 'both' else 'nclt_town01'
    thresh_str = f"{SIMILARITY_THRESHOLD:.2f}".replace('.', '')
    out_path = OUTPUT_DIR / f"spectral_graph_{suffix}_thresh{thresh_str}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
