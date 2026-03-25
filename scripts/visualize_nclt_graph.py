"""
Visualize trajectory with temporal and similarity edges.
Supports NCLT and HeLiPR datasets.

Usage:
    python visualize_nclt_graph.py                    # NCLT 2012-01-08
    python visualize_nclt_graph.py --dataset town01    # HeLiPR Town01
    python visualize_nclt_graph.py --dataset both      # Both side by side
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.collections as mcoll
from pathlib import Path
from scipy.spatial.transform import Rotation
import csv
import faiss

# ── Config ──────────────────────────────────────────────────
OUTPUT_DIR = Path("/workspace/Neural-Spectral-Codec/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISTANCE_THRESHOLD = 3.0
TEMPORAL_NEIGHBORS = 15  # half-window = 7
SIMILARITY_K = 5


# ── Pose loaders ───────────────────────────────────────────
def load_nclt_poses(gt_file: Path) -> np.ndarray:
    """Load NCLT ground truth CSV → (N, 3) [x, y, z]"""
    poses = []
    with open(gt_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            try:
                vals = list(map(float, row))
            except ValueError:
                continue
            if np.isnan(vals[1]) or np.isnan(vals[2]):
                continue
            poses.append([vals[1], vals[2], vals[3]])
    return np.array(poses)


def load_helipr_poses(gt_file: Path) -> np.ndarray:
    """Load HeLiPR ground truth (quaternion) → (N, 3) [x, y, z]"""
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
            poses.append([vals[1], vals[2], vals[3]])
    return np.array(poses)


# ── Graph construction ─────────────────────────────────────
def select_keyframes(positions: np.ndarray, dist_thresh: float) -> np.ndarray:
    keyframe_indices = [0]
    last_pos = positions[0]
    for i in range(1, len(positions)):
        if np.linalg.norm(positions[i] - last_pos) >= dist_thresh:
            keyframe_indices.append(i)
            last_pos = positions[i]
    return np.array(keyframe_indices)


def build_temporal_edges(n_nodes, temporal_neighbors):
    edges = []
    half_window = temporal_neighbors // 2
    for i in range(n_nodes):
        for offset in range(-half_window, half_window + 1):
            if offset == 0:
                continue
            j = i + offset
            if 0 <= j < n_nodes:
                edges.append((i, j))
    return edges


def build_similarity_edges(positions_2d, similarity_k, temporal_neighbor_sets):
    n = len(positions_2d)
    pos_f32 = positions_2d.astype(np.float32).copy()
    index = faiss.IndexFlatL2(pos_f32.shape[1])
    index.add(pos_f32)
    max_temporal = max((len(s) for s in temporal_neighbor_sets.values()), default=0)
    fetch_k = min(similarity_k + max_temporal + 1, n)
    _, indices = index.search(pos_f32, fetch_k)

    edges = []
    for i in range(n):
        count = 0
        temporal_set = temporal_neighbor_sets.get(i, set())
        for j_pos in range(fetch_k):
            j = int(indices[i, j_pos])
            if j == i or j in temporal_set:
                continue
            edges.append((i, j))
            count += 1
            if count >= similarity_k:
                break
    return edges


# ── Plotting ───────────────────────────────────────────────
def plot_dual_edge_graph(ax, positions_3d, title, dist_thresh=DISTANCE_THRESHOLD):
    """Plot trajectory with temporal + similarity edges on given axes."""
    half_window = TEMPORAL_NEIGHBORS // 2

    # Subsample for trajectory line
    step = max(1, len(positions_3d) // 50000)
    traj_xy = positions_3d[::step, :2]

    # Select keyframes
    kf_indices = select_keyframes(positions_3d, dist_thresh)
    n_kf = len(kf_indices)
    kf_xy = positions_3d[kf_indices, :2]

    # Build temporal edges + sets
    temporal_edges = build_temporal_edges(n_kf, TEMPORAL_NEIGHBORS)
    temporal_set = {}
    for i in range(n_kf):
        neighbors = set()
        for offset in range(-half_window, half_window + 1):
            if offset == 0:
                continue
            j = i + offset
            if 0 <= j < n_kf:
                neighbors.add(j)
        temporal_set[i] = neighbors

    # Build similarity edges
    sim_edges = build_similarity_edges(kf_xy, SIMILARITY_K, temporal_set)
    revisit_edges = [(i, j) for i, j in sim_edges if abs(i - j) > 2 * half_window]

    # Deduplicate revisit
    revisit_set = set()
    for i, j in revisit_edges:
        revisit_set.add((min(i, j), max(i, j)))
    revisit_unique = sorted(revisit_set)

    print(f"  [{title}]")
    print(f"    GT poses: {len(positions_3d):,}")
    print(f"    Keyframes: {n_kf} (Δd={dist_thresh}m)")
    print(f"    Temporal edges: {len(temporal_edges):,}")
    print(f"    Similarity edges: {len(sim_edges):,}")
    print(f"    Revisit edges (|Δi|>{2*half_window}): {len(revisit_unique)}")

    # 1. Full trajectory
    ax.plot(traj_xy[:, 0], traj_xy[:, 1],
            color='#D0D0D0', linewidth=0.8, alpha=0.5, zorder=1)

    # 2. Temporal chain (±1)
    chain_segments = [[kf_xy[i], kf_xy[i + 1]] for i in range(n_kf - 1)]
    ax.add_collection(mcoll.LineCollection(
        chain_segments, colors='#3377BB', linewidths=1.0, alpha=0.5, zorder=2))

    # Extended temporal (±half_window)
    ext_segments = []
    for i in range(n_kf):
        for offset in [half_window, -half_window]:
            j = i + offset
            if 0 <= j < n_kf:
                ext_segments.append([kf_xy[i], kf_xy[j]])
    if ext_segments:
        ax.add_collection(mcoll.LineCollection(
            ext_segments, colors='#3377BB', linewidths=0.3, alpha=0.12, zorder=2))

    # 3. Similarity edges (all revisits)
    if revisit_unique:
        sim_segments = [[kf_xy[i], kf_xy[j]] for i, j in revisit_unique]
        ax.add_collection(mcoll.LineCollection(
            sim_segments, colors='#DD3333', linewidths=0.5, alpha=0.4,
            linestyles='dashed', zorder=3))

    # 4. Keyframe nodes (colored by temporal order)
    scatter = ax.scatter(
        kf_xy[:, 0], kf_xy[:, 1],
        c=np.arange(n_kf), cmap='viridis', s=12, zorder=5,
        edgecolors='white', linewidths=0.2, alpha=0.85)

    # 5. Start / End
    ax.scatter(kf_xy[0, 0], kf_xy[0, 1], c='#22CC44', s=350, zorder=7,
               marker='^', edgecolors='black', linewidths=1.5)
    ax.scatter(kf_xy[-1, 0], kf_xy[-1, 1], c='#DD3333', s=350, zorder=7,
               marker='s', edgecolors='black', linewidths=1.5)

    # Legend
    handles = [
        mlines.Line2D([], [], color='#3377BB', linewidth=2, alpha=0.6,
                       label=f'Temporal edge (±{half_window})'),
        mlines.Line2D([], [], color='#DD3333', linewidth=2, alpha=0.65,
                       linestyle='--', label=f'Similarity edge (k={SIMILARITY_K})'),
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
    stats = (
        f"GT: {len(positions_3d):,}\n"
        f"KF: {n_kf:,} (Δd={dist_thresh}m)\n"
        f"Temporal: {len(temporal_edges):,}\n"
        f"Similarity: {len(sim_edges):,}\n"
        f" └revisit: {len(revisit_unique):,}"
    )
    ax.text(0.98, 0.02, stats, transform=ax.transAxes,
            fontsize=10, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.92),
            fontfamily='monospace')

    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=11)

    return scatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='both',
                        choices=['nclt', 'town01', 'both'])
    args = parser.parse_args()

    datasets = {}
    if args.dataset in ('nclt', 'both'):
        gt = Path("/workspace/data/nclt/ground_truth/2012-01-08.csv")
        datasets['NCLT 2012-01-08'] = load_nclt_poses(gt)
    if args.dataset in ('town01', 'both'):
        gt = Path("/workspace/data/helipr/Town01/Town01/LiDAR_GT/Velodyne_gt.txt")
        datasets['HeLiPR Town01'] = load_helipr_poses(gt)

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(22 * n, 20))
    fig.patch.set_facecolor('white')
    if n == 1:
        axes = [axes]

    for ax, (name, positions) in zip(axes, datasets.items()):
        scatter = plot_dual_edge_graph(ax, positions, name)

    # Colorbar outside plot area
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.4, pad=0.03,
                         location='right', aspect=30)
    cbar.set_label('Keyframe index (temporal order)', fontsize=13)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle('Dual-Edge Pose Graph — Temporal (blue) + Similarity (red)',
                 fontsize=22, fontweight='bold', y=1.01)

    suffix = args.dataset if args.dataset != 'both' else 'nclt_town01'
    out_path = OUTPUT_DIR / f"dual_edge_graph_{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
