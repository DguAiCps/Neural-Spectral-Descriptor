"""
Visualize ground truth trajectories for all sequences.
Shows start point (green), end point (red), and intermediate markers every 100 frames.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import csv
from scipy.spatial.transform import Rotation

# ── Data roots ──────────────────────────────────────────────
KITTI_ROOT = Path("/workspace/data/kitti")
NCLT_ROOT = Path("/workspace/data/nclt")
HELIPR_ROOT = Path("/workspace/data/helipr")
OUTPUT_DIR = Path("/workspace/Neural-Spectral-Codec/outputs/trajectories")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MARKER_INTERVAL = 100  # frames


# ── Pose loaders ────────────────────────────────────────────

def load_kitti_poses(seq_id: str) -> np.ndarray:
    """Load KITTI poses from poses.txt → (N, 4, 4)"""
    pose_file = KITTI_ROOT / "dataset" / "poses" / f"{seq_id}.txt"
    if not pose_file.exists():
        pose_file = KITTI_ROOT / "poses" / f"{seq_id}.txt"
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return np.array(poses)


def load_nclt_poses(date: str) -> np.ndarray:
    """Load NCLT ground truth CSV → (N, 4, 4)"""
    gt_file = NCLT_ROOT / date / f"groundtruth_{date}.csv"
    poses = []
    with open(gt_file) as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header if present
        for row in reader:
            if len(row) < 7:
                continue
            try:
                vals = list(map(float, row))
            except ValueError:
                continue
            # timestamp, x, y, z, roll, pitch, yaw
            x, y, z = vals[1], vals[2], vals[3]
            roll, pitch, yaw = vals[4], vals[5], vals[6]
            R = Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            poses.append(T)
    return np.array(poses)


def load_helipr_poses(seq_name: str) -> np.ndarray:
    """Load HeLiPR ground truth (quaternion) → (N, 4, 4)"""
    gt_file = HELIPR_ROOT / seq_name / seq_name / "LiDAR_GT" / "Velodyne_gt.txt"
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
            # timestamp, x, y, z, qx, qy, qz, qw
            x, y, z = vals[1], vals[2], vals[3]
            qx, qy, qz, qw = vals[4], vals[5], vals[6], vals[7]
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            poses.append(T)
    return np.array(poses)


# ── Plotting ────────────────────────────────────────────────

def plot_trajectory(ax, poses: np.ndarray, title: str, axes_idx=(0, 1)):
    """Plot a single trajectory on given axes.
    axes_idx: which translation components to plot.
              (0,1) = X vs Y for NCLT/HeLiPR,  (0,2) = X vs Z for KITTI.
    """
    xy = poses[:, :3, 3][:, list(axes_idx)]  # Extract selected axes
    n = len(xy)

    # Trajectory line
    ax.plot(xy[:, 0], xy[:, 1], color='#4488CC', linewidth=0.8, alpha=0.7, zorder=1)

    # Intermediate markers every MARKER_INTERVAL frames
    marker_indices = list(range(MARKER_INTERVAL, n - 1, MARKER_INTERVAL))
    if marker_indices:
        mx = xy[marker_indices, 0]
        my = xy[marker_indices, 1]
        ax.scatter(mx, my, c='#888888', s=12, zorder=3, marker='o', edgecolors='white', linewidths=0.3)
        # Label frame number
        for idx in marker_indices:
            ax.annotate(str(idx), (xy[idx, 0], xy[idx, 1]),
                        fontsize=4, color='#666666', ha='left', va='bottom',
                        xytext=(2, 2), textcoords='offset points')

    # Start point (green)
    ax.scatter(xy[0, 0], xy[0, 1], c='#22CC44', s=60, zorder=5,
               marker='^', edgecolors='black', linewidths=0.8, label='Start')

    # End point (red)
    ax.scatter(xy[-1, 0], xy[-1, 1], c='#DD3333', s=60, zorder=5,
               marker='s', edgecolors='black', linewidths=0.8, label='End')

    ax.set_title(f"{title}  ({n:,} frames)", fontsize=9, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=6)
    ax.set_xlabel('X (m)', fontsize=7)
    ax.set_ylabel('Y (m)', fontsize=7)


# ── Sequence definitions ────────────────────────────────────

SEQUENCES = {
    'kitti': {
        'train_val': ['00', '05', '08', '09', '10'],
    },
    'nclt': {
        'train_val': ['2012-01-08', '2013-01-10'],
    },
    'helipr': {
        'train_val': [
            'Town01', 'Town02', 'Town03',
            'Roundabout01', 'Roundabout02', 'Roundabout03',
            'Bridge01', 'Bridge02', 'Bridge03', 'Bridge04',
            'KAIST04', 'KAIST05', 'KAIST06',
            'DCC04', 'DCC05', 'DCC06',
            'Riverside04', 'Riverside05', 'Riverside06',
        ],
    }
}


def main():
    # ── 1. KITTI ──
    kitti_seqs = SEQUENCES['kitti']['train_val']
    fig, axes = plt.subplots(1, len(kitti_seqs), figsize=(4 * len(kitti_seqs), 4))
    if len(kitti_seqs) == 1:
        axes = [axes]
    fig.suptitle('KITTI Ground Truth Trajectories', fontsize=13, fontweight='bold')
    for ax, seq in zip(axes, kitti_seqs):
        poses = load_kitti_poses(seq)
        plot_trajectory(ax, poses, f"KITTI {seq}", axes_idx=(0, 2))  # X vs Z (Y is height)
    axes[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / "kitti_trajectories.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

    # ── 2. NCLT ──
    nclt_seqs = SEQUENCES['nclt']['train_val']
    fig, axes = plt.subplots(1, len(nclt_seqs), figsize=(5 * len(nclt_seqs), 5))
    if len(nclt_seqs) == 1:
        axes = [axes]
    fig.suptitle('NCLT Ground Truth Trajectories', fontsize=13, fontweight='bold')
    for ax, date in zip(axes, nclt_seqs):
        poses = load_nclt_poses(date)
        plot_trajectory(ax, poses, f"NCLT {date}")
    axes[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / "nclt_trajectories.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

    # ── 3. HeLiPR (grid layout) ──
    helipr_seqs = SEQUENCES['helipr']['train_val']
    n_helipr = len(helipr_seqs)
    ncols = 5
    nrows = (n_helipr + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = axes.flatten()
    fig.suptitle('HeLiPR Ground Truth Trajectories', fontsize=14, fontweight='bold')

    for i, seq in enumerate(helipr_seqs):
        gt_file = HELIPR_ROOT / seq / seq / "LiDAR_GT" / "Velodyne_gt.txt"
        if not gt_file.exists():
            axes_flat[i].set_title(f"{seq} (NOT FOUND)", fontsize=9, color='red')
            axes_flat[i].axis('off')
            continue
        poses = load_helipr_poses(seq)
        plot_trajectory(axes_flat[i], poses, seq)

    # Hide unused subplots
    for i in range(n_helipr, len(axes_flat)):
        axes_flat[i].axis('off')

    axes_flat[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUTPUT_DIR / "helipr_trajectories.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

    print(f"\nAll trajectories saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
