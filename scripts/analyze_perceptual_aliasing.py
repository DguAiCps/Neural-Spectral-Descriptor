"""
Perceptual Aliasing Analysis for Spectral Descriptors.

Investigates cases where structurally different projected images produce
nearly identical spectral descriptors (perceptual aliasing).

Supports both range image and BEV polar grid projection modes.

Identifies false positive pairs (high cosine similarity, large GT distance)
and analyzes:
  1. Which rows / frequency bins cause aliasing
  2. Projected image structural differences for aliased pairs
  3. Per-row similarity breakdown
  4. Energy distribution across frequency bins

Usage (run inside container):
    python scripts/analyze_perceptual_aliasing.py --dataset all
    python scripts/analyze_perceptual_aliasing.py --dataset nclt --top-k 15
    python scripts/analyze_perceptual_aliasing.py --projection bev --dataset kitti_00
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import faiss
from pathlib import Path
from scipy.spatial.distance import cdist
import csv
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from encoding.spectral_encoder import SpectralEncoder
from encoding.range_image import RangeImageProjector, interpolate_range_image
from encoding.bev_image import BEVProjector, interpolate_bev_image

OUTPUT_DIR = Path("/workspace/Neural-Spectral-Codec/outputs")

DISTANCE_THRESHOLD = 3.0
TEMPORAL_NEIGHBORS = 15
COSINE_THRESHOLD = 0.993  # Current config threshold
FALSE_POSITIVE_GT_MIN = 20.0  # GT distance to consider "clearly different place"
PROJECTION_TYPE = 'bev'  # 'range_image' or 'bev'
N_BINS = 4  # Number of frequency bins (4 for BEV, 16 for range_image)

SENSOR_CONFIGS = {
    'kitti': {
        'elevation_range': (-24.8, 2.0),
        'n_elevation': 64,
    },
    'nclt': {
        'elevation_range': (-30.67, 10.67),
        'n_elevation': 32,
    },
    'helipr': {
        'elevation_range': (-15.0, 15.0),
        'n_elevation': 16,
    },
}


# ── Tee output helper ─────────────────────────────────────
class TeeOutput:
    """Write to both stdout and a file."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# ── Point cloud loaders ────────────────────────────────────
def load_nclt_bin(filepath):
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
    dt = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('intensity', np.float32), ('ring', np.uint16), ('time', np.float32)
    ])
    data = np.fromfile(filepath, dtype=dt)
    pts = np.stack([data['x'], data['y'], data['z'], data['intensity']], axis=-1)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts[:, :3]) < 200).all(axis=1)
    return pts[valid]


def load_kitti_bin(filepath):
    pts = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts[:, :3]) < 200).all(axis=1)
    return pts[valid]


# ── Pose loaders ───────────────────────────────────────────
def load_kitti_poses(poses_file):
    """Load KITTI poses.txt → (N, 4, 4) SE(3) matrices → extract (N, 3) xyz."""
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            values = np.array([float(x) for x in line.strip().split()])
            if len(values) != 12:
                continue
            pose_3x4 = values.reshape(3, 4)
            pose = np.eye(4)
            pose[:3, :] = pose_3x4
            poses.append(pose)
    poses = np.array(poses)
    # Return (N, 4) format: [index, x, y, z] for consistency
    # But KITTI doesn't have timestamps in poses → use frame index
    xyz = poses[:, :3, 3]
    return poses, xyz


def load_nclt_poses(gt_file):
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
    idx = np.searchsorted(gt_ts, scan_ts)
    idx = np.clip(idx, 1, len(gt_ts) - 1)
    left = np.abs(gt_ts[idx - 1] - scan_ts)
    right = np.abs(gt_ts[idx] - scan_ts)
    return np.where(left < right, idx - 1, idx)


def select_keyframes_by_distance(positions, dist_thresh):
    kf_idx = [0]
    last = positions[0]
    for i in range(1, len(positions)):
        if np.linalg.norm(positions[i] - last) >= dist_thresh:
            kf_idx.append(i)
            last = positions[i]
    return np.array(kf_idx)


# ── Encoding with intermediates ────────────────────────────
def encode_with_intermediates(projector, points, target_rows=16, n_bins=16, alpha=2.0,
                              projection_type='range_image'):
    """
    Encode point cloud and return all intermediate representations.

    Returns:
        projected_image: (n_rows, 360) after pooling + interpolation
        fft_magnitudes: (n_rows, 181) FFT magnitude spectrum
        histogram_2d: (n_rows, n_bins) binned histogram (before flatten/normalize)
        descriptor: (n_rows * n_bins,) final normalized descriptor
    """
    n_azimuth = 360
    n_freqs = n_azimuth // 2 + 1  # 181
    epsilon = 1e-8

    # 1. Project to 2D image
    image_raw, _ = projector.project(points, keep_intensity=False)

    # 2. Row binning (range_image: pool to target_rows; BEV: use full resolution)
    if projection_type == 'bev':
        projected_image = image_raw.copy()
        target_rows = image_raw.shape[0]  # Use actual n_rings
    elif image_raw.shape[0] != target_rows:
        ri_tensor = torch.from_numpy(image_raw).float().unsqueeze(0).unsqueeze(0)
        ri_tensor = torch.nn.functional.adaptive_avg_pool2d(
            ri_tensor, (target_rows, n_azimuth)
        ).squeeze().numpy()
        projected_image = ri_tensor
    else:
        projected_image = image_raw.copy()

    # 3. Interpolate empty pixels
    if projection_type == 'bev':
        projected_image = interpolate_bev_image(projected_image, method='linear')
    else:
        projected_image = interpolate_range_image(projected_image, method='linear')

    # 4. FFT along azimuth
    fft_output = np.fft.rfft(projected_image, axis=1, norm='ortho')
    fft_magnitudes = np.abs(fft_output) * np.sqrt(n_azimuth)

    # 5. Exponential frequency binning
    t = np.linspace(0, 1, n_bins + 1)
    bin_edges = (np.exp(alpha * t) - 1) / (np.exp(alpha) - 1 + epsilon) * n_freqs

    histogram_2d = np.zeros((target_rows, n_bins), dtype=np.float32)
    freq_indices = np.arange(n_freqs)

    for b in range(n_bins):
        mask = (freq_indices >= bin_edges[b]) & (freq_indices < bin_edges[b + 1])
        if mask.any():
            histogram_2d[:, b] = fft_magnitudes[:, mask].sum(axis=1)

    # 6. Global normalization
    descriptor = histogram_2d.flatten()
    desc_sum = descriptor.sum()
    if desc_sum > epsilon:
        descriptor = descriptor / (desc_sum + epsilon)
    else:
        descriptor = np.ones_like(descriptor) / descriptor.size

    return projected_image, fft_magnitudes, histogram_2d, descriptor


# ── Dataset preparation with intermediates ─────────────────
def prepare_dataset_full(name):
    """Load poses, compute descriptors WITH intermediates."""
    print(f"\n{'='*60}")
    print(f"Preparing {name} (with intermediates)...")

    if name.startswith('kitti_'):
        seq = name.split('_')[1]  # e.g. '00', '05', '08'
        kitti_root = Path("/workspace/data/kitti/dataset")
        seq_dir = kitti_root / "sequences" / seq
        scan_dir = seq_dir / "velodyne"
        poses_file = seq_dir / "poses.txt"
        if not poses_file.exists():
            poses_file = kitti_root / "poses" / f"{seq}.txt"
        cfg = SENSOR_CONFIGS['kitti']
        load_bin = load_kitti_bin
        poses_4x4, positions_all = load_kitti_poses(poses_file)

        scan_files = sorted(scan_dir.glob("*.bin"))
        # KITTI: 1:1 correspondence between scans and poses
        n_scans = min(len(scan_files), len(positions_all))
        scan_files = scan_files[:n_scans]
        positions = positions_all[:n_scans]

        print(f"  Scans: {len(scan_files):,}, Poses: {len(positions_all):,}")

        # Select keyframes
        kf_scan_idx = select_keyframes_by_distance(positions, DISTANCE_THRESHOLD)
        n_kf = len(kf_scan_idx)
        kf_positions_3d = positions[kf_scan_idx]
        print(f"  Keyframes: {n_kf} (delta_d={DISTANCE_THRESHOLD}m)")

    elif name == 'nclt':
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

        print(f"  Scans: {len(scan_files):,}, GT: {len(gt_ts):,}")

        # Select keyframes
        kf_scan_idx = select_keyframes_by_distance(positions, DISTANCE_THRESHOLD)
        n_kf = len(kf_scan_idx)
        kf_positions_3d = positions[kf_scan_idx]
        print(f"  Keyframes: {n_kf} (delta_d={DISTANCE_THRESHOLD}m)")

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

        print(f"  Scans: {len(scan_files):,}, GT: {len(gt_ts):,}")

        # Select keyframes
        kf_scan_idx = select_keyframes_by_distance(positions, DISTANCE_THRESHOLD)
        n_kf = len(kf_scan_idx)
        kf_positions_3d = positions[kf_scan_idx]
        print(f"  Keyframes: {n_kf} (delta_d={DISTANCE_THRESHOLD}m)")

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Projector: BEV or range image
    if PROJECTION_TYPE == 'bev':
        projector = BEVProjector(
            n_sectors=360,
            max_range=80.0,
            min_range=1.0,
            z_min=-3.0
        )
        target_rows = projector.n_rings
    else:
        projector = RangeImageProjector(
            n_elevation=cfg['n_elevation'],
            n_azimuth=360,
            elevation_range=cfg['elevation_range']
        )
        target_rows = 16

    # Compute descriptors with intermediates
    all_projected_images = []
    all_fft_mags = []
    all_histograms_2d = []
    all_descriptors = []

    print(f"  Computing descriptors with intermediates ({PROJECTION_TYPE})...", flush=True)

    for i, si in enumerate(kf_scan_idx):
        pts = load_bin(scan_files[si])
        img, fft_mag, hist_2d, desc = encode_with_intermediates(
            projector, pts, target_rows=target_rows,
            n_bins=N_BINS, projection_type=PROJECTION_TYPE
        )
        all_projected_images.append(img)
        all_fft_mags.append(fft_mag)
        all_histograms_2d.append(hist_2d)
        all_descriptors.append(desc)

        if (i + 1) % 200 == 0 or i == n_kf - 1:
            print(f"    [{i+1}/{n_kf}]", flush=True)

    return {
        'name': name,
        'kf_positions_3d': kf_positions_3d,
        'descriptors': np.array(all_descriptors),
        'range_images': np.array(all_projected_images),
        'fft_magnitudes': np.array(all_fft_mags),
        'histograms_2d': np.array(all_histograms_2d),
        'n_keyframes': n_kf,
        'scan_files': [scan_files[si] for si in kf_scan_idx],
        'load_bin': load_bin,
        'projector': projector,
    }


# ── Find false positive pairs ─────────────────────────────
def find_false_positive_pairs(descriptors, gt_dist_matrix, n_kf,
                               cos_threshold=COSINE_THRESHOLD,
                               gt_min_dist=FALSE_POSITIVE_GT_MIN,
                               temporal_neighbors=TEMPORAL_NEIGHBORS,
                               fetch_k=200):
    """Find pairs with high cosine similarity but large GT distance."""
    hw = temporal_neighbors // 2

    # FAISS search
    desc_f32 = descriptors.astype(np.float32).copy()
    norms = np.linalg.norm(desc_f32, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    desc_norm = desc_f32 / norms

    fetch_k = min(fetch_k, n_kf)
    index = faiss.IndexFlatIP(desc_norm.shape[1])
    index.add(desc_norm)
    sims, indices = index.search(desc_norm, fetch_k)

    false_positives = []  # (i, j, cos_sim, gt_dist)
    true_positives = []
    seen = set()

    for i in range(n_kf):
        for jp in range(fetch_k):
            j = int(indices[i, jp])
            if j == i:
                continue
            cos_sim = float(sims[i, jp])
            if cos_sim < cos_threshold:
                break

            # Skip temporal neighbors
            if abs(i - j) <= hw:
                continue

            pair = (min(i, j), max(i, j))
            if pair in seen:
                continue
            seen.add(pair)

            gt_dist = gt_dist_matrix[i, j]

            if gt_dist >= gt_min_dist:
                false_positives.append((i, j, cos_sim, gt_dist))
            elif gt_dist < 5.0:
                true_positives.append((i, j, cos_sim, gt_dist))

    # Sort by cosine similarity descending (most confident false positives first)
    false_positives.sort(key=lambda x: x[2], reverse=True)
    true_positives.sort(key=lambda x: x[2], reverse=True)

    return false_positives, true_positives


# ── Per-elevation aliasing analysis ────────────────────────
def analyze_per_elevation_aliasing(data, false_positives, true_positives):
    """Analyze which elevation rows cause the most aliasing."""
    histograms = data['histograms_2d']  # (n_kf, 16, 16)
    n_elev = histograms.shape[1]

    # Per-elevation cosine similarity for FP vs TP pairs
    fp_per_elev_sim = np.zeros((len(false_positives), n_elev))
    tp_per_elev_sim = np.zeros((len(true_positives), n_elev))

    for idx, (i, j, _, _) in enumerate(false_positives):
        for e in range(n_elev):
            h_i = histograms[i, e, :]
            h_j = histograms[j, e, :]
            norm_i = np.linalg.norm(h_i)
            norm_j = np.linalg.norm(h_j)
            if norm_i > 1e-8 and norm_j > 1e-8:
                fp_per_elev_sim[idx, e] = np.dot(h_i, h_j) / (norm_i * norm_j)
            else:
                fp_per_elev_sim[idx, e] = 1.0  # Both zero → "same"

    for idx, (i, j, _, _) in enumerate(true_positives):
        for e in range(n_elev):
            h_i = histograms[i, e, :]
            h_j = histograms[j, e, :]
            norm_i = np.linalg.norm(h_i)
            norm_j = np.linalg.norm(h_j)
            if norm_i > 1e-8 and norm_j > 1e-8:
                tp_per_elev_sim[idx, e] = np.dot(h_i, h_j) / (norm_i * norm_j)
            else:
                tp_per_elev_sim[idx, e] = 1.0

    return fp_per_elev_sim, tp_per_elev_sim


def analyze_energy_distribution(data):
    """Analyze how energy is distributed across elevation rows and frequency bins."""
    histograms = data['histograms_2d']  # (n_kf, 16, 16)

    # Mean energy per elevation row (averaged across all keyframes)
    mean_energy_per_elev = histograms.mean(axis=0).sum(axis=1)  # (16,)

    # Mean energy per frequency bin (averaged across all keyframes)
    mean_energy_per_bin = histograms.mean(axis=0).sum(axis=0)  # (16,)

    # Variance of energy per elevation row across keyframes
    energy_per_elev = histograms.sum(axis=2)  # (n_kf, 16)
    var_energy_per_elev = energy_per_elev.var(axis=0)  # (16,)

    # Variance of energy per bin across keyframes
    energy_per_bin = histograms.sum(axis=1)  # (n_kf, 16)
    var_energy_per_bin = energy_per_bin.var(axis=0)  # (16,)

    return {
        'mean_energy_per_elev': mean_energy_per_elev,
        'mean_energy_per_bin': mean_energy_per_bin,
        'var_energy_per_elev': var_energy_per_elev,
        'var_energy_per_bin': var_energy_per_bin,
    }


# ── Range image structural difference ─────────────────────
def compute_range_image_diff(ri_a, ri_b):
    """Compute structural difference metrics between two range images."""
    # Normalize both to [0, 1] by max range (80m)
    max_range = 80.0
    a = ri_a / max_range
    b = ri_b / max_range

    # L2 distance (pixel-level)
    l2 = np.sqrt(((a - b) ** 2).mean())

    # Occupancy difference
    occ_a = (ri_a > 0).sum() / ri_a.size
    occ_b = (ri_b > 0).sum() / ri_b.size
    occ_diff = abs(occ_a - occ_b)

    # Per-row L2 (captures which elevation rows differ most)
    row_l2 = np.sqrt(((a - b) ** 2).mean(axis=1))

    return {
        'l2': l2,
        'occ_diff': occ_diff,
        'row_l2': row_l2,
    }


# ── Visualization: Individual pair ─────────────────────────
def plot_pair(data, i, j, cos_sim, gt_dist, rank, output_path):
    """Visualize a single false positive pair with all intermediate representations."""
    ri_a = data['range_images'][i]
    ri_b = data['range_images'][j]
    fft_a = data['fft_magnitudes'][i]
    fft_b = data['fft_magnitudes'][j]
    hist_a = data['histograms_2d'][i]
    hist_b = data['histograms_2d'][j]

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3,
                           width_ratios=[1, 1, 0.6])

    title = (f"False Positive #{rank}: cos_sim={cos_sim:.4f}, GT dist={gt_dist:.1f}m\n"
             f"KF {i} vs KF {j} ({data['name']})")
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Row 1: Range images
    ax1a = fig.add_subplot(gs[0, 0])
    im1a = ax1a.imshow(ri_a, aspect='auto', cmap='viridis', vmin=0, vmax=80)
    ax1a.set_title(f'Range Image (KF {i})', fontsize=11)
    ax1a.set_ylabel('Elevation')
    plt.colorbar(im1a, ax=ax1a, fraction=0.046, pad=0.04, label='Range (m)')

    ax1b = fig.add_subplot(gs[0, 1])
    im1b = ax1b.imshow(ri_b, aspect='auto', cmap='viridis', vmin=0, vmax=80)
    ax1b.set_title(f'Range Image (KF {j})', fontsize=11)
    plt.colorbar(im1b, ax=ax1b, fraction=0.046, pad=0.04, label='Range (m)')

    # Range image difference
    ax1c = fig.add_subplot(gs[0, 2])
    ri_diff = np.abs(ri_a - ri_b)
    im1c = ax1c.imshow(ri_diff, aspect='auto', cmap='hot', vmin=0,
                        vmax=max(ri_diff.max(), 1.0))
    ax1c.set_title('|Difference|', fontsize=11)
    plt.colorbar(im1c, ax=ax1c, fraction=0.046, pad=0.04, label='m')

    # Row 2: FFT magnitudes (log scale for visibility)
    fft_a_log = np.log1p(fft_a)
    fft_b_log = np.log1p(fft_b)
    vmax_fft = max(fft_a_log.max(), fft_b_log.max())

    ax2a = fig.add_subplot(gs[1, 0])
    im2a = ax2a.imshow(fft_a_log, aspect='auto', cmap='magma', vmin=0, vmax=vmax_fft)
    ax2a.set_title(f'FFT Magnitude (KF {i})', fontsize=11)
    ax2a.set_ylabel('Elevation')
    ax2a.set_xlabel('Frequency')
    plt.colorbar(im2a, ax=ax2a, fraction=0.046, pad=0.04, label='log(1+mag)')

    ax2b = fig.add_subplot(gs[1, 1])
    im2b = ax2b.imshow(fft_b_log, aspect='auto', cmap='magma', vmin=0, vmax=vmax_fft)
    ax2b.set_title(f'FFT Magnitude (KF {j})', fontsize=11)
    ax2b.set_xlabel('Frequency')
    plt.colorbar(im2b, ax=ax2b, fraction=0.046, pad=0.04, label='log(1+mag)')

    # FFT difference
    ax2c = fig.add_subplot(gs[1, 2])
    fft_diff = np.abs(fft_a_log - fft_b_log)
    im2c = ax2c.imshow(fft_diff, aspect='auto', cmap='hot', vmin=0,
                        vmax=max(fft_diff.max(), 0.1))
    ax2c.set_title('|FFT Diff|', fontsize=11)
    ax2c.set_xlabel('Frequency')
    plt.colorbar(im2c, ax=ax2c, fraction=0.046, pad=0.04)

    # Row 3: Histograms (16×16)
    vmax_hist = max(hist_a.max(), hist_b.max())
    if vmax_hist < 1e-8:
        vmax_hist = 1.0

    ax3a = fig.add_subplot(gs[2, 0])
    im3a = ax3a.imshow(hist_a, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax_hist)
    ax3a.set_title(f'Histogram 16x16 (KF {i})', fontsize=11)
    ax3a.set_ylabel('Elevation')
    ax3a.set_xlabel('Freq Bin')
    plt.colorbar(im3a, ax=ax3a, fraction=0.046, pad=0.04)

    ax3b = fig.add_subplot(gs[2, 1])
    im3b = ax3b.imshow(hist_b, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax_hist)
    ax3b.set_title(f'Histogram 16x16 (KF {j})', fontsize=11)
    ax3b.set_xlabel('Freq Bin')
    plt.colorbar(im3b, ax=ax3b, fraction=0.046, pad=0.04)

    # Histogram difference
    ax3c = fig.add_subplot(gs[2, 2])
    hist_diff = np.abs(hist_a - hist_b)
    im3c = ax3c.imshow(hist_diff, aspect='auto', cmap='hot', vmin=0,
                        vmax=max(hist_diff.max(), 1e-6))
    ax3c.set_title('|Hist Diff|', fontsize=11)
    ax3c.set_xlabel('Freq Bin')
    plt.colorbar(im3c, ax=ax3c, fraction=0.046, pad=0.04)

    # Row 4: Per-elevation similarity + descriptor comparison
    ax4a = fig.add_subplot(gs[3, 0])
    # Per-elevation cosine similarity
    per_elev_sim = []
    for e in range(hist_a.shape[0]):
        h_i = hist_a[e, :]
        h_j = hist_b[e, :]
        ni = np.linalg.norm(h_i)
        nj = np.linalg.norm(h_j)
        if ni > 1e-8 and nj > 1e-8:
            per_elev_sim.append(np.dot(h_i, h_j) / (ni * nj))
        else:
            per_elev_sim.append(1.0)

    colors = ['red' if s > 0.99 else 'orange' if s > 0.95 else 'green'
              for s in per_elev_sim]
    ax4a.barh(range(len(per_elev_sim)), per_elev_sim, color=colors, height=0.7)
    ax4a.set_xlim(0.8, 1.01)
    ax4a.set_xlabel('Cosine Similarity')
    ax4a.set_ylabel('Elevation Row')
    ax4a.set_title('Per-Elevation Similarity', fontsize=11)
    ax4a.axvline(0.993, color='gray', linestyle='--', alpha=0.5, label='0.993')
    ax4a.invert_yaxis()

    # Per-elevation energy comparison
    ax4b = fig.add_subplot(gs[3, 1])
    energy_a = hist_a.sum(axis=1)
    energy_b = hist_b.sum(axis=1)
    x_pos = np.arange(len(energy_a))
    width = 0.35
    ax4b.bar(x_pos - width/2, energy_a, width, label=f'KF {i}', color='steelblue', alpha=0.8)
    ax4b.bar(x_pos + width/2, energy_b, width, label=f'KF {j}', color='coral', alpha=0.8)
    ax4b.set_xlabel('Elevation Row')
    ax4b.set_ylabel('Energy (sum)')
    ax4b.set_title('Per-Elevation Energy', fontsize=11)
    ax4b.legend(fontsize=9)

    # Per-bin energy comparison
    ax4c = fig.add_subplot(gs[3, 2])
    bin_energy_a = hist_a.sum(axis=0)
    bin_energy_b = hist_b.sum(axis=0)
    x_pos = np.arange(len(bin_energy_a))
    ax4c.bar(x_pos - width/2, bin_energy_a, width, label=f'KF {i}', color='steelblue', alpha=0.8)
    ax4c.bar(x_pos + width/2, bin_energy_b, width, label=f'KF {j}', color='coral', alpha=0.8)
    ax4c.set_xlabel('Freq Bin')
    ax4c.set_ylabel('Energy')
    ax4c.set_title('Per-Bin Energy', fontsize=11)
    ax4c.legend(fontsize=9)

    fig.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ── Visualization: Summary ─────────────────────────────────
def plot_summary(data, false_positives, true_positives,
                 fp_per_elev_sim, tp_per_elev_sim,
                 energy_info, gt_dist_matrix, output_path):
    """Plot aggregate aliasing analysis."""
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    n_fp = len(false_positives)
    n_tp = len(true_positives)

    fig.suptitle(
        f"{data['name'].upper()} — Perceptual Aliasing Analysis\n"
        f"FP pairs (cos>={COSINE_THRESHOLD}, GT>={FALSE_POSITIVE_GT_MIN}m): {n_fp:,}  |  "
        f"TP pairs (cos>={COSINE_THRESHOLD}, GT<5m): {n_tp:,}",
        fontsize=14, fontweight='bold', y=0.99)

    # 1. GT distance distribution for high-sim pairs
    ax1 = fig.add_subplot(gs[0, 0])
    all_fp_gt = [fp[3] for fp in false_positives]
    all_tp_gt = [tp[3] for tp in true_positives]
    if all_fp_gt:
        ax1.hist(all_fp_gt, bins=50, alpha=0.7, color='red', label=f'FP ({n_fp})')
    if all_tp_gt:
        ax1.hist(all_tp_gt, bins=50, alpha=0.7, color='green', label=f'TP ({n_tp})')
    ax1.set_xlabel('GT Distance (m)')
    ax1.set_ylabel('Count')
    ax1.set_title('GT Distance Distribution\n(pairs with cos >= 0.993)')
    ax1.legend()
    ax1.axvline(5.0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(FALSE_POSITIVE_GT_MIN, color='gray', linestyle=':', alpha=0.5)

    # 2. Cosine similarity distribution for FP vs TP
    ax2 = fig.add_subplot(gs[0, 1])
    fp_sims = [fp[2] for fp in false_positives]
    tp_sims = [tp[2] for tp in true_positives]
    if fp_sims:
        ax2.hist(fp_sims, bins=50, alpha=0.7, color='red', label=f'FP')
    if tp_sims:
        ax2.hist(tp_sims, bins=50, alpha=0.7, color='green', label=f'TP')
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Count')
    ax2.set_title('Cosine Sim Distribution')
    ax2.legend()

    # 3. Range image L2 distance: FP vs TP
    ax3 = fig.add_subplot(gs[0, 2])
    ri_data = data['range_images']
    fp_ri_l2 = []
    tp_ri_l2 = []

    for (i, j, _, _) in false_positives[:500]:  # Cap at 500 for speed
        diff = compute_range_image_diff(ri_data[i], ri_data[j])
        fp_ri_l2.append(diff['l2'])
    for (i, j, _, _) in true_positives[:500]:
        diff = compute_range_image_diff(ri_data[i], ri_data[j])
        tp_ri_l2.append(diff['l2'])

    box_data = []
    box_labels = []
    if fp_ri_l2:
        box_data.append(fp_ri_l2)
        box_labels.append(f'FP (n={len(fp_ri_l2)})')
    if tp_ri_l2:
        box_data.append(tp_ri_l2)
        box_labels.append(f'TP (n={len(tp_ri_l2)})')
    if box_data:
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors_box = ['#ff6b6b', '#51cf66']
        for patch, color in zip(bp['boxes'], colors_box[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax3.set_ylabel('Range Image L2 Distance')
    ax3.set_title('Range Image Difference\n(FP vs TP)')

    # 4. Per-elevation cosine similarity: FP vs TP
    ax4 = fig.add_subplot(gs[1, 0])
    n_elev = fp_per_elev_sim.shape[1] if n_fp > 0 else 16
    x_pos = np.arange(n_elev)

    if n_fp > 0:
        fp_mean = fp_per_elev_sim.mean(axis=0)
        fp_std = fp_per_elev_sim.std(axis=0)
        ax4.fill_between(x_pos, fp_mean - fp_std, fp_mean + fp_std,
                         alpha=0.2, color='red')
        ax4.plot(x_pos, fp_mean, 'r-o', markersize=4, label='FP mean')

    if n_tp > 0:
        tp_mean = tp_per_elev_sim.mean(axis=0)
        tp_std = tp_per_elev_sim.std(axis=0)
        ax4.fill_between(x_pos, tp_mean - tp_std, tp_mean + tp_std,
                         alpha=0.2, color='green')
        ax4.plot(x_pos, tp_mean, 'g-s', markersize=4, label='TP mean')

    ax4.set_xlabel('Elevation Row')
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('Per-Elevation Cosine Similarity\n(FP vs TP)')
    ax4.legend()
    ax4.set_ylim(0.7, 1.02)
    ax4.axhline(0.993, color='gray', linestyle='--', alpha=0.3)

    # 5. Energy distribution per elevation
    ax5 = fig.add_subplot(gs[1, 1])
    me = energy_info['mean_energy_per_elev']
    ve = energy_info['var_energy_per_elev']
    ax5.bar(x_pos, me, color='steelblue', alpha=0.7, label='Mean energy')
    ax5_twin = ax5.twinx()
    # Coefficient of variation = std / mean
    cv = np.sqrt(ve) / (me + 1e-10)
    ax5_twin.plot(x_pos, cv, 'ro-', markersize=4, label='CV (std/mean)')
    ax5.set_xlabel('Elevation Row')
    ax5.set_ylabel('Mean Energy', color='steelblue')
    ax5_twin.set_ylabel('Coefficient of Variation', color='red')
    ax5.set_title('Energy per Elevation\n(Low CV = less discriminative)')
    ax5.legend(loc='upper left', fontsize=9)
    ax5_twin.legend(loc='upper right', fontsize=9)

    # 6. Energy distribution per frequency bin
    ax6 = fig.add_subplot(gs[1, 2])
    mb = energy_info['mean_energy_per_bin']
    vb = energy_info['var_energy_per_bin']
    ax6.bar(np.arange(len(mb)), mb, color='coral', alpha=0.7, label='Mean energy')
    ax6_twin = ax6.twinx()
    cv_b = np.sqrt(vb) / (mb + 1e-10)
    ax6_twin.plot(np.arange(len(mb)), cv_b, 'bo-', markersize=4, label='CV')
    ax6.set_xlabel('Frequency Bin')
    ax6.set_ylabel('Mean Energy', color='coral')
    ax6_twin.set_ylabel('Coefficient of Variation', color='blue')
    ax6.set_title('Energy per Frequency Bin\n(Low CV = less discriminative)')
    ax6.legend(loc='upper left', fontsize=9)
    ax6_twin.legend(loc='upper right', fontsize=9)

    # 7. Heatmap: mean per-elevation per-bin histogram (all keyframes)
    ax7 = fig.add_subplot(gs[2, 0])
    mean_hist = data['histograms_2d'].mean(axis=0)
    im7 = ax7.imshow(mean_hist, aspect='auto', cmap='YlOrRd')
    ax7.set_xlabel('Freq Bin')
    ax7.set_ylabel('Elevation Row')
    ax7.set_title('Mean Histogram (all KFs)')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    # 8. Heatmap: variance of per-elevation per-bin histogram
    ax8 = fig.add_subplot(gs[2, 1])
    var_hist = data['histograms_2d'].var(axis=0)
    im8 = ax8.imshow(var_hist, aspect='auto', cmap='YlOrRd')
    ax8.set_xlabel('Freq Bin')
    ax8.set_ylabel('Elevation Row')
    ax8.set_title('Histogram Variance (all KFs)\n(Low var = high aliasing risk)')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    # 9. Discriminative power: CV per cell
    ax9 = fig.add_subplot(gs[2, 2])
    cv_hist = np.sqrt(var_hist) / (mean_hist + 1e-10)
    im9 = ax9.imshow(cv_hist, aspect='auto', cmap='RdYlGn')  # Green = discriminative
    ax9.set_xlabel('Freq Bin')
    ax9.set_ylabel('Elevation Row')
    ax9.set_title('CV per Cell\n(Green = discriminative, Red = aliasing-prone)')
    plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

    fig.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Summary plot saved: {output_path}")


# ── Console statistics ─────────────────────────────────────
def print_statistics(data, false_positives, true_positives,
                     fp_per_elev_sim, energy_info):
    name = data['name'].upper()
    n_kf = data['n_keyframes']
    n_fp = len(false_positives)
    n_tp = len(true_positives)

    print(f"\n{'='*70}")
    print(f"  PERCEPTUAL ALIASING ANALYSIS — {name}")
    print(f"{'='*70}")
    print(f"  Keyframes:       {n_kf:,}")
    print(f"  Threshold:       cos >= {COSINE_THRESHOLD}")
    print(f"  False Positives: {n_fp:,} (GT >= {FALSE_POSITIVE_GT_MIN}m)")
    print(f"  True Positives:  {n_tp:,} (GT < 5m)")

    if n_fp + n_tp > 0:
        fp_ratio = n_fp / (n_fp + n_tp)
        print(f"  FP Ratio:        {fp_ratio:.1%} of all high-sim pairs")

    if n_fp > 0:
        print(f"\n  --- False Positive Stats ---")
        fp_gt_dists = [fp[3] for fp in false_positives]
        fp_cos_sims = [fp[2] for fp in false_positives]
        print(f"  GT distance: median={np.median(fp_gt_dists):.1f}m, "
              f"max={np.max(fp_gt_dists):.1f}m")
        print(f"  Cos similarity: median={np.median(fp_cos_sims):.4f}, "
              f"max={np.max(fp_cos_sims):.4f}")

        # Per-elevation analysis
        print(f"\n  --- Per-Elevation Aliasing (FP pairs) ---")
        print(f"  {'Elev':>5} | {'Mean Sim':>9} | {'Std':>6} | {'% > 0.99':>9} | Assessment")
        print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*6}-+-{'-'*9}-+----------")

        for e in range(fp_per_elev_sim.shape[1]):
            mean_s = fp_per_elev_sim[:, e].mean()
            std_s = fp_per_elev_sim[:, e].std()
            pct_high = (fp_per_elev_sim[:, e] > 0.99).mean() * 100

            if pct_high > 80:
                assess = "ALIASING"
            elif pct_high > 50:
                assess = "moderate"
            else:
                assess = "ok"

            print(f"  {e:>5} | {mean_s:>9.4f} | {std_s:>6.4f} | {pct_high:>8.1f}% | {assess}")

    # Energy distribution analysis
    print(f"\n  --- Energy Distribution ---")
    me = energy_info['mean_energy_per_elev']
    ve = energy_info['var_energy_per_elev']
    cv = np.sqrt(ve) / (me + 1e-10)

    total_energy = me.sum()
    top3_elev = np.argsort(me)[-3:][::-1]
    top3_pct = me[top3_elev].sum() / total_energy * 100

    print(f"  Top-3 energy elevations: {list(top3_elev)} "
          f"({top3_pct:.1f}% of total energy)")

    low_cv_elev = np.where(cv < np.median(cv))[0]
    print(f"  Low-CV elevations (aliasing-prone): {list(low_cv_elev)}")

    mb = energy_info['mean_energy_per_bin']
    bin0_pct = mb[0] / mb.sum() * 100
    print(f"  Bin 0 (DC/low-freq) energy: {bin0_pct:.1f}% of total")
    top2_bin_pct = mb[:2].sum() / mb.sum() * 100
    print(f"  Bin 0-1 energy: {top2_bin_pct:.1f}% of total")

    # Top false positive pairs
    if n_fp > 0:
        print(f"\n  --- Top-10 False Positive Pairs ---")
        print(f"  {'Rank':>4} | {'KF_i':>5} {'KF_j':>5} | {'Cos Sim':>8} | {'GT dist':>8} | {'RI L2':>6}")
        print(f"  {'-'*4}-+-{'-'*11}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")

        ri_data = data['range_images']
        for rank, (i, j, cos_sim, gt_dist) in enumerate(false_positives[:10]):
            diff = compute_range_image_diff(ri_data[i], ri_data[j])
            print(f"  {rank+1:>4} | {i:>5} {j:>5} | {cos_sim:>8.4f} | {gt_dist:>7.1f}m | {diff['l2']:>6.4f}")


# ── Main ───────────────────────────────────────────────────
def main():
    ALL_DATASETS = ['kitti_00', 'kitti_05', 'kitti_08', 'nclt', 'town01']

    parser = argparse.ArgumentParser(
        description='Analyze perceptual aliasing in spectral descriptors')
    parser.add_argument('--dataset', default='all',
                        choices=ALL_DATASETS + ['all', 'both'],
                        help='Dataset to analyze (default: all)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top FP pairs to visualize')
    parser.add_argument('--fp-gt-min', type=float, default=20.0,
                        help='Minimum GT distance for false positive (m)')
    parser.add_argument('--cos-threshold', type=float, default=0.993,
                        help='Cosine similarity threshold for high-sim pairs')
    parser.add_argument('--projection', type=str, default='bev',
                        choices=['range_image', 'bev'],
                        help='Projection type (default: bev)')
    parser.add_argument('--n-bins', type=int, default=4,
                        help='Number of frequency bins (default: 4)')
    parser.add_argument('--save-report', type=str,
                        default=str(OUTPUT_DIR / 'perceptual_aliasing_report.txt'),
                        help='Path to save text report')
    args = parser.parse_args()

    global FALSE_POSITIVE_GT_MIN, COSINE_THRESHOLD, PROJECTION_TYPE, N_BINS
    FALSE_POSITIVE_GT_MIN = args.fp_gt_min
    COSINE_THRESHOLD = args.cos_threshold
    PROJECTION_TYPE = args.projection
    N_BINS = args.n_bins

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set up tee output
    tee = TeeOutput(args.save_report)
    sys.stdout = tee

    names = []
    if args.dataset == 'all':
        names = ALL_DATASETS
    elif args.dataset == 'both':
        names = ['nclt', 'town01']
    else:
        names = [args.dataset]

    for name in names:
        data = prepare_dataset_full(name)
        n_kf = data['n_keyframes']

        # GT pairwise distances
        print(f"  Computing GT pairwise distances ({n_kf}x{n_kf})...")
        gt_dist_matrix = cdist(data['kf_positions_3d'], data['kf_positions_3d'],
                               metric='euclidean')

        # Find FP and TP pairs
        print(f"  Finding false positive pairs...")
        false_positives, true_positives = find_false_positive_pairs(
            data['descriptors'], gt_dist_matrix, n_kf,
            cos_threshold=COSINE_THRESHOLD,
            gt_min_dist=FALSE_POSITIVE_GT_MIN)

        print(f"  False positives: {len(false_positives):,}")
        print(f"  True positives: {len(true_positives):,}")

        # Per-elevation aliasing analysis
        print(f"  Analyzing per-elevation aliasing...")
        fp_per_elev_sim, tp_per_elev_sim = analyze_per_elevation_aliasing(
            data, false_positives, true_positives)

        # Energy distribution analysis
        energy_info = analyze_energy_distribution(data)

        # Print statistics
        print_statistics(data, false_positives, true_positives,
                         fp_per_elev_sim, energy_info)

        # Plot summary
        out_summary = OUTPUT_DIR / f"perceptual_aliasing_{name}_summary.png"
        plot_summary(data, false_positives, true_positives,
                     fp_per_elev_sim, tp_per_elev_sim,
                     energy_info, gt_dist_matrix, out_summary)

        # Plot top-K individual FP pairs (sorted by cosine sim)
        n_plot = min(args.top_k, len(false_positives))
        if n_plot > 0:
            print(f"\n  Generating {n_plot} pair visualizations (by cos_sim)...")
            for rank, (i, j, cos_sim, gt_dist) in enumerate(false_positives[:n_plot]):
                out_pair = OUTPUT_DIR / f"perceptual_aliasing_{name}_pair_{rank+1:02d}.png"
                plot_pair(data, i, j, cos_sim, gt_dist, rank + 1, out_pair)
                print(f"    Pair #{rank+1}: KF {i} vs {j}, "
                      f"cos={cos_sim:.4f}, GT={gt_dist:.1f}m → {out_pair.name}")

        # Plot top-K FP pairs sorted by RANGE IMAGE L2 (most different RI but similar descriptor)
        if len(false_positives) > 0:
            ri_data = data['range_images']
            fp_with_ri_l2 = []
            for (i, j, cos_sim, gt_dist) in false_positives:
                diff = compute_range_image_diff(ri_data[i], ri_data[j])
                fp_with_ri_l2.append((i, j, cos_sim, gt_dist, diff['l2']))
            fp_with_ri_l2.sort(key=lambda x: x[4], reverse=True)  # RI L2 descending

            # RI L2 distribution statistics
            all_ri_l2 = np.array([x[4] for x in fp_with_ri_l2])
            print(f"\n  --- RI L2 Distribution (all {len(all_ri_l2)} FP pairs) ---")
            print(f"  min={all_ri_l2.min():.4f}  median={np.median(all_ri_l2):.4f}  "
                  f"mean={all_ri_l2.mean():.4f}  max={all_ri_l2.max():.4f}  std={all_ri_l2.std():.4f}")
            thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
            print(f"  {'Threshold':>10} | {'Count':>6} | {'% of FP':>7} | Description")
            print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*7}-+-{'-'*30}")
            for th in thresholds:
                cnt = (all_ri_l2 >= th).sum()
                pct = cnt / len(all_ri_l2) * 100
                desc = ""
                if th == 0.10:
                    desc = "<-- moderate RI difference"
                elif th == 0.15:
                    desc = "<-- clear RI difference"
                elif th == 0.20:
                    desc = "<-- very clear RI difference"
                print(f"  RI L2>={th:.2f} | {cnt:>6} | {pct:>6.1f}% | {desc}")

            n_plot_ri = min(args.top_k, len(fp_with_ri_l2))
            print(f"\n  --- Top-{n_plot_ri} FP by Range Image L2 (most different RI) ---")
            print(f"  {'Rank':>4} | {'KF_i':>5} {'KF_j':>5} | {'Cos Sim':>8} | {'GT dist':>8} | {'RI L2':>6}")
            print(f"  {'-'*4}-+-{'-'*11}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
            for rank, (i, j, cos_sim, gt_dist, ri_l2) in enumerate(fp_with_ri_l2[:n_plot_ri]):
                print(f"  {rank+1:>4} | {i:>5} {j:>5} | {cos_sim:>8.4f} | {gt_dist:>7.1f}m | {ri_l2:>6.4f}")

            print(f"\n  Generating {n_plot_ri} pair visualizations (by RI L2)...")
            for rank, (i, j, cos_sim, gt_dist, ri_l2) in enumerate(fp_with_ri_l2[:n_plot_ri]):
                out_pair = OUTPUT_DIR / f"perceptual_aliasing_{name}_ri_diff_{rank+1:02d}.png"
                plot_pair(data, i, j, cos_sim, gt_dist, rank + 1, out_pair)
                print(f"    RI-Diff #{rank+1}: KF {i} vs {j}, "
                      f"cos={cos_sim:.4f}, GT={gt_dist:.1f}m, RI_L2={ri_l2:.4f} → {out_pair.name}")

    print(f"\n{'='*70}")
    print(f"  Analysis complete. Outputs in {OUTPUT_DIR}")
    print(f"  Text report: {args.save_report}")
    print(f"{'='*70}")

    # Restore stdout and close tee
    sys.stdout = tee.stdout
    tee.close()
    print(f"Report saved to {args.save_report}")


if __name__ == "__main__":
    main()
