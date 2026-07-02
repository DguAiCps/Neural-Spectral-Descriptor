"""
Compact Descriptor Test: Bins 0-1 vs Full 544D

Tests raw R@1 using different subsets of frequency bins to verify
that high-frequency bins (2-8) contribute no useful information.

Descriptor layout per ring (34D):
  [0..8]   bin0_mean .. bin8_mean
  [9..17]  bin0_std  .. bin8_std
  [18..25] diff_mean(1-0) .. diff_mean(8-7)
  [26..33] diff_std(1-0)  .. diff_std(8-7)

Bin subsets tested:
  - DC only (bin 0):     32D  (16 rings × 2)
  - Bins 0-1:            96D  (16 rings × 6)
  - Bins 0-2:           160D  (16 rings × 10)
  - Full (bins 0-8):    544D  (16 rings × 34)

Usage:
    docker exec nsc python3 scripts/test_compact_descriptor.py
"""

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ── Config ────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("data/preprocessed")
N_RINGS = 16
PER_RING_DIM = 34
SKIP_FRAMES = 30
POS_DIST = 5.0

DATASETS = {
    "KITTI_00":            ("094303a1", "kitti_val_00",            "HDL-64E"),
    "KITTI_05":            ("094303a1", "kitti_val_05",            "HDL-64E"),
    "KITTI_08":            ("094303a1", "kitti_val_08",            "HDL-64E"),
    "NCLT_01-08":          ("76c42cd7", "nclt_val_2012-01-08",    "HDL-32E"),
    "NCLT_13-01":          ("76c42cd7", "nclt_val_2013-01-10",    "HDL-32E"),
    "HeLiPR_Town01":       ("9cde251e", "helipr_val_Town01",      "VLP-16"),
    "MulRan_DCC03":        ("c6f48230", "mulran_val_DCC03",       "OS1-64"),
    "MulRan_KAIST03":      ("c6f48230", "mulran_val_KAIST03",     "OS1-64"),
    "MulRan_Riverside03":  ("c6f48230", "mulran_val_Riverside03", "OS1-64"),
}

RAW_RECALL = {
    "KITTI_00": 0.9288, "KITTI_05": 0.8196, "KITTI_08": 0.3574,
    "NCLT_01-08": 0.3190, "NCLT_13-01": 0.1768,
    "HeLiPR_Town01": 0.1583,
    "MulRan_DCC03": 0.5947, "MulRan_KAIST03": 0.9715, "MulRan_Riverside03": 0.7139,
}


# ── Bin extraction ────────────────────────────────────────────────────────────

def extract_bins(descriptors, n_bins):
    """Extract first n_bins from 544D descriptor.

    Args:
        descriptors: (N, 544) full descriptors
        n_bins: number of bins to keep (1=DC only, 2=bins 0-1, etc.)

    Returns:
        compact: (N, dim) where dim = 16 * (2*n_bins + 2*max(0, n_bins-1))
    """
    N = descriptors.shape[0]
    desc_rings = descriptors.reshape(N, N_RINGS, PER_RING_DIM)

    parts = []
    for r in range(N_RINGS):
        ring = desc_rings[:, r, :]  # (N, 34)

        # Intra-bin: means [0..n_bins-1], stds [9..9+n_bins-1]
        means = ring[:, :n_bins]           # (N, n_bins)
        stds = ring[:, 9:9+n_bins]         # (N, n_bins)
        parts.append(means)
        parts.append(stds)

        # Inter-bin diffs (only if n_bins > 1)
        if n_bins > 1:
            n_diffs = n_bins - 1
            diff_means = ring[:, 18:18+n_diffs]  # (N, n_diffs)
            diff_stds = ring[:, 26:26+n_diffs]   # (N, n_diffs)
            parts.append(diff_means)
            parts.append(diff_stds)

    return np.concatenate(parts, axis=1)


# ── Recall computation ────────────────────────────────────────────────────────

def compute_recall_at_k(descriptors, poses, k_values=(1, 5)):
    """Compute R@K using FAISS cosine similarity, matching trainer logic.

    Returns dict of {k: recall_value}.
    """
    N = len(descriptors)
    positions = poses[:, :3, 3].astype(np.float64)

    # Find revisit queries: frame j with any i within POS_DIST and gap > SKIP_FRAMES
    tree = cKDTree(positions)
    query_indices = []
    for j in range(SKIP_FRAMES, N):
        nearby = tree.query_ball_point(positions[j], POS_DIST)
        if any(i <= j - SKIP_FRAMES for i in nearby):
            query_indices.append(j)

    if not query_indices:
        return {k: 0.0 for k in k_values}, 0

    # L2-normalize for cosine similarity
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    desc_norm = (descriptors / norms).astype(np.float32)

    # Build FAISS index
    dim = desc_norm.shape[1]
    max_k = max(k_values)

    if HAS_FAISS:
        index = faiss.IndexFlatIP(dim)
        index.add(desc_norm)
    else:
        # Fallback: brute-force with numpy
        pass

    recalls = {k: 0 for k in k_values}
    n_queries = len(query_indices)

    for j in query_indices:
        # True positives: frames within POS_DIST and gap > SKIP_FRAMES
        nearby = tree.query_ball_point(positions[j], POS_DIST)
        true_pos = set(i for i in nearby if i <= j - SKIP_FRAMES)

        if not true_pos:
            n_queries -= 1
            continue

        if HAS_FAISS:
            # Exclude temporal neighbors (within SKIP_FRAMES)
            # Search more to allow filtering
            search_k = max_k + 2 * SKIP_FRAMES + 1
            D, I = index.search(desc_norm[j:j+1], min(search_k, N))
            I = I[0]
        else:
            sims = desc_norm @ desc_norm[j]
            I = np.argsort(-sims)

        # Filter out temporal neighbors
        retrieved = []
        for idx in I:
            if abs(idx - j) > SKIP_FRAMES:
                retrieved.append(int(idx))
                if len(retrieved) >= max_k:
                    break

        for k in k_values:
            top_k = set(retrieved[:k])
            if top_k & true_pos:
                recalls[k] += 1

    if n_queries > 0:
        recalls = {k: v / n_queries for k, v in recalls.items()}

    return recalls, n_queries


# ── HDL-32E elevation mapping analysis ────────────────────────────────────────

def analyze_elevation_mapping():
    """Print HDL-32E elevation mapping comparison."""
    print("\n" + "=" * 90)
    print("HDL-32E ELEVATION MAPPING ANALYSIS")
    print("=" * 90)

    sensors = {
        "HDL-64E (KITTI)":   {"beams": 64, "fov": (-24.8, 2.0)},
        "HDL-32E (NCLT)":    {"beams": 32, "fov": (-30.67, 10.67)},
        "OS1-64 (MulRan)":   {"beams": 64, "fov": (-16.6, 16.6)},
        "VLP-16 (HeLiPR)":   {"beams": 16, "fov": (-15.0, 15.0)},
    }
    target_bins = 16

    print(f"\n{'Sensor':22s} | {'Beams':>5s} | {'FoV':>12s} | {'Total°':>6s} | "
          f"{'°/beam':>7s} | {'Pool':>5s} | {'°/bin':>6s} | {'Horizon bin':>11s}")
    print("-" * 100)

    for name, info in sensors.items():
        beams = info["beams"]
        fov_min, fov_max = info["fov"]
        total_fov = fov_max - fov_min
        deg_per_beam = total_fov / beams
        pool_ratio = beams / target_bins
        deg_per_bin = total_fov / target_bins

        # Which bin contains the horizon (0°)?
        horizon_norm = (0.0 - fov_min) / total_fov
        horizon_bin = int(horizon_norm * target_bins)
        horizon_bin = min(horizon_bin, target_bins - 1)

        print(f"{name:22s} | {beams:5d} | [{fov_min:+6.1f},{fov_max:+5.1f}] | "
              f"{total_fov:6.1f} | {deg_per_beam:7.3f} | {pool_ratio:4.1f}x | "
              f"{deg_per_bin:6.2f} | bin {horizon_bin:2d} ({horizon_norm*100:.0f}%)")

    # HDL-32E specific analysis
    print(f"\n--- HDL-32E Detail ---")
    fov_min, fov_max = -30.67, 10.67
    total_fov = fov_max - fov_min
    print(f"FoV: [{fov_min}°, {fov_max}°] = {total_fov:.2f}°")
    print(f"Horizon (0°) at normalized position: {(0 - fov_min) / total_fov:.2f}")
    print(f"Below horizon: {abs(fov_min):.1f}° ({abs(fov_min)/total_fov*100:.0f}%)")
    print(f"Above horizon: {fov_max:.1f}° ({fov_max/total_fov*100:.0f}%)")
    print(f"→ 74% of bins look DOWN (ground), only 26% look UP (structures)")
    print(f"→ Ground-heavy bins are largely redundant for place recognition")
    print(f"→ Compare: KITTI has 92% below horizon — but 64 beams → 4x pooling averages it out")

    # Per-bin elevation ranges for each sensor
    print(f"\n--- Per-Bin Elevation Ranges (16 bins) ---")
    for name, info in sensors.items():
        fov_min, fov_max = info["fov"]
        total_fov = fov_max - fov_min
        bin_width = total_fov / target_bins
        print(f"\n{name}:")
        for b in range(target_bins):
            lo = fov_min + b * bin_width
            hi = lo + bin_width
            label = ""
            if lo <= 0 <= hi:
                label = " ← HORIZON"
            elif lo < -20:
                label = " (deep ground)"
            elif hi > 5:
                label = " (sky/structures)"
            print(f"  bin {b:2d}: [{lo:+6.2f}°, {hi:+6.2f}°]{label}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("COMPACT DESCRIPTOR TEST: Bin Subsets vs Full 544D")
    print("=" * 90)

    bin_configs = [
        (1, "DC only"),
        (2, "Bins 0-1"),
        (3, "Bins 0-2"),
        (9, "Full (0-8)"),
    ]

    # Header
    header_parts = [f"{'Dataset':22s}", f"{'Sensor':8s}", f"{'Queries':>7s}"]
    for n_bins, label in bin_configs:
        if n_bins == 1:
            dim = N_RINGS * 2
        else:
            dim = N_RINGS * (2 * n_bins + 2 * (n_bins - 1))
        header_parts.append(f"{label} ({dim}D)")
    header_parts.append("Ref R@1")
    print("\n" + " | ".join(header_parts))
    print("-" * 120)

    all_recalls = {label: {} for _, label in bin_configs}

    for ds_name, (cache_hash, cache_suffix, sensor) in DATASETS.items():
        cache_path = CACHE_DIR / f"cache_{cache_hash}_{cache_suffix}.npz"
        if not cache_path.exists():
            print(f"  [SKIP] {ds_name}: cache not found")
            continue

        data = np.load(cache_path, allow_pickle=True)
        descriptors = data["descriptors"]  # (N, 544)
        poses = data["poses"]              # (N, 4, 4)

        row_parts = [f"{ds_name:22s}", f"{sensor:8s}"]
        n_queries_val = None

        for n_bins, label in bin_configs:
            compact = extract_bins(descriptors, n_bins)
            recalls, n_queries = compute_recall_at_k(compact, poses, k_values=(1,))
            r1 = recalls[1]
            all_recalls[label][ds_name] = r1
            if n_queries_val is None:
                n_queries_val = n_queries
            row_parts.append(f"{r1:.4f}")

        row_parts.insert(2, f"{n_queries_val:7d}")
        row_parts.append(f"{RAW_RECALL.get(ds_name, 0):.4f}")
        print(" | ".join(row_parts))

    # Summary by sensor
    print("\n" + "=" * 90)
    print("AVERAGE R@1 BY SENSOR")
    print("=" * 90)

    sensor_map = {}
    for ds_name, (_, _, sensor) in DATASETS.items():
        sensor_map.setdefault(sensor, []).append(ds_name)

    header = f"{'Sensor':12s}"
    for _, label in bin_configs:
        header += f" | {label:>15s}"
    header += f" | {'Ref (full)':>12s}"
    print(header)
    print("-" * 90)

    for sensor, ds_list in sorted(sensor_map.items()):
        row = f"{sensor:12s}"
        for _, label in bin_configs:
            vals = [all_recalls[label].get(ds, 0) for ds in ds_list if ds in all_recalls[label]]
            avg = np.mean(vals) if vals else 0
            row += f" | {avg:>15.4f}"
        ref_vals = [RAW_RECALL.get(ds, 0) for ds in ds_list]
        row += f" | {np.mean(ref_vals):>12.4f}"
        print(row)

    # Overall average
    row = f"{'OVERALL':12s}"
    for _, label in bin_configs:
        vals = list(all_recalls[label].values())
        avg = np.mean(vals) if vals else 0
        row += f" | {avg:>15.4f}"
    ref_avg = np.mean(list(RAW_RECALL.values()))
    row += f" | {ref_avg:>12.4f}"
    print(row)

    # Diagnosis
    print("\n" + "=" * 90)
    print("DIAGNOSIS")
    print("=" * 90)
    full_vals = list(all_recalls["Full (0-8)"].values())
    bins01_vals = list(all_recalls["Bins 0-1"].values())
    dc_vals = list(all_recalls["DC only"].values())

    if full_vals and bins01_vals:
        full_avg = np.mean(full_vals)
        bins01_avg = np.mean(bins01_vals)
        dc_avg = np.mean(dc_vals)
        delta_01 = (bins01_avg - full_avg) / full_avg * 100

        print(f"DC only ({N_RINGS*2}D) avg R@1:  {dc_avg:.4f}")
        print(f"Bins 0-1 (96D) avg R@1:  {bins01_avg:.4f}")
        print(f"Full 544D avg R@1:       {full_avg:.4f}")
        print(f"Bins 0-1 vs Full:        {delta_01:+.1f}%")
        print()

        if abs(delta_01) < 5:
            print("→ Bins 2-8 contribute <5% to R@1. High-frequency bins are NOISE.")
            print("  Compact 96D descriptor is sufficient.")
        elif delta_01 < -10:
            print("→ Bins 2-8 contribute significantly. High-frequency info IS useful.")
        else:
            print(f"→ Moderate difference ({delta_01:+.1f}%). Some high-freq info helps marginally.")

    # Run HDL-32E analysis
    analyze_elevation_mapping()


if __name__ == "__main__":
    main()
