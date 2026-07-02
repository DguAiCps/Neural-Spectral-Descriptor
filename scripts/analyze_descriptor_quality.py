"""
Range Image Descriptor Quality Analysis

Diagnoses range image representation bottlenecks using cached descriptor data.
No training required — runs in ~10 minutes on cached .npz files.

Analyses:
  1. Positive/Negative L2 distance distributions + aliasing rate
  2. Per-ring (16 elevation bins) discriminability (Fisher ratio)
  3. FFT energy distribution per ring
  4. Spectral entropy vs raw R@1 correlation

Usage:
    python scripts/analyze_descriptor_quality.py [--cache-dir data/preprocessed]
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import pearsonr


# ── Dataset registry ──────────────────────────────────────────────────────────

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

# Known raw R@1 from ir_conv_k8 benchmark (2026-04-10)
RAW_RECALL = {
    "KITTI_00": 0.9288, "KITTI_05": 0.8196, "KITTI_08": 0.3574,
    "NCLT_01-08": 0.3190, "NCLT_13-01": 0.1768,
    "HeLiPR_Town01": 0.1583,
    "MulRan_DCC03": 0.5947, "MulRan_KAIST03": 0.9715, "MulRan_Riverside03": 0.7139,
}

SKIP_FRAMES = 30
POS_DIST = 5.0    # meters
NEG_DIST = 10.0   # meters
N_SAMPLE = 5000   # max pairs per category
N_RINGS = 16
PER_RING_DIM = 34  # 9 bins × 2 stats + 8 inter × 2 stats = 34


# ── Analysis functions ────────────────────────────────────────────────────────

def sample_pairs(poses, rng, n_sample=N_SAMPLE):
    """Extract positive and negative frame pairs from poses.

    Positive: GT distance < POS_DIST and frame gap > SKIP_FRAMES
    Negative: GT distance > NEG_DIST (random sample)
    """
    n = len(poses)
    positions = poses[:, :3, 3].astype(np.float64)
    tree = cKDTree(positions)

    # Positive pairs
    pos_pairs = []
    for j in range(SKIP_FRAMES, n):
        nearby = tree.query_ball_point(positions[j], POS_DIST)
        for i in nearby:
            if i <= j - SKIP_FRAMES:
                pos_pairs.append((j, i))
                break
    pos_pairs = np.array(pos_pairs) if pos_pairs else np.zeros((0, 2), dtype=int)

    # Subsample positives if too many
    if len(pos_pairs) > n_sample:
        idx = rng.choice(len(pos_pairs), n_sample, replace=False)
        pos_pairs = pos_pairs[idx]

    # Negative pairs: random frame pairs with GT distance > NEG_DIST
    neg_pairs = []
    attempts = 0
    max_attempts = n_sample * 20
    while len(neg_pairs) < n_sample and attempts < max_attempts:
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if abs(i - j) <= SKIP_FRAMES:
            attempts += 1
            continue
        d = np.linalg.norm(positions[i] - positions[j])
        if d > NEG_DIST:
            neg_pairs.append((i, j))
        attempts += 1
    neg_pairs = np.array(neg_pairs) if neg_pairs else np.zeros((0, 2), dtype=int)

    return pos_pairs, neg_pairs


def analyze_distance_distributions(descriptors, pos_pairs, neg_pairs):
    """Analysis 1: Positive/Negative L2 distance distributions + aliasing rate."""
    if len(pos_pairs) == 0 or len(neg_pairs) == 0:
        return {"pos_mean": 0, "neg_mean": 0, "aliasing_rate": 1.0, "n_pos": 0, "n_neg": 0}

    pos_dists = np.linalg.norm(
        descriptors[pos_pairs[:, 0]] - descriptors[pos_pairs[:, 1]], axis=1
    )
    neg_dists = np.linalg.norm(
        descriptors[neg_pairs[:, 0]] - descriptors[neg_pairs[:, 1]], axis=1
    )

    # Aliasing rate: fraction of negatives closer than median positive distance
    median_pos = np.median(pos_dists)
    aliasing_rate = np.mean(neg_dists < median_pos)

    # Separation: how much gap between distributions
    # d-prime: (μ_neg - μ_pos) / sqrt(0.5 * (σ²_pos + σ²_neg))
    var_sum = pos_dists.var() + neg_dists.var()
    d_prime = (neg_dists.mean() - pos_dists.mean()) / np.sqrt(0.5 * var_sum + 1e-10)

    return {
        "pos_mean": pos_dists.mean(),
        "pos_std": pos_dists.std(),
        "neg_mean": neg_dists.mean(),
        "neg_std": neg_dists.std(),
        "median_pos": median_pos,
        "aliasing_rate": aliasing_rate,
        "d_prime": d_prime,
        "n_pos": len(pos_pairs),
        "n_neg": len(neg_pairs),
    }


def analyze_per_ring_discriminability(descriptors, pos_pairs, neg_pairs):
    """Analysis 2: Fisher discriminant ratio per elevation ring."""
    if len(pos_pairs) == 0 or len(neg_pairs) == 0:
        return np.zeros(N_RINGS)

    # Reshape: (N, 544) → (N, 16, 34)
    n = descriptors.shape[0]
    desc_rings = descriptors.reshape(n, N_RINGS, PER_RING_DIM)

    fisher_scores = np.zeros(N_RINGS)
    for r in range(N_RINGS):
        ring_desc = desc_rings[:, r, :]  # (N, 34)

        # Positive pair distances per ring
        pos_d = np.linalg.norm(
            ring_desc[pos_pairs[:, 0]] - ring_desc[pos_pairs[:, 1]], axis=1
        )
        # Negative pair distances per ring
        neg_d = np.linalg.norm(
            ring_desc[neg_pairs[:, 0]] - ring_desc[neg_pairs[:, 1]], axis=1
        )

        # Fisher ratio
        var_sum = pos_d.var() + neg_d.var()
        if var_sum > 1e-10:
            fisher_scores[r] = (neg_d.mean() - pos_d.mean()) ** 2 / var_sum
        else:
            fisher_scores[r] = 0.0

    return fisher_scores


def analyze_fft_energy(fft_magnitudes):
    """Analysis 3: Per-ring frequency energy profile.

    Args:
        fft_magnitudes: (N, 16, 181) raw FFT magnitude spectra

    Returns:
        per_ring_energy: (16, 181) mean energy per frequency
        eff_bandwidth: (16,) frequency index where 90% cumulative energy is reached
    """
    # Energy = mean(|FFT|²) across all frames
    energy = np.mean(fft_magnitudes ** 2, axis=0)  # (16, 181)

    # Effective bandwidth per ring: freq where cumsum reaches 90%
    eff_bandwidth = np.zeros(N_RINGS, dtype=int)
    for r in range(N_RINGS):
        cumsum = np.cumsum(energy[r])
        total = cumsum[-1]
        if total > 0:
            eff_bandwidth[r] = np.searchsorted(cumsum, 0.9 * total) + 1
        else:
            eff_bandwidth[r] = 0

    # DC (f=0) energy fraction
    dc_fraction = energy[:, 0] / (energy.sum(axis=1) + 1e-10)

    # Low-freq (f<=4, octave bin 1) energy fraction
    low_freq_fraction = energy[:, :5].sum(axis=1) / (energy.sum(axis=1) + 1e-10)

    return {
        "energy": energy,
        "eff_bandwidth": eff_bandwidth,
        "dc_fraction_per_ring": dc_fraction,
        "low_freq_fraction_per_ring": low_freq_fraction,
        "mean_eff_bandwidth": eff_bandwidth.mean(),
        "dc_fraction_mean": dc_fraction.mean(),
        "low_freq_fraction_mean": low_freq_fraction.mean(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Range Image Descriptor Quality Analysis")
    parser.add_argument("--cache-dir", default="data/preprocessed", help="Cache directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    rng = np.random.default_rng(args.seed)

    print("=" * 100)
    print("RANGE IMAGE DESCRIPTOR QUALITY ANALYSIS")
    print("=" * 100)

    # ── Collect results ───────────────────────────────────────────────────
    all_results = {}

    for ds_name, (cache_hash, cache_suffix, sensor) in DATASETS.items():
        cache_path = cache_dir / f"cache_{cache_hash}_{cache_suffix}.npz"
        if not cache_path.exists():
            print(f"  [SKIP] {ds_name}: {cache_path} not found")
            continue

        data = np.load(cache_path, allow_pickle=True)
        descriptors = data["descriptors"]    # (N, 544)
        poses = data["poses"]                # (N, 4, 4)
        fft_mags = data["fft_magnitudes"]    # (N, 16, 181)
        entropies = data["spectral_entropies"]  # (N,)

        print(f"\n{'─' * 80}")
        print(f"  {ds_name} ({sensor}) — {len(descriptors)} keyframes, dim={descriptors.shape[1]}")
        print(f"{'─' * 80}")

        # Sample pairs
        pos_pairs, neg_pairs = sample_pairs(poses, rng)
        print(f"  Pairs: {len(pos_pairs)} positive, {len(neg_pairs)} negative")

        # Analysis 1: Distance distributions
        dist_result = analyze_distance_distributions(descriptors, pos_pairs, neg_pairs)
        print(f"  [Dist] pos_L2={dist_result['pos_mean']:.4f}±{dist_result['pos_std']:.4f}, "
              f"neg_L2={dist_result['neg_mean']:.4f}±{dist_result['neg_std']:.4f}")
        print(f"         aliasing_rate={dist_result['aliasing_rate']:.4f}, "
              f"d'={dist_result['d_prime']:.3f}")

        # Analysis 2: Per-ring discriminability
        fisher = analyze_per_ring_discriminability(descriptors, pos_pairs, neg_pairs)
        top_rings = np.argsort(fisher)[::-1][:5]
        bottom_rings = np.argsort(fisher)[:5]
        print(f"  [Ring] Fisher scores: {fisher.round(3).tolist()}")
        print(f"         Top-5 rings: {top_rings.tolist()} (F={fisher[top_rings].round(3).tolist()})")
        print(f"         Bot-5 rings: {bottom_rings.tolist()} (F={fisher[bottom_rings].round(3).tolist()})")

        # Analysis 3: FFT energy
        fft_result = analyze_fft_energy(fft_mags)
        print(f"  [FFT]  Mean eff_bandwidth={fft_result['mean_eff_bandwidth']:.1f}/181, "
              f"DC_frac={fft_result['dc_fraction_mean']:.3f}, "
              f"low_freq_frac={fft_result['low_freq_fraction_mean']:.3f}")
        print(f"         Eff_BW per ring: {fft_result['eff_bandwidth'].tolist()}")

        # Analysis 4: Entropy stats
        print(f"  [Entropy] mean={entropies.mean():.4f}, std={entropies.std():.4f}")

        all_results[ds_name] = {
            "sensor": sensor,
            "n_keyframes": len(descriptors),
            "raw_recall": RAW_RECALL.get(ds_name, 0),
            "dist": dist_result,
            "fisher": fisher,
            "fft": fft_result,
            "entropy_mean": entropies.mean(),
            "entropy_std": entropies.std(),
        }

    # ── Summary tables ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY: ALIASING RATE & SEPARATION BY SENSOR")
    print("=" * 100)
    print(f"{'Dataset':22s} | {'Sensor':8s} | {'raw R@1':>8s} | {'Alias%':>7s} | "
          f"{'d-prime':>8s} | {'pos_L2':>7s} | {'neg_L2':>7s} | {'Entropy':>8s}")
    print("-" * 100)

    sensor_groups = {}
    for ds_name, r in sorted(all_results.items()):
        sensor = r["sensor"]
        alias_pct = r["dist"]["aliasing_rate"] * 100
        print(f"{ds_name:22s} | {sensor:8s} | {r['raw_recall']:>8.4f} | {alias_pct:>6.2f}% | "
              f"{r['dist']['d_prime']:>8.3f} | {r['dist']['pos_mean']:>7.4f} | "
              f"{r['dist']['neg_mean']:>7.4f} | {r['entropy_mean']:>8.4f}")
        sensor_groups.setdefault(sensor, []).append(r)

    # Per-sensor averages
    print("-" * 100)
    for sensor, results in sorted(sensor_groups.items()):
        avg_alias = np.mean([r["dist"]["aliasing_rate"] for r in results]) * 100
        avg_dprime = np.mean([r["dist"]["d_prime"] for r in results])
        avg_raw = np.mean([r["raw_recall"] for r in results])
        avg_entropy = np.mean([r["entropy_mean"] for r in results])
        n = len(results)
        print(f"{'  AVG ' + sensor:22s} | {sensor:8s} | {avg_raw:>8.4f} | {avg_alias:>6.2f}% | "
              f"{avg_dprime:>8.3f} | {'':>7s} | {'':>7s} | {avg_entropy:>8.4f}  (n={n})")

    # ── Per-ring discriminability summary ─────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY: PER-RING FISHER DISCRIMINABILITY")
    print("=" * 100)
    print(f"{'Dataset':22s} | {'Sensor':8s} | " +
          " ".join(f"R{i:02d}" for i in range(N_RINGS)) + " | Useful")
    print("-" * (38 + N_RINGS * 5 + 10))
    for ds_name, r in sorted(all_results.items()):
        fisher = r["fisher"]
        useful = np.sum(fisher > 0.1)  # rings with Fisher > 0.1
        bar = " ".join(f"{f:4.2f}" for f in fisher)
        print(f"{ds_name:22s} | {r['sensor']:8s} | {bar} | {useful:2d}/16")

    # ── FFT bandwidth summary ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY: FFT EFFECTIVE BANDWIDTH (90% energy)")
    print("=" * 100)
    print(f"{'Dataset':22s} | {'Sensor':8s} | {'Mean BW':>8s} | {'DC%':>6s} | {'LowF%':>6s} | "
          f"BW per ring (0..15)")
    print("-" * 100)
    for ds_name, r in sorted(all_results.items()):
        bw = r["fft"]["eff_bandwidth"]
        print(f"{ds_name:22s} | {r['sensor']:8s} | {r['fft']['mean_eff_bandwidth']:>7.1f} | "
              f"{r['fft']['dc_fraction_mean']*100:>5.1f}% | "
              f"{r['fft']['low_freq_fraction_mean']*100:>5.1f}% | "
              f"{bw.tolist()}")

    # ── Entropy ↔ R@1 correlation ────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY: ENTROPY vs RAW R@1 CORRELATION")
    print("=" * 100)
    entropies_list = [r["entropy_mean"] for r in all_results.values()]
    recalls_list = [r["raw_recall"] for r in all_results.values()]
    if len(entropies_list) >= 3:
        r_corr, p_val = pearsonr(entropies_list, recalls_list)
        print(f"Pearson r = {r_corr:.4f}, p = {p_val:.4f}")
        for ds_name, res in sorted(all_results.items()):
            print(f"  {ds_name:22s}: entropy={res['entropy_mean']:.4f}, raw_R@1={res['raw_recall']:.4f}")
    else:
        print("Not enough datasets for correlation")

    # ── Diagnosis ─────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("DIAGNOSIS")
    print("=" * 100)

    # Check if aliasing varies by sensor
    sensor_alias = {}
    for r in all_results.values():
        sensor_alias.setdefault(r["sensor"], []).append(r["dist"]["aliasing_rate"])
    for sensor, rates in sorted(sensor_alias.items()):
        avg = np.mean(rates) * 100
        print(f"  {sensor:8s} avg aliasing: {avg:.1f}%")

    # Check per-ring useful count
    sensor_useful = {}
    for r in all_results.values():
        useful = np.sum(r["fisher"] > 0.1)
        sensor_useful.setdefault(r["sensor"], []).append(useful)
    print()
    for sensor, counts in sorted(sensor_useful.items()):
        avg = np.mean(counts)
        print(f"  {sensor:8s} avg useful rings: {avg:.1f}/16")

    # Check FFT bandwidth
    sensor_bw = {}
    for r in all_results.values():
        sensor_bw.setdefault(r["sensor"], []).append(r["fft"]["mean_eff_bandwidth"])
    print()
    for sensor, bws in sorted(sensor_bw.items()):
        avg = np.mean(bws)
        print(f"  {sensor:8s} avg eff_bandwidth: {avg:.1f}/181")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
