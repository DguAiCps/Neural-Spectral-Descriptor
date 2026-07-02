"""
Similarity Edge Impact Diagnosis.

Quantifies how cosine-Bayesian similarity edges affect per-dataset R@1 vs
the baseline (no-sim-edge) GNN. Joins:
  - Per-dataset R@1 from baseline log (training_20260417_070324.log, R@1=0.7309)
  - Per-dataset R@1 from cosine_bayesian log (training_20260429_074110.log, R@1=0.7244)
  - Per-dataset sim edge density built with current production config

Outputs scatter (sim_edge_density vs delta) + table + JSON.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import hashlib
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ── Hard-coded comparison from training logs ─────────────────────
# Baseline: training_20260417_070324.log, ep96 best
# Cosine Bayesian: training_20260429_074110.log, ep93 best
PER_DATASET = [
    # (name,                baseline_r1, cosbay_r1, raw_r1, queries)
    ("KITTI_00",            0.9035, 0.9462, 0.9288,  632),
    ("KITTI_05",            0.8249, 0.8302, 0.8196,  377),
    ("KITTI_08",            0.3277, 0.3702, 0.3574,  235),
    ("NCLT_2012-01-08",     0.4024, 0.4269, 0.3190, 1834),
    ("NCLT_2013-01-10",     0.3536, 0.2376, 0.1768,  181),
    ("HeLiPR_Town01",       0.5126, 0.4685, 0.1583, 1586),
    ("MulRan_DCC03",        0.7474, 0.7359, 0.5947, 2344),
    ("MulRan_KAIST03",      0.9948, 0.9914, 0.9715, 2909),
    ("MulRan_Riverside03",  0.7883, 0.7740, 0.7139, 2796),
]


def compute_cache_key(config):
    enc, kf = config['encoding'], config['keyframe']
    pt = enc.get('projection_type', 'range_image')
    p = {
        'projection_type': pt,
        'n_azimuth': enc['n_azimuth'],
        'n_bins': enc['n_bins'],
        'binning_strategy': enc.get('binning_strategy', 'exponential'),
        'bin_statistics': enc.get('bin_statistics', ['sum']),
        'inter_bin_statistics': enc.get('inter_bin_statistics', []),
        'max_range': enc.get('max_range', 80.0),
        'min_range': enc.get('min_range', 1.0),
        'zero_center': enc.get('zero_center', False),
        'log_magnitude': enc.get('log_magnitude', False),
        'normalize_channels': enc.get('normalize_channels', True),
        'distance_threshold': kf['distance_threshold'],
        'rotation_threshold': kf['rotation_threshold'],
        'overlap_threshold': kf['overlap_threshold'],
        'temporal_threshold': kf['temporal_threshold'],
    }
    if pt == 'bev':
        p['bev'] = enc.get('bev', {})
    else:
        p['n_elevation'] = enc['n_elevation']
        p['elevation_range'] = enc['elevation_range']
        p['sensor_elevation_ranges'] = enc.get('sensor_elevation_ranges', {})
        p['target_elevation_bins'] = enc['target_elevation_bins']
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:8]


def load_keyframes_cache(path):
    from keyframe.selector import Keyframe
    with np.load(path) as data:
        descs, poses, ts = data['descriptors'], data['poses'], data['timestamps']
        scan_ids, kf_ids = data['scan_ids'], data['keyframe_ids']
        kfs = []
        for i in range(len(scan_ids)):
            kfs.append(Keyframe(
                keyframe_id=int(kf_ids[i]), scan_id=int(scan_ids[i]),
                points=np.empty((0, 3)), pose=poses[i],
                timestamp=float(ts[i]), descriptor=descs[i],
            ))
        return kfs


def build_val_graph(keyframes, config, similarity_dist, std_stats):
    from keyframe.graph_manager import build_graph_from_keyframes_batch
    g_cfg = config['keyframe'].get('graph', {})
    poses = np.array([kf.pose for kf in keyframes])
    descs = np.array([kf.descriptor for kf in keyframes])
    return build_graph_from_keyframes_batch(
        keyframes,
        temporal_neighbors=config['keyframe']['temporal_neighbors'],
        device='cpu',
        poses=poses,
        descriptors=descs,
        similarity_threshold=g_cfg.get('similarity_threshold', 0.993),
        similarity_max_k=g_cfg.get('similarity_max_k', 10),
        similarity_exclude_temporal=g_cfg.get('similarity_exclude_temporal', True),
        similarity_dist=similarity_dist,
        similarity_metric=g_cfg.get('similarity_metric', 'cosine'),
        standardization_stats=std_stats,
        confidence_level=g_cfg.get('confidence_level', 0.95),
        base_prior=g_cfg.get('base_prior', 0.01),
        density_k=g_cfg.get('density_k', 50),
        density_beta=g_cfg.get('density_beta', 10.0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/training_multi_dataset.yaml')
    parser.add_argument('--checkpoint-dir', default='results/ctx128_cosine_bayesian')
    parser.add_argument('--output-dir', default='results/sim_edge_impact')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cache_key = compute_cache_key(config)
    cache_dir = config['data'].get('cache_dir', 'data/preprocessed')
    print(f"Cache key: {cache_key}")

    # Load similarity distribution + standardization (production config: cosine + bayesian)
    from utils.similarity_stats import SimilarityDistribution
    sim_metric = config['keyframe'].get('graph', {}).get('similarity_metric', 'cosine')
    sim_dist = None
    sim_dist_path = os.path.join(args.checkpoint_dir, 'similarity_dist.npz')
    if os.path.exists(sim_dist_path):
        sim_dist = SimilarityDistribution(metric=sim_metric).load(sim_dist_path)
        print(f"Loaded similarity distribution: {sim_dist_path}")

    std_stats = None
    if sim_metric == 'l2':
        from utils.standardization_stats import StandardizationStats
        std_path = os.path.join(args.checkpoint_dir, 'standardization_stats.npz')
        if os.path.exists(std_path):
            std_stats = StandardizationStats().load(std_path)

    # Load each val sequence + build graph + count sim edges
    name_to_cache = {
        'KITTI_00': 'kitti_val_00',
        'KITTI_05': 'kitti_val_05',
        'KITTI_08': 'kitti_val_08',
        'NCLT_2012-01-08': 'nclt_val_2012-01-08',
        'NCLT_2013-01-10': 'nclt_val_2013-01-10',
        'HeLiPR_Town01': 'helipr_val_Town01',
        'MulRan_DCC03': 'mulran_val_DCC03',
        'MulRan_KAIST03': 'mulran_val_KAIST03',
        'MulRan_Riverside03': 'mulran_val_Riverside03',
    }

    rows = []
    for (name, base_r1, cb_r1, raw_r1, q) in PER_DATASET:
        seq = name_to_cache[name]
        path = Path(cache_dir) / f"cache_{cache_key}_{seq}.npz"
        if not path.exists():
            print(f"  [SKIP] {name}: {path} not found")
            continue
        kfs = load_keyframes_cache(str(path))
        graph = build_val_graph(kfs, config, sim_dist, std_stats)
        n_temp = int((graph.edge_type == 0).sum())
        n_sim = int((graph.edge_type == 1).sum())
        n_nodes = graph.num_nodes
        sim_per_node = n_sim / max(n_nodes, 1)
        delta = cb_r1 - base_r1
        rows.append({
            'name': name,
            'baseline_r1': base_r1,
            'cosbay_r1': cb_r1,
            'raw_r1': raw_r1,
            'delta_r1': delta,
            'queries': q,
            'n_nodes': n_nodes,
            'n_sim_edges': n_sim,
            'n_temporal_edges': n_temp,
            'sim_per_node': sim_per_node,
            'cosbay_gain_vs_raw': cb_r1 - raw_r1,
            'baseline_gain_vs_raw': base_r1 - raw_r1,
        })
        print(f"  {name:22s} | base={base_r1:.4f} cb={cb_r1:.4f} Δ={delta:+.4f} | "
              f"sim/node={sim_per_node:5.2f} ({n_sim}/{n_nodes})")

    # Compute correlations
    if rows:
        sim_density = np.array([r['sim_per_node'] for r in rows])
        deltas = np.array([r['delta_r1'] for r in rows])
        raw_r1s = np.array([r['raw_r1'] for r in rows])
        queries = np.array([r['queries'] for r in rows])

        from scipy.stats import pearsonr, spearmanr
        pear_d, pear_p = pearsonr(sim_density, deltas)
        sp_d, sp_p = spearmanr(sim_density, deltas)
        pear_r, _ = pearsonr(raw_r1s, deltas)
        sp_r, _ = spearmanr(raw_r1s, deltas)

        # Net frame-weighted impact
        net_change = sum(r['delta_r1'] * r['queries'] for r in rows)
        total_q = sum(r['queries'] for r in rows)

        summary = {
            'n_datasets': len(rows),
            'total_queries': int(total_q),
            'avg_baseline_r1': float(sum(r['baseline_r1'] * r['queries'] for r in rows) / total_q),
            'avg_cosbay_r1': float(sum(r['cosbay_r1'] * r['queries'] for r in rows) / total_q),
            'net_frame_change': float(net_change),
            'pearson_sim_density_vs_delta': {'r': float(pear_d), 'p': float(pear_p)},
            'spearman_sim_density_vs_delta': {'r': float(sp_d), 'p': float(sp_p)},
            'pearson_raw_r1_vs_delta': {'r': float(pear_r)},
            'spearman_raw_r1_vs_delta': {'r': float(sp_r)},
            'datasets': rows,
        }

        out_json = Path(args.output_dir) / 'summary.json'
        with open(out_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {out_json}")

        # Stdout summary
        print("\n" + "=" * 90)
        print("PARADOX DECOMPOSITION (cosine Bayesian vs baseline)")
        print("=" * 90)
        print(f"{'Dataset':22s} {'base':>7s} {'cosbay':>7s} {'Δ':>8s} {'sim/n':>7s} {'Δ×Q':>8s}")
        print("-" * 90)
        rows_sorted = sorted(rows, key=lambda r: r['delta_r1'] * r['queries'])
        for r in rows_sorted:
            print(f"{r['name']:22s} {r['baseline_r1']:>7.4f} {r['cosbay_r1']:>7.4f} "
                  f"{r['delta_r1']:>+8.4f} {r['sim_per_node']:>7.2f} "
                  f"{r['delta_r1'] * r['queries']:>+8.1f}")
        print(f"{'TOTAL':22s} {summary['avg_baseline_r1']:>7.4f} {summary['avg_cosbay_r1']:>7.4f} "
              f"{summary['avg_cosbay_r1'] - summary['avg_baseline_r1']:>+8.4f} {'':>7s} "
              f"{net_change:>+8.1f}")
        print()
        print(f"Pearson  (sim_density vs Δ): r={pear_d:+.3f} p={pear_p:.3f}")
        print(f"Spearman (sim_density vs Δ): r={sp_d:+.3f} p={sp_p:.3f}")
        print(f"Pearson  (raw_R1     vs Δ): r={pear_r:+.3f}  (low raw → bigger sim-edge harm?)")
        print(f"Spearman (raw_R1     vs Δ): r={sp_r:+.3f}")

        # Scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for r in rows:
            color = 'tab:green' if r['delta_r1'] > 0 else 'tab:red'
            ax.scatter(r['sim_per_node'], r['delta_r1'], s=r['queries']/30, c=color, alpha=0.7)
            ax.annotate(r['name'].replace('MulRan_', 'M_').replace('NCLT_', 'N_').replace('HeLiPR_', 'H_'),
                        (r['sim_per_node'], r['delta_r1']), fontsize=8,
                        xytext=(5, 5), textcoords='offset points')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlabel('sim edges per node')
        ax.set_ylabel('Δ R@1 (cosine_bayesian − baseline)')
        ax.set_title(f'sim density vs Δ (Pearson r={pear_d:+.2f})')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for r in rows:
            color = 'tab:green' if r['delta_r1'] > 0 else 'tab:red'
            ax.scatter(r['raw_r1'], r['delta_r1'], s=r['queries']/30, c=color, alpha=0.7)
            ax.annotate(r['name'].replace('MulRan_', 'M_').replace('NCLT_', 'N_').replace('HeLiPR_', 'H_'),
                        (r['raw_r1'], r['delta_r1']), fontsize=8,
                        xytext=(5, 5), textcoords='offset points')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlabel('raw R@1 (descriptor quality)')
        ax.set_ylabel('Δ R@1 (cosine_bayesian − baseline)')
        ax.set_title(f'raw R@1 vs Δ (Pearson r={pear_r:+.2f})')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = Path(args.output_dir) / 'scatter.png'
        plt.savefig(out_png, dpi=120)
        print(f"\nSaved: {out_png}")


if __name__ == '__main__':
    main()
