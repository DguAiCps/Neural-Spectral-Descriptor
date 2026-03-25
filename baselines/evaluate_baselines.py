"""
Baseline LiDAR Place Recognition Evaluation.

Compares handcrafted + learned baselines against NSC on identical keyframes
and evaluation criteria (R@1, R@5, R@10 with revisit queries).

Usage (run inside container):
    # Quick test on KITTI_00 only
    python baselines/evaluate_baselines.py --methods sc++ m2dp fresco nsc_raw \
        --dataset-filter KITTI_00

    # All handcrafted + NSC on all val datasets
    python baselines/evaluate_baselines.py --methods sc++ m2dp fresco nsc_raw nsc_gnn

    # Full evaluation including test sets
    python baselines/evaluate_baselines.py --methods all --include-test
"""

import sys
import os
import argparse
import hashlib
import json
import gc
import time
import csv as csv_mod
import numpy as np
import torch
import yaml
from pathlib import Path
from collections import OrderedDict

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from baselines.eval_utils import compute_recall_multi_k

# Import baseline methods (triggers @register)
import baselines.scan_context
import baselines.fresco
import baselines.m2dp
import baselines.lidar_iris
import baselines.nsc
from baselines import REGISTRY, get_method


# ── Config/cache helpers (from train_multi_dataset.py) ─────
def compute_cache_key(config):
    enc = config['encoding']
    kf = config['keyframe']
    key_params = {
        'n_elevation': enc['n_elevation'], 'n_azimuth': enc['n_azimuth'],
        'n_bins': enc['n_bins'], 'elevation_range': enc['elevation_range'],
        'sensor_elevation_ranges': enc.get('sensor_elevation_ranges', {}),
        'target_elevation_bins': enc['target_elevation_bins'],
        'binning_strategy': enc.get('binning_strategy', 'exponential'),
        'bin_statistics': enc.get('bin_statistics', ['sum']),
        'inter_bin_statistics': enc.get('inter_bin_statistics', []),
        'zero_center': enc.get('zero_center', False),
        'log_magnitude': enc.get('log_magnitude', False),
        'normalize_channels': enc.get('normalize_channels', True),
        'distance_threshold': kf['distance_threshold'],
        'rotation_threshold': kf['rotation_threshold'],
        'overlap_threshold': kf['overlap_threshold'],
        'temporal_threshold': kf['temporal_threshold'],
    }
    return hashlib.sha256(json.dumps(key_params, sort_keys=True).encode()).hexdigest()[:8]


def get_cache_path(cache_dir, cache_key, dataset_name):
    return Path(cache_dir) / f"cache_{cache_key}_{dataset_name}.npz"


def load_cache_raw(path):
    """Load cache as raw numpy arrays (no Keyframe reconstruction)."""
    with np.load(path) as data:
        return {
            'descriptors': data['descriptors'],
            'poses': data['poses'],
            'timestamps': data['timestamps'],
            'scan_ids': data['scan_ids'],
            'keyframe_ids': data['keyframe_ids'],
        }


# ── Data loader factory ───────────────────────────────────
def create_loader(dataset_type, root, sequence):
    """Create the appropriate data loader."""
    if dataset_type == 'kitti':
        from data.kitti_loader import KITTILoader
        return KITTILoader(root, sequence, lazy_load=True)
    elif dataset_type == 'nclt':
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(root, sequence, lazy_load=True)
    elif dataset_type == 'helipr':
        from data.helipr_loader import HeLiPRLoader
        seq_path = os.path.join(root, sequence, sequence)
        return HeLiPRLoader(seq_path, lazy_load=True)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_point_clouds_at_indices(loader, scan_ids, progress_interval=500):
    """Load point clouds at specific keyframe scan indices."""
    point_clouds = []
    for i, sid in enumerate(scan_ids):
        data = loader[int(sid)]
        point_clouds.append(data['points'])
        if progress_interval and (i + 1) % progress_interval == 0:
            print(f"    Loading point clouds: [{i+1}/{len(scan_ids)}]")
    return point_clouds


# ── Test set preprocessing ────────────────────────────────
def preprocess_test_set(config, cache_key, cache_dir, dataset_type, root, sequence, device):
    """Generate cache for test sets that weren't preprocessed during training."""
    from encoding.spectral_encoder import SpectralEncoder
    from keyframe.selector import KeyframeSelector

    ds_name = f"{dataset_type}_test_{sequence}"
    cache_path = get_cache_path(cache_dir, cache_key, ds_name)

    if cache_path.exists():
        print(f"  Cache exists: {cache_path}")
        return cache_path

    print(f"  Preprocessing {ds_name}...")
    enc_cfg = config['encoding']
    kf_cfg = config['keyframe']

    elevation_range = enc_cfg.get('sensor_elevation_ranges', {}).get(
        dataset_type, tuple(enc_cfg['elevation_range']))
    if isinstance(elevation_range, list):
        elevation_range = tuple(elevation_range)

    encoder = SpectralEncoder(
        n_elevation=enc_cfg['n_elevation'],
        n_azimuth=enc_cfg['n_azimuth'],
        n_bins=enc_cfg['n_bins'],
        alpha=enc_cfg['alpha'],
        learnable_alpha=False,
        target_elevation_bins=enc_cfg['target_elevation_bins'],
        elevation_range=elevation_range,
        bin_statistics=enc_cfg.get('bin_statistics', ['sum']),
        inter_bin_statistics=enc_cfg.get('inter_bin_statistics', []),
        device=device,
        zero_center=enc_cfg.get('zero_center', False),
        log_magnitude=enc_cfg.get('log_magnitude', False),
        binning_strategy=enc_cfg.get('binning_strategy', 'exponential'),
        normalize_channels=enc_cfg.get('normalize_channels', True),
    )
    encoder.eval()

    keyframe_selector = KeyframeSelector(
        distance_threshold=kf_cfg['distance_threshold'],
        rotation_threshold=kf_cfg.get('rotation_threshold', 20.0),
        overlap_threshold=kf_cfg.get('overlap_threshold', 0.65),
        temporal_threshold=kf_cfg.get('temporal_threshold', 30.0),
        voxel_size=kf_cfg.get('voxel_size', 0.2),
        max_keyframes=kf_cfg.get('max_keyframes', 10000000),
    )

    loader = create_loader(dataset_type, root, sequence)

    keyframes = []
    for idx in range(len(loader)):
        data = loader[idx]
        with torch.no_grad():
            desc = encoder.encode_points(data['points']).cpu().numpy()
        accepted = keyframe_selector.process_scan(
            scan_id=data['idx'],
            points=data['points'],
            pose=data['pose'],
            timestamp=data['timestamp'],
            descriptor=desc,
        )
        if accepted:
            keyframes.append(keyframe_selector.keyframes[-1])
        if (idx + 1) % 500 == 0:
            print(f"    [{idx+1}/{len(loader)}] keyframes: {len(keyframes)}")

    print(f"    Total keyframes: {len(keyframes)}")

    # Save cache
    descriptors = np.array([kf.descriptor for kf in keyframes])
    poses = np.array([kf.pose for kf in keyframes])
    timestamps = np.array([kf.timestamp for kf in keyframes])
    scan_ids = np.array([kf.scan_id for kf in keyframes])
    keyframe_ids = np.array([kf.keyframe_id for kf in keyframes])
    np.savez(cache_path, descriptors=descriptors, poses=poses,
             timestamps=timestamps, scan_ids=scan_ids, keyframe_ids=keyframe_ids)
    print(f"    Saved cache: {cache_path}")

    del loader, keyframes, encoder, keyframe_selector
    gc.collect()
    return cache_path


# ── Dataset definition ────────────────────────────────────
def build_eval_datasets(config, cache_key, cache_dir, include_test=False, device='cuda'):
    """
    Build list of evaluation datasets from config.

    Returns:
        List of dicts: {display_name, dataset_type, root, sequence, cache_path}
    """
    datasets = []

    # Validation sets
    for ds_cfg in config['data']['datasets'].get('val', []):
        dtype = ds_cfg['type']
        root = ds_cfg['root']
        for seq in ds_cfg['sequences']:
            ds_name = f"{dtype}_val_{seq}"
            cache_path = get_cache_path(cache_dir, cache_key, ds_name)
            if not cache_path.exists():
                print(f"  WARNING: Cache missing for {ds_name}: {cache_path}")
                continue
            display = f"{dtype.upper()}_{seq}"
            datasets.append({
                'display_name': display,
                'dataset_type': dtype,
                'root': root,
                'sequence': seq,
                'cache_path': cache_path,
                'split': 'val',
            })

    # Test sets
    if include_test:
        for ds_cfg in config['data']['datasets'].get('test', []):
            dtype = ds_cfg['type']
            root = ds_cfg['root']
            for seq in ds_cfg['sequences']:
                cache_path = preprocess_test_set(
                    config, cache_key, cache_dir, dtype, root, seq, device)
                display = f"{dtype.upper()}_{seq}"
                datasets.append({
                    'display_name': display,
                    'dataset_type': dtype,
                    'root': root,
                    'sequence': seq,
                    'cache_path': cache_path,
                    'split': 'test',
                })

    return datasets


# ── Main evaluation ───────────────────────────────────────
def evaluate_method_on_dataset(method_name, method_instance, point_clouds,
                               cache_data, poses, config, checkpoint_path, device):
    """
    Evaluate a single method on a single dataset.

    Returns:
        dict with recalls, n_queries, encoding_time_ms
    """
    k_values = [1, 5, 10]

    if method_name == 'nsc_raw':
        # Use cached descriptors directly
        descriptors = cache_data['descriptors']
        encoding_time_ms = 0.0

    elif method_name == 'nsc_gnn':
        # GNN forward pass
        nsc_gnn = baselines.nsc.NSCGNN()
        t0 = time.perf_counter()
        descriptors = nsc_gnn.compute_embeddings(
            cache_data, config, checkpoint_path, device)
        encoding_time_ms = (time.perf_counter() - t0) * 1000 / len(poses)

    else:
        # Standard baseline: encode from point clouds
        descriptors = method_instance.encode_sequence(point_clouds)
        encoding_time_ms = method_instance.last_encode_time_ms

    recalls, n_queries = compute_recall_multi_k(
        descriptors, poses, k_values=k_values,
        distance_threshold=5.0, skip_frames=30)

    return {
        'recalls': recalls,
        'n_queries': n_queries,
        'encoding_time_ms': encoding_time_ms,
    }


def print_results_table(all_results, methods_order, datasets_order):
    """Print formatted comparison table."""
    # Group datasets for display
    print(f"\n{'='*100}")
    print(f"  BASELINE EVALUATION — R@K (distance < 5m, skip > 30 frames)")
    print(f"{'='*100}")

    # Header
    header = f"{'Method':<16} {'Dim':>4}"
    for ds in datasets_order:
        header += f"  {'R@1':>5} {'R@5':>5} {'R@10':>5}"
    header += f"  {'ms':>6}"
    print(header)

    # Sub-header with dataset names
    sub = f"{'':>21}"
    for ds in datasets_order:
        name = ds['display_name']
        # Center the name across 17 chars (5+1+5+1+5)
        sub += f"  {name:^17}"
    print(sub)
    print(f"{'─'*100}")

    for method_name in methods_order:
        method_cls = get_method(method_name)
        inst = method_cls()
        dim = inst.descriptor_dim

        row = f"{inst.name:<16} {dim:>4}"
        all_times = []

        for ds in datasets_order:
            key = (method_name, ds['display_name'])
            if key in all_results:
                r = all_results[key]
                r1 = r['recalls'].get(1, 0)
                r5 = r['recalls'].get(5, 0)
                r10 = r['recalls'].get(10, 0)
                row += f"  {r1:>5.3f} {r5:>5.3f} {r10:>5.3f}"
                all_times.append(r['encoding_time_ms'])
            else:
                row += f"  {'---':>5} {'---':>5} {'---':>5}"

        avg_ms = np.mean(all_times) if all_times else 0.0
        row += f"  {avg_ms:>6.1f}"
        print(row)

    print(f"{'='*100}")

    # Print query counts
    print(f"\n  Queries per dataset:")
    for ds in datasets_order:
        for method_name in methods_order:
            key = (method_name, ds['display_name'])
            if key in all_results:
                print(f"    {ds['display_name']}: {all_results[key]['n_queries']} queries")
                break


def save_csv(all_results, methods_order, datasets_order, output_path):
    """Save results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv_mod.writer(f)
        writer.writerow([
            'method', 'short_name', 'dim', 'dataset', 'split',
            'recall_at_1', 'recall_at_5', 'recall_at_10',
            'n_queries', 'encoding_time_ms'
        ])

        for method_name in methods_order:
            inst = get_method(method_name)()
            for ds in datasets_order:
                key = (method_name, ds['display_name'])
                if key not in all_results:
                    continue
                r = all_results[key]
                writer.writerow([
                    inst.name, inst.short_name, inst.descriptor_dim,
                    ds['display_name'], ds['split'],
                    f"{r['recalls'].get(1, 0):.4f}",
                    f"{r['recalls'].get(5, 0):.4f}",
                    f"{r['recalls'].get(10, 0):.4f}",
                    r['n_queries'],
                    f"{r['encoding_time_ms']:.2f}",
                ])

    print(f"\n  CSV saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline LiDAR PR methods')
    parser.add_argument('--config', default='configs/training_multi_dataset.yaml',
                        help='Training config file')
    parser.add_argument('--checkpoint', default='src/checkpoints/best_model.pth',
                        help='GNN checkpoint for nsc_gnn method')
    parser.add_argument('--methods', nargs='+', default=['all'],
                        help='Methods to evaluate (sc++, m2dp, fresco, nsc_raw, nsc_gnn, all)')
    parser.add_argument('--include-test', action='store_true',
                        help='Include test sets (KITTI 09/10)')
    parser.add_argument('--dataset-filter', nargs='+', default=None,
                        help='Only evaluate on these datasets (e.g., KITTI_00)')
    parser.add_argument('--output', default='outputs/baseline_results.csv',
                        help='Output CSV path')
    parser.add_argument('--device', default='cuda',
                        help='Device for GNN inference')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    cache_dir = config['data']['cache_dir']
    cache_key = compute_cache_key(config)
    print(f"Config: {args.config}")
    print(f"Cache key: {cache_key}, dir: {cache_dir}")

    # Resolve methods
    if 'all' in args.methods:
        methods_order = list(REGISTRY.keys())
    else:
        methods_order = args.methods

    # Check availability
    available_methods = []
    for m in methods_order:
        inst = get_method(m)()
        if inst.is_available():
            available_methods.append(m)
            print(f"  [{m}] {inst.name} ({inst.descriptor_dim}D) — available")
        else:
            print(f"  [{m}] {inst.name} — SKIPPED (dependencies missing)")
    methods_order = available_methods

    # Build dataset list
    device = args.device if torch.cuda.is_available() else 'cpu'
    datasets = build_eval_datasets(
        config, cache_key, cache_dir,
        include_test=args.include_test, device=device)

    if args.dataset_filter:
        datasets = [d for d in datasets if d['display_name'] in args.dataset_filter]

    print(f"\nDatasets ({len(datasets)}):")
    for ds in datasets:
        print(f"  {ds['display_name']} [{ds['split']}] — {ds['cache_path']}")

    # Evaluate
    all_results = {}  # (method_name, display_name) → result dict

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {ds['display_name']} ({ds['split']})")
        print(f"{'='*60}")

        # Load cache
        cache_data = load_cache_raw(ds['cache_path'])
        poses = cache_data['poses']
        scan_ids = cache_data['scan_ids']
        n_kf = len(scan_ids)
        print(f"  Keyframes: {n_kf}")

        # Check which methods need point clouds
        need_point_clouds = any(
            m not in ('nsc_raw', 'nsc_gnn') for m in methods_order
        )

        point_clouds = None
        if need_point_clouds:
            print(f"  Loading point clouds at {n_kf} keyframe indices...")
            loader = create_loader(ds['dataset_type'], ds['root'], ds['sequence'])
            point_clouds = load_point_clouds_at_indices(loader, scan_ids)
            del loader
            gc.collect()

        for method_name in methods_order:
            inst = get_method(method_name)()
            print(f"\n  [{inst.short_name}] {inst.name} ({inst.descriptor_dim}D)...")

            result = evaluate_method_on_dataset(
                method_name, inst, point_clouds,
                cache_data, poses, config, args.checkpoint, device)

            r = result['recalls']
            print(f"    R@1={r.get(1,0):.4f}  R@5={r.get(5,0):.4f}  "
                  f"R@10={r.get(10,0):.4f}  "
                  f"({result['n_queries']} queries, "
                  f"{result['encoding_time_ms']:.1f} ms/scan)")

            all_results[(method_name, ds['display_name'])] = result

        # Free point clouds
        del point_clouds, cache_data
        gc.collect()

    # Print summary table
    print_results_table(all_results, methods_order, datasets)

    # Save CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(all_results, methods_order, datasets, output_path)


if __name__ == "__main__":
    main()
