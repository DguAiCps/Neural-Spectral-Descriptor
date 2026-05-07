"""
Rotation invariance evaluation (paper Table 5: tab:rotation).

For each baseline + NSC, on a single sequence (default KITTI 00):
    Stability = mean similarity between d(x) and d(R_Δφ x) over 24 yaw offsets,
                averaged over a random subset of n_stability_samples scans.
    R@1 (random yaw) = recall@1 with each query augmented by an independent
                       random yaw, querying the unrotated database.

Float-descriptor methods use cosine similarity for Stability. Binary-template
methods (LiDAR-Iris) use 1 - normalized_min_shift_hamming.

Usage (inside the nsc container):
    python -u baselines/evaluate_rotation_invariance.py \
        --methods sc++ m2dp fresco lidar_iris nsc_raw nsc_gnn \
        --dataset KITTI_00 --n_rotations 24 --n_stability_samples 200
"""

import argparse
import csv as csv_mod
import gc
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from baselines import REGISTRY, get_method  # noqa: E402
import baselines.fresco  # noqa: E402,F401
import baselines.lidar_iris  # noqa: E402,F401
import baselines.m2dp  # noqa: E402,F401
import baselines.nsc  # noqa: E402,F401
import baselines.scan_context  # noqa: E402,F401
from baselines.evaluate_baselines import (  # noqa: E402
    build_eval_datasets, compute_cache_key, create_loader, load_cache_raw,
    load_point_clouds_at_indices,
)
from baselines.eval_utils import _find_revisit_queries, _score_recalls_from_ranked  # noqa: E402
from baselines.lidar_iris import _hamming_min_shift  # noqa: E402


def _yaw_rotate(points: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=points.dtype)
    out = points.copy()
    out[:, :3] = points[:, :3] @ R.T
    return out


def _build_nsc_encoder(config, device):
    from encoding.spectral_encoder import SpectralEncoder
    enc_cfg = config['encoding']
    elev = enc_cfg.get('sensor_elevation_ranges', {}).get(
        'kitti', tuple(enc_cfg['elevation_range']))
    if isinstance(elev, list):
        elev = tuple(elev)
    enc = SpectralEncoder(
        n_elevation=enc_cfg['n_elevation'], n_azimuth=enc_cfg['n_azimuth'],
        n_bins=enc_cfg['n_bins'], alpha=enc_cfg['alpha'],
        learnable_alpha=False,
        target_elevation_bins=enc_cfg['target_elevation_bins'],
        elevation_range=elev,
        bin_statistics=enc_cfg.get('bin_statistics', ['sum']),
        inter_bin_statistics=enc_cfg.get('inter_bin_statistics', []),
        device=device,
        zero_center=enc_cfg.get('zero_center', False),
        log_magnitude=enc_cfg.get('log_magnitude', False),
        binning_strategy=enc_cfg.get('binning_strategy', 'exponential'),
        normalize_channels=enc_cfg.get('normalize_channels', True),
    )
    enc.eval()
    return enc


def _encode_method(method_name, point_clouds, inst, cache_data, config,
                   checkpoint_path, device, return_aux=False):
    """Returns (descriptors, aux_dict_or_None). aux_dict has 'templates' for
    LiDAR-Iris (binary case)."""
    if method_name == 'nsc_raw':
        # NSC raw uses cached descriptors when point_clouds correspond to
        # the cache scan_ids; otherwise encode from scratch.
        if point_clouds is None:
            return cache_data['descriptors'].astype(np.float32), None
        encoder = _build_nsc_encoder(config, device)
        with torch.no_grad():
            d = np.stack([
                encoder.encode_points(p).cpu().numpy() for p in point_clouds
            ], axis=0).astype(np.float32)
        return d, None
    if method_name == 'nsc_gnn':
        encoder = _build_nsc_encoder(config, device)
        with torch.no_grad():
            raw = np.stack([
                encoder.encode_points(p).cpu().numpy() for p in point_clouds
            ], axis=0).astype(np.float32)
        # nsc_gnn requires graph-aligned poses/scan_ids/timestamps. When
        # point_clouds is a subset, the cache must be subset to match. Caller
        # passes the index list as cache_data['_subset_idx'] when needed.
        sub_idx = cache_data.get('_subset_idx', None)
        if sub_idx is None:
            cache_local = {**cache_data, 'descriptors': raw}
        else:
            cache_local = {
                'descriptors': raw,
                'poses': cache_data['poses'][sub_idx],
                'timestamps': cache_data['timestamps'][sub_idx],
                'scan_ids': cache_data['scan_ids'][sub_idx],
                'keyframe_ids': cache_data['keyframe_ids'][sub_idx],
            }
        d = baselines.nsc.NSCGNN().compute_embeddings(
            cache_local, config, checkpoint_path, device).astype(np.float32)
        return d, None
    if method_name == 'lidar_iris':
        d, aux_list = inst.encode_sequence_with_aux(point_clouds, progress_interval=0)
        templates = np.stack([a['templates'] for a in aux_list], axis=0)
        return d, templates if return_aux else None
    return inst.encode_sequence(point_clouds, progress_interval=0), None


def evaluate_method_rotation(method_name, point_clouds, poses, rotations,
                             cache_data, config, checkpoint_path, device,
                             n_stability_samples):
    """Returns dict with mean_stability, recall_at_1, n_queries."""
    inst = get_method(method_name)() if method_name in REGISTRY else None
    is_binary = (method_name == 'lidar_iris')

    print(f"    Encoding originals...")
    t0 = time.perf_counter()
    descs_orig, templates_orig = _encode_method(
        method_name, point_clouds, inst, cache_data, config,
        checkpoint_path, device, return_aux=is_binary,
    )
    enc_time_orig = time.perf_counter() - t0
    n = len(descs_orig)

    # Stability: subsample n_stability_samples scans, encode at each rotation.
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=min(n_stability_samples, n), replace=False)
    sample_pcs = [point_clouds[i] for i in sample_idx]

    sims_per_scan = np.zeros(len(sample_idx), dtype=np.float64)
    # For nsc_gnn, the GNN context depends on the trajectory graph. To measure
    # stability faithfully, rotate ALL keyframes by theta and run the full GNN,
    # then compare per-scan in the sample subset. For other methods, encoding
    # is per-scan so we encode just the subset.
    for ri, theta in enumerate(rotations):
        if method_name == 'nsc_gnn':
            rot_pcs_full = [_yaw_rotate(pc, theta) for pc in point_clouds]
            d_rot_full, _ = _encode_method(
                method_name, rot_pcs_full, inst, cache_data, config,
                checkpoint_path, device, return_aux=False,
            )
            d_rot = d_rot_full[sample_idx]
            del rot_pcs_full, d_rot_full
        else:
            rot_pcs = [_yaw_rotate(pc, theta) for pc in sample_pcs]
            d_rot, t_rot = _encode_method(
                method_name, rot_pcs, inst, cache_data, config,
                checkpoint_path, device, return_aux=is_binary,
            )
            del rot_pcs

        if is_binary:
            for j, j_idx in enumerate(sample_idx):
                d = _hamming_min_shift(
                    templates_orig[j_idx], t_rot[j], max_shift=180
                )
                sims_per_scan[j] += (1.0 - d)
        else:
            for j, j_idx in enumerate(sample_idx):
                a = descs_orig[j_idx]; b = d_rot[j]
                an = a / (np.linalg.norm(a) + 1e-8)
                bn = b / (np.linalg.norm(b) + 1e-8)
                sims_per_scan[j] += float(np.dot(an, bn))
        print(f"      stab rot {ri+1}/{len(rotations)} done")
        del d_rot
        gc.collect()
    stability = float(np.mean(sims_per_scan / len(rotations)))

    # Random yaw R@1: encode all scans at one random yaw each, query against orig DB.
    print(f"    Encoding random-yaw queries (full {n})...")
    rand_thetas = rng.uniform(-np.pi, np.pi, size=n)
    rot_all_pcs = [_yaw_rotate(pc, t) for pc, t in zip(point_clouds, rand_thetas)]
    descs_query, templates_query = _encode_method(
        method_name, rot_all_pcs, inst, cache_data, config,
        checkpoint_path, device, return_aux=is_binary,
    )
    del rot_all_pcs

    positions = poses[:, :3, 3].astype(np.float64)
    queries = _find_revisit_queries(positions, distance_threshold=5.0, skip_frames=30)
    if not queries:
        return {'mean_stability': stability, 'recall_at_1': 0.0,
                'n_queries': 0, 'enc_time_ms_per_scan': enc_time_orig * 1000 / n}

    # Build cosine FAISS over the unrotated descriptors as the database.
    import faiss
    db = descs_orig.astype(np.float32).copy()
    faiss.normalize_L2(db)
    qv = descs_query.astype(np.float32).copy()
    faiss.normalize_L2(qv)
    idx = faiss.IndexFlatIP(db.shape[1]); idx.add(db)
    max_k = 1
    search_k = min(max_k + 60, len(db))

    ranked_lists = []
    for q_idx, _ in queries:
        _, ind = idx.search(qv[q_idx:q_idx + 1], search_k)
        valid = np.abs(ind[0] - q_idx) > 30
        cands = ind[0][valid][:max_k]
        if is_binary and len(cands) > 0:
            t_q = templates_query[q_idx]
            cdists = np.array([
                _hamming_min_shift(t_q, templates_orig[c], max_shift=180)
                for c in cands
            ])
            cands = cands[np.argsort(cdists)]
        ranked_lists.append(cands)

    recalls = _score_recalls_from_ranked(
        queries, ranked_lists, positions, [1], distance_threshold=5.0
    )
    return {
        'mean_stability': stability,
        'recall_at_1': recalls[1],
        'n_queries': len(queries),
        'enc_time_ms_per_scan': enc_time_orig * 1000 / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/training_paper_table4.yaml')
    parser.add_argument('--checkpoint', default='src/checkpoints/best_model.pth')
    parser.add_argument('--methods', nargs='+',
                        default=['sc++', 'm2dp', 'fresco', 'lidar_iris',
                                 'nsc_raw', 'nsc_gnn'])
    parser.add_argument('--dataset', default='KITTI_00')
    parser.add_argument('--n_rotations', type=int, default=24)
    parser.add_argument('--n_stability_samples', type=int, default=200)
    parser.add_argument('--output', default='outputs/rotation_invariance_v2.csv')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cache-key', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    cache_dir = config['data']['cache_dir']
    cache_key = args.cache_key if args.cache_key else compute_cache_key(config)
    print(f"Cache key: {cache_key}{' (override)' if args.cache_key else ''}")

    device = args.device if torch.cuda.is_available() else 'cpu'
    datasets = build_eval_datasets(config, cache_key, cache_dir,
                                   include_test=False, device=device)
    target = [d for d in datasets if d['display_name'] == args.dataset]
    if not target:
        raise ValueError(
            f"Dataset {args.dataset} not found. Available: "
            f"{[d['display_name'] for d in datasets]}"
        )
    ds = target[0]
    print(f"Evaluating rotation invariance on {ds['display_name']}")

    cache_data = load_cache_raw(ds['cache_path'])
    poses = cache_data['poses']
    scan_ids = cache_data['scan_ids']
    print(f"  Keyframes: {len(scan_ids)}")

    loader = create_loader(ds['dataset_type'], ds['root'], ds['sequence'])
    point_clouds = load_point_clouds_at_indices(loader, scan_ids)
    del loader

    rotations = np.linspace(0, 2 * np.pi, args.n_rotations, endpoint=False).tolist()

    results = {}
    for method_name in args.methods:
        if method_name not in REGISTRY and method_name not in ('nsc_raw', 'nsc_gnn'):
            print(f"  Skipping unknown method: {method_name}")
            continue
        print(f"\n[{method_name}]")
        r = evaluate_method_rotation(
            method_name, point_clouds, poses, rotations,
            cache_data, config, args.checkpoint, device,
            args.n_stability_samples,
        )
        print(f"  Stability={r['mean_stability']:.4f}  R@1={r['recall_at_1']:.4f}  "
              f"({r['n_queries']} queries)")
        results[method_name] = r

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv_mod.writer(f)
        w.writerow(['method', 'dataset', 'mean_stability', 'recall_at_1',
                    'n_queries', 'enc_time_ms_per_scan'])
        for m, r in results.items():
            w.writerow([m, args.dataset, f"{r['mean_stability']:.4f}",
                        f"{r['recall_at_1']:.4f}", r['n_queries'],
                        f"{r['enc_time_ms_per_scan']:.2f}"])
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
