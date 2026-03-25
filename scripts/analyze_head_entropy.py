#!/usr/bin/env python3
"""
Attention Head Entropy Analysis for DiffAttnConv

Evaluates whether multi-head attention heads learn meaningfully different patterns
by computing per-head entropy of attention distributions.

Usage:
    python scripts/analyze_head_entropy.py \
        --checkpoint src/checkpoints/best_model.pth \
        --config configs/training_multi_dataset.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import hashlib
import json
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_cache_key(config):
    enc = config['encoding']
    kf = config['keyframe']
    projection_type = enc.get('projection_type', 'range_image')
    key_params = {
        'projection_type': projection_type,
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
    if projection_type == 'bev':
        key_params['bev'] = enc.get('bev', {})
    else:
        key_params['n_elevation'] = enc['n_elevation']
        key_params['elevation_range'] = enc['elevation_range']
        key_params['sensor_elevation_ranges'] = enc.get('sensor_elevation_ranges', {})
        key_params['target_elevation_bins'] = enc['target_elevation_bins']
    return hashlib.sha256(json.dumps(key_params, sort_keys=True).encode()).hexdigest()[:8]


def load_keyframes_cache(path):
    from keyframe.selector import Keyframe
    with np.load(path) as data:
        descriptors = data['descriptors']
        poses = data['poses']
        timestamps = data['timestamps']
        scan_ids = data['scan_ids']
        keyframe_ids = data['keyframe_ids']
        keyframes = []
        for i in range(len(scan_ids)):
            kf = Keyframe(
                keyframe_id=int(keyframe_ids[i]),
                scan_id=int(scan_ids[i]),
                points=np.empty((0, 3)),
                pose=poses[i],
                timestamp=float(timestamps[i]),
                descriptor=descriptors[i],
            )
            keyframes.append(kf)
        return keyframes


def compute_per_node_entropy(attn_weights, edge_index, n_nodes, n_heads):
    """
    Compute per-node, per-head attention entropy.

    Args:
        attn_weights: (n_edges, n_heads) attention scores (already softmaxed per target node)
        edge_index: (2, n_edges) [src, tgt]
        n_nodes: total number of nodes
        n_heads: number of attention heads

    Returns:
        entropy: (n_nodes, n_heads) — H(h, i) = -Σ_j α_{ij}^{(h)} log α_{ij}^{(h)}
    """
    tgt = edge_index[1]  # target nodes
    entropy = torch.zeros(n_nodes, n_heads, device=attn_weights.device)

    # -α log(α), with 0·log(0) = 0
    elem_entropy = -attn_weights * torch.log(attn_weights + 1e-10)

    # Scatter-add per target node
    tgt_expanded = tgt.unsqueeze(-1).expand_as(elem_entropy)
    entropy.scatter_add_(0, tgt_expanded, elem_entropy)

    return entropy


def main():
    parser = argparse.ArgumentParser(description='Attention Head Entropy Analysis')
    parser.add_argument('--checkpoint', type=str, default='src/checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/training_multi_dataset.yaml')
    parser.add_argument('--output-dir', type=str, default='outputs/head_entropy')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ========================================================================
    # Load validation data from cache
    # ========================================================================
    cache_dir = config['data'].get('cache_dir', 'data/preprocessed')
    cache_key = compute_cache_key(config)
    print(f"Cache key: {cache_key}")

    val_datasets_config = config['data']['datasets']['val']
    val_datasets = {}

    for dataset_cfg in val_datasets_config:
        dataset_type = dataset_cfg['type']
        sequences = dataset_cfg['sequences']
        for seq in sequences:
            if dataset_type == 'kitti':
                ds_name = f"kitti_val_{seq}"
                display_name = f"KITTI_{seq}"
            elif dataset_type == 'nclt':
                ds_name = f"nclt_val_{seq}"
                display_name = f"NCLT_{seq}"
            elif dataset_type == 'helipr':
                ds_name = f"helipr_val_{seq}"
                display_name = f"HeLiPR_{seq}"
            else:
                continue

            cache_path = Path(cache_dir) / f"cache_{cache_key}_{ds_name}.npz"
            if not cache_path.exists():
                print(f"  Cache not found: {cache_path} — skipping")
                continue

            keyframes = load_keyframes_cache(cache_path)
            val_datasets[display_name] = keyframes
            print(f"  Loaded {display_name}: {len(keyframes)} keyframes")

    if not val_datasets:
        print("ERROR: No cached validation data found. Run training first.")
        return

    # ========================================================================
    # Build graphs
    # ========================================================================
    from keyframe.graph_manager import build_graph_from_keyframes_batch
    from utils.similarity_stats import SimilarityDistribution

    graph_config = config['keyframe'].get('graph', {})
    edge_method = graph_config.get('edge_method', 'threshold')

    similarity_metric = graph_config.get('similarity_metric', 'cosine')
    standardization_stats = None
    if similarity_metric == 'l2':
        from utils.standardization_stats import StandardizationStats
        std_path = os.path.join(os.path.dirname(args.checkpoint), 'standardization_stats.npz')
        if os.path.exists(std_path):
            standardization_stats = StandardizationStats().load(std_path)
            print(f"  Loaded standardization stats from {std_path}")

    similarity_dist = None
    bayesian_config = {}
    if edge_method == 'bayesian':
        dist_cache_path = os.path.join(os.path.dirname(args.checkpoint), 'similarity_dist.npz')
        if os.path.exists(dist_cache_path):
            similarity_dist = SimilarityDistribution(metric=similarity_metric).load(dist_cache_path)
            print(f"  Loaded Bayesian distribution from {dist_cache_path}")
        bayesian_config = {
            'confidence_level': graph_config.get('confidence_level', 0.95),
            'base_prior': graph_config.get('base_prior', 0.01),
            'density_k': graph_config.get('density_k', 50),
            'density_beta': graph_config.get('density_beta', 10.0),
        }

    val_graphs = {}
    for name, keyframes in val_datasets.items():
        poses = np.array([kf.pose for kf in keyframes])
        descs = np.array([kf.descriptor for kf in keyframes])
        graph = build_graph_from_keyframes_batch(
            keyframes,
            temporal_neighbors=config['keyframe']['temporal_neighbors'],
            device=device,
            poses=poses,
            descriptors=descs,
            similarity_threshold=graph_config.get('similarity_threshold', 0.993),
            similarity_max_k=graph_config.get('similarity_max_k', 10),
            similarity_exclude_temporal=graph_config.get('similarity_exclude_temporal', True),
            similarity_dist=similarity_dist,
            similarity_metric=similarity_metric,
            standardization_stats=standardization_stats,
            **bayesian_config,
        )
        val_graphs[name] = graph
        n_temp = int((graph.edge_type == 0).sum()) if hasattr(graph, 'edge_type') else 0
        n_sim = int((graph.edge_type == 1).sum()) if hasattr(graph, 'edge_type') else 0
        print(f"  {name} graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges (temp={n_temp}, sim={n_sim})")

    # ========================================================================
    # Load GNN model
    # ========================================================================
    from gnn.model import create_spectral_gnn

    edge_enc_config = config['gnn'].get('edge_encoding', None)
    gnn_model = create_spectral_gnn(
        input_dim=config['gnn']['input_dim'],
        hidden_dim=config['gnn']['hidden_dim'],
        context_dim=config['gnn']['context_dim'],
        n_layers=config['gnn']['n_layers'],
        n_heads=config['gnn'].get('n_heads', 4),
        dropout=config['gnn']['dropout'],
        edge_encoder_config=edge_enc_config,
        gradient_checkpointing=False
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    gnn_model.load_state_dict(state_dict)
    gnn_model.eval()
    print(f"\nLoaded checkpoint: {args.checkpoint}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}, Best R@1: {checkpoint.get('best_r1', 'N/A')}")

    n_heads = config['gnn'].get('n_heads', 4)
    n_layers = config['gnn']['n_layers']
    # Access base GNN (unwrap LocalUpdateGNN if needed)
    base_gnn = gnn_model.gnn if hasattr(gnn_model, 'gnn') else gnn_model

    # ========================================================================
    # Compute attention entropy
    # ========================================================================
    print(f"\n{'='*60}")
    print("ATTENTION HEAD ENTROPY ANALYSIS")
    print(f"{'='*60}")

    all_results = {}

    for name, graph in val_graphs.items():
        print(f"\n--- {name} ---")
        with torch.no_grad():
            _, attn_weights_list = base_gnn.forward_with_attention(graph)

        edge_type = graph.edge_type if hasattr(graph, 'edge_type') else None
        n_nodes = graph.num_nodes

        layer_results = []
        for layer_idx, (edge_index, attn) in enumerate(attn_weights_list):
            # attn shape: (n_edges, n_heads)
            entropy = compute_per_node_entropy(attn, edge_index, n_nodes, n_heads)
            entropy_np = entropy.cpu().numpy()

            print(f"\n  Layer {layer_idx}:")
            for h in range(n_heads):
                h_ent = entropy_np[:, h]
                print(f"    Head {h}: mean={h_ent.mean():.4f}, std={h_ent.std():.4f}, "
                      f"median={np.median(h_ent):.4f}, min={h_ent.min():.4f}, max={h_ent.max():.4f}")

            # Edge-type breakdown
            if edge_type is not None:
                tgt = edge_index[1].cpu()
                for etype, etype_name in [(0, 'temporal'), (1, 'similarity')]:
                    mask = (edge_type == etype)
                    if mask.sum() == 0:
                        continue
                    masked_attn = attn[mask]
                    masked_edge_index = edge_index[:, mask]
                    etype_entropy = compute_per_node_entropy(
                        masked_attn, masked_edge_index, n_nodes, n_heads
                    )
                    etype_np = etype_entropy.cpu().numpy()
                    # Only nodes with edges of this type
                    has_edges = etype_np.sum(axis=1) > 0
                    if has_edges.sum() == 0:
                        continue
                    print(f"\n    [{etype_name} edges only] (nodes with edges: {has_edges.sum()})")
                    for h in range(n_heads):
                        h_ent = etype_np[has_edges, h]
                        print(f"      Head {h}: mean={h_ent.mean():.4f}, std={h_ent.std():.4f}")

            layer_results.append(entropy_np)
        all_results[name] = layer_results

    # ========================================================================
    # Visualization
    # ========================================================================
    print(f"\n{'='*60}")
    print("Generating plots...")

    # 1. Box plot: per-head entropy distribution (aggregated across all datasets)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        head_data = []
        for name, layer_results in all_results.items():
            head_data.append(layer_results[layer_idx])
        # Stack across datasets: (total_nodes, n_heads)
        stacked = np.concatenate(head_data, axis=0)

        bp = ax.boxplot(
            [stacked[:, h] for h in range(n_heads)],
            labels=[f'Head {h}' for h in range(n_heads)],
            patch_artist=True
        )
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for patch, color in zip(bp['boxes'], colors[:n_heads]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f'Layer {layer_idx}')
        ax.set_ylabel('Entropy (nats)')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Per-Head Attention Entropy Distribution', fontsize=14)
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'head_entropy_boxplot.png')
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()

    # 2. Per-dataset bar chart
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    dataset_names = list(all_results.keys())
    x = np.arange(len(dataset_names))
    width = 0.8 / n_heads

    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        for h in range(n_heads):
            means = [all_results[name][layer_idx][:, h].mean() for name in dataset_names]
            ax.bar(x + h * width, means, width, label=f'Head {h}', alpha=0.8)
        ax.set_xticks(x + width * (n_heads - 1) / 2)
        ax.set_xticklabels(dataset_names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Mean Entropy')
        ax.set_title(f'Layer {layer_idx}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Mean Attention Entropy per Dataset', fontsize=14)
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'head_entropy_per_dataset.png')
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()

    # 3. Head-pair correlation heatmap
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        all_entropy = np.concatenate(
            [all_results[name][layer_idx] for name in dataset_names], axis=0
        )
        # Pearson correlation between heads
        corr = np.corrcoef(all_entropy.T)  # (n_heads, n_heads)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_heads))
        ax.set_xticklabels([f'H{h}' for h in range(n_heads)])
        ax.set_yticklabels([f'H{h}' for h in range(n_heads)])
        ax.set_title(f'Layer {layer_idx}')
        for i in range(n_heads):
            for j in range(n_heads):
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle('Head Entropy Correlation (Pearson)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'head_entropy_correlation.png')
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()

    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == '__main__':
    main()
