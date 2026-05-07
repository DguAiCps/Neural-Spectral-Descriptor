"""
NSC (Neural Spectral Codec) wrappers for baseline comparison.

Two variants:
- NSCRaw: 992D raw spectral descriptor (loaded from cache, no encoding needed)
          = 16×16×2 (mean+std per bin) + 16×15×2 (inter-bin diff) = 512 + 480
- NSCGNN: 1248D GNN-enhanced descriptor (requires model checkpoint)
          = cat(L2_norm(raw_992), L2_norm(ctx_256))
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baselines.base import BaselineEncoder
from baselines import register


@register
class NSCRaw(BaselineEncoder):
    """NSC raw spectral histogram descriptor (992D). Loaded from cache."""

    @property
    def name(self):
        return "NSC (raw)"

    @property
    def short_name(self):
        return "nsc_raw"

    @property
    def descriptor_dim(self):
        return 992  # 16×16×2 (mean+std) + 16×15×2 (inter-diff) = 512 + 480

    def encode(self, points):
        raise NotImplementedError("NSCRaw uses cached descriptors, not encode()")


@register
class NSCGNN(BaselineEncoder):
    """NSC + GNN enhanced descriptor (1248D). Requires checkpoint."""

    def __init__(self):
        self._model = None
        self._device = None

    @property
    def name(self):
        return "NSC + GNN"

    @property
    def short_name(self):
        return "nsc_gnn"

    @property
    def descriptor_dim(self):
        return 1248  # cat(L2_norm(raw_992), L2_norm(ctx_256))

    def encode(self, points):
        raise NotImplementedError("NSCGNN uses graph-based inference, not encode()")

    def compute_embeddings(self, cache_data, config, checkpoint_path, device='cuda'):
        """
        Run GNN forward pass on cached keyframes.

        Args:
            cache_data: dict with 'descriptors', 'poses', 'timestamps',
                       'scan_ids', 'keyframe_ids'
            config: training config dict
            checkpoint_path: path to best_model.pth
            device: 'cuda' or 'cpu'

        Returns:
            (n, 512) GNN-enhanced embeddings
        """
        from keyframe.selector import Keyframe
        from keyframe.graph_manager import build_graph_from_keyframes_batch
        from gnn.model import create_spectral_gnn

        descriptors = cache_data['descriptors']
        poses = cache_data['poses']
        timestamps = cache_data['timestamps']
        scan_ids = cache_data['scan_ids']
        keyframe_ids = cache_data['keyframe_ids']

        # Reconstruct Keyframe objects
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

        # Build graph
        gnn_cfg = config['gnn']
        graph_cfg = config['keyframe']['graph']

        # Load standardization stats if using L2 metric
        similarity_metric = graph_cfg.get('similarity_metric', 'cosine')
        standardization_stats = None
        if similarity_metric == 'l2':
            from utils.standardization_stats import StandardizationStats
            std_path = os.path.join(
                os.path.dirname(checkpoint_path), 'standardization_stats.npz'
            )
            if os.path.exists(std_path):
                standardization_stats = StandardizationStats().load(std_path)

        # Load Bayesian similarity distribution if available
        similarity_dist = None
        edge_method = graph_cfg.get('edge_method', 'threshold')
        bayesian_config = {}
        if edge_method == 'bayesian':
            from utils.similarity_stats import SimilarityDistribution
            dist_path = os.path.join(
                os.path.dirname(checkpoint_path), 'similarity_dist.npz'
            )
            if os.path.exists(dist_path):
                similarity_dist = SimilarityDistribution(metric=similarity_metric).load(dist_path)
                bayesian_config = {
                    'confidence_level': graph_cfg.get('confidence_level', 0.95),
                    'base_prior': graph_cfg.get('base_prior', 0.01),
                    'density_k': graph_cfg.get('density_k', 50),
                    'density_beta': graph_cfg.get('density_beta', 10.0),
                }

        graph = build_graph_from_keyframes_batch(
            keyframes,
            temporal_neighbors=config['keyframe']['temporal_neighbors'],
            device=device,
            poses=poses,
            descriptors=descriptors,
            similarity_threshold=graph_cfg['similarity_threshold'],
            similarity_max_k=graph_cfg['similarity_max_k'],
            similarity_exclude_temporal=graph_cfg.get('similarity_exclude_temporal', True),
            similarity_dist=similarity_dist,
            similarity_metric=similarity_metric,
            standardization_stats=standardization_stats,
            **bayesian_config,
        )

        # Create and load GNN model
        edge_cfg = gnn_cfg.get('edge_encoding', None)
        gnn = create_spectral_gnn(
            input_dim=gnn_cfg['input_dim'],
            hidden_dim=gnn_cfg['hidden_dim'],
            context_dim=gnn_cfg['context_dim'],
            n_layers=gnn_cfg['n_layers'],
            n_heads=gnn_cfg.get('n_heads', 4),
            dropout=gnn_cfg['dropout'],
            use_local_updates=gnn_cfg.get('use_local_updates', True),
            local_update_hops=gnn_cfg.get('local_update_hops', 3),
            edge_encoder_config=edge_cfg,
        )
        gnn = gnn.to(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        gnn.load_state_dict(state_dict, strict=False)
        gnn.eval()

        with torch.no_grad():
            embeddings = gnn(graph.to(device)).cpu().numpy()

        return embeddings
