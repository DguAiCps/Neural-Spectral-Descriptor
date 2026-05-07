"""
Graph Manager for Dual-Edge Keyframe Graph (Temporal + Similarity)

Manages PyTorch Geometric graph structure with:
- Node features: 256D per-elevation spectral histograms (16 elevations × 16 bins)
- Temporal edges (type 0): Connect temporally adjacent keyframes (half-window neighbors)
- Similarity edges (type 1): Bayesian posterior or cosine-threshold neighbors
- Edge features: 5D [dist_norm, rot_norm, cos_sim, l2_dist_norm, posterior]
- Edge type: LongTensor (0=temporal, 1=similarity)
- Sliding window (max 1000 active nodes)
- Local updates (3-hop neighborhoods)

Graph Structure:
    Temporal:   t-7 ← ... ← t-1 ← t → t+1 → ... → t+7 (half-window)
    Similarity: query ←→ Bayesian posterior or cosine threshold neighbors
"""

import logging
import numpy as np
import torch
from collections import defaultdict
from torch_geometric.data import Data
from typing import List, Tuple, Optional, Set
from keyframe.selector import Keyframe

logger = logging.getLogger(__name__)


def _compute_multiscale_consistency(
    descs_normed: np.ndarray,
    i: int,
    j: int,
    channel_splits: List[Tuple[int, int]],
) -> float:
    """Compute multi-scale similarity consistency between two descriptors.

    True same-place matches show consistent high similarity across all
    descriptor sub-bands (mean, std, inter-bin diff channels). Spectrally
    aliased pairs tend to match only in low-order statistics but diverge
    in higher-order ones.

    Args:
        descs_normed: (N, D) L2-normalized descriptors
        i, j: Node indices
        channel_splits: List of (start, end) index pairs for each sub-channel

    Returns:
        Consistency score in [0, 1]. Higher = more consistent across scales.
        Computed as 1 - std(per_channel_cosine_sims).
    """
    if len(channel_splits) <= 1:
        return 1.0

    channel_sims = []
    for start, end in channel_splits:
        ch_i = descs_normed[i, start:end]
        ch_j = descs_normed[j, start:end]
        norm_i = np.linalg.norm(ch_i)
        norm_j = np.linalg.norm(ch_j)
        if norm_i < 1e-8 or norm_j < 1e-8:
            channel_sims.append(0.0)
        else:
            channel_sims.append(float(np.dot(ch_i, ch_j) / (norm_i * norm_j)))

    return float(1.0 - np.std(channel_sims))


def _build_similarity_edges(
    descriptors: np.ndarray,
    temporal_neighbor_sets: dict,
    sequence_ids: np.ndarray = None,
    similarity_threshold: float = 0.993,
    similarity_max_k: int = 10,
    similarity_min_k: int = 0,
    similarity_exclude_temporal: bool = True,
    similarity_dist=None,
    confidence_level: float = 0.95,
    density_k: int = 50,
    density_beta: float = 10.0,
    base_prior: float = 0.01,
    spectral_entropies: np.ndarray = None,
    prior_signal: str = 'density',
    channel_splits: List[Tuple[int, int]] = None,
    multiscale_min_consistency: float = 0.0,
    similarity_metric: str = 'cosine',
    standardization_stats=None,
) -> List[Tuple[int, int, float, float, float]]:
    """
    Build similarity edges using Bayesian posterior or cosine threshold.

    Mode selection:
    - If similarity_dist is provided: Bayesian posterior with adaptive prior
    - Otherwise: Fixed cosine threshold (legacy fallback)

    Metric selection:
    - 'cosine': FAISS IndexFlatIP on L2-normalized descriptors (original)
    - 'l2': FAISS IndexFlatL2 on z-scored descriptors (standardized Euclidean)

    Args:
        descriptors: (n_nodes, D) descriptor matrix
        temporal_neighbor_sets: {node_idx: set of temporal neighbor indices}
        sequence_ids: Optional (n_nodes,) sequence ID per descriptor. When provided,
            similarity candidates are restricted to the same sequence because poses
            and revisit labels are sequence-local.
        similarity_threshold: Minimum cosine similarity (fallback mode)
        similarity_max_k: Maximum neighbors per node (safety cap)
        similarity_min_k: Minimum neighbors per node to keep via relaxed top-k
            fallback. Default 0 preserves the original strict thresholding.
        similarity_exclude_temporal: If True, exclude temporal neighbors
        similarity_dist: SimilarityDistribution instance (enables Bayesian mode)
        confidence_level: Minimum posterior P(same|obs) for edge (Bayesian mode)
        density_k: k for local density estimation via FAISS k-NN
        density_beta: Sensitivity of prior signal → prior mapping
        base_prior: Maximum P(same) prior
        spectral_entropies: (n_nodes,) per-node spectral entropy
        prior_signal: 'density' or 'entropy'
        channel_splits: List of (start, end) tuples for multi-scale consistency check
        multiscale_min_consistency: Minimum consistency score (0=disabled)
        similarity_metric: 'cosine' or 'l2' (standardized Euclidean)
        standardization_stats: StandardizationStats instance (required for metric='l2')

    Returns:
        List of (src, dst, cos_sim, l2_dist, posterior) tuples.
        posterior=1.0 when using fallback threshold mode.
    """
    import faiss

    n_nodes, d = descriptors.shape
    descs_f32 = descriptors.astype(np.float32).copy()
    similarity_max_k = max(0, int(similarity_max_k))
    similarity_min_k = max(0, min(int(similarity_min_k), similarity_max_k))

    # L2 normalize for cosine sim (always needed for edge_attr)
    norms = np.linalg.norm(descs_f32, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    descs_normed = descs_f32 / norms

    # Over-fetch to compensate for temporal exclusion + self
    max_temporal = max((len(s) for s in temporal_neighbor_sets.values()), default=0)
    fetch_k_base = similarity_max_k + max_temporal + 1
    if similarity_min_k > 0:
        fetch_k_base = max(fetch_k_base, similarity_min_k * 4 + max_temporal + 1)

    # Bayesian mode needs more candidates (threshold is adaptive)
    bayesian = similarity_dist is not None and similarity_dist.fitted
    if bayesian:
        fetch_k = min(max(fetch_k_base, density_k + 1), n_nodes)
    else:
        fetch_k = min(fetch_k_base, n_nodes)

    use_l2 = similarity_metric == 'l2'

    if use_l2:
        # Standardized Euclidean: z-score then L2
        if standardization_stats is None or not standardization_stats.fitted:
            raise RuntimeError(
                "standardization_stats required for similarity_metric='l2'"
            )
        descs_std = standardization_stats.transform(descs_f32)
        d_std = descs_std.shape[1]
        index = faiss.IndexFlatL2(d_std)
        index.add(descs_std)
        dists_sq, indices = index.search(descs_std, fetch_k)
        # Convert squared L2 to L2 distance
        faiss_dists = np.sqrt(np.maximum(dists_sq, 0.0))
    else:
        # Cosine similarity via inner product on L2-normalized
        index = faiss.IndexFlatIP(d)
        index.add(descs_normed)
        faiss_dists, indices = index.search(descs_normed, fetch_k)
        # faiss_dists = cosine similarities (descending)

    use_multiscale = (
        channel_splits is not None
        and len(channel_splits) > 1
        and multiscale_min_consistency > 0
    )

    if bayesian:
        # === Bayesian mode: adaptive posterior ===

        # Select prior signal
        use_entropy = (
            prior_signal == 'entropy'
            and spectral_entropies is not None
            and len(spectral_entropies) == n_nodes
        )

        if use_entropy:
            adaptive_priors = similarity_dist.compute_entropy_adaptive_priors(
                spectral_entropies, base_prior=base_prior, beta=density_beta
            )
            logger.info(
                f"Bayesian similarity edges ({similarity_metric}, entropy prior): "
                f"confidence={confidence_level}, "
                f"entropy range=[{spectral_entropies.min():.3f}, {spectral_entropies.max():.3f}], "
                f"prior range=[{adaptive_priors.min():.6f}, {adaptive_priors.max():.6f}]"
            )
        else:
            # Legacy: density-based prior
            if use_l2:
                # For L2, density = mean distance to k-NN (lower = denser)
                # Negate so higher value = higher density (same sign as cosine)
                k_density = min(density_k, fetch_k)
                local_densities = -faiss_dists[:, 1:k_density + 1].mean(axis=1)
            else:
                k_density = min(density_k, fetch_k)
                local_densities = faiss_dists[:, 1:k_density + 1].mean(axis=1)
            adaptive_priors = similarity_dist.compute_adaptive_priors(
                local_densities, base_prior=base_prior, beta=density_beta
            )
            logger.info(
                f"Bayesian similarity edges ({similarity_metric}, density prior): "
                f"confidence={confidence_level}, density_k={density_k}, "
                f"prior range=[{adaptive_priors.min():.6f}, {adaptive_priors.max():.6f}]"
            )

        # Pre-compute threshold for early termination
        threshold = similarity_dist.confidence_threshold(
            confidence_level, prior=base_prior
        )
        if use_l2:
            logger.info(f"  ceiling_l2={threshold:.4f}")
        else:
            threshold = max(threshold, similarity_threshold)
            logger.info(
                f"  floor_sim={threshold:.6f} "
                f"(posterior floor plus cosine floor {similarity_threshold:.6f})"
            )

        edges = []
        n_multiscale_filtered = 0
        n_relaxed_added = 0
        for i in range(n_nodes):
            accepted = []
            relaxed = []
            temporal_set = temporal_neighbor_sets.get(i, set())
            prior_i = float(adaptive_priors[i])

            for j_pos in range(fetch_k):
                j = int(indices[i, j_pos])
                if j == i:
                    continue
                if sequence_ids is not None and sequence_ids[j] != sequence_ids[i]:
                    continue
                if similarity_exclude_temporal and j in temporal_set:
                    continue

                obs = float(faiss_dists[i, j_pos])
                below_strict_floor = False

                if use_l2:
                    # L2: ascending order, break when above ceiling
                    if obs > threshold:
                        below_strict_floor = True
                else:
                    # Cosine: descending order, break when below floor
                    if obs < threshold:
                        below_strict_floor = True

                post = float(similarity_dist.posterior(obs, prior=prior_i))
                posterior_ok = post >= confidence_level

                # Multi-scale consistency check
                if use_multiscale:
                    consistency = _compute_multiscale_consistency(
                        descs_normed, i, j, channel_splits
                    )
                    if consistency < multiscale_min_consistency:
                        n_multiscale_filtered += 1
                        continue

                # Edge features always from original descriptor space
                cos_sim = float(np.dot(descs_normed[i], descs_normed[j]))
                l2 = float(np.linalg.norm(descs_f32[i] - descs_f32[j]))
                candidate = (i, j, cos_sim, l2, post)

                if not below_strict_floor and posterior_ok:
                    accepted.append(candidate)
                    if len(accepted) >= similarity_max_k:
                        break
                elif similarity_min_k > 0 and posterior_ok:
                    relaxed.append(candidate)

                if (
                    similarity_min_k > 0
                    and len(accepted) < similarity_min_k
                    and len(accepted) + len(relaxed) >= similarity_max_k
                ):
                    break

                if below_strict_floor and similarity_min_k == 0:
                    break

            if len(accepted) < similarity_min_k and relaxed:
                need = min(similarity_min_k - len(accepted), similarity_max_k - len(accepted))
                accepted.extend(relaxed[:need])
                n_relaxed_added += need

            edges.extend(accepted[:similarity_max_k])

        if use_multiscale and n_multiscale_filtered > 0:
            logger.info(
                f"  Multi-scale consistency filtered {n_multiscale_filtered} candidate edges "
                f"(min_consistency={multiscale_min_consistency:.2f})"
            )
        if n_relaxed_added > 0:
            logger.info(
                f"  Relaxed similarity min-k added {n_relaxed_added} edges "
                f"(min_k={similarity_min_k}, max_k={similarity_max_k})"
            )

    else:
        # === Fallback: fixed cosine threshold (cosine metric only) ===
        edges = []
        n_relaxed_added = 0
        for i in range(n_nodes):
            accepted = []
            relaxed = []
            temporal_set = temporal_neighbor_sets.get(i, set())
            for j_pos in range(fetch_k):
                j = int(indices[i, j_pos])
                if j == i:
                    continue
                if sequence_ids is not None and sequence_ids[j] != sequence_ids[i]:
                    continue
                if similarity_exclude_temporal and j in temporal_set:
                    continue

                cos_sim = float(faiss_dists[i, j_pos])
                if cos_sim < similarity_threshold:
                    if similarity_min_k == 0:
                        break
                    l2 = float(np.linalg.norm(descs_f32[i] - descs_f32[j]))
                    relaxed.append((i, j, cos_sim, l2, 1.0))
                    if len(accepted) + len(relaxed) >= similarity_max_k:
                        break
                    continue

                l2 = float(np.linalg.norm(descs_f32[i] - descs_f32[j]))
                accepted.append((i, j, cos_sim, l2, 1.0))

                if len(accepted) >= similarity_max_k:
                    break

            if len(accepted) < similarity_min_k and relaxed:
                need = min(similarity_min_k - len(accepted), similarity_max_k - len(accepted))
                accepted.extend(relaxed[:need])
                n_relaxed_added += need
            edges.extend(accepted[:similarity_max_k])

        if n_relaxed_added > 0:
            logger.info(
                f"  Relaxed similarity min-k added {n_relaxed_added} edges "
                f"(min_k={similarity_min_k}, max_k={similarity_max_k})"
            )

    return edges


def build_similarity_edges_from_poses(
    poses: np.ndarray,
    descriptors: np.ndarray,
    temporal_neighbor_sets: dict,
    sequence_ids: np.ndarray = None,
    pos_dist: float = 5.0,
    min_temporal_gap: int = 30,
    similarity_max_k: int = 10,
) -> List[Tuple[int, int, float, float, float]]:
    """Ground-truth pose-based similarity edges ("same-place" by definition).

    For each node i, select up to `similarity_max_k` nearest neighbors j
    satisfying: pose_dist(i,j) < pos_dist, |i-j| >= min_temporal_gap,
    and j not already a temporal neighbor. Cross-sequence pairs are excluded
    when `sequence_ids` is provided (poses live in sequence-local frames).

    Returns the same 5-tuple format as _build_similarity_edges:
    (src, dst, cos_sim, l2, posterior=1.0). Descriptor-based cos_sim/l2 are
    computed so the edge encoder sees consistent features.
    """
    from scipy.spatial import cKDTree

    positions = poses[:, :3, 3].astype(np.float32)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    descs_normed = (descriptors / norms).astype(np.float32)

    def _select_within_segment(segment_indices: np.ndarray) -> None:
        """Build KD-tree on segment_indices and append edges for each node."""
        if len(segment_indices) < 2:
            return
        seg_positions = positions[segment_indices]
        tree = cKDTree(seg_positions)
        for local_i, global_i in enumerate(segment_indices):
            neighbor_locals = tree.query_ball_point(seg_positions[local_i], pos_dist)
            temporal_set = temporal_neighbor_sets.get(int(global_i), set())
            candidates: List[Tuple[float, int]] = []
            for local_j in neighbor_locals:
                global_j = int(segment_indices[local_j])
                if global_i == global_j:
                    continue
                if abs(global_i - global_j) < min_temporal_gap:
                    continue
                if global_j in temporal_set:
                    continue
                d = float(np.linalg.norm(positions[global_i] - positions[global_j]))
                candidates.append((d, global_j))
            candidates.sort()
            for d, j in candidates[:similarity_max_k]:
                cos_sim = float(np.dot(descs_normed[int(global_i)], descs_normed[j]))
                l2 = float(np.linalg.norm(descriptors[int(global_i)] - descriptors[j]))
                edges.append((int(global_i), j, cos_sim, l2, 1.0))

    edges: List[Tuple[int, int, float, float, float]] = []
    if sequence_ids is not None:
        for seq_id in np.unique(sequence_ids):
            seg = np.where(sequence_ids == seq_id)[0]
            _select_within_segment(seg)
    else:
        _select_within_segment(np.arange(len(positions)))

    return edges


def attach_pose_gt_similarity_edges(
    graph: Data,
    poses: np.ndarray,
    descriptors: np.ndarray,
    sequence_ids: np.ndarray = None,
    pos_dist: float = 5.0,
    min_temporal_gap: int = 30,
    similarity_max_k: int = 10,
) -> Tuple[Data, int]:
    """Replace the graph's similarity edges (type=1) with ground-truth
    pose-based "same-place" edges. Temporal edges (type=0) are preserved.

    This is a train-time supervision signal: poses are available during
    training, so the GNN sees perfectly clean same-place message-passing
    edges. At inference we do NOT add these edges; val/test graphs run with
    temporal-only structure.

    Returns (graph, n_similarity_edges). Graph is modified in place.
    """
    device = graph.edge_index.device

    # Reconstruct temporal_neighbor_sets from preserved edges (type 0)
    temporal_mask = (graph.edge_type == 0).cpu().numpy()
    ei_cpu = graph.edge_index.cpu().numpy()
    attr_cpu = graph.edge_attr.cpu().numpy()
    type_cpu = graph.edge_type.cpu().numpy()

    temporal_ei = ei_cpu[:, temporal_mask]
    temporal_neighbor_sets = defaultdict(set)
    for k in range(temporal_ei.shape[1]):
        src, dst = int(temporal_ei[0, k]), int(temporal_ei[1, k])
        temporal_neighbor_sets[src].add(dst)
        temporal_neighbor_sets[dst].add(src)

    # Build GT pose edges
    sim_edges = build_similarity_edges_from_poses(
        poses, descriptors, temporal_neighbor_sets,
        sequence_ids=sequence_ids,
        pos_dist=pos_dist,
        min_temporal_gap=min_temporal_gap,
        similarity_max_k=similarity_max_k,
    )

    # Combine preserved temporal + new GT similarity edges
    kept_ei = temporal_ei
    kept_attr = attr_cpu[temporal_mask]
    kept_type = type_cpu[temporal_mask]

    if len(sim_edges) > 0:
        sim_ei = np.empty((2, len(sim_edges)), dtype=np.int64)
        sim_attr = np.empty((len(sim_edges), 5), dtype=np.float32)
        sim_type = np.ones(len(sim_edges), dtype=np.int64)
        for i, (src, dst, cos_sim, l2, posterior) in enumerate(sim_edges):
            sim_ei[0, i] = src
            sim_ei[1, i] = dst
            l2_norm = np.log1p(l2) / 5.0
            sim_attr[i] = [0.0, 0.0, cos_sim, l2_norm, posterior]

        all_ei = np.concatenate([kept_ei, sim_ei], axis=1)
        all_attr = np.concatenate([kept_attr, sim_attr], axis=0)
        all_type = np.concatenate([kept_type, sim_type], axis=0)
    else:
        all_ei = kept_ei
        all_attr = kept_attr
        all_type = kept_type

    graph.edge_index = torch.from_numpy(all_ei).long().to(device)
    graph.edge_attr = torch.from_numpy(all_attr).float().to(device)
    graph.edge_type = torch.from_numpy(all_type).long().to(device)

    return graph, len(sim_edges)


def rebuild_similarity_edges(
    graph: Data,
    new_descriptors: np.ndarray,
    similarity_threshold: float = 0.993,
    similarity_max_k: int = 10,
    similarity_min_k: int = 0,
    similarity_exclude_temporal: bool = True,
    similarity_dist=None,
    confidence_level: float = 0.95,
    density_k: int = 50,
    density_beta: float = 10.0,
    base_prior: float = 0.01,
    spectral_entropies: np.ndarray = None,
    prior_signal: str = 'density',
    channel_splits: List[Tuple[int, int]] = None,
    multiscale_min_consistency: float = 0.0,
    similarity_metric: str = 'l2',
    standardization_stats=None,
    sequence_ids: np.ndarray = None,
) -> Tuple[Data, int]:
    """Replace similarity edges (edge_type=1) on a graph with newly computed ones.

    Used by two-pass refinement: after training begins, call this periodically
    with embeddings from the current GNN (typically the ctx portion) to rebuild
    similarity edges in a discriminative space.

    Temporal edges (edge_type=0) are preserved unchanged.

    Args:
        graph: Existing PyG Data with edge_index, edge_attr (N, 5), edge_type.
        new_descriptors: (N, D) descriptor matrix to build similarity edges from.
            Typically GNN ctx output of shape (N, context_dim).
        Remaining kwargs: forwarded to _build_similarity_edges.

    Returns:
        (graph, n_similarity_edges) — graph is modified in place.
    """
    device = graph.edge_index.device
    n_nodes = graph.num_nodes
    if sequence_ids is None and hasattr(graph, 'sequence_ids'):
        sequence_ids = graph.sequence_ids.detach().cpu().numpy()

    # Reconstruct temporal_neighbor_sets from preserved edges (type 0)
    temporal_mask = (graph.edge_type == 0).cpu().numpy()
    ei_cpu = graph.edge_index.cpu().numpy()
    attr_cpu = graph.edge_attr.cpu().numpy()
    type_cpu = graph.edge_type.cpu().numpy()

    temporal_ei = ei_cpu[:, temporal_mask]  # (2, n_temporal)
    temporal_neighbor_sets = defaultdict(set)
    for k in range(temporal_ei.shape[1]):
        src, dst = int(temporal_ei[0, k]), int(temporal_ei[1, k])
        temporal_neighbor_sets[src].add(dst)
        temporal_neighbor_sets[dst].add(src)

    # Build new similarity edges via existing helper
    sim_edges = _build_similarity_edges(
        new_descriptors, temporal_neighbor_sets,
        sequence_ids=sequence_ids,
        similarity_threshold=similarity_threshold,
        similarity_max_k=similarity_max_k,
        similarity_min_k=similarity_min_k,
        similarity_exclude_temporal=similarity_exclude_temporal,
        similarity_dist=similarity_dist,
        confidence_level=confidence_level,
        density_k=density_k,
        density_beta=density_beta,
        base_prior=base_prior,
        spectral_entropies=spectral_entropies,
        prior_signal=prior_signal,
        channel_splits=channel_splits,
        multiscale_min_consistency=multiscale_min_consistency,
        similarity_metric=similarity_metric,
        standardization_stats=standardization_stats,
    )

    # Combine preserved temporal + new similarity
    kept_ei = temporal_ei
    kept_attr = attr_cpu[temporal_mask]
    kept_type = type_cpu[temporal_mask]

    if len(sim_edges) > 0:
        sim_ei = np.empty((2, len(sim_edges)), dtype=np.int64)
        sim_attr = np.empty((len(sim_edges), 5), dtype=np.float32)
        sim_type = np.ones(len(sim_edges), dtype=np.int64)
        for i, (src, dst, cos_sim, l2, posterior) in enumerate(sim_edges):
            sim_ei[0, i] = src
            sim_ei[1, i] = dst
            l2_norm = np.log1p(l2) / 5.0
            sim_attr[i] = [0.0, 0.0, cos_sim, l2_norm, posterior]

        all_ei = np.concatenate([kept_ei, sim_ei], axis=1)
        all_attr = np.concatenate([kept_attr, sim_attr], axis=0)
        all_type = np.concatenate([kept_type, sim_type], axis=0)
    else:
        all_ei = kept_ei
        all_attr = kept_attr
        all_type = kept_type

    graph.edge_index = torch.from_numpy(all_ei).long().to(device)
    graph.edge_attr = torch.from_numpy(all_attr).float().to(device)
    graph.edge_type = torch.from_numpy(all_type).long().to(device)

    return graph, len(sim_edges)


class TemporalGraphManager:
    """
    Manages temporal graph of keyframes for GNN processing

    Creates and maintains a PyTorch Geometric graph where:
    - Nodes are keyframes with spectral histogram features
    - Edges connect temporally adjacent keyframes (M=5 neighbors)
    - Sliding window maintains max 1000 active nodes
    - Old nodes are frozen (embeddings cached, removed from active computation)
    """

    def __init__(
        self,
        temporal_neighbors: int = 5,
        max_active_nodes: int = 1000,
        feature_dim: int = 256,
        device: str = 'cpu'
    ):
        """
        Initialize graph manager

        Args:
            temporal_neighbors: Number of temporal neighbors (M=5)
            max_active_nodes: Maximum active nodes in sliding window
            feature_dim: Feature dimension (256 = 16 elevations × 16 bins)
            device: Device for PyTorch tensors
        """
        self.temporal_neighbors = temporal_neighbors
        self.max_active_nodes = max_active_nodes
        self.feature_dim = feature_dim
        self.device = device

        # Active graph
        self.graph: Optional[Data] = None
        self.keyframes: List[Keyframe] = []

        # Frozen nodes (beyond sliding window)
        self.frozen_keyframes: List[Keyframe] = []
        self.frozen_embeddings: Optional[torch.Tensor] = None

        # Node index mapping
        self.keyframe_id_to_node_idx = {}

    def reset(self):
        """Reset graph state"""
        self.graph = None
        self.keyframes.clear()
        self.frozen_keyframes.clear()
        self.frozen_embeddings = None
        self.keyframe_id_to_node_idx.clear()

    def add_keyframe(self, keyframe: Keyframe) -> int:
        """
        Add new keyframe to graph

        Args:
            keyframe: Keyframe to add (must have descriptor set)

        Returns:
            Node index in active graph
        """
        if keyframe.descriptor is None:
            raise ValueError("Keyframe must have descriptor computed before adding to graph")

        # Add to keyframes list
        self.keyframes.append(keyframe)
        node_idx = len(self.keyframes) - 1

        # Update mapping
        self.keyframe_id_to_node_idx[keyframe.keyframe_id] = node_idx

        # Rebuild graph
        self._rebuild_graph()

        # Check sliding window constraint
        if len(self.keyframes) > self.max_active_nodes:
            self._freeze_oldest_node()

        return node_idx

    def _rebuild_graph(self):
        """
        Rebuild PyTorch Geometric graph from current keyframes
        """
        n_nodes = len(self.keyframes)

        if n_nodes == 0:
            self.graph = None
            return

        # Extract features (descriptors)
        features = torch.stack([
            torch.from_numpy(kf.descriptor).float()
            for kf in self.keyframes
        ], dim=0).to(self.device)  # (n_nodes, feature_dim)

        # Build temporal edges (M nearest neighbors)
        edge_index = self._build_temporal_edges(n_nodes)

        # Create PyG Data object
        self.graph = Data(
            x=features,
            edge_index=edge_index,
            num_nodes=n_nodes
        ).to(self.device)

    def _build_temporal_edges(self, n_nodes: int) -> torch.Tensor:
        """
        Build temporal edge connections (M=5 nearest neighbors in time)

        Args:
            n_nodes: Number of nodes

        Returns:
            (2, num_edges) edge index tensor
        """
        edges = []

        for i in range(n_nodes):
            # Connect to M/2 past neighbors and M/2 future neighbors
            half_window = self.temporal_neighbors // 2

            for offset in range(-half_window, half_window + 1):
                if offset == 0:
                    continue

                neighbor_idx = i + offset

                # Check bounds
                if 0 <= neighbor_idx < n_nodes:
                    # Bidirectional edge
                    edges.append([i, neighbor_idx])

        if len(edges) == 0:
            # No edges (single node)
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        # Convert to tensor
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()

        return edge_index

    def _freeze_oldest_node(self):
        """
        Freeze oldest node and move to frozen storage

        Freezing means:
        - Remove from active graph
        - Cache embedding (if computed)
        - Keep descriptor for retrieval
        """
        if len(self.keyframes) == 0:
            return

        # Remove oldest keyframe
        oldest_kf = self.keyframes.pop(0)
        self.frozen_keyframes.append(oldest_kf)

        # Update node index mapping
        del self.keyframe_id_to_node_idx[oldest_kf.keyframe_id]

        # Shift all indices down by 1
        for kf_id in self.keyframe_id_to_node_idx:
            self.keyframe_id_to_node_idx[kf_id] -= 1

        # Cache embedding if available
        if oldest_kf.embedding is not None:
            embedding = torch.from_numpy(oldest_kf.embedding).float().to(self.device)

            if self.frozen_embeddings is None:
                self.frozen_embeddings = embedding.unsqueeze(0)
            else:
                self.frozen_embeddings = torch.cat([
                    self.frozen_embeddings,
                    embedding.unsqueeze(0)
                ], dim=0)

        # Rebuild graph without oldest node
        self._rebuild_graph()

    def get_graph(self) -> Optional[Data]:
        """Get current active graph"""
        return self.graph

    def add_loop_closure_edge(
        self,
        query_keyframe_id: int,
        match_keyframe_id: int,
        pose_query: np.ndarray = None,
        pose_match: np.ndarray = None
    ) -> bool:
        """
        Add verified loop closure edge to graph (bidirectional).

        Called after geometric verification confirms a loop closure.
        This connects spatially related keyframes that may be temporally distant.

        Args:
            query_keyframe_id: Query keyframe ID
            match_keyframe_id: Matched keyframe ID
            pose_query: Optional (4, 4) SE(3) pose of query
            pose_match: Optional (4, 4) SE(3) pose of match

        Returns:
            True if edge was added, False if keyframes not in active graph
        """
        query_idx = self.keyframe_id_to_node_idx.get(query_keyframe_id)
        match_idx = self.keyframe_id_to_node_idx.get(match_keyframe_id)

        if query_idx is None or match_idx is None:
            return False

        if self.graph is None:
            return False

        # Create new edges (bidirectional)
        new_edges = torch.tensor(
            [[query_idx, match_idx], [match_idx, query_idx]],
            dtype=torch.long,
            device=self.device
        ).t()

        # Compute edge features if poses available
        new_edge_attr = None
        if pose_query is not None and pose_match is not None and self.graph.edge_attr is not None:
            # Distance
            dist = np.linalg.norm(pose_query[:3, 3] - pose_match[:3, 3])
            norm_dist = np.log1p(dist) / 5.0

            # Rotation
            R_rel = pose_match[:3, :3] @ pose_query[:3, :3].T
            trace_val = np.clip(np.trace(R_rel), -1.0, 3.0)
            angle_rad = np.arccos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
            norm_rot = angle_rad / np.pi

            # Create edge attr for both directions (5D: temporal format)
            new_edge_attr = torch.tensor(
                [[norm_dist, norm_rot, 0.0, 0.0, 1.0],
                 [norm_dist, norm_rot, 0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=self.device
            )

        # Append to existing edges
        self.graph.edge_index = torch.cat([self.graph.edge_index, new_edges], dim=1)

        if new_edge_attr is not None and self.graph.edge_attr is not None:
            self.graph.edge_attr = torch.cat([self.graph.edge_attr, new_edge_attr], dim=0)

        # Append edge types (temporal = 0 for loop closure edges since they have pose info)
        if hasattr(self.graph, 'edge_type') and self.graph.edge_type is not None:
            new_edge_type = torch.zeros(2, dtype=torch.long, device=self.device)
            self.graph.edge_type = torch.cat([self.graph.edge_type, new_edge_type])

        return True

    def get_node_index(self, keyframe_id: int) -> Optional[int]:
        """
        Get node index for keyframe ID in active graph

        Args:
            keyframe_id: Keyframe ID

        Returns:
            Node index or None if not in active graph
        """
        return self.keyframe_id_to_node_idx.get(keyframe_id, None)

    def get_k_hop_neighbors(self, node_idx: int, k: int) -> Set[int]:
        """
        Get k-hop neighborhood of a node

        Args:
            node_idx: Node index
            k: Number of hops

        Returns:
            Set of node indices in k-hop neighborhood
        """
        if self.graph is None or k <= 0:
            return {node_idx}

        # BFS to find k-hop neighbors
        neighbors = {node_idx}
        current_layer = {node_idx}

        edge_index = self.graph.edge_index.cpu().numpy()

        for _ in range(k):
            next_layer = set()

            for node in current_layer:
                # Find neighbors
                outgoing = edge_index[1, edge_index[0] == node]
                next_layer.update(outgoing.tolist())

            neighbors.update(next_layer)
            current_layer = next_layer

            if len(current_layer) == 0:
                break

        return neighbors

    def get_local_subgraph(self, node_idx: int, k_hops: int = 3) -> Tuple[Data, dict]:
        """
        Extract k-hop local subgraph around a node

        Args:
            node_idx: Center node index
            k_hops: Number of hops (default: 3)

        Returns:
            subgraph: Local subgraph as PyG Data
            mapping: Mapping from original to subgraph indices
        """
        if self.graph is None:
            raise ValueError("Graph is empty")

        # Get k-hop neighbors
        neighbor_indices = sorted(list(self.get_k_hop_neighbors(node_idx, k_hops)))

        # Create mapping
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(neighbor_indices)}

        # Extract subgraph features
        subgraph_features = self.graph.x[neighbor_indices]

        # Extract subgraph edges
        edge_index = self.graph.edge_index.cpu().numpy()
        subgraph_edges = []

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]

            if src in neighbor_indices and dst in neighbor_indices:
                # Remap to subgraph indices
                new_src = mapping[src]
                new_dst = mapping[dst]
                subgraph_edges.append([new_src, new_dst])

        if len(subgraph_edges) > 0:
            subgraph_edge_index = torch.tensor(
                subgraph_edges,
                dtype=torch.long,
                device=self.device
            ).t()
        else:
            subgraph_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        # Create subgraph
        subgraph = Data(
            x=subgraph_features,
            edge_index=subgraph_edge_index,
            num_nodes=len(neighbor_indices)
        ).to(self.device)

        return subgraph, mapping

    def update_embeddings(self, embeddings: torch.Tensor):
        """
        Update keyframe embeddings from GNN output

        Args:
            embeddings: (n_nodes, embedding_dim) GNN embeddings
        """
        if len(embeddings) != len(self.keyframes):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) != keyframe count ({len(self.keyframes)})"
            )

        # Update keyframe embeddings
        embeddings_np = embeddings.detach().cpu().numpy()

        for i, kf in enumerate(self.keyframes):
            kf.embedding = embeddings_np[i]

    def get_all_keyframes(self) -> List[Keyframe]:
        """Get all keyframes (active + frozen)"""
        return self.frozen_keyframes + self.keyframes

    def get_all_descriptors(self) -> np.ndarray:
        """
        Get all descriptors (active + frozen)

        Returns:
            (total_keyframes, feature_dim) array
        """
        all_kfs = self.get_all_keyframes()

        descriptors = np.array([kf.descriptor for kf in all_kfs])

        return descriptors

    def get_all_embeddings(self) -> Optional[np.ndarray]:
        """
        Get all embeddings (active + frozen)

        Returns:
            (total_keyframes, embedding_dim) array or None
        """
        all_kfs = self.get_all_keyframes()

        if all_kfs[0].embedding is None:
            return None

        embeddings = np.array([kf.embedding for kf in all_kfs])

        return embeddings

    def get_statistics(self) -> dict:
        """Get graph statistics"""
        return {
            'num_active_nodes': len(self.keyframes),
            'num_frozen_nodes': len(self.frozen_keyframes),
            'total_nodes': len(self.keyframes) + len(self.frozen_keyframes),
            'num_edges': self.graph.edge_index.shape[1] if self.graph is not None else 0,
            'avg_degree': (
                self.graph.edge_index.shape[1] / len(self.keyframes)
                if self.graph is not None and len(self.keyframes) > 0
                else 0.0
            )
        }


def build_graph_from_keyframes(
    keyframes: List[Keyframe],
    temporal_neighbors: int = 5,
    device: str = 'cpu'
) -> Data:
    """
    Build PyTorch Geometric graph from keyframe list

    Args:
        keyframes: List of keyframes with descriptors
        temporal_neighbors: Number of temporal neighbors
        device: Device for tensors

    Returns:
        PyG Data object
    """
    manager = TemporalGraphManager(
        temporal_neighbors=temporal_neighbors,
        max_active_nodes=len(keyframes),  # No freezing for offline construction
        device=device
    )

    for kf in keyframes:
        manager.add_keyframe(kf)

    return manager.get_graph()


def build_graph_from_keyframes_batch(
    keyframes: List[Keyframe],
    temporal_neighbors: int = 5,
    device: str = 'cpu',
    poses: np.ndarray = None,
    loop_closures: List[Tuple[int, int]] = None,
    descriptors: np.ndarray = None,
    similarity_threshold: float = 0.993,
    similarity_max_k: int = 10,
    similarity_min_k: int = 0,
    similarity_exclude_temporal: bool = True,
    similarity_dist=None,
    confidence_level: float = 0.95,
    density_k: int = 50,
    density_beta: float = 10.0,
    base_prior: float = 0.01,
    spectral_entropies: np.ndarray = None,
    prior_signal: str = 'density',
    channel_splits: List[Tuple[int, int]] = None,
    multiscale_min_consistency: float = 0.0,
    similarity_metric: str = 'cosine',
    standardization_stats=None,
    sequence_ids: np.ndarray = None,
    temporal_edge_mode: str = 'bidirectional',
    temporal_direction_mode: str = 'none',
) -> Data:
    """
    O(n) batch graph construction with temporal + similarity edges.

    Builds dual-edge-type graph:
    - Temporal edges (type 0): half-window temporal neighbors with pose-based features
    - Similarity edges (type 1): Bayesian posterior or cosine threshold descriptor edges

    Args:
        keyframes: List of keyframes with descriptors
        temporal_neighbors: Number of temporal neighbors (M)
        device: Device for tensors ('cuda' or 'cpu')
        poses: Optional (n_keyframes, 4, 4) SE(3) poses for temporal edge features
        loop_closures: Optional list of (query_idx, match_idx) verified loop closure pairs
        descriptors: (n_keyframes, D) d_local descriptors for similarity edges
        similarity_threshold: Minimum cosine similarity (fallback mode)
        similarity_max_k: Maximum neighbors per node (safety cap)
        similarity_min_k: Minimum neighbors per node via relaxed top-k fallback
            (0 disables; default preserves existing behavior)
        similarity_exclude_temporal: Exclude temporal neighbors from similarity results
        similarity_dist: SimilarityDistribution instance (enables Bayesian mode)
        confidence_level: Minimum posterior for edge (Bayesian mode)
        density_k: k for local density estimation
        density_beta: Prior signal → prior sensitivity
        base_prior: Maximum P(same) prior
        spectral_entropies: (n_keyframes,) per-node spectral entropy (for prior_signal='entropy')
        prior_signal: 'density' or 'entropy' — which signal drives the adaptive prior
        channel_splits: List of (start, end) tuples for multi-scale consistency filtering
        multiscale_min_consistency: Minimum consistency score to accept a similarity edge (0=disabled)
        sequence_ids: Optional (n_keyframes,) sequence ID per keyframe. When provided,
            temporal and similarity edges never cross sequence boundaries.
        temporal_edge_mode: 'bidirectional' (legacy), 'past_to_current', or
            'future_to_current'. Causal modes are used for test-time forward/reverse
            graph ablations.
        temporal_direction_mode: 'none' or 'signed_distance'. The latter keeps
            edge_attr 5D/checkpoint-compatible but signs temporal dist by edge
            direction (dst index - src index).

    Returns:
        PyG Data object with:
            x: (n_nodes, D) features
            edge_index: (2, n_edges) connectivity
            edge_attr: (n_edges, 5) [dist_norm, rot_norm, cos_sim, l2_dist_norm, posterior]
            edge_type: (n_edges,) LongTensor (0=temporal, 1=similarity)
    """
    import torch
    from torch_geometric.data import Data

    n_nodes = len(keyframes)

    if n_nodes == 0:
        return None

    valid_temporal_modes = {'bidirectional', 'past_to_current', 'future_to_current'}
    if temporal_edge_mode not in valid_temporal_modes:
        raise ValueError(
            f"Unknown temporal_edge_mode={temporal_edge_mode!r}; "
            f"expected one of {sorted(valid_temporal_modes)}"
        )
    valid_direction_modes = {'none', 'signed_distance'}
    if temporal_direction_mode not in valid_direction_modes:
        raise ValueError(
            f"Unknown temporal_direction_mode={temporal_direction_mode!r}; "
            f"expected one of {sorted(valid_direction_modes)}"
        )

    # 1. Extract all features at once - O(n)
    features = torch.stack([
        torch.from_numpy(kf.descriptor).float()
        for kf in keyframes
    ], dim=0).to(device)

    # 2. Build temporal edges - O(n * M)
    edges = []
    edge_attrs = []  # 5D: [dist_norm, rot_norm, cos_sim, l2_dist_norm, posterior]
    edge_types = []
    temporal_neighbor_sets = defaultdict(set)

    M = temporal_neighbors
    half_window = M // 2

    def _append_temporal_edge(src: int, dst: int) -> None:
        edges.append([src, dst])
        edge_types.append(0)  # temporal

        # Compute pose-based features
        if poses is not None:
            pos_src = poses[src, :3, 3]
            pos_dst = poses[dst, :3, 3]
            dist = np.linalg.norm(pos_src - pos_dst)
            norm_dist = np.log1p(dist) / 5.0
            if temporal_direction_mode == 'signed_distance':
                norm_dist *= float(np.sign(dst - src))

            R_src = poses[src, :3, :3]
            R_dst = poses[dst, :3, :3]
            R_rel = R_dst @ R_src.T
            trace_val = np.clip(np.trace(R_rel), -1.0, 3.0)
            angle_rad = np.arccos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
            norm_rot = angle_rad / np.pi

            edge_attrs.append([norm_dist, norm_rot, 0.0, 0.0, 1.0])
        else:
            edge_attrs.append([0.0, 0.0, 0.0, 0.0, 1.0])

    for i in range(n_nodes):
        for offset in range(-half_window, half_window + 1):
            if offset == 0:
                continue

            neighbor_idx = i + offset

            if 0 <= neighbor_idx < n_nodes:
                if sequence_ids is not None and sequence_ids[neighbor_idx] != sequence_ids[i]:
                    continue
                # Similarity exclusion remains symmetric even when message
                # passing is causal; immediate temporal neighbors should not
                # re-enter as descriptor-similarity edges.
                temporal_neighbor_sets[i].add(neighbor_idx)
                temporal_neighbor_sets[neighbor_idx].add(i)

                if temporal_edge_mode == 'bidirectional':
                    _append_temporal_edge(i, neighbor_idx)
                elif temporal_edge_mode == 'past_to_current' and neighbor_idx < i:
                    _append_temporal_edge(neighbor_idx, i)
                elif temporal_edge_mode == 'future_to_current' and neighbor_idx > i:
                    _append_temporal_edge(neighbor_idx, i)

    # 2.5. Add verified loop closure edges (bidirectional, type 0)
    if loop_closures is not None and len(loop_closures) > 0:
        for query_idx, match_idx in loop_closures:
            if 0 <= query_idx < n_nodes and 0 <= match_idx < n_nodes:
                for src, dst in [(query_idx, match_idx), (match_idx, query_idx)]:
                    temporal_neighbor_sets[src].add(dst)
                    temporal_neighbor_sets[dst].add(src)
                    _append_temporal_edge(src, dst)

    n_temporal_edges = len(edges)

    # 3. Build similarity edges (Bayesian posterior or cosine threshold)
    if descriptors is not None and n_nodes > 1:
        sim_edges = _build_similarity_edges(
            descriptors,
            temporal_neighbor_sets,
            sequence_ids=sequence_ids,
            similarity_threshold=similarity_threshold,
            similarity_max_k=similarity_max_k,
            similarity_min_k=similarity_min_k,
            similarity_exclude_temporal=similarity_exclude_temporal,
            similarity_dist=similarity_dist,
            confidence_level=confidence_level,
            density_k=density_k,
            density_beta=density_beta,
            base_prior=base_prior,
            spectral_entropies=spectral_entropies,
            prior_signal=prior_signal,
            channel_splits=channel_splits,
            multiscale_min_consistency=multiscale_min_consistency,
            similarity_metric=similarity_metric,
            standardization_stats=standardization_stats,
        )
        for src, dst, cos_sim, l2, posterior in sim_edges:
            edges.append([src, dst])
            l2_norm = np.log1p(l2) / 5.0
            edge_attrs.append([0.0, 0.0, cos_sim, l2_norm, posterior])
            edge_types.append(1)  # similarity

    n_similarity_edges = len(edges) - n_temporal_edges

    # 4. Convert to tensors
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).to(device)
        edge_type = torch.tensor(edge_types, dtype=torch.long).to(device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 5), dtype=torch.float32, device=device)
        edge_type = torch.empty(0, dtype=torch.long, device=device)

    # 4.5 Compute pose-GT labels for each edge (used by EdgeConfidenceGate aux loss).
    # label = 1.0 if endpoints are same-place (pose_dist < pos_dist threshold), else 0.0.
    # Temporal edges always 1.0 (always informative). Requires poses.
    if poses is not None and len(edges) > 0:
        edges_arr = np.array(edges, dtype=np.int64)
        positions = poses[:, :3, 3]
        pose_dists = np.linalg.norm(
            positions[edges_arr[:, 0]] - positions[edges_arr[:, 1]], axis=1
        )
        # Same-place threshold = 5m (matches positive_distance_max)
        labels = (pose_dists < 5.0).astype(np.float32)
        # Temporal edges (type=0): force label=1.0
        types_arr = np.array(edge_types, dtype=np.int64)
        labels[types_arr == 0] = 1.0
        edge_pose_label = torch.from_numpy(labels).to(device)
    else:
        edge_pose_label = torch.empty(0, dtype=torch.float32, device=device)

    # 5. Create graph
    graph = Data(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        num_nodes=n_nodes
    ).to(device)
    # Attach pose-GT labels separately (PyG Data supports arbitrary tensor attrs)
    graph.edge_pose_label = edge_pose_label
    if sequence_ids is not None:
        graph.sequence_ids = torch.from_numpy(np.asarray(sequence_ids)).long().to(device)
    graph.temporal_edge_mode = temporal_edge_mode
    graph.temporal_direction_mode = temporal_direction_mode

    return graph
