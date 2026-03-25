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
    similarity_threshold: float = 0.993,
    similarity_max_k: int = 10,
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
        similarity_threshold: Minimum cosine similarity (fallback mode)
        similarity_max_k: Maximum neighbors per node (safety cap)
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

    # L2 normalize for cosine sim (always needed for edge_attr)
    norms = np.linalg.norm(descs_f32, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    descs_normed = descs_f32 / norms

    # Over-fetch to compensate for temporal exclusion + self
    max_temporal = max((len(s) for s in temporal_neighbor_sets.values()), default=0)
    fetch_k_base = similarity_max_k + max_temporal + 1

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
            logger.info(f"  floor_sim={threshold:.6f}")

        edges = []
        n_multiscale_filtered = 0
        for i in range(n_nodes):
            count = 0
            temporal_set = temporal_neighbor_sets.get(i, set())
            prior_i = float(adaptive_priors[i])

            for j_pos in range(fetch_k):
                j = int(indices[i, j_pos])
                if j == i:
                    continue
                if similarity_exclude_temporal and j in temporal_set:
                    continue

                obs = float(faiss_dists[i, j_pos])

                if use_l2:
                    # L2: ascending order, break when above ceiling
                    if obs > threshold:
                        break
                else:
                    # Cosine: descending order, break when below floor
                    if obs < threshold:
                        break

                post = float(similarity_dist.posterior(obs, prior=prior_i))
                if post < confidence_level:
                    continue

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
                edges.append((i, j, cos_sim, l2, post))

                count += 1
                if count >= similarity_max_k:
                    break

        if use_multiscale and n_multiscale_filtered > 0:
            logger.info(
                f"  Multi-scale consistency filtered {n_multiscale_filtered} candidate edges "
                f"(min_consistency={multiscale_min_consistency:.2f})"
            )

    else:
        # === Fallback: fixed cosine threshold (cosine metric only) ===
        edges = []
        for i in range(n_nodes):
            count = 0
            temporal_set = temporal_neighbor_sets.get(i, set())
            for j_pos in range(fetch_k):
                j = int(indices[i, j_pos])
                if j == i:
                    continue
                if similarity_exclude_temporal and j in temporal_set:
                    continue

                cos_sim = float(faiss_dists[i, j_pos])
                if cos_sim < similarity_threshold:
                    break

                l2 = float(np.linalg.norm(descs_f32[i] - descs_f32[j]))
                edges.append((i, j, cos_sim, l2, 1.0))

                count += 1
                if count >= similarity_max_k:
                    break

    return edges


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

    for i in range(n_nodes):
        for offset in range(-half_window, half_window + 1):
            if offset == 0:
                continue

            neighbor_idx = i + offset

            if 0 <= neighbor_idx < n_nodes:
                edges.append([i, neighbor_idx])
                temporal_neighbor_sets[i].add(neighbor_idx)
                edge_types.append(0)  # temporal

                # Compute pose-based features
                if poses is not None:
                    pos_i = poses[i, :3, 3]
                    pos_j = poses[neighbor_idx, :3, 3]
                    dist = np.linalg.norm(pos_i - pos_j)
                    norm_dist = np.log1p(dist) / 5.0

                    R_i = poses[i, :3, :3]
                    R_j = poses[neighbor_idx, :3, :3]
                    R_rel = R_j @ R_i.T
                    trace_val = np.clip(np.trace(R_rel), -1.0, 3.0)
                    angle_rad = np.arccos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
                    norm_rot = angle_rad / np.pi

                    edge_attrs.append([norm_dist, norm_rot, 0.0, 0.0, 1.0])
                else:
                    edge_attrs.append([0.0, 0.0, 0.0, 0.0, 1.0])

    # 2.5. Add verified loop closure edges (bidirectional, type 0)
    if loop_closures is not None and len(loop_closures) > 0:
        for query_idx, match_idx in loop_closures:
            if 0 <= query_idx < n_nodes and 0 <= match_idx < n_nodes:
                for src, dst in [(query_idx, match_idx), (match_idx, query_idx)]:
                    edges.append([src, dst])
                    temporal_neighbor_sets[src].add(dst)
                    edge_types.append(0)  # temporal

                    if poses is not None:
                        pos_i = poses[query_idx, :3, 3]
                        pos_j = poses[match_idx, :3, 3]
                        dist = np.linalg.norm(pos_i - pos_j)
                        norm_dist = np.log1p(dist) / 5.0

                        R_i = poses[query_idx, :3, :3]
                        R_j = poses[match_idx, :3, :3]
                        R_rel = R_j @ R_i.T
                        trace_val = np.clip(np.trace(R_rel), -1.0, 3.0)
                        angle_rad = np.arccos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0))
                        norm_rot = angle_rad / np.pi

                        edge_attrs.append([norm_dist, norm_rot, 0.0, 0.0, 1.0])
                    else:
                        edge_attrs.append([0.0, 0.0, 0.0, 0.0, 1.0])

    n_temporal_edges = len(edges)

    # 3. Build similarity edges (Bayesian posterior or cosine threshold)
    if descriptors is not None and n_nodes > 1:
        sim_edges = _build_similarity_edges(
            descriptors, temporal_neighbor_sets,
            similarity_threshold, similarity_max_k,
            similarity_exclude_temporal,
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

    # 5. Create graph
    graph = Data(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        num_nodes=n_nodes
    ).to(device)

    return graph
