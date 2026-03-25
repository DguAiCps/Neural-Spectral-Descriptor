"""
Evaluation utilities for baseline comparison.

compute_recall_multi_k is adapted from src/gnn/trainer.py:_compute_recall_multi_k
with identical logic: KD-Tree spatial queries + FAISS cosine similarity search.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_recall_multi_k(
    embeddings: np.ndarray,
    poses: np.ndarray,
    k_values: List[int] = [1, 5, 10],
    distance_threshold: float = 5.0,
    skip_frames: int = 30,
) -> Tuple[Dict[int, float], int]:
    """
    Compute Recall@K for multiple K values.

    Only evaluates on "revisit" queries: frames that return within
    distance_threshold of a previously visited location, with at least
    skip_frames temporal gap.

    Args:
        embeddings: (n, D) descriptors (will be L2-normalized internally)
        poses: (n, 4, 4) SE(3) poses
        k_values: List of K values for Recall@K
        distance_threshold: GT distance for positive match (meters)
        skip_frames: Minimum temporal gap for loop closure

    Returns:
        recalls: {k: recall_value}
        n_queries: Number of loop closure queries found
    """
    from scipy.spatial import cKDTree
    import faiss

    n = len(embeddings)
    max_k = max(k_values)

    # Extract positions from SE(3) poses
    positions = poses[:, :3, 3].astype(np.float64)

    # Find loop closure queries using KD-Tree
    spatial_tree = cKDTree(positions)
    queries = []
    for j in range(skip_frames, n):
        nearby_indices = spatial_tree.query_ball_point(positions[j], distance_threshold)
        for i in nearby_indices:
            if i <= j - skip_frames:
                queries.append((j, i))
                break  # Only first revisit per query frame

    if len(queries) == 0:
        return {k: 0.0 for k in k_values}, 0

    # Build FAISS index for cosine similarity search
    embeddings_f32 = embeddings.astype(np.float32).copy()
    d = embeddings_f32.shape[1]
    faiss.normalize_L2(embeddings_f32)
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(embeddings_f32)

    search_k = min(max_k + 2 * skip_frames, n)

    correct_at_k = {k: 0 for k in k_values}

    for query_idx, true_match_idx in queries:
        query_emb = embeddings_f32[query_idx:query_idx + 1]
        distances, indices = faiss_index.search(query_emb, search_k)

        # Filter temporal neighbors
        valid_mask = np.abs(indices[0] - query_idx) > skip_frames
        valid_indices = indices[0][valid_mask]

        if len(valid_indices) == 0:
            continue

        valid_indices_max = valid_indices[:max_k]
        geo_dists = np.linalg.norm(
            positions[valid_indices_max] - positions[query_idx], axis=1
        )

        for k in k_values:
            if np.any(geo_dists[:k] < distance_threshold):
                correct_at_k[k] += 1

    recalls = {k: correct_at_k[k] / len(queries) for k in k_values}
    return recalls, len(queries)
