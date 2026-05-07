"""
Evaluation utilities for baseline comparison.

Two retrieval paths:
- compute_recall_multi_k(...)             — single-vector cosine FAISS (default)
- compute_recall_cosine_then_rerank(...)  — cosine pre-filter + custom rerank
                                            (used by SC++, LiDAR-Iris)

Both share the same revisit-query selection and recall-counting logic, and
both accept an optional `per_query_records` list to populate with diagnostic
fields (geo distance, rank, |Δyaw|) used for yaw-conditioned R@1 analysis.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def _find_revisit_queries(
    positions: np.ndarray,
    distance_threshold: float = 5.0,
    skip_frames: int = 30,
) -> List[Tuple[int, int]]:
    """
    Identify loop-closure query frames.

    A query (j, i) means frame j returned within distance_threshold of an
    earlier frame i, with i <= j - skip_frames temporal gap.

    Returns:
        List of (query_idx, ground_truth_match_idx). Only the first revisit
        per query frame is kept (matching the original eval protocol).
    """
    from scipy.spatial import cKDTree

    n = len(positions)
    spatial_tree = cKDTree(positions)
    queries = []
    for j in range(skip_frames, n):
        nearby_indices = spatial_tree.query_ball_point(positions[j], distance_threshold)
        for i in nearby_indices:
            if i <= j - skip_frames:
                queries.append((j, i))
                break
    return queries


def _score_recalls_from_ranked(
    queries: List[Tuple[int, int]],
    ranked_lists: List[np.ndarray],
    positions: np.ndarray,
    k_values: List[int],
    distance_threshold: float = 5.0,
    poses: Optional[np.ndarray] = None,
    per_query_records: Optional[List[dict]] = None,
) -> Dict[int, float]:
    """
    Count how many queries had a correct match (within distance_threshold) in
    their top-K ranked candidate lists.

    Args:
        queries: list of (query_idx, gt_idx). gt_idx is the anchored true match
                 from `_find_revisit_queries`; used for rank/yaw bookkeeping
                 when `per_query_records` is provided.
        ranked_lists: same length as queries; each is an int array of database
                      indices in retrieval order (already temporal-skip-filtered).
        positions: (n, 3) keyframe positions.
        k_values: list of K to compute recall at.
        poses: (n, 4, 4) SE(3); only used when `per_query_records` is provided
               (to extract |Δyaw| between query and gt match).
        per_query_records: if provided, append per-query diagnostic dicts
                           (matching src/gnn/trainer.py format) for downstream
                           yaw-conditioned analysis.

    Returns:
        {k: recall_value}
    """
    correct_at_k = {k: 0 for k in k_values}
    max_k = max(k_values)
    n_queries = len(queries)
    if n_queries == 0:
        return {k: 0.0 for k in k_values}

    for (query_idx, true_match_idx), ranked in zip(queries, ranked_lists):
        if len(ranked) == 0:
            if per_query_records is not None:
                per_query_records.append({
                    'query_idx': int(query_idx),
                    'true_match_idx': int(true_match_idx),
                    'top1_idx': -1,
                    'top1_geo_dist_m': float('nan'),
                    'true_match_rank': -1,
                    'success_at_k1': False,
                    'delta_yaw_deg': _delta_yaw_deg(poses, query_idx, true_match_idx)
                                       if poses is not None else float('nan'),
                })
            continue
        ranked_top = ranked[:max_k]
        geo_dists = np.linalg.norm(
            positions[ranked_top] - positions[query_idx], axis=1
        )
        for k in k_values:
            if np.any(geo_dists[:k] < distance_threshold):
                correct_at_k[k] += 1

        if per_query_records is not None:
            rank_arr = np.where(ranked_top == true_match_idx)[0]
            true_rank = int(rank_arr[0]) + 1 if rank_arr.size > 0 else -1
            per_query_records.append({
                'query_idx': int(query_idx),
                'true_match_idx': int(true_match_idx),
                'top1_idx': int(ranked_top[0]),
                'top1_geo_dist_m': float(geo_dists[0]),
                'true_match_rank': true_rank,
                'success_at_k1': bool(geo_dists[0] < distance_threshold),
                'delta_yaw_deg': _delta_yaw_deg(poses, query_idx, true_match_idx)
                                   if poses is not None else float('nan'),
            })

    return {k: correct_at_k[k] / n_queries for k in k_values}


def _delta_yaw_deg(poses: np.ndarray, i: int, j: int) -> float:
    """Signed yaw difference between two SE(3) poses, in degrees, in [-180, 180]."""
    R_i = poses[i, :3, :3]
    R_j = poses[j, :3, :3]
    yaw_i = np.arctan2(R_i[1, 0], R_i[0, 0])
    yaw_j = np.arctan2(R_j[1, 0], R_j[0, 0])
    return float(np.degrees(np.arctan2(np.sin(yaw_i - yaw_j),
                                       np.cos(yaw_i - yaw_j))))


def compute_recall_multi_k(
    embeddings: np.ndarray,
    poses: np.ndarray,
    k_values: List[int] = [1, 5, 10],
    distance_threshold: float = 5.0,
    skip_frames: int = 30,
    per_query_records: Optional[List[dict]] = None,
) -> Tuple[Dict[int, float], int]:
    """
    Single-vector cosine FAISS retrieval (default path).

    Args:
        embeddings: (n, D) descriptors (will be L2-normalized internally)
        poses: (n, 4, 4) SE(3) poses
        k_values: list of K for Recall@K
        distance_threshold: GT positive radius (m)
        skip_frames: minimum temporal gap for valid loop closure
        per_query_records: optional list to populate with per-query records
                           (geo distance, rank, |Δyaw|) for downstream analysis.
    """
    import faiss

    positions = poses[:, :3, 3].astype(np.float64)
    queries = _find_revisit_queries(positions, distance_threshold, skip_frames)
    if len(queries) == 0:
        return {k: 0.0 for k in k_values}, 0

    embeddings_f32 = embeddings.astype(np.float32).copy()
    d = embeddings_f32.shape[1]
    faiss.normalize_L2(embeddings_f32)
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(embeddings_f32)

    n = len(embeddings_f32)
    max_k = max(k_values)
    search_k = min(max_k + 2 * skip_frames, n)

    ranked_lists = []
    for query_idx, _ in queries:
        query_emb = embeddings_f32[query_idx:query_idx + 1]
        _, indices = faiss_index.search(query_emb, search_k)
        valid_mask = np.abs(indices[0] - query_idx) > skip_frames
        ranked_lists.append(indices[0][valid_mask][:max_k])

    recalls = _score_recalls_from_ranked(
        queries, ranked_lists, positions, k_values, distance_threshold,
        poses=poses, per_query_records=per_query_records,
    )
    return recalls, len(queries)


def compute_recall_cosine_then_rerank(
    coarse_descriptors: np.ndarray,
    rerank_fn: Callable[[int, np.ndarray], np.ndarray],
    poses: np.ndarray,
    k_values: List[int] = [1, 5, 10],
    distance_threshold: float = 5.0,
    skip_frames: int = 30,
    n_coarse: int = 200,
    per_query_records: Optional[List[dict]] = None,
) -> Tuple[Dict[int, float], int]:
    """
    Two-stage retrieval: cosine FAISS pre-filter → method-specific rerank.

    Args:
        coarse_descriptors: (n, D_coarse) rotation-invariant pre-filter
                            descriptors (e.g., Ring Key for SC++, ring-mean
                            signature for LiDAR-Iris). L2-normalized internally.
        rerank_fn(query_idx, candidate_indices) -> reranked_indices
            Method-specific distance computation; returns the candidates sorted
            ascending by distance.
        poses, k_values, distance_threshold, skip_frames: as for
            compute_recall_multi_k.
        n_coarse: how many candidates to pull from the cosine pre-filter
                  before reranking.

    Returns:
        ({k: recall}, n_queries)
    """
    import faiss

    positions = poses[:, :3, 3].astype(np.float64)
    queries = _find_revisit_queries(positions, distance_threshold, skip_frames)
    if len(queries) == 0:
        return {k: 0.0 for k in k_values}, 0

    coarse_f32 = coarse_descriptors.astype(np.float32).copy()
    faiss.normalize_L2(coarse_f32)
    faiss_index = faiss.IndexFlatIP(coarse_f32.shape[1])
    faiss_index.add(coarse_f32)

    n = len(coarse_f32)
    search_k = min(n_coarse + 2 * skip_frames, n)

    ranked_lists = []
    for query_idx, _ in queries:
        query_emb = coarse_f32[query_idx:query_idx + 1]
        _, indices = faiss_index.search(query_emb, search_k)
        valid_mask = np.abs(indices[0] - query_idx) > skip_frames
        candidates = indices[0][valid_mask][:n_coarse]
        if len(candidates) == 0:
            ranked_lists.append(np.array([], dtype=np.int64))
            continue
        reranked = rerank_fn(query_idx, candidates)
        ranked_lists.append(np.asarray(reranked, dtype=np.int64))

    recalls = _score_recalls_from_ranked(
        queries, ranked_lists, positions, k_values, distance_threshold,
        poses=poses, per_query_records=per_query_records,
    )
    return recalls, len(queries)
