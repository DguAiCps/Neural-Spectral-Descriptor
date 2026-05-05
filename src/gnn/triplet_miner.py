"""
Triplet Miner for Hard Negative Mining

Implements hard negative mining strategy for triplet loss training:

Positive pairs:
- Same location: distance < 5m
- Different time: >30 frames apart

Hard negatives:
- Different location: distance > 10m (no upper bound)
- Smallest distance in chosen metric (most confusing)
- Supports L2 (for online mining with GNN output) and Wasserstein metrics

No upper distance bound ensures the model learns to reject perceptual aliasing
(visually similar but distant places), not just nearby different locations.

This mining strategy is critical for learning discriminative embeddings.
"""

import numpy as np
import torch
import logging
import time
from typing import List, Tuple, Optional
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from data.pose_utils import euclidean_distance
from retrieval.wasserstein import (
    wasserstein_distance_1d_numpy,
    wasserstein_distance_batch_numpy,
    precompute_cdfs_numpy,
    wasserstein_from_cdfs_numpy,
)


def _mine_sequence_worker(args):
    """
    Worker function for parallel triplet mining (CPU only).

    Must be a top-level function for multiprocessing pickle compatibility.
    Uses batch KD-tree queries + vectorized numpy + precomputed CDFs/L2.
    Note: Cannot use GPU here (CUDA context doesn't survive fork).
    """
    (seq_id, seq_indices, seq_descriptors, seq_poses,
     positive_distance_max, positive_temporal_min,
     negative_distance_min,
     negative_temporal_min, mining_strategy, n_triplets_per_anchor,
     metric) = args

    triplets = []
    n_seq = len(seq_indices)

    if n_seq < 3:
        return seq_id, triplets

    # Extract positions from poses (seq_poses is already sequence-specific)
    # Opt 9: Vectorized slicing instead of Python loop
    seq_positions = seq_poses[:, :3, 3]

    # Build KD-Tree
    tree = cKDTree(seq_positions)

    # Batch KD-tree queries (replaces N individual calls)
    too_close = tree.query_ball_tree(tree, r=negative_distance_min)
    pos_candidates_all = tree.query_ball_tree(tree, r=positive_distance_max)

    # Precompute data for distance computation
    if mining_strategy != "random":
        if metric == 'wasserstein':
            seq_data = precompute_cdfs_numpy(seq_descriptors)
        else:  # l2
            seq_data = seq_descriptors
    else:
        seq_data = None

    all_indices = np.arange(n_seq)

    for local_anchor in range(n_seq):
        # Vectorized positive filtering
        pos_local = np.array(pos_candidates_all[local_anchor])
        if len(pos_local) == 0:
            continue
        temporal_gaps = np.abs(pos_local - local_anchor)
        positive_local = pos_local[
            (pos_local != local_anchor) & (temporal_gaps >= positive_temporal_min)
        ]
        if len(positive_local) == 0:
            continue

        # Vectorized negative filtering
        tc_arr = np.array(too_close[local_anchor])
        tc_mask = np.zeros(n_seq, dtype=bool)
        tc_mask[tc_arr] = True
        temporal_mask = np.abs(all_indices - local_anchor) >= negative_temporal_min
        neg_mask = temporal_mask & ~tc_mask
        neg_mask[local_anchor] = False
        negative_local = np.where(neg_mask)[0]

        if len(negative_local) == 0:
            continue

        for _ in range(n_triplets_per_anchor):
            local_positive = np.random.choice(positive_local)

            # Hard negative selection
            if mining_strategy == "random":
                local_negative = np.random.choice(negative_local)
            else:
                anchor_data = seq_data[local_anchor]
                neg_data = seq_data[negative_local]

                if metric == 'wasserstein':
                    distances = wasserstein_from_cdfs_numpy(anchor_data, neg_data)
                else:  # l2 — squared L2 (monotonic, skip sqrt for argmin)
                    distances = np.sum((neg_data - anchor_data[None, :]) ** 2, axis=1)

                if mining_strategy == "hard":
                    local_negative = negative_local[np.argmin(distances)]
                else:  # semi-hard
                    local_negative = negative_local[np.argsort(distances)[len(distances) // 2]]

            # Convert local indices back to global indices
            triplets.append((
                seq_indices[local_anchor],
                seq_indices[local_positive],
                seq_indices[local_negative]
            ))

    return seq_id, triplets


class TripletMiner:
    """
    Mines triplets (anchor, positive, negative) for training.

    Loop Closure Mining Strategy:
    - Positive: Same location (< 5m) but different time (>= skip_frames apart)
    - Negative: Different location (> 10m, no upper bound) AND different time (>= skip_frames apart)

    No upper distance bound ensures hard negatives include perceptually similar
    but distant places (perceptual aliasing), not just nearby different locations.
    """

    def __init__(
        self,
        positive_distance_max: float = 5.0,
        positive_temporal_min: int = 30,
        negative_distance_min: float = 10.0,
        negative_temporal_min: int = 30,
        mining_strategy: str = "hard",
        device: str = "auto",
        metric: str = "l2"
    ):
        """
        Initialize triplet miner for loop closure learning.

        Args:
            positive_distance_max: Max distance for positive pairs (meters)
            positive_temporal_min: Min temporal gap for positives (keyframes)
            negative_distance_min: Min distance for negatives (meters)
            negative_temporal_min: Min temporal gap for negatives (keyframes)
            mining_strategy: Mining strategy (hard, semi-hard, random)
            device: Device for GPU-accelerated mining ('auto', 'cuda', 'cpu')
            metric: Distance metric for hard negative selection ('l2' or 'wasserstein')
        """
        self.positive_distance_max = positive_distance_max
        self.positive_temporal_min = positive_temporal_min
        self.negative_distance_min = negative_distance_min
        self.negative_temporal_min = negative_temporal_min
        self.mining_strategy = mining_strategy
        self.metric = metric

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def mine_triplets(
        self,
        descriptors: np.ndarray,
        poses: np.ndarray,
        n_triplets_per_anchor: int = 1,
        sequence_ids: np.ndarray = None,
        parallel: bool = True,
        n_workers: int = None
    ) -> List[Tuple[int, int, int]]:
        """
        Mine triplets from keyframe data (per-sequence to avoid cross-sequence pairs)

        Args:
            descriptors: (n_keyframes, n_bins) spectral histograms
            poses: (n_keyframes, 4, 4) SE(3) poses
            n_triplets_per_anchor: Number of triplets per anchor
            sequence_ids: (n_keyframes,) sequence ID for each keyframe (optional)
            parallel: Use parallel processing for sequence mining (default: True)
            n_workers: Number of parallel workers (default: min(n_sequences, cpu_count/2))

        Returns:
            List of (anchor_idx, positive_idx, negative_idx) tuples
        """
        n_keyframes = len(descriptors)
        triplets = []

        # If sequence_ids provided, mine per sequence (much faster)
        if sequence_ids is not None:
            unique_seqs = np.unique(sequence_ids)

            # Use parallel mining if enabled and multiple sequences
            if parallel and len(unique_seqs) > 1:
                return self._mine_triplets_parallel(
                    descriptors, poses, n_triplets_per_anchor,
                    sequence_ids, unique_seqs, n_workers
                )

            # Sequential mining
            logging.info(f"Mining triplets per sequence ({len(unique_seqs)} sequences)...")

            for seq_idx, seq_id in enumerate(unique_seqs):
                seq_start = time.perf_counter()
                seq_mask = sequence_ids == seq_id
                seq_indices = np.where(seq_mask)[0]

                if len(seq_indices) < 3:
                    continue

                # Mine within this sequence
                seq_triplets = self._mine_sequence_triplets(
                    seq_indices, descriptors, poses, n_triplets_per_anchor
                )
                triplets.extend(seq_triplets)
                seq_time = time.perf_counter() - seq_start

                logging.info(
                    f"  Seq {seq_idx+1}/{len(unique_seqs)} (id={seq_id}): "
                    f"{len(seq_indices):,} keyframes -> {len(seq_triplets):,} triplets "
                    f"({seq_time:.1f}s, {len(seq_triplets)/seq_time:.0f}/s)"
                )

            return triplets

        # Original O(n²) approach if no sequence_ids
        for anchor_idx in range(n_keyframes):
            positive_candidates = self._find_positive_candidates(
                anchor_idx, poses, n_keyframes
            )

            if len(positive_candidates) == 0:
                continue

            negative_candidates = self._find_negative_candidates(
                anchor_idx, poses, n_keyframes
            )

            if len(negative_candidates) == 0:
                continue

            for _ in range(n_triplets_per_anchor):
                positive_idx = np.random.choice(positive_candidates)
                negative_idx = self._select_hard_negative(
                    anchor_idx, negative_candidates, descriptors
                )
                triplets.append((anchor_idx, positive_idx, negative_idx))

        return triplets

    def _mine_triplets_parallel(
        self,
        descriptors: np.ndarray,
        poses: np.ndarray,
        n_triplets_per_anchor: int,
        sequence_ids: np.ndarray,
        unique_seqs: np.ndarray,
        n_workers: int = None
    ) -> List[Tuple[int, int, int]]:
        """
        Mine triplets in parallel across sequences.

        Opt 2: Uses ThreadPoolExecutor instead of ProcessPoolExecutor to avoid
        fork-related memory duplication. Numpy operations release the GIL, so
        threads achieve real parallelism for the heavy compute (KD-tree, L2).

        Args:
            descriptors: All descriptors
            poses: All poses
            n_triplets_per_anchor: Triplets per anchor
            sequence_ids: Sequence ID for each keyframe
            unique_seqs: Unique sequence IDs
            n_workers: Number of workers

        Returns:
            List of triplets
        """
        from concurrent.futures import ThreadPoolExecutor

        if n_workers is None:
            n_workers = min(len(unique_seqs), 8)

        logging.info(f"Mining triplets in PARALLEL ({len(unique_seqs)} sequences, {n_workers} thread workers)...")

        # Prepare arguments for each sequence (only pass sequence-specific data)
        args_list = []
        for seq_id in unique_seqs:
            seq_mask = sequence_ids == seq_id
            seq_indices = np.where(seq_mask)[0]

            if len(seq_indices) < 3:
                continue

            # Extract only this sequence's data to avoid copying entire arrays
            seq_descriptors = descriptors[seq_indices]
            seq_poses = poses[seq_indices]

            args_list.append((
                seq_id,
                seq_indices,
                seq_descriptors,  # Only this sequence's descriptors
                seq_poses,        # Only this sequence's poses
                self.positive_distance_max,
                self.positive_temporal_min,
                self.negative_distance_min,
                self.negative_temporal_min,
                self.mining_strategy,
                n_triplets_per_anchor,
                self.metric
            ))

        # Execute in parallel using threads (numpy releases GIL)
        all_triplets = []
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_mine_sequence_worker, args_list))

        for seq_id, triplets in results:
            all_triplets.extend(triplets)

        elapsed = time.perf_counter() - start_time
        logging.info(
            f"  Parallel mining complete: {len(all_triplets):,} triplets "
            f"in {elapsed:.1f}s ({len(all_triplets)/elapsed:.0f}/s)"
        )

        return all_triplets

    def _mine_sequence_gpu(
        self,
        seq_indices: np.ndarray,
        seq_data: np.ndarray,
        pos_candidates_all: list,
        too_close: list,
        n_seq: int,
        n_triplets_per_anchor: int
    ) -> List[Tuple[int, int, int]]:
        """
        GPU-accelerated hard negative mining for a single sequence.

        Uses batched distance computation on GPU for exact hardest
        negative selection without subsampling.
        Supports L2 (torch.cdist) and Wasserstein (CDF L1) metrics.
        """
        device = self.device

        # Move data to GPU (N, D) — CDFs for wasserstein, embeddings for L2
        data_gpu = torch.from_numpy(seq_data).float().to(device)

        # Precompute too_close as padded tensor for efficient GPU masking
        max_tc_len = max(len(tc) for tc in too_close) if too_close else 0
        tc_padded = np.full((n_seq, max(max_tc_len, 1)), -1, dtype=np.int64)
        tc_lengths = np.zeros(n_seq, dtype=np.int64)
        for i, tc in enumerate(too_close):
            if len(tc) > 0:
                tc_padded[i, :len(tc)] = tc
                tc_lengths[i] = len(tc)

        tc_padded_gpu = torch.from_numpy(tc_padded).to(device)
        tc_lengths_gpu = torch.from_numpy(tc_lengths).to(device)

        # Identify productive anchors (those with valid positives AND negatives)
        all_indices_np = np.arange(n_seq)
        productive_anchors = []
        anchor_positives = {}

        for local_anchor in range(n_seq):
            pos_local = np.array(pos_candidates_all[local_anchor])
            if len(pos_local) == 0:
                continue
            temporal_gaps = np.abs(pos_local - local_anchor)
            pos_filtered = pos_local[
                (pos_local != local_anchor) & (temporal_gaps >= self.positive_temporal_min)
            ]
            if len(pos_filtered) == 0:
                continue
            productive_anchors.append(local_anchor)
            anchor_positives[local_anchor] = pos_filtered

        if len(productive_anchors) == 0:
            return []

        # GPU batched Wasserstein computation
        BATCH_SIZE = 32
        indices_gpu = torch.arange(n_seq, device=device)
        triplets = []

        for batch_start in range(0, len(productive_anchors), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(productive_anchors))
            batch_locals = productive_anchors[batch_start:batch_end]
            B = len(batch_locals)
            batch_locals_gpu = torch.tensor(batch_locals, dtype=torch.long, device=device)

            # Batch data: (B, D)
            batch_data = data_gpu[batch_locals_gpu]

            # Distance computation based on metric
            if self.metric == 'l2':
                # L2 distances: (B, N)
                dists = torch.cdist(batch_data, data_gpu)
            else:
                # Wasserstein: (B, 1, D) - (1, N, D) → abs → sum → (B, N)
                dists = torch.abs(
                    batch_data.unsqueeze(1) - data_gpu.unsqueeze(0)
                ).sum(dim=2)

            # Build negative mask per batch on-the-fly: (B, N)
            # Temporal: |anchor_idx - j| >= temporal_min
            temporal_ok = torch.abs(
                batch_locals_gpu.unsqueeze(1) - indices_gpu.unsqueeze(0)
            ) >= self.negative_temporal_min

            # Spatial: exclude too_close nodes
            spatial_far = torch.ones(B, n_seq, dtype=torch.bool, device=device)
            for k in range(B):
                local_anchor = batch_locals[k]
                tc_len = tc_lengths[local_anchor]
                if tc_len > 0:
                    tc_indices = tc_padded_gpu[local_anchor, :tc_len]
                    spatial_far[k, tc_indices] = False

            # Exclude self
            neg_mask = temporal_ok & spatial_far
            for k in range(B):
                neg_mask[k, batch_locals[k]] = False

            # Mask non-negatives with inf
            dists[~neg_mask] = float('inf')

            # Check if any valid negatives exist per anchor
            has_negatives = neg_mask.any(dim=1)  # (B,)

            if self.mining_strategy == "hard":
                hardest_neg = dists.argmin(dim=1).cpu().numpy()
            else:  # semi-hard: pick median difficulty
                sorted_dists, sorted_idx = torch.sort(dists, dim=1)
                n_valid = neg_mask.sum(dim=1)
                median_pos = (n_valid // 2).clamp(min=0)
                hardest_neg = sorted_idx[
                    torch.arange(B, device=device), median_pos
                ].cpu().numpy()

            # Create triplets
            for k in range(B):
                if not has_negatives[k]:
                    continue
                local_anchor = batch_locals[k]
                anchor_idx = seq_indices[local_anchor]
                positives = anchor_positives[local_anchor]

                for _ in range(n_triplets_per_anchor):
                    local_positive = np.random.choice(positives)
                    positive_idx = seq_indices[local_positive]
                    negative_idx = seq_indices[hardest_neg[k]]
                    triplets.append((anchor_idx, positive_idx, negative_idx))

        return triplets

    def _mine_sequence_triplets(
        self,
        seq_indices: np.ndarray,
        descriptors: np.ndarray,
        poses: np.ndarray,
        n_triplets_per_anchor: int
    ) -> List[Tuple[int, int, int]]:
        """
        Mine triplets within a single sequence using KD-Tree.

        Loop Closure Mining:
        - Positive: distance < 5m AND temporal_gap >= 30 (same place, different time)
        - Negative: distance > 10m (no upper bound) AND temporal_gap >= 30

        Uses GPU-accelerated Wasserstein computation when available,
        with vectorized CPU fallback.
        """
        n_seq = len(seq_indices)

        # Extract positions from poses (translation component)
        seq_positions = np.array([
            poses[idx][:3, 3] for idx in seq_indices
        ])  # (n_seq, 3)

        # Build KD-Tree for fast spatial queries
        tree = cKDTree(seq_positions)

        # Batch KD-tree queries (replaces N individual query_ball_point calls)
        too_close = tree.query_ball_tree(tree, r=self.negative_distance_min)
        pos_candidates_all = tree.query_ball_tree(tree, r=self.positive_distance_max)

        # Precompute data for distance computation
        seq_descriptors = descriptors[seq_indices]
        if self.metric == 'wasserstein':
            seq_data = precompute_cdfs_numpy(seq_descriptors)
        else:  # l2
            seq_data = seq_descriptors

        # GPU path: exact hard negatives via batched GPU distance computation
        use_gpu = (
            self.device.startswith("cuda")
            and torch.cuda.is_available()
            and self.mining_strategy != "random"
        )

        if use_gpu:
            return self._mine_sequence_gpu(
                seq_indices, seq_data, pos_candidates_all, too_close,
                n_seq, n_triplets_per_anchor
            )

        # CPU vectorized fallback
        triplets = []
        all_indices = np.arange(n_seq)

        for local_anchor in range(n_seq):
            anchor_idx = seq_indices[local_anchor]

            # Vectorized positive filtering
            pos_local = np.array(pos_candidates_all[local_anchor])
            if len(pos_local) == 0:
                continue
            temporal_gaps = np.abs(pos_local - local_anchor)
            pos_filtered = pos_local[
                (pos_local != local_anchor) & (temporal_gaps >= self.positive_temporal_min)
            ]
            if len(pos_filtered) == 0:
                continue
            positive_candidates = seq_indices[pos_filtered]

            # Vectorized negative filtering
            tc_arr = np.array(too_close[local_anchor])
            tc_mask = np.zeros(n_seq, dtype=bool)
            tc_mask[tc_arr] = True
            temporal_mask = np.abs(all_indices - local_anchor) >= self.negative_temporal_min
            neg_mask = temporal_mask & ~tc_mask
            neg_mask[local_anchor] = False
            negative_local = np.where(neg_mask)[0]

            if len(negative_local) == 0:
                continue

            for _ in range(n_triplets_per_anchor):
                positive_idx = np.random.choice(positive_candidates)

                if self.mining_strategy == "random":
                    negative_idx = seq_indices[np.random.choice(negative_local)]
                else:
                    anchor_data = seq_data[local_anchor]
                    neg_data = seq_data[negative_local]

                    if self.metric == 'wasserstein':
                        distances = wasserstein_from_cdfs_numpy(anchor_data, neg_data)
                    else:  # l2 — squared L2 (monotonic, skip sqrt for argmin)
                        distances = np.sum((neg_data - anchor_data[None, :]) ** 2, axis=1)

                    if self.mining_strategy == "hard":
                        negative_idx = seq_indices[negative_local[np.argmin(distances)]]
                    else:  # semi-hard
                        median = np.argsort(distances)[len(distances) // 2]
                        negative_idx = seq_indices[negative_local[median]]

                triplets.append((anchor_idx, positive_idx, negative_idx))

        return triplets

    def _find_positive_candidates(
        self,
        anchor_idx: int,
        poses: np.ndarray,
        n_keyframes: int
    ) -> List[int]:
        """
        Find positive candidates for anchor

        Criteria:
        - Distance < 5m
        - Temporal gap > 30 frames

        Args:
            anchor_idx: Anchor index
            poses: All poses
            n_keyframes: Total keyframes

        Returns:
            List of positive candidate indices
        """
        candidates = []

        anchor_pose = poses[anchor_idx]

        for i in range(n_keyframes):
            if i == anchor_idx:
                continue

            # Check temporal gap
            temporal_gap = abs(i - anchor_idx)
            if temporal_gap < self.positive_temporal_min:
                continue

            # Check spatial distance
            distance = euclidean_distance(anchor_pose, poses[i])
            if distance <= self.positive_distance_max:
                candidates.append(i)

        return candidates

    def _find_negative_candidates(
        self,
        anchor_idx: int,
        poses: np.ndarray,
        n_keyframes: int
    ) -> List[int]:
        """
        Find negative candidates for anchor (loop closure style).

        Criteria:
        - distance > 10m (different location, no upper bound)
        - temporal_gap >= negative_temporal_min (not a temporal neighbor)

        Args:
            anchor_idx: Anchor index
            poses: All poses
            n_keyframes: Total keyframes

        Returns:
            List of negative candidate indices
        """
        candidates = []

        anchor_pose = poses[anchor_idx]

        for i in range(n_keyframes):
            if i == anchor_idx:
                continue

            # Check temporal gap (exclude temporal neighbors)
            temporal_gap = abs(i - anchor_idx)
            if temporal_gap < self.negative_temporal_min:
                continue

            # Check spatial distance (no upper bound)
            distance = euclidean_distance(anchor_pose, poses[i])

            if distance >= self.negative_distance_min:
                candidates.append(i)

        return candidates

    def _select_hard_negative(
        self,
        anchor_idx: int,
        negative_candidates: List[int],
        descriptors: np.ndarray
    ) -> int:
        """
        Select hard negative from candidates

        Hard negative = smallest Wasserstein distance (most confusing)

        Args:
            anchor_idx: Anchor index
            negative_candidates: List of negative candidate indices
            descriptors: All descriptors

        Returns:
            Index of hard negative
        """
        if self.mining_strategy == "random":
            return np.random.choice(negative_candidates)

        # Compute Wasserstein distances to all candidates (vectorized)
        anchor_descriptor = descriptors[anchor_idx]
        neg_descriptors = descriptors[negative_candidates]  # (n_neg, n_bins)
        distances = wasserstein_distance_batch_numpy(anchor_descriptor, neg_descriptors)

        if self.mining_strategy == "hard":
            # Smallest distance = hardest negative
            hardest_idx = np.argmin(distances)
            return negative_candidates[hardest_idx]

        elif self.mining_strategy == "semi-hard":
            # Semi-hard: closer than positive but not too close
            # For simplicity, select median difficulty
            median_idx = np.argsort(distances)[len(distances) // 2]
            return negative_candidates[median_idx]

        else:
            raise ValueError(f"Unknown mining strategy: {self.mining_strategy}")


def mine_pos_neg_lists(
    descriptors: np.ndarray,
    poses: np.ndarray,
    sequence_ids: np.ndarray,
    pos_dist_max: float = 5.0,
    neg_dist_min: float = 10.0,
    temporal_min: int = 30,
    n_neg_per_anchor: int = 32,
    metric: str = 'l2',
) -> Tuple[np.ndarray, dict, dict]:
    """Mine positive/negative pools per anchor for SmoothAP training.

    For each anchor, returns:
    - All same-sequence positives (pose_dist < pos_dist_max, |i-j| >= temporal_min)
    - Top-N hardest negatives (pose_dist > neg_dist_min, smallest descriptor distance)

    Cross-sequence pairs are not connected (poses live in sequence-local frames).

    Args:
        descriptors: (N, D) raw descriptors (used for hard negative ranking).
        poses: (N, 4, 4) SE(3) poses.
        sequence_ids: (N,) sequence ID per node.
        pos_dist_max: Same-place threshold (meters).
        neg_dist_min: Different-place threshold (meters).
        temporal_min: Minimum frame gap to consider as revisit.
        n_neg_per_anchor: Hard negatives kept per anchor.
        metric: 'l2' descriptor distance for hard negative selection.

    Returns:
        anchors: array of anchor indices that have at least 1 positive.
        pos_pool: dict {anchor_idx: np.array of positive indices (variable length)}
        neg_pool: dict {anchor_idx: np.array of (top-N hard) negative indices (length n_neg_per_anchor or less)}
    """
    pos_pool = {}
    neg_pool = {}
    valid_anchors = []

    positions = poses[:, :3, 3]
    unique_seqs = np.unique(sequence_ids)

    for seq_id in unique_seqs:
        seq_mask = sequence_ids == seq_id
        seq_indices = np.where(seq_mask)[0]
        n_seq = len(seq_indices)
        if n_seq < 3:
            continue

        seq_positions = positions[seq_indices]
        seq_descriptors = descriptors[seq_indices]
        tree = cKDTree(seq_positions)
        all_local = np.arange(n_seq)

        # Vectorized neighbor queries
        pos_candidates_all = tree.query_ball_tree(tree, r=pos_dist_max)
        too_close_all = tree.query_ball_tree(tree, r=neg_dist_min)

        for local_anchor in range(n_seq):
            global_anchor = int(seq_indices[local_anchor])

            # Positives: pos_dist < pos_dist_max, frame_gap >= temporal_min, exclude self
            pos_local = np.array(pos_candidates_all[local_anchor], dtype=np.int64)
            if len(pos_local) == 0:
                continue
            tgaps = np.abs(pos_local - local_anchor)
            keep = (pos_local != local_anchor) & (tgaps >= temporal_min)
            pos_local = pos_local[keep]
            if len(pos_local) == 0:
                continue

            # Negatives: pose_dist > neg_dist_min, frame_gap >= temporal_min
            tc_arr = np.array(too_close_all[local_anchor], dtype=np.int64)
            tc_mask = np.zeros(n_seq, dtype=bool)
            tc_mask[tc_arr] = True
            neg_temporal = np.abs(all_local - local_anchor) >= temporal_min
            neg_mask = neg_temporal & ~tc_mask
            neg_mask[local_anchor] = False
            negative_local = np.where(neg_mask)[0]
            if len(negative_local) == 0:
                continue

            # Top-N hardest negatives by descriptor L2
            anchor_desc = seq_descriptors[local_anchor]
            neg_descs = seq_descriptors[negative_local]
            if metric == 'l2':
                d2 = np.sum((neg_descs - anchor_desc[None, :]) ** 2, axis=1)
            else:
                # Fallback: random
                d2 = np.random.rand(len(negative_local))
            top = min(n_neg_per_anchor, len(negative_local))
            top_local = negative_local[np.argsort(d2)[:top]]

            pos_global = seq_indices[pos_local].astype(np.int64)
            neg_global = seq_indices[top_local].astype(np.int64)

            pos_pool[global_anchor] = pos_global
            neg_pool[global_anchor] = neg_global
            valid_anchors.append(global_anchor)

    return np.array(valid_anchors, dtype=np.int64), pos_pool, neg_pool


class BatchTripletMiner:
    """
    Mines triplets from a batch of embeddings during training

    Uses online hard mining within a batch for efficiency.
    """

    def __init__(
        self,
        margin: float = 0.1,
        mining_strategy: str = "hard"
    ):
        """
        Initialize batch triplet miner

        Args:
            margin: Triplet loss margin
            mining_strategy: Mining strategy (hard, semi-hard, all)
        """
        self.margin = margin
        self.mining_strategy = mining_strategy

    def mine_batch_triplets(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine triplets from batch

        Args:
            embeddings: (batch_size, embedding_dim) embeddings
            labels: (batch_size,) labels (keyframe IDs or cluster IDs)

        Returns:
            anchors: (n_triplets, embedding_dim)
            positives: (n_triplets, embedding_dim)
            negatives: (n_triplets, embedding_dim)
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Compute pairwise distances
        distances = self._pairwise_distances(embeddings)

        # Create label masks
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_not_equal = ~label_equal

        # For each anchor
        anchors = []
        positives = []
        negatives = []

        for i in range(batch_size):
            # Find positives (same label, excluding self)
            positive_mask = label_equal[i].clone()
            positive_mask[i] = False

            if not positive_mask.any():
                continue  # No valid positives

            # Find negatives (different label)
            negative_mask = label_not_equal[i]

            if not negative_mask.any():
                continue  # No valid negatives

            # Select positive
            if self.mining_strategy == "hard":
                # Hardest positive = farthest
                positive_distances = distances[i].clone()
                positive_distances[~positive_mask] = -1
                positive_idx = positive_distances.argmax()
            else:
                # Random positive
                positive_indices = positive_mask.nonzero(as_tuple=True)[0]
                positive_idx = positive_indices[torch.randint(len(positive_indices), (1,))]

            # Select negative
            if self.mining_strategy == "hard":
                # Hardest negative = closest
                negative_distances = distances[i].clone()
                negative_distances[~negative_mask] = float('inf')
                negative_idx = negative_distances.argmin()
            elif self.mining_strategy == "semi-hard":
                # Semi-hard: d(a,n) > d(a,p) but < d(a,p) + margin
                positive_dist = distances[i, positive_idx]
                negative_distances = distances[i].clone()
                negative_distances[~negative_mask] = float('inf')

                # Find semi-hard negatives
                semi_hard_mask = (
                    (negative_distances > positive_dist) &
                    (negative_distances < positive_dist + self.margin)
                )

                if semi_hard_mask.any():
                    semi_hard_distances = negative_distances.clone()
                    semi_hard_distances[~semi_hard_mask] = float('inf')
                    negative_idx = semi_hard_distances.argmin()
                else:
                    # Fall back to hardest
                    negative_idx = negative_distances.argmin()
            else:
                # Random negative
                negative_indices = negative_mask.nonzero(as_tuple=True)[0]
                negative_idx = negative_indices[torch.randint(len(negative_indices), (1,))]

            # Add triplet
            anchors.append(embeddings[i])
            positives.append(embeddings[positive_idx])
            negatives.append(embeddings[negative_idx])

        if len(anchors) == 0:
            # No valid triplets
            return (
                torch.zeros((0, embeddings.shape[1]), device=device),
                torch.zeros((0, embeddings.shape[1]), device=device),
                torch.zeros((0, embeddings.shape[1]), device=device)
            )

        return (
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives)
        )

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise L2 distances

        Args:
            embeddings: (batch_size, embedding_dim)

        Returns:
            (batch_size, batch_size) distance matrix
        """
        # Efficient pairwise distance computation
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b

        dot_product = torch.mm(embeddings, embeddings.t())
        squared_norms = torch.diag(dot_product).unsqueeze(0)

        distances = squared_norms + squared_norms.t() - 2 * dot_product
        distances = torch.clamp(distances, min=0.0)  # Numerical stability

        return torch.sqrt(distances)


def create_triplet_miner(
    positive_distance_max: float = 5.0,
    positive_temporal_min: int = 30,
    negative_distance_min: float = 10.0,
    negative_temporal_min: int = 30,
    mining_strategy: str = "hard",
    device: str = "auto",
    metric: str = "l2"
) -> TripletMiner:
    """
    Factory function to create triplet miner for loop closure learning.

    Args:
        positive_distance_max: Max distance for positives (meters)
        positive_temporal_min: Min temporal gap for positives (keyframes)
        negative_distance_min: Min distance for negatives (meters)
        negative_temporal_min: Min temporal gap for negatives (keyframes)
        mining_strategy: Mining strategy (hard, semi-hard, random)
        device: Device for GPU-accelerated mining ('auto', 'cuda', 'cpu')
        metric: Distance metric for hard negative selection ('l2' or 'wasserstein')

    Returns:
        TripletMiner instance configured for loop closure learning
    """
    return TripletMiner(
        positive_distance_max=positive_distance_max,
        positive_temporal_min=positive_temporal_min,
        negative_distance_min=negative_distance_min,
        negative_temporal_min=negative_temporal_min,
        mining_strategy=mining_strategy,
        device=device,
        metric=metric
    )
