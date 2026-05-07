"""
Two-Stage Loop Closing - Algorithm 5

Stage 1 (Global Retrieval):
- Compute Wasserstein distance to database (O(n) via sorted histograms)
- Spatial filtering >50m
- Context injection with last 10 keyframes
- Select top-K=10 candidates

Stage 2 (Geometric Verification):
- GICP registration on top-K candidates
- Fitness >0.3, RMSE <0.5m thresholds
- Output: relative pose & information matrix

Target latency: 27ms @ 100K database
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from retrieval.wasserstein import wasserstein_distance_batch_numpy, WassersteinRetriever
from retrieval.geometric_verification import GeometricVerifier, compute_pose_graph_edge
from keyframe.selector import Keyframe


@dataclass
class LoopClosureCandidate:
    """Loop closure candidate"""
    database_idx: int
    distance: float  # Wasserstein distance
    verified: bool = False
    transform: Optional[np.ndarray] = None
    fitness: Optional[float] = None
    rmse: Optional[float] = None
    information_matrix: Optional[np.ndarray] = None


class TwoStageRetrieval:
    """
    Two-stage loop closing system

    Combines fast global retrieval with precise geometric verification.
    """

    def __init__(
        self,
        top_k: int = 10,
        spatial_filter_distance: float = 50.0,
        context_window: int = 10,
        fitness_threshold: float = 0.3,
        rmse_threshold: float = 0.5,
        verification_method: str = "gicp",
        use_torch: bool = False,
        device: str = 'cpu'
    ):
        """
        Initialize two-stage retrieval

        Args:
            top_k: Number of candidates from global retrieval
            spatial_filter_distance: Spatial filtering threshold (meters)
            context_window: Number of recent keyframes for context
            fitness_threshold: ICP fitness threshold
            rmse_threshold: ICP RMSE threshold
            verification_method: ICP method ("icp" or "gicp")
            use_torch: Use PyTorch for Wasserstein computation
            device: Device for PyTorch
        """
        self.top_k = top_k
        self.spatial_filter_distance = spatial_filter_distance
        self.context_window = context_window

        # Stage 1: Global retrieval
        self.retriever = WassersteinRetriever(
            use_torch=use_torch,
            device=device
        )

        # Stage 2: Geometric verification
        self.verifier = GeometricVerifier(
            method=verification_method,
            fitness_threshold=fitness_threshold,
            rmse_threshold=rmse_threshold
        )

        # Database
        self.keyframes: List[Keyframe] = []

    def add_keyframe(self, keyframe: Keyframe):
        """
        Add keyframe to database

        Args:
            keyframe: Keyframe with descriptor computed
        """
        if keyframe.descriptor is None:
            raise ValueError("Keyframe must have descriptor before adding to database")

        self.keyframes.append(keyframe)

        # Add descriptor to retrieval database
        descriptor = keyframe.descriptor.reshape(1, -1)
        self.retriever.add_to_database(descriptor)

    def query(
        self,
        query_keyframe: Keyframe,
        query_points: Optional[np.ndarray] = None,
        verify: bool = True
    ) -> List[LoopClosureCandidate]:
        """
        Query for loop closures

        Args:
            query_keyframe: Query keyframe with descriptor
            query_points: Query point cloud (required if verify=True)
            verify: Run geometric verification (Stage 2)

        Returns:
            List of loop closure candidates (verified if verify=True)
        """
        if query_keyframe.descriptor is None:
            raise ValueError("Query keyframe must have descriptor")

        # Stage 1: Global retrieval
        candidates = self._global_retrieval(query_keyframe)

        if len(candidates) == 0:
            return []

        # Stage 2: Geometric verification
        if verify:
            if query_points is None:
                query_points = query_keyframe.points

            candidates = self._geometric_verification(
                query_points,
                candidates
            )

        return candidates

    def _global_retrieval(
        self,
        query_keyframe: Keyframe
    ) -> List[LoopClosureCandidate]:
        """
        Stage 1: Global retrieval with Wasserstein distance

        Args:
            query_keyframe: Query keyframe

        Returns:
            List of top-K candidates (unverified)
        """
        # Spatial filtering: exclude nearby keyframes
        valid_indices = []

        for i, kf in enumerate(self.keyframes):
            # Spatial filter
            from data.pose_utils import euclidean_distance

            if query_keyframe.pose is not None and kf.pose is not None:
                dist = euclidean_distance(query_keyframe.pose, kf.pose)

                if dist < self.spatial_filter_distance:
                    continue  # Too close spatially

            valid_indices.append(i)

        if len(valid_indices) == 0:
            return []

        # Query retrieval database
        top_k = min(self.top_k, len(valid_indices))

        if top_k == 0:
            return []

        # Get top-K matches
        indices, distances = self.retriever.query(
            query_keyframe.descriptor,
            top_k=len(self.keyframes)  # Get all, will filter manually
        )

        # Filter to valid indices and take top-K
        candidates = []

        for idx, dist in zip(indices, distances):
            if idx in valid_indices:
                candidate = LoopClosureCandidate(
                    database_idx=int(idx),
                    distance=float(dist)
                )
                candidates.append(candidate)

                if len(candidates) >= top_k:
                    break

        return candidates

    def _geometric_verification(
        self,
        query_points: np.ndarray,
        candidates: List[LoopClosureCandidate]
    ) -> List[LoopClosureCandidate]:
        """
        Stage 2: Geometric verification with ICP/GICP

        Args:
            query_points: Query point cloud
            candidates: Unverified candidates from Stage 1

        Returns:
            Verified candidates
        """
        verified_candidates = []

        for candidate in candidates:
            # Get candidate point cloud
            candidate_kf = self.keyframes[candidate.database_idx]
            candidate_points = candidate_kf.points

            # Run ICP/GICP
            verified, transform, info = self.verifier.verify(
                query_points,
                candidate_points
            )

            # Update candidate
            candidate.verified = verified
            candidate.transform = transform
            candidate.fitness = info['fitness']
            candidate.rmse = info['rmse']
            candidate.information_matrix = info.get('information_matrix', None)

            if verified:
                verified_candidates.append(candidate)

        return verified_candidates

    def get_loop_closures(
        self,
        query_keyframe: Keyframe,
        query_points: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Get verified loop closures in g2o format

        Args:
            query_keyframe: Query keyframe
            query_points: Query point cloud

        Returns:
            List of loop closure dictionaries for g2o
        """
        candidates = self.query(
            query_keyframe,
            query_points=query_points,
            verify=True
        )

        loop_closures = []

        for candidate in candidates:
            if not candidate.verified:
                continue

            # Create loop closure edge
            candidate_kf = self.keyframes[candidate.database_idx]

            edge = compute_pose_graph_edge(
                source_pose=query_keyframe.pose,
                target_pose=candidate_kf.pose,
                relative_transform=candidate.transform,
                information_matrix=candidate.information_matrix
            )

            # Add IDs
            edge['source_id'] = query_keyframe.keyframe_id
            edge['target_id'] = candidate_kf.keyframe_id
            edge['fitness'] = candidate.fitness
            edge['rmse'] = candidate.rmse
            edge['wasserstein_distance'] = candidate.distance

            loop_closures.append(edge)

        return loop_closures

    def clear_database(self):
        """Clear database"""
        self.keyframes.clear()
        self.retriever.clear_database()


def create_two_stage_retrieval(
    top_k: int = 10,
    spatial_filter_distance: float = 50.0,
    use_gpu: bool = False
) -> TwoStageRetrieval:
    """
    Factory function to create two-stage retrieval

    Args:
        top_k: Number of candidates
        spatial_filter_distance: Spatial filtering threshold
        use_gpu: Use GPU for Wasserstein computation

    Returns:
        TwoStageRetrieval instance
    """
    return TwoStageRetrieval(
        top_k=top_k,
        spatial_filter_distance=spatial_filter_distance,
        use_torch=use_gpu,
        device='cuda' if use_gpu else 'cpu'
    )


def batch_loop_closing(
    query_keyframes: List[Keyframe],
    database_keyframes: List[Keyframe],
    top_k: int = 10,
    spatial_filter_distance: float = 50.0,
    verify: bool = True
) -> Dict[int, List[Dict]]:
    """
    Batch loop closing for multiple queries

    Args:
        query_keyframes: List of query keyframes
        database_keyframes: List of database keyframes
        top_k: Number of candidates per query
        spatial_filter_distance: Spatial filtering threshold
        verify: Run geometric verification

    Returns:
        Dictionary mapping query_idx -> list of loop closures
    """
    # Create retrieval system
    retrieval = create_two_stage_retrieval(
        top_k=top_k,
        spatial_filter_distance=spatial_filter_distance
    )

    # Add database keyframes
    for kf in database_keyframes:
        retrieval.add_keyframe(kf)

    # Query for each keyframe
    results = {}

    for i, query_kf in enumerate(query_keyframes):
        loop_closures = retrieval.get_loop_closures(query_kf)
        results[i] = loop_closures

    return results
