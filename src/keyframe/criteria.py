"""
Keyframe Selection Criteria

Implements the 4-criterion keyframe selection strategy:
1. Distance criterion: translation > 0.5m
2. Rotation criterion: rotation > 15°
3. Geometric novelty: IoU < 0.7 @ 0.2m voxel resolution
4. Temporal criterion: time difference > 5s

These criteria prevent redundant keyframes while ensuring adequate coverage.
"""

import numpy as np
from typing import Optional, Tuple
from data.pose_utils import euclidean_distance, rotation_angle_degrees, compute_overlap


class KeyframeSelectionCriteria:
    """
    Manages keyframe selection criteria

    Uses 4 criteria to determine if a new scan should become a keyframe:
    - Distance: translation from last keyframe
    - Rotation: rotation from last keyframe
    - Geometric novelty: overlap with last keyframe
    - Temporal: time since last keyframe
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        rotation_threshold: float = 15.0,
        overlap_threshold: float = 0.7,
        temporal_threshold: float = 5.0,
        voxel_size: float = 0.2
    ):
        """
        Initialize selection criteria

        Args:
            distance_threshold: Minimum translation in meters (default: 0.5m)
            rotation_threshold: Minimum rotation in degrees (default: 15°)
            overlap_threshold: Maximum IoU for geometric novelty (default: 0.7)
            temporal_threshold: Minimum time difference in seconds (default: 5s)
            voxel_size: Voxel size for overlap computation (default: 0.2m)
        """
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.overlap_threshold = overlap_threshold
        self.temporal_threshold = temporal_threshold
        self.voxel_size = voxel_size

    def check_distance(
        self,
        pose_current: np.ndarray,
        pose_last: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if distance criterion is satisfied

        Args:
            pose_current: (4, 4) current pose
            pose_last: (4, 4) last keyframe pose

        Returns:
            satisfied: True if distance > threshold
            distance: Actual distance in meters
        """
        distance = euclidean_distance(pose_current, pose_last)
        satisfied = distance > self.distance_threshold

        return satisfied, distance

    def check_rotation(
        self,
        pose_current: np.ndarray,
        pose_last: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if rotation criterion is satisfied

        Args:
            pose_current: (4, 4) current pose
            pose_last: (4, 4) last keyframe pose

        Returns:
            satisfied: True if rotation > threshold
            rotation: Actual rotation in degrees
        """
        rotation = rotation_angle_degrees(pose_current, pose_last)
        satisfied = rotation > self.rotation_threshold

        return satisfied, rotation

    def check_geometric_novelty(
        self,
        points_current: np.ndarray,
        points_last: np.ndarray,
        pose_current: np.ndarray,
        pose_last: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if geometric novelty criterion is satisfied

        Computes IoU (Intersection over Union) between current and last point clouds.
        Novel geometry means IoU < threshold (low overlap).

        Args:
            points_current: (N, 3) current point cloud
            points_last: (M, 3) last keyframe point cloud
            pose_current: (4, 4) current pose
            pose_last: (4, 4) last keyframe pose

        Returns:
            satisfied: True if IoU < threshold (novel geometry)
            overlap: Actual IoU value
        """
        # Compute relative transformation from last to current
        from data.pose_utils import relative_pose
        T_rel = relative_pose(pose_last, pose_current)

        # Compute overlap
        overlap = compute_overlap(
            points_last,
            points_current,
            T_rel,
            voxel_size=self.voxel_size
        )

        # Novel if overlap is LOW (< threshold)
        satisfied = overlap < self.overlap_threshold

        return satisfied, overlap

    def check_temporal(
        self,
        timestamp_current: float,
        timestamp_last: float
    ) -> Tuple[bool, float]:
        """
        Check if temporal criterion is satisfied

        Args:
            timestamp_current: Current timestamp in seconds
            timestamp_last: Last keyframe timestamp in seconds

        Returns:
            satisfied: True if time difference > threshold
            time_diff: Actual time difference in seconds
        """
        time_diff = abs(timestamp_current - timestamp_last)
        satisfied = time_diff > self.temporal_threshold

        return satisfied, time_diff

    def should_select_keyframe(
        self,
        pose_current: np.ndarray,
        timestamp_current: float,
        points_current: Optional[np.ndarray],
        pose_last: np.ndarray,
        timestamp_last: float,
        points_last: Optional[np.ndarray],
        require_all: bool = False
    ) -> Tuple[bool, dict]:
        """
        Determine if current scan should be a keyframe (optimized with early termination)

        Args:
            pose_current: (4, 4) current pose
            timestamp_current: Current timestamp
            points_current: (N, 3) current point cloud (optional for geometric check)
            pose_last: (4, 4) last keyframe pose
            timestamp_last: Last keyframe timestamp
            points_last: (M, 3) last keyframe point cloud (optional for geometric check)
            require_all: If True, all criteria must be satisfied; if False, any criterion

        Returns:
            selected: True if should be keyframe
            details: Dictionary with criterion results
        """
        details = {}

        # Check distance (fast)
        dist_satisfied, dist_value = self.check_distance(pose_current, pose_last)
        details['distance'] = {
            'satisfied': dist_satisfied,
            'value': dist_value,
            'threshold': self.distance_threshold
        }

        # Check rotation (fast)
        rot_satisfied, rot_value = self.check_rotation(pose_current, pose_last)
        details['rotation'] = {
            'satisfied': rot_satisfied,
            'value': rot_value,
            'threshold': self.rotation_threshold
        }

        # Check temporal (fast)
        temp_satisfied, temp_value = self.check_temporal(timestamp_current, timestamp_last)
        details['temporal'] = {
            'satisfied': temp_satisfied,
            'value': temp_value,
            'threshold': self.temporal_threshold
        }

        # Determine selection (NO geometric check - removed for memory efficiency)
        if require_all:
            # AND logic: all criteria must be satisfied
            selected = dist_satisfied and rot_satisfied and temp_satisfied
        else:
            # OR logic: any criterion triggers selection
            selected = dist_satisfied or rot_satisfied or temp_satisfied

        details['selected'] = selected

        return selected, details


def estimate_keyframe_rate(
    distance_threshold: float = 0.5,
    rotation_threshold: float = 15.0,
    avg_velocity: float = 5.0,
    avg_angular_velocity: float = 10.0
) -> float:
    """
    Estimate keyframe selection rate based on criteria and motion

    Args:
        distance_threshold: Distance threshold in meters
        rotation_threshold: Rotation threshold in degrees
        avg_velocity: Average linear velocity in m/s
        avg_angular_velocity: Average angular velocity in deg/s

    Returns:
        Estimated keyframe rate in Hz
    """
    # Time to satisfy distance criterion
    time_distance = distance_threshold / avg_velocity if avg_velocity > 0 else float('inf')

    # Time to satisfy rotation criterion
    time_rotation = rotation_threshold / avg_angular_velocity if avg_angular_velocity > 0 else float('inf')

    # Keyframe selected when first criterion is satisfied (OR logic)
    time_to_keyframe = min(time_distance, time_rotation)

    # Rate = 1 / time
    keyframe_rate = 1.0 / time_to_keyframe if time_to_keyframe > 0 else 0.0

    return keyframe_rate


def analyze_keyframe_spacing(
    poses: np.ndarray,
    timestamps: np.ndarray,
    selected_indices: np.ndarray
) -> dict:
    """
    Analyze spacing between selected keyframes

    Args:
        poses: (N, 4, 4) all poses
        timestamps: (N,) all timestamps
        selected_indices: (K,) indices of selected keyframes

    Returns:
        Dictionary with statistics
    """
    if len(selected_indices) < 2:
        return {
            'num_keyframes': len(selected_indices),
            'mean_distance': 0.0,
            'mean_time': 0.0
        }

    distances = []
    time_diffs = []

    for i in range(len(selected_indices) - 1):
        idx1 = selected_indices[i]
        idx2 = selected_indices[i + 1]

        # Distance
        dist = euclidean_distance(poses[idx1], poses[idx2])
        distances.append(dist)

        # Time difference
        time_diff = timestamps[idx2] - timestamps[idx1]
        time_diffs.append(time_diff)

    return {
        'num_keyframes': len(selected_indices),
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'mean_time': np.mean(time_diffs),
        'std_time': np.std(time_diffs),
        'min_time': np.min(time_diffs),
        'max_time': np.max(time_diffs),
        'avg_keyframe_rate': 1.0 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0.0
    }
