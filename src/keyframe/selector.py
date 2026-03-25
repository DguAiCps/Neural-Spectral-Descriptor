"""
Keyframe Selector - Algorithm 2

Implements incremental keyframe selection with 4-criterion strategy:
1. Distance > 0.5m
2. Rotation > 15°
3. Geometric novelty (IoU < 0.7 @ 0.2m)
4. Temporal > 5s

Maintains keyframe database and supports both online and offline processing.
Target: ~1Hz keyframe rate (10x reduction from 10Hz raw scans)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from keyframe.criteria import KeyframeSelectionCriteria


@dataclass
class Keyframe:
    """
    Keyframe data structure

    Contains all information for a selected keyframe:
    - Scan data
    - Pose and timestamp
    - Descriptor (computed later by encoder)
    """
    keyframe_id: int
    scan_id: int  # Original scan index in sequence
    points: np.ndarray  # (N, 3) or (N, 4) point cloud
    pose: np.ndarray  # (4, 4) SE(3) pose
    timestamp: float
    descriptor: Optional[np.ndarray] = None  # Spectral histogram (set later)
    embedding: Optional[np.ndarray] = None  # GNN embedding (set later)
    spectral_entropy: Optional[float] = None  # Shannon entropy of FFT magnitudes (set later)


class KeyframeSelector:
    """
    Manages incremental keyframe selection from LiDAR scans

    Uses 4-criterion strategy to select keyframes while traversing a sequence.
    Achieves ~10x compression (10Hz -> 1Hz keyframe rate).
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        rotation_threshold: float = 15.0,
        overlap_threshold: float = 0.7,
        temporal_threshold: float = 5.0,
        voxel_size: float = 0.2,
        max_keyframes: int = 10000
    ):
        """
        Initialize keyframe selector

        Args:
            distance_threshold: Min translation for keyframe (meters)
            rotation_threshold: Min rotation for keyframe (degrees)
            overlap_threshold: Max IoU for geometric novelty
            temporal_threshold: Min time between keyframes (seconds)
            voxel_size: Voxel size for overlap computation (meters)
            max_keyframes: Maximum number of keyframes to store
        """
        self.criteria = KeyframeSelectionCriteria(
            distance_threshold=distance_threshold,
            rotation_threshold=rotation_threshold,
            overlap_threshold=overlap_threshold,
            temporal_threshold=temporal_threshold,
            voxel_size=voxel_size
        )

        self.max_keyframes = max_keyframes

        # Keyframe database
        self.keyframes: List[Keyframe] = []
        self.keyframe_id_counter = 0

        # Last keyframe for incremental selection
        self.last_keyframe: Optional[Keyframe] = None

        # Statistics
        self.total_scans_processed = 0
        self.selection_details_history: List[dict] = []

    def reset(self):
        """Reset selector state"""
        self.keyframes.clear()
        self.keyframe_id_counter = 0
        self.last_keyframe = None
        self.total_scans_processed = 0
        self.selection_details_history.clear()

    def process_scan(
        self,
        scan_id: int,
        points: np.ndarray,
        pose: np.ndarray,
        timestamp: float,
        force_first: bool = True
    ) -> Tuple[bool, Optional[Keyframe], dict]:
        """
        Process a single scan and decide if it should be a keyframe

        Args:
            scan_id: Scan index in sequence
            points: (N, 3) or (N, 4) point cloud
            pose: (4, 4) SE(3) pose
            timestamp: Timestamp in seconds
            force_first: If True, force first scan to be keyframe

        Returns:
            selected: True if keyframe was created
            keyframe: Keyframe object if selected, None otherwise
            details: Dictionary with selection criteria details
        """
        self.total_scans_processed += 1

        # First scan is always a keyframe
        if self.last_keyframe is None:
            if force_first:
                keyframe = self._create_keyframe(scan_id, points, pose, timestamp)
                self.last_keyframe = keyframe
                self.keyframes.append(keyframe)

                details = {
                    'selected': True,
                    'reason': 'First keyframe',
                    'keyframe_id': keyframe.keyframe_id
                }

                self.selection_details_history.append(details)
                return True, keyframe, details
            else:
                return False, None, {'selected': False, 'reason': 'Not forcing first'}

        # Check selection criteria
        selected, details = self.criteria.should_select_keyframe(
            pose_current=pose,
            timestamp_current=timestamp,
            points_current=points,
            pose_last=self.last_keyframe.pose,
            timestamp_last=self.last_keyframe.timestamp,
            points_last=self.last_keyframe.points,
            require_all=False  # OR logic: any criterion triggers selection
        )

        if selected:
            # Create new keyframe
            keyframe = self._create_keyframe(scan_id, points, pose, timestamp)
            self.last_keyframe = keyframe
            self.keyframes.append(keyframe)

            # Enforce max keyframes limit
            if len(self.keyframes) > self.max_keyframes:
                # Remove oldest keyframe
                self.keyframes.pop(0)

            details['keyframe_id'] = keyframe.keyframe_id
            self.selection_details_history.append(details)

            return True, keyframe, details
        else:
            self.selection_details_history.append(details)
            return False, None, details

    def _create_keyframe(
        self,
        scan_id: int,
        points: np.ndarray,
        pose: np.ndarray,
        timestamp: float
    ) -> Keyframe:
        """
        Create a new keyframe

        Args:
            scan_id: Original scan index
            points: Point cloud
            pose: SE(3) pose
            timestamp: Timestamp

        Returns:
            Keyframe object
        """
        keyframe = Keyframe(
            keyframe_id=self.keyframe_id_counter,
            scan_id=scan_id,
            points=points,
            pose=pose,
            timestamp=timestamp
        )

        self.keyframe_id_counter += 1

        return keyframe

    def process_sequence(
        self,
        points_list: List[np.ndarray],
        poses: np.ndarray,
        timestamps: np.ndarray
    ) -> List[Keyframe]:
        """
        Process entire sequence offline

        Args:
            points_list: List of (N, 3) point clouds
            poses: (M, 4, 4) SE(3) poses
            timestamps: (M,) timestamps

        Returns:
            List of selected keyframes
        """
        self.reset()

        for scan_id in range(len(points_list)):
            self.process_scan(
                scan_id=scan_id,
                points=points_list[scan_id],
                pose=poses[scan_id],
                timestamp=timestamps[scan_id]
            )

        return self.keyframes

    def get_keyframe_by_id(self, keyframe_id: int) -> Optional[Keyframe]:
        """Get keyframe by ID"""
        for kf in self.keyframes:
            if kf.keyframe_id == keyframe_id:
                return kf
        return None

    def get_keyframe_by_scan_id(self, scan_id: int) -> Optional[Keyframe]:
        """Get keyframe by original scan ID"""
        for kf in self.keyframes:
            if kf.scan_id == scan_id:
                return kf
        return None

    def get_statistics(self) -> dict:
        """
        Get selection statistics

        Returns:
            Dictionary with statistics
        """
        if len(self.keyframes) == 0:
            return {
                'num_keyframes': 0,
                'num_scans': self.total_scans_processed,
                'compression_ratio': 0.0
            }

        # Compute compression ratio
        compression_ratio = self.total_scans_processed / len(self.keyframes)

        # Compute avg keyframe rate
        if len(self.keyframes) > 1:
            first_time = self.keyframes[0].timestamp
            last_time = self.keyframes[-1].timestamp
            duration = last_time - first_time

            if duration > 0:
                avg_rate = (len(self.keyframes) - 1) / duration
            else:
                avg_rate = 0.0
        else:
            avg_rate = 0.0

        # Analyze criteria satisfaction
        criteria_counts = {
            'distance': 0,
            'rotation': 0,
            'temporal': 0,
            'geometric': 0
        }

        for details in self.selection_details_history:
            if details.get('selected', False):
                if 'distance' in details and details['distance']['satisfied']:
                    criteria_counts['distance'] += 1
                if 'rotation' in details and details['rotation']['satisfied']:
                    criteria_counts['rotation'] += 1
                if 'temporal' in details and details['temporal']['satisfied']:
                    criteria_counts['temporal'] += 1
                if 'geometric' in details and details['geometric'].get('satisfied', False):
                    criteria_counts['geometric'] += 1

        return {
            'num_keyframes': len(self.keyframes),
            'num_scans': self.total_scans_processed,
            'compression_ratio': compression_ratio,
            'avg_keyframe_rate_hz': avg_rate,
            'criteria_counts': criteria_counts
        }

    def export_keyframe_poses(self) -> np.ndarray:
        """
        Export keyframe poses as array

        Returns:
            (K, 4, 4) array of SE(3) poses
        """
        poses = np.array([kf.pose for kf in self.keyframes])
        return poses

    def export_keyframe_timestamps(self) -> np.ndarray:
        """
        Export keyframe timestamps

        Returns:
            (K,) array of timestamps
        """
        timestamps = np.array([kf.timestamp for kf in self.keyframes])
        return timestamps

    def export_keyframe_descriptors(self) -> Optional[np.ndarray]:
        """
        Export keyframe descriptors (if computed)

        Returns:
            (K, D) array of descriptors or None if not computed
        """
        if len(self.keyframes) == 0:
            return None

        if self.keyframes[0].descriptor is None:
            return None

        descriptors = np.array([kf.descriptor for kf in self.keyframes])
        return descriptors

    def attach_descriptors(self, descriptors: np.ndarray):
        """
        Attach descriptors to keyframes

        Args:
            descriptors: (K, D) array of descriptors
        """
        assert len(descriptors) == len(self.keyframes), \
            f"Descriptor count ({len(descriptors)}) != keyframe count ({len(self.keyframes)})"

        for i, kf in enumerate(self.keyframes):
            kf.descriptor = descriptors[i]

    def attach_embeddings(self, embeddings: np.ndarray):
        """
        Attach GNN embeddings to keyframes

        Args:
            embeddings: (K, D) array of embeddings
        """
        assert len(embeddings) == len(self.keyframes), \
            f"Embedding count ({len(embeddings)}) != keyframe count ({len(self.keyframes)})"

        for i, kf in enumerate(self.keyframes):
            kf.embedding = embeddings[i]


def select_keyframes_from_kitti(
    kitti_loader,
    distance_threshold: float = 0.5,
    rotation_threshold: float = 15.0,
    overlap_threshold: float = 0.7,
    temporal_threshold: float = 5.0
) -> List[Keyframe]:
    """
    Convenience function to select keyframes from KITTI sequence

    Args:
        kitti_loader: KITTILoader instance
        distance_threshold: Distance criterion
        rotation_threshold: Rotation criterion
        overlap_threshold: Geometric novelty criterion
        temporal_threshold: Temporal criterion

    Returns:
        List of selected keyframes
    """
    selector = KeyframeSelector(
        distance_threshold=distance_threshold,
        rotation_threshold=rotation_threshold,
        overlap_threshold=overlap_threshold,
        temporal_threshold=temporal_threshold
    )

    # Process all scans
    for scan_id in range(len(kitti_loader)):
        data = kitti_loader[scan_id]

        selector.process_scan(
            scan_id=scan_id,
            points=data['points'],
            pose=data['pose'],
            timestamp=data['timestamp']
        )

    # Print statistics
    stats = selector.get_statistics()
    print(f"Selected {stats['num_keyframes']} keyframes from {stats['num_scans']} scans")
    print(f"Compression ratio: {stats['compression_ratio']:.1f}x")
    print(f"Avg keyframe rate: {stats['avg_keyframe_rate_hz']:.2f} Hz")

    return selector.keyframes
