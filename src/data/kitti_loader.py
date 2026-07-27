"""
KITTI Dataset Loader for LiDAR Loop Closing

Loads KITTI odometry sequences with point clouds and ground truth poses.
Supports lazy loading of point clouds to reduce memory usage.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


class KITTILoader:
    """
    KITTI Odometry Dataset Loader

    Directory structure expected:
    data_root/
        sequences/
            00/
                velodyne/
                    000000.bin
                    000001.bin
                    ...
                poses.txt (or calib.txt + poses.txt for ground truth)
    """

    def __init__(self, data_root: str, sequence: str, lazy_load: bool = True):
        """
        Initialize KITTI loader

        Args:
            data_root: Path to KITTI dataset root (contains 'sequences' folder)
            sequence: Sequence ID as string (e.g., '00', '05', '09')
            lazy_load: If True, load point clouds on demand; if False, preload all
        """
        self.data_root = Path(data_root)
        self.sequence = sequence
        self.lazy_load = lazy_load

        self.sequence_path = self.data_root / "sequences" / sequence
        self.velodyne_path = self.sequence_path / "velodyne"

        # Check poses.txt in multiple locations
        # 1. In sequence directory (with symlink): sequences/00/poses.txt
        # 2. In poses directory: poses/00.txt
        self.poses_file = self.sequence_path / "poses.txt"
        if not self.poses_file.exists():
            poses_alt = self.data_root / "poses" / f"{sequence}.txt"
            if poses_alt.exists():
                self.poses_file = poses_alt

        # Validate paths
        if not self.sequence_path.exists():
            raise FileNotFoundError(f"Sequence path not found: {self.sequence_path}")
        if not self.velodyne_path.exists():
            raise FileNotFoundError(f"Velodyne path not found: {self.velodyne_path}")

        # Load frame list
        self.frame_files = sorted(self.velodyne_path.glob("*.bin"))
        self.num_frames = len(self.frame_files)

        if self.num_frames == 0:
            raise ValueError(f"No .bin files found in {self.velodyne_path}")

        # Load poses if available
        self.poses = self._load_poses()

        # Generate timestamps (KITTI is 10Hz)
        self.timestamps = np.arange(self.num_frames) * 0.1  # 10Hz = 0.1s interval

        # Preload point clouds if not lazy
        if not lazy_load:
            print(f"Preloading {self.num_frames} point clouds...")
            self.point_clouds = [self._load_point_cloud(i) for i in range(self.num_frames)]
        else:
            self.point_clouds = None

    def _load_poses(self) -> Optional[np.ndarray]:
        """
        Load ground truth poses from poses.txt

        Returns:
            Array of shape (N, 4, 4) containing SE(3) transformation matrices
            or None if poses file doesn't exist
        """
        if not self.poses_file.exists():
            print(f"Warning: Poses file not found at {self.poses_file}")
            return None

        poses = []
        with open(self.poses_file, 'r') as f:
            for line in f:
                # Each line contains 12 values representing a 3x4 transformation matrix
                values = np.array([float(x) for x in line.strip().split()])
                if len(values) != 12:
                    continue

                # Reshape to 3x4 and add bottom row [0, 0, 0, 1]
                pose_3x4 = values.reshape(3, 4)
                pose = np.eye(4)
                pose[:3, :] = pose_3x4
                poses.append(pose)

        return np.array(poses) if poses else None

    def _load_point_cloud(self, idx: int) -> np.ndarray:
        """
        Load a single point cloud from bin file

        Args:
            idx: Frame index

        Returns:
            Point cloud array of shape (N, 4) with [x, y, z, intensity]
        """
        bin_file = self.frame_files[idx]

        # KITTI .bin format: float32 array with [x, y, z, intensity] per point
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

        return points

    def __len__(self) -> int:
        """Return number of frames in sequence"""
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        """
        Get frame data by index

        Args:
            idx: Frame index

        Returns:
            Dictionary containing:
                - 'points': (N, 4) array [x, y, z, intensity]
                - 'pose': (4, 4) SE(3) transformation matrix (if available)
                - 'timestamp': float timestamp in seconds
                - 'idx': frame index
        """
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of range [0, {self.num_frames})")

        # Load point cloud (from cache or file)
        if self.lazy_load:
            points = self._load_point_cloud(idx)
        else:
            points = self.point_clouds[idx]

        result = {
            'points': points,
            'timestamp': self.timestamps[idx],
            'idx': idx,
        }

        # Add pose if available
        if self.poses is not None:
            result['pose'] = self.poses[idx]

        return result

    def get_point_cloud(self, idx: int) -> np.ndarray:
        """
        Get point cloud by index (convenience method)

        Args:
            idx: Frame index

        Returns:
            Point cloud array of shape (N, 4)
        """
        return self.__getitem__(idx)['points']

    def get_pose(self, idx: int) -> Optional[np.ndarray]:
        """
        Get pose by index (convenience method)

        Args:
            idx: Frame index

        Returns:
            (4, 4) SE(3) transformation matrix or None
        """
        if self.poses is None:
            return None
        return self.poses[idx]

    def get_relative_pose(self, idx1: int, idx2: int) -> Optional[np.ndarray]:
        """
        Compute relative pose from frame idx1 to idx2

        Args:
            idx1: Source frame index
            idx2: Target frame index

        Returns:
            (4, 4) relative transformation T_12 = T_1^{-1} @ T_2
            or None if poses not available
        """
        if self.poses is None:
            return None

        T1 = self.poses[idx1]
        T2 = self.poses[idx2]

        # T_12 transforms points from frame 2 to frame 1
        T_12 = np.linalg.inv(T1) @ T2

        return T_12

    def get_distance(self, idx1: int, idx2: int) -> Optional[float]:
        """
        Compute Euclidean distance between two frames

        Args:
            idx1: First frame index
            idx2: Second frame index

        Returns:
            Euclidean distance in meters or None if poses not available
        """
        if self.poses is None:
            return None

        pos1 = self.poses[idx1][:3, 3]
        pos2 = self.poses[idx2][:3, 3]

        return np.linalg.norm(pos2 - pos1)

    def get_frames_in_range(
        self,
        query_idx: int,
        min_distance: float,
        max_distance: float
    ) -> List[int]:
        """
        Get all frame indices within a distance range from query frame

        Args:
            query_idx: Query frame index
            min_distance: Minimum distance in meters
            max_distance: Maximum distance in meters

        Returns:
            List of frame indices within the distance range
        """
        if self.poses is None:
            return []

        query_pos = self.poses[query_idx][:3, 3]

        indices = []
        for i in range(self.num_frames):
            if i == query_idx:
                continue

            pos = self.poses[i][:3, 3]
            dist = np.linalg.norm(pos - query_pos)

            if min_distance <= dist <= max_distance:
                indices.append(i)

        return indices


def load_kitti_sequence(
    data_root: str,
    sequence: str,
    lazy_load: bool = True
) -> KITTILoader:
    """
    Factory function to load a KITTI sequence

    Args:
        data_root: Path to KITTI dataset root
        sequence: Sequence ID (e.g., '00', '05')
        lazy_load: Whether to use lazy loading

    Returns:
        KITTILoader instance
    """
    return KITTILoader(data_root, sequence, lazy_load)
