"""
HeLiPR Dataset Loader

Loads Velodyne VLP-16 point clouds from HeLiPR dataset.
Format: 22 bytes per point (x,y,z,intensity as float32, ring as uint16, time as float32)
"""

import numpy as np
from pathlib import Path
from typing import Optional
from scipy.spatial.transform import Rotation


class HeLiPRLoader:
    """
    HeLiPR Velodyne VLP-16 data loader

    Args:
        root: Path to HeLiPR sequence (e.g., /data/helipr/Sample)
        lazy_load: If True, load point clouds on demand
    """

    def __init__(self, root: str, lazy_load: bool = True):
        self.root = Path(root)
        self.lazy_load = lazy_load

        # Paths
        self.velodyne_dir = self.root / "LiDAR" / "Velodyne"
        self.gt_file = self.root / "LiDAR_GT" / "Velodyne_gt.txt"

        if not self.velodyne_dir.exists():
            raise ValueError(f"Velodyne directory not found: {self.velodyne_dir}")

        # Load ground truth poses
        self._load_ground_truth()

        # Get scan files
        self._get_scan_files()

        # Pre-load if not lazy
        if not lazy_load:
            self._preload_all()

        print(f"HeLiPR: Loaded {len(self.scan_files)} scans from {root}")

    def _load_ground_truth(self):
        """Load ground truth poses from Velodyne_gt.txt"""
        self.timestamps_gt = []
        self.poses = []

        if not self.gt_file.exists():
            raise ValueError(f"Ground truth file not found: {self.gt_file}")

        with open(self.gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue

                timestamp = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                # Convert quaternion to rotation matrix
                rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

                # Build SE(3) matrix
                pose = np.eye(4)
                pose[:3, :3] = rot
                pose[:3, 3] = [x, y, z]

                self.timestamps_gt.append(timestamp)
                self.poses.append(pose)

        self.timestamps_gt = np.array(self.timestamps_gt)
        self.poses = np.array(self.poses)

    def _get_scan_files(self):
        """Get list of scan files and match with poses"""
        # Get all .bin files
        all_files = sorted(self.velodyne_dir.glob("*.bin"))

        self.scan_files = []
        self.scan_timestamps = []
        self.scan_poses = []
        self.scan_pose_indices = []

        for f in all_files:
            # Extract timestamp from filename
            timestamp = int(f.stem)

            # Find closest GT pose using binary search
            idx = np.searchsorted(self.timestamps_gt, timestamp)
            idx = np.clip(idx, 0, len(self.timestamps_gt) - 1)

            # Check if close enough (within 100ms = 100,000,000 ns)
            time_diff = abs(timestamp - self.timestamps_gt[idx])
            if time_diff > 100_000_000:
                # Also check idx-1
                if idx > 0:
                    time_diff_prev = abs(timestamp - self.timestamps_gt[idx-1])
                    if time_diff_prev < time_diff:
                        idx = idx - 1
                        time_diff = time_diff_prev

            if time_diff <= 100_000_000:  # 100ms tolerance
                self.scan_files.append(f)
                self.scan_timestamps.append(timestamp)
                self.scan_poses.append(self.poses[idx])
                self.scan_pose_indices.append(idx)

        self.scan_poses = np.array(self.scan_poses)

    def _preload_all(self):
        """Pre-load all point clouds"""
        self.point_clouds = []
        for f in self.scan_files:
            self.point_clouds.append(self._load_velodyne(f))

    def _load_velodyne(self, filepath: Path) -> np.ndarray:
        """
        Load Velodyne VLP-16 point cloud

        Format: 22 bytes per point
        - x, y, z: float32 (4 bytes each)
        - intensity: float32 (4 bytes)
        - ring: uint16 (2 bytes)
        - time: float32 (4 bytes)

        Returns:
            (N, 4) array [x, y, z, intensity]
        """
        dt = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
            ('ring', np.uint16),
            ('time', np.float32)
        ])

        data = np.fromfile(filepath, dtype=dt)

        # Extract x, y, z, intensity
        points = np.stack([
            data['x'],
            data['y'],
            data['z'],
            data['intensity']
        ], axis=-1).astype(np.float32)

        return points

    def __len__(self) -> int:
        return len(self.scan_files)

    def __getitem__(self, idx: int) -> dict:
        """
        Get scan data by index

        Returns:
            dict with:
                - 'points': (N, 4) array [x, y, z, intensity]
                - 'pose': (4, 4) SE(3) transformation
                - 'timestamp': float timestamp in seconds
                - 'idx': frame index
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Load point cloud
        if self.lazy_load:
            points = self._load_velodyne(self.scan_files[idx])
        else:
            points = self.point_clouds[idx]

        # Convert timestamp to seconds
        timestamp_sec = self.scan_timestamps[idx] / 1e9

        return {
            'points': points,
            'pose': self.scan_poses[idx],
            'timestamp': timestamp_sec,
            'idx': idx
        }

    def get_all_poses(self) -> np.ndarray:
        """Get all poses as (N, 4, 4) array"""
        return self.scan_poses
