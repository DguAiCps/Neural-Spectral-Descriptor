"""
NCLT Dataset Loader for LiDAR Loop Closing

Loads NCLT (Michigan Campus) dataset with point clouds and ground truth poses.
NCLT uses Velodyne HDL-32E (32-ring) sensor with seasonal variations.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List


class NCLTLoader:
    """
    NCLT Dataset Loader

    Directory structure expected:
    data_root/
        2012-01-08/
            velodyne_sync/
                1326059182636482.bin
                1326059182736482.bin
                ...
        ground_truth/
            2012-01-08.csv
    """

    def __init__(self, data_root: str, date: str, lazy_load: bool = True):
        """
        Initialize NCLT loader

        Args:
            data_root: Path to NCLT dataset root
            date: Date string (e.g., '2012-01-08')
            lazy_load: If True, load point clouds on demand
        """
        self.data_root = Path(data_root)
        self.date = date
        self.lazy_load = lazy_load

        self.sequence_path = self.data_root / date
        self.velodyne_path = self.sequence_path / "velodyne_sync"
        self.gt_file = self.sequence_path / f"groundtruth_{date}.csv"

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

        # Extract timestamps from filenames (microseconds)
        self.timestamps = self._extract_timestamps()

        # Load ground truth poses if available
        self.poses = self._load_poses()

        # Preload point clouds if not lazy
        if not lazy_load:
            print(f"Preloading {self.num_frames} point clouds...")
            self.point_clouds = [self._load_point_cloud(i) for i in range(self.num_frames)]
        else:
            self.point_clouds = None

    def _extract_timestamps(self) -> np.ndarray:
        """
        Extract timestamps from NCLT filenames

        NCLT filenames are microsecond timestamps: 1326059182636482.bin

        Returns:
            Array of timestamps in seconds
        """
        timestamps = []
        for f in self.frame_files:
            # Extract timestamp from filename (microseconds)
            ts_us = int(f.stem)
            # Convert to seconds
            ts_s = ts_us / 1e6
            timestamps.append(ts_s)

        timestamps = np.array(timestamps)
        # Normalize to start from 0
        timestamps = timestamps - timestamps[0]

        return timestamps

    def _load_poses(self) -> Optional[np.ndarray]:
        """
        Load ground truth poses from CSV file

        NCLT ground truth format (no header):
        timestamp (us), x, y, z, roll, pitch, yaw

        Returns:
            Array of shape (N, 4, 4) containing SE(3) transformation matrices
            or None if poses file doesn't exist
        """
        if not self.gt_file.exists():
            print(f"Warning: Ground truth file not found at {self.gt_file}")
            return None

        try:
            # Read CSV (no header) without dtype specification
            df = pd.read_csv(self.gt_file, header=None, low_memory=False)

            # Assign column names
            df.columns = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']

            # Remove rows with NaN values
            df = df.dropna()

            # Convert to numeric, coercing errors
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['x'] = pd.to_numeric(df['x'], errors='coerce')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df['z'] = pd.to_numeric(df['z'], errors='coerce')
            df['roll'] = pd.to_numeric(df['roll'], errors='coerce')
            df['pitch'] = pd.to_numeric(df['pitch'], errors='coerce')
            df['yaw'] = pd.to_numeric(df['yaw'], errors='coerce')

            # Drop any rows that couldn't be converted
            df = df.dropna()

            # Extract pose columns
            timestamps_gt = df['timestamp'].values.astype(np.int64)
            x = df['x'].values.astype(np.float64)
            y = df['y'].values.astype(np.float64)
            z = df['z'].values.astype(np.float64)
            roll = df['roll'].values.astype(np.float64)
            pitch = df['pitch'].values.astype(np.float64)
            yaw = df['yaw'].values.astype(np.float64)

            # Create a mapping from velodyne timestamp to ground truth pose
            # We need to match velodyne timestamps to ground truth timestamps
            self.gt_timestamps = timestamps_gt

            # OPTIMIZED: Use searchsorted for O(n log m) instead of O(n × m)
            vel_timestamps = np.array([int(f.stem) for f in self.frame_files], dtype=np.int64)

            # searchsorted finds insertion points in sorted array
            insert_indices = np.searchsorted(timestamps_gt, vel_timestamps)

            # Clamp to valid range
            insert_indices = np.clip(insert_indices, 1, len(timestamps_gt) - 1)

            # Compare with left and right neighbors to find nearest
            left_indices = insert_indices - 1
            right_indices = insert_indices

            left_diff = np.abs(vel_timestamps - timestamps_gt[left_indices])
            right_diff = np.abs(vel_timestamps - timestamps_gt[right_indices])

            # Choose closer one
            nearest_indices = np.where(left_diff <= right_diff, left_indices, right_indices)

            # Vectorized pose construction
            poses = np.array([
                self._euler_to_se3(x[idx], y[idx], z[idx], roll[idx], pitch[idx], yaw[idx])
                for idx in nearest_indices
            ])

            print(f"Loaded {len(poses)} matched poses (from {len(timestamps_gt)} ground truth)")
            return poses

        except Exception as e:
            print(f"Warning: Failed to load ground truth: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _euler_to_se3(self, x, y, z, roll, pitch, yaw) -> np.ndarray:
        """
        Convert Euler angles to SE(3) transformation matrix

        Args:
            x, y, z: Translation
            roll, pitch, yaw: Rotation angles in radians

        Returns:
            4x4 SE(3) matrix
        """
        # Rotation matrices
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Combined rotation matrix (ZYX order)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])

        # SE(3) matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        return T

    def _load_point_cloud(self, idx: int) -> np.ndarray:
        """
        Load a single point cloud from NCLT bin file (vectorized)

        NCLT actual format (12 bytes per point):
        - x, y, z: uint16 (stored as cm, centered at 32767)
        - intensity: uint8
        - padding: uint8
        - extra: uint32 (timestamp or ring)

        Args:
            idx: Frame index

        Returns:
            Point cloud array of shape (N, 4) with [x, y, z, intensity]
        """
        bin_file = self.frame_files[idx]

        try:
            # Define structured dtype for NCLT format (12 bytes per point)
            nclt_dtype = np.dtype([
                ('x', '<u2'),      # uint16 little-endian
                ('y', '<u2'),
                ('z', '<u2'),
                ('intensity', 'u1'),  # uint8
                ('padding', 'u1'),
                ('extra', '<u4')   # uint32
            ])

            # Read entire file at once with numpy (vectorized)
            raw_data = np.fromfile(bin_file, dtype=nclt_dtype)

            if len(raw_data) == 0:
                return np.zeros((0, 4), dtype=np.float32)

            # Vectorized conversion to meters (NCLT format: scale 0.005, offset -100)
            # Reference: https://github.com/aljosaosep/NCLT-dataset-tools
            x = raw_data['x'].astype(np.float32) * 0.005 - 100.0
            y = raw_data['y'].astype(np.float32) * 0.005 - 100.0
            z = raw_data['z'].astype(np.float32) * 0.005 - 100.0
            # NCLT stores points in the body frame with z pointing down;
            # convert to z-up to match the KITTI/HeLiPR/MulRan convention.
            z = -z
            intensity = raw_data['intensity'].astype(np.float32) / 255.0

            # Stack into (N, 4) array
            points = np.column_stack([x, y, z, intensity])

        except Exception as e:
            print(f"Warning: Failed to parse NCLT format: {e}")
            return np.zeros((0, 4), dtype=np.float32)

        # Filter invalid points (vectorized)
        if len(points) > 0:
            valid_mask = (
                np.isfinite(points[:, :3]).all(axis=1) &
                (np.abs(points[:, 0]) < 200.0) &
                (np.abs(points[:, 1]) < 200.0) &
                (np.abs(points[:, 2]) < 200.0)
            )
            points = points[valid_mask]

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
                - 'dataset': 'nclt'
        """
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of range [0, {self.num_frames})")

        # Load point cloud
        if self.lazy_load:
            points = self._load_point_cloud(idx)
        else:
            points = self.point_clouds[idx]

        result = {
            'points': points,
            'timestamp': self.timestamps[idx],
            'idx': idx,
            'dataset': 'nclt',
            'date': self.date
        }

        # Add pose if available
        if self.poses is not None and idx < len(self.poses):
            result['pose'] = self.poses[idx]
        else:
            result['pose'] = np.eye(4)  # Identity if not available

        return result

    def get_sequence_info(self) -> dict:
        """Get information about this sequence"""
        return {
            'dataset': 'nclt',
            'date': self.date,
            'num_frames': self.num_frames,
            'duration': self.timestamps[-1] if len(self.timestamps) > 0 else 0.0,
            'has_poses': self.poses is not None,
            'sensor': 'Velodyne HDL-32E',
            'rings': 32
        }
