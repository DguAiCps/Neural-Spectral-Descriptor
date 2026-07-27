"""
MulRan Dataset Loader

Loads Ouster OS1-64 point clouds from the MulRan dataset.

Directory structure:
    root/
        <sequence>/                 e.g. DCC01, KAIST02, Riverside03, Sejong01
            Ouster/
                <ns_timestamp>.bin  raw float32 xyzi (65536 points per scan)
            global_pose.csv         pose per high-rate timestamp
                format: ts, r11,r12,r13,tx, r21,r22,r23,ty, r31,r32,r33,tz
            ouster_front_stamp.csv  list of Ouster scan timestamps (optional)

Sensor: Ouster OS1-64 (64 channels, vertical FoV ±16.6°)
"""

import numpy as np
from pathlib import Path
from typing import Optional, List


class MulRanLoader:
    """
    MulRan Ouster OS1-64 data loader

    Args:
        root: Path to MulRan dataset root (contains <sequence> folders)
        sequence: Sequence name (e.g. 'DCC01', 'KAIST02')
        lazy_load: If True, load point clouds on demand
        pose_time_tolerance_ns: Max nanoseconds between scan and pose timestamps
    """

    POINTS_PER_SCAN = 65536  # 64 channels × 1024 columns

    def __init__(
        self,
        root: str,
        sequence: str,
        lazy_load: bool = True,
        pose_time_tolerance_ns: int = 100_000_000,  # 100ms
    ):
        self.root = Path(root)
        self.sequence = sequence
        self.lazy_load = lazy_load
        self.pose_time_tolerance_ns = pose_time_tolerance_ns

        self.sequence_path = self.root / sequence
        self.ouster_dir = self.sequence_path / "Ouster"
        self.pose_file = self.sequence_path / "global_pose.csv"

        if not self.sequence_path.exists():
            raise FileNotFoundError(f"MulRan sequence path not found: {self.sequence_path}")
        if not self.ouster_dir.exists():
            raise FileNotFoundError(f"Ouster directory not found: {self.ouster_dir}")
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")

        self._load_poses()
        self._match_scans_to_poses()

        if not lazy_load:
            self.point_clouds = [self._load_point_cloud(i) for i in range(len(self.scan_files))]
        else:
            self.point_clouds = None

        print(f"MulRan: Loaded {len(self.scan_files)} scans from {self.sequence_path}")

    def _load_poses(self):
        """
        Load poses from global_pose.csv.

        Each line: ts, r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz
        Poses are for the Ouster sensor in the world frame.
        """
        ts_list: List[int] = []
        pose_list: List[np.ndarray] = []

        with open(self.pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 13:
                    continue
                try:
                    ts = int(parts[0])
                    vals = np.array([float(x) for x in parts[1:13]], dtype=np.float64)
                except ValueError:
                    continue
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :] = vals.reshape(3, 4)
                ts_list.append(ts)
                pose_list.append(pose)

        if not ts_list:
            raise ValueError(f"No valid poses parsed from {self.pose_file}")

        self.pose_timestamps = np.array(ts_list, dtype=np.int64)
        self.all_poses = np.stack(pose_list, axis=0)

        # global_pose.csv is time-sorted in MulRan, but enforce it for searchsorted
        order = np.argsort(self.pose_timestamps)
        self.pose_timestamps = self.pose_timestamps[order]
        self.all_poses = self.all_poses[order]

    def _match_scans_to_poses(self):
        """
        Match each Ouster scan to the nearest pose via binary search.
        Scans without a pose within tolerance are dropped.
        """
        all_bins = sorted(self.ouster_dir.glob("*.bin"))

        self.scan_files: List[Path] = []
        self.scan_timestamps: List[int] = []
        scan_poses: List[np.ndarray] = []

        for f in all_bins:
            try:
                ts = int(f.stem)
            except ValueError:
                continue

            idx = np.searchsorted(self.pose_timestamps, ts)
            idx = int(np.clip(idx, 0, len(self.pose_timestamps) - 1))
            time_diff = abs(ts - int(self.pose_timestamps[idx]))
            if idx > 0:
                prev_diff = abs(ts - int(self.pose_timestamps[idx - 1]))
                if prev_diff < time_diff:
                    idx -= 1
                    time_diff = prev_diff

            if time_diff > self.pose_time_tolerance_ns:
                continue

            self.scan_files.append(f)
            self.scan_timestamps.append(ts)
            scan_poses.append(self.all_poses[idx])

        if not self.scan_files:
            raise ValueError(
                f"No Ouster scans matched to poses within "
                f"{self.pose_time_tolerance_ns}ns tolerance in {self.sequence_path}"
            )

        self.scan_poses = np.stack(scan_poses, axis=0)
        self.scan_timestamps_ns = np.array(self.scan_timestamps, dtype=np.int64)

        # MulRan poses are in absolute UTM (e.g. x~355630, y~4026791), which loses
        # ~3cm precision when cast to float32 downstream. Re-center on the first
        # scan's translation so poses are in local coordinates like KITTI/NCLT/HeLiPR.
        # Rotations are unchanged; relative distances are preserved exactly.
        self.pose_origin = self.scan_poses[0, :3, 3].copy()
        self.scan_poses[:, :3, 3] -= self.pose_origin

    def _load_point_cloud(self, idx: int) -> np.ndarray:
        """
        Load an Ouster scan as (N, 4) float32 array [x, y, z, intensity].

        MulRan Ouster bin format: raw float32 xyzi, 65536 points per scan.
        """
        f = self.scan_files[idx]
        data = np.fromfile(f, dtype=np.float32)
        if data.size % 4 != 0:
            raise ValueError(f"Unexpected MulRan bin size in {f}: {data.size} floats")
        return data.reshape(-1, 4)

    def __len__(self) -> int:
        return len(self.scan_files)

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        if self.lazy_load:
            points = self._load_point_cloud(idx)
        else:
            points = self.point_clouds[idx]

        return {
            'points': points,
            'pose': self.scan_poses[idx],
            'timestamp': self.scan_timestamps_ns[idx] / 1e9,
            'idx': idx,
        }

    def get_all_poses(self) -> np.ndarray:
        return self.scan_poses
