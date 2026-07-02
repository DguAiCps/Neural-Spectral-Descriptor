"""KITTI odometry dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np


class KITTILoader:
    """Lazy loader for KITTI odometry sequences.

    Expected layout after extracting official KITTI odometry archives:
    root/
      poses/00.txt
      sequences/00/times.txt
      sequences/00/velodyne/000000.bin
    """

    def __init__(
        self,
        data_root: Union[str, Path] = None,
        sequence: Union[str, int] = "00",
        lazy_load: bool = True,
        root: Union[str, Path] = None,
    ):
        if data_root is None:
            data_root = root
        if data_root is None:
            raise ValueError("data_root is required")

        self.root = Path(data_root)
        self.sequence = f"{int(sequence):02d}" if isinstance(sequence, int) else str(sequence)
        self.lazy_load = lazy_load

        self.sequence_dir = self.root / "sequences" / self.sequence
        self.velodyne_dir = self.sequence_dir / "velodyne"
        self.poses_path = self.root / "poses" / f"{self.sequence}.txt"
        self.times_path = self.sequence_dir / "times.txt"

        if not self.velodyne_dir.is_dir():
            raise FileNotFoundError(f"KITTI velodyne directory not found: {self.velodyne_dir}")
        if not self.poses_path.is_file():
            raise FileNotFoundError(f"KITTI poses file not found: {self.poses_path}")

        self.scan_files: List[Path] = sorted(self.velodyne_dir.glob("*.bin"))
        if not self.scan_files:
            raise FileNotFoundError(f"No KITTI .bin scans found in {self.velodyne_dir}")
        self.scan_indices = np.asarray([int(p.stem) for p in self.scan_files], dtype=np.int64)

        self.poses = self._load_poses(self.poses_path)
        if len(self.poses) <= int(self.scan_indices.max()):
            raise ValueError(
                f"Pose count ({len(self.poses)}) does not cover max scan index "
                f"{int(self.scan_indices.max())}"
            )
        self.timestamps = self._load_timestamps()

        self._points_cache = None
        if not lazy_load:
            self._points_cache = [self._load_points(path) for path in self.scan_files]

    def __len__(self) -> int:
        return len(self.scan_files)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if self._points_cache is None:
            points = self._load_points(self.scan_files[idx])
        else:
            points = self._points_cache[idx]
        return {
            "points": points,
            "pose": self.poses[self.scan_indices[idx]].copy(),
            "timestamp": float(self.timestamps[self.scan_indices[idx]]),
            "scan_id": int(self.scan_indices[idx]),
            "sequence": self.sequence,
        }

    @staticmethod
    def _load_points(path: Path) -> np.ndarray:
        points = np.fromfile(path, dtype=np.float32)
        if points.size % 4 != 0:
            raise ValueError(f"Invalid KITTI point cloud size in {path}")
        return points.reshape(-1, 4)

    @staticmethod
    def _load_poses(path: Path) -> np.ndarray:
        raw = np.loadtxt(path, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] != 12:
            raise ValueError(f"KITTI pose file must have 12 columns: {path}")
        poses = np.tile(np.eye(4, dtype=np.float64), (raw.shape[0], 1, 1))
        poses[:, :3, :4] = raw.reshape(-1, 3, 4)
        return poses

    def _load_timestamps(self) -> np.ndarray:
        if self.times_path.is_file():
            times = np.loadtxt(self.times_path, dtype=np.float64)
            if times.ndim == 0:
                times = times.reshape(1)
            if len(times) > int(self.scan_indices.max()):
                return times
        return np.arange(len(self.poses), dtype=np.float64) / 10.0

    def get_pose(self, idx: int) -> np.ndarray:
        return self.poses[int(self.scan_indices[int(idx)])].copy()

    def get_timestamp(self, idx: int) -> float:
        return float(self.timestamps[int(self.scan_indices[int(idx)])])

    def get_scan_path(self, idx: int) -> Path:
        return self.scan_files[int(idx)]
