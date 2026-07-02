"""NCLT dataset loader.

Expected layouts are intentionally permissive because local mirrors and the
official tarballs differ slightly after extraction:

  root/
    2012-01-08/velodyne_sync/*.bin
    2012-01-08/groundtruth_2012-01-08.csv

or:

  root/
    ground_truth/2012-01-08.csv
    2012-01-08/velodyne_sync/*.bin

Each item returns the same fields as :class:`data.kitti_loader.KITTILoader` so
the multi-dataset training and diagnostic scripts can share one code path.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Union

import numpy as np


def _euler_zyx_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return Rz(yaw) Ry(pitch) Rx(roll), matching the NCLT convention."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


class NCLTLoader:
    """Lazy loader for NCLT Velodyne scans and ground-truth poses."""

    def __init__(
        self,
        data_root: Union[str, Path] = None,
        date: str = "2012-01-08",
        lazy_load: bool = True,
        root: Union[str, Path] = None,
    ):
        if data_root is None:
            data_root = root
        if data_root is None:
            raise ValueError("data_root is required")

        self.root = Path(data_root)
        self.date = str(date)
        self.lazy_load = lazy_load

        self.sequence_dir = self.root / self.date
        self.velodyne_dir = self._find_velodyne_dir()
        self.groundtruth_path = self._find_groundtruth_path()

        self.scan_files: List[Path] = sorted(self.velodyne_dir.glob("*.bin"))
        if not self.scan_files:
            raise FileNotFoundError(f"No NCLT .bin scans found in {self.velodyne_dir}")

        gt_timestamps, gt_poses = self._load_groundtruth(self.groundtruth_path)
        self.scan_timestamps = np.asarray([int(p.stem) for p in self.scan_files], dtype=np.float64)
        matched = self._match_timestamps(self.scan_timestamps, gt_timestamps)
        self.poses = gt_poses[matched]
        self.timestamps = self.scan_timestamps.astype(np.float64)

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
            "pose": self.poses[idx].copy(),
            "timestamp": float(self.timestamps[idx]),
            "scan_id": int(self.scan_timestamps[idx]),
            "sequence": self.date,
        }

    def _find_velodyne_dir(self) -> Path:
        candidates = [
            self.sequence_dir / "velodyne_sync",
            self.sequence_dir / self.date / "velodyne_sync",
            self.sequence_dir / "velodyne",
            self.sequence_dir / "velodyne_data",
            self.sequence_dir,
        ]
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.bin")):
                return candidate
        raise FileNotFoundError(
            "NCLT velodyne directory not found. Checked: "
            + ", ".join(str(p) for p in candidates)
        )

    def _find_groundtruth_path(self) -> Path:
        candidates = [
            self.sequence_dir / f"groundtruth_{self.date}.csv",
            self.sequence_dir / "groundtruth.csv",
            self.root / f"groundtruth_{self.date}.csv",
            self.root / "ground_truth" / f"{self.date}.csv",
            self.root / "ground_truth" / f"groundtruth_{self.date}.csv",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "NCLT ground-truth file not found. Checked: "
            + ", ".join(str(p) for p in candidates)
        )

    @staticmethod
    def _load_points(path: Path) -> np.ndarray:
        nclt_dtype = np.dtype(
            [
                ("x", "<u2"),
                ("y", "<u2"),
                ("z", "<u2"),
                ("intensity", "u1"),
                ("padding", "u1"),
                ("extra", "<u4"),
            ]
        )
        raw = np.fromfile(path, dtype=nclt_dtype)
        x = raw["x"].astype(np.float64) * 0.005 - 100.0
        y = raw["y"].astype(np.float64) * 0.005 - 100.0
        z = raw["z"].astype(np.float64) * 0.005 - 100.0
        intensity = raw["intensity"].astype(np.float32) / 255.0
        points = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)
        valid = np.isfinite(points).all(axis=1) & (np.abs(points[:, :3]) < 200.0).all(axis=1)
        return points[valid]

    @staticmethod
    def _load_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray]:
        timestamps, poses = [], []
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 7:
                    continue
                try:
                    vals = [float(x) for x in row[:7]]
                except ValueError:
                    continue
                if not np.isfinite(vals).all():
                    continue
                timestamp, x, y, z, roll, pitch, yaw = vals
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = _euler_zyx_matrix(roll, pitch, yaw)
                pose[:3, 3] = [x, y, z]
                timestamps.append(timestamp)
                poses.append(pose)

        if not timestamps:
            raise ValueError(f"No valid NCLT ground-truth rows in {path}")

        order = np.argsort(np.asarray(timestamps, dtype=np.float64))
        return np.asarray(timestamps, dtype=np.float64)[order], np.asarray(poses, dtype=np.float64)[order]

    @staticmethod
    def _match_timestamps(scan_ts: np.ndarray, gt_ts: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(gt_ts, scan_ts)
        idx = np.clip(idx, 1, len(gt_ts) - 1)
        left = np.abs(gt_ts[idx - 1] - scan_ts)
        right = np.abs(gt_ts[idx] - scan_ts)
        return np.where(left < right, idx - 1, idx)

    def get_pose(self, idx: int) -> np.ndarray:
        return self.poses[int(idx)].copy()

    def get_timestamp(self, idx: int) -> float:
        return float(self.timestamps[int(idx)])

    def get_scan_path(self, idx: int) -> Path:
        return self.scan_files[int(idx)]
