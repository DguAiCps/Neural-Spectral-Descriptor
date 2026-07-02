"""HeLiPR dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np


def _quat_xyzw_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert an xyzw quaternion to a rotation matrix."""
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class HeLiPRLoader:
    """Lazy loader for HeLiPR Velodyne sequences.

    The loader accepts either the sequence root itself:

      ``/data/helipr/Town01/Town01``

    or the dataset root plus sequence:

      ``HeLiPRLoader('/data/helipr', 'Town01')``
    """

    def __init__(
        self,
        data_root: Union[str, Path] = None,
        sequence: str | None = None,
        lazy_load: bool = True,
        root: Union[str, Path] = None,
    ):
        if data_root is None:
            data_root = root
        if data_root is None:
            raise ValueError("data_root is required")

        self.root = Path(data_root)
        self.sequence = sequence
        self.lazy_load = lazy_load
        self.sequence_dir = self._find_sequence_dir()
        self.velodyne_dir = self._find_velodyne_dir()
        self.groundtruth_path = self._find_groundtruth_path()

        self.scan_files: List[Path] = sorted(self.velodyne_dir.glob("*.bin"))
        if not self.scan_files:
            raise FileNotFoundError(f"No HeLiPR .bin scans found in {self.velodyne_dir}")

        gt_timestamps, gt_poses = self._load_groundtruth(self.groundtruth_path)
        self.scan_timestamps = np.asarray([int(p.stem) for p in self.scan_files], dtype=np.float64)
        matched = self._match_timestamps(self.scan_timestamps, gt_timestamps)
        self.poses = gt_poses[matched]
        self.timestamps = self.scan_timestamps.astype(np.float64)

        if self.sequence is None:
            self.sequence = self.sequence_dir.name

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
            "sequence": str(self.sequence),
        }

    def _find_sequence_dir(self) -> Path:
        if self.sequence is None:
            return self.root

        candidates = [
            self.root / self.sequence / self.sequence,
            self.root / self.sequence,
            self.root,
        ]
        for candidate in candidates:
            if (candidate / "LiDAR").is_dir() or (candidate / "LiDAR_GT").is_dir():
                return candidate
        raise FileNotFoundError(
            "HeLiPR sequence directory not found. Checked: "
            + ", ".join(str(p) for p in candidates)
        )

    def _find_velodyne_dir(self) -> Path:
        candidates = [
            self.sequence_dir / "LiDAR" / "Velodyne",
            self.sequence_dir / "Velodyne",
            self.sequence_dir,
        ]
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.bin")):
                return candidate
        raise FileNotFoundError(
            "HeLiPR Velodyne directory not found. Checked: "
            + ", ".join(str(p) for p in candidates)
        )

    def _find_groundtruth_path(self) -> Path:
        candidates = [
            self.sequence_dir / "LiDAR_GT" / "Velodyne_gt.txt",
            self.sequence_dir / "Velodyne_gt.txt",
            self.sequence_dir.parent / "LiDAR_GT" / "Velodyne_gt.txt",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "HeLiPR ground-truth file not found. Checked: "
            + ", ".join(str(p) for p in candidates)
        )

    @staticmethod
    def _load_points(path: Path) -> np.ndarray:
        helipr_dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("intensity", np.float32),
                ("ring", np.uint16),
                ("time", np.float32),
            ]
        )
        data = np.fromfile(path, dtype=helipr_dtype)
        if data.size == 0:
            fallback = np.fromfile(path, dtype=np.float32)
            if fallback.size % 4 != 0:
                raise ValueError(f"Invalid HeLiPR point cloud size in {path}")
            points = fallback.reshape(-1, 4)
        else:
            points = np.stack(
                [data["x"], data["y"], data["z"], data["intensity"]],
                axis=-1,
            ).astype(np.float32)
        valid = np.isfinite(points).all(axis=1) & (np.abs(points[:, :3]) < 200.0).all(axis=1)
        return points[valid]

    @staticmethod
    def _load_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray]:
        timestamps, poses = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                try:
                    vals = [float(x) for x in parts[:8]]
                except ValueError:
                    continue
                if not np.isfinite(vals).all():
                    continue
                timestamp, x, y, z, qx, qy, qz, qw = vals
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = _quat_xyzw_matrix(qx, qy, qz, qw)
                pose[:3, 3] = [x, y, z]
                timestamps.append(timestamp)
                poses.append(pose)

        if not timestamps:
            raise ValueError(f"No valid HeLiPR ground-truth rows in {path}")

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
