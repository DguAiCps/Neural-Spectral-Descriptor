"""MulRan (Ouster OS1-64) loader, reimplemented to match the NSD loader interface.

Raw format (verified): DCC03/Ouster/<timestamp>.bin, each = 65536 points x 4 float32
[x, y, z, intensity] in sensor frame. Poses in DCC03/global_pose.csv:
  timestamp, r00,r01,r02,tx, r10,r11,r12,ty, r20,r21,r22,tz   (3x4 row-major)
Scan timestamp = .bin file stem. Poses matched to scans by nearest timestamp.
Mirrors NCLTLoader / HeLiPRLoader: __len__, __getitem__ -> {points,pose,timestamp,scan_id,sequence}.
"""
from pathlib import Path
from typing import Dict, List, Union
import numpy as np


class MulRanLoader:
    def __init__(
        self,
        data_root: Union[str, Path] = None,
        sequence: str = None,
        lazy_load: bool = True,
        root: Union[str, Path] = None,
        **kwargs,
    ):
        if data_root is None:
            data_root = root
        if data_root is None:
            raise ValueError("data_root is required")
        self.root = Path(data_root)
        self.sequence = sequence
        self.lazy_load = lazy_load
        self.sequence_dir = self._find_sequence_dir()
        self.ouster_dir = self._find_ouster_dir()

        self.scan_files: List[Path] = sorted(self.ouster_dir.glob("*.bin"))
        if not self.scan_files:
            raise FileNotFoundError(f"No MulRan Ouster .bin scans in {self.ouster_dir}")
        self.scan_timestamps = np.asarray([int(p.stem) for p in self.scan_files], dtype=np.float64)

        gt_ts, gt_poses = self._load_global_pose(self.sequence_dir / "global_pose.csv")
        matched = self._match_timestamps(self.scan_timestamps, gt_ts)
        self.poses = gt_poses[matched]
        self.timestamps = self.scan_timestamps.astype(np.float64)

        if self.sequence is None:
            self.sequence = self.sequence_dir.name
        self._points_cache = None
        if not lazy_load:
            self._points_cache = [self._load_points(p) for p in self.scan_files]

    def __len__(self) -> int:
        return len(self.scan_files)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        points = self._points_cache[idx] if self._points_cache is not None else self._load_points(self.scan_files[idx])
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
        for c in [self.root / self.sequence, self.root / self.sequence / self.sequence, self.root]:
            if (c / "Ouster").is_dir() or (c / "global_pose.csv").is_file():
                return c
        raise FileNotFoundError(f"MulRan sequence dir not found under {self.root} / {self.sequence}")

    def _find_ouster_dir(self) -> Path:
        for c in [self.sequence_dir / "Ouster", self.sequence_dir]:
            if c.is_dir() and any(c.glob("*.bin")):
                return c
        raise FileNotFoundError(f"MulRan Ouster dir with .bin not found under {self.sequence_dir}")

    @staticmethod
    def _load_points(path: Path) -> np.ndarray:
        a = np.fromfile(path, dtype=np.float32)
        if a.size % 4 != 0:
            raise ValueError(f"Invalid MulRan Ouster size in {path}: {a.size}")
        pts = a.reshape(-1, 4)  # x,y,z,intensity
        xyz = pts[:, :3]
        r = np.linalg.norm(xyz, axis=1)
        valid = np.isfinite(pts).all(axis=1) & (r > 1e-3) & (r < 200.0)
        return pts[valid].astype(np.float32)

    @staticmethod
    def _load_global_pose(path: Path):
        ts, poses = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 13:
                    continue
                try:
                    vals = [float(x) for x in parts[:13]]
                except ValueError:
                    continue
                t = vals[0]
                m = np.array(vals[1:13], dtype=np.float64).reshape(3, 4)
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :4] = m
                ts.append(t); poses.append(pose)
        if not ts:
            raise ValueError(f"No valid MulRan poses in {path}")
        ts = np.asarray(ts, dtype=np.float64); poses = np.asarray(poses, dtype=np.float64)
        order = np.argsort(ts)
        return ts[order], poses[order]

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
