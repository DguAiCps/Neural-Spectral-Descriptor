"""
MCD Dataset Loader (extracted MulRan-style layout)

Loads Ouster OS1-64 (gen2) point clouds from the extracted MCD sequences.

Directory structure:
    root/
        <sequence>/                 e.g. tuhh_night_09, kth_day_06
            Ouster/
                <ns_timestamp>.bin  raw float32 xyzi (65536 points per scan)
            scan_poses.csv          header 'timestamp_ns,x,y,z'; one row per
                                    extracted scan (ground-truth position at
                                    the scan timestamp; no orientation)

Sensor: Ouster OS1-64 gen2 (64 channels, vertical FoV ±22.5°).

Pose note: scan_poses.csv provides positions only, so every pose is a pure
translation with identity rotation. Relative distances (all that place
recognition and keyframe distance/overlap/temporal selection use) are exact;
the keyframe rotation threshold simply never fires. Positions are re-centered
on the first scan like the MulRan loader so float32 casts keep precision.
"""

import numpy as np
from pathlib import Path
from typing import List


class MCDLoader:
    """
    MCD Ouster OS1-64 (gen2) data loader

    Args:
        root: Path to the extracted MCD root (contains <sequence> folders)
        sequence: Sequence name (e.g. 'tuhh_night_09')
        lazy_load: If True, load point clouds on demand
    """

    POINTS_PER_SCAN = 65536  # 64 channels × 1024 columns

    def __init__(self, root: str, sequence: str, lazy_load: bool = True):
        self.root = Path(root)
        self.sequence = sequence
        self.lazy_load = lazy_load

        self.sequence_path = self.root / sequence
        self.ouster_dir = self.sequence_path / "Ouster"
        self.pose_file = self.sequence_path / "scan_poses.csv"

        if not self.sequence_path.exists():
            raise FileNotFoundError(f"MCD sequence path not found: {self.sequence_path}")
        if not self.ouster_dir.exists():
            raise FileNotFoundError(f"Ouster directory not found: {self.ouster_dir}")
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")

        self._load_scans()

        if not lazy_load:
            self.point_clouds = [self._load_point_cloud(i) for i in range(len(self.scan_files))]
        else:
            self.point_clouds = None

        print(
            f"MCD: {sequence}: {len(self.scan_files)} scans "
            f"({self.n_dropped_no_pose} bins without pose row)"
        )

    def _load_scans(self):
        """Parse scan_poses.csv and match each Ouster bin by exact timestamp."""
        pose_by_ts = {}
        with open(self.pose_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 4:
                    continue
                try:
                    ts = int(parts[0])
                    xyz = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
                except ValueError:
                    continue  # header row
                pose_by_ts[ts] = xyz
        if not pose_by_ts:
            raise ValueError(f"No valid pose rows parsed from {self.pose_file}")

        all_bins = sorted(self.ouster_dir.glob("*.bin"))
        self.scan_files: List[Path] = []
        ts_list: List[int] = []
        xyz_list: List[np.ndarray] = []
        self.n_dropped_no_pose = 0
        for f in all_bins:
            try:
                ts = int(f.stem)
            except ValueError:
                continue
            if ts not in pose_by_ts:
                self.n_dropped_no_pose += 1
                continue
            self.scan_files.append(f)
            ts_list.append(ts)
            xyz_list.append(pose_by_ts[ts])

        if not self.scan_files:
            raise ValueError(
                f"No Ouster scans matched a scan_poses.csv row in {self.sequence_path}"
            )

        poses = np.tile(np.eye(4, dtype=np.float64), (len(self.scan_files), 1, 1))
        poses[:, :3, 3] = np.stack(xyz_list, axis=0)
        # Re-center on the first scan's translation (MulRan-loader convention).
        self.pose_origin = poses[0, :3, 3].copy()
        poses[:, :3, 3] -= self.pose_origin

        self.scan_poses = poses
        self.scan_timestamps_ns = np.array(ts_list, dtype=np.int64)

    def _load_point_cloud(self, idx: int) -> np.ndarray:
        """Load a scan as (N, 4) float32 array [x, y, z, intensity]."""
        f = self.scan_files[idx]
        data = np.fromfile(f, dtype=np.float32)
        if data.size % 4 != 0:
            raise ValueError(f"Unexpected MCD bin size in {f}: {data.size} floats")
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
