"""
KITTI-360 Dataset Loader

Loads KITTI-360 velodyne scans (HDL-64E, KITTI bin format: float32 x,y,z,i)
with per-scan poses derived from the drive's poses.txt.

Directory structure:
    root/
        2013_05_28_drive_0002_sync/
            velodyne_points/
                data/
                    0000004391.bin      10-digit velodyne frame index
                    ...
            poses.txt                   rows: frame_idx r11 r12 r13 tx r21 ... tz

Velodyne frame indices and pose frame indices are each a sparse subset of the
drive's frame counter: poses.txt does not cover every velodyne frame and vice
versa. For each velodyne frame the translation is linearly interpolated between
the two bracketing pose rows (in frame-index space); the rotation is taken from
the nearest row. Scans outside pose coverage are dropped, and coverage stats
are recorded in ``self.coverage``.

Frame-of-reference note: poses.txt is the IMU/GNSS pose in the world frame,
not the velodyne pose. For place recognition at a 5 m distance threshold only
relative distances between scan positions matter, and the constant
IMU->velodyne lever arm cancels in relative distances, so no extrinsic
calibration is applied.
"""

import numpy as np
from pathlib import Path
from typing import List


class Kitti360Loader:
    """
    KITTI-360 velodyne data loader

    Args:
        root: Path to KITTI-360 root (contains 2013_05_28_drive_XXXX_sync folders)
        sequence: Drive id, either '0002' or the full '2013_05_28_drive_0002_sync'
        lazy_load: If True, load point clouds on demand
    """

    def __init__(self, root: str, sequence: str, lazy_load: bool = True):
        self.root = Path(root)
        seq = str(sequence)
        if not seq.startswith("2013_05_28_drive"):
            seq = f"2013_05_28_drive_{int(seq):04d}_sync"
        self.sequence = seq
        self.lazy_load = lazy_load

        self.sequence_path = self.root / seq
        self.velodyne_dir = self.sequence_path / "velodyne_points" / "data"
        self.pose_file = self.sequence_path / "poses.txt"

        if not self.sequence_path.exists():
            raise FileNotFoundError(f"KITTI-360 sequence path not found: {self.sequence_path}")
        if not self.velodyne_dir.exists():
            raise FileNotFoundError(f"Velodyne directory not found: {self.velodyne_dir}")
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_file}")

        self._load_poses()
        self._assign_scan_poses()

        if not lazy_load:
            self.point_clouds = [self._load_point_cloud(i) for i in range(len(self.scan_files))]
        else:
            self.point_clouds = None

        print(
            f"KITTI-360: {seq}: {self.coverage['n_with_pose']}/{self.coverage['n_bins']} "
            f"scans with pose (dropped {self.coverage['n_dropped_outside']} outside "
            f"pose frames {self.coverage['pose_frame_range']})"
        )

    def _load_poses(self):
        """Load poses.txt rows: frame_idx + 12 values (3x4 world pose)."""
        frames: List[int] = []
        rotations: List[np.ndarray] = []
        translations: List[np.ndarray] = []
        with open(self.pose_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 13:
                    continue
                mat = np.array([float(x) for x in parts[1:]], dtype=np.float64).reshape(3, 4)
                frames.append(int(parts[0]))
                rotations.append(mat[:, :3])
                translations.append(mat[:, 3])
        if not frames:
            raise ValueError(f"No valid poses parsed from {self.pose_file}")

        order = np.argsort(frames)
        self.pose_frames = np.asarray(frames, dtype=np.int64)[order]
        self.pose_rotations = np.stack(rotations, axis=0)[order]
        self.pose_translations = np.stack(translations, axis=0)[order]

    def _assign_scan_poses(self):
        """Interpolate a pose for every velodyne frame inside pose coverage.

        Translation: linear interpolation between the bracketing pose rows in
        frame-index space. Rotation: nearest pose row (pose rows are dense, so
        this is nearly exact and only feeds the keyframe rotation threshold).
        """
        all_bins = sorted(self.velodyne_dir.glob("*.bin"))
        if not all_bins:
            raise ValueError(f"No .bin files found in {self.velodyne_dir}")
        bin_frames = np.asarray([int(f.stem) for f in all_bins], dtype=np.int64)

        pf = self.pose_frames
        in_coverage = (bin_frames >= pf[0]) & (bin_frames <= pf[-1])
        kept = np.where(in_coverage)[0]
        if len(kept) == 0:
            raise ValueError(
                f"No velodyne frames inside pose coverage [{pf[0]}, {pf[-1]}] "
                f"in {self.sequence_path}"
            )
        frames = bin_frames[kept]

        hi = np.searchsorted(pf, frames, side="left")
        exact = pf[hi] == frames
        lo = np.where(exact, hi, hi - 1)
        span = np.maximum(pf[hi] - pf[lo], 1).astype(np.float64)
        t = np.where(exact, 0.0, (frames - pf[lo]) / span)

        translations = (1.0 - t)[:, None] * self.pose_translations[lo] \
            + t[:, None] * self.pose_translations[hi]
        rotations = self.pose_rotations[np.where(t < 0.5, lo, hi)]

        poses = np.tile(np.eye(4, dtype=np.float64), (len(frames), 1, 1))
        poses[:, :3, :3] = rotations
        poses[:, :3, 3] = translations
        # Re-center on the first kept scan so downstream float32 casts keep
        # centimeter precision, like the MulRan loader. Relative distances are
        # preserved exactly.
        self.pose_origin = poses[0, :3, 3].copy()
        poses[:, :3, 3] -= self.pose_origin

        self.scan_files = [all_bins[i] for i in kept]
        self.scan_frames = frames
        self.scan_poses = poses
        # Velodyne runs at 10 Hz; frame indices are the drive's 10 Hz counter,
        # so gaps in the bin list become real time gaps.
        self.timestamps = frames.astype(np.float64) * 0.1

        interp_gaps = (pf[hi] - pf[lo])[~exact]
        self.coverage = {
            "n_bins": int(len(all_bins)),
            "n_with_pose": int(len(kept)),
            "n_dropped_outside": int(len(all_bins) - len(kept)),
            "n_exact_pose": int(exact.sum()),
            "n_interpolated": int((~exact).sum()),
            "max_interp_gap_frames": int(interp_gaps.max()) if len(interp_gaps) else 0,
            "pose_frame_range": [int(pf[0]), int(pf[-1])],
            "bin_frame_range": [int(bin_frames[0]), int(bin_frames[-1])],
        }

    def _load_point_cloud(self, idx: int) -> np.ndarray:
        """Load a scan as (N, 4) float32 array [x, y, z, intensity]."""
        data = np.fromfile(self.scan_files[idx], dtype=np.float32)
        if data.size % 4 != 0:
            raise ValueError(
                f"Unexpected KITTI-360 bin size in {self.scan_files[idx]}: {data.size} floats"
            )
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
            'timestamp': float(self.timestamps[idx]),
            'idx': idx,
        }

    def get_all_poses(self) -> np.ndarray:
        return self.scan_poses
