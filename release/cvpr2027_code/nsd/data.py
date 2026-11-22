"""Dataset readers: LiDAR scans, scan timestamps and reference poses.

Each reader returns a `Sequence`: the ordered list of scans of one recording, their timestamps in
seconds, and the dataset's reference poses (used for evaluation only; the method itself runs on
LiDAR odometry, see `nsd.odometry`). Scan index i of a `Sequence` is the `scan_id` stored in the
per-sequence output.

`root` is the directory of one dataset as distributed, with these layouts:

    kitti      <root>/sequences/<seq>/velodyne/*.bin, <root>/poses/<seq>.txt
    kitti360   <root>/2013_05_28_drive_<seq>_sync/velodyne_points/data/*.bin, .../poses.txt
    nclt       <root>/<date>/velodyne_sync/*.bin, <root>/<date>/groundtruth_<date>.csv
    helipr     <root>/<seq>/LiDAR/Velodyne/*.bin, <root>/<seq>/LiDAR_GT/Velodyne_gt.txt
    mulran     <root>/<seq>/Ouster/*.bin, <root>/<seq>/global_pose.csv
    mcd        <root>/<seq>/Ouster/*.bin, <root>/<seq>/scan_poses.csv
               (scans and per-scan positions extracted from the recorded bags, one float32
               x, y, z, intensity file per scan named by its nanosecond timestamp)

Conventions kept from the evaluated configuration:
  * scans without a reference pose are dropped (KITTI-360 outside pose coverage; HeLiPR and
    MulRan scans more than 100 ms from a pose; MCD scans without a pose row);
  * MulRan, MCD and KITTI-360 positions are shifted so that the first scan is at the origin;
  * MCD provides positions only, so its reference rotations are the identity;
  * the NCLT HDL-32E is mounted upside down; z is negated so that every cloud is z-up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

DATASETS = ("kitti", "kitti360", "nclt", "helipr", "mulran", "mcd")
POSE_TOLERANCE_NS = 100_000_000  # HeLiPR / MulRan scan-to-pose matching tolerance (100 ms)


@dataclass
class Sequence:
    """One recording: scan files, timestamps [s], reference poses [N, 4, 4]."""
    dataset: str
    name: str
    scan_files: List[Path]
    timestamps: np.ndarray
    poses_gt: np.ndarray
    _read: Callable[[Path], np.ndarray]
    lidar_to_pose: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.scan_files)

    def points(self, i: int) -> np.ndarray:
        """Point cloud of scan i as float32 [M, 3] (x, y, z in the sensor frame, z up)."""
        return self._read(self.scan_files[i])


# ----------------------------------------------------------------------------- point formats
def _read_xyzi_float32(path: Path) -> np.ndarray:
    """KITTI, KITTI-360, MulRan, MCD: float32 records (x, y, z, intensity)."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


_HELIPR_VELODYNE = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32),
                             ("intensity", np.float32), ("ring", np.uint16),
                             ("time", np.float32)])


def _read_helipr_velodyne(path: Path) -> np.ndarray:
    """HeLiPR Velodyne VLP-16: 22-byte records (x, y, z, intensity, ring, time)."""
    data = np.fromfile(path, dtype=_HELIPR_VELODYNE)
    return np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)


_NCLT_HIT = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"),
                      ("intensity", "u1"), ("laser", "u1")])


def _read_nclt_velodyne(path: Path) -> np.ndarray:
    """NCLT velodyne_sync: uint16 coordinates, metres = value * 0.005 - 100; z flipped to point up."""
    raw = np.fromfile(path, dtype=_NCLT_HIT)
    x = raw["x"].astype(np.float32) * 0.005 - 100.0
    y = raw["y"].astype(np.float32) * 0.005 - 100.0
    z = -(raw["z"].astype(np.float32) * 0.005 - 100.0)
    xyz = np.column_stack([x, y, z])
    keep = np.isfinite(xyz).all(axis=1) & (np.abs(xyz) < 200.0).all(axis=1)
    return xyz[keep]


# ----------------------------------------------------------------------------- pose helpers
def _se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    poses = np.tile(np.eye(4), (len(translation), 1, 1))
    poses[:, :3, :3] = rotation
    poses[:, :3, 3] = translation
    return poses


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from quaternions (qx, qy, qz, qw) [N, 4] -> [N, 3, 3], normalised first."""
    x, y, z, w = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    xx, yy, zz, ww = x * x, y * y, z * z, w * w
    xy, xz, xw, yz, yw, zw = x * y, x * z, x * w, y * z, y * w, z * w
    return np.stack([
        np.stack([xx - yy - zz + ww, 2 * (xy - zw), 2 * (xz + yw)], -1),
        np.stack([2 * (xy + zw), -xx + yy - zz + ww, 2 * (yz - xw)], -1),
        np.stack([2 * (xz - yw), 2 * (yz + xw), -xx - yy + zz + ww], -1),
    ], axis=1)


def _euler_to_matrix(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll), [N] -> [N, 3, 3]."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.stack([
        np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
        np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
        np.stack([-sp, cp * sr, cp * cr], -1),
    ], axis=1)


def _read_stamped_rows(path: Path, sep, n_fields: int):
    """Text rows 'timestamp_ns v1 v2 ...': int64 stamps [N] and float64 values [N, n_fields - 1].
    Rows with fewer fields or a non-integer stamp (headers) are skipped."""
    stamps, values = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(sep)
            if len(parts) < n_fields:
                continue
            try:
                stamps.append(int(parts[0]))
                values.append([float(v) for v in parts[1:n_fields]])
            except ValueError:
                continue
    return np.array(stamps, dtype=np.int64), np.array(values, dtype=np.float64)


def _bin_files(directory: Path) -> List[Path]:
    files = sorted(directory.glob("*.bin"))
    if not files:
        raise FileNotFoundError(f"no .bin scans in {directory}")
    return files


# ----------------------------------------------------------------------------- readers
def _kitti(root: Path, seq: str) -> Sequence:
    files = _bin_files(root / "sequences" / seq / "velodyne")
    pose_file = root / "sequences" / seq / "poses.txt"
    if not pose_file.exists():
        pose_file = root / "poses" / f"{seq}.txt"
    rows = np.loadtxt(pose_file, dtype=np.float64).reshape(-1, 3, 4)
    poses = _se3(rows[:, :, :3], rows[:, :, 3])
    timestamps = np.arange(len(files)) * 0.1  # 10 Hz
    return Sequence("kitti", seq, files, timestamps, poses, _read_xyzi_float32,
                    lidar_to_pose=_kitti_calibration(root / "sequences" / seq / "calib.txt"))


def _kitti_calibration(path: Path) -> Optional[np.ndarray]:
    """4x4 `Tr` (LiDAR -> camera 0) from a KITTI calib.txt, or None if the file is absent."""
    if not path.exists():
        return None
    (line,) = [l for l in path.read_text().splitlines() if l.startswith("Tr:")]
    tr = np.eye(4)
    tr[:3] = np.array([float(v) for v in line.split(":", 1)[1].split()]).reshape(3, 4)
    return tr


def _kitti360(root: Path, seq: str) -> Sequence:
    """Translation interpolated linearly in frame index between bracketing pose rows, rotation
    of the nearest row; scans outside the pose rows are dropped."""
    drive = root / f"2013_05_28_drive_{int(seq):04d}_sync"
    all_files = _bin_files(drive / "velodyne_points" / "data")
    rows = np.loadtxt(drive / "poses.txt", dtype=np.float64)
    rows = rows[np.argsort(rows[:, 0].astype(np.int64))]
    pose_frames = rows[:, 0].astype(np.int64)
    mats = rows[:, 1:].reshape(-1, 3, 4)
    bin_frames = np.array([int(f.stem) for f in all_files], dtype=np.int64)
    kept = np.where((bin_frames >= pose_frames[0]) & (bin_frames <= pose_frames[-1]))[0]
    frames = bin_frames[kept]
    hi = np.searchsorted(pose_frames, frames, side="left")
    exact = pose_frames[hi] == frames
    lo = np.where(exact, hi, hi - 1)
    span = np.maximum(pose_frames[hi] - pose_frames[lo], 1).astype(np.float64)
    t = np.where(exact, 0.0, (frames - pose_frames[lo]) / span)
    translation = (1.0 - t)[:, None] * mats[lo, :, 3] + t[:, None] * mats[hi, :, 3]
    rotation = mats[np.where(t < 0.5, lo, hi), :, :3]
    poses = _se3(rotation, translation)
    poses[:, :3, 3] -= poses[0, :3, 3].copy()
    return Sequence("kitti360", seq, [all_files[i] for i in kept],
                    frames.astype(np.float64) * 0.1, poses, _read_xyzi_float32)


def _nclt(root: Path, date: str) -> Sequence:
    """Reference pose of the nearest ground-truth row in time (ties go to the earlier row)."""
    files = _bin_files(root / date / "velodyne_sync")
    stamps_us = np.array([int(f.stem) for f in files], dtype=np.int64)
    timestamps = stamps_us / 1e6
    timestamps = timestamps - timestamps[0]
    gt = np.genfromtxt(root / date / f"groundtruth_{date}.csv", delimiter=",", dtype=np.float64)
    gt = gt[np.isfinite(gt).all(axis=1)]
    gt_us = gt[:, 0].astype(np.int64)
    right = np.clip(np.searchsorted(gt_us, stamps_us), 1, len(gt_us) - 1)
    left = right - 1
    nearest = np.where(np.abs(stamps_us - gt_us[left]) <= np.abs(stamps_us - gt_us[right]),
                       left, right)
    g = gt[nearest]
    poses = _se3(_euler_to_matrix(g[:, 4], g[:, 5], g[:, 6]), g[:, 1:4])
    return Sequence("nclt", date, files, timestamps, poses, _read_nclt_velodyne)


def _helipr(root: Path, seq: str) -> Sequence:
    """Reference pose of the first ground-truth row at or after the scan time, or of the row
    before it when that one is closer and the first is beyond the 100 ms tolerance."""
    base = root / seq
    if not (base / "LiDAR").is_dir():
        base = base / seq
    files_all = _bin_files(base / "LiDAR" / "Velodyne")
    gt_ns, gt_rows = _read_stamped_rows(base / "LiDAR_GT" / "Velodyne_gt.txt", None, 8)
    gt_poses = _se3(_quaternion_to_matrix(gt_rows[:, 3:7]), gt_rows[:, 0:3])
    files, stamps, idx_kept = [], [], []
    for f in files_all:
        ts = int(f.stem)
        idx = int(np.clip(np.searchsorted(gt_ns, ts), 0, len(gt_ns) - 1))
        diff = abs(ts - int(gt_ns[idx]))
        if diff > POSE_TOLERANCE_NS and idx > 0 and abs(ts - int(gt_ns[idx - 1])) < diff:
            idx -= 1
            diff = abs(ts - int(gt_ns[idx]))
        if diff <= POSE_TOLERANCE_NS:
            files.append(f)
            stamps.append(ts)
            idx_kept.append(idx)
    timestamps = np.array(stamps, dtype=np.int64) / 1e9
    return Sequence("helipr", seq, files, timestamps, gt_poses[idx_kept], _read_helipr_velodyne)


def _mulran(root: Path, seq: str) -> Sequence:
    """Reference pose of the nearest global_pose.csv row within 100 ms; origin at the first scan."""
    pose_ns, rows = _read_stamped_rows(root / seq / "global_pose.csv", ",", 13)
    order = np.argsort(pose_ns)
    pose_ns, rows = pose_ns[order], rows[order]
    mats = rows[:, :12].reshape(-1, 3, 4)
    files, stamps, idx_kept = [], [], []
    for f in _bin_files(root / seq / "Ouster"):
        ts = int(f.stem)
        idx = int(np.clip(np.searchsorted(pose_ns, ts), 0, len(pose_ns) - 1))
        diff = abs(ts - int(pose_ns[idx]))
        if idx > 0 and abs(ts - int(pose_ns[idx - 1])) < diff:
            idx -= 1
            diff = abs(ts - int(pose_ns[idx]))
        if diff <= POSE_TOLERANCE_NS:
            files.append(f)
            stamps.append(ts)
            idx_kept.append(idx)
    poses = _se3(mats[idx_kept, :, :3], mats[idx_kept, :, 3])
    poses[:, :3, 3] -= poses[0, :3, 3].copy()
    timestamps = np.array(stamps, dtype=np.int64) / 1e9
    return Sequence("mulran", seq, files, timestamps, poses, _read_xyzi_float32)


def _mcd(root: Path, seq: str) -> Sequence:
    """Positions only (identity rotation), matched by exact timestamp; origin at the first scan."""
    pose_by_ns = {}
    with open(root / seq / "scan_poses.csv") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) == 4 and parts[0].isdigit():
                pose_by_ns[int(parts[0])] = [float(v) for v in parts[1:4]]
    files, stamps = [], []
    for f in _bin_files(root / seq / "Ouster"):
        if f.stem.isdigit() and int(f.stem) in pose_by_ns:
            files.append(f)
            stamps.append(int(f.stem))
    translation = np.array([pose_by_ns[t] for t in stamps], dtype=np.float64)
    poses = _se3(np.eye(3), translation)
    poses[:, :3, 3] -= poses[0, :3, 3].copy()
    timestamps = np.array(stamps, dtype=np.int64) / 1e9
    return Sequence("mcd", seq, files, timestamps, poses, _read_xyzi_float32)


_READERS = {"kitti": _kitti, "kitti360": _kitti360, "nclt": _nclt,
            "helipr": _helipr, "mulran": _mulran, "mcd": _mcd}


def open_sequence(dataset: str, root: str, sequence: str) -> Sequence:
    """Open one recording of `dataset` stored under `root` (layouts in the module docstring)."""
    if dataset not in _READERS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    return _READERS[dataset](Path(root), sequence)
