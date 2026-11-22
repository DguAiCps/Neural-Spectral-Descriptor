"""Keyframe selection.

The first scan is a keyframe. Every later scan becomes a keyframe when, relative to the previous
keyframe, it has moved more than 0.8 m, turned more than 20 degrees, or is more than 30 s later
(any one of the three). Poses are the LiDAR odometry poses.
"""
from __future__ import annotations

import numpy as np

DISTANCE_M = 0.8
ROTATION_DEG = 20.0
INTERVAL_S = 30.0


def _translation_m(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b[:3, 3] - a[:3, 3]))


def _rotation_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the relative rotation R_a^T R_b, from its trace."""
    cos_theta = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos_theta)))


def select_keyframes(poses: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Scan indices of the keyframes, in order, for poses [N, 4, 4] and timestamps [N]."""
    selected = [0]
    for i in range(1, len(poses)):
        last = selected[-1]
        if (_translation_m(poses[i], poses[last]) > DISTANCE_M
                or _rotation_deg(poses[i], poses[last]) > ROTATION_DEG
                or abs(timestamps[i] - timestamps[last]) > INTERVAL_S):
            selected.append(i)
    return np.asarray(selected, dtype=np.int64)
