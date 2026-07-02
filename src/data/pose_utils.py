"""SE(3) pose and point-cloud utility functions."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def euclidean_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """Euclidean distance between pose translations."""
    return float(np.linalg.norm(np.asarray(pose_a)[:3, 3] - np.asarray(pose_b)[:3, 3]))


def rotation_angle(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """Unsigned relative rotation angle in radians."""
    r_a = np.asarray(pose_a)[:3, :3]
    r_b = np.asarray(pose_b)[:3, :3]
    r_rel = r_b @ r_a.T
    cos_angle = (np.trace(r_rel) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def rotation_angle_degrees(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """Unsigned relative rotation angle in degrees."""
    return float(np.degrees(rotation_angle(pose_a, pose_b)))


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    """Invert a homogeneous 4x4 pose."""
    pose = np.asarray(pose)
    inv = np.eye(4, dtype=pose.dtype)
    r = pose[:3, :3]
    t = pose[:3, 3]
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ t
    return inv


def compose_poses(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """Compose two homogeneous 4x4 poses."""
    return np.asarray(pose_a) @ np.asarray(pose_b)


def relative_pose(pose_from: np.ndarray, pose_to: np.ndarray) -> np.ndarray:
    """Transform from pose_from coordinates to pose_to coordinates."""
    return inverse_pose(pose_from) @ np.asarray(pose_to)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to Nx3 or NxD points, preserving extra columns."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {points.shape}")
    transform = np.asarray(transform)
    xyz_h = np.ones((len(points), 4), dtype=np.result_type(points, transform))
    xyz_h[:, :3] = points[:, :3]
    transformed_xyz = (transform @ xyz_h.T).T[:, :3]
    if points.shape[1] == 3:
        return transformed_xyz
    return np.concatenate([transformed_xyz, points[:, 3:]], axis=1)


def is_valid_transformation(pose: np.ndarray, atol: float = 1e-6) -> bool:
    """Validate a homogeneous SE(3) matrix."""
    pose = np.asarray(pose)
    if pose.shape != (4, 4):
        return False
    if not np.allclose(pose[3], np.array([0, 0, 0, 1]), atol=atol):
        return False
    r = pose[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=atol):
        return False
    if not np.isclose(np.linalg.det(r), 1.0, atol=atol):
        return False
    return True


def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    """Convert Cartesian XYZ to spherical coordinates (range, azimuth, elevation)."""
    xyz = np.asarray(points)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    radius = np.linalg.norm(xyz[:, :3], axis=1)
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(np.divide(z, radius, out=np.zeros_like(z), where=radius > 0))
    return np.stack([radius, azimuth, elevation], axis=1)


def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates (range, azimuth, elevation) to Cartesian XYZ."""
    sph = np.asarray(spherical)
    radius, azimuth, elevation = sph[:, 0], sph[:, 1], sph[:, 2]
    cos_el = np.cos(elevation)
    x = radius * cos_el * np.cos(azimuth)
    y = radius * cos_el * np.sin(azimuth)
    z = radius * np.sin(elevation)
    return np.stack([x, y, z], axis=1)


def pose_difference(pose_a: np.ndarray, pose_b: np.ndarray) -> Tuple[float, float]:
    """Return translation distance and rotation angle in degrees."""
    return euclidean_distance(pose_a, pose_b), rotation_angle_degrees(pose_a, pose_b)


def _rotation_matrix_to_quaternion(r: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to quaternion [w, x, y, z]."""
    tr = float(np.trace(r))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s,
                         (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    idx = int(np.argmax(np.diag(r)))
    if idx == 0:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        return np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s,
                         (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
    if idx == 1:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        return np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                         0.25 * s, (r[1, 2] + r[2, 1]) / s])
    s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
    return np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                     (r[1, 2] + r[2, 1]) / s, 0.25 * s])


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a rotation matrix."""
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def interpolate_poses(pose_a: np.ndarray, pose_b: np.ndarray, t: float) -> np.ndarray:
    """Interpolate two poses with linear translation and quaternion slerp."""
    t = float(np.clip(t, 0.0, 1.0))
    pose_a = np.asarray(pose_a)
    pose_b = np.asarray(pose_b)
    out = np.eye(4, dtype=np.result_type(pose_a, pose_b))
    out[:3, 3] = (1.0 - t) * pose_a[:3, 3] + t * pose_b[:3, 3]

    q0 = _rotation_matrix_to_quaternion(pose_a[:3, :3])
    q1 = _rotation_matrix_to_quaternion(pose_b[:3, :3])
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
        theta = theta_0 * t
        q = (math.sin(theta_0 - theta) * q0 + math.sin(theta) * q1) / math.sin(theta_0)
    out[:3, :3] = _quaternion_to_rotation_matrix(q)
    return out


def compute_overlap(
    points_a: np.ndarray,
    points_b: np.ndarray,
    transform_a_to_b: np.ndarray,
    voxel_size: float = 0.2,
) -> float:
    """Compute voxel IoU after transforming points_a into points_b's frame."""
    if points_a is None or points_b is None or len(points_a) == 0 or len(points_b) == 0:
        return 0.0
    a = transform_points(points_a[:, :3], transform_a_to_b)[:, :3]
    b = np.asarray(points_b)[:, :3]
    va = {tuple(v) for v in np.floor(a / voxel_size).astype(np.int64)}
    vb = {tuple(v) for v in np.floor(b / voxel_size).astype(np.int64)}
    union = len(va | vb)
    if union == 0:
        return 0.0
    return float(len(va & vb) / union)


def pose_to_7dof(pose: np.ndarray) -> np.ndarray:
    """Convert 4x4 pose to [x, y, z, qw, qx, qy, qz]."""
    pose = np.asarray(pose)
    q = _rotation_matrix_to_quaternion(pose[:3, :3])
    return np.concatenate([pose[:3, 3], q]).astype(np.float64)
