"""
SE(3) Pose Utilities for LiDAR SLAM

Provides utility functions for working with 6-DOF poses:
- SE(3) transformations
- Distance and rotation calculations
- Pose interpolation
- Coordinate transformations
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation


def pose_to_transformation_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Convert position and rotation to SE(3) transformation matrix

    Args:
        position: (3,) translation vector [x, y, z]
        rotation: (3, 3) rotation matrix or (4,) quaternion [w, x, y, z]

    Returns:
        (4, 4) SE(3) transformation matrix
    """
    T = np.eye(4)
    T[:3, 3] = position

    if rotation.shape == (3, 3):
        T[:3, :3] = rotation
    elif rotation.shape == (4,):
        # Quaternion to rotation matrix
        R = Rotation.from_quat(rotation[[1, 2, 3, 0]]).as_matrix()  # scipy uses [x,y,z,w]
        T[:3, :3] = R
    else:
        raise ValueError(f"Invalid rotation shape: {rotation.shape}")

    return T


def transformation_matrix_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract position and rotation from SE(3) transformation matrix

    Args:
        T: (4, 4) SE(3) transformation matrix

    Returns:
        position: (3,) translation vector
        rotation: (3, 3) rotation matrix
    """
    position = T[:3, 3]
    rotation = T[:3, :3]
    return position, rotation


def inverse_pose(T: np.ndarray) -> np.ndarray:
    """
    Compute inverse of SE(3) transformation

    Args:
        T: (4, 4) SE(3) transformation matrix

    Returns:
        (4, 4) inverse transformation T^{-1}
    """
    T_inv = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]

    # T^{-1} = [R^T, -R^T @ t; 0, 1]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def compose_poses(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    Compose two SE(3) transformations: T_result = T1 @ T2

    Args:
        T1: (4, 4) first transformation
        T2: (4, 4) second transformation

    Returns:
        (4, 4) composed transformation
    """
    return T1 @ T2


def relative_pose(T_source: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """
    Compute relative pose from source to target

    Args:
        T_source: (4, 4) source pose in world frame
        T_target: (4, 4) target pose in world frame

    Returns:
        (4, 4) relative transformation T_rel = T_source^{-1} @ T_target
    """
    return inverse_pose(T_source) @ T_target


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform point cloud by SE(3) transformation

    Args:
        points: (N, 3) or (N, 4) point cloud [x, y, z] or [x, y, z, intensity]
        T: (4, 4) SE(3) transformation matrix

    Returns:
        Transformed points with same shape as input
    """
    if points.shape[1] == 3:
        # Homogeneous coordinates
        points_hom = np.hstack([points, np.ones((len(points), 1))])
        transformed = (T @ points_hom.T).T
        return transformed[:, :3]
    elif points.shape[1] == 4:
        # Keep intensity unchanged
        xyz = points[:, :3]
        intensity = points[:, 3:4]

        xyz_hom = np.hstack([xyz, np.ones((len(xyz), 1))])
        transformed = (T @ xyz_hom.T).T

        return np.hstack([transformed[:, :3], intensity])
    else:
        raise ValueError(f"Invalid point cloud shape: {points.shape}")


def euclidean_distance(T1: np.ndarray, T2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two poses

    Args:
        T1: (4, 4) first pose
        T2: (4, 4) second pose

    Returns:
        Euclidean distance in meters
    """
    pos1 = T1[:3, 3]
    pos2 = T2[:3, 3]
    return np.linalg.norm(pos2 - pos1)


def rotation_angle(T1: np.ndarray, T2: np.ndarray) -> float:
    """
    Compute rotation angle between two poses

    Args:
        T1: (4, 4) first pose
        T2: (4, 4) second pose

    Returns:
        Rotation angle in radians
    """
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]

    # Relative rotation
    R_rel = R1.T @ R2

    # Angle from trace of rotation matrix
    # trace(R) = 1 + 2*cos(theta)
    trace = np.trace(R_rel)
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)  # Handle numerical errors

    return np.arccos(cos_theta)


def rotation_angle_degrees(T1: np.ndarray, T2: np.ndarray) -> float:
    """
    Compute rotation angle between two poses in degrees

    Args:
        T1: (4, 4) first pose
        T2: (4, 4) second pose

    Returns:
        Rotation angle in degrees
    """
    return np.degrees(rotation_angle(T1, T2))


def interpolate_poses(T1: np.ndarray, T2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Interpolate between two poses using SLERP for rotation

    Args:
        T1: (4, 4) first pose
        T2: (4, 4) second pose
        alpha: Interpolation parameter in [0, 1]

    Returns:
        (4, 4) interpolated pose
    """
    # Linear interpolation for translation
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    t_interp = (1 - alpha) * t1 + alpha * t2

    # SLERP for rotation
    R1 = Rotation.from_matrix(T1[:3, :3])
    R2 = Rotation.from_matrix(T2[:3, :3])

    # Scipy's slerp
    from scipy.spatial.transform import Slerp
    key_times = [0, 1]
    key_rots = Rotation.concatenate([R1, R2])
    slerp = Slerp(key_times, key_rots)
    R_interp = slerp([alpha])[0]

    # Construct interpolated pose
    T_interp = np.eye(4)
    T_interp[:3, :3] = R_interp.as_matrix()
    T_interp[:3, 3] = t_interp

    return T_interp


def pose_difference(T1: np.ndarray, T2: np.ndarray) -> Tuple[float, float]:
    """
    Compute translation and rotation differences between two poses

    Args:
        T1: (4, 4) first pose
        T2: (4, 4) second pose

    Returns:
        translation_diff: Euclidean distance in meters
        rotation_diff: Rotation angle in radians
    """
    trans_diff = euclidean_distance(T1, T2)
    rot_diff = rotation_angle(T1, T2)
    return trans_diff, rot_diff


def is_valid_transformation(T: np.ndarray, epsilon: float = 1e-6) -> bool:
    """
    Check if matrix is a valid SE(3) transformation

    Args:
        T: (4, 4) transformation matrix
        epsilon: Tolerance for checks

    Returns:
        True if valid SE(3) transformation
    """
    if T.shape != (4, 4):
        return False

    # Check bottom row is [0, 0, 0, 1]
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=epsilon):
        return False

    # Check rotation matrix is orthogonal
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=epsilon):
        return False

    # Check determinant is 1 (not -1, which would be reflection)
    if not np.isclose(np.linalg.det(R), 1.0, atol=epsilon):
        return False

    return True


def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to spherical coordinates

    Args:
        points: (N, 3) array of [x, y, z] coordinates

    Returns:
        (N, 3) array of [range, azimuth, elevation]
        - range: distance from origin
        - azimuth: angle in xy-plane from +x axis [-pi, pi]
        - elevation: angle from xy-plane [-pi/2, pi/2]
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Range
    r = np.sqrt(x**2 + y**2 + z**2)

    # Azimuth (yaw) in [-pi, pi]
    azimuth = np.arctan2(y, x)

    # Elevation (pitch) in [-pi/2, pi/2]
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))

    return np.stack([r, azimuth, elevation], axis=1)


def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates

    Args:
        spherical: (N, 3) array of [range, azimuth, elevation]

    Returns:
        (N, 3) array of [x, y, z] coordinates
    """
    r = spherical[:, 0]
    azimuth = spherical[:, 1]
    elevation = spherical[:, 2]

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)

    return np.stack([x, y, z], axis=1)


def compute_overlap(
    points1: np.ndarray,
    points2: np.ndarray,
    T_12: np.ndarray,
    voxel_size: float = 0.2,
    max_points: int = 5000
) -> float:
    """
    Compute geometric overlap between two point clouds (optimized)

    Args:
        points1: (N, 3) first point cloud
        points2: (M, 3) second point cloud
        T_12: (4, 4) transformation from cloud1 to cloud2
        voxel_size: Voxel size for overlap computation
        max_points: Maximum points to use (downsampling for speed)

    Returns:
        IoU (Intersection over Union) ratio
    """
    # Downsample for speed (20x faster with 5000 vs 100000 points)
    if len(points1) > max_points:
        idx1 = np.random.choice(len(points1), max_points, replace=False)
        points1 = points1[idx1]

    if len(points2) > max_points:
        idx2 = np.random.choice(len(points2), max_points, replace=False)
        points2 = points2[idx2]

    # Transform points1 to frame of points2
    points1_transformed = transform_points(points1, T_12)

    # Voxelize both clouds using numpy (faster than set for large arrays)
    def voxelize_fast(points, voxel_size):
        # Filter invalid points
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]

        if len(points) == 0:
            # Return empty structured array
            dtype = [('x', np.int32), ('y', np.int32), ('z', np.int32)]
            return np.array([], dtype=dtype)

        # Clip values to prevent overflow
        points = np.clip(points, -1e6, 1e6)

        voxel_coords = np.floor(points / voxel_size).astype(np.int32)
        # Use structured array for fast unique
        dtype = [('x', np.int32), ('y', np.int32), ('z', np.int32)]
        voxels = np.empty(len(voxel_coords), dtype=dtype)
        voxels['x'] = voxel_coords[:, 0]
        voxels['y'] = voxel_coords[:, 1]
        voxels['z'] = voxel_coords[:, 2]
        return np.unique(voxels)

    voxels1 = voxelize_fast(points1_transformed, voxel_size)
    voxels2 = voxelize_fast(points2, voxel_size)

    # Convert to sets for IoU (only after unique, much smaller)
    set1 = set(map(tuple, [(v['x'], v['y'], v['z']) for v in voxels1]))
    set2 = set(map(tuple, [(v['x'], v['y'], v['z']) for v in voxels2]))

    # Compute IoU
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0
