"""
M2DP baseline — Multi-view 2D Projection descriptor.

Reference: He, Li, et al. (2016) "M2DP: A Novel 3D Point Cloud Descriptor
and Its Application in Loop Closure Detection"

Projects 3D points onto multiple 2D planes defined by (azimuth, elevation)
pairs, computes polar histograms on each plane, then extracts the principal
components via SVD as the descriptor.
"""

import numpy as np
from baselines.base import BaselineEncoder
from baselines import register


@register
class M2DP(BaselineEncoder):
    """M2DP: Multi-view 2D Projection descriptor (192D)."""

    def __init__(self, n_azimuth_planes=4, n_elevation_planes=16,
                 n_distance_bins=8, n_angle_bins=16, max_range=80.0,
                 max_points=4096):
        self.p = n_azimuth_planes
        self.q = n_elevation_planes
        self.l = n_distance_bins
        self.t = n_angle_bins
        self.max_range = max_range
        self.max_points = max_points

    @property
    def name(self):
        return "M2DP"

    @property
    def short_name(self):
        return "m2dp"

    @property
    def descriptor_dim(self):
        return self.p * self.q + self.l * self.t  # 64 + 128 = 192

    @staticmethod
    def _orthonormal_basis(normal):
        """Compute two orthonormal vectors perpendicular to normal."""
        n = normal / (np.linalg.norm(normal) + 1e-10)
        # Choose a vector not parallel to n
        if abs(n[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, ref)
        u = u / (np.linalg.norm(u) + 1e-10)
        v = np.cross(n, u)
        v = v / (np.linalg.norm(v) + 1e-10)
        return u, v

    def encode(self, points):
        xyz = points[:, :3].astype(np.float64)

        # Filter by range
        dists = np.linalg.norm(xyz, axis=1)
        valid = (dists > 0.1) & (dists < self.max_range)
        xyz = xyz[valid]

        if len(xyz) < 10:
            return np.zeros(self.descriptor_dim, dtype=np.float32)

        # Downsample for speed (original M2DP paper uses ~4096 points)
        if len(xyz) > self.max_points:
            indices = np.random.choice(len(xyz), self.max_points, replace=False)
            xyz = xyz[indices]

        # In LiDAR SLAM the sensor is at the origin, so no centering is applied.
        # The original M2DP paper centers by cloud centroid for object recognition
        # (arbitrary pose), but for place recognition the sensor-centric frame is
        # already consistent across scans — centering by the point cloud centroid
        # would shift the reference away from the sensor and discard directional
        # structure, hurting discrimination.
        xyz_c = xyz  # (N, 3), sensor at origin

        # Max distance for histogram binning
        max_dist = np.percentile(np.linalg.norm(xyz_c, axis=1), 99) + 1e-6

        # Precompute all plane bases (vectorized)
        azimuths = np.linspace(0, np.pi, self.p, endpoint=False)
        elevations = np.linspace(0, np.pi / 2, self.q, endpoint=False)

        n_planes = self.p * self.q
        normals = np.zeros((n_planes, 3))
        us = np.zeros((n_planes, 3))
        vs = np.zeros((n_planes, 3))

        for ai, az in enumerate(azimuths):
            for ei, el in enumerate(elevations):
                idx = ai * self.q + ei
                cos_el, sin_el = np.cos(el), np.sin(el)
                cos_az, sin_az = np.cos(az), np.sin(az)
                normal = np.array([cos_el * cos_az, cos_el * sin_az, sin_el])
                u, v = self._orthonormal_basis(normal)
                normals[idx] = normal
                us[idx] = u
                vs[idx] = v

        # Batch project: (n_planes, N) = (n_planes, 3) @ (3, N)
        proj_u = us @ xyz_c.T  # (n_planes, N)
        proj_v = vs @ xyz_c.T  # (n_planes, N)

        r_2d = np.sqrt(proj_u ** 2 + proj_v ** 2)
        theta_2d = np.arctan2(proj_v, proj_u) + np.pi  # [0, 2*pi]

        r_bin = np.clip((r_2d / max_dist * self.l).astype(np.int32), 0, self.l - 1)
        t_bin = np.clip((theta_2d / (2 * np.pi) * self.t).astype(np.int32), 0, self.t - 1)
        linear_idx = r_bin * self.t + t_bin  # (n_planes, N)

        # Build histograms for all planes at once
        A = np.zeros((n_planes, self.l * self.t), dtype=np.float64)
        for pi in range(n_planes):
            np.add.at(A[pi], linear_idx[pi], 1)

        # SVD decomposition
        U, S, Vt = np.linalg.svd(A, full_matrices=False)

        # Descriptor: first left + first right singular vectors
        descriptor = np.concatenate([U[:, 0], Vt[0, :]])

        norm = np.linalg.norm(descriptor)
        if norm > 1e-8:
            descriptor = descriptor / norm
        else:
            descriptor = np.zeros(self.descriptor_dim, dtype=np.float64)

        return descriptor.astype(np.float32)
