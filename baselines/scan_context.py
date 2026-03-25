"""
Scan Context++ baseline — Ring Key descriptor.

Reference: Kim et al. (2022) "Scan Context++: Structural Place Recognition
Robust to Rotation and Lateral Variations"

Uses the rotation-invariant Ring Key (per-ring mean of max z-height)
for FAISS-based cosine similarity retrieval.

Note: This uses single-stage Ring Key retrieval only. The full SC++ pipeline
includes column-shift re-ranking for rotation alignment, which is not used here.
"""

import numpy as np
from baselines.base import BaselineEncoder
from baselines import register


def build_scan_context(points, n_rings=20, n_sectors=60, max_range=80.0, z_min=-3.0):
    """
    Build BEV polar grid (Scan Context matrix).

    Args:
        points: (N, 3+) point cloud
        n_rings: Number of radial divisions
        n_sectors: Number of angular divisions
        max_range: Maximum range in meters
        z_min: Minimum z for ground filtering

    Returns:
        (n_rings, n_sectors) Scan Context matrix (max z-height per cell)
    """
    xyz = points[:, :3]
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    valid = (r < max_range) & (r > 0.1) & (xyz[:, 2] > z_min)
    xyz = xyz[valid]
    r = r[valid]

    if len(xyz) == 0:
        return np.zeros((n_rings, n_sectors), dtype=np.float32)

    theta = np.arctan2(xyz[:, 1], xyz[:, 0]) + np.pi  # [0, 2*pi]

    ring_idx = np.clip((r / max_range * n_rings).astype(int), 0, n_rings - 1)
    sector_idx = np.clip((theta / (2 * np.pi) * n_sectors).astype(int), 0, n_sectors - 1)

    sc = np.full((n_rings, n_sectors), -np.inf, dtype=np.float32)
    np.maximum.at(sc, (ring_idx, sector_idx), xyz[:, 2].astype(np.float32))
    sc[sc == -np.inf] = 0.0

    return sc


@register
class ScanContextPP(BaselineEncoder):
    """Scan Context++ with Ring Key descriptor."""

    def __init__(self, n_rings=20, n_sectors=60, max_range=80.0, z_min=-3.0):
        self.n_rings = n_rings
        self.n_sectors = n_sectors
        self.max_range = max_range
        self.z_min = z_min

    @property
    def name(self):
        return "Scan Context++"

    @property
    def short_name(self):
        return "sc++"

    @property
    def descriptor_dim(self):
        return self.n_rings

    def encode(self, points):
        sc = build_scan_context(
            points, self.n_rings, self.n_sectors, self.max_range, self.z_min
        )
        # Ring Key: mean of each ring row (rotation-invariant)
        ring_key = sc.mean(axis=1)  # (n_rings,)
        norm = np.linalg.norm(ring_key)
        if norm > 1e-8:
            ring_key = ring_key / norm
        else:
            ring_key = np.zeros(self.n_rings, dtype=np.float32)
        return ring_key.astype(np.float32)
