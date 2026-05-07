"""
BEV (Bird's Eye View) Polar Grid Projection for LiDAR Point Clouds

Converts 3D LiDAR point clouds to 2D BEV polar grids:
- Rows represent radial distance at 1m resolution
- Columns represent angular sectors at 1° resolution
- Values represent max z-height per cell

Grid size: (max_range - min_range) × n_sectors (e.g., 79 × 360 for 1m-80m).
Sensor-agnostic: no elevation range calibration needed.
"""

import numpy as np
from typing import Tuple, Optional


def interpolate_bev_image(bev_image: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Interpolate empty cells (zeros) in BEV image along sector (azimuth) direction.

    Same logic as interpolate_range_image but for BEV polar grids.
    Empty cells cause FFT distortion; interpolation ensures clean spectral analysis.

    Args:
        bev_image: (n_rings, n_sectors) BEV image with 0 for empty cells
        method: 'linear' (sector direction, circular) or 'nearest'

    Returns:
        Interpolated BEV image with no empty cells
    """
    result = bev_image.copy()
    n_rings, n_sectors = bev_image.shape

    for row in range(n_rings):
        row_data = result[row]
        valid_mask = row_data != 0

        if not np.any(valid_mask):
            continue

        if np.all(valid_mask):
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_values = row_data[valid_mask]
        invalid_indices = np.where(~valid_mask)[0]

        if method == 'linear':
            # Circular linear interpolation (azimuth wraps around)
            extended_indices = np.concatenate([
                valid_indices - n_sectors,
                valid_indices,
                valid_indices + n_sectors
            ])
            extended_values = np.tile(valid_values, 3)
            interpolated = np.interp(invalid_indices, extended_indices, extended_values)
            result[row, invalid_indices] = interpolated

        elif method == 'nearest':
            for idx in invalid_indices:
                distances = np.minimum(
                    np.abs(valid_indices - idx),
                    n_sectors - np.abs(valid_indices - idx)
                )
                nearest_valid = valid_indices[np.argmin(distances)]
                result[row, idx] = row_data[nearest_valid]

    # Handle completely empty rings by copying from nearest non-empty ring
    for row in range(n_rings):
        if not np.any(result[row] != 0):
            for offset in range(1, n_rings):
                if row - offset >= 0 and np.any(result[row - offset] != 0):
                    result[row] = result[row - offset]
                    break
                if row + offset < n_rings and np.any(result[row + offset] != 0):
                    result[row] = result[row + offset]
                    break

    return result


class BEVProjector:
    """
    Projects 3D point clouds to 2D BEV polar grids.

    Creates a 2D representation where:
    - Rows represent radial distance at 1m resolution
    - Columns represent angular sectors at 1° resolution
    - Values represent either max z-height or LiDAR-Iris style height code

    Height encoding mode ('iris'):
    - Discretizes vertical space into n_height_layers binary layers
    - Each cell gets an integer code: sum(2^k for k where layer k is occupied)
    - Encodes full vertical occupancy profile (e.g., 8 layers -> 0~255)

    Grid size is determined by range: n_rows = int(max_range - min_range), n_cols = n_sectors.
    Full resolution is preserved (no pooling); FFT operates on all rows directly.

    Inherently sensor-agnostic: no elevation range needed.
    """

    def __init__(
        self,
        n_sectors: int = 360,
        max_range: float = 80.0,
        min_range: float = 1.0,
        z_min: float = -3.0,
        height_encoding: str = 'max',
        n_height_layers: int = 8,
        z_max: float = 5.0
    ):
        self.n_sectors = n_sectors
        self.max_range = max_range
        self.min_range = min_range
        self.z_min = z_min
        self.height_encoding = height_encoding
        self.n_height_layers = n_height_layers
        self.z_max = z_max
        # 1m radial resolution: each row covers 1m
        self.n_rings = int(max_range - min_range)

    def set_elevation_range(self, elevation_range: Tuple[float, float]):
        """No-op for BEV projection (sensor-agnostic)."""
        pass

    def project(
        self,
        points: np.ndarray,
        keep_intensity: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Project 3D point cloud to BEV polar grid.

        Args:
            points: (N, 3+) array of [x, y, z, ...]
            keep_intensity: Ignored (interface compatibility)

        Returns:
            bev_image: (n_rings, n_sectors) values per cell
                - 'max': max z-height (float)
                - 'iris': binary height code 0~2^n_height_layers-1 (float)
            None (no intensity image for BEV)
        """
        xyz = points[:, :3]

        # Filter invalid coordinates
        valid_coords = np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1]) & np.isfinite(xyz[:, 2])
        xyz = xyz[valid_coords]

        if len(xyz) == 0:
            return np.zeros((self.n_rings, self.n_sectors), dtype=np.float32), None

        # Horizontal range
        r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)

        # Filter by range and z_min
        valid = (r >= self.min_range) & (r < self.max_range) & (xyz[:, 2] > self.z_min)
        xyz = xyz[valid]
        r = r[valid]

        if len(xyz) == 0:
            return np.zeros((self.n_rings, self.n_sectors), dtype=np.float32), None

        # Azimuth: [0, 2*pi]
        theta = np.arctan2(xyz[:, 1], xyz[:, 0])
        theta = (theta + np.pi) % (2 * np.pi)

        # Bin into polar grid (1m radial resolution)
        ring_idx = np.clip(
            np.floor(r - self.min_range).astype(int),
            0, self.n_rings - 1
        )
        sector_idx = np.clip(
            np.floor(theta / (2 * np.pi) * self.n_sectors).astype(int),
            0, self.n_sectors - 1
        )

        if self.height_encoding == 'iris':
            return self._project_iris(xyz, ring_idx, sector_idx), None
        else:
            return self._project_max_height(xyz, ring_idx, sector_idx), None

    def _project_max_height(
        self,
        xyz: np.ndarray,
        ring_idx: np.ndarray,
        sector_idx: np.ndarray
    ) -> np.ndarray:
        """Fill grid with max z-height per cell."""
        linear_idx = ring_idx * self.n_sectors + sector_idx
        flat_bev = np.full(self.n_rings * self.n_sectors, -np.inf, dtype=np.float32)
        np.maximum.at(flat_bev, linear_idx, xyz[:, 2].astype(np.float32))

        bev_image = flat_bev.reshape(self.n_rings, self.n_sectors)
        bev_image[bev_image == -np.inf] = 0.0
        return bev_image

    def _project_iris(
        self,
        xyz: np.ndarray,
        ring_idx: np.ndarray,
        sector_idx: np.ndarray
    ) -> np.ndarray:
        """
        LiDAR-Iris style height encoding.

        Discretizes [z_min, z_max] into n_height_layers binary layers.
        Each cell gets an integer code encoding which layers are occupied:
            code = sum(2^k for each occupied layer k)

        E.g., n_height_layers=8, z_min=-3, z_max=5 -> 1m per layer
        A cell with points at z=-2 (layer 1) and z=3 (layer 6) gets code = 2^1 + 2^6 = 66
        """
        n_layers = self.n_height_layers
        z_heights = xyz[:, 2].astype(np.float32)

        # Assign each point to a height layer
        layer_idx = np.clip(
            np.floor((z_heights - self.z_min) / (self.z_max - self.z_min) * n_layers).astype(int),
            0, n_layers - 1
        )

        # Compute bit value for each point: 2^layer_idx
        bit_values = (1 << layer_idx).astype(np.float32)

        # Accumulate bits per cell using bitwise OR via maximum on unique bit positions
        # For each (ring, sector, layer) -> mark as occupied
        bev_image = np.zeros((self.n_rings, self.n_sectors), dtype=np.int32)
        linear_idx = ring_idx * self.n_sectors + sector_idx
        flat_bev = np.zeros(self.n_rings * self.n_sectors, dtype=np.int32)

        # Bitwise OR: for each point, set the corresponding bit
        np.bitwise_or.at(flat_bev, linear_idx, (1 << layer_idx).astype(np.int32))

        bev_image = flat_bev.reshape(self.n_rings, self.n_sectors).astype(np.float32)
        return bev_image
