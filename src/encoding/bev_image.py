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


def interpolate_bev_image(
    bev_image: np.ndarray,
    method: str = 'linear',
    n_channels: int = 1,
) -> np.ndarray:
    """
    Interpolate empty cells (zeros) in BEV image along sector (azimuth) direction.

    Same logic as interpolate_range_image but for BEV polar grids.
    Empty cells cause FFT distortion; interpolation ensures clean spectral analysis.

    Args:
        bev_image: (n_rings, n_sectors) BEV image with 0 for empty cells
        method: 'linear' (sector direction, circular) or 'nearest'
        n_channels: Number of row-stacked physical channels. For physics3 this
            must be 3 so empty-ring copying never crosses channel boundaries.

    Returns:
        Interpolated BEV image with no empty cells
    """
    result = bev_image.copy()
    n_rings, n_sectors = bev_image.shape
    n_channels = max(1, int(n_channels))
    if n_rings % n_channels != 0:
        raise ValueError(
            f"BEV rows ({n_rings}) must be divisible by n_channels={n_channels}"
        )
    rows_per_channel = n_rings // n_channels

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

    # Handle completely empty rings by copying from nearest non-empty ring, but
    # never cross row-stacked channel boundaries (critical for physics3).
    for ch in range(n_channels):
        start = ch * rows_per_channel
        end = start + rows_per_channel
        for row in range(start, end):
            if not np.any(result[row] != 0):
                for offset in range(1, rows_per_channel):
                    if row - offset >= start and np.any(result[row - offset] != 0):
                        result[row] = result[row - offset]
                        break
                    if row + offset < end and np.any(result[row + offset] != 0):
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
            bev_image: (n_rings, n_sectors) values per cell, or
                (3 * n_rings, n_sectors) for 'physics3'
                - 'max': max z-height (float)
                - 'iris': binary height code 0~2^n_height_layers-1 (float)
                - 'physics3': stacked [normalized max height, log occupancy
                  density, normalized vertical span]. This keeps deterministic
                  physical sensor cues without learning an encoder.
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
        elif self.height_encoding == 'physics3':
            return self._project_physics3(xyz, ring_idx, sector_idx), None
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

    def _project_physics3(
        self,
        xyz: np.ndarray,
        ring_idx: np.ndarray,
        sector_idx: np.ndarray,
    ) -> np.ndarray:
        """Stack three deterministic physical BEV channels.

        The goal is to mimic part of multi-sensor fine-tuning with domain
        knowledge instead of a large learned encoder:
        - max height: geometry/structure cue;
        - occupancy density: sampling/visibility cue, normalized by polar cell
          area so sparse sensors are less biased toward near-range returns;
        - vertical span: object/vegetation/building thickness cue independent of
          absolute ground height.

        The channels are row-stacked as ``(3 * n_rings, n_sectors)`` so all
        existing FFT/phase-sketch code can consume them unchanged. Use
        ``bev_row_pool=48`` and ``bev_freqs=4`` to keep the 384D phase budget:
        ``48 rows * 4 freqs * 2 = 384``.
        """
        n_cells = self.n_rings * self.n_sectors
        linear_idx = ring_idx * self.n_sectors + sector_idx
        z = xyz[:, 2].astype(np.float32)

        flat_max = np.full(n_cells, -np.inf, dtype=np.float32)
        flat_min = np.full(n_cells, np.inf, dtype=np.float32)
        flat_count = np.zeros(n_cells, dtype=np.float32)
        np.maximum.at(flat_max, linear_idx, z)
        np.minimum.at(flat_min, linear_idx, z)
        np.add.at(flat_count, linear_idx, 1.0)

        occupied = flat_count > 0
        flat_max[~occupied] = 0.0
        flat_min[~occupied] = 0.0

        height = np.zeros(n_cells, dtype=np.float32)
        span = np.zeros(n_cells, dtype=np.float32)
        z_range = max(float(self.z_max - self.z_min), 1e-6)
        height[occupied] = np.clip((flat_max[occupied] - self.z_min) / z_range, 0.0, 1.0)
        span[occupied] = np.clip((flat_max[occupied] - flat_min[occupied]) / z_range, 0.0, 1.0)

        # Polar cells cover larger physical area at larger radius. Divide by
        # ring radius to turn raw point count into a rough sampling density.
        ring_centers = self.min_range + np.arange(self.n_rings, dtype=np.float32) + 0.5
        density_norm = np.repeat(ring_centers, self.n_sectors)
        density = np.log1p(flat_count / np.maximum(density_norm, 1.0)).astype(np.float32)
        if density.max() > 0:
            density = density / density.max()

        stacked = np.stack(
            [
                height.reshape(self.n_rings, self.n_sectors),
                density.reshape(self.n_rings, self.n_sectors),
                span.reshape(self.n_rings, self.n_sectors),
            ],
            axis=0,
        )
        return stacked.reshape(3 * self.n_rings, self.n_sectors).astype(np.float32)
