"""
Panoramic Range Image Projection for LiDAR Point Clouds

Converts 3D LiDAR point clouds to 2D panoramic range images
following the KITTI HDL-64E specifications:
- 64 elevation rings
- 360 azimuth bins (1 degree resolution)
"""

import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


def interpolate_range_image(range_image: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Interpolate empty pixels (zeros) in range image.

    This is critical for sensor-invariant FFT-based descriptors.
    Empty pixels cause FFT distortion - same place with different sensor
    density produces completely different frequency responses.

    Args:
        range_image: (n_elevation, n_azimuth) range image with 0 for empty pixels
        method: 'linear' (azimuth direction) or 'nearest'

    Returns:
        Interpolated range image with no empty pixels
    """
    result = range_image.copy()
    n_elevation, n_azimuth = range_image.shape

    for row in range(n_elevation):
        row_data = result[row]
        valid_mask = row_data > 0

        if not np.any(valid_mask):
            # Completely empty row - skip or use neighbor rows
            continue

        if np.all(valid_mask):
            # No interpolation needed
            continue

        # Get valid indices and values
        valid_indices = np.where(valid_mask)[0]
        valid_values = row_data[valid_mask]

        # Get invalid indices
        invalid_indices = np.where(~valid_mask)[0]

        if method == 'linear':
            # Circular linear interpolation (azimuth wraps around)
            # Extend valid data for circular boundary handling
            extended_indices = np.concatenate([
                valid_indices - n_azimuth,
                valid_indices,
                valid_indices + n_azimuth
            ])
            extended_values = np.tile(valid_values, 3)

            # Interpolate
            interpolated = np.interp(invalid_indices, extended_indices, extended_values)
            result[row, invalid_indices] = interpolated

        elif method == 'nearest':
            # Nearest neighbor interpolation
            for idx in invalid_indices:
                # Find nearest valid pixel (circular)
                distances = np.minimum(
                    np.abs(valid_indices - idx),
                    n_azimuth - np.abs(valid_indices - idx)
                )
                nearest_valid = valid_indices[np.argmin(distances)]
                result[row, idx] = row_data[nearest_valid]

    # Handle completely empty rows by copying from nearest non-empty row
    for row in range(n_elevation):
        if not np.any(result[row] > 0):
            # Find nearest non-empty row
            for offset in range(1, n_elevation):
                if row - offset >= 0 and np.any(result[row - offset] > 0):
                    result[row] = result[row - offset]
                    break
                if row + offset < n_elevation and np.any(result[row + offset] > 0):
                    result[row] = result[row + offset]
                    break

    return result


class RangeImageProjector:
    """
    Projects 3D point clouds to 2D panoramic range images

    Creates a 2D representation where:
    - Rows represent elevation rings (64 for HDL-64E)
    - Columns represent azimuth bins (360 for 1-degree resolution)
    - Values represent range (distance from sensor)
    """

    def __init__(
        self,
        n_elevation: int = 64,
        n_azimuth: int = 360,
        elevation_range: Tuple[float, float] = (-24.8, 2.0),
        max_range: float = 80.0,
        min_range: float = 1.0
    ):
        """
        Initialize range image projector

        Args:
            n_elevation: Number of elevation rings (64 for HDL-64E)
            n_azimuth: Number of azimuth bins (360 for 1-degree resolution)
            elevation_range: (min, max) elevation angles in degrees
            max_range: Maximum valid range in meters
            min_range: Minimum valid range in meters
        """
        self.n_elevation = n_elevation
        self.n_azimuth = n_azimuth
        self.max_range = max_range
        self.min_range = min_range

        # Convert elevation range to radians
        self.elevation_min = np.deg2rad(elevation_range[0])
        self.elevation_max = np.deg2rad(elevation_range[1])

    def set_elevation_range(self, elevation_range: Tuple[float, float]):
        """Update elevation range (in degrees) for a different sensor"""
        self.elevation_min = np.deg2rad(elevation_range[0])
        self.elevation_max = np.deg2rad(elevation_range[1])

    def project(
        self,
        points: np.ndarray,
        keep_intensity: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Project 3D point cloud to 2D range image

        Args:
            points: (N, 3) or (N, 4) array of [x, y, z] or [x, y, z, intensity]
            keep_intensity: If True, also return intensity image

        Returns:
            range_image: (n_elevation, n_azimuth) range values
            intensity_image: (n_elevation, n_azimuth) intensity values (if keep_intensity)
        """
        # Extract coordinates
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Filter invalid coordinates first (NaN, Inf)
        valid_coords = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x = x[valid_coords]
        y = y[valid_coords]
        z = z[valid_coords]
        points = points[valid_coords]

        # Compute spherical coordinates with overflow protection
        # Range - use clipping to prevent overflow
        x_sq = np.clip(x**2, 0, 1e10)
        y_sq = np.clip(y**2, 0, 1e10)
        z_sq = np.clip(z**2, 0, 1e10)
        range_vals = np.sqrt(x_sq + y_sq + z_sq)

        # Azimuth: angle in xy-plane from +x axis
        # Map from [-pi, pi] to [0, 2*pi] for easier binning
        azimuth = np.arctan2(y, x)  # [-pi, pi]
        azimuth = (azimuth + np.pi) % (2 * np.pi)  # [0, 2*pi]

        # Elevation: angle from xy-plane
        xy_range = np.sqrt(x_sq + y_sq)
        elevation = np.arctan2(z, xy_range)

        # Filter points by range
        valid_mask = (range_vals >= self.min_range) & (range_vals <= self.max_range) & np.isfinite(range_vals)
        range_vals = range_vals[valid_mask]
        azimuth = azimuth[valid_mask]
        elevation = elevation[valid_mask]

        if points.shape[1] == 4:
            intensity = points[:, 3][valid_mask]
        else:
            intensity = None

        # Bin into 2D grid
        # Elevation bins: map [-24.8°, 2.0°] to [0, 63]
        elev_norm = (elevation - self.elevation_min) / (self.elevation_max - self.elevation_min)
        elev_bins = np.clip(
            np.floor(elev_norm * self.n_elevation).astype(int),
            0,
            self.n_elevation - 1
        )

        # Azimuth bins: map [0, 2*pi] to [0, 359]
        azim_bins = np.clip(
            np.floor(azimuth / (2 * np.pi) * self.n_azimuth).astype(int),
            0,
            self.n_azimuth - 1
        )

        # Vectorized range image filling using np.minimum.at
        # Convert 2D indices to linear indices for vectorized operation
        linear_idx = elev_bins * self.n_azimuth + azim_bins

        # Initialize flat array with inf
        flat_range = np.full(self.n_elevation * self.n_azimuth, np.inf, dtype=np.float32)

        # Use minimum.at for keeping closest point at each pixel (vectorized)
        np.minimum.at(flat_range, linear_idx, range_vals)

        # Reshape to 2D image
        range_image = flat_range.reshape(self.n_elevation, self.n_azimuth)

        # Replace inf with 0 for empty cells
        range_image[range_image == np.inf] = 0.0

        # Handle intensity image if needed
        if keep_intensity and intensity is not None:
            # For intensity, we need to get the intensity of the closest point
            # Find which points are the closest at each pixel
            flat_intensity = np.zeros(self.n_elevation * self.n_azimuth, dtype=np.float32)

            # Create mask for points that are the closest at their pixel
            closest_mask = (range_vals == flat_range[linear_idx])

            # Use maximum.at to handle multiple closest points (take any one)
            np.maximum.at(flat_intensity, linear_idx[closest_mask], intensity[closest_mask])

            intensity_image = flat_intensity.reshape(self.n_elevation, self.n_azimuth)
        else:
            intensity_image = None

        return range_image, intensity_image

    def unproject(
        self,
        range_image: np.ndarray,
        intensity_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert range image back to 3D point cloud

        Args:
            range_image: (n_elevation, n_azimuth) range values
            intensity_image: Optional (n_elevation, n_azimuth) intensity values

        Returns:
            points: (N, 3) or (N, 4) point cloud
        """
        # Get valid points (non-zero range)
        valid_mask = range_image > 0

        # Compute elevation and azimuth for each pixel
        elevation_bins = np.arange(self.n_elevation)
        azimuth_bins = np.arange(self.n_azimuth)

        # Create meshgrid
        elev_grid, azim_grid = np.meshgrid(elevation_bins, azimuth_bins, indexing='ij')

        # Convert bins to angles
        # Elevation: [0, 63] -> [-24.8°, 2.0°]
        elevation = (
            self.elevation_min +
            (elev_grid / self.n_elevation) * (self.elevation_max - self.elevation_min)
        )

        # Azimuth: [0, 359] -> [0, 2*pi]
        azimuth = (azim_grid / self.n_azimuth) * 2 * np.pi

        # Get ranges at valid locations
        ranges = range_image[valid_mask]
        elevations = elevation[valid_mask]
        azimuths = azimuth[valid_mask]

        # Convert to Cartesian
        x = ranges * np.cos(elevations) * np.cos(azimuths)
        y = ranges * np.cos(elevations) * np.sin(azimuths)
        z = ranges * np.sin(elevations)

        if intensity_image is not None:
            intensities = intensity_image[valid_mask]
            points = np.stack([x, y, z, intensities], axis=1)
        else:
            points = np.stack([x, y, z], axis=1)

        return points

    def visualize_range_image(self, range_image: np.ndarray) -> np.ndarray:
        """
        Convert range image to normalized visualization

        Args:
            range_image: (n_elevation, n_azimuth) range values

        Returns:
            (n_elevation, n_azimuth) normalized image in [0, 1]
        """
        # Normalize to [0, 1] based on max_range
        vis = np.clip(range_image / self.max_range, 0, 1)
        return vis


def project_to_range_image(
    points: np.ndarray,
    n_elevation: int = 64,
    n_azimuth: int = 360
) -> np.ndarray:
    """
    Convenience function to project points to range image

    Args:
        points: (N, 3) or (N, 4) point cloud
        n_elevation: Number of elevation rings
        n_azimuth: Number of azimuth bins

    Returns:
        (n_elevation, n_azimuth) range image
    """
    projector = RangeImageProjector(
        n_elevation=n_elevation,
        n_azimuth=n_azimuth
    )
    range_image, _ = projector.project(points, keep_intensity=False)
    return range_image


def compute_range_image_difference(
    range_image1: np.ndarray,
    range_image2: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute normalized difference between two range images

    Args:
        range_image1: First range image
        range_image2: Second range image
        threshold: Difference threshold in meters

    Returns:
        Fraction of pixels with difference > threshold
    """
    # Only compare valid pixels (both non-zero)
    valid_mask = (range_image1 > 0) & (range_image2 > 0)

    if valid_mask.sum() == 0:
        return 1.0  # Completely different

    diff = np.abs(range_image1 - range_image2)
    different_pixels = (diff[valid_mask] > threshold).sum()

    return different_pixels / valid_mask.sum()
