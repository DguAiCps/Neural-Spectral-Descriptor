"""Tests for BEV polar grid projection."""

import numpy as np
import pytest
from encoding.bev_image import BEVProjector, interpolate_bev_image


class TestBEVProjector:

    def test_output_shape(self, random_point_cloud):
        proj = BEVProjector(n_sectors=360, max_range=80.0, min_range=1.0)
        bev, intensity = proj.project(random_point_cloud)
        assert bev.shape == (79, 360)
        assert intensity is None

    def test_n_rings_calculation(self):
        proj = BEVProjector(max_range=50.0, min_range=5.0)
        assert proj.n_rings == 45

    def test_empty_point_cloud(self):
        proj = BEVProjector()
        bev, _ = proj.project(np.empty((0, 3), dtype=np.float32))
        assert bev.shape == (79, 360)
        assert np.all(bev == 0)

    def test_max_height_encoding(self, random_point_cloud):
        proj = BEVProjector(height_encoding='max')
        bev, _ = proj.project(random_point_cloud)
        # Max height cells should be finite float values
        nonzero = bev[bev != 0]
        assert len(nonzero) > 0
        assert np.all(np.isfinite(nonzero))

    def test_iris_height_encoding(self, random_point_cloud):
        proj = BEVProjector(height_encoding='iris', n_height_layers=8)
        bev, _ = proj.project(random_point_cloud)
        nonzero = bev[bev != 0]
        assert len(nonzero) > 0
        # Iris values should be 0~255 (2^8 - 1)
        assert np.all(nonzero >= 0)
        assert np.all(nonzero <= 255)

    def test_range_filtering(self):
        """Points outside [min_range, max_range) are excluded."""
        proj = BEVProjector(max_range=10.0, min_range=5.0)
        # Point at range=3 (too close) and range=12 (too far)
        points = np.array([[3.0, 0, 0], [12.0, 0, 0], [7.0, 0, 0.5]], dtype=np.float32)
        bev, _ = proj.project(points)
        # Only the point at range=7 should be in the grid
        assert bev.sum() > 0
        # ring_idx for r=7: floor(7-5)=2
        assert bev[2, :].sum() > 0

    def test_z_min_filtering(self):
        """Points below z_min are filtered out."""
        proj = BEVProjector(z_min=-3.0)
        below = np.array([[10.0, 0, -5.0]], dtype=np.float32)
        bev, _ = proj.project(below)
        assert np.all(bev == 0)

    def test_set_elevation_range_noop(self):
        """BEV projector ignores elevation range (sensor-agnostic)."""
        proj = BEVProjector()
        proj.set_elevation_range((-30.0, 10.0))  # should not error

    def test_dtype_float32(self, random_point_cloud):
        proj = BEVProjector()
        bev, _ = proj.project(random_point_cloud)
        assert bev.dtype == np.float32

    def test_azimuth_coverage(self):
        """Points at 0° and 180° should land in different sectors."""
        proj = BEVProjector(n_sectors=360)
        points = np.array([
            [10.0, 0.0, 1.0],    # ~180° (arctan2(0,10)+π = π), z=1 so nonzero
            [-10.0, 0.0, 1.0],   # ~0° (arctan2(0,-10)+π = 2π → 0), z=1 so nonzero
        ], dtype=np.float32)
        bev, _ = proj.project(points)
        nonzero_cols = np.where(bev.sum(axis=0) > 0)[0]
        assert len(nonzero_cols) == 2


class TestInterpolateBEV:

    def test_fills_empty_cells(self):
        """Interpolation should fill zero cells between valid ones."""
        bev = np.zeros((5, 10), dtype=np.float32)
        bev[0, 0] = 1.0
        bev[0, 5] = 2.0
        result = interpolate_bev_image(bev, method='linear')
        # Cells between 0 and 5 should now be interpolated
        assert result[0, 2] > 0

    def test_no_change_if_full(self):
        """Full image should be unchanged."""
        bev = np.ones((5, 10), dtype=np.float32)
        result = interpolate_bev_image(bev)
        np.testing.assert_array_equal(bev, result)

    def test_nearest_method(self):
        bev = np.zeros((3, 10), dtype=np.float32)
        bev[0, 0] = 5.0
        result = interpolate_bev_image(bev, method='nearest')
        # All cells in row 0 should be filled with 5.0
        assert np.all(result[0] == 5.0)

    def test_empty_ring_copy(self):
        """Completely empty ring copies from nearest non-empty ring."""
        bev = np.zeros((3, 10), dtype=np.float32)
        bev[0, :] = np.arange(10, dtype=np.float32) + 1
        # Row 1 is empty
        result = interpolate_bev_image(bev)
        assert np.any(result[1] != 0)

    def test_does_not_modify_input(self):
        """Interpolation should not modify the input array."""
        bev = np.zeros((3, 10), dtype=np.float32)
        bev[0, 0] = 1.0
        original = bev.copy()
        _ = interpolate_bev_image(bev)
        np.testing.assert_array_equal(bev, original)
