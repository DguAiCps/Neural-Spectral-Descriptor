"""Tests for StandardizationStats (z-score normalization)."""

import numpy as np
import pytest
import tempfile
import os
from utils.standardization_stats import StandardizationStats


class TestStandardizationStats:

    @pytest.fixture
    def training_data(self):
        """(100, 64) random descriptors with known mean/std."""
        rng = np.random.RandomState(42)
        return rng.randn(100, 64).astype(np.float32) * 3 + 5

    def test_fit(self, training_data):
        stats = StandardizationStats()
        stats.fit(training_data)
        assert stats.fitted is True
        assert stats.mean.shape == (64,)
        assert stats.std.shape == (64,)

    def test_transform_shape(self, training_data):
        stats = StandardizationStats().fit(training_data)
        transformed = stats.transform(training_data)
        assert transformed.shape == training_data.shape
        assert transformed.dtype == np.float32

    def test_transform_zero_mean(self, training_data):
        stats = StandardizationStats().fit(training_data)
        transformed = stats.transform(training_data)
        means = transformed.mean(axis=0)
        np.testing.assert_allclose(means, 0, atol=0.05)

    def test_transform_unit_std(self, training_data):
        stats = StandardizationStats().fit(training_data)
        transformed = stats.transform(training_data)
        stds = transformed.std(axis=0)
        np.testing.assert_allclose(stds, 1.0, atol=0.15)

    def test_transform_single_vector(self, training_data):
        stats = StandardizationStats().fit(training_data)
        single = training_data[0]
        transformed = stats.transform(single)
        assert transformed.shape == (64,)

    def test_not_fitted_raises(self):
        stats = StandardizationStats()
        with pytest.raises(RuntimeError, match="Not fitted"):
            stats.transform(np.zeros((10, 64)))

    def test_near_zero_std_clamped(self):
        """Constant columns should have std clamped to epsilon."""
        data = np.zeros((50, 4), dtype=np.float32)
        data[:, 0] = 1.0  # constant
        data[:, 1] = np.arange(50)  # varying
        stats = StandardizationStats(epsilon=1e-8).fit(data)
        # Column 0 std should be clamped
        assert stats.std[0] == pytest.approx(1e-8)
        # Transform should not produce inf/nan
        transformed = stats.transform(data)
        assert np.all(np.isfinite(transformed))

    def test_save_load(self, training_data):
        stats = StandardizationStats().fit(training_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'stats.npz')
            stats.save(path)
            loaded = StandardizationStats().load(path)
            assert loaded.fitted is True
            np.testing.assert_array_equal(stats.mean, loaded.mean)
            np.testing.assert_array_equal(stats.std, loaded.std)

    def test_chaining(self, training_data):
        """fit() returns self for method chaining."""
        result = StandardizationStats().fit(training_data)
        assert isinstance(result, StandardizationStats)
