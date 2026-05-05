"""Tests for SimilarityDistribution (Bayesian edge selection)."""

import numpy as np
import pytest
import tempfile
import os
from utils.similarity_stats import SimilarityDistribution, _sigmoid


class TestSigmoid:

    def test_range(self):
        x = np.linspace(-10, 10, 100)
        y = _sigmoid(x)
        assert np.all(y >= 0)
        assert np.all(y <= 1)

    def test_midpoint(self):
        assert _sigmoid(np.array([0.0])) == pytest.approx(0.5)

    def test_numerical_stability(self):
        """Large positive/negative values should not overflow."""
        assert np.isfinite(_sigmoid(np.array([1000.0])))
        assert np.isfinite(_sigmoid(np.array([-1000.0])))


class TestSimilarityDistribution:

    @pytest.fixture
    def loop_data_cosine(self):
        """Synthetic data with clear same/different place pairs."""
        rng = np.random.RandomState(42)
        n = 200
        descriptors = rng.randn(n, 64).astype(np.float32)
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors /= np.maximum(norms, 1e-8)

        poses = np.zeros((n, 4, 4))
        for i in range(n):
            poses[i] = np.eye(4)
            # Circular trajectory: revisits at i and i+100
            angle = 2 * np.pi * i / 100
            poses[i, 0, 3] = 20 * np.cos(angle)
            poses[i, 1, 3] = 20 * np.sin(angle)
        return descriptors, poses

    @pytest.fixture
    def loop_data_l2(self):
        """Synthetic z-scored descriptors for L2 metric."""
        rng = np.random.RandomState(42)
        n = 200
        descriptors = rng.randn(n, 64).astype(np.float32)
        poses = np.zeros((n, 4, 4))
        for i in range(n):
            poses[i] = np.eye(4)
            angle = 2 * np.pi * i / 100
            poses[i, 0, 3] = 20 * np.cos(angle)
            poses[i, 1, 3] = 20 * np.sin(angle)
        return descriptors, poses

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric"):
            SimilarityDistribution(metric='invalid')

    def test_cosine_fit(self, loop_data_cosine):
        desc, poses = loop_data_cosine
        dist = SimilarityDistribution(metric='cosine').fit(
            desc, poses, pos_dist=5.0, neg_dist=10.0,
            min_temporal_gap=5, n_samples=50000,
        )
        assert dist.fitted is True

    def test_l2_fit(self, loop_data_l2):
        desc, poses = loop_data_l2
        dist = SimilarityDistribution(metric='l2').fit(
            desc, poses, pos_dist=5.0, neg_dist=10.0,
            min_temporal_gap=5, n_samples=50000,
        )
        assert dist.fitted is True

    def test_posterior_range(self, loop_data_cosine):
        desc, poses = loop_data_cosine
        dist = SimilarityDistribution(metric='cosine').fit(
            desc, poses, pos_dist=5.0, neg_dist=10.0,
            min_temporal_gap=5, n_samples=50000,
        )
        if not dist.fitted:
            pytest.skip("Distribution not fitted (insufficient pairs)")
        post = dist.posterior(0.99, prior=0.01)
        assert 0 <= float(post) <= 1

    def test_posterior_array(self, loop_data_cosine):
        desc, poses = loop_data_cosine
        dist = SimilarityDistribution(metric='cosine').fit(
            desc, poses, pos_dist=5.0, neg_dist=10.0,
            min_temporal_gap=5, n_samples=50000,
        )
        if not dist.fitted:
            pytest.skip("Distribution not fitted")
        observations = np.array([0.5, 0.8, 0.99])
        posteriors = dist.posterior(observations, prior=0.01)
        assert posteriors.shape == (3,)
        assert np.all(posteriors >= 0) and np.all(posteriors <= 1)

    def test_not_fitted_raises(self):
        dist = SimilarityDistribution(metric='cosine')
        with pytest.raises(RuntimeError, match="not fitted"):
            dist.posterior(0.9)

    def test_confidence_threshold_cosine(self, loop_data_cosine):
        desc, poses = loop_data_cosine
        dist = SimilarityDistribution(metric='cosine').fit(
            desc, poses, pos_dist=5.0, neg_dist=10.0,
            min_temporal_gap=5, n_samples=50000,
        )
        if not dist.fitted:
            pytest.skip("Distribution not fitted")
        threshold = dist.confidence_threshold(0.95, prior=0.01)
        assert -1 <= threshold <= 1

    def test_confidence_threshold_l2(self, loop_data_l2):
        desc, poses = loop_data_l2
        dist = SimilarityDistribution(metric='l2').fit(
            desc, poses, pos_dist=5.0, neg_dist=10.0,
            min_temporal_gap=5, n_samples=50000,
        )
        if not dist.fitted:
            pytest.skip("Distribution not fitted")
        threshold = dist.confidence_threshold(0.95, prior=0.01)
        assert threshold >= 0

    def test_adaptive_priors_density(self, loop_data_cosine):
        desc, poses = loop_data_cosine
        dist = SimilarityDistribution(metric='cosine').fit(
            desc, poses, pos_dist=5.0, neg_dist=10.0,
            min_temporal_gap=5, n_samples=50000,
        )
        densities = np.random.rand(50)
        priors = dist.compute_adaptive_priors(densities, base_prior=0.01, beta=10.0)
        assert priors.shape == (50,)
        assert np.all(priors > 0) and np.all(priors <= 0.01)

    def test_adaptive_priors_entropy(self, loop_data_cosine):
        desc, poses = loop_data_cosine
        dist = SimilarityDistribution(metric='cosine').fit(
            desc, poses, pos_dist=5.0, neg_dist=10.0,
            min_temporal_gap=5, n_samples=50000,
        )
        entropies = np.random.rand(50) * 5
        priors = dist.compute_entropy_adaptive_priors(entropies, base_prior=0.01, beta=10.0)
        assert priors.shape == (50,)
        assert np.all(priors > 0) and np.all(priors <= 0.01)

    def test_save_load(self, loop_data_cosine):
        desc, poses = loop_data_cosine
        dist = SimilarityDistribution(metric='cosine').fit(
            desc, poses, pos_dist=5.0, neg_dist=10.0,
            min_temporal_gap=5, n_samples=50000,
        )
        if not dist.fitted:
            pytest.skip("Distribution not fitted")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'dist.npz')
            dist.save(path)
            loaded = SimilarityDistribution(metric='cosine').load(path)
            assert loaded.fitted is True
            assert loaded.mu_same == pytest.approx(dist.mu_same)
            assert loaded.metric == 'cosine'
