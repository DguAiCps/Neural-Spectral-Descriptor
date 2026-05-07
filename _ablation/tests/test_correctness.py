"""Reviewer-facing correctness tests for paper-code alignment.

These tests pin the closed-form hard-octave encoder claims separately from the
learnable spectral-policy path. They use synthetic inputs only.
"""

import importlib
import math

import numpy as np
import pytest
import torch

from encoding.spectral_encoder import SpectralEncoder


def _hard_octave_encoder(**overrides):
    kwargs = dict(
        n_elevation=16,
        n_azimuth=360,
        n_bins=16,
        alpha=2.0,
        learnable_alpha=True,
        target_elevation_bins=16,
        elevation_range=(-24.8, 2.0),
        bin_statistics=["mean", "std"],
        inter_bin_statistics=["diff"],
        projection_type="range_image",
        binning_strategy="octave",
        zero_center=False,
        log_magnitude=False,
        normalize_channels=False,
    )
    kwargs.update(overrides)
    return SpectralEncoder(**kwargs)


def _synthetic_range_image(seed=0):
    generator = torch.Generator().manual_seed(seed)
    image = torch.rand((16, 360), generator=generator)
    # Add low-frequency structure so bins are non-degenerate.
    az = torch.linspace(0, 2 * math.pi, 360)
    image = image + 10.0 + 2.0 * torch.cos(az).unsqueeze(0)
    return image.float()


def _cosine(a, b):
    return torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b) + 1e-8)


def _fractional_circular_shift(image, shift_columns):
    n = image.shape[1]
    freqs = torch.fft.rfftfreq(n, d=1.0)
    phase = torch.exp(-2j * math.pi * freqs * shift_columns)
    spectrum = torch.fft.rfft(image, dim=1)
    return torch.fft.irfft(spectrum * phase.unsqueeze(0), n=n, dim=1).real


def test_prop1_integer_azimuth_shift_invariance():
    encoder = _hard_octave_encoder()
    image = _synthetic_range_image()
    descriptor = encoder.encode_projected_image(image)

    for shift in (1, 7, 31, 90, 179):
        shifted = torch.roll(image, shifts=shift, dims=1)
        shifted_descriptor = encoder.encode_projected_image(shifted)
        assert _cosine(descriptor, shifted_descriptor) > 1.0 - 1e-6


def test_prop1_fractional_azimuth_shift_stability():
    encoder = _hard_octave_encoder()
    image = _synthetic_range_image(seed=1)
    descriptor = encoder.encode_projected_image(image)

    shifted = _fractional_circular_shift(image, shift_columns=0.37).float()
    shifted_descriptor = encoder.encode_projected_image(shifted)
    relative_error = torch.linalg.norm(descriptor - shifted_descriptor) / torch.linalg.norm(descriptor)
    assert relative_error < 5e-5


def test_hard_octave_encoder_has_no_trainable_parameters():
    encoder = _hard_octave_encoder(learnable_alpha=True)
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    assert n_trainable == 0
    assert encoder.alpha.requires_grad is False


def test_hard_octave_edges_and_544d_dimension_are_deterministic():
    encoder = _hard_octave_encoder()
    expected_edges = torch.tensor([0, 1, 2, 4, 8, 16, 32, 64, 128, 181], dtype=torch.float32)

    edges_a = encoder._compute_bin_edges(encoder.alpha).detach().cpu()
    edges_b = encoder._compute_bin_edges(encoder.alpha).detach().cpu()

    assert encoder.n_bins == 9
    assert encoder.output_dim == 544
    assert torch.equal(edges_a, expected_edges)
    assert torch.equal(edges_b, expected_edges)


def test_aliasrate_definition_matches_far_pair_collision_probability(monkeypatch):
    module = importlib.import_module("scripts.compute_aliasrate")

    descriptors = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 4, axis=0)
    poses[:, 0, 3] = np.array([0.0, 30.0, 60.0, 1.0])
    poses[:, 1, 3] = 0.0
    poses[2, 0, 0] = -1.0
    poses[2, 1, 1] = -1.0

    # Four sampled pairs: three far pairs, one near pair. One of the far pairs
    # collides in descriptor space, so AliasRate = 1/3.
    monkeypatch.setattr(
        module,
        "sample_pairs",
        lambda n, n_pairs, rng: (
            np.array([0, 0, 1, 0], dtype=np.int64),
            np.array([1, 2, 2, 3], dtype=np.int64),
        ),
    )

    result = module.aliasrate_with_yaw(
        descriptors,
        poses,
        eps=0.1,
        gamma=25.0,
        n_pairs=4,
        rng=np.random.default_rng(0),
    )

    assert result["n_far"] == 3
    assert result["n_collide"] == 1
    assert result["all"] == pytest.approx(1.0 / 3.0)


def test_bayesian_posterior_matches_paper_eq7_with_fisher_z():
    """Paper Eq. 7: posterior under Fisher z-transformed cosine similarity.

    P(same | s) = N(z; mu_+, sigma_+) * pi
                  / [N(z; mu_+, sigma_+) * pi + N(z; mu_-, sigma_-) * (1 - pi)],
    where z = arctanh(s).
    """
    from utils.similarity_stats import SimilarityDistribution
    from scipy.stats import norm

    dist = SimilarityDistribution(metric='cosine')
    dist.mu_same, dist.sigma_same = 1.5, 0.3   # high z = high cosine similarity
    dist.mu_diff, dist.sigma_diff = 0.2, 0.4
    dist.fitted = True

    s = 0.92
    pi = 0.05
    z = np.arctanh(np.clip(s, -0.9999, 0.9999))
    p_same = norm.pdf(z, dist.mu_same, dist.sigma_same)
    p_diff = norm.pdf(z, dist.mu_diff, dist.sigma_diff)
    expected = (p_same * pi) / (p_same * pi + p_diff * (1 - pi))

    actual = dist.posterior(observation=s, prior=pi)
    assert abs(actual - expected) < 1e-12


def test_density_adaptive_prior_matches_paper_eq9_logistic_form():
    """Paper Eq. 9 (corrected to match code): pi_i = pi_0 * sigmoid(-beta * (rho_i - rho_ref)).

    Verifies:
      (a) monotone decreasing in rho_i,
      (b) bounded in (0, pi_0),
      (c) pi_i = pi_0 / 2 at rho_i = median(rho).
    """
    from utils.similarity_stats import SimilarityDistribution

    dist = SimilarityDistribution(metric='cosine')
    rng = np.random.default_rng(7)
    rho = np.sort(rng.uniform(0.0, 1.0, size=100))  # ascending densities

    pi_0 = 0.01
    beta = 10.0
    priors = dist.compute_adaptive_priors(rho, base_prior=pi_0, beta=beta)

    # (a) monotone non-increasing
    assert np.all(np.diff(priors) <= 1e-12), \
        "Adaptive prior must be monotone non-increasing in rho"

    # (b) bounded in (0, pi_0)
    assert priors.min() > 0.0
    assert priors.max() < pi_0

    # (c) pi(rho = median) ~= pi_0 / 2
    median_rho = float(np.median(rho))
    pi_at_median = float(dist.compute_adaptive_priors(
        np.array([median_rho]), base_prior=pi_0, beta=beta)[0])
    assert abs(pi_at_median - pi_0 / 2.0) < 1e-6


def test_full_gnn_pipeline_output_dimension_672_when_pyg_available():
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data
    from gnn.model import create_spectral_gnn

    model = create_spectral_gnn(
        input_dim=544,
        hidden_dim=128,
        context_dim=128,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        edge_encoder_config=None,
        use_local_updates=False,
    )
    graph = Data(
        x=torch.randn(8, 544),
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]],
            dtype=torch.long,
        ),
    )

    with torch.no_grad():
        out = model(graph)

    assert out.shape == (8, 672)
