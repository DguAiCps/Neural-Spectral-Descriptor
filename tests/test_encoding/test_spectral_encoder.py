"""Tests for SpectralEncoder (PyTorch) and SpectralEncoderNumpy."""

import numpy as np
import torch
import pytest
from encoding.spectral_encoder import SpectralEncoder


def _make_encoder(device='cpu', **overrides):
    """Helper to create a BEV-mode encoder with production defaults."""
    kwargs = dict(
        n_elevation=16, n_azimuth=360, n_bins=4, alpha=2.0,
        learnable_alpha=True, target_elevation_bins=16,
        elevation_range=(-24.8, 2.0),
        bin_statistics=['mean', 'std'],
        inter_bin_statistics=['diff'],
        projection_type='bev',
        max_range=80.0, min_range=1.0,
        z_min=-3.0, height_encoding='iris',
        n_height_layers=8, z_max=5.0,
        binning_strategy='octave',
        zero_center=False, log_magnitude=False,
        normalize_channels=False,
    )
    kwargs.update(overrides)
    return SpectralEncoder(**kwargs).to(device)


class TestSpectralEncoderOutputDim:

    def test_bev_exponential_4bins(self):
        """BEV exponential 4-bin: 79×4×2 + 79×3×2 = 632+474 = 1106."""
        enc = _make_encoder(n_bins=4, binning_strategy='exponential')
        assert enc.output_dim == 1106

    def test_bev_octave_always_9bins(self):
        """BEV octave always gives 9 freq bins: 79×9×2 + 79×8×2 = 2686."""
        enc = _make_encoder(n_bins=4, binning_strategy='octave')
        assert enc.output_dim == 2686

    def test_range_image_output_dim(self):
        """Range image exponential 16-bin: 16×16×2 + 16×15×2 = 512+480 = 992."""
        enc = _make_encoder(projection_type='range_image', n_bins=16,
                            n_elevation=64, target_elevation_bins=16,
                            binning_strategy='exponential')
        assert enc.output_dim == 992

    def test_output_dim_single_stat(self):
        """Exponential 4-bin, only mean: 79×4×1 + 79×3×1 = 316+237 = 553."""
        enc = _make_encoder(n_bins=4, bin_statistics=['mean'],
                            inter_bin_statistics=['diff'],
                            binning_strategy='exponential')
        assert enc.output_dim == 553


class TestSpectralEncoderEncode:

    def test_encode_points_shape(self, random_point_cloud):
        enc = _make_encoder()
        desc, entropy = enc.encode_points(random_point_cloud, return_entropy=True)
        assert desc.shape == (enc.output_dim,)
        assert isinstance(entropy, float)

    def test_encode_points_no_entropy(self, random_point_cloud):
        enc = _make_encoder()
        desc = enc.encode_points(random_point_cloud, return_entropy=False)
        assert desc.shape == (enc.output_dim,)

    def test_encode_points_finite(self, random_point_cloud):
        enc = _make_encoder()
        desc = enc.encode_points(random_point_cloud)
        assert torch.all(torch.isfinite(desc))

    def test_rotation_invariance(self, random_point_cloud):
        """Rotating the point cloud should produce the same descriptor."""
        enc = _make_encoder()
        desc_orig = enc.encode_points(random_point_cloud).detach().cpu().numpy()

        # Rotate 90° around Z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        rotated = (R @ random_point_cloud.T).T
        desc_rot = enc.encode_points(rotated).detach().cpu().numpy()

        # FFT magnitude is rotation-invariant up to discretization error
        cos_sim = np.dot(desc_orig, desc_rot) / (
            np.linalg.norm(desc_orig) * np.linalg.norm(desc_rot) + 1e-8
        )
        assert cos_sim > 0.85, f"Rotation invariance broken: cos_sim={cos_sim:.4f}"


class TestComputeFFTMagnitudes:

    def test_fft_magnitudes_shape(self, random_point_cloud):
        enc = _make_encoder()
        fft_mag = enc.compute_fft_magnitudes(random_point_cloud)
        # BEV: n_rows=79, n_freqs=181 (rfft of 360)
        assert fft_mag.shape == (79, 181)

    def test_fft_magnitudes_dtype(self, random_point_cloud):
        enc = _make_encoder()
        fft_mag = enc.compute_fft_magnitudes(random_point_cloud)
        assert fft_mag.dtype == np.float32

    def test_fft_magnitudes_nonnegative(self, random_point_cloud):
        enc = _make_encoder()
        fft_mag = enc.compute_fft_magnitudes(random_point_cloud)
        assert np.all(fft_mag >= 0)

    def test_fft_magnitudes_with_log(self, random_point_cloud):
        enc = _make_encoder(log_magnitude=True)
        fft_mag = enc.compute_fft_magnitudes(random_point_cloud)
        assert fft_mag.shape == (79, 181)
        assert np.all(np.isfinite(fft_mag))

    def test_fft_magnitudes_with_zero_center(self, random_point_cloud):
        enc = _make_encoder(zero_center=True)
        fft_mag = enc.compute_fft_magnitudes(random_point_cloud)
        assert fft_mag.shape == (79, 181)

    def test_range_image_fft_shape(self, random_point_cloud):
        enc = _make_encoder(projection_type='range_image')
        fft_mag = enc.compute_fft_magnitudes(random_point_cloud)
        # Pooled to target_elevation_bins=16, rfft of 360=181
        assert fft_mag.shape == (16, 181)


class TestBinning:

    def test_exponential_binning(self, random_point_cloud):
        enc = _make_encoder(binning_strategy='exponential')
        desc = enc.encode_points(random_point_cloud)
        assert desc.shape == (enc.output_dim,)

    def test_octave_binning(self, random_point_cloud):
        enc = _make_encoder(binning_strategy='octave')
        desc = enc.encode_points(random_point_cloud)
        assert desc.shape == (enc.output_dim,)

    def test_alpha_is_fixed_for_octave_even_if_requested_learnable(self):
        enc = _make_encoder(binning_strategy='octave', learnable_alpha=True)
        assert enc.alpha.requires_grad is False

    def test_alpha_is_fixed_for_exponential_when_requested_fixed(self):
        enc = _make_encoder(binning_strategy='exponential', learnable_alpha=False)
        assert enc.alpha.requires_grad is False

    def test_alpha_is_learnable_for_legacy_exponential(self):
        enc = _make_encoder(binning_strategy='exponential', learnable_alpha=True)
        assert enc.alpha.requires_grad is True


class TestSpectralEncoderNumpy:

    def test_numpy_encoder_output(self, random_point_cloud):
        from encoding.spectral_encoder import SpectralEncoderNumpy
        enc = SpectralEncoderNumpy(
            n_elevation=16, n_azimuth=360, n_bins=4, alpha=2.0,
            target_elevation_bins=16,
            bin_statistics=['mean', 'std'],
            inter_bin_statistics=['diff'],
            projection_type='bev',
            max_range=80.0, min_range=1.0,
            z_min=-3.0, height_encoding='iris',
            n_height_layers=8, z_max=5.0,
            binning_strategy='exponential',
            normalize_channels=False,
        )
        desc = enc.encode_points(random_point_cloud)
        assert isinstance(desc, np.ndarray)
        assert desc.shape == (1106,)
        assert np.all(np.isfinite(desc))

    def test_numpy_fft_magnitudes(self, random_point_cloud):
        from encoding.spectral_encoder import SpectralEncoderNumpy
        enc = SpectralEncoderNumpy(
            n_elevation=16, n_azimuth=360, n_bins=4, alpha=2.0,
            target_elevation_bins=16,
            projection_type='bev', max_range=80.0, min_range=1.0,
            z_min=-3.0, height_encoding='iris', n_height_layers=8, z_max=5.0,
            binning_strategy='exponential', normalize_channels=False,
        )
        fft_mag = enc.compute_fft_magnitudes(random_point_cloud)
        assert fft_mag.shape == (79, 181)
        assert fft_mag.dtype == np.float32
