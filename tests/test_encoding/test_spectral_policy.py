"""Tests for all 5 spectral policy types."""

import torch
import pytest
from encoding.spectral_policy import (
    LearnedFilterbank,
    ConvSpectralPool,
    CrossAttentionPool,
    SoftBinning,
    GatedFrequencySelection,
    create_spectral_policy,
    POLICY_REGISTRY,
)

N_RINGS = 79
N_FREQS = 181
BATCH = 4


@pytest.fixture
def fft_input():
    return torch.randn(BATCH, N_RINGS, N_FREQS)


# ---------------------------------------------------------------------------
# Option A: LearnedFilterbank
# ---------------------------------------------------------------------------

class TestLearnedFilterbank:

    def test_output_shape_shared(self, fft_input):
        policy = LearnedFilterbank(N_RINGS, N_FREQS, output_dim=1106, shared_across_rings=True)
        out = policy(fft_input)
        assert out.shape == (BATCH, 1106)

    def test_output_shape_per_ring(self, fft_input):
        policy = LearnedFilterbank(N_RINGS, N_FREQS, output_dim=1106, shared_across_rings=False)
        out = policy(fft_input)
        assert out.shape == (BATCH, 1106)

    def test_invalid_output_dim(self):
        with pytest.raises(ValueError, match="divisible"):
            LearnedFilterbank(N_RINGS, N_FREQS, output_dim=1000)

    def test_gradient_flow(self, fft_input):
        policy = LearnedFilterbank(N_RINGS, N_FREQS, output_dim=1106)
        fft_input.requires_grad_(True)
        out = policy(fft_input)
        loss = out.sum()
        loss.backward()
        assert fft_input.grad is not None


# ---------------------------------------------------------------------------
# Option B: ConvSpectralPool
# ---------------------------------------------------------------------------

class TestConvSpectralPool:

    def test_output_shape(self, fft_input):
        policy = ConvSpectralPool(N_RINGS, N_FREQS, output_dim=1106, kernel_size=7)
        out = policy(fft_input)
        assert out.shape == (BATCH, 1106)

    def test_different_kernel_sizes(self, fft_input):
        for k in [3, 7, 15]:
            policy = ConvSpectralPool(N_RINGS, N_FREQS, output_dim=1106, kernel_size=k)
            out = policy(fft_input)
            assert out.shape == (BATCH, 1106)

    def test_gradient_flow(self, fft_input):
        policy = ConvSpectralPool(N_RINGS, N_FREQS, output_dim=1106)
        fft_input.requires_grad_(True)
        out = policy(fft_input)
        out.sum().backward()
        assert fft_input.grad is not None


# ---------------------------------------------------------------------------
# Option C: CrossAttentionPool
# ---------------------------------------------------------------------------

class TestCrossAttentionPool:

    def test_output_shape(self, fft_input):
        policy = CrossAttentionPool(
            N_RINGS, N_FREQS, output_dim=1106,
            n_queries=7, n_heads=2, head_dim=32, d_pe=16,
        )
        out = policy(fft_input)
        assert out.shape == (BATCH, 1106)

    def test_gradient_flow(self, fft_input):
        policy = CrossAttentionPool(N_RINGS, N_FREQS, output_dim=1106)
        fft_input.requires_grad_(True)
        out = policy(fft_input)
        out.sum().backward()
        assert fft_input.grad is not None


# ---------------------------------------------------------------------------
# Option D: SoftBinning
# ---------------------------------------------------------------------------

class TestSoftBinning:

    def test_output_shape_shared(self, fft_input):
        policy = SoftBinning(
            N_RINGS, N_FREQS, output_dim=1106,
            n_soft_bins=4, stats=['mean', 'std'],
            inter_stats=['diff'], shared_across_rings=True,
        )
        out = policy(fft_input)
        assert out.shape == (BATCH, 1106)

    def test_output_shape_per_ring(self, fft_input):
        policy = SoftBinning(
            N_RINGS, N_FREQS, output_dim=1106,
            n_soft_bins=4, shared_across_rings=False,
        )
        out = policy(fft_input)
        assert out.shape == (BATCH, 1106)

    def test_init_from_fixed(self):
        policy = SoftBinning(
            N_RINGS, N_FREQS, output_dim=1106,
            init_from_fixed=True, alpha=2.0,
        )
        # Centers should be between 0 and N_FREQS
        c = policy.centers.detach()
        assert c.min() >= 0
        assert c.max() <= N_FREQS

    def test_init_uniform(self):
        policy = SoftBinning(
            N_RINGS, N_FREQS, output_dim=1106,
            init_from_fixed=False,
        )
        c = policy.centers.detach()
        assert c.min() >= 0

    def test_gradient_flow(self, fft_input):
        policy = SoftBinning(N_RINGS, N_FREQS, output_dim=1106)
        fft_input.requires_grad_(True)
        out = policy(fft_input)
        out.sum().backward()
        assert fft_input.grad is not None
        assert policy.centers.grad is not None
        assert policy.log_widths.grad is not None

    def test_per_ring_dim_calculation(self):
        """4 bins, [mean,std], [diff] → 4*2 + 3*1*2 = 14 per ring."""
        policy = SoftBinning(
            N_RINGS, N_FREQS, output_dim=1106,
            n_soft_bins=4, stats=['mean', 'std'], inter_stats=['diff'],
        )
        assert policy.per_ring_dim == 14

    def test_with_projection(self, fft_input):
        """When output_dim doesn't match n_rings*per_ring_dim, a projection is used."""
        policy = SoftBinning(
            N_RINGS, N_FREQS, output_dim=512,
            n_soft_bins=4,
        )
        out = policy(fft_input)
        assert out.shape == (BATCH, 512)
        assert policy.proj is not None


# ---------------------------------------------------------------------------
# Option E: GatedFrequencySelection
# ---------------------------------------------------------------------------

class TestGatedFrequencySelection:

    def test_output_shape(self, fft_input):
        policy = GatedFrequencySelection(N_RINGS, N_FREQS, output_dim=1106)
        out = policy(fft_input)
        assert out.shape == (BATCH, 1106)

    def test_gradient_flow(self, fft_input):
        policy = GatedFrequencySelection(N_RINGS, N_FREQS, output_dim=1106)
        fft_input.requires_grad_(True)
        out = policy(fft_input)
        out.sum().backward()
        assert fft_input.grad is not None

    def test_no_projection_when_matching(self):
        """n_rings * 3 = 237 → no projection needed."""
        policy = GatedFrequencySelection(N_RINGS, N_FREQS, output_dim=N_RINGS * 3)
        assert policy.proj is None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:

    @pytest.mark.parametrize("policy_type", list(POLICY_REGISTRY.keys()))
    def test_create_all_types(self, policy_type, fft_input):
        config = {
            'type': policy_type,
            'output_dim': 1106,
            'shared_across_rings': True,
            'soft_binning': {'n_soft_bins': 4, 'stats': ['mean', 'std'], 'inter_stats': ['diff']},
            'linear': {'d_per_ring': 14},
            'conv1d': {'channels_per_group': 2, 'kernel_size': 7},
            'attention': {'n_queries': 7, 'n_heads': 2, 'head_dim': 32, 'd_pe': 16},
            'gated': {'gate_hidden': 64},
        }
        policy = create_spectral_policy(config, n_rings=N_RINGS, n_freqs=N_FREQS)
        out = policy(fft_input)
        assert out.shape == (BATCH, 1106)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_spectral_policy({'type': 'nonexistent'}, N_RINGS, N_FREQS)

    def test_default_type_is_soft_binning(self):
        config = {'output_dim': 1106}
        policy = create_spectral_policy(config, N_RINGS, N_FREQS)
        assert isinstance(policy, SoftBinning)


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------

class TestPolicyCrossCutting:

    @pytest.mark.parametrize("policy_type", list(POLICY_REGISTRY.keys()))
    def test_output_is_finite(self, policy_type, fft_input):
        config = {
            'type': policy_type, 'output_dim': 1106,
            'shared_across_rings': True,
            'soft_binning': {'n_soft_bins': 4},
            'conv1d': {'channels_per_group': 2, 'kernel_size': 7},
            'attention': {'n_queries': 7, 'n_heads': 2, 'head_dim': 32},
            'gated': {'gate_hidden': 64},
        }
        policy = create_spectral_policy(config, N_RINGS, N_FREQS)
        out = policy(fft_input)
        assert torch.all(torch.isfinite(out))

    @pytest.mark.parametrize("policy_type", list(POLICY_REGISTRY.keys()))
    def test_batch_size_1(self, policy_type):
        x = torch.randn(1, N_RINGS, N_FREQS)
        config = {
            'type': policy_type, 'output_dim': 1106,
            'shared_across_rings': True,
            'soft_binning': {'n_soft_bins': 4},
            'conv1d': {'channels_per_group': 2, 'kernel_size': 7},
            'attention': {'n_queries': 7, 'n_heads': 2, 'head_dim': 32},
            'gated': {'gate_hidden': 64},
        }
        policy = create_spectral_policy(config, N_RINGS, N_FREQS)
        out = policy(x)
        assert out.shape == (1, 1106)
