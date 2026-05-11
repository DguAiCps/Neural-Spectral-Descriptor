"""Yaw-invariance and module-shape tests for phase-coherence and dual-stream."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from encoding.phase_coherence import (
    ClosedFormPhaseEdgeBias,
    phase_correlation_peak,
    reshape_complex,
)
from encoding.phase_features import (
    phase_features_from_layouts,
    prepare_raw_complex_phase_config,
)
from gnn.phase_diff_conv import (
    bispectrum_features,
    feature_dim,
    phase_invariant_features,
    power_features,
)


# -- helpers -----------------------------------------------------------------


def _coeffs(signal: np.ndarray, n_freqs: int) -> np.ndarray:
    F = np.fft.rfft(signal, axis=-1, norm="ortho")
    return F[..., 1 : n_freqs + 1]


def _pack(F: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [F.real.reshape(-1), F.imag.reshape(-1)]
    ).astype(np.float32)


def _make_x_phase(N: int, n_rows: int, n_freqs: int, A: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    sigs = rng.standard_normal((N, n_rows, A)).astype(np.float32)
    F = _coeffs(sigs, n_freqs)
    return torch.tensor(np.stack([_pack(F[i]) for i in range(N)]))


def _rotate_per_node(
    x_phase: torch.Tensor,
    n_rows: int,
    n_freqs: int,
    n_per_node,
    A: int,
) -> torch.Tensor:
    half = n_rows * n_freqs
    real = x_phase[..., :half].reshape(*x_phase.shape[:-1], n_rows, n_freqs)
    imag = x_phase[..., half : 2 * half].reshape(*x_phase.shape[:-1], n_rows, n_freqs)
    z = torch.complex(real, imag)
    k = torch.arange(1, n_freqs + 1, dtype=torch.float32)
    n_per_node = torch.as_tensor(n_per_node, dtype=torch.float32).view(-1, 1, 1)
    phase = torch.exp(-2j * math.pi * k.view(1, 1, -1) * n_per_node / A).to(z.dtype)
    z = z * phase
    return torch.cat(
        [
            z.real.reshape(*x_phase.shape[:-1], -1),
            z.imag.reshape(*x_phase.shape[:-1], -1),
        ],
        dim=-1,
    )


# -- ClosedFormPhaseEdgeBias --------------------------------------------------


def test_phase_correlation_peak_is_yaw_invariant():
    """PoC peak ≈ 1 for any integer yaw shift; ≈ 1 for fractional shift too."""
    n_rows, n_freqs, A = 16, 8, 64
    rng = np.random.default_rng(7)
    x = rng.standard_normal((n_rows, A)).astype(np.float32)
    F_a = _coeffs(x, n_freqs)
    xp_a = torch.tensor(_pack(F_a))[None]

    for n0 in [0, 1, 17, 31, 63]:
        F_b = _coeffs(np.roll(x, n0, axis=-1), n_freqs)
        xp_b = torch.tensor(_pack(F_b))[None]
        z_a = reshape_complex(xp_a, n_rows, n_freqs)
        z_b = reshape_complex(xp_b, n_rows, n_freqs)
        peak = phase_correlation_peak(z_a, z_b, pad_factor=8, mode="poc").item()
        assert peak > 0.999, f"PoC peak under shift {n0} = {peak} (expected ~1)"

    k = np.arange(1, n_freqs + 1, dtype=np.float32)
    F_b = F_a * np.exp(-2j * np.pi * k[None, :] * 17.5 / A)
    xp_b = torch.tensor(_pack(F_b))[None]
    peak = phase_correlation_peak(
        reshape_complex(xp_a, n_rows, n_freqs),
        reshape_complex(xp_b, n_rows, n_freqs),
        pad_factor=8,
        mode="poc",
    ).item()
    assert peak > 0.99, f"PoC peak under fractional shift 17.5 = {peak}"


def test_closed_form_bias_invariant_under_per_node_yaw():
    """Bias matrix must not change when each node is rotated by an arbitrary yaw."""
    n_rows, n_freqs, A = 16, 8, 64
    N = 6
    x_phase = _make_x_phase(N, n_rows, n_freqs, A, seed=11)
    mod = ClosedFormPhaseEdgeBias(
        n_rows=n_rows,
        n_freqs=n_freqs,
        scale=2.0,
        mode="poc",
        pad_factor=4,
        similarity_only=True,
        center=True,
    )
    pairs = torch.tensor(
        [[i for i in range(N) for _ in range(N)],
         [j for _ in range(N) for j in range(N)]],
        dtype=torch.long,
    )
    edge_type = torch.ones(N * N, dtype=torch.long)
    bias_a = mod(x_phase, pairs, edge_type=edge_type)

    rng = np.random.default_rng(23)
    n_per = rng.integers(0, A, size=N).tolist()
    x_phase_rot = _rotate_per_node(x_phase, n_rows, n_freqs, n_per, A)
    bias_b = mod(x_phase_rot, pairs, edge_type=edge_type)

    delta = (bias_a - bias_b).abs().max().item()
    assert delta < 5e-2, f"max |Δbias| under per-node yaw = {delta}"


def test_closed_form_bias_zeros_temporal_when_similarity_only():
    n_rows, n_freqs = 4, 4
    x_phase = torch.randn(3, 2 * n_rows * n_freqs)
    mod = ClosedFormPhaseEdgeBias(
        n_rows=n_rows,
        n_freqs=n_freqs,
        scale=2.0,
        similarity_only=True,
    )
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_type = torch.tensor([0, 1])  # 0 = temporal, 1 = similarity
    bias = mod(x_phase, edge_index, edge_type=edge_type)
    assert torch.equal(bias[0], torch.zeros_like(bias[0]))
    assert not torch.equal(bias[1], torch.zeros_like(bias[1]))


# -- phase_diff_conv helpers -------------------------------------------------


def test_power_features_invariance():
    n_rows, n_freqs, A = 8, 4, 32
    N = 5
    x_phase = _make_x_phase(N, n_rows, n_freqs, A, seed=2)
    pw = power_features(x_phase, n_rows, n_freqs)
    rng = np.random.default_rng(2)
    n_per = rng.integers(0, A, size=N).tolist()
    pw_rot = power_features(
        _rotate_per_node(x_phase, n_rows, n_freqs, n_per, A), n_rows, n_freqs
    )
    assert (pw - pw_rot).abs().max().item() < 1e-5


def test_bispectrum_features_invariance():
    n_rows, n_freqs, A = 4, 6, 32
    N = 4
    x_phase = _make_x_phase(N, n_rows, n_freqs, A, seed=3)
    bi = bispectrum_features(x_phase, n_rows, n_freqs)
    rng = np.random.default_rng(3)
    n_per = rng.integers(0, A, size=N).tolist()
    bi_rot = bispectrum_features(
        _rotate_per_node(x_phase, n_rows, n_freqs, n_per, A), n_rows, n_freqs
    )
    assert (bi - bi_rot).abs().max().item() < 5e-4


def test_feature_dim_matches_actual():
    n_rows, n_freqs = 16, 8
    expected = feature_dim(n_rows, n_freqs, use_bispectrum=True)
    x_phase = torch.randn(2, 2 * n_rows * n_freqs)
    fi = phase_invariant_features(x_phase, n_rows, n_freqs, use_bispectrum=True)
    assert fi.shape[-1] == expected


def test_raw_complex_phase_config_preserves_complex_geometry_and_rejects_cross():
    n_rows, n_freqs, A = 4, 3, 16
    consumer = {"phase_coherence": {"enabled": True, "n_rows": n_rows, "n_freqs": n_freqs}}
    cfg = prepare_raw_complex_phase_config({"source": "bev_complex"}, consumer)
    assert cfg["apply_log_compression"] is False
    assert cfg["bev_rows"] == n_rows
    assert cfg["bev_freqs"] == n_freqs

    rng = np.random.default_rng(31)
    bev_layout = rng.standard_normal((n_rows, A)).astype(np.float32)
    expected = _pack(_coeffs(bev_layout, n_freqs))
    actual = phase_features_from_layouts(None, bev_layout, cfg)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    with pytest.raises(ValueError, match="requires raw complex phase features"):
        prepare_raw_complex_phase_config({"source": "bev_cross"}, consumer)


# -- DualStreamSpectralGNN end-to-end ----------------------------------------


def test_dual_stream_output_dim_and_invariance():
    Data = pytest.importorskip("torch_geometric.data").Data
    from gnn.model import create_spectral_gnn

    n_rows, n_freqs, A = 8, 4, 32
    input_dim = 16
    context_dim = 8
    N, E = 6, 12

    rng = np.random.default_rng(1)
    x_mag = torch.tensor(rng.standard_normal((N, input_dim)).astype(np.float32))
    x_phase = _make_x_phase(N, n_rows, n_freqs, A, seed=1)

    edge_index = torch.tensor(rng.integers(0, N, size=(2, E)), dtype=torch.long)
    edge_attr = torch.tensor(rng.standard_normal((E, 5)).astype(np.float32))
    edge_type = torch.ones(E, dtype=torch.long)
    data = Data(
        x=x_mag,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        x_phase=x_phase,
    )

    edge_encoder_config = {
        "d_edge": 16,
        "n_edge_types": 2,
        "d_type_embed": 8,
        "d_rot_encode": 8,
        "dropout": 0.0,
    }

    model = create_spectral_gnn(
        input_dim=input_dim,
        hidden_dim=16,
        context_dim=context_dim,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        use_local_updates=False,
        edge_encoder_config=edge_encoder_config,
        norm_type="layer_norm",
        use_residual_gate=True,
        gate_initial_alpha=0.1,
        gradient_checkpointing=False,
        dual_stream_config={
            "enabled": True,
            "n_rows": n_rows,
            "n_freqs": n_freqs,
            "use_bispectrum": True,
            "hidden_dim": 16,
            "context_dim": context_dim,
            "n_layers": 1,
            "n_heads": 2,
            "fuse_initial_alpha": 0.5,
            "fuse_per_node": True,
            "norm_type": "layer_norm",
        },
    )
    model.eval()

    with torch.no_grad():
        out = model(data)
    assert out.shape == (N, input_dim + context_dim), out.shape

    # All-nodes-rotated-by-same-yaw → ctx_phase identical → fused output identical.
    n0 = 7
    x_phase_rot = _rotate_per_node(x_phase, n_rows, n_freqs, [n0] * N, A)
    data_rot = Data(
        x=x_mag,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        x_phase=x_phase_rot,
    )
    with torch.no_grad():
        out_rot = model(data_rot)
    delta = (out - out_rot).abs().max().item()
    assert delta < 1e-4, f"Dual-stream output max|Δ| under shared yaw = {delta}"
