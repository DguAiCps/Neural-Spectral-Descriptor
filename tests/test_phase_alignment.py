import math

import torch

from encoding.phase_alignment import (
    feature_dim,
    phase_alignment_edge_features,
)


def _make_shifted_phase(n_nodes=3, n_rows=4, n_freqs=3, n_sectors=16):
    torch.manual_seed(7)
    base = torch.randn(n_rows, n_freqs, dtype=torch.complex64)
    freqs = torch.arange(1, n_freqs + 1).float()
    shifts = torch.tensor([0, 3, 7], dtype=torch.float32)[:n_nodes]
    nodes = []
    for shift in shifts:
        phase = torch.exp(-2j * math.pi * freqs * shift / float(n_sectors))
        z = base * phase.unsqueeze(0)
        nodes.append(z)
    z = torch.stack(nodes, dim=0)
    return torch.cat([z.real.reshape(n_nodes, -1), z.imag.reshape(n_nodes, -1)], dim=-1)


def test_phase_alignment_default_is_leakage_safe_dim():
    x_phase = _make_shifted_phase()
    edge_index = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
    edge_type = torch.tensor([1, 1], dtype=torch.long)
    feats = phase_alignment_edge_features(
        x_phase,
        edge_index,
        edge_type,
        n_rows=4,
        n_freqs=3,
        n_sectors=16,
        include_score=False,
    )
    assert feats.shape == (2, 4)
    assert feature_dim(include_score=False) == 4


def test_phase_alignment_score_is_optional():
    x_phase = _make_shifted_phase()
    edge_index = torch.tensor([[1], [0]], dtype=torch.long)
    edge_type = torch.tensor([1], dtype=torch.long)
    feats = phase_alignment_edge_features(
        x_phase,
        edge_index,
        edge_type,
        n_rows=4,
        n_freqs=3,
        n_sectors=16,
        include_score=True,
    )
    assert feats.shape == (1, 5)
    assert feature_dim(include_score=True) == 5
    assert feats[0, 0] > 0.99


def test_phase_alignment_temporal_edges_are_zeroed_by_default():
    x_phase = _make_shifted_phase()
    edge_index = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
    edge_type = torch.tensor([0, 1], dtype=torch.long)
    feats = phase_alignment_edge_features(
        x_phase,
        edge_index,
        edge_type,
        n_rows=4,
        n_freqs=3,
        n_sectors=16,
        include_score=False,
        similarity_only=True,
    )
    assert torch.allclose(feats[0], torch.zeros_like(feats[0]))
    assert feats[1].abs().sum() > 0
