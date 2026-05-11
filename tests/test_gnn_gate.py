import torch
import pytest

Data = pytest.importorskip("torch_geometric.data").Data

from gnn.model import create_spectral_gnn


def test_context_gate_initial_alpha_controls_output_scale():
    model = create_spectral_gnn(
        input_dim=8,
        hidden_dim=16,
        context_dim=4,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        use_local_updates=False,
        edge_encoder_config=None,
        norm_type="layer_norm",
        use_residual_gate=True,
        gate_initial_alpha=0.0625,
    )
    model.eval()

    data = Data(
        x=torch.randn(6, 8),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    with torch.no_grad():
        out = model(data)

    assert out.shape == (6, 12)
    assert model._last_alpha is not None
    assert torch.allclose(
        model._last_alpha,
        torch.full_like(model._last_alpha, 0.0625),
        atol=1e-5,
    )


def test_attention_path_uses_same_context_gate():
    model = create_spectral_gnn(
        input_dim=8,
        hidden_dim=16,
        context_dim=4,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        use_local_updates=False,
        edge_encoder_config=None,
        norm_type="layer_norm",
        use_residual_gate=True,
        gate_initial_alpha=0.125,
    )
    model.eval()

    data = Data(
        x=torch.randn(6, 8),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    with torch.no_grad():
        out = model(data)
        out_attn, _ = model.forward_with_attention(data)

    assert torch.allclose(out, out_attn, atol=1e-6)
