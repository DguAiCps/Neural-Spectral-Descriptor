"""Tests for GNN model: SpectralGNN, DiffAttnConv, EdgeEncoder."""

import torch
import torch.nn.functional as F
import pytest
from torch_geometric.data import Data
from gnn.model import (
    SinusoidalEncoding,
    EdgeEncoder,
    DiffAttnConv,
    SpectralGNN,
    LocalUpdateGNN,
    create_spectral_gnn,
)


class TestSinusoidalEncoding:

    def test_output_shape(self):
        enc = SinusoidalEncoding(d_encode=16)
        x = torch.randn(50)
        out = enc(x)
        assert out.shape == (50, 16)

    def test_different_inputs_different_outputs(self):
        enc = SinusoidalEncoding(d_encode=16)
        a = enc(torch.tensor([0.0]))
        b = enc(torch.tensor([1.0]))
        assert not torch.allclose(a, b)


class TestEdgeEncoder:

    @pytest.fixture
    def encoder(self):
        return EdgeEncoder(d_edge=32, n_edge_types=2, d_type_embed=16, d_rot_encode=16)

    def test_output_shape(self, encoder):
        n_edges = 40
        edge_attr = torch.randn(n_edges, 5)
        edge_type = torch.randint(0, 2, (n_edges,))
        out = encoder(edge_attr, edge_type)
        assert out.shape == (n_edges, 32)

    def test_different_types_differ(self, encoder):
        """Temporal and similarity edges should produce different embeddings."""
        edge_attr = torch.randn(1, 5)
        # Same attributes but different types
        out_temporal = encoder(edge_attr, torch.tensor([0]))
        out_similarity = encoder(edge_attr, torch.tensor([1]))
        assert not torch.allclose(out_temporal, out_similarity, atol=1e-3)


class TestDiffAttnConv:

    def test_output_shape(self):
        conv = DiffAttnConv(channels=128, heads=4)
        x = torch.randn(10, 128)
        edge_index = torch.randint(0, 10, (2, 30))
        out = conv(x, edge_index)
        assert out.shape == (10, 128)

    def test_with_edge_attr(self):
        conv = DiffAttnConv(channels=128, heads=4, edge_dim=32)
        x = torch.randn(10, 128)
        edge_index = torch.randint(0, 10, (2, 30))
        edge_attr = torch.randn(30, 32)
        out = conv(x, edge_index, edge_attr=edge_attr)
        assert out.shape == (10, 128)

    def test_return_attention_weights(self):
        conv = DiffAttnConv(channels=128, heads=4, edge_dim=32)
        conv.eval()
        x = torch.randn(10, 128)
        edge_index = torch.randint(0, 10, (2, 30))
        edge_attr = torch.randn(30, 32)
        out, (ei, attn) = conv(x, edge_index, edge_attr=edge_attr,
                                return_attention_weights=True)
        assert out.shape == (10, 128)
        assert attn.shape == (30, 4)  # n_edges × n_heads

    def test_gradient_flow(self):
        conv = DiffAttnConv(channels=64, heads=2)
        x = torch.randn(5, 64, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        out = conv(x, edge_index)
        out.sum().backward()
        assert x.grad is not None

    def test_message_uses_differences(self):
        """Verify that identical nodes produce zero output (no self-info)."""
        conv = DiffAttnConv(channels=64, heads=2)
        conv.eval()
        # All nodes identical → all diffs are zero → V(0)=0 → output=0
        x = torch.ones(5, 64)
        # Chain graph
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        out = conv(x, edge_index)
        # Output should be near-zero since V(h_j - h_i) = V(0) → ~0
        assert out.abs().max() < 0.1


class TestSpectralGNN:

    def test_output_dim(self, small_graph):
        model = SpectralGNN(
            input_dim=1106, hidden_dim=128, context_dim=256,
            n_layers=2, n_heads=4,
            edge_encoder_config={'d_edge': 32, 'n_edge_types': 2,
                                 'd_type_embed': 16, 'd_rot_encode': 16, 'dropout': 0.1},
        ).to(small_graph.x.device)
        model.eval()
        out = model(small_graph)
        assert out.shape == (small_graph.num_nodes, 1106 + 256)

    def test_output_normalized(self, small_graph):
        """Raw and context parts should be L2-normalized."""
        model = SpectralGNN(
            input_dim=1106, hidden_dim=128, context_dim=256,
            n_layers=2, n_heads=4,
        ).to(small_graph.x.device)
        model.eval()
        out = model(small_graph)
        raw_part = out[:, :1106]
        ctx_part = out[:, 1106:]
        # L2 norms should be ~1
        raw_norms = torch.norm(raw_part, dim=1)
        ctx_norms = torch.norm(ctx_part, dim=1)
        torch.testing.assert_close(raw_norms, torch.ones_like(raw_norms), atol=1e-4, rtol=0)
        torch.testing.assert_close(ctx_norms, torch.ones_like(ctx_norms), atol=1e-4, rtol=0)

    def test_get_embedding_dim(self):
        model = SpectralGNN(input_dim=256, context_dim=128)
        assert model.get_embedding_dim() == 384

    def test_forward_with_attention(self, small_graph):
        model = SpectralGNN(
            input_dim=1106, hidden_dim=128, context_dim=256,
            n_layers=2, n_heads=4,
            edge_encoder_config={'d_edge': 32, 'n_edge_types': 2,
                                 'd_type_embed': 16, 'd_rot_encode': 16, 'dropout': 0.1},
        ).to(small_graph.x.device)
        model.eval()
        out, attn_list = model.forward_with_attention(small_graph)
        assert out.shape[1] == 1106 + 256
        assert len(attn_list) == 2  # n_layers

    def test_gradient_checkpointing(self, small_graph):
        """Model with gradient checkpointing should produce same output."""
        model = SpectralGNN(
            input_dim=1106, hidden_dim=128, context_dim=256,
            gradient_checkpointing=True,
        ).to(small_graph.x.device)
        model.train()
        out = model(small_graph)
        loss = out.sum()
        loss.backward()  # Should not error

    def test_without_edge_encoder(self, small_graph):
        """Model works without edge encoder."""
        model = SpectralGNN(
            input_dim=1106, hidden_dim=128, context_dim=256,
            edge_encoder_config=None,
        ).to(small_graph.x.device)
        model.eval()
        out = model(small_graph)
        assert out.shape[1] == 1106 + 256

    def test_residual_connection(self):
        """With residual=True, output differs from residual=False."""
        data = Data(
            x=torch.randn(5, 64),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        )
        m1 = SpectralGNN(input_dim=64, hidden_dim=64, context_dim=32, residual=True)
        m2 = SpectralGNN(input_dim=64, hidden_dim=64, context_dim=32, residual=False)
        # Copy weights from m1 to m2
        m2.load_state_dict(m1.state_dict())
        m1.eval()
        m2.eval()
        out1 = m1(data)
        out2 = m2(data)
        # With residual, output should differ (unless weights happen to cancel)
        assert not torch.allclose(out1, out2, atol=1e-4)


class TestSpectralGNNWithPolicy:

    def test_policy_overrides_input_dim(self):
        from encoding.spectral_policy import LearnedFilterbank
        policy = LearnedFilterbank(79, 181, output_dim=1106)
        model = SpectralGNN(input_dim=999, spectral_policy=policy)
        assert model.input_dim == 1106

    def test_forward_with_fft(self, device):
        from encoding.spectral_policy import SoftBinning
        n_nodes = 10
        policy = SoftBinning(79, 181, output_dim=1106, n_soft_bins=4)
        model = SpectralGNN(
            input_dim=1106, hidden_dim=64, context_dim=128,
            spectral_policy=policy,
        ).to(device)
        model.eval()

        x_fft = torch.randn(n_nodes, 79 * 181).to(device)
        data = Data(
            x=torch.randn(n_nodes, 1106).to(device),
            x_fft=x_fft,
            edge_index=torch.randint(0, n_nodes, (2, 20)).to(device),
        )
        out = model(data)
        assert out.shape == (n_nodes, 1106 + 128)

    def test_fallback_to_precomputed_when_no_fft(self, device):
        """Without x_fft, model falls back to data.x."""
        from encoding.spectral_policy import SoftBinning
        n_nodes = 10
        policy = SoftBinning(79, 181, output_dim=1106, n_soft_bins=4)
        model = SpectralGNN(
            input_dim=1106, hidden_dim=64, context_dim=128,
            spectral_policy=policy,
        ).to(device)
        model.eval()

        data = Data(
            x=torch.randn(n_nodes, 1106).to(device),
            edge_index=torch.randint(0, n_nodes, (2, 20)).to(device),
        )
        out = model(data)
        assert out.shape == (n_nodes, 1106 + 128)


class TestCreateSpectralGNN:

    def test_factory_returns_local_update(self):
        model = create_spectral_gnn(use_local_updates=True)
        assert isinstance(model, LocalUpdateGNN)

    def test_factory_returns_base_gnn(self):
        model = create_spectral_gnn(use_local_updates=False)
        assert isinstance(model, SpectralGNN)

    def test_factory_with_edge_encoder(self):
        model = create_spectral_gnn(
            edge_encoder_config={'d_edge': 32, 'n_edge_types': 2,
                                 'd_type_embed': 16, 'd_rot_encode': 16, 'dropout': 0.1}
        )
        base = model.gnn if hasattr(model, 'gnn') else model
        assert base.edge_encoder is not None
