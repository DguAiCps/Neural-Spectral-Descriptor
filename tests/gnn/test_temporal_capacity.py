"""용량을 맞춘 투영이 이웃 정보를 사용하지 않는지 확인한다."""
import sys
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_temporal_capacity import create_capacity_model


def test_capacity_active_parameters_and_neighbor_independence():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    cfg = yaml.safe_load((REPO / "results/pdv2_temporal3_20261017/training.yaml").read_text())
    model = create_capacity_model(input_dim=288, hidden_dim=128, context_dim=128, n_layers=2,
        n_heads=4, dropout=.1, norm_type="batch_norm", use_residual_gate=True, gate_hidden_dim=64,
        gate_initial_alpha=.0625, edge_encoder_config=cfg["gnn"]["edge_encoding"]).eval()
    assert sum(p.numel() for p in model.parameters()) == 167705
    x = torch.randn(8, 288)
    edges = torch.tensor([[1, 2, 3], [0, 0, 0]])
    a = model(Data(x=x, edge_index=edges))
    changed = x.clone()
    changed[1:] += 10
    b = model(Data(x=changed, edge_index=edges))
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
    assert a.shape == (8, 416) and torch.isfinite(a).all()
    a[:, 288:].square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for layer in model.gnn.convs for p in layer.parameters())
