"""학습 투영의 이웃 독립성과 학습 가능한 416차원 출력을 확인한다."""
import torch
from torch_geometric.data import Data

from gnn.graph_free import create_graph_free_model


def make_model():
    torch.manual_seed(11)
    return create_graph_free_model(input_dim=288, hidden_dim=128, context_dim=128,
                                   n_layers=2, dropout=0.1, norm_type="batch_norm",
                                   use_residual_gate=True, gate_initial_alpha=0.0625)


def test_graph_free_neighbor_and_edge_independence():
    model = make_model().eval()
    x = torch.rand(8, 288)
    full = Data(x=x, edge_index=torch.tensor([[0, 2, 4], [1, 1, 1]]))
    with torch.no_grad():
        expected = model(full)[1]
        changed = full.clone()
        changed.x[[0, 2, 3, 4, 5, 6, 7]] *= 20
        changed.edge_index = torch.tensor([[3, 7], [1, 1]])
        torch.testing.assert_close(model(changed)[1], expected, rtol=0, atol=0)
        isolated = Data(x=x[1:2], edge_index=torch.empty((2, 0), dtype=torch.long))
        torch.testing.assert_close(model(isolated)[0], expected, rtol=1e-5, atol=1e-6)
        assert expected.shape == (416,) and torch.isfinite(expected).all()
        torch.testing.assert_close(expected[:288], torch.nn.functional.normalize(x[1], dim=0))
        torch.testing.assert_close(expected[288:].norm(), torch.tensor(0.0625))


def test_graph_free_projection_receives_gradients():
    model = make_model().train()
    graph = Data(x=torch.rand(12, 288), edge_index=torch.empty((2, 0), dtype=torch.long))
    before = model.gnn.convs[0].linear.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(graph)
    (output[:, 288:] * torch.randn_like(output[:, 288:])).sum().backward()
    grad = model.gnn.convs[0].linear.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.gnn.convs[0].linear.weight)
