"""복소 전달에 연결하는 차이 주의 층의 원래 수식 재현 검사."""

import torch

from gnn.geometry_interaction import GeometryInteractionConv
from gnn.learned_spectral_difference import ControlledRelationConv
from gnn.model import DiffAttnConv


def test_controlled_diff_matches_original_forward_and_gradient():
    """실제 복수 이웃에서 순전파와 입력 기울기가 원 수식과 같다."""
    torch.manual_seed(19)
    raw = DiffAttnConv(128, heads=4, edge_dim=32, dropout=0.0)
    reference = GeometryInteractionConv(raw, "interaction").double().eval()
    controlled = ControlledRelationConv(reference, "diff").eval()
    x = torch.randn(13, 128, dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([[0, 2, 4, 3, 9, 6, 4, 7], [1, 1, 1, 5, 5, 10, 10, 10]])
    attr = torch.randn(8, 32, dtype=torch.float64)
    y = reference(x, edges, attr)
    z = controlled(x, edges, attr)
    torch.testing.assert_close(y, z, rtol=1e-12, atol=1e-12)
    a = torch.autograd.grad(y.square().sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad(z.square().sum(), x)[0]
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
