"""차이 값 변환의 기준 재현·영 변화·유한 미분을 확인한다."""

import copy

import torch

from gnn.model import DiffAttnConv
from gnn.geometry_interaction import GeometryInteractionConv
from gnn.moment_difference import MomentDifferenceConv


def test_signed_matches_reference_and_gradients():
    torch.manual_seed(15)
    ref = (
        GeometryInteractionConv(
            DiffAttnConv(16, heads=4, edge_dim=5, dropout=0), "interaction"
        )
        .double()
        .eval()
    )
    new = MomentDifferenceConv(copy.deepcopy(ref), "signed").double().eval()
    x = torch.randn(10, 16, dtype=torch.float64, requires_grad=True)
    target = torch.arange(10).repeat_interleave(2)
    source = (target + torch.tensor([-1, 1]).repeat(10)) % 10
    edges = torch.stack((source, target))
    attr = torch.randn(20, 5, dtype=torch.float64)
    a = ref(x, edges, attr)
    b = new(x, edges, attr)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    ga = torch.autograd.grad(a.square().sum(), x, retain_graph=True)[0]
    gb = torch.autograd.grad(b.square().sum(), x)[0]
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)


def test_square_nonnegative_and_zero_difference():
    torch.manual_seed(19)
    ref = GeometryInteractionConv(
        DiffAttnConv(16, heads=4, edge_dim=5, dropout=0), "interaction"
    )
    edge = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    attr = torch.randn(4, 5)
    x = torch.randn(4, 16, requires_grad=True)
    for mode in ("square", "odd_even"):
        net = MomentDifferenceConv(copy.deepcopy(ref), mode).eval()
        v = net(x, edge, attr)
        assert torch.isfinite(v).all()
        if mode == "square":
            assert (v >= 0).all()
        zero = net(torch.ones_like(x), edge, attr)
        torch.testing.assert_close(zero, torch.zeros_like(zero))
        v.square().sum().backward()
        assert torch.isfinite(x.grad).all() and x.grad.norm() > 0
