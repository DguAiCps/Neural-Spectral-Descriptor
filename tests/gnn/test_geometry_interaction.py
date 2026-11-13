"""점수 결합의 기준 동등성·상수 불변성·실제 상호작용을 검산한다."""
import copy

import pytest
import torch
from torch_geometric.utils import softmax

from gnn.geometry_interaction import GeometryInteractionConv, combine_scores
from gnn.model import DiffAttnConv
from gnn.value_relation import ValueRelationConv


def test_additive_replays_output_and_gradients():
    ref = DiffAttnConv(8, heads=2, dropout=.1, edge_dim=3).double()
    with torch.no_grad():
        ref.bias_mlp[-1].weight.normal_(0., .1)
    old = ValueRelationConv(copy.deepcopy(ref), "tanh")
    new = GeometryInteractionConv(copy.deepcopy(ref), "additive")
    x = torch.randn(6, 8, dtype=torch.double)
    edge = torch.tensor([[0, 2, 4, 5], [1, 1, 3, 3]])
    attr = torch.randn(4, 3, dtype=torch.double)
    outputs, grads = [], []
    for net in (old, new):
        torch.manual_seed(17)
        out = net(x, edge, attr)
        out.square().sum().backward()
        outputs.append(out)
        grads.append({n: p.grad for n, p in net.named_parameters()})
    assert old.state_dict().keys() == new.state_dict().keys()
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    for key in grads[0]:
        torch.testing.assert_close(grads[0][key], grads[1][key], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["interaction", "temperature"])
def test_zero_bias_starts_at_reference_and_has_finite_gradients(mode):
    ref = DiffAttnConv(8, heads=2, dropout=0., edge_dim=3).double()
    old = ValueRelationConv(copy.deepcopy(ref), "tanh").eval()
    new = GeometryInteractionConv(copy.deepcopy(ref), mode)
    x = torch.randn(5, 8, dtype=torch.double)
    edge = torch.tensor([[0, 2, 3, 4], [1, 1, 2, 2]])
    attr = torch.randn(4, 3, dtype=torch.double)
    out = new(x, edge, attr)
    torch.testing.assert_close(out, old(x, edge, attr), atol=0, rtol=0)
    out.square().sum().backward()
    assert out.shape == x.shape and out.dtype == x.dtype
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
    assert new.bias_mlp[-1].weight.grad.norm() > 0
    torch.testing.assert_close(new(x, edge[:, :0], attr[:0]), torch.zeros_like(x))


def test_interaction_preserves_score_shift_and_rejects_uniform_geometry():
    target = torch.tensor([0, 0, 1, 1, 1])
    c = torch.randn(5, 2, dtype=torch.double)
    b = torch.randn(5, 2, dtype=torch.double)
    original = combine_scores(c, b, target, 2, "interaction")
    shifted = combine_scores(c + 7, b - 3, target, 2, "interaction")
    torch.testing.assert_close(softmax(original, target), softmax(shifted, target),
                               atol=1e-13, rtol=1e-13)
    common = torch.tensor([[.5, -.4], [.8, .2]], dtype=torch.double)[target]
    mixed = combine_scores(c, common, target, 2, "interaction")
    torch.testing.assert_close(mixed, c + common, atol=1e-15, rtol=0)
    temperature = combine_scores(c, common, target, 2, "temperature")
    assert not torch.allclose(softmax(mixed, target), softmax(temperature, target))


def test_interaction_has_feature_dependent_geometry_gradient_and_dense_reference():
    target = torch.tensor([0, 0, 0, 1, 1])
    c = torch.randn(5, 2, dtype=torch.double, requires_grad=True)
    b = torch.randn(5, 2, dtype=torch.double, requires_grad=True)
    actual = combine_scores(c, b, target, 2, "interaction")
    expected = torch.empty_like(actual)
    for node in (0, 1):
        mask = target == node
        cc, bb = c[mask], b[mask]
        expected[mask] = cc + bb + (cc - cc.mean(0)) * (bb - bb.mean(0)).tanh()
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    assert torch.autograd.gradcheck(lambda a, g: combine_scores(a, g, target, 2, "interaction"), (c, b))
    weights = torch.randn_like(c)
    g1 = torch.autograd.grad((actual * weights).sum(), b, retain_graph=True)[0]
    g2 = torch.autograd.grad((combine_scores(c * 2, b, target, 2, "interaction") * weights).sum(), b)[0]
    assert not torch.allclose(g1, g2)
