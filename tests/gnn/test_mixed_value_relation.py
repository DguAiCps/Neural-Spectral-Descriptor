"""고정 혼합 값의 기존 연산 재현·큰 차이 보존·집계 기울기를 확인한다."""
import copy

import pytest
import torch

from gnn.model import DiffAttnConv
from gnn.mixed_value_relation import MixedValueRelationConv
from gnn.value_relation import ValueRelationConv


@pytest.mark.parametrize("relation", ["diff", "standard"])
@pytest.mark.parametrize("weight,mode", [(0., "tanh"), (1., "linear")])
def test_endpoints_replay_values_attention_and_gradients(relation, weight, mode):
    torch.manual_seed(42)
    reference = DiffAttnConv(8, heads=2, dropout=.1, edge_dim=3,
                             value_source=relation).double()
    old = ValueRelationConv(copy.deepcopy(reference), mode)
    new = MixedValueRelationConv(copy.deepcopy(reference), weight)
    assert old.state_dict().keys() == new.state_dict().keys()
    for name in old.state_dict():
        torch.testing.assert_close(old.state_dict()[name], new.state_dict()[name], atol=0, rtol=0)
    x = torch.randn(6, 8, dtype=torch.float64)
    edges = torch.tensor([[0, 2, 4, 5], [1, 1, 3, 3]])
    attrs = torch.randn(4, 3, dtype=torch.float64)
    outputs, gradients, attentions = [], [], []
    for net in (old, new):
        torch.manual_seed(17)
        value, (_, attention) = net(x, edges, attrs, return_attention_weights=True)
        value.square().sum().backward()
        outputs.append(value)
        attentions.append(attention)
        gradients.append({name: p.grad for name, p in net.named_parameters()})
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    torch.testing.assert_close(*attentions, atol=0, rtol=0)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=0, rtol=0)


def test_odd_zero_local_slope_and_unbounded_tail():
    net = MixedValueRelationConv(DiffAttnConv(8, heads=2, dropout=0.)).double()
    z = torch.tensor([-100., -3., 0., 3., 100.], dtype=torch.float64, requires_grad=True)
    value = net.transform_value(z)
    torch.testing.assert_close(value, -net.transform_value(-z), atol=0, rtol=0)
    assert value[2] == 0 and value[-1] > value[-2] > 1
    value.sum().backward()
    torch.testing.assert_close(z.grad[[0, 2, 4]], torch.tensor([.5, 1., .5]).double())
    x = torch.ones(5, 8, dtype=torch.float64)
    edges = torch.tensor([[0, 2, 4], [1, 1, 3]])
    torch.testing.assert_close(net(x, edges), torch.zeros_like(x), atol=0, rtol=0)
    torch.testing.assert_close(net(x, edges[:, :0]), torch.zeros_like(x), atol=0, rtol=0)


@pytest.mark.parametrize("relation", ["diff", "standard"])
def test_dense_reference_permutation_gradients_and_amp(relation):
    torch.manual_seed(29)
    net = MixedValueRelationConv(DiffAttnConv(8, heads=2, dropout=0., edge_dim=3,
                                             value_source=relation)).double()
    x = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([[0, 2, 4, 5], [1, 1, 3, 3]])
    attrs = torch.randn(4, 3, dtype=torch.float64)
    gate = torch.rand(4, 2, dtype=torch.float64)
    actual = net(x, edges, attrs, edge_value_gate=gate)
    expected = torch.zeros_like(x)
    for node in (1, 3):
        mask = edges[1] == node
        relation_x = x[edges[0, mask]] - x[node] if relation == "diff" else x[edges[0, mask]]
        q = net.W_q(x[node]).reshape(2, 4)
        k = net.W_k(relation_x).tanh().reshape(-1, 2, 4)
        score = (q * k).sum(-1) / 2 + net.bias_mlp(attrs[mask])
        z = net.W_v(relation_x).reshape(-1, 2, 4)
        v = (z + torch.tanh(z)) / 2
        expected[node] = (score.softmax(0)[..., None] * v * gate[mask, :, None]).sum(0).flatten()
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    permutation = torch.tensor([3, 0, 4, 1, 5, 2])
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(actual[permutation],
        net(x[permutation], inverse[edges], attrs, edge_value_gate=gate), atol=1e-13, rtol=0)
    actual.square().sum().backward()
    assert actual.shape == x.shape and actual.dtype == x.dtype
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    net = net.float()
    net.zero_grad()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = net(x.detach().float(), edges, attrs.float())
        loss = result.float().square().mean()
    loss.backward()
    assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


@pytest.mark.parametrize("weight", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_weights(weight):
    with pytest.raises(ValueError):
        MixedValueRelationConv(DiffAttnConv(8, heads=2), weight)
