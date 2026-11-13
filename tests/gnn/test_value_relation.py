"""집계 전 값 처리의 동등성·대칭성·표현 차이·기울기를 검산한다."""
import copy

import pytest
import torch

from gnn.model import DiffAttnConv
from gnn.nonlinear_relation import NonlinearRelationConv
from gnn.value_relation import ValueRelationConv


@pytest.mark.parametrize("relation", ["diff", "standard"])
def test_linear_replays_existing_operator_and_gradients(relation):
    torch.manual_seed(42)
    ref = DiffAttnConv(8, heads=2, dropout=.1, edge_dim=3,
                       value_source=relation).double()
    old = NonlinearRelationConv(copy.deepcopy(ref))
    new = ValueRelationConv(copy.deepcopy(ref), "linear")
    assert old.state_dict().keys() == new.state_dict().keys()
    x = torch.randn(6, 8, dtype=torch.float64)
    edge = torch.tensor([[0, 2, 4, 5], [1, 1, 3, 3]])
    attr = torch.randn(4, 3, dtype=torch.float64)
    outputs, grads = [], []
    for net in (old, new):
        torch.manual_seed(17)
        y = net(x, edge, attr)
        y.square().sum().backward()
        outputs.append(y)
        grads.append({n: p.grad for n, p in net.named_parameters()})
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    for name in grads[0]:
        torch.testing.assert_close(grads[0][name], grads[1][name], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["linear", "tanh", "odd_even"])
def test_constant_neighborhood_empty_edges_and_permutation(mode):
    torch.manual_seed(7)
    net = ValueRelationConv(DiffAttnConv(8, heads=2, dropout=0.), mode).double().eval()
    edge = torch.tensor([[0, 2, 3, 4], [1, 1, 2, 2]])
    x = torch.randn(1, 8, dtype=torch.float64).repeat(5, 1)
    torch.testing.assert_close(net(x, edge), torch.zeros_like(x), atol=0, rtol=0)
    x = torch.randn(5, 8, dtype=torch.float64)
    torch.testing.assert_close(net(x, edge[:, :0]), torch.zeros_like(x))
    perm = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.argsort(perm)
    torch.testing.assert_close(net(x, edge)[perm], net(x[perm], inverse[edge]),
                               atol=1e-13, rtol=0)


def test_equal_mean_neighborhoods_have_distinct_even_messages():
    edge = torch.tensor([[1, 2], [0, 0]])
    x = torch.tensor([[0., 0.], [-.2, -.2], [.2, .2]], dtype=torch.float64)
    y = x * 3
    for mode in ("linear", "tanh", "odd_even"):
        net = ValueRelationConv(DiffAttnConv(2, heads=1, dropout=0.), mode).double().eval()
        with torch.no_grad():
            net.W_q.weight.zero_()
            net.W_v.weight.copy_(torch.eye(2))
        a, b = net(x, edge), net(y, edge)
        assert a[0, 0] == b[0, 0] == 0
        if mode == "odd_even":
            assert b[0, 1] > a[0, 1] > 0
        else:
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("relation", ["diff", "standard"])
@pytest.mark.parametrize("mode", ["tanh", "odd_even"])
def test_dense_reference_gradients_and_amp(relation, mode):
    torch.manual_seed(29)
    net = ValueRelationConv(DiffAttnConv(8, heads=2, dropout=0., edge_dim=3,
                                        value_source=relation), mode).double()
    x = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
    edge = torch.tensor([[0, 2, 4, 5], [1, 1, 3, 3]])
    attr = torch.randn(4, 3, dtype=torch.float64)
    gate = torch.rand(4, 2, dtype=torch.float64)
    actual = net(x, edge, attr, edge_value_gate=gate)
    expected = torch.zeros_like(x)
    for i in (1, 3):
        mask = edge[1] == i
        d = x[edge[0, mask]] - x[i] if relation == "diff" else x[edge[0, mask]]
        q = net.W_q(x[i]).reshape(2, 4)
        k = net.W_k(d).tanh().reshape(-1, 2, 4)
        score = (q * k).sum(-1) / 2 + net.bias_mlp(attr[mask])
        v = net.W_v(d).tanh().reshape(-1, 2, 4)
        if mode == "odd_even":
            v = torch.stack((v[..., 0], v[..., 1], v[..., 2] ** 2,
                             v[..., 3] ** 2), dim=-1)
        expected[i] = (score.softmax(0)[..., None] * v * gate[mask, :, None]).sum(0).flatten()
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    actual.square().sum().backward()
    assert actual.shape == x.shape and actual.dtype == x.dtype
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    net = net.float()
    net.zero_grad()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = net(x.detach().float(), edge, attr.float())
        loss = result.float().square().mean()
    loss.backward()
    assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
