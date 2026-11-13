"""attention 공유 범위만 바뀌는지 독립 계산과 반례로 확인한다."""
import copy

import pytest
import torch

from gnn.model import DiffAttnConv
from gnn.nonlinear_relation import NonlinearRelationConv
from gnn.grouped_relation import GroupedRelationConv


@pytest.mark.parametrize("relation", ["diff", "standard"])
def test_one_group_exactly_matches_reference(relation):
    torch.manual_seed(71)
    ref = DiffAttnConv(8, heads=2, dropout=.1, value_source=relation)
    old = NonlinearRelationConv(copy.deepcopy(ref))
    new = GroupedRelationConv(copy.deepcopy(ref), 1)
    x = torch.randn(5, 8)
    edges = torch.tensor([[0, 2, 3, 4], [1, 1, 2, 2]])
    outputs = []
    for net in (old, new):
        torch.manual_seed(19)
        output = net(x, edges)
        output.square().sum().backward()
        outputs.append(output)
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    for (n, p), (n2, p2) in zip(old.named_parameters(), new.named_parameters()):
        assert n == n2
        torch.testing.assert_close(p.grad, p2.grad, atol=0, rtol=0)


@pytest.mark.parametrize("relation", ["diff", "standard"])
def test_independent_groups_dense_equation_and_gradient(relation):
    torch.manual_seed(43)
    net = GroupedRelationConv(DiffAttnConv(8, heads=2, edge_dim=3, dropout=0.,
                                         value_source=relation), 2).double()
    x = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([[0, 2, 3, 4], [1, 1, 2, 2]])
    attr = torch.randn(4, 3, dtype=torch.float64)
    gate = torch.rand(4, 2, dtype=torch.float64)
    bias = torch.randn(4, 1, dtype=torch.float64)
    actual = net(x, edges, attr, bias, gate)
    expected = torch.zeros_like(x)
    for i in (1, 2):
        keep = edges[1] == i
        r = x[edges[0, keep]] - x[i] if relation == "diff" else x[edges[0, keep]]
        for head in (0, 1):
            for group in (0, 1):
                s = slice(head * 4 + group * 2, head * 4 + group * 2 + 2)
                q = net.W_q(x[i])[s]
                k = net.W_k(r).tanh()[:, s]
                score = (q * k).sum(-1) / 2 ** .5 + net.bias_mlp(attr[keep])[:, head] + bias[keep, 0]
                value = net.W_v(r)[:, s]
                expected[i, s] = (score.softmax(0)[:, None] * value * gate[keep, head, None]).sum(0)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_groups_can_choose_different_neighbors_and_zero_difference():
    net = GroupedRelationConv(DiffAttnConv(2, heads=1, dropout=0.), 2).double().eval()
    with torch.no_grad():
        for projection in (net.W_q, net.W_k, net.W_v):
            projection.weight.copy_(torch.eye(2))
    x = torch.tensor([[1., 1.], [2., 0.], [0., 2.]], dtype=torch.float64)
    edges = torch.tensor([[1, 2], [0, 0]])
    output, (_, weight) = net(x, edges, return_attention_weights=True)
    assert weight[0, 0] > weight[1, 0] and weight[0, 1] < weight[1, 1]
    assert output.shape == x.shape and torch.isfinite(output).all()
    torch.testing.assert_close(net(torch.ones_like(x), edges), torch.zeros_like(x), atol=0, rtol=0)
    torch.testing.assert_close(net(x, edges[:, :0]), torch.zeros_like(x), atol=0, rtol=0)


def test_amp_parameter_count_and_permutation():
    torch.manual_seed(77)
    ref = DiffAttnConv(8, heads=2, dropout=0.)
    net = GroupedRelationConv(copy.deepcopy(ref), 2)
    assert net.state_dict().keys() == NonlinearRelationConv(ref).state_dict().keys()
    assert sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in ref.parameters())
    x = torch.randn(5, 8, requires_grad=True)
    edges = torch.tensor([[0, 2, 3, 4], [1, 1, 2, 2]])
    perm = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.argsort(perm)
    torch.testing.assert_close(net(x, edges)[perm], net(x[perm], inverse[edges]))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = net(x, edges)
        loss = output.float().square().mean()
    loss.backward()
    assert output.dtype == torch.bfloat16 and torch.isfinite(x.grad).all()
