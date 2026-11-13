"""중심 보존의 값 항등식·같은 주의 가중치·학습 경로를 검산한다."""
import copy

import pytest
import torch

from gnn.model import DiffAttnConv
from gnn.nonlinear_relation import NonlinearRelationConv
from gnn.center_preserving_relation import CenterPreservingRelationConv


@pytest.mark.parametrize("beta", [0.5, 1.])
def test_fixed_attention_dense_value_identity_and_gradients(beta):
    torch.manual_seed(521)
    x = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    edge = torch.tensor([[0, 2, 3, 4, 6], [1, 1, 1, 5, 5]])
    attr = torch.randn(5, 3, dtype=torch.float64)
    old = NonlinearRelationConv(DiffAttnConv(8, heads=2, edge_dim=3, dropout=0.)).double().eval()
    net = CenterPreservingRelationConv(copy.deepcopy(old), beta).double().eval()
    difference, (_, a) = old(x, edge, attr, return_attention_weights=True)
    output, (_, b) = net(x, edge, attr, return_attention_weights=True)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    center = old.W_v(x)
    expected = torch.zeros_like(x)
    for node in (1, 5):
        mask = edge[1] == node
        values = old.W_v(x[edge[0, mask]]).reshape(-1, 2, 4)
        average = (a[mask, :, None]*values).sum(0).flatten()
        expected[node] = average-(1-beta)*center[node]
        torch.testing.assert_close(output[node], difference[node]+beta*center[node], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(output, expected, atol=1e-13, rtol=1e-13)
    # 기준 주의 가중치도 미분 가능한 독립 목적 노드 계산으로 확인한다.
    dense = torch.zeros_like(x)
    for node in (1, 5):
        mask = edge[1] == node
        delta = x[edge[0, mask]]-x[node]
        query = net.W_q(x[node]).reshape(2, 4)
        key = net.W_k(delta).tanh().reshape(-1, 2, 4)
        scores = (query*key).sum(-1)/2+net.bias_mlp(attr[mask])
        values = (net.W_v(x[edge[0, mask]])-(1-beta)*net.W_v(x[node])).reshape(-1, 2, 4)
        dense[node] = (scores.softmax(0)[..., None]*values).sum(0).flatten()
    actual_grad = torch.autograd.grad(output.square().sum(), (x, *net.parameters()), retain_graph=True)
    dense_grad = torch.autograd.grad(dense.square().sum(), (x, *net.parameters()))
    for actual, reference in zip(actual_grad, dense_grad):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
        assert torch.isfinite(actual).all()


def test_zero_exact_dropout_replay_and_same_parameters():
    torch.manual_seed(522)
    old = NonlinearRelationConv(DiffAttnConv(8, heads=2, dropout=.2)).train()
    net = CenterPreservingRelationConv(copy.deepcopy(old), 0.).train()
    assert old.state_dict().keys() == net.state_dict().keys()
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in net.parameters())
    x = torch.randn(5, 8, requires_grad=True)
    edge = torch.tensor([[0, 1, 2, 3], [2, 2, 4, 4]])
    torch.manual_seed(523)
    a = old(x, edge)
    torch.manual_seed(523)
    b = net(x, edge)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(a.sum(), x)[0], torch.autograd.grad(b.sum(), x)[0], atol=0, rtol=0)


@pytest.mark.parametrize("beta", [.5, 1.])
def test_autocast_empty_edges_and_finite_backward(beta):
    torch.manual_seed(524)
    net = CenterPreservingRelationConv(NonlinearRelationConv(DiffAttnConv(8, heads=2, dropout=.1)), beta)
    x = torch.randn(5, 8, requires_grad=True)
    edge = torch.tensor([[0, 1, 2, 3], [2, 2, 4, 4]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = net(x, edge)
        loss = output.float().square().sum()
        empty = net(x, torch.empty((2, 0), dtype=torch.long))
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    torch.testing.assert_close(empty, torch.zeros_like(empty), atol=0, rtol=0)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
