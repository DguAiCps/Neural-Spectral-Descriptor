"""attention 점수 제거가 값·저장 가중치를 바꾸지 않는지 검사한다."""
import copy

import pytest
import torch
from torch import nn

from gnn.attention_probe import attach_attention_probe
from gnn.model import DiffAttnConv
from gnn.value_relation import ValueRelationConv


def network():
    model = nn.Module()
    model.gnn = nn.Module()
    model.gnn.convs = nn.ModuleList([
        ValueRelationConv(DiffAttnConv(8, heads=2, dropout=0., edge_dim=3), "tanh")
        for _ in range(2)])
    return model.double().eval()


@pytest.mark.parametrize("mode", ["full", "content", "geometry", "uniform"])
def test_dense_scores_state_and_handle_removal(mode):
    torch.manual_seed(7)
    model = network()
    with torch.no_grad():
        for conv in model.gnn.convs:
            conv.bias_mlp[-1].weight.normal_()
    reference = copy.deepcopy(model)
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    handles = attach_attention_probe(model, mode)
    x = torch.randn(5, 8, dtype=torch.float64)
    edge = torch.tensor([[0, 2, 4], [1, 1, 1]])
    attr = torch.randn(3, 3, dtype=torch.float64)
    for conv, original in zip(model.gnn.convs, reference.gnn.convs):
        result, (_, attn) = conv(x, edge, attr, return_attention_weights=True)
        delta = x[edge[0]] - x[1]
        query = original.W_q(x[1]).reshape(2, 4)
        key = original.W_k(delta).tanh().reshape(3, 2, 4)
        content = (query * key).sum(-1) / 2
        bias = original.bias_mlp(attr)
        logits = torch.zeros_like(bias)
        if mode in ("full", "content"):
            logits += content
        if mode in ("full", "geometry"):
            logits += bias
        expected_attention = logits.softmax(0)
        torch.testing.assert_close(attn, expected_attention, atol=1e-13, rtol=0)
        expected = torch.zeros_like(x)
        expected[1] = (expected_attention[..., None] * original.W_v(delta).tanh().reshape(3, 2, 4)).sum(0).flatten()
        torch.testing.assert_close(result, expected, atol=1e-13, rtol=0)
        assert result.dtype == x.dtype and torch.isfinite(result).all()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
    for handle in handles:
        handle.remove()
    for conv, original in zip(model.gnn.convs, reference.gnn.convs):
        torch.testing.assert_close(conv(x, edge, attr), original(x, edge, attr), atol=0, rtol=0)


def test_training_is_rejected():
    model = network()
    attach_attention_probe(model, "uniform")
    model.train()
    with pytest.raises(RuntimeError, match="추론"):
        model.gnn.convs[0](torch.ones(3, 8).double(), torch.tensor([[0], [1]]),
                          torch.ones(1, 3).double())
