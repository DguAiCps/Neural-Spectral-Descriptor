"""게이트 강도·기울기와 전체 키의 크기 불변성을 검산한다."""
import torch
import torch.nn.functional as F

from retrieval.gated_context import gated_context_score
from retrieval.nonrevisit_learning import context_score


def test_gate_product_and_gradient_survive_key_normalization():
    alpha = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
    raw = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    context = torch.tensor([[1., 0.], [0.6, 0.8]], dtype=torch.float64)
    key = F.normalize(torch.cat((raw, alpha[:, None] * context), dim=-1), dim=-1)
    plan = {"query": torch.tensor([[[0]]]), "candidate": torch.tensor([[[1]]]),
            "weight": torch.ones(1, 1, 1, dtype=torch.float64)}
    score = gated_context_score(key, plan, 2)
    torch.testing.assert_close(score, torch.tensor([0.126], dtype=torch.float64))
    gradient, = torch.autograd.grad(score.sum(), alpha, retain_graph=True)
    torch.testing.assert_close(gradient, torch.tensor([0.42, 0.18], dtype=torch.float64))
    old = context_score(key, plan, 2)
    old_gradient, = torch.autograd.grad(old.sum(), alpha)
    torch.testing.assert_close(old_gradient, torch.zeros_like(alpha), atol=1e-14, rtol=0)
    torch.testing.assert_close(gated_context_score(key * 3, plan, 2), score)


def test_zero_input_is_finite_and_neutral():
    plan = {"query": torch.tensor([[[0]]]), "candidate": torch.tensor([[[1]]]),
            "weight": torch.ones(1, 1, 1)}
    key = torch.zeros(2, 416, requires_grad=True)
    value = gated_context_score(key, plan)
    value.sum().backward()
    assert value.item() == 0 and torch.isfinite(key.grad).all()
