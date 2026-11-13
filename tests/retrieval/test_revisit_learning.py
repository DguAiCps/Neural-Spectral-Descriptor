"""실제 재방문 보조 손실의 정답별 제약·패딩·문맥 분리를 확인한다."""

import math

import pytest
import torch

from src.retrieval.revisit_learning import revisit_terms


def test_each_positive_competes_and_padding_has_no_gradient():
    x = torch.tensor(
        [[1.0, 0.0], [0.8, 0.6], [-0.6, 0.8], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    out = revisit_terms(
        x,
        torch.tensor([0]),
        torch.tensor([[1, 2, -1]]),
        torch.tensor([[3, -1]]),
        temperature=1.0,
    )
    expected = (math.log1p(math.exp(-0.8)) + math.log1p(math.exp(0.6))) / 2
    torch.testing.assert_close(out["loss"], torch.tensor([expected], dtype=x.dtype))
    assert not out["all_positive_win"].item()
    out["loss"].sum().backward()
    assert x.grad[1, 0] < 0 and x.grad[2, 0] < 0 and x.grad[3, 0] > 0
    assert (x.grad[4] == 0).all() and torch.isfinite(x.grad).all()


def test_context_loss_ignores_raw_and_is_scale_invariant():
    torch.manual_seed(29)
    x = torch.randn(4, 416, dtype=torch.float64, requires_grad=True)
    args = (torch.tensor([0]), torch.tensor([[1]]), torch.tensor([[2, 3]]))
    first = revisit_terms(x, *args, context_only=True)["loss"]
    changed = x.detach().clone()
    changed[:, :288] += 100
    changed[:, 288:] *= torch.tensor([2.0, 3.0, 4.0, 5.0])[:, None]
    torch.testing.assert_close(
        first, revisit_terms(changed, *args, context_only=True)["loss"]
    )
    first.sum().backward()
    assert (x.grad[:, :288] == 0).all() and torch.isfinite(x.grad).all()


def test_gradcheck_and_zero_vectors_are_finite():
    torch.manual_seed(4)
    args = (
        torch.tensor([0, 1]),
        torch.tensor([[1, -1], [0, 2]]),
        torch.tensor([[3], [3]]),
    )
    x = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: revisit_terms(z, *args)["loss"], (x,))
    out = revisit_terms(torch.zeros(4, 5), *args)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_empty_positive_or_negative_is_rejected():
    with pytest.raises(ValueError):
        revisit_terms(
            torch.eye(3), torch.tensor([0]), torch.tensor([[-1]]), torch.tensor([[2]])
        )
