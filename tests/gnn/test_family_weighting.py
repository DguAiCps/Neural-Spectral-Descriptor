"""계열 비중 외에 대조 후보·수식·기울기가 바뀌지 않는지 검사한다."""
import pytest
import torch

from gnn.family_weighting import equal_family_mean, infonce_rows
from gnn.trainer import InfoNCELoss


def test_rows_preserve_original_candidates_loss_and_gradients():
    generator = torch.Generator().manual_seed(51)
    tensors = [torch.randn(7, 9, generator=generator, dtype=torch.float64,
                           requires_grad=True) for _ in range(3)]
    original = InfoNCELoss(0.1)(*tensors)
    rows = infonce_rows(*tensors)
    assert rows.shape == (7,) and rows.dtype == torch.float64
    assert torch.isfinite(rows).all()
    torch.testing.assert_close(rows.mean(), original, atol=1e-14, rtol=1e-14)
    a = torch.autograd.grad(original, tensors, retain_graph=True)
    b = torch.autograd.grad(rows.mean(), tensors)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=1e-14, rtol=1e-14)


def test_balancing_gives_half_weight_to_each_family():
    rows = torch.tensor([2., 4., 6., 8.], dtype=torch.float64, requires_grad=True)
    families = torch.tensor([0, 1, 1, 1])
    balanced = equal_family_mean(rows, families)
    torch.testing.assert_close(balanced, torch.tensor(4., dtype=torch.float64))
    gradient, = torch.autograd.grad(balanced, rows)
    torch.testing.assert_close(gradient, torch.tensor([.5, 1/6, 1/6, 1/6], dtype=torch.float64))
    # 같은 행별 오류의 개수만 늘리면 계열 평균은 그대로다.
    repeated = torch.tensor([2., 4., 6., 8., 4., 6., 8.], dtype=torch.float64)
    torch.testing.assert_close(equal_family_mean(repeated, torch.tensor([0, 1, 1, 1, 1, 1, 1])), balanced)


def test_equal_counts_reduce_to_original_mean():
    rows = torch.tensor([1., 3., 2., 8.])
    torch.testing.assert_close(equal_family_mean(rows, torch.tensor([0, 0, 1, 1])), rows.mean())


def test_invalid_or_missing_family_is_rejected():
    with pytest.raises(ValueError):
        equal_family_mean(torch.ones(3), torch.ones(3, dtype=torch.long))
    with pytest.raises(ValueError):
        equal_family_mean(torch.ones(3), torch.tensor([0, 1, 2]))
    with pytest.raises(ValueError):
        equal_family_mean(torch.ones(3, 1), torch.tensor([0, 1, 1]))
