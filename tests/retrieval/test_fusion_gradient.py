"""정규화 폭 대리 기울기의 순전파 정합성과 실제 미분 차이를 검증한다."""
import pytest
import torch

from src.retrieval.candidate_learning import masked_minmax
from src.retrieval.fusion_gradient import fixed_width_minmax, fixed_width_terms
from src.retrieval.fusion_guard import fusion_guard_terms, fusion_guard_loss


def test_forward_equal_with_padding_constant_and_ties():
    key = torch.tensor([[.2, .3, .8, 9.], [.4, .4, .4, 9.], [.1, .1, .8, 9.]], requires_grad=True)
    valid = torch.tensor([[True, True, True, False]]).expand(3, -1)
    got = fixed_width_minmax(key, valid)
    torch.testing.assert_close(got, masked_minmax(key, valid), rtol=0, atol=0)
    assert got.shape == key.shape and got.dtype == key.dtype
    got.sum().backward()
    assert torch.isfinite(key.grad).all() and (key.grad[:, -1] == 0).all()


def test_surrogate_matches_held_width_derivative_not_true_derivative():
    key = torch.tensor([[0., .4, 1.]], dtype=torch.float64, requires_grad=True)
    valid = torch.ones_like(key, dtype=torch.bool)
    # 점수 차이는 공통 최솟값의 미분을 상쇄한다. 분모 미분만 남는 차이를 확인한다.
    exact, surrogate = masked_minmax(key, valid), fixed_width_minmax(key, valid)
    ge = torch.autograd.grad(exact[0, 1] - exact[0, 0], key)[0]
    gs = torch.autograd.grad(surrogate[0, 1] - surrogate[0, 0], key)[0]
    torch.testing.assert_close(gs, torch.tensor([[-1., 1., 0.]], dtype=torch.float64))
    torch.testing.assert_close(ge, torch.tensor([[-.6, 1., -.4]], dtype=torch.float64))
    torch.testing.assert_close(gs.sum(), torch.tensor(0., dtype=torch.float64))


@pytest.mark.parametrize('filtered,protected', [(False, False), (True, True)])
def test_all_terms_forward_equal_and_gradient_finite(filtered, protected):
    torch.manual_seed(13)
    key = torch.rand(8, 12, requires_grad=True)
    teacher = torch.rand(8, 12, requires_grad=True)
    static = torch.rand(8, 12, 3, requires_grad=True)
    valid = torch.ones(8, 12, dtype=torch.bool)
    valid[:, -2:] = False
    correct = torch.zeros_like(valid)
    correct[:, :3] = True
    correct[0] = False
    correct[1] = valid[1]
    args = (key, teacher, static, valid, correct, (1., 1., 2., 1.))
    reference, actual = fusion_guard_terms(*args), fixed_width_terms(*args)
    for name in reference:
        torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
    loss = fusion_guard_loss(actual, filtered, protected)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(key.grad).all()
    assert teacher.grad is None and static.grad is None
    assert (key.grad[:, -2:] == 0).all()


def test_empty_row_rejected():
    with pytest.raises(ValueError):
        fixed_width_minmax(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool))
