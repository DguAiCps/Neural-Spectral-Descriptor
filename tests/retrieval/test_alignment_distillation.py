"""정렬 교사의 위치 정답 보호와 미분 경로를 확인한다."""
import torch

from retrieval.alignment_distillation import alignment_ranking_loss


def test_wrong_teacher_is_not_distilled_and_static_has_no_gradient():
    key = torch.tensor([[.2, .6, .8, .1]], requires_grad=True)
    aux = torch.tensor([[.3, .7, .9, .2]], requires_grad=True)
    teacher = torch.tensor([[.4, .9, .8, .1]], requires_grad=True)
    correct = torch.tensor([[True, False, False, False]])
    present = torch.ones_like(correct)
    joint, _ = alignment_ranking_loss(key, aux, teacher, correct, present, present, (1., .5), 'joint')
    learned, info = alignment_ranking_loss(key, aux, teacher, correct, present, present, (1., .5), 'teacher')
    torch.testing.assert_close(joint, learned, rtol=0, atol=0)
    assert info['trusted'].item() == 0
    learned.backward()
    assert key.grad is not None and torch.isfinite(key.grad).all() and key.grad.abs().sum() > 0
    assert aux.grad is None and teacher.grad is None


def test_trusted_teacher_gradient_and_gray_candidate_are_finite():
    key = torch.tensor([[.4, .6, .9, .1]], requires_grad=True)
    aux = torch.tensor([[.6, .8, .9, .2]])
    teacher = torch.tensor([[1., .5, .7, .1]])
    correct = torch.tensor([[True, False, False, False]])
    present = torch.ones_like(correct)
    eligible = torch.tensor([[True, True, False, True]])
    loss, info = alignment_ranking_loss(key, aux, teacher, correct, present, eligible, (1., .5), 'teacher')
    assert info['trusted'].item() == 1 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(key.grad).all()
    assert key.grad[0, 0] < 0  # 정답 키 점수를 올리는 방향이다.
    changed = key.detach() - .01 * key.grad
    next_loss, _ = alignment_ranking_loss(changed, aux, teacher, correct, present, eligible, (1., .5), 'teacher')
    assert next_loss < loss


def test_no_positive_produces_zero_loss_and_finite_gradient():
    key = torch.tensor([[.2, .6, .9]], requires_grad=True)
    present = torch.ones_like(key, dtype=torch.bool)
    correct = torch.zeros_like(present)
    loss, info = alignment_ranking_loss(key, key.detach(), key.detach(), correct, present, present, (1., .5), 'teacher')
    loss.backward()
    assert loss.item() == 0 and info['active'].item() == 0
    assert torch.isfinite(key.grad).all() and key.grad.abs().sum() == 0
