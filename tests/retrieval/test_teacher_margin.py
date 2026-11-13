"""순위 보호 손실의 방향과 기준 모델의 기울기 차단을 검사한다."""
import torch

from retrieval.teacher_margin import teacher_margin_loss, triplet_margin


def test_guard_only_lost_correct_margin_and_detaches_teacher():
    current = torch.tensor([0.1, 0.4, -0.3], requires_grad=True)
    teacher = torch.tensor([0.2, 0.3, -0.1], requires_grad=True)
    value = teacher_margin_loss(current, teacher)
    torch.testing.assert_close(value, torch.tensor(1 / 3))
    value.backward()
    assert current.grad[0] < 0
    assert torch.equal(current.grad[1:], torch.zeros(2))
    assert teacher.grad is None


def test_triplet_values_and_empty_backward_are_finite():
    key = torch.tensor([[1., 0.], [0.8, 0.6], [-1., 0.]], requires_grad=True)
    ids = torch.tensor([[0, 1, 2]])
    margin = triplet_margin(key, ids)
    torch.testing.assert_close(margin, torch.tensor([1.8]))
    teacher_margin_loss(margin, torch.tensor([1.9])).backward()
    assert torch.isfinite(key.grad).all() and key.grad.abs().sum() > 0
    empty = torch.empty(0, requires_grad=True)
    loss = teacher_margin_loss(empty, torch.empty(0))
    assert loss == 0 and torch.isfinite(loss)
    loss.backward()
