"""별도 오답 손실의 기울기와 배치 크기 의존성을 확인한다."""
import torch
import torch.nn.functional as F

from gnn.stratified_contrastive import StratifiedContrastiveLoss


def test_stratified_loss_matches_binary_cross_entropy_and_has_gradients():
    generator = torch.Generator().manual_seed(17)
    tensors = [torch.randn(7, 9, generator=generator, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    loss = StratifiedContrastiveLoss(.2)(*tensors)
    a, p, n = (F.normalize(t, dim=1) for t in tensors)
    cross = a @ p.T / .2
    logits = torch.stack((cross.diagonal(), (a*n).sum(1)/.2), 1)
    expected = .5*(F.cross_entropy(cross, torch.arange(7)) + F.cross_entropy(logits, torch.zeros(7, dtype=torch.long)))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert loss.ndim == 0 and loss.dtype == torch.float64 and torch.isfinite(loss)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in tensors)


def test_explicit_negative_gradient_does_not_shrink_with_batch_size():
    # 모든 쌍 점수가 0인 조건에서, 원래 혼합 softmax는 명시적 오답의
    # 질의당 기울기가 1/(B+1), 분리 항은 0.25다(배치 평균을 제외).
    for batch in (4, 64):
        cross = torch.zeros(batch, batch, dtype=torch.float64, requires_grad=True)
        hard = torch.zeros(batch, dtype=torch.float64, requires_grad=True)
        loss = .5*(F.cross_entropy(cross, torch.arange(batch)) + F.softplus(hard-cross.diagonal()).mean())
        grad, = torch.autograd.grad(loss, hard)
        torch.testing.assert_close(grad*batch, torch.full_like(hard, .25))
