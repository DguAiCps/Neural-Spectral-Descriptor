"""문맥 상대 점수 손실의 방향·패딩·무라벨 정의를 확인한다."""
import torch

from src.retrieval.context_consistency import context_pair_scores, relational_consistency


def test_identity_constant_shift_and_padding():
    clean = torch.tensor([[.1, .4, .8, 99.]], requires_grad=True)
    changed = torch.tensor([[.3, .6, 1., -99.]], requires_grad=True)
    valid = torch.tensor([[True, True, True, False]])
    loss = relational_consistency(clean, changed, valid)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-11, rtol=0)
    loss.sum().backward()
    assert clean.grad is None and changed.grad[0, -1] == 0


def test_gradient_restores_relative_scores():
    clean = torch.tensor([[.8, .2]], requires_grad=True)
    changed = torch.tensor([[.3, .7]], requires_grad=True)
    loss = relational_consistency(clean, changed, torch.ones(1, 2, dtype=torch.bool))
    loss.sum().backward()
    assert clean.grad is None and changed.grad[0, 0] < 0 < changed.grad[0, 1]
    assert torch.isfinite(changed.grad).all()
    torch.testing.assert_close(changed.grad.sum(), torch.tensor(0.))


def test_context_scores_only_and_zero_vectors_finite():
    torch.manual_seed(19)
    embedding = torch.randn(5, 416, requires_grad=True)
    queries, candidates = torch.tensor([0, 1]), torch.tensor([[2, 3, -1], [0, 4, -1]])
    result = context_pair_scores(embedding, queries, candidates)
    assert result.shape == (2, 3) and result.dtype == embedding.dtype and torch.isfinite(result).all()
    result.sum().backward()
    assert (embedding.grad[:, :288] == 0).all()
    zeros = context_pair_scores(torch.zeros(5, 416), queries, candidates)
    assert torch.isfinite(zeros).all()
