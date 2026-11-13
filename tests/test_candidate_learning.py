"""후보 학습의 여러 정답·거절·패딩·문맥 기울기 계약을 검사한다."""
import torch

from src.retrieval.candidate_learning import CandidateScorer, candidate_loss, pair_features


def test_multiple_positives_and_reject_gradients():
    logits = torch.tensor([[0., 0., 1.], [1., 2., 3.]], requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    correct = torch.tensor([[True, True, False], [False, False, False]])
    loss = candidate_loss(logits, correct, valid)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert (logits.grad[0, :2] < 0).all()  # 두 정답 모두 강화한다.
    assert logits.grad[0, 2] > 0 and (logits.grad[1] > 0).all()


def test_padding_and_candidate_permutation_invariance():
    torch.manual_seed(7)
    emb = torch.nn.functional.normalize(torch.randn(6, 416), dim=1)
    q = torch.tensor([5, 4])
    c = torch.tensor([[0, 1, 2], [1, 2, 3]])
    static = torch.rand(2, 3, 3)
    valid = torch.ones(2, 3, dtype=torch.bool)
    correct = torch.tensor([[True, False, True], [False, False, False]])
    head = CandidateScorer()
    features = pair_features(emb, q, c, static, valid)
    logits = head(features)
    reference = candidate_loss(logits, correct, valid)
    p = torch.tensor([2, 0, 1])
    torch.testing.assert_close(reference, candidate_loss(logits[:, p], correct[:, p], valid[:, p]))
    cpad = torch.cat([c, torch.full((2, 2), -1)], dim=1)
    vpad = cpad >= 0
    spad = torch.cat([static, torch.full((2, 2, 3), 100.)], dim=1)
    fpad = pair_features(emb, q, cpad, spad, vpad)
    torch.testing.assert_close(features, fpad[:, :3])
    ypad = torch.cat([correct, torch.zeros(2, 2, dtype=torch.bool)], dim=1)
    torch.testing.assert_close(reference, candidate_loss(head(fpad), ypad, vpad))


def test_context_receives_gradient_without_future_or_labels_as_input():
    torch.manual_seed(11)
    raw = torch.randn(80, 416, requires_grad=True)
    emb = torch.nn.functional.normalize(raw, dim=1)
    q, c = torch.tensor([50]), torch.tensor([[1, 3, 20]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    static = torch.rand(1, 3, 3)
    head = CandidateScorer()
    features = pair_features(emb, q, c, static, valid)
    loss = candidate_loss(head(features), torch.tensor([[True, False, False]]), valid)
    loss.backward()
    assert raw.grad[50, 288:].abs().sum() > 0
    assert raw.grad[66:].abs().sum() == 0
    altered = emb.detach().clone()
    altered[66:] = 100.
    torch.testing.assert_close(features.detach(), pair_features(altered, q, c, static, valid))
