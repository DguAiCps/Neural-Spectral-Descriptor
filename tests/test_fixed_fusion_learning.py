"""기존 융합 보존·순위 기울기·정답 없는 행 처리를 검사한다."""
import numpy as np
import torch

from src.retrieval.fixed_fusion_learning import fixed_fusion_scores, positive_ranking_loss


def test_fixed_fusion_matches_numpy_and_ignores_padding():
    torch.manual_seed(31)
    key = torch.rand(3, 7)
    static = torch.rand(3, 7, 3)
    valid = torch.ones(3, 7, dtype=torch.bool)
    valid[1, 4:] = False
    for weights in [(1, 1, 2, 1), (2, 1, 2, 1)]:
        result = fixed_fusion_scores(key, static, valid, weights)
        for row in range(3):
            mask = valid[row].numpy()
            channels = [key[row].numpy()[mask]] + [static[row, :, j].numpy()[mask] for j in range(3)]
            ref = sum(w * (x - x.min()) / max(float(x.max()) - float(x.min()), 1e-8)
                      for w, x in zip(weights, channels))
            np.testing.assert_allclose(result[row].numpy()[mask], ref, atol=1e-6, rtol=1e-6)
        padded_key = torch.cat([key, torch.full((3, 2), 900.)], 1)
        padded_static = torch.cat([static, torch.full((3, 2, 3), 900.)], 1)
        padded_valid = torch.cat([valid, torch.zeros(3, 2, dtype=torch.bool)], 1)
        torch.testing.assert_close(result, fixed_fusion_scores(padded_key, padded_static, padded_valid, weights)[:, :7])


def test_multiple_positives_missing_positive_and_all_positive_are_finite():
    scores = torch.tensor([[.2, .3, .4], [.2, .3, .4], [.2, .3, .4]], requires_grad=True)
    correct = torch.tensor([[True, True, False], [False, False, False], [True, True, True]])
    valid = torch.ones_like(correct)
    losses, active = positive_ranking_loss(scores, correct, valid)
    losses.mean().backward()
    assert active.tolist() == [True, False, False]
    assert torch.isfinite(losses).all() and torch.isfinite(scores.grad).all()
    assert (scores.grad[0, :2] < 0).all() and scores.grad[0, 2] > 0
    assert (scores.grad[1:] == 0).all()
    permutation = torch.tensor([2, 0, 1])
    other, _ = positive_ranking_loss(scores[:, permutation], correct[:, permutation], valid[:, permutation])
    torch.testing.assert_close(losses, other)


def test_static_offsets_change_training_signal_but_receive_no_gradient():
    key = torch.tensor([[.1, .4, .6, .9]], requires_grad=True)
    static = torch.tensor([[[.9, .9, .9], [.2, .1, .2], [.8, .8, .8], [.1, .1, .1]]], requires_grad=True)
    valid = torch.ones_like(key, dtype=torch.bool)
    correct = torch.tensor([[False, True, False, False]])
    scores = fixed_fusion_scores(key, static, valid, (1, 1, 2, 1))
    loss, _ = positive_ranking_loss(scores, correct, valid)
    loss.mean().backward()
    grad = key.grad.clone()
    assert static.grad is None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    key.grad.zero_()
    baseline = fixed_fusion_scores(key, static, valid, (1, 1, 2, 1), False)
    positive_ranking_loss(baseline, correct, valid)[0].mean().backward()
    assert not torch.allclose(grad, key.grad)
