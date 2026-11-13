"""배치 오답 노출과 문맥 출력 기울기를 독립 계산과 대조한다."""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _pdv2_train_failure_lib import (
    batch_geometry, batch_membership, cosine_descent_response,
    normalization_gradient, wrong_place_neighbors,
)


def test_nearby_excludes_future_and_other_sessions() -> None:
    pos = np.zeros((100, 3))
    seq = np.repeat([0, 1], 50)
    pos[:, 0] = np.tile(np.arange(50), 2)
    winner = np.full(100, -1)
    winner[[40, 90]] = [5, 55]
    nearby = wrong_place_neighbors(pos, seq, winner)
    np.testing.assert_array_equal(nearby[40].indices, np.arange(11))
    np.testing.assert_array_equal(nearby[90].indices, np.arange(50, 61))
    batch = np.array([[40, 41, 5], [90, 5, 55], [5, 40, 6], [42, 6, 7]])
    masks = batch_membership(batch, winner, nearby)
    assert masks["exact"][0, 1] and masks["hard_exact"][0]
    assert masks["nearby"][0, 3] and not masks["nearby"][1, 1]
    assert masks["reverse"][2, 2] == 0
    for row in range(len(batch)):
        a, _, n = batch[row]
        for col, p in enumerate(batch[:, 1]):
            expected = row != col and winner[a] >= 0 and winner[a] == p
            assert masks["exact"][row, col] == expected


def test_geometry_matches_autograd_with_duplicate_frames() -> None:
    torch.manual_seed(43)
    raw = torch.randn(11, 9, dtype=torch.float64, requires_grad=True)
    batch = torch.tensor([[0, 1, 4], [0, 2, 5], [3, 1, 4], [4, 6, 0]])
    unit = F.normalize(raw, dim=1)
    loss, prob, ga, gp, gn = batch_geometry(unit.detach(), batch)
    a, p, n = batch.T
    logits = torch.cat((unit[a] @ unit[p].T, (unit[a] * unit[n]).sum(1, keepdim=True)), 1) / .1
    expected_loss = F.cross_entropy(logits, torch.arange(len(a)))
    expected = torch.autograd.grad(expected_loss, raw)[0]
    combined = torch.zeros_like(raw)
    for ids, gradient in ((a, ga), (p, gp), (n, gn)):
        combined.index_add_(0, ids, gradient / len(a))
    actual = normalization_gradient(unit.detach(), raw.detach().norm(dim=1, keepdim=True), combined)
    torch.testing.assert_close(loss.mean(), expected_loss)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert torch.isfinite(actual).all() and prob.shape == (4, 5)


def test_context_response_matches_finite_difference() -> None:
    torch.manual_seed(9)
    raw = torch.randn(7, 8, dtype=torch.float64)
    gradient = torch.randn_like(raw)
    q, j = torch.tensor([0, 3, 5]), torch.tensor([1, 2, 0])
    actual = cosine_descent_response(raw, gradient, q, j, context_start=4)
    direction = gradient.clone()
    direction[:, :4] = 0
    eps = 1e-6
    plus, minus = F.normalize(raw-eps*direction, dim=1), F.normalize(raw+eps*direction, dim=1)
    finite = ((plus[q]*plus[j]).sum(1)-(minus[q]*minus[j]).sum(1))/(2*eps)
    torch.testing.assert_close(actual, finite, rtol=1e-7, atol=1e-9)
