"""좌표계·3D 거리 경계·제외된 항의 기울기를 독립 확인한다."""
import pytest
import torch
import torch.nn.functional as F

from gnn.negative_exclusion import exclusion_masks, excluded_infonce
from gnn.trainer import InfoNCELoss


def test_coordinate_frames_three_dimensions_and_closed_boundary():
    pos = torch.tensor([[0., 0., 0.], [0., 0., 5.], [0., 0., 5.001],
                        [0., 0., 0.], [0., 0., 25.], [0., 0., 25.001]], dtype=torch.float64)
    frames = torch.tensor([0, 0, 0, 1, 0, 0])
    triplets = torch.tensor([[0, 1, 4], [0, 2, 5], [3, 3, 0]])
    for radius in (5., 25.):
        cross, hard = exclusion_masks(pos, frames, triplets, radius)
        expected = torch.zeros(3, 3, dtype=torch.bool)
        expected_hard = torch.zeros(3, dtype=torch.bool)
        for i, (a, _, n) in enumerate(triplets.tolist()):
            expected_hard[i] = frames[a] == frames[n] and float((pos[a]-pos[n]).norm()) <= radius
            for j, (_, p, _) in enumerate(triplets.tolist()):
                expected[i, j] = i != j and frames[a] == frames[p] and float((pos[a]-pos[p]).norm()) <= radius
        assert torch.equal(cross, expected) and torch.equal(hard, expected_hard)
    assert hard.tolist() == [True, False, False]


def test_empty_mask_replays_original_loss_and_gradients():
    torch.manual_seed(3)
    values = [torch.randn(4, 7, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    plain = InfoNCELoss(.1)(*values)
    changed = excluded_infonce(*values, torch.zeros(4, 4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
    assert torch.equal(plain, changed)
    a = torch.autograd.grad(plain, values, retain_graph=True)
    b = torch.autograd.grad(changed, values)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_independent_ragged_denominators_and_gradients():
    torch.manual_seed(4)
    values = [torch.randn(3, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    mask = torch.tensor([[False, True, False], [True, False, True], [True, True, False]])
    hard_mask = torch.tensor([False, True, True])
    actual = excluded_infonce(*values, mask, hard_mask)
    a, p, n = [F.normalize(x, dim=1) for x in values]
    losses = []
    for i in range(3):
        logits = [(a[i]*p[j]).sum()/.1 for j in range(3) if not mask[i, j]]
        if not hard_mask[i]:
            logits.append((a[i]*n[i]).sum()/.1)
        losses.append(torch.logsumexp(torch.stack(logits), 0)-(a[i]*p[i]).sum()/.1)
    expected = torch.stack(losses).mean()
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual, values, retain_graph=True)
    gb = torch.autograd.grad(expected, values)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y)
        assert torch.isfinite(x).all()
    assert torch.equal(ga[2][1:], torch.zeros_like(ga[2][1:]))


def test_autocast_and_no_remaining_negative_are_finite():
    torch.manual_seed(6)
    values = [torch.randn(3, 8, requires_grad=True) for _ in range(3)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = excluded_infonce(*values, ~torch.eye(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
    assert loss == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in values)


def test_positive_diagonal_cannot_be_removed():
    with pytest.raises(ValueError):
        excluded_infonce(*[torch.ones(2, 3)]*3, torch.eye(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
