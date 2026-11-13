"""방향 쌍의 좌표계·시간 조건과 거짓 오답 제거를 확인한다."""

import numpy as np
import torch

from gnn.pair_sampling import (
    masked_infonce, positive_pools, same_place_mask, sample_triplets,
)
from gnn.trainer import InfoNCELoss


def test_opposite_pairs_respect_coordinate_frames_and_time():
    track = np.column_stack((np.arange(50), np.zeros((50, 2)))).astype(float)
    positions = np.concatenate((track, track[::-1], track[::-1]))
    sequences = np.repeat([0, 1, 2], 50)
    frames = np.repeat([0, 0, 1], 50)
    intra, opposite = positive_pools(positions, sequences, frames)
    assert not any(len(p) for p in intra)
    assert len(opposite[20]) > 0
    assert np.all(sequences[opposite[20]] == 1)
    assert len(opposite[0]) == 0  # 과거 이동 방향이 아직 정해지지 않는다.
    assert not any(len(p) for p in opposite[100:])


def test_sampling_keeps_budget_and_uses_negative_of_replaced_anchor():
    intra = [np.array([1]), np.array([0]), np.array([], dtype=int)]
    opposite = [np.array([], dtype=int), np.array([], dtype=int), np.array([0])]
    negatives = np.array([8, 9, 10])
    baseline = sample_triplets(intra, opposite, negatives, 7)
    changed = sample_triplets(intra, opposite, negatives, 7, 0.5)
    assert baseline.shape == changed.shape == (6, 3)
    assert changed.dtype == np.int64
    assert (changed[:, 0] == 2).sum() == 3
    np.testing.assert_array_equal(changed[:, 2], negatives[changed[:, 0]])
    np.testing.assert_array_equal(changed, sample_triplets(intra, opposite, negatives, 7, 0.5))


def test_mask_excludes_known_places_only_and_has_finite_gradients():
    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    frame = torch.tensor([0, 0, 1])
    mask = same_place_mask(pos, pos, frame, frame)
    assert mask.tolist() == [[False, True, False], [True, False, False], [False, False, False]]
    torch.manual_seed(1)
    a, p, n = [torch.randn(3, 8, requires_grad=True) for _ in range(3)]
    plain = InfoNCELoss(0.1)(a, p, n)
    torch.testing.assert_close(plain, masked_infonce(a, p, n, torch.zeros_like(mask)))
    changed = masked_infonce(a, p, n, mask)
    assert changed <= plain
    changed.backward()
    assert all(torch.isfinite(x.grad).all() for x in (a, p, n))
