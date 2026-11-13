"""같은 장소·접근 방향에 따른 학습 쌍 실험용 함수다."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


def past_heading(positions: np.ndarray, sequence_ids: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """시퀀스 경계를 넘지 않는 과거 5프레임의 3차원 이동 방향을 구한다."""
    heading = np.zeros_like(positions, dtype=np.float64)
    valid = np.zeros(len(positions), dtype=bool)
    for seq in np.unique(sequence_ids):
        indices = np.flatnonzero(sequence_ids == seq)
        delta = positions[indices[5:]] - positions[indices[:-5]]
        length = np.linalg.norm(delta, axis=1)
        heading[indices[5:]] = delta / np.maximum(length[:, None], 1e-12)
        valid[indices[5:]] = length >= 1.0
    return heading, valid


def positive_pools(positions: np.ndarray, sequence_ids: np.ndarray,
                   frame_ids: np.ndarray, radius: float = 5.0,
                   temporal_gap: int = 30
                   ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """기존 시퀀스 내 양성과 좌표계가 같은 반대 방향 양성을 구한다.

    frame_ids는 검증된 좌표계 그룹이다. 같은 숫자라는 이유만으로 서로
    다른 사이트의 지역 좌표를 비교하지 않도록 호출자가 명시해야 한다.
    """
    heading, valid = past_heading(positions, sequence_ids)
    intra = [np.empty(0, dtype=np.int64) for _ in positions]
    opposite = [np.empty(0, dtype=np.int64) for _ in positions]
    local_index = np.zeros(len(positions), dtype=np.int64)
    for seq in np.unique(sequence_ids):
        ix = np.flatnonzero(sequence_ids == seq)
        local_index[ix] = np.arange(len(ix))
    for frame in np.unique(frame_ids):
        ix = np.flatnonzero(frame_ids == frame)
        tree = cKDTree(positions[ix])
        for local, candidates in enumerate(tree.query_ball_tree(tree, radius)):
            anchor = ix[local]
            pos = ix[np.asarray(candidates, dtype=np.int64)]
            same = sequence_ids[pos] == sequence_ids[anchor]
            allowed = ~same | (np.abs(local_index[pos] - local_index[anchor]) >= temporal_gap)
            pos, same = pos[allowed], same[allowed]
            intra[anchor] = pos[same]
            dot = heading[pos] @ heading[anchor]
            opposite[anchor] = pos[valid[anchor] & valid[pos] & (dot <= -0.5)]
    return intra, opposite


def sample_triplets(intra: List[np.ndarray], opposite: List[np.ndarray],
                    hard_negative: np.ndarray, seed: int,
                    opposite_fraction: float = 0.0) -> np.ndarray:
    """기존 3쌍/앵커 예산 중 지정 비율을 반대 방향 쌍으로 교체한다."""
    if not 0 <= opposite_fraction <= 1:
        raise ValueError("opposite_fraction은 0~1이어야 한다")
    rng = np.random.default_rng(seed)
    anchors = np.repeat([i for i, p in enumerate(intra)
                         if len(p) and hard_negative[i] >= 0], 3)
    positives = np.asarray([rng.choice(intra[i]) for i in anchors], dtype=np.int64)
    count = int(round(len(anchors) * opposite_fraction))
    if count:
        eligible = np.asarray([i for i, p in enumerate(opposite)
                               if len(p) and hard_negative[i] >= 0])
        if not len(eligible):
            raise ValueError("반대 방향 양성 쌍이 없다")
        replace = rng.choice(len(anchors), count, replace=False)
        anchors[replace] = rng.choice(eligible, count, replace=True)
        positives[replace] = [rng.choice(opposite[i]) for i in anchors[replace]]
    return np.stack((anchors, positives, hard_negative[anchors]), axis=1).astype(np.int64)


def same_place_mask(anchor_positions: torch.Tensor, positive_positions: torch.Tensor,
                    anchor_frames: torch.Tensor, positive_frames: torch.Tensor,
                    radius: float = 5.0) -> torch.Tensor:
    """검증된 공통 좌표계에서 5m 이내인 배치 오답 항을 제외한다."""
    distance = torch.cdist(anchor_positions.double(), positive_positions.double(),
                           compute_mode="donot_use_mm_for_euclid_dist")
    mask = (distance <= radius) & (anchor_frames[:, None] == positive_frames[None, :])
    mask.fill_diagonal_(False)
    return mask


def masked_infonce(anchors: torch.Tensor, positives: torch.Tensor,
                   negatives: torch.Tensor, mask: torch.Tensor,
                   temperature: float = 0.1) -> torch.Tensor:
    """단일 정답 InfoNCE에서 알려진 같은 장소의 오답 항만 제외한다."""
    a, p, n = [F.normalize(x, dim=1) for x in (anchors, positives, negatives)]
    logits = (a @ p.T) / temperature
    logits = logits.masked_fill(mask, float("-inf"))
    hard = (a * n).sum(dim=1, keepdim=True) / temperature
    return F.cross_entropy(torch.cat((logits, hard), dim=1),
                           torch.arange(len(a), device=a.device))
