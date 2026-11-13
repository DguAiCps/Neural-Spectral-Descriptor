"""검증된 좌표계의 가까운 학습 쌍을 오답 분모에서 제외한다."""
from typing import Tuple

import torch
import torch.nn.functional as F


def exclusion_masks(
    positions: torch.Tensor, frames: torch.Tensor, triplets: torch.Tensor,
    radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """같은 좌표계·3D 반경 안의 배치 오답과 지정 오답을 표시한다.

    지정 양성인 대각선은 유지한다. 시간 간격은 적용하지 않는다.
    가까운 시간 이웃도 서로 다른 장소라는 오답 라벨을 주지 않는다.
    """
    if radius < 0 or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("3D 위치와 음수가 아닌 반경이 필요하다")
    a, p, n = triplets.T
    distance = torch.cdist(positions[a].double(), positions[p].double(),
                           compute_mode="donot_use_mm_for_euclid_dist")
    cross = (distance <= radius) & (frames[a, None] == frames[p][None, :])
    cross.fill_diagonal_(False)
    hard_distance = torch.linalg.vector_norm(positions[a].double()-positions[n].double(), dim=1)
    hard = (hard_distance <= radius) & (frames[a] == frames[n])
    return cross, hard


def excluded_infonce(
    anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor,
    cross_mask: torch.Tensor, hard_mask: torch.Tensor, temperature: float = .1,
) -> torch.Tensor:
    """표본과 양성 항을 유지하고 알려진 가까운 오답 항만 제외한다."""
    size = len(anchors)
    if cross_mask.shape != (size, size) or hard_mask.shape != (size,):
        raise ValueError("마스크 모양이 배치와 맞지 않는다")
    if cross_mask.dtype != torch.bool or hard_mask.dtype != torch.bool:
        raise ValueError("마스크는 bool이어야 한다")
    if bool(cross_mask.diagonal().any()) or temperature <= 0:
        raise ValueError("지정 양성을 유지하고 양수 온도를 써야 한다")
    a, p, n = [F.normalize(x, dim=1) for x in (anchors, positives, negatives)]
    cross = ((a @ p.T)/temperature).masked_fill(cross_mask, float("-inf"))
    hard = ((a*n).sum(1, keepdim=True)/temperature).masked_fill(hard_mask[:, None], float("-inf"))
    return F.cross_entropy(torch.cat((cross, hard), dim=1), torch.arange(size, device=a.device))
