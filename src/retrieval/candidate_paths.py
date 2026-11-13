"""제한된 시간 창의 장소 후보를 이동거리 제약으로 함께 선택한다."""
from typing import Dict

import numpy as np
import torch


def transition_cost(previous: torch.Tensor, current: torch.Tensor,
                    query_step: torch.Tensor, direction: int,
                    weight: float = .25, null_cost: float = .15) -> torch.Tensor:
    """누적거리의 부호 있는 증가량과 대응 없음의 전환 비용을 계산한다."""
    delta = direction * (current[:, None, :] - previous[:, :, None])
    step = query_step[:, None, None]
    cost = weight * torch.clamp(torch.abs(delta-step) / torch.clamp(step, min=.8), max=3.)
    invalid = (delta < -.8) | (delta > 3.*step+3.2)
    cost = cost.masked_fill(invalid, torch.inf)
    # 마지막 상태는 대응 없음이다. 이 상태에서는 이동 제약을 끊는다.
    cost[:, -1, :] = null_cost
    cost[:, :, -1] = null_cost
    cost[:, -1, -1] = 0.
    return cost


@torch.no_grad()
def select_paths(indices: np.ndarray, unary: np.ndarray, travel: np.ndarray,
                 queries: np.ndarray, radius: int = 5, weight: float = .25,
                 null_unary: float = .5, null_cost: float = .15,
                 device: str = "cuda", batch_size: int = 256) -> Dict[str, np.ndarray]:
    """정·역방향 사슬의 중심 최대 주변 점수로 후보와 대응 없음을 선택한다.

    indices/unary는 프레임별 상위 K 후보와 고정 융합 점수다.
    GT 좌표나 정답은 받지 않는다. 각 창은 독립적이며 이전 창의 출력을
    입력으로 재사용하지 않아 관측 의존성이 창 밖으로 늘어나지 않는다.
    """
    ids = np.asarray(indices, dtype=np.int64)
    values = np.asarray(unary, dtype=np.float64)
    path = np.asarray(travel, dtype=np.float64)
    q = np.asarray(queries, dtype=np.int64)
    if ids.ndim != 2 or ids.shape != values.shape or len(ids) != len(path):
        raise ValueError("후보·점수·이동거리 모양이 맞지 않는다")
    if (np.diff(path) < 0).any() or not np.isfinite(path).all():
        raise ValueError("누적 이동거리는 유한하고 감소하지 않아야 한다")
    if ((q-radius < 0) | (q+radius >= len(path))).any():
        raise ValueError("질의 창이 관측 범위 밖이다")
    if not np.isfinite(values[ids >= 0]).all():
        raise ValueError("유효 후보 점수는 유한해야 한다")
    k = ids.shape[1]
    outputs = {key: [] for key in ("forced", "joint", "direction", "margin", "objective")}
    for start in range(0, len(q), batch_size):
        query = q[start:start+batch_size]
        frames = query[:, None] + np.arange(-radius, radius+1)[None]
        ii = ids[frames]
        uu = np.where(ii >= 0, values[frames], -np.inf)
        uu = np.concatenate([uu, np.full((*uu.shape[:2], 1), null_unary)], axis=-1)
        pp = path[np.maximum(ii, 0)]
        pp = np.concatenate([pp, np.zeros((*pp.shape[:2], 1))], axis=-1)
        u = torch.as_tensor(uu, dtype=torch.float64, device=device)
        p = torch.as_tensor(pp, dtype=torch.float64, device=device)
        steps = torch.as_tensor(np.diff(path[frames], axis=1), device=device)
        center_scores = []
        for direction in (1, -1):
            left = u[:, 0]
            for t in range(1, radius+1):
                cost = transition_cost(p[:, t-1], p[:, t], steps[:, t-1],
                                       direction, weight, null_cost)
                left = u[:, t] + (left[:, :, None]-cost).max(dim=1).values
            right = u[:, -1]
            for t in range(2*radius-1, radius-1, -1):
                cost = transition_cost(p[:, t], p[:, t+1], steps[:, t],
                                       direction, weight, null_cost)
                right = u[:, t] + (right[:, None, :]-cost).max(dim=2).values
            # 유효하지 않은 중심은 무한대끼리의 뺄셈을 피한다.
            center = left+right-torch.where(torch.isfinite(u[:, radius]), u[:, radius], 0.)
            center_scores.append(center)
        all_scores = torch.stack(center_scores, dim=1)
        # 후보 동점은 융합 순서, 방향 동점은 정방향으로 고정한다.
        by_candidate, directions = all_scores.max(dim=1)
        forced_slot = by_candidate[:, :k].argmax(dim=1)
        chosen = by_candidate.argmax(dim=1)
        row = torch.arange(len(query), device=device)
        forced_np = forced_slot.cpu().numpy()
        chosen_np = chosen.cpu().numpy()
        forced = ids[query, forced_np]
        joint = np.where(chosen_np == k, -1, ids[query, np.minimum(chosen_np, k-1)])
        outputs["forced"].append(forced)
        outputs["joint"].append(joint)
        outputs["direction"].append(np.where(directions[row, forced_slot].cpu().numpy() == 0, 1, -1))
        outputs["margin"].append((by_candidate[row, forced_slot]-by_candidate[:, -1]).cpu().numpy())
        outputs["objective"].append(by_candidate[row, chosen].cpu().numpy())
    return {key: np.concatenate(value) for key, value in outputs.items()}
