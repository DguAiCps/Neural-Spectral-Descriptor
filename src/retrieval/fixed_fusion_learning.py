"""추론의 고정 융합 규칙을 유지하며 검색 표현만 학습한다."""
from typing import Sequence, Tuple

import torch

from .candidate_learning import masked_minmax


def fixed_fusion_scores(key: torch.Tensor, static: torch.Tensor,
                        valid: torch.Tensor, weights: Sequence[float],
                        use_static: bool = True) -> torch.Tensor:
    """기존 네 점수의 질의별 정규화와 고정 가중합을 미분 가능하게 계산한다."""
    score = float(weights[0]) * masked_minmax(key, valid)
    if use_static:
        for index in range(3):
            score = score + float(weights[index + 1]) * masked_minmax(
                static[..., index].detach(), valid)
    return score


def positive_ranking_loss(scores: torch.Tensor, correct: torch.Tensor,
                          valid: torch.Tensor, temperature: float = .1
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """후보 안에 정답·오답이 모두 있는 행에서 정답 확률 합을 높인다.

    정답 없는 행에는 거절 점수를 새로 만들지 않고 기울기 0을 준다.
    반환값은 행별 손실과 실제 학습 신호가 있는 행의 마스크다.
    """
    positive = correct & valid
    active = positive.any(1) & (valid & ~correct).any(1)
    # 제외 행에도 유한한 logsumexp를 계산해 NaN 역전파를 막는다.
    safe_positive = positive | ((~active)[:, None] & valid)
    logits = scores / temperature
    denominator = torch.logsumexp(logits.masked_fill(~valid, -float("inf")), 1)
    numerator = torch.logsumexp(logits.masked_fill(~safe_positive, -float("inf")), 1)
    return torch.where(active, denominator - numerator, scores.sum(1) * 0), active
