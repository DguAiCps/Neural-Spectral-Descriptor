"""추론 점수는 유지하고 학습 시 후보 점수 폭의 기울기만 분리한다."""
from typing import Dict, Sequence

import torch

from .candidate_learning import masked_minmax
from .fixed_fusion_learning import positive_ranking_loss
from .fusion_guard import fusion_guard_terms


def fixed_width_minmax(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """현재 폭으로 정규화하되 그 폭은 역전파에서 상수로 취급한다.

    순전파는 기존 min-max와 같다. 실제 min-max 함수의 정확한 미분이 아닌
    대리 기울기다. 폭은 매 순전파 다시 계산하며 저장하거나 추론에서 고정하지 않는다.
    """
    if not bool(valid.any(1).all()):
        raise ValueError('질의마다 유효 후보가 필요하다')
    lo = values.masked_fill(~valid, float('inf')).amin(1, keepdim=True)
    hi = values.masked_fill(~valid, -float('inf')).amax(1, keepdim=True)
    width = (hi - lo).clamp_min(1e-8).detach()
    return ((values - lo) / width).masked_fill(~valid, 0.)


def fixed_width_terms(key: torch.Tensor, teacher_key: torch.Tensor,
                      static: torch.Tensor, valid: torch.Tensor,
                      correct: torch.Tensor, weights: Sequence[float],
                      temperature: float = .1, margin_cap: float = .1
                      ) -> Dict[str, torch.Tensor]:
    """기존 두 손실·필터·보호 기준을 유지하고 현재 키의 분모 미분만 끊는다."""
    terms = fusion_guard_terms(key, teacher_key, static, valid, correct,
                               weights, temperature, margin_cap)
    scores = float(weights[0]) * fixed_width_minmax(key, valid)
    for i in range(3):
        scores = scores + float(weights[i + 1]) * masked_minmax(static[..., i].detach(), valid)
    rank, active = positive_ranking_loss(scores / float(weights[0]), correct, valid, temperature)
    positive = (correct & valid) | ((~active)[:, None] & valid)
    negative = (~correct & valid) | ((~active)[:, None] & valid)
    margin = (scores.masked_fill(~positive, -float('inf')).amax(1)
              - scores.masked_fill(~negative, -float('inf')).amax(1)) / float(weights[0])
    guard = torch.where(terms['protected'],
        ((terms['target_margin'] - margin) / temperature).relu().square(), scores.sum(1) * 0)
    terms.update(rank=rank, guard=guard, margin=margin)
    return terms
