"""실제 후보의 융합 순위에서 불가능한 교정과 기존 정답 보호를 분리한다."""
from typing import Dict, Sequence

import torch

from .fixed_fusion_learning import fixed_fusion_scores, positive_ranking_loss


def fusion_guard_terms(
    key: torch.Tensor,
    teacher_key: torch.Tensor,
    static: torch.Tensor,
    valid: torch.Tensor,
    correct: torch.Tensor,
    weights: Sequence[float],
    temperature: float = 0.1,
    margin_cap: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """질의별 순위 손실과 기준 모델 정답의 한쪽 방향 보호 손실을 계산한다.

    admissible은 키 점수를 자유롭게 [0,1]에 둘 수 있어도 불가능한 행을
    제외하는 필요조건이다. 현재 인코더가 그 순위를 실현한다는 보장은 아니다.
    teacher_key와 보조 점수는 고정하며 min-max의 분모까지 미분한다.
    """
    if temperature <= 0 or margin_cap <= 0 or float(weights[0]) <= 0:
        raise ValueError("온도·보호 상한·키 가중치는 양수여야 한다")
    if not bool(valid.any(1).all()):
        raise ValueError("질의마다 유효 후보가 필요하다")
    score = fixed_fusion_scores(key, static, valid, weights)
    reference = fixed_fusion_scores(teacher_key.detach(), static, valid, weights).detach()
    rank, active = positive_ranking_loss(score / float(weights[0]), correct, valid, temperature)
    positive, negative = correct & valid, ~correct & valid
    safe_positive = positive | ((~active)[:, None] & valid)
    safe_negative = negative | ((~active)[:, None] & valid)

    def margin(values: torch.Tensor) -> torch.Tensor:
        return (values.masked_fill(~safe_positive, -float('inf')).amax(1)
                - values.masked_fill(~safe_negative, -float('inf')).amax(1))

    static_score = fixed_fusion_scores(torch.zeros_like(key), static, valid, weights)
    upper_bound = margin(static_score) + float(weights[0])
    admissible = active & (upper_bound > 0)
    current_margin = margin(score) / float(weights[0])
    reference_margin = margin(reference) / float(weights[0])
    protected = active & (reference_margin > 0)
    target = reference_margin.clamp(min=0., max=margin_cap)
    guard = torch.where(protected, ((target - current_margin) / temperature).relu().square(),
                        score.sum(1) * 0)
    return {'rank': rank, 'guard': guard, 'active': active, 'admissible': admissible,
            'protected': protected, 'margin': current_margin, 'target_margin': target,
            'reference_margin': reference_margin, 'static_upper_bound': upper_bound.detach()}


def fusion_guard_loss(terms: Dict[str, torch.Tensor], filter_impossible: bool,
                      protect_correct: bool) -> torch.Tensor:
    """동일 배치 분모를 유지하며 두 제약만 켜거나 끈다."""
    rank = terms['rank'] * (terms['admissible'] if filter_impossible else terms['active'])
    return (rank + (terms['guard'] if protect_correct else 0)).mean()
