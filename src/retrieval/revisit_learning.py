"""실제 재방문 정답 각각을 먼 오답과 구분하는 보조 학습 손실."""

from typing import Dict

import torch
import torch.nn.functional as F


def revisit_terms(
    embedding: torch.Tensor,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    context_only: bool = False,
    temperature: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """정답마다 같은 오답 집합과 비교하고 정답 수와 무관하게 질의를 평균한다."""
    if temperature <= 0:
        raise ValueError("온도는 양수여야 한다")
    pv, nv = positives >= 0, negatives >= 0
    if not bool(pv.any(1).all() and nv.any(1).all()):
        raise ValueError("각 질의에 정답과 오답이 하나 이상 있어야 한다")
    value = F.normalize(embedding[:, 288:] if context_only else embedding, dim=1)
    query = value[anchors, None]
    positive = (query * value[positives.clamp_min(0)]).sum(-1)
    negative = (query * value[negatives.clamp_min(0)]).sum(-1)
    negative_logsum = (negative / temperature).masked_fill(~nv, -torch.inf).logsumexp(1)
    pair_loss = F.softplus(negative_logsum[:, None] - positive / temperature)
    loss = pair_loss.masked_fill(~pv, 0).sum(1) / pv.sum(1)
    best_negative = negative.masked_fill(~nv, -torch.inf).max(1).values
    return {
        "loss": loss,
        "mean_margin": (
            (positive - best_negative[:, None]).masked_fill(~pv, 0).sum(1) / pv.sum(1)
        ),
        "all_positive_win": ((positive > best_negative[:, None]) | ~pv).all(1),
    }
