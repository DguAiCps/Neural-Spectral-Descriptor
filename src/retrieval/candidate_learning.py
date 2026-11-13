"""실제 검색 후보의 순위와 후보 없음 판단을 함께 학습한다."""
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

FEATURE_NAMES = ("raw_cos", "context_cos", "key_cos", "aux_cos", "c_spec", "c_col",
                 "key_minmax", "aux_minmax", "spec_minmax", "col_minmax")


def masked_minmax(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """패딩을 제외하고 한 질의 안에서만 점수를 정규화한다."""
    lo = values.masked_fill(~valid, float("inf")).amin(dim=1, keepdim=True)
    hi = values.masked_fill(~valid, -float("inf")).amax(dim=1, keepdim=True)
    return ((values - lo) / (hi - lo).clamp_min(1e-8)).masked_fill(~valid, 0.)


def pair_features(embedding: torch.Tensor, queries: torch.Tensor,
                  candidates: torch.Tensor, static: torch.Tensor,
                  valid: torch.Tensor) -> torch.Tensor:
    """정답·시퀀스 길이 없이 현재 키와 이미 계산한 매칭 점수만 조립한다."""
    query = embedding[queries, None, :]
    database = embedding[candidates.clamp_min(0)]
    raw = (F.normalize(query[..., :288], dim=-1)
           * F.normalize(database[..., :288], dim=-1)).sum(-1)
    context = (F.normalize(query[..., 288:], dim=-1)
               * F.normalize(database[..., 288:], dim=-1)).sum(-1)
    key = (query * database).sum(-1)
    aux, spectrum, column = static.unbind(-1)
    channels = [raw, context, key, aux, spectrum, column]
    channels.extend(masked_minmax(value, valid) for value in (key, aux, spectrum, column))
    return torch.stack(channels, dim=-1).masked_fill(~valid[..., None], 0.)


class CandidateScorer(nn.Module):
    """후보별 같은 장소 점수를 출력하는 작은 판정기다."""

    def __init__(self, hidden: Tuple[int, int] = (32, 16)) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden[0]), nn.ReLU(),
                                    nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                                    nn.Linear(hidden[1], 1))
        nn.init.constant_(self.layers[-1].bias, -3.)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """질의·후보별 수락 점수를 반환한다."""
        return self.layers(features).squeeze(-1)


def candidate_loss(logits: torch.Tensor, correct: torch.Tensor,
                   valid: torch.Tensor, pair_weight: float = .5) -> torch.Tensor:
    """여러 정답 또는 후보 없음의 목록 손실과 쌍 판정 손실을 더한다.

    후보 없음의 logit은 0으로 고정한다. 정답이 여럿이면 정답 확률 합을
    최대화하며, 같은 장소를 배치의 다른 오답으로 취급하지 않는다.
    """
    if not bool(valid.any(dim=1).all()):
        raise ValueError("각 질의에는 유효 후보가 하나 이상 있어야 한다")
    positive = correct & valid
    negative = ~correct & valid
    has_positive = positive.any(dim=1)
    denominator = torch.logsumexp(torch.cat([
        logits.masked_fill(~valid, -float("inf")), torch.zeros_like(logits[:, :1])], dim=1), dim=1)
    # 정답 없는 행도 미분 가능한 유한한 값을 거쳐 logsumexp를 계산한다.
    numerator_terms = logits.masked_fill(~positive, -float("inf"))
    numerator_terms = torch.cat([numerator_terms,
                                 torch.where(has_positive, -float("inf"), 0.)[:, None]], dim=1)
    ranking = denominator - torch.logsumexp(numerator_terms, dim=1)
    pos_bce = (F.softplus(-logits) * positive).sum(1) / positive.sum(1).clamp_min(1)
    neg_bce = (F.softplus(logits) * negative).sum(1) / negative.sum(1).clamp_min(1)
    groups = has_positive.float() + negative.any(dim=1).float()
    pair = (pos_bce + neg_bce) / groups.clamp_min(1)
    return (ranking + pair_weight * pair).mean()
