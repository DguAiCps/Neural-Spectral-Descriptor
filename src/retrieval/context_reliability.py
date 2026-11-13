"""기존 문맥 차이로 시간 대응별 스펙트럼 증거의 신뢰도를 조절한다."""

from typing import Dict, Tuple

import torch


def reliability_score(
    embedding: torch.Tensor,
    plan: Dict[str, torch.Tensor],
    peaks: torch.Tensor,
    gamma: float,
    mode: str,
    descriptor_dim: int = 288,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """문맥 거리의 지수 감쇠를 감점 또는 정규화한 평균 비중으로 쓴다.

    peaks는 [질의, 방향2, 중심/이웃11, 원본spec/col/보정spec/col]이다.
    gamma=0은 기존 가중치·방향별 최대·채널 혼합을 그대로 돌려준다.
    반환값의 두 번째 항목은 경로별 신뢰 질량이며 확률 보장은 없다.
    """
    if gamma < 0 or mode not in ("attenuate", "reweight"):
        raise ValueError("지원하지 않는 신뢰도 규칙")
    if peaks.shape[:-1] != plan["weight"].shape or peaks.shape[-1] != 4:
        raise ValueError("스펙트럼과 대응 계획의 모양이 다르다")
    raw_norm = embedding[:, :descriptor_dim].norm(dim=-1, keepdim=True)
    context = embedding[:, descriptor_dim:] / raw_norm.clamp_min(1e-12)
    context = torch.where(raw_norm > 1e-12, context, torch.zeros_like(context))
    distance = 0.5 * (context[plan["query"]] - context[plan["candidate"]]).square().sum(
        -1
    )
    trust = torch.exp(-float(gamma) * distance)
    weight = plan["weight"] * trust
    mass = weight.sum(-1)
    if mode == "reweight":
        weight = weight / mass.clamp_min(1e-12).unsqueeze(-1)
    channel = (weight.unsqueeze(-1) * peaks).sum(-2).max(dim=1).values
    score = 0.5 * (
        30 * (0.75 * channel[:, 0] + 0.25 * channel[:, 2])
        + 0.75 * channel[:, 1]
        + 0.25 * channel[:, 3]
    )
    return score, mass
