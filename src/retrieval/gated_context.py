"""정규화된 검색 키에서 게이트 강도를 보존하는 문맥 판정."""
from typing import Dict

import torch


def gated_context_score(embedding: torch.Tensor, plan: Dict[str, torch.Tensor],
                        descriptor_dim: int = 288) -> torch.Tensor:
    """원 특징 노름으로 전체 정규화만 풀어 alpha 곱을 보존한다.

    입력 키는 normalize([unit_raw, alpha * unit_context])다. 따라서
    문맥 블록을 원 특징 블록의 노름으로 나누면 alpha * unit_context다.
    원 특징이 영벡터인 퇴화 입력은 문맥 증거도 0으로 처리한다.
    """
    raw_norm = embedding[:, :descriptor_dim].norm(dim=-1, keepdim=True)
    context = embedding[:, descriptor_dim:] / raw_norm.clamp_min(1e-12)
    context = torch.where(raw_norm > 1e-12, context, torch.zeros_like(context))
    dots = (context[plan["query"]] * context[plan["candidate"]]).sum(-1)
    return (dots * plan["weight"]).sum(-1).max(-1).values
