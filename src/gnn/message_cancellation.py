"""시간 이웃 메시지의 좌표별 부호 상쇄를 분해하는 진단 함수."""
from typing import Dict

import torch


def cancellation_statistics(messages: torch.Tensor, edge_index: torch.Tensor,
                            num_nodes: int) -> Dict[str, torch.Tensor]:
    """L1 질량 손실을 같은 시간 방향 내부와 과거/미래 사이로 분해한다.

    messages는 주의 가중치까지 곱한 (간선, 채널) 텐서다. 결과는
    노드별 진단값이며 정보 손실률이나 성능 기여율을 뜻하지 않는다.
    """
    if messages.ndim != 2 or edge_index.shape != (2, len(messages)):
        raise ValueError("메시지와 간선 모양이 맞지 않는다")
    src, tgt = edge_index
    if bool((src == tgt).any()):
        raise ValueError("자기 간선 없는 시간 그래프를 진단한다")
    total = messages.new_zeros((num_nodes, messages.shape[1]))
    total.index_add_(0, tgt, messages)
    past = torch.zeros_like(total)
    before = src < tgt
    past.index_add_(0, tgt[before], messages[before])
    future = torch.zeros_like(total)
    future.index_add_(0, tgt[~before], messages[~before])
    mass = messages.new_zeros(num_nodes)
    mass.index_add_(0, tgt, messages.abs().sum(-1))
    surviving = total.abs().sum(-1)
    sides = past.abs().sum(-1) + future.abs().sum(-1)
    valid = mass > 1e-12
    safe = mass.clamp_min(1e-12)
    before_norm = torch.linalg.vector_norm(past, dim=-1)
    after_norm = torch.linalg.vector_norm(future, dim=-1)
    both = (before_norm > 1e-12) & (after_norm > 1e-12)
    cosine = (past * future).sum(-1) / (before_norm * after_norm).clamp_min(1e-12)
    return {
        "total": total,
        "mass": mass,
        "surviving": surviving,
        "cancellation": torch.where(valid, 1 - surviving / safe, 0.).clamp(0, 1),
        "within": torch.where(valid, (mass - sides) / safe, 0.).clamp(0, 1),
        "between": torch.where(valid, (sides - surviving) / safe, 0.).clamp(0, 1),
        "side_cosine": torch.where(both, cosine, 0.).clamp(-1, 1),
        "valid": valid,
        "both_sides": both,
    }
