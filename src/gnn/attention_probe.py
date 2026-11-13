"""가중치를 보존한 채 attention 점수의 두 성분을 분리하는 추론 진단."""
from typing import List

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


def _zero_output(module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
    if module.training:
        raise RuntimeError("attention 성분 제거는 추론 진단에서만 쓴다")
    return torch.zeros_like(output)


def attach_attention_probe(model: nn.Module, mode: str) -> List[RemovableHandle]:
    """full/content/geometry/uniform 점수 진단을 설치하고 제거용 핸들을 반환한다."""
    if mode not in ("full", "content", "geometry", "uniform"):
        raise ValueError("지원하지 않는 attention 진단")
    gnn = model.gnn
    if len(gnn.convs) != 2:
        raise ValueError("두 attention 층을 전제로 한 진단이다")
    for name in ("edge_gate", "phase_edge_bias", "phase_alignment_bias", "phase_coherence_bias"):
        if getattr(gnn, name, None) is not None:
            raise ValueError("별도 간선 점수가 꺼진 모델에만 적용한다")
    if not all(conv.bias_mlp is not None for conv in gnn.convs):
        raise ValueError("두 층 모두 간선 편향이 있어야 한다")
    handles = []
    for conv in gnn.convs:
        if mode in ("geometry", "uniform"):
            handles.append(conv.W_q.register_forward_hook(_zero_output))
        if mode in ("content", "uniform"):
            handles.append(conv.bias_mlp.register_forward_hook(_zero_output))
    return handles
