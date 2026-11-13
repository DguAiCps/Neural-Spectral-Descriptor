"""기존 특징·기하 점수의 결합만 바꾸는 시간 attention 대조."""
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import softmax

from .model import DiffAttnConv
from .value_relation import ValueRelationConv


def incoming_mean(value: torch.Tensor, target: torch.Tensor,
                  num_nodes: int) -> torch.Tensor:
    """같은 중심으로 들어오는 간선 평균을 각 간선에 돌려준다."""
    work = value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    total = torch.zeros(num_nodes, value.shape[1], device=value.device, dtype=work.dtype)
    total.index_add_(0, target, work)
    count = torch.bincount(target, minlength=num_nodes).clamp_min(1)
    return (total / count[:, None])[target].to(value.dtype)


def combine_scores(content: torch.Tensor, geometry: torch.Tensor,
                   target: torch.Tensor, num_nodes: int, mode: str) -> torch.Tensor:
    """이웃별 기하 결합과 중심별 온도 효과를 같은 용량에서 분리한다."""
    result = content + geometry
    if mode == "additive":
        return result
    centered_content = content - incoming_mean(content, target, num_nodes)
    mean_geometry = incoming_mean(geometry, target, num_nodes)
    if mode == "interaction":
        modulation = geometry - mean_geometry
    elif mode == "temperature":
        modulation = mean_geometry
    else:
        raise ValueError("지원하지 않는 기하 점수 결합")
    return result + centered_content * modulation.tanh()


class GeometryInteractionConv(ValueRelationConv):
    """Q/K/V·기하 MLP를 그대로 쓰고 점수의 상호작용만 추가한다."""

    def __init__(self, reference: DiffAttnConv, mode: str) -> None:
        super().__init__(reference, "tanh")
        if mode not in ("additive", "interaction", "temperature"):
            raise ValueError("지원하지 않는 기하 점수 결합")
        self.geometry_mode = mode
        self.geometry_enabled = True

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_logit: Optional[torch.Tensor] = None,
        edge_value_gate: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, tuple]]:
        """시간 간선·차이 값·출력 형태를 유지하며 점수만 결합한다."""
        src, tgt = edge_index
        center = x[tgt]
        relation = x[src] - center if self.value_source == "diff" else x[src]

        def scores(center_in, relation_in, attr_in, logit_in):
            query = self.W_q(center_in).view(-1, self.heads, self.head_dim)
            key = self.W_k(relation_in).tanh().view(-1, self.heads, self.head_dim)
            content = torch.einsum("ehd,ehd->eh", query, key) / self.scale
            if attr_in is not None and self.bias_mlp is not None and self.geometry_enabled:
                result = combine_scores(content, self.bias_mlp(attr_in), tgt,
                                        len(x), self.geometry_mode)
            else:
                result = content
            if logit_in is not None:
                result = result + logit_in
            return result

        if self.training and torch.is_grad_enabled():
            logits = checkpoint(scores, center, relation, edge_attr, edge_logit,
                                use_reentrant=False)
        else:
            logits = scores(center, relation, edge_attr, edge_logit)
        attention = self.dropout(softmax(logits, tgt, num_nodes=len(x)))
        self._attn_weights = attention.detach() if return_attention_weights else None
        value = self.W_v(relation).view(-1, self.heads, self.head_dim).tanh()
        message = attention.unsqueeze(-1) * value
        if edge_value_gate is not None:
            message = message * edge_value_gate.unsqueeze(-1)
        output = torch.zeros(len(x), self.channels, dtype=message.dtype, device=message.device)
        output.scatter_add_(0, tgt[:, None].expand(-1, self.channels),
                            message.reshape(-1, self.channels))
        if return_attention_weights:
            return output, (edge_index, self._attn_weights)
        return output


def attach_geometry_interaction(model: nn.Module, mode: str) -> None:
    """학습 매개변수를 추가하지 않고 두 층의 점수 결합을 교체한다."""
    layers = model.gnn.convs
    if len(layers) != 2 or not all(isinstance(layer, DiffAttnConv) for layer in layers):
        raise ValueError("원래 두 DiffAttnConv 층에만 적용한다")
    for index, layer in enumerate(layers):
        layers[index] = GeometryInteractionConv(layer, mode)
