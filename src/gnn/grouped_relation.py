"""관계 키의 attention 가중치를 head 안의 작은 채널 묶음별로 만든다."""
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import softmax

from .model import DiffAttnConv
from .nonlinear_relation import NonlinearRelationConv


class GroupedRelationConv(NonlinearRelationConv):
    """동일 Q/K/V와 간선 편향에서 attention 공유 범위만 줄인다."""

    def __init__(self, reference: DiffAttnConv, groups_per_head: int = 8) -> None:
        super().__init__(reference)
        if groups_per_head < 1 or self.head_dim % groups_per_head:
            raise ValueError("묶음 수는 head 채널 수의 양의 약수여야 한다")
        self.groups_per_head = groups_per_head
        self.group_dim = self.head_dim // groups_per_head

    def transform_value(self, value: torch.Tensor) -> torch.Tensor:
        """값은 선형으로 유지하며 진단 도구의 공통 인터페이스를 제공한다."""
        return value

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_logit: Optional[torch.Tensor] = None,
        edge_value_gate: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, tuple]]:
        """같은 head의 간선 편향을 공유하며 묶음별로 이웃을 정규화한다."""
        if self.groups_per_head == 1:
            return super().forward(x, edge_index, edge_attr, edge_logit,
                                   edge_value_gate, return_attention_weights)
        src, tgt = edge_index
        center = x[tgt]
        relation = x[src] - center if self.value_source == "diff" else x[src]
        groups = self.groups_per_head

        def scores(center_in, relation_in, attr_in, logit_in):
            shape = (-1, self.heads, groups, self.group_dim)
            query = self.W_q(center_in).view(shape)
            key = self.W_k(relation_in).tanh().view(shape)
            score = (query * key).sum(-1) / self.group_dim ** .5
            if attr_in is not None and self.bias_mlp is not None:
                score = score + self.bias_mlp(attr_in).unsqueeze(-1)
            if logit_in is not None:
                score = score + logit_in.unsqueeze(-1)
            return score.flatten(1)

        if self.training and torch.is_grad_enabled():
            logits = checkpoint(scores, center, relation, edge_attr, edge_logit,
                                use_reentrant=False)
        else:
            logits = scores(center, relation, edge_attr, edge_logit)
        attention = self.dropout(softmax(logits, tgt, num_nodes=len(x)))
        self._attn_weights = attention.detach() if return_attention_weights else None
        value = self.W_v(relation).view(-1, self.heads, groups, self.group_dim)
        message = attention.view(-1, self.heads, groups, 1) * value
        if edge_value_gate is not None:
            message = message * edge_value_gate[:, :, None, None]
        output = torch.zeros(len(x), self.channels, dtype=message.dtype, device=x.device)
        output.scatter_add_(0, tgt[:, None].expand(-1, self.channels),
                            message.reshape(-1, self.channels))
        if return_attention_weights:
            return output, (edge_index, self._attn_weights)
        return output


def attach_grouped_relation(model: nn.Module, groups_per_head: int) -> None:
    """두 층의 행렬을 재사용해 묶음별 attention으로 교체한다."""
    layers = model.gnn.convs
    if len(layers) != 2 or not all(isinstance(x, DiffAttnConv) for x in layers):
        raise ValueError("원래 두 DiffAttnConv 층에만 적용한다")
    for i, layer in enumerate(layers):
        layers[i] = GroupedRelationConv(layer, groups_per_head)
