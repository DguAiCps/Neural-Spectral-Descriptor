"""관계 값의 선형·비선형·부호/크기 처리를 같은 채널 수로 비교한다."""
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import softmax

from .model import DiffAttnConv
from .nonlinear_relation import NonlinearRelationConv


class ValueRelationConv(NonlinearRelationConv):
    """tanh 관계 키를 유지하며 집계 전 값의 변환만 바꾼다."""

    def __init__(self, reference: DiffAttnConv, value_mode: str) -> None:
        super().__init__(reference)
        if value_mode not in ("linear", "tanh", "odd_even"):
            raise ValueError("지원하지 않는 값 변환")
        if self.head_dim % 2:
            raise ValueError("각 head의 채널 수는 짝수여야 한다")
        self.value_mode = value_mode

    def transform_value(self, projected: torch.Tensor) -> torch.Tensor:
        """head별 앞 절반은 부호, 뒤 절반은 제곱 크기를 보존한다."""
        if self.value_mode == "linear":
            return projected
        bounded = projected.tanh()
        if self.value_mode == "tanh":
            return bounded
        midpoint = self.head_dim // 2
        return torch.cat((bounded[..., :midpoint],
                          bounded[..., midpoint:].square()), dim=-1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_logit: Optional[torch.Tensor] = None,
        edge_value_gate: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, tuple]]:
        """기존 관계 키와 동일한 attention으로 변환된 값을 집계한다."""
        src, tgt = edge_index
        center = x[tgt]
        relation = x[src] - center if self.value_source == "diff" else x[src]

        def scores(center_in, relation_in, attr_in, logit_in):
            query = self.W_q(center_in).view(-1, self.heads, self.head_dim)
            key = torch.tanh(self.W_k(relation_in)).view(
                -1, self.heads, self.head_dim)
            result = torch.einsum("ehd,ehd->eh", query, key) / self.scale
            if attr_in is not None and self.bias_mlp is not None:
                result = result + self.bias_mlp(attr_in)
            if logit_in is not None:
                result = result + logit_in
            return result

        if self.training and torch.is_grad_enabled():
            logits = checkpoint(scores, center, relation, edge_attr, edge_logit,
                                use_reentrant=False)
        else:
            logits = scores(center, relation, edge_attr, edge_logit)
        attention = self.dropout(softmax(logits, tgt, num_nodes=len(x)))
        self._attn_weights = (attention.detach()
                              if return_attention_weights else None)
        value = self.W_v(relation).view(-1, self.heads, self.head_dim)
        message = attention.unsqueeze(-1) * self.transform_value(value)
        if edge_value_gate is not None:
            message = message * edge_value_gate.unsqueeze(-1)
        output = torch.zeros(len(x), self.channels, dtype=message.dtype,
                             device=message.device)
        output.scatter_add_(0, tgt[:, None].expand(-1, self.channels),
                            message.reshape(-1, self.channels))
        if return_attention_weights:
            return output, (edge_index, self._attn_weights)
        return output


def attach_value_relation(model: nn.Module, value_mode: str) -> None:
    """공통 초기 가중치를 유지한 채 두 attention 층의 값 처리를 바꾼다."""
    layers = model.gnn.convs
    if len(layers) != 2 or not all(isinstance(x, DiffAttnConv) for x in layers):
        raise ValueError("원래 두 DiffAttnConv 층에만 적용한다")
    for index, layer in enumerate(layers):
        layers[index] = ValueRelationConv(layer, value_mode)
