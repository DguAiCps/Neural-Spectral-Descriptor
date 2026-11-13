"""같은 Q/K/V에서 관계 키의 비선형성만 분리하는 실험 연산."""
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import softmax

from .model import DiffAttnConv


class NonlinearRelationConv(nn.Module):
    """K=tanh(Wk·관계), V=Wv·관계로 고정한 매개변수 추가 없는 대조.

    관계는 이웃-중심 또는 이웃 자체다. 비선형 차이 연산 자체의
    신규성을 주장하는 모듈이 아니며 두 관계의 효과를 분리한다.
    """

    def __init__(self, reference: DiffAttnConv) -> None:
        super().__init__()
        if reference.value_source not in ("diff", "standard"):
            raise ValueError("차이 또는 표준 관계만 비교한다")
        for name in ("W_q", "W_k", "W_v", "bias_mlp", "dropout"):
            setattr(self, name, getattr(reference, name))
        for name in ("channels", "heads", "head_dim", "scale", "value_source"):
            setattr(self, name, getattr(reference, name))
        self._attn_weights = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                edge_logit: Optional[torch.Tensor] = None,
                edge_value_gate: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, tuple]]:
        """차이 키의 tanh를 제외하면 기준과 같은 집계·드롭아웃이다."""
        src, tgt = edge_index
        center = x[tgt]
        relation = x[src]-center if self.value_source == "diff" else x[src]

        def scores(center_in, relation_in, attr_in, logit_in):
            query = self.W_q(center_in).view(-1, self.heads, self.head_dim)
            key = torch.tanh(self.W_k(relation_in)).view(-1, self.heads, self.head_dim)
            result = torch.einsum("ehd,ehd->eh", query, key)/self.scale
            if attr_in is not None and self.bias_mlp is not None:
                result = result+self.bias_mlp(attr_in)
            if logit_in is not None:
                result = result+logit_in
            return result

        if self.training and torch.is_grad_enabled():
            logits = checkpoint(scores, center, relation, edge_attr, edge_logit,
                                use_reentrant=False)
        else:
            logits = scores(center, relation, edge_attr, edge_logit)
        attention = self.dropout(softmax(logits, tgt, num_nodes=len(x)))
        self._attn_weights = attention.detach() if return_attention_weights else None
        value = self.W_v(relation).view(-1, self.heads, self.head_dim)
        message = attention.unsqueeze(-1)*value
        if edge_value_gate is not None:
            message = message*edge_value_gate.unsqueeze(-1)
        output = torch.zeros(len(x), self.channels, dtype=message.dtype, device=message.device)
        output.scatter_add_(0, tgt[:, None].expand(-1, self.channels),
                            message.reshape(-1, self.channels))
        return (output, (edge_index, self._attn_weights)) if return_attention_weights else output


def attach_nonlinear_relation(model: nn.Module) -> None:
    """기존 두 층의 가중치를 그대로 옮겨 관계 키의 비선형성만 바꾼다."""
    layers = model.gnn.convs
    if len(layers) != 2 or not all(isinstance(layer, DiffAttnConv) for layer in layers):
        raise ValueError("현재 두 attention 층만 지원한다")
    for index, layer in enumerate(layers):
        layers[index] = NonlinearRelationConv(layer)
