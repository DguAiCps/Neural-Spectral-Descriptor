"""차이 attention의 값에 부호 또는 제곱 크기를 보존하는 대조다."""

from typing import Optional, Tuple, Union

import torch
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import softmax

from .geometry_interaction import GeometryInteractionConv, combine_scores


class MomentDifferenceConv(GeometryInteractionConv):
    """기존 Q/K·기하 결합을 유지하며 차이 값의 형태만 바꾼다."""

    def __init__(self, reference: GeometryInteractionConv, mode: str) -> None:
        super().__init__(reference, reference.geometry_mode)
        if mode not in ("signed", "square", "odd_even"):
            raise ValueError(mode)
        self.moment_mode = mode

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_logit: Optional[torch.Tensor] = None,
        edge_value_gate: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, tuple]]:
        """중심 차이의 평균·제곱 평균·부호/크기 분할을 동일 용량으로 구한다."""
        src, tgt = edge_index
        center = x[tgt]
        relation = x[src] - center

        def logits(center_in, relation_in, attr_in, logit_in):
            q = self.W_q(center_in).view(-1, self.heads, self.head_dim)
            k = self.W_k(relation_in).tanh().view(-1, self.heads, self.head_dim)
            content = torch.einsum("ehd,ehd->eh", q, k) / self.scale
            result = (
                combine_scores(
                    content, self.bias_mlp(attr_in), tgt, len(x), self.geometry_mode
                )
                if attr_in is not None
                and self.bias_mlp is not None
                and self.geometry_enabled
                else content
            )
            return result if logit_in is None else result + logit_in

        value = (
            checkpoint(
                logits, center, relation, edge_attr, edge_logit, use_reentrant=False
            )
            if self.training and torch.is_grad_enabled()
            else logits(center, relation, edge_attr, edge_logit)
        )
        attention = self.dropout(softmax(value, tgt, num_nodes=len(x)))
        v = self.W_v(relation).view(-1, self.heads, self.head_dim).tanh()
        if self.moment_mode == "square":
            v = v.square()
        elif self.moment_mode == "odd_even":
            half = self.head_dim // 2
            v = torch.cat((v[..., :half], v[..., half:].square()), -1)
        message = attention.unsqueeze(-1) * v
        if edge_value_gate is not None:
            message = message * edge_value_gate.unsqueeze(-1)
        output = torch.zeros(
            len(x), self.channels, dtype=message.dtype, device=message.device
        )
        output.scatter_add_(
            0,
            tgt[:, None].expand(-1, self.channels),
            message.reshape(-1, self.channels),
        )
        self._attn_weights = attention.detach() if return_attention_weights else None
        if return_attention_weights:
            return output, (edge_index, self._attn_weights)
        return output
