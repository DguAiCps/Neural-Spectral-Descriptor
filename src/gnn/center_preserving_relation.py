"""차이 주의 키를 유지한 채 선형 값의 중심 보존량만 조절한다."""
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import softmax

from .nonlinear_relation import NonlinearRelationConv


class CenterPreservingRelationConv(NonlinearRelationConv):
    """K=tanh(Wk(hj-hi)), V=Wv(hj-(1-beta)hi)의 같은 용량 대조.

    beta는 고정 실험 설정이며 학습 파라미터가 아니다. 평가 중 dropout과
    값 게이트가 없을 때 집계는 D+beta*C다. beta=1도 차이 키는 유지한다.
    """

    def __init__(self, reference: NonlinearRelationConv, beta: float) -> None:
        if not isinstance(reference, NonlinearRelationConv) or reference.value_source != "diff":
            raise ValueError("비선형 차이 키 기준층이 필요하다")
        if not 0. <= beta <= 1.:
            raise ValueError("중심 보존량은 0과 1 사이여야 한다")
        super().__init__(reference)
        self.beta = float(beta)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                edge_logit: Optional[torch.Tensor] = None,
                edge_value_gate: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, tuple]]:
        """주의 가중치는 기존 차이 키로 만들고 값의 중심항만 보존한다."""
        if self.beta == 0.:
            return super().forward(x, edge_index, edge_attr, edge_logit,
                                   edge_value_gate, return_attention_weights)
        src, tgt = edge_index
        center = x[tgt]
        relation = x[src]-center

        def scores(center_in, relation_in, attr_in, logit_in):
            query = self.W_q(center_in).view(-1, self.heads, self.head_dim)
            key = self.W_k(relation_in).tanh().view(-1, self.heads, self.head_dim)
            result = torch.einsum("ehd,ehd->eh", query, key)/self.scale
            if attr_in is not None and self.bias_mlp is not None:
                result = result+self.bias_mlp(attr_in)
            if logit_in is not None:
                result = result+logit_in
            return result

        if self.training and torch.is_grad_enabled():
            logits = checkpoint(scores, center, relation, edge_attr, edge_logit, use_reentrant=False)
        else:
            logits = scores(center, relation, edge_attr, edge_logit)
        attention = self.dropout(softmax(logits, tgt, num_nodes=len(x)))
        self._attn_weights = attention.detach() if return_attention_weights else None
        # 투영은 한 번만 한다. 혼합 입력과 D+beta*C는 실수 연산에서 같다.
        value_input = x[src] if self.beta == 1. else x[src]-(1.-self.beta)*center
        value = self.W_v(value_input).view(-1, self.heads, self.head_dim)
        message = attention[..., None]*value
        if edge_value_gate is not None:
            message = message*edge_value_gate[..., None]
        output = torch.zeros(len(x), self.channels, dtype=message.dtype, device=x.device)
        output.scatter_add_(0, tgt[:, None].expand(-1, self.channels), message.reshape(-1, self.channels))
        return (output, (edge_index, self._attn_weights)) if return_attention_weights else output


def attach_center_preservation(model: nn.Module, beta: float) -> None:
    """기존 두 차이 층의 가중치를 유지하며 값 보존량을 고정한다."""
    if len(model.gnn.convs) != 2:
        raise ValueError("현재 두 시간 attention 층만 지원한다")
    for index, conv in enumerate(model.gnn.convs):
        model.gnn.convs[index] = CenterPreservingRelationConv(conv, beta)
