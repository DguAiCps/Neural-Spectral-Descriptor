"""기존 차이 attention과 게이트를 복소 스펙트럼 차이 전달에 연결한다."""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import softmax

from .geometry_interaction import GeometryInteractionConv, combine_scores
from .spectral_difference import transport_difference


class ControlledRelationConv(GeometryInteractionConv):
    """같은 가중치·비선형성·기하 결합에서 K/V의 차이 여부만 바꾼다."""

    def __init__(self, reference: GeometryInteractionConv, mode: str) -> None:
        super().__init__(reference, reference.geometry_mode)
        if mode not in ("diff", "abs_k", "abs_kv"):
            raise ValueError(mode)
        self.relation_mode = mode
        self.attention = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """학습용 주의 가중치를 미분 연결 상태로 보존한다."""
        src, tgt = edge_index
        center = x[tgt]
        delta = x[src] - center
        key_input = delta if self.relation_mode == "diff" else x[src]
        value_input = x[src] if self.relation_mode == "abs_kv" else delta
        q = self.W_q(center).view(-1, self.heads, self.head_dim)
        k = self.W_k(key_input).tanh().view(-1, self.heads, self.head_dim)
        content = (q * k).sum(-1) / self.scale
        logits = (
            combine_scores(
                content, self.bias_mlp(edge_attr), tgt, len(x), self.geometry_mode
            )
            if edge_attr is not None and self.bias_mlp is not None
            else content
        )
        self.attention = self.dropout(softmax(logits, tgt, num_nodes=len(x)))
        v = self.W_v(value_input).tanh().view(-1, self.heads, self.head_dim)
        messages = (v * self.attention[..., None]).flatten(1)
        out = torch.zeros(len(x), self.channels, dtype=x.dtype, device=x.device)
        return out.index_add(0, tgt, messages)


class LearnedSpectralDifference(nn.Module):
    """추가 층 없이 기존 두 층의 attention·게이트로 복소 차이를 전달한다."""

    def __init__(self, reference: nn.Module, kind: str) -> None:
        super().__init__()
        self.kind = kind
        gnn = reference.gnn
        for name in ("input_proj", "input_norm", "batch_norms", "gate", "edge_encoder"):
            setattr(self, name, getattr(gnn, name))
        if kind == "point_gate":
            self.convs = gnn.convs
            self.edge_encoder = None
        else:
            mode = "diff" if kind == "gate_only" else kind
            self.convs = nn.ModuleList(
                ControlledRelationConv(c, mode) for c in gnn.convs
            )
            with torch.no_grad():
                for conv in self.convs:
                    conv.W_q.weight.zero_()
                    if conv.W_q.bias is not None:
                        conv.W_q.bias.zero_()
                    final = [
                        v for v in conv.bias_mlp.modules() if isinstance(v, nn.Linear)
                    ][-1]
                    final.weight.zero_()
                    final.bias.zero_()
        with torch.no_grad():
            last = [v for v in self.gate.modules() if isinstance(v, nn.Linear)][-1]
            last.weight.zero_()
            last.bias.zero_()
        for name, param in self.named_parameters():
            param.requires_grad_(kind != "gate_only" or name.startswith("gate."))
        self.last_strengths = None

    def forward(self, graph) -> torch.Tensor:
        """최대 ±10키프레임의 출력으로 최종 q+15 조건을 보존한다."""
        h = F.relu(self.input_norm(self.input_proj(graph.x)))
        first = h
        attr = (
            self.edge_encoder(graph.edge_attr, graph.edge_type)
            if self.edge_encoder is not None
            else None
        )
        weights = []
        src, tgt = graph.edge_index
        for layer, (conv, norm) in enumerate(zip(self.convs, self.batch_norms)):
            previous = h
            h = norm(conv(h, graph.edge_index, edge_attr=attr))
            if layer == 0:
                h = F.relu(h)
            h = h + previous
            if self.kind != "point_gate":
                dense = torch.zeros(len(h), 11, 4, dtype=h.dtype, device=h.device)
                dense[tgt, src - tgt + 5] = conv.attention
                weights.append(dense)
        # 첫 전달의 세기는 점별 h0, 둘째는 ±10의 h2만 사용한다.
        # h2를 첫 전달에 쓰면 복소 이웃 전달까지 합쳐 미래 범위가 늘어난다.
        strengths = 0.5 * torch.stack(
            (self.gate(first).sigmoid(), self.gate(h).sigmoid())
        )
        self.last_strengths = strengths
        attention = torch.stack(weights) if weights else None
        return transport_difference(
            graph.spectrum,
            graph.transport_indices,
            graph.transport_mask,
            graph.transport_phase,
            strengths,
            attention,
        )
