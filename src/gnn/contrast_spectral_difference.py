"""이웃 공통 성분을 빼는 복소 차이 전달의 같은 용량 학습 대조."""

import torch
from torch import nn
from torch.nn import functional as F

from .learned_spectral_difference import LearnedSpectralDifference
from .spectral_difference import transport_difference


class ContrastSpectralDifference(LearnedSpectralDifference):
    """기존 attention·게이트를 쓰며 전달 세기를 -1~0으로 바꾼다."""

    def forward(self, graph) -> torch.Tensor:
        """초기 세기 -.5에서 이웃 공통 성분을 제거한다."""
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
        strengths = -torch.stack((self.gate(first).sigmoid(), self.gate(h).sigmoid()))
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


class GlobalContrastDifference(nn.Module):
    """스캔과 무관한 층별 공통 세기 두 개만 학습한다."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2, 1, 1))
        self.last_strengths = None

    def forward(self, graph) -> torch.Tensor:
        """학습되는 공통 세기와 균등 시간 주의로 차이를 강조한다."""
        self.last_strengths = -self.logits.sigmoid().expand(-1, len(graph.x), -1)
        return transport_difference(
            graph.spectrum,
            graph.transport_indices,
            graph.transport_mask,
            graph.transport_phase,
            self.last_strengths,
        )
