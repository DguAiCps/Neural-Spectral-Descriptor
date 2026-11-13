"""표 4의 그래프 없는 학습 투영: 각 스캔의 불변 키만 처리한다."""
from typing import Optional

import torch
from torch import nn

from .model import create_spectral_gnn as _create_reference


class PointwiseProjection(nn.Module):
    """간선 입력을 사용하지 않는 노드별 선형 변환이다."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """스캔마다 같은 선형 변환을 적용한다."""
        return self.linear(x)


def create_graph_free_model(**kwargs) -> nn.Module:
    """정제기의 메시지 전달을 노드별 투영으로 바꾸고 나머지 연산은 유지한다.

    차원·정규화·잔차·드롭아웃·출력 투영·게이트는 기준 모델과 같다.
    파라미터 수를 맞춘 대조군은 아니다. 학습 중 배치 정규화는 기준
    학습기와 같고, 추론에서는 고정된 통계를 사용하므로 스캔끼리 섞이지 않는다.
    """
    assert kwargs.get("spectral_policy") is None
    for name in ("phase_token_config", "phase_edge_config", "phase_alignment_config",
                 "phase_coherence_config", "dual_stream_config", "sensor_gate_config"):
        assert not (kwargs.get(name) or {}).get("enabled", False), name
    assert not kwargs.get("use_edge_confidence_gate", False)
    model = _create_reference(**kwargs)
    base = model.gnn if hasattr(model, "gnn") else model
    base.edge_encoder = None
    base.convs = nn.ModuleList(PointwiseProjection(base.hidden_dim)
                              for _ in range(base.n_layers))
    return model
