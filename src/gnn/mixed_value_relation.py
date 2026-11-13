"""비선형 차이 값에 선형 성분을 남기는 고정 혼합 대조."""
import math

import torch
from torch import nn

from .model import DiffAttnConv
from .value_relation import ValueRelationConv


class MixedValueRelationConv(ValueRelationConv):
    """값을 a*z+(1-a)*tanh(z)로 만들고 기존 attention을 유지한다."""

    def __init__(self, reference: DiffAttnConv, linear_weight: float = .5) -> None:
        super().__init__(reference, "linear")
        if not math.isfinite(linear_weight) or not 0 <= linear_weight <= 1:
            raise ValueError("선형 비중은 0 이상 1 이하의 유한한 값이어야 한다")
        self.linear_weight = float(linear_weight)

    def transform_value(self, projected: torch.Tensor) -> torch.Tensor:
        """원점 기울기 1, 큰 입력 기울기 a인 홀함수로 값을 변환한다."""
        if self.linear_weight == 1:
            return projected
        bounded = projected.tanh()
        if self.linear_weight == 0:
            return bounded
        return self.linear_weight * projected + (1 - self.linear_weight) * bounded


def attach_mixed_value_relation(model: nn.Module, linear_weight: float = .5) -> None:
    """두 차이 attention 층의 가중치를 유지하며 값 변환만 교체한다."""
    layers = model.gnn.convs
    if len(layers) != 2 or not all(isinstance(layer, DiffAttnConv) for layer in layers):
        raise ValueError("원래 두 DiffAttnConv 층에만 적용한다")
    for index, layer in enumerate(layers):
        layers[index] = MixedValueRelationConv(layer, linear_weight)
