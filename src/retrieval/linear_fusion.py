"""기존 네 원시 점수의 가중합을 후보 선택과 거절에 함께 사용한다."""
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

FEATURE_NAMES = ("key_cos", "aux_cos", "c_spec", "c_col")


def fusion_features(embedding: torch.Tensor, queries: torch.Tensor,
                    candidates: torch.Tensor, static: torch.Tensor,
                    valid: torch.Tensor) -> torch.Tensor:
    """후보 목록의 최솟값·최댓값을 쓰지 않고 기존 네 점수를 조립한다."""
    query = embedding[queries, None, :]
    database = embedding[candidates.clamp_min(0)]
    key = (query * database).sum(-1)
    return torch.cat([key[..., None], static], dim=-1).masked_fill(~valid[..., None], 0.)


class LinearFusionScorer(nn.Module):
    """양의 가중치 네 개와 절편 하나만 학습하는 단순 가중합이다."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        std_tensor = torch.as_tensor(std, dtype=torch.float32)
        if mean_tensor.shape != (4,) or std_tensor.shape != (4,):
            raise ValueError("네 점수의 학습셋 평균·표준편차가 필요하다")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all() or (std_tensor <= 0).any():
            raise ValueError("정규화 상수는 유한하고 표준편차는 양수여야 한다")
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)
        # 표준화된 좌표에서 가중치 1, 절편 -3으로 모든 조건을 초기화한다.
        self.weight_logits = nn.Parameter(torch.log(torch.expm1(torch.ones(4))))
        self.bias = nn.Parameter(torch.tensor(-3.))

    def raw_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """고정 정규화를 흡수한 원시 점수용 가중치·절편을 반환한다."""
        weights = F.softplus(self.weight_logits) / self.std
        intercept = self.bias - (weights * self.mean).sum()
        return weights, intercept

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """학습·평가 모두 배포 시와 같은 원시 점수 가중합을 계산한다."""
        weights, intercept = self.raw_coefficients()
        return (features * weights).sum(-1) + intercept

    def export(self) -> dict:
        """추론에 필요한 상수 다섯 개를 내보낸다."""
        weights, intercept = self.raw_coefficients()
        return {"features": list(FEATURE_NAMES), "weights": weights.detach().cpu().tolist(),
                "intercept": float(intercept.detach()), "trainable_parameters": 5}
