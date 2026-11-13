"""시간 간선 안에서 앞뒤 거리 보간과 저차 변화 정보를 계산한다."""
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def derivative_plan(
    positions: np.ndarray,
    sequences: np.ndarray,
    mode: str = "arc",
    radius: int = 5,
    scales: Tuple[float, float] = (1.6, 3.2),
    minimum_radius: float = 0.4,
) -> Dict[str, np.ndarray]:
    """±radius 간선의 선형 보간으로 같은 거리의 1차·2차 차분을 만든다.

    양쪽에서 도달 가능한 거리로 각 규모를 줄인다. 최소 거리보다
    짧거나 한쪽이 없는 노드는 차분을 0으로 둔다. 같은 누적 거리의
    관측은 평균하여 정지 구간의 앞뒤 순서를 뒤집어도 같은 규칙을 쓴다.
    frame은 같은 연산을 키프레임 좌표와 2·4프레임 규모에 적용한다.
    """
    if mode not in ("arc", "frame"):
        raise ValueError(mode)
    positions = np.asarray(positions, dtype=np.float64)
    sequences = np.asarray(sequences)
    n = len(positions)
    if positions.shape != (n, 3) or sequences.shape != (n,):
        raise ValueError("positions=(N,3), sequences=(N,) 이어야 한다")
    if not np.isfinite(positions).all() or radius < 1:
        raise ValueError("유한한 좌표와 양의 창 크기가 필요하다")
    offsets = np.arange(-radius, radius + 1)
    indices = np.clip(np.arange(n)[:, None] + offsets, 0, max(n - 1, 0))
    coeff = np.zeros((n, 4, len(offsets)), np.float64)
    used_radii = np.zeros((n, 2), np.float64)
    valid = np.zeros((n, 2), bool)
    if not n:
        return dict(indices=indices, coefficients=coeff.astype(np.float32),
                    radii=used_radii, valid=valid)
    # 시퀀스는 연속된 블록이어야 한다. 서로 다른 주행 사이의 거리를 더하지 않는다.
    cuts = np.r_[0, np.flatnonzero(sequences[1:] != sequences[:-1]) + 1, n]
    if len(np.unique(sequences)) != len(cuts) - 1:
        raise ValueError("시퀀스별 노드가 연속되어야 한다")
    targets = scales if mode == "arc" else (2., 4.)
    lower_bound = minimum_radius if mode == "arc" else 1.
    for start, end in zip(cuts[:-1], cuts[1:]):
        path = (np.r_[0., np.cumsum(np.linalg.norm(
            np.diff(positions[start:end], axis=0), axis=1))]
            if mode == "arc" else np.arange(end - start, dtype=float))
        for i in range(start + 1, end - 1):
            lo, hi = max(start, i - radius), min(end - 1, i + radius)
            local = np.arange(lo, hi + 1)
            distance = path[local - start] - path[i - start]
            reach = min(-distance[0], distance[-1])
            if reach < lower_bound:
                continue

            def interpolate(target: float) -> np.ndarray:
                unique, inverse, count = np.unique(
                    distance, return_inverse=True, return_counts=True)
                right = min(int(np.searchsorted(unique, target)), len(unique) - 1)
                left = max(right - 1, 0)
                weights = np.zeros(len(unique))
                if unique[right] == target or left == right:
                    weights[right] = 1.
                else:
                    fraction = (target - unique[left]) / (unique[right] - unique[left])
                    weights[left], weights[right] = 1. - fraction, fraction
                result = np.zeros(len(offsets))
                result[local - i + radius] = weights[inverse] / count[inverse]
                return result

            for k, wanted in enumerate(targets):
                r = min(wanted, reach)
                plus, minus = interpolate(r), interpolate(-r)
                coeff[i, 2*k] = (plus - minus) / (2. * r)
                coeff[i, 2*k + 1] = (plus + minus) / (r*r)
                coeff[i, 2*k + 1, radius] -= 2. / (r*r)
                valid[i, k], used_radii[i, k] = True, r
    if not np.isfinite(coeff).all():
        raise ValueError("차분 계수가 유한하지 않다")
    return dict(indices=indices, coefficients=coeff.astype(np.float32),
                radii=used_radii, valid=valid)


def attach_derivatives(graph, positions: np.ndarray, sequences: np.ndarray,
                       mode: str = "arc") -> dict:
    """실제 시간 간선 범위 안의 고정 차분 계획을 그래프에 붙인다."""
    plan = derivative_plan(positions, sequences, mode)
    edges = graph.edge_index.detach().cpu().numpy()
    n = len(positions)
    available = np.zeros((n, 11), bool)
    available[:, 5] = True
    source, target = edges
    offsets = source - target + 5
    if np.any((offsets < 0) | (offsets > 10)):
        raise ValueError("±5 시간 간선만 지원한다")
    available[target, offsets] = True
    required = np.any(plan["coefficients"] != 0, axis=1)
    if np.any(required & ~available):
        raise ValueError("차분 계획이 실제 시간 간선 밖의 관측을 사용한다")
    device = graph.x.device
    graph.derivative_indices = torch.as_tensor(plan["indices"], device=device)
    graph.derivative_coefficients = torch.as_tensor(plan["coefficients"], device=device)
    return plan


def temporal_differences(x: torch.Tensor, indices: torch.Tensor,
                         coefficients: torch.Tensor) -> torch.Tensor:
    """정지 구간에서도 유한한 앞뒤 차분을 FP32 이상으로 계산한다."""
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        neighbors = x[indices].to(dtype)
        return torch.einsum("nko,noc->nkc", coefficients.to(dtype), neighbors)


class TemporalDerivativeConv(nn.Module):
    """중심·두 거리의 절대 1차 변화·부호 있는 2차 변화를 압축한다."""

    def __init__(self, channels: int = 128, width: int = 68) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(5 * channels, width),
                                    nn.ReLU(), nn.Linear(width, channels))

    def forward(self, x: torch.Tensor, indices: torch.Tensor,
                coefficients: torch.Tensor) -> torch.Tensor:
        """앞뒤 반전 불변인 다섯 블록을 만든다."""
        changes = temporal_differences(x, indices, coefficients)
        features = torch.cat((x.float(), changes[:, 0].abs(), changes[:, 1],
                              changes[:, 2].abs(), changes[:, 3]), dim=-1)
        return self.layers(features.to(x.dtype))


class TemporalDerivativeGNN(nn.Module):
    """기준 모델의 입출력·정규화·게이트를 보존하고 두 집계 층을 교체한다."""

    def __init__(self, reference: nn.Module) -> None:
        super().__init__()
        for name in ("input_proj", "input_norm", "batch_norms", "output_proj", "gate"):
            setattr(self, name, getattr(reference, name))
        self.register_parameter("key_scale", reference.key_scale)
        for name in ("input_dim", "hidden_dim", "context_dim", "n_layers", "dropout",
                     "residual", "gradient_checkpointing", "use_residual_gate",
                     "key_transform"):
            setattr(self, name, getattr(reference, name))
        self.convs = nn.ModuleList(TemporalDerivativeConv(self.hidden_dim)
                                   for _ in range(self.n_layers))
        self.spectral_policy = None
        self._last_alpha = None
        self._last_alpha_for_loss = None

    def get_embedding_dim(self) -> int:
        """기준 검색 키와 같은 출력 차원을 반환한다."""
        return self.input_dim + self.context_dim

    def forward(self, data) -> torch.Tensor:
        """두 층 모두 같은 ±5 시간 간선의 차분을 사용한다."""
        x = data.x
        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_gpu_dtype())
        h = F.relu(self.input_norm(self.input_proj(x)))
        for i, (conv, norm) in enumerate(zip(self.convs, self.batch_norms)):
            previous = h
            if torch.is_autocast_enabled():
                h = h.to(torch.get_autocast_gpu_dtype())
            args = (h, data.derivative_indices, data.derivative_coefficients)
            h = (checkpoint(conv, *args, use_reentrant=False)
                 if self.gradient_checkpointing and self.training else conv(*args))
            h = norm(h)
            if i < self.n_layers - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
            if self.residual:
                h = h + previous
        context = F.normalize(self.output_proj(h), p=2, dim=-1)
        if self.use_residual_gate:
            alpha = torch.sigmoid(self.gate(h))
            context = alpha * context
            self._last_alpha_for_loss, self._last_alpha = alpha, alpha.detach()
        raw = torch.log1p(x) if self.key_transform == "log1p" else x
        if self.key_scale is not None:
            raw = raw * self.key_scale
        return torch.cat((F.normalize(raw, p=2, dim=-1), context), dim=-1)
