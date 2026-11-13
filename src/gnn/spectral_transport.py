"""시간 간선에서 각도 푸리에 계수를 회전 보정하며 전달한다."""
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def lidar_yaw(poses: np.ndarray, lidar_to_pose: np.ndarray = None) -> np.ndarray:
    """자세의 센서 축을 맞추고 첫 라이다 좌표계에서 yaw를 구한다."""
    rotations = np.asarray(poses, dtype=np.float64)[:, :3, :3]
    if lidar_to_pose is not None:
        rotations = rotations @ np.asarray(lidar_to_pose, dtype=np.float64)[:3, :3]
    relative = rotations[0].T[None] @ rotations
    return np.arctan2(relative[:, 1, 0], relative[:, 0, 0])


def attach_transport(graph, spectrum: np.ndarray, yaw: np.ndarray,
                     sequences: np.ndarray) -> Dict[str, float]:
    """실제 ±5 시간 간선과 스캔별 복소 특징·방향을 그래프에 붙인다."""
    n = graph.num_nodes
    spectrum = np.ascontiguousarray(spectrum, dtype=np.complex64)
    yaw = np.asarray(yaw, dtype=np.float64)
    sequences = np.asarray(sequences)
    if spectrum.shape != (n, 16, 8) or yaw.shape != (n,):
        raise ValueError("복소 특징 (N,16,8)과 yaw (N,)가 필요하다")
    if not np.isfinite(spectrum).all() or not np.isfinite(yaw).all():
        raise ValueError("입력은 유한해야 한다")
    source, target = graph.edge_index.detach().cpu().numpy()
    offset = source - target + 5
    if np.any((offset < 0) | (offset > 10) | (source == target)):
        raise ValueError("자기 자신을 제외한 ±5 시간 간선만 지원한다")
    if np.any(sequences[source] != sequences[target]):
        raise ValueError("서로 다른 시퀀스 사이의 간선은 허용하지 않는다")
    if not bool((graph.edge_type == 0).all()):
        raise ValueError("시간 간선만 사용한다")
    index = np.broadcast_to(np.arange(n)[:, None], (n, 11)).copy()
    mask = np.zeros((n, 11), np.float32)
    index[target, offset], mask[target, offset] = source, 1.
    if int(mask.sum()) != len(source):
        raise ValueError("중복된 시간 간선은 허용하지 않는다")
    device = graph.x.device
    graph.transport_indices = torch.as_tensor(index, device=device)
    graph.transport_mask = torch.as_tensor(mask, device=device)
    graph.transport_phase = torch.as_tensor(
        np.exp(1j * yaw[:, None] * np.arange(1, 9)[None]).astype(np.complex64),
        device=device)
    graph.spectrum = torch.as_tensor(spectrum, device=device)
    return {"zero_spectrum_fraction": float(np.mean(np.linalg.norm(
        spectrum.reshape(n, -1), axis=1) == 0)),
        "isolated_fraction": float(np.mean(mask.sum(1) == 0))}


def complex_unit_rms(z: torch.Tensor) -> torch.Tensor:
    """노드마다 복소 특징의 제곱평균제곱근을 1로 맞춘다."""
    scale = (z.real.square() + z.imag.square()).mean((1, 2), keepdim=True)
    return z / scale.clamp_min(1e-12).sqrt()


def spectral_mean(z: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor,
                  phase: torch.Tensor, mode: str) -> torch.Tensor:
    """이웃 계수를 중심 좌표로 옮겨 평균하거나 대조 연산을 수행한다.

    X_i=e^(ik*yaw_i) X_world이므로 j→i의 회전은
    e^(ik*(yaw_i-yaw_j))다. 전역 중간 좌표는 이 상대 회전의 인수분해다.
    """
    if mode == "spectral_local":
        return z
    if mode not in ("untransported", "transported"):
        raise ValueError(mode)
    world = z * phase.conj()[:, None] if mode == "transported" else z
    total = torch.zeros_like(world)
    for slot in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10):
        total = total + world[indices[:, slot]] * mask[:, slot, None, None]
    total = total / mask.sum(1).clamp_min(1.)[:, None, None]
    return total * phase[:, None] if mode == "transported" else total


class SpectralTransportConv(nn.Module):
    """주파수별 복소 전달과 그 크기를 쓰는 실수 특징 변환 한 층이다."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.self_weight = nn.Parameter(torch.empty(2, 8, 16, 16))
        self.neighbor_weight = nn.Parameter(torch.empty(2, 8, 16, 16))
        nn.init.normal_(self.self_weight, std=1 / np.sqrt(32.))
        nn.init.normal_(self.neighbor_weight, std=1 / np.sqrt(32.))
        self.layers = nn.Sequential(nn.Linear(384, 86), nn.ReLU(), nn.Linear(86, 128))

    def forward(self, h: torch.Tensor, z: torch.Tensor, indices: torch.Tensor,
                mask: torch.Tensor, phase: torch.Tensor) -> tuple:
        """복소 위상을 두 층 사이에 보존하고 크기만 실수 가지로 읽는다."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            mean = spectral_mean(z, indices, mask, phase, self.mode)
            wself = torch.complex(self.self_weight[0], self.self_weight[1])
            wneighbor = torch.complex(self.neighbor_weight[0], self.neighbor_weight[1])
            updated = complex_unit_rms(
                z + torch.einsum("nrk,krs->nsk", z, wself) +
                torch.einsum("nrk,krs->nsk", mean, wneighbor))
            invariant = torch.cat((h.float(), z.abs().flatten(1),
                                   updated.abs().flatten(1)), dim=-1)
        return self.layers(invariant.to(h.dtype)), updated


class SpectralTransportGNN(nn.Module):
    """기준 문맥 모델의 두 집계 층을 복소 전달 층으로 대체한다."""

    def __init__(self, reference: nn.Module, mode: str) -> None:
        super().__init__()
        for name in ("input_proj", "input_norm", "batch_norms", "output_proj", "gate"):
            setattr(self, name, getattr(reference, name))
        self.register_parameter("key_scale", reference.key_scale)
        for name in ("input_dim", "hidden_dim", "context_dim", "n_layers", "dropout",
                     "residual", "gradient_checkpointing", "use_residual_gate",
                     "key_transform"):
            setattr(self, name, getattr(reference, name))
        self.convs = nn.ModuleList(SpectralTransportConv(mode) for _ in range(self.n_layers))
        self.spectral_policy = None
        self._last_alpha = self._last_alpha_for_loss = None

    def get_embedding_dim(self) -> int:
        """원 특징과 문맥을 합친 출력 차원을 반환한다."""
        return self.input_dim + self.context_dim

    def forward(self, data) -> torch.Tensor:
        """최대 앞뒤 10키프레임에서 복소 문맥을 만들어 기존 키에 붙인다."""
        x = data.x
        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_gpu_dtype())
        h = F.relu(self.input_norm(self.input_proj(x)))
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = complex_unit_rms(data.spectrum)
        for i, (conv, norm) in enumerate(zip(self.convs, self.batch_norms)):
            previous = h
            if torch.is_autocast_enabled():
                h = h.to(torch.get_autocast_gpu_dtype())
            args = (h, z, data.transport_indices, data.transport_mask, data.transport_phase)
            h, z = (checkpoint(conv, *args, use_reentrant=False)
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
