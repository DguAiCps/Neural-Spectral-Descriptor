"""기존 128차원 문맥을 위상을 보존하는 스펙트럼 이득으로 읽는다."""

from typing import Dict, Tuple

import torch
from torch.nn import functional as F


def spectral_gain(embedding: torch.Tensor, strength: float) -> torch.Tensor:
    """정규화 키에서 게이트 문맥을 복원하고 양의 16×8 이득을 만든다."""
    raw_norm = embedding[:, :288].norm(dim=-1, keepdim=True)
    context = embedding[:, 288:] / raw_norm.clamp_min(1e-12)
    context = torch.where(raw_norm > 1e-12, context, torch.zeros_like(context))
    return torch.exp(strength * (128.0**0.5 * context).tanh()).reshape(-1, 16, 8)


def spectral_bank(coefficients: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """복소 계수로 기존 60각도 정렬의 미분 가능한 두 채널을 만든다."""
    spectrum = F.pad(coefficients, (1, 22))
    norm = coefficients.abs().flatten(1).norm(dim=1).clamp_min(1e-8)
    grid = torch.fft.irfft(spectrum, n=60, dim=-1)
    column_norm = grid.norm(dim=1, keepdim=True)
    mask = (column_norm[:, 0] > 1e-8).to(grid.dtype)
    columns = torch.fft.rfft(grid / column_norm.clamp_min(1e-8), dim=-1)
    return spectrum, norm, columns, torch.fft.rfft(mask, dim=-1)


def pair_peaks(
    bank: tuple, query: torch.Tensor, candidate: torch.Tensor
) -> torch.Tensor:
    """각 쌍의 스펙트럼·열 정렬 피크를 순서대로 반환한다."""
    spectrum, norm, columns, mask = bank
    cross = spectrum[candidate] * spectrum[query].conj()
    profile = torch.fft.irfft(cross, n=60, dim=-1).sum(-2)
    spec = profile.max(-1).values / (norm[candidate] * norm[query])
    profile = torch.fft.irfft(columns[candidate] * columns[query].conj(), n=60).sum(-2)
    count = torch.fft.irfft(mask[candidate] * mask[query].conj(), n=60)
    col = torch.where(count > 0.5, profile / count.clamp_min(1.0), -1.0).max(-1).values
    return torch.stack((spec, col), -1)


def gain_score(
    embedding: torch.Tensor,
    coefficients: torch.Tensor,
    plan: Dict[str, torch.Tensor],
    strength: float,
) -> torch.Tensor:
    """두 스펙트럼의 위상을 보존하고 기존 시간 판정까지 미분한다.

    coefficients는 (N,2,16,8)의 제곱근 진폭 원본·상관 보정 계수다.
    plan의 query/candidate/weight는 (B,2,11)이다. 추론에도 같은 식을 쓴다.
    """
    q, j, weight = (plan[k] for k in ("query", "candidate", "weight"))
    ids, inverse = torch.unique(
        torch.cat((q.flatten(), j.flatten())), return_inverse=True
    )
    iq, ij = inverse.chunk(2)
    gain = spectral_gain(embedding[ids], strength)
    peaks = []
    for index in range(2):
        bank = spectral_bank(coefficients[ids, index] * gain)
        p = pair_peaks(bank, iq, ij).reshape(*q.shape, 2)
        peaks.append((p * weight[..., None]).sum(-2).max(-2).values)
    mixed = 0.75 * peaks[0] + 0.25 * peaks[1]
    return (30.0 * mixed[:, 0] + mixed[:, 1]) / 2.0
