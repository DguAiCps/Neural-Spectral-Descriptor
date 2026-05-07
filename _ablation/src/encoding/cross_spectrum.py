"""Yaw-invariant adjacent-row cross-spectrum features."""

from __future__ import annotations

import torch


def adjacent_cross_spectrum_dim(n_rows: int, n_freqs: int) -> int:
    """Return flattened real/imag dimension for adjacent row-pair features."""
    if n_rows < 2:
        return 0
    if n_freqs <= 0:
        return 0
    return (n_rows - 1) * n_freqs * 2


def normalized_adjacent_cross_spectrum(
    fft_rows: torch.Tensor,
    n_freqs: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute compact phase-preserving, yaw-invariant row-pair features.

    For a yaw shift by tau, every row-wise Fourier coefficient receives the same
    phase factor exp(-i 2*pi*k*tau/A). The product
    F_e(k) * conj(F_{e+1}(k)) cancels that factor and preserves the relative
    phase between adjacent elevation rows. DC is skipped because its phase is
    not informative.

    Args:
        fft_rows: Complex tensor with shape (rows, rfft_freqs).
        n_freqs: Number of non-DC low frequencies to keep.
        eps: Stabilizer for magnitude normalization.

    Returns:
        Flattened tensor with shape ((rows - 1) * n_freqs * 2,). Real and
        imaginary channels are interleaved in the final dimension.
    """
    if n_freqs <= 0 or fft_rows.shape[0] < 2:
        return fft_rows.new_empty((0,), dtype=torch.float32)
    max_freqs = fft_rows.shape[1] - 1
    if n_freqs > max_freqs:
        raise ValueError(f"n_freqs={n_freqs} exceeds available non-DC frequencies {max_freqs}")

    low = fft_rows[:, 1 : n_freqs + 1]
    prod = low[:-1] * torch.conj(low[1:])
    denom = torch.abs(low[:-1]) * torch.abs(low[1:]) + eps
    cross = prod / denom
    features = torch.stack((cross.real, cross.imag), dim=-1)
    return features.reshape(-1).to(torch.float32)
