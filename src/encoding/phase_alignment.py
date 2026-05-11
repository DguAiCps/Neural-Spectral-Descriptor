"""Edge-wise phase-alignment features for phase-conditioned GAT.

This module computes the same cyclic-shift evidence used by the compact
phase-sketch reranker, but exposes only low-leakage confidence features by
default. The raw alignment score is optional because feeding the teacher score
directly to the GAT can collapse into sketch-score mimicry.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def reshape_complex_phase(
    x_phase: torch.Tensor,
    n_rows: int,
    n_freqs: int,
) -> torch.Tensor:
    """Return ``(N, rows, freqs)`` complex phase coefficients.

    ``x_phase`` must be laid out as ``[real || imag]`` over a single raw complex
    Fourier grid. Sources such as ``bev_cross`` are already transformed and are
    not valid inputs for this function.
    """
    half = int(n_rows) * int(n_freqs)
    if x_phase.shape[-1] < 2 * half:
        raise ValueError(
            f"x_phase has trailing dim {x_phase.shape[-1]}, "
            f"but raw complex phase needs at least {2 * half} "
            f"for shape ({n_rows}, {n_freqs})"
        )
    real = x_phase[..., :half].reshape(*x_phase.shape[:-1], n_rows, n_freqs)
    imag = x_phase[..., half : 2 * half].reshape(*x_phase.shape[:-1], n_rows, n_freqs)
    return torch.complex(real.float(), imag.float())


def phase_alignment_edge_features(
    x_phase: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: Optional[torch.Tensor],
    n_rows: int,
    n_freqs: int,
    n_sectors: int,
    include_score: bool = False,
    similarity_only: bool = True,
    entropy_temperature: float = 0.05,
) -> torch.Tensor:
    """Compute leakage-controlled phase-alignment features per graph edge.

    Features are:

    - optionally ``best_score`` (teacher-like cyclic-shift cosine; off by default)
    - ``sin(best_shift)``, ``cos(best_shift)``
    - ``peak_margin`` between the best and second-best shifts
    - normalized shift entropy in ``[0, 1]``

    Args:
        x_phase: ``(N, 2 * rows * freqs)`` raw complex Fourier sketch.
        edge_index: ``(2, E)`` graph edges; source is candidate neighbor.
        edge_type: optional ``(E,)`` where 1 means similarity edge.
        n_rows/n_freqs: raw complex sketch shape.
        n_sectors: DFT length used when the sketch was computed.
        include_score: include raw best alignment score as the first feature.
        similarity_only: zero features for temporal edges.
        entropy_temperature: softmax temperature over cyclic shifts.
    """
    if n_sectors < n_freqs + 1:
        raise ValueError(
            f"n_sectors={n_sectors} must be >= n_freqs+1={n_freqs + 1}"
        )
    if entropy_temperature <= 0:
        raise ValueError("entropy_temperature must be positive")

    z = reshape_complex_phase(x_phase, n_rows, n_freqs)
    src, tgt = edge_index[0], edge_index[1]
    z_src = z[src]
    z_tgt = z[tgt]

    # Row-summed cross spectrum. Direction is consistent with the reranker:
    # target/query against source/candidate.
    cross = (torch.conj(z_tgt) * z_src).sum(dim=1)                 # (E, K)
    padded = torch.zeros(
        cross.shape[0],
        n_sectors,
        dtype=cross.dtype,
        device=cross.device,
    )
    padded[:, 1 : n_freqs + 1] = cross
    corr = torch.fft.fft(padded, dim=-1).real                      # (E, S)

    q_norm = z_tgt.reshape(z_tgt.shape[0], -1).abs().square().sum(dim=-1).sqrt()
    c_norm = z_src.reshape(z_src.shape[0], -1).abs().square().sum(dim=-1).sqrt()
    sims = corr / (q_norm * c_norm).clamp_min(1e-8).unsqueeze(-1)

    if n_sectors >= 2:
        top2 = torch.topk(sims, k=2, dim=-1).values
        best_score = top2[:, 0]
        peak_margin = top2[:, 0] - top2[:, 1]
    else:
        best_score = sims[:, 0]
        peak_margin = torch.zeros_like(best_score)

    best_shift = torch.argmax(sims, dim=-1).float()
    angle = 2.0 * math.pi * best_shift / float(n_sectors)
    shift_sin = torch.sin(angle)
    shift_cos = torch.cos(angle)

    probs = torch.softmax(sims / entropy_temperature, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    entropy = entropy / math.log(float(max(n_sectors, 2)))

    parts = []
    if include_score:
        parts.append(best_score.unsqueeze(-1))
    parts.extend([
        shift_sin.unsqueeze(-1),
        shift_cos.unsqueeze(-1),
        peak_margin.unsqueeze(-1),
        entropy.unsqueeze(-1),
    ])
    features = torch.cat(parts, dim=-1)

    if similarity_only and edge_type is not None:
        is_similarity = (edge_type == 1).unsqueeze(-1)
        features = torch.where(is_similarity, features, torch.zeros_like(features))
    return features


def feature_dim(include_score: bool = False) -> int:
    """Feature dimension for ``phase_alignment_edge_features``."""
    return 5 if include_score else 4
