"""Phase-stream helpers for the dual-stream GNN.

Provides yaw-invariant per-node features derived from compact Fourier phase
sketches. These features are then refined by the existing real-valued
:class:`gnn.model.DiffAttnConv` — there is no new conv operator. Phase-aware
behaviour at attention time is supplied separately by
:class:`encoding.phase_coherence.ClosedFormPhaseEdgeBias` (closed-form,
parameter-free) and the existing :class:`gnn.model.PhaseEdgeBias` (learned).

Why no conjugate-product conv
-----------------------------
A conjugate-product edge feature ``z_j · conj(z_i)`` carries the *relative*
yaw of the two nodes as a phase ramp ``exp(-2πi · k · (n_j − n_i) / A)`` and
is therefore yaw-equivariant, not yaw-invariant. Feeding it into a
``DiffAttnConv`` style attention yields a yaw-equivariant output — which
breaks the contract that retrieval keys are yaw-invariant. The principled
phase-aware attention path is to use yaw-invariant *node* features
(``|z|^2`` and friends) for the conv input, and to express phase agreement
between two nodes through an additive **edge logit** that is yaw-invariant
by construction (the IFFT-peak coherence in
:mod:`encoding.phase_coherence`).
"""

from __future__ import annotations

from typing import Tuple

import torch


def _reshape_complex(
    x_phase: torch.Tensor, n_rows: int, n_freqs: int
) -> torch.Tensor:
    """Flat ``[real || imag]`` layout → ``(..., rows, freqs)`` complex tensor."""
    half = n_rows * n_freqs
    if x_phase.shape[-1] < 2 * half:
        raise ValueError(
            f"x_phase has trailing dim {x_phase.shape[-1]}, "
            f"need at least {2 * half} for n_rows={n_rows}, n_freqs={n_freqs}"
        )
    real = x_phase[..., :half].reshape(*x_phase.shape[:-1], n_rows, n_freqs)
    imag = x_phase[..., half : 2 * half].reshape(*x_phase.shape[:-1], n_rows, n_freqs)
    return torch.complex(real, imag)


def power_features(
    x_phase: torch.Tensor, n_rows: int, n_freqs: int
) -> torch.Tensor:
    """Yaw-invariant per-node features: ``log(1 + |z|^2)`` flattened.

    ``|z|^2`` is exactly invariant under any per-node azimuthal rotation, so
    every downstream linear/conv operation that consumes this tensor stays
    yaw-invariant. The ``log1p`` compresses the dynamic range so that high-
    energy frequencies do not dominate the attention scores.

    Returns
    -------
    Real ``(N, n_rows * n_freqs)`` tensor.
    """
    z = _reshape_complex(x_phase, n_rows, n_freqs)
    return torch.log1p(z.abs().square()).reshape(*x_phase.shape[:-1], n_rows * n_freqs)


def bispectrum_features(
    x_phase: torch.Tensor,
    n_rows: int,
    n_freqs: int,
    pairs: Tuple[Tuple[int, int], ...] | None = None,
) -> torch.Tensor:
    """Yaw-invariant per-node bispectrum coefficients, real and imag stacked.

    Bispectrum: ``B[r; k1, k2] = z[r,k1] · z[r,k2] · conj(z[r,k1+k2])``.
    Under per-node rotation ``z[r,k] ↦ z[r,k] · exp(-2πi · k · n / A)`` the
    three exponents sum to zero, so ``B`` is exactly yaw-invariant.

    Args:
        pairs: optional ``((k1, k2), …)`` index list (1-based). Only pairs
            with ``k1 + k2 <= n_freqs`` are kept. If ``None`` we enumerate
            all such ordered pairs with ``k1 <= k2``.

    Returns
    -------
    Real ``(N, n_rows * 2 * P)`` tensor where ``P`` is the number of valid
    pairs.
    """
    if pairs is None:
        pairs = tuple(
            (k1, k2)
            for k1 in range(1, n_freqs + 1)
            for k2 in range(k1, n_freqs + 1)
            if k1 + k2 <= n_freqs
        )
    if not pairs:
        return x_phase.new_zeros(*x_phase.shape[:-1], 0)

    z = _reshape_complex(x_phase, n_rows, n_freqs)                 # (N, R, K)
    # 1-based pair indices → 0-based array indices; z does NOT include DC,
    # so frequency k corresponds to index k-1.
    parts = []
    for k1, k2 in pairs:
        if k1 + k2 > n_freqs:
            continue
        b = z[..., k1 - 1] * z[..., k2 - 1] * z[..., k1 + k2 - 1].conj()  # (N, R)
        parts.append(b)
    if not parts:
        return x_phase.new_zeros(*x_phase.shape[:-1], 0)
    bi = torch.stack(parts, dim=-1)                                # (N, R, P) complex
    feat = torch.cat([bi.real, bi.imag], dim=-1)                   # (N, R, 2P)
    return feat.reshape(*x_phase.shape[:-1], -1)


def phase_invariant_features(
    x_phase: torch.Tensor,
    n_rows: int,
    n_freqs: int,
    use_bispectrum: bool = True,
) -> torch.Tensor:
    """Concatenated yaw-invariant per-node feature vector for the phase stream.

    Components:

    * ``log(1 + |z|^2)`` (always)
    * bispectrum coefficients if ``use_bispectrum`` (typically a few dozens
      of features per row).

    Returns
    -------
    Real ``(N, F)`` tensor with ``F`` reported by :func:`feature_dim`.
    """
    parts = [power_features(x_phase, n_rows, n_freqs)]
    if use_bispectrum:
        parts.append(bispectrum_features(x_phase, n_rows, n_freqs))
    return torch.cat(parts, dim=-1)


def feature_dim(n_rows: int, n_freqs: int, use_bispectrum: bool = True) -> int:
    """Output dim of :func:`phase_invariant_features` for the given config."""
    dim = n_rows * n_freqs
    if use_bispectrum:
        n_pairs = sum(
            1
            for k1 in range(1, n_freqs + 1)
            for k2 in range(k1, n_freqs + 1)
            if k1 + k2 <= n_freqs
        )
        dim += n_rows * 2 * n_pairs
    return dim
