"""Closed-form phase-correlation edge bias.

Computes a deterministic, yaw-invariant edge logit from two nodes' compact
Fourier phase sketches. Stacks alongside the learned ``PhaseEdgeBias`` so
that the GAT receives both a math-guaranteed phase-agreement signal and a
learned correction term.

Why a peak-of-IFFT — not a raw inner product
-------------------------------------------
For two scans of the same place differing by an integer azimuthal shift
``n0`` (yaw rotation) one has::

    F_b[r, k] = F_a[r, k] · exp(-2πi · k · n0 / A)

so the per-frequency conjugate product carries the rotation as a phase ramp::

    F_a[r, k] · conj(F_b[r, k]) = |F_a[r, k]|^2 · exp(2πi · k · n0 / A)

A naive sum ``|Σ_k F_a · F_b*|`` therefore depends on ``n0`` and is **not**
yaw-invariant. The fix is to take the inverse FFT of the cross spectrum and
read off its peak magnitude::

    peak[r] = max_n | IFFT_n( F_a[r, :] · conj(F_b[r, :]) ) |

For matched scenes (regardless of yaw) the peak is concentrated at the bin
that corresponds to ``n0`` and has magnitude ``≈ 1`` (after L2 normalization);
for unrelated scenes the peak is ``O(1/√K)``. The peak is yaw-invariant by
construction — the shift only relocates the peak, not its magnitude.

Variants
--------
* ``mode="poc"`` (default): phase-only correlation. Each cross spectrum entry
  is normalized to unit magnitude *before* IFFT, so the result is sensitive to
  phase agreement only. Yields a sharp delta at the matching shift; gives
  near-1 peak for matched scenes regardless of energy distribution.
* ``mode="ncc"``: normalized cross-correlation. The peak is the cosine
  similarity of the phase-aligned signals, weighted by their energy spectra.

The module is parameter-free. It expects ``x_phase`` laid out as
``[real_part || imag_part]`` over a flat ``(n_rows, n_freqs)`` grid — the
layout produced by :func:`encoding.phase_features._complex_features`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def reshape_complex(x_phase: torch.Tensor, n_rows: int, n_freqs: int) -> torch.Tensor:
    """Flat real ``[real | imag]`` tensor → complex ``(N, rows, freqs)`` tensor."""
    half = n_rows * n_freqs
    if x_phase.shape[-1] < 2 * half:
        raise ValueError(
            f"x_phase has trailing dim {x_phase.shape[-1]}, "
            f"need at least {2 * half} for complex layout "
            f"(n_rows={n_rows}, n_freqs={n_freqs})"
        )
    real = x_phase[..., :half].reshape(*x_phase.shape[:-1], n_rows, n_freqs)
    imag = x_phase[..., half : 2 * half].reshape(*x_phase.shape[:-1], n_rows, n_freqs)
    return torch.complex(real, imag)


def phase_correlation_peak(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    pad_factor: int = 4,
    mode: str = "poc",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-row peak of cross-correlation magnitude, mean over rows.

    Args:
        z_a, z_b: complex ``(..., rows, freqs)``.
        pad_factor: zero-pad multiple for the IFFT (peak resolution).
        mode: ``"poc"`` for phase-only correlation, ``"ncc"`` for energy-weighted.

    Returns:
        Real ``(...,)`` in ``[0, 1]`` (poc), ``[0, 1]`` (ncc).
    """
    cross = z_a * z_b.conj()                                  # (..., rows, K)

    if mode == "poc":
        cross = cross / (cross.abs() + eps)                   # phase-only
    elif mode != "ncc":
        raise ValueError(f"Unknown mode {mode!r}; expected 'poc' or 'ncc'.")

    K = cross.shape[-1]
    L = max(K * max(int(pad_factor), 1), K)
    if L > K:
        cross = F.pad(cross, (0, L - K))

    corr = torch.fft.ifft(cross, dim=-1)                      # (..., rows, L)
    peak = corr.abs().amax(dim=-1)                            # (..., rows)

    if mode == "poc":
        # IFFT with norm='backward' returns Σ_k g_k exp(2πi k n / L) / L. A
        # phase-only signal (|g_k|=1) aligned at some shift n* therefore
        # peaks at K / L. Rescale by L/K so matched ⇒ ~1.
        peak = peak * (L / max(K, 1))
        sim = peak.clamp(0.0, 1.0)
    else:  # ncc
        # Same IFFT 1/L factor, but the energy reference is now ||F_a|| ||F_b||
        # measured over the budget freqs. Matched scenes have peak·L = ||F||²
        # so sim → 1; unrelated peaks scale as 1/√K.
        norm_a = z_a.abs().square().sum(dim=-1).sqrt()
        norm_b = z_b.abs().square().sum(dim=-1).sqrt()
        sim = (peak * L) / (norm_a * norm_b + eps)
        sim = sim.clamp(0.0, 1.0)

    return sim.mean(dim=-1)


class ClosedFormPhaseEdgeBias(nn.Module):
    """Parameter-free phase coherence → per-edge attention bias.

    Args:
        n_rows: rows in the underlying complex phase sketch.
        n_freqs: per-row frequency budget.
        scale: maximum absolute logit. ``c=1`` → ``+scale``; ``c=0`` → ``-scale``.
        mode: ``"poc"`` (phase-only) or ``"ncc"``.
        pad_factor: zero-pad multiple for IFFT resolution.
        similarity_only: if ``True`` and ``edge_type`` is provided, the bias is
            only emitted for similarity edges (``edge_type == 1``); temporal
            edges receive zero bias so the existing temporal prior is unchanged.
        center: if ``True`` (default), map ``c → 2c − 1`` so the bias is
            zero-mean; if ``False``, return ``scale * c`` directly (always
            non-negative — boosts in-scene candidates only).
    """

    def __init__(
        self,
        n_rows: int,
        n_freqs: int,
        scale: float = 2.0,
        mode: str = "poc",
        pad_factor: int = 4,
        similarity_only: bool = True,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.n_rows = int(n_rows)
        self.n_freqs = int(n_freqs)
        self.scale = float(scale)
        self.mode = str(mode)
        self.pad_factor = int(pad_factor)
        self.similarity_only = bool(similarity_only)
        self.center = bool(center)

    def forward(
        self,
        x_phase: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = reshape_complex(x_phase, self.n_rows, self.n_freqs)
        src, tgt = edge_index[0], edge_index[1]
        c = phase_correlation_peak(
            z[src], z[tgt],
            pad_factor=self.pad_factor,
            mode=self.mode,
        )                                                    # (E,) ∈ [0, 1]
        if self.center:
            logit = self.scale * (2.0 * c - 1.0)
        else:
            logit = self.scale * c
        logit = logit.unsqueeze(-1)                          # (E, 1)
        if self.similarity_only and edge_type is not None:
            is_similarity = (edge_type == 1).unsqueeze(-1)
            logit = torch.where(is_similarity, logit, torch.zeros_like(logit))
        return logit
