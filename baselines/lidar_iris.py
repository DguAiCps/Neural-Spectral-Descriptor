"""
LiDAR-Iris baseline — original binary + Hamming + column-shift formulation.

Reference:
    Wang, Sun, Xu, Sarma, Yang, Kong. "LiDAR Iris for Loop-Closure Detection",
    IROS 2020. arXiv:1911.08488. Reference impl: JoestarK/LiDAR-Iris.

Pipeline (paper §III-A through §III-C):
    1. Iris image (paper §III-A): 80×360 polar BEV with 8-bit height encoding.
       For each cell, divide [z_min, z_max] into 8 z-strata; bit k = 1 iff a
       point falls in stratum k. The cell value is uint8 in [0, 255].
    2. Per-row 1D Log-Gabor (paper §III-B Eq.6): apply 4 wavelength scales to
       each 360-column row of the iris image. The complex response yields one
       binary template per scale via real-part and imag-part sign thresholds
       — total 4 × 2 = 8 binary templates of shape (80, 360) per scan.
    3. Matching (paper §III-C Eq.10): minimum Hamming distance over column
       shifts in {-50, ..., +50} (= ±50°) for rotation invariance.

Coarse pre-filter (engineering choice): a 640D rotation-invariant ring-mean
signature drives a FAISS cosine pre-filter to top-200 candidates; the full
Hamming + column-shift rerank is then applied only on those candidates.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from baselines import register
from baselines.base import BaselineEncoder


def _log_gabor_1d(n: int, wavelength: float, sigma_f: float = 0.55) -> np.ndarray:
    """1D Log-Gabor bandpass filter in the frequency domain (rfft bins)."""
    freqs = np.fft.rfftfreq(n).astype(np.float64)
    center = 1.0 / wavelength
    freqs[0] = 1e-10
    log_gabor = np.exp(-(np.log(freqs / center)) ** 2 / (2.0 * sigma_f ** 2))
    log_gabor[0] = 0.0  # zero DC
    return log_gabor.astype(np.float64)


def _build_iris_8bit(
    points: np.ndarray,
    n_rings: int = 80,
    n_sectors: int = 360,
    max_range: float = 80.0,
    z_min: float = -1.0,
    z_max: float = 5.0,
) -> np.ndarray:
    """
    Build 8-bit iris image (paper §III-A).

    Each cell [r, s] is a uint8 whose bit k is set iff there exists a point
    with z in stratum k of the (r, s) BEV cell. 8 z-strata uniformly span
    [z_min, z_max].
    """
    iris = np.zeros((n_rings, n_sectors), dtype=np.uint8)

    xyz = points[:, :3].astype(np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x ** 2 + y ** 2)
    valid = (r > 0.5) & (r < max_range) & (z >= z_min) & (z < z_max)
    if not valid.any():
        return iris
    x, y, z, r = x[valid], y[valid], z[valid], r[valid]

    ring_idx = np.clip((r / max_range * n_rings).astype(np.int64), 0, n_rings - 1)
    theta = (np.arctan2(y, x) + np.pi) / (2.0 * np.pi)
    sector_idx = np.clip((theta * n_sectors).astype(np.int64), 0, n_sectors - 1)
    z_bin = np.clip(((z - z_min) / (z_max - z_min) * 8).astype(np.int64), 0, 7)

    bit_values = (np.uint8(1) << z_bin.astype(np.uint8))
    np.bitwise_or.at(iris, (ring_idx, sector_idx), bit_values)
    return iris


def _make_templates(
    iris_img: np.ndarray,
    wavelengths: Tuple[float, ...] = (18.0, 24.0, 32.0, 42.0),
    sigma_f: float = 0.55,
) -> np.ndarray:
    """
    Apply per-row 1D Log-Gabor at multiple scales, threshold real/imag parts
    by sign to produce binary templates (paper §III-B).

    Args:
        iris_img: (H, W) uint8 image.
        wavelengths: 4 wavelengths.
        sigma_f: log-domain bandwidth.

    Returns:
        (8, H, W) uint8 binary templates (values in {0, 1}). Template layout:
        [scale0_real, scale0_imag, scale1_real, scale1_imag, ...].
    """
    h, w = iris_img.shape
    iris_f = iris_img.astype(np.float64) / 255.0
    iris_fft = np.fft.rfft(iris_f, axis=1)  # (H, W//2+1) complex

    templates = np.empty((2 * len(wavelengths), h, w), dtype=np.uint8)
    for s, wl in enumerate(wavelengths):
        f = _log_gabor_1d(w, wl, sigma_f)
        filtered = iris_fft * f[None, :]
        response = np.fft.irfft(filtered, n=w, axis=1)
        # Real part: just the filtered signal. Imag part: Hilbert transform-
        # like quadrature (paper uses both for orthogonal phase information).
        # We get imag via odd-symmetric filter: multiply rfft bins by -1j.
        response_im = np.fft.irfft(filtered * (-1j), n=w, axis=1)
        templates[2 * s] = (response > 0).astype(np.uint8)
        templates[2 * s + 1] = (response_im > 0).astype(np.uint8)
    return templates


def _coarse_signature(templates: np.ndarray) -> np.ndarray:
    """
    Rotation-invariant 640D pre-filter signature: per-template per-ring mean
    of binary values. Sector-axis mean is shift-invariant.
    """
    return templates.mean(axis=2).reshape(-1).astype(np.float32)


def _hamming_batched_min_shift(
    t_q: np.ndarray, t_cands: np.ndarray, max_shift: int = 50
) -> np.ndarray:
    """
    Batched min-Hamming-over-shift between one query template stack and an
    array of candidate template stacks.

    Args:
        t_q: (K, H, W) uint8 query templates.
        t_cands: (B, K, H, W) uint8 candidate templates.
        max_shift: max column shift in the rotation alignment (paper §III-C).

    Returns:
        (B,) float — normalized Hamming distance per candidate.
    """
    w = t_q.shape[-1]
    total_bits = t_q.size

    t_q_f = t_q.astype(np.float32)
    t_c_f = t_cands.astype(np.float32)

    F_q = np.fft.rfft(t_q_f, axis=-1)            # (K, H, n_freq)
    F_c = np.fft.rfft(t_c_f, axis=-1)            # (B, K, H, n_freq)
    cross = np.fft.irfft(F_c * np.conj(F_q[None]), n=w, axis=-1)  # (B, K, H, W)
    inner = cross.sum(axis=(1, 2))                # (B, W)
    sum_q = float(t_q_f.sum())
    sum_c = t_c_f.sum(axis=(1, 2, 3))             # (B,)
    counts = sum_q + sum_c[:, None] - 2.0 * inner  # (B, W)

    if max_shift >= w // 2:
        best = counts.min(axis=1)
    else:
        idx = np.concatenate([
            np.arange(0, max_shift + 1),
            np.arange(w - max_shift, w),
        ])
        best = counts[:, idx].min(axis=1)
    return best.astype(np.float32) / total_bits


def _hamming_min_shift(
    t1: np.ndarray, t2: np.ndarray, max_shift: int = 50
) -> float:
    """
    Minimum normalized Hamming distance over column shifts in [-max_shift,
    +max_shift] (paper §III-C Eq.10).

    Computed in O(K * H * W * log W) via rfft cross-correlation along the
    sector axis. For binary 0/1 templates,
        bitcount(t1 XOR roll(t2, tau)) = sum(t1) + sum(t2) - 2 * inner(tau)
    where inner(tau) = sum over (template, ring, sector) of
        t1[s] * t2[(s - tau) mod W]
    is computed for ALL tau in O(W log W) per (template, ring) row via
        irfft(F1 * conj(F2)).
    """
    w = t1.shape[-1]
    total_bits = t1.size

    t1f = t1.astype(np.float32)
    t2f = t2.astype(np.float32)
    F1 = np.fft.rfft(t1f, axis=-1)
    F2 = np.fft.rfft(t2f, axis=-1)
    # irfft(F2 * conj(F1))[..., tau] = sum_s t2[..., s] * t1[..., (s + tau) mod W]
    #                                = sum_s' t1[..., s'] * t2[..., (s' - tau) mod W]
    # which equals inner(t1, roll(t2, tau)) along the last axis.
    cross = np.fft.irfft(F2 * np.conj(F1), n=w, axis=-1)
    # Sum over template + ring axes -> (W,) per-shift inner.
    sum_axes = tuple(range(cross.ndim - 1))
    inner = cross.sum(axis=sum_axes)  # (W,)
    sum_t1 = float(t1f.sum())
    sum_t2 = float(t2f.sum())
    counts_all = sum_t1 + sum_t2 - 2.0 * inner  # (W,)

    if max_shift >= w // 2:
        best = counts_all.min()
    else:
        # Only consider shifts in [-max_shift, max_shift].
        # In FFT convention, shift tau in [0, W) maps to: positive tau (mod W)
        # corresponds to roll-by-tau. Taus in (-max_shift, 0) are W - |tau|.
        idx = np.concatenate([
            np.arange(0, max_shift + 1),
            np.arange(w - max_shift, w),
        ])
        best = counts_all[idx].min()
    return float(best) / total_bits


@register
class LiDARIris(BaselineEncoder):
    """
    LiDAR-Iris (paper-faithful binary + Hamming + col-shift).
    """

    def __init__(
        self,
        n_rings: int = 80,
        n_sectors: int = 360,
        max_range: float = 80.0,
        z_min: float = -1.0,
        z_max: float = 5.0,
        wavelengths: Tuple[float, ...] = (18.0, 24.0, 32.0, 42.0),
        sigma_f: float = 0.55,
        max_shift: int = 50,
        n_coarse: int = 200,
    ):
        self.n_rings = n_rings
        self.n_sectors = n_sectors
        self.max_range = max_range
        self.z_min = z_min
        self.z_max = z_max
        self.wavelengths = wavelengths
        self.sigma_f = sigma_f
        self.max_shift = max_shift
        self.n_coarse = n_coarse

    @property
    def name(self) -> str:
        return "LiDAR-Iris"

    @property
    def short_name(self) -> str:
        return "lidar_iris"

    @property
    def descriptor_dim(self) -> int:
        # Reported dim is the 640D coarse pre-filter signature; the rerank
        # uses the full (8, n_rings, n_sectors) binary template stack
        # (8×80×360 bits = 28,800 bytes).
        return 2 * len(self.wavelengths) * self.n_rings

    def encode(self, points: np.ndarray) -> np.ndarray:
        iris = _build_iris_8bit(
            points, self.n_rings, self.n_sectors,
            self.max_range, self.z_min, self.z_max,
        )
        templates = _make_templates(iris, self.wavelengths, self.sigma_f)
        sig = _coarse_signature(templates)
        norm = np.linalg.norm(sig)
        if norm > 1e-8:
            sig = sig / norm
        return sig.astype(np.float32)

    def encode_with_aux(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        iris = _build_iris_8bit(
            points, self.n_rings, self.n_sectors,
            self.max_range, self.z_min, self.z_max,
        )
        templates = _make_templates(iris, self.wavelengths, self.sigma_f)
        sig = _coarse_signature(templates)
        norm = np.linalg.norm(sig)
        sig_norm = sig / norm if norm > 1e-8 else np.zeros_like(sig)
        # Precompute rfft along sector axis once per scan: reused by every
        # rerank pair the scan participates in.
        F_t = np.fft.rfft(templates.astype(np.float32), axis=-1).astype(
            np.complex64
        )
        sum_t = float(templates.sum())
        return sig_norm.astype(np.float32), {
            'templates': templates,
            'fft_templates': F_t,
            'template_sum': sum_t,
        }

    def compute_recalls(
        self,
        point_clouds: List[np.ndarray],
        poses: np.ndarray,
        k_values: List[int] = [1, 5, 10],
        distance_threshold: float = 5.0,
        skip_frames: int = 30,
        per_query_records=None,
    ):
        from baselines.eval_utils import compute_recall_cosine_then_rerank

        descriptors, aux_list = self.encode_sequence_with_aux(point_clouds)
        F_templates = np.stack(
            [a['fft_templates'] for a in aux_list], axis=0
        )  # (N, K, H, n_freq) complex64
        sums = np.array([a['template_sum'] for a in aux_list], dtype=np.float32)
        n_bits = aux_list[0]['templates'].size  # K * H * W
        w = aux_list[0]['templates'].shape[-1]
        max_shift = self.max_shift

        if max_shift >= w // 2:
            shift_idx = None
        else:
            shift_idx = np.concatenate([
                np.arange(0, max_shift + 1),
                np.arange(w - max_shift, w),
            ])

        def rerank_fn(query_idx: int, candidates: np.ndarray) -> np.ndarray:
            F_q = F_templates[query_idx]                # (K, H, n_freq)
            F_c = F_templates[candidates]               # (B, K, H, n_freq)
            cross = np.fft.irfft(
                F_c * np.conj(F_q[None]), n=w, axis=-1
            )  # (B, K, H, W)
            inner = cross.sum(axis=(1, 2))              # (B, W)
            counts = sums[query_idx] + sums[candidates, None] - 2.0 * inner
            if shift_idx is None:
                best = counts.min(axis=1)
            else:
                best = counts[:, shift_idx].min(axis=1)
            distances = (best / n_bits).astype(np.float32)
            return candidates[np.argsort(distances)]

        return compute_recall_cosine_then_rerank(
            descriptors, rerank_fn, poses,
            k_values=k_values,
            distance_threshold=distance_threshold,
            skip_frames=skip_frames,
            n_coarse=self.n_coarse,
            per_query_records=per_query_records,
        )
