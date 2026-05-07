"""
FreSCo baseline — Frequency-Domain Scan Context with Fourier-Mellin
translation+rotation invariance.

Reference:
    Fan, He, Lu, Liu. "FreSCo: Frequency-Domain Scan Context for LiDAR-based
    Place Recognition with Translation and Rotation Invariance", ICARCV 2022.
    arXiv:2206.12628. Reference impl: soytony/FreSCo.

Pipeline (paper §III-B and §III-C):
    1. SC matrix (max-z BEV polar grid)         — paper §III-A baseline.
    2. M = |FFT2(SC)|                           — rotation invariance (§III-B).
    3. M_lp = log_polar_warp(M)                 — radial scaling/translation
                                                  becomes horizontal shift.
    4. D = |rFFT2(M_lp)|                        — second FFT magnitude is
                                                  shift-invariant → also
                                                  translation invariant.
    5. log(1 + D) for dynamic-range compression — paper §III-C.

Output dim is preserved at 620 by setting the log-polar grid to (20, 60) so
that rFFT2 yields (20, 31) = 620.
"""

import numpy as np

from baselines import register
from baselines.base import BaselineEncoder
from baselines.scan_context import build_scan_context


def _log_polar_warp(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Log-polar warp via numpy bilinear sampling. No OpenCV dependency.

    For each output pixel (rho_idx, phi_idx) in the (out_h, out_w) grid:
        phi = 2*pi * phi_idx / out_w
        rho = exp(rho_idx * log_max / (out_h - 1))   in [1, max_radius]
        sample at (cy + rho*sin(phi), cx + rho*cos(phi))

    The log-polar transform turns a radial scale/translation in the input
    image into a horizontal shift in the output, so the magnitude of a 2nd
    FFT becomes invariant to that shift.
    """
    h, w = image.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_radius = min(cy, cx)
    if max_radius < 1.0:
        max_radius = max(cy, cx)

    rho_idx = np.arange(out_h, dtype=np.float64)
    phi_idx = np.arange(out_w, dtype=np.float64)
    log_max = np.log(max_radius + 1.0)
    rho = np.exp(rho_idx * log_max / max(out_h - 1, 1)) - 1.0  # [0, max_radius]
    phi = 2.0 * np.pi * phi_idx / out_w

    # Sampling coordinates: (out_h, out_w)
    cos_phi = np.cos(phi)[None, :]
    sin_phi = np.sin(phi)[None, :]
    rho_grid = rho[:, None]
    sample_y = cy + rho_grid * sin_phi
    sample_x = cx + rho_grid * cos_phi

    # Bilinear sampling with edge clamping.
    y0 = np.floor(sample_y).astype(np.int64)
    x0 = np.floor(sample_x).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = sample_y - y0
    wx = sample_x - x0

    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)

    out = (
        image[y0, x0] * (1 - wy) * (1 - wx)
        + image[y0, x1] * (1 - wy) * wx
        + image[y1, x0] * wy * (1 - wx)
        + image[y1, x1] * wy * wx
    )
    return out.astype(np.float32)


@register
class FreSCo(BaselineEncoder):
    """FreSCo: 2D FFT magnitude of Scan Context + Fourier-Mellin transform."""

    def __init__(self, n_rings=20, n_sectors=60, max_range=80.0, z_min=-3.0,
                 lp_h: int = 20, lp_w: int = 60):
        self.n_rings = n_rings
        self.n_sectors = n_sectors
        self.max_range = max_range
        self.z_min = z_min
        # Log-polar grid: (lp_h, lp_w). Setting lp_w=60 keeps the rFFT2 output
        # size at (lp_h, lp_w//2 + 1) = (20, 31) = 620 — same dimension as the
        # rotation-only variant, so paper Table tab:results column "Dim=620"
        # is preserved.
        self.lp_h = lp_h
        self.lp_w = lp_w
        self._n_freq_cols = lp_w // 2 + 1

    @property
    def name(self):
        return "FreSCo"

    @property
    def short_name(self):
        return "fresco"

    @property
    def descriptor_dim(self):
        return self.lp_h * self._n_freq_cols

    def encode(self, points):
        sc = build_scan_context(
            points, self.n_rings, self.n_sectors, self.max_range, self.z_min
        )
        # 2D FFT magnitude of SC (paper §III-B): rotation invariance via the
        # circular shift theorem. Use full FFT (not rfft) and fftshift to put
        # the DC at center, which makes the subsequent log-polar warp pivot
        # around the (rotation/scale) symmetry center.
        fft_full = np.fft.fft2(sc)
        magnitude = np.abs(np.fft.fftshift(fft_full)).astype(np.float32)

        # Log-polar warp (paper §III-C): radial translation/scale → horizontal
        # shift in the warped image.
        m_lp = _log_polar_warp(magnitude, self.lp_h, self.lp_w)

        # Second FFT magnitude over log-polar image: rFFT2 keeps unique
        # frequency content (real input → conjugate symmetric FFT), giving
        # (lp_h, lp_w//2 + 1) bins. The magnitude is invariant to horizontal
        # shifts in the log-polar image, hence translation/scale-invariant in
        # the original BEV.
        d = np.abs(np.fft.rfft2(m_lp)).astype(np.float32)
        d_log = np.log1p(d)  # paper §III-C dynamic-range compression

        descriptor = d_log.flatten()
        norm = np.linalg.norm(descriptor)
        if norm > 1e-8:
            descriptor = descriptor / norm
        else:
            descriptor = np.zeros(self.descriptor_dim, dtype=np.float32)
        return descriptor.astype(np.float32)
