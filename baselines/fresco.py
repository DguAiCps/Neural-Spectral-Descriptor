"""
FreSCo baseline — Frequency-domain Scan Context.

Rotation invariance via 2D FFT magnitude of the Scan Context BEV polar grid.
Rotation in physical space = column shift in SC matrix = phase shift in FFT.
Taking magnitude discards phase, achieving exact rotation invariance.

Key difference from NSC:
- FreSCo: BEV polar grid (max z-height) → 2D FFT → magnitude → 620D
- NSC: Range image (16,360) → per-row 1D FFT → exp binning → 256D
"""

import numpy as np
from baselines.base import BaselineEncoder
from baselines.scan_context import build_scan_context
from baselines import register


@register
class FreSCo(BaselineEncoder):
    """FreSCo: 2D FFT magnitude of Scan Context matrix."""

    def __init__(self, n_rings=20, n_sectors=60, max_range=80.0, z_min=-3.0):
        self.n_rings = n_rings
        self.n_sectors = n_sectors
        self.max_range = max_range
        self.z_min = z_min
        self._n_freq_cols = n_sectors // 2 + 1  # rfft2 output

    @property
    def name(self):
        return "FreSCo"

    @property
    def short_name(self):
        return "fresco"

    @property
    def descriptor_dim(self):
        return self.n_rings * self._n_freq_cols

    def encode(self, points):
        sc = build_scan_context(
            points, self.n_rings, self.n_sectors, self.max_range, self.z_min
        )
        # 2D real FFT → magnitude (rotation invariant)
        fft_2d = np.fft.rfft2(sc)  # (n_rings, n_sectors//2+1)
        magnitude = np.abs(fft_2d).astype(np.float32)

        descriptor = magnitude.flatten()
        norm = np.linalg.norm(descriptor)
        if norm > 1e-8:
            descriptor = descriptor / norm
        else:
            descriptor = np.zeros(self.descriptor_dim, dtype=np.float32)
        return descriptor
