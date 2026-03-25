"""
LiDAR-Iris baseline — adapted for cosine similarity evaluation.

Original paper:
    Wang et al., "LiDAR Iris for Loop-Closure Detection", IROS 2020.
    https://arxiv.org/abs/1911.08488

Method:
    1. Project point cloud onto a 2D binary "iris image" via cylindrical
       projection (elevation_bins × azimuth_bins).
       Rows = elevation angle bins (matching the paper's LiDAR beam structure).
       Columns = azimuth angle bins.
    2. Apply row-wise 1D Log-Gabor bandpass filters (3 wavelength scales).
    3. Compute FFT magnitude of each filtered row — rotation-invariant because
       a rotation = column shift = phase shift in FFT, and magnitude is
       phase-invariant.
    4. Subsample frequency bins and flatten to a fixed-length float descriptor.

Original paper's iris image:
    Row i = LiDAR beam channel i (fixed elevation angle per beam).
    For HDL-64E: 64 rows. For VLP-16: 16 rows.
    Here we unify sensors by using a fixed elevation range (elev_min, elev_max)
    divided into n_height_bins = 64 uniform angle bins, approximating the
    beam-channel structure in a sensor-agnostic way.
    Default range [-30°, 15°] covers HDL-64E, HDL-32E, and VLP-16.

Adaptation note:
    The original paper uses binary XOR encoding + Hamming distance matching
    with column-shift alignment for rotation invariance.
    Here we use the pre-binarization float magnitudes and FFT magnitude
    (phase-invariant) to enable cosine similarity in the FAISS framework.

Descriptor dimensions:
    n_height_bins × n_wavelengths × n_freq_keep = 64 × 3 × 30 = 5760D
"""

import numpy as np
from baselines.base import BaselineEncoder
from baselines import register


def _log_gabor_1d(n: int, wavelength: float, sigma_f: float = 0.75) -> np.ndarray:
    """
    1D Log-Gabor bandpass filter in the frequency domain (pure numpy).

    Args:
        n: Signal length (number of azimuth bins).
        wavelength: Filter center wavelength in pixels (e.g. 18 = 20 cycles/row).
        sigma_f: Bandwidth parameter (log-domain σ). Higher = narrower band.

    Returns:
        (n//2 + 1,) float32 filter magnitudes for rfft output bins.
    """
    freqs = np.fft.rfftfreq(n).astype(np.float64)  # [0, 1/n, ..., 1/2]
    center = 1.0 / wavelength

    # Avoid log(0) at DC bin
    freqs[0] = 1e-10

    log_gabor = np.exp(-(np.log(freqs / center)) ** 2 / (2.0 * sigma_f ** 2))
    log_gabor[0] = 0.0  # Zero DC component

    return log_gabor.astype(np.float32)


@register
class LiDARIris(BaselineEncoder):
    """
    LiDAR-Iris (float adaptation for cosine similarity).

    Elevation-angle-based iris image + row-wise Log-Gabor magnitude features.
    """

    def __init__(
        self,
        n_height_bins: int = 64,
        n_azimuth_bins: int = 360,
        max_range: float = 50.0,
        elev_min_deg: float = -30.0,
        elev_max_deg: float = 15.0,
        wavelengths: tuple = (18, 36, 72),
        sigma_f: float = 0.75,
        n_freq_keep: int = 30,
    ):
        self.n_height_bins = n_height_bins
        self.n_azimuth_bins = n_azimuth_bins
        self.max_range = max_range
        self._elev_min_rad = np.radians(elev_min_deg)
        self._elev_max_rad = np.radians(elev_max_deg)
        self._elev_range_rad = self._elev_max_rad - self._elev_min_rad
        self.wavelengths = wavelengths
        self.sigma_f = sigma_f
        self.n_freq_keep = n_freq_keep

        # Precompute Log-Gabor filters (shape: n_wavelengths × n_rfft_bins)
        n_rfft = n_azimuth_bins // 2 + 1
        self._filters = np.stack([
            _log_gabor_1d(n_azimuth_bins, wl, sigma_f)
            for wl in wavelengths
        ], axis=0)  # (n_wavelengths, n_rfft)

        # Subsample indices: n_freq_keep evenly-spaced points over (0, n_rfft-1]
        # Start from index 1 to exclude DC (already zeroed by filter, but avoids
        # wasting a bin on the guaranteed-zero DC component)
        self._freq_indices = np.round(
            np.linspace(1, n_rfft - 1, n_freq_keep)
        ).astype(int)

    @property
    def name(self) -> str:
        return "LiDAR-Iris"

    @property
    def short_name(self) -> str:
        return "lidar_iris"

    @property
    def descriptor_dim(self) -> int:
        return self.n_height_bins * len(self.wavelengths) * self.n_freq_keep

    def _build_iris_image(self, points: np.ndarray) -> np.ndarray:
        """
        Build binary iris image from point cloud using elevation angle rows.

        Each row corresponds to an elevation angle bin, mirroring the LiDAR
        beam-channel structure from the original paper (e.g. 64 rows for
        HDL-64E). Points are projected via cylindrical coordinates:
            elevation = arctan2(z, r_horizontal)
            azimuth   = arctan2(y, x)

        Args:
            points: (N, 3+) array [x, y, z, ...].

        Returns:
            (n_height_bins, n_azimuth_bins) float32 binary occupancy image.
        """
        iris = np.zeros((self.n_height_bins, self.n_azimuth_bins), dtype=np.float32)

        xyz = points[:, :3].astype(np.float64)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # Horizontal range filter
        r_horiz = np.sqrt(x ** 2 + y ** 2)
        valid = (r_horiz >= 0.5) & (r_horiz <= self.max_range)
        x, y, z, r_horiz = x[valid], y[valid], z[valid], r_horiz[valid]

        if len(x) == 0:
            return iris

        # Elevation angle: arctan2(z, r_horizontal) ∈ (-π/2, π/2)
        # Maps to row index within [elev_min_rad, elev_max_rad]
        elev_angle = np.arctan2(z, r_horiz)
        h_frac = (elev_angle - self._elev_min_rad) / self._elev_range_rad
        h_idx = (h_frac * self.n_height_bins).astype(int)
        # Clip: points outside elevation range are silently discarded
        in_elev_range = (h_idx >= 0) & (h_idx < self.n_height_bins)
        h_idx = h_idx[in_elev_range]
        x, y = x[in_elev_range], y[in_elev_range]

        if len(h_idx) == 0:
            return iris

        # Azimuth: atan2(y, x) ∈ (-π, π] → [0, n_azimuth_bins)
        azimuth = (np.arctan2(y, x) + np.pi) / (2.0 * np.pi)  # [0, 1)
        az_idx = (azimuth * self.n_azimuth_bins).astype(int) % self.n_azimuth_bins

        # Mark occupied cells
        iris[h_idx, az_idx] = 1.0

        return iris

    def encode(self, points: np.ndarray) -> np.ndarray:
        """
        Encode a single point cloud as a LiDAR-Iris float descriptor.

        Args:
            points: (N, 3) or (N, 4) point cloud.

        Returns:
            (descriptor_dim,) float32 L2-normalized descriptor.
        """
        iris = self._build_iris_image(points)

        # Batch FFT over all rows at once: (n_height_bins, n_rfft)
        row_fft_mag = np.abs(np.fft.rfft(iris, axis=1)).astype(np.float32)

        # Apply each Log-Gabor filter and subsample:
        # row_fft_mag: (H, F), filters: (W, F) → filtered: (H, W, F)
        # Since filter is real and ≥ 0: |FFT × filter| = |FFT| × filter
        filtered = row_fft_mag[:, np.newaxis, :] * self._filters[np.newaxis, :, :]
        # (H, n_wavelengths, n_rfft) → subsample freq axis
        subsampled = filtered[:, :, self._freq_indices]
        # Shape: (H, n_wavelengths, n_freq_keep) → flatten

        descriptor = subsampled.reshape(-1).astype(np.float32)

        norm = np.linalg.norm(descriptor)
        if norm > 1e-8:
            return descriptor / norm
        return np.zeros(self.descriptor_dim, dtype=np.float32)
