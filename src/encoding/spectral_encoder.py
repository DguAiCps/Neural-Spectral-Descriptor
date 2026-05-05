"""
Spectral Histogram Encoder - Algorithm 1

Implements the core Neural Spectral Codec algorithm:
1. Project point cloud to 2D image:
   - 'range_image': panoramic range image (n_elevation × 360), pooled to target_elevation_bins
   - 'bev': BEV polar grid (n_rings × 360) with 1m radial × 1° angular resolution
2. Apply row-wise 1D FFT along azimuth for rotation invariance
3. Aggregate FFT magnitudes into per-row histograms via exponential binning
4. Normalize and prepare for quantization

Key features:
- Rotation invariance via magnitude spectrum (phase discarded)
- Per-row histogram preserves spatial information
- Adaptive exponential frequency binning (learnable α parameter)
- BEV mode: sensor-agnostic (no elevation calibration needed)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from encoding.range_image import RangeImageProjector, interpolate_range_image
from encoding.bev_image import BEVProjector, interpolate_bev_image


VALID_BIN_STATISTICS = ('sum', 'mean', 'std', 'max', 'min')
VALID_INTER_BIN_STATISTICS = ('diff', 'ratio')


class SpectralEncoder(nn.Module):
    """
    Neural Spectral Histogram Encoder

    Compresses LiDAR point clouds to per-row spectral histograms.
    Supports two projection modes:
    - 'range_image': panoramic range image, pooled to target_elevation_bins rows
    - 'bev': BEV polar grid with 1m radial resolution (n_rows = max_range - min_range)

    Output dimension: n_rows × n_bins × n_stats + inter-bin terms
    """

    def __init__(
        self,
        n_elevation: int = 64,
        n_azimuth: int = 360,
        n_bins: int = 16,
        alpha: float = 2.0,
        learnable_alpha: bool = True,
        epsilon: float = 1e-8,
        target_elevation_bins: int = 16,
        interpolate_empty: bool = True,
        elevation_range: tuple = (-24.8, 2.0),
        bin_statistics: Optional[List[str]] = None,
        inter_bin_statistics: Optional[List[str]] = None,
        device: str = 'cpu',
        projection_type: str = 'range_image',
        max_range: float = 80.0,
        min_range: float = 1.0,
        z_min: float = -3.0,
        height_encoding: str = 'max',
        n_height_layers: int = 8,
        z_max: float = 5.0,
        zero_center: bool = False,
        log_magnitude: bool = False,
        binning_strategy: str = 'exponential',
        normalize_channels: bool = True
    ):
        """
        Initialize spectral encoder

        Args:
            n_elevation: Number of elevation rings (64 for HDL-64E)
            n_azimuth: Number of azimuth bins (360 for 1-degree resolution)
            n_bins: Number of histogram bins (16 for target compression)
            alpha: Exponential warping parameter for frequency binning
            learnable_alpha: If True, α is learned during training
            epsilon: Small constant for numerical stability
            target_elevation_bins: Target elevation bins for sensor-agnostic binning (16 for compatibility)
            interpolate_empty: If True, interpolate empty pixels before FFT (critical for sensor invariance)
            elevation_range: (min, max) elevation angles in degrees (sensor-specific, range_image only)
            bin_statistics: List of statistics to extract per bin.
                Valid: 'sum', 'mean', 'std', 'max', 'min'. Default: ['sum']
            inter_bin_statistics: List of inter-bin (adjacent-bin) statistics to compute.
                Applied to every base stat channel. Valid: 'diff', 'ratio'. Default: []
                - 'diff': diff[e,b] = stat[e,b+1] - stat[e,b]  (240D per base stat)
                - 'ratio': log(stat[e,b+1]+ε) - log(stat[e,b]+ε)  (240D per base stat)
            device: Device for tensor operations
            projection_type: 'range_image' or 'bev'. BEV uses polar grid with max z-height.
            max_range: Maximum LiDAR range in meters
            min_range: Minimum LiDAR range in meters
            z_min: Minimum z-height for BEV ground filtering
            zero_center: If True, subtract row mean before FFT to remove DC component
            log_magnitude: If True, use log(|FFT| + ε) for compressed dynamic range
            binning_strategy: 'exponential' (alpha-warped) or 'octave' (log2 scale, auto n_bins)
            normalize_channels: If True, apply per-channel L2 normalization (default).
                Set False to preserve magnitude for standardized Euclidean distance edges.
        """
        super().__init__()

        self.n_elevation = n_elevation
        self.n_azimuth = n_azimuth
        self.epsilon = epsilon
        self.binning_strategy = binning_strategy
        self.normalize_channels = normalize_channels

        # Octave binning: auto-compute n_bins from n_freqs
        n_freqs = n_azimuth // 2 + 1
        if binning_strategy == 'octave':
            # Bin edges: [0, 1, 2, 4, 8, ..., 2^k, n_freqs]
            power = 0
            while 2 ** power < n_freqs:
                power += 1
            self.n_bins = power + 1  # +1 for the DC bin [0, 1)
        else:
            self.n_bins = n_bins
        self.target_elevation_bins = target_elevation_bins
        self.interpolate_empty = interpolate_empty
        self._device = device
        self.projection_type = projection_type
        self.zero_center = zero_center
        self.log_magnitude = log_magnitude

        # Validate and store bin statistics
        if bin_statistics is None:
            bin_statistics = ['sum']
        for stat in bin_statistics:
            if stat not in VALID_BIN_STATISTICS:
                raise ValueError(f"Invalid bin statistic '{stat}'. Valid: {VALID_BIN_STATISTICS}")
        self.bin_statistics = bin_statistics
        self.n_stats = len(bin_statistics)

        # Validate and store inter-bin statistics
        if inter_bin_statistics is None:
            inter_bin_statistics = []
        for stat in inter_bin_statistics:
            if stat not in VALID_INTER_BIN_STATISTICS:
                raise ValueError(f"Invalid inter-bin statistic '{stat}'. Valid: {VALID_INTER_BIN_STATISTICS}")
        self.inter_bin_statistics = inter_bin_statistics
        self.n_inter_stats = len(inter_bin_statistics)

        # Learnable alpha parameter
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

        # Projector: BEV polar grid or range image
        if projection_type == 'bev':
            self.projector = BEVProjector(
                n_sectors=n_azimuth,
                max_range=max_range,
                min_range=min_range,
                z_min=z_min,
                height_encoding=height_encoding,
                n_height_layers=n_height_layers,
                z_max=z_max
            )
            # BEV: use full resolution (1m per row), no pooling
            self.target_elevation_bins = self.projector.n_rings
        else:
            self.projector = RangeImageProjector(
                n_elevation=n_elevation,
                n_azimuth=n_azimuth,
                elevation_range=elevation_range,
                max_range=max_range,
                min_range=min_range
            )

        # Precompute number of FFT frequencies
        # Real FFT outputs (n_azimuth // 2 + 1) frequencies
        self.n_freqs = n_azimuth // 2 + 1  # 181 for 360 azimuth bins

        # Output dimension: rows × bins × statistics
        self.base_dim = self.target_elevation_bins * self.n_bins
        self.inter_base_dim = self.target_elevation_bins * (self.n_bins - 1)
        self.output_dim = (self.base_dim * self.n_stats
                           + self.inter_base_dim * self.n_inter_stats * self.n_stats)

    def set_elevation_range(self, elevation_range: tuple):
        """Update elevation range for a different sensor (no-op for BEV)"""
        self.projector.set_elevation_range(elevation_range)

    def _compute_bin_edges(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency bin edges.

        Strategies:
            'exponential': Maps [0, n_freqs] using exponential warping with alpha
            'octave': Log2 scale bins [0, 1, 2, 4, 8, ..., n_freqs]

        Args:
            alpha: Warping parameter (used only for exponential)

        Returns:
            (n_bins + 1,) bin edges in frequency space
        """
        if self.binning_strategy == 'octave':
            edges = [0]
            power = 0
            while 2 ** power < self.n_freqs:
                edges.append(2 ** power)
                power += 1
            edges.append(self.n_freqs)
            return torch.tensor(edges, dtype=torch.float32, device=alpha.device)

        # Exponential warping
        t = torch.linspace(0, 1, self.n_bins + 1, device=alpha.device)
        bin_edges = (torch.exp(alpha * t) - 1) / (torch.exp(alpha) - 1 + self.epsilon)
        bin_edges = bin_edges * self.n_freqs
        return bin_edges

    def _bin_fft_magnitudes(
        self,
        fft_magnitudes: torch.Tensor,
        bin_edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Bin FFT magnitudes into per-elevation histograms using adaptive edges.
        Extracts multiple statistics per bin based on self.bin_statistics.

        Args:
            fft_magnitudes: (n_elevation, n_freqs) FFT magnitude spectrum
            bin_edges: (n_bins + 1,) bin edges in frequency space

        Returns:
            (n_stats * n_elevation * n_bins,) flattened multi-stat histograms.
            Statistics are concatenated channel-wise: [stat0_256D, stat1_256D, ...]
        """
        n_elevation = fft_magnitudes.shape[0]
        device = fft_magnitudes.device
        total_bins = n_elevation * self.n_bins

        # Frequency indices: (n_freqs,)
        freq_indices = torch.arange(self.n_freqs, dtype=torch.float32, device=device)

        # Assign each frequency to a bin using searchsorted (vectorized)
        bin_assignments = torch.searchsorted(bin_edges, freq_indices, right=True) - 1
        bin_assignments = torch.clamp(bin_assignments, 0, self.n_bins - 1)

        # Global bin indices: encode (elevation, bin) as a single index
        # offsets: [0, n_bins, 2*n_bins, ...] for each elevation
        offsets = torch.arange(n_elevation, device=device) * self.n_bins  # (n_elevation,)
        global_bins = bin_assignments.unsqueeze(0) + offsets.unsqueeze(1)  # (n_elevation, n_freqs)
        global_bins_flat = global_bins.reshape(-1).long()  # (n_elevation * n_freqs,)
        values_flat = fft_magnitudes.reshape(-1)  # (n_elevation * n_freqs,)

        # Precompute sums and counts (reused by sum, mean, std)
        need_sum = any(s in self.bin_statistics for s in ('sum', 'mean', 'std'))
        sums = None
        counts = None

        if need_sum:
            sums = torch.zeros(total_bins, device=device)
            sums.scatter_add_(0, global_bins_flat, values_flat)
            counts = torch.zeros(total_bins, device=device)
            counts.scatter_add_(0, global_bins_flat, torch.ones_like(values_flat))

        # Precompute means (reused by mean, std)
        means = None
        if any(s in self.bin_statistics for s in ('mean', 'std')):
            means = sums / counts.clamp(min=1)

        # Compute each requested statistic
        channels = []
        for stat_name in self.bin_statistics:
            if stat_name == 'sum':
                channels.append(sums)
            elif stat_name == 'mean':
                channels.append(means)
            elif stat_name == 'std':
                # Two-pass: scatter (x - mean)^2, then sqrt
                per_freq_means = means[global_bins_flat]
                sq_diffs = (values_flat - per_freq_means) ** 2
                variance = torch.zeros(total_bins, device=device)
                variance.scatter_add_(0, global_bins_flat, sq_diffs)
                variance = variance / counts.clamp(min=1)
                channels.append(torch.sqrt(variance + self.epsilon))
            elif stat_name == 'max':
                max_vals = torch.full((total_bins,), -float('inf'), device=device)
                max_vals.scatter_reduce_(0, global_bins_flat, values_flat, reduce='amax', include_self=False)
                max_vals[max_vals == -float('inf')] = 0.0
                channels.append(max_vals)
            elif stat_name == 'min':
                min_vals = torch.full((total_bins,), float('inf'), device=device)
                min_vals.scatter_reduce_(0, global_bins_flat, values_flat, reduce='amin', include_self=False)
                min_vals[min_vals == float('inf')] = 0.0
                channels.append(min_vals)

        # Inter-bin channels: computed from intra-bin channels only (snapshot before appending)
        if self.inter_bin_statistics:
            intra_channels = list(channels)  # snapshot: only base stat channels (256D each)
            for inter_type in self.inter_bin_statistics:
                for ch in intra_channels:
                    hist_2d = ch.view(n_elevation, self.n_bins)  # (n_elev, n_bins)
                    if inter_type == 'diff':
                        inter_ch = hist_2d[:, 1:] - hist_2d[:, :-1]  # (n_elev, n_bins-1)
                    else:  # 'ratio'
                        inter_ch = (torch.log(hist_2d[:, 1:] + self.epsilon)
                                    - torch.log(hist_2d[:, :-1] + self.epsilon))
                    channels.append(inter_ch.flatten())  # (n_elev*(n_bins-1),)

        # Concatenate channels: (n_stats * total_bins + n_inter_stats * n_stats * inter_total,)
        return torch.cat(channels, dim=0)

    @staticmethod
    def _compute_spectral_entropy(fft_magnitudes: torch.Tensor, epsilon: float = 1e-8) -> float:
        """Compute spectral entropy from FFT magnitudes.

        Measures how uniformly energy is distributed across frequencies.
        Low entropy = energy concentrated in few frequencies = repetitive structure.
        High entropy = energy spread across frequencies = unique structure.

        Args:
            fft_magnitudes: (n_rows, n_freqs) FFT magnitude spectrum
            epsilon: Numerical stability constant

        Returns:
            Scalar spectral entropy averaged across rows (in nats).
        """
        # Power spectrum per row
        power = fft_magnitudes ** 2  # (n_rows, n_freqs)
        power_sum = power.sum(dim=1, keepdim=True).clamp(min=epsilon)
        # Normalize to probability distribution per row
        p = power / power_sum  # (n_rows, n_freqs)
        # Shannon entropy per row: H = -sum(p * log(p))
        log_p = torch.log(p + epsilon)
        entropy_per_row = -(p * log_p).sum(dim=1)  # (n_rows,)
        return float(entropy_per_row.mean())

    def encode_projected_image(self, projected_image: torch.Tensor, return_entropy: bool = False):
        """
        Encode 2D projected image to per-row spectral histogram

        Args:
            projected_image: (n_rows, n_azimuth) projected image (range image or BEV)
            return_entropy: If True, also return spectral entropy (float)

        Returns:
            If return_entropy=False: (output_dim,) normalized descriptor.
            If return_entropy=True: ((output_dim,) descriptor, float entropy).
        """
        # Sensor-agnostic row binning (range_image only; BEV already has correct n_rows)
        if self.projection_type != 'bev' and projected_image.shape[0] != self.target_elevation_bins:
            projected_image = torch.nn.functional.adaptive_avg_pool2d(
                projected_image.unsqueeze(0).unsqueeze(0),
                (self.target_elevation_bins, projected_image.shape[1])
            ).squeeze()

        # Zero-center rows to remove DC component (average range per row)
        if self.zero_center:
            projected_image = projected_image - projected_image.mean(dim=1, keepdim=True)

        # Apply row-wise 1D FFT along azimuth dimension
        fft_output = torch.fft.rfft(projected_image, dim=1, norm='ortho')

        # Take magnitude (discard phase for rotation invariance)
        fft_magnitudes = torch.abs(fft_output)  # (target_elevation_bins, n_freqs)

        # Normalize by sqrt(n_azimuth) to maintain scale
        fft_magnitudes = fft_magnitudes * np.sqrt(self.n_azimuth)

        # Compute spectral entropy before log transform (on raw magnitudes)
        entropy = None
        if return_entropy:
            entropy = self._compute_spectral_entropy(fft_magnitudes, self.epsilon)

        # Log-magnitude for compressed dynamic range
        if self.log_magnitude:
            fft_magnitudes = torch.log(fft_magnitudes + self.epsilon)

        # Compute adaptive bin edges
        bin_edges = self._compute_bin_edges(self.alpha)

        # Bin FFT magnitudes into per-elevation histograms
        histogram = self._bin_fft_magnitudes(fft_magnitudes, bin_edges)
        # histogram shape: (output_dim,) = intra channels + inter channels

        # Normalization
        if self.n_stats == 1 and self.bin_statistics[0] == 'sum' and self.n_inter_stats == 0:
            # Backward-compatible: global sum-to-1
            histogram_sum = histogram.sum()
            if histogram_sum > self.epsilon:
                histogram = histogram / (histogram_sum + self.epsilon)
            else:
                histogram = torch.ones_like(histogram) / histogram.numel()
        elif self.normalize_channels:
            # Per-channel L2 normalization for each channel independently.
            # Channel layout: [intra_ch × n_stats | inter_ch × n_inter_stats × n_stats]
            # Intra channels: each base_dim (256), inter channels: each inter_base_dim (240)
            channel_dims = (
                [self.base_dim] * self.n_stats
                + [self.inter_base_dim] * (self.n_inter_stats * self.n_stats)
            )
            normalized = []
            offset = 0
            for dim in channel_dims:
                ch = histogram[offset:offset + dim]
                normalized.append(torch.nn.functional.normalize(ch, p=2, dim=0))
                offset += dim
            histogram = torch.cat(normalized)
        # else: no normalization — raw magnitudes preserved for standardized L2 edges

        if return_entropy:
            return histogram, entropy
        return histogram

    def encode_points(self, points: np.ndarray, return_entropy: bool = False):
        """
        Encode point cloud to spectral histogram

        Args:
            points: (N, 3) or (N, 4) numpy array [x, y, z] or [x, y, z, intensity]
            return_entropy: If True, also return spectral entropy (float)

        Returns:
            If return_entropy=False: (output_dim,) normalized spectral histogram
            If return_entropy=True: ((output_dim,) histogram, float entropy)
        """
        # Project to 2D image (range image or BEV polar grid)
        image_2d, _ = self.projector.project(points, keep_intensity=False)

        # Interpolate empty pixels for clean FFT
        if self.interpolate_empty:
            if self.projection_type == 'bev':
                image_2d = interpolate_bev_image(image_2d, method='linear')
            else:
                image_2d = interpolate_range_image(image_2d, method='linear')

        # Convert to torch tensor and move to same device as model
        image_tensor = torch.from_numpy(image_2d).float().to(self.alpha.device)

        # Encode
        return self.encode_projected_image(image_tensor, return_entropy=return_entropy)

    def compute_fft_magnitudes(self, points: np.ndarray) -> np.ndarray:
        """
        Compute FFT magnitude spectrum without binning (for caching).

        Runs the same pipeline as encode_projected_image up to (and including)
        the optional log transform, but stops before binning.

        Args:
            points: (N, 3) or (N, 4) point cloud

        Returns:
            (n_rows, n_freqs) float32 FFT magnitude spectrum
        """
        image_2d, _ = self.projector.project(points, keep_intensity=False)
        if self.interpolate_empty:
            if self.projection_type == 'bev':
                image_2d = interpolate_bev_image(image_2d, method='linear')
            else:
                image_2d = interpolate_range_image(image_2d, method='linear')

        image_tensor = torch.from_numpy(image_2d).float().to(self.alpha.device)

        # Sensor-agnostic row binning (range_image only)
        if self.projection_type != 'bev' and image_tensor.shape[0] != self.target_elevation_bins:
            image_tensor = torch.nn.functional.adaptive_avg_pool2d(
                image_tensor.unsqueeze(0).unsqueeze(0),
                (self.target_elevation_bins, image_tensor.shape[1])
            ).squeeze()

        if self.zero_center:
            image_tensor = image_tensor - image_tensor.mean(dim=1, keepdim=True)

        fft_output = torch.fft.rfft(image_tensor, dim=1, norm='ortho')
        fft_magnitudes = torch.abs(fft_output) * np.sqrt(self.n_azimuth)

        if self.log_magnitude:
            fft_magnitudes = torch.log(fft_magnitudes + self.epsilon)

        return fft_magnitudes.detach().cpu().numpy().astype(np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (for use in neural network training)

        Args:
            x: (batch, n_rows, n_azimuth) batch of projected images

        Returns:
            (batch, output_dim) batch of spectral histograms
        """
        batch_size = x.shape[0]
        histograms = []

        for i in range(batch_size):
            histogram = self.encode_projected_image(x[i])
            histograms.append(histogram)

        return torch.stack(histograms, dim=0)

    def encode_batch(self, projected_images: torch.Tensor) -> torch.Tensor:
        """
        Encode batch of projected images

        Args:
            projected_images: (batch, n_rows, n_azimuth) projected images

        Returns:
            (batch, output_dim) spectral histograms
        """
        return self.forward(projected_images)

    # Backward compatibility alias
    def encode_range_image(self, range_image: torch.Tensor) -> torch.Tensor:
        return self.encode_projected_image(range_image)


class SpectralEncoderNumpy:
    """
    Numpy-only version of SpectralEncoder for inference without PyTorch

    Supports both 'range_image' and 'bev' projection modes.
    """

    def __init__(
        self,
        n_elevation: int = 64,
        n_azimuth: int = 360,
        n_bins: int = 16,
        alpha: float = 2.0,
        epsilon: float = 1e-8,
        bin_statistics: Optional[List[str]] = None,
        inter_bin_statistics: Optional[List[str]] = None,
        projection_type: str = 'range_image',
        target_elevation_bins: int = 16,
        max_range: float = 80.0,
        min_range: float = 1.0,
        z_min: float = -3.0,
        height_encoding: str = 'max',
        n_height_layers: int = 8,
        z_max: float = 5.0,
        zero_center: bool = False,
        log_magnitude: bool = False,
        binning_strategy: str = 'exponential',
        normalize_channels: bool = True
    ):
        """
        Initialize numpy spectral encoder

        Args:
            n_elevation: Number of elevation rings
            n_azimuth: Number of azimuth bins
            n_bins: Number of histogram bins
            alpha: Exponential warping parameter
            epsilon: Numerical stability constant
            bin_statistics: List of statistics per bin. Default: ['sum']
            inter_bin_statistics: List of inter-bin statistics. Valid: 'diff', 'ratio'. Default: []
            projection_type: 'range_image' or 'bev'
            target_elevation_bins: Target rows (elevation bins or rings)
            max_range: Maximum LiDAR range in meters
            min_range: Minimum LiDAR range in meters
            z_min: Minimum z-height for BEV ground filtering
            height_encoding: 'max' or 'iris' (LiDAR-Iris style binary height code)
            n_height_layers: Number of binary height layers for iris encoding
            z_max: Maximum z-height for iris encoding
            zero_center: If True, subtract row mean before FFT to remove DC component
            log_magnitude: If True, use log(|FFT| + ε) for compressed dynamic range
            binning_strategy: 'exponential' or 'octave' (log2 scale)
            normalize_channels: If True, apply per-channel L2 normalization (default).
                Set False to preserve magnitude for standardized Euclidean distance edges.
        """
        self.n_elevation = n_elevation
        self.n_azimuth = n_azimuth
        self.alpha = alpha
        self.epsilon = epsilon
        self.projection_type = projection_type
        self.target_elevation_bins = target_elevation_bins
        self.zero_center = zero_center
        self.log_magnitude = log_magnitude
        self.binning_strategy = binning_strategy
        self.normalize_channels = normalize_channels

        # Octave binning: auto-compute n_bins
        n_freqs_val = n_azimuth // 2 + 1
        if binning_strategy == 'octave':
            power = 0
            while 2 ** power < n_freqs_val:
                power += 1
            self.n_bins = power + 1
        else:
            self.n_bins = n_bins

        if bin_statistics is None:
            bin_statistics = ['sum']
        for stat in bin_statistics:
            if stat not in VALID_BIN_STATISTICS:
                raise ValueError(f"Invalid bin statistic '{stat}'. Valid: {VALID_BIN_STATISTICS}")
        self.bin_statistics = bin_statistics
        self.n_stats = len(bin_statistics)

        if inter_bin_statistics is None:
            inter_bin_statistics = []
        for stat in inter_bin_statistics:
            if stat not in VALID_INTER_BIN_STATISTICS:
                raise ValueError(f"Invalid inter-bin statistic '{stat}'. Valid: {VALID_INTER_BIN_STATISTICS}")
        self.inter_bin_statistics = inter_bin_statistics
        self.n_inter_stats = len(inter_bin_statistics)

        if projection_type == 'bev':
            self.projector = BEVProjector(
                n_sectors=n_azimuth,
                max_range=max_range,
                min_range=min_range,
                z_min=z_min,
                height_encoding=height_encoding,
                n_height_layers=n_height_layers,
                z_max=z_max
            )
            n_rows = self.projector.n_rings
        else:
            self.projector = RangeImageProjector(
                n_elevation=n_elevation,
                n_azimuth=n_azimuth
            )
            n_rows = n_elevation

        self.base_dim = n_rows * self.n_bins
        self.inter_base_dim = n_rows * (self.n_bins - 1)
        self.output_dim = (self.base_dim * self.n_stats
                           + self.inter_base_dim * self.n_inter_stats * self.n_stats)

        self.n_freqs = n_azimuth // 2 + 1

    def _compute_bin_edges(self) -> np.ndarray:
        """Compute frequency bin edges (exponential or octave)"""
        if self.binning_strategy == 'octave':
            edges = [0]
            power = 0
            while 2 ** power < self.n_freqs:
                edges.append(2 ** power)
                power += 1
            edges.append(self.n_freqs)
            return np.array(edges, dtype=np.float64)

        t = np.linspace(0, 1, self.n_bins + 1)
        bin_edges = (np.exp(self.alpha * t) - 1) / (np.exp(self.alpha) - 1 + self.epsilon)
        bin_edges = bin_edges * self.n_freqs
        return bin_edges

    @staticmethod
    def _compute_spectral_entropy(fft_magnitudes: np.ndarray, epsilon: float = 1e-8) -> float:
        """Compute spectral entropy from FFT magnitudes (numpy version)."""
        power = fft_magnitudes ** 2
        power_sum = power.sum(axis=1, keepdims=True)
        power_sum = np.maximum(power_sum, epsilon)
        p = power / power_sum
        log_p = np.log(p + epsilon)
        entropy_per_row = -(p * log_p).sum(axis=1)
        return float(entropy_per_row.mean())

    def encode_range_image(self, range_image: np.ndarray, return_entropy: bool = False):
        """
        Encode range image to spectral histogram with multiple statistics

        Args:
            range_image: (n_elevation, n_azimuth) range values
            return_entropy: If True, also return spectral entropy (float)

        Returns:
            If return_entropy=False: (output_dim,) normalized descriptor
            If return_entropy=True: ((output_dim,) descriptor, float entropy)
        """
        n_elevation = range_image.shape[0]

        # Zero-center rows to remove DC component
        if self.zero_center:
            range_image = range_image - range_image.mean(axis=1, keepdims=True)

        # Apply ring-wise 1D FFT
        fft_output = np.fft.rfft(range_image, axis=1, norm='ortho')

        # Take magnitude
        fft_magnitudes = np.abs(fft_output)  # (n_elevation, n_freqs)

        # Normalize
        fft_magnitudes = fft_magnitudes * np.sqrt(self.n_azimuth)

        # Compute spectral entropy before log transform
        entropy = None
        if return_entropy:
            entropy = self._compute_spectral_entropy(fft_magnitudes, self.epsilon)

        # Log-magnitude for compressed dynamic range
        if self.log_magnitude:
            fft_magnitudes = np.log(fft_magnitudes + self.epsilon)

        # Compute bin edges
        bin_edges = self._compute_bin_edges()

        # Bin magnitudes with multiple statistics
        freq_indices = np.arange(self.n_freqs)

        channels = []
        for stat_name in self.bin_statistics:
            channel = np.zeros((n_elevation, self.n_bins))
            for i in range(self.n_bins):
                mask = (freq_indices >= bin_edges[i]) & (freq_indices < bin_edges[i + 1])
                if mask.any():
                    bin_values = fft_magnitudes[:, mask]  # (n_elevation, n_freq_in_bin)
                    if stat_name == 'sum':
                        channel[:, i] = bin_values.sum(axis=1)
                    elif stat_name == 'mean':
                        channel[:, i] = bin_values.mean(axis=1)
                    elif stat_name == 'std':
                        channel[:, i] = bin_values.std(axis=1) + self.epsilon
                    elif stat_name == 'max':
                        channel[:, i] = bin_values.max(axis=1)
                    elif stat_name == 'min':
                        channel[:, i] = bin_values.min(axis=1)
            channels.append(channel.flatten())  # (n_elevation * n_bins,)

        # Inter-bin channels: adjacent-bin statistics from each intra-bin channel (snapshot before appending)
        if self.inter_bin_statistics:
            intra_channels = list(channels)  # snapshot: only base stat channels (256D each)
            for inter_type in self.inter_bin_statistics:
                for ch in intra_channels:
                    hist_2d = ch.reshape(n_elevation, self.n_bins)
                    if inter_type == 'diff':
                        inter_ch = hist_2d[:, 1:] - hist_2d[:, :-1]  # (n_elev, n_bins-1)
                    else:  # 'ratio'
                        inter_ch = (np.log(hist_2d[:, 1:] + self.epsilon)
                                    - np.log(hist_2d[:, :-1] + self.epsilon))
                    channels.append(inter_ch.flatten())  # (n_elevation * (n_bins-1),)

        histogram = np.concatenate(channels)

        # Normalization
        if self.n_stats == 1 and self.bin_statistics[0] == 'sum' and self.n_inter_stats == 0:
            # Backward-compatible: global sum-to-1
            histogram_sum = histogram.sum()
            if histogram_sum > self.epsilon:
                histogram = histogram / (histogram_sum + self.epsilon)
            else:
                histogram = np.ones_like(histogram) / histogram.size
        elif self.normalize_channels:
            # Per-channel L2 normalization for each channel independently
            channel_dims = (
                [self.base_dim] * self.n_stats
                + [self.inter_base_dim] * (self.n_inter_stats * self.n_stats)
            )
            normalized = []
            offset = 0
            for dim in channel_dims:
                ch = histogram[offset:offset + dim]
                norm = np.linalg.norm(ch)
                normalized.append(ch / max(norm, self.epsilon))
                offset += dim
            histogram = np.concatenate(normalized)
        # else: no normalization — raw magnitudes preserved for standardized L2 edges

        if return_entropy:
            return histogram, entropy
        return histogram

    def encode_points(self, points: np.ndarray, return_entropy: bool = False):
        """
        Encode point cloud to spectral histogram

        Args:
            points: (N, 3) or (N, 4) point cloud
            return_entropy: If True, also return spectral entropy (float)

        Returns:
            If return_entropy=False: (output_dim,) spectral histogram
            If return_entropy=True: ((output_dim,) histogram, float entropy)
        """
        image_2d, _ = self.projector.project(points, keep_intensity=False)
        if self.projection_type == 'bev':
            image_2d = interpolate_bev_image(image_2d, method='linear')
        else:
            image_2d = interpolate_range_image(image_2d, method='linear')
        return self.encode_range_image(image_2d, return_entropy=return_entropy)

    def compute_fft_magnitudes(self, points: np.ndarray) -> np.ndarray:
        """
        Compute FFT magnitude spectrum without binning (for caching).

        Args:
            points: (N, 3) or (N, 4) point cloud

        Returns:
            (n_rows, n_freqs) float32 FFT magnitude spectrum
        """
        image_2d, _ = self.projector.project(points, keep_intensity=False)
        if self.projection_type == 'bev':
            image_2d = interpolate_bev_image(image_2d, method='linear')
        else:
            image_2d = interpolate_range_image(image_2d, method='linear')

        # Sensor-agnostic row binning (range_image only)
        if self.projection_type != 'bev':
            from scipy.ndimage import zoom
            n_rows_current = image_2d.shape[0]
            target = getattr(self, 'target_elevation_bins', n_rows_current)
            if n_rows_current != target:
                ratio = target / n_rows_current
                image_2d = zoom(image_2d, (ratio, 1), order=1)

        if self.zero_center:
            image_2d = image_2d - image_2d.mean(axis=1, keepdims=True)

        fft_output = np.fft.rfft(image_2d, axis=1, norm='ortho')
        fft_magnitudes = np.abs(fft_output) * np.sqrt(self.n_azimuth)

        if self.log_magnitude:
            fft_magnitudes = np.log(fft_magnitudes + self.epsilon)

        return fft_magnitudes.astype(np.float32)


def test_rotation_invariance(
    encoder: SpectralEncoder,
    points: np.ndarray,
    n_rotations: int = 8
) -> float:
    """
    Test rotation invariance of spectral encoder

    Args:
        encoder: SpectralEncoder instance
        points: (N, 3) point cloud
        n_rotations: Number of rotations to test

    Returns:
        Maximum histogram difference across rotations
    """
    from data.pose_utils import transform_points

    histograms = []

    for i in range(n_rotations):
        # Rotate around z-axis
        angle = 2 * np.pi * i / n_rotations

        # Rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Rotate points
        rotated_points = transform_points(points, R)

        # Encode
        histogram = encoder.encode_points(rotated_points)
        histograms.append(histogram.detach().numpy())

    # Compute maximum difference
    histograms = np.array(histograms)
    max_diff = 0.0

    for i in range(n_rotations):
        for j in range(i + 1, n_rotations):
            diff = np.abs(histograms[i] - histograms[j]).max()
            max_diff = max(max_diff, diff)

    return max_diff
