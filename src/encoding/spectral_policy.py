"""
Learnable Spectral Sampling Policies

Replaces fixed binning + statistics with differentiable NN modules that
learn to extract descriptors from FFT magnitude spectra end-to-end.

All policies share the interface:
    Input:  (B, n_rings, n_freqs)  FFT magnitude spectrum
    Output: (B, output_dim)        descriptor vector

Five policy types:
    A. LearnedFilterbank  — Linear projection (simplest)
    B. ConvSpectralPool   — 1D convolution along frequency axis
    C. CrossAttentionPool — Learned queries attend to spectrum
    D. SoftBinning        — Differentiable Gaussian bin kernels
    E. GatedFrequencySelection — Learned per-ring frequency masks
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class SpectralPolicyBase(nn.Module):
    """Abstract base class for spectral sampling policies."""

    def __init__(self, n_rings: int, n_freqs: int, output_dim: int):
        super().__init__()
        self.n_rings = n_rings
        self.n_freqs = n_freqs
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_rings, n_freqs) FFT magnitude spectrum
        Returns:
            (B, output_dim) descriptor
        """
        raise NotImplementedError


# =========================================================================
# Option A: Learned Filterbank (Linear Projection)
# =========================================================================

class LearnedFilterbank(SpectralPolicyBase):
    """Each row of the weight matrix is a learned spectral filter.

    Shared across all rings by default. Per-ring variant uses separate
    Linear layers for each ring.
    """

    def __init__(
        self,
        n_rings: int,
        n_freqs: int,
        output_dim: int,
        shared_across_rings: bool = True,
    ):
        super().__init__(n_rings, n_freqs, output_dim)
        self.shared = shared_across_rings

        if shared_across_rings:
            self.d_per_ring = output_dim // n_rings
            if self.d_per_ring * n_rings != output_dim:
                raise ValueError(
                    f"output_dim ({output_dim}) must be divisible by n_rings ({n_rings}) "
                    f"for shared filterbank"
                )
            self.linear = nn.Linear(n_freqs, self.d_per_ring)
        else:
            self.d_per_ring = output_dim // n_rings
            if self.d_per_ring * n_rings != output_dim:
                raise ValueError(
                    f"output_dim ({output_dim}) must be divisible by n_rings ({n_rings}) "
                    f"for per-ring filterbank"
                )
            self.linears = nn.ModuleList([
                nn.Linear(n_freqs, self.d_per_ring) for _ in range(n_rings)
            ])

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.shared:
            # (B, n_rings, n_freqs) -> (B, n_rings, d_per_ring) -> (B, output_dim)
            out = self.linear(x)
            out = out.reshape(B, -1)
        else:
            parts = []
            for i, lin in enumerate(self.linears):
                parts.append(lin(x[:, i, :]))
            out = torch.cat(parts, dim=1)
        return self.norm(out)


# =========================================================================
# Option B: 1D Conv Spectral Pooling
# =========================================================================

class ConvSpectralPool(SpectralPolicyBase):
    """Conv1d along frequency axis with adaptive pooling.

    Uses grouped convolution (groups=n_rings) so each ring has
    independent conv filters, then pools to target dimension.
    """

    def __init__(
        self,
        n_rings: int,
        n_freqs: int,
        output_dim: int,
        channels_per_group: int = 2,
        kernel_size: int = 7,
    ):
        super().__init__(n_rings, n_freqs, output_dim)
        self.channels_per_group = channels_per_group
        out_channels = n_rings * channels_per_group

        # Grouped conv: each ring gets its own set of filters
        self.conv = nn.Conv1d(
            in_channels=n_rings,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=n_rings,
        )

        # Adaptive pool to match target dim per ring
        # Total output = n_rings * channels_per_group * pool_size
        # pool_size = output_dim / (n_rings * channels_per_group)
        total_features = n_rings * channels_per_group
        if output_dim % total_features != 0:
            # Use a projection layer if not evenly divisible
            self.pool_size = max(1, output_dim // total_features)
            self.proj = nn.Linear(total_features * self.pool_size, output_dim)
        else:
            self.pool_size = output_dim // total_features
            self.proj = None

        self.pool = nn.AdaptiveAvgPool1d(self.pool_size)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # x: (B, n_rings, n_freqs) — n_rings acts as channels
        h = F.relu(self.conv(x))  # (B, n_rings*C, n_freqs)
        h = self.pool(h)  # (B, n_rings*C, pool_size)
        h = h.reshape(B, -1)  # (B, n_rings*C*pool_size)
        if self.proj is not None:
            h = self.proj(h)
        return self.norm(h)


# =========================================================================
# Option C: Cross-Attention Frequency Pooling
# =========================================================================

class CrossAttentionPool(SpectralPolicyBase):
    """Learned query vectors attend to frequency spectrum per ring.

    Uses sinusoidal positional encoding for frequency awareness.
    """

    def __init__(
        self,
        n_rings: int,
        n_freqs: int,
        output_dim: int,
        n_queries: int = 7,
        n_heads: int = 2,
        head_dim: int = 32,
        d_pe: int = 16,
    ):
        super().__init__(n_rings, n_freqs, output_dim)
        self.n_queries = n_queries
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = n_heads * head_dim

        # Frequency positional encoding (sinusoidal, fixed)
        pe = self._make_freq_pe(n_freqs, d_pe)
        self.register_buffer('freq_pe', pe)  # (n_freqs, d_pe)

        # Input: magnitude (1D) + positional encoding (d_pe)
        input_dim = 1 + d_pe

        # Key/Value projections (shared across rings)
        self.proj_k = nn.Linear(input_dim, self.d_model)
        self.proj_v = nn.Linear(input_dim, self.d_model)

        # Learned queries
        self.queries = nn.Parameter(torch.randn(n_queries, self.d_model) * 0.02)

        # Output projection: per-ring queries → flat descriptor
        total_query_dim = n_queries * self.d_model
        per_ring_out = output_dim // n_rings
        if per_ring_out * n_rings != output_dim:
            self.out_proj = nn.Linear(n_rings * total_query_dim, output_dim)
        else:
            self.out_proj = nn.Linear(total_query_dim, per_ring_out)

        self.per_ring_out = per_ring_out if per_ring_out * n_rings == output_dim else None
        self.norm = nn.LayerNorm(output_dim)
        self.scale = math.sqrt(head_dim)

    @staticmethod
    def _make_freq_pe(n_freqs: int, d_pe: int) -> torch.Tensor:
        """Sinusoidal positional encoding for frequency indices."""
        position = torch.arange(n_freqs, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_pe, 2, dtype=torch.float32) * -(math.log(10000.0) / d_pe)
        )
        pe = torch.zeros(n_freqs, d_pe)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, R, F_ = x.shape

        # Chunked processing for large batches to avoid OOM
        # K/V projections create (B*R, F, d_model) tensors — 132GB at 178K×16 rings
        max_batch_rings = 4096
        if B * R > max_batch_rings:
            return self._forward_chunked(x, max_batch_rings)

        return self._forward_core(x)

    def _forward_chunked(self, x: torch.Tensor, max_batch_rings: int) -> torch.Tensor:
        """Process in chunks along batch dimension to bound memory."""
        B, R, F_ = x.shape
        chunk_size = max(1, max_batch_rings // R)
        outputs = []
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            out_chunk = self._forward_core(x[start:end])
            outputs.append(out_chunk)
        return torch.cat(outputs, dim=0)

    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:
        """Core attention forward for a single chunk."""
        B, R, F_ = x.shape
        # Expand PE: (F, d_pe) → (B, R, F, d_pe)
        pe = self.freq_pe.unsqueeze(0).unsqueeze(0).expand(B, R, -1, -1)
        # Concat magnitude + PE: (B, R, F, 1+d_pe)
        mag = x.unsqueeze(-1)  # (B, R, F, 1)
        kv_input = torch.cat([mag, pe], dim=-1)  # (B, R, F, 1+d_pe)

        # Reshape for batched attention: (B*R, F, 1+d_pe)
        kv_input = kv_input.reshape(B * R, F_, -1)
        K = self.proj_k(kv_input)  # (B*R, F, d_model)
        V = self.proj_v(kv_input)  # (B*R, F, d_model)

        # Queries: (n_queries, d_model) → (B*R, n_queries, d_model)
        Q = self.queries.unsqueeze(0).expand(B * R, -1, -1)

        # Multi-head attention
        # Reshape for heads: (B*R, seq, n_heads, head_dim) → (B*R, n_heads, seq, head_dim)
        Q = Q.view(B * R, self.n_queries, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B * R, F_, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B * R, F_, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B*R, heads, nq, F)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # (B*R, heads, nq, head_dim)

        # Merge heads: (B*R, nq, d_model)
        out = out.transpose(1, 2).reshape(B * R, self.n_queries, self.d_model)
        out = out.reshape(B * R, -1)  # (B*R, nq * d_model)

        if self.per_ring_out is not None:
            out = self.out_proj(out)  # (B*R, per_ring_out)
            out = out.reshape(B, -1)  # (B, R * per_ring_out) = (B, output_dim)
        else:
            out = out.reshape(B, -1)  # (B, R * nq * d_model)
            out = self.out_proj(out)  # (B, output_dim)

        return self.norm(out)


# =========================================================================
# Option D: Soft Binning (Differentiable)
# =========================================================================

class SoftBinning(SpectralPolicyBase):
    """Differentiable Gaussian bin kernels replacing hard binning.

    Each soft bin is a Gaussian kernel centered at a learnable position
    with learnable width. Statistics (mean, std, inter-bin diff) are
    computed as weighted aggregations.
    """

    def __init__(
        self,
        n_rings: int,
        n_freqs: int,
        output_dim: int,
        n_soft_bins: int = 4,
        stats: Optional[List[str]] = None,
        inter_stats: Optional[List[str]] = None,
        init_from_fixed: bool = True,
        alpha: float = 2.0,
        shared_across_rings: bool = True,
        init_mode: Optional[str] = None,
        inter_ring: Optional[dict] = None,
    ):
        # Auto-compute output_dim from n_soft_bins to avoid projection layer
        _stats = stats if stats is not None else ['mean', 'std']
        _inter = inter_stats if inter_stats is not None else ['diff']
        n_stats = len(_stats)
        n_inter_stats = len(_inter)
        per_ring_dim = n_soft_bins * n_stats + (n_soft_bins - 1) * n_inter_stats * n_stats

        # Inter-ring 1D Conv: captures vertical structure across rings
        ir_channels = 0
        if inter_ring and inter_ring.get('enabled', False):
            ir_channels = inter_ring.get('channels', 8)

        auto_output_dim = n_rings * per_ring_dim + n_rings * ir_channels

        # Use auto-computed dim (ignore config output_dim to prevent projection)
        super().__init__(n_rings, n_freqs, auto_output_dim)
        self.n_soft_bins = n_soft_bins
        self.stats = _stats
        self.inter_stats = _inter
        self.shared = shared_across_rings
        self.per_ring_dim = per_ring_dim
        self.proj = None  # Never use projection — dim always matches

        # Inter-ring Conv1d along ring axis
        self.inter_ring_conv = None
        self.inter_ring_channels = ir_channels
        if ir_channels > 0:
            ir_kernel = inter_ring.get('kernel_size', 3)
            self.inter_ring_conv = nn.Conv1d(
                in_channels=per_ring_dim,
                out_channels=ir_channels,
                kernel_size=ir_kernel,
                padding=ir_kernel // 2,
            )

        # Learnable bin parameters
        n_param_sets = 1 if shared_across_rings else n_rings

        # Resolve init mode (backward compat: init_from_fixed maps to exponential/uniform)
        if init_mode is None:
            init_mode = 'exponential' if init_from_fixed else 'uniform'

        if init_mode == 'octave':
            centers, log_widths = self._init_from_octave(
                n_soft_bins, n_freqs, n_param_sets
            )
        elif init_mode == 'exponential':
            centers, log_widths = self._init_from_exponential(
                n_soft_bins, n_freqs, alpha, n_param_sets
            )
        else:  # uniform
            centers = torch.linspace(0, n_freqs - 1, n_soft_bins).unsqueeze(0).expand(n_param_sets, -1)
            log_widths = torch.full((n_param_sets, n_soft_bins), math.log(n_freqs / n_soft_bins / 2))

        self.centers = nn.Parameter(centers.clone())  # (n_sets, n_soft_bins)
        self.log_widths = nn.Parameter(log_widths.clone())  # (n_sets, n_soft_bins)

        # Frequency indices buffer
        self.register_buffer('freq_idx', torch.arange(n_freqs, dtype=torch.float32))

        self.norm = nn.LayerNorm(auto_output_dim)

    @staticmethod
    def _init_from_exponential(n_bins, n_freqs, alpha, n_sets):
        """Initialize centers/widths from exponential bin edges."""
        t = torch.linspace(0, 1, n_bins + 1)
        edges = (torch.exp(alpha * t) - 1) / (torch.exp(torch.tensor(alpha)) - 1) * n_freqs
        centers = (edges[:-1] + edges[1:]) / 2  # midpoints
        widths = (edges[1:] - edges[:-1]) / 2   # half-widths
        widths = torch.clamp(widths, min=1.0)    # ensure positive
        centers = centers.unsqueeze(0).expand(n_sets, -1)
        log_widths = torch.log(widths).unsqueeze(0).expand(n_sets, -1)
        return centers, log_widths

    @staticmethod
    def _init_from_octave(n_bins, n_freqs, n_sets):
        """Initialize centers/widths from octave bin edges.

        Octave edges: [0, 1, 2, 4, 8, 16, ..., n_freqs].
        If n_bins != number of octave bins, interpolates to match.
        """
        edges = [0]
        p = 0
        while 2 ** p < n_freqs:
            edges.append(2 ** p)
            p += 1
        edges.append(n_freqs)
        edges_t = torch.tensor(edges, dtype=torch.float32)

        # Interpolate if n_bins doesn't match natural octave count
        n_octave_bins = len(edges_t) - 1
        if n_octave_bins != n_bins:
            import numpy as np
            t_orig = torch.linspace(0, 1, len(edges_t))
            t_new = torch.linspace(0, 1, n_bins + 1)
            edges_t = torch.from_numpy(
                np.interp(t_new.numpy(), t_orig.numpy(), edges_t.numpy())
            ).float()

        centers = (edges_t[:-1] + edges_t[1:]) / 2
        widths = (edges_t[1:] - edges_t[:-1]) / 2
        widths = torch.clamp(widths, min=1.0)
        centers = centers.unsqueeze(0).expand(n_sets, -1)
        log_widths = torch.log(widths).unsqueeze(0).expand(n_sets, -1)
        return centers, log_widths

    def _compute_weights(self, set_idx: int = 0) -> torch.Tensor:
        """Compute soft bin assignment weights.

        Returns:
            (n_soft_bins, n_freqs) normalized Gaussian weights
        """
        c = self.centers[set_idx]     # (n_soft_bins,)
        w = torch.exp(self.log_widths[set_idx])  # (n_soft_bins,)

        # Gaussian kernels: (n_soft_bins, n_freqs)
        diff = self.freq_idx.unsqueeze(0) - c.unsqueeze(1)  # (nb, nf)
        weights = torch.exp(-0.5 * (diff / w.unsqueeze(1)) ** 2)

        # Normalize so each bin's weights sum to 1
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, R, F_ = x.shape
        eps = 1e-8
        nb = self.n_soft_bins

        # Chunk large batches to limit peak memory from float32 std computation.
        # 178K nodes × 16 rings × 181 freqs × 4 bytes (x^2 in fp32) = 2GB if unchunked.
        MAX_NODES = 32768
        if B > MAX_NODES:
            chunks = []
            for start in range(0, B, MAX_NODES):
                chunk_x = x[start:start + MAX_NODES]
                cB = chunk_x.shape[0]
                if self.shared:
                    chunks.append(self._forward_shared(chunk_x, cB, R, F_, eps, nb))
                else:
                    chunks.append(self._forward_per_ring(chunk_x, cB, R, F_, eps, nb))
            return torch.cat(chunks, dim=0)

        if self.shared:
            return self._forward_shared(x, B, R, F_, eps, nb)
        else:
            return self._forward_per_ring(x, B, R, F_, eps, nb)

    def _forward_shared(self, x, B, R, F_, eps, nb):
        """Vectorized forward when all rings share parameters."""
        weights = self._compute_weights(0)  # (nb, F) — float32

        # Batch all rings: (B, R, F) → (B*R, F)
        x_flat = x.reshape(B * R, F_)

        # Soft mean: weights @ x_flat^T = (nb, B*R) → (B*R, nb)
        soft_mean = torch.matmul(weights, x_flat.t()).t()

        base_channels = []
        stat_parts = []

        for stat_name in self.stats:
            if stat_name == 'mean':
                ch = soft_mean
            elif stat_name == 'std':
                # x^2 overflows float16 (~16K^2 > 65504) — compute in float32
                with torch.cuda.amp.autocast(enabled=False):
                    x_f32 = x_flat.float()
                    w_f32 = weights.float()
                    sm_f32 = soft_mean.float()
                    soft_mean_sq = torch.matmul(w_f32, (x_f32 ** 2).t()).t()
                    soft_var = soft_mean_sq - sm_f32 ** 2
                    ch = torch.sqrt(torch.clamp(soft_var, min=eps))
                ch = ch.to(soft_mean.dtype)  # back to AMP dtype
            elif stat_name == 'sum':
                ch = soft_mean * F_
            else:
                ch = soft_mean
            stat_parts.append(ch)  # (B*R, nb)
            base_channels.append(ch)

        # Inter-bin: reshape to (B, R, nb) for diff, then back to (B*R, nb-1)
        for inter_type in self.inter_stats:
            for base_ch in base_channels:
                ch_3d = base_ch.reshape(B, R, nb)
                if inter_type == 'diff':
                    inter = ch_3d[:, :, 1:] - ch_3d[:, :, :-1]
                elif inter_type == 'ratio':
                    inter = (torch.log(ch_3d[:, :, 1:] + eps)
                             - torch.log(ch_3d[:, :, :-1] + eps))
                else:
                    inter = ch_3d[:, :, 1:] - ch_3d[:, :, :-1]
                stat_parts.append(inter.reshape(B * R, nb - 1))

        # Concat: (B*R, per_ring_dim) → (B, R, per_ring_dim)
        per_ring = torch.cat(stat_parts, dim=1)  # (B*R, per_ring_dim)
        base_2d = per_ring.reshape(B, R, -1)     # (B, R, per_ring_dim)

        # Inter-ring conv: capture vertical structure across rings
        if self.inter_ring_conv is not None:
            # (B, R, D) → (B, D, R) for Conv1d → (B, k, R) → (B, R, k)
            ir_in = base_2d.transpose(1, 2)
            ir_out = F.relu(self.inter_ring_conv(ir_in))
            ir_out = ir_out.transpose(1, 2)  # (B, R, k)
            combined = torch.cat([base_2d, ir_out], dim=2)  # (B, R, D+k)
        else:
            combined = base_2d

        out = combined.reshape(B, -1)  # (B, R*(D+k))

        if self.proj is not None:
            out = self.proj(out)
        return self.norm(out)

    def _forward_per_ring(self, x, B, R, F_, eps, nb):
        """Per-ring forward when each ring has its own parameters."""
        all_ring_features = []
        for r in range(R):
            weights = self._compute_weights(r)  # float32
            mag = x[:, r, :]  # (B, F)

            ring_stats = []
            base_channels = []

            soft_mean = torch.matmul(weights, mag.t()).t()  # (B, nb)
            for stat_name in self.stats:
                if stat_name == 'mean':
                    ch = soft_mean
                elif stat_name == 'std':
                    # x^2 overflows float16 — compute in float32
                    with torch.cuda.amp.autocast(enabled=False):
                        mag_f32 = mag.float()
                        w_f32 = weights.float()
                        sm_f32 = soft_mean.float()
                        soft_mean_sq = torch.matmul(w_f32, (mag_f32 ** 2).t()).t()
                        soft_var = soft_mean_sq - sm_f32 ** 2
                        ch = torch.sqrt(torch.clamp(soft_var, min=eps))
                    ch = ch.to(soft_mean.dtype)
                elif stat_name == 'sum':
                    ch = soft_mean * F_
                else:
                    ch = soft_mean
                ring_stats.append(ch)
                base_channels.append(ch)

            for inter_type in self.inter_stats:
                for base_ch in base_channels:
                    if inter_type == 'diff':
                        inter_ch = base_ch[:, 1:] - base_ch[:, :-1]
                    elif inter_type == 'ratio':
                        inter_ch = (torch.log(base_ch[:, 1:] + eps)
                                    - torch.log(base_ch[:, :-1] + eps))
                    else:
                        inter_ch = base_ch[:, 1:] - base_ch[:, :-1]
                    ring_stats.append(inter_ch)

            all_ring_features.append(torch.cat(ring_stats, dim=1))

        # Stack per-ring features: (B, R, per_ring_dim)
        base_2d = torch.stack(all_ring_features, dim=1)

        # Inter-ring conv: capture vertical structure across rings
        if self.inter_ring_conv is not None:
            ir_in = base_2d.transpose(1, 2)  # (B, D, R)
            ir_out = F.relu(self.inter_ring_conv(ir_in))  # (B, k, R)
            ir_out = ir_out.transpose(1, 2)  # (B, R, k)
            combined = torch.cat([base_2d, ir_out], dim=2)  # (B, R, D+k)
        else:
            combined = base_2d

        out = combined.reshape(B, -1)
        if self.proj is not None:
            out = self.proj(out)
        return self.norm(out)


# =========================================================================
# Option E: Gated Frequency Selection
# =========================================================================

class GatedFrequencySelection(SpectralPolicyBase):
    """Per-ring learned soft frequency masks.

    A small MLP generates a sigmoid gate from per-ring summary statistics,
    then pools the gated spectrum into fixed-dim features.
    """

    def __init__(
        self,
        n_rings: int,
        n_freqs: int,
        output_dim: int,
        gate_hidden: int = 64,
    ):
        super().__init__(n_rings, n_freqs, output_dim)

        # Gate network: per-ring stats → sigmoid mask
        # Input: [mean, std, max] per ring = 3 features
        self.gate_mlp = nn.Sequential(
            nn.Linear(3, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, n_freqs),
            nn.Sigmoid(),
        )

        # After gating, pool to [mean, std, max] = 3 features per ring
        pool_dim = n_rings * 3
        if pool_dim != output_dim:
            self.proj = nn.Linear(pool_dim, output_dim)
        else:
            self.proj = None

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, R, F_ = x.shape

        # Per-ring summary statistics for gate input
        ring_mean = x.mean(dim=2)   # (B, R)
        ring_std = x.std(dim=2)     # (B, R)
        ring_max = x.max(dim=2)[0]  # (B, R)

        # Gate input: (B, R, 3)
        gate_input = torch.stack([ring_mean, ring_std, ring_max], dim=2)

        # Generate gates: (B, R, 3) → (B*R, 3) → MLP → (B*R, F) → (B, R, F)
        gate_flat = gate_input.reshape(B * R, 3)
        gate = self.gate_mlp(gate_flat).reshape(B, R, F_)

        # Apply gate
        masked = x * gate  # (B, R, F)

        # Pool statistics
        pool_mean = masked.mean(dim=2)   # (B, R)
        pool_std = masked.std(dim=2)     # (B, R)
        pool_max = masked.max(dim=2)[0]  # (B, R)

        out = torch.cat([pool_mean, pool_std, pool_max], dim=1)  # (B, R*3)

        if self.proj is not None:
            out = self.proj(out)

        return self.norm(out)


# =========================================================================
# Factory
# =========================================================================

POLICY_REGISTRY = {
    'linear': LearnedFilterbank,
    'conv1d': ConvSpectralPool,
    'attention': CrossAttentionPool,
    'soft_binning': SoftBinning,
    'gated': GatedFrequencySelection,
}


def create_spectral_policy(
    config: dict,
    n_rings: int = 79,
    n_freqs: int = 181,
) -> SpectralPolicyBase:
    """Create a spectral policy from YAML config dict.

    Args:
        config: The 'spectral_policy' subsection of encoding config.
        n_rings: Number of rings (rows) in FFT magnitude.
        n_freqs: Number of frequency bins in FFT magnitude.

    Returns:
        SpectralPolicyBase instance.
    """
    policy_type = config.get('type', 'soft_binning')
    output_dim = config.get('output_dim', 1106)
    shared = config.get('shared_across_rings', True)

    if policy_type not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown spectral policy type '{policy_type}'. "
            f"Available: {list(POLICY_REGISTRY.keys())}"
        )

    # Type-specific kwargs
    if policy_type == 'linear':
        sub = config.get('linear', {})
        d_per_ring = sub.get('d_per_ring', None)
        # Auto-compute output_dim from d_per_ring if specified
        if d_per_ring is not None and shared:
            output_dim = n_rings * d_per_ring
        return LearnedFilterbank(
            n_rings=n_rings, n_freqs=n_freqs, output_dim=output_dim,
            shared_across_rings=shared,
        )

    elif policy_type == 'conv1d':
        sub = config.get('conv1d', {})
        return ConvSpectralPool(
            n_rings=n_rings, n_freqs=n_freqs, output_dim=output_dim,
            channels_per_group=sub.get('channels_per_group', 2),
            kernel_size=sub.get('kernel_size', 7),
        )

    elif policy_type == 'attention':
        sub = config.get('attention', {})
        return CrossAttentionPool(
            n_rings=n_rings, n_freqs=n_freqs, output_dim=output_dim,
            n_queries=sub.get('n_queries', 7),
            n_heads=sub.get('n_heads', 2),
            head_dim=sub.get('head_dim', 32),
            d_pe=sub.get('d_pe', 16),
        )

    elif policy_type == 'soft_binning':
        sub = config.get('soft_binning', {})
        return SoftBinning(
            n_rings=n_rings, n_freqs=n_freqs, output_dim=output_dim,
            n_soft_bins=sub.get('n_soft_bins', 4),
            stats=sub.get('stats', ['mean', 'std']),
            inter_stats=sub.get('inter_stats', ['diff']),
            init_from_fixed=sub.get('init_from_fixed', True),
            alpha=sub.get('alpha', 2.0),
            shared_across_rings=shared,
            init_mode=sub.get('init_mode', None),
            inter_ring=sub.get('inter_ring', None),
        )

    elif policy_type == 'gated':
        sub = config.get('gated', {})
        return GatedFrequencySelection(
            n_rings=n_rings, n_freqs=n_freqs, output_dim=output_dim,
            gate_hidden=sub.get('gate_hidden', 64),
        )
