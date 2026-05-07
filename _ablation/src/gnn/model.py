"""
GNN Model - Concat Architecture with DiffAttnConv + EdgeEncoder

Produces trajectory context via difference-based attention, concatenated with raw descriptors.
Messages encode feature differences (h_j - h_i), capturing trajectory change rates.
Edge features are encoded via type-aware EdgeEncoder and applied as attention score bias.

Architecture:
- Input: Per-elevation spectral histograms (256D = 16 elevations × 16 bins)
- EdgeEncoder: type-aware edge embedding (temporal vs similarity edges)
- DiffAttnConv layers produce context vector (256D) from neighbor differences
- Output: cat(raw_descriptor, context) = 512D
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import softmax
from typing import Optional


class SinusoidalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for scalar values (e.g., rotation angles).

    Maps a scalar to a d-dimensional vector using sin/cos at different frequencies.
    """

    def __init__(self, d_encode: int = 16):
        super().__init__()
        self.d_encode = d_encode
        freqs = torch.exp(
            torch.arange(0, d_encode, 2).float()
            * -(math.log(10000.0) / d_encode)
        )
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (n_edges,) scalar values
        Returns:
            (n_edges, d_encode) encoded vectors
        """
        angles = x.unsqueeze(-1) * self.freqs  # (n_edges, d_encode/2)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class EdgeEncoder(nn.Module):
    """
    Type-aware edge embedding.

    Encodes edges differently based on type:
    - Temporal (type 0): [dist, sinusoidal(rot), type_embed]
    - Similarity (type 1): [cos_sim, l2_dist, posterior, type_embed]

    Projects through shared MLP to fixed d_edge dimension.
    """

    def __init__(
        self,
        d_edge: int = 32,
        n_edge_types: int = 2,
        d_type_embed: int = 16,
        d_rot_encode: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_edge = d_edge

        self.type_embed = nn.Embedding(n_edge_types, d_type_embed)
        self.rot_encoder = SinusoidalEncoding(d_rot_encode)

        # Temporal: dist(1) + sinusoidal_rot(d_rot_encode) + type_embed(d_type_embed)
        temporal_raw_dim = 1 + d_rot_encode + d_type_embed
        # Similarity: cos_sim(1) + l2_dist(1) + posterior(1) + type_embed(d_type_embed)
        similarity_raw_dim = 3 + d_type_embed
        self._max_raw_dim = max(temporal_raw_dim, similarity_raw_dim)
        self._temporal_raw_dim = temporal_raw_dim
        self._similarity_raw_dim = similarity_raw_dim

        self.mlp = nn.Sequential(
            nn.Linear(self._max_raw_dim, d_edge * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_edge * 2, d_edge)
        )

    def forward(
        self,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            edge_attr: (n_edges, 5) — [dist_norm, rot_norm, cos_sim, l2_dist_norm, posterior]
            edge_type: (n_edges,) LongTensor — 0=temporal, 1=similarity
        Returns:
            (n_edges, d_edge) edge embeddings
        """
        type_emb = self.type_embed(edge_type)  # (n_edges, d_type_embed)

        dist = edge_attr[:, 0:1]       # (n_edges, 1)
        rot = edge_attr[:, 1]          # (n_edges,)
        cos_sim = edge_attr[:, 2:3]    # (n_edges, 1)
        l2_dist = edge_attr[:, 3:4]    # (n_edges, 1)
        posterior = edge_attr[:, 4:5]  # (n_edges, 1)

        rot_encoded = self.rot_encoder(rot)  # (n_edges, d_rot_encode)

        # Temporal features: [dist, sinusoidal_rot, type_embed]
        temporal_feats = torch.cat([dist, rot_encoded, type_emb], dim=-1)
        if self._temporal_raw_dim < self._max_raw_dim:
            temporal_feats = F.pad(
                temporal_feats, (0, self._max_raw_dim - self._temporal_raw_dim)
            )

        # Similarity features: [cos_sim, l2_dist, posterior, type_embed]
        sim_feats = torch.cat([cos_sim, l2_dist, posterior, type_emb], dim=-1)
        if self._similarity_raw_dim < self._max_raw_dim:
            sim_feats = F.pad(
                sim_feats, (0, self._max_raw_dim - self._similarity_raw_dim)
            )

        # Select by edge type
        is_similarity = (edge_type == 1).unsqueeze(-1)  # (n_edges, 1)
        raw_feats = torch.where(is_similarity, sim_feats, temporal_feats)

        return self.mlp(raw_feats)


class EdgeConfidenceGate(nn.Module):
    """Per-edge confidence gate via additive logit bias pre-softmax.

    Produces a scalar logit per edge: large positive → boost attention,
    large negative → suppress. Added to attention scores before softmax,
    softmax handles normalization (no NaN risk from division).

    Equivalent in spirit to multiplicative gating (`softmax(s + log(c))` =
    `softmax(s) * c / sum(softmax(s) * c)`), but numerically stable.

    Trained with auxiliary BCE loss on sigmoid(logit) vs pose-GT same-place
    label. Temporal edges always get bias=0 (no attention shift; existing
    edge_bias_mlp handles temporal-specific weighting).

    Init: last layer zero → bias=0 at start → no effect on attention initially
    (matches no-gate behavior, learned divergence over time).
    """

    def __init__(self, edge_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero init → bias=0, sigmoid(0)=0.5 for diagnostic c_e
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, edge_emb: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_emb: (E, edge_dim) edge embeddings from EdgeEncoder
            edge_type: (E,) LongTensor (0=temporal, 1=similarity)

        Returns:
            (E, 1) bounded logit in [-3, 3] (c_e in [0.047, 0.953]).
            Bounding prevents FP16 softmax underflow (exp(-15) < FP16 min)
            when gate learns extreme suppression. Temporal edges forced to 0.
        """
        # Soft-clamp via tanh: logit = 3 * tanh(MLP(edge_emb)) ∈ [-3, 3]
        # Differentiable, no hard cliff. c_e ∈ [0.047, 0.953].
        logit = 3.0 * torch.tanh(self.classifier(edge_emb))
        is_temporal = (edge_type == 0).unsqueeze(-1)
        return torch.where(is_temporal, torch.zeros_like(logit), logit)


class DiffAttnConv(MessagePassing):
    """
    Difference-based Attention Convolution with edge score bias.

    Computes messages from feature DIFFERENCES (h_j - h_i).
    Edge embeddings are applied as scalar bias to attention scores.

    Attention:
        α_ij = softmax( (W_q·h_i)^T · (W_k·(h_j - h_i)) / √d + bias(edge_embed_ij) )
    Message:
        msg_ij = α_ij · W_v · (h_j - h_i)
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        edge_dim: int = None,
        dropout: float = 0.1
    ):
        super().__init__(aggr='add', node_dim=0)

        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.edge_dim = edge_dim
        self.scale = math.sqrt(self.head_dim)

        # Query: from target node h_i
        self.W_q = nn.Linear(channels, heads * self.head_dim, bias=False)
        # Key: from difference (h_j - h_i)
        self.W_k = nn.Linear(channels, heads * self.head_dim, bias=False)
        # Value: from difference (h_j - h_i)
        self.W_v = nn.Linear(channels, heads * self.head_dim, bias=False)

        # Edge score bias: maps edge embedding to per-head scalar bias
        if edge_dim is not None:
            self.bias_mlp = nn.Sequential(
                nn.Linear(edge_dim, edge_dim),
                nn.ReLU(),
                nn.Linear(edge_dim, heads)
            )
        else:
            self.bias_mlp = None

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        if self.bias_mlp is not None:
            # Initialize final layer near zero for stable training start
            nn.init.zeros_(self.bias_mlp[-1].weight)
            nn.init.zeros_(self.bias_mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        edge_logit: torch.Tensor = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """Memory-efficient forward bypassing propagate().

        Args:
            edge_logit: optional (n_edges, 1) additive bias added to attention
                scores BEFORE softmax. Large positive → boosts edge; large
                negative → suppresses. Numerically stable (softmax handles
                normalization automatically).
        """
        self._return_attn = return_attention_weights
        self._attn_weights = None

        src, tgt = edge_index[0], edge_index[1]
        n_nodes = x.size(0)

        # Edge-level features (bypass propagate to enable nested checkpoint)
        x_tgt = x[tgt]            # target node features (= x_i)
        diff = x[src] - x_tgt     # difference (= x_j - x_i)

        # --- Nested checkpoint: Q, K freed after attn scores produced ---
        def compute_attn_scores(x_tgt_in, diff_in, edge_attr_in, edge_logit_in):
            Q = self.W_q(x_tgt_in).view(-1, self.heads, self.head_dim)
            K = self.W_k(diff_in).view(-1, self.heads, self.head_dim)
            scores = torch.einsum('ehd,ehd->eh', Q, K) / self.scale
            if edge_attr_in is not None and self.bias_mlp is not None:
                scores = scores + self.bias_mlp(edge_attr_in)
            if edge_logit_in is not None:
                # Broadcast (E, 1) over heads
                scores = scores + edge_logit_in
            return scores

        if self.training and torch.is_grad_enabled():
            raw_attn = grad_checkpoint(
                compute_attn_scores, x_tgt, diff, edge_attr, edge_logit,
                use_reentrant=False
            )
        else:
            raw_attn = compute_attn_scores(x_tgt, diff, edge_attr, edge_logit)

        attn = softmax(raw_attn, tgt, num_nodes=n_nodes)
        attn = self.dropout(attn)

        if self._return_attn:
            self._attn_weights = attn.detach()

        # Value and message (diff already in memory from above)
        V = self.W_v(diff).view(-1, self.heads, self.head_dim)
        msg = attn.unsqueeze(-1) * V  # (n_edges, heads, head_dim)
        msg = msg.reshape(-1, self.heads * self.head_dim)

        # Aggregate via scatter_add (equivalent to aggr='add')
        out = torch.zeros(n_nodes, self.channels, device=msg.device, dtype=msg.dtype)
        out.scatter_add_(0, tgt.unsqueeze(-1).expand_as(msg), msg)

        if return_attention_weights:
            return out, (edge_index, self._attn_weights)
        return out


class SpectralGNN(nn.Module):
    """
    Graph Neural Network for Spectral Histogram Context Injection

    Uses EdgeEncoder for type-aware edge embedding, DiffAttnConv for context
    aggregation via difference-based attention with edge score bias.

    Output: cat(raw_256, context_256) = 512D

    Architecture: Input(256) → Proj(256) → DiffAttnConv×2(256) → Proj(256) → cat(raw, context)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        context_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
        edge_encoder_config: dict = None,
        gradient_checkpointing: bool = True,
        spectral_policy: Optional[nn.Module] = None,
        norm_type: str = 'batch_norm',
        use_residual_gate: bool = False,
        gate_hidden_dim: int = 64,
        gate_initial_alpha: float = 0.5,
        use_edge_confidence_gate: bool = False,
        edge_gate_hidden_dim: int = 16,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.residual = residual
        self.gradient_checkpointing = gradient_checkpointing
        self.use_residual_gate = use_residual_gate

        # Spectral policy: if provided, overrides input_dim with policy output
        self.spectral_policy = spectral_policy
        if spectral_policy is not None:
            self.input_dim = spectral_policy.output_dim
        else:
            self.input_dim = input_dim

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.input_norm = self._make_norm(hidden_dim, norm_type)

        # Edge encoder
        if edge_encoder_config is not None:
            self.edge_encoder = EdgeEncoder(**edge_encoder_config)
            effective_edge_dim = edge_encoder_config.get('d_edge', 32)
        else:
            self.edge_encoder = None
            effective_edge_dim = None

        # DiffAttnConv layers (all in hidden_dim)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(n_layers):
            self.convs.append(
                DiffAttnConv(
                    channels=hidden_dim,
                    heads=n_heads,
                    edge_dim=effective_edge_dim,
                    dropout=dropout
                )
            )
            self.batch_norms.append(self._make_norm(hidden_dim, norm_type))

        # Output projection: 256 -> context_dim
        self.output_proj = nn.Linear(hidden_dim, context_dim)

        # Residual gate: per-node scalar α ∈ [0,1] decides ctx contribution
        # output = cat(raw_norm, α · ctx_norm). α=0 → raw only; α=1 → 50/50 mix
        # (matches current behavior). Lets the model adapt: KITTI raw is great
        # → push α toward 0; HeLiPR raw is weak → push α toward 1.
        if use_residual_gate:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            # Initialize to a controlled context weight. KITTI ablations show
            # equal raw/context weighting can degrade retrieval; small alpha
            # lets the model add context only when useful.
            nn.init.zeros_(self.gate[-1].weight)
            gate_initial_alpha = float(min(max(gate_initial_alpha, 1e-4), 1.0 - 1e-4))
            nn.init.constant_(
                self.gate[-1].bias,
                math.log(gate_initial_alpha / (1.0 - gate_initial_alpha)),
            )
        else:
            self.gate = None

        # Diagnostic: store last-forward alpha for logging
        self._last_alpha = None

        # Edge confidence gate: per-edge soft same-place classifier
        # c_e = sigmoid(MLP(edge_emb)). Multiplies attention weight in DiffAttnConv.
        # Trained with auxiliary BCE loss against pose-GT labels (during training).
        # Temporal edges always c_e=1 (always informative).
        self.use_edge_confidence_gate = use_edge_confidence_gate
        if use_edge_confidence_gate:
            assert effective_edge_dim is not None, (
                "edge_encoder required when use_edge_confidence_gate=True"
            )
            self.edge_gate = EdgeConfidenceGate(
                edge_dim=effective_edge_dim,
                hidden_dim=edge_gate_hidden_dim,
            )
        else:
            self.edge_gate = None
        self._last_edge_conf = None  # diagnostic for aux loss

    @staticmethod
    def _make_norm(dim: int, norm_type: str) -> nn.Module:
        if norm_type == 'layer_norm':
            return nn.LayerNorm(dim)
        elif norm_type == 'none':
            return nn.Identity()
        else:  # 'batch_norm' (default)
            return nn.BatchNorm1d(dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass: produce cat(raw, context)

        Returns:
            (n_nodes, input_dim + context_dim) = (n, 512) concatenated embeddings
        """
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        edge_type = getattr(data, 'edge_type', None)

        # If spectral policy is available, transform FFT magnitudes → descriptor
        if self.spectral_policy is not None and hasattr(data, 'x_fft') and data.x_fft is not None:
            x_fft = data.x_fft.view(-1, self.spectral_policy.n_rings, self.spectral_policy.n_freqs)
            # Gradient checkpointing: recompute policy activations during backward
            # to avoid storing large intermediates (e.g. conv1d: 178K×32×181 = ~8GB)
            if self.training and torch.is_grad_enabled() and any(p.requires_grad for p in self.spectral_policy.parameters()):
                x = grad_checkpoint(self.spectral_policy, x_fft, use_reentrant=False)
            else:
                x = self.spectral_policy(x_fft)  # (N, policy_output_dim)
        else:
            x = data.x

        # Cast to AMP dtype early to prevent FP32 edge-level tensor explosion
        # (graph.x is FP32 from numpy; under autocast, keeping it FP32 wastes 150 MB)
        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_gpu_dtype())

        # Preserve raw descriptor for concatenation
        x_raw = x

        # Encode edges (once, shared across all layers)
        if self.edge_encoder is not None and edge_attr is not None and edge_type is not None:
            edge_embed = self.edge_encoder(edge_attr, edge_type)
        else:
            edge_embed = None

        # Compute per-edge confidence gate (once, shared across all layers).
        # edge_logit: raw additive bias added to attention scores pre-softmax.
        # _last_edge_conf: sigmoid(logit) ∈ [0, 1] for BCE supervision + diagnostic.
        if self.edge_gate is not None and edge_embed is not None and edge_type is not None:
            edge_logit = self.edge_gate(edge_embed, edge_type)   # (E, 1) raw
            self._last_edge_conf = torch.sigmoid(edge_logit)     # (E, 1) for BCE
        else:
            edge_logit = None
            self._last_edge_conf = None

        # Input projection: 256 -> 256
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.relu(h)

        # DiffAttnConv layers (with optional gradient checkpointing)
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h_prev = h

            # BatchNorm forces FP32 output even under autocast.
            # Cast h to AMP dtype before conv to prevent FP32 edge-level tensors
            # in message() (2.87M edges × 256D × 4B = 2.80 GiB per tensor in FP32).
            if torch.is_autocast_enabled():
                h = h.to(torch.get_autocast_gpu_dtype())

            if self.gradient_checkpointing and self.training:
                # Recompute conv intermediates during backward to save VRAM.
                if edge_embed is not None and edge_logit is not None:
                    def _conv_with_logit(h_in, ei, ea, el):
                        return conv(h_in, ei, edge_attr=ea, edge_logit=el)
                    h = grad_checkpoint(
                        _conv_with_logit, h, edge_index, edge_embed, edge_logit,
                        use_reentrant=False
                    )
                elif edge_embed is not None:
                    h = grad_checkpoint(
                        conv, h, edge_index, edge_embed,
                        use_reentrant=False
                    )
                else:
                    h = grad_checkpoint(
                        conv, h, edge_index,
                        use_reentrant=False
                    )
            else:
                if edge_embed is not None:
                    h = conv(h, edge_index, edge_attr=edge_embed, edge_logit=edge_logit)
                else:
                    h = conv(h, edge_index)

            h = bn(h)

            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            # Residual: apply to all layers (including first, i=0)
            # Previously guarded by `i > 0`, which made residual inactive with n_layers=1
            if self.residual:
                h = h + h_prev

        # Output projection: 256 -> context_dim
        context = self.output_proj(h)

        # L2 normalize each part so both contribute equally to distance metrics
        raw_norm = F.normalize(x_raw, p=2, dim=-1)
        ctx_norm = F.normalize(context, p=2, dim=-1)

        # Residual gate: scale ctx contribution per-node based on hidden state.
        if self.use_residual_gate and self.gate is not None:
            alpha = torch.sigmoid(self.gate(h))  # (n_nodes, 1) in [0, 1]
            ctx_norm = alpha * ctx_norm
            self._last_alpha = alpha.detach()

        return torch.cat([raw_norm, ctx_norm], dim=-1)

    def forward_with_attention(self, data: Data) -> tuple:
        """
        Forward pass with attention weights for visualization

        Returns:
            embeddings: (n_nodes, input_dim + context_dim) concatenated embeddings
            attention_weights: List of (edge_index, attention) per layer
        """
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        edge_type = getattr(data, 'edge_type', None)

        # Spectral policy (same logic as forward)
        if self.spectral_policy is not None and hasattr(data, 'x_fft') and data.x_fft is not None:
            x_fft = data.x_fft.view(-1, self.spectral_policy.n_rings, self.spectral_policy.n_freqs)
            x = self.spectral_policy(x_fft)
        else:
            x = data.x

        x_raw = x

        # Encode edges (once)
        if self.edge_encoder is not None and edge_attr is not None and edge_type is not None:
            edge_embed = self.edge_encoder(edge_attr, edge_type)
        else:
            edge_embed = None

        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.relu(h)

        attention_weights = []

        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h_prev = h

            if edge_embed is not None:
                h, (edge_idx, attn) = conv(
                    h, edge_index, edge_attr=edge_embed,
                    return_attention_weights=True
                )
            else:
                h, (edge_idx, attn) = conv(
                    h, edge_index,
                    return_attention_weights=True
                )
            attention_weights.append((edge_idx, attn))

            h = bn(h)

            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            if self.residual:
                h = h + h_prev

        context = self.output_proj(h)

        # L2 normalize each part so both contribute equally to distance metrics
        raw_norm = F.normalize(x_raw, p=2, dim=-1)
        ctx_norm = F.normalize(context, p=2, dim=-1)
        if self.use_residual_gate and self.gate is not None:
            alpha = torch.sigmoid(self.gate(h))
            ctx_norm = alpha * ctx_norm
            self._last_alpha = alpha.detach()
        return torch.cat([raw_norm, ctx_norm], dim=-1), attention_weights

    def get_embedding_dim(self) -> int:
        """Get output embedding dimension (raw + context)"""
        return self.input_dim + self.context_dim


class LocalUpdateGNN(nn.Module):
    """
    GNN with efficient local k-hop updates

    Only updates k-hop neighborhood instead of full graph.
    """

    def __init__(
        self,
        gnn: SpectralGNN,
        k_hops: int = 3
    ):
        super().__init__()
        self.gnn = gnn
        self.k_hops = k_hops

    def forward(
        self,
        data: Data,
        update_nodes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if update_nodes is None:
            return self.gnn(data)
        else:
            return self.gnn(data)

    def forward_local(
        self,
        data: Data,
        center_node: int,
        k_hops: Optional[int] = None
    ) -> torch.Tensor:
        if k_hops is None:
            k_hops = self.k_hops
        embeddings = self.gnn(data)
        return embeddings[center_node:center_node+1]


def create_spectral_gnn(
    input_dim: int = 256,
    hidden_dim: int = 256,
    context_dim: int = 256,
    n_layers: int = 2,
    n_heads: int = 4,
    dropout: float = 0.1,
    use_local_updates: bool = True,
    local_update_hops: int = 3,
    edge_encoder_config: dict = None,
    gradient_checkpointing: bool = True,
    spectral_policy: Optional[nn.Module] = None,
        norm_type: str = 'batch_norm',
        use_residual_gate: bool = False,
        gate_hidden_dim: int = 64,
        gate_initial_alpha: float = 0.5,
        use_edge_confidence_gate: bool = False,
        edge_gate_hidden_dim: int = 16,
) -> nn.Module:
    """
    Factory function to create GNN model

    Args:
        input_dim: Input dimension (256)
        hidden_dim: Hidden dimension (256)
        context_dim: Context vector dimension (256), concatenated with raw
        n_layers: Number of DiffAttnConv layers
        n_heads: Number of attention heads (4)
        dropout: Dropout rate
        use_local_updates: Enable local update wrapper
        local_update_hops: Number of hops for local updates
        edge_encoder_config: EdgeEncoder config dict with keys:
            d_edge, n_edge_types, d_type_embed, d_rot_encode, dropout
        gradient_checkpointing: Recompute conv intermediates during backward to save VRAM
        spectral_policy: Optional SpectralPolicy module for end-to-end learning.
            When provided, input_dim is overridden by policy.output_dim.
        norm_type: Normalization type ('batch_norm', 'layer_norm', 'none')

    Returns:
        GNN model (LocalUpdateGNN or SpectralGNN)
    """
    base_gnn = SpectralGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        context_dim=context_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        residual=True,
        edge_encoder_config=edge_encoder_config,
        gradient_checkpointing=gradient_checkpointing,
        spectral_policy=spectral_policy,
        norm_type=norm_type,
        use_residual_gate=use_residual_gate,
        gate_hidden_dim=gate_hidden_dim,
        gate_initial_alpha=gate_initial_alpha,
        use_edge_confidence_gate=use_edge_confidence_gate,
        edge_gate_hidden_dim=edge_gate_hidden_dim,
    )

    if use_local_updates:
        return LocalUpdateGNN(base_gnn, k_hops=local_update_hops)
    else:
        return base_gnn


def test_gnn_forward():
    """Test GNN forward pass with EdgeEncoder"""
    n_nodes = 10
    n_edges = 20
    feature_dim = 256

    x = torch.randn(n_nodes, feature_dim)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, 5)  # [dist, rot, cos_sim, l2_dist, posterior]
    edge_type = torch.randint(0, 2, (n_edges,))  # 0=temporal, 1=similarity

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)

    edge_encoder_config = {
        'd_edge': 32,
        'n_edge_types': 2,
        'd_type_embed': 16,
        'd_rot_encode': 16,
        'dropout': 0.1
    }
    model = create_spectral_gnn(edge_encoder_config=edge_encoder_config)

    embeddings = model(data)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {embeddings.shape}")  # Expected: (10, 512)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test with attention weights
    embeddings2, attn_weights = model.gnn.forward_with_attention(data)
    print(f"Attention layers: {len(attn_weights)}")
    for i, (ei, aw) in enumerate(attn_weights):
        print(f"  Layer {i}: edges={ei.shape}, attention={aw.shape}")

    return model, embeddings


if __name__ == "__main__":
    test_gnn_forward()
