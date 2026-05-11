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

from encoding.phase_coherence import ClosedFormPhaseEdgeBias
from encoding.phase_alignment import (
    feature_dim as phase_alignment_feature_dim,
    phase_alignment_edge_features,
)


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


class PhaseTokenProjector(nn.Module):
    """Trainable compressor for compact phase features."""

    def __init__(
        self,
        input_dim: int,
        token_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        mode: str = "mlp",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.mode = mode
        if mode == "identity":
            if self.token_dim != self.input_dim:
                raise ValueError(
                    "identity phase projector requires token_dim == input_dim "
                    f"(got token_dim={self.token_dim}, input_dim={self.input_dim})"
                )
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.token_dim),
            )

    def forward(self, x_phase: torch.Tensor) -> torch.Tensor:
        return self.net(x_phase)


class PhaseEdgeBias(nn.Module):
    """Phase-aware attention bias for graph edges.

    This module does not append phase to the final descriptor. It learns a
    scalar edge logit from phase consistency between source/target nodes and
    injects it into GAT attention before softmax.
    """

    def __init__(
        self,
        input_dim: int,
        key_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        max_logit: float = 2.0,
        similarity_only: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.key_dim = int(key_dim)
        self.max_logit = float(max_logit)
        self.similarity_only = bool(similarity_only)

        self.norm = nn.LayerNorm(self.input_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.key_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * self.key_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Starts exactly as the old GAT; training must earn any phase effect.
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)

    def forward(
        self,
        x_phase: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, tgt = edge_index[0], edge_index[1]
        z = self.node_proj(self.norm(x_phase))
        z = F.normalize(z, p=2, dim=-1)
        z_src = z[src]
        z_tgt = z[tgt]
        cos = (z_src * z_tgt).sum(dim=-1, keepdim=True)
        signed_delta = z_src - z_tgt
        feats = torch.cat([
            torch.abs(signed_delta),
            signed_delta,
            z_src * z_tgt,
            cos,
        ], dim=-1)
        logit = self.max_logit * torch.tanh(self.edge_mlp(feats))

        if self.similarity_only and edge_type is not None:
            is_similarity = (edge_type == 1).unsqueeze(-1)
            logit = torch.where(is_similarity, logit, torch.zeros_like(logit))
        return logit


class PhaseAlignmentEdgeBias(nn.Module):
    """Leakage-controlled phase-alignment bias/value gate for GAT edges.

    Unlike :class:`PhaseEdgeBias`, this module does not learn phase consistency
    from node embeddings. It computes explicit cyclic-shift alignment features
    from raw complex Fourier phase coefficients and feeds only a small feature
    vector to an MLP. The raw alignment score is disabled by default to avoid
    trivially copying the closed-form phase-sketch reranker.
    """

    def __init__(
        self,
        n_rows: int,
        n_freqs: int,
        n_sectors: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        max_logit: float = 2.0,
        include_score: bool = False,
        similarity_only: bool = True,
        entropy_temperature: float = 0.05,
    ):
        super().__init__()
        self.n_rows = int(n_rows)
        self.n_freqs = int(n_freqs)
        self.n_sectors = int(n_sectors)
        self.include_score = bool(include_score)
        self.similarity_only = bool(similarity_only)
        self.entropy_temperature = float(entropy_temperature)
        self.max_logit = float(max_logit)
        self.input_dim = phase_alignment_feature_dim(self.include_score)

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Start as old GAT; training must earn any phase-conditioned effect.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def features(
        self,
        x_phase: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return phase_alignment_edge_features(
            x_phase=x_phase,
            edge_index=edge_index,
            edge_type=edge_type,
            n_rows=self.n_rows,
            n_freqs=self.n_freqs,
            n_sectors=self.n_sectors,
            include_score=self.include_score,
            similarity_only=self.similarity_only,
            entropy_temperature=self.entropy_temperature,
        )

    def forward(
        self,
        x_phase: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x_phase.float(), edge_index, edge_type)
        logit = self.max_logit * torch.tanh(self.mlp(feats))
        if self.similarity_only and edge_type is not None:
            is_similarity = (edge_type == 1).unsqueeze(-1)
            logit = torch.where(is_similarity, logit, torch.zeros_like(logit))
        return logit, feats


class DiffAttnConv(MessagePassing):
    """
    Difference-based Attention Convolution with edge score bias.

    Computes attention from feature DIFFERENCES (h_j - h_i). By default the
    value/message path also uses differences for backward compatibility. The
    optional ``value_source='abs_diff'`` adds an absolute-neighbor value branch
    W_abs h_j so the layer keeps keyframe identity while retaining diff-based
    attention.
    Edge embeddings are applied as scalar bias to attention scores.

    Attention:
        α_ij = softmax( (W_q·h_i)^T · (W_k·(h_j - h_i)) / √d + bias(edge_embed_ij) )
    Message:
        msg_ij = α_ij · (W_v · (h_j - h_i) [+ W_abs · h_j])
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        edge_dim: int = None,
        dropout: float = 0.1,
        value_source: str = "diff",
    ):
        super().__init__(aggr='add', node_dim=0)

        if value_source not in {"diff", "abs_diff"}:
            raise ValueError(
                f"Unknown DiffAttnConv value_source={value_source!r}; "
                "expected 'diff' or 'abs_diff'."
            )

        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.edge_dim = edge_dim
        self.scale = math.sqrt(self.head_dim)
        self.value_source = value_source

        # Query: from target node h_i
        self.W_q = nn.Linear(channels, heads * self.head_dim, bias=False)
        # Key: from difference (h_j - h_i)
        self.W_k = nn.Linear(channels, heads * self.head_dim, bias=False)
        # Value: from difference (h_j - h_i)
        self.W_v = nn.Linear(channels, heads * self.head_dim, bias=False)
        # Optional absolute-neighbor value branch. This fixes the information
        # loss of pure differences without changing attention semantics.
        if value_source == "abs_diff":
            self.W_v_abs = nn.Linear(channels, heads * self.head_dim, bias=False)
        else:
            self.W_v_abs = None

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
        if self.W_v_abs is not None:
            # Start exactly as the original diff-only operator. Training must
            # earn any absolute-neighbor contribution, which keeps ablations
            # stable and prevents random value noise at epoch 0.
            nn.init.zeros_(self.W_v_abs.weight)
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
        edge_value_gate: torch.Tensor = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """Memory-efficient forward bypassing propagate().

        Args:
            edge_logit: optional (n_edges, 1) additive bias added to attention
                scores BEFORE softmax. Large positive → boosts edge; large
                negative → suppresses. Numerically stable (softmax handles
                normalization automatically).
            edge_value_gate: optional (n_edges, 1) multiplicative gate applied
                to messages AFTER attention. Used for phase-conditioned
                difference propagation without changing descriptor dimension.
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
        value = self.W_v(diff)
        if self.W_v_abs is not None:
            value = value + self.W_v_abs(x[src])
        V = value.view(-1, self.heads, self.head_dim)
        msg = attn.unsqueeze(-1) * V  # (n_edges, heads, head_dim)
        if edge_value_gate is not None:
            msg = msg * edge_value_gate.unsqueeze(-1)
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
        phase_token_config: Optional[dict] = None,
        phase_edge_config: Optional[dict] = None,
        phase_alignment_config: Optional[dict] = None,
        phase_coherence_config: Optional[dict] = None,
        sensor_gate_config: Optional[dict] = None,
        diffattn_value_source: str = "diff",
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
        self.phase_token_config = phase_token_config or {}
        self.phase_edge_config = phase_edge_config or {}
        self.phase_alignment_config = phase_alignment_config or {}
        self.phase_coherence_config = phase_coherence_config or {}
        self.sensor_gate_config = sensor_gate_config or {}
        self.diffattn_value_source = diffattn_value_source

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
                    dropout=dropout,
                    value_source=diffattn_value_source,
                )
            )
            self.batch_norms.append(self._make_norm(hidden_dim, norm_type))

        # Output projection: 256 -> context_dim
        self.output_proj = nn.Linear(hidden_dim, context_dim)

        # Residual gate: per-node scalar α ∈ [0,1] decides ctx contribution
        # output = cat(raw_norm, α · ctx_norm). α=0 → raw only; α=1 → 50/50 mix
        # (matches current behavior). Lets the model adapt: KITTI raw is great
        # → push α toward 0; HeLiPR raw is weak → push α toward 1.
        self.sensor_gate_enabled = bool(self.sensor_gate_config.get("enabled", False))
        self.sensor_embed = None
        self.beam_embed = None
        self._sensor_name_to_id = {
            "kitti": 0,
            "nclt": 1,
            "helipr": 2,
            "mulran": 3,
        }
        gate_input_dim = hidden_dim
        if self.sensor_gate_enabled:
            self.num_sensors = int(self.sensor_gate_config.get("num_sensors", 4))
            self.unknown_sensor_id = self.num_sensors
            sensor_embed_dim = int(self.sensor_gate_config.get("sensor_embed_dim", 8))
            self.sensor_embed = nn.Embedding(self.num_sensors + 1, sensor_embed_dim)
            gate_input_dim += sensor_embed_dim

            if bool(self.sensor_gate_config.get("use_beam_count", True)):
                beam_embed_dim = int(self.sensor_gate_config.get("beam_embed_dim", 4))
                self.beam_embed = nn.Sequential(
                    nn.Linear(1, beam_embed_dim),
                    nn.ReLU(),
                    nn.Linear(beam_embed_dim, beam_embed_dim),
                )
                gate_input_dim += beam_embed_dim
            self.default_beam_count = float(self.sensor_gate_config.get("default_beam_count", 64.0))
            self.alpha_max_by_sensor = self._make_sensor_scalar_table(
                self.sensor_gate_config.get("alpha_max", 1.0),
                default=1.0,
            )
        else:
            self.alpha_max_by_sensor = None

        if use_residual_gate:
            self.gate = nn.Sequential(
                nn.Linear(gate_input_dim, gate_hidden_dim),
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
        self._last_alpha_for_loss = None

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

        if self.phase_edge_config.get("enabled", False):
            phase_edge_input_dim = int(self.phase_edge_config["input_dim"])
            self.phase_edge_bias = PhaseEdgeBias(
                input_dim=phase_edge_input_dim,
                key_dim=int(self.phase_edge_config.get("key_dim", 32)),
                hidden_dim=int(self.phase_edge_config.get("hidden_dim", 64)),
                dropout=float(self.phase_edge_config.get("dropout", dropout)),
                max_logit=float(self.phase_edge_config.get("max_logit", 2.0)),
                similarity_only=bool(self.phase_edge_config.get("similarity_only", True)),
            )
        else:
            self.phase_edge_bias = None
        self._last_phase_edge_conf = None
        self._last_phase_edge_logit = None
        self.phase_edge_value_scale = float(self.phase_edge_config.get("value_scale", 0.0))

        if self.phase_alignment_config.get("enabled", False):
            self.phase_alignment_bias = PhaseAlignmentEdgeBias(
                n_rows=int(self.phase_alignment_config["n_rows"]),
                n_freqs=int(self.phase_alignment_config["n_freqs"]),
                n_sectors=int(self.phase_alignment_config.get("n_sectors", 60)),
                hidden_dim=int(self.phase_alignment_config.get("hidden_dim", 32)),
                dropout=float(self.phase_alignment_config.get("dropout", dropout)),
                max_logit=float(self.phase_alignment_config.get("max_logit", 2.0)),
                include_score=bool(self.phase_alignment_config.get("include_score", False)),
                similarity_only=bool(self.phase_alignment_config.get("similarity_only", True)),
                entropy_temperature=float(
                    self.phase_alignment_config.get("entropy_temperature", 0.05)
                ),
            )
        else:
            self.phase_alignment_bias = None
        self.phase_alignment_value_scale = float(
            self.phase_alignment_config.get("value_scale", 0.0)
        )
        self._last_phase_alignment_logit = None
        self._last_phase_alignment_conf = None
        self._last_phase_alignment_features = None

        # Closed-form (parameter-free) phase coherence edge bias.
        # Stacks additively with the learned PhaseEdgeBias above.
        if self.phase_coherence_config.get("enabled", False):
            self.phase_coherence_bias = ClosedFormPhaseEdgeBias(
                n_rows=int(self.phase_coherence_config["n_rows"]),
                n_freqs=int(self.phase_coherence_config["n_freqs"]),
                scale=float(self.phase_coherence_config.get("scale", 2.0)),
                mode=str(self.phase_coherence_config.get("mode", "poc")),
                pad_factor=int(self.phase_coherence_config.get("pad_factor", 4)),
                similarity_only=bool(self.phase_coherence_config.get("similarity_only", True)),
                center=bool(self.phase_coherence_config.get("center", True)),
            )
        else:
            self.phase_coherence_bias = None
        self._last_phase_coherence_logit = None
        self._last_edge_logit = None

        if self.phase_token_config.get("enabled", False):
            phase_input_dim = int(self.phase_token_config["input_dim"])
            phase_token_dim = int(self.phase_token_config.get("token_dim", 64))
            self.phase_projector = PhaseTokenProjector(
                input_dim=phase_input_dim,
                token_dim=phase_token_dim,
                hidden_dim=int(self.phase_token_config.get("hidden_dim", 128)),
                dropout=float(self.phase_token_config.get("dropout", dropout)),
                mode=str(self.phase_token_config.get("mode", "mlp")),
            )
            self.phase_token_dim = phase_token_dim
            phase_initial_alpha = float(self.phase_token_config.get("initial_alpha", 0.25))
            phase_initial_alpha = min(max(phase_initial_alpha, 1e-4), 1.0 - 1e-4)
            self.phase_logit = nn.Parameter(
                torch.tensor(math.log(phase_initial_alpha / (1.0 - phase_initial_alpha)))
            )
            self._last_phase_alpha = None
        else:
            self.phase_projector = None
            self.phase_token_dim = 0
            self.phase_logit = None
            self._last_phase_alpha = None

    @staticmethod
    def _make_norm(dim: int, norm_type: str) -> nn.Module:
        if norm_type == 'layer_norm':
            return nn.LayerNorm(dim)
        elif norm_type == 'none':
            return nn.Identity()
        else:  # 'batch_norm' (default)
            return nn.BatchNorm1d(dim)

    def _gate_input(self, h: torch.Tensor, data: Data) -> torch.Tensor:
        """Build residual-gate input with optional sensor metadata.

        ``sensor_id`` is a per-node long tensor. Unknown or missing ids map to
        the final embedding row. ``beam_count`` is normalized continuously so a
        new sensor can still fall back on beam density even without a learned
        discrete token.
        """
        if not self.sensor_gate_enabled:
            return h

        n_nodes = h.size(0)
        parts = [h]

        if self.sensor_embed is not None:
            sensor_id = getattr(data, "sensor_id", None)
            if sensor_id is None:
                sensor_id = torch.full(
                    (n_nodes,),
                    self.unknown_sensor_id,
                    device=h.device,
                    dtype=torch.long,
                )
            elif not torch.is_tensor(sensor_id):
                sensor_id = torch.as_tensor(sensor_id, device=h.device, dtype=torch.long)
            else:
                sensor_id = sensor_id.to(device=h.device, dtype=torch.long)

            sensor_id = sensor_id.reshape(-1)
            if sensor_id.numel() == 1:
                sensor_id = sensor_id.expand(n_nodes)
            elif sensor_id.numel() != n_nodes:
                raise RuntimeError(
                    f"sensor_id must have 1 or {n_nodes} entries, got {sensor_id.numel()}"
                )
            sensor_id = sensor_id.clamp(min=0, max=self.unknown_sensor_id)
            parts.append(self.sensor_embed(sensor_id))

        if self.beam_embed is not None:
            beam_count = getattr(data, "beam_count", None)
            if beam_count is None:
                beam_count = torch.full(
                    (n_nodes, 1),
                    self.default_beam_count,
                    device=h.device,
                    dtype=h.dtype,
                )
            elif not torch.is_tensor(beam_count):
                beam_count = torch.as_tensor(beam_count, device=h.device, dtype=h.dtype)
            else:
                beam_count = beam_count.to(device=h.device, dtype=h.dtype)

            beam_count = beam_count.reshape(-1, 1)
            if beam_count.size(0) == 1:
                beam_count = beam_count.expand(n_nodes, 1)
            elif beam_count.size(0) != n_nodes:
                raise RuntimeError(
                    f"beam_count must have 1 or {n_nodes} entries, got {beam_count.size(0)}"
                )
            beam_norm = torch.log(torch.clamp(beam_count, min=1.0)) / math.log(64.0)
            parts.append(self.beam_embed(beam_norm))

        return torch.cat(parts, dim=-1)

    def _make_sensor_scalar_table(self, value, default: float) -> torch.Tensor:
        """Build a length ``num_sensors + 1`` table from scalar/list/dict config."""
        table = torch.full((self.num_sensors + 1,), float(default), dtype=torch.float32)
        if isinstance(value, (int, float)):
            table.fill_(float(value))
            return table
        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value[: self.num_sensors]):
                table[idx] = float(item)
            return table
        if isinstance(value, dict):
            fallback = value.get("default", default)
            table.fill_(float(fallback))
            for key, item in value.items():
                if key == "default":
                    continue
                idx = self._sensor_name_to_id.get(str(key).lower(), None)
                if idx is None:
                    try:
                        idx = int(key)
                    except (TypeError, ValueError):
                        continue
                if 0 <= idx < self.num_sensors:
                    table[idx] = float(item)
            return table
        return table

    def _sensor_ids_for_nodes(self, data: Data, n_nodes: int, device: torch.device) -> torch.Tensor:
        """Return clamped per-node sensor ids, using unknown id when missing."""
        sensor_id = getattr(data, "sensor_id", None)
        if sensor_id is None:
            sensor_id = torch.full(
                (n_nodes,), self.unknown_sensor_id, dtype=torch.long, device=device
            )
        elif not torch.is_tensor(sensor_id):
            sensor_id = torch.as_tensor(sensor_id, dtype=torch.long, device=device)
        else:
            sensor_id = sensor_id.to(device=device, dtype=torch.long)
        sensor_id = sensor_id.reshape(-1)
        if sensor_id.numel() == 1:
            sensor_id = sensor_id.expand(n_nodes)
        elif sensor_id.numel() != n_nodes:
            raise ValueError(f"sensor_id must have 1 or {n_nodes} entries, got {sensor_id.numel()}")
        return sensor_id.clamp(min=0, max=self.unknown_sensor_id)

    def _apply_alpha_cap(self, alpha: torch.Tensor, data: Data) -> torch.Tensor:
        """Apply optional per-sensor upper bounds to the residual context gate."""
        if not self.sensor_gate_enabled or self.alpha_max_by_sensor is None:
            return alpha
        n_nodes = alpha.size(0)
        sensor_id = self._sensor_ids_for_nodes(data, n_nodes, alpha.device)
        alpha_max = self.alpha_max_by_sensor.to(device=alpha.device, dtype=alpha.dtype)[sensor_id]
        return alpha * alpha_max.unsqueeze(-1)

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

        if self.phase_edge_bias is not None:
            x_phase_edge = getattr(data, 'x_phase', None)
            if x_phase_edge is None:
                raise RuntimeError("phase_edge enabled but graph has no x_phase")
            if torch.is_autocast_enabled():
                x_phase_edge = x_phase_edge.to(torch.get_autocast_gpu_dtype())
            phase_edge_logit = self.phase_edge_bias(x_phase_edge, edge_index, edge_type)
            edge_logit = phase_edge_logit if edge_logit is None else edge_logit + phase_edge_logit
            edge_value_gate = None
            if self.phase_edge_value_scale > 0:
                edge_value_gate = 1.0 + self.phase_edge_value_scale * torch.tanh(phase_edge_logit)
            self._last_phase_edge_logit = phase_edge_logit.detach()
            self._last_phase_edge_conf = torch.sigmoid(phase_edge_logit)
        else:
            self._last_phase_edge_logit = None
            self._last_phase_edge_conf = None
            edge_value_gate = None

        if self.phase_alignment_bias is not None:
            x_phase_align = getattr(data, 'x_phase', None)
            if x_phase_align is None:
                raise RuntimeError("phase_alignment_edge enabled but graph has no x_phase")
            align_logit, align_feats = self.phase_alignment_bias(
                x_phase_align.float(), edge_index, edge_type
            )
            edge_logit = align_logit if edge_logit is None else edge_logit + align_logit.to(edge_logit.dtype)
            if self.phase_alignment_value_scale > 0:
                align_value_gate = 1.0 + self.phase_alignment_value_scale * torch.tanh(align_logit)
                edge_value_gate = (
                    align_value_gate
                    if edge_value_gate is None
                    else edge_value_gate * align_value_gate.to(edge_value_gate.dtype)
            )
            self._last_phase_alignment_logit = align_logit.detach()
            self._last_phase_alignment_conf = torch.sigmoid(align_logit)
            self._last_phase_alignment_features = align_feats.detach()
        else:
            self._last_phase_alignment_logit = None
            self._last_phase_alignment_conf = None
            self._last_phase_alignment_features = None

        # Closed-form coherence bias: deterministic, yaw-invariant, stacks additively.
        if self.phase_coherence_bias is not None:
            x_phase_coh = getattr(data, 'x_phase', None)
            if x_phase_coh is None:
                raise RuntimeError("phase_coherence enabled but graph has no x_phase")
            # Coherence path runs in FP32 (FFT precision-sensitive).
            coh_logit = self.phase_coherence_bias(
                x_phase_coh.float(), edge_index, edge_type
            )
            if edge_logit is None:
                edge_logit = coh_logit
            else:
                edge_logit = edge_logit + coh_logit.to(edge_logit.dtype)
            self._last_phase_coherence_logit = coh_logit.detach()
        else:
            self._last_phase_coherence_logit = None
        self._last_edge_logit = edge_logit.detach() if edge_logit is not None else None

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
                    def _conv_with_logit(h_in, ei, ea, el, ev):
                        return conv(
                            h_in, ei, edge_attr=ea,
                            edge_logit=el, edge_value_gate=ev
                        )
                    h = grad_checkpoint(
                        _conv_with_logit, h, edge_index, edge_embed, edge_logit, edge_value_gate,
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
                    h = conv(
                        h, edge_index, edge_attr=edge_embed,
                        edge_logit=edge_logit,
                        edge_value_gate=edge_value_gate,
                    )
                else:
                    h = conv(h, edge_index, edge_value_gate=edge_value_gate)

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
            alpha = torch.sigmoid(self.gate(self._gate_input(h, data)))  # (n_nodes, 1)
            alpha = self._apply_alpha_cap(alpha, data)
            ctx_norm = alpha * ctx_norm
            self._last_alpha_for_loss = alpha
            self._last_alpha = alpha.detach()

        parts = [raw_norm, ctx_norm]
        if self.phase_projector is not None:
            x_phase = getattr(data, 'x_phase', None)
            if x_phase is None:
                raise RuntimeError("phase_token enabled but graph has no x_phase")
            if torch.is_autocast_enabled():
                x_phase = x_phase.to(torch.get_autocast_gpu_dtype())
            phase_norm = F.normalize(self.phase_projector(x_phase), p=2, dim=-1)
            phase_alpha = torch.sigmoid(self.phase_logit)
            phase_norm = phase_alpha * phase_norm
            self._last_phase_alpha = phase_alpha.detach()
            parts.append(phase_norm)

        return torch.cat(parts, dim=-1)

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

        edge_logit = None
        edge_value_gate = None
        if self.phase_edge_bias is not None:
            x_phase_edge = getattr(data, 'x_phase', None)
            if x_phase_edge is None:
                raise RuntimeError("phase_edge enabled but graph has no x_phase")
            phase_edge_logit = self.phase_edge_bias(x_phase_edge, edge_index, edge_type)
            edge_logit = phase_edge_logit
            if self.phase_edge_value_scale > 0:
                edge_value_gate = 1.0 + self.phase_edge_value_scale * torch.tanh(phase_edge_logit)
            self._last_phase_edge_logit = phase_edge_logit.detach()
            self._last_phase_edge_conf = torch.sigmoid(phase_edge_logit)

        if self.phase_alignment_bias is not None:
            x_phase_align = getattr(data, 'x_phase', None)
            if x_phase_align is None:
                raise RuntimeError("phase_alignment_edge enabled but graph has no x_phase")
            align_logit, align_feats = self.phase_alignment_bias(
                x_phase_align.float(), edge_index, edge_type
            )
            edge_logit = align_logit if edge_logit is None else edge_logit + align_logit.to(edge_logit.dtype)
            if self.phase_alignment_value_scale > 0:
                align_value_gate = 1.0 + self.phase_alignment_value_scale * torch.tanh(align_logit)
                edge_value_gate = (
                    align_value_gate
                    if edge_value_gate is None
                    else edge_value_gate * align_value_gate.to(edge_value_gate.dtype)
            )
            self._last_phase_alignment_logit = align_logit.detach()
            self._last_phase_alignment_conf = torch.sigmoid(align_logit)
            self._last_phase_alignment_features = align_feats.detach()
        else:
            self._last_phase_alignment_logit = None
            self._last_phase_alignment_conf = None
            self._last_phase_alignment_features = None

        if self.phase_coherence_bias is not None:
            x_phase_coh = getattr(data, 'x_phase', None)
            if x_phase_coh is None:
                raise RuntimeError("phase_coherence enabled but graph has no x_phase")
            coh_logit = self.phase_coherence_bias(
                x_phase_coh.float(), edge_index, edge_type
            )
            edge_logit = coh_logit if edge_logit is None else edge_logit + coh_logit.to(edge_logit.dtype)
            self._last_phase_coherence_logit = coh_logit.detach()
        else:
            self._last_phase_coherence_logit = None
        self._last_edge_logit = edge_logit.detach() if edge_logit is not None else None

        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.relu(h)

        attention_weights = []

        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h_prev = h

            if edge_embed is not None:
                h, (edge_idx, attn) = conv(
                    h, edge_index, edge_attr=edge_embed,
                    edge_logit=edge_logit,
                    edge_value_gate=edge_value_gate,
                    return_attention_weights=True
                )
            else:
                h, (edge_idx, attn) = conv(
                    h, edge_index, edge_logit=edge_logit,
                    edge_value_gate=edge_value_gate,
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
            alpha = torch.sigmoid(self.gate(self._gate_input(h, data)))
            alpha = self._apply_alpha_cap(alpha, data)
            ctx_norm = alpha * ctx_norm
            self._last_alpha_for_loss = alpha
            self._last_alpha = alpha.detach()
        parts = [raw_norm, ctx_norm]
        if self.phase_projector is not None:
            x_phase = getattr(data, 'x_phase', None)
            if x_phase is None:
                raise RuntimeError("phase_token enabled but graph has no x_phase")
            phase_norm = F.normalize(self.phase_projector(x_phase), p=2, dim=-1)
            phase_alpha = torch.sigmoid(self.phase_logit)
            phase_norm = phase_alpha * phase_norm
            self._last_phase_alpha = phase_alpha.detach()
            parts.append(phase_norm)
        return torch.cat(parts, dim=-1), attention_weights

    def get_embedding_dim(self) -> int:
        """Get output embedding dimension (raw + context)"""
        return self.input_dim + self.context_dim + self.phase_token_dim


class PhaseStreamGNN(nn.Module):
    """Yaw-invariant phase context encoder (sibling of SpectralGNN's mag stream).

    Two ``DiffAttnConv`` layers operating on yaw-invariant per-node phase
    features (``log(1+|z|^2)`` plus optional bispectrum coefficients), output
    a ``context_dim`` per-node vector. The conv operator is reused unchanged;
    yaw-invariance comes from the inputs, not from a new attention primitive.
    """

    def __init__(
        self,
        phase_input_dim: int,
        hidden_dim: int = 128,
        context_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        edge_encoder_config: Optional[dict] = None,
        norm_type: str = "layer_norm",
    ) -> None:
        super().__init__()
        self.phase_input_dim = int(phase_input_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)

        self.input_proj = nn.Linear(self.phase_input_dim, hidden_dim)
        self.input_norm = SpectralGNN._make_norm(hidden_dim, norm_type)

        if edge_encoder_config is not None:
            self.edge_encoder = EdgeEncoder(**edge_encoder_config)
            edge_dim = edge_encoder_config.get("d_edge", 32)
        else:
            self.edge_encoder = None
            edge_dim = None

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                DiffAttnConv(
                    channels=hidden_dim,
                    heads=n_heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
            )
            self.norms.append(SpectralGNN._make_norm(hidden_dim, norm_type))

        self.output_proj = nn.Linear(hidden_dim, context_dim)
        self.dropout = float(dropout)
        self.n_layers = int(n_layers)

    def forward(
        self,
        phase_x_inv: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
        edge_logit: Optional[torch.Tensor] = None,
        edge_value_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_proj(phase_x_inv)
        h = self.input_norm(h)
        h = F.relu(h)

        if self.edge_encoder is not None and edge_attr is not None and edge_type is not None:
            edge_embed = self.edge_encoder(edge_attr, edge_type)
        else:
            edge_embed = None

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = h
            h = conv(
                h,
                edge_index,
                edge_attr=edge_embed,
                edge_logit=edge_logit,
                edge_value_gate=edge_value_gate,
            )
            h = norm(h)
            if i < self.n_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_prev

        return self.output_proj(h)


class DualStreamSpectralGNN(nn.Module):
    """Magnitude + phase dual-stream GNN with gated context fusion.

    The magnitude stream is a regular :class:`SpectralGNN` on ``data.x`` (the
    yaw-invariant 288D magnitude key); the phase stream is a
    :class:`PhaseStreamGNN` on yaw-invariant features derived from
    ``data.x_phase`` (``log(1+|z|^2)`` + optional bispectrum). Their context
    vectors are mixed by a per-node gate ``α``::

        ctx_fused = (1 - α) · ctx_mag + α · ctx_phase

    so the model defaults to magnitude behaviour at init (``α ≈ initial_alpha``,
    typically 0.1) and grows phase contribution when training finds it useful.

    Output layout matches :meth:`SpectralGNN.forward`::

        cat( raw_mag, ctx_fused, [optional phase_token tail from mag_gnn] )

    Backward compat: pass ``data`` exactly as the magnitude stream expects;
    the phase stream additionally consumes ``data.x_phase``.
    """

    def __init__(
        self,
        mag_gnn: "SpectralGNN",
        phase_gnn: "PhaseStreamGNN",
        n_rows: int,
        n_freqs: int,
        use_bispectrum: bool = True,
        fuse_initial_alpha: float = 0.1,
        fuse_per_node: bool = True,
        fuse_hidden_dim: int = 64,
        scale_phase_by_mag_gate: bool = True,
    ) -> None:
        super().__init__()
        if mag_gnn.context_dim != phase_gnn.context_dim:
            raise ValueError(
                f"context_dim mismatch: mag={mag_gnn.context_dim}, "
                f"phase={phase_gnn.context_dim}; fusion expects equal dims."
            )
        self.mag_gnn = mag_gnn
        self.phase_gnn = phase_gnn
        self.n_rows = int(n_rows)
        self.n_freqs = int(n_freqs)
        self.use_bispectrum = bool(use_bispectrum)
        self.scale_phase_by_mag_gate = bool(scale_phase_by_mag_gate)

        ctx_dim = mag_gnn.context_dim
        init_alpha = float(min(max(fuse_initial_alpha, 1e-4), 1.0 - 1e-4))
        init_logit = math.log(init_alpha / (1.0 - init_alpha))

        self.fuse_per_node = bool(fuse_per_node)
        if self.fuse_per_node:
            self.fuse_gate = nn.Sequential(
                nn.Linear(2 * ctx_dim, fuse_hidden_dim),
                nn.ReLU(),
                nn.Linear(fuse_hidden_dim, 1),
            )
            nn.init.zeros_(self.fuse_gate[-1].weight)
            nn.init.constant_(self.fuse_gate[-1].bias, init_logit)
            self.fuse_logit = None
        else:
            self.fuse_gate = None
            self.fuse_logit = nn.Parameter(torch.tensor(init_logit))

        self._last_fuse_alpha: Optional[torch.Tensor] = None

    def get_embedding_dim(self) -> int:
        return self.mag_gnn.get_embedding_dim()

    # Property forwards keep call-sites that introspect `base_gnn.input_dim` /
    # `base_gnn.context_dim` working unchanged (see train_multi_dataset.py).
    @property
    def input_dim(self) -> int:
        return self.mag_gnn.input_dim

    @property
    def context_dim(self) -> int:
        return self.mag_gnn.context_dim

    @property
    def phase_token_dim(self) -> int:
        return self.mag_gnn.phase_token_dim

    def _phase_invariants(self, x_phase: torch.Tensor) -> torch.Tensor:
        # Imported lazily to avoid circular imports at module load time.
        from gnn.phase_diff_conv import phase_invariant_features

        return phase_invariant_features(
            x_phase.float(),
            self.n_rows,
            self.n_freqs,
            use_bispectrum=self.use_bispectrum,
        )

    def forward(self, data: Data) -> torch.Tensor:
        x_phase = getattr(data, "x_phase", None)
        if x_phase is None:
            raise RuntimeError("DualStreamSpectralGNN requires data.x_phase")

        # ---- Magnitude stream (existing pipeline, unchanged) ----
        mag_out = self.mag_gnn(data)
        phase_edge_logit = getattr(self.mag_gnn, "_last_edge_logit", None)
        mag_dim = self.mag_gnn.input_dim
        ctx_dim = self.mag_gnn.context_dim
        mag_raw = mag_out[..., :mag_dim]
        mag_ctx = mag_out[..., mag_dim : mag_dim + ctx_dim]
        tail = mag_out[..., mag_dim + ctx_dim :]                   # phase token, if any

        # ---- Phase stream ----
        phase_inv = self._phase_invariants(x_phase)
        if torch.is_autocast_enabled():
            phase_inv = phase_inv.to(torch.get_autocast_gpu_dtype())
        ctx_phase_raw = self.phase_gnn(
            phase_inv,
            edge_index=data.edge_index,
            edge_attr=getattr(data, "edge_attr", None),
            edge_type=getattr(data, "edge_type", None),
            edge_logit=phase_edge_logit,
        )
        ctx_phase = F.normalize(ctx_phase_raw, p=2, dim=-1)
        if self.scale_phase_by_mag_gate:
            mag_alpha = getattr(self.mag_gnn, "_last_alpha", None)
            if mag_alpha is not None:
                ctx_phase = mag_alpha.to(ctx_phase.dtype) * ctx_phase

        # ---- Fuse ----
        if self.fuse_gate is not None:
            alpha = torch.sigmoid(
                self.fuse_gate(torch.cat([mag_ctx, ctx_phase], dim=-1))
            )                                                       # (N, 1)
        else:
            alpha = torch.sigmoid(self.fuse_logit)                  # ()
        self._last_fuse_alpha = alpha.detach()
        ctx_fused = (1.0 - alpha) * mag_ctx + alpha * ctx_phase

        return torch.cat([mag_raw, ctx_fused, tail], dim=-1)


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
    phase_token_config: Optional[dict] = None,
    phase_edge_config: Optional[dict] = None,
    phase_alignment_config: Optional[dict] = None,
    phase_coherence_config: Optional[dict] = None,
    dual_stream_config: Optional[dict] = None,
    sensor_gate_config: Optional[dict] = None,
    diffattn_value_source: str = "diff",
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
        phase_token_config=phase_token_config,
        phase_edge_config=phase_edge_config,
        phase_alignment_config=phase_alignment_config,
        phase_coherence_config=phase_coherence_config,
        sensor_gate_config=sensor_gate_config,
        diffattn_value_source=diffattn_value_source,
    )

    # Optional dual-stream wrapper: magnitude (existing) + phase (yaw-invariant)
    # streams with gated context fusion. The magnitude SpectralGNN above is
    # reused as-is; only ctx_mag is fused with ctx_phase.
    if dual_stream_config is not None and dual_stream_config.get("enabled", False):
        from gnn.phase_diff_conv import feature_dim as _phase_feature_dim

        ds = dual_stream_config
        n_rows = int(ds["n_rows"])
        n_freqs = int(ds["n_freqs"])
        use_bispectrum = bool(ds.get("use_bispectrum", True))
        phase_input_dim = _phase_feature_dim(n_rows, n_freqs, use_bispectrum)
        phase_gnn = PhaseStreamGNN(
            phase_input_dim=phase_input_dim,
            hidden_dim=int(ds.get("hidden_dim", 128)),
            context_dim=int(ds.get("context_dim", context_dim)),
            n_layers=int(ds.get("n_layers", 2)),
            n_heads=int(ds.get("n_heads", n_heads)),
            dropout=float(ds.get("dropout", dropout)),
            edge_encoder_config=ds.get("edge_encoder_config"),
            norm_type=str(ds.get("norm_type", "layer_norm")),
        )
        base_gnn = DualStreamSpectralGNN(
            mag_gnn=base_gnn,
            phase_gnn=phase_gnn,
            n_rows=n_rows,
            n_freqs=n_freqs,
            use_bispectrum=use_bispectrum,
            fuse_initial_alpha=float(ds.get("fuse_initial_alpha", 0.1)),
            fuse_per_node=bool(ds.get("fuse_per_node", True)),
            fuse_hidden_dim=int(ds.get("fuse_hidden_dim", 64)),
            scale_phase_by_mag_gate=bool(ds.get("scale_phase_by_mag_gate", True)),
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
