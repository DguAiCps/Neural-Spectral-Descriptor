"""Small learned rerankers for compact phase-sketch candidates.

The model here is intentionally narrow: it does not change the NSD encoder or
GNN descriptor. It consumes the candidate-level cyclic phase-correlation curve
computed from stored compact phase coefficients and learns a pairwise score for
top-N reranking.
"""

from __future__ import annotations

import torch
from torch import nn


class PhaseCorrelationReranker(nn.Module):
    """MLP reranker over shift-correlation curves and embedding similarity.

    Args:
        n_shifts: number of cyclic shifts in the correlation curve.
        hidden_dim: hidden width of the scoring MLP.
        dropout: dropout used inside the MLP.

    Input shapes:
        shift_corr: ``(B, N, n_shifts)`` normalized correlation per candidate.
        emb_sim: ``(B, N)`` cosine similarity from the frozen retrieval key.

    Returns:
        ``(B, N)`` logits. Higher is better.
    """

    def __init__(
        self,
        n_shifts: int = 60,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        base_phase_weight: float = 10.0,
        base_embedding_weight: float = 1.0,
        adaptive_residual_gate: bool = False,
        gate_hidden_dim: int = 16,
        gate_initial_alpha: float = 0.25,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_shifts = int(n_shifts)
        self.adaptive_residual_gate = bool(adaptive_residual_gate)
        self.residual_scale = float(residual_scale)
        self.base_phase_weight = nn.Parameter(torch.tensor(float(base_phase_weight)))
        self.base_embedding_weight = nn.Parameter(torch.tensor(float(base_embedding_weight)))
        stat_dim = 5  # emb_sim, max, mean, std, margin
        self.net = nn.Sequential(
            nn.Linear(self.n_shifts + stat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Start from the strong closed-form phase reranker and let training learn
        # only a correction. Random logits alone are unstable on KITTI 00/05.
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        gate_alpha = min(max(float(gate_initial_alpha), 1e-4), 1.0 - 1e-4)
        gate_bias = torch.logit(torch.tensor(gate_alpha))
        self.gate_net = nn.Sequential(
            nn.Linear(3, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        gate_last = self.gate_net[-1]
        if isinstance(gate_last, nn.Linear):
            nn.init.zeros_(gate_last.weight)
            nn.init.constant_(gate_last.bias, float(gate_bias))

    def forward(
        self,
        shift_corr: torch.Tensor,
        emb_sim: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if shift_corr.ndim != 3:
            raise ValueError(f"shift_corr must be (B,N,S), got {tuple(shift_corr.shape)}")
        if shift_corr.shape[-1] != self.n_shifts:
            raise ValueError(
                f"Expected {self.n_shifts} shifts, got {shift_corr.shape[-1]}"
            )
        if emb_sim.shape != shift_corr.shape[:2]:
            raise ValueError(
                f"emb_sim shape {tuple(emb_sim.shape)} incompatible with "
                f"shift_corr {tuple(shift_corr.shape)}"
            )

        top2 = torch.topk(shift_corr, k=min(2, self.n_shifts), dim=-1).values
        peak = top2[..., 0]
        if top2.shape[-1] > 1:
            margin = top2[..., 0] - top2[..., 1]
        else:
            margin = torch.zeros_like(peak)
        stats = torch.stack(
            [
                emb_sim,
                peak,
                shift_corr.mean(dim=-1),
                shift_corr.std(dim=-1),
                margin,
            ],
            dim=-1,
        )
        feat = torch.cat([shift_corr, stats], dim=-1)
        residual = self.net(feat).squeeze(-1)
        base_score = (
            self.base_phase_weight * peak
            + self.base_embedding_weight * emb_sim
        )
        if self.adaptive_residual_gate:
            masked_base = base_score
            if valid_mask is not None:
                masked_base = masked_base.masked_fill(~valid_mask, -1e9)
            top2_base = torch.topk(masked_base, k=min(2, masked_base.shape[1]), dim=1).values
            top1 = top2_base[:, 0]
            if top2_base.shape[1] > 1:
                query_margin = top2_base[:, 0] - top2_base[:, 1]
                top2 = top2_base[:, 1]
            else:
                query_margin = torch.zeros_like(top1)
                top2 = torch.zeros_like(top1)
            gate_feat = torch.stack([top1, top2, query_margin], dim=-1)
            gate = torch.sigmoid(self.gate_net(gate_feat)).squeeze(-1).unsqueeze(-1)
            residual = gate * residual
        logits = base_score + self.residual_scale * residual
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -1e9)
        return logits
