"""
GNN Trainer - Algorithm 4

Implements training loop for GNN with InfoNCE loss:
- 50 epochs on KITTI sequences [0-8]
- Validation on sequence [9]
- Hard negative mining
- Adam optimizer, lr=5e-4
- InfoNCE loss with temperature=0.07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import json
import os
import logging

from gnn.model import SpectralGNN, create_spectral_gnn
from gnn.triplet_miner import TripletMiner, BatchTripletMiner
from keyframe.graph_manager import TemporalGraphManager


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for ranking-aware metric learning.

    For each anchor_i, positive_i should rank #1 among:
    - all other positives_j (in-batch cross-negatives, B-1)
    - hard-mined negative_i (1)
    Total B negatives per anchor.

    loss = CrossEntropy(sim_matrix / tau, labels=arange(B))
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchors: (B, D)
            positives: (B, D)
            negatives: (B, D)

        Returns:
            Scalar loss
        """
        anchors = F.normalize(anchors, dim=1)
        positives = F.normalize(positives, dim=1)
        negatives = F.normalize(negatives, dim=1)

        # In-batch similarity: (B, B) — anchor_i vs all positives_j
        # Diagonal (i,i) = positive pair
        cross_sim = (anchors @ positives.T) / self.temperature

        # Hard-mined negative: (B, 1)
        hard_neg_sim = (anchors * negatives).sum(dim=1, keepdim=True) / self.temperature

        # Logits: (B, B+1) = [cross_sim | hard_neg]
        logits = torch.cat([cross_sim, hard_neg_sim], dim=1)

        # Label: anchor_i's positive is at index i in cross_sim
        labels = torch.arange(anchors.shape[0], device=anchors.device)

        return F.cross_entropy(logits, labels)


class SmoothAPLoss(nn.Module):
    """Smooth Average Precision loss (Brown et al., ECCV 2020).

    For each anchor with K positives and M negatives, compute differentiable AP:
        rank_pos(k) = 1 + Σ_{k' != k} σ((s_{k'} - s_k) / τ)
        rank_all(k) = rank_pos(k) + Σ_m σ((s_m - s_k) / τ)
        AP = (1/K_valid) Σ_k rank_pos(k) / rank_all(k)
        Loss = 1 - mean_anchors(AP)

    Where σ is sigmoid (smooth indicator). Vectorized — all anchors in batch
    processed in parallel.

    Padding handled via masks: padded entries contribute 0 to ranks and are
    excluded from the AP average.
    """

    def __init__(self, tau: float = 0.01):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        anchor_emb: torch.Tensor,    # (B, D)
        positive_emb: torch.Tensor,  # (B, K, D)
        negative_emb: torch.Tensor,  # (B, M, D)
        pos_mask: torch.Tensor,      # (B, K) bool, True = real positive
        neg_mask: torch.Tensor,      # (B, M) bool, True = real negative
    ) -> torch.Tensor:
        # L2-normalize for cosine similarity (matches retrieval evaluation)
        a = F.normalize(anchor_emb, p=2, dim=-1)
        p = F.normalize(positive_emb, p=2, dim=-1)
        n = F.normalize(negative_emb, p=2, dim=-1)

        # Cosine similarities
        s_pos = torch.einsum('bd,bkd->bk', a, p)  # (B, K)
        s_neg = torch.einsum('bd,bmd->bm', a, n)  # (B, M)

        # Pairwise differences for sigmoid indicators
        # diff_pp[b, k1, k2] = s_pos[b, k2] - s_pos[b, k1]   (col=k2, row=k1)
        # diff_pn[b, k, m]   = s_neg[b, m]  - s_pos[b, k]
        diff_pp = s_pos.unsqueeze(2) - s_pos.unsqueeze(1)
        diff_pn = s_neg.unsqueeze(1) - s_pos.unsqueeze(2)

        sig_pp = torch.sigmoid(diff_pp / self.tau)  # (B, K, K)
        sig_pn = torch.sigmoid(diff_pn / self.tau)  # (B, K, M)

        # Mask self (k == k') in positive-positive ranks
        B, K = s_pos.shape
        eye_mask = torch.eye(K, dtype=torch.bool, device=s_pos.device).unsqueeze(0)
        sig_pp = sig_pp.masked_fill(eye_mask, 0.0)

        # Mask padded entries from contributing to other entries' ranks
        # pos_mask broadcast over k1 (rows): only valid k2 (cols) count toward rank_pos[k1]
        sig_pp = sig_pp * pos_mask.unsqueeze(1).float()
        # neg_mask broadcast over k (rows): only valid m (cols) count toward rank_all[k]
        sig_pn = sig_pn * neg_mask.unsqueeze(1).float()

        # Ranks
        rank_pos = 1.0 + sig_pp.sum(dim=2)         # (B, K) — rank within positives
        rank_all = rank_pos + sig_pn.sum(dim=2)    # (B, K) — rank within positives + negatives

        # AP per (anchor, positive)
        ap_per_pos = rank_pos / rank_all           # (B, K)

        # Average AP per anchor over valid positives only
        ap_per_pos = ap_per_pos * pos_mask.float()
        n_valid_pos = pos_mask.sum(dim=1).clamp(min=1).float()
        ap_per_anchor = ap_per_pos.sum(dim=1) / n_valid_pos  # (B,)

        return 1.0 - ap_per_anchor.mean()


class GNNTrainer:
    """
    Trainer for Spectral GNN with InfoNCE loss
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10,
        use_multi_gpu: bool = True,
        patience: int = 10,
        use_amp: bool = True,  # Mixed precision training
        policy_lr_scale: float = 1.0,
        policy_warmup_epochs: int = 0,
        loss_type: str = 'infonce',
        smoothap_tau: float = 0.01,
        smoothap_n_pos: int = 8,
        smoothap_n_neg: int = 32,
        smoothap_batch_anchors: int = 64,
        edge_aux_lambda: float = 0.0,
        phase_edge_aux_lambda: float = 0.0,
        phase_edge_aux_balance: bool = False,
        phase_edge_aux_focal_gamma: float = 0.0,
        phase_alignment_aux_lambda: float = 0.0,
        phase_alignment_aux_balance: bool = False,
        phase_alignment_aux_focal_gamma: float = 0.0,
        context_aux_lambda: float = 0.0,
        phase_token_aux_lambda: float = 0.0,
        checkpoint_metric: str = 'average_recall@1',
        recall_k_values: Optional[List[int]] = None,
    ):
        """
        Initialize trainer

        Args:
            model: GNN model
            device: Device for training
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            temperature: InfoNCE temperature (lower = sharper ranking)
            checkpoint_dir: Directory for checkpoints
            log_interval: Logging interval (iterations)
            use_multi_gpu: Use multiple GPUs if available
            patience: Early stopping patience (epochs without improvement)
            policy_lr_scale: Learning rate multiplier for spectral policy params
            policy_warmup_epochs: Epochs to keep policy frozen before training
        """
        self.model = model.to(device)
        self.device = device
        self.policy_warmup_epochs = policy_warmup_epochs

        # Multi-GPU support
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        else:
            print(f"Using single GPU for training")

        self.patience = patience
        self.epochs_without_improvement = 0

        # Separate parameter groups for spectral policy (if present)
        base_model = model.gnn if hasattr(model, 'gnn') else model
        has_policy = hasattr(base_model, 'spectral_policy') and base_model.spectral_policy is not None
        if has_policy and policy_lr_scale != 1.0:
            policy_params = list(base_model.spectral_policy.parameters())
            policy_param_ids = {id(p) for p in policy_params}
            other_params = [p for p in model.parameters() if id(p) not in policy_param_ids]
            param_groups = [
                {'params': other_params, 'lr': learning_rate},
                {'params': policy_params, 'lr': learning_rate * policy_lr_scale},
            ]
            self.optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
            logging.info(f"  Optimizer: policy lr_scale={policy_lr_scale}, warmup={policy_warmup_epochs} epochs")
        else:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

        # Mixed precision training (50% VRAM reduction)
        self.use_amp = use_amp and device == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            logging.info("Mixed precision training enabled (FP16)")
        else:
            self.scaler = None

        self.loss_type = loss_type
        self.smoothap_tau = smoothap_tau
        self.smoothap_n_pos = smoothap_n_pos
        self.smoothap_n_neg = smoothap_n_neg
        self.smoothap_batch_anchors = smoothap_batch_anchors

        if loss_type == 'smoothap':
            self.criterion = SmoothAPLoss(tau=smoothap_tau)
        else:
            self.criterion = InfoNCELoss(temperature=temperature)

        # Edge confidence gate auxiliary loss weight (BCE on pose-GT labels).
        # 0.0 disables aux loss; gate still runs forward but is unsupervised.
        self.edge_aux_lambda = edge_aux_lambda
        self.phase_edge_aux_lambda = phase_edge_aux_lambda
        self.phase_edge_aux_balance = phase_edge_aux_balance
        self.phase_edge_aux_focal_gamma = phase_edge_aux_focal_gamma
        self.phase_alignment_aux_lambda = phase_alignment_aux_lambda
        self.phase_alignment_aux_balance = phase_alignment_aux_balance
        self.phase_alignment_aux_focal_gamma = phase_alignment_aux_focal_gamma
        self.context_aux_lambda = context_aux_lambda
        self.phase_token_aux_lambda = phase_token_aux_lambda
        self.checkpoint_metric = checkpoint_metric
        self.recall_k_values = sorted(set(int(k) for k in (recall_k_values or [1, 5, 10])))
        if 1 not in self.recall_k_values:
            self.recall_k_values.insert(0, 1)

        # SmoothAP mining cache (anchors, pos_pool, neg_pool)
        self._smoothap_anchors = None
        self._smoothap_pos_pool = None
        self._smoothap_neg_pool = None

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = log_interval

        # LR scheduler (created in train() once n_epochs is known)
        self.scheduler = None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0

        # Triplet mining cache (Opt 1: mine_every_n_epochs)
        self._cached_triplets = None
        self._last_mine_epoch = -1

        # Two-pass refinement state (populated by _refine_similarity_edges,
        # consumed by validate() to apply matching edge refinement to val graphs)
        self._refined_similarity_dist = None
        self._refined_std_stats = None
        self._refined_edge_kwargs = None
        self._refined_space = 'ctx'

        # History
        self.train_losses = []
        self.val_metrics = []

    @staticmethod
    def _sensor_name_to_id() -> Dict[str, int]:
        return {
            'kitti': 0,
            'nclt': 1,
            'helipr': 2,
            'mulran': 3,
        }

    @staticmethod
    def _sensor_label_from_dataset_name(dataset_name: str) -> str:
        return dataset_name.split('_', 1)[0].lower()

    def _sensor_scalar_table_from_config(
        self,
        value,
        num_sensors: int,
        default: float,
    ) -> torch.Tensor:
        table = torch.full((num_sensors + 1,), float(default), dtype=torch.float32, device=self.device)
        if isinstance(value, (int, float)):
            table.fill_(float(value))
            return table
        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value[:num_sensors]):
                table[idx] = float(item)
            return table
        if isinstance(value, dict):
            fallback = float(value.get('default', default))
            table.fill_(fallback)
            name_to_id = self._sensor_name_to_id()
            for key, item in value.items():
                if key == 'default':
                    continue
                idx = name_to_id.get(str(key).lower(), None)
                if idx is None:
                    try:
                        idx = int(key)
                    except (TypeError, ValueError):
                        continue
                if 0 <= idx < num_sensors:
                    table[idx] = float(item)
        return table

    def _gate_alpha_regularization(self, graph: Data) -> Tuple[Optional[torch.Tensor], float]:
        """Regularize residual gate alpha toward sensor-specific targets."""
        base_model = self.model.gnn if hasattr(self.model, 'gnn') else self.model
        alpha = getattr(base_model, '_last_alpha_for_loss', None)
        cfg = getattr(base_model, 'sensor_gate_config', {}) or {}
        if alpha is None or not cfg.get('enabled', False):
            return None, 0.0

        start_weight = float(cfg.get('regularization_weight', 0.0))
        if start_weight <= 0:
            return None, 0.0
        final_weight = float(cfg.get('regularization_final_weight', start_weight))
        anneal_epochs = int(cfg.get('regularization_anneal_epochs', 0))
        if anneal_epochs > 0:
            t = min(max(self.epoch, 0) / float(max(anneal_epochs - 1, 1)), 1.0)
            weight = start_weight + (final_weight - start_weight) * t
        else:
            weight = start_weight

        num_sensors = int(cfg.get('num_sensors', 4))
        target_cfg = cfg.get('target_alpha', cfg.get('target', 0.0625))
        target_table = self._sensor_scalar_table_from_config(
            target_cfg, num_sensors=num_sensors, default=0.0625
        )

        sensor_id = getattr(graph, 'sensor_id', None)
        if sensor_id is None:
            sensor_id = torch.full(
                (alpha.size(0),), num_sensors, dtype=torch.long, device=alpha.device
            )
        else:
            sensor_id = sensor_id.to(device=alpha.device, dtype=torch.long).reshape(-1)
            if sensor_id.numel() == 1:
                sensor_id = sensor_id.expand(alpha.size(0))
        sensor_id = sensor_id.clamp(min=0, max=num_sensors)
        target = target_table.to(device=alpha.device, dtype=alpha.dtype)[sensor_id].unsqueeze(-1)
        reg = F.mse_loss(alpha.float(), target.float())
        return reg, weight

    def _checkpoint_metric_value(self, all_metrics: Dict[str, Dict[str, float]]) -> Tuple[str, float]:
        """Select validation metric for early stopping/checkpointing."""
        key = (self.checkpoint_metric or 'average_recall@1').lower()
        aliases = {
            'avg': ('_average', 'recall@1'),
            'average': ('_average', 'recall@1'),
            'average_recall@1': ('_average', 'recall@1'),
            'query_weighted_recall@1': ('_average', 'recall@1'),
            'sequence_macro': ('_sequence_macro', 'recall@1'),
            'sequence_macro_recall@1': ('_sequence_macro', 'recall@1'),
            'sensor_macro': ('_sensor_macro', 'recall@1'),
            'sensor_macro_recall@1': ('_sensor_macro', 'recall@1'),
        }
        if key in aliases:
            section, metric = aliases[key]
            return f"{section}.{metric}", float(all_metrics.get(section, {}).get(metric, 0.0))
        if '.' in key:
            section, metric = key.split('.', 1)
            if not section.startswith('_'):
                section = f"_{section}"
            return f"{section}.{metric}", float(all_metrics.get(section, {}).get(metric, 0.0))
        return '_average.recall@1', float(all_metrics.get('_average', {}).get('recall@1', 0.0))

    def _balance_triplets(
        self,
        triplets: np.ndarray,
        graph: Data,
        sequence_ids: Optional[np.ndarray],
        sampling_cfg: Optional[Dict],
    ) -> np.ndarray:
        """Downsample mined triplets so no sensor/sequence dominates the loss."""
        if sampling_cfg is None or len(triplets) == 0:
            return triplets
        rng = np.random.default_rng(seed=self.epoch)

        def _balanced_by_labels(labels: np.ndarray, cap_key: str, name: str) -> np.ndarray:
            unique = [u for u in np.unique(labels) if np.sum(labels == u) > 0]
            if not unique:
                return triplets
            counts = {int(u): int(np.sum(labels == u)) for u in unique}
            cap = int(sampling_cfg.get(cap_key, 0) or 0)
            target = min(counts.values())
            if cap > 0:
                target = min(target, cap)
            selected = []
            for u in unique:
                idx = np.where(labels == u)[0]
                if len(idx) > target:
                    idx = rng.choice(idx, size=target, replace=False)
                selected.append(idx)
            keep = np.concatenate(selected) if selected else np.arange(len(triplets))
            rng.shuffle(keep)
            balanced = triplets[keep]
            logging.info(
                f"  Triplet {name} balance: before={counts}, target={target:,}, "
                f"after={len(balanced):,}"
            )
            return balanced

        balanced = triplets
        if sampling_cfg.get('balanced_by_sensor', False):
            sensor_id = getattr(graph, 'sensor_id', None)
            if sensor_id is None:
                logging.warning("  Triplet sensor balance requested but graph.sensor_id is missing")
            else:
                sensor_np = sensor_id.detach().cpu().numpy().reshape(-1)
                labels = sensor_np[balanced[:, 0]]
                triplets = balanced
                balanced = _balanced_by_labels(
                    labels,
                    cap_key='max_triplets_per_sensor',
                    name='sensor',
                )
        if sampling_cfg.get('balanced_by_sequence', False) and sequence_ids is not None:
            labels = sequence_ids[balanced[:, 0]]
            triplets = balanced
            balanced = _balanced_by_labels(
                labels,
                cap_key='max_triplets_per_sequence',
                name='sequence',
            )
        return balanced

    def _diagnose_false_negatives(
        self,
        anchor_indices: np.ndarray,
        positive_indices: np.ndarray,
        negative_indices: np.ndarray,
        embeddings: torch.Tensor,
        poses: np.ndarray,
        sequence_ids: Optional[np.ndarray],
        threshold: float = 5.0,
    ) -> None:
        """Diagnose false negative ratio in InfoNCE cross_sim matrix.

        Runs on a single batch. Logs:
        - Same-seq / cross-seq pair counts
        - False negative count & ratio (same-seq, dist < threshold, off-diag)
        - Cosine similarity stats: true-pos vs false-neg vs true-neg
        - Softmax probability mass landing on false negatives (= wasted gradient)
        """
        from scipy.spatial.distance import cdist as scipy_cdist

        B = len(anchor_indices)
        if B < 2:
            return

        # Positions
        if poses.ndim == 3 and poses.shape[1:] == (4, 4):
            positions = poses[:, :3, 3]
        else:
            positions = poses[:, :3]

        anchor_pos = positions[anchor_indices]    # (B, 3)
        positive_pos = positions[positive_indices]  # (B, 3)

        # Sequence awareness
        if sequence_ids is not None:
            anchor_seq = sequence_ids[anchor_indices]
            positive_seq = sequence_ids[positive_indices]
            same_seq = anchor_seq[:, None] == positive_seq[None, :]  # (B, B)
        else:
            same_seq = np.ones((B, B), dtype=bool)

        diag = np.eye(B, dtype=bool)

        # GT pairwise distances (meaningful only for same-seq pairs)
        gt_dists = scipy_cdist(anchor_pos, positive_pos)

        # False negatives: same sequence, close, off-diagonal
        fn_mask = same_seq & (gt_dists < threshold) & ~diag
        n_fn = int(fn_mask.sum())
        n_same_off = int((same_seq & ~diag).sum())
        n_cross = int((~same_seq).sum())

        # Cosine similarities
        with torch.no_grad():
            anc = F.normalize(embeddings[anchor_indices].float(), dim=1)
            pos = F.normalize(embeddings[positive_indices].float(), dim=1)
            sim = (anc @ pos.T).cpu().numpy()  # (B, B)

        tp_sims = sim[diag]
        fn_sims = sim[fn_mask] if n_fn > 0 else np.array([0.0])
        tn_mask = ~diag & ~fn_mask
        tn_sims = sim[tn_mask]

        # Softmax probability mass on false negatives (= wasted correct signal)
        with torch.no_grad():
            logits = (anc @ pos.T) / self.criterion.temperature  # (B, B)
            # Include hard negative column for realistic softmax
            neg = F.normalize(embeddings[negative_indices].float(), dim=1)
            hard_neg_col = (anc * neg).sum(dim=1, keepdim=True) / self.criterion.temperature
            full_logits = torch.cat([logits, hard_neg_col], dim=1)  # (B, B+1)
            probs = torch.softmax(full_logits, dim=1)[:, :B].cpu().numpy()  # (B, B) cross part

        fn_prob_per_anchor = (probs * fn_mask).sum(axis=1)  # (B,)
        tp_prob_per_anchor = probs[diag]                     # (B,) = P(correct)
        n_anchors_with_fn = int((fn_prob_per_anchor > 0).sum())

        fn_ratio_of_same = n_fn / max(n_same_off, 1) * 100
        logging.info(
            f"  [FN Diag] B={B} | same_seq_offdiag={n_same_off:,} cross_seq={n_cross:,} | "
            f"false_neg={n_fn:,} ({fn_ratio_of_same:.1f}% of same-seq)"
        )
        logging.info(
            f"  [FN Diag] cos_sim: TP={tp_sims.mean():.4f}±{tp_sims.std():.4f}, "
            f"FN={fn_sims.mean():.4f}±{fn_sims.std():.4f} (n={n_fn}), "
            f"TN={tn_sims.mean():.4f}±{tn_sims.std():.4f}"
        )
        logging.info(
            f"  [FN Diag] softmax: P(TP)={tp_prob_per_anchor.mean():.4f}, "
            f"P(FN)={fn_prob_per_anchor.mean():.4f}, "
            f"anchors_with_FN={n_anchors_with_fn}/{B} "
            f"(max_P(FN)={fn_prob_per_anchor.max():.4f})"
        )

        # --- Genuine Alias (GA) Diagnostic ---
        # GA = geographically distant but descriptor-similar (structural aliasing)
        neg_dist = 10.0  # matches config negative_distance_min
        ga_sim_thresh = float(tp_sims.mean() - 2 * tp_sims.std())

        # Same-seq GA: confirmed different place (dist > neg_dist) + high sim
        ga_same = same_seq & (gt_dists > neg_dist) & ~diag & (sim > ga_sim_thresh)
        # Cross-seq: always different place → high sim = alias
        ga_cross = (~same_seq) & (sim > ga_sim_thresh)
        ga_mask = ga_same | ga_cross
        n_ga_same = int(ga_same.sum())
        n_ga_cross = int(ga_cross.sum())
        n_ga = n_ga_same + n_ga_cross
        ga_sims = sim[ga_mask] if n_ga > 0 else np.array([0.0])

        # Softmax mass on genuine aliases
        ga_prob_per_anchor = (probs * ga_mask).sum(axis=1)
        n_anchors_with_ga = int((ga_prob_per_anchor > 0).sum())

        # Top-1 confusor per anchor: what type is the most confusing negative?
        off_diag_probs = probs.copy()
        off_diag_probs[diag] = 0.0
        top1_idx = np.argmax(off_diag_probs, axis=1)  # (B,)
        top1_is_fn = np.array([fn_mask[i, top1_idx[i]] for i in range(B)])
        top1_is_ga = np.array([ga_mask[i, top1_idx[i]] for i in range(B)])
        n_top_fn = int(top1_is_fn.sum())
        n_top_ga = int(top1_is_ga.sum())
        n_top_other = B - n_top_fn - n_top_ga

        logging.info(
            f"  [GA Diag] sim_thresh=TP_mean-2σ={ga_sim_thresh:.4f} | "
            f"same_seq={n_ga_same:,} cross_seq={n_ga_cross:,} total={n_ga:,}"
        )
        logging.info(
            f"  [GA Diag] cos_sim: GA={ga_sims.mean():.4f}±{ga_sims.std():.4f} | "
            f"P(GA)={ga_prob_per_anchor.mean():.4f}, "
            f"anchors_with_GA={n_anchors_with_ga}/{B}"
        )
        logging.info(
            f"  [GA Diag] top-1 confusor: FN={n_top_fn}/{B} GA={n_top_ga}/{B} "
            f"other={n_top_other}/{B}"
        )

    def train_epoch(
        self,
        graph: Data,
        triplet_miner: TripletMiner,
        poses: np.ndarray,
        descriptors: np.ndarray,
        sequence_ids: np.ndarray = None,
        n_triplets_per_anchor: int = 1,
        mine_every_n_epochs: int = 1,
        triplet_sampling_config: Optional[Dict] = None,
    ) -> float:
        """
        Train for one epoch

        Args:
            graph: PyG graph with keyframe features
            triplet_miner: Triplet miner for hard negative mining
            poses: (n_keyframes, 4, 4) poses for mining
            descriptors: (n_keyframes, n_bins) descriptors for mining
            sequence_ids: (n_keyframes,) sequence IDs for per-sequence mining
            n_triplets_per_anchor: Number of triplets per anchor
            mine_every_n_epochs: Re-mine triplets every N epochs (cache between)

        Returns:
            Average loss for epoch
        """
        self.model.train()

        # Spectral policy warmup: freeze policy for first N epochs
        base_model = self.model.gnn if hasattr(self.model, 'gnn') else self.model
        has_policy = hasattr(base_model, 'spectral_policy') and base_model.spectral_policy is not None
        if has_policy and self.policy_warmup_epochs > 0:
            should_freeze = (self.epoch < self.policy_warmup_epochs)
            for p in base_model.spectral_policy.parameters():
                p.requires_grad = not should_freeze
            if self.epoch == self.policy_warmup_epochs:
                logging.info(f"  Spectral policy unfrozen at epoch {self.epoch + 1}")

        # Move graph to device once (needed for both mining and training)
        graph = graph.to(self.device)

        # SmoothAP loss path: completely separate mining + batching from triplet/InfoNCE
        if self.loss_type == 'smoothap':
            return self._train_epoch_smoothap(
                graph, poses, descriptors, sequence_ids, mine_every_n_epochs,
            )

        # Mine triplets using raw descriptors, NOT the GNN output.
        # Rationale: mining on the combined cat(raw, ctx) embedding creates a
        # feedback loop — a poorly-trained ctx distorts which pairs are "hard",
        # causing the model to learn ctx in the wrong direction (ctx_sim < 0 for
        # same-place pairs).  Mining on stable raw features gives a clean signal:
        # ctx is trained purely to improve upon what raw descriptors already do.
        #
        # Opt 1: Cache triplets for mine_every_n_epochs. Raw descriptors are
        # precomputed and stable — mining results are identical across epochs.
        should_mine = (
            self._cached_triplets is None
            or self.epoch % mine_every_n_epochs == 0
        )

        if should_mine:
            logging.info(f"Mining triplets for epoch {self.epoch + 1}...")
            mining_start = time.perf_counter()

            # Opt 7: Use descriptors param directly (already numpy on CPU)
            # instead of graph.x.cpu().numpy() which triggers GPU→CPU sync
            mining_embeddings = descriptors

            triplets = triplet_miner.mine_triplets(
                descriptors=mining_embeddings,
                poses=poses,
                n_triplets_per_anchor=n_triplets_per_anchor,
                sequence_ids=sequence_ids,
            )
            mining_time = time.perf_counter() - mining_start

            if len(triplets) == 0:
                logging.warning("No valid triplets mined!")
                return 0.0

            triplets_np = np.array(triplets, dtype=np.int64)
            triplets_np = self._balance_triplets(
                triplets_np, graph, sequence_ids, triplet_sampling_config
            )
            self._cached_triplets = triplets_np
            self._last_mine_epoch = self.epoch
            logging.info(
                f"Mined {len(triplets):,} triplets in {mining_time:.2f}s "
                f"({len(triplets)/mining_time:.0f} triplets/sec); "
                f"using {len(self._cached_triplets):,} after balancing"
            )
        else:
            logging.info(f"Reusing cached triplets from epoch {self._last_mine_epoch + 1} "
                         f"({len(self._cached_triplets):,} triplets, next re-mine at epoch {self._last_mine_epoch + mine_every_n_epochs + 1})")

        if self._cached_triplets is None or len(self._cached_triplets) == 0:
            logging.warning("No valid triplets available!")
            return 0.0

        # Shuffle cached triplets (copy to avoid mutating cache)
        triplets = self._cached_triplets.copy()
        np.random.shuffle(triplets)

        # Process triplets in mini-batches with gradient accumulation.
        # Single forward per accumulation window: within a window the weights
        # are frozen (optimizer.step only at window end), so all batches share
        # identical embeddings. Combined loss → single backward avoids
        # retain_graph and cuts forward passes by accumulation_steps×.
        batch_size = 1024
        accumulation_steps = 4
        n_batches = (len(triplets) + batch_size - 1) // batch_size

        epoch_losses = []
        self.optimizer.zero_grad()

        for window_start in range(0, n_batches, accumulation_steps):
            window_end = min(window_start + accumulation_steps, n_batches)
            n_in_window = window_end - window_start

            # One forward pass per accumulation window
            # Opt 5: CUDA memory tracking only on first window of each epoch
            track_vram = (window_start == 0)
            if track_vram:
                try:
                    torch.cuda.reset_peak_memory_stats(self.device)
                except (RuntimeError, ValueError):
                    pass
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    embeddings = self.model(graph)
            else:
                embeddings = self.model(graph)
            if track_vram:
                try:
                    fwd_peak = torch.cuda.max_memory_allocated(self.device) / 1e9
                    logging.info(f"  VRAM after forward: {torch.cuda.memory_allocated()/1e9:.2f} GB, "
                                 f"peak: {fwd_peak:.2f} GB")
                except (RuntimeError, ValueError):
                    pass

            # Auxiliary edge confidence loss (BCE on pose-GT labels). Computed once
            # per forward pass since edge_conf is stable for this window.
            base_model = self.model.gnn if hasattr(self.model, 'gnn') else self.model
            aux_terms = []
            if (self.edge_aux_lambda > 0
                    and getattr(base_model, '_last_edge_conf', None) is not None
                    and hasattr(graph, 'edge_pose_label')
                    and graph.edge_pose_label.numel() > 0):
                # Force FP32 for BCE (AMP makes _last_edge_conf FP16 inside autocast)
                edge_conf = base_model._last_edge_conf.squeeze(-1).float()  # (E,)
                edge_label = graph.edge_pose_label.float()                   # (E,)
                edge_type_t = graph.edge_type
                # Only supervise similarity edges (type==1); temporal are forced to 1.
                sim_mask = (edge_type_t == 1)
                if sim_mask.any():
                    edge_aux = F.binary_cross_entropy(
                        edge_conf[sim_mask].clamp(1e-7, 1.0 - 1e-7),
                        edge_label[sim_mask],
                    )
                    aux_terms.append(('edge', self.edge_aux_lambda, edge_aux, sim_mask, edge_label))

            if (self.phase_edge_aux_lambda > 0
                    and getattr(base_model, '_last_phase_edge_conf', None) is not None
                    and hasattr(graph, 'edge_pose_label')
                    and graph.edge_pose_label.numel() > 0):
                phase_edge_conf = base_model._last_phase_edge_conf.squeeze(-1).float()
                edge_label = graph.edge_pose_label.float()
                edge_type_t = graph.edge_type
                sim_mask = (edge_type_t == 1)
                if sim_mask.any():
                    phase_pred = phase_edge_conf[sim_mask].clamp(1e-7, 1.0 - 1e-7)
                    phase_target = edge_label[sim_mask]
                    phase_bce = F.binary_cross_entropy(
                        phase_pred,
                        phase_target,
                        reduction='none',
                    )
                    phase_weight = torch.ones_like(phase_bce)
                    if self.phase_edge_aux_balance:
                        pos = phase_target.sum().clamp(min=1.0)
                        neg = (1.0 - phase_target).sum().clamp(min=1.0)
                        pos_weight = (neg / pos).clamp(min=1.0, max=20.0)
                        phase_weight = torch.where(
                            phase_target > 0.5,
                            phase_weight * pos_weight,
                            phase_weight,
                        )
                    if self.phase_edge_aux_focal_gamma > 0:
                        pt = torch.where(phase_target > 0.5, phase_pred, 1.0 - phase_pred)
                        phase_weight = phase_weight * (1.0 - pt).pow(self.phase_edge_aux_focal_gamma)
                    phase_edge_aux = (phase_bce * phase_weight).sum() / phase_weight.sum().clamp(min=1.0)
                    aux_terms.append((
                        'phase-edge', self.phase_edge_aux_lambda,
                        phase_edge_aux, sim_mask, edge_label
                    ))

            if (self.phase_alignment_aux_lambda > 0
                    and getattr(base_model, '_last_phase_alignment_conf', None) is not None
                    and hasattr(graph, 'edge_pose_label')
                    and graph.edge_pose_label.numel() > 0):
                align_conf = base_model._last_phase_alignment_conf.squeeze(-1).float()
                edge_label = graph.edge_pose_label.float()
                edge_type_t = graph.edge_type
                sim_mask = (edge_type_t == 1)
                if sim_mask.any():
                    align_pred = align_conf[sim_mask].clamp(1e-7, 1.0 - 1e-7)
                    align_target = edge_label[sim_mask]
                    align_bce = F.binary_cross_entropy(
                        align_pred,
                        align_target,
                        reduction='none',
                    )
                    align_weight = torch.ones_like(align_bce)
                    if self.phase_alignment_aux_balance:
                        pos = align_target.sum().clamp(min=1.0)
                        neg = (1.0 - align_target).sum().clamp(min=1.0)
                        pos_weight = (neg / pos).clamp(min=1.0, max=20.0)
                        align_weight = torch.where(
                            align_target > 0.5,
                            align_weight * pos_weight,
                            align_weight,
                        )
                    if self.phase_alignment_aux_focal_gamma > 0:
                        pt = torch.where(align_target > 0.5, align_pred, 1.0 - align_pred)
                        align_weight = align_weight * (1.0 - pt).pow(
                            self.phase_alignment_aux_focal_gamma
                        )
                    align_aux = (
                        (align_bce * align_weight).sum()
                        / align_weight.sum().clamp(min=1.0)
                    )
                    aux_terms.append((
                        'phase-align', self.phase_alignment_aux_lambda,
                        align_aux, sim_mask, edge_label
                    ))

            # Accumulate loss across all batches in this window
            total_loss = torch.tensor(0.0, device=self.device)
            for batch_idx in range(window_start, window_end):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(triplets))
                batch_triplets = triplets[start_idx:end_idx]

                anchor_indices = batch_triplets[:, 0]
                positive_indices = batch_triplets[:, 1]
                negative_indices = batch_triplets[:, 2]

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        anchors = embeddings[anchor_indices]
                        positives = embeddings[positive_indices]
                        negatives = embeddings[negative_indices]
                        batch_loss = self.criterion(anchors, positives, negatives)
                        if self.context_aux_lambda > 0:
                            base_model_for_ctx = self.model.gnn if hasattr(self.model, 'gnn') else self.model
                            raw_dim = base_model_for_ctx.input_dim
                            ctx_end = raw_dim + base_model_for_ctx.context_dim
                            ctx_loss = self.criterion(
                                embeddings[anchor_indices, raw_dim:ctx_end],
                                embeddings[positive_indices, raw_dim:ctx_end],
                                embeddings[negative_indices, raw_dim:ctx_end],
                            )
                            batch_loss = batch_loss + self.context_aux_lambda * ctx_loss
                        if self.phase_token_aux_lambda > 0:
                            base_model_for_phase = self.model.gnn if hasattr(self.model, 'gnn') else self.model
                            phase_dim = getattr(base_model_for_phase, 'phase_token_dim', 0)
                            if phase_dim > 0:
                                phase_start = base_model_for_phase.input_dim + base_model_for_phase.context_dim
                                phase_end = phase_start + phase_dim
                                phase_loss = self.criterion(
                                    embeddings[anchor_indices, phase_start:phase_end],
                                    embeddings[positive_indices, phase_start:phase_end],
                                    embeddings[negative_indices, phase_start:phase_end],
                                )
                                batch_loss = batch_loss + self.phase_token_aux_lambda * phase_loss
                else:
                    anchors = embeddings[anchor_indices]
                    positives = embeddings[positive_indices]
                    negatives = embeddings[negative_indices]
                    batch_loss = self.criterion(anchors, positives, negatives)
                    if self.context_aux_lambda > 0:
                        base_model_for_ctx = self.model.gnn if hasattr(self.model, 'gnn') else self.model
                        raw_dim = base_model_for_ctx.input_dim
                        ctx_end = raw_dim + base_model_for_ctx.context_dim
                        ctx_loss = self.criterion(
                            embeddings[anchor_indices, raw_dim:ctx_end],
                            embeddings[positive_indices, raw_dim:ctx_end],
                            embeddings[negative_indices, raw_dim:ctx_end],
                        )
                        batch_loss = batch_loss + self.context_aux_lambda * ctx_loss
                    if self.phase_token_aux_lambda > 0:
                        base_model_for_phase = self.model.gnn if hasattr(self.model, 'gnn') else self.model
                        phase_dim = getattr(base_model_for_phase, 'phase_token_dim', 0)
                        if phase_dim > 0:
                            phase_start = base_model_for_phase.input_dim + base_model_for_phase.context_dim
                            phase_end = phase_start + phase_dim
                            phase_loss = self.criterion(
                                embeddings[anchor_indices, phase_start:phase_end],
                                embeddings[positive_indices, phase_start:phase_end],
                                embeddings[negative_indices, phase_start:phase_end],
                            )
                            batch_loss = batch_loss + self.phase_token_aux_lambda * phase_loss

                total_loss = total_loss + batch_loss / n_in_window
                epoch_losses.append(batch_loss.item())
                self.global_step += 1

            # Add aux loss once (after all batches in window) — affects the same
            # backward pass. Scaled to match the main-loss accumulation.
            if aux_terms:
                for aux_name, aux_lambda, aux_loss, sim_mask, edge_label in aux_terms:
                    total_loss = total_loss + aux_lambda * aux_loss
                    if track_vram:
                        logging.info(
                            f"  Aux {aux_name} BCE: {aux_loss.item():.4f} "
                            f"(λ={aux_lambda}, "
                            f"sim_edges={int(sim_mask.sum())}, "
                            f"label_pos_rate={edge_label[sim_mask].mean().item():.3f})"
                        )

            gate_reg, gate_reg_weight = self._gate_alpha_regularization(graph)
            if gate_reg is not None and gate_reg_weight > 0:
                total_loss = total_loss + gate_reg_weight * gate_reg
                if track_vram:
                    alpha = getattr(base_model, '_last_alpha', None)
                    alpha_mean = float(alpha.float().mean().item()) if alpha is not None else float('nan')
                    logging.info(
                        f"  Aux gate-alpha MSE: {gate_reg.item():.4f} "
                        f"(λ={gate_reg_weight:.4f}, alpha_mean={alpha_mean:.3f})"
                    )

                # Opt 6: False negative diagnostic every 5 epochs (expensive scipy cdist)
                if batch_idx == 0 and self.epoch % 5 == 0:
                    self._diagnose_false_negatives(
                        anchor_indices, positive_indices, negative_indices,
                        embeddings, poses, sequence_ids,
                    )

                if self.global_step % self.log_interval == 0:
                    logging.info(
                        f"Epoch {self.epoch + 1} | Batch {batch_idx + 1}/{n_batches} | "
                        f"Loss: {epoch_losses[-1]:.4f}"
                    )

            # Single backward pass (no retain_graph needed)
            if track_vram:
                try:
                    torch.cuda.reset_peak_memory_stats(self.device)
                except (RuntimeError, ValueError):
                    pass
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            if track_vram:
                try:
                    bwd_peak = torch.cuda.max_memory_allocated(self.device) / 1e9
                    logging.info(f"  VRAM after backward: peak {bwd_peak:.2f} GB")
                except (RuntimeError, ValueError):
                    pass
            self.optimizer.zero_grad()

            del embeddings, total_loss

        # Opt 5: Single empty_cache at epoch end (not per-window)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)

        return avg_loss

    def _train_epoch_smoothap(
        self,
        graph: Data,
        poses: np.ndarray,
        descriptors: np.ndarray,
        sequence_ids: np.ndarray,
        mine_every_n_epochs: int,
    ) -> float:
        """SmoothAP training epoch: per-anchor multi-positive AP optimization."""
        from gnn.triplet_miner import mine_pos_neg_lists

        # Mine pos/neg pools once (raw-based, stable across epochs)
        should_mine = (
            self._smoothap_anchors is None
            or (self.epoch % mine_every_n_epochs == 0 and self._last_mine_epoch != self.epoch)
        )
        if should_mine:
            logging.info(f"Mining SmoothAP pos/neg pools for epoch {self.epoch + 1}...")
            t0 = time.perf_counter()
            anchors, pos_pool, neg_pool = mine_pos_neg_lists(
                descriptors=descriptors,
                poses=poses,
                sequence_ids=sequence_ids,
                pos_dist_max=5.0,
                neg_dist_min=10.0,
                temporal_min=30,
                n_neg_per_anchor=self.smoothap_n_neg,
                metric='l2',
            )
            self._smoothap_anchors = anchors
            self._smoothap_pos_pool = pos_pool
            self._smoothap_neg_pool = neg_pool
            self._last_mine_epoch = self.epoch
            n_pos_total = sum(len(v) for v in pos_pool.values())
            logging.info(
                f"Mined {len(anchors):,} valid anchors, {n_pos_total:,} total positives "
                f"({n_pos_total / max(len(anchors), 1):.1f}/anchor) in {time.perf_counter() - t0:.1f}s"
            )
        else:
            logging.info(f"Reusing SmoothAP pools from epoch {self._last_mine_epoch + 1}")

        anchors_arr = self._smoothap_anchors
        pos_pool = self._smoothap_pos_pool
        neg_pool = self._smoothap_neg_pool
        if len(anchors_arr) == 0:
            logging.warning("No valid SmoothAP anchors!")
            return 0.0

        # Sample anchor batches for this epoch
        rng = np.random.default_rng(seed=self.epoch)
        shuffled = rng.permutation(anchors_arr)
        batch_anchors = self.smoothap_batch_anchors
        accumulation_steps = 4
        n_batches = (len(shuffled) + batch_anchors - 1) // batch_anchors

        epoch_losses = []
        self.optimizer.zero_grad()
        K = self.smoothap_n_pos
        M = self.smoothap_n_neg

        for window_start in range(0, n_batches, accumulation_steps):
            window_end = min(window_start + accumulation_steps, n_batches)
            n_in_window = window_end - window_start

            # Forward once per accumulation window
            track_vram = (window_start == 0)
            if track_vram:
                try:
                    torch.cuda.reset_peak_memory_stats(self.device)
                except (RuntimeError, ValueError):
                    pass
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    embeddings = self.model(graph)
            else:
                embeddings = self.model(graph)
            if track_vram:
                try:
                    fwd_peak = torch.cuda.max_memory_allocated(self.device) / 1e9
                    logging.info(f"  VRAM after forward: {torch.cuda.memory_allocated()/1e9:.2f} GB, "
                                 f"peak: {fwd_peak:.2f} GB")
                except (RuntimeError, ValueError):
                    pass

            total_loss = torch.tensor(0.0, device=self.device)
            for batch_idx in range(window_start, window_end):
                start_idx = batch_idx * batch_anchors
                end_idx = min((batch_idx + 1) * batch_anchors, len(shuffled))
                batch_a = shuffled[start_idx:end_idx]
                B = len(batch_a)

                # Build padded pos/neg index tensors
                pos_idx = np.zeros((B, K), dtype=np.int64)
                neg_idx = np.zeros((B, M), dtype=np.int64)
                pos_mask = np.zeros((B, K), dtype=bool)
                neg_mask = np.zeros((B, M), dtype=bool)
                for i, a in enumerate(batch_a):
                    pos_arr = pos_pool[int(a)]
                    neg_arr = neg_pool[int(a)]
                    K_real = min(len(pos_arr), K)
                    M_real = min(len(neg_arr), M)
                    if K_real > 0:
                        sampled_pos = rng.choice(pos_arr, K_real, replace=False)
                        pos_idx[i, :K_real] = sampled_pos
                        pos_mask[i, :K_real] = True
                        # pad rest with first valid (ignored by mask)
                        if K_real < K:
                            pos_idx[i, K_real:] = sampled_pos[0]
                    if M_real > 0:
                        sampled_neg = rng.choice(neg_arr, M_real, replace=False) if len(neg_arr) >= M_real else neg_arr
                        neg_idx[i, :M_real] = sampled_neg
                        neg_mask[i, :M_real] = True
                        if M_real < M:
                            neg_idx[i, M_real:] = sampled_neg[0]

                a_t = torch.from_numpy(batch_a.astype(np.int64)).to(self.device)
                p_t = torch.from_numpy(pos_idx).to(self.device)
                n_t = torch.from_numpy(neg_idx).to(self.device)
                pmask_t = torch.from_numpy(pos_mask).to(self.device)
                nmask_t = torch.from_numpy(neg_mask).to(self.device)

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        a_emb = embeddings[a_t]                  # (B, D)
                        p_emb = embeddings[p_t]                  # (B, K, D)
                        n_emb = embeddings[n_t]                  # (B, M, D)
                        batch_loss = self.criterion(a_emb, p_emb, n_emb, pmask_t, nmask_t)
                else:
                    a_emb = embeddings[a_t]
                    p_emb = embeddings[p_t]
                    n_emb = embeddings[n_t]
                    batch_loss = self.criterion(a_emb, p_emb, n_emb, pmask_t, nmask_t)

                total_loss = total_loss + batch_loss / n_in_window
                epoch_losses.append(batch_loss.item())
                self.global_step += 1

                if self.global_step % self.log_interval == 0:
                    logging.info(
                        f"Epoch {self.epoch + 1} | Batch {batch_idx + 1}/{n_batches} | "
                        f"SmoothAP Loss: {epoch_losses[-1]:.4f}"
                    )

            if track_vram:
                try:
                    torch.cuda.reset_peak_memory_stats(self.device)
                except (RuntimeError, ValueError):
                    pass
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            if track_vram:
                try:
                    bwd_peak = torch.cuda.max_memory_allocated(self.device) / 1e9
                    logging.info(f"  VRAM after backward: peak {bwd_peak:.2f} GB")
                except (RuntimeError, ValueError):
                    pass
            self.optimizer.zero_grad()

            del embeddings, total_loss

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = float(np.mean(epoch_losses))
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def extract_all_embeddings(self, graph: Data) -> np.ndarray:
        """Full-graph forward pass to extract current GNN embeddings on CPU.

        Used by two-pass similarity edge refinement to rebuild edges in the
        learned ctx space. The model is set to eval() during the forward pass
        (disables dropout) and restored to train() before returning.

        Args:
            graph: PyG Data object (moved to device automatically)

        Returns:
            (N, embedding_dim) float32 numpy array on CPU.
        """
        was_training = self.model.training
        self.model.eval()
        graph = graph.to(self.device)
        embeddings = self.model(graph)
        emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
        del embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if was_training:
            self.model.train()
        return emb_np

    def _refine_similarity_edges(
        self,
        graph: Data,
        poses: np.ndarray,
        refine_space: str = 'ctx',
        fit_kwargs: Optional[Dict] = None,
        edge_kwargs: Optional[Dict] = None,
    ):
        """Two-pass similarity edge refinement.

        1. Extract current GNN embeddings (full-graph forward).
        2. Select refinement subspace (ctx / full / raw).
        3. Refit Bayesian similarity distribution on that subspace.
        4. Rebuild similarity edges in-place; preserve temporal edges.

        Args:
            graph: Training graph (mutated in place).
            poses: (N, 4, 4) SE(3) training poses for Bayesian fit.
            refine_space: 'ctx' (last context_dim), 'full' (all dims), or 'raw' (input_dim).
            fit_kwargs: Forwarded to SimilarityDistribution.fit (pos_dist, neg_dist,
                min_temporal_gap, n_samples).
            edge_kwargs: Forwarded to rebuild_similarity_edges (confidence_level,
                similarity_max_k, etc.).
        """
        from utils.similarity_stats import SimilarityDistribution
        from utils.standardization_stats import StandardizationStats
        from keyframe.graph_manager import rebuild_similarity_edges

        fit_kwargs = dict(fit_kwargs or {})
        edge_kwargs = dict(edge_kwargs or {})

        refine_start = time.perf_counter()
        logging.info(f"[2-pass] Refining similarity edges at epoch {self.epoch + 1}...")

        # 1. Extract embeddings
        embeddings = self.extract_all_embeddings(graph)  # (N, embedding_dim)

        base_model = self.model.gnn if hasattr(self.model, 'gnn') else self.model
        raw_dim = base_model.input_dim
        ctx_end = raw_dim + base_model.context_dim

        # 2. Select refinement subspace
        if refine_space == 'ctx':
            refine_descs = embeddings[:, raw_dim:ctx_end]
        elif refine_space == 'full':
            refine_descs = embeddings
        elif refine_space == 'raw':
            refine_descs = embeddings[:, :raw_dim]
        elif refine_space == 'phase':
            refine_descs = embeddings[:, ctx_end:]
        else:
            raise ValueError(f"Unknown refine_space: {refine_space}")
        logging.info(f"[2-pass] refine_space='{refine_space}', shape={refine_descs.shape}")

        # 3. Standardize (Bayesian L2 metric requires z-scored descriptors)
        std_stats = StandardizationStats().fit(refine_descs)
        refine_descs_std = std_stats.transform(refine_descs)

        # 4. Refit Bayesian distribution on new subspace
        similarity_metric = edge_kwargs.get('similarity_metric', 'l2')
        graph_sequence_ids = None
        if hasattr(graph, 'sequence_ids'):
            graph_sequence_ids = graph.sequence_ids.detach().cpu().numpy()

        new_dist = SimilarityDistribution(metric=similarity_metric).fit(
            refine_descs_std, poses,
            sequence_ids=graph_sequence_ids,
            pos_dist=fit_kwargs.get('pos_dist', 5.0),
            neg_dist=fit_kwargs.get('neg_dist', 10.0),
            min_temporal_gap=fit_kwargs.get('min_temporal_gap', 30),
            n_samples=fit_kwargs.get('n_samples', 1_000_000),
        )

        if not new_dist.fitted:
            logging.warning(
                "[2-pass] Refit failed (too few same-place pairs in new space). "
                "Keeping current edges."
            )
            return

        # 5. Rebuild similarity edges (preserves temporal edges of type 0)
        edge_kwargs_merged = dict(edge_kwargs)
        edge_kwargs_merged['similarity_dist'] = new_dist
        edge_kwargs_merged['standardization_stats'] = std_stats
        # refine_descs is pre-standardized logic inside _build_similarity_edges
        # expects unnormalized input when metric='l2' + std_stats — so pass raw refine_descs.
        _, n_sim = rebuild_similarity_edges(
            graph, refine_descs, **edge_kwargs_merged,
        )

        # Save state so validate() can apply matching refinement to val graphs.
        # This keeps train and val graphs structurally consistent (same edge
        # building policy in the same ctx subspace).
        self._refined_similarity_dist = new_dist
        self._refined_std_stats = std_stats
        self._refined_edge_kwargs = dict(edge_kwargs)
        self._refined_space = refine_space

        n_temporal = int((graph.edge_type == 0).sum())
        refine_time = time.perf_counter() - refine_start
        logging.info(
            f"[2-pass] Rebuilt edges: {n_temporal:,} temporal + {n_sim:,} similarity "
            f"in {refine_time:.1f}s"
        )

    def validate(
        self,
        val_graph: Data,
        val_poses: np.ndarray,
        distance_threshold: float = 5.0,
        skip_frames: int = 30,
        per_query_records: Optional[List[dict]] = None,
    ) -> Dict[str, float]:
        """
        Validate model on validation set using loop closure evaluation.

        Only evaluates on "revisit" queries - frames that return to a previously
        visited location (within distance_threshold) after at least skip_frames.

        Args:
            val_graph: Validation graph
            val_poses: Validation poses
            distance_threshold: Distance threshold for positive match (meters)
            skip_frames: Minimum temporal gap to consider as loop closure

        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        base_model = self.model.gnn if hasattr(self.model, 'gnn') else self.model

        # Two-pass refinement: if a trained similarity distribution exists,
        # apply the same edge rebuild policy to the val graph before retrieval.
        # This keeps train/val graph structure consistent — without this, the
        # GNN would see dense (refined) graphs during training but temporal-only
        # graphs at eval time, causing a train-test distribution shift.
        val_refinement_active = self._refined_similarity_dist is not None

        with torch.no_grad():
            val_graph = val_graph.to(self.device)

            if val_refinement_active:
                # First forward: extract ctx embeddings from current val graph
                # (temporal-only initially; cached refined edges on subsequent passes).
                probe_emb = self.model(val_graph)
                raw_dim = base_model.input_dim
                ctx_end = raw_dim + base_model.context_dim
                if self._refined_space == 'ctx':
                    probe_descs = probe_emb[:, raw_dim:ctx_end].detach().cpu().numpy().astype(np.float32)
                elif self._refined_space == 'full':
                    probe_descs = probe_emb.detach().cpu().numpy().astype(np.float32)
                elif self._refined_space == 'raw':
                    probe_descs = probe_emb[:, :raw_dim].detach().cpu().numpy().astype(np.float32)
                elif self._refined_space == 'phase':
                    probe_descs = probe_emb[:, ctx_end:].detach().cpu().numpy().astype(np.float32)
                else:
                    probe_descs = probe_emb[:, raw_dim:ctx_end].detach().cpu().numpy().astype(np.float32)
                del probe_emb

                # Rebuild val similarity edges using train-fitted distribution.
                # (Reusing trained stats keeps train/val policy identical.)
                from keyframe.graph_manager import rebuild_similarity_edges
                edge_kwargs = dict(self._refined_edge_kwargs)
                edge_kwargs['similarity_dist'] = self._refined_similarity_dist
                edge_kwargs['standardization_stats'] = self._refined_std_stats
                _, n_val_sim = rebuild_similarity_edges(
                    val_graph, probe_descs, **edge_kwargs,
                )

            # Final forward pass (with refined edges if applicable)
            embeddings = self.model(val_graph)
            embeddings_np = embeddings.cpu().numpy()
            raw_descriptors_np = val_graph.x.cpu().numpy()

            # Residual gate diagnostic: log per-dataset α statistics
            gate_alpha_np = None
            if hasattr(base_model, '_last_alpha') and base_model._last_alpha is not None:
                gate_alpha_np = base_model._last_alpha.cpu().numpy().squeeze(-1)

            # If spectral policy is active, also extract policy-only descriptors
            policy_descriptors_np = None
            has_policy = hasattr(base_model, 'spectral_policy') and base_model.spectral_policy is not None
            if has_policy and hasattr(val_graph, 'x_fft') and val_graph.x_fft is not None:
                x_fft = val_graph.x_fft.view(-1, base_model.spectral_policy.n_rings, base_model.spectral_policy.n_freqs)
                policy_out = base_model.spectral_policy(x_fft)
                policy_descriptors_np = policy_out.cpu().numpy()

            # Free GPU tensors and move val graph back to CPU
            del embeddings
            val_graph.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # GNN output: configured R@K values in one pass (single KD-Tree + FAISS build)
        recalls, n_queries = self._compute_recall_multi_k(
            embeddings_np, val_poses,
            k_values=self.recall_k_values,
            distance_threshold=distance_threshold,
            skip_frames=skip_frames,
            per_query_records=per_query_records,
        )

        # Raw baseline (precomputed descriptors): R@1
        raw_recalls, _ = self._compute_recall_multi_k(
            raw_descriptors_np, val_poses,
            k_values=[1],
            distance_threshold=distance_threshold,
            skip_frames=skip_frames
        )

        # Policy-only baseline (if active): R@1
        policy_recall_1 = None
        if policy_descriptors_np is not None:
            policy_recalls, _ = self._compute_recall_multi_k(
                policy_descriptors_np, val_poses,
                k_values=[1],
                distance_threshold=distance_threshold,
                skip_frames=skip_frames
            )
            policy_recall_1 = policy_recalls[1]

        # GNN context only: slice ctx_norm from cat(raw_norm, ctx_norm)
        raw_dim = base_model.input_dim
        ctx_end = raw_dim + base_model.context_dim
        ctx_only_np = embeddings_np[:, raw_dim:ctx_end]
        ctx_recalls, _ = self._compute_recall_multi_k(
            ctx_only_np, val_poses,
            k_values=[1],
            distance_threshold=distance_threshold,
            skip_frames=skip_frames
        )

        result = {
            'recall@1': recalls[1],
            'raw_recall@1': raw_recalls[1],
            'ctx_recall@1': ctx_recalls[1],
            'n_queries': n_queries
        }
        for k in self.recall_k_values:
            result[f'recall@{k}'] = recalls[k]
        if getattr(base_model, 'phase_token_dim', 0) > 0:
            phase_only_np = embeddings_np[:, ctx_end:]
            phase_recalls, _ = self._compute_recall_multi_k(
                phase_only_np, val_poses,
                k_values=[1],
                distance_threshold=distance_threshold,
                skip_frames=skip_frames
            )
            result['phase_recall@1'] = phase_recalls[1]
        if policy_recall_1 is not None:
            result['policy_recall@1'] = policy_recall_1
        if gate_alpha_np is not None:
            result['gate_alpha_mean'] = float(gate_alpha_np.mean())
            result['gate_alpha_p10'] = float(np.percentile(gate_alpha_np, 10))
            result['gate_alpha_p90'] = float(np.percentile(gate_alpha_np, 90))
        return result

    def validate_all(
        self,
        val_datasets: Dict[str, Dict],
        distance_threshold: float = 5.0,
        skip_frames: int = 30,
        per_query_dump_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Validate model on multiple validation datasets.

        Args:
            val_datasets: Dict of {dataset_name: {'graph': Data, 'poses': np.array}}
            distance_threshold: Distance threshold for positive match (meters)
            skip_frames: Minimum temporal gap to consider as loop closure

        Returns:
            Dictionary with per-dataset metrics and aggregated metrics
        """
        all_metrics = {}
        total_correct_at_1 = 0
        total_raw_correct_at_1 = 0
        total_ctx_correct_at_1 = 0
        total_phase_correct_at_1 = 0
        has_phase_metrics = False
        total_queries = 0

        if per_query_dump_dir is not None:
            os.makedirs(per_query_dump_dir, exist_ok=True)

        logging.info("Validation (per-dataset):")
        for dataset_name, info in val_datasets.items():
            records = [] if per_query_dump_dir is not None else None
            metrics = self.validate(
                info['graph'],
                info['poses'],
                distance_threshold=distance_threshold,
                skip_frames=skip_frames,
                per_query_records=records,
            )
            all_metrics[dataset_name] = metrics

            if records is not None:
                dump_path = os.path.join(per_query_dump_dir, f"{dataset_name}.json")
                with open(dump_path, 'w') as f:
                    json.dump({
                        'dataset': dataset_name,
                        'distance_threshold_m': distance_threshold,
                        'skip_frames': skip_frames,
                        'n_queries': len(records),
                        'records': records,
                    }, f, indent=2)
                logging.info(f"  [per-query dump] {dump_path} ({len(records)} records)")

            # Log per-dataset results
            policy_str = ""
            if 'policy_recall@1' in metrics:
                policy_str = f", policy: {metrics['policy_recall@1']:.4f}"
            phase_str = ""
            if 'phase_recall@1' in metrics:
                phase_str = f", phase: {metrics['phase_recall@1']:.4f}"
            gate_str = ""
            if 'gate_alpha_mean' in metrics:
                gate_str = (f" | α: mean={metrics['gate_alpha_mean']:.3f} "
                            f"[p10={metrics['gate_alpha_p10']:.3f}, "
                            f"p90={metrics['gate_alpha_p90']:.3f}]")
            extra_recall_str = ""
            if 5 in self.recall_k_values and 'recall@5' in metrics:
                extra_recall_str += f" | R@5: {metrics['recall@5']:.4f}"
            if 10 in self.recall_k_values and 'recall@10' in metrics:
                extra_recall_str += f" | R@10: {metrics['recall@10']:.4f}"
            if 800 in self.recall_k_values and 'recall@800' in metrics:
                extra_recall_str += f" | R@800: {metrics['recall@800']:.4f}"
            logging.info(
                f"  {dataset_name:20s} | R@1: {metrics['recall@1']:.4f} "
                f"(raw: {metrics['raw_recall@1']:.4f}, ctx: {metrics['ctx_recall@1']:.4f}"
                f"{phase_str}{policy_str})"
                f"{extra_recall_str} | Queries: {metrics['n_queries']}{gate_str}"
            )

            # Accumulate for weighted average
            total_correct_at_1 += metrics['recall@1'] * metrics['n_queries']
            total_raw_correct_at_1 += metrics['raw_recall@1'] * metrics['n_queries']
            total_ctx_correct_at_1 += metrics['ctx_recall@1'] * metrics['n_queries']
            if 'phase_recall@1' in metrics:
                total_phase_correct_at_1 += metrics['phase_recall@1'] * metrics['n_queries']
                has_phase_metrics = True
            total_queries += metrics['n_queries']

        # Compute weighted average
        if total_queries > 0:
            avg_recall_at_1 = total_correct_at_1 / total_queries
            avg_raw_recall_at_1 = total_raw_correct_at_1 / total_queries
            avg_ctx_recall_at_1 = total_ctx_correct_at_1 / total_queries
            avg_phase_recall_at_1 = total_phase_correct_at_1 / total_queries
        else:
            avg_recall_at_1 = 0.0
            avg_raw_recall_at_1 = 0.0
            avg_ctx_recall_at_1 = 0.0
            avg_phase_recall_at_1 = 0.0

        all_metrics['_average'] = {
            'recall@1': avg_recall_at_1,
            'raw_recall@1': avg_raw_recall_at_1,
            'ctx_recall@1': avg_ctx_recall_at_1,
            'n_queries': total_queries
        }
        if has_phase_metrics:
            all_metrics['_average']['phase_recall@1'] = avg_phase_recall_at_1
            phase_avg_str = f", phase: {avg_phase_recall_at_1:.4f}"
        else:
            phase_avg_str = ""
        logging.info(
            f"  {'AVERAGE':20s} | R@1: {avg_recall_at_1:.4f} "
            f"(raw: {avg_raw_recall_at_1:.4f}, ctx: {avg_ctx_recall_at_1:.4f}"
            f"{phase_avg_str}) | Total Queries: {total_queries}"
        )

        dataset_metrics = {
            name: metrics for name, metrics in all_metrics.items()
            if not name.startswith('_') and metrics.get('n_queries', 0) > 0
        }
        if dataset_metrics:
            seq_macro = float(np.mean([m['recall@1'] for m in dataset_metrics.values()]))
            raw_seq_macro = float(np.mean([m['raw_recall@1'] for m in dataset_metrics.values()]))
            ctx_seq_macro = float(np.mean([m['ctx_recall@1'] for m in dataset_metrics.values()]))
        else:
            seq_macro = raw_seq_macro = ctx_seq_macro = 0.0

        sensor_groups = {}
        for dataset_name, metrics in dataset_metrics.items():
            sensor_label = self._sensor_label_from_dataset_name(dataset_name)
            sensor_groups.setdefault(sensor_label, []).append(metrics)
        if sensor_groups:
            sensor_macro = float(np.mean([
                np.mean([m['recall@1'] for m in group])
                for group in sensor_groups.values()
            ]))
            raw_sensor_macro = float(np.mean([
                np.mean([m['raw_recall@1'] for m in group])
                for group in sensor_groups.values()
            ]))
            ctx_sensor_macro = float(np.mean([
                np.mean([m['ctx_recall@1'] for m in group])
                for group in sensor_groups.values()
            ]))
        else:
            sensor_macro = raw_sensor_macro = ctx_sensor_macro = 0.0

        all_metrics['_sequence_macro'] = {
            'recall@1': seq_macro,
            'raw_recall@1': raw_seq_macro,
            'ctx_recall@1': ctx_seq_macro,
            'n_datasets': len(dataset_metrics),
        }
        all_metrics['_sensor_macro'] = {
            'recall@1': sensor_macro,
            'raw_recall@1': raw_sensor_macro,
            'ctx_recall@1': ctx_sensor_macro,
            'n_sensors': len(sensor_groups),
        }
        logging.info(
            f"  {'SEQ-MACRO':20s} | R@1: {seq_macro:.4f} "
            f"(raw: {raw_seq_macro:.4f}, ctx: {ctx_seq_macro:.4f}) | "
            f"Datasets: {len(dataset_metrics)}"
        )
        logging.info(
            f"  {'SENSOR-MACRO':20s} | R@1: {sensor_macro:.4f} "
            f"(raw: {raw_sensor_macro:.4f}, ctx: {ctx_sensor_macro:.4f}) | "
            f"Sensors: {len(sensor_groups)}"
        )

        return all_metrics

    def _compute_recall_multi_k(
        self,
        embeddings: np.ndarray,
        poses: np.ndarray,
        k_values: List[int],
        distance_threshold: float,
        skip_frames: int = 30,
        per_query_records: Optional[List[dict]] = None,
    ) -> Tuple[Dict[int, float], int]:
        """
        Compute Recall@K for multiple K values in a single pass.

        Builds KD-Tree and FAISS index once, evaluates all K values.
        Only evaluates on "revisit" queries - frames that return to a previously
        visited location. This measures actual place recognition ability,
        not temporal similarity.

        Args:
            embeddings: (n, embedding_dim) embeddings
            poses: (n, 4, 4) poses
            k_values: List of K values for Recall@K (e.g., [1, 5, 10])
            distance_threshold: Distance threshold for positive match (meters)
            skip_frames: Minimum temporal gap to consider as loop closure

        Returns:
            recalls: {k: recall_value} for each k in k_values
            n_queries: Number of loop closure queries found
        """
        from scipy.spatial import cKDTree
        import faiss

        n = len(embeddings)
        max_k = max(k_values)

        # Extract positions from SE(3) poses
        positions = poses[:, :3, 3].astype(np.float64)

        # Build KD-Tree for spatial queries (memory efficient)
        spatial_tree = cKDTree(positions)

        # Find loop closure queries using KD-Tree (O(n log n) instead of O(n²))
        queries = []
        for j in range(skip_frames, n):
            # Find frames within distance_threshold of frame j
            nearby_indices = spatial_tree.query_ball_point(positions[j], distance_threshold)
            # Filter: only consider frames at least skip_frames earlier
            for i in nearby_indices:
                if i <= j - skip_frames:
                    queries.append((j, i))
                    break  # Only count first revisit per query frame

        if len(queries) == 0:
            return {k: 0.0 for k in k_values}, 0

        # Build FAISS index for cosine similarity search (consistent with cosine loss)
        embeddings_f32 = embeddings.astype(np.float32)
        d = embeddings_f32.shape[1]
        faiss.normalize_L2(embeddings_f32)          # in-place L2 normalize
        faiss_index = faiss.IndexFlatIP(d)           # inner product = cosine sim
        faiss_index.add(embeddings_f32)

        # For each query, get top-(max_k + 2*skip_frames) candidates to filter temporal neighbors
        search_k = min(max_k + 2 * skip_frames, n)

        # Opt 4: Batched FAISS search (all queries at once instead of per-query loop)
        query_indices = np.array([q[0] for q in queries], dtype=np.int64)
        query_embs = embeddings_f32[query_indices]  # (n_queries, d)
        all_distances, all_indices = faiss_index.search(query_embs, search_k)

        # Evaluate each query
        correct_at_k = {k: 0 for k in k_values}

        for i, (query_idx, true_match_idx) in enumerate(queries):
            # Filter out temporal neighbors (vectorized)
            valid_mask = np.abs(all_indices[i] - query_idx) > skip_frames
            valid_indices = all_indices[i][valid_mask]

            if len(valid_indices) == 0:
                continue

            # Take top-max_k valid candidates and compute geo distances once
            valid_indices_max = valid_indices[:max_k]
            geo_dists = np.linalg.norm(positions[valid_indices_max] - positions[query_idx], axis=1)

            # Check each k value
            for k in k_values:
                if np.any(geo_dists[:k] < distance_threshold):
                    correct_at_k[k] += 1

            if per_query_records is not None:
                rank_arr = np.where(valid_indices == true_match_idx)[0]
                true_rank = int(rank_arr[0]) + 1 if rank_arr.size > 0 else -1
                top1_cosine = float(all_distances[i][valid_mask][0])
                R_q = poses[query_idx, :3, :3]
                R_m = poses[true_match_idx, :3, :3]
                yaw_q = np.arctan2(R_q[1, 0], R_q[0, 0])
                yaw_m = np.arctan2(R_m[1, 0], R_m[0, 0])
                dyaw = np.degrees(np.arctan2(np.sin(yaw_q - yaw_m), np.cos(yaw_q - yaw_m)))
                per_query_records.append({
                    'query_idx': int(query_idx),
                    'true_match_idx': int(true_match_idx),
                    'top1_idx': int(valid_indices_max[0]),
                    'top1_cosine_sim': top1_cosine,
                    'top1_geo_dist_m': float(geo_dists[0]),
                    'true_match_rank': true_rank,
                    'success_at_k1': bool(geo_dists[0] < distance_threshold),
                    'delta_yaw_deg': float(dyaw),
                })

        recalls = {k: correct_at_k[k] / len(queries) for k in k_values}
        return recalls, len(queries)

    def train(
        self,
        train_graph: Data,
        train_poses: np.ndarray,
        train_descriptors: np.ndarray,
        train_sequence_ids: np.ndarray = None,
        val_datasets: Optional[Dict[str, Dict]] = None,
        n_epochs: int = 50,
        triplet_miner: Optional[TripletMiner] = None,
        mine_every_n_epochs: int = 1,
        validate_every_n_epochs: int = 1,
        n_triplets_per_anchor: int = 1,
        two_pass_cfg: Optional[Dict] = None,
        refine_edge_kwargs: Optional[Dict] = None,
        refine_fit_kwargs: Optional[Dict] = None,
        temperature_schedule: Optional[Dict] = None,
        triplet_sampling_config: Optional[Dict] = None,
    ):
        """
        Full training loop

        Args:
            train_graph: Training graph
            train_poses: Training poses
            train_descriptors: Training descriptors
            train_sequence_ids: Sequence IDs for per-sequence triplet mining
            val_datasets: Dict of {dataset_name: {'graph': Data, 'poses': np.array}}
            n_epochs: Number of epochs
            triplet_miner: Triplet miner (created if None)
            mine_every_n_epochs: Re-mine triplets every N epochs (cache between)
            validate_every_n_epochs: Run validation every N epochs
            n_triplets_per_anchor: Number of mined triplets per anchor
            temperature_schedule: Optional linear InfoNCE temperature schedule
        """
        if triplet_miner is None:
            from gnn.triplet_miner import create_triplet_miner
            triplet_miner = create_triplet_miner()

        logging.info(f"Starting training for {n_epochs} epochs...")
        logging.info(f"Training graph: {train_graph.num_nodes:,} nodes, {train_graph.edge_index.shape[1]:,} edges")
        if train_sequence_ids is not None:
            logging.info(f"Per-sequence mining enabled: {len(np.unique(train_sequence_ids))} sequences")

        # LR scheduler: 3-epoch linear warmup → cosine decay to 1e-6
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        n_warmup = min(3, n_epochs)
        warmup_sched = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_warmup)
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=max(n_epochs - n_warmup, 1), eta_min=1e-6)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[n_warmup]
        )
        logging.info(f"LR scheduler: {n_warmup}-epoch warmup → cosine decay (eta_min=1e-6)")
        if mine_every_n_epochs > 1:
            logging.info(f"Triplet mining cache: re-mine every {mine_every_n_epochs} epochs")
        logging.info(f"Triplet mining: {n_triplets_per_anchor} triplets/anchor")
        if validate_every_n_epochs > 1:
            logging.info(f"Validation frequency: every {validate_every_n_epochs} epochs")
        logging.info(f"Checkpoint metric: {self.checkpoint_metric}")
        logging.info(f"Validation recall K values: {self.recall_k_values}")

        temp_schedule = dict(temperature_schedule or {})
        temp_schedule_enabled = (
            bool(temp_schedule.get('enabled', False))
            and hasattr(self.criterion, 'temperature')
        )
        if temp_schedule_enabled:
            temp_start = float(temp_schedule.get('start', self.criterion.temperature))
            temp_end = float(temp_schedule.get('end', self.criterion.temperature))
            logging.info(
                f"InfoNCE temperature schedule: {temp_start:.4f} -> {temp_end:.4f} "
                f"over {n_epochs} epochs"
            )

        # Two-pass similarity edge refinement setup
        two_pass_enabled = bool(two_pass_cfg and two_pass_cfg.get('enabled', False))
        if two_pass_enabled:
            warmup_epochs = int(two_pass_cfg.get('warmup_epochs', 10))
            refine_every = int(two_pass_cfg.get('refine_every', 5))
            refine_space = two_pass_cfg.get('refine_space', 'ctx')
            logging.info(
                f"[2-pass] Enabled: warmup={warmup_epochs} ep, refine_every={refine_every} ep, "
                f"refine_space='{refine_space}'"
            )

        total_training_start = time.perf_counter()

        for epoch in range(n_epochs):
            self.epoch = epoch
            epoch_start = time.perf_counter()

            if temp_schedule_enabled:
                progress = epoch / max(n_epochs - 1, 1)
                self.criterion.temperature = temp_start + (temp_end - temp_start) * progress
                logging.info(f"InfoNCE temperature: {self.criterion.temperature:.4f}")

            # Two-pass: refine similarity edges before this epoch
            if (two_pass_enabled
                and epoch >= warmup_epochs
                and (epoch - warmup_epochs) % refine_every == 0):
                self._refine_similarity_edges(
                    train_graph, train_poses,
                    refine_space=refine_space,
                    fit_kwargs=refine_fit_kwargs or {},
                    edge_kwargs=refine_edge_kwargs or {},
                )

            # Train
            avg_loss = self.train_epoch(
                train_graph,
                triplet_miner,
                train_poses,
                train_descriptors,
                sequence_ids=train_sequence_ids,
                n_triplets_per_anchor=n_triplets_per_anchor,
                mine_every_n_epochs=mine_every_n_epochs,
                triplet_sampling_config=triplet_sampling_config,
            )
            train_time = time.perf_counter() - epoch_start

            # Opt 3: Validate every N epochs, plus first/last/near-early-stop
            should_validate = (
                val_datasets is not None and len(val_datasets) > 0
                and (
                    (epoch + 1) % validate_every_n_epochs == 0
                    or epoch == 0
                    or epoch == n_epochs - 1
                    or self.epochs_without_improvement >= self.patience - 2
                )
            )

            if should_validate:
                val_start = time.perf_counter()
                all_metrics = self.validate_all(val_datasets)
                avg_metrics = all_metrics['_average']
                metric_name, selected_val_metric = self._checkpoint_metric_value(all_metrics)
                self.val_metrics.append(all_metrics)
                val_time = time.perf_counter() - val_start

                epoch_total = time.perf_counter() - epoch_start
                logging.info(
                    f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f} | "
                    f"Avg R@1: {avg_metrics['recall@1']:.4f} | "
                    f"{metric_name}: {selected_val_metric:.4f} | "
                    f"Time: {epoch_total:.1f}s (train={train_time:.1f}s, val={val_time:.1f}s)"
                )

                # Save best model using the configured checkpoint metric.
                if selected_val_metric > self.best_val_metric:
                    self.best_val_metric = selected_val_metric
                    self.save_checkpoint('best_model.pth')
                    logging.info(
                        f"  -> New best model! {metric_name}: {self.best_val_metric:.4f}"
                    )
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    logging.info(f"  -> No improvement for {self.epochs_without_improvement} epoch(s)")

                # Early stopping
                if self.epochs_without_improvement >= self.patience:
                    logging.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                    logging.info(f"Best validation metric ({self.checkpoint_metric}): {self.best_val_metric:.4f}")
                    break
            else:
                epoch_total = time.perf_counter() - epoch_start
                logging.info(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f} | Time: {epoch_total:.1f}s (train={train_time:.1f}s, val=skipped)")

            # LR scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
                logging.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

        # Save final model
        total_time = time.perf_counter() - total_training_start
        self.save_checkpoint('final_model.pth')
        logging.info(
            f"Training complete! Total time: {total_time/3600:.2f}h | "
            f"Best {self.checkpoint_metric}: {self.best_val_metric:.4f}"
        )

    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'checkpoint_metric': self.checkpoint_metric,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'epochs_without_improvement': self.epochs_without_improvement
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        logging.info(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, filename: str):
        """Load checkpoint"""
        load_path = self.checkpoint_dir / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)

        missing, unexpected = self.model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        if missing:
            logging.warning(f"Missing keys in checkpoint (new params): {missing}")
        if unexpected:
            logging.warning(f"Unexpected keys in checkpoint (removed params): {unexpected}")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

        logging.info(f"Loaded checkpoint: {load_path}")
        logging.info(f"Epoch: {self.epoch} | Best R@1: {self.best_val_metric:.4f}")


def create_trainer(
    model: Optional[nn.Module] = None,
    device: str = 'cuda',
    **kwargs
) -> GNNTrainer:
    """
    Factory function to create trainer

    Args:
        model: GNN model (created if None)
        device: Device for training
        **kwargs: Additional arguments for trainer

    Returns:
        GNNTrainer instance
    """
    if model is None:
        model = create_spectral_gnn()

    return GNNTrainer(model=model, device=device, **kwargs)
