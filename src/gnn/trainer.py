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
        use_amp: bool = True  # Mixed precision training
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
        """
        self.model = model.to(device)
        self.device = device

        # Multi-GPU support
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        else:
            print(f"Using single GPU for training")

        self.patience = patience
        self.epochs_without_improvement = 0

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

        self.criterion = InfoNCELoss(temperature=temperature)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = log_interval

        # LR scheduler (created in train() once n_epochs is known)
        self.scheduler = None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0

        # History
        self.train_losses = []
        self.val_metrics = []

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
        n_triplets_per_anchor: int = 1
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

        Returns:
            Average loss for epoch
        """
        # Move graph to device once (needed for both mining and training)
        graph = graph.to(self.device)

        # Mine triplets using raw descriptors (graph.x), NOT the GNN output.
        # Rationale: mining on the combined cat(raw, ctx) embedding creates a
        # feedback loop — a poorly-trained ctx distorts which pairs are "hard",
        # causing the model to learn ctx in the wrong direction (ctx_sim < 0 for
        # same-place pairs).  Mining on stable raw features gives a clean signal:
        # ctx is trained purely to improve upon what raw descriptors already do.
        logging.info(f"Mining triplets for epoch {self.epoch + 1}...")
        mining_start = time.perf_counter()

        mining_embeddings = graph.x.cpu().float().numpy()  # raw descriptor (precomputed, stable)

        triplets = triplet_miner.mine_triplets(
            descriptors=mining_embeddings,
            poses=poses,
            n_triplets_per_anchor=n_triplets_per_anchor,
            sequence_ids=sequence_ids,
            parallel=False  # Sequential to avoid fork memory issues
        )
        mining_time = time.perf_counter() - mining_start

        if len(triplets) == 0:
            logging.warning("No valid triplets mined!")
            return 0.0

        logging.info(f"Mined {len(triplets):,} triplets in {mining_time:.2f}s ({len(triplets)/mining_time:.0f} triplets/sec)")

        # Shuffle triplets and convert to numpy array for efficient indexing
        np.random.shuffle(triplets)
        triplets = np.array(triplets)

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
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except RuntimeError:
                pass  # Non-critical monitoring, skip if CUDA stats unavailable
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    embeddings = self.model(graph)
            else:
                embeddings = self.model(graph)
            fwd_peak = torch.cuda.max_memory_allocated(self.device) / 1e9
            logging.info(f"  VRAM after forward: {torch.cuda.memory_allocated()/1e9:.2f} GB, "
                         f"peak: {fwd_peak:.2f} GB")

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
                else:
                    anchors = embeddings[anchor_indices]
                    positives = embeddings[positive_indices]
                    negatives = embeddings[negative_indices]
                    batch_loss = self.criterion(anchors, positives, negatives)

                total_loss = total_loss + batch_loss / n_in_window
                epoch_losses.append(batch_loss.item())
                self.global_step += 1

                # False negative diagnostic: first batch of each epoch
                if batch_idx == 0:
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
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except RuntimeError:
                pass
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            bwd_peak = torch.cuda.max_memory_allocated(self.device) / 1e9
            logging.info(f"  VRAM after backward: peak {bwd_peak:.2f} GB")
            self.optimizer.zero_grad()

            del embeddings, total_loss
            torch.cuda.empty_cache()

        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(
        self,
        val_graph: Data,
        val_poses: np.ndarray,
        distance_threshold: float = 5.0,
        skip_frames: int = 30
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

        with torch.no_grad():
            val_graph = val_graph.to(self.device)
            embeddings = self.model(val_graph)
            embeddings_np = embeddings.cpu().numpy()
            raw_descriptors_np = val_graph.x.cpu().numpy()

        # GNN output: R@1, R@5, R@10 in one pass (single KD-Tree + FAISS build)
        recalls, n_queries = self._compute_recall_multi_k(
            embeddings_np, val_poses,
            k_values=[1, 5, 10],
            distance_threshold=distance_threshold,
            skip_frames=skip_frames
        )

        # Raw baseline: R@1 only in one pass
        raw_recalls, _ = self._compute_recall_multi_k(
            raw_descriptors_np, val_poses,
            k_values=[1],
            distance_threshold=distance_threshold,
            skip_frames=skip_frames
        )

        # GNN context only: slice ctx_norm from cat(raw_norm, ctx_norm)
        ctx_only_np = embeddings_np[:, raw_descriptors_np.shape[1]:]
        ctx_recalls, _ = self._compute_recall_multi_k(
            ctx_only_np, val_poses,
            k_values=[1],
            distance_threshold=distance_threshold,
            skip_frames=skip_frames
        )

        return {
            'recall@1': recalls[1],
            'recall@5': recalls[5],
            'recall@10': recalls[10],
            'raw_recall@1': raw_recalls[1],
            'ctx_recall@1': ctx_recalls[1],
            'n_queries': n_queries
        }

    def validate_all(
        self,
        val_datasets: Dict[str, Dict],
        distance_threshold: float = 5.0,
        skip_frames: int = 30
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
        total_queries = 0

        logging.info("Validation (per-dataset):")
        for dataset_name, info in val_datasets.items():
            metrics = self.validate(
                info['graph'],
                info['poses'],
                distance_threshold=distance_threshold,
                skip_frames=skip_frames
            )
            all_metrics[dataset_name] = metrics

            # Log per-dataset results
            logging.info(
                f"  {dataset_name:20s} | R@1: {metrics['recall@1']:.4f} (raw: {metrics['raw_recall@1']:.4f}, ctx: {metrics['ctx_recall@1']:.4f}) | "
                f"R@5: {metrics['recall@5']:.4f} | Queries: {metrics['n_queries']}"
            )

            # Accumulate for weighted average
            total_correct_at_1 += metrics['recall@1'] * metrics['n_queries']
            total_raw_correct_at_1 += metrics['raw_recall@1'] * metrics['n_queries']
            total_ctx_correct_at_1 += metrics['ctx_recall@1'] * metrics['n_queries']
            total_queries += metrics['n_queries']

        # Compute weighted average
        if total_queries > 0:
            avg_recall_at_1 = total_correct_at_1 / total_queries
            avg_raw_recall_at_1 = total_raw_correct_at_1 / total_queries
            avg_ctx_recall_at_1 = total_ctx_correct_at_1 / total_queries
        else:
            avg_recall_at_1 = 0.0
            avg_raw_recall_at_1 = 0.0
            avg_ctx_recall_at_1 = 0.0

        all_metrics['_average'] = {
            'recall@1': avg_recall_at_1,
            'raw_recall@1': avg_raw_recall_at_1,
            'ctx_recall@1': avg_ctx_recall_at_1,
            'n_queries': total_queries
        }
        logging.info(
            f"  {'AVERAGE':20s} | R@1: {avg_recall_at_1:.4f} (raw: {avg_raw_recall_at_1:.4f}, ctx: {avg_ctx_recall_at_1:.4f}) | Total Queries: {total_queries}"
        )

        return all_metrics

    def _compute_recall_multi_k(
        self,
        embeddings: np.ndarray,
        poses: np.ndarray,
        k_values: List[int],
        distance_threshold: float,
        skip_frames: int = 30
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

        # Evaluate each query
        correct_at_k = {k: 0 for k in k_values}

        for query_idx, true_match_idx in queries:
            # Search for nearest embeddings
            query_emb = embeddings_f32[query_idx:query_idx+1]
            distances, indices = faiss_index.search(query_emb, search_k)

            # Filter out temporal neighbors (vectorized)
            valid_mask = np.abs(indices[0] - query_idx) > skip_frames
            valid_indices = indices[0][valid_mask]

            if len(valid_indices) == 0:
                continue

            # Take top-max_k valid candidates and compute geo distances once
            valid_indices_max = valid_indices[:max_k]
            geo_dists = np.linalg.norm(positions[valid_indices_max] - positions[query_idx], axis=1)

            # Check each k value
            for k in k_values:
                if np.any(geo_dists[:k] < distance_threshold):
                    correct_at_k[k] += 1

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
        triplet_miner: Optional[TripletMiner] = None
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

        total_training_start = time.perf_counter()

        for epoch in range(n_epochs):
            self.epoch = epoch
            epoch_start = time.perf_counter()

            # Train
            avg_loss = self.train_epoch(
                train_graph,
                triplet_miner,
                train_poses,
                train_descriptors,
                sequence_ids=train_sequence_ids
            )
            train_time = time.perf_counter() - epoch_start

            # Validate
            val_start = time.perf_counter()
            if val_datasets is not None and len(val_datasets) > 0:
                all_metrics = self.validate_all(val_datasets)
                avg_metrics = all_metrics['_average']
                self.val_metrics.append(all_metrics)
                val_time = time.perf_counter() - val_start

                epoch_total = time.perf_counter() - epoch_start
                logging.info(
                    f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f} | "
                    f"Avg R@1: {avg_metrics['recall@1']:.4f} | "
                    f"Time: {epoch_total:.1f}s (train={train_time:.1f}s, val={val_time:.1f}s)"
                )

                # Save best model (based on average R@1 across all datasets)
                if avg_metrics['recall@1'] > self.best_val_metric:
                    self.best_val_metric = avg_metrics['recall@1']
                    self.save_checkpoint('best_model.pth')
                    logging.info(f"  -> New best model! Avg R@1: {self.best_val_metric:.4f}")
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    logging.info(f"  -> No improvement for {self.epochs_without_improvement} epoch(s)")

                # Early stopping
                if self.epochs_without_improvement >= self.patience:
                    logging.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                    logging.info(f"Best validation Avg R@1: {self.best_val_metric:.4f}")
                    break
            else:
                epoch_total = time.perf_counter() - epoch_start
                logging.info(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f} | Time: {epoch_total:.1f}s")

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
        logging.info(f"Training complete! Total time: {total_time/3600:.2f}h | Best R@1: {self.best_val_metric:.4f}")

    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
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

        checkpoint = torch.load(load_path, map_location=self.device)

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
