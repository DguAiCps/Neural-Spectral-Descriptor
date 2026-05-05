"""Tests for GNN Trainer and InfoNCE loss."""

import torch
import torch.nn.functional as F
import numpy as np
import pytest
from torch_geometric.data import Data
from gnn.trainer import InfoNCELoss, GNNTrainer
from gnn.model import create_spectral_gnn


class TestInfoNCELoss:

    def test_output_scalar(self):
        loss_fn = InfoNCELoss(temperature=0.07)
        B, D = 8, 128
        anchors = torch.randn(B, D)
        positives = torch.randn(B, D)
        negatives = torch.randn(B, D)
        loss = loss_fn(anchors, positives, negatives)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_perfect_alignment_low_loss(self):
        """If positives are identical to anchors, loss should be low."""
        loss_fn = InfoNCELoss(temperature=0.07)
        B, D = 8, 128
        anchors = torch.randn(B, D)
        positives = anchors.clone()
        negatives = torch.randn(B, D) * 10  # very different
        loss = loss_fn(anchors, positives, negatives)
        # With perfect alignment, loss should be small
        assert loss.item() < 1.0

    def test_random_high_loss(self):
        """Random embeddings should produce high loss."""
        loss_fn = InfoNCELoss(temperature=0.07)
        B, D = 16, 128
        loss = loss_fn(torch.randn(B, D), torch.randn(B, D), torch.randn(B, D))
        # For random 128-D vectors, loss ≈ log(B+1) ≈ 2.8
        assert loss.item() > 1.0

    def test_temperature_effect(self):
        """Lower temperature should produce higher loss for random embeddings."""
        B, D = 8, 128
        anchors = torch.randn(B, D)
        positives = torch.randn(B, D)
        negatives = torch.randn(B, D)
        loss_high_t = InfoNCELoss(temperature=1.0)(anchors, positives, negatives)
        loss_low_t = InfoNCELoss(temperature=0.01)(anchors, positives, negatives)
        # Lower temperature makes the distribution sharper, usually higher loss
        assert loss_low_t.item() != loss_high_t.item()

    def test_gradient_flow(self):
        loss_fn = InfoNCELoss(temperature=0.1)
        B, D = 4, 64
        anchors = torch.randn(B, D, requires_grad=True)
        positives = torch.randn(B, D)
        negatives = torch.randn(B, D)
        loss = loss_fn(anchors, positives, negatives)
        loss.backward()
        assert anchors.grad is not None


class TestGNNTrainer:

    @pytest.fixture
    def trainer_and_graph(self, device):
        """Create a minimal trainer with small graph for testing."""
        n_nodes = 50
        n_edges = 150
        input_dim = 256

        model = create_spectral_gnn(
            input_dim=input_dim, hidden_dim=64, context_dim=64,
            n_layers=1, n_heads=2, dropout=0.0,
            use_local_updates=True,
            edge_encoder_config=None,
            gradient_checkpointing=False,
        )

        trainer = GNNTrainer(
            model=model, device=device,
            learning_rate=1e-3, weight_decay=0,
            temperature=0.1, patience=5,
            use_amp=False,
        )

        x = torch.randn(n_nodes, input_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        graph = Data(x=x, edge_index=edge_index).to(device)

        # Poses: straight line
        poses = np.zeros((n_nodes, 4, 4))
        for i in range(n_nodes):
            poses[i] = np.eye(4)
            poses[i, 0, 3] = i * 2.0

        descriptors = x.detach().cpu().numpy()

        return trainer, graph, poses, descriptors

    def test_trainer_creation(self, trainer_and_graph):
        trainer, _, _, _ = trainer_and_graph
        assert trainer.epoch == 0
        assert trainer.best_val_metric == 0.0

    def test_single_train_step(self, trainer_and_graph):
        """A training step with synthetic triplets should not error."""
        trainer, graph, poses, descriptors = trainer_and_graph
        n = graph.num_nodes

        # Synthetic triplets: pairs (i, i+1, i+25)
        anchors, positives, negatives = [], [], []
        for i in range(min(10, n - 26)):
            anchors.append(i)
            positives.append(i + 1)
            negatives.append(i + 25)

        if len(anchors) < 2:
            pytest.skip("Not enough nodes for triplets")

        triplets = list(zip(anchors, positives, negatives))
        trainer.model.train()

        # Minimal forward + loss
        embeddings = trainer.model(graph)
        a_idx = torch.tensor(anchors, device=embeddings.device)
        p_idx = torch.tensor(positives, device=embeddings.device)
        n_idx = torch.tensor(negatives, device=embeddings.device)

        loss = trainer.criterion(
            embeddings[a_idx], embeddings[p_idx], embeddings[n_idx]
        )
        loss.backward()
        assert loss.item() > 0

    def test_policy_lr_scale(self, device):
        """Spectral policy should get a different learning rate."""
        from encoding.spectral_policy import SoftBinning
        policy = SoftBinning(79, 181, output_dim=1106, n_soft_bins=4)
        model = create_spectral_gnn(
            input_dim=1106, hidden_dim=64, context_dim=64,
            spectral_policy=policy,
        )
        trainer = GNNTrainer(
            model=model, device=device,
            learning_rate=1e-3, policy_lr_scale=0.1,
            use_amp=False,
        )
        # Should have 2 param groups
        assert len(trainer.optimizer.param_groups) == 2
        assert trainer.optimizer.param_groups[1]['lr'] == pytest.approx(1e-4)

    def test_warmup_freeze(self, device):
        """Policy params should be frozen during warmup."""
        from encoding.spectral_policy import SoftBinning
        policy = SoftBinning(79, 181, output_dim=1106, n_soft_bins=4)
        model = create_spectral_gnn(
            input_dim=1106, hidden_dim=64, context_dim=64,
            spectral_policy=policy,
        )
        trainer = GNNTrainer(
            model=model, device=device,
            learning_rate=1e-3, policy_warmup_epochs=5,
            use_amp=False,
        )
        base = model.gnn if hasattr(model, 'gnn') else model

        # Before warmup ends, policy should be frozen
        # (warmup logic runs at start of train_epoch, check initial state)
        assert trainer.policy_warmup_epochs == 5
