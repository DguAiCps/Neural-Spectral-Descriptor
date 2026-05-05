"""Integration tests: end-to-end pipeline from point cloud to GNN output."""

import numpy as np
import torch
import pytest


def _make_pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = np.cos(yaw); T[0, 1] = -np.sin(yaw)
    T[1, 0] = np.sin(yaw); T[1, 1] = np.cos(yaw)
    T[0, 3] = x; T[1, 3] = y; T[2, 3] = z
    return T


class TestEndToEndBEV:
    """Full pipeline: Point Cloud → BEV → FFT → Binning → GNN."""

    @pytest.fixture
    def encoder(self, device):
        from encoding.spectral_encoder import SpectralEncoder
        return SpectralEncoder(
            n_elevation=16, n_azimuth=360, n_bins=4, alpha=2.0,
            learnable_alpha=True, target_elevation_bins=16,
            elevation_range=(-24.8, 2.0),
            bin_statistics=['mean', 'std'],
            inter_bin_statistics=['diff'],
            projection_type='bev',
            max_range=80.0, min_range=1.0,
            z_min=-3.0, height_encoding='iris',
            n_height_layers=8, z_max=5.0,
            binning_strategy='octave',
            normalize_channels=False,
        ).to(device)

    def test_encode_to_gnn(self, encoder, random_point_cloud, device):
        """Encoder → descriptor → GNN forward."""
        from gnn.model import create_spectral_gnn
        from torch_geometric.data import Data

        # Encode multiple point clouds
        n_kf = 10
        descriptors = []
        for _ in range(n_kf):
            desc = encoder.encode_points(random_point_cloud).detach().cpu().numpy()
            descriptors.append(desc)
        descriptors = np.array(descriptors)

        # Build simple graph
        x = torch.tensor(descriptors, dtype=torch.float32).to(device)
        edges = []
        for i in range(n_kf - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)

        data = Data(x=x, edge_index=edge_index)

        # GNN forward
        model = create_spectral_gnn(
            input_dim=encoder.output_dim, hidden_dim=64, context_dim=64,
            n_layers=1, n_heads=2,
            edge_encoder_config=None,
            gradient_checkpointing=False,
        ).to(device)
        model.eval()

        out = model(data)
        assert out.shape == (n_kf, encoder.output_dim + 64)
        assert torch.all(torch.isfinite(out))

    def test_fft_magnitudes_to_policy_to_gnn(self, encoder, random_point_cloud, device):
        """FFT magnitudes → Policy → GNN (end-to-end with gradient)."""
        from encoding.spectral_policy import SoftBinning
        from gnn.model import SpectralGNN
        from torch_geometric.data import Data

        n_kf = 8
        fft_mags = []
        descriptors = []
        for _ in range(n_kf):
            fft = encoder.compute_fft_magnitudes(random_point_cloud)
            desc = encoder.encode_points(random_point_cloud).detach().cpu().numpy()
            fft_mags.append(fft)
            descriptors.append(desc)

        fft_mags = np.array(fft_mags)
        descriptors = np.array(descriptors)
        n_rings, n_freqs = fft_mags[0].shape

        # Create policy
        policy = SoftBinning(
            n_rings, n_freqs, output_dim=encoder.output_dim,
            n_soft_bins=4, init_from_fixed=True,
        )

        # Create GNN with policy
        model = SpectralGNN(
            input_dim=encoder.output_dim, hidden_dim=64, context_dim=64,
            n_layers=1, n_heads=2,
            spectral_policy=policy,
        ).to(device)

        # Build graph
        x = torch.tensor(descriptors, dtype=torch.float32).to(device)
        x_fft = torch.tensor(fft_mags.reshape(n_kf, -1), dtype=torch.float32).to(device)
        edges = []
        for i in range(n_kf - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)
        data = Data(x=x, x_fft=x_fft, edge_index=edge_index)

        # Forward with gradient
        model.train()
        out = model(data)
        assert out.shape == (n_kf, encoder.output_dim + 64)

        # Gradient should flow to policy parameters
        loss = out.sum()
        loss.backward()
        assert policy.centers.grad is not None
        assert policy.log_widths.grad is not None


class TestEndToEndInfoNCE:
    """End-to-end: encoder → GNN → InfoNCE loss."""

    def test_training_step(self, random_point_cloud, device):
        from encoding.spectral_encoder import SpectralEncoder
        from gnn.model import create_spectral_gnn
        from gnn.trainer import InfoNCELoss
        from torch_geometric.data import Data

        encoder = SpectralEncoder(
            n_elevation=16, n_azimuth=360, n_bins=4, alpha=2.0,
            learnable_alpha=True, target_elevation_bins=16,
            elevation_range=(-24.8, 2.0),
            bin_statistics=['mean', 'std'],
            inter_bin_statistics=['diff'],
            projection_type='bev',
            max_range=80.0, min_range=1.0,
            z_min=-3.0, height_encoding='iris',
            n_height_layers=8, z_max=5.0,
            binning_strategy='octave',
            normalize_channels=False,
        ).to(device)

        # Encode
        n_kf = 20
        descs = []
        for _ in range(n_kf):
            d = encoder.encode_points(random_point_cloud).detach().cpu().numpy()
            descs.append(d)
        descs = np.array(descs)

        # Graph
        x = torch.tensor(descs, dtype=torch.float32).to(device)
        edges = []
        for i in range(n_kf - 1):
            edges.extend([[i, i + 1], [i + 1, i]])
        edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)
        data = Data(x=x, edge_index=edge_index)

        # Model
        model = create_spectral_gnn(
            input_dim=encoder.output_dim, hidden_dim=64, context_dim=64,
            n_layers=1, n_heads=2,
            edge_encoder_config=None,
            gradient_checkpointing=False,
        ).to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = InfoNCELoss(temperature=0.1)

        # Training step
        embeddings = model(data)
        B = 4
        anchors = embeddings[:B]
        positives = embeddings[1:B + 1]
        negatives = embeddings[n_kf - B:]

        loss = loss_fn(anchors, positives, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert torch.all(torch.isfinite(embeddings))


class TestCacheRoundtrip:
    """Test keyframe cache save/load preserves FFT magnitudes."""

    def test_save_load_with_fft(self, random_point_cloud, tmp_path):
        from keyframe.selector import Keyframe

        # Create keyframes with FFT magnitudes
        kfs = []
        for i in range(5):
            kf = Keyframe(
                keyframe_id=i, scan_id=i,
                points=np.empty((0, 3)),
                pose=_make_pose(x=i * 5.0),
                timestamp=float(i),
                descriptor=np.random.randn(1106).astype(np.float32),
                spectral_entropy=float(np.random.rand()),
                fft_magnitudes=np.random.randn(79, 181).astype(np.float32),
            )
            kfs.append(kf)

        # Import save/load from training script
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from train_multi_dataset import save_keyframes_cache, load_keyframes_cache

        cache_path = tmp_path / "test_cache.npz"
        save_keyframes_cache(cache_path, kfs)
        loaded = load_keyframes_cache(cache_path)

        assert len(loaded) == 5
        for orig, load in zip(kfs, loaded):
            np.testing.assert_array_almost_equal(orig.descriptor, load.descriptor)
            assert load.fft_magnitudes is not None
            np.testing.assert_array_almost_equal(orig.fft_magnitudes, load.fft_magnitudes)
            assert load.spectral_entropy == pytest.approx(orig.spectral_entropy, abs=1e-5)

    def test_save_load_without_fft(self, random_point_cloud, tmp_path):
        """Cache without FFT magnitudes (backward compat)."""
        from keyframe.selector import Keyframe
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from train_multi_dataset import save_keyframes_cache, load_keyframes_cache

        kfs = []
        for i in range(3):
            kf = Keyframe(
                keyframe_id=i, scan_id=i,
                points=np.empty((0, 3)),
                pose=_make_pose(x=i * 5.0),
                timestamp=float(i),
                descriptor=np.random.randn(1106).astype(np.float32),
            )
            kfs.append(kf)

        cache_path = tmp_path / "test_cache_no_fft.npz"
        save_keyframes_cache(cache_path, kfs)
        loaded = load_keyframes_cache(cache_path)

        assert len(loaded) == 3
        for load in loaded:
            assert load.fft_magnitudes is None
