"""
Shared fixtures for Neural Spectral Codec tests.

All tests use synthetic data — no real dataset files required.
"""

import sys
import os
import numpy as np
import torch
import pytest

# Mirror the sys.path hack from train_multi_dataset.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Deterministic seeds
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def seed_everything():
    """Fix random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Synthetic point clouds
# ---------------------------------------------------------------------------

@pytest.fixture
def random_point_cloud():
    """(N, 3) random points in LiDAR-like range."""
    n_points = 5000
    r = np.random.uniform(2.0, 70.0, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    z = np.random.uniform(-2.0, 4.0, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


@pytest.fixture
def random_point_cloud_4d():
    """(N, 4) random points with intensity."""
    n_points = 5000
    r = np.random.uniform(2.0, 70.0, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    z = np.random.uniform(-2.0, 4.0, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    intensity = np.random.uniform(0, 1, n_points)
    return np.stack([x, y, z, intensity], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# SE(3) poses
# ---------------------------------------------------------------------------

def _make_pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    """Create a SE(3) pose from x, y, z, yaw (rad)."""
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = np.cos(yaw)
    T[0, 1] = -np.sin(yaw)
    T[1, 0] = np.sin(yaw)
    T[1, 1] = np.cos(yaw)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


@pytest.fixture
def identity_pose():
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def trajectory_poses():
    """20 poses along a straight line, 2m apart, 10Hz."""
    return np.array([_make_pose(x=i * 2.0) for i in range(20)])


@pytest.fixture
def trajectory_timestamps():
    """Timestamps at 10Hz for 20 poses."""
    return np.arange(20) * 0.1


@pytest.fixture
def loop_trajectory_poses():
    """50 poses in a loop — returns near start at the end."""
    poses = []
    for i in range(50):
        angle = 2 * np.pi * i / 50
        x = 20 * np.cos(angle)
        y = 20 * np.sin(angle)
        poses.append(_make_pose(x=x, y=y, yaw=angle + np.pi / 2))
    return np.array(poses)


@pytest.fixture
def loop_trajectory_timestamps():
    return np.arange(50) * 0.5


# ---------------------------------------------------------------------------
# Encoding config helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def bev_encoding_config():
    """Minimal BEV encoding config matching production defaults."""
    return {
        'projection_type': 'bev',
        'n_elevation': 16,
        'n_azimuth': 360,
        'n_bins': 4,
        'alpha': 2.0,
        'learnable_alpha': True,
        'target_elevation_bins': 16,
        'elevation_range': [-24.8, 2.0],
        'max_range': 80.0,
        'min_range': 1.0,
        'bev': {
            'z_min': -3.0,
            'height_encoding': 'iris',
            'n_height_layers': 8,
            'z_max': 5.0,
        },
        'bin_statistics': ['mean', 'std'],
        'inter_bin_statistics': ['diff'],
        'binning_strategy': 'octave',
        'zero_center': False,
        'log_magnitude': False,
        'normalize_channels': False,
    }


@pytest.fixture
def gnn_config():
    """GNN config matching production defaults."""
    return {
        'input_dim': 1106,
        'hidden_dim': 128,
        'context_dim': 256,
        'n_layers': 2,
        'n_heads': 4,
        'dropout': 0.1,
        'edge_encoding': {
            'd_edge': 32,
            'n_edge_types': 2,
            'd_type_embed': 16,
            'd_rot_encode': 16,
            'dropout': 0.1,
        },
    }


# ---------------------------------------------------------------------------
# Small graph fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def small_graph(device):
    """A tiny PyG Data object for GNN tests."""
    from torch_geometric.data import Data

    n_nodes = 20
    n_edges = 60
    input_dim = 1106

    x = torch.randn(n_nodes, input_dim)
    edge_index = torch.stack([
        torch.randint(0, n_nodes, (n_edges,)),
        torch.randint(0, n_nodes, (n_edges,)),
    ])
    edge_attr = torch.randn(n_edges, 5)
    edge_attr[:, 0] = torch.abs(edge_attr[:, 0])  # dist >= 0
    edge_attr[:, 1] = torch.abs(edge_attr[:, 1])  # rot >= 0
    edge_attr[:, 2] = torch.clamp(edge_attr[:, 2], -1, 1)  # cos_sim
    edge_attr[:, 3] = torch.abs(edge_attr[:, 3])  # l2_dist
    edge_attr[:, 4] = torch.clamp(edge_attr[:, 4], 0, 1)  # posterior
    edge_type = torch.randint(0, 2, (n_edges,))

    data = Data(
        x=x, edge_index=edge_index,
        edge_attr=edge_attr, edge_type=edge_type,
    ).to(device)
    return data
