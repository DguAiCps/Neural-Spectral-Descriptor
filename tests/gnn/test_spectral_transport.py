"""복소 전달의 회전 부호·좌표 불변성·관측 범위와 기울기를 확인한다."""
import numpy as np
import torch

from gnn.model import create_spectral_gnn
from gnn.spectral_transport import (
    SpectralTransportGNN, attach_transport, complex_unit_rms, lidar_yaw, spectral_mean,
)


def graph(raw, z, yaw, sequence=None):
    from torch_geometric.data import Data
    n = len(raw)
    seq = np.zeros(n, int) if sequence is None else sequence
    edges = [(j, i) for i in range(n) for j in range(max(0, i-5), min(n, i+6))
             if i != j and seq[j] == seq[i]]
    g = Data(x=torch.tensor(raw, dtype=torch.float32),
             edge_index=torch.tensor(edges, dtype=torch.long).T,
             edge_type=torch.zeros(len(edges), dtype=torch.long))
    attach_transport(g, z, yaw, seq)
    return g


def test_fft_rotation_sign_and_camera_frame():
    rng = np.random.default_rng(17)
    world = rng.uniform(size=(16, 60))
    shift = np.array([0, 2, -4, 9, 5, -7, 3])
    # 센서가 +yaw 회전하면 같은 세계의 영상은 -shift 이동한다.
    z = np.stack([np.fft.rfft(np.roll(world, -int(s), axis=1), axis=1)[:, 1:9]
                  for s in shift])
    yaw = shift * 2*np.pi/60
    g = graph(np.ones((len(z), 288)), z, yaw)
    aligned = spectral_mean(g.spectrum, g.transport_indices, g.transport_mask,
                            g.transport_phase, "transported")
    torch.testing.assert_close(aligned, g.spectrum, atol=1e-5, rtol=2e-6)
    unaligned = spectral_mean(g.spectrum, g.transport_indices, g.transport_mask,
                              g.transport_phase, "untransported")
    assert (unaligned-g.spectrum).abs().mean() > 1.
    r = np.zeros((len(yaw), 4, 4))
    r[:, 3, 3] = r[:, 2, 2] = 1.
    r[:, 0, 0] = r[:, 1, 1] = np.cos(yaw)
    r[:, 1, 0], r[:, 0, 1] = np.sin(yaw), -np.sin(yaw)
    extrinsic = np.array([[0., -1, 0], [0, 0, -1], [1, 0, 0]])
    camera = r.copy()
    camera[:, :3, :3] = extrinsic @ r[:, :3, :3] @ extrinsic.T
    np.testing.assert_allclose(lidar_yaw(camera, extrinsic), yaw, atol=1e-14)


def test_sequence_boundary_zero_and_isolation():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(13, 16, 8)).astype(np.complex64)
    g = graph(np.ones((13, 288)), z, np.zeros(13), np.r_[np.zeros(5), np.ones(7), 2])
    a = spectral_mean(g.spectrum, g.transport_indices, g.transport_mask,
                      g.transport_phase, "transported")
    for i in range(13):
        ids = g.edge_index[0, g.edge_index[1] == i]
        expected = g.spectrum[ids].mean(0) if len(ids) else torch.zeros_like(a[i])
        torch.testing.assert_close(a[i], expected)
    z0 = torch.zeros_like(g.spectrum, requires_grad=True)
    complex_unit_rms(z0).abs().sum().backward()
    assert torch.isfinite(z0.grad).all()


def test_model_local_gauge_reversal_and_prefix():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    ref = create_spectral_gnn(input_dim=288, hidden_dim=128, context_dim=128,
                             n_layers=2, use_residual_gate=True, gate_initial_alpha=.0625)
    ref.gnn = SpectralTransportGNN(ref.gnn, "transported")
    assert sum(p.numel() for p in ref.parameters()) == 167469
    rng = np.random.default_rng(4)
    raw = rng.uniform(size=(52, 288)).astype(np.float32)
    z = (rng.normal(size=(52, 16, 8)) + 1j*rng.normal(size=(52, 16, 8))).astype(np.complex64)
    yaw, change = rng.uniform(-3, 3, (2, 52))
    phased = z * np.exp(1j*change[:, None, None]*np.arange(1, 9))
    ref.eval()
    full = ref(graph(raw, z, yaw))
    moved = ref(graph(raw, phased, yaw+change))
    reverse = ref(graph(raw[::-1].copy(), z[::-1], yaw[::-1])).flip(0)
    prefix = ref(graph(raw[:36], z[:36], yaw[:36]))
    torch.testing.assert_close(full, moved, atol=3e-7, rtol=3e-5)
    torch.testing.assert_close(full, reverse, atol=3e-7, rtol=3e-5)
    torch.testing.assert_close(full[20], prefix[20], atol=3e-7, rtol=3e-5)
    assert full.shape == (52, 416) and torch.isfinite(full).all()
    ref.train()
    ref(graph(raw, z, yaw))[:, 288:].square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in ref.parameters())


def test_local_control_ignores_neighbor_spectrum():
    torch.manual_seed(8)
    ref = create_spectral_gnn(input_dim=288, hidden_dim=128, context_dim=128,
                             n_layers=2, use_residual_gate=True, gate_initial_alpha=.0625)
    ref.gnn = SpectralTransportGNN(ref.gnn, "spectral_local")
    ref.eval()
    rng = np.random.default_rng(15)
    raw = rng.uniform(size=(20, 288)).astype(np.float32)
    z = rng.normal(size=(20, 16, 8)).astype(np.complex64)
    a = ref(graph(raw, z, np.zeros(20)))
    z[:10] *= -1j
    z[11:] *= 10j
    b = ref(graph(raw, z, np.zeros(20)))
    torch.testing.assert_close(a[10], b[10], atol=0, rtol=0)
