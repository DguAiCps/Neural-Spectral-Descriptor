"""앞뒤 거리 차분의 기하·반전·입력 범위·학습 가능성을 확인한다."""
import numpy as np
import torch

from gnn.temporal_derivatives import (
    TemporalDerivativeGNN, attach_derivatives, derivative_plan, temporal_differences,
)
from gnn.model import create_spectral_gnn


def graph(positions, raw=None, seq=None):
    from torch_geometric.data import Data
    n = len(positions)
    seq = np.zeros(n, int) if seq is None else seq
    edges = [(j, i) for i in range(n) for j in range(max(0, i-5), min(n, i+6))
             if i != j and seq[i] == seq[j]]
    data = Data(x=torch.tensor(raw if raw is not None else positions, dtype=torch.float32),
                edge_index=torch.tensor(edges, dtype=torch.long).T,
                edge_type=torch.zeros(len(edges), dtype=torch.long))
    plan = attach_derivatives(data, positions, seq)
    return data, plan


def test_linear_quadratic_constant_and_distances():
    pos = np.column_stack((np.arange(21)*.8, np.zeros((21, 2))))
    p = derivative_plan(pos, np.zeros(21, int))
    x = torch.tensor(np.column_stack((pos[:, 0], pos[:, 0]**2, np.ones(21))),
                     dtype=torch.float64)
    y = temporal_differences(x, torch.tensor(p["indices"]), torch.tensor(p["coefficients"]))
    # 1.6/3.2m 모두 실제 관측 위치이므로 다항식 차분이 정확하다.
    for k in (0, 2):
        np.testing.assert_allclose(y[10, k], [1., 2*pos[10, 0], 0.], atol=1e-8)
        np.testing.assert_allclose(y[10, k+1], [0., 2., 0.], atol=1e-8)
    np.testing.assert_allclose(p["radii"][10], [1.6, 3.2])
    assert y.shape == (21, 4, 3) and torch.isfinite(y).all()


def test_reversal_stops_boundaries_and_sequence_separation():
    rng = np.random.default_rng(71)
    s = np.array([0., .8, .8, 1.6, 2.4, 2.4, 3.2, 4., 5.6, 6.4, 7.2])
    pos = np.column_stack((s, np.zeros((len(s), 2))))
    raw = torch.tensor(rng.normal(size=(len(s), 4)), dtype=torch.float64)
    a = derivative_plan(pos, np.zeros(len(s), int))
    b = derivative_plan(pos[::-1], np.zeros(len(s), int))
    ya = temporal_differences(raw, torch.tensor(a["indices"]), torch.tensor(a["coefficients"]))
    yb = temporal_differences(raw.flip(0), torch.tensor(b["indices"]), torch.tensor(b["coefficients"])).flip(0)
    torch.testing.assert_close(ya[:, [0, 2]], -yb[:, [0, 2]], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(ya[:, [1, 3]], yb[:, [1, 3]], atol=1e-7, rtol=1e-7)
    assert not a["valid"][[0, -1]].any()
    stop = derivative_plan(np.zeros((8, 3)), np.zeros(8, int))
    assert not stop["valid"].any() and not stop["coefficients"].any()
    seq = np.r_[np.zeros(5, int), np.ones(6, int)]
    split = derivative_plan(pos, seq)
    for i in range(len(pos)):
        used = np.any(split["coefficients"][i] != 0, axis=0)
        assert np.all(seq[split["indices"][i, used]] == seq[i])


def test_nonuniform_linear_interpolation():
    s = np.array([0., .7, 1.9, 2.8, 4.2, 5.4, 6.1, 8.2, 9.5, 11., 13.])
    pos = np.column_stack((s, np.zeros((len(s), 2))))
    p = derivative_plan(pos, np.zeros(len(s), int))
    y = temporal_differences(torch.tensor(s[:, None]), torch.tensor(p["indices"]),
                             torch.tensor(p["coefficients"]))
    for k in range(2):
        use = p["valid"][:, k]
        np.testing.assert_allclose(y[use, 2*k, 0], 1., atol=1e-6)
        np.testing.assert_allclose(y[use, 2*k+1, 0], 0., atol=1e-6)


def test_model_reversal_prefix_and_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    net = create_spectral_gnn(input_dim=288, hidden_dim=128, context_dim=128,
                             n_layers=2, n_heads=4, dropout=.1,
                             use_residual_gate=True, gate_initial_alpha=.0625)
    net.gnn = TemporalDerivativeGNN(net.gnn)
    assert sum(p.numel() for p in net.parameters()) == 167433
    n = 50
    pos = np.column_stack((np.arange(n)*.8, np.zeros((n, 2))))
    raw = np.random.default_rng(8).normal(size=(n, 288)).astype(np.float32)
    full, _ = graph(pos, raw)
    reverse, _ = graph(pos[::-1], raw[::-1].copy())
    prefix, _ = graph(pos[:36], raw[:36])
    net.eval()
    a, b, c = net(full), net(reverse).flip(0), net(prefix)
    torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(a[20], c[20], atol=2e-7, rtol=2e-5)
    assert a.shape == (n, 416) and torch.isfinite(a).all()
    net.train()
    net(full)[:, 288:].square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in net.parameters())
