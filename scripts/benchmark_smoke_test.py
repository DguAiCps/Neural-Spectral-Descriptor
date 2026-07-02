#!/usr/bin/env python3
"""
Benchmark Smoke Test — 실제 데이터 없이 합성 데이터로 전체 파이프라인 검증.

각 policy type에 대해:
  1. 합성 keyframes + FFT magnitudes 생성
  2. Policy 생성 → GNN 모델 생성
  3. Graph 빌드
  4. 1 epoch 학습 (InfoNCE)
  5. Validation (R@1 계산)
  6. Gradient flow 확인 (policy params에 grad 존재)

모든 policy가 에러 없이 학습/검증되는지 빠르게 확인.

Usage:
  python scripts/benchmark_smoke_test.py
  python scripts/benchmark_smoke_test.py --device cpu
  python scripts/benchmark_smoke_test.py --policies soft_binning linear
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import time
import numpy as np
import torch
from torch_geometric.data import Data

POLICY_TYPES = ['soft_binning', 'linear', 'conv1d', 'attention', 'gated']

# Synthetic data dimensions (matching BEV mode)
N_RINGS = 79
N_FREQS = 181
N_KEYFRAMES = 50
DESC_DIM = 1106
N_EPOCHS = 2


def make_synthetic_data(n_kf: int, device: str):
    """Create synthetic graph data with FFT magnitudes."""
    # Descriptors (fixed binning output)
    descriptors = np.random.randn(n_kf, DESC_DIM).astype(np.float32)

    # FFT magnitudes
    fft_mags = np.abs(np.random.randn(n_kf, N_RINGS, N_FREQS)).astype(np.float32)

    # Poses: straight line with loop
    poses = np.zeros((n_kf, 4, 4), dtype=np.float64)
    for i in range(n_kf):
        poses[i] = np.eye(4)
        angle = 2 * np.pi * i / n_kf
        poses[i, 0, 3] = 20 * np.cos(angle)  # circular trajectory
        poses[i, 1, 3] = 20 * np.sin(angle)

    # Build graph
    x = torch.tensor(descriptors, dtype=torch.float32).to(device)
    x_fft = torch.tensor(fft_mags.reshape(n_kf, -1), dtype=torch.float32).to(device)

    # Temporal edges (k=5 window)
    edges = []
    for i in range(n_kf):
        for j in range(max(0, i - 5), min(n_kf, i + 6)):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)

    # Edge attributes: [dist, rot, cos_sim, l2_dist, posterior]
    n_edges = edge_index.shape[1]
    edge_attr = torch.rand(n_edges, 5, device=device)
    edge_type = torch.zeros(n_edges, dtype=torch.long, device=device)

    graph = Data(
        x=x, x_fft=x_fft, edge_index=edge_index,
        edge_attr=edge_attr, edge_type=edge_type,
    )

    return graph, poses, descriptors


def make_policy(policy_type: str, output_dim: int = DESC_DIM):
    """Create a spectral policy."""
    from encoding.spectral_policy import create_spectral_policy

    config = {
        'enabled': True,
        'type': policy_type,
        'output_dim': output_dim,
        'shared_across_rings': True,
        'soft_binning': {
            'n_soft_bins': 4,
            'stats': ['mean', 'std'],
            'inter_stats': ['diff'],
            'init_from_fixed': True,
            'alpha': 2.0,
        },
        'linear': {'d_per_ring': 14},
        'conv1d': {'channels_per_group': 2, 'kernel_size': 7},
        'attention': {'n_queries': 7, 'n_heads': 2, 'head_dim': 32, 'd_pe': 16},
        'gated': {'gate_hidden': 64},
    }

    return create_spectral_policy(config, n_rings=N_RINGS, n_freqs=N_FREQS)


def run_smoke_test(policy_type: str, device: str) -> dict:
    """Run full smoke test for a single policy type."""
    from gnn.model import create_spectral_gnn
    from gnn.trainer import InfoNCELoss

    result = {
        'policy_type': policy_type,
        'status': 'pending',
    }

    try:
        t0 = time.perf_counter()

        # 1. Create data
        graph, poses, descriptors = make_synthetic_data(N_KEYFRAMES, device)

        # 2. Create policy + model
        policy = make_policy(policy_type)
        n_policy_params = sum(p.numel() for p in policy.parameters())
        result['policy_params'] = n_policy_params

        model = create_spectral_gnn(
            input_dim=DESC_DIM,
            hidden_dim=64,
            context_dim=64,
            n_layers=1,
            n_heads=2,
            edge_encoder_config=None,
            spectral_policy=policy,
        ).to(device)

        n_total_params = sum(p.numel() for p in model.parameters())
        result['total_params'] = n_total_params

        base_model = model.gnn if hasattr(model, 'gnn') else model
        result['effective_input_dim'] = base_model.input_dim

        # 3. Forward pass
        model.train()
        out = model(graph)
        result['output_shape'] = list(out.shape)
        assert out.shape[0] == N_KEYFRAMES
        assert torch.all(torch.isfinite(out)), "Non-finite output"

        # 4. Loss + backward
        loss_fn = InfoNCELoss(temperature=0.1)
        B = 8
        anchors = out[:B]
        positives = out[1:B + 1]
        negatives = out[N_KEYFRAMES - B:]

        loss = loss_fn(anchors, positives, negatives)
        loss.backward()

        result['loss'] = loss.item()
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss is not finite"

        # 5. Check gradient flow to policy
        has_policy_grad = False
        for name, p in policy.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_policy_grad = True
                break
        result['policy_gradient_flow'] = has_policy_grad

        # 6. Optimizer step
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.step()
        optimizer.zero_grad()

        # 7. Validation forward
        model.eval()
        with torch.no_grad():
            val_out = model(graph)
            assert torch.all(torch.isfinite(val_out)), "Non-finite validation output"

            # Policy-only output
            x_fft = graph.x_fft.view(-1, N_RINGS, N_FREQS)
            policy_out = base_model.spectral_policy(x_fft)
            assert torch.all(torch.isfinite(policy_out)), "Non-finite policy output"
            result['policy_output_dim'] = policy_out.shape[1]

        elapsed = time.perf_counter() - t0
        result['elapsed'] = elapsed
        result['status'] = 'PASS'

    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()

    return result


def run_baseline_smoke_test(device: str) -> dict:
    """Smoke test for baseline (no policy)."""
    from gnn.model import create_spectral_gnn
    from gnn.trainer import InfoNCELoss

    result = {'policy_type': 'baseline (no policy)', 'status': 'pending'}

    try:
        t0 = time.perf_counter()

        graph, poses, descriptors = make_synthetic_data(N_KEYFRAMES, device)

        model = create_spectral_gnn(
            input_dim=DESC_DIM,
            hidden_dim=64,
            context_dim=64,
            n_layers=1,
            n_heads=2,
            edge_encoder_config=None,
            spectral_policy=None,
        ).to(device)

        result['total_params'] = sum(p.numel() for p in model.parameters())

        model.train()
        out = model(graph)
        result['output_shape'] = list(out.shape)

        loss_fn = InfoNCELoss(temperature=0.1)
        B = 8
        loss = loss_fn(out[:B], out[1:B+1], out[N_KEYFRAMES-B:])
        loss.backward()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.step()

        result['loss'] = loss.item()
        result['elapsed'] = time.perf_counter() - t0
        result['status'] = 'PASS'

    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description='Benchmark Smoke Test')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--policies', nargs='+', default=None,
                        help='Specific policies to test')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    policies_to_test = args.policies or POLICY_TYPES

    print("=" * 80)
    print("BENCHMARK SMOKE TEST")
    print(f"Device: {device}")
    print(f"Policies: {['baseline'] + policies_to_test}")
    print(f"Synthetic data: {N_KEYFRAMES} keyframes, {N_RINGS}x{N_FREQS} FFT")
    print("=" * 80)

    # Seed
    np.random.seed(42)
    torch.manual_seed(42)

    results = []

    # Baseline
    print(f"\n[1/{len(policies_to_test)+1}] Testing: baseline (no policy)...")
    r = run_baseline_smoke_test(device)
    results.append(r)
    status = r['status']
    print(f"  -> {status} | params={r.get('total_params', '?'):,} | "
          f"loss={r.get('loss', '?'):.4f} | {r.get('elapsed', 0):.2f}s")
    if status == 'FAIL':
        print(f"  ERROR: {r.get('error', '?')}")

    # Each policy
    for idx, policy_type in enumerate(policies_to_test, 2):
        print(f"\n[{idx}/{len(policies_to_test)+1}] Testing: {policy_type}...")
        r = run_smoke_test(policy_type, device)
        results.append(r)
        status = r['status']
        grad_str = "grad=OK" if r.get('policy_gradient_flow') else "grad=NONE"
        print(f"  -> {status} | policy_params={r.get('policy_params', '?'):,} | "
              f"total_params={r.get('total_params', '?'):,} | "
              f"loss={r.get('loss', '?'):.4f} | {grad_str} | {r.get('elapsed', 0):.2f}s")
        if status == 'FAIL':
            print(f"  ERROR: {r.get('error', '?')}")
            if 'traceback' in r:
                for line in r['traceback'].split('\n')[-5:]:
                    print(f"    {line}")

    # Summary
    n_pass = sum(1 for r in results if r['status'] == 'PASS')
    n_fail = sum(1 for r in results if r['status'] == 'FAIL')

    print("\n" + "=" * 80)
    print(f"SMOKE TEST SUMMARY: {n_pass} PASS, {n_fail} FAIL / {len(results)} total")
    print("=" * 80)

    print(f"\n{'Policy':<25} {'Status':<8} {'Params':>10} {'Pol Params':>12} "
          f"{'Output Dim':>12} {'Grad Flow':>10} {'Loss':>8}")
    print("-" * 90)

    for r in results:
        name = r['policy_type']
        status = r['status']
        params = r.get('total_params', 0)
        pol_params = r.get('policy_params', 0)
        out_dim = r.get('effective_input_dim', r.get('output_shape', ['?', '?'])[1] if 'output_shape' in r else '?')
        grad = "OK" if r.get('policy_gradient_flow') else ("n/a" if 'baseline' in name else "NONE")
        loss = r.get('loss', 0)

        print(f"{name:<25} {status:<8} {params:>10,} {pol_params:>12,} "
              f"{str(out_dim):>12} {grad:>10} {loss:>8.4f}")

    print()

    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
