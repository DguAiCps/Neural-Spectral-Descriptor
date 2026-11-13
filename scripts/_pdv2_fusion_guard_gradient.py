#!/usr/bin/env python3
"""추가 학습 시작점에서 큰 손실을 내는 불가능한 질의의 기울기를 분리한다."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_fusion_guard as run
from _pdv2_no_context_ablation import sha256, write_json


def vector_gradient(loss: torch.Tensor, parameters: list, retain: bool) -> torch.Tensor:
    """사용하지 않는 매개변수에는 0을 넣어 공통 좌표의 기울기를 만든다."""
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain)
    flat = torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                      for p, g in zip(parameters, gradients)])
    assert torch.isfinite(flat).all()
    return flat


def relation(first: np.ndarray, second: np.ndarray) -> dict:
    """두 기울기의 크기와 방향을 기록한다. 실제 Adam 갱신 비중은 아니다."""
    nf, ns = np.linalg.norm(first), np.linalg.norm(second)
    return {'admissible_norm': float(nf), 'impossible_norm': float(ns),
        'total_norm': float(np.linalg.norm(first + second)),
        'impossible_to_admissible_norm': float(ns / nf) if nf else None,
        'cosine': float(first @ second / nf / ns) if nf and ns else None}


def main() -> None:
    """고정된 시작 모델과 실제 모든 학습 배치에서 손실 집단을 나눠 미분한다."""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = run.protocol(True)
    audit = run.base.read(run.OUT / 'completion_validation.json')
    assert audit['passed']
    output = run.OUT / 'gradient_probe'
    output.mkdir(exist_ok=True)
    source = dict(audit['sources'])
    source[str(Path(__file__).relative_to(REPO))] = sha256(Path(__file__))
    write_json(output / 'protocol.json', {'sources': source, 'posthoc': True,
        'scope': '추가 학습 시작 가중치에서 1에포크의 모든63배치를 고정하여 기울기를 분리한다. '
                 '업데이트 없이 각 배치 평균 손실의 기울기를 구하고63으로 나눠 합한다. '
                 '실제 순차 Adam 갱신이나 최종 성능의 기여율이 아니다.', 'test_used': False})
    net = run.initial_model()
    parameters = [p for p in net.parameters() if p.requires_grad]
    dimension = sum(p.numel() for p in parameters)
    totals = np.zeros((2, dimension), dtype=np.float64)
    per_sequence, batch_records = {}, []
    maximum_additivity_error = 0.
    for ordinal, sequence in enumerate(spec['train']):
        n = spec['pool_counts'][sequence]['queries']
        if not n:
            continue
        graph = torch.load(run.pool_path(sequence) / 'graph.pt', map_location='cuda', weights_only=False)
        pool = run.load_pool(sequence)
        order = np.random.default_rng(1000 + ordinal).permutation(n)
        sequence_gradients = np.zeros_like(totals)
        for start in range(0, n, run.CFG['batch_queries']):
            rows = torch.as_tensor(order[start:start + run.CFG['batch_queries']], device='cuda')
            embedding = F.normalize(net(graph), dim=1)
            terms = run.terms_for(embedding, pool, rows, tuple(spec['confidence']['1']['tuple']))
            feasible = (terms['rank'] * terms['admissible']).mean()
            impossible = (terms['rank'] * (terms['active'] & ~terms['admissible'])).mean()
            combined = vector_gradient(feasible + impossible, parameters, True) if start == 0 else None
            g0 = vector_gradient(feasible, parameters, True)
            g1 = vector_gradient(impossible, parameters, False)
            if combined is not None:
                error = float((g0 + g1 - combined).norm() / combined.norm().clamp_min(1e-12))
                assert error < 1e-4
                maximum_additivity_error = max(maximum_additivity_error, error)
            values = torch.stack((g0, g1)).detach().double().cpu().numpy()
            sequence_gradients += values
            batch_records.append({'sequence': sequence, 'rows': len(rows),
                'impossible_queries': int((terms['active'] & ~terms['admissible']).sum()),
                **relation(*values)})
            del embedding, terms, feasible, impossible, g0, g1, combined
        totals += sequence_gradients
        per_sequence[sequence] = relation(*(sequence_gradients / spec['pool_counts'][sequence]['steps_per_epoch']))
        del graph, pool
    assert len(batch_records) == spec['updates_per_epoch']
    totals /= len(batch_records)
    np.savez_compressed(output / 'gradient_vectors.npz', admissible=totals[0], impossible=totals[1])
    result = {'initial_step_mean': relation(*totals), 'per_sequence': per_sequence,
        'batches': batch_records, 'batch_count': len(batch_records),
        'negative_cosine_batches': sum(b['cosine'] is not None and b['cosine'] < 0 for b in batch_records),
        'maximum_gradient_additivity_error': maximum_additivity_error,
        'parameter_dimension': dimension, 'new_updates': 0, 'posthoc': True, 'test_used': False}
    write_json(output / 'summary.json', result)
    for path in (output / 'protocol.json', output / 'summary.json', output / 'gradient_vectors.npz'):
        source[str(path.relative_to(REPO))] = sha256(path)
    for path, digest in source.items():
        assert sha256(REPO / path) == digest, path
    write_json(output / 'completion_validation.json', {'passed': True, 'sources': source,
        'batches': len(batch_records), 'gradient_additivity_checks': len(per_sequence), 'new_updates': 0})
    print('GUARD_GRADIENT_COMPLETE', result['initial_step_mean'], result['negative_cosine_batches'], flush=True)


if __name__ == '__main__':
    main()
