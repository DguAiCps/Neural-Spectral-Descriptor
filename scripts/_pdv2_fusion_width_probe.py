#!/usr/bin/env python3
"""같은 초기 가중치에서 점수 차이와 정규화 폭의 기울기를 분해한다."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_fusion_width as run
from _pdv2_fusion_guard_gradient import vector_gradient
from _pdv2_no_context_ablation import sha256, write_json
from retrieval.fusion_guard import fusion_guard_loss


def compare(exact: np.ndarray, direct: np.ndarray) -> dict:
    """정확 미분을 대리 미분과 폭 미분으로 나눠 크기·방향을 기록한다."""
    width = exact - direct
    norms = [np.linalg.norm(g) for g in (exact, direct, width)]
    return {'exact_norm': float(norms[0]), 'direct_norm': float(norms[1]),
        'width_norm': float(norms[2]), 'width_to_direct_norm': float(norms[2]/norms[1]),
        'width_direct_cosine': float(width @ direct / max(norms[1]*norms[2], 1e-30)),
        'exact_direct_cosine': float(exact @ direct / max(norms[0]*norms[1], 1e-30))}


def main() -> None:
    """최초63배치에서 가중치 갱신 없이 두 손실의 전체 파라미터 기울기를 잰다."""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = run.protocol(True)
    folder = run.OUT / 'gradient_probe'
    folder.mkdir(exist_ok=True)
    assert not (folder / 'summary.json').exists()
    net = run.initial_model()
    parameters = [p for p in net.parameters() if p.requires_grad]
    dimension = sum(p.numel() for p in parameters)
    totals = {a: np.zeros((2, dimension), np.float64) for a in ('rank_all', 'rank_filtered_guarded')}
    records = []
    for ordinal, seq in enumerate(spec['train']):
        n = spec['pool_counts'][seq]['queries']
        if not n:
            continue
        graph = torch.load(run.pool_path(seq) / 'graph.pt', map_location='cuda', weights_only=False)
        pool = run.load_pool(seq)
        order = np.random.default_rng(1000 + ordinal).permutation(n)
        for start in range(0, n, run.CFG['batch_queries']):
            rows = torch.as_tensor(order[start:start+run.CFG['batch_queries']], device='cuda')
            embedding = F.normalize(net(graph), dim=1)
            exact = run.prior.terms_for(embedding, pool, rows, tuple(spec['confidence']['1']['tuple']))
            direct = run.terms_for(embedding, pool, rows, tuple(spec['confidence']['1']['tuple']))
            for key in exact:
                torch.testing.assert_close(exact[key], direct[key], rtol=0, atol=0)
            record = {'sequence': seq, 'rows': len(rows)}
            for index, arm in enumerate(totals):
                guarded = arm == 'rank_filtered_guarded'
                ge = vector_gradient(fusion_guard_loss(exact, guarded, guarded), parameters, True)
                gd = vector_gradient(fusion_guard_loss(direct, guarded, guarded), parameters, index == 0)
                array = torch.stack((ge, gd)).detach().double().cpu().numpy()
                totals[arm] += array
                record[arm] = compare(*array)
                del ge, gd
            records.append(record)
            del embedding, exact, direct
        del graph, pool
        print('WIDTH_GRAD_DONE', seq, flush=True)
    assert len(records) == spec['updates_per_epoch']
    for arm in totals:
        totals[arm] /= len(records)
    result = {'initial_batch_mean': {a: compare(*v) for a, v in totals.items()},
        'negative_width_direct_batches': {a: sum(r[a]['width_direct_cosine'] < 0 for r in records) for a in totals},
        'batches': records, 'batch_count': len(records), 'parameter_dimension': dimension,
        'scope': '초기 가중치에서 배치 평균 손실 기울기의63배치 평균이다. 실제 Adam 업데이트·'
                 '순차 학습·일반화 손실의 원인 비율이 아니다. 보호 항은 시작점에서0이다.',
        'new_updates': 0, 'test_used': False}
    write_json(folder / 'summary.json', result)
    np.savez_compressed(folder / 'vectors.npz', **totals)
    write_json(folder / 'completion_validation.json', {'passed': True, 'forward_terms_bitwise_equal': True,
        'batches': len(records), 'sources': {str(p.relative_to(REPO)): sha256(p) for p in
        (Path(__file__), REPO / 'scripts/_pdv2_fusion_guard_gradient.py', run.OUT / 'protocol.json',
         folder / 'summary.json', folder / 'vectors.npz')}, 'test_used': False})
    print('WIDTH_GRAD_COMPLETE', result['initial_batch_mean'], flush=True)


if __name__ == '__main__':
    main()
