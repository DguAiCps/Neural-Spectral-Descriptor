#!/usr/bin/env python3
"""방향별 분모를 독립 집계하고 대칭 시간 그래프의 역재생을 검증한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_context_transfer_diagnose as diagnosis
from _pdv2_input_alignment_lib import make_graph
from _pdv2_no_context_ablation import sha256, write_json

prior, OUT = diagnosis.prior, diagnosis.OUT


def summarize(row: dict, mask: np.ndarray) -> dict:
    """누락 후보를 분모에서 빼지 않고 정답 손익을 집계한다."""
    old, new = row['old_correct'] & mask, row['new_correct'] & mask
    eligible = mask & row['eligible']
    gains, losses = new & ~old, old & ~new
    result = {'positive': int(mask.sum()), 'old_correct': int(old.sum()),
        'new_correct': int(new.sum()), 'gained': int(gains.sum()), 'lost': int(losses.sum()),
        'gain_rate_given_wrong': float(gains.sum() / (mask & ~old).sum()) if (mask & ~old).any() else None,
        'loss_rate_given_correct': float(losses.sum() / old.sum()) if old.any() else None,
        'fixed_pool_eligible': int(eligible.sum()),
        'fixed_pool_gained': int((eligible & ~old & row['fixed_correct']).sum()),
        'fixed_pool_lost': int((eligible & old & ~row['fixed_correct']).sum()),
        'raw_pair_margin_positive': int((eligible & (row['raw_cos_margin'] > 0)).sum()),
        'local_raw_pair_margin_positive': int((eligible & (row['neighbor_raw_margin'] > 0)).sum()),
        'lost_context_margin_negative_delta': int((losses & (row['context_contribution_delta'] < 0)).sum()),
        'lost_recovered_old_pool': int((losses & row['fixed_correct']).sum()),
        'lost_additionally_recovered_old_width': int((losses & ~row['fixed_correct'] & row['old_width_correct']).sum()),
        'means': {}}
    for cohort, chosen in (('all', eligible), ('gained', gains & eligible), ('lost', losses & eligible)):
        result['means'][cohort] = {k: float(row[k][chosen].mean()) if chosen.any() else None for k in
            ('context_contribution_delta', 'raw_contribution_delta', 'key_width_delta',
             'raw_cos_margin', 'neighbor_raw_margin', 'smoothness', 'step_m', 'initial_margin')}
    return result


def main() -> None:
    """기존 배열에서 집계를 재계산하고 두 고정 가중치의 순서 반전을 검산한다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = prior.protocol(True)
    audit = prior.base.read(OUT / 'completion_validation.json')
    assert audit['passed']
    sources = dict(audit['sources'])
    for p in (Path(__file__), REPO / 'scripts/_pdv2_input_alignment_lib.py',
              prior.OUT / 'summary.json', prior.OUT / diagnosis.ARM / 'epoch4.pth'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    assert not (OUT / 'report.json').exists()
    write_json(OUT / 'report_protocol.json', {'sources': sources, 'test_used': False,
        'scope': '진단 배열의 모든 양성을 독립 재집계한다. 최초 summary.json의 gained/lost는 '
                 '상세 통계 딕셔너리로 덮였으므로 건수는 본 보고서의 별도 필드에서 읽는다. '
                 '원 배열·분모·모델 평가에는 영향이 없다. 역재생은 인코더만 비교하며 후보 과거성 '
                 '평가나 배포 조건을 뒤집지 않는다.',
        'reversal_models': ['initial', 'rank_filtered_guarded_epoch4'], 'roles': ['train', 'val']})
    all_rows, result = {}, {'roles': {}, 'sequences': {}, 'reversal': {}}
    for role in ('train', 'val'):
        rows = [prior.base.exp.arrays(OUT / role / (s.replace('/', '_') + '.npz')) for s in spec[role]]
        all_rows[role] = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        result['sequences'][role] = {s: summarize(r, np.ones(len(r['query']), bool)) for s, r in zip(spec[role], rows)}
    quartiles = np.quantile(all_rows['train']['smoothness'], [.25, .5, .75])
    for role, row in all_rows.items():
        result['roles'][role] = {'all': summarize(row, np.ones(len(row['query']), bool)),
            'direction': {str(g): summarize(row, row['group'] == g) for g in range(3)},
            'raw_margin_bins': {str(g): summarize(row, row['eligible'] &
                (np.digitize(row['raw_cos_margin'], [0., .01, .05], right=True) == g)) for g in range(4)},
            'smoothness_bins': {str(g): summarize(row, np.digitize(row['smoothness'], quartiles, right=True) == g)
                                for g in range(4)}}
        total = result['roles'][role]['all']
        expected = prior.base.read(prior.OUT / 'summary.json')['candidate_transitions'][diagnosis.ARM][role]['4']
        assert total['gained'] == expected['gained'] and total['lost'] == expected['lost']
        for k in ('positive', 'old_correct', 'new_correct', 'gained', 'lost'):
            assert total[k] == sum(v[k] for v in result['roles'][role]['direction'].values())
    net = prior.initial_model()
    total_nodes = total_edges = 0
    max_key_error = max_attribute_error = 0.
    for label in ('initial', 'rank_filtered_guarded_epoch4'):
        if label != 'initial':
            cp = prior.OUT / diagnosis.ARM / 'epoch4.pth'
            state = torch.load(cp, map_location='cpu', weights_only=False)['model_state_dict']
            current = net.state_dict()
            current.update(state)
            net.load_state_dict(current)
        result['reversal'][label] = {}
        for role in ('train', 'val'):
            for sequence in spec[role]:
                cache = prior.base.exp.arrays(prior.factor.paths_for(role, sequence)['cache'])
                raw, pose = cache['descriptors'], cache['poses']
                n = len(raw)
                g0 = make_graph(raw, pose, np.zeros(n, np.int64))
                g1 = make_graph(raw[::-1].copy(), pose[::-1].copy(), np.zeros(n, np.int64))
                e0, e1 = g0.edge_index.numpy(), n - 1 - g1.edge_index.numpy()
                order0 = np.argsort(e0[0]*n + e0[1])
                order1 = np.argsort(e1[0]*n + e1[1])
                np.testing.assert_array_equal(e0[:, order0], e1[:, order1])
                attr_error = float(np.abs(g0.edge_attr.numpy()[order0] - g1.edge_attr.numpy()[order1]).max())
                assert attr_error < 1e-6
                with torch.no_grad():
                    k0 = prior.base.normalize(net(g0.cuda()).float().cpu().numpy())
                    k1 = prior.base.normalize(net(g1.cuda()).float().cpu().numpy())[::-1]
                error = float(np.abs(k0 - k1).max())
                np.testing.assert_allclose(k0, k1, atol=4e-6, rtol=4e-5)
                saved = (prior.factor.key_path(role, 'interaction', sequence) if label == 'initial' else
                    prior.OUT / diagnosis.ARM / 'epoch4' / role / ('keys_' + sequence.replace('/', '_') + '.npz'))
                np.testing.assert_allclose(k0, prior.base.exp.arrays(saved)['embedding'], atol=4e-6, rtol=4e-5)
                sources[str(saved.relative_to(REPO))] = sha256(saved)
                result['reversal'][label][role + '/' + sequence] = {'nodes': n, 'edges': e0.shape[1],
                    'key_max_abs_error': error, 'edge_attribute_max_abs_error': attr_error}
                max_key_error, max_attribute_error = max(max_key_error, error), max(max_attribute_error, attr_error)
                total_nodes += n
                total_edges += e0.shape[1]
                del g0, g1
                print('REVERSAL_DONE', label, role, sequence, error, flush=True)
    result.update(smoothness_train_quartiles=quartiles.tolist(), posthoc=True, test_used=False,
        reversal_total_nodes=total_nodes, reversal_total_edges=total_edges,
        reversal_max_key_error=max_key_error, reversal_max_attribute_error=max_attribute_error,
        interpretation='역재생은 대칭 시간 그래프를 같은 관계로 재번호화한 입력이다. '
        '평가 모드에서 정보 증가가 없다. 학습 dropout 난수나 유한 정밀도까지 모든 갱신이 동일하다는 뜻은 아니다. '
        '주행 방향과 장면 반복성의 집단별 차이는 관찰 연관이며 독립 인과 효과가 아니다.')
    write_json(OUT / 'report.json', result)
    for path in (OUT / 'report_protocol.json', OUT / 'report.json'):
        sources[str(path.relative_to(REPO))] = sha256(path)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest, path
    write_json(OUT / 'report_validation.json', {'passed': True, 'sources': sources,
        'cohort_count_matches_previous_ledgers': True, 'reversal_runs': 34,
        'nodes': total_nodes, 'edges': total_edges, 'maximum_key_error': max_key_error,
        'maximum_attribute_error': max_attribute_error, 'new_updates': 0, 'test_used': False})
    print('TRANSFER_REPORT_COMPLETE', max_key_error, flush=True)


if __name__ == '__main__':
    main()
