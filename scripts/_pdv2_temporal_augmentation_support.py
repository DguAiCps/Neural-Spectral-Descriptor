#!/usr/bin/env python3
"""현재 학습 캐시에서 실제 반대 방향 관측과 순위 손실의 노출 범위를 구분한다."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_temporal_augmentation as run
from _pdv2_context_diagnosis import heading
from _pdv2_no_context_ablation import sha256, write_json


def empty() -> dict:
    """서로 겹칠 수 있는 후보 존재 조건의 집계 항목을 만든다."""
    return dict(queries=0, positive=0, same_available=0, opposite_available=0,
                both_available=0, opposite_only=0, positive_pairs=0, opposite_pairs=0)


def add(row: dict, direction: np.ndarray, valid: np.ndarray) -> None:
    """서로 다른 방향 정답이 공존하는 질의를 따로 센다."""
    same, opposite = valid & (direction >= .5), valid & (direction <= -.5)
    row['queries'] += 1
    row['positive'] += bool(len(direction))
    row['same_available'] += bool(same.any())
    row['opposite_available'] += bool(opposite.any())
    row['both_available'] += bool(same.any() and opposite.any())
    row['opposite_only'] += bool(len(direction) and opposite.all())
    row['positive_pairs'] += len(direction)
    row['opposite_pairs'] += int(opposite.sum())


def main() -> None:
    """같은 질의 범위에서 세션·시간 제한을 풀 때의 학습 관측을 조사한다."""
    spec = run.protocol(True)
    output = run.OUT / 'training_support'
    output.mkdir(exist_ok=True)
    assert not (output / 'summary.json').exists()
    sources = {**spec['runtime'], **spec['inputs']}
    sources[str(Path(__file__).relative_to(REPO))] = sha256(Path(__file__))
    sources['scripts/_pdv2_context_diagnosis.py'] = sha256(REPO / 'scripts/_pdv2_context_diagnosis.py')
    write_json(output / 'protocol.json', {'sources': sources, 'test_used': False, 'posthoc': True,
        'scope': '현재 GT학습8의 동일질의 q30..n-16에서 같은세션과거, 같은세션양방향, '
                 '검증된공통좌표의다른학습세션포함을 비교한다. 후보는5m. 같은세션은시간차30이상. '
                 'Bridge01~04만공통좌표, KITTI는시퀀스별분리. 학습 가능 관측의 진단이며 온라인 후보 규칙을 바꾸지 않는다.'})
    sequence_arrays = []
    for ordinal, seq in enumerate(spec['train']):
        cache = run.base.exp.arrays(run.factor.paths_for('train', seq)['cache'])
        position = cache['poses'][:, :3, 3].astype(np.float64)
        h, valid = heading(position)
        sequence_arrays.append({'position': position, 'heading': h, 'valid': valid,
            'sequence': np.full(len(position), ordinal), 'index': np.arange(len(position)),
            'coordinate': np.full(len(position), 100 if seq.startswith('helipr/') else ordinal)})
    full = {k: np.concatenate([a[k] for a in sequence_arrays]) for k in sequence_arrays[0]}
    trees = {}
    for coordinate in np.unique(full['coordinate']):
        ids = np.flatnonzero(full['coordinate'] == coordinate)
        trees[int(coordinate)] = (cKDTree(full['position'][ids]), ids)
    names = ('same_session_past', 'same_session_both_time', 'shared_coordinate_all_sessions')
    aggregate = {name: empty() for name in names}
    sequences, pool_totals = {}, {'queries': 0, 'opposite_available': 0, 'same_and_opposite': 0,
        'opposite_only': 0, 'best_positive_opposite': 0, 'conditional_positive_mass_opposite_sum': 0.,
        'conditional_positive_mass_opposite_sum_on_both': 0.}
    for ordinal, (sequence, local) in enumerate(zip(spec['train'], sequence_arrays)):
        rows = {name: empty() for name in names}
        tree, ids = trees[int(local['coordinate'][0])]
        queries = np.arange(30, len(local['position'])-15)
        for q in queries:
            candidates = ids[np.asarray(tree.query_ball_point(local['position'][q], 5.), dtype=np.int64)]
            same_sequence = full['sequence'][candidates] == ordinal
            index = full['index'][candidates]
            masks = (same_sequence & (index <= q-30), same_sequence & (np.abs(index-q) >= 30),
                     ~same_sequence | (np.abs(index-q) >= 30))
            cosine = full['heading'][candidates] @ local['heading'][q]
            valid = full['valid'][candidates] & local['valid'][q]
            for name, mask in zip(names, masks):
                add(rows[name], cosine[mask], valid[mask])
        for name in names:
            for k, value in rows[name].items():
                aggregate[name][k] += value
        pool = run.base.exp.arrays(run.pool_path(sequence) / 'pool.npz')
        totals = dict.fromkeys(pool_totals, 0)
        for i, q in enumerate(pool['queries']):
            valid = pool['valid'][i]
            candidates = pool['candidates'][i, valid]
            correct = pool['correct'][i, valid]
            values = np.column_stack((pool['teacher_key'][i, valid], pool['static'][i, valid]))
            score, _ = run.fuse_with_range(values, tuple(spec['confidence']['1']['tuple']))
            logits = score[correct].astype(np.float64) / spec['confidence']['1']['tuple'][0] / run.CFG['temperature']
            probabilities = np.exp(logits-logits.max())
            probabilities /= probabilities.sum()
            cosine = local['heading'][candidates[correct]] @ local['heading'][q]
            direction_valid = local['valid'][candidates[correct]] & local['valid'][q]
            same, opposite = direction_valid & (cosine >= .5), direction_valid & (cosine <= -.5)
            totals['queries'] += 1
            totals['opposite_available'] += bool(opposite.any())
            totals['same_and_opposite'] += bool(same.any() and opposite.any())
            totals['opposite_only'] += bool(opposite.all())
            totals['best_positive_opposite'] += bool(opposite[np.argmax(logits)])
            totals['conditional_positive_mass_opposite_sum'] += float(probabilities[opposite].sum())
            if same.any() and opposite.any():
                totals['conditional_positive_mass_opposite_sum_on_both'] += float(probabilities[opposite].sum())
        for k, value in totals.items():
            pool_totals[k] += value
        sequences[sequence] = {'available': rows, 'initial_fixed_pool': totals}
        print('TRAIN_SUPPORT_DONE', sequence, rows['shared_coordinate_all_sessions']['opposite_available'], flush=True)
    assert aggregate['same_session_past']['queries'] == 71946
    assert aggregate['same_session_past']['positive'] == 30772
    assert aggregate['same_session_past']['opposite_only'] == 62
    assert pool_totals['queries'] == 30726
    pool_totals['mean_opposite_mass_all_queries'] = pool_totals['conditional_positive_mass_opposite_sum']/pool_totals['queries']
    pool_totals['mean_opposite_mass_when_both_available'] = (pool_totals['conditional_positive_mass_opposite_sum_on_both']/pool_totals['same_and_opposite']
                                                          if pool_totals['same_and_opposite'] else None)
    result = {'aggregate': aggregate, 'initial_fixed_pool': pool_totals, 'sequences': sequences,
        'test_used': False, 'posthoc': True, 'new_updates': 0,
        'interpretation': '반대 방향만 정답인 질의62개는 반대 방향 관측 전체 개수가 아니다. '
                         '정답 로그합항의 후보 확률질량은 실제 전체 파라미터 기울기의 방향별 비중이 아니다. '
                         '보강 학습의 이득이나 손실을 측정하지 않는다.'}
    write_json(output / 'summary.json', result)
    for p in (output / 'protocol.json', output / 'summary.json'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for path, value in sources.items():
        assert sha256(REPO / path) == value, path
    write_json(output / 'completion_validation.json', {'passed': True, 'sources': sources,
        'previous_cohort_counts_reproduced': True, 'new_updates': 0, 'test_used': False})
    print('TRAIN_SUPPORT_COMPLETE', aggregate, pool_totals, flush=True)


if __name__ == '__main__':
    main()
