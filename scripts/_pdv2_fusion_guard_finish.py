#!/usr/bin/env python3
"""동일 학습 예산·후보 재검색·최종 지표와 보호 제약의 손익을 검산한다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_fusion_guard as run
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """모든 체크포인트의 검증과 선택 모델의 학습 결과를 따로 집계한다."""
    spec = run.protocol(True)
    selection = run.base.read(run.OUT / 'selection.json')
    baseline = run.base.read(run.factor.OUT / 'summary.json')['results']
    sources = {**spec['runtime'], **spec['inputs']}
    results = {role: {'start': baseline[role]['new']} for role in ('train', 'val')}
    transitions, pool_diagnostics = {}, {}
    count = total_rows = metrics = scalar_checks = prefixes = 0
    maximum_metric_error = maximum_score_error = 0.
    reference_orders = None
    updates = 0
    for arm in run.CFG['arms']:
        complete = run.base.read(run.OUT / arm / 'training_complete.json')
        assert complete['initial'] == spec['initial']
        assert complete['updates'] == run.CFG['epochs'] * spec['updates_per_epoch']
        assert complete['bn_and_remetric_frozen']
        if reference_orders is None:
            reference_orders = complete['orders']
        assert complete['orders'] == reference_orders
        updates += complete['updates']
        for name, digest in complete['checkpoints'].items():
            path = run.OUT / arm / name
            assert sha256(path) == digest
            sources[str(path.relative_to(REPO))] = digest
        path = run.OUT / arm / 'training_complete.json'
        sources[str(path.relative_to(REPO))] = sha256(path)
        chosen = selection['selected_epoch'][arm]
        transitions[arm], pool_diagnostics[arm] = {}, {}
        for role in ('train', 'val'):
            epochs = [chosen] if role == 'train' else run.CFG['checkpoint_epochs']
            transitions[arm][role] = {}
            for epoch in epochs:
                per_sequence, by_sequence = {}, {}
                total = np.zeros((3, 3), dtype=np.int64)
                for sequence in spec[role]:
                    path = run.OUT / arm / f'epoch{epoch}' / role / (sequence.replace('/', '_') + '.npz')
                    info = run.base.read(path.with_suffix('.json'))
                    key_path = path.with_name('keys_' + path.name)
                    assert sha256(path) == info['sha256'] and sha256(key_path) == info['keys_sha256']
                    row = run.base.exp.arrays(path)
                    key = run.base.exp.arrays(key_path)['embedding']
                    reference_path = run.factor.OUT / role / sequence.replace('/', '_') / 'new.npz'
                    reference = run.base.exp.arrays(reference_path)
                    q, j = row['query_idx'], row['j_star']
                    np.testing.assert_array_equal(q, reference['query_idx'])
                    assert np.all(j <= q - 30) and key.shape[1] == 416
                    assert np.isfinite(key).all() and np.isfinite(row[run.METHOD]).all()
                    paths = run.factor.paths_for(role, sequence)
                    cache = run.base.exp.arrays(paths['cache'])
                    position = cache['poses' if role == 'train' else 'poses_gt'][:, :3, 3]
                    np.testing.assert_allclose(row['top1_geo_dist'], np.linalg.norm(
                        position[q].astype(np.float64) - position[j].astype(np.float64), axis=1), atol=1e-10)
                    np.testing.assert_allclose(row['min_past_gt_dist'], reference['min_past_gt_dist'], atol=1e-10)
                    for method in ('raw', run.METHOD):
                        actual = run.base.exp.metrics(row[method], row)
                        for metric, value in actual.items():
                            expected = info['metrics'][method][metric]
                            if value is None:
                                assert expected is None
                            else:
                                maximum_metric_error = max(maximum_metric_error, abs(value - expected))
                                assert abs(value - expected) < 5e-13
                        metrics += 1
                    per_sequence[sequence] = info['metrics'][run.METHOD]
                    positive = row['min_past_gt_dist'] <= 5
                    before = np.where(reference['top1_geo_dist'] <= 5, 2,
                                      np.where(reference['union_hit'], 1, 0))
                    after = np.where(row['top1_geo_dist'] <= 5, 2, np.where(row['union_hit'], 1, 0))
                    assert not np.any((after == 2) & ~row['union_hit'])
                    matrix = np.bincount(3 * before[positive] + after[positive], minlength=9).reshape(3, 3)
                    total += matrix
                    by_sequence[sequence] = matrix.tolist()
                    with np.load(paths['spec']) as z:
                        bank = run.base.exp.old.base.bank_for(run.base.exp.OP, z)
                    xyz = cache['poses'][:, :3, 3]
                    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
                    conf = spec['confidence']['1']
                    for index in (0, len(q) - 1):
                        t = int(q[index])
                        ids = np.unique(np.r_[run.base.prefix_topk(key, t, t - 29, 100),
                            run.base.prefix_topk(bank[0], t, t - 29, 100)])
                        channels = (key[ids] @ key[t], bank[0][ids] @ bank[0][t],
                            run.base.align_scores(bank[1], bank[2], t, ids),
                            run.base.column_align_scores(bank[3], t, ids))
                        fused = sum(float(w) * run.base.minmax(v) for w, v in zip(conf['tuple'], channels))
                        winner = int(ids[np.argmax(fused)])
                        assert winner == j[index]
                        raw = float(channels[3 if conf['decision'] == 'c_col' else 2][np.argmax(fused)])
                        direct = run.factor.scalar_score(bank, travel, t, winner, conf['decision'], raw)
                        error = abs(direct - row[run.METHOD][index])
                        maximum_score_error = max(maximum_score_error, error)
                        assert error < 1e-12
                        scalar_checks += 1
                    for record in info['prefixes']:
                        assert record['winner_equal'] and record['score_error'] == 0
                        assert record['key_max_error'] < 4e-6
                        prefixes += 1
                    if role == 'train' and info['training_fixed_pool']:
                        pool_diagnostics[arm][sequence] = info['training_fixed_pool']
                    for p in (path, path.with_suffix('.json'), key_path, reference_path):
                        sources[str(p.relative_to(REPO))] = sha256(p)
                    count += 1
                    total_rows += len(q)
                aggregate = run.base.macro(per_sequence)
                if epoch == chosen:
                    results[role][arm] = aggregate
                if role == 'val':
                    expected = selection['results'][arm][str(epoch)]
                    for metric, value in aggregate['macro'].items():
                        assert abs(value - expected['macro'][metric]) < 5e-13
                transitions[arm][role][str(epoch)] = {'matrix': total.tolist(), 'by_sequence': by_sequence,
                    'gained': int(total[:2, 2].sum()), 'lost': int(total[2, :2].sum())}
    checks, effects = {}, {}
    for arm in run.CFG['arms']:
        checks[arm], effects[arm] = {}, {}
        result = results['val'][arm]
        for ref in ('start', 'rank_all'):
            other = results['val'][ref]
            delta = {m: 100 * (result['macro'][m] - other['macro'][m]) for m in run.base.exp.METRICS}
            effects[arm][ref] = delta
            checks[arm][ref] = {
                'AP': delta['AP'] >= 100 * run.CFG['promotion']['ap_gain_min'],
                'F1': delta['max_F1'] >= -100 * run.CFG['promotion']['f1_loss_max'],
                'R100': delta['R@100P'] >= -100 * run.CFG['promotion']['r100_loss_max'],
                'family_AP': all(result['families'][f]['AP'] >= other['families'][f]['AP']
                    - run.CFG['promotion']['family_ap_loss_max'] for f in other['families'])}
    same_epoch = {}
    for epoch in run.CFG['checkpoint_epochs']:
        d = {a: selection['results'][a][str(epoch)]['macro'] for a in run.CFG['arms']}
        same_epoch[str(epoch)] = {metric: 100 * (d['rank_filtered_guarded'][metric]
            - d['rank_filtered'][metric] - d['rank_guarded'][metric] + d['rank_all'][metric])
            for metric in run.base.exp.METRICS}
    eligible = [a for a in run.CFG['arms'] if a != 'rank_all'
                and all(all(c.values()) for c in checks[a].values())]
    summary = {'results': results, 'selected_epoch': selection['selected_epoch'],
        'all_validation_checkpoints': selection['results'], 'minus_reference_pp': effects,
        'promotion_checks': checks, 'eligible_candidates': eligible,
        'same_epoch_factorial_interaction_pp': same_epoch, 'candidate_transitions': transitions,
        'training_fixed_pool': pool_diagnostics, 'total_new_updates': updates,
        'initial_and_orders_equal': True, 'test_used': False,
        'scope': '시드1 검증 예비. 고정 후보의 순위 학습이며 최종 결과는 후보를 다시 검색한다. '
                 '시험·원고 승격 없음. 필터와 보호의 효과는 같은 예산 대조 범위다.'}
    write_json(run.OUT / 'summary.json', summary)
    for p in (run.OUT / 'summary.json', run.OUT / 'protocol.json', run.OUT / 'selection.json',
              run.factor.OUT / 'summary.json'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for p, digest in sources.items():
        assert sha256(REPO / p) == digest, p
    write_json(run.OUT / 'completion_validation.json', {'passed': True, 'sources': sources,
        'ledgers': count, 'rows': total_rows, 'metric_sets': metrics,
        'scalar_checks': scalar_checks, 'prefix_checks': prefixes,
        'maximum_metric_error': maximum_metric_error, 'maximum_score_error': maximum_score_error,
        'new_training_updates': updates, 'test_used': False})
    print('FUSION_GUARD_AUDIT_PASSED', count, total_rows, 'eligible', eligible, flush=True)


if __name__ == '__main__':
    main()
