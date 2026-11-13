#!/usr/bin/env python3
"""기하 결합 세 조건의 최종 원장과 후보·점수·접두 일관성을 검산한다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_geometry_interaction as run
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_temporal_test_control_finish import scalar_score
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """동일 질의에서 후보를 다시 검색한 결과를 별도 계산으로 확인한다."""
    spec = run.protocol(True)
    summary = run.base.read(run.OUT / 'summary.json')
    sources = {**spec['runtime'], **spec['inputs']}
    counts, measured, states = {}, {}, {}
    max_metric_error = max_score_error = max_prefix_error = 0.
    ledgers = rows_total = metrics_total = scalar_checks = prefixes = 0
    for arm in run.CFG['arms']:
        counts[arm], measured[arm], states[arm] = {}, {m: {} for m in ('raw', run.METHOD)}, {}
        root = run.base.root(1, arm)
        complete = run.base.read(root / 'training_complete.json')
        assert complete['updates'] == 900
        for name, digest in complete['checkpoints'].items():
            p = root / 'checkpoints' / name
            assert sha256(p) == digest
            sources[str(p.relative_to(REPO))] = digest
        for seq in spec['val']:
            p = root / 'epoch20' / (seq.replace('/', '_') + '.npz')
            info = run.base.read(p.with_suffix('.json'))
            kp, cp = p.with_name('keys_' + p.name), p.with_name('coverage_' + p.name)
            expected = [(p, info['sha256']), (kp, info['keys_sha256']),
                        (cp, run.base.read(cp.with_suffix('.json'))['sha256'])]
            for path, digest in expected:
                assert sha256(path) == digest
                sources[str(path.relative_to(REPO))] = digest
            row, keydata, coverage = [run.base.exp.arrays(path) for path in (p, kp, cp)]
            q, j, key = row['query_idx'], row['j_star'], keydata['embedding']
            np.testing.assert_array_equal(q, np.arange(30, len(key) - 15))
            np.testing.assert_array_equal(q, coverage['query_idx'])
            np.testing.assert_array_equal(j, coverage['j_star'])
            np.testing.assert_array_equal(coverage['union_hit'], coverage['key_hit'] | coverage['aux_hit'])
            assert np.all(j <= q - 30)
            _, _, paths, _ = run.base.exp.old.base.inputs('val', seq)
            cache = run.base.exp.arrays(paths['cache'])
            np.testing.assert_array_equal(keydata['scan_ids'], cache['scan_ids'])
            np.testing.assert_allclose(row['top1_geo_dist'], np.linalg.norm(
                cache['poses_gt'][q, :3, 3] - cache['poses_gt'][j, :3, 3], axis=1), atol=1e-10)
            for method in measured[arm]:
                actual = independent_metrics(row, row[method])
                measured[arm][method][seq] = actual
                for metric, value in actual.items():
                    reference = info['metrics'][method][metric]
                    if value is None:
                        assert reference is None
                    else:
                        max_metric_error = max(max_metric_error, abs(value - reference))
                        assert abs(value - reference) < 5e-13
                metrics_total += 1
            pos, correct = row['min_past_gt_dist'] <= 5, row['top1_geo_dist'] <= 5
            assert not np.any(correct & ~coverage['union_hit'])
            state = np.where(correct, 2, np.where(coverage['union_hit'], 1, 0))
            states[arm][seq] = (q, pos, state)
            counts[arm][seq] = np.bincount(state[pos], minlength=3).tolist()
            with np.load(paths['spec']) as z:
                bank = run.base.exp.old.base.bank_for(run.base.exp.OP, z)
            xyz = cache['poses'][:, :3, 3]
            travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
            conf = spec['confidence']['1']
            for index in (0, len(q) // 3, 2 * len(q) // 3, len(q) - 1):
                t = int(q[index])
                ids = np.unique(np.r_[run.base.prefix_topk(key, t, t - 29, 100),
                                      run.base.prefix_topk(bank[0], t, t - 29, 100)])
                channels = (key[ids] @ key[t], bank[0][ids] @ bank[0][t],
                    run.base.align_scores(bank[1], bank[2], t, ids), run.base.column_align_scores(bank[3], t, ids))
                fused = sum(float(w) * run.base.minmax(v) for w, v in zip(conf['tuple'], channels))
                winner = int(ids[np.argsort(-fused, kind='stable')[0]])
                assert winner == j[index]
                channel = 2 if conf['decision'] == 'c_spec' else 3
                raw = float(channels[channel][np.flatnonzero(ids == winner)[0]])
                assert abs(raw - row['raw'][index]) < 1e-12
                score = scalar_score(bank, travel, t, winner, conf['decision'], raw)
                max_score_error = max(max_score_error, abs(score - row[run.METHOD][index]))
                assert abs(score - row[run.METHOD][index]) < 1e-12
                scalar_checks += 1
            for prefix in info['prefixes']:
                assert prefix['winner_equal'] and prefix['score_error'] == 0.
                assert prefix['key_max_error'] < 4e-6
                max_prefix_error = max(max_prefix_error, prefix['key_max_error'])
                prefixes += 1
            for path in (p.with_suffix('.json'), cp.with_suffix('.json')):
                sources[str(path.relative_to(REPO))] = sha256(path)
            ledgers += 1
            rows_total += len(q)
        for method, rows in measured[arm].items():
            actual = run.base.macro(rows)
            expected = summary['results'][arm][method]
            for group in ('macro',):
                for metric, value in actual[group].items():
                    np.testing.assert_allclose(value, expected[group][metric], atol=5e-13, rtol=0)
            for family, values in actual['families'].items():
                for metric, value in values.items():
                    np.testing.assert_allclose(value, expected['families'][family][metric], atol=5e-13, rtol=0)
    transitions = {}
    for arm in run.CFG['arms'][1:]:
        transition = np.zeros((3, 3), np.int64)
        per_sequence = {}
        for seq in spec['val']:
            q, mask, before = states['additive'][seq]
            q2, mask2, after = states[arm][seq]
            np.testing.assert_array_equal(q, q2)
            np.testing.assert_array_equal(mask, mask2)
            matrix = np.bincount(3 * before[mask] + after[mask], minlength=9).reshape(3, 3)
            transition += matrix
            per_sequence[seq] = matrix.tolist()
        transitions[arm] = {'total': transition.tolist(), 'by_sequence': per_sequence}
    write_json(run.OUT / 'candidate_diagnosis.json', {'states': ['missing', 'present_but_wrong', 'correct'],
        'counts_by_sequence': counts, 'reference_to_arm': transitions, 'test_used': False})
    for p in (run.OUT / 'summary.json', run.OUT / 'amp_smoke.json',
              run.OUT / 'candidate_diagnosis.json', run.OUT / 'protocol.json'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest
    write_json(run.OUT / 'completion_validation.json', {'passed': True, 'ledgers': ledgers,
        'rows': rows_total, 'metric_sets': metrics_total, 'scalar_checks': scalar_checks,
        'prefixes': prefixes, 'maximum_metric_error': max_metric_error,
        'maximum_score_error': max_score_error, 'maximum_prefix_key_error': max_prefix_error,
        'sources': sources, 'test_used': False})
    print('INTERACTION_AUDIT_PASSED', ledgers, rows_total, flush=True)


if __name__ == '__main__':
    main()
