#!/usr/bin/env python3
"""후보·표현·정규화 대조의 최종 원장과 모든 변경 질의를 집계한다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'scripts')]
import _pdv2_fusion_factor as run
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """학습과 검증을 따로 보고하고 독립 지표·원천 해시를 검산한다."""
    spec = run.protocol(True)
    sources = {**spec['runtime'], **spec['inputs']}
    key_check = run.base.read(run.OUT / 'training_keys.json')
    for p, digest in key_check['outputs'].items():
        assert sha256(REPO / p) == digest
        sources[p] = digest
    results, cohorts, transitions, changed = {}, {}, {}, {}
    n_ledgers = n_rows = n_metrics = n_prefixes = n_scalar = 0
    maximum_metric_error = 0.
    for role in ('train', 'val'):
        metrics = {v: {} for v in run.VARIANTS}
        cohorts[role], changed[role] = {}, {}
        transitions[role] = {v: np.zeros((3, 3), np.int64) for v in run.VARIANTS if v != 'new'}
        for seq in spec[role]:
            folder = run.OUT / role / seq.replace('/', '_')
            report = run.base.read(folder / 'diagnosis.json')
            cohorts[role][seq] = report['cohorts']
            states = {}
            for variant in run.VARIANTS:
                path = folder / (variant + '.npz')
                assert sha256(path) == report['outputs'][path.name]
                sources[str(path.relative_to(REPO))] = sha256(path)
                row = run.base.exp.arrays(path)
                q, j = row['query_idx'], row['j_star']
                assert len(q) == report['queries'] and np.all(j <= q - 30)
                np.testing.assert_array_equal(q, np.arange(30, 30 + len(q)))
                assert all(np.isfinite(row[k]).all() for k in ('top1_geo_dist', 'raw', run.METHOD))
                positive = row['min_past_gt_dist'] <= 5
                correct = row['top1_geo_dist'] <= 5
                assert not np.any(correct & ~row['union_hit'])
                state = np.where(correct, 2, np.where(row['union_hit'], 1, 0))
                states[variant] = state[positive]
                actual = independent_metrics(row, row[run.METHOD])
                canonical = run.base.exp.metrics(row[run.METHOD], row)
                metrics[variant][seq] = actual
                for k in run.base.exp.METRICS:
                    if actual[k] is None:
                        assert canonical[k] is None
                    else:
                        error = abs(actual[k] - canonical[k])
                        maximum_metric_error = max(maximum_metric_error, error)
                        assert error < 5e-13
                        assert abs(actual[k] - report['metrics'][variant][k]) < 5e-13
                n_ledgers += 1
                n_rows += len(q)
                n_metrics += 1
            for variant, matrix in transitions[role].items():
                matrix += np.bincount(3 * states['new'] + states[variant], minlength=9).reshape(3, 3)
            case_summary = {}
            for change in ('loss', 'gain'):
                cases = [c for c in report['changed_correctness'] if c['change'] == change]
                sign = 1 if change == 'gain' else -1
                case_summary[change] = {
                    'count': len(cases), 'farther_than_25m': sum(c['false_distance'] > 25 for c in cases),
                    'true_in_new': sum(c['true_in_new'] for c in cases),
                    'context_margin_deteriorates': sum(c['margins']['new']['context'] < c['margins']['old']['context'] for c in cases),
                    'key_margin_deteriorates': sum(c['margins']['new']['total'] < c['margins']['old']['total'] for c in cases),
                    'new_key_pair_prefers_true': sum(c['margins']['new']['total'] > 0 for c in cases),
                    'correct_by_variant': {v: sum(c['correct_by_variant'][v] for c in cases) for v in run.VARIANTS},
                }
                expected = int(((states['old'] == 2) & (states['new'] != 2)).sum()) if sign < 0 else int(((states['old'] != 2) & (states['new'] == 2)).sum())
                assert len(cases) == expected
            changed[role][seq] = case_summary
            for p in report['prefix_checks']:
                assert p['score_error'] == 0.
            for p in report['scalar_checks']:
                assert p['maximum_error'] < 2e-6
            n_prefixes += len(report['prefix_checks'])
            n_scalar += len(report['scalar_checks'])
            sources[str((folder / 'diagnosis.json').relative_to(REPO))] = sha256(folder / 'diagnosis.json')
        results[role] = {v: run.base.macro(rows) for v, rows in metrics.items()}
        for variant, result in results[role].items():
            print('FINAL', role, variant, result['macro'], flush=True)
    decomposition = {}
    for role, rows in results.items():
        decomposition[role] = {}
        for name in ('macro', *rows['old']['families']):
            scores = {v: (r['macro'] if name == 'macro' else r['families'][name])['AP'] for v, r in rows.items()}
            keys = .5 * ((scores['old_pool_new_key'] - scores['old']) + (scores['new'] - scores['new_pool_old_key']))
            pools = .5 * ((scores['new_pool_old_key'] - scores['old']) + (scores['new'] - scores['old_pool_new_key']))
            assert abs(keys + pools - scores['new'] + scores['old']) < 1e-12
            decomposition[role][name] = {'key_score_change_pp': 100 * keys, 'pool_change_pp': 100 * pools,
                'total_pp': 100 * (scores['new'] - scores['old']),
                'old_ranges_minus_new_pp': 100 * (scores['new_pool_old_ranges'] - scores['new']),
                'raw_scoring_minus_new_pp': 100 * (scores['new_pool_raw_key'] - scores['new'])}
    summary = {'results': results, 'cohorts': cohorts, 'changed_correctness': changed,
        'new_to_variant_transitions': {role: {v: m.tolist() for v, m in variants.items()} for role, variants in transitions.items()},
        'ap_arithmetic_decomposition': decomposition, 'test_used': False, 'training_updates': 0,
        'scope': '후보·점수교차진단의최종AP. 원고방법성능이나독립모듈기여로사용하지않는다. 학습8과검증9는계열과입력이달라절대값을일반화차이로직접비교하지않는다.'}
    write_json(run.OUT / 'summary.json', summary)
    for p in (run.OUT / 'summary.json', run.OUT / 'protocol.json', run.OUT / 'training_keys.json'):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for p, digest in sources.items():
        assert sha256(REPO / p) == digest, p
    write_json(run.OUT / 'completion_validation.json', {'passed': True, 'sources': sources,
        'ledgers': n_ledgers, 'rows': n_rows, 'metric_sets': n_metrics,
        'score_prefix_checks': n_prefixes, 'scalar_fusion_checks': n_scalar,
        'train_key_prefix_checks': sum(len(v) for v in key_check['prefix_checks'].values()),
        'maximum_metric_error': maximum_metric_error, 'test_used': False, 'training_updates': 0})
    print('FUSION_FACTOR_AUDIT_PASSED', n_ledgers, n_rows, flush=True)


if __name__ == '__main__':
    main()
