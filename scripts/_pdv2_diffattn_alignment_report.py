#!/usr/bin/env python3
"""중간 검색과 정렬 열화 진단의 학습 전후 차이를 집계한다."""
from __future__ import annotations

import numpy as np
import _pdv2_diffattn_alignment_stages as run


def main() -> None:
    """모든 조건과 조건부 짝지은 구간을 빠짐없이 남긴다."""
    spec = run.prepare()
    outputs = {}
    for role in ('val', 'test'):
        summary = run.run.READ(run.OUT / role / 'summary.json')
        results = summary['results']
        contrasts = {}
        for label in results:
            if not label.startswith('frozen__'):
                continue
            suffix = label[len('frozen__'):]
            contrasts[suffix] = {}
            for model in ('convs', 'whole'):
                learned = results[model + '__' + suffix]
                frozen = results[label]
                contrasts[suffix][model] = dict(
                    mean_delta_pp={m: 100 * (learned['mean'][m] - frozen['mean'][m]) for m in run.run.base.METRICS},
                    per_seed_delta_pp={s: {m: 100 * (learned['per_seed'][s]['macro'][m] - frozen['per_seed'][s]['macro'][m])
                        for m in run.run.base.METRICS} for s in ('1', '2', '3')})
        changes, checks, ledgers, queries = [], 0, 0, {}
        for seed in spec['seeds']:
            queries[str(seed)] = 0
            for seq in spec[role]:
                meta = run.run.READ(run.OUT / role / f's{seed}' / (seq.replace('/', '_') + '.json'))
                checks += len(meta['checks'])
                queries[str(seed)] += meta['queries']
                for model, record in meta['records'].items():
                    ledgers += len(record['conditions'])
                    for change in record.get('numerical_replay_changes', []):
                        changes.append(dict(seed=seed, sequence=seq, model=model, **change))
        outputs[role] = dict(contrasts=contrasts, recall=summary['recall'],
            validation=dict(status='passed', jobs=3 * len(spec[role]), ledgers=ledgers,
                            prefix_model_checks=checks, queries_per_seed=queries,
                            numerical_replay_changes=changes))
        for model in run.MODELS:
            np.testing.assert_allclose(summary['recall'][model]['mean']['R1'],
                results[model + '__key__score']['mean']['R@1'], atol=1e-12, rtol=0)
            assert summary['recall'][model]['mean']['union'] >= summary['recall'][model]['mean']['R100']
    # 고정된 세 시드의 평균을 두고 같은 도시 안에서 시퀀스를 짝지어 다시 뽑는다.
    # 학습 난수나 새 미노출 시험의 불확실성은 포함하지 않는다.
    test = run.run.READ(run.OUT / 'test/summary.json')['results']
    rng = np.random.default_rng(20261104)
    city_samples = {}
    for city, sequences in run.run.current.CITY.items():
        keep = [s for s in sequences if test['frozen__full__score']['per_seed']['1']['sequences'][s]['n_pos']]
        city_samples[city] = (keep, rng.integers(len(keep), size=(10000, len(keep))))
    intervals = {}
    for model in ('convs', 'whole'):
        for suffix in ('key__score', 'pre__score', 'no_alignment__score',
                       'full__center', 'full__spectral', 'full__score',
                       'angle12__center', 'angle12__spectral', 'angle12__score', 'interaction'):
            metric_results = {}
            for metric in run.run.base.METRICS:
                draws = []
                for city, (sequences, sample) in city_samples.items():
                    def delta(seq: str, arm: str) -> float:
                        return float(np.mean([test[model + '__' + arm]['per_seed'][s]['sequences'][seq][metric]
                            - test['frozen__' + arm]['per_seed'][s]['sequences'][seq][metric] for s in ('1', '2', '3')]))
                    values = np.asarray([delta(seq, suffix) if suffix != 'interaction' else
                        delta(seq, 'angle12__score') - delta(seq, 'full__score') for seq in sequences])
                    draws.append(values[sample].mean(1))
                metric_results[metric] = (100 * np.quantile(np.mean(draws, axis=0), [.025, .975])).tolist()
            intervals[model + '__' + suffix] = metric_results
    reference = run.run.READ(run.seeds.PARENT / 'fixed_test/summary.json')['mean']
    conv_reference = run.run.READ(run.seeds.PARENT / 'convs_fixed_test/summary.json')['mean']
    replay_deltas = {}
    for model, ref in (('frozen', reference['frozen_same']), ('whole', reference['learned']),
                       ('convs', conv_reference['learned'])):
        replay_deltas[model] = {m: 100 * (test[model + '__full__score']['mean'][m] - ref[m]) for m in run.run.base.METRICS}
    run.run.WRITE(run.OUT / 'integrated_report.json', dict(**outputs, conditional_bootstrap_95_pp=intervals,
        test_reference_replay_delta_pp=replay_deltas, bev_native=reference['bev_native'],
        bev_temporal=reference['bev_temporal'], test_used_for_selection=False,
        interval_scope='초기3시드 고정 평균·도시 내 시퀀스 짝지은10000재표집. 학습 난수 불확실성·독립 미노출 확증 아님.',
        source_sha256=run.run.HASH(run.Path(__file__)), protocol_sha256=run.run.HASH(run.OUT / 'protocol.json')))
    print('REPORT_PASSED', {r: outputs[r]['validation']['ledgers'] for r in outputs})


if __name__ == '__main__':
    main()
