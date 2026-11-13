#!/usr/bin/env python3
"""정렬 교사와 세기 고정 대조의 최종 원장·무결성을 검산한다."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import _pdv2_diffattn_alignment_teacher as exp


def main() -> None:
    """두 실험의 원장에서 최종 지표를 재계산하고 완료 기록을 쓴다."""
    roots = (exp.OUT, exp.ROOT / 'results/pdv2_diffattn_fixed_gate_teacher_20261104')
    reports, total_rows, total_ledgers, prefix_checks = {}, 0, 0, 0
    for root in roots:
        spec = exp.run.READ(root / 'protocol.json')
        assert exp.run.READ(root / 'status.json')['status'] == 'validation_complete'
        for name, digest in spec['sources'].items():
            assert exp.run.HASH(exp.ROOT / name) == digest, name
        records = list(root.glob('*/epoch*/val/*.json'))
        for path in records:
            meta = exp.run.READ(path)
            ledger = path.with_suffix('.npz')
            assert exp.run.HASH(ledger) == meta['ledger_sha256']
            checkpoint = root / meta['arm'] / f'epoch{meta["epoch"]}.pth'
            assert exp.run.HASH(checkpoint) == meta['checkpoint_sha256']
            row = exp.run.ARR(ledger)
            for score in ('score', 'spectral'):
                metrics = exp.run.base.independent_metrics(row, row[score])
                for key in exp.run.base.METRICS:
                    if metrics[key] is not None:
                        np.testing.assert_allclose(metrics[key], meta['metrics'][score][key], rtol=0, atol=1e-12)
            assert meta['test_used'] is False
            total_rows += len(row['query_idx'])
            prefix_checks += len(meta['prefix_checks'])
        training = []
        for arm in spec['arms']:
            done = exp.run.READ(root / arm / 'training_complete.json')
            assert done['updates'] == 448
            assert all(n.startswith('gnn.convs.') for n in done['changed_parameters'])
            for name, digest in done['checkpoints'].items():
                assert exp.run.HASH(root / arm / name) == digest
            training.append(dict(arm=arm, updates=done['updates'], changed_tensors=len(done['changed_parameters'])))
        reports[root.name] = dict(summary=exp.run.READ(root / 'summary.json'), training=training,
                                 ledgers=len(records), protocol_sha256=exp.run.HASH(root / 'protocol.json'))
        total_ledgers += len(records)
    audit = exp.run.READ(roots[0] / 'gate_audit.json')['records']['val']['macro']
    base = reports[roots[0].name]['summary']
    np.testing.assert_allclose(audit['frozen']['key_R1'], base['frozen_intermediate']['R1'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(audit['learned']['key_R1'], base['results']['teacher_lr4_8']['intermediate']['key_R1'], rtol=0, atol=1e-12)
    figures = exp.run.READ(exp.ROOT / 'results/pdv2_learning_plan_20261104/restoration.json')['figure_sha256']
    for name, digest in figures.items():
        assert exp.run.HASH(exp.ROOT / 'docs/paper/body_v2.2' / name) == digest, name
    record = dict(status='passed', experiments=reports, rows=total_rows, ledgers=total_ledgers,
                  prefix_checks=prefix_checks, figure_hashes_unchanged=len(figures), test_used=False,
                  source_sha256=exp.run.HASH(Path(__file__)),
                  conclusion='검증 탐색의 최종 판정과 중간 검색을 구분한다. 같은 동결 대비 학습 기여를 판단하며, 원고 수치·시험 결과로 대체하지 않는다.')
    assert total_ledgers == 153 and total_rows == 1687012 and prefix_checks == 306
    exp.run.WRITE(roots[1] / 'integrated_validation.json', record)
    print('VERIFIED', total_ledgers, total_rows, prefix_checks, 'figures', len(figures), flush=True)


if __name__ == '__main__':
    exp.run.configure()
    main()
