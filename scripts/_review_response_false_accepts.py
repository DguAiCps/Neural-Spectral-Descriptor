#!/usr/bin/env python3
"""이미 고정한 검증 기준값의 오수락 위치를 분해한다. 설정을 고르지 않는다."""
import json
from pathlib import Path

import numpy as np

import _paper_fixed_threshold as fixed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/pdv2_review_response_20261104/false_accepts.json'


def main() -> None:
    """원 결과의 FP·TP를 정확히 재현하고 시퀀스·거리 구간을 기록한다."""
    freeze = fixed.read(fixed.OUT/'freeze.json')
    reference = fixed.read(fixed.OUT/'summary.json')['fixed']
    runs = fixed.test_runs()
    result = {}
    for method in ('nsd', 'bevplace'):
        result[method] = {}
        for point in ('P99', 'P100', 'F1'):
            per_seed = {}
            for seed in ('1', '2', '3'):
                tau = freeze['thresholds'][method][seed][point]['tau']
                seqs, events, tp, fp = {}, [], 0, 0
                bins = {'5_to_10_m': 0, '10_to_25_m': 0, 'above_25_m': 0}
                negative_fp = 0
                for sequence, (score, distance, qc) in runs[method][seed].items():
                    accept = (qc['positive'] | qc['negative']) & np.isfinite(score) & (score >= tau)
                    good = accept & qc['positive'] & (distance <= 5)
                    bad = accept & ~good
                    tp += int(good.sum()); fp += int(bad.sum())
                    negative_fp += int((bad & qc['negative']).sum())
                    bins['5_to_10_m'] += int((bad & (distance <= 10)).sum())
                    bins['10_to_25_m'] += int((bad & (distance > 10) & (distance <= 25)).sum())
                    bins['above_25_m'] += int((bad & (distance > 25)).sum())
                    if bad.any():
                        seqs[sequence] = dict(FP=int(bad.sum()), negative_query_FP=int((bad & qc['negative']).sum()))
                        if point == 'P99':
                            sensor, name = sequence.split('/')
                            with np.load(fixed.POWER/'test'/f's{seed}'/f'{sensor}_{name}_{fixed.ARM}.npz') as data:
                                q = data['query_idx']
                            events.extend(dict(sequence=sequence, query=int(q[i]), score=float(score[i]),
                                               returned_distance_m=float(distance[i]),
                                               query_kind='negative' if qc['negative'][i] else 'positive')
                                          for i in np.flatnonzero(bad))
                expected = reference[method][point]['per_seed'][seed]
                assert tp == expected['TP'] and fp == expected['FP']
                assert negative_fp == expected['negative_query_FP']
                assert sum(bins.values()) == fp
                per_seed[seed] = dict(tau=tau, TP=tp, FP=fp, negative_query_FP=negative_fp,
                                      precision=tp/(tp+fp) if tp+fp else 1., distance_bins=bins,
                                      sequences=seqs, P99_events=events)
            result[method][point] = per_seed
    fixed.write(OUT, dict(scope='검증에서 이미 고정한 원래 NSD/BEVPlace++ 기준값. 동일 시간 판정 BEV 대조와 구분.',
                         original_counts_exactly_reproduced=True, freeze_sha256=fixed.digest(fixed.OUT/'freeze.json'),
                         methods=result))
    for method in result:
        print(method, {seed: {k:v for k,v in row.items() if k != 'P99_events'}
                       for seed,row in result[method]['P99'].items()})


if __name__ == '__main__':
    main()
