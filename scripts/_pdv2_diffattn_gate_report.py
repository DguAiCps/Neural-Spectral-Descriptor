#!/usr/bin/env python3
"""게이트 학습의 고정 시험 결과와 같은 후보 대조를 통합 검산한다."""
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import _pdv2_diffattn_gate_learning as experiment


def main() -> None:
    """시험을 선택에 사용하지 않고 완료된 두 대조의 차이를 보고한다."""
    run=experiment.run
    root=experiment.OUT
    whole=run.READ(root/'fixed_test/summary.json')
    conv=run.READ(root/'convs_fixed_test/summary.json')
    for folder in ('fixed_test','convs_fixed_test'):
        assert run.READ(root/folder/'completion_validation.json')['status']=='passed'
    labels={'whole':whole['per_seed']['learned'],'convs':conv['per_seed']['learned'],
            'frozen':whole['per_seed']['frozen_same'],'bev_native':whole['per_seed']['bev_native'],
            'bev_temporal':whole['per_seed']['bev_temporal']}
    baseline_checks={}
    for name in ('frozen_same','frozen_retuned','bev_native','bev_temporal'):
        for seed in ('1','2','3'):
            a,b=whole['per_seed'][name][seed]['macro'],conv['per_seed'][name][seed]['macro']
            baseline_checks[f'{name}_s{seed}']={m:float(b[m]-a[m]) for m in run.base.METRICS}
            if name.startswith('bev'):assert a==b
    per_seed_delta={}
    city_delta={}
    intervals={}
    winner_diagnostic={}
    for name in ('whole','convs'):
        own=labels[name]
        controls=whole if name=='whole' else conv
        frozen=controls['per_seed']['frozen_same']
        per_seed_delta[name]={s:{m:100*(own[s]['macro'][m]-frozen[s]['macro'][m]) for m in run.base.METRICS} for s in own}
        city_delta[name]={}
        for city,sequences in run.current.CITY.items():
            city_delta[name][city]={m:float(np.mean([100*(own[s]['sequences'][q][m]-frozen[s]['sequences'][q][m])
                for s in own for q in sequences if own[s]['sequences'][q]['n_pos']])) for m in run.base.METRICS}
        for reference in ('frozen','bev_native','bev_temporal'):
            other=frozen if reference=='frozen' else labels[reference]
            rng=np.random.default_rng(20261104)
            draws=[]
            for city,sequences in run.current.CITY.items():
                available=[q for q in sequences if own['1']['sequences'][q]['n_pos']]
                matrix=np.array([[np.mean([100*(own[s]['sequences'][q][m]-other[s]['sequences'][q][m]) for s in own])
                                  for m in run.base.METRICS] for q in available])
                index=rng.integers(0,len(available),size=(10000,len(available)))
                draws.append(matrix[index].mean(1))
            boot=np.stack(draws).mean(0)
            intervals[f'{name}_vs_{reference}']={m:np.quantile(boot[:,i],[.025,.975]).tolist() for i,m in enumerate(run.base.METRICS)}
        folder=root/('fixed_test' if name=='whole' else 'convs_fixed_test')
        aggregates={}
        for city,sequences in run.current.CITY.items():
            count=same_count=gained=lost=0
            shifts={'correct':[],'wrong_place':[],'nonrevisit':[]}
            for seed in ('1','2','3'):
                for seq in sequences:
                    meta=run.READ(folder/f's{seed}/test'/(seq.replace('/','_')+'.json'))
                    a=run.ARR(ROOT/meta['records']['learned']['path'])
                    b=run.ARR(ROOT/meta['records']['frozen']['path'])
                    same=a['j_star']==b['j_star']
                    count+=len(same);same_count+=int(same.sum())
                    positive=a['min_past_gt_dist']<=5
                    ca=positive&(a['top1_geo_dist']<=5)
                    cb=positive&(b['top1_geo_dist']<=5)
                    gained+=int((ca&~cb).sum());lost+=int((cb&~ca).sum())
                    np.testing.assert_allclose(a['spectral'][same],b['spectral'][same],atol=2e-6,rtol=0)
                    for label,keep in (('correct',same&ca),('wrong_place',same&positive&~ca),
                                       ('nonrevisit',same&(a['min_past_gt_dist']>25))):
                        shifts[label].append((a['score']-b['score'])[keep])
            aggregates[city]=dict(queries=count,same_winners=same_count,gained_correct=gained,lost_correct=lost,
                same_winner_score_shift={k:dict(count=sum(len(x) for x in v),mean=float(np.concatenate(v).mean()) if sum(len(x) for x in v) else None) for k,v in shifts.items()})
        winner_diagnostic[name]=aggregates
    paths=[Path(__file__),root/'fixed_test/summary.json',root/'convs_fixed_test/summary.json',root/'three_seed_validation/summary.json']
    report=dict(whole=whole['mean'],convs=conv['mean'],whole_delta=whole['learned_delta_pp'],
        convs_delta=conv['learned_delta_pp'],per_seed_learning_delta_pp=per_seed_delta,city_learning_delta_pp=city_delta,
        baseline_reproduction_difference=baseline_checks,conditional_bootstrap_95_pp=intervals,
        winner_diagnostic=winner_diagnostic,
        interval_scope='고정3초기시드 평균의 도시별 짝지은 시퀀스 재표집10000회. 학습 실행 불확실성·개발 노출·다중 비교를 해결하지 않는 탐색적 조건부 구간. 비유의 차이로 동등성을 주장하지 않음.',
        test_used_for_selection=False,sources={str(p.relative_to(ROOT)):run.HASH(p) for p in paths})
    run.WRITE(root/'integrated_report.json',report)
    print('INTEGRATED',report['per_seed_learning_delta_pp'],report['conditional_bootstrap_95_pp'],flush=True)


if __name__=='__main__':main()
