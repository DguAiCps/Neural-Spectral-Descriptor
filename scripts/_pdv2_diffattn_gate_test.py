#!/usr/bin/env python3
"""반복 시드 검증을 통과한 고정 게이트 학습과 동결 대조를 시험한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import _pdv2_diffattn_gate_seeds as replication
import _pdv2_diffattn_comparable_test as engine

experiment=replication.experiment
run=replication.run
OUT=replication.PARENT/'fixed_test'


def prepare() -> dict:
    """3시드 검증 통과와 모든 선택 가중치·척도를 시험 전에 고정한다."""
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        spec=run.READ(path)
        for p,h in spec['sources'].items():assert run.HASH(ROOT/p)==h,p
        return spec
    common=replication.prepare()
    summary=run.READ(replication.OUT/'summary.json')
    assert summary['advance_to_test'],'3시드 검증 확장 기준 미달'
    selected=common['selected']
    roots={str(s):(ROOT/common['seed1_origin'] if s==1 else replication.OUT/f's{s}') for s in (1,2,3)}
    paths=[Path(__file__),Path(engine.__file__),replication.OUT/'protocol.json',replication.OUT/'summary.json',
           replication.OUT/'completion_validation.json']
    models={}
    for seed,root in roots.items():
        checkpoint=root/selected['arm']/f"epoch{selected['epoch']}.pth"
        models[seed]=dict(root=str(root.relative_to(ROOT)),checkpoint=str(checkpoint.relative_to(ROOT)),
                         checkpoint_sha256=run.HASH(checkpoint))
        paths += [checkpoint,root/'protocol.json',root/'gate_statistics.json',root/'calibration_recenter.json',
                  ROOT/common['initial'][seed]]
    spec=dict(created_unix=time.time(),selection=common,models=models,test=['/'.join(s) for s in run.current.TEST],
        policy='3시드 검증의 주선택을 그대로 시험46에 적용. 같은 중심화 초기모델과 같은 비중/검증선택비중 대조. 시험 결과로 설정을 고르지 않으며 원고 변경 없음.',
        sources={str(p.relative_to(ROOT)):run.HASH(p) for p in paths})
    run.WRITE(path,spec)
    return spec


def activate(seed: int) -> dict:
    """기존 시험 엔진에 고정 초기 시드와 문맥 점수만 공급한다."""
    spec=prepare();selected=spec['selection']['selected'];model=spec['models'][str(seed)]
    root=ROOT/model['root']
    experiment.OUT=root
    run.OUT=root
    run.prepare=lambda:run.READ(root/'protocol.json')
    run.model=lambda checkpoint=None:experiment.load_model(checkpoint,selected['arm'])
    run.context_score=experiment.gated_context_score
    run.calibration=lambda:run.READ(root/'calibration_recenter.json')
    selection=dict(selected=selected,selected_checkpoint=model['checkpoint'],checkpoint_sha256=model['checkpoint_sha256'],
        frozen_best_coefficient=spec['selection']['frozen_coefficient'],test=spec['test'])
    protocol=dict(selection=selection,parent_sha256=run.HASH(OUT/'protocol.json'))
    engine.OUT=OUT/f's{seed}'
    engine.OUT.mkdir(exist_ok=True)
    path=engine.OUT/'protocol.json'
    if path.exists():assert run.READ(path)==protocol
    else:run.WRITE(path,protocol)
    engine.experiment=SimpleNamespace(OUT=root)
    engine.prepare=lambda:protocol
    return spec


def preflight() -> None:
    """모든 시드의 시험 엔진이 먼저 검증 결과를 재현하는지 검사한다."""
    spec=prepare();selected=spec['selection']['selected'];seq='kitti360/0004'
    for seed in (1,2,3):
        activate(seed)
        engine.worker('val',seq)
        root=ROOT/spec['models'][str(seed)]['root']
        frozen=replication.PARENT if seed==1 else root
        meta=run.READ(OUT/f's{seed}/val/kitti360_0004.json')
        for arm,source in (('learned',root/selected['arm']/f"epoch{selected['epoch']}/val/kitti360_0004.npz"),
                           ('frozen',frozen/'recenter_frozen/epoch0/val/kitti360_0004.npz')):
            actual=run.ARR(ROOT/meta['records'][arm]['path']);expected=run.ARR(source)
            np.testing.assert_array_equal(actual['j_star'],expected['j_star'])
            np.testing.assert_allclose(actual['score'],expected[f"lambda_{selected['coefficient']:g}"],atol=2e-6,rtol=0)
    run.WRITE(OUT/'preflight.json',dict(status='passed',seeds=3,validation_sequence=seq,test_used=False))


def finish() -> None:
    """최종 시험 성능을 시퀀스·도시·시드 순서로 집계한다."""
    spec=prepare()
    results={k:{} for k in ('learned','frozen_same','frozen_retuned','bev_native','bev_temporal')}
    queries={}
    for seed in (1,2,3):
        per={k:{} for k in results};queries[str(seed)]=0
        for seq in spec['test']:
            name=seq.replace('/','_')
            meta=run.READ(OUT/f's{seed}/test'/(name+'.json'));assert meta['status']=='passed'
            queries[str(seed)]+=meta['queries']
            reference=None
            for arm in ('learned','frozen'):
                record=meta['records'][arm];path=ROOT/record['path']
                assert run.HASH(path)==record['sha256'] and len(record['prefix_checks'])==2
                row=run.ARR(path);reference=row
                for field,label in ([('score','learned')] if arm=='learned' else [('score','frozen_same'),('score_retuned','frozen_retuned')]):
                    metric=run.base.independent_metrics(row,row[field]);assert metric==record['metrics'][field]
                    per[label][seq]=metric
            path=replication.BEV/f'test/s{seed}'/(name+'.npz')
            bev=run.ARR(path);bmeta=run.READ(path.with_suffix('.json'))
            assert run.HASH(path)==bmeta['sha256']
            for field in ('query_idx','min_past_gt_dist'):np.testing.assert_array_equal(bev[field],reference[field])
            for label,field in (('bev_native','score_negl2'),('bev_temporal','score')):
                per[label][seq]=run.base.independent_metrics(bev,bev[field])
        for label in results:
            results[label][str(seed)]=dict(**run.current.adoption.previous.macro(per[label],run.current.CITY),sequences=per[label])
    mean={k:{m:float(np.mean([r['macro'][m] for r in v.values()])) for m in run.base.METRICS} for k,v in results.items()}
    std={k:{m:float(np.std([r['macro'][m] for r in v.values()],ddof=1)) for m in run.base.METRICS} for k,v in results.items()}
    delta={k:{m:100*(mean['learned'][m]-v[m]) for m in run.base.METRICS} for k,v in mean.items() if k!='learned'}
    run.WRITE(OUT/'summary.json',dict(per_seed=results,mean=mean,std=std,learned_delta_pp=delta,
        aggregation='시험46→양성45→도시6→초기시드3',test_used_for_selection=False))
    assert set(queries.values())=={195726}
    run.WRITE(OUT/'completion_validation.json',dict(status='passed',seeds=3,test_sequences=46,
        queries=queries,new_ledgers=276,prefix_checks=552,paired_labels=True,no_parameter_reselection=True))
    run.WRITE(OUT/'status.json',dict(status='complete'))
    print('TEST_RESULT',mean,'DELTA',delta,flush=True)


def campaign() -> None:
    spec=prepare();preflight();logs=OUT/'logs';logs.mkdir(exist_ok=True)
    def evaluate(job: tuple) -> None:
        seed,seq=job
        with (logs/f"s{seed}_{seq.replace('/','_')}.log").open('w') as log:
            subprocess.run([sys.executable,__file__,'worker','--seed',str(seed),'--sequence',seq],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    jobs=[(s,q) for s in (1,2,3) for q in spec['test']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for i,future in enumerate(concurrent.futures.as_completed([pool.submit(evaluate,j) for j in jobs]),1):
            future.result();run.WRITE(OUT/'status.json',dict(status='running',done=i,total=len(jobs)))
            print('TEST',i,len(jobs),flush=True)
    finish()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','preflight','worker','finish','run'))
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--sequence',default='kitti/00')
    args=parser.parse_args();run.configure()
    if args.action=='prepare':prepare()
    elif args.action=='preflight':preflight()
    elif args.action=='worker':activate(args.seed);engine.worker('test',args.sequence)
    elif args.action=='finish':finish()
    else:campaign()
