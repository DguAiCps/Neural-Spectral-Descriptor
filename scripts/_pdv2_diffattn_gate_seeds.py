#!/usr/bin/env python3
"""시드1에서 선택한 게이트 학습을 재선택 없이 다른 두 초기 시드에 검증한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import _pdv2_diffattn_gate_learning as experiment
import _pdv2_diffattn_gate_balance as balance

run=experiment.run
PARENT=experiment.OUT
BALANCE=balance.OUT
OUT=PARENT/'three_seed_validation'
BEV=experiment.previous.BEV


def prepare() -> dict:
    """시험 전에 반복 시드 검증 범위와 통과 기준을 고정한다."""
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        spec=run.READ(path)
        for p,h in spec['sources'].items():assert run.HASH(ROOT/p)==h,p
        return spec
    chosen=run.READ(BALANCE/'validation_summary.json')
    selected=chosen['selected']
    assert selected is not None,'검증 확장 기준을 통과한 학습 후보가 없다'
    arm=selected['arm']
    origin=BALANCE if arm in balance.ARMS else PARENT
    checkpoint=origin/arm/f"epoch{selected['epoch']}.pth"
    main=run.READ(run.power.OUT/'protocol.json')
    initial={str(s):run.source.root(s,'octave288')/'checkpoints'/f"checkpoint_epoch_{main['models'][str(s)]}.pth" for s in (1,2,3)}
    paths=[Path(__file__),Path(balance.__file__),Path(experiment.__file__),
           PARENT/'validation_summary.json',BALANCE/'validation_summary.json',
           PARENT/'protocol.json',BALANCE/'protocol.json',checkpoint,*initial.values()]
    spec=dict(created_unix=time.time(),selected=selected,conv_control=chosen['best']['recenter_convs'],seed1_origin=str(origin.relative_to(ROOT)),
        initial={s:str(p.relative_to(ROOT)) for s,p in initial.items()},confidence=main['confidence'],
        frozen_coefficient=chosen['best'][experiment.mode(arm)+'_frozen']['coefficient'],
        rule='시드1 검증에서 고른 조건·epoch·추론 비중을 시드2·3에서 그대로 반복. conv만 학습한 대조도 시드1의 선택 epoch·비중으로 별도 반복하며 주후보를 바꾸는 데 쓰지 않음. 게이트 중심화와 점수 척도만 각 초기모델의 학습8로 동일하게 적합. 시드별 검증 재선택 없음. 학습 전은 같은 비중과 시드1 검증에서 선택한 공통 비중을 모두 보고.',
        advancement='3시드 검증 평균 AP가 같은비중 및 학습전 공통선택비중 모두보다+.1%p 이상, 같은비중 학습이득이 최소2시드에서 양수, BEV 동일시간 판정 대비 평균 AP/F1 -2%p 이내·R100/R1 -3%p 이내일 때만 시험 확장. 학습 설정 재선택 금지. 이는 탐색 확장 기준이며 통계적 동등성 아님.',
        test_used=False,sources={str(p.relative_to(ROOT)):run.HASH(p) for p in paths})
    run.WRITE(path,spec)
    return spec


def seed_prepare(seed: int) -> dict:
    """시드별 입력과 초기 가중치를 고정한다."""
    common=prepare()
    root=OUT/f's{seed}';root.mkdir(exist_ok=True)
    path=root/'protocol.json'
    if path.exists():return run.READ(path)
    spec=run.READ(PARENT/'protocol.json')
    spec.update(seed=seed,initial=common['initial'][str(seed)],confidence=common['confidence'][str(seed)],
        epochs=[common['selected']['epoch']],arms=[common['selected']['arm']],sources=common['sources'],
        fixed_replication=True,parent_sha256=run.HASH(OUT/'protocol.json'))
    run.WRITE(path,spec)
    pool=experiment.previous.OUT/'three_seed'/f's{seed}'/'pools'
    (root/'pools').symlink_to(pool,target_is_directory=True)
    return spec


def activate(seed: int, arm: str) -> dict:
    """원래 구현에 시드 경로와 선택 손실을 공급한다."""
    spec=seed_prepare(seed)
    experiment.OUT=OUT/f's{seed}'
    experiment.prepare=lambda:seed_prepare(seed)
    epoch=prepare()['conv_control']['epoch'] if arm=='recenter_convs' else spec['epochs'][0]
    experiment.EPOCHS=(epoch,)
    experiment.previous.PRIOR=experiment.previous.OUT/'three_seed'/f's{seed}'
    experiment.activate(arm)
    if arm=='recenter_balanced':
        run.acceptance_loss=lambda logits,correct,*a,**kw:spec['acceptance_gain']*balance.balanced_loss(logits,correct)
    elif arm=='recenter_nonrevisit':
        run.acceptance_loss=lambda logits,correct,positive,*a,**kw:spec['acceptance_gain']*experiment.LOSS(logits,correct,positive,True)
    return spec


def finish() -> None:
    """공통 비중의 3시드 검증 결과를 집계하고 시험 여부를 결정한다."""
    spec=prepare();selected=spec['selected']
    arm,epoch,coefficient=selected['arm'],selected['epoch'],selected['coefficient']
    frozen=experiment.mode(arm)+'_frozen'
    val=run.READ(PARENT/'protocol.json')['val']
    results={name:{} for name in ('learned','convs','frozen_same','frozen_retuned','bev_native','bev_temporal')}
    count=0
    for seed in (1,2,3):
        own=ROOT/spec['seed1_origin'] if seed==1 else OUT/f's{seed}'
        frozen_root=PARENT if seed==1 else own
        conv_root=PARENT if seed==1 else own
        conv=spec['conv_control']
        per={name:{} for name in results}
        for seq in val:
            name=seq.replace('/','_')
            reference=None
            for label,folder,weight in (
                ('learned',own/arm/f'epoch{epoch}/val',coefficient),
                ('convs',conv_root/'recenter_convs'/f"epoch{conv['epoch']}/val",conv['coefficient']),
                ('frozen_same',frozen_root/frozen/'epoch0/val',coefficient),
                ('frozen_retuned',frozen_root/frozen/'epoch0/val',spec['frozen_coefficient'])):
                path=folder/(name+'.npz');meta=run.READ(path.with_suffix('.json'));row=run.ARR(path)
                assert run.HASH(path)==meta['ledger_sha256'] and len(meta['prefix_checks'])==2
                metric=run.base.independent_metrics(row,row[f'lambda_{weight:g}'])
                assert metric==meta['metrics'][f'lambda_{weight:g}']
                per[label][seq]=metric;reference=row;count+=1
            bev=run.ARR(BEV/f'val/s{seed}'/(name+'.npz'))
            meta=run.READ(BEV/f'val/s{seed}'/(name+'.json'))
            assert run.HASH(BEV/f'val/s{seed}'/(name+'.npz'))==meta['sha256']
            for field in ('query_idx','min_past_gt_dist'):np.testing.assert_array_equal(bev[field],reference[field])
            for label,score in (('bev_native','score_negl2'),('bev_temporal','score')):
                per[label][seq]=run.base.independent_metrics(bev,bev[score])
        for label in results:results[label][str(seed)]=dict(**run.base.macro(per[label]),sequences=per[label])
    means={label:{m:float(np.mean([v['macro'][m] for v in seeds.values()])) for m in run.base.METRICS}
           for label,seeds in results.items()}
    deltas={s:results['learned'][s]['macro']['AP']-results['frozen_same'][s]['macro']['AP'] for s in ('1','2','3')}
    criteria={m:means['learned'][m]>=means['bev_temporal'][m]-tol
              for m,tol in {'AP':.02,'max_F1':.02,'R@100P':.03,'R@1':.03}.items()}
    criteria.update(over_same=means['learned']['AP']>=means['frozen_same']['AP']+.001,
                    over_retuned=means['learned']['AP']>=means['frozen_retuned']['AP']+.001,
                    positive_two_seeds=sum(d>0 for d in deltas.values())>=2)
    run.WRITE(OUT/'summary.json',dict(per_seed=results,mean=means,learned_ap_delta=deltas,
        advancement_checks=criteria,advance_to_test=all(criteria.values()),test_used=False))
    run.WRITE(OUT/'completion_validation.json',dict(status='passed',new_ledgers=54,
        metric_rows=count,prefix_checks=108,paired_labels=True,no_parameter_reselection=True,test_used=False))
    run.WRITE(OUT/'status.json',dict(status='complete',advance_to_test=all(criteria.values()),test_used=False))
    print('THREE_SEED_VALIDATION',means,'DELTA',deltas,'ADVANCE',all(criteria.values()),flush=True)


def campaign() -> None:
    spec=prepare();selected=spec['selected'];logs=OUT/'logs';logs.mkdir(exist_ok=True)
    def call(args: list,name: str) -> None:
        with (logs/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,__file__,*args],stdout=log,stderr=subprocess.STDOUT,check=True)
    for seed in (2,3):
        call(['train','--seed',str(seed)],f'train_s{seed}')
        call(['train','--seed',str(seed),'--arm','recenter_convs'],f'train_convs_s{seed}')
    val=run.READ(PARENT/'protocol.json')['val']
    jobs=[(s,a,e,q) for s in (2,3) for a,e in ((selected['arm'],selected['epoch']),('recenter_convs',spec['conv_control']['epoch']),(experiment.mode(selected['arm'])+'_frozen',0)) for q in val]
    def evaluate(job: tuple) -> None:
        s,a,e,q=job
        call(['val','--seed',str(s),'--arm',a,'--epoch',str(e),'--sequence',q],f"val_s{s}_{a}_{q.replace('/','_')}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for i,future in enumerate(concurrent.futures.as_completed([pool.submit(evaluate,j) for j in jobs]),1):
            future.result()
            run.WRITE(OUT/'status.json',dict(status='running',phase='validation',done=i,total=len(jobs)))
            print('VALIDATION',i,len(jobs),flush=True)
    finish()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','train','val','finish','run'))
    parser.add_argument('--seed',type=int,choices=(2,3),default=2)
    parser.add_argument('--arm',default=None)
    parser.add_argument('--epoch',type=int,default=4)
    parser.add_argument('--sequence',default='kitti/09')
    args=parser.parse_args();run.configure()
    if args.action=='prepare':prepare()
    elif args.action=='train':
        spec=prepare();arm=args.arm or spec['selected']['arm'];activate(args.seed,arm);experiment.setup();activate(args.seed,arm);run.train(arm)
    elif args.action=='val':activate(args.seed,args.arm);run.validate(args.arm,args.epoch,args.sequence)
    elif args.action=='finish':finish()
    else:campaign()
