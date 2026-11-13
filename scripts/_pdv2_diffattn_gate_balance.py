#!/usr/bin/env python3
"""포화 완화 게이트의 점수 상승을 비재방문·정오 균형 학습으로 대조한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import _pdv2_diffattn_gate_learning as parent

run = parent.run
PARENT = parent.OUT
OUT = ROOT / "results/pdv2_diffattn_gate_balance_20261104"
ARMS = ("recenter_nonrevisit", "recenter_balanced")
ACTIVATE = parent.activate


def balanced_loss(logits: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """정답·오답 승자 두 집단의 평균 BCE에 같은 비중을 준다."""
    loss = F.binary_cross_entropy_with_logits(logits, correct.to(logits.dtype), reduction="none")
    terms = [loss[mask].mean() for mask in (correct.bool(), ~correct.bool()) if mask.any()]
    return torch.stack(terms).mean()


def prepare() -> dict:
    """학습 자료의 게이트 상승 진단에 근거한 후속 대조를 고정한다."""
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/"protocol.json"
    if target.exists():
        spec=run.READ(target)
        for path,digest in spec["sources"].items():
            assert run.HASH(ROOT/path)==digest,path
        return spec
    spec=run.READ(PARENT/"protocol.json")
    paths=[ROOT/p for p in spec["sources"]]
    paths += [Path(__file__),PARENT/"protocol.json",PARENT/"gate_statistics.json",
              PARENT/"calibration_recenter.json",PARENT/"training_diagnostic.json"]
    spec.update(created_unix=time.time(),arms=ARMS,
        objective="게이트 포화 완화·전체 문맥 모듈에서 모든 적격 질의 BCE / 정답·오답 승자 집단의 동일 비중 BCE를 비교. 초기 모델·게이트 보정·학습8 쌍·배치·512갱신·학습률·BCE계수·추론 비중은 부모와 같음. 4·8epoch 검증.",
        trigger="부모의 검증 전체 결과를 보기 전에 학습8 고정 승자의 진단에서 정답 BCE 감소와 비재방문 BCE 상승을 확인해 추가. 검증에서는 원래 양성 질의 전용·동결·conv만·gate만 대조를 함께 보고 선택 기회 증가를 명시.",
        selection="부모와 같은 검증 확장 기준. 부모 조건까지 포함해 최고AP를 보고. 후속 학습은 전체 모듈 효과이며 conv 고유 우위로 해석하지 않음. 단일 검증 선택 후 반복 시드 검증을 먼저 확인하며 시험은 조건 고정 후에만 사용.",
        sources={str(p.relative_to(ROOT)):run.HASH(p) for p in paths})
    run.WRITE(target,spec)
    for name in ("pools","gate_statistics.json","calibration_native.json","calibration_recenter.json",*parent.FROZEN):
        (OUT/name).symlink_to(PARENT/name,target_is_directory=(name=="pools" or name in parent.FROZEN))
    return spec


def activate(arm: str) -> dict:
    """점수·모델은 부모와 같고 수락 손실의 질의 가중치만 바꾼다."""
    parent.OUT=OUT
    parent.prepare=prepare
    spec=ACTIVATE(arm)
    if arm=="recenter_balanced":
        run.acceptance_loss=lambda logits,correct,*a,**kw: spec["acceptance_gain"]*balanced_loss(logits,correct)
    else:
        run.acceptance_loss=lambda logits,correct,positive,*a,**kw: spec["acceptance_gain"]*parent.LOSS(logits,correct,positive,True)
    return spec


def finish() -> None:
    """검증 최종 성능과 학습 전 대조의 차이를 재검산한다."""
    spec=prepare()
    original=run.READ(PARENT/"validation_summary.json")
    results=dict(original["all_results"])
    checks=0
    for arm in ARMS:
        for epoch in parent.EPOCHS:
            per={lam:{} for lam in run.LAMBDAS}
            for seq in spec["val"]:
                path=OUT/arm/f"epoch{epoch}/val"/(seq.replace('/','_')+'.json')
                meta,row=run.READ(path),run.ARR(path.with_suffix('.npz'))
                assert meta['status']=='passed' and len(meta['prefix_checks'])==2
                assert run.HASH(path.with_suffix('.npz'))==meta['ledger_sha256']
                assert run.HASH(OUT/arm/f'epoch{epoch}.pth')==meta['checkpoint_sha256']
                for lam in run.LAMBDAS:
                    label=f'lambda_{lam:g}'
                    metric=run.base.independent_metrics(row,row[label])
                    assert metric==meta['metrics'][label]
                    per[lam][seq]=metric; checks+=1
            for lam in run.LAMBDAS:
                results[f'{arm}_{epoch}_{lam:g}']=dict(arm=arm,epoch=epoch,coefficient=lam,
                    **run.base.macro(per[lam]),sequences=per[lam])
    order=lambda v:(-round(v['macro']['AP'],12),v['coefficient'],v['epoch'],v['arm'])
    best={a:sorted([r for r in results.values() if r['arm']==a],key=order)[0]
          for a in (*parent.FROZEN,*parent.ARMS,*ARMS)}
    frozen=max(original['frozen_best']['macro']['AP'],original['prior_normalized_frozen']['macro']['AP'])
    eligible,tests=[],{}
    for name,row in results.items():
        if row['arm'] in parent.FROZEN: continue
        same=results[f"{parent.mode(row['arm'])}_frozen_0_{row['coefficient']:g}"]
        criteria={m:row['macro'][m]>=original['bev']['temporal']['macro'][m]-tol
                  for m,tol in {'AP':.02,'max_F1':.02,'R@100P':.03,'R@1':.03}.items()}
        criteria.update(over_frozen=row['macro']['AP']>=frozen+.001,
                        over_same=row['macro']['AP']>=same['macro']['AP']+.001)
        tests[name]=criteria
        if all(criteria.values()):eligible.append(row)
    selected=sorted(eligible,key=order)[0] if eligible else None
    training=[run.READ(OUT/a/'training_complete.json') for a in ARMS]
    reference=run.READ(PARENT/'recenter_all/training_complete.json')
    reference_order=[r['batch_hashes'] for r in reference['records']]
    assert all([r['batch_hashes'] for r in t['records']]==reference_order for t in training)
    run.WRITE(OUT/'validation_summary.json',dict(all_results=results,best=best,selected=selected,
        advancement_checks=tests,bev=original['bev'],frozen_best=original['frozen_best'],
        prior_normalized_frozen=original['prior_normalized_frozen'],test_used=False))
    run.WRITE(OUT/'validation_checks.json',dict(status='passed',new_ledgers=36,metric_rows=checks,
        prefix_checks=72,equal_training_samples=True,test_used=False))
    run.WRITE(OUT/'status.json',dict(status='validation_complete',selected=selected,test_used=False))
    print('RESULT',{a:v['macro'] for a,v in best.items()},'SELECTED',None if selected is None else selected['arm'],flush=True)


def campaign() -> None:
    """추가 두 학습과 검증을 완료한다."""
    spec=prepare()
    logs=OUT/'logs'; logs.mkdir(exist_ok=True)
    def call(args: list,name: str) -> None:
        with (logs/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,__file__,*args],stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        for arm in ARMS:
            run.WRITE(OUT/'status.json',dict(status='running',phase='training',arm=arm))
            call(['train','--arm',arm],'train_'+arm)
        jobs=[(a,e,s) for a in ARMS for e in parent.EPOCHS for s in spec['val']]
        def evaluate(job: tuple) -> None:
            a,e,s=job
            call(['val','--arm',a,'--epoch',str(e),'--sequence',s],f"val_{a}_{e}_{s.replace('/','_')}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for i,future in enumerate(concurrent.futures.as_completed([pool.submit(evaluate,j) for j in jobs]),1):
                future.result()
                run.WRITE(OUT/'status.json',dict(status='running',phase='validation',done=i,total=len(jobs)))
                print('VALIDATION',i,len(jobs),flush=True)
        finish()
    except Exception as exc:
        run.WRITE(OUT/'status.json',dict(status='failed',error=str(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','train','val','finish','run'))
    parser.add_argument('--arm',choices=ARMS,default=ARMS[0])
    parser.add_argument('--epoch',type=int,default=8)
    parser.add_argument('--sequence',default='kitti/09')
    args=parser.parse_args(); run.configure()
    if args.action=='prepare':prepare()
    elif args.action=='train':activate(args.arm);run.train(args.arm)
    elif args.action=='val':activate(args.arm);run.validate(args.arm,args.epoch,args.sequence)
    elif args.action=='finish':finish()
    else:campaign()
