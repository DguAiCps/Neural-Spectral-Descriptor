#!/usr/bin/env python3
"""DiffAttnConv 두 층만 학습한 고정 보조 대조를 시험46에 적용한다."""
import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import _pdv2_diffattn_gate_test as primary

run=primary.run
replication=primary.replication
OUT=replication.PARENT/'convs_fixed_test'


def prepare() -> dict:
    """주후보 시험 성능을 보기 전에 conv만 대조의 설정을 고정한다."""
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        spec=run.READ(path)
        for p,h in spec['sources'].items():assert run.HASH(ROOT/p)==h,p
        return spec
    common=dict(replication.prepare())
    summary=run.READ(replication.OUT/'summary.json')
    assert all(summary['per_seed']['convs'][s]['macro']['AP']>summary['per_seed']['frozen_same'][s]['macro']['AP'] for s in ('1','2','3'))
    assert summary['mean']['convs']['AP']>summary['mean']['frozen_same']['AP']+.001
    common['selected']=common['conv_control']
    common['seed1_origin']=str(replication.PARENT.relative_to(ROOT))
    selected=common['selected']
    models={}
    paths=[Path(__file__),Path(primary.__file__),replication.OUT/'protocol.json',replication.OUT/'summary.json',
           replication.OUT/'checkpoint_checks.json',primary.OUT/'protocol.json']
    for seed in (1,2,3):
        root=replication.PARENT if seed==1 else replication.OUT/f's{seed}'
        checkpoint=root/'recenter_convs'/f"epoch{selected['epoch']}.pth"
        models[str(seed)]=dict(root=str(root.relative_to(ROOT)),checkpoint=str(checkpoint.relative_to(ROOT)),
                              checkpoint_sha256=run.HASH(checkpoint))
        paths += [checkpoint,root/'protocol.json',root/'gate_statistics.json',root/'calibration_recenter.json',
                  ROOT/common['initial'][str(seed)]]
    spec=dict(created_unix=time.time(),selection=common,models=models,test=['/'.join(s) for s in run.current.TEST],
        policy='주후보의 시험 결과를 읽기 전에 추가한 보조 확증. 3시드 검증에서 conv만 AP가 모두 향상되어 사용자 우선 목표인 DiffAttnConv만의 최종 개선을 분리한다. epoch8·비중.5 고정. 이 결과로 주후보를 바꾸지 않음.',
        primary_test_results_used=False,sources={str(p.relative_to(ROOT)):run.HASH(p) for p in paths})
    run.WRITE(path,spec)
    return spec


def activate() -> None:
    """같은 평가 엔진의 결과 폴더와 고정 후보만 분리한다."""
    primary.OUT=OUT
    primary.prepare=prepare


def campaign() -> None:
    spec=prepare();activate();primary.preflight()
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    jobs=[(s,q) for s in (1,2,3) for q in spec['test']]
    def worker(job: tuple) -> None:
        seed,seq=job
        with (logs/f"s{seed}_{seq.replace('/','_')}.log").open('w') as log:
            subprocess.run([sys.executable,__file__,'worker','--seed',str(seed),'--sequence',seq],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for i,future in enumerate(concurrent.futures.as_completed([pool.submit(worker,j) for j in jobs]),1):
            future.result();run.WRITE(OUT/'status.json',dict(status='running',done=i,total=len(jobs)))
            print('CONV_TEST',i,len(jobs),flush=True)
    primary.finish()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','worker','finish','run'))
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--sequence',default='kitti/00')
    args=parser.parse_args();run.configure()
    if args.action=='prepare':prepare()
    elif args.action=='worker':activate();primary.activate(args.seed);primary.engine.worker('test',args.sequence)
    elif args.action=='finish':activate();primary.finish()
    else:campaign()
