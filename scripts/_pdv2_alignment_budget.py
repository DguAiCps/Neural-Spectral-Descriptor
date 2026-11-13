#!/usr/bin/env python3
"""검증에서 정렬 후보 예산과 신경망 기여를 교차한다. 시험·재학습은 하지 않는다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import _review_response_nsd as lib
from retrieval.sequence_evidence import neighbor_plans, path_means, repeated_candidate_weights

ROOT=lib.ROOT
OUT=ROOT/'results/pdv2_alignment_budget_20261104'
BUDGETS=(1,5,20)
MODELS=('learned','no_neural')
READ,WRITE,ARR,HASH=lib.READ,lib.WRITE,lib.ARR,lib.HASH
base,cal,power,white=lib.base,lib.cal,lib.power,lib.white


def prepare() -> dict:
    """모든 예산과 동일 검증 선택 규칙을 먼저 고정한다."""
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        spec=READ(path)
        for p,h in spec['sources'].items(): assert HASH(ROOT/p)==h,p
        return spec
    sources=[Path(__file__),Path(lib.__file__),Path(cal.__file__),Path(lib.adoption.__file__),
             ROOT/'src/retrieval/sequence_evidence.py',power.OUT/'protocol.json',white.OUT/'statistics.npz']
    spec=dict(created_unix=time.time(),seeds=[1,2,3],validation=['/'.join(s) for s in lib.VAL],
              models=MODELS,budgets=BUDGETS,full_budget=200,fusions=cal.FUSIONS,
              shortlist='같은 100+100 합집합에서 키/보조의 minmax 점수를 주모델 aux_half 가중치로 합해 상위 L. 동률 작은 ID. 정렬은 L개에만 적용.',
              ranking='단축 목록 안에서 네 점수를 다시 minmax한 뒤 기존 9개 융합 각각으로 승자 선정.',
              final='압축0.5·상관보정25%·두채널·같은 시간 판정 유지. 별도 no_alignment 행은 L1 승자의 전체 검색키 코사인 시간 판정만 사용.',
              comparison='288D 재계량 키 대 기존 416D 학습 키. 학습 없는 보정 통계는 공통. 모델·후보예산별 동일9개 융합 검증 평균AP 최대 선택, 동률 목록순. 선택과 평가는 같은 반복 검증이므로 탐색 결과다.',
              scope='검증9·3시드. 시험 없음, 재학습 없음, 원고 주모델 자동 교체 없음. 계산한 쌍 수는 연산량 지표이며 실측시간 아님.',
              advancement='학습 유무의 격차뿐 아니라 학습 조건의 절대AP/F1/R100/R1와 정렬쌍 수를 함께 보고. 성능을 일부러 낮춰 유리한 대조만 선택하지 않음.',
              sources={str(p.relative_to(ROOT)):HASH(p) for p in sources})
    WRITE(path,spec);return spec


def shortlist_winners(key: np.ndarray, bank: tuple, weights: np.ndarray, q: int) -> tuple:
    """최대 20후보의 정렬만 계산해 세 예산의 9개 융합 승자를 구한다."""
    ids=np.unique(np.r_[base.prefix_topk(key,q,q-29,100),base.prefix_topk(bank[0],q,q-29,100)])
    key_value=key[ids]@key[q];aux_value=bank[0][ids]@bank[0][q]
    pre=weights[cal.FUSIONS.index('aux_half'),0]*base.minmax(key_value)+weights[cal.FUSIONS.index('aux_half'),1]*base.minmax(aux_value)
    order=np.lexsort((ids,-pre))[:max(BUDGETS)]
    chosen=ids[order]
    spectral=base.align_scores(bank[1],bank[2],q,chosen)
    column=base.column_align_scores(bank[3],q,chosen)
    winners=[]
    for budget in BUDGETS:
        take=np.argsort(chosen[:budget],kind='stable')
        values=np.stack([base.minmax(v[:budget][take]) for v in (key_value[order],aux_value[order],spectral,column)],axis=1)
        score=values@weights.T
        winners.append(chosen[:budget][take][np.argmax(score,axis=0)])
    return np.concatenate(winners),len(ids),min(len(ids),max(BUDGETS))


def key_temporal(key: np.ndarray, travel: np.ndarray, q: np.ndarray, j: np.ndarray) -> np.ndarray:
    """정렬 없는 조건은 같은 거리 대응으로 전체 키 코사인을 집계한다."""
    values=[]
    for start in range(0,len(q),256):
        sl=slice(start,start+256)
        center=np.sum(key[q[sl]]*key[j[sl]],axis=1).astype(np.float64)
        qn,jn,valid=neighbor_plans(travel,q[sl],j[sl])['arc_unique']
        pair=np.zeros(qn.shape,np.float32)
        pair[valid]=np.sum(key[qn[valid]]*key[jn[valid]],axis=1)
        means=path_means(center,pair,valid,repeated_candidate_weights(jn,valid))
        beta=11/15
        values.append((1-beta)*center+beta*means.max(0))
    return np.concatenate(values)


def worker(seed: int, sequence: str) -> None:
    """한 시퀀스에서 두 표현과 모든 예산을 최종 지표까지 평가한다."""
    spec=prepare();name=sequence.replace('/','_');folder=OUT/f's{seed}'
    folder.mkdir(exist_ok=True);target=folder/(name+'.json')
    if target.exists(): assert READ(target)['status']=='passed';return
    tick=time.time();cfg=READ(power.OUT/'protocol.json')
    refpath=ROOT/'results/paper_fixed_threshold_20261101/nsd_val'/f's{seed}'/(name+'.npz')
    ref=ARR(refpath);q=ref['query_idx']
    _,_,paths,_=white.prior.BASE.exp.old.base.inputs('val',sequence)
    cache=ARR(paths['cache']);pos=cache['poses_gt'][:,:3,3];xyz=cache['poses'][:,:3,3]
    travel=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
    keypath=base.source_folder(seed,'val',cfg)/('keys_'+name+'.npz');keys=ARR(keypath)
    np.testing.assert_array_equal(keys['scan_ids'],cache['scan_ids'])
    raw=cache['descriptors'].astype(np.float32)*np.load(ROOT/'results/pdv2_spectral_keys_20261025/remetric_octave288.npy').astype(np.float32)
    raw/=np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12)
    x=white.coefficients(paths['spec']);wx=white.transform(x,ARR(white.OUT/'statistics.npz')['full'])
    bank=base.variant_inputs(x,8);nb,nc=[base.variant_inputs(power.transform(v,.5),8) for v in (x,wx)]
    weights=cal.fusion_weights(cfg['confidence'][str(seed)])
    records={};counts={};checks=[]
    def save(arm: str, ids: np.ndarray, scores: np.ndarray) -> None:
        row=dict(ref,j_star=ids,top1_geo_dist=np.linalg.norm(pos[q]-pos[ids],axis=1))
        assert (ids<=q-30).all() and np.isfinite(scores).all()
        path=folder/(name+'_'+arm+'.npz')
        np.savez_compressed(path,**{k:row[k] for k in base.FIELDS},score=scores)
        metric=base.independent_metrics(row,scores)
        check=base.white.prior.BASE.exp.metrics(scores,row)
        for m in base.METRICS:
            if metric[m] is not None: np.testing.assert_allclose(metric[m],check[m],rtol=0,atol=1e-12)
        records[arm]=dict(path=str(path.relative_to(ROOT)),sha256=HASH(path),metrics=metric)
    for model,key in (('learned',keys['embedding']),('no_neural',raw)):
        chosen=[];union=[]
        for t in q:
            ids,n,_=shortlist_winners(key,bank,weights,int(t));chosen.append(ids);union.append(n)
        ids=np.stack(chosen)
        pair,inv=np.unique((q[:,None]*len(key)+ids).ravel(),return_inverse=True)
        score=lib.adoption.scores(nb,nc,travel,pair//len(key),pair%len(key))['score'][inv].reshape(ids.shape)
        if model=='learned':
            full=ARR(cal.OUT/'val'/f's{seed}'/(name+'.npz'))
            power_values=ARR(power.OUT/'val'/f's{seed}'/(name+'.npz'))
            full_ids=full['candidates'];full_scores=cal.confidence(power_values['p050_c_spec'],power_values['p050_c_col'],'equal','c_col')
        else:
            columns=[ARR(lib.OUT/'val'/f's{seed}'/(name+'_no_context__'+f+'.npz')) for f in cal.FUSIONS]
            full_ids=np.stack([v['j_star'] for v in columns],axis=1);full_scores=np.stack([v['score'] for v in columns],axis=1)
        for fi,fusion in enumerate(cal.FUSIONS):
            save(model+'__L200__'+fusion,full_ids[:,fi],full_scores[:,fi])
        for bi,budget in enumerate(BUDGETS):
            for fi,fusion in enumerate(cal.FUSIONS):
                col=bi*len(cal.FUSIONS)+fi
                save(model+f'__L{budget}__'+fusion,ids[:,col],score[:,col])
        noalign=key_temporal(key,travel,q,ids[:,0])
        save(model+'__no_alignment__key_temporal',ids[:,0],noalign)
        for at in (len(q)//3,2*len(q)//3):
            t=int(q[at]);end=t+16
            np.testing.assert_array_equal(cal.candidates(key,bank,weights,t),full_ids[at])
            np.testing.assert_array_equal(shortlist_winners(key[:end],lib.adoption.previous.source.base.exp.truncate_bank(bank,end),weights,t)[0],ids[at])
            short=[lib.adoption.previous.source.base.exp.truncate_bank(b,end) for b in (nb,nc)]
            np.testing.assert_array_equal(lib.adoption.scores(*short,travel[:end],np.full(ids.shape[1],t),ids[at])['score'],score[at])
            np.testing.assert_array_equal(key_temporal(key[:end],travel[:end],q[at:at+1],ids[at:at+1,0]),noalign[at:at+1])
            checks.append(dict(model=model,query=t,full_replay=True,prefix_exact=True))
        counts[model]=dict(full_mean=float(np.mean(union)),shortlist_mean={str(b):float(np.mean(np.minimum(union,b))) for b in BUDGETS})
        print('MODEL_DONE',seed,sequence,model,round(time.time()-tick,1),flush=True)
    WRITE(target,dict(status='passed',seed=seed,sequence=sequence,records=records,alignment_candidate_counts=counts,checks=checks,
                      protocol_sha256=HASH(OUT/'protocol.json'),elapsed_s=time.time()-tick,
                      inputs={str(p.relative_to(ROOT)):HASH(p) for p in (refpath,keypath,paths['cache'],paths['spec'])}))


def aggregate() -> dict:
    """각 조건의 같은 9융합 선택과 고정 융합을 함께 보고한다."""
    spec=prepare();all_rows={};source=[]
    for seed in spec['seeds']:
        for seq in spec['validation']:
            record=READ(OUT/f's{seed}'/(seq.replace('/','_')+'.json'));source.append(record)
            for arm,entry in record['records'].items():
                assert HASH(ROOT/entry['path'])==entry['sha256']
                all_rows.setdefault(arm,{}).setdefault(str(seed),{})[seq]=entry['metrics']
    results={}
    for arm,seeds in all_rows.items():
        per={s:base.macro(rows) for s,rows in seeds.items()}
        results[arm]=dict(per_seed=per,mean={m:float(np.mean([v['macro'][m] for v in per.values()])) for m in base.METRICS})
    selected={}
    for model in MODELS:
        for budget in (200,*BUDGETS):
            arms=[model+f'__L{budget}__'+f for f in cal.FUSIONS]
            best=max(arms,key=lambda a:results[a]['mean']['AP'])
            selected[model+f'__L{budget}']=dict(arm=best,**results[best],fixed=results[model+f'__L{budget}__aux_half'])
    summary=dict(selected=selected,all_arms=results,scope=spec['scope'],test_used=False,
                 alignment_counts={model:{'full_mean':float(np.mean([r['alignment_candidate_counts'][model]['full_mean'] for r in source])),
                  'shortlist_means':{str(b):float(np.mean([r['alignment_candidate_counts'][model]['shortlist_mean'][str(b)] for r in source])) for b in BUDGETS}} for model in MODELS})
    WRITE(OUT/'summary.json',summary);return summary


def run(workers: int) -> None:
    """시퀀스별 독립 프로세스로 검증만 완료한다."""
    spec=prepare();logs=OUT/'logs';logs.mkdir(exist_ok=True)
    jobs=[(seed,seq) for seed in spec['seeds'] for seq in spec['validation']]
    def call(job: tuple) -> None:
        seed,seq=job
        with (logs/f's{seed}_{seq.replace("/","_")}.log').open('w') as f:
            subprocess.run([sys.executable,__file__,'worker','--seed',str(seed),'--sequence',seq],stdout=f,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(concurrent.futures.as_completed([pool.submit(call,j) for j in jobs]),1):
            f.result();WRITE(OUT/'status.json',dict(status='running',done=i,total=len(jobs)));print('PROGRESS',i,len(jobs),flush=True)
    aggregate();WRITE(OUT/'status.json',dict(status='complete',jobs=len(jobs),test_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','worker','run','aggregate'));p.add_argument('--seed',type=int,default=1);p.add_argument('--sequence',default='kitti/09');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.action=='worker':worker(a.seed,a.sequence)
    elif a.action=='run':run(a.workers)
    elif a.action=='aggregate':aggregate()
    else:prepare()
