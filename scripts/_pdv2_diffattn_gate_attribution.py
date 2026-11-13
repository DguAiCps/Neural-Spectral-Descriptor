#!/usr/bin/env python3
"""포화 완화 동결 모델의 문맥 방향과 게이트 강도를 검증에서 분리한다."""
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import _pdv2_diffattn_gate_learning as experiment
from retrieval.nonrevisit_learning import context_score


def scores(key: torch.Tensor, plan: dict) -> dict:
    """같은 대응에서 방향만·강도만·둘의 결합을 계산한다."""
    alpha=key[:,288:].norm(dim=-1)/key[:,:288].norm(dim=-1).clamp_min(1e-12)
    products=alpha[plan['query']]*alpha[plan['candidate']]
    return dict(direction=context_score(key,plan),
        strength=(products*plan['weight']).sum(-1).max(-1).values,
        combined=experiment.gated_context_score(key,plan))


def main() -> None:
    """같은 동결 후보의 판정 성분만 분리하며 시험은 읽지 않는다."""
    run=experiment.run
    run.configure()
    spec=experiment.activate('recenter_frozen')
    out=experiment.OUT/'attribution';out.mkdir(exist_ok=True)
    checkpoint=experiment.OUT/'recenter_frozen/epoch0.pth'
    protocol=dict(created_unix=time.time(),model=str(checkpoint.relative_to(ROOT)),
        definition='포화 완화 동결 모델의 같은 후보에서 방향 코사인만 / 게이트 곱만 / 곱한 문맥 점수를 비교. 각 척도는 학습8의 같은 초기 승자로 적합. 검증9는 성분 진단이며 시험이나 주모델 선택을 하지 않음.',
        sources={str(p.relative_to(ROOT)):run.HASH(p) for p in (Path(__file__),checkpoint,experiment.OUT/'protocol.json')})
    target=out/'protocol.json'
    if target.exists():
        for p,h in run.READ(target)['sources'].items():assert run.HASH(ROOT/p)==h,p
    else:run.WRITE(target,protocol)
    net=run.model(checkpoint)
    parts={k:[] for k in ('direction','strength','combined')};weights=[]
    for seq in spec['train']:
        folder=experiment.OUT/'pools'/seq.replace('/','_')
        pool=run.ARR(folder/'pool.npz')
        graph=torch.load(folder/'graph.pt',map_location='cuda',weights_only=False)
        with torch.no_grad():
            key=F.normalize(net(graph),dim=1)
            plan={k:torch.from_numpy(pool['plan_'+k]).cuda() for k in ('query','candidate','weight')}
            batches=[scores(key,{k:v[i:i+256] for k,v in plan.items()}) for i in range(0,len(plan['query']),256)]
            for name in parts:parts[name].append(torch.cat([v[name] for v in batches]).cpu().numpy().astype(np.float64))
        weights.append(np.full(len(plan['query']),1./len(plan['query'])))
        del graph,key,plan,batches
    w=np.concatenate(weights);w/=w.sum()
    calibration={}
    for name,values in parts.items():
        x=np.concatenate(values);mean=float(w@x)
        calibration[name]=dict(mean=mean,scale=max(float(np.sqrt(w@((x-mean)**2))),.05))
    fit=run.calibration()
    np.testing.assert_allclose([calibration['combined']['mean'],calibration['combined']['scale']],
                              [fit['context_mean'],fit['context_scale']],atol=1e-9,rtol=0)
    per={name:{l:{} for l in run.LAMBDAS} for name in parts}
    checks=0
    for seq in spec['val']:
        name=seq.replace('/','_')
        row=run.ARR(experiment.OUT/'recenter_frozen/epoch0/val'/(name+'.npz'))
        cache,_,_,travel,*_=run.inputs('val',seq)
        key=torch.from_numpy(run.source.make_keys(net,cache['descriptors'],cache['poses']))
        q,j=row['query_idx'],row['j_star']
        values={k:[] for k in parts}
        with torch.no_grad():
            for i in range(0,len(q),256):
                plan={k:torch.from_numpy(v) for k,v in run.context_plan(travel,q[i:i+256],j[i:i+256]).items()}
                batch=scores(key,plan)
                for k in values:values[k].append(batch[k].numpy())
        values={k:np.concatenate(v).astype(np.float64) for k,v in values.items()}
        np.testing.assert_allclose(values['combined'],row['context'],atol=2e-6,rtol=0)
        base=run.predict_score(row['spectral'][:,None],fit['spectral'])
        for k,x in values.items():
            residual=(x-calibration[k]['mean'])/calibration[k]['scale']
            for l in run.LAMBDAS:
                per[k][l][seq]=run.base.independent_metrics(row,base+l*residual)
                if k=='combined':
                    np.testing.assert_allclose(base+l*residual,row[f'lambda_{l:g}'],atol=2e-6,rtol=0);checks+=1
    results={k:{str(l):dict(**run.base.macro(v),sequences=v) for l,v in by.items()} for k,by in per.items()}
    best={k:max(v,key=lambda l:v[l]['macro']['AP']) for k,v in results.items()}
    run.WRITE(out/'summary.json',dict(results=results,best=best,calibration=calibration,
        combined_reproduction_checks=checks,fixed_candidate=True,test_used=False))
    print('ATTRIBUTION',{k:(l,results[k][l]['macro']) for k,l in best.items()},flush=True)


if __name__=='__main__':main()
