#!/usr/bin/env python3
"""제외 장소의 학습 접근·학습량 대조·최종 원장과 적합 점수를 검산한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src'),str(REPO/'scripts')]
import _pdv2_encoder_place as run
import _pdv2_encoder_place_eval as ev
from _pdv2_encoder_place_lib import temporal_subgraph
from _pdv2_context_accept_lib import labels, unit
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from retrieval.sequence_evidence import neighbor_plans, repeated_candidate_weights


def context_reference(key: np.ndarray, path: np.ndarray, row: dict, index: int) -> float:
    """한 질의의 문맥 이웃 가중합을 별도 스칼라 반복으로 계산한다."""
    q,j=int(row['query_idx'][index]),int(row['j_star'][index])
    c=unit(key[:,288:])
    qn,jn,valid=neighbor_plans(path,np.array([q]),np.array([j]))['arc_unique']
    weights=repeated_candidate_weights(jn,valid)
    direction=int(row['direction'][index]);v=valid[direction,0];w=weights[direction,0]
    center=np.dot(c[q],c[j])
    # path_means의 중심 가중치 1과 유효 이웃 가중치 합을 명시한다.
    total=float(center);denominator=1.
    for k in np.flatnonzero(v):
        total+=float(w[k])*float(np.dot(c[qn[direction,0,k]],c[jn[direction,0,k]]))
        denominator+=float(w[k])
    beta=(2./3.)*(11./10.)
    return (1-beta)*center+beta*total/denominator


def main() -> None:
    """고정 원천과 장소 제외·적합·최종 지표를 확인한다."""
    spec=run.protocol();ev.prepare()
    sources={**spec['runtime'],**spec['inputs'],**run.read(run.OUT/'evaluation_protocol.json')['sources']}
    for p,h in sources.items():assert sha256(REPO/p)==h,p
    state=torch.load(run.previous.PREPARED,map_location='cpu',weights_only=False)
    graph=torch.load(run.previous.GRAPH,map_location='cpu',weights_only=False)
    initial=set();models=0;prefix=0;ledgers=0;rows=0;context_checks=0
    for arm,plan in spec['plans'].items():
        root=run.folder(arm);ids=np.load(root/'training_node_ids.npy')
        absent=[spec['train'].index(s) for s in plan['heldout']]
        assert not np.isin(state['sequences'][ids],absent).any()
        np.testing.assert_array_equal(ids,np.flatnonzero(~np.isin(state['sequences'],absent)))
        part=temporal_subgraph(graph,ids)
        assert part.num_nodes==plan['nodes'] and part.edge_type.eq(0).all()
        for k,h in plan['graph_hashes'].items():assert array_hash(getattr(part,k).numpy())==h
        edges=part.edge_index.numpy();global_edges=ids[edges]
        assert np.all(state['sequences'][global_edges[0]]==state['sequences'][global_edges[1]])
        assert np.max(np.abs(global_edges[0]-global_edges[1]))<=5
        for e in np.linspace(0,edges.shape[1]-1,5).astype(int):
            a,b=global_edges[:,e];poses=state['poses']
            distance=np.linalg.norm(poses[a,:3,3]-poses[b,:3,3])
            rotation=poses[b,:3,:3]@poses[a,:3,:3].T
            angle=np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))
            expected=[np.log1p(distance)/5,angle/np.pi,0,0,1]
            np.testing.assert_allclose(part.edge_attr[e].numpy(),expected,atol=1e-7,rtol=1e-5)
        for epoch in run.REFRESH:
            trips=np.load(root/f'triplets_epoch{epoch}.npz')['triplets']
            original=np.load(run.SAMPLES/f'original_epoch{epoch}.npz')['triplets']
            global_trips=ids[trips]
            if arm.startswith('matched_'):
                selected=np.random.default_rng(71000+epoch).choice(len(original),len(trips),replace=False)
                assert len(np.unique(selected))==len(selected)
                expected=original[selected]
            else:
                included=np.zeros(len(state['sequences']),bool);included[ids]=True
                expected=original[np.all(included[original],axis=1)]
            np.testing.assert_array_equal(global_trips,expected)
            assert len(trips)==plan['triplets_per_epoch']
        completed=run.read(root/'training_complete.json')
        checkpoint=root/'checkpoints/final_model.pth'
        assert sha256(checkpoint)==completed['checkpoint_sha256']
        blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert blob['epoch']==19 and blob['val_metrics']==[]
        assert all(torch.isfinite(v).all() for v in blob['model_state_dict'].values())
        steps={int(v['step']) for v in blob['optimizer_state_dict']['state'].values()}
        assert steps=={20*plan['updates_per_epoch']} and completed['updates'] in steps
        initial.add(sha256(root/'initial_model.json'))
        key_info=run.read(root/'keys_complete.json');prefix+=len(key_info['prefix'])
        assert sha256(root/'retrieval_keys.npy')==key_info['sha256']
        models+=1
        sequences=spec['train'] if arm=='full' else spec['places'][plan['budget_place']]
        for sequence in sequences:
            p=ev.feature_path(arm,sequence);row=ev.load(p);info=run.read(p.with_suffix('.json'))
            assert sha256(p)==info['sha256']
            assert ev.accept.exp.metrics(row['baseline'],row)==info['metrics']
            fixed=root/'fixed'/(sequence.replace('/','_')+'.npz')
            assert sha256(fixed)==info['fixed_sha256']
            reference=ev.load(ev.accept.feature_file('train',1,sequence));frow=ev.load(fixed)
            for field in ev.FIELDS:np.testing.assert_array_equal(frow[field],reference[field])
            ids_seq=np.flatnonzero(state['sequences']==spec['train'].index(sequence))
            key=np.load(root/'retrieval_keys.npy',mmap_mode='r')[ids_seq]
            positions=state['poses'][ids_seq,:3,3].astype(float)
            path=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
            for r in (row,frow):
                for i in (len(r['query_idx'])//3,2*len(r['query_idx'])//3):
                    np.testing.assert_allclose(context_reference(key,path,r,i),r['context_arc'][i],atol=3e-7,rtol=0)
                    q,j=r['query_idx'][i],r['j_star'][i]
                    assert j<=q-30 and q<len(key)-15
                    np.testing.assert_allclose(np.linalg.norm(positions[q]-positions[j]),r['top1_geo_dist'][i],atol=1e-8)
                    context_checks+=1
            rows+=len(row['query_idx']);ledgers+=1
            for source in (p,p.with_suffix('.json'),fixed):sources[str(source.relative_to(REPO))]=sha256(source)
        for p in (checkpoint,root/'training_complete.json',root/'initial_model.json',root/'keys_complete.json'):
            sources[str(p.relative_to(REPO))]=sha256(p)
    assert len(initial)==1
    for pair in spec['pairs'].values():
        a,b=[run.read(run.folder(pair[k])/'training_complete.json')['updates'] for k in ('leave','matched')]
        assert a==b
    frozen=run.read(run.OUT/'calibration_freeze.json')
    for name,fit in frozen['models'].items():
        xs,ys,ws=[],[],[]
        for place in spec['places']:
            for sequence in spec['places'][place]:
                arm='full' if name=='in_sample' else spec['pairs'][place]['leave']
                row=ev.load(ev.feature_path(arm,sequence));keep,y=labels(row)
                xs.append(np.column_stack([row[f][keep] for f in fit['fields']]))
                ys.append(y[keep])
                count=sum(s.split('/')[0]==sequence.split('/')[0] for s in spec['train'])
                ws.append(np.full(int(keep.sum()),1/(2*count*keep.sum())))
        x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w/=w.sum()
        mean=w@x;scale=np.maximum(np.sqrt(w@((x-mean)**2)),1e-6)
        np.testing.assert_allclose(mean,fit['mean'],atol=1e-12)
        np.testing.assert_allclose(scale,fit['scale'],atol=1e-12)
        z=(x-mean)/scale;coef=np.asarray(fit['coefficients'])
        r=w*(expit(z@coef+fit['intercept'])-y);grad=z.T@r+.001*coef
        projected=np.where(coef<=1e-6,np.minimum(grad,0),grad)
        assert max(np.abs(projected).max(),abs(r.sum()))<2e-6
    summary=run.read(run.OUT/'summary.json');vrows=0
    for sensor,seq in ev.accept.VAL:
        sequence=sensor+'/'+seq;p=ev.feature_path('full',sequence,'val');row=ev.load(p)
        vrows+=len(row['query_idx'])
        assert sha256(p)==summary['outputs'][str(p.relative_to(REPO))]
        for name,fit in frozen['models'].items():
            coef=np.asarray(fit['coefficients'])/np.asarray(fit['scale'])
            bias=fit['intercept']-np.dot(fit['mean'],coef)
            expected=np.column_stack([row[f] for f in fit['fields']])@coef+bias
            np.testing.assert_allclose(expected,row[name],atol=3e-12,rtol=0)
        for name in summary['per_sequence']:
            assert ev.accept.exp.metrics(row[name],row)==summary['per_sequence'][name][sequence]
        keypath=run.folder('full')/'val_keys'/(sequence.replace('/','_')+'.npz')
        keyinfo=run.read(keypath.with_suffix('.json'));prefix+=len(keyinfo['prefix'])
        assert sha256(keypath)==keyinfo['sha256']
        _,_,paths,_=ev.focus.base.inputs('val',sequence)
        cache=ev.load(paths['cache']);key=ev.load(keypath)['embedding']
        xyz=cache['poses'][:,:3,3]
        path=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        for i in (len(row['query_idx'])//3,2*len(row['query_idx'])//3):
            np.testing.assert_allclose(context_reference(key,path,row,i),row['context_arc'][i],atol=3e-7,rtol=0)
            q,j=row['query_idx'][i],row['j_star'][i]
            assert j<=q-30 and q<len(key)-15
            np.testing.assert_allclose(np.linalg.norm(cache['poses_gt'][q,:3,3]-cache['poses_gt'][j,:3,3]),row['top1_geo_dist'][i],atol=1e-5)
            context_checks+=1
        for source in (p,p.with_suffix('.json'),keypath,keypath.with_suffix('.json')):sources[str(source.relative_to(REPO))]=sha256(source)
    assert vrows==99236
    for method,per in summary['per_sequence'].items():
        assert ev.accept.macro(per)==summary['aggregate'][method]
    for p in (run.OUT/'calibration_freeze.json',run.OUT/'summary.json',Path(__file__)):
        sources[str(p.relative_to(REPO))]=sha256(p)
    report={'status':'passed','models':models,'training_ledgers':ledgers,'training_rows':rows,
        'validation_ledgers':9,'validation_rows':vrows,'gpu_prefix_checks':prefix,
        'scalar_context_checks':context_checks,'test_used':False,'sources':sources,
        'scope':'인코더 장소제외·같은초기화/업데이트·표본추가반복없음·GT/KISS입력구분·최종지표·적합KKT·원점수가중합 검산'}
    write_json(run.OUT/'completion_validation.json',report)
    print('VERIFIED',models,ledgers,rows,vrows,prefix,context_checks,len(sources),flush=True)


if __name__=='__main__':main()
