#!/usr/bin/env python3
"""정수bin 회전의 raw격자·yaw전달·검색·정렬·최종판정 차이를 분리한다."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import _review_direction_expand as expansion
from _pdv2_input_alignment_lib import make_graph, pose_attributes, temporal_edges
from gnn.spectral_transport import attach_transport

base, yaw = expansion.base, expansion.yaw
ROOT = base.ROOT
OUT = expansion.OUT / "causes"
ANGLES = (6., 90., 180.)
ARMS = ("actual", "ideal_grid", "ideal_transport", "ideal_both", "ideal_both_exact_geometry", "direct_output_phase")


def freeze() -> dict:
    """성능 결과를 보지 않고 모든 시퀀스의16질의와 인과 대조를 고정한다."""
    parent = expansion.freeze()
    path = OUT / "protocol.json"
    sources = {str(p.relative_to(ROOT)):base.digest(p) for p in (Path(__file__), Path(yaw.__file__), expansion.OUT/'yaw_protocol.json',
        ROOT/'src/gnn/spectral_transport.py', ROOT/'src/gnn/spectral_difference.py', ROOT/'scripts/_xs_eval_lib.py')}
    if path.exists():
        spec=base.read(path)
        assert spec['sources']==sources
        return spec
    spec=dict(created_unix=time.time(),angles=ANGLES,arms=ARMS,
        queries={s:q[::4] for s,q in parent['queries'].items()}, cases=parent['cases'], sources=sources,
        scope='46시퀀스 모두, 기존64균등질의의 매4번째16개=736질의. GT·성능으로 고르지 않음. 3시작점·6/90/180도·6조건. raw회전 주평가2944질의와 분리한 원인 대조.',
        interventions=dict(actual='실제 원시 회전 인코딩과 일관된 T Rz(-theta) 자세.',
          ideal_grid='원래288D와 원BEV계수의 해석적 위상회전으로 raw격자 오차만 제거. 자세는 실제 변환.',
          ideal_transport='실제 raw회전 특징을 유지하고 transport yaw를 원yaw-theta로 고정. 실제 변환자세의 기하속성 유지.',
          ideal_both='해석적 raw특징과 원yaw-theta transport. 실제 변환자세의 기하속성 유지.',
          ideal_both_exact_geometry='ideal_both에 기하속성도 원본 그대로 사용하여 자세행렬 연산 반올림 제거.',
          direct_output_phase='GNN 결과를 직접 해석적 위상회전. 원래 검색키·BEV크기. downstream만 시험.'),
        stages='후보집합 변경과 원래 고정후보집합에서4채널 원점수/minmax/융합 변화 및 승자를 모두 기록. 최종점수는 새 승자와 원래 승자 양쪽에서 계산. grid×transport 2x2이며 효과의 가산성은 가정하지 않음.',
        validity='ideal_*는 원인분리용 내부 대체이며 실제 raw평가를 대체하거나 배포 수정으로 채택하지 않는다. direct_output_phase에서 남는 차이는 수치·downstream 효과다. 원시/물리검산과 actual 원장 재현을 먼저 확인한다.')
    base.write(path,spec)
    return spec


@torch.no_grad()
def infer(net, raw: np.ndarray, coeff: np.ndarray, poses: np.ndarray, first: np.ndarray, override=None) -> np.ndarray:
    """실제 또는 고정한 transport 위상을 쓰되 나머지 모델은 유지한다."""
    seq=np.zeros(len(raw),np.int64)
    graph=make_graph(raw,poses,seq).cuda()
    relative=first.T[None].astype(np.float64)@poses[:,:3,:3].astype(np.float64)
    angle=np.arctan2(relative[:,1,0],relative[:,0,0]) if override is None else override
    attach_transport(graph,coeff,angle,seq)
    return net(graph).cpu().numpy()


def channels(ids, qkey, qaux, qz, keys, hkeys, gallery) -> np.ndarray:
    """같은 후보에서 네 원점수를 순서대로 반환한다."""
    bank=yaw.current.run.base.variant_inputs(np.concatenate((gallery[ids],qz[None])),8)
    j=np.arange(len(ids));q=len(ids)
    return np.stack((keys[ids]@qkey,hkeys[ids]@qaux,
        yaw.current.run.base.align_scores(bank[1],bank[2],q,j),
        yaw.current.run.base.column_align_scores(bank[3],q,j)),axis=1)


def normalized(values: np.ndarray) -> np.ndarray:
    """현재 후보별 minmax를 네 채널에 그대로 적용한다."""
    return np.stack([yaw.current.run.base.minmax(values[:,k]) for k in range(4)],axis=1)


def fusion(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """현재 원장과 같은 float32 덧셈 순서를 유지한다."""
    return weights[0]*values[:,0]+weights[1]*values[:,1]+weights[2]*values[:,2]+weights[3]*values[:,3]


@torch.no_grad()
def final_scores(q, ids, qz, gallery, travel) -> np.ndarray:
    """새 승자·원래 승자를 같은 query 특징으로 따로 판정한다."""
    plan=yaw.current.e.prior.base.prior.plan(travel,np.full(len(ids),q,np.int64),np.asarray(ids,np.int64))
    qi,ji=np.unique(plan['query']),np.unique(plan['candidate'])
    compact=np.concatenate((qz[qi-q+15],gallery[ji]))
    pp=dict(query=np.searchsorted(qi,plan['query']),candidate=np.searchsorted(ji,plan['candidate'])+len(qi),weight=plan['weight'])
    return yaw.current.e.prior.score(torch.from_numpy(compact).cuda(),{k:torch.from_numpy(v).cuda() for k,v in pp.items()}).cpu().numpy()


def evaluate(q, raw, rawcoeff, output, keys, hkeys, gallery, travel, weights, reference=None) -> dict:
    """실제 검색과 원래 후보 고정 재정렬을 동시에 계산한다."""
    qkey=yaw.raw_key(raw[15:16])[0]
    qaux=yaw.current.run.base.variant_inputs(rawcoeff[15:16],8)[0][0]
    oldkey,oldaux=keys[q].copy(),hkeys[q].copy()
    keys[q],hkeys[q]=qkey,qaux
    keyids=yaw.current.run.base.prefix_topk(keys,q,q-29,100)
    auxids=yaw.current.run.base.prefix_topk(hkeys,q,q-29,100)
    keys[q],hkeys[q]=oldkey,oldaux
    ids=np.union1d(keyids,auxids)
    rawscores=channels(ids,qkey,qaux,output[15],keys,hkeys,gallery)
    norm=normalized(rawscores)
    fused=fusion(norm,weights)
    winner=int(ids[np.argmax(fused)])
    if reference is None:
        fixed_ids,fixed_raw,fixed_norm=ids,rawscores,norm
        baseline=winner
    else:
        fixed_ids=reference['ids']
        fixed_raw=channels(fixed_ids,qkey,qaux,output[15],keys,hkeys,gallery)
        fixed_norm=normalized(fixed_raw)
        baseline=reference['winner']
    fixed_fused=fusion(fixed_norm,weights)
    values=final_scores(q,[winner,baseline],output,gallery,travel)
    sorted_fused=np.sort(fused)
    return dict(ids=ids,winner=winner,fixed_winner=int(fixed_ids[np.argmax(fixed_fused)]),
        keyids=keyids,auxids=auxids,raw=fixed_raw,norm=fixed_norm,fused=fixed_fused,
        confidence=float(values[0]),fixed_confidence=float(values[1]),
        margin=float(sorted_fused[-1]-sorted_fused[-2]) if len(ids)>1 else 0.)


def unit(z: np.ndarray) -> np.ndarray:
    """프레임별 복소계수의 단위벡터를 만든다."""
    return z/np.maximum(np.linalg.norm(z.reshape(len(z),-1),axis=1),1e-12)[:,None,None]


def run_sequence(sequence: str, smoke: bool=False) -> None:
    """사전고정된모든 대조를 한 시퀀스의 세 시작점에 적용한다."""
    expansion.activate();spec=freeze()
    name=sequence.replace('/','_')
    target=OUT/(name+('_smoke' if smoke else '')+'.json')
    if target.exists():return
    folder=expansion.OUT/'yaw'/name
    encoded={angle:dict(np.load(folder/f'encoded_{angle:g}.npz')) for angle in (0.,*ANGLES)}
    cache,z,travel=yaw.original_inputs(sequence)
    keys,hkeys=yaw.raw_key(cache['descriptors']),yaw.current.run.base.variant_inputs(z,8)[0]
    weights=yaw.current.run.cal.fusion_weights(base.read(base.MAIN/'protocol.json')['confidence'])[yaw.current.run.cal.FUSIONS.index('aux_half')]
    rows=[];checks=[]
    queries=spec['queries'][sequence][:1] if smoke else spec['queries'][sequence]
    started=time.time()
    for tag,case in spec['cases'].items():
        net=yaw.current.load_model(case)
        gallery=yaw.infer(net,cache['descriptors'],z,cache['poses'],cache['poses'][0,:3,:3])
        actual={angle:dict(np.load(folder/f'{tag}_{angle:g}.npz')) for angle in (0.,*ANGLES)}
        for q in queries:
            take=np.searchsorted(encoded[0.]['frames'],np.arange(q-15,q+16))
            x0,z0=encoded[0.]['raw'][take],encoded[0.]['coeff'][take]
            p0=cache['poses'][q-15:q+16].copy();first=cache['poses'][0,:3,:3]
            r0=first.T[None]@p0[:,:3,:3]
            yaw0=np.arctan2(r0[:,1,0],r0[:,0,0])
            baseoutput=infer(net,x0,z0,p0,first)
            ref=evaluate(q,x0,z0,baseoutput,keys,hkeys,gallery,travel,weights)
            ix=np.searchsorted(actual[0.]['query_idx'],q)
            assert ref['winner']==actual[0.]['j_star'][ix]
            assert abs(ref['confidence']-actual[0.]['score'][ix])<=8e-6
            for angle in ANGLES:
                take=np.searchsorted(encoded[angle]['frames'],np.arange(q-15,q+16))
                xa,za=encoded[angle]['raw'][take],encoded[angle]['coeff'][take]
                phase=np.exp(-1j*np.deg2rad(angle)*np.arange(1,9))[None,None].astype(np.complex64)
                zi=z0*phase
                pa=p0.copy();pa[:,:3,:3]=pa[:,:3,:3]@yaw.rotz(angle).T
                rel=first.T[None]@pa[:,:3,:3]
                ya=np.arctan2(rel[:,1,0],rel[:,0,0]);expected_yaw=yaw0-np.deg2rad(angle)
                residual=np.rad2deg(np.angle(np.exp(1j*(ya-expected_yaw))))
                edges=temporal_edges(np.zeros(31,np.int64))
                geom_error=float(np.max(abs(pose_attributes(pa,edges)-pose_attributes(p0,edges))))
                for arm in ARMS:
                    ideal_grid=arm in ('ideal_grid','ideal_both','ideal_both_exact_geometry','direct_output_phase')
                    x,coeff=(x0,zi) if ideal_grid else (xa,za)
                    override=expected_yaw if arm in ('ideal_transport','ideal_both','ideal_both_exact_geometry') else None
                    poses=p0 if arm=='ideal_both_exact_geometry' else pa
                    output=baseoutput*phase if arm=='direct_output_phase' else infer(net,x,coeff,poses,first,override)
                    value=evaluate(q,x,coeff,output,keys,hkeys,gallery,travel,weights,ref)
                    if arm=='actual':
                        ix=np.searchsorted(actual[angle]['query_idx'],q)
                        assert value['winner']==actual[angle]['j_star'][ix],(sequence,tag,q,angle)
                        error=abs(value['confidence']-actual[angle]['score'][ix])
                        assert error<=8e-6,(sequence,tag,q,angle,error)
                        checks.append(error)
                    output_error=np.linalg.norm((unit(output[10:21])-unit(baseoutput[10:21]*phase)).reshape(11,-1),axis=1)
                    row=dict(sequence=sequence,seed=int(tag[-1]),query=int(q),angle=angle,arm=arm,
                        input_key_l2=float(np.linalg.norm(yaw.raw_key(x[15:16])-yaw.raw_key(x0[15:16]))),
                        input_coefficient_phase_l2=float(np.linalg.norm(unit(coeff[15:16])-unit(zi[15:16]))),
                        output_phase_l2_max=float(output_error.max()),yaw_phase_residual_deg_max=float(abs(residual).max()),geometry_attribute_error=geom_error,
                        candidate_symmetric_difference=int(len(np.setxor1d(value['ids'],ref['ids']))),
                        key_candidates_changed=int(not np.array_equal(value['keyids'],ref['keyids'])),
                        auxiliary_candidates_changed=int(not np.array_equal(value['auxids'],ref['auxids'])),
                        raw_channel_max_error=np.max(abs(value['raw']-ref['raw']),axis=0).tolist(),
                        normalized_channel_max_error=np.max(abs(value['norm']-ref['norm']),axis=0).tolist(),
                        baseline_channel_range=np.ptp(ref['raw'],axis=0).tolist(),
                        fixed_pool_fusion_max_error=float(np.max(abs(value['fused']-ref['fused']))),
                        baseline_margin=ref['margin'],baseline_candidates=len(ref['ids']),winner_changed=value['winner']!=ref['winner'],fixed_pool_winner_changed=value['fixed_winner']!=ref['winner'],
                        winner=value['winner'],baseline_winner=ref['winner'],
                        confidence_change=value['confidence']-ref['confidence'],fixed_winner_confidence_change=value['fixed_confidence']-ref['confidence'])
                    rows.append(row)
        del net,gallery
    base.write(target,dict(status='passed',sequence=sequence,rows=rows,actual_ledger_checks=len(checks),actual_max_score_error=max(checks),seconds=time.time()-started,
        protocol_sha256=base.digest(OUT/'protocol.json')))
    print('CAUSE',sequence,'rows',len(rows),'seconds',round(time.time()-started,1),flush=True)


def main() -> None:
    """smoke 뒤 모든 고정 원인 대조를 실행한다."""
    parser=argparse.ArgumentParser();parser.add_argument('--sequence');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=freeze()
    for seq in ([args.sequence] if args.sequence else spec['queries']):run_sequence(seq,args.smoke)


if __name__=='__main__':main()
