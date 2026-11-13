#!/usr/bin/env python3
"""장소 제외 인코더를 고정 승자와 최종 검색에서 대조하고 검증에 적용한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src'),str(REPO/'scripts')]
import _pdv2_encoder_place as run
from _pdv2_encoder_place_lib import context_on_fixed_winner
from _pdv2_context_accept_lib import pair_features, discrimination, labels
from _pdv2_context_place_lib import fit_fold, score
from _pdv2_no_context_ablation import sha256, write_json, evaluate_keys
from _pdv2_pair_evaluate import ArrayCache
from _xs_eval_lib import normalize
from _causal_pr_dump import dump_op

OUT=run.OUT
accept, previous, focus = run.accept, run.previous, run.focus
FIELDS=accept.FIELDS


def load(path: Path) -> dict:
    """NPZ를 독립 배열로 읽는다."""
    with np.load(path) as z:return {k:z[k] for k in z.files if k!='meta'}


def feature_path(arm: str, sequence: str, role: str='train') -> Path:
    """최종 승자를 재검색한 문맥 특징의 저장 위치다."""
    return OUT/arm/role/(sequence.replace('/','_')+'.npz')


def prepare() -> None:
    """새 점수 계산 전 평가 구현과 검증 적용 규칙을 동결한다."""
    run.protocol()
    target=OUT/'evaluation_protocol.json'
    if target.exists():
        for p,h in run.read(target)['sources'].items():assert sha256(REPO/p)==h,p
        return
    sources={Path(__file__),REPO/'scripts/_pdv2_encoder_place_lib.py',OUT/'protocol.json',
             accept.OUT/'protocol.json',REPO/'scripts/_pdv2_context_accept_lib.py',
             REPO/'scripts/_pdv2_context_place_lib.py',REPO/'scripts/_pdv2_sequence_fixed_test.py',
             REPO/'scripts/_pdv2_temporal3.py',REPO/'scripts/_lag_branch_fusion.py'}
    for sensor,seq in accept.VAL:
        _,_,paths,_=focus.base.inputs('val',sensor+'/'+seq)
        sources.update(paths.values())
        sources.add(accept.feature_file('val',1,sensor+'/'+seq))
    sources.update(accept.feature_file('train',1,s) for s in run.protocol()['train'])
    sources.update((accept.PRIOR/'retrieval_keys.npy',accept.PRIOR/'training_metadata.npz'))
    write_json(target,{'created_unix':time.time(),'seed':1,'test_used':False,
        'selection':'학습20에포크·추가계수100% 고정. 학습 원장8에서full/OOF두적합 후검증9. 주조건OOF. 기준은training protocol과동일.',
        'sources':{str(p.relative_to(REPO)):sha256(p) for p in sorted(sources)}})
    print('EVALUATION_FROZEN',flush=True)


def training_features(arm: str, sequence: str, spec: dict) -> None:
    """같은 승자의 문맥 진단과 해당 모델의 실제 최종 검색을 기록한다."""
    target=feature_path(arm,sequence)
    if target.exists():
        assert sha256(target)==run.read(target.with_suffix('.json'))['sha256']
        return
    root=run.folder(arm)
    meta=load(root/'training_metadata.npz')
    ids=np.flatnonzero(meta['sequences']==spec['train'].index(sequence))
    key=np.load(root/'retrieval_keys.npy',mmap_mode='r')[ids]
    pos=meta['positions'][ids].astype(np.float64)
    path=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(pos,axis=0),axis=1))]
    old=load(accept.feature_file('train',1,sequence))
    fixed=context_on_fixed_winner(key,path,old)
    fixedpath=root/'fixed'/(sequence.replace('/','_')+'.npz')
    fixedpath.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(fixedpath,**fixed)
    # replay에는 학습 자료만 주며 쓰기 위치는 새 모델 폴더다.
    selected=run.read(accept.OUT/'protocol.json')['seeds']['1']
    previous.replay(sequence,{'train_roster':spec['train'],'selected':selected})
    ledger=load(root/('ledger_'+sequence.replace('/','_')+'.npz'))
    with np.load(accept.spec_path(sequence)) as z:bank=focus.base.bank_for('pbev128_main',z)
    features=[]
    for start in range(0,len(ledger['query_idx']),256):
        sl=slice(start,start+256)
        features.append(pair_features(key,bank,path,ledger['query_idx'][sl],ledger['j_star'][sl],
                         selected['confidence']['decision'],ledger[selected['confidence']['decision']][sl]))
    row={k:np.concatenate([x[k] for x in features]) for k in features[0]}
    row.update({f:ledger[f] for f in FIELDS})
    np.testing.assert_array_equal(row['baseline'],ledger[accept.BASELINE])
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**row)
    fixed_d=discrimination(fixed,fixed,selected['new_tau']['tau'])
    for cohort in fixed_d.values():
        cohort['features']={k:v for k,v in cohort['features'].items()
                            if k in ('baseline','context_center','context_arc')}
    write_json(target.with_suffix('.json'),{'sequence':sequence,'sha256':sha256(target),
        'fixed_sha256':sha256(fixedpath),'metrics':accept.exp.metrics(row['baseline'],row),
        'fixed_discrimination':fixed_d,
        'new_discrimination':discrimination(row,row,selected['new_tau']['tau'])})
    print('TRAIN_FINAL',arm,sequence,flush=True)


def train_report(spec: dict) -> None:
    """자료량이 같은 장소 제외 대조와 OOF 적합을 검증 전에 완료한다."""
    target=OUT/'calibration_freeze.json'
    if target.exists():return
    paired={}
    for place,pair in spec['pairs'].items():
        per={}
        for mode in ('leave','matched'):
            arm=pair[mode]
            rows={s:run.read(feature_path(arm,s).with_suffix('.json')) for s in spec['places'][place]}
            metrics={s:r['metrics'] for s,r in rows.items()}
            positives=[r for r in metrics.values() if r['n_pos']]
            per[mode]={'arm':arm,'per_sequence':rows,
                'macro':{m:float(np.mean([r[m] for r in positives])) for m in accept.exp.METRICS} if positives else None}
        paired[place]=per
    full={s:load(feature_path('full',s)) for s in spec['train']}
    cross={s:load(feature_path(spec['pairs'][p]['leave'],s)) for p in spec['places'] for s in spec['places'][p]}
    models={'in_sample':fit_fold(full,(),'context_arc'),'crossfit':fit_fold(cross,(),'context_arc')}
    sources={feature_path('full',s) for s in spec['train']}
    sources.update(feature_path(spec['pairs'][p]['leave'],s) for p in spec['places'] for s in spec['places'][p])
    write_json(target,{'created_unix':time.time(),'models':models,'paired':paired,
        'full_final':accept.macro({s:accept.exp.metrics(r['baseline'],r) for s,r in full.items()}),
        'oof_final':accept.macro({s:accept.exp.metrics(r['baseline'],r) for s,r in cross.items()}),
        'test_used':False,'validation_scored_before_freeze':False,
        'sources':{str(p.relative_to(REPO)):sha256(p) for p in sorted(sources)}})
    print('CALIBRATION_FROZEN',flush=True)


def validation_keys(sequence: str) -> None:
    """검증 KISS-ICP 키를 만들고 q+15 접두 관측을 확인한다."""
    assert (OUT/'calibration_freeze.json').exists()
    root=run.folder('full')/'val_keys'
    root.mkdir(parents=True,exist_ok=True)
    target=root/(sequence.replace('/','_')+'.npz')
    if target.exists():return
    sensor,seq,paths,_=focus.base.inputs('val',sequence)
    cache=ArrayCache(load(paths['cache']))
    net=run.model().cuda().eval()
    checkpoint=run.folder('full')/'checkpoints/final_model.pth'
    assert sha256(checkpoint)==run.read(run.folder('full')/'training_complete.json')['checkpoint_sha256']
    net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    net.gnn.key_scale=torch.nn.Parameter(torch.from_numpy(np.load(focus.base.RVEC)).float().cuda(),requires_grad=False)
    cfg=yaml.safe_load((focus.OUT/'standard.yaml').read_text())
    raw=cache['descriptors'].astype(np.float32)
    key,graph=focus.old.embedding(cache,sensor,raw,cfg,net,'temporal')
    checks=[]
    for q in (len(raw)//3,2*len(raw)//3):
        end=q+16
        cut=ArrayCache({k:v[:end] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(raw) else v for k,v in cache.items()})
        small,_=focus.old.embedding(cut,sensor,raw[:end],cfg,net,'temporal')
        np.testing.assert_allclose(small[q],key[q],atol=3e-6,rtol=3e-5)
        checks.append({'query':q,'max_error':float(np.abs(small[q]-key[q]).max())})
    np.testing.assert_allclose(normalize(key[:,:288]),normalize(raw*np.load(focus.base.RVEC)[None]),atol=3e-7)
    np.savez_compressed(target,embedding=key,scan_ids=cache['scan_ids'])
    write_json(target.with_suffix('.json'),{'sha256':sha256(target),'prefix':checks,'graph':graph})
    del net
    torch.cuda.empty_cache()
    print('VAL_KEYS',sequence,flush=True)


def validation_score(sequence: str) -> None:
    """새 full 모델로 최종 승자를 고르고 동결된 두 가중합을 평가한다."""
    target=feature_path('full',sequence,'val')
    if target.exists():return
    frozen=run.read(OUT/'calibration_freeze.json')
    sensor,seq,paths,min_past=focus.base.inputs('val',sequence)
    cache=load(paths['cache'])
    key=load(run.folder('full')/'val_keys'/(sequence.replace('/','_')+'.npz'))['embedding']
    selected=run.read(accept.OUT/'protocol.json')['seeds']['1']
    conf=selected['confidence']
    with np.load(paths['spec']) as z:
        np.testing.assert_array_equal(z['scan_ids'],cache['scan_ids'])
        bank=focus.base.bank_for('pbev128_main',z)
    positions=cache['poses_gt'][:,:3,3]
    ledger=evaluate_keys(key,bank,positions,min_past,{1:tuple(conf['tuple'])})[1]
    use=ledger['query_idx']<len(key)-15
    ledger={k:v[use] for k,v in ledger.items()}
    reference=load(accept.feature_file('val',1,sequence))
    for field in ('query_idx','min_past_gt_dist'):
        np.testing.assert_array_equal(ledger[field],reference[field])
    # 판정의 거리 대응에는 기준 자세가 아니라 KISS-ICP 자세를 쓴다.
    xyz=cache['poses'][:,:3,3]
    path=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
    chunks=[]
    for start in range(0,len(ledger['query_idx']),256):
        sl=slice(start,start+256)
        chunks.append(pair_features(key,bank,path,ledger['query_idx'][sl],ledger['j_star'][sl],conf['decision'],ledger[conf['decision']][sl]))
    row={k:np.concatenate([r[k] for r in chunks]) for k in chunks[0]}
    row.update({f:ledger[f] for f in FIELDS})
    for method,fitted in frozen['models'].items():row[method]=score(row,fitted,1.)
    # 표준 평가기와 실제 접두의 최종 승자를 대조한다.
    n=min(160,len(key)); small=accept.exp.truncate_bank(bank,n)
    exact=dump_op(key[:n],*small,positions[:n],min_past[:n],tuple(conf['tuple']),100)
    for field,value in exact.items():np.testing.assert_allclose(ledger[field][:len(value)],value,atol=1e-12,rtol=0)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**row)
    write_json(target.with_suffix('.json'),{'sha256':sha256(target),'rows':len(row['query_idx']),
        'metrics':{m:accept.exp.metrics(row[m],row) for m in ('baseline','in_sample','crossfit')},
        'canonical_prefix_rows':len(exact['query_idx'])})
    print('VAL_FINAL',sequence,flush=True)


def report() -> None:
    """같은 full 모델의 최종 지표와 사전 확장 기준을 보고한다."""
    per={m:{} for m in ('baseline','in_sample','crossfit')}
    outputs={}
    for sensor,seq in accept.VAL:
        sequence=sensor+'/'+seq
        p=feature_path('full',sequence,'val')
        record=run.read(p.with_suffix('.json'))
        assert sha256(p)==record['sha256']
        for m in per:per[m][sequence]=record['metrics'][m]
        outputs[str(p.relative_to(REPO))]=sha256(p)
    aggregate={m:accept.macro(rows) for m,rows in per.items()}
    now=aggregate['crossfit']; passes={}
    for name in ('baseline','in_sample'):
        ref=aggregate[name]
        passes[name]=(now['macro']['AP']>=ref['macro']['AP']+.001 and
            now['macro']['R@100P']>=ref['macro']['R@100P']-.001 and
            all(now['families'][f]['AP']>=ref['families'][f]['AP']-.0025 for f in ref['families']))
    current={s+'/'+q:load(accept.feature_file('val',1,s+'/'+q)) for s,q in accept.VAL}
    record={'seed':1,'epochs':20,'aggregate':aggregate,'per_sequence':per,
        'current_selected_reference':accept.macro({s:accept.exp.metrics(r['baseline'],r) for s,r in current.items()}),
        'expansion_checks':passes,'eligible_for_three_seed_confirmation':all(passes.values()),
        'test_used':False,'outputs':outputs,'calibration_sha256':sha256(OUT/'calibration_freeze.json')}
    write_json(OUT/'summary.json',record)
    for name,result in aggregate.items():print('FINAL_VALIDATION',name,result['macro'],flush=True)


def main(action: str) -> None:
    """학습 내부 분석을 마친 뒤 동결된 적합만 검증에 적용한다."""
    prepare()
    if action=='prepare':return
    spec=run.protocol()
    assert run.read(OUT/'status.json')['status']=='training_complete'
    for arm in spec['plans']:
        sequences=spec['train'] if arm=='full' else spec['places'][spec['plans'][arm]['budget_place']]
        previous.OUT=run.folder(arm)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda s:training_features(arm,s,spec),sequences))
    train_report(spec)
    for sensor,seq in accept.VAL:validation_keys(sensor+'/'+seq)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(validation_score,[s+'/'+q for s,q in accept.VAL]))
    report()
    write_json(OUT/'evaluation_status.json',{'status':'complete','test_used':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));a=p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:main(a.action)
    except Exception as error:
        write_json(OUT/'evaluation_status.json',{'status':'failed','error':repr(error)})
        raise
