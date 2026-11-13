#!/usr/bin/env python3
"""장소 제외 적합·선택·최종 지표와 높은 정밀도 경계를 독립 검산한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_context_place as run
import _pdv2_context_place_smallstep as followup
from _pdv2_context_accept_lib import labels
from _pdv2_context_place_lib import FACTORS, FEATURES, PLACES
from _pdv2_matching_mine import cache_path
from _pdv2_no_context_ablation import sha256, write_json


def raw_prediction(row: dict, model: dict, factor: float) -> np.ndarray:
    """표준화 가중합을 원 점수의 계수로 펼쳐 별도로 계산한다."""
    coef = np.asarray(model["coefficients"])*np.array([1., factor])
    weight = coef/np.asarray(model["scale"])
    bias = model["intercept"]-np.dot(np.asarray(model["mean"]), weight)
    return np.column_stack([row[f] for f in model["fields"]]) @ weight+bias


def main() -> None:
    """선택 자료의 분리와 점수·집계·형태 진단을 검증한다."""
    spec = run.prepare()
    followup.prepare()
    fit = run.old.exp.read(run.OUT/"oof_fits.json")
    oof = run.old.exp.read(run.OUT/"oof_summary.json")
    frozen = run.old.exp.read(run.OUT/"freeze.json")
    small = run.old.exp.read(followup.OUT/"freeze.json")
    source_hashes = dict(spec["sources"])
    rows, fit_count = 0, 0
    for seed in (1,2,3):
        data = run.data_for("train", seed, spec)
        for place, heldout in PLACES.items():
            for feature in FEATURES:
                model = fit[str(seed)][place][feature]
                assert set(model["heldout"]) == set(heldout)
                assert set(model["training"]) == set(spec["train"])-set(heldout)
                xs, ys, ws = [], [], []
                families = {s.split('/')[0] for s in model["training"]}
                for sequence in model["training"]:
                    row = data[sequence]
                    keep, y = labels(row)
                    xs.append(np.column_stack([row[f][keep] for f in model["fields"]]))
                    ys.append(y[keep])
                    count = sum(s.split('/')[0] == sequence.split('/')[0] for s in model["training"])
                    ws.append(np.full(int(keep.sum()), 1/(len(families)*count*keep.sum())))
                x,y,w = np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)
                w /= w.sum()
                mean = np.average(x, axis=0, weights=w)
                scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
                np.testing.assert_allclose(mean,model['mean'],atol=1e-12)
                np.testing.assert_allclose(scale,model['scale'],atol=1e-12)
                z=(x-mean)/scale;coef=np.asarray(model['coefficients'])
                residual=w*(expit(z@coef+model['intercept'])-y)
                gradient=z.T@residual+.001*coef
                projected=np.where(coef<=1e-6,np.minimum(gradient,0),gradient)
                assert max(np.max(np.abs(projected)),abs(residual.sum())) < 2e-6
                fit_count += 1
            for sequence in heldout:
                path=run.OUT/f"oof/s{seed}/{sequence.replace('/', '_')}.npz"
                a=run.old.exp.arrays(path);row=data[sequence]
                source_hashes[str(path.relative_to(REPO))]=sha256(path)
                rows+=len(a['query_idx'])
                for field in run.old.FIELDS:
                    np.testing.assert_array_equal(a[field],row[field])
                for feature in FEATURES:
                    for factor in FACTORS:
                        model=fit[str(seed)][place][feature]
                        value=a[run.key(feature,factor)]
                        np.testing.assert_allclose(value,raw_prediction(row,model,factor),atol=2e-12,rtol=0)
                        result=run.old.exp.metrics(value,row)
                        stored=oof['statistics'][str(seed)][place][feature][str(factor)]['per_sequence'][sequence]
                        assert all(result[m]==stored[m] for m in run.old.exp.METRICS)
                        if factor==0:
                            assert result==run.old.exp.metrics(row['baseline'],row)
    summaries={}
    for root in (run.OUT,followup.OUT):
        summary=run.old.exp.read(root/'summary.json');summaries[str(root)]=summary
        for p,h in summary['outputs'].items():
            assert sha256(REPO/p)==h
            a=run.old.exp.arrays(REPO/p)
            parts=Path(p).parts;seed=int(parts[-2][1:]);sensor,sequence=Path(p).stem.split('_',1)
            seq=sensor+'/'+sequence
            source=run.old.exp.arrays(run.old.feature_file('val',seed,seq))
            for field in run.old.FIELDS:
                np.testing.assert_array_equal(a[field],source[field])
            for method in summary['per_seed'][str(seed)]:
                metrics=run.old.exp.metrics(a[method],a)
                assert metrics==summary['per_seed'][str(seed)][method]['per_sequence'][seq]
                if method=='baseline':
                    np.testing.assert_array_equal(a[method],source['baseline'])
                else:
                    condition=frozen['conditions'][method] if root==run.OUT else ['context_arc',small['factor']]
                    model=frozen['full_models'][str(seed)][condition[0]]
                    np.testing.assert_allclose(a[method],raw_prediction(source,model,condition[1]),atol=2e-12,rtol=0)
        for method,aggregate in summary['mean_std'].items():
            for metric,value in aggregate['macro'].items():
                values=[summary['per_seed'][str(s)][method]['macro'][metric] for s in (1,2,3)]
                np.testing.assert_allclose([np.mean(values),np.std(values,ddof=1)],[value['mean'],value['std']],atol=1e-15)
    # 정수 해상도 후속 선택은 학습 내부 원장만으로 재계산한다.
    allowed=[]
    for candidate in small['evaluated']:
        factor=candidate['factor'];checks=[]
        for seed in (1,2,3):
            grid=oof['grids']['context_arc'];current=next(v for v in grid if v['factor']==factor)['per_seed'][seed-1];base=grid[0]['per_seed'][seed-1]
            checks.append(current['AP']>=base['AP']+.001 and current['R@100P']>=base['R@100P'])
            for place,seqs in PLACES.items():
                for seq in seqs:
                    record=oof['statistics'][str(seed)][place]['context_arc']
                    a,b=record[str(factor)]['per_sequence'][seq],record['0.0']['per_sequence'][seq]
                    if a['n_pos']:
                        loss=round((b['R@100P']-a['R@100P'])*a['n_pos'])
                        checks.append(a['AP']>=b['AP']-.001 and loss<=1)
                    else:
                        checks.append(a['incorrect_accept']/a['incorrect']-b['incorrect_accept']/b['incorrect']<=.001)
        if factor>0 and all(checks):allowed.append(factor)
    assert allowed==small['eligible_factors'] and small['factor']==(min(allowed) if allowed else 0.)
    diagnostic=run.old.exp.read(run.OUT/'high_precision_geometry.json')
    for seq,record in diagnostic['results'].items():
        row=run.old.exp.arrays(run.OUT/f"oof/s1/{seq.replace('/','_')}.npz")
        keep,y=labels(row);correct=keep&(y==1);wrong=keep&(y==0)
        positions=run.old.exp.arrays(cache_path(seq))['poses'][:,:3,3].astype(float)
        for factor in (0.,.05):
            value=row[run.key('context_arc',factor)]
            threshold=value[wrong].max();first=np.flatnonzero(wrong&(value==threshold))
            assert int((correct&(value>threshold)).sum())==record[str(factor)]['R100_tp']
            assert len(first)==len(record[str(factor)]['first_incorrect'])
            for i,expected in zip(first,record[str(factor)]['first_incorrect']):
                q,j=int(row['query_idx'][i]),int(row['j_star'][i])
                assert (q,j)==(expected['query'],expected['winner'])
                np.testing.assert_allclose(np.linalg.norm(positions[q]-positions[j]),expected['winner_distance_m'],atol=1e-12)
        source_hashes[str(cache_path(seq).relative_to(REPO))]=sha256(cache_path(seq))
    for root in (run.OUT,followup.OUT):
        for name in ('protocol.json','freeze.json','summary.json'):
            source_hashes[str((root/name).relative_to(REPO))]=sha256(root/name)
    for p in (Path(__file__),run.OUT/'oof_fits.json',run.OUT/'oof_summary.json',run.OUT/'high_precision_boundary.json',run.OUT/'high_precision_geometry.json'):
        source_hashes[str(p.relative_to(REPO))]=sha256(p)
    record={'status':'passed','oof_rows':rows,'oof_ledgers':24,'fold_fits':fit_count,'validation_rows_per_experiment':3*99236,
        'validation_ledgers_per_experiment':27,'test_used':False,'sources':source_hashes,'source_count':len(source_hashes),
        'scope':'인코더동결. 적합장소분리·정규화·KKT·원점수계수·원장·최종지표·후속선택·첫위양성거리검산'}
    write_json(run.OUT/'completion_validation.json',record)
    write_json(followup.OUT/'completion_validation.json',record)
    print('VERIFIED',rows,fit_count,len(source_hashes),flush=True)


if __name__=='__main__':
    main()
