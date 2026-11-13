#!/usr/bin/env python3
"""46시퀀스 원시 yaw 확장과 단계별 원인 대조를 검산·집계한다."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import _review_direction_metrics as base
from _review_direction_report import stats, unit

OUT=base.OUT/'expanded46'
ARMS=('actual','ideal_grid','ideal_transport','ideal_both','ideal_both_exact_geometry','direct_output_phase')


def collect_learning() -> dict:
    """6도시 분모를 유지하는 블록 민감도 구간과 모든 비교CSV를 쓴다."""
    folder=base.OUT/'direction_learning'
    block=base.read(folder/'summary.json');sequence=base.read(folder/'sequence_summary.json')
    samples=np.load(folder/'samples.npz')['cities']
    methods=('identity','fixed','diff','convs_only');groups=('all','forward_only','reverse_only')
    comparisons={}
    for a,b in [('convs_only','fixed'),('diff','fixed'),('diff','identity')]:
        comparisons[a+'-'+b]={}
        for gi,group in enumerate(groups):
            valid=np.isfinite(samples[:,:,gi]).all(axis=(1,2,3))
            delta=100*(samples[valid,:,gi,methods.index(a)].mean(1)-samples[valid,:,gi,methods.index(b)].mean(1))
            metrics={m:dict(delta_pp=block['comparisons'][a+'-'+b][group]['overall'][m]['delta_pp'],ci95_pp=np.percentile(delta[:,k],[2.5,97.5]).tolist()) for k,m in enumerate(base.KEYS)}
            comparisons[a+'-'+b][group]=dict(overall=metrics,complete_city_replicates=int(valid.sum()),incomplete_city_replicates=int((~valid).sum()))
    result=dict(status='passed',comparisons=comparisons,
        scope='추가 민감도 분석: 원래2000회 공통블록 재표집 중 해당방향 양성이6도시모두 남은replicate만 사용. 성능값이 아닌 분모존재만으로 필터. 기존available-city구간summary.json을보존. 점추정과city균등가중치유지. 도시별구간은기존summary.json그대로.',
        source_sha256={str(p.relative_to(base.ROOT)):base.digest(p) for p in [folder/'samples.npz',folder/'summary.json',folder/'sequence_summary.json',Path(__file__)]})
    base.write(folder/'fixed_city_block_summary.json',result)
    with (folder/'learning_intervals.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['estimator','comparison','direction','level','city','metric','delta_pp','ci95_low_pp','ci95_high_pp','valid_replicates'])
        for label,source,total in [('sequence_only',sequence,10000),('sequence_and_query_blocks',block,2000),('blocks_complete_six_cities',result,2000)]:
            for pair,directions in source['comparisons'].items():
                for group,row in directions.items():
                    for metric,value in row['overall'].items():writer.writerow([label,pair,group,'overall','',metric,value['delta_pp'],*value['ci95_pp'],row.get('complete_city_replicates',total)])
                    for city,metrics in row.get('cities',{}).items():
                        for metric,value in metrics.items():writer.writerow([label,pair,group,'city',city,metric,value['delta_pp'],*value['ci95_pp'],value.get('valid_replicates',total)])
    return result


def collect_yaw() -> dict:
    """2944질의의 모든 각도·시작점을 집계하고 원시 재생 출처를 확인한다."""
    spec=base.read(OUT/'yaw_protocol.json')
    baseline=base.read(OUT/'yaw_baseline_validation.json')
    assert baseline['status']=='passed' and baseline['queries']==2944
    for path,digest in spec['sources'].items():assert base.digest(base.ROOT/path)==digest,path
    sequences=list(spec['queries'])
    datasets={s:[q for q in sequences if q.startswith(s+'/')] for s in sorted({q.split('/')[0] for q in sequences})}
    results,sources,counts,principle={},{},{},{}
    coordinate_audit={}
    for angle in spec['angles']:
        per_seed={};pairs=[];enc_samples={};per_enc={}
        for tag in spec['cases']:
            seqmetrics,paired={},{ }
            for seq in sequences:
                folder=OUT/'yaw'/seq.replace('/','_');path=folder/f'{tag}_{angle:g}.npz'
                meta=base.read(path.with_suffix('.json'));assert base.digest(path)==meta['sha256']
                sources[str(path.relative_to(base.ROOT))]=meta['sha256']
                row=dict(np.load(path));zero=dict(np.load(folder/f'{tag}_0.npz'))
                for key in ('query_idx','min_past_gt_dist'):np.testing.assert_array_equal(row[key],zero[key])
                np.testing.assert_array_equal(row['query_idx'],spec['queries'][seq])
                assert np.all(row['j_star']<=row['query_idx']-30)
                assert np.isfinite(row['score']).all() and np.isfinite(row['top1_geo_dist']).all()
                assert 1 <= row['n_candidates'].min() <= row['n_candidates'].max() <= 200
                value=base.metric(row['score'],row['top1_geo_dist'],row['min_past_gt_dist'])
                if value is None:assert meta['metrics'] is None
                else:
                    for k,v in value.items():np.testing.assert_allclose(v,meta['metrics'][k],atol=1e-12,rtol=0)
                seqmetrics[seq]=value
                qc=base.classify_queries(row['min_past_gt_dist'])
                before=(zero['top1_geo_dist']<=5)&qc['positive'];after=(row['top1_geo_dist']<=5)&qc['positive']
                pair=dict(queries=len(row['query_idx']),winner_agreement=int(np.sum(row['j_star']==zero['j_star'])),
                    correct_before=int(before.sum()),correct_after=int(after.sum()),
                    recovered=int((~before&after).sum()),regressed=int((before&~after).sum()),
                    gallery_raw_overlap_count=int(row['gallery_raw_support_overlaps_query'].sum()),
                    score_absolute_difference=stats(abs(row['score']-zero['score'])))
                paired[seq]=pair;pairs.append(pair)
                if angle==0 and tag=='diff_s1':counts[seq]=dict(queries=len(row['query_idx']),positive=qc['n_pos'],negative=qc['n_neg'],ambiguous=qc['n_ambiguous'])
            cities=base.aggregate(seqmetrics,base.CITY)
            valid=[v for v in cities.values() if v is not None]
            per_seed[tag]=dict(macro={k:float(np.mean([v[k] for v in valid])) for k in base.KEYS},cities=cities,datasets=base.aggregate(seqmetrics,datasets),sequences=seqmetrics,paired=paired)
        mean_std={k:dict(mean=float(np.mean([v['macro'][k] for v in per_seed.values()])),std=float(np.std([v['macro'][k] for v in per_seed.values()],ddof=1))) for k in base.KEYS}
        results[str(angle)]=dict(mean_std=mean_std,per_seed=per_seed,pooled_query_seed_pairs={k:sum(v[k] for v in pairs) for k in ('queries','winner_agreement','correct_before','correct_after','recovered','regressed','gallery_raw_overlap_count')})
        coordinate_audit[str(angle)]={}
        for seq in sequences:
            folder=OUT/'yaw'/seq.replace('/','_');path=folder/f'encoded_{angle:g}.npz'
            meta=base.read(path.with_suffix('.json'));original=base.read(folder/'encoded_0.json')
            assert meta['points_sha256']==original['points_sha256']
            assert base.digest(path)==meta['sha256']
            sources[str(path.relative_to(base.ROOT))]=meta['sha256']
            row=dict(np.load(path));zero=dict(np.load(folder/'encoded_0.npz'))
            np.testing.assert_array_equal(row['frames'],zero['frames'])
            phase=np.exp(1j*np.deg2rad(angle)*np.arange(1,9))[None,None]
            samples=dict(range_key_cosine=np.sum(unit(row['raw'])*unit(zero['raw']),axis=1),
                bev_magnitude_cosine=np.sum(unit(abs(row['coeff']))*unit(abs(zero['coeff'])),axis=1),
                bev_phase_compensated_l2=np.linalg.norm(unit(row['coeff']*phase)-unit(zero['coeff']),axis=1))
            per_enc[seq]={k:stats(v) for k,v in samples.items()}
            per_enc[seq]['world_coordinate_first_scan_max_error_m']=meta['world_coordinate_max_error']
            for k,v in samples.items():enc_samples.setdefault(k,[]).append(v)
            with np.load(base.cache_path(seq)) as cache:poses=cache['poses']
            relative=poses[0,:3,:3].T[None]@poses[row['frames'],:3,:3]
            rad=np.deg2rad(angle);c,s=np.cos(rad),np.sin(rad)
            rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            changed=relative@rotation.T
            residual=np.angle(np.exp(1j*(np.arctan2(changed[:,1,0],changed[:,0,0])-np.arctan2(relative[:,1,0],relative[:,0,0])+rad)))
            coordinate_audit[str(angle)][seq]=dict(world_coordinate_first_scan_max_error_m=meta['world_coordinate_max_error'],
                yaw_transport_phase_residual_deg=stats(abs(np.rad2deg(residual))))
        principle[str(angle)]=dict(pooled={k:stats(np.concatenate(v)) for k,v in enc_samples.items()},sequences=per_enc)
    for angle,row in results.items():row['delta_vs_zero_pp']={k:100*(row['mean_std'][k]['mean']-results['0.0']['mean_std'][k]['mean']) for k in base.KEYS}
    totals={k:sum(v[k] for v in counts.values()) for k in next(iter(counts.values()))}
    assert totals['queries']==2944
    direction_counts={label:0 for label in base.LABELS}
    for seq in sequences:
        with np.load(base.OUT/'directions'/(seq.replace('/','_')+'.npz')) as directions:
            selected=directions['group'][np.searchsorted(directions['query_idx'],spec['queries'][seq])]
        for code,label in enumerate(base.LABELS):direction_counts[label]+=int((selected==code).sum())
    result=dict(status='passed',counts=totals,per_sequence_counts=counts,final_recognition=results,encoding_principle=principle,
        sampled_positive_direction_counts=direction_counts,coordinate_audit=coordinate_audit,
        zero_positive_sequences=[seq for seq,row in counts.items() if row['positive']==0],
        sample_axis_clarification='np.linspace로 각 시퀀스의 유효 query_idx 배열에서64개 순서 위치를 등간격 선택한다. 초 단위 시간이나 이동거리의 등간격 표본은 아니다.',
        scope=spec['scope'],baseline_validation={k:v for k,v in baseline.items() if k!='sequences'},
        protocol_sha256=base.digest(OUT/'yaw_protocol.json'),source_sha256=sources,
        raw_support_frames=spec['total_raw_support_frames'],raw_encode_calls=7*spec['total_raw_support_frames'],
        coordinate_check='시퀀스별 첫 지원 스캔의 모든 점×7각도. 모든원시프레임 세계좌표검산은 아닌322스캔-각도 검사이며 Tprime pprime=Tp 대수적 일치는 전체에 적용.',
        gallery_overlap_accounting='gallery_raw_overlap_count는 선택 승자의 보수적 원시 지원 상한 j+20>=q-15인 건수다. 실제 거리 대응 이웃별 의존을 추적한 정확한 겹침 건수가 아니다. 이 경계에서도 gallery전체의 인코딩·수정계수·시간문맥은 원래값을 유지하고query표현만 별도로 재인코딩한다.',
        limitations=['전체46시퀀스를 포함하지만195726질의전수회전이 아닌2944질의표본이다.','query좌표회전은 실제반대방향시점변화와 다르다. gallery와후보시간문맥은 모두원본고정.','KITTI07등 표본양성없는시퀀스는AP/재현율평균에서제외하고음성질의와승자변화기록은보존한다.'])
    base.write(OUT/'yaw_summary.json',result)
    with (OUT/'yaw_metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['angle',*base.KEYS,*[k+'_delta_pp' for k in base.KEYS],'winner_agreement_percent'])
        for angle,row in results.items():
            p=row['pooled_query_seed_pairs'];writer.writerow([angle,*[100*row['mean_std'][k]['mean'] for k in base.KEYS],*[row['delta_vs_zero_pp'][k] for k in base.KEYS],100*p['winner_agreement']/p['queries']])
    with (OUT/'yaw_city_dataset.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['angle','seed','level','name',*base.KEYS,'positive_queries','negative_queries'])
        for angle,row in results.items():
            for seed,run in row['per_seed'].items():
                for level in ('cities','datasets','sequences'):
                    for name,metrics in run[level].items():
                        if metrics is not None:
                            negatives=metrics.get('n_neg')
                            if negatives is None:negatives=sum(counts[seq]['negative'] for seq in metrics['sequences'])
                            writer.writerow([angle,seed,level,name,*[100*metrics[k] for k in base.KEYS],metrics['n_pos'],negatives])
    return result


def collect_causes() -> dict:
    """6조건의 제거 대조와 단계별 변화가 actual원장을 재현하는지 확인한다."""
    folder=OUT/'causes';spec=base.read(folder/'protocol.json')
    rows=[];checks=0;worst=0.;sources={}
    for seq in spec['queries']:
        path=folder/(seq.replace('/','_')+'.json');record=base.read(path)
        assert record['status']=='passed' and len(record['rows'])==864
        assert record['actual_max_score_error']<=8e-6
        assert record['protocol_sha256']==base.digest(folder/'protocol.json')
        sources[str(path.relative_to(base.ROOT))]=base.digest(path)
        checks+=record['actual_ledger_checks'];worst=max(worst,record['actual_max_score_error']);rows+=record['rows']
    fields=('input_key_l2','input_coefficient_phase_l2','output_phase_l2_max','yaw_phase_residual_deg_max','geometry_attribute_error','candidate_symmetric_difference','fixed_pool_fusion_max_error','confidence_change','fixed_winner_confidence_change')
    results={}
    for angle in spec['angles']:
        arms={};grouped={}
        for arm in spec['arms']:
            selected=[r for r in rows if r['angle']==angle and r['arm']==arm]
            assert len(selected)==2208
            grouped[arm]={(r['sequence'],r['seed'],r['query']):r for r in selected}
            entry=dict(n=len(selected),winner_changes=sum(r['winner_changed'] for r in selected),fixed_pool_winner_changes=sum(r['fixed_pool_winner_changed'] for r in selected),
                candidate_set_changes=sum(r['candidate_symmetric_difference']>0 for r in selected),key_candidate_list_changes=sum(r['key_candidates_changed'] for r in selected),auxiliary_candidate_list_changes=sum(r['auxiliary_candidates_changed'] for r in selected),
                confidence_changes_gt_8e6=sum(abs(r['confidence_change'])>8e-6 for r in selected),fixed_winner_confidence_changes_gt_8e6=sum(abs(r['fixed_winner_confidence_change'])>8e-6 for r in selected),
                measurements={f:stats(np.array([abs(r[f]) for r in selected])) for f in fields})
            entry['raw_channels']={str(k):stats(np.array([r['raw_channel_max_error'][k] for r in selected])) for k in range(4)}
            entry['minmax_channels']={str(k):stats(np.array([r['normalized_channel_max_error'][k] for r in selected])) for k in range(4)}
            changed=[r for r in selected if r['fixed_pool_winner_changed'] and r['baseline_candidates']>1]
            if changed:
                assert all(r['baseline_margin']<=2*r['fixed_pool_fusion_max_error']+1e-6 for r in changed)
                entry['changed_fixed_pool_margins']=stats(np.array([r['baseline_margin'] for r in changed]))
            arms[arm]=entry
        actual=grouped['actual'];changed=[k for k,r in actual.items() if r['winner_changed']]
        restored={arm:sum(not grouped[arm][k]['winner_changed'] for k in changed) for arm in ARMS[1:]}
        contrasts={}
        for a,b in [('actual','ideal_grid'),('actual','ideal_transport'),('ideal_grid','ideal_both'),('ideal_transport','ideal_both'),('ideal_both','ideal_both_exact_geometry'),('ideal_both_exact_geometry','direct_output_phase')]:
            contrasts[a+' vs '+b]=dict(winner_differences=sum(grouped[a][k]['winner']!=grouped[b][k]['winner'] for k in actual),fixed_winner_score_difference=stats(np.array([abs(grouped[a][k]['fixed_winner_confidence_change']-grouped[b][k]['fixed_winner_confidence_change']) for k in actual])))
        results[str(angle)]=dict(arms=arms,actual_changed_winners=len(changed),actual_changes_restored=restored,paired_contrasts=contrasts)
    transport_edges={}
    for angle in spec['angles']:
        samples=[];rad=np.deg2rad(angle);c,s=np.cos(rad),np.sin(rad)
        rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        for seq,queries in spec['queries'].items():
            with np.load(base.cache_path(seq)) as cache:poses=cache['poses']
            relative=poses[0,:3,:3].T[None]@poses[:,:3,:3]
            changed=relative@rotation.T
            residual=np.angle(np.exp(1j*(np.arctan2(changed[:,1,0],changed[:,0,0])-np.arctan2(relative[:,1,0],relative[:,0,0])+rad)))
            for query in queries:
                part=residual[query-15:query+16]
                edges=np.concatenate([np.angle(np.exp(1j*(part[offset:]-part[:-offset]))) for offset in range(1,6)])
                samples.append(float(np.rad2deg(abs(edges)).max()))
        assert len(samples)==736
        transport_edges[str(angle)]=stats(np.array(samples))
    result=dict(status='passed',results=results,actual_ledger_checks=checks,actual_max_score_error=worst,rows=len(rows),
        transport_edge_residual_deg_max_per_query=transport_edges,
        transport_explanation='변환yaw잔차δ_i가모든이웃에서같은상수면공통위상이상쇄된다. 실제복소전달에영향을주는오차는δ_i−δ_j다. 같은736질의q±15의±5간선에서원래자세로이상대잔차를CPU계산했다. GPU원인대조의입력이나설정은변경하지않는다.',
        channel_order=['원키코사인','원BEV크기코사인','수정계수스펙트럼정렬','수정계수열정렬'],
        protocol_sha256=base.digest(folder/'protocol.json'),source_sha256=sources)
    base.write(folder/'summary.json',result)
    return result


def collect_grid() -> dict:
    """점별 할당 고정으로 격자 차이가 제거되는지 수치로 집계한다."""
    folder=OUT/'grid_probe';spec=base.read(folder/'protocol.json');rows=[]
    for seq in spec['queries']:
        rec=base.read(folder/(seq.replace('/','_')+'.json'))
        assert rec['status']=='passed' and len(rec['rows'])==24
        rows+=rec['rows']
    results={}
    for angle in spec['angles']:
        results[str(angle)]={}
        for projector in ('range','bev'):
            values=[r for r in rows if r['angle']==angle and r['projector']==projector]
            assert len(values)==184
            counts={k:sum(r[k] for r in values) for k in ('points','valid_points','angular_assignment_mismatch','angular_assignment_mismatch_float64','row_assignment_mismatch','range_mask_mismatch')}
            comparisons={k:{f:stats(np.array([r[k][f] for r in values])) for f in ('max_absolute','l2','changed_cells')} for k in ('actual_grid_vs_roll','angle_locked_vs_roll','all_bins_locked_vs_roll')}
            f='coefficient_errors' if projector=='bev' else 'descriptor_errors'
            feature={arm:stats(np.array([r[f][arm]['max_absolute'] for r in values])) for arm in ('actual','angle_locked','all_bins_locked')}
            results[str(angle)][projector]=dict(counts=counts,grid_comparisons=comparisons,feature_max_errors=feature)
    result=dict(status='passed',results=results,rows=len(rows),protocol_sha256=base.digest(folder/'protocol.json'))
    base.write(folder/'summary.json',result)
    return result


def main() -> None:
    """완료된 부분만 선택적으로 또는 전체를 검산한다."""
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=['yaw','causes','grid','learning','all']);args=parser.parse_args()
    for part,function in [('yaw',collect_yaw),('causes',collect_causes),('grid',collect_grid),('learning',collect_learning)]:
        if args.part in (part,'all'):
            result=function();print('REPORT',part,result['status'],flush=True)


if __name__=='__main__':main()
