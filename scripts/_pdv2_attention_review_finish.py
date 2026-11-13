#!/usr/bin/env python3
"""attention 추론 개입의 최종 원장·후보·접두·독립 지표를 검산한다."""
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'src'), str(REPO/'scripts')]
import _pdv2_attention_review as run
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_temporal_test_control_finish import scalar_score
from _pdv2_relation_precision import records as precision_records
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """동일 체크포인트의 36원장을 독립적으로 검산하고 결과를 기록한다."""
    spec, summary = run.protocol(True), run.read(run.OUT/'summary.json')
    status = run.read(run.OUT/'status.json')
    assert status['status'] in ('evaluation_complete','complete')
    sources = {**spec['runtime'], **spec['inputs']}
    for p in (Path(__file__),run.OUT/'summary.json',run.OUT/'protocol.json',
              REPO/'scripts/_pdv2_context_tail_finish.py',REPO/'scripts/_pdv2_temporal_test_control_finish.py',
              REPO/'scripts/_pdv2_relation_precision.py'):
        sources[str(p.relative_to(REPO))]=sha256(p)
    assert summary['protocol_sha256']==sha256(run.OUT/'protocol.json')
    ledgers=rows_count=metric_sets=scalar_checks=prefixes=exact_replays=0
    metric_error=scalar_error=prefix_error=reference_key_error=0.
    counts,rows_by_arm,replay_differences,precision={},{},[],{}
    for arm in run.ARMS:
        root=run.base.root(1,arm)
        checkpoint=root/'checkpoints/checkpoint_epoch_20.pth'
        assert sha256(checkpoint)==spec['checkpoint_sha256']
        sources[str(checkpoint.relative_to(REPO))]=sha256(checkpoint)
        measured={m:{} for m in ('raw',run.METHOD)}
        counts[arm],rows_by_arm[arm]={},{}
        for seq in spec['val']:
            name=seq.replace('/','_')+'.npz'
            p=root/'epoch20'/name; kp=p.with_name('keys_'+name);cp=p.with_name('coverage_'+name)
            info=run.read(p.with_suffix('.json'))
            for path,expected in ((p,info['sha256']),(kp,info['keys_sha256']),
                (cp,run.read(cp.with_suffix('.json'))['sha256'])):
                assert sha256(path)==expected
                sources[str(path.relative_to(REPO))]=expected
            for path in (p.with_suffix('.json'),cp.with_suffix('.json')):
                sources[str(path.relative_to(REPO))]=sha256(path)
            row,key_data,coverage=(run.base.exp.arrays(path) for path in (p,kp,cp))
            key=key_data['embedding'];q,j=row['query_idx'],row['j_star']
            _,_,paths,_=run.base.exp.old.base.inputs('val',seq)
            cache=run.base.exp.arrays(paths['cache']);xyz=cache['poses_gt'][:,:3,3]
            np.testing.assert_array_equal(q,np.arange(30,len(key)-15))
            np.testing.assert_array_equal(key_data['scan_ids'],cache['scan_ids'])
            assert key.shape==(len(xyz),416) and np.isfinite(key).all()
            np.testing.assert_allclose(np.linalg.norm(key,axis=1),1.,atol=2e-6)
            assert np.all((j>=0)&(j<=q-30))
            np.testing.assert_allclose(np.linalg.norm(xyz[q]-xyz[j],axis=1),row['top1_geo_dist'],atol=1e-10,rtol=0)
            for f in ('query_idx','j_star','top1_geo_dist','min_past_gt_dist'):
                np.testing.assert_array_equal(row[f],coverage[f])
            np.testing.assert_array_equal(coverage['union_hit'],coverage['key_hit']|coverage['aux_hit'])
            ref_folder=run.source.OUT/'s1/diff_tanh/epoch20'
            reference=run.base.exp.arrays(ref_folder/name)
            for f in ('query_idx','min_past_gt_dist','group'):
                np.testing.assert_array_equal(row[f],reference[f])
            extra_checks=[]
            if arm=='full':
                old_key=run.base.exp.arrays(ref_folder/('keys_'+name))['embedding']
                np.testing.assert_allclose(key,old_key,atol=4e-6,rtol=4e-5)
                reference_key_error=max(reference_key_error,float(abs(key-old_key).max()))
                changed=np.flatnonzero(np.logical_or.reduce([row[f]!=reference[f] for f in row]))
                if len(changed):
                    for ix in changed:
                        replay_differences.append({'sequence':seq,'query':int(q[ix]),
                            'current':{f:v[ix].item() for f,v in row.items()},
                            'reference':{f:v[ix].item() for f,v in reference.items()}})
                    extra_checks=changed.tolist()
                else: exact_replays+=1
            for method in measured:
                values=independent_metrics(row,row[method]);measured[method][seq]=values
                for metric,value in values.items():
                    expected=info['metrics'][method][metric]
                    if value is None: assert expected is None
                    else:
                        metric_error=max(metric_error,abs(value-expected))
                        np.testing.assert_allclose(value,expected,atol=5e-13,rtol=0)
                metric_sets+=1
            with np.load(paths['spec']) as z:bank=run.base.exp.old.base.bank_for(run.base.exp.OP,z)
            pos=cache['poses'][:,:3,3];travel=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(pos,axis=0),axis=1))]
            conf=spec['confidence']['1']
            for ix in sorted(set([0,len(q)//3,2*len(q)//3,len(q)-1,*extra_checks])):
                t=int(q[ix]);ids=np.unique(np.r_[run.base.prefix_topk(key,t,t-29,100),run.base.prefix_topk(bank[0],t,t-29,100)])
                channels=(key[ids]@key[t],bank[0][ids]@bank[0][t],run.base.align_scores(bank[1],bank[2],t,ids),run.base.column_align_scores(bank[3],t,ids))
                fused=sum(float(w)*run.base.minmax(v) for w,v in zip(conf['tuple'],channels))
                winner=int(ids[np.argsort(-fused,kind='stable')[0]])
                assert winner==int(j[ix])
                channel=2 if conf['decision']=='c_spec' else 3
                raw=float(channels[channel][np.flatnonzero(ids==winner)[0]])
                np.testing.assert_allclose(raw,row['raw'][ix],atol=1e-12,rtol=0)
                value=scalar_score(bank,travel,t,winner,conf['decision'],raw)
                scalar_error=max(scalar_error,abs(value-row[run.METHOD][ix]))
                np.testing.assert_allclose(value,row[run.METHOD][ix],atol=1e-12,rtol=0)
                scalar_checks+=1
            for check in info['prefixes']:
                assert check['winner_equal'] and check['score_error']==0
                prefix_error=max(prefix_error,check['key_max_error']);prefixes+=1
            positive=row['min_past_gt_dist']<=5;correct=positive&(row['top1_geo_dist']<=5)
            assert not np.any(correct&~coverage['union_hit'])
            counts[arm][seq]={'positive':int(positive.sum()),'correct':int(correct.sum()),
                'absent':int((positive&~coverage['union_hit']).sum()),
                'wrong_in_pool':int((positive&coverage['union_hit']&~correct).sum())}
            rows_by_arm[arm][seq]=row
            ledgers+=1;rows_count+=len(q)
        for method,per_seq in measured.items():
            aggregate=run.base.macro(per_seq)
            for metric in run.base.exp.METRICS:
                np.testing.assert_allclose(aggregate['macro'][metric],summary['results'][arm][method]['macro'][metric],atol=5e-13,rtol=0)
        precision[arm]=precision_records(root,20)
    totals={arm:{k:sum(v[k] for v in seq.values()) for k in next(iter(seq.values()))} for arm,seq in counts.items()}
    transitions={}
    for arm in run.ARMS:
        transitions[arm]={'gained':0,'lost':0}
        for seq,old in rows_by_arm['full'].items():
            new=rows_by_arm[arm][seq];pos=old['min_past_gt_dist']<=5
            transitions[arm]['gained']+=int((pos&(old['top1_geo_dist']>5)&(new['top1_geo_dist']<=5)).sum())
            transitions[arm]['lost']+=int((pos&(old['top1_geo_dist']<=5)&(new['top1_geo_dist']>5)).sum())
    original=run.read(run.source.OUT/'summary.json')['per_seed']['1']['diff_tanh']['selected']
    deltas={m:100*(summary['results']['full'][run.METHOD]['macro'][m]-original[run.METHOD]['macro'][m]) for m in run.base.exp.METRICS}
    assert max(abs(v) for v in deltas.values())<.001, deltas
    write_json(run.OUT/'diagnostics.json',{'totals':totals,'sequences':counts,'transitions':transitions,
        'precision':precision,'reference_exact_ledgers':exact_replays,'reference_differences':replay_differences,
        'reference_macro_delta_pp':deltas,'test_used':False})
    statistics=run.read(run.OUT/'attention_statistics.json')
    sources.update(statistics['sources'])
    for name in ('attention_statistics.json','diagnostics.json'):
        p=run.OUT/name;sources[str(p.relative_to(REPO))]=sha256(p)
    with (run.OUT/'final_metrics.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['arm','aggregation',*run.base.exp.METRICS])
        for arm,r in summary['results'].items():
            for group,values in {'macro':r[run.METHOD]['macro'],**r[run.METHOD]['families']}.items():
                writer.writerow([arm,group,*[100*values[m] for m in run.base.exp.METRICS]])
    sources[str((run.OUT/'final_metrics.csv').relative_to(REPO))]=sha256(run.OUT/'final_metrics.csv')
    for p,h in sources.items():assert sha256(REPO/p)==h,p
    result={'status':'passed','ledgers':ledgers,'rows':rows_count,'metric_sets':metric_sets,
        'scalar_checks':scalar_checks,'prefix_checks':prefixes,'max_metric_error':metric_error,
        'max_scalar_error':scalar_error,'max_prefix_key_error':prefix_error,
        'exact_reference_ledger_replays':exact_replays,'reference_changed_queries':len(replay_differences),
        'max_reference_key_error':reference_key_error,'source_count':len(sources),'sources':sources,
        'test_used':False,'training_performed':False}
    write_json(run.OUT/'completion_validation.json',result)
    write_json(run.OUT/'status.json',{**status,'status':'complete','audit_passed':True})
    print('PASSED',{k:v for k,v in result.items() if k!='sources'},flush=True)

if __name__=='__main__':
    main()
