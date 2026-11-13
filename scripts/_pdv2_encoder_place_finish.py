#!/usr/bin/env python3
"""인코더 장소 제외의 최종 지표와 두 높은 점수 오답을 정리한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src'),str(REPO/'scripts')]
import _pdv2_encoder_place as run
import _pdv2_encoder_place_eval as ev
from _pdv2_context_accept_lib import unit
from _pdv2_no_context_ablation import sha256,write_json


def main() -> None:
    """검산된 결과를 그림과 해석용 숫자로 저장한다."""
    completed=run.read(run.OUT/'completion_validation.json')
    assert completed['status']=='passed'
    spec=run.protocol();frozen=run.read(run.OUT/'calibration_freeze.json')
    summary=run.read(run.OUT/'summary.json')
    place_rows={}
    for place,pair in spec['pairs'].items():
        entries={}
        for mode in ('matched','leave'):
            arm=pair[mode];rows=[];auc=[];negative_fp=negative_n=0
            for sequence in spec['places'][place]:
                detail=run.read(ev.feature_path(arm,sequence).with_suffix('.json'))
                if detail['metrics']['n_pos']:rows.append(detail['metrics'])
                value=detail['fixed_discrimination']['eligible']['features']['context_arc']['auc']
                if value is not None:auc.append(value)
                replay=run.read(run.folder(arm)/('replay_'+sequence.replace('/','_')+'.json'))
                negative_fp+=replay['counts']['negative_false_accept'];negative_n+=replay['counts']['negative']
            entries[mode]={'arm':arm,'updates':20*spec['plans'][arm]['updates_per_epoch'],
                'AP':float(np.mean([r['AP'] for r in rows])) if rows else None,
                'R100':float(np.mean([r['R@100P'] for r in rows])) if rows else None,
                'fixed_context_auc':float(np.mean(auc)) if auc else None,
                'negative_false_accept_rate':negative_fp/negative_n if negative_n else None}
        place_rows[place]=entries
    cases={}
    # 앞선 두 사례의 같은 질의·승자 쌍을 새 모델에서도 직접 확인한다.
    pairs={'helipr/Bridge02':(6835,6765),'helipr/Bridge03':(1099,1008)}
    for sequence,(q,j) in pairs.items():
        records={}
        paths={'current_selected':ev.accept.feature_file('train',1,sequence)}
        for arm in ('full','leave_helipr_bridge','matched_helipr_bridge'):
            paths[arm]=run.folder(arm)/'fixed'/(sequence.replace('/','_')+'.npz')
        for name,p in paths.items():
            row=ev.load(p);idx=np.flatnonzero(row['query_idx']==q)
            assert len(idx)==1 and row['j_star'][idx[0]]==j
            i=int(idx[0])
            records[name]={'context_arc':float(row['context_arc'][i]),
                           'context_center':float(row['context_center'][i]),
                           'actual_distance_m':float(row['top1_geo_dist'][i])}
        cases[sequence]={'query':q,'winner':j,'models':records}
    ratios={}
    for name,m in frozen['models'].items():
        raw=np.asarray(m['coefficients'])/np.asarray(m['scale'])
        ratios[name]={'standardized_coefficients':m['coefficients'],
            'raw_coefficients':raw.tolist(),'context_to_baseline_ratio':float(raw[1]/raw[0])}
    probe_path=run.OUT/'representation_probe.json'
    probe=run.read(probe_path)
    for p,h in probe['sources'].items():assert sha256(REPO/p)==h,p
    meta=ev.load(ev.accept.PRIOR/'training_metadata.npz')
    for sequence,models in probe['results'].items():
        ids=np.flatnonzero(meta['sequences']==spec['train'].index(sequence))
        for arm,expected in models.items():
            key=np.load(run.folder(arm)/'retrieval_keys.npy',mmap_mode='r')[ids]
            c=unit(key[:,288:]).astype(float);mean=c.mean(0)
            np.testing.assert_allclose(np.linalg.norm(mean),expected['mean_direction_length'],atol=1e-12)
            centered=c-mean
            eig=np.maximum(np.linalg.eigvalsh(centered.T@centered/len(c)),0)
            probability=eig/max(eig.sum(),1e-30)
            rank=np.exp(-np.sum(probability*np.log(np.maximum(probability,1e-30))))
            np.testing.assert_allclose(rank,expected['centered_effective_rank'],atol=1e-10)
            np.testing.assert_allclose(cases[sequence]['models'][arm]['context_arc'],expected['context_arc'],atol=1e-12)
    record={'seed':1,'epochs':20,'place_comparisons':place_rows,'fixed_false_pairs':cases,
            'calibration':ratios,'validation':summary['aggregate'],
            'representation_probe':probe['results'],
            'expansion_eligible':summary['eligible_for_three_seed_confirmation'],'test_used':False,
            'limits':'1시드 예비. 제외 장소·학습 자료량·분포 이동이 다름. matched는같은업데이트수지만동일그래프크기/자료구성이아님. 고정재계량/융합의장소미관측확증아님.'}
    write_json(run.OUT/'report.json',record)
    plot(record)
    files=[Path(__file__),run.OUT/'completion_validation.json',run.OUT/'summary.json',
           run.OUT/'calibration_freeze.json',run.OUT/'report.json',probe_path,
           run.OUT/'encoder_place.png',run.OUT/'encoder_place.pdf']
    write_json(run.OUT/'report_validation.json',{'status':'passed','test_used':False,
               'sources':{str(p.relative_to(REPO)):sha256(p) for p in files}})
    print('REPORT_COMPLETE',flush=True)


def plot(record: dict) -> None:
    """최종 검증·장소별 손실·문맥 구분력·오답 사례를 함께 표시한다."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('/tmp/nsd_context_accept_font.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.4),constrained_layout=True)
    ax=axes[0,0];metrics=('AP','max_F1','R@100P');x=np.arange(3)
    for i,(method,label,color) in enumerate((('baseline','기존 시간 판정','#64748b'),
            ('in_sample','같은 장소 출력으로 적합','#d97706'),('crossfit','제외 장소 출력으로 적합','#2563eb'))):
        values=[record['validation'][method]['macro'][m]*100 for m in metrics]
        ax.bar(x+(i-1)*.26,values,.26,label=label,color=color)
    ax.set_xticks(x,['AP','최대 F1','R@100P']);ax.set_ylim(0,100)
    ax.set_title('최종 검증 9 · 같은 새 full 모델');ax.set_ylabel('%');ax.legend(fontsize=8)
    places=('kitti_route02','kitti_urban','helipr_bridge');names=('KITTI 02','KITTI 05·06','Bridge01–04')
    for ax,metric,title in ((axes[0,1],'AP','해당 장소의 최종 검색 AP'),
                            (axes[1,0],'fixed_context_auc','승자 고정 · 문맥 점수의 정답/오답 구분력')):
        for i,(mode,label,color) in enumerate((('matched','장소 포함 · 학습량 대조','#64748b'),
                                              ('leave','장소 제외','#2563eb'))):
            values=[record['place_comparisons'][p][mode][metric]*100 for p in places]
            ax.bar(np.arange(3)+(i-.5)*.32,values,.32,label=label,color=color)
            for j,v in enumerate(values):ax.text(j+(i-.5)*.32,v+1,f'{v:.1f}',ha='center',fontsize=8)
        ax.set_xticks(np.arange(3),names);ax.set_ylim(0,105);ax.set_title(title);ax.set_ylabel('%')
        ax.legend(fontsize=8)
    ax=axes[1,1]
    model_names=('current_selected','full','matched_helipr_bridge','leave_helipr_bridge')
    for sequence,color,marker in (('helipr/Bridge02','#d97706','o'),('helipr/Bridge03','#2563eb','s')):
        values=[record['fixed_false_pairs'][sequence]['models'][m]['context_arc'] for m in model_names]
        ax.plot(np.arange(4),values,marker=marker,color=color,label=sequence.split('/')[1])
    ax.set_xticks(np.arange(4),['현재 선택 모델','새 full 모델','학습량 대조','Bridge 제외'])
    ax.set_ylim(-1,1.05);ax.set_ylabel('문맥 이웃 코사인');ax.set_title('같은 두 오답 쌍의 문맥 유사도');ax.legend()
    for ax in axes.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('인코더 장소 제외 · 1시드 · 20에포크 고정\n학습 내부는 GT, 검증은 KISS-ICP · 시험 미사용 · Bridge 제외/대조는 40업데이트',fontsize=12)
    fig.savefig(run.OUT/'encoder_place.png',dpi=160)
    fig.savefig(run.OUT/'encoder_place.pdf')
    plt.close(fig)


if __name__=='__main__':main()
