#!/usr/bin/env python3
"""고정 쌍 진단을 집계하고 논문용 그림·수치·캡션을 만든다."""
from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import rankdata

import _pdv2_pair_visualization as experiment

OUT = experiment.OUT
DEST = experiment.REPO/'docs/paper/body_v2.2/figs/pair_analysis'
NAMES = dict(scpp='SC++', ringsharp='RING#-L', helios='HeLiOS', seqot='SeqOT',
             bevplace='BEVPlace++', nsd='NSD (ours)')
XLABEL = dict(scpp='Negative A-PC distance', ringsharp='Negative translation error',
              helios='Descriptor cosine', seqot='Negative descriptor L2',
              bevplace='Negative descriptor L2', nsd='Final pair confidence')
COLOURS = dict(positive='#196B24', random='#7B8790', hard='#B34E19')


def weighted_auc(label: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    """동점에 절반을 주는 가중 정답·오답 순서 확률이다."""
    _, index = np.unique(score, return_inverse=True)
    positive = np.bincount(index, weights=weight*label)
    negative = np.bincount(index, weights=weight*(~label.astype(bool)))
    below = np.cumsum(negative)-negative
    value = np.sum(positive*(below+.5*negative))/(positive.sum()*negative.sum())
    return float(np.clip(value, 0, 1))


def group_weights(query: np.ndarray, sequence: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """시퀀스마다 같게, 그 안에서 질의마다 같게 가중한다."""
    weights = np.zeros(len(query), dtype=float)
    for seq in np.unique(sequence[keep]):
        selected = np.flatnonzero(keep & (sequence == seq))
        qs, count = np.unique(query[selected], return_counts=True)
        sizes = dict(zip(qs, count))
        weights[selected] = [1/(len(qs)*sizes[q]) for q in query[selected]]
    weights /= max(weights.sum(), 1)
    return weights


def collect() -> tuple:
    """검증된 여섯 방법을 동일한 쌍 순서로 결합한다."""
    per = []
    verification = []
    for seqid, sequence in enumerate(experiment.SEQUENCES):
        target, pairs, _ = experiment.inputs(sequence)
        row = dict(query=pairs['query'], reference=pairs['reference'], flags=pairs['flags'],
                   distance=pairs['distance'], sequence=np.full(len(pairs['query']), seqid))
        for method in experiment.METHODS:
            meta = experiment.read(target/f'scores_{method}.json')
            assert meta['status'] == 'passed'
            assert meta['pairs_sha256'] == experiment.sha(target/'pairs.npz')
            assert meta['score_sha256'] == experiment.sha(target/f'scores_{method}.npz')
            with np.load(target/f'scores_{method}.npz') as data:
                row[method] = data['score']
            verification.append(dict(sequence=sequence, method=method, **meta))
        with np.load(target/'nsd_components.npz') as data:
            row['nsd_original'] = data['original']
        per.append(row)
    arrays = {k: np.concatenate([r[k] for r in per]) for k in per[0]}
    summary = {'scope': '3 test sequences, 64 revisit queries each, seed 1; forced-pair scoring, not Table 2 retrieval',
               'counts': {}, 'methods': {}, 'nsd_correction': {}, 'verification': verification}
    masks = {name: (arrays['flags'] & bit) != 0 for name, bit in
             [('positive', experiment.POSITIVE), ('random', experiment.RANDOM), ('hard', experiment.HARD)]}
    weights = {name: group_weights(arrays['query'], arrays['sequence'], mask)
               for name, mask in masks.items()}
    for name, mask in masks.items():
        summary['counts'][name] = int(mask.sum())
    summary['counts']['queries'] = 192
    summary['counts']['all_scored_pairs_including_checks'] = len(arrays['query'])
    metrics = []
    for method in (*experiment.METHODS, 'nsd_original'):
        values = arrays[method]
        entry = {}
        for kind in ('random', 'hard'):
            negative = masks[kind]
            qids = arrays['sequence']*1000000+arrays['query']
            relevant = np.unique(qids[negative])
            positive = masks['positive'] & np.isin(qids, relevant)
            keep = positive | negative
            positive_weight = group_weights(arrays['query'], arrays['sequence'], positive)
            entry['auc_'+kind] = weighted_auc(positive[keep], values[keep],
                                               (positive_weight+weights[kind])[keep])
            by_sequence, margin_rows = [], []
            for seqid in range(3):
                outcomes = []
                for q in np.unique(arrays['query'][negative & (arrays['sequence'] == seqid)]):
                    same = (arrays['sequence'] == seqid) & (arrays['query'] == q)
                    pos_score = float(values[same & positive].max())
                    neg_score = float(values[same & negative].max())
                    outcome = float(pos_score > neg_score) + .5*float(pos_score == neg_score)
                    outcomes.append(outcome)
                    margin_rows.append({'sequence': experiment.SEQUENCES[seqid], 'query': int(q),
                                        'margin': pos_score-neg_score, 'correct': outcome})
                if outcomes:
                    by_sequence.append(float(np.mean(outcomes)))
                    local = keep & (arrays['sequence'] == seqid)
                    entry.setdefault('by_sequence',{}).setdefault(experiment.SEQUENCES[seqid],{}).update({
                        'auc_'+kind: weighted_auc(positive[local],values[local],(positive_weight+weights[kind])[local]),
                        'best_positive_beats_'+kind: float(np.mean(outcomes)),
                        'queries_'+kind:len(outcomes)})
            entry['best_positive_beats_'+kind] = float(np.mean(by_sequence))
            entry['margins_'+kind] = margin_rows
        if method == 'nsd_original':
            summary['nsd_correction']['before'] = entry
        else:
            summary['methods'][method] = entry
        metrics.append(dict(method=method, **{k:v for k,v in entry.items() if not k.startswith('margins') and k!='by_sequence'}))
    summary['nsd_correction']['after'] = summary['methods']['nsd']
    np.savez_compressed(OUT/'all_pair_scores.npz', **arrays)
    experiment.write(OUT/'summary.json', summary)
    with (OUT/'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    local_rows = [dict(method=method,sequence=seq,**values) for method,entry in summary['methods'].items()
                  for seq,values in entry['by_sequence'].items()]
    with (OUT/'per_sequence_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(local_rows[0]));writer.writeheader();writer.writerows(local_rows)
    with (OUT/'pair_scores.csv').open('w', newline='') as handle:
        columns = ['sequence', 'query', 'reference', 'distance', 'flags', *experiment.METHODS, 'nsd_original']
        writer = csv.writer(handle); writer.writerow(columns)
        for i in range(len(arrays['query'])):
            writer.writerow([experiment.SEQUENCES[arrays['sequence'][i]], *[arrays[k][i] for k in columns[1:]]])
    return arrays, masks, weights, summary


def save(fig: plt.Figure, stem: str) -> None:
    """벡터 PDF·SVG와 검토용 PNG를 함께 보존한다."""
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(DEST/f'{stem}.{suffix}', dpi=220, facecolor='white', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)


def matrix_figure(arrays: dict) -> dict:
    """동일 질의의 여섯 방법 점수 순위와 실제 LiDAR 입력을 표시한다."""
    target = experiment.folder('kitti/08')
    matrix = experiment.read(target/'matrix.json')
    query, reference = matrix['query'], matrix['reference']
    match = np.array(matrix['gt_distance']) <= 5
    lookup = {(int(q),int(j)): i for i,(q,j,s) in enumerate(zip(arrays['query'], arrays['reference'], arrays['sequence'])) if s == 0}
    fig = plt.figure(figsize=(3.35,4.35))
    gs = fig.add_gridspec(3, 2, hspace=.30, wspace=.18,
                         left=.17,right=.94,top=.83,bottom=.16)
    thumb = fig.add_gridspec(1,5,left=.10,right=.96,top=.97,bottom=.86,wspace=.12)
    with np.load(target/'thumbnail_points.npz') as points:
        for col,q in enumerate(matrix['centres']):
            ax = fig.add_subplot(thumb[0,col])
            xyz = points[str(q)]
            keep = (np.abs(xyz[:,:2])<40).all(1) & (xyz[:,2]>-2) & (xyz[:,2]<8)
            xyz = xyz[keep]
            ax.scatter(xyz[:,0],xyz[:,1],c=xyz[:,2],s=.09,cmap='cividis',vmin=-2,vmax=8,
                       rasterized=True,linewidths=0)
            ax.plot(0,0,'+',color='#B34E19',ms=3,lw=.8)
            ax.set(xlim=(-40,40),ylim=(-40,40),aspect='equal',xticks=[],yticks=[])
            ax.set_title(f'Place {chr(65+col)}',fontsize=6.5,pad=2)
            for spine in ax.spines.values():spine.set_color('#D4DEE5')
            small, small_ax = plt.subplots(figsize=(1.15,1.0))
            small_ax.scatter(xyz[:,0],xyz[:,1],c=xyz[:,2],s=.12,cmap='cividis',vmin=-2,vmax=8,
                             rasterized=True,linewidths=0)
            small_ax.set(xlim=(-40,40),ylim=(-40,40),aspect='equal',xticks=[],yticks=[])
            small_ax.set_title(f'Place {chr(65+col)}',fontsize=8,pad=2)
            small.tight_layout(pad=.15)
            small.savefig(DEST/f'thumbnail_{chr(65+col)}.png',dpi=220)
            plt.close(small)
    records = {}
    labels = list('ABCDE')
    for panel,method in enumerate(experiment.METHODS):
        ax = fig.add_subplot(gs[panel//2,panel%2])
        score = np.array([[arrays[method][lookup[(q,j)]] for j in reference] for q in query])
        ranks = (rankdata(score, method='average', axis=1)-1)/(len(reference)-1)
        img = ax.imshow(ranks,cmap='viridis',vmin=0,vmax=1,interpolation='nearest')
        correct = int(match[np.arange(len(query)),np.argmax(score,axis=1)].sum())
        for row,col in zip(*np.nonzero(match)):
            ax.add_patch(Rectangle((col-.5,row-.5),1,1,fill=False,edgecolor='white',lw=.7))
        best = np.argmax(score,axis=1)
        ax.scatter(best,np.arange(len(query)),s=9,marker='x',c='#F29C33',linewidths=.65)
        for edge in np.arange(1.5,9,2):
            ax.axvline(edge,c='white',lw=.25,alpha=.55); ax.axhline(edge,c='white',lw=.25,alpha=.55)
        ax.set_title(f'{NAMES[method]}  ({correct}/10)',fontsize=7.5,pad=2,
                     color='#196B24' if method=='nsd' else '#0E2841',fontweight='bold' if method=='nsd' else 'normal')
        ax.set_xticks(np.arange(5)*2+.5,labels,fontsize=6.7)
        ax.set_yticks(np.arange(5)*2+.5,labels,fontsize=6.7)
        ax.tick_params(length=0,pad=1)
        records[method] = {'native_score': score.tolist(),'row_rank': ranks.tolist(),'correct_top_rank':correct}
    fig.text(.11,.49,'Query revisit',va='center',rotation=90,fontsize=6.7)
    fig.text(.55,.105,'Past reference',ha='center',fontsize=6.7)
    cax = fig.add_axes([.23,.035,.61,.012])
    fig.colorbar(img,cax=cax,orientation='horizontal',ticks=[0,.5,1])
    cax.tick_params(labelsize=7,length=2)
    cax.set_xlabel('Within-query score rank ↑',fontsize=7,labelpad=2)
    fig.text(.55,.073,'White: same place (≤5 m)   ×: highest score',ha='center',fontsize=6.5)
    save(fig,'fig_pair_matrix')
    experiment.write(OUT/'matrix_results.json',{'selection':matrix,'methods':records})
    return records


def distributions(arrays: dict, masks: dict, weights: dict, summary: dict) -> None:
    """그룹마다 면적이 1인 원점수 분포를 동일 스캔 쌍에서 비교한다."""
    fig, axes = plt.subplots(3,2,figsize=(3.35,4.0))
    fig.subplots_adjust(left=.16,right=.94,bottom=.13,top=.86,wspace=.36,hspace=.86)
    labels = dict(positive='Same place',random='Random negative',hard='Hard negative')
    legend = [Line2D([0],[0],color=COLOURS[k],lw=1.5,label=labels[k]) for k in labels]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,1.005),ncol=3,
               frameon=False,fontsize=6.5,handlelength=1.3,columnspacing=.8,handletextpad=.4)
    fig.text(.54,.95,'192 queries · KITTI / MulRan / NCLT · seed 1',ha='center',fontsize=6.5)
    chart_data = {}
    for ax, method in zip(axes.flat,experiment.METHODS):
        selected = masks['positive']|masks['random']|masks['hard']
        values = arrays[method]
        bins = np.linspace(values[selected].min(),values[selected].max()+1e-8,33)
        chart_data[method] = {'bin_edges':bins.tolist(),'groups':{}}
        for kind in ('random','hard','positive'):
            keep = masks[kind]
            ax.hist(values[keep],bins=bins,weights=weights[kind][keep],density=True,
                    histtype='step',lw=1.05,color=COLOURS[kind])
            density = np.histogram(values[keep],bins=bins,weights=weights[kind][keep],density=True)[0]
            chart_data[method]['groups'][kind] = density.tolist()
        metric = summary['methods'][method]['best_positive_beats_hard']*100
        ax.set_title(f'{NAMES[method]}',fontsize=7.5,pad=11,
                     color='#196B24' if method=='nsd' else '#0E2841',fontweight='bold' if method=='nsd' else 'normal')
        ax.text(.5,1.01,f'Hard-pool win: {metric:.1f}%',transform=ax.transAxes,
                ha='center',va='bottom',fontsize=6.0)
        ax.set_xlabel(XLABEL[method]+' ↑',fontsize=6.2,labelpad=1)
        ax.tick_params(labelsize=6.7,length=2,pad=1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#E8E8E8',lw=.5)
        ax.set_ylim(bottom=0)
    fig.text(.015,.49,'Density',va='center',rotation=90,fontsize=7)
    fig.text(.54,.028,'Unit-area densities; equal sequence/query weights.\nHard-pool win: 87 queries.',
             ha='center',fontsize=6.5,linespacing=1.25)
    save(fig,'fig_pair_distributions')
    experiment.write(OUT/'distribution_plot_data.json',chart_data)


def documents(summary: dict, matrix: dict) -> None:
    """실험 범위를 명시한 캡션·LaTeX 삽입본과 한국어 검토 자료를 쓴다."""
    (DEST/'data').mkdir(exist_ok=True)
    for name in ('pair_scores.csv', 'metrics.csv', 'per_sequence_metrics.csv', 'protocol.json',
                 'summary.json', 'matrix_results.json', 'distribution_plot_data.json'):
        shutil.copy2(OUT/name, DEST/'data'/name)
    captions = {
        'fig_pair_matrix': (
            'Matching responses at five KITTI 08 places. Colours: within-query score ranks; '
            'white outlines: pairs within 5 m; crosses: top-scoring references. '
            'Parentheses count correct matches among the ten displayed queries.'),
        'fig_pair_distributions': (
            'Native score distributions on identical positive, random-negative, and hard-negative '
            'pairs from KITTI, MulRan, and NCLT. Each distribution has unit area. Higher scores '
            'are better; score scales differ across methods.'),
    }
    for stem, caption in captions.items():
        content = ('% scripts/_pdv2_pair_visualization_finish.py가 검증된 쌍 점수에서 생성한다.\n'
                   '\\begin{figure}[t]\n\\centering\n'
                   f'\\includegraphics[width=0.95\\columnwidth]{{figs/pair_analysis/{stem}.pdf}}\n'
                   '\\caption{\\cmedit{'+caption.replace('%','\\%').replace(' <= ',' $\\leq$ ')+'}}\n'
                   '\\label{fig:'+stem[4:]+'}\n\\end{figure}\n')
        (DEST/f'{stem}.tex').write_text(content)
    experiment.write(DEST/'captions.json', captions)
    body = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>NSD · LiDAR 스캔 쌍 비교</title><style>body{font:16px/1.7 system-ui;color:#172638;max-width:1100px;margin:30px auto;padding:0 20px}img{width:100%;border:1px solid #ddd}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#155a98}.note{background:#f1f5f9;padding:18px}</style>',
            '<h1>LiDAR 스캔 쌍의 매칭 점수 비교</h1>',
            '<p>2026-10-30 · SC++ / RING#-L / HeLiOS / SeqOT / BEVPlace++ / NSD</p>',
            '<p class="note">세 시험 시퀀스에서 고정한 같은 쌍을 평가했다. 학습 방법은 시드 1이다. 후보를 찾는 단계를 생략하고 매칭 점수의 분리 능력을 보는 진단이며 표 2의 최종 검색 성능과 구분한다. 특히 NSD의 GNN 검색 키 기여를 직접 분리한 실험이 아니다.</p>',
            '<h2>1. 유사도 행렬과 실제 입력</h2><a href="fig_pair_matrix.pdf"><img src="fig_pair_matrix.png" alt="실제 라이다 입력과 여섯 방법의 쌍 점수 행렬"></a>',
            '<p>흰 테두리는 정답 쌍이고 주황색 ×는 표시된 참조 중 최고 점수다. 색은 질의 안 순위이므로 서로 다른 방법의 원점수 크기를 비교하지 않는다. 장소는 모델 점수와 무관한 위치 간격으로 골랐다.</p>',
            '<h2>2. 정답·오답 점수 분포</h2><a href="fig_pair_distributions.pdf"><img src="fig_pair_distributions.png" alt="같은 장소와 다른 장소의 원점수 분포"></a>',
            '<p>초록은 같은 장소, 회색은 무작위 다른 장소, 주황은 여섯 방법이 실제 혼동한 다른 장소다. 방법마다 가로축 단위가 다르다. 어려운 오답이 있는 질의에서 최고 정답이 모든 어려운 오답보다 높으면 hard-pool win으로 센다. 실제 검색 성공률은 아니다.</p>',
            '<table><tr><th>방법</th><th>행렬의 정답 최고순위</th><th>어려운 후보군 내 정답 우위</th><th>정답/어려운 오답 AUC</th></tr>']
    for method in experiment.METHODS:
        entry = summary['methods'][method]
        body.append(f'<tr><td>{NAMES[method]}</td><td>{matrix[method]["correct_top_rank"]}/10</td>'
                    f'<td>{100*entry["best_positive_beats_hard"]:.2f}%</td><td>{entry["auc_hard"]:.4f}</td></tr>')
    body.extend(['</table><h2>NSD 보정 전후</h2>'])
    before, after = summary['nsd_correction']['before'],summary['nsd_correction']['after']
    body.append(f'<p>같은 진단 쌍의 어려운 후보군 내 정답 우위는 보정 전 {100*before["best_positive_beats_hard"]:.2f}% → 보정 후 {100*after["best_positive_beats_hard"]:.2f}%다. '
                '이는 고정 쌍의 점수 순서이며 실제 파이프라인의 최종 승자는 25% 보정으로 바뀌지 않는다. 따라서 여기서 후보 순서가 바뀌는 것을 표 2의 R@1 개선으로 해석하지 않는다.</p>')
    body.append('<h2>원고용 파일</h2><p>각 그림의 PDF는 벡터 축·텍스트를 포함하고, SVG·PNG·LaTeX 캡션도 같은 폴더에 있다. 본문 실험 절은 두 LaTeX 조각을 불러와 그림과 캡션을 배치한다.</p>')
    body.append('<p><a href="pair_figures_editable.pptx">편집 가능한 PPT</a> · <a href="data/pair_scores.csv">원점수 CSV</a> · <a href="data/per_sequence_metrics.csv">데이터셋별 수치</a></p>')
    for stem, caption in captions.items():
        body.append(f'<p><a href="{stem}.pdf">{stem} PDF</a> · <a href="{stem}.svg">SVG</a> · <a href="{stem}.tex">LaTeX</a></p><p>{html.escape(caption)}</p>')
    body.append('<p>원천: results/pdv2_pair_visualization_20261030/{protocol.json,pair_scores.csv,summary.json,completion_validation.json}. 같은 질의의 기존 표 2 승자 점수를 별도로 재생성해 원장과 대조했다.</p></html>')
    (DEST/'index.html').write_text('\n'.join(body))


def validate(summary: dict, matrix: dict) -> None:
    """고정 원천·시간 규칙·정답 라벨·수치 출력의 무결성을 검사한다."""
    protocol=experiment.read(OUT/'protocol.json')
    for relative,digest in protocol['sources'].items():
        assert experiment.sha(experiment.REPO/relative)==digest,relative
    for sequence in experiment.SEQUENCES:
        target,pairs,_=experiment.inputs(sequence)
        with np.load(experiment.REPO/protocol['sequences_info'][sequence]['cache']) as data:
            xyz=data['poses_gt'][:,:3,3]
            ids=data['scan_ids']
        q,j=pairs['query'],pairs['reference']
        assert np.all(j<=q-30) and np.all(q+15<len(ids))
        np.testing.assert_array_equal(ids[pairs['indices']],pairs['scan_ids'])
        np.testing.assert_allclose(np.linalg.norm(xyz[q]-xyz[j],axis=1),pairs['distance'],atol=1e-10,rtol=0)
        assert np.all(pairs['distance'][pairs['flags']&experiment.POSITIVE!=0]<=5)
        assert np.all(pairs['distance'][pairs['flags']&(experiment.RANDOM|experiment.HARD)!=0]>25)
        assert len(pairs['selected_queries'])==64
    source_files=[experiment.REPO/'scripts'/name for name in
                  ['_pdv2_pair_visualization.py','_pdv2_pair_visualization_run.py',
                   '_pdv2_pair_visualization_finish.py','pdv2_pair_visualization_scpp.m','_paper_pair_figures_ppt.py']]
    validation={
        'status':'passed','methods':6,'sequences':3,'selected_queries':192,
        'counts':summary['counts'],'score_files':18,
        'ledger_replay_checks':sum(r['ledger_replays'] for r in summary['verification']),
        'max_score_error_by_method':{m:max(r['max_score_error'] for r in summary['verification'] if r['method']==m) for m in experiment.METHODS},
        'fixed_input_hashes_checked':len(protocol['sources']),
        'causal_candidate_and_qplus15_checks':True,'ground_truth_and_scan_id_checks':True,
        'matrix_correct_top_rank':{m:v['correct_top_rank'] for m,v in matrix.items()},
        'source_sha256':{str(p.relative_to(experiment.REPO)):experiment.sha(p) for p in source_files},
        'artifact_sha256':{str(p.relative_to(experiment.REPO)):experiment.sha(p) for p in DEST.rglob('*') if p.is_file()},
        'main_manuscript_includes_figures':all(
            'figs/pair_analysis/'+stem in (experiment.REPO/'docs/paper/body_v2.2/cvpr2027_body.tex').read_text()
            for stem in ('fig_pair_matrix','fig_pair_distributions')),
        'limitations':['시드1·세 시험 시퀀스의 선택된 재방문 질의 진단이다.',
                       '후보 생성과 GNN 검색 키를 우회한 쌍 점수이며 표2 AP·R@1의 대체 결과가 아니다.',
                       '기존 개발에 노출된 시험 자료를 해석한 것으로 새 미노출 확증이나 사전 등록을 주장하지 않는다.'],
    }
    experiment.write(OUT/'completion_validation.json',validation)
    experiment.write(OUT/'status.json',{'status':'complete','artifact_directory':str(DEST.relative_to(experiment.REPO)),
                                       'validation':str((OUT/'completion_validation.json').relative_to(experiment.REPO))})


def legacy_main() -> None:
    """검증을 통과한 실측으로만 두 그림을 내보낸다."""
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Carlito','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','axes.labelcolor':'#0E2841','text.color':'#0E2841',
                         'axes.edgecolor':'#6D7780','axes.linewidth':.6})
    arrays,masks,weights,summary = collect()
    matrix = matrix_figure(arrays)
    distributions(arrays,masks,weights,summary)
    documents(summary,matrix)
    validate(summary,matrix)
    print(json.dumps({'counts':summary['counts'],'matrix_correct':{m:r['correct_top_rank'] for m,r in matrix.items()},
                      'hard_win':{m:r['best_positive_beats_hard'] for m,r in summary['methods'].items()}},indent=2))


def main() -> None:
    """현재 원고의 사례·ROC 생성기로 연결한다. 원천 점수는 유지한다."""
    from _paper_pair_diagnostics import main as current_main
    current_main()


if __name__ == '__main__':
    main()
