#!/usr/bin/env python3
"""저장된 공통 쌍 점수에서 혼동 사례와 ROC 그림을 만든다. 재추론하지 않는다."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

import _pdv2_pair_visualization as exp
from _pdv2_pair_visualization_finish import DEST, NAMES, group_weights

OUT = exp.REPO / 'results/paper_pair_redesign_20261030'
COLOURS = dict(scpp='#B34E19', ringsharp='#156082', helios='#7B586F',
               seqot='#94791C', bevplace='#753A85', nsd='#196B24')
LINESTYLES = dict(scpp='--', ringsharp='-.', helios=':', seqot=(0, (4, 1, 1, 1)),
                  bevplace='-', nsd='-')
CAPTIONS = {
    'fig_pair_matrix': 'Selected revisit examples with retrieved distractors. '
        'The grid marks the higher-scoring reference: same place (P) or distractor (N).',
    'fig_pair_distributions': 'ROC curves on identical scan pairs with random negatives '
        '(a) or retrieved distractors (b). Bold green: NSD. Upper left is better.',
}


def roc_curve(scores: np.ndarray, positive: np.ndarray, pw: np.ndarray,
              nw: np.ndarray) -> dict:
    """동점을 묶고 클래스별 가중치를 정규화한 ROC를 반환한다."""
    order = np.argsort(-scores, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(order)-1]
    tpr = np.r_[0., np.cumsum(pw[order])[ends] / pw.sum()]
    fpr = np.r_[0., np.cumsum(nw[order])[ends] / nw.sum()]
    tpr[-1] = fpr[-1] = 1.
    assert np.all(np.diff(tpr) >= -1e-12) and np.all(np.diff(fpr) >= -1e-12)
    auc = float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
    return dict(fpr=fpr.tolist(), tpr=tpr.tolist(), auc=auc)


def prepare_rocs(a: dict) -> dict:
    """기존 전체 표본과 가중치를 보존하고 저장된 AUC와 대조한다."""
    summary = exp.read(exp.OUT/'summary.json')
    output = {}
    qids = a['sequence']*1000000+a['query']
    for kind, bit in [('random', exp.RANDOM), ('hard', exp.HARD)]:
        negative = (a['flags'] & bit) != 0
        positive = ((a['flags'] & exp.POSITIVE) != 0) & np.isin(qids, np.unique(qids[negative]))
        assert not np.any(positive & negative)
        pw = group_weights(a['query'], a['sequence'], positive)
        nw = group_weights(a['query'], a['sequence'], negative)
        np.testing.assert_allclose([pw.sum(), nw.sum()], [1., 1.], atol=1e-12)
        keep = positive | negative
        output[kind] = dict(positive_pairs=int(positive.sum()), negative_pairs=int(negative.sum()),
                            queries=int(len(np.unique(qids[keep]))), methods={})
        for m in exp.METHODS:
            curve = roc_curve(a[m][keep], positive[keep], pw[keep], nw[keep])
            np.testing.assert_allclose(curve['auc'], summary['methods'][m]['auc_'+kind], atol=1e-12)
            output[kind]['methods'][m] = curve
    return output


def select_cases(a: dict) -> dict:
    """성공 예시 중 방법 간 판단이 갈리는 사례를 명시적인 규칙으로 고른다."""
    selected, audit = [], {}
    for sid, sequence in enumerate(exp.SEQUENCES):
        ledger, _, _ = exp.ledger('nsd', sequence)
        actual = dict(zip(ledger['query_idx'].tolist(), ledger['top1_geo_dist'].tolist()))
        hard = (a['sequence'] == sid) & ((a['flags'] & exp.HARD) != 0)
        eligible = []
        for q in np.unique(a['query'][hard]):
            if actual[int(q)] > 5:
                continue
            same = (a['sequence'] == sid) & (a['query'] == q)
            positives = np.flatnonzero(same & ((a['flags'] & exp.POSITIVE) != 0))
            p = min(positives, key=lambda i: (a['distance'][i], a['reference'][i]))
            options = []
            for n in np.flatnonzero(same & hard):
                if a['nsd'][p] <= a['nsd'][n]:
                    continue
                disagree = sum(a[m][p] < a[m][n] for m in exp.METHODS if m != 'nsd')
                if disagree:
                    options.append((int(disagree), int(a['reference'][n]), int(n)))
            if not options:
                continue
            count, _, n = sorted(options, key=lambda row: (-row[0], row[1]))[0]
            eligible.append(dict(query=int(q), positive_index=int(p), negative_index=n,
                                 comparison_methods_preferring_negative=count))
        assert eligible, sequence
        maximum = max(row['comparison_methods_preferring_negative'] for row in eligible)
        pool = [row for row in eligible if row['comparison_methods_preferring_negative'] == maximum]
        chosen = dict(pool[len(pool)//2])
        p, n = chosen['positive_index'], chosen['negative_index']
        chosen.update(case=chr(65+sid), sequence=sequence,
                      positive_reference=int(a['reference'][p]), negative_reference=int(a['reference'][n]),
                      positive_distance=float(a['distance'][p]), negative_distance=float(a['distance'][n]),
                      actual_nsd_top1_distance=actual[chosen['query']], methods={})
        for m in exp.METHODS:
            ps, ns = float(a[m][p]), float(a[m][n])
            chosen['methods'][m] = dict(positive_score=ps, negative_score=ns,
                                        preference='P' if ps > ns else ('N' if ps < ns else '='))
        assert chosen['positive_distance'] <= 5 < 25 < chosen['negative_distance']
        assert max(chosen['positive_reference'], chosen['negative_reference']) <= chosen['query']-30
        selected.append(chosen)
        audit[sequence] = dict(eligible=eligible, maximum_disagreement=maximum,
                               maximum_disagreement_queries=[row['query'] for row in pool])
    return dict(selection='시퀀스별 실제 NSD 최종 검색 성공 질의 중 최근접 양성 점수가 어려운 오답보다 '
                '높고 다른 방법이 오답을 선호하는 사례를 고른다. 질의별로 오답을 선호하는 비교 방법 수가 '
                '최대인 오답을 고르고 동률이면 참조 ID가 작은 것을 쓴다. 시퀀스 안에서도 그 수가 최대인 '
                '질의들의 시간순 상위 중앙값을 고른다. 점수를 본 뒤 고른 성공 사례이며 대표 정확도 표본이 아니다.',
                positive_selection='기존 양성 중 GT 최근접; 동률이면 참조 ID 오름차순',
                cases=selected, audit=audit)


def points_for(sequence: str, keyframe: int) -> np.ndarray:
    """점수 계산 때 보존한 실제 XYZ에서 해당 키프레임만 읽는다."""
    folder = exp.folder(sequence)
    with np.load(folder/'pairs.npz') as data:
        indices = data['indices']
    counts = np.fromfile(folder/'chunk_000000.cnt', dtype=np.int64)
    assert len(counts) == len(indices)
    index = int(np.flatnonzero(indices == keyframe)[0])
    offset = int(counts[:index].sum())*12
    xyz = np.fromfile(folder/'chunk_000000.xyz', dtype=np.float32,
                      count=int(counts[index])*3, offset=offset).reshape(-1, 3)
    keep = np.isfinite(xyz).all(1) & (np.abs(xyz[:, :2]) < 40).all(1) & (xyz[:, 2] > -2) & (xyz[:, 2] < 8)
    return xyz[keep]


def draw_cloud(ax: plt.Axes, xyz: np.ndarray) -> None:
    """센서 좌표의 80m 정사각형을 동일한 높이 색으로 표시한다."""
    ax.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2], s=.12, cmap='cividis', vmin=-2, vmax=8,
               rasterized=True, linewidths=0)
    ax.plot(0, 0, '+', color='#B34E19', ms=3, mew=.7)
    ax.set(xlim=(-40, 40), ylim=(-40, 40), aspect='equal', xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_color('#D4DEE5')


def save(fig: plt.Figure, stem: str) -> None:
    """벡터 그림과 검토용 래스터를 같은 범위로 저장한다."""
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(DEST/f'{stem}.{suffix}', dpi=240, facecolor='white', bbox_inches='tight', pad_inches=.025)
    plt.close(fig)


def case_figure(cases: dict) -> None:
    """실제 점군 세 쌍과 여섯 방법의 선호를 한 단에 배치한다."""
    fig = plt.figure(figsize=(3.35, 4.15))
    lefts = [.20, .465, .73]
    for col, title in enumerate(['Query', 'Same place (P)', 'Distractor (N)']):
        fig.text(lefts[col]+.12, .98, title, ha='center', va='top', fontsize=7.2)
    for row, case in enumerate(cases['cases']):
        y = .765-row*.205
        label = ['KITTI\n08', 'MulRan\nDCC01', 'NCLT\n01-08'][row]
        fig.text(.09, y+.09, f"{case['case']}\n{label}", fontsize=6.6, ha='center', va='center')
        for col, (kind, key) in enumerate([('query', 'query'), ('positive', 'positive_reference'),
                                           ('negative', 'negative_reference')]):
            xyz = points_for(case['sequence'], case[key])
            ax = fig.add_axes([lefts[col], y, .24, .16])
            draw_cloud(ax, xyz)
            if col:
                ax.text(.5, -.04, f"{case[kind+'_distance']:.1f} m", transform=ax.transAxes,
                        ha='center', va='top', fontsize=6.5)
            small, sax = plt.subplots(figsize=(1., 1.))
            draw_cloud(sax, xyz)
            small.subplots_adjust(left=.015, right=.985, bottom=.015, top=.985)
            small.savefig(DEST/f"case_{case['case']}_{kind}.png", dpi=220)
            plt.close(small)
    fig.text(.575, .308, 'Higher-scoring reference', fontsize=7.3, ha='center', weight='bold')
    ax = fig.add_axes([.03, .005, .94, .29])
    ax.set(xlim=(0, 3.5), ylim=(0, 7)); ax.axis('off')
    for j, case in enumerate(cases['cases']):
        ax.text(1.85+j*.62, 6.58, case['case'], ha='center', va='center', fontsize=7.5, weight='bold')
    for i, m in enumerate(exp.METHODS):
        y = 5.75-i*.96
        ax.text(.05, y, NAMES[m], ha='left', va='center', fontsize=7.6,
                color=COLOURS[m] if m == 'nsd' else '#0E2841', weight='bold' if m == 'nsd' else 'normal')
        for j, case in enumerate(cases['cases']):
            v = case['methods'][m]['preference']; x = 1.85+j*.62
            fill, ink = ('#D9F2D0', '#196B24') if v == 'P' else ('#FBE2D5', '#B34E19')
            ax.add_patch(Rectangle((x-.26, y-.38), .52, .76, facecolor=fill, edgecolor='none'))
            ax.text(x, y, v, color=ink, fontsize=8, weight='bold', ha='center', va='center')
    save(fig, 'fig_pair_matrix')


def roc_figure(rocs: dict) -> None:
    """전체 ROC를 유지하고 NSD 직접 표기와 백분율 축으로 비교한다."""
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.70))
    fig.subplots_adjust(left=.14, right=.985, top=.86, bottom=.39, wspace=.24)
    for panel, (ax, kind) in enumerate(zip(axes, ['random', 'hard'])):
        ax.plot([0, 100], [0, 100], color='#CDD2D6', lw=.6, zorder=0)
        for m in exp.METHODS:
            c = rocs[kind]['methods'][m]
            ax.plot(np.array(c['fpr'])*100, np.array(c['tpr'])*100,
                    color=COLOURS[m], ls=LINESTYLES[m],
                    lw=2.1 if m == 'nsd' else .95, zorder=5 if m == 'nsd' else 2)
        ax.set(xlim=(0, 100), ylim=(0, 100), xticks=[0, 50, 100], yticks=[0, 50, 100])
        ax.set_title(['(a) Random negatives', '(b) Retrieved distractors'][panel], fontsize=7.2, pad=6)
        ax.tick_params(labelsize=7, pad=2, length=2)
        ax.grid(color='#ECEFF1', lw=.45)
        if panel:
            ax.set_yticklabels([])
        c = rocs[kind]['methods']['nsd']
        x, y = np.array(c['fpr']), np.array(c['tpr'])
        index = np.flatnonzero(x <= .05+1e-12)[-1]
        point = (100*x[index], 100*y[index])
        ax.scatter(*point, s=18, c=COLOURS['nsd'], edgecolor='white', lw=.5, zorder=6)
        ax.annotate('NSD', xy=point, xytext=(40, 45 if panel else 65),
                    fontsize=8, fontweight='bold', color=COLOURS['nsd'],
                    arrowprops={'arrowstyle': '-', 'color': COLOURS['nsd'], 'lw': .75})
    fig.text(.022, .63, 'True match rate (%)', rotation=90, va='center', ha='center', fontsize=7.5)
    fig.text(.56, .285, 'False match rate (%)', ha='center', fontsize=7.5)
    legend_order = ['nsd', 'bevplace', 'ringsharp', 'scpp', 'helios', 'seqot']
    handles = [Line2D([0], [0], color=COLOURS[m], ls=LINESTYLES[m],
                       lw=2.1 if m == 'nsd' else .95, label=NAMES[m]) for m in legend_order]
    legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .035),
                        ncol=3, frameon=False, fontsize=7, columnspacing=.8,
                        handlelength=1.6, handletextpad=.35)
    for text in legend.get_texts():
        if text.get_text() == NAMES['nsd']:
            text.set_weight('bold')
            text.set_color(COLOURS['nsd'])
    save(fig, 'fig_pair_distributions')


def render_rocs() -> None:
    """원점수 재집계와 고정 좌표를 대조하고 ROC 자산만 갱신한다."""
    rocs = exp.read(OUT/'roc_plot_data.json')
    with np.load(exp.OUT/'all_pair_scores.npz') as source:
        rebuilt = prepare_rocs({key: source[key] for key in source.files})
    for kind in rocs:
        for method in exp.METHODS:
            for field in ('fpr', 'tpr', 'auc'):
                np.testing.assert_allclose(rocs[kind]['methods'][method][field],
                                           rebuilt[kind]['methods'][method][field], atol=1e-12)
    with plt.rc_context({'font.family': 'Carlito', 'font.size': 8, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'text.color': '#0E2841',
                         'axes.labelcolor': '#0E2841', 'axes.edgecolor': '#6D7780',
                         'axes.linewidth': .6}):
        roc_figure(rocs)
    caption = CAPTIONS['fig_pair_distributions']
    (DEST/'fig_pair_distributions.tex').write_text(
        '% scripts/_paper_pair_diagnostics.py가 저장된 공통 쌍 점수에서 생성한다.\n'
        '\\begin{figure}[t]\n\\centering\n'
        '\\includegraphics[width=0.95\\columnwidth]{figs/pair_analysis/fig_pair_distributions.pdf}\n'
        f'\\caption{{\\cmedit{{{caption}}}}}\n\\label{{fig:pair_distributions}}\n\\end{{figure}}\n')
    captions = exp.read(DEST/'captions.json')
    captions['fig_pair_distributions'] = caption
    exp.write(DEST/'captions.json', captions)
    path = DEST/'index.html'
    html = path.read_text()
    old = '전체 고정 표본을 사용하며 왼쪽 위가 좋다.'
    new = ('전체 고정 표본과 0–100% 축 범위를 사용하며 왼쪽 위가 좋다. '
           'NSD는 굵은 녹색과 직접 표기로 구분한다. '
           '이 표본의 두 AUC에서 NSD는 BEVPlace++와 RING#-L 다음이며 전체 우위를 뜻하지 않는다.')
    path.write_text(html.replace(old, new))


def documents(cases: dict, rocs: dict) -> None:
    """짧은 캡션과 재현 자료, 한국어 검토 화면을 함께 쓴다."""
    for stem, caption in CAPTIONS.items():
        label = 'fig:pair_matrix' if stem.endswith('matrix') else 'fig:pair_distributions'
        (DEST/f'{stem}.tex').write_text(
            '% scripts/_paper_pair_diagnostics.py가 저장된 공통 쌍 점수에서 생성한다.\n'
            '\\begin{figure}[t]\n\\centering\n'
            f'\\includegraphics[width=0.95\\columnwidth]{{figs/pair_analysis/{stem}.pdf}}\n'
            f'\\caption{{\\cmedit{{{caption}}}}}\n\\label{{{label}}}\n\\end{{figure}}\n')
    exp.write(DEST/'captions.json', CAPTIONS)
    for name, value in [('selected_cases.json', cases), ('roc_plot_data.json', rocs)]:
        exp.write(OUT/name, value)
        shutil.copy2(OUT/name, DEST/'data'/name)
    with (DEST/'data/selected_case_scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'sequence', 'query', 'positive', 'negative', 'method', 'positive_score', 'negative_score', 'preference'])
        for c in cases['cases']:
            for m, v in c['methods'].items():
                writer.writerow([c['case'], c['sequence'], c['query'], c['positive_reference'], c['negative_reference'], m,
                                 v['positive_score'], v['negative_score'], v['preference']])
    rows = ''.join(f"<tr><td>{NAMES[m]}</td><td>{rocs['random']['methods'][m]['auc']:.4f}</td>"
                   f"<td>{rocs['hard']['methods'][m]['auc']:.4f}</td></tr>" for m in exp.METHODS)
    (DEST/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>NSD 공통 쌍 진단</title><style>body{max-width:1000px;margin:30px auto;font:17px sans-serif;line-height:1.6}'
        'img{max-width:450px;vertical-align:top}td,th{padding:5px 18px}table{border-collapse:collapse}</style>'
        '<h1>혼동 사례와 공통 축 ROC</h1><p>Fig. 4는 선택한 NSD 성공 예시다. P는 같은 장소, N은 검색된 다른 장소이며, '
        '표는 둘 중 점수가 높은 것을 표시한다. 전체 정확도로 읽지 않는다. 점군은 센서 좌표의 80m 정사각형이며 높이에 따라 색을 입혔다.</p>'
        '<img src="fig_pair_matrix.png" alt="세 데이터셋의 혼동 사례와 방법별 선호">'
        '<img src="fig_pair_distributions.png" alt="무작위 오답과 검색된 오답에 대한 ROC">'
        '<p>Fig. 5 가로축은 다른 장소를 같은 장소로 잘못 수락하는 비율(FPR), 세로축은 같은 장소를 수락하는 비율(TPR)이다. '
        '전체 고정 표본을 사용하며 왼쪽 위가 좋다. 시퀀스·질의·참조 순으로 같은 가중치를 준다. '
        '무작위 비교 192질의, 어려운 오답 비교 87질의다. 세 시험 시퀀스·시드 1의 쌍 점수 진단이며 표 2 검색 성능과 다르다.</p>'
        '<table><tr><th>방법</th><th>무작위 AUC</th><th>어려운 오답 AUC</th></tr>'+rows+'</table>'
        '<p><a href="pair_figures_editable.pptx">편집용 PPT</a> · <a href="data/selected_cases.json">사례 선정 규칙</a> · '
        '<a href="data/selected_case_scores.csv">사례 원점수</a> · <a href="data/roc_plot_data.json">ROC 좌표와 AUC</a></p></html>')


def legacy_main() -> None:
    """수치 검증 뒤 그림을 출력하고 원천 점수 보존을 확인한다."""
    OUT.mkdir(exist_ok=True); DEST.mkdir(exist_ok=True); (DEST/'data').mkdir(exist_ok=True)
    source = exp.OUT/'all_pair_scores.npz'
    digest = exp.sha(source)
    with np.load(source) as data:
        arrays = {k: data[k] for k in data.files}
    plt.rcParams.update({'font.family': 'Carlito', 'font.size': 8, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.color': '#0E2841',
                         'axes.labelcolor': '#0E2841', 'axes.edgecolor': '#6D7780', 'axes.linewidth': .6})
    cases, rocs = select_cases(arrays), prepare_rocs(arrays)
    case_figure(cases); roc_figure(rocs); documents(cases, rocs)
    assert exp.sha(source) == digest
    validation = dict(status='passed', original_scores_sha256=digest, models_changed=False,
                      cases=[{k: v for k, v in c.items() if k != 'methods'} for c in cases['cases']],
                      roc_auc_matches_all_12_original_values=True,
                      roc_counts={k: {n: v for n, v in r.items() if n != 'methods'} for k, r in rocs.items()},
                      roc_ties_grouped=True, roc_class_weights_sum_to_one=True,
                      case_selection_is_posthoc_illustrative=True)
    exp.write(OUT/'numeric_validation.json', validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


def main() -> None:
    """현재 그림 4의 혼동 구간 행렬과 그림 5의 최종 판정 비교를 출력한다."""
    from _paper_pair_matrix_alternatives import install as install_matrix
    from _paper_precision_figure import install
    install_matrix()
    install()


if __name__ == '__main__':
    main()
