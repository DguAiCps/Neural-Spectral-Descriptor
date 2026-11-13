#!/usr/bin/env python3
"""방법별 혼동이 겹치는 재방문 구간을 공통 후보 행렬로 시각화한다."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import rankdata

import _pdv2_pair_visualization as exp
from _pdv2_pair_visualization_finish import DEST, NAMES

SOURCE = exp.REPO / 'results/pdv2_pair_visualization_20261030'
OUT = exp.REPO / 'results/paper_pair_ambiguous_matrix_20261030'
SEQUENCE = 'mulran/DCC01'
CAPTION = ('Selected ambiguous revisit in MulRan DCC01. Colours: within-query '
           'score ranks; white outlines: same-place pairs; crosses: highest scores.')


def prepare() -> None:
    """원장으로 질의 구간과 후보 전체를 고정하고 재계산 입력을 만든다."""
    target = OUT/'mulran_DCC01'
    if (OUT/'freeze.json').exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    protocol = exp.read(SOURCE/'protocol.json')
    cache = exp.REPO/protocol['sequences_info'][SEQUENCE]['cache']
    with np.load(cache) as data:
        xyz, scan_ids = data['poses_gt'][:, :3, 3], data['scan_ids']
    ledgers = {m: exp.ledger(m, SEQUENCE) for m in exp.METHODS}
    query = ledgers['nsd'][0]['query_idx']
    refs, distances = [], []
    for m in exp.METHODS:
        values = ledgers[m][0]
        ix = np.searchsorted(values['query_idx'], query)
        np.testing.assert_array_equal(values['query_idx'][ix], query)
        j = values['j_star'][ix]
        d = np.linalg.norm(xyz[query]-xyz[j], axis=1)
        d[j < 0] = np.inf
        refs.append(j); distances.append(d)
    refs, distances = np.array(refs), np.array(distances)
    far = (distances[:-1] > 25).sum(0)
    eligible = (distances[-1] <= 5) & (far >= 3)
    windows = []
    for start in query[eligible]:
        ix = np.flatnonzero(eligible & (query >= start) & (query < start+30))
        windows.append((len(ix), int(far[ix].sum()), -int(start), ix))
    window = max(windows, key=lambda row: row[:3])
    assert window[0] >= 10
    selected = window[3][np.linspace(0, window[0]-1, 10).round().astype(int)]
    qs = query[selected]
    positive = sorted(set(refs[-1, selected].tolist()))
    assert len(positive) == 5
    negative = sorted({int(j) for j in refs[:-1, selected].ravel()
                       if 0 <= j <= qs.min()-30 and
                       np.min(np.linalg.norm(xyz[qs]-xyz[j], axis=1)) > 25})
    pool = sorted(positive+negative)
    assert set(positive).isdisjoint(negative) and len(negative) >= 5
    pairs = {(int(q), int(j)): exp.MATRIX for q in qs for j in pool}
    replay = {}
    for m, (values, score, _) in ledgers.items():
        replay[m] = []
        for q in qs:
            i = int(np.flatnonzero(values['query_idx'] == q)[0])
            j = int(values['j_star'][i])
            pairs[(int(q), j)] = pairs.get((int(q), j), 0) | exp.VERIFY
            replay[m].append([int(q), j, float(values[score][i])])
    q, j = np.array(sorted(pairs), dtype=np.int64).T
    indices = np.unique(np.r_[q, j])
    assert np.all(j <= q-30) and np.all(q+15 < len(xyz))
    np.savez_compressed(target/'pairs.npz', query=q, reference=j,
                        flags=np.array([pairs[k] for k in sorted(pairs)], dtype=np.uint8),
                        distance=np.linalg.norm(xyz[q]-xyz[j], axis=1),
                        indices=indices, scan_ids=scan_ids[indices], selected_queries=qs)
    lookup = {int(k): i for i, k in enumerate(indices)}
    with (target/'scpp_pairs.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['query', 'reference', 'query_index', 'reference_index'])
        writer.writerows((a, b, lookup[int(a)], lookup[int(b)]) for a, b in zip(q, j))
    exp.write(target/'scpp_input.json', dict(indices=indices.tolist(), lidar_height=1.693))
    exp.write(target/'ledger_replay.json', replay)
    matrix = dict(sequence=SEQUENCE, query=qs.tolist(), reference=pool,
                  positive_references=positive, negative_pool=negative,
                  gt_distance=np.linalg.norm(xyz[qs, None]-xyz[np.array(pool)[None]], axis=2).tolist())
    exp.write(target/'matrix.json', matrix)
    exp.write(OUT/'protocol.json', protocol)
    canonical = OUT/'canonical_seqot'
    canonical.mkdir(exist_ok=True)
    (canonical/'mulran_DCC01.npz').symlink_to(
        os.path.relpath(SOURCE/'canonical_seqot/mulran_DCC01.npz', canonical))
    exp.write(OUT/'freeze.json', dict(matrix=matrix, models_changed=False,
        query_selection='NSD 실제 검색이 5m 이내이고 비교 방법 3개 이상이 25m 초과인 질의를 '
        '모은다. 30키프레임 창 중 해당 질의 수, 비교 방법의 총 원거리 오검색 수, 이른 시작점 '
        '순으로 선택하고 시간순 균등 10질의를 쓴다. MulRan DCC01의 사후 선택 성공 사례다.',
        reference_selection='정답은 해당 질의에서 NSD가 반환한 참조 5개다. 모든 표시 질의에서 '
        '25m 초과이고 공통 과거 후보 범위를 만족하는 실제 오검색 참조를 전부 점수화한다. '
        '그 뒤 서로 다른 비교 방법의 점수 혼동을 포함하도록 오답 5개를 선택한다. '
        '새 설정 선택이나 대표 정확도 표본으로 해석하지 않는다.',
        window_start=-window[2], eligible_queries_in_window=window[0],
        comparator_far_errors_in_window=window[1], all_eligible_queries=query[eligible].tolist(),
        source_sha256={str(p.relative_to(exp.REPO)): exp.sha(p)
                       for p in [cache, *[row[2] for row in ledgers.values()]]},
        pairs_sha256=exp.sha(target/'pairs.npz')))
    print('FROZEN', matrix, 'frames', len(indices), flush=True)


def score(method: str) -> None:
    """고정 모델과 공식 전처리·매칭으로 새 쌍을 계산한다."""
    exp.OUT = OUT
    if method in ('bevplace', 'helios', 'seqot'):
        exp.vector_method(SEQUENCE, method)
    elif method == 'scpp':
        exp.collect_scpp(SEQUENCE)
    else:
        getattr(exp, method)(SEQUENCE)


def collect() -> dict:
    """전체 후보 점수를 검증하고 비교 방법의 혼동을 드러내는 열을 고른다."""
    target = OUT/'mulran_DCC01'
    pool = exp.read(target/'matrix.json')
    with np.load(target/'pairs.npz') as data:
        q, j = data['query'], data['reference']
    lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(q, j))}
    scores, checks = {}, {}
    for method in exp.METHODS:
        meta = exp.read(target/f'scores_{method}.json')
        assert meta['status'] == 'passed'
        assert meta['pairs_sha256'] == exp.sha(target/'pairs.npz')
        assert meta['score_sha256'] == exp.sha(target/f'scores_{method}.npz')
        with np.load(target/f'scores_{method}.npz') as data:
            values = data['score']
        scores[method] = np.array([[values[lookup[(a, b)]] for b in pool['reference']]
                                   for a in pool['query']])
        checks[method] = meta
    gt = np.array(pool['gt_distance']) <= 5
    coverage = {}
    for candidate in pool['negative_pool']:
        col = pool['reference'].index(candidate)
        coverage[candidate] = {(m, i) for m in exp.METHODS if m != 'nsd' for i in range(10)
                               if scores[m][i, col] > scores[m][i, gt[i]].max()}
    selected, covered, covered_methods = [], set(), set()
    for _ in range(5):
        remaining = [j for j in pool['negative_pool'] if j not in selected]
        def priority(j: int) -> tuple:
            events = coverage[j]-covered
            return (len({m for m, _ in events}-covered_methods), len(events), -j)
        candidate = max(remaining, key=priority)
        selected.append(candidate)
        covered |= coverage[candidate]
        covered_methods |= {m for m, _ in coverage[candidate]}
    reference = sorted(pool['positive_references']+selected)
    cols = [pool['reference'].index(j) for j in reference]
    same = gt[:, cols]
    result = dict(selection=dict(sequence=SEQUENCE, query=pool['query'], reference=reference,
                   row_labels=[str(i+1) for i in range(10)],
                   column_labels=[str(i+1) for i in range(10)],
                   gt_distance=np.array(pool['gt_distance'])[:, cols].tolist(),
                   positive_references=pool['positive_references'], selected_distractors=selected,
                   posthoc_illustrative_selection=True), methods={})
    for method in exp.METHODS:
        native = scores[method][:, cols]
        result['methods'][method] = dict(native_score=native.tolist(),
            row_rank=((rankdata(native, method='average', axis=1)-1)/9).tolist(),
            correct_top_rank=int(same[np.arange(10), np.argmax(native, axis=1)].sum()))
    for p, digest in exp.read(OUT/'freeze.json')['source_sha256'].items():
        assert exp.sha(exp.REPO/p) == digest, p
    exp.write(OUT/'matrix_results.json', result)
    exp.write(OUT/'pool_scores.json', dict(selection=pool, methods={m: s.tolist() for m, s in scores.items()},
              negative_coverage={str(j): sorted(coverage[j]) for j in coverage}))
    exp.write(OUT/'numeric_validation.json', dict(status='passed', same_candidates_for_all_methods=True,
              rows=10, columns=10, full_pool_columns=len(pool['reference']),
              comparator_methods_with_included_scoring_errors=sorted(covered_methods),
              checks=checks, models_changed=False, posthoc_illustrative_selection=True))
    print({m: row['correct_top_rank'] for m, row in result['methods'].items()}, flush=True)
    return result


def render() -> None:
    """행·열·점군에 일관된 숫자 번호를 쓴 여섯 매트릭스를 그린다."""
    result = collect()
    selection = result['selection']
    same = np.array(selection['gt_distance']) <= 5
    plt.rcParams.update({'font.family': 'Carlito', 'font.size': 8, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'text.color': '#0E2841'})
    fig = plt.figure(figsize=(3.0, 3.08))
    grid = fig.add_gridspec(2, 3, left=.075, right=.99, top=.805, bottom=.201,
                           hspace=.30, wspace=.20)
    thumbs = fig.add_gridspec(1, 10, left=.075, right=.99, top=.968, bottom=.864,
                             wspace=.10)
    from _paper_pair_diagnostics import draw_cloud
    with np.load(OUT/'mulran_DCC01/thumbnail_points.npz') as points:
        for i, q in enumerate(selection['query']):
            xyz = points[str(q)]
            keep = (np.abs(xyz[:, :2]) < 40).all(1) & (xyz[:, 2] > -2) & (xyz[:, 2] < 8)
            ax = fig.add_subplot(thumbs[0, i])
            draw_cloud(ax, xyz[keep])
            ax.set_title(str(i+1), fontsize=6.2, pad=1)
            small, sax = plt.subplots(figsize=(.8, .8))
            draw_cloud(sax, xyz[keep])
            small.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
            small.savefig(DEST/f'query_{i+1:02d}.png', dpi=220)
            plt.close(small)
    for panel, method in enumerate(exp.METHODS):
        ax = fig.add_subplot(grid[panel//3, panel % 3])
        row = result['methods'][method]
        native, ranks = np.array(row['native_score']), np.array(row['row_rank'])
        img = ax.imshow(ranks, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        for i, j in zip(*np.nonzero(same)):
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor='white', lw=.65))
        ax.scatter(np.argmax(native, axis=1), np.arange(10), s=10, marker='x',
                   c='#F29C33', linewidths=.8)
        ax.set_title(NAMES[method], fontsize=7.5, pad=3,
                     color='#196B24' if method == 'nsd' else '#0E2841',
                     weight='bold' if method == 'nsd' else 'normal')
        ax.set_xticks(np.arange(10), selection['column_labels'], fontsize=5.8)
        ax.set_yticks(np.arange(10), selection['row_labels'], fontsize=5.8)
        ax.tick_params(length=0, pad=1.3)
    fig.text(.005, .503, 'Query scan', rotation=90, va='center', fontsize=6.7)
    fig.text(.53, .143, 'Reference scan', ha='center', fontsize=6.7)
    fig.text(.53, .097, 'White: same place (≤5 m)   ×: highest score', ha='center', fontsize=6.5)
    cax = fig.add_axes([.23, .058, .61, .015])
    fig.colorbar(img, cax=cax, orientation='horizontal', ticks=[0, .5, 1])
    cax.tick_params(labelsize=6.5, length=2)
    cax.set_xlabel('Within-query score rank ↑', fontsize=6.8, labelpad=2)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(DEST/f'fig_pair_matrix.{suffix}', dpi=240, facecolor='white',
                    bbox_inches='tight', pad_inches=.025)
    plt.close(fig)
    (DEST/'fig_pair_matrix.tex').write_text(
        '% scripts/_paper_pair_ambiguous_matrix.py가 공통 후보의 실측 점수로 생성한다.\n'
        '\\begin{figure}[t]\n\\centering\n'
        '\\includegraphics[width=0.95\\columnwidth]{figs/pair_analysis/fig_pair_matrix.pdf}\n'
        f'\\caption{{\\cmedit{{{CAPTION}}}}}\n\\label{{fig:pair_matrix}}\n\\end{{figure}}\n')
    captions = exp.read(DEST/'captions.json')
    captions['fig_pair_matrix'] = CAPTION
    exp.write(DEST/'captions.json', captions)
    shutil.copy2(OUT/'matrix_results.json', DEST/'data/ambiguous_matrix_results.json')
    html = (DEST/'index.html').read_text()
    start, end = html.index('<h1>'), html.index('<img src="fig_pair_matrix.png"')
    html = html[:start] + (
        '<h1>혼동이 겹치는 재방문 구간의 점수 행렬</h1><p>Fig. 4는 MulRan DCC01에서 선택한 '
        '질의 10개와 공통 참조 10개의 점수 행렬이다. 질의 행과 참조 열은 각각 1–10으로 번호를 '
        '붙이고, 썸네일 번호는 질의 행과 맞춘다. 같은 번호끼리 반드시 같은 장소라는 뜻은 아니다. '
        '흰 테두리가 5m 이내 정답이고 ×는 최고 점수다. 색은 행 안 점수 순위다.</p>'
        '<p>NSD 실제 검색 성공과 여러 비교 방법의 오검색이 겹치는 구간을 사후 선택했다. '
        '후보 28개를 모두 점수화한 뒤 서로 다른 비교 방법의 혼동을 포함하도록 오답 5개를 '
        '고르고 정답 참조 5개와 합쳤다. NSD는 표시 후보에서 9/10, 비교 방법들은 0/10이며, '
        'NSD의 나머지 한 최고 점수는 GT 거리 5.082m로 정답 경계를 조금 벗어난다. '
        '이 수는 전체 검색 성능이 아니라 선택한 후보군의 점수 비교다.</p>') + html[end:]
    html = html.replace('동일 질의와 혼동 후보의 여섯 점수 행렬', 'MulRan 혼동 구간의 여섯 점수 행렬')
    html = html.replace('data/hard_matrix_results.json', 'data/ambiguous_matrix_results.json')
    (DEST/'index.html').write_text(html)


def main() -> None:
    """질의 선정·전체 후보 점수화·표시 후보 선정·그리기를 분리한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'collect', 'render', 'raw', *exp.METHODS])
    args = parser.parse_args()
    if args.action in ('prepare', 'collect', 'render'):
        globals()[args.action]()
    else:
        score(args.action)


if __name__ == '__main__':
    main()
