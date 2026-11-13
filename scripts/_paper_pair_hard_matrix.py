#!/usr/bin/env python3
"""원래 질의 10개에서 참조 후보만 바꾼 동일 쌍 행렬을 만든다."""
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
OUT = exp.REPO / 'results/paper_pair_hard_matrix_20261030'
SEQUENCE = 'kitti/08'
CAPTION = ('Matching responses with retrieved distractors. Colours: within-query '
           'score ranks; white outlines: same-place pairs; crosses: highest scores.')


def prepare() -> None:
    """기존 질의와 양성을 유지하고 실제 오검색 후보를 점수 계산 전에 고정한다."""
    target = OUT / 'kitti_08'
    if (target / 'matrix.json').exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    original = exp.read(SOURCE / 'kitti_08/matrix.json')
    protocol = exp.read(SOURCE / 'protocol.json')
    cache_path = exp.REPO / protocol['sequences_info'][SEQUENCE]['cache']
    with np.load(cache_path) as data:
        xyz, scan_ids = data['poses_gt'][:, :3, 3], data['scan_ids']
    query = np.array(original['query'], dtype=np.int64)
    positive = original['reference'][::2]
    ledgers = {m: exp.ledger(m, SEQUENCE) for m in exp.METHODS}
    negatives, provenance = [], []
    for group in range(5):
        # 장소별 첫 질의부터 검사한다. 가장 많은 방법이 반환한 오답을
        # 고르고 동률이면 참조 ID가 작은 것을 고른다. 새 쌍 점수는 안 쓴다.
        for q in query[2*group:2*group+2]:
            candidates = {}
            for method, (values, _, _) in ledgers.items():
                i = int(np.flatnonzero(values['query_idx'] == q)[0])
                j = int(values['j_star'][i])
                if (j < 0 or j > query.min()-30 or j in negatives or
                        np.min(np.linalg.norm(xyz[query]-xyz[j], axis=1)) <= 25):
                    continue
                candidates.setdefault(j, []).append(method)
            if candidates:
                j = min(candidates, key=lambda key: (-len(candidates[key]), key))
                negatives.append(j)
                provenance.append(dict(query=int(q), reference=j,
                                       returned_by=candidates[j]))
                break
        else:
            raise ValueError(f'장소 {group}에 공통 시간·거리 조건을 만족하는 오답이 없다.')
    reference = positive + negatives
    distance = np.linalg.norm(xyz[query, None]-xyz[np.array(reference)[None]], axis=2)
    assert len(set(reference)) == 10
    assert np.all(distance[:, 5:] > 25)
    assert np.all((distance[:, :5] <= 5).sum(1) == 1)
    matrix = dict(centres=original['centres'], query=query.tolist(), reference=reference,
                  column_labels=list('ABCDE') + [f'H{i+1}' for i in range(5)],
                  gt_distance=distance.tolist(), hard_reference_provenance=provenance,
                  original_query_unchanged=True, original_positive_reference_subset=positive,
                  selection='기존 GT 간격으로 고른 질의 10개를 유지한다. 장소마다 기존 첫 양성 참조를 '
                  '남긴다. 두 질의의 시간순으로 실제 반환된 참조를 검사하고, 모든 질의보다 30키프레임 '
                  '이상 과거이며 모든 표시 질의에서 25m 초과인 오답 중 반환 방법 수가 가장 큰 것을 '
                  '고른다. 동률은 참조 ID 오름차순이며 이미 고른 오답을 제외한다. 새 쌍 점수로 고르지 않는다.')
    pairs = {(int(q), j): exp.MATRIX for q in query for j in reference}
    replay = {}
    for method, (values, score, _) in ledgers.items():
        replay[method] = []
        for q in original['centres']:
            i = int(np.flatnonzero(values['query_idx'] == q)[0])
            j, value = int(values['j_star'][i]), float(values[score][i])
            pairs[(q, j)] = pairs.get((q, j), 0) | exp.VERIFY
            replay[method].append([q, j, value])
    with np.load(SOURCE / 'kitti_08/pairs.npz') as data:
        indices = data['indices']
        np.testing.assert_array_equal(scan_ids[indices], data['scan_ids'])
    q, j = np.array(sorted(pairs), dtype=np.int64).T
    assert np.isin(q, indices).all() and np.isin(j, indices).all()
    assert np.all(j <= q-30) and np.all(q+15 < len(xyz))
    np.savez_compressed(target/'pairs.npz', query=q, reference=j,
                        flags=np.array([pairs[k] for k in sorted(pairs)], dtype=np.uint8),
                        distance=np.linalg.norm(xyz[q]-xyz[j], axis=1),
                        indices=indices, scan_ids=scan_ids[indices], selected_queries=query)
    lookup = {int(k): i for i, k in enumerate(indices)}
    with (target/'scpp_pairs.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['query', 'reference', 'query_index', 'reference_index'])
        writer.writerows((a, b, lookup[int(a)], lookup[int(b)]) for a, b in zip(q, j))
    features = ['features_bevplace.npy', 'features_helios.npy', 'features_seqot.npy',
                'features_ringsharp.npz', 'desc_000000.mat', 'raw_complete.json', 'scpp_input.json']
    for name in features:
        source = SOURCE / 'kitti_08' / name
        (target/name).symlink_to(os.path.relpath(source, target))
    exp.write(target/'ledger_replay.json', replay)
    exp.write(target/'matrix.json', matrix)
    exp.write(OUT/'protocol.json', protocol)
    exp.write(OUT/'freeze.json', dict(matrix=matrix, models_changed=False,
              source_sha256={str(p.relative_to(exp.REPO)): exp.sha(p) for p in
                             [SOURCE/'all_pair_scores.npz', cache_path,
                              *[SOURCE/'kitti_08'/name for name in features]]},
              pairs_sha256=exp.sha(target/'pairs.npz')))
    print('FROZEN', query.tolist(), reference, 'scored pairs including replays', len(q), flush=True)


def score(method: str) -> None:
    """검증된 공식 매칭 함수를 새 고정 쌍에 실행한다. 특징은 재사용한다."""
    exp.OUT = OUT
    if method in ('bevplace', 'helios', 'seqot'):
        target, pairs, lookup = exp.inputs(SEQUENCE)
        path = target/f'features_{method}.npy'
        descriptors = np.load(path)
        a = descriptors[[lookup[int(q)] for q in pairs['query']]]
        b = descriptors[[lookup[int(j)] for j in pairs['reference']]]
        values = np.sum(a*b, axis=1) if method == 'helios' else -np.linalg.norm(a-b, axis=1)
        exp.save_scores(SEQUENCE, method, values,
                        dict(feature_sha256=exp.sha(path), cached_features_read_only=True))
    elif method == 'scpp':
        exp.collect_scpp(SEQUENCE)
    else:
        getattr(exp, method)(SEQUENCE)


def collect() -> dict:
    """원장 재현과 기존 공통 쌍 점수를 확인하고 10×10 행렬을 조립한다."""
    target = OUT/'kitti_08'
    result = dict(selection=exp.read(target/'matrix.json'), methods={})
    with np.load(target/'pairs.npz') as data:
        q, j = data['query'], data['reference']
    lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(q, j))}
    with np.load(SOURCE/'all_pair_scores.npz') as data:
        old = {k: data[k] for k in data.files}
    overlap = [(i, lookup[(int(a), int(b))]) for i, (a, b, s) in enumerate(
        zip(old['query'], old['reference'], old['sequence']))
        if s == 0 and (int(a), int(b)) in lookup]
    validation = {}
    same = np.array(result['selection']['gt_distance']) <= 5
    for method in exp.METHODS:
        meta = exp.read(target/f'scores_{method}.json')
        assert meta['status'] == 'passed'
        assert meta['pairs_sha256'] == exp.sha(target/'pairs.npz')
        assert meta['score_sha256'] == exp.sha(target/f'scores_{method}.npz')
        with np.load(target/f'scores_{method}.npz') as data:
            values = data['score']
        error = max(abs(float(old[method][a])-float(values[b])) for a, b in overlap)
        assert error <= meta['tolerance'], (method, error)
        native = np.array([[values[lookup[(a, b)]] for b in result['selection']['reference']]
                           for a in result['selection']['query']])
        ranks = (rankdata(native, method='average', axis=1)-1)/9
        result['methods'][method] = dict(native_score=native.tolist(), row_rank=ranks.tolist(),
            correct_top_rank=int(same[np.arange(10), np.argmax(native, axis=1)].sum()))
        validation[method] = dict(overlap_pairs=len(overlap), max_overlap_error=error,
                                  ledger_replays=meta['ledger_replays'],
                                  max_ledger_error=meta['max_score_error'])
    for p, digest in exp.read(OUT/'freeze.json')['source_sha256'].items():
        assert exp.sha(exp.REPO/p) == digest, p
    exp.write(OUT/'matrix_results.json', result)
    exp.write(OUT/'numeric_validation.json', dict(status='passed', query_unchanged=True,
              same_candidates_for_all_methods=True, methods=validation,
              matrix_pairs=100, rows=10, columns=10, true_pairs=int(same.sum())))
    return result


def render() -> None:
    """기존 여섯 패널 행렬의 형식과 한 단 크기를 유지한다."""
    result = collect()
    matrix = result['selection']
    same = np.array(matrix['gt_distance']) <= 5
    plt.rcParams.update({'font.family': 'Carlito', 'font.size': 8, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'text.color': '#0E2841'})
    fig = plt.figure(figsize=(3.35, 4.35))
    grid = fig.add_gridspec(3, 2, left=.17, right=.97, top=.81, bottom=.18,
                           hspace=.43, wspace=.20)
    thumbs = fig.add_gridspec(1, 5, left=.10, right=.96, top=.97, bottom=.86, wspace=.12)
    with np.load(SOURCE/'kitti_08/thumbnail_points.npz') as points:
        for col, q in enumerate(matrix['centres']):
            from _paper_pair_diagnostics import draw_cloud
            ax = fig.add_subplot(thumbs[0, col])
            xyz = points[str(q)]
            keep = (np.abs(xyz[:, :2]) < 40).all(1) & (xyz[:, 2] > -2) & (xyz[:, 2] < 8)
            draw_cloud(ax, xyz[keep])
            ax.set_title('Place '+chr(65+col), fontsize=6.5, pad=2)
    for panel, method in enumerate(exp.METHODS):
        ax = fig.add_subplot(grid[panel//2, panel % 2])
        row = result['methods'][method]
        native, ranks = np.array(row['native_score']), np.array(row['row_rank'])
        img = ax.imshow(ranks, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        for i, j in zip(*np.nonzero(same)):
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor='white', lw=.8))
        ax.scatter(np.argmax(native, axis=1), np.arange(10), s=10, marker='x',
                   c='#F29C33', linewidths=.75)
        ax.axvline(4.5, color='white', lw=.8)
        for edge in np.arange(1.5, 9, 2):
            ax.axhline(edge, color='white', lw=.3, alpha=.55)
        ax.set_title(NAMES[method], fontsize=7.5, pad=3,
                     color='#196B24' if method == 'nsd' else '#0E2841',
                     weight='bold' if method == 'nsd' else 'normal')
        ax.set_xticks(np.arange(10), matrix['column_labels'], fontsize=5.2, rotation=0)
        for label in ax.get_xticklabels()[5:]:
            label.set_rotation(90)
        ax.set_yticks(np.arange(5)*2+.5, list('ABCDE'), fontsize=6.7)
        ax.tick_params(length=0, pad=1)
    fig.text(.10, .49, 'Query revisit', rotation=90, va='center', fontsize=6.7)
    fig.text(.56, .138, 'Past reference: A–E same places | H1–H5 distractors',
             ha='center', fontsize=6.1)
    fig.text(.56, .105, 'White: same place (≤5 m)   ×: highest score', ha='center', fontsize=6.5)
    cax = fig.add_axes([.23, .058, .61, .012])
    fig.colorbar(img, cax=cax, orientation='horizontal', ticks=[0, .5, 1])
    cax.tick_params(labelsize=6.5, length=2)
    cax.set_xlabel('Within-query score rank ↑', fontsize=6.8, labelpad=2)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(DEST/f'fig_pair_matrix.{suffix}', dpi=240, facecolor='white',
                    bbox_inches='tight', pad_inches=.025)
    plt.close(fig)
    (DEST/'fig_pair_matrix.tex').write_text(
        '% scripts/_paper_pair_hard_matrix.py가 공통 후보의 실측 점수로 생성한다.\n'
        '\\begin{figure}[t]\n\\centering\n'
        '\\includegraphics[width=0.95\\columnwidth]{figs/pair_analysis/fig_pair_matrix.pdf}\n'
        f'\\caption{{\\cmedit{{{CAPTION}}}}}\n\\label{{fig:pair_matrix}}\n\\end{{figure}}\n')
    captions = exp.read(DEST/'captions.json')
    captions['fig_pair_matrix'] = CAPTION
    exp.write(DEST/'captions.json', captions)
    shutil.copy2(OUT/'matrix_results.json', DEST/'data/hard_matrix_results.json')
    html = (DEST/'index.html').read_text()
    start, end = html.index('<h1>'), html.index('<img src="fig_pair_matrix.png"')
    html = html[:start] + (
        '<h1>혼동 후보의 점수 행렬과 공통 축 ROC</h1><p>Fig. 5는 기존 KITTI 질의 10개를 유지한 '
        '10×10 매칭 행렬이다. 열 A–E는 장소별 정답 참조, H1–H5는 실제 잘못 검색된 오답 참조다. '
        '여섯 방법에 같은 질의·후보를 준다. 흰 테두리는 5m 이내 정답, ×는 최고 점수다. '
        '색은 행 안 점수 순위로 방법 간 원점수 크기를 비교하지 않는다. 선택한 후보 집합의 진단이며 '
        '전체 검색 정확도로 읽지 않는다.</p>') + html[end:]
    html = html.replace('세 데이터셋의 혼동 사례와 방법별 선호', '동일 질의와 혼동 후보의 여섯 점수 행렬')
    html = html.replace('<a href="data/selected_cases.json">사례 선정 규칙</a> · '
                        '<a href="data/selected_case_scores.csv">사례 원점수</a>',
                        '<a href="data/hard_matrix_results.json">행렬 후보·원점수·순위</a>')
    (DEST/'index.html').write_text(html)
    print({m: r['correct_top_rank'] for m, r in result['methods'].items()}, flush=True)


def main() -> None:
    """선정·실측·시각화를 분리해 원천 점수를 보존한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'render', *exp.METHODS])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'render':
        render()
    else:
        score(args.action)


if __name__ == '__main__':
    main()
