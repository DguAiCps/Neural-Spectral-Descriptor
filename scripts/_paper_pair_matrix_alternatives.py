#!/usr/bin/env python3
"""그림 4 대안 세트의 점수 행렬: 성공·실패 대칭 쌍(1안)과 점수와 무관한 무작위 구간(2안).

선정 규칙은 `select`에서 원장만으로 고정하고 freeze.json에 남긴 뒤 점수를 계산한다.
점수 계산은 `_pdv2_pair_visualization`의 공식 경로와 원장 재대조를 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import _pdv2_pair_visualization as exp

SOURCE = exp.REPO/'results/pdv2_pair_visualization_20261030'
ROOT = exp.REPO/'results/paper_pair_matrix_alternatives_20261031'
SEED = 20261031
SETS = ('option1_win', 'option1_loss', 'option2_random')
ADOPTED = 'option1_win'  # 원고 그림 4에 쓰는 세트
COMPARATORS = tuple(m for m in exp.METHODS if m != 'nsd')
RULES = {
    'common': '세 시험 시퀀스(KITTI 08, MulRan DCC01, NCLT 2012-01-08)의 시드 1 원장에서 여섯 방법 공통 질의 '
              '중 q+15 < N인 질의만 쓴다. 재방문은 j ≤ q−30 안에 GT 5 m 이내 참조가 있는 질의다. 성공은 최종 '
              '1등이 5 m 이내, 오검색은 25 m 초과다. 표시 참조는 모두 j ≤ min(q)−30을 만족한다.',
    'option1_rows': '이긴 쪽 패널: NSD 성공이고 비교 방법 3개 이상이 오검색인 재방문 질의. 진 쪽 패널: NSD 오검색이고 '
                    '비교 방법 3개 이상이 성공인 재방문 질의. 각 방향에서 세 시퀀스 전체의 30키프레임 창 중 해당 질의 수, '
                    '실패 쪽 총 오검색 수, 이른 시작점 순으로 한 창을 고르고 시간순 균등 10질의를 쓴다.',
    'option2_rows': 'MulRan DCC01(현재 그림 4와 같은 시퀀스)에서 재방문 질의가 10개 이상인 30키프레임 창의 시작점 전체 중 '
                    f'RNG {SEED}로 하나를 고르고 그 안의 재방문 질의를 시간순 균등 10개 쓴다. 점수와 성공 여부를 보지 않는다.',
    'positive_columns': '정답 열 5개는 GT로만 정한다. 각 질의의 공통 과거 범위 안 GT 최근접 참조의 합집합에서 시간순 균등 '
                        '5개를 쓰고, 5개 미만이면 질의 순서대로 다음 최근접(5 m 이내)을 더한다.',
    'negative_columns': '오답 열 5개는 실제 오검색 참조 중 모든 표시 질의에서 25 m 초과인 것을 반환 횟수 순(동률은 이른 '
                        '키프레임)으로 쓴다. 1안은 실패 쪽만 센다(이긴 쪽 패널은 비교 방법 5개, 진 쪽 패널은 NSD). '
                        f'2안은 여섯 방법 모두를 센다. 5개 미만이면 25 m 초과 과거 참조에서 RNG {SEED}로 채운다.',
    'scoring': '후보 생성 없이 고정 쌍에 각 방법의 원래 매칭 점수를 적용한다. 각 방법의 실제 최종 승자 점수를 원장과 '
               '재대조한다. 학습 방법은 시드 1이다. 대표 정확도 표본이나 표 2의 대체가 아니다.',
}


def outcomes(sequence: str) -> dict:
    """여섯 방법 원장의 공통 질의, 최종 1등, GT 거리를 모은다."""
    protocol = exp.read(SOURCE/'protocol.json')
    cache = exp.REPO/protocol['sequences_info'][sequence]['cache']
    with np.load(cache) as data:
        xyz, scan_ids = data['poses_gt'][:, :3, 3], data['scan_ids']
    ledgers = {m: exp.ledger(m, sequence) for m in exp.METHODS}
    common = set.intersection(*(set(v[0]['query_idx'].tolist()) for v in ledgers.values()))
    q = np.array(sorted(i for i in common if i+15 < len(xyz)))
    revisit = np.array([i >= 30 and np.linalg.norm(xyz[:i-29]-xyz[i], axis=1).min() <= 5 for i in q])
    top, dist, score = {}, {}, {}
    for m, (values, name, _) in ledgers.items():
        ix = np.searchsorted(values['query_idx'], q)
        np.testing.assert_array_equal(values['query_idx'][ix], q)
        np.testing.assert_array_equal(values['min_past_gt_dist'][ix] <= 5, revisit)
        top[m] = values['j_star'][ix].astype(int)
        dist[m] = np.linalg.norm(xyz[q]-xyz[np.clip(top[m], 0, None)], axis=1)
        dist[m][top[m] < 0] = np.inf
        score[m] = values[name][ix]
    return dict(sequence=sequence, xyz=xyz, scan_ids=scan_ids, cache=cache, ledgers=ledgers,
                q=q, revisit=revisit, top=top, dist=dist, score=score)


def best_window(o: dict, eligible: np.ndarray, failures: np.ndarray) -> tuple:
    """해당 질의 수, 실패 쪽 오검색 수, 이른 시작점 순으로 30키프레임 창을 고른다."""
    rows = [(0, 0, 0, np.array([], int))]
    for start in o['q'][eligible]:
        ix = np.flatnonzero(eligible & (o['q'] >= start) & (o['q'] < start+30))
        rows.append((len(ix), int(failures[ix].sum()), -int(start), ix))
    return max(rows, key=lambda row: row[:3])


def even(values: list, count: int) -> list:
    """정렬한 값에서 시간순 균등 count개를 고른다."""
    values = sorted(values)
    return [values[i] for i in np.linspace(0, len(values)-1, count).round().astype(int)]


def columns(o: dict, qs: np.ndarray, failing: tuple) -> dict:
    """GT 정답 5개와 실패 쪽 실제 오검색 5개를 고른다."""
    rng = np.random.default_rng(SEED)
    xyz, limit = o['xyz'], int(qs.min())-30
    distance = np.linalg.norm(xyz[None, :limit+1]-xyz[qs][:, None], axis=2)
    order = np.argsort(distance, axis=1, kind='stable')
    union = sorted({int(order[i, 0]) for i in range(len(qs))})
    assert all(distance[i, order[i, 0]] <= 5 for i in range(len(qs)))
    positive = even(union, 5) if len(union) >= 5 else list(union)
    rank = 1
    while len(positive) < 5:
        for i in range(len(qs)):
            j = int(order[i, rank])
            if distance[i, j] <= 5 and j not in positive and len(positive) < 5:
                positive.append(j)
        rank += 1
    far = distance.min(0) > 25
    rows = np.searchsorted(o['q'], qs)
    counts: dict[int, int] = {}
    for m in failing:
        for r in rows:
            j = int(o['top'][m][r])
            if 0 <= j <= limit and far[j]:
                counts[j] = counts.get(j, 0)+1
    negative = [j for j, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
    filled = []
    if len(negative) < 5:
        pool = np.setdiff1d(np.flatnonzero(far), negative)
        filled = sorted(int(j) for j in rng.choice(pool, 5-len(negative), replace=False))
        negative += filled
    return dict(positive=sorted(positive), negative=sorted(negative), random_filled=filled,
                error_counts={str(j): c for j, c in sorted(counts.items())})


def freeze_set(name: str, o: dict, qs: np.ndarray, picked: dict, window: dict) -> dict:
    """선정한 쌍과 원장 재대조 쌍을 기존 점수 경로의 입력 형식으로 쓴다."""
    out = ROOT/name
    target = out/o['sequence'].replace('/', '_')
    target.mkdir(parents=True, exist_ok=True)
    xyz = o['xyz']
    reference = sorted(picked['positive']+picked['negative'])
    pairs = {(int(q), int(j)): exp.MATRIX for q in qs for j in reference}
    replay = {}
    for m in exp.METHODS:
        replay[m] = []
        for q in qs:
            r = int(np.searchsorted(o['q'], q))
            j, value = int(o['top'][m][r]), float(o['score'][m][r])
            if j >= 0 and np.isfinite(value):
                pairs[(int(q), j)] = pairs.get((int(q), j), 0) | exp.VERIFY
                replay[m].append([int(q), j, value])
    q, j = np.array(sorted(pairs), dtype=np.int64).T
    assert np.all(j <= q-30) and np.all(q+15 < len(xyz))
    indices = np.unique(np.r_[q, j])
    np.savez_compressed(target/'pairs.npz', query=q, reference=j,
                        flags=np.array([pairs[k] for k in sorted(pairs)], dtype=np.uint8),
                        distance=np.linalg.norm(xyz[q]-xyz[j], axis=1),
                        indices=indices, scan_ids=o['scan_ids'][indices], selected_queries=qs)
    lookup = {int(k): i for i, k in enumerate(indices)}
    with (target/'scpp_pairs.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['query', 'reference', 'query_index', 'reference_index'])
        writer.writerows((a, b, lookup[int(a)], lookup[int(b)]) for a, b in zip(q, j))
    height = exp.read(SOURCE/o['sequence'].replace('/', '_')/'scpp_input.json')['lidar_height']
    exp.write(target/'scpp_input.json', dict(indices=indices.tolist(), lidar_height=height))
    exp.write(target/'ledger_replay.json', replay)
    matrix = dict(sequence=o['sequence'], query=[int(v) for v in qs], reference=reference,
                  positive_references=picked['positive'], negative_references=picked['negative'],
                  gt_distance=np.linalg.norm(xyz[qs, None]-xyz[np.array(reference)[None]], axis=2).tolist())
    exp.write(target/'matrix.json', matrix)
    exp.write(out/'protocol.json', exp.read(SOURCE/'protocol.json'))
    seqot = SOURCE/'canonical_seqot'/f"{o['sequence'].replace('/', '_')}.npz"
    link = out/'canonical_seqot'/seqot.name
    if seqot.exists() and not link.exists():
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(os.path.relpath(seqot, link.parent))
    rows = np.searchsorted(o['q'], qs)
    return dict(sequence=o['sequence'], window=window, query=[int(v) for v in qs], **picked,
                ledger_top1_within_5m={m: int((o['dist'][m][rows] <= 5).sum()) for m in exp.METHODS},
                ledger_top1_beyond_25m={m: int((o['dist'][m][rows] > 25).sum()) for m in exp.METHODS},
                frames=len(indices), pairs=len(q), pairs_sha256=exp.sha(target/'pairs.npz'))


def select() -> None:
    """원장만으로 세 세트의 행과 열을 고정한다. 점수는 보지 않는다."""
    if (ROOT/'freeze.json').exists():
        print('FROZEN_ALREADY'); return
    data = {s: outcomes(s) for s in exp.SEQUENCES}
    frozen, sources = {}, {}
    for o in data.values():
        sources[str(o['cache'].relative_to(exp.REPO))] = exp.sha(o['cache'])
        for _, _, path in o['ledgers'].values():
            sources[str(path.relative_to(exp.REPO))] = exp.sha(path)
    for name, nsd_ok in (('option1_win', True), ('option1_loss', False)):
        best = None
        for o in data.values():
            comparators_far = sum((o['dist'][m] > 25).astype(int) for m in COMPARATORS)
            comparators_ok = sum((o['dist'][m] <= 5).astype(int) for m in COMPARATORS)
            if nsd_ok:
                eligible = o['revisit'] & (o['dist']['nsd'] <= 5) & (comparators_far >= 3)
                failures = comparators_far
            else:
                eligible = o['revisit'] & (o['dist']['nsd'] > 25) & (comparators_ok >= 3)
                failures = (o['dist']['nsd'] > 25).astype(int)
            row = best_window(o, eligible, failures)
            print(name, o['sequence'], 'qualifying', int(eligible.sum()), 'window', row[:3], flush=True)
            if best is None or row[:2] > best[1][:2]:
                best = (o, row)
        o, row = best
        assert row[0] >= 10, (name, row[:3])
        qs = o['q'][row[3][np.linspace(0, row[0]-1, 10).round().astype(int)]]
        picked = columns(o, qs, COMPARATORS if nsd_ok else ('nsd',))
        frozen[name] = freeze_set(name, o, qs, picked, dict(start=-row[2], qualifying_queries=row[0],
                                                             failing_side_errors=row[1]))
    o = data['mulran/DCC01']
    starts = [int(s) for s in o['q'][o['revisit']]
              if (o['revisit'] & (o['q'] >= s) & (o['q'] < s+30)).sum() >= 10]
    start = int(np.random.default_rng(SEED).choice(starts))
    ix = np.flatnonzero(o['revisit'] & (o['q'] >= start) & (o['q'] < start+30))
    qs = o['q'][ix[np.linspace(0, len(ix)-1, 10).round().astype(int)]]
    picked = columns(o, qs, exp.METHODS)
    frozen['option2_random'] = freeze_set('option2_random', o, qs, picked,
                                          dict(start=start, candidate_starts=len(starts), revisit_queries=len(ix)))
    exp.write(ROOT/'freeze.json', dict(rules=RULES, seed=SEED, sets=frozen, source_sha256=sources,
                                       script_sha256=exp.sha(Path(__file__)), models_changed=False,
                                       test_used_for_model_selection=False))
    for name, value in frozen.items():
        print('FROZEN', name, value['sequence'], value['window'], 'frames', value['frames'], flush=True)


def config(name: str) -> tuple:
    """세트별 출력 폴더를 점수 경로에 연결한다."""
    sequence = exp.read(ROOT/'freeze.json')['sets'][name]['sequence']
    exp.OUT = ROOT/name
    return sequence, exp.folder(sequence)


def score(name: str, method: str) -> None:
    """기존 공식 경로로 원본 점군 또는 방법 점수를 만든다."""
    sequence, target = config(name)
    if method in exp.METHODS and (target/f'scores_{method}.json').exists():
        print('VERIFIED_CACHE', name, method); return
    if method in ('bevplace', 'helios', 'seqot'):
        exp.vector_method(sequence, method)
    elif method == 'scpp':
        exp.collect_scpp(sequence)
    else:
        getattr(exp, method)(sequence)


def collect(name: str) -> dict:
    """점수 검증 기록을 확인하고 10×10 행렬과 행 안 순위를 만든다."""
    sequence, target = config(name)
    matrix = exp.read(target/'matrix.json')
    with np.load(target/'pairs.npz') as data:
        lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(data['query'], data['reference']))}
    same = np.array(matrix['gt_distance']) <= 5
    result = dict(selection=dict(set=name, **matrix, row_labels=[str(i+1) for i in range(10)],
                                 column_labels=[str(i+1) for i in range(10)]), methods={})
    for m in exp.METHODS:
        meta = exp.read(target/f'scores_{m}.json')
        assert meta['status'] == 'passed' and meta['pairs_sha256'] == exp.sha(target/'pairs.npz')
        assert meta['score_sha256'] == exp.sha(target/f'scores_{m}.npz')
        with np.load(target/f'scores_{m}.npz') as data:
            values = data['score']
        native = np.array([[values[lookup[(a, b)]] for b in matrix['reference']] for a in matrix['query']])
        result['methods'][m] = dict(native_score=native.tolist(),
            row_rank=((rankdata(native, method='average', axis=1)-1)/9).tolist(),
            correct_top_rank=int(same[np.arange(10), np.argmax(native, axis=1)].sum()),
            ledger_replays=meta['ledger_replays'], max_score_error=meta['max_score_error'])
    exp.write(ROOT/name/'matrix_results.json', result)
    print(name, {m: v['correct_top_rank'] for m, v in result['methods'].items()}, flush=True)
    return result


# 그림 4의 글자·표시 설정. 기본값은 기존 배치이고, 원고 생성기가 style로 덮어쓸 수 있다.
MATRIX_STYLE = dict(
    thumb_font=6.2, title_font=7.5, tick_font=5.8, axis_font=6.7, note_font=6.5,
    cbar_tick_font=6.5, cbar_label_font=6.8,
    grid=dict(left=.075, right=.99, top=.805, bottom=.201, hspace=.30, wspace=.20),
    thumbs=dict(left=.075, right=.99, top=.968, bottom=.864, wspace=.10),
    outer_labels_only=False, xtick_positions=None,
    cross_color='#F29C33', cross_size=10, cross_lw=.8, outline_halo=None,
    query_label=None, note='White: same place (≤5 m)   ×: highest score',
    text_y=dict(reference=.143, note=.097), cax=[.23, .058, .61, .015])


def draw(name: str, stem: Path | None = None, suffixes: tuple = ('pdf', 'png'), style: dict | None = None) -> None:
    """현재 그림 4와 같은 압축 배치로 세트별 행렬을 저장한다."""
    import matplotlib.pyplot as plt
    fig = figure(name, style)
    stem = stem or ROOT/name/f'fig_pair_matrix_{name}'
    for suffix in suffixes:
        fig.savefig(f'{stem}.{suffix}', dpi=240, facecolor='white',
                    bbox_inches='tight', pad_inches=.025)
    plt.close(fig)


def figure(name: str, style: dict | None = None):
    """썸네일 10개, 행렬 6개, 색 막대 순서의 축을 가진 그림 4 배치를 만든다."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle
    s = {**MATRIX_STYLE, **(style or {})}
    from _pdv2_pair_visualization_finish import NAMES
    from _paper_pair_diagnostics import draw_cloud
    result = collect(name)
    selection = result['selection']
    same = np.array(selection['gt_distance']) <= 5
    plt.rcParams.update({'font.family': 'Carlito', 'font.size': 8, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'text.color': '#0E2841'})
    fig = plt.figure(figsize=(3.0, 3.08))
    grid = fig.add_gridspec(2, 3, **s['grid'])
    thumbs = fig.add_gridspec(1, 10, **s['thumbs'])
    with np.load(exp.folder(selection['sequence'])/'thumbnail_points.npz') as points:
        for i, q in enumerate(selection['query']):
            xyz = points[str(q)]
            keep = (np.abs(xyz[:, :2]) < 40).all(1) & (xyz[:, 2] > -2) & (xyz[:, 2] < 8)
            ax = fig.add_subplot(thumbs[0, i])
            draw_cloud(ax, xyz[keep])
            ax.set_title(str(i+1), fontsize=s['thumb_font'], pad=1)
    if s['query_label']:
        fig.text(.005, (s['thumbs']['top'] + s['thumbs']['bottom']) / 2, s['query_label'],
                 rotation=90, va='center', fontsize=s['axis_font'])
    for panel, method in enumerate(exp.METHODS):
        ax = fig.add_subplot(grid[panel//3, panel % 3])
        row = result['methods'][method]
        native, ranks = np.array(row['native_score']), np.array(row['row_rank'])
        img = ax.imshow(ranks, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        halo = ([pe.withStroke(linewidth=s['outline_halo'][0], foreground=s['outline_halo'][1])]
                if s['outline_halo'] else None)
        for i, j in zip(*np.nonzero(same)):
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor='white', lw=.65, path_effects=halo))
        ax.scatter(np.argmax(native, axis=1), np.arange(10), s=s['cross_size'], marker='x', c=s['cross_color'],
                   linewidths=s['cross_lw'], zorder=3)
        ax.set_title(NAMES[method], fontsize=s['title_font'], pad=3, color='#196B24' if method == 'nsd' else '#0E2841',
                     weight='bold' if method == 'nsd' else 'normal')
        xpos = s['xtick_positions'] if s['xtick_positions'] is not None else list(range(10))
        ax.set_xticks(xpos, [selection['column_labels'][k] for k in xpos], fontsize=s['tick_font'])
        ax.set_yticks(np.arange(10), selection['row_labels'], fontsize=s['tick_font'])
        ax.tick_params(length=0, pad=1.3)
        if s['outer_labels_only']:
            ax.tick_params(labelbottom=panel // 3 == 1, labelleft=panel % 3 == 0)
    fig.text(.005, (s['grid']['top'] + s['grid']['bottom']) / 2, 'Query scan', rotation=90, va='center',
             fontsize=s['axis_font'])
    fig.text(.53, s['text_y']['reference'], 'Reference scan', ha='center', fontsize=s['axis_font'])
    fig.text(.53, s['text_y']['note'], s['note'], ha='center', fontsize=s['note_font'])
    cax = fig.add_axes(s['cax'])
    fig.colorbar(img, cax=cax, orientation='horizontal', ticks=[0, .5, 1])
    cax.tick_params(labelsize=s['cbar_tick_font'], length=2)
    cax.set_xlabel('Within-query score rank ↑', fontsize=s['cbar_label_font'], labelpad=2)
    return fig


def install(name: str = ADOPTED, style: dict | None = None) -> None:
    """채택한 세트를 작업 원고의 그림 4로 설치한다. 캡션과 썸네일 질의는 기존 그림과 같다."""
    import shutil
    from _paper_pair_ambiguous_matrix import CAPTION
    from _pdv2_pair_visualization_finish import DEST
    draw(name, DEST/'fig_pair_matrix', ('pdf', 'svg', 'png'), style)
    (DEST/'fig_pair_matrix.tex').write_text(
        '% scripts/_paper_pair_matrix_alternatives.py install이 고정 쌍의 실측 점수로 생성한다.\n'
        '\\begin{figure}[t]\n\\centering\n'
        '\\includegraphics[width=0.95\\columnwidth]{figs/pair_analysis/fig_pair_matrix.pdf}\n'
        f'\\caption{{\\cmedit{{{CAPTION}}}}}\n\\label{{fig:pair_matrix}}\n\\end{{figure}}\n')
    captions = exp.read(DEST/'captions.json')
    captions['fig_pair_matrix'] = CAPTION
    exp.write(DEST/'captions.json', captions)
    shutil.copy2(ROOT/name/'matrix_results.json', DEST/'data/ambiguous_matrix_results.json')
    counts = {m: v['correct_top_rank'] for m, v in exp.read(ROOT/name/'matrix_results.json')['methods'].items()}
    html = (DEST/'index.html').read_text()
    start, end = html.index('<h1>'), html.index('<img src="fig_pair_matrix.png"')
    html = html[:start] + (
        '<h1>혼동이 겹치는 재방문 구간의 점수 행렬</h1><p>Fig. 4는 MulRan DCC01에서 선택한 '
        '질의 10개와 공통 참조 10개의 점수 행렬이다. 질의 행과 참조 열은 각각 1–10으로 번호를 '
        '붙이고, 썸네일 번호는 질의 행과 맞춘다. 같은 번호끼리 반드시 같은 장소라는 뜻은 아니다. '
        '흰 테두리가 5m 이내 정답이고 ×는 최고 점수다. 색은 행 안 점수 순위다.</p>'
        '<p>세 시퀀스에서 NSD가 성공하고 비교 방법 3개 이상이 25m 밖을 반환한 질의가 가장 많은 '
        '30키프레임 창을 원장만으로 고르고, 점수 계산 전에 열을 고정했다. 정답 열 5개는 GT 최근접 '
        '참조, 오답 열 5개는 비교 방법들이 이 질의에서 가장 자주 반환한 25m 밖 참조다. '
        f"표시 열 안 최고 점수가 정답인 질의는 NSD {counts['nsd']}/10, BEVPlace++ {counts['bevplace']}/10, "
        f"SC++ {counts['scpp']}/10, RING#-L {counts['ringsharp']}/10, HeLiOS {counts['helios']}/10, "
        f"SeqOT {counts['seqot']}/10이다. "
        '이 수는 전체 검색 성능이 아니라 선택한 후보군의 점수 비교다.</p>') + html[end:]
    (DEST/'index.html').write_text(html)


def main() -> None:
    """선정 고정·원본 점군·방법별 점수·그리기를 단계별로 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['select', 'raw', *exp.METHODS, 'collect', 'render', 'install'])
    parser.add_argument('sets', nargs='*', help=f'기본값은 전체 {SETS}')
    args = parser.parse_args()
    assert set(args.sets) <= set(SETS), args.sets
    if args.action == 'select':
        select(); return
    if args.action == 'install':
        install(*args.sets); return
    for name in args.sets or SETS:
        if args.action == 'collect':
            collect(name)
        elif args.action == 'render':
            draw(name)
        else:
            score(name, args.action)
    if args.action == 'render' and not args.sets:
        review()


def review() -> None:
    """세 세트의 그림을 제목과 함께 한 장의 검토용 PNG로 붙인다."""
    from PIL import Image, ImageDraw, ImageFont
    titles = {'option1_win': 'Option 1a  NSD succeeds, >=3 comparators fail',
              'option1_loss': 'Option 1b  NSD fails, >=3 comparators succeed',
              'option2_random': 'Option 2  score-independent random window'}
    images = [Image.open(ROOT/name/f'fig_pair_matrix_{name}.png').convert('RGB') for name in SETS]
    font = ImageFont.truetype('/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf', 26)
    small = ImageFont.truetype('/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf', 24)
    head, gap = 90, 40
    sheet = Image.new('RGB', (sum(i.width for i in images)+gap*(len(images)+1),
                              max(i.height for i in images)+head+gap), 'white')
    x = gap
    for name, image in zip(SETS, images):
        sequence = exp.read(ROOT/name/'matrix_results.json')['selection']['sequence']
        ImageDraw.Draw(sheet).text((x, 12), titles[name], fill='#0E2841', font=font)
        ImageDraw.Draw(sheet).text((x, 50), sequence, fill='#0E2841', font=small)
        sheet.paste(image, (x, head))
        x += image.width+gap
    sheet.save(ROOT/'review_three_sets.png')


if __name__ == '__main__':
    main()
