"""새 보고모델(2026-11-16) 결과에서 시간 창·무지연 표 세 개를 만든다(v2.4).

scripts/_paper_temporal_tables_v23.py에서 출발했다. v2.3은 원고 문장에 적힌 수치만 옮긴 temporal_source.json에서 표를 만들었고
문장에 없는 절대값은 대시로 두었다. v2.4는 결과 JSON에 모든 절대값과 짝지은 구간이 있으므로 build()가 결과 JSON에서
temporal_source.json(schema 2)을 다시 쓰고, generate()가 그 파일만 읽어 표를 만든다.

남기는 행(사용자 결정 2026-11-16): 학습 없는 균등 감산 대조(대칭, 앞 5·뒤 5의 같은 세기/선택 세기, 정제 없음+양방향 수락),
대칭 NSD, Past-only NSD(과거 10), 과거 10 균등 감산, 정제 없음+과거 5 수락. 사전학습 기반 학습 변형은 모두 뺀다.
narratives(원고 문단)는 기존 temporal_source.json의 것을 옮기고, 원고에서 고친 문장(NARRATIVE_FIXES)만 적용한다.
"""
from pathlib import Path
from decimal import Decimal
import argparse
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional

GROUPS = (("all", "Overall"), ("forward_only", "Forward"), ("reverse_only", "Reverse"))
ROOT = Path(__file__).resolve().parents[1]
# 원고(v2.7) temporal_controls.tex에서 고친 문장. build()가 옮기는 narratives에 적용한다(이미 고친 원천이면 그대로 둔다).
NARRATIVE_FIXES = {
    'window_setup': (('two-sided temporal acceptance are shared', 'two-sided temporal evidence are shared'),
                     ('Validation AP selects each one-sided strength from $\\{-0.125,-0.25,-0.5,-1\\}$: $-0.125$ for the preceding window (80.69\\%) and $-0.25$ for the following window (80.97\\%), versus 83.34\\% for the symmetric window at $-0.5$.',
                      'The one-sided strengths are set empirically: $-0.125$ for the preceding window and $-0.25$ for the following window.'),
                     # 조건표를 결과표에 합치고(2026-11-21) 부록의 방향별 표(tab:direction)는 본문 방향 표로 대신한다.
                     (r'Table~\ref{tab:temporal_window_controls} distinguishes the refining, acceptance, and training conditions. Tables~\ref{tab:temporal_window_results} and~\ref{tab:temporal_window_deltas} give test values and paired differences',
                      r'Table~\ref{tab:temporal_window_results} lists the refining, acceptance, and training conditions with test values, and Table~\ref{tab:temporal_window_deltas} gives paired differences'),
                     (r'Direction groups follow Table~\ref{tab:direction}.', r'Direction groups follow Table~\ref{tab:direction_all}.'),
                     # 2026-11-22 구성 변경: 같은 세기 대조와 줄인 세기 대조를 문단으로 나눈다(세기 문장과 역방향 설명은 window_readout으로).
                     (' The one-sided strengths are set empirically: $-0.125$ for the preceding window and $-0.25$ for the following window.', ''),
                     (', including the equal-strength controls at $-0.5$', ''),
                     (' On a reverse revisit, one-sided windows contrast the query and stored place against opposite sides of the road.', '')),
    'window_readout': ((r' and a median of 14.69\,m of travel', ''),
                       ('Selecting the preceding strength on validation narrows the reverse-only gap',
                        'At its own strength of $-0.125$, the preceding window narrows the reverse-only gap'),
                       ('The equal-strength preceding window has', 'At the same strength of $-0.5$, the preceding window has'),
                       ('the following window shows the same direction asymmetry. At its own strength of $-0.125$, the preceding window narrows',
                        'the following window shows the same direction asymmetry. On a reverse revisit, one-sided windows contrast the query and '
                        'stored place against opposite sides of the road.\n\nWe also evaluate each one-sided window at a reduced strength, set '
                        'empirically to $-0.125$ for the preceding window and $-0.25$ for the following window. The preceding window then narrows'),
                       ('stays close to no refining. Symmetric NSD uses', 'stays close to no refining.\n\nSymmetric NSD uses'),
                       ('On a reverse revisit, one-sided windows contrast the query and stored place against opposite sides of the road.',
                        'On a reverse revisit, a one-sided window contrasts the query and the stored place against opposite stretches of the '
                        'road: one against the segment before the place, the other against the segment after it.')),
    'zero_setup': ((r", matching NSD's per-layer neighbor count", ''),
                   ('Following the rule of twice the maximum temporal offset, the candidate-side correspondence radius is 20 keyframes, increased from 10.',
                    'As in NSD, the candidate-side correspondence radius is twice the maximum temporal offset, which gives 20 keyframes instead of 10.'),
                   (' Query-side acceptance also uses ten preceding keyframes.', ''),
                   ('the 75:25 mixture are shared with NSD; the mean center weight',
                    r'the mixing weight $\lambda$ are shared with NSD; the mean weight of the center pair'),
                   ('for query-side acceptance. These three configurations use no keyframes after the query.',
                    'for query-side temporal evidence.'),
                   (r'\paragraph{Zero-lag configurations.}', r'\paragraph{Past-only configurations.}')),
    'zero_readout': (('than symmetric NSD.\n\n',
                      'than symmetric NSD. Among the zero-lag configurations, the training-free configuration without refining has the highest '
                      'AP and R@100P over all revisits. Past-only NSD is 0.66 AP points below past-only uniform subtraction ($[-1.02, -0.31]$), '
                      'while the R@100P interval includes zero ($-0.25$ $[-1.72, 1.23]$). The gains from refining and learning reported for NSD therefore '
                      'hold for the symmetric window.\n\n'),
                     ('Among the zero-lag configurations', 'Among the past-only configurations'),
                     ('therefore hold for the symmetric window.', 'are therefore specific to the symmetric window.')),
}


def normalized(item: Dict[str, Any]) -> Dict[str, str]:
    """비교 방향에 맞게 차이와 구간 끝점을 변환한다."""
    d = item['original']
    x, lo, hi = (Decimal(d[k]) for k in ('estimate', 'lower', 'upper'))
    if item['reverse_direction']:
        x, lo, hi = -x, -hi, -lo
    assert lo <= hi
    return {'estimate': f'{x:+.2f}', 'lower': f'{lo:.2f}', 'upper': f'{hi:.2f}'}


def cell(item: Optional[Dict[str, Any]]) -> str:
    """절대값이나 차이 한 칸을 조판한다."""
    if item is None:
        return r'\textemdash'
    if 'original' in item:
        d = normalized(item)
        return '${estimate}\\;[{lower}, {upper}]$'.format(**d)
    return '$' + item['value'] + (r'\pm' + item['sd'] if item.get('sd') else '') + '$'


def start_table(caption: str, label: str) -> List[str]:
    """표의 공통 머리 부분을 만든다."""
    return ['% temporal_source.json에서 생성. 숫자는 생성 결과를 직접 고치지 않는다.',
            r'\begin{table*}[!t]', r'\caption{' + caption + r'}\label{' + label + '}',
            r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',
            r'\renewcommand{\arraystretch}{1.12}']


def absolute_header() -> List[str]:
    """조건(세기·수락·지연) 열과 전체·순방향·역방향 지표 열을 만든다."""
    return [r'\begin{tabular}{@{}lclccccccc@{}}\toprule',
            r'Variant & Strength & Acceptance & Lag & \multicolumn{2}{c}{Overall} & \multicolumn{2}{c}{Forward-only} & \multicolumn{2}{c}{Reverse-only} \\',
            r'\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}',
            r' & & & & AP & R@100P & AP & R@100P & AP & R@100P \\ \midrule']


def absolute_row(row: Dict[str, Any], condition: Dict[str, Any]) -> str:
    """조건과 절대 성능 한 행을 만든다."""
    assert condition['acceptance'] and condition['lag']
    parts = [row['label'], condition['strength'] or 'n/a', condition['acceptance'], condition['lag']]
    for g in ('overall', 'forward', 'reverse'):
        for m in ('AP', 'R100'):
            parts.append(cell(row['metrics'].get(g + '_' + m)))
    return ' & '.join(parts) + r' \\'


def comparison_rows(rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    """같은 비교의 전체·순방향·역방향 값을 한 행으로 묶는다."""
    groups = {}
    for row in rows:
        if row['metrics'][metric] is not None:
            groups.setdefault(row['comparison'], {})[row['group']] = row['metrics'][metric]
    return groups


# ----------------------------------------------------------------- 결과 JSON → temporal_source.json
def _absolute(value_pct: float, sd_pct: Optional[float], source: Dict[str, Any]) -> Dict[str, Any]:
    return {'value': f'{value_pct:.2f}', 'sd': None if sd_pct is None else f'{sd_pct:.2f}', 'result_source': source}


def _difference(delta: float, ci: List[float], source: Dict[str, Any]) -> Dict[str, Any]:
    return {'original': {'estimate': f'{delta:+.2f}', 'lower': f'{ci[0]:.2f}', 'upper': f'{ci[1]:.2f}'},
            'reverse_direction': False, 'result_source': dict(source, value={'delta_pp': delta, 'ci95_pp': list(ci)})}


def build(path: Path, summary: dict, controls: dict, window: dict, selection: dict, extra: dict, paths: Dict[str, Path]) -> dict:
    """결과 JSON에서 temporal_source.json(schema 2)을 다시 쓴다. narratives는 기존 파일의 것에 NARRATIVE_FIXES만 적용한다."""
    previous = json.loads(path.read_text())
    narratives = previous['narratives']
    for item in narratives:
        for old, new in NARRATIVE_FIXES.get(item['id'], ()):
            assert old in item['tex'] or new in item['tex'], (item['id'], old)
            item['tex'] = item['tex'].replace(old, new, 1)
    sel = selection['selected']
    assert selection['test_used'] is False and window['test_used_for_selection'] is False and window['status'] == 'passed'
    assert sel == window['selection']['selected'] and window['arms']['past_selected'] == f"past_{str(-sel['past']).replace('.', 'p')}"
    assert window['arms']['future_selected'] == f"future_{str(-sel['future']).replace('.', 'p')}"
    assert extra['status'] == 'passed' and controls['updates'] == 512
    strength = lambda v: f'${v:g}$'
    # NSD·균등 감산은 세 원천에서 같은 원장이다.
    for g, _ in GROUPS:
        for k in ('AP', 'max_F1', 'R@100P', 'R@1'):
            nsd = controls['test'][g]['arms']['nsd'][k]['mean']
            assert abs(window['groups'][g]['nsd_percent'][k] - 100 * nsd) < 1e-9
            assert abs(window['groups'][g]['percent']['symmetric'][k] - 100 * controls['test'][g]['arms']['uniform'][k]['mean']) < 1e-9
            if g == 'all':
                assert abs(nsd - summary['mean']['diff'][k]) < 1e-12
    pre, fol = '$-5,\\ldots,-1$', '$+1,\\ldots,+5$'
    # Lag는 원고 temporal_source.json(2026-11-18 원고에서 고친 조건)을 따른다: 질의 뒤에 필요한 최대 키프레임 수.
    conditions = [
        dict(label='Symmetric uniform', window='$\\pm1,\\ldots,\\pm5$', scope='None', strength='$-0.5$', acceptance='Two-sided', lag='$15$', group='training_free'),
        dict(label='Preceding five, equal', window=pre, scope='None', strength='$-0.5$', acceptance='Two-sided', lag='$5$', group='training_free'),
        dict(label='Preceding five, selected', window=pre, scope='None', strength=strength(sel['past']), acceptance='Two-sided', lag='$5$', group='training_free'),
        dict(label='Following five, equal', window=fol, scope='None', strength='$-0.5$', acceptance='Two-sided', lag='$15$', group='training_free'),
        dict(label='Following five, selected', window=fol, scope='None', strength=strength(sel['future']), acceptance='Two-sided', lag='$15$', group='training_free'),
        dict(label='No refining, two-sided acceptance', window='None', scope='None', strength=None, acceptance='Two-sided', lag='$5$', group='training_free'),
        # 새 보고모델은 대비 세기를 −0.5로 고정해 학습한다(게이트 고정).
        dict(label='Symmetric NSD', window='$\\pm1,\\ldots,\\pm5$', scope='Full', strength='$-0.5$', acceptance='Two-sided', lag='$15$', group='learned'),
        dict(label='Past-only NSD', window='$-10,\\ldots,-1$', scope='Full', strength='$-0.5$', acceptance='Preceding ten', lag='$0$', group='zero_lag'),
        dict(label='Preceding ten, uniform', window='$-10,\\ldots,-1$', scope='None', strength='$-0.5$', acceptance='Preceding ten', lag='$0$', group='zero_lag'),
        dict(label='No refining, past-five acceptance', window='None', scope='None', strength=None, acceptance='Preceding five', lag='$0$', group='zero_lag'),
    ]
    # 이전 판과 같은 조건 행은 조건 문자열(lag 포함)이 같아야 한다(세기·범위만 새 구성에 맞춘다).
    old_conditions = {r['label']: r for r in previous['conditions']}
    for row in conditions:
        if row['label'] in old_conditions and row['group'] == 'training_free':
            assert all(row[k] == old_conditions[row['label']][k] for k in ('window', 'scope', 'strength', 'acceptance', 'lag')), row['label']

    def window_row(label, arm):
        metrics = {}
        for g, name in GROUPS:
            for m, short in (('AP', 'AP'), ('R@100P', 'R100')):
                v = window['groups'][g]['percent'][arm][m]
                metrics[name.lower() + '_' + short] = _absolute(v, None, {'file_id': 'window', 'key': ['groups', g, 'percent', arm, m]})
        return dict(label=label, metrics=metrics, group='training_free')

    def control_row(label, arm, group):
        metrics = {}
        for g, name in GROUPS:
            for m, short in (('AP', 'AP'), ('R@100P', 'R100')):
                a = controls['test'][g]['arms'][arm][m]
                metrics[name.lower() + '_' + short] = _absolute(100 * a['mean'], None if a['sd'] is None else 100 * a['sd'],
                                                                {'file_id': 'controls', 'key': ['test', g, 'arms', arm, m]})
        return dict(label=label, metrics=metrics, group=group)

    window_absolute = [window_row('Symmetric uniform', 'symmetric'), window_row('Preceding five, equal', 'past_same'),
                       window_row('Preceding five, selected', 'past_selected'), window_row('Following five, equal', 'future_same'),
                       window_row('Following five, selected', 'future_selected'),
                       window_row('No refining, two-sided acceptance', 'no_refining'),
                       control_row('Symmetric NSD', 'nsd', 'learned')]
    zero_absolute = [control_row('Past-only NSD', 'pastonly', 'zero_lag'),
                     control_row('Preceding ten, uniform', 'uniform_past10', 'zero_lag'),
                     window_row('No refining, past-five acceptance', 'no_refining_past5_acceptance')]
    zero_absolute[-1]['group'] = 'zero_lag'

    def diff_rows(comparison, getter, file_id, key):
        out = []
        for g, name in GROUPS:
            metrics = {}
            for m in ('AP', 'F1', 'R@100P'):
                k = 'max_F1' if m == 'F1' else m
                delta, ci = getter(g, k)
                metrics[m] = _difference(delta, ci, {'file_id': file_id, 'key': key(g, k)})
            out.append(dict(comparison=comparison, group=name, metrics=metrics))
        return out

    ws = lambda arm: (lambda g, k: (window['groups'][g]['minus_symmetric'][arm][k]['mean_pp'], window['groups'][g]['minus_symmetric'][arm][k]['ci95_pp']))
    wkey = lambda arm: (lambda g, k: ['groups', g, 'minus_symmetric', arm, k])
    cs = lambda pair: (lambda g, k: (controls['test'][g]['differences'][pair]['point_pp'][k], controls['test'][g]['differences'][pair]['conditional_95_pp'][k]))
    ckey = lambda pair: (lambda g, k: ['test', g, 'differences', pair, k])
    window_differences = (diff_rows('Preceding five, equal $-$ symmetric uniform', ws('past_same'), 'window', wkey('past_same'))
                          + diff_rows('Following five, equal $-$ symmetric uniform', ws('future_same'), 'window', wkey('future_same'))
                          + diff_rows('Preceding five, selected $-$ symmetric uniform', ws('past_selected'), 'window', wkey('past_selected')))
    zero_differences = (diff_rows('Symmetric NSD $-$ Past-only NSD', cs('nsd-pastonly'), 'controls', ckey('nsd-pastonly'))
                        + diff_rows('Past-only NSD $-$ preceding ten, uniform', cs('pastonly-uniform_past10'), 'controls', ckey('pastonly-uniform_past10'))
                        + diff_rows('Past-only NSD $-$ no refining, past-five acceptance',
                                    lambda g, k: (extra['groups'][g]['delta_pp'][k], extra['groups'][g]['conditional_95_pp'][k]),
                                    'extra', lambda g, k: ['groups', g, k]))
    # 차이의 점추정은 두 절대값의 차이와 같아야 한다.
    for g, _ in GROUPS:
        for k in ('AP', 'R@100P'):
            a = controls['test'][g]['arms']
            assert abs(controls['test'][g]['differences']['nsd-pastonly']['point_pp'][k] - 100 * (a['nsd'][k]['mean'] - a['pastonly'][k]['mean'])) < 1e-9
            assert abs(extra['groups'][g]['delta_pp'][k]
                       - (100 * a['pastonly'][k]['mean'] - window['groups'][g]['percent']['no_refining_past5_acceptance'][k])) < 1e-9
    sources = {name: {'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for name, p in paths.items()}
    d = {'schema_version': 2,
         'note': '2026-11-17 새 보고모델(시드 4·5·6) 결과 JSON에서 scripts/_paper_temporal_tables_v24.py build()가 생성. narratives는 기존 파일의 것을 그대로 옮긴다(상태는 narratives_status).',
         'missing_value_policy': 'All absolute metrics come from the result JSONs; no value is reconstructed from differences.',
         'direction_policy': 'All displayed differences are first named condition minus second.',
         'result_sources': sources, 'selection': selection,
         'conditions': conditions, 'window_absolute': window_absolute, 'zero_absolute': zero_absolute,
         'window_differences': window_differences, 'zero_differences': zero_differences, 'extra_differences': [],
         'narratives': narratives,
         'narratives_status': previous.get('narratives_status', 'v2.3 문단 그대로. 수치·조건이 새 보고모델과 다르다(원고 단위가 고친다).')}
    path.write_text(json.dumps(d, ensure_ascii=False, indent=1) + '\n')
    return d


def generate(source: Path, out: Path, prefix: str, rv: Callable[[str], str] = lambda s: s) -> Dict[str, Any]:
    """구조화 원천에서 세 표와 교체 문단을 생성한다. rv는 v2.3에서 바뀐 캡션 구절 표시 함수다."""
    d = json.loads(source.read_text())
    assert d['schema_version'] == 2
    out.mkdir(parents=True, exist_ok=True)
    cap = (r'Temporal refining and acceptance conditions. Lag denotes the maximum required look-ahead after the query, in keyframes, derived from the refining and acceptance windows; n/a denotes inapplicable strength. '
           r'Full learning updates the NSD modules' + rv(' from random initialization with the strength fixed') + '.')
    t = start_table(cap, 'tab:temporal_window_controls')
    t += [r'\begin{tabular}{@{}llllll@{}}\toprule',
          r'Variant & Refining offsets & Learning & Strength & Acceptance & Lag \\ \midrule']
    last = None
    for row in d['conditions']:
        if row['group'] != last:
            if last is not None:
                t.append(r'\addlinespace[2pt]')
            heading = {'training_free': 'Training-free uniform-subtraction controls', 'learned': 'Learned model with two-sided acceptance',
                       'zero_lag': 'Past-only configurations'}[row['group']]
            t.append(r'\multicolumn{6}{@{}l}{\emph{' + heading + r'}} \\')
            last = row['group']
        t.append(' & '.join(row[k] if row[k] is not None else ('not stated' if k == 'lag' else 'n/a')
                            for k in ('label', 'window', 'scope', 'strength', 'acceptance', 'lag')) + r' \\')
    t += [r'\bottomrule\end{tabular}', r'\end{table*}']
    (out / 'appdx_temporal_window_controls.tex').write_text('\n'.join(t) + '\n')
    # 원고는 조건표를 불러오지 않는다(2026-11-21 결과표에 합침). 조건 열과 정의는 결과표가 싣는다.
    cap = (r'Conditions and absolute performance for temporal-window and past-only controls (\%). '
           r'Lag denotes the maximum required look-ahead after the query, in keyframes, derived from the refining and acceptance windows; n/a denotes inapplicable strength. '
           r'Learned models update the NSD modules' + rv(' from random initialization with the strength fixed')
           + r'; their values are mean $\pm$ SD over three ' + rv('seeds') + '. '
           + rv(r'Symmetric NSD is the model of Table~\ref{tab:results}.'))
    conditions = {row['label']: row for row in d['conditions']}
    assert [row['label'] for row in d['window_absolute'] + d['zero_absolute']] == list(conditions)
    # 조건 열을 더해 전체 폭을 넘으므로 다른 부록 표처럼 폭에 맞춰 줄인다(자연 크기보다 키우지 않는다).
    t = start_table(cap, 'tab:temporal_window_results') + [r'\sbox0{%'] + absolute_header()
    # 같은 세기(-0.5) 대조를 먼저 묶고, 세기를 줄인 한쪽 창은 따로 묶는다. 줄 이름의 세기 구분은 Strength 열이 대신한다.
    shown = {'Preceding five, equal': 'Preceding five', 'Preceding five, selected': 'Preceding five',
             'Following five, equal': 'Following five', 'Following five, selected': 'Following five',
             'No refining, two-sided acceptance': 'No refining'}
    by_label = {row['label']: row for row in d['window_absolute']}
    same = ['No refining, two-sided acceptance', 'Symmetric uniform', 'Preceding five, equal', 'Following five, equal', 'Symmetric NSD']
    reduced = ['Preceding five, selected', 'Following five, selected']
    assert sorted(same + reduced) == sorted(by_label)
    line = lambda label: absolute_row(dict(by_label[label], label=shown.get(label, label)), conditions[label])
    t.append(r'\multicolumn{10}{@{}l}{\emph{Two-sided acceptance}} \\')
    t += [line(label) for label in same]
    t += [r'\addlinespace[2pt]', r'\multicolumn{10}{@{}l}{\emph{Two-sided acceptance, one-sided windows at reduced strength}} \\']
    t += [line(label) for label in reduced]
    t += [r'\addlinespace[2pt]', r'\multicolumn{10}{@{}l}{\emph{Past-only acceptance}} \\']
    t += [absolute_row(row, conditions[row['label']]) for row in d['zero_absolute']]
    t += [r'\bottomrule\end{tabular}}', r'\ifdim\wd0>\textwidth\resizebox{\textwidth}{!}{\usebox0}\else\usebox0\fi', r'\end{table*}']
    (out / 'appdx_temporal_window_results.tex').write_text('\n'.join(t) + '\n')
    cap = (r'Paired temporal-window and past-only differences in percentage points, reported as first named condition minus second. '
           r'The 95\% intervals use paired sequence resampling; rows use strength $-0.5$ unless stated, and uniform denotes uniform subtraction. '
           + rv('Window and acceptance conditions') + r' are listed in Table~\ref{tab:temporal_window_results}.')
    t = start_table(cap, 'tab:temporal_window_deltas')
    t += [r'\label{tab:zero_lag_controls}', r'\begin{tabular}{@{}lccc@{}}\toprule',
          r'Comparison & Overall & Forward-only & Reverse-only \\ \midrule']
    rows = d['window_differences'] + d['zero_differences'] + d['extra_differences']
    for metric in ('AP', 'R@100P'):
        if metric != 'AP':
            t.append(r'\midrule')
        t.append(r'\multicolumn{4}{@{}l}{\textbf{' + metric + r'}} \\')
        for comparison, groups in comparison_rows(rows, metric).items():
            comparison = (comparison.replace(', equal $-$', ' $-$')
                          .replace('Preceding five, selected $-$', r'Preceding five ($\eta=' + conditions['Preceding five, selected']['strength'].strip('$') + '$) $-$'))
            t.append(comparison + ' & ' + ' & '.join(cell(groups.get(g)) for g in ('Overall', 'Forward', 'Reverse')) + r' \\')
    t += [r'\bottomrule\end{tabular}', r'\end{table*}']
    (out / 'appdx_temporal_window_deltas.tex').write_text('\n'.join(t) + '\n')
    n = {item['id']: item['tex'] for item in d['narratives']}
    inp = lambda name: r'\input{' + (prefix.rstrip('/') + '/' if prefix else '') + name + '}'
    chunks = [n['window_setup'], n['window_learning'], inp('appdx_temporal_window_results'),
              inp('appdx_temporal_window_deltas'), n['window_readout'], n['zero_setup'], n['zero_readout']]
    chunks = [c for c in chunks if c]  # 원고에서 뺀 문단은 빈 문자열로 둔다(2026-11-20 window_learning).
    (out / 'temporal_controls.tex').write_text('\n\n'.join(chunks) + '\n')
    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    paper = ROOT / 'docs/paper/body_v2.4'
    parser.add_argument('--source', type=Path, default=paper / 'temporal_source.json')
    parser.add_argument('--output-dir', type=Path, default=paper / 'tables_v3')
    parser.add_argument('--input-prefix', default='tables_v3')
    args = parser.parse_args()
    generate(args.source, args.output_dir, args.input_prefix)
