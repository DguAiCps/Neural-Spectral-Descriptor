"""새 보고모델(2026-11-16) 결과에서 시간 창·무지연 표 세 개를 만든다(v2.4).

scripts/_paper_temporal_tables_v23.py에서 출발했다. v2.3은 원고 문장에 적힌 수치만 옮긴 temporal_source.json에서 표를 만들었고
문장에 없는 절대값은 대시로 두었다. v2.4는 결과 JSON에 모든 절대값과 짝지은 구간이 있으므로 build()가 결과 JSON에서
temporal_source.json(schema 2)을 다시 쓰고, generate()가 그 파일만 읽어 표를 만든다.

남기는 행(사용자 결정 2026-11-16): 학습 없는 균등 감산 대조(대칭, 앞 5·뒤 5의 같은 세기/선택 세기, 정제 없음+양방향 수락),
대칭 NSD, Past-only NSD(과거 10), 과거 10 균등 감산, 정제 없음+과거 5 수락. 사전학습 기반 학습 변형은 모두 뺀다.
narratives(원고 문단)는 손대지 않고 기존 temporal_source.json의 것을 그대로 옮긴다(문장 수정은 원고 단위가 한다).
"""
from pathlib import Path
from decimal import Decimal
import argparse
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional

GROUPS = (("all", "Overall"), ("forward_only", "Forward"), ("reverse_only", "Reverse"))
ROOT = Path(__file__).resolve().parents[1]


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
    """전체·순방향·역방향 지표 열을 만든다."""
    return [r'\begin{tabular}{@{}lcccccc@{}}\toprule',
            r'Variant & \multicolumn{2}{c}{Overall} & \multicolumn{2}{c}{Forward-only} & \multicolumn{2}{c}{Reverse-only} \\',
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
            r' & AP & R@100P & AP & R@100P & AP & R@100P \\ \midrule']


def absolute_row(row: Dict[str, Any]) -> str:
    """절대 성능 한 행을 만든다."""
    parts = [row['label']]
    for g in ('overall', 'forward', 'reverse'):
        for m in ('AP', 'R100'):
            parts.append(cell(row['metrics'].get(g + '_' + m)))
    return ' & '.join(parts) + r' \\'


def stack_cell(item: Optional[Dict[str, Any]]) -> str:
    """차이와 신뢰구간을 두 줄로 조판한다."""
    if item is None:
        return r'\textemdash'
    d = normalized(item)
    return (r'$\vcenter{\hbox{\shortstack{$' + d['estimate'] + r'$\\'
            r'{\scriptsize $[' + d['lower'] + ', ' + d['upper'] + r']$}}}}$')


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
    """결과 JSON에서 temporal_source.json(schema 2)을 다시 쓴다. narratives는 기존 파일의 것을 그대로 둔다."""
    previous = json.loads(path.read_text())
    narratives = previous['narratives']
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
    conditions = [
        dict(label='Symmetric uniform', window='$\\pm1,\\ldots,\\pm5$', scope='None', strength='$-0.5$', acceptance='Two-sided', lag=None, group='training_free'),
        dict(label='Preceding five, equal', window=pre, scope='None', strength='$-0.5$', acceptance='Two-sided', lag='$5$', group='training_free'),
        dict(label='Preceding five, selected', window=pre, scope='None', strength=strength(sel['past']), acceptance='Two-sided', lag='$5$', group='training_free'),
        dict(label='Following five, equal', window=fol, scope='None', strength='$-0.5$', acceptance='Two-sided', lag=None, group='training_free'),
        dict(label='Following five, selected', window=fol, scope='None', strength=strength(sel['future']), acceptance='Two-sided', lag=None, group='training_free'),
        dict(label='No refining, two-sided acceptance', window='None', scope='None', strength=None, acceptance='Two-sided', lag=None, group='training_free'),
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
    cap = (r'Temporal refining and acceptance conditions. Lag is the number of keyframes observed after the query; n/a denotes inapplicable strength. '
           r'Full learning updates the NSD modules' + rv(' from scratch with the strength fixed') + '.')
    t = start_table(cap, 'tab:temporal_window_controls')
    t += [r'\begin{tabular}{@{}llllll@{}}\toprule',
          r'Variant & Refining offsets & Learning & Strength & Acceptance & Lag \\ \midrule']
    last = None
    for row in d['conditions']:
        if row['group'] != last:
            if last is not None:
                t.append(r'\addlinespace[2pt]')
            heading = {'training_free': 'Training-free uniform-subtraction controls', 'learned': 'Learned model with two-sided acceptance',
                       'zero_lag': 'Zero-lag configurations: past-only acceptance'}[row['group']]
            t.append(r'\multicolumn{6}{@{}l}{\emph{' + heading + r'}} \\')
            last = row['group']
        t.append(' & '.join(row[k] if row[k] is not None else ('not stated' if k == 'lag' else 'n/a')
                            for k in ('label', 'window', 'scope', 'strength', 'acceptance', 'lag')) + r' \\')
    t += [r'\bottomrule\end{tabular}',
          r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize',
          r'Past-only NSD expands the candidate-side correspondence radius from 10 to 20 keyframes; the radius is twice the maximum temporal offset. '
          r'Learned ' + rv('models') + ' use three ' + rv('seeds') + ' and 512 ' + rv('training') + ' updates.',
          r'\end{minipage}', r'\end{table*}']
    (out / 'appdx_temporal_window_controls.tex').write_text('\n'.join(t) + '\n')
    cap = (r'Absolute performance for temporal-window and zero-lag controls (\%). Learned values are mean $\pm$ SD over three ' + rv('seeds') + '. '
           + rv(r'Symmetric NSD is the model of Table~\ref{tab:results}.'))
    t = start_table(cap, 'tab:temporal_window_results') + absolute_header()
    t.append(r'\multicolumn{7}{@{}l}{\emph{Two-sided acceptance: training-free models and NSD}} \\')
    t += [absolute_row(row) for row in d['window_absolute']]
    t += [r'\addlinespace[2pt]', r'\multicolumn{7}{@{}l}{\emph{Past-only acceptance (zero lag)}} \\']
    t += [absolute_row(row) for row in d['zero_absolute']]
    t += [r'\bottomrule\end{tabular}', r'\end{table*}']
    (out / 'appdx_temporal_window_results.tex').write_text('\n'.join(t) + '\n')
    cap = (r'Paired temporal-window and zero-lag differences in percentage points, reported as first named condition minus second. '
           r'Conditional 95\% intervals use paired sequence resampling; equal-strength rows use $-0.5$, and uniform denotes uniform subtraction. '
           + rv('Window and acceptance conditions') + r' are listed in Table~\ref{tab:temporal_window_controls}.')
    t = start_table(cap, 'tab:temporal_window_deltas')
    t += [r'\label{tab:zero_lag_controls}', r'\begin{tabular}{@{}lccc@{}}\toprule',
          r'Comparison & Overall & Forward-only & Reverse-only \\ \midrule']
    rows = d['window_differences'] + d['zero_differences'] + d['extra_differences']
    for metric in ('AP', 'R@100P'):
        if metric != 'AP':
            t.append(r'\midrule')
        t.append(r'\multicolumn{4}{@{}l}{\textbf{' + metric + r'}} \\')
        for comparison, groups in comparison_rows(rows, metric).items():
            t.append(comparison + ' & ' + ' & '.join(stack_cell(groups.get(g)) for g in ('Overall', 'Forward', 'Reverse')) + r' \\')
            t.append(r'\addlinespace[1.5pt]')
    t += [r'\bottomrule\end{tabular}', r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize']
    f1 = next(r['metrics']['F1'] for r in rows if r['comparison'].startswith('Preceding five, selected') and r['group'] == 'Overall')
    t += ['For preceding five, selected $-$ symmetric uniform, the overall F1 difference is ' + cell(f1) + '.',
          r'\end{minipage}', r'\end{table*}']
    (out / 'appdx_temporal_window_deltas.tex').write_text('\n'.join(t) + '\n')
    n = {item['id']: item['tex'] for item in d['narratives']}
    inp = lambda name: r'\input{' + (prefix.rstrip('/') + '/' if prefix else '') + name + '}'
    chunks = [n['window_setup'], n['window_learning'], inp('appdx_temporal_window_controls'), inp('appdx_temporal_window_results'),
              inp('appdx_temporal_window_deltas'), n['window_readout'], n['zero_setup'], n['zero_readout']]
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
