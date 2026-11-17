"""구조화된 원문 수치에서 시간창·무지연 표와 교체 문단을 생성한다."""
from pathlib import Path
from decimal import Decimal
import argparse
import json
from typing import Any, Dict, List, Optional


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
        return '${estimate}\;[{lower}, {upper}]$'.format(**d)
    return '$' + item['value'] + (r'\pm' + item['sd'] if item.get('sd') else '') + '$'


def start_table(caption: str, label: str) -> List[str]:
    """표의 공통 머리 부분을 만든다."""
    return ['% temporal_source.json에서 생성. 숫자는 생성 결과를 직접 고치지 않는다.',
            r'\begin{table*}[!t]',r'\caption{' + caption + r'}\label{' + label + '}',
            r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',
            r'\renewcommand{\arraystretch}{1.12}']


def absolute_header(with_strength: bool) -> List[str]:
    """전체·순방향·역방향 지표 열을 만든다."""
    n = 2 if with_strength else 1
    col = 'llcccccc' if with_strength else 'lcccccc'
    lead = 'Variant & Strength & ' if with_strength else 'Variant & '
    cmid = r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}' if with_strength else r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}'
    return [r'\begin{tabular}{@{}'+col+r'@{}}\toprule',
            lead+r'\multicolumn{2}{c}{Overall} & \multicolumn{2}{c}{Forward-only} & \multicolumn{2}{c}{Reverse-only} \\',
            cmid,
            (' & ' * n) + r'AP & R@100P & AP & R@100P & AP & R@100P \\ \midrule']


def absolute_row(row: Dict[str, Any], strength: bool = False) -> str:
    """절대 성능 한 행을 만든다."""
    parts=[row['label']]
    if strength:
        parts.append('$'+row['strength']+'$' if row.get('strength') is not None else r'\textemdash')
    for g in ('overall','forward','reverse'):
        for m in ('AP','R100'):
            parts.append(cell(row['metrics'].get(g+'_'+m)))
    return ' & '.join(parts) + r' \\'


def difference_panel(rows: List[Dict[str, Any]], include_f1: bool) -> List[str]:
    """방향별 행으로 차이 패널을 만든다."""
    metrics=['AP','F1','R@100P'] if include_f1 else ['AP','R@100P']
    columns=1+len(metrics)
    out=[r'\begin{tabular}{@{}l' + 'c'*len(metrics) + r'@{}}\toprule',
         'Query group & ' + ' & '.join(metrics) + r' \\ \midrule']
    last=None
    for row in rows:
        if row['comparison'] != last:
            if last is not None:out.append(r'\addlinespace[2pt]')
            out.append(r'\multicolumn{'+str(columns)+r'}{@{}l}{\emph{'+row['comparison']+r'}} \\')
            last=row['comparison']
        out.append(row['group']+' & '+' & '.join(cell(row['metrics'][m]) for m in metrics)+r' \\')
    out.append(r'\bottomrule\end{tabular}')
    return out


def stack_cell(item: Optional[Dict[str, Any]]) -> str:
    """차이와 신뢰구간을 두 줄로 조판한다."""
    if item is None:
        return r'\textemdash'
    d = normalized(item)
    return (r'$\vcenter{\hbox{\shortstack{$'+d['estimate']+r'$\\'
            r'{\scriptsize $['+d['lower']+', '+d['upper']+r']$}}}}$')


def comparison_rows(rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    """같은 비교의 전체·순방향·역방향 값을 한 행으로 묶는다."""
    groups={}
    for row in rows:
        if row['metrics'][metric] is not None:
            groups.setdefault(row['comparison'],{})[row['group']]=row['metrics'][metric]
    return groups


def generate(source: Path, out: Path, prefix: str) -> Dict[str, Any]:
    """구조화 원천에서 세 표와 교체 문단을 생성한다."""
    d=json.loads(source.read_text())
    out.mkdir(parents=True,exist_ok=True)
    cap=(r'Temporal refining and acceptance conditions. Lag is the number of keyframes observed after the query; n/a denotes inapplicable strength or pretraining. Full learning updates the NSD modules, and DiffAttnConv learning updates the two DiffAttnConv layers.')
    t=start_table(cap,'tab:temporal_window_controls')
    t += [r'\begin{tabular}{@{}lllllll@{}}\toprule',
          r'Variant & Refining offsets & Learning & Strength & Acceptance & Lag & Pretraining \\ \midrule']
    last=None
    for row in d['conditions']:
        if row['group']!=last:
            if last is not None:t.append(r'\addlinespace[2pt]')
            heading={'training_free':'Training-free uniform-subtraction controls', 'learned':'Learned controls with two-sided acceptance', 'zero_lag':'Zero-lag configurations: past-only acceptance'}[row['group']]
            t.append(r'\multicolumn{7}{@{}l}{\emph{'+heading+r'}} \\')
            last=row['group']
        t.append(' & '.join(row[k] if row[k] is not None else ('not stated' if k == 'lag' else 'n/a') for k in ('label','window','scope','strength','acceptance','lag','pretraining'))+r' \\')
    t += [r'\bottomrule\end{tabular}',
          r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize',
          r'Past-only NSD expands the candidate-side correspondence radius from 10 to 20 keyframes; the radius is twice the maximum temporal offset. Learned controls use three starts and 512 fine-tuning updates.',
          r'\end{minipage}',r'\end{table*}']
    (out/'appdx_temporal_window_controls.tex').write_text('\n'.join(t)+'\n')
    cap=(r'Absolute performance for temporal-window and zero-lag controls (\%). Learned values are mean $\pm$ SD over three starts; an em dash denotes an unreported metric. Symmetric NSD direction values are in Table~\ref{tab:direction}; the checkpoints of Table~\ref{tab:results} are the paired reference in Table~\ref{tab:temporal_window_deltas}.')
    t=start_table(cap,'tab:temporal_window_results')+absolute_header(False)
    t.append(r'\multicolumn{7}{@{}l}{\emph{Two-sided acceptance: training-free or symmetric-pretrained models}} \\')
    t += [absolute_row(row) for row in d['window_absolute']]
    t += [r'\addlinespace[2pt]',r'\multicolumn{7}{@{}l}{\emph{Past-only acceptance: past-pretrained or training-free models (zero lag)}} \\']
    t += [absolute_row(row) for row in d['zero_absolute']]
    t += [r'\bottomrule\end{tabular}',r'\end{table*}']
    (out/'appdx_temporal_window_results.tex').write_text('\n'.join(t)+'\n')
    cap=(r'Paired temporal-window and zero-lag differences in percentage points, reported as first named condition minus second. Conditional 95\% intervals use paired sequence resampling; equal-strength rows use $-0.5$, and uniform denotes uniform subtraction. Window, acceptance, and pretraining conditions are listed in Table~\ref{tab:temporal_window_controls}.')
    t=start_table(cap,'tab:temporal_window_deltas')
    t += [r'\label{tab:zero_lag_controls}',r'\begin{tabular}{@{}lccc@{}}\toprule',
          r'Comparison & Overall & Forward-only & Reverse-only \\ \midrule']
    rows=d['window_differences']+d['zero_differences']+d['extra_differences']
    for metric in ('AP','R@100P'):
        if metric!='AP':t.append(r'\midrule')
        t.append(r'\multicolumn{4}{@{}l}{\textbf{'+metric+r'}} \\')
        for comparison,groups in comparison_rows(rows,metric).items():
            t.append(comparison+' & '+' & '.join(stack_cell(groups.get(g)) for g in ('Overall','Forward','Reverse'))+r' \\')
            t.append(r'\addlinespace[1.5pt]')
    t += [r'\bottomrule\end{tabular}',r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize']
    f1=next(r['metrics']['F1'] for r in rows if r['metrics']['F1'] is not None)
    t += ['For preceding five, selected $-$ symmetric uniform, the overall F1 difference is '+cell(f1)+'.',
          r'\end{minipage}',r'\end{table*}']
    (out/'appdx_temporal_window_deltas.tex').write_text('\n'.join(t)+'\n')
    n={item['id']:item['tex'] for item in d['narratives']}
    inp=lambda name:r'\input{'+(prefix.rstrip('/')+'/' if prefix else '')+name+'}'
    chunks=[n['window_setup'],n['window_learning'],inp('appdx_temporal_window_controls'),inp('appdx_temporal_window_results'),inp('appdx_temporal_window_deltas'),n['window_readout'],n['zero_setup'],n['zero_readout']]
    (out/'temporal_controls.tex').write_text('\n\n'.join(chunks)+'\n')
    return d


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    paper = Path(__file__).resolve().parents[1] / 'docs/paper/body_v2.3'
    parser.add_argument('--source',type=Path,default=paper / 'temporal_source.json')
    parser.add_argument('--output-dir',type=Path,default=paper / 'tables_v3')
    parser.add_argument('--input-prefix',default='tables_v3')
    args=parser.parse_args()
    generate(args.source,args.output_dir,args.input_prefix)
