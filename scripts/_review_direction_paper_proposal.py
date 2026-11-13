#!/usr/bin/env python3
"""현재 원고를 보존하며 방향·회전·구간의 영문 삽입안을 생성한다."""
from pathlib import Path
import time
import _review_direction_metrics as base

OUT = base.OUT / 'expanded46/paper_proposal'
PAPER = base.ROOT / 'docs/paper/body_v2.3'
GROUPS = [('all', 'All'), ('forward_only', 'Forward-only'), ('reverse_only', 'Reverse-only')]


def write(name: str, text: str) -> None:
    """제안 폴더에만 텍스트를 쓴다."""
    (OUT / name).write_text(text.strip() + '\n', encoding='utf-8')


def table(name: str, caption: str, label: str, columns: str,
          header: list, rows: list) -> None:
    """현재 원고의 booktabs·자동 폭 제한 형식으로 별도 표를 만든다."""
    lines = ['% 결과 JSON에서 생성한 삽입 제안. 원고 원본은 수정하지 않는다.',
             r'\begin{table*}[t]', r'\caption{' + caption + r'}\label{' + label + '}',
             r'\centering\footnotesize\setlength{\tabcolsep}{4pt}', r'\sbox0{%',
             r'\begin{tabular}{@{}' + columns + r'@{}}\toprule',
             ' & '.join(header) + r' \\', r'\midrule']
    lines += [' & '.join(row) + r' \\' for row in rows]
    lines += [r'\bottomrule\end{tabular}}',
              r'\ifdim\wd0>\textwidth\resizebox{\textwidth}{!}{\usebox0}\else\usebox0\fi',
              r'\end{table*}']
    write(name, '\n'.join(lines))


def interval(value: dict) -> str:
    """차이와 조건부 구간을 percentage-point 단위로 쓴다."""
    lo, hi = value['ci95_pp']
    return f"${value['delta_pp']:+.3f}\;[{lo:+.3f},{hi:+.3f}]$"


def score_number(value: float) -> str:
    """작은 점수 오차는 LaTeX 과학적 표기로 쓴다."""
    if value >= .001:
        return f'${value:.3f}$'
    mantissa, exponent = f'{value:.2e}'.split('e')
    return '$' + mantissa + r'\times10^{' + str(int(exponent)) + '}$'


def main() -> None:
    """네 표·영문 조각·한국어 적용안과 읽은 원고의 해시를 보존한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [base.OUT/'direction_summary.json',
             base.OUT/'direction_learning/sequence_summary.json',
             base.OUT/'direction_learning/summary.json',
             base.OUT/'direction_learning/fixed_city_block_summary.json',
             base.OUT/'expanded46/yaw_summary.json',
             base.OUT/'expanded46/causes/summary.json',
             base.OUT/'expanded46/completion_validation.json']
    direction, sequence, block, six, yaw, cause, validation = [base.read(p) for p in paths]
    assert validation['status'] == 'passed'
    manuscript = sorted(p for p in PAPER.rglob('*') if p.is_file() and (
        p.suffix in ('.tex', '.bib', '.sty') or 'figs' in p.relative_to(PAPER).parts))
    paper_hashes = {str(p.relative_to(base.ROOT)):base.digest(p) for p in manuscript}
    initial_hashes = paper_hashes
    if (OUT/'source_manifest.json').exists():
        previous = base.read(OUT/'source_manifest.json')
        initial_hashes = previous.get('initial_manuscript_files', previous['preserved_manuscript_files'])
    protected = [p for p in initial_hashes if '/figs/' in p or '/tables_v3/' in p
                 or Path(p).name in ('cvpr2027_body.tex', 'cvpr2027_supplement_body.tex')]
    assert all(paper_hashes.get(p) == initial_hashes[p] for p in protected), '연구 본문·표·그림 원천이 바뀌었다.'
    concurrent = {p:dict(before=initial_hashes.get(p), after=paper_hashes.get(p))
                  for p in sorted(set(initial_hashes) | set(paper_hashes))
                  if initial_hashes.get(p) != paper_hashes.get(p)}
    rows = []
    for group, label in [('forward_only', 'Forward-only'), ('reverse_only', 'Reverse-only'), ('mixed', 'Mixed')]:
        for method, name in [('diff', 'NSD'), ('bev_native', 'BEVPlace++')]:
            metrics = direction['methods'][method]['mean_std'][group]
            rows.append([label, name, *[f"{100*metrics[k]['mean']:.3f}" for k in base.KEYS]])
    table('table_direction.tex',
          r'Traversal-direction breakdown on the full 195,726-query evaluation (percent). '
          r'Forward-only, reverse-only, and mixed groups contain 33,031, 23,422, and 2,997 positive queries. '
          r'Each group uses common nonrevisit negatives and excludes other positive groups. '
          r'Positive-bearing sequences are averaged within cities, then across cities and three starts; '
          r'the mixed group has five cities, and the other groups have six.',
          'tab:rdyr_direction', 'llrrrr',
          ['Group', 'Method', r'AP (\%)', r'Max F1 (\%)', r'R@100P (\%)', r'Revisit R@1 (\%)'], rows)
    rows = []
    for angle, result in yaw['final_recognition'].items():
        p = result['pooled_query_seed_pairs']
        rows.append([f'${float(angle):g}^\circ$',
                     *[f"{100*result['mean_std'][k]['mean']:.3f}" for k in base.KEYS],
                     f"{100*p['winner_agreement']/p['queries']:.3f}"])
    table('table_yaw.tex',
          r'Raw-point query-frame rotation on 2,944 fixed queries (64 per sequence) and three starts. '
          r'Gallery features and candidate-side temporal context remain unchanged. '
          r'Metrics use the same city macro aggregation; winner agreement covers all 8,832 query--start pairs at each angle. '
          r'This sample is distinct from the full-query results and does not synthesize reverse-view occlusion.',
          'tab:rdyr_yaw', 'rrrrrr',
          ['Yaw', r'AP (\%)', r'Max F1 (\%)', r'R@100P (\%)', r'Revisit R@1 (\%)', r'Winner agreement (\%)'], rows)
    arm_names = [('actual', 'Actual raw rotation'), ('ideal_grid', 'Ideal rasterized features'),
                 ('ideal_transport', 'Covariant transport phase'), ('ideal_both', 'Both replacements'),
                 ('ideal_both_exact_geometry', 'Both + original edge geometry'),
                 ('direct_output_phase', 'Direct output-phase rotation')]
    rows = []
    for arm, label in arm_names:
        values = [cause['results'][a]['arms'][arm] for a in ('6.0', '90.0', '180.0')]
        maximum = max(v['measurements']['fixed_winner_confidence_change']['max'] for v in values)
        rows.append([label, *[str(v['winner_changes']) for v in values], score_number(maximum)])
    table('table_causes.tex',
          r'Mechanism diagnostics on 736 fixed queries (every fourth query in the yaw sample). '
          r'Winner-change counts are out of 2,208 query--start pairs per angle. '
          r'The last column gives the maximum absolute score change for the original $0^\circ$ winner across all three angles. '
          r'Internal replacements isolate mechanisms; they are not evaluated deployment variants.',
          'tab:rdyr_causes', 'lrrrr',
          ['Condition', r'$6^\circ$', r'$90^\circ$', r'$180^\circ$', r'Max $|\Delta S|$'], rows)
    rows = []
    for group, label in GROUPS:
        for name, source in [('Sequence only (10,000)', sequence), ('Sequence + blocks (2,000)', block),
                             ('Blocks, all six cities', six)]:
            result = source['comparisons']['convs_only-fixed'][group]
            rows.append([label, name, *[interval(result['overall'][k]) for k in base.KEYS]])
    table('table_sensitivity.tex',
          r'DiffAttnConv learning minus uniform subtraction: point differences and conditional 95\% intervals '
          r'(percentage points). The original sequence-only procedure is retained. The block procedure '
          r'also changes the sequence sampling population and the handling of zero-positive strata. '
          r'The six-city restriction retains 1,994, 1,557, and 1,868 block replicates for all, forward-only, '
          r'and reverse-only queries, respectively; point estimates remain the original six-city means.',
          'tab:rdyr_sensitivity', 'llcccc',
          ['Group', 'Resampling', 'AP', 'Max F1', 'R@100P', 'Revisit R@1'], rows)
    write('main_text.tex', r'''
% 본문 결과 절의 짧은 추가 문단. 기존 표2·4의 수치와 구간은 유지한다.
\paragraph{Direction and query-yaw diagnostics.}
On reverse-only revisits, NSD exceeds BEVPlace++ in AP, F1, and R@100P, but trails in R@1 by 0.455 percentage points; BEVPlace++ leads on all four forward-only metrics (Table~\ref{tab:rdyr_direction}). Separately, raw-point rotations of 2,944 fixed queries spanning all 46 sequences yield AP changes from $-0.358$ to $+0.346$ points over six nonzero angles (Table~\ref{tab:rdyr_yaw}). These are gallery-fixed coordinate changes, not newly observed reverse views. Rasterization boundaries and non-covariant Euler-yaw transport explain the observed integer-bin deviations (Table~\ref{tab:rdyr_causes}); the DFT shift property does not imply exact end-to-end invariance.

\paragraph{Resampling sensitivity.}
The reported sequence-only intervals remain unchanged. A paired sequence-and-query-block sensitivity analysis gives AP intervals including zero for both all-query and reverse-only DiffAttnConv learning gains (Table~\ref{tab:rdyr_sensitivity}). This procedure also changes the sequence sampling population and zero-positive denominators, so the difference cannot be attributed solely to temporal dependence. The positive original AP interval is conditional on its stated resampling design.
''')
    write('method_scope.tex', r'''
% 선택적 문장 교체·보충안: 정확한 격자 이동에 대한 기존 Proposition은 그대로 둔다.
% 앞 문단: Introduction의 integer-bin rotation 설명을 아래 두 문장으로 교체할 수 있다.
An exact integer circular shift of each rasterized row preserves its DFT magnitude. Raw-point rotation need not produce that exact shift because points near rasterization boundaries can change bins.

% 다음 문단: 공통 yaw frame 설명 뒤의 범위 보충에 사용한다.
The current transport angle is extracted from the pose relative to the first LiDAR orientation. Under a sensor-frame change $R'=R R_z(-\theta)$, the Euler yaw $\psi(R)=\operatorname{atan2}(R_{10},R_{00})$ does not generally satisfy $\psi(R')=\psi(R)-\theta$ when roll and pitch are present. A constant phase residual cancels, whereas residuals that differ across temporal neighbors change their relative phase. We assess this limitation with fixed-input diagnostics rather than claiming general end-to-end yaw invariance.
''')
    write('supplement_text.tex', r'''
% 부록 삽입안. 같은 폴더의 table_*.tex를 함께 삽입한다.
\section{Traversal direction, raw-point yaw, and resampling sensitivity}
\label{app:rdyr_diagnostics}
\paragraph{Direction strata.}
For each positive query, we consider every ground-truth match within 5\,m and eligible history $j\leq q-30$. A heading uses three-dimensional ground-truth displacement over the preceding five keyframes and requires at least 1\,m of motion. Forward-only queries require valid query and match headings with every angle at most $60^\circ$; reverse-only queries require the same validity with every angle at least $120^\circ$. Mixed queries have at least one match in each angular range; other matches may coexist. The remaining 2,232 positives are outside these three groups. Each group retains common negatives whose nearest eligible ground-truth distance exceeds 25\,m, excluding other positives and the ambiguous distance band. Correctness still depends on the returned match being within 5\,m, not its heading. Table~\ref{tab:rdyr_direction} uses all 46 sequences, with five eligible cities for mixed queries and six for the other groups. Full-minus-unmodified differences include the contribution of uniform subtraction; isolated learning is measured against that subtraction control.

\paragraph{Raw-point protocol.}
We fix 64 evenly spaced positions in each sequence's eligible-query index array, rather than uniform seconds or traveled distance. Angles are $0,0.5,1,6,7.3,90,180$ degrees, using three fixed checkpoints. Each query re-encodes the raw scans in $q\pm15$. With column-vector coordinates, $p'=R_z(\theta)p$ and $T'=T\,\operatorname{diag}(R_z(-\theta),1)$ preserve $T'p'=Tp$, using homogeneous coordinates for pose application. All gallery indexes, modified spectra, and candidate-side context remain original, including boundaries where the conservative candidate support $j+20$ overlaps $q-15$. This is neither a whole-trajectory rerun nor a new reverse-view observation. The sample has 779 positives, 1,616 negatives, and 549 ambiguous queries; only KITTI~07 lacks sampled positives. Table~\ref{tab:rdyr_yaw} therefore does not replace full-query accuracy.

\paragraph{Mechanism controls.}
Range-image and BEV azimuth bins span $1^\circ$ and $6^\circ$, respectively. On every fourth selected query, we compare actual raw rotation with ideal rasterized features (the original key and analytically phase-shifted original BEV coefficients), a transport phase fixed to $\psi_i-\theta$, their combination, the combination with original edge geometry, and direct phase rotation of the original network output. Table~\ref{tab:rdyr_causes} separates candidate selection from the score of the original winner. Raw bin/range boundaries change raster values; Euler-yaw residual differences change temporal transport. Both replacements restore all 6,624 original winners, with maximum score error $2.38\times10^{-6}$, below the fixed $8\times10^{-6}$ tolerance. Direct output-phase controls likewise preserve all winners. These counterfactuals explain the measured deviations without establishing the performance of a corrected model or a theorem for arbitrary observations.

\paragraph{Replay and discretization checks.}
At $0^\circ$, all 8,832 winners reproduce the full-query ledgers, with maximum score error $1.91\times10^{-6}$. Independent threshold enumeration reproduces all metrics for 966 ledgers and 61,824 predictions. World-coordinate checks on the first support scan of each sequence at each angle give maximum point error $7.72\times10^{-6}$\,m; this numerical check does not cover every raw point. On 184 fixed scans, exact grid shifts commute with the circular interpolation in all 552 checks; maximum normalized range-key and phase-compensated BEV errors are $1.38\times10^{-7}$ and $1.53\times10^{-7}$. This grid-level check is separate from raw-point recognition.

\paragraph{Intervals under different resampling procedures.}
The original 10,000 sequence-only resamples draw from positive-bearing sequences within each city, retaining equal city weights and the fixed mean of three starts. For sensitivity, 2,000 paired replicates draw from all sequences in each city and resample circular blocks of 100 consecutive queries. All conditions and starts share the same draws. Sequences without positives in a stratum are omitted; a city without remaining positives is omitted for that replicate. We additionally retain only replicates with positives in all six cities, as shown in Table~\ref{tab:rdyr_sensitivity}. These procedures differ in the sequence population and denominator handling as well as query resampling. They neither invalidate the original conditional intervals nor isolate the effect of temporal dependence alone. All-query and reverse-only AP intervals include zero under both block summaries. Their F1 and R@100P intervals remain positive, but the all-query F1 lower endpoint is only $+0.002$ points. All intervals remain conditional on fixed models and cities, without accounting for new training, development selection, or multiple comparisons.
''')
    write('preview.tex', r'''
\documentclass[10pt,twocolumn]{article}
\usepackage[letterpaper,margin=0.75in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\usepackage[hidelinks]{hyperref}
\title{Direction and Yaw Evidence: Insertion Proposal}
\author{}\date{}
\begin{document}
\maketitle
\section*{Optional main-text addition}
\input{main_text}
\section*{Optional method clarification}
\input{method_scope}
\clearpage
\input{supplement_text}
\input{table_direction}
\input{table_yaw}
\input{table_causes}
\input{table_sensitivity}
\clearpage
\end{document}
''')
    write('반영안.txt', '''현재 v2.3에 대한 방향·회전·불확실성 삽입 제안

범위
- 원고·그림 원본은 읽기만 했다. 이 폴더의 tex 조각과 preview.pdf는 반영 전 제안이다.
- 기존 표2·표4, appdx_intervals의 수치와 10,000회 시퀀스 구간을 교체하지 않는다.
- 표 번호를 고정하지 않고 rdyr 접두사의 고유 label을 사용했다. 기존 table4의 Uniform subtraction / DiffAttnConv learning 이름을 따른다.
- 새 표의 성능은 백분율, 구간은 %p다. 본문의 0~1 단위 표2·4와 단위를 섞지 않는다.

권장 위치
1. main_text.tex 첫 문단: cvpr2027_body.tex의 sec:pair_responses 안에서 현재 진단 해석 뒤에 삽입한다. 실제 방향 비교와 gallery 고정 원시 좌표 회전을 구분한다.
2. main_text.tex의 Resampling sensitivity 문단: sec:ablation의 DiffAttnConv 학습 효과 문단 뒤에 둔다. 기존 양의 시퀀스 구간과 점추정은 그대로 유지한다.
3. method_scope.tex 첫 문단: Introduction의 “integer-bin yaw rotation becomes a circular shift” 문장에 대한 선택적 교체안이다. prop:rotation은 이미 정확한 rasterized-row shift를 조건으로 하므로 보존한다.
4. method_scope.tex 둘째 문단: 공통 yaw frame 설명 뒤에 보충한다. 원고의 기존 yaw-only 설명과 충돌하지 않으며 일반 roll/pitch에서의 좌표 공변성 한계를 명확히 한다.
5. supplement_text.tex와 네 table_*.tex: cvpr2027_supplement_body.tex의 Statistical and geographic breakdowns 뒤에 함께 넣는다. 실제 삽입 시 해당 위치의 상대 input 경로만 조정한다. 기존 app:statistics와 구간 표는 유지한다.
6. 기존 abstract/conclusion의 학습 기여 문장을 강화하지 않는다. 시퀀스 재표집 조건부 근거를 유지하면서 본문에 민감도 문단을 연결한다.

해석상 반드시 유지할 내용
- 방향 평가는 전체 195,726질의다. 순방향 33,031·역방향 23,422·혼합 2,997·나머지 양성 2,232개를 구분한다. 방향별 양성+공통 음성만 평가하며 다른 방향 양성과 모호 거리 구간을 제외한다. 혼합 평균은 5도시다.
- 회전은 46시퀀스×64질의=2,944표본이다. 초·거리 균등 표본이나 전체 질의 전수 회전으로 쓰지 않는다. gallery의 후보 시간 문맥까지 고정하며 실제 역방향의 새 가림·시야를 만들지 않는다.
- 원인 대조는 736질의이므로 회전 표의 승자 변화 분모와 다르다. 내부 특징/위상 대체를 새 배포 모델이나 성능 개선 방법으로 쓰지 않는다.
- 기존 시퀀스 구간은 방향 양성이 있는 시퀀스만 추출한다. 블록 민감도는 도시의 전체 시퀀스에서 뽑고 양성 소실에 따라 시퀀스/도시를 제외한다. 6도시 유지 버전은 분모가 남는 반복만 사용한다. 두 절차 차이를 시간 상관 하나로 해석하지 않는다.
- 전체·역방향 AP 블록 구간의 0 포함은 공개한다. 기존 양의 시퀀스 구간을 무효라고 쓰거나, 추가 분석이 독립 확증이라고 쓰지 않는다. 전체 F1 블록 하단 +0.002%p도 함께 알린다.
- 공변 각도 후보식의 수학적 검사는 별도 원천에 있지만 원고 제안에 새 방법으로 넣지 않았다. 해당 수정의 학습·검증 성능과 일반화는 미측정이다.

원천과 재생
- 모든 표는 scripts/_review_direction_paper_proposal.py가 결과 JSON에서 생성한다.
- 방향: direction_summary.json; 구간: direction_learning/{sequence_summary,summary,fixed_city_block_summary}.json.
- 회전·원인: expanded46/yaw_summary.json, expanded46/causes/summary.json, expanded46/completion_validation.json.
- 세부 도시·데이터셋·전체 비교는 direction_city_dataset.csv, direction_learning/learning_intervals.csv, expanded46/yaw_city_dataset.csv에 보존한다.
- source_manifest.json은 입력 결과·생성기·읽은 원고와 그림의 해시를 담는다. validation.json은 원고 보존·표 재생·별도 PDF 빌드·위키 검사 결과를 담는다.
- preview.tex는 검토용 자체 문서다. 실제 CVPR 원고의 지면이나 최종 배치 검증을 대신하지 않는다.
''')
    base.write(OUT/'source_manifest.json', dict(created_unix=time.time(), status='proposal',
        sources={str(p.relative_to(base.ROOT)):base.digest(p) for p in [*paths, Path(__file__)]},
        initial_manuscript_files=initial_hashes, preserved_manuscript_files=paper_hashes,
        protected_research_files=protected, concurrent_format_and_disclosure_changes=concurrent,
        concurrent_change_note='부모가 사용자 요청의 공식 CVPR 2027 양식·LLM 사용 고지 변경임을 확인했다. 이 에이전트는 원고 파일에 쓰지 않았고 연구 본문·표·그림은 최초 스냅샷과 같다.',
        proposal_sha256={p.name:base.digest(p) for p in OUT.iterdir() if p.suffix == '.tex' or p.name == '반영안.txt'},
        scope='CPU 문서 제안만 생성. docs/paper/body_v2.3의 원고·표·그림은 수정하지 않음.'))
    print('PROPOSAL', OUT)


if __name__ == '__main__':
    main()
