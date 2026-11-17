#!/usr/bin/env python3
"""분할 v2 예비 결과 JSON에서 원고 표와 출처 목록을 만든다.

대칭 문맥 주 결과와 공통 질의 비교, 별도 과거 전용·오프라인 진단의
분할 v2 결과만 사용한다. 이전 분할 비교 블록은 읽지 않는다.
진단 표 세 개와 tables_rev/templates의 빈 표 틀 일곱 개를 채운 사본을
만든다. 원고 진입 파일이 표 배치를 정하며 틀 원본은 고치지 않는다.
호스트의 파이썬 표준 라이브러리만으로 실행할 수 있다.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from _paper_current_nsd import apply_current_main


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "docs/paper/body_v2.2/tables_v2"
TEMPLATE_DIR = REPO / "docs/paper/body_v2.2/tables_rev/templates"
# NSD's BEV auxiliary grid is clipped at the sensor plane (scripts/_bevclip_spec.py).
OFFLINE_PATH = "results/pdv2_bevclip/summary.json"
MAIN_PATH = "results/pdv2_symmetric3_20261013/summary.json"
MAIN_BASELINE_PATH = "results/pdv2_main_lag15_20261014/baselines_common.json"
# Official-code comparison methods (seed 1) on the same common queries (scripts/_pdv2_main_baselines_official.py)
OFFICIAL_BASELINE_PATH = "results/pdv2_main_official_baselines/baselines_common.json"
SEEDS_PATH = "results/pdv2_baseline_seeds/summary.json"            # scripts/_pdv2_baseline_seeds.py
APPENDIX_PATH = "results/pdv2_appendix_baselines/summary.json"      # scripts/_pdv2_appendix_baselines.py
BOOTSTRAP_PATH = "results/pdv2_paired_bootstrap/summary.json"       # scripts/_pdv2_paired_bootstrap.py
OFFICIAL_ROWS = (("SC++", "scpp"), ("RING", "ring"), ("RING++", "ringpp"),
                 (r"RING\#-L", "ringsharp"), (r"DeepRING~\citep{lu2022deepring}", "deepring"),
                 (r"SeqOT~\citep{ma2022seqot}", "seqot"), (r"DiSCO~\citep{xu2021disco}", "disco"),
                 ("HeLiOS", "helios"), ("ImLPR", "imlpr"), ("BEVPlace++", "bevplace"))
TABLE4_PATH = "results/pdv2_table4_20261014/summary.json"
PAST_COMMON_PATH = "results/pdv2_paper2_common_queries_20261014/summary.json"
RING_SELECTION_PATH = "results/pdv2_ring_official/ringsharp_ckpt_selection.json"
# 과거 전용 표의 비교 방법: 주표와 같은 공식 시드 1 실행(scripts/_pdv2_causal_baselines_official.py)
CAUSAL_OFFICIAL_PATH = "results/pdv2_causal_official_baselines/pr_summary_baselines.json"
ONLINE_SEEDS = (1, 2, 3)
# NSD/NSD-H: selection refit per seed, decision statistic selected over ALIGN_STATS
# on validation (scripts/_pdv2_bevclip.py); the original ω is kept. The
# reference-pose diagnostic has seed 1 only.
NSD_PATHS = {
    "odom": tuple("results/pdv2_c3_bevclip_s{}/odom/pr_summary_align.json".format(s)
                  for s in ONLINE_SEEDS),
    "gt": ("results/pdv2_c3_bevclip_s1/gt/pr_summary_align.json",),
}
OFFLINE_METRICS = (
    "city_macro6", "city_macro5_without_nclt", "dataset_macro6",
)
CAUSAL_METRICS = ("AP", "max_F1", "R@100P")
CAUSAL_PR_METRICS = ("AP", "max_F1", "R@100P", "R@99P", "R@95P")
OPERATING_POINTS = (("NSD", "pbev128_main"), ("NSD-H", "hspec12"))
# 오프라인 표의 데이터셋 열과 시퀀스 이름 접두사. "kitti/"는 KITTI-360을 포함하지 않는다.
DATASET_COLUMNS = (
    ("KITTI", "kitti/"), ("KITTI-360", "kitti360/"), ("HeLiPR", "helipr/"),
    ("MulRan", "mulran/"), ("NCLT", "nclt/"), ("MCD", "mcd/"),
)
DASH = r"\cmedit{---}"
FILLED_HEADER = ("% scripts/_paper_tables_pdv2.py가 tables_rev 틀에 분할 v2 예비 결과를 "
                 "채운다. 숫자를 직접 고치지 않는다.")


def read_source(relative_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """결과 JSON과 원본 바이트의 출처·해시를 반환한다."""
    raw = (REPO / relative_path).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return data, {
        "path": relative_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def checked_number(value: Any) -> float:
    """표의 지표가 0 이상 1 이하인 유한한 수인지 확인한다."""
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("지표 범위를 벗어났다: {!r}".format(value))
    return number


def validate_roster(data: Dict[str, Any], condition: str = "", seed: int = 1) -> None:
    """예비 실행의 시드·조건·시험 시퀀스 수를 확인한다."""
    meta = data["_meta"]
    if meta["n_test_sequences"] != 46:
        raise ValueError("시험 시퀀스가 46개인 결과만 사용할 수 있다")
    if "seed" in meta and meta["seed"] != seed:
        raise ValueError("시드 {} 결과가 아니다".format(seed))
    if condition and meta["condition"] != condition:
        raise ValueError("요청한 자세 조건과 결과 파일이 다르다")


def seed_runs(runs: Sequence[Dict[str, Any]], condition: str) -> None:
    """NSD 결과가 시드 1, 2, …의 순서로 모두 있는지 확인한다."""
    for seed, data in enumerate(runs, start=1):
        validate_roster(data, condition, seed)
        if data["_meta"].get("seed") != seed:
            raise ValueError("인과 NSD 결과의 시드 정보가 없다")


def seed_values(runs: Sequence[Dict[str, Any]], key: str, metric: str) -> List[float]:
    """시드별 도시 6묶음 평균 지표를 모은다."""
    return [checked_number(data[key]["city_macro6"][metric]) for data in runs]


def offline_table(data: Dict[str, Any]) -> str:
    """오프라인 예비 결과의 세 집계 방식 표를 만든다."""
    validate_roster(data)
    if data["_meta"].get("seed") != 1:
        raise ValueError("오프라인 결과의 시드 정보가 없다")
    if data["_meta"].get("config") != "configs/training_place_disjoint_v2.yaml":
        raise ValueError("분할 v2 설정으로 만든 결과가 아니다")
    rows = []
    for name, key in OPERATING_POINTS:
        values = [checked_number(data[key][metric]) for metric in OFFLINE_METRICS]
        if len(data[key]["per_sequence"]) != 46:
            raise ValueError("오프라인 시퀀스별 결과가 완전하지 않다")
        rows.append(name + " & " + " & ".join("{:.3f}".format(v) for v in values)
                    + r" \\")
    return r"""% scripts/_paper_tables_pdv2.py가 생성한다. 숫자를 직접 고치지 않는다.
% 출처: results/pdv2_bevclip/summary.json; source_manifest.json에 해시를 기록한다.
\begin{table*}[t]
\color{blue}
\caption{\cmedit{Offline retrieval over complete trajectories: seed 1, 46 test sequences, Recall@1 within 5\,m. The study was not preregistered. Three aggregate views are reported because the grouping rule remains unspecified.}}
\label{tab:v2_offline_pilot}
\centering\small
\begin{tabular}{lccc}
\toprule
Method & Six-city macro & Five-city macro without NCLT & Six-dataset-group macro \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.97\textwidth}\footnotesize
\cmedit{Each aggregate averages sequences within a group and weights groups equally. The six cities are Karlsruhe, Seoul, Daejeon, Ann Arbor (NCLT), Stockholm, and Hamburg. The six dataset groups are KITTI with KITTI-360, HeLiPR Town, HeLiPR Daejeon, MulRan Daejeon, NCLT, and MCD (both cities). Retrieval uses complete trajectories, reference poses, and a union of two top-100 candidate lists. Results use one training seed, with no uncertainty estimate or complete baseline comparison.}
\end{minipage}
\end{table*}
"""


def causal_table(sources: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]],
                 ring_selection: Dict[str, Any]) -> str:
    """주행 추정·기준 자세 조건의 인과 평가 표를 만든다.

    비교 방법은 주표와 같은 공식 시드 1 실행이다. 기준 자세로는 다시 돌리지 않아 그 열은 빈칸이다.
    """
    methods = (
        ("SC++", "scpp", 1), ("NSD", "pbev128_main", 0),
        ("NSD-H", "hspec12", 0), (r"RING\#-L", "ringsharp", 1),
        ("SeqOT (trailing window)", "seqot", 1),
    )
    for condition, (runs, _) in sources.items():
        seed_runs(runs, condition)
    official = sources["odom"][1]
    rows = []
    for name, key, source_index in methods:
        cells = []
        for condition in ("odom", "gt"):
            if source_index == 1 and condition == "gt":
                cells.extend(["---"] * len(CAUSAL_METRICS))
                continue
            # NSD/NSD-H: seed 1 in both conditions (the GT diagnostic has only seed 1).
            data = (sources[condition][0][0][key] if source_index == 0       # 시드 1 표: 비교 방법도 시드 1
                    else official[key]["per_seed"]["-" if official[key]["seeds"] == ["-"] else "1"])
            if len(data["per_sequence"]) != 46:
                raise ValueError("인과 시퀀스별 결과가 완전하지 않다")
            cells.extend("{:.3f}".format(checked_number(data["city_macro6"][m]))
                         for m in CAUSAL_METRICS)
        rows.append(name + " & " + " & ".join(cells) + r" \\")
    ring_note = (r" RING\#-L is trained with its official trainer for {} epochs; epoch {} is selected by"
                 r" validation Recall@1 pooled within sensor families.").format(
        int(ring_selection["official_final_epoch"][1:]), int(ring_selection["selected_epoch"][1:]))
    return r"""% scripts/_paper_tables_pdv2.py가 생성한다. 숫자를 직접 고치지 않는다.
% 출처: results/pdv2_c3_bevclip_s1/{odom,gt}/pr_summary_align.json, """ + CAUSAL_OFFICIAL_PATH + r""".
\begin{table*}[t]
\color{blue}
\caption{\cmedit{Past-only place recognition with odometry and reference poses: 46 complete test sequences, seed 1 for every learned method (Table~\ref{tab:causal_pr} gives three-seed means under odometry), and six-city macro averages. These past-only NSD models differ from the symmetric models in Table~\ref{tab:results}. Comparison methods are the official seed-1 runs of Table~\ref{tab:results}, not rerun with reference poses (dashes); Table~\ref{tab:causal_pr} lists all of them under odometry. SC++ requires no training. The study was not preregistered.}}
\label{tab:v2_causal_pilot}
\centering\small
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{KISS-ICP odometry} & \multicolumn{3}{c}{Reference poses (GT)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Method & AP & Max F1 & R@100P & AP & Max F1 & R@100P \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.97\textwidth}\footnotesize
\cmedit{The causal refiner and edge selector are trained with reference poses, which also supply refiner checkpoint selection. At inference, the odometry condition uses KISS-ICP poses for keyframe selection and temporal-edge attributes; ground truth supplies evaluation labels. Fusion tuples, confidence statistics, and decision thresholds are selected separately for each condition on the nine validation sequences. The comparison therefore includes condition-specific scoring. SeqOT runs the released code with trailing windows of raw scans, and its checkpoints are selected on validation with the same windows.""" + ring_note + r""" Grouping is exploratory. Within each condition, all methods use the same keyframes. AP and F1 measure loop-closure decisions, whereas offline Recall@1 measures retrieval accuracy.}
\end{minipage}
\end{table*}
"""


def cell(value: float) -> str:
    """0–1 지표를 소수 셋째 자리 파란 칸으로 쓴다."""
    return r"\cmedit{" + "{:.3f}".format(checked_number(value)) + "}"


def cell_pm(values: Sequence[float]) -> str:
    """시드 평균과 표본 표준편차를 '평균 아래첨자 ±표준편차'의 파란 칸으로 쓴다."""
    return (r"\cmedit{$" + "{:.3f}".format(statistics.mean(values)) + r"_{\pm"
            + "{:.3f}".format(statistics.stdev(values)) + "}$}")


def table_row(label: str, cells: Sequence[str]) -> str:
    """표의 한 행을 만든다."""
    return label + " & " + " & ".join(cells) + r" \\"


def replace_once(text: str, old: str, new: str, after: Optional[str] = None) -> str:
    """old를 한 번만 바꾼다. after가 있으면 그 표지 뒤에서 처음 나오는 것을 바꾼다."""
    if after is None:
        if text.count(old) != 1:
            raise ValueError("틀에서 바꿀 문자열이 하나가 아니다: " + old)
        start = 0
    else:
        if text.count(after) != 1:
            raise ValueError("틀의 구획 표지가 하나가 아니다: " + after)
        start = text.index(after)
    index = text.find(old, start)
    if index < 0:
        raise ValueError("틀에 바꿀 문자열이 없다: " + old)
    return text[:index] + new + text[index + len(old):]


def fill_template(name: str,
                  edits: Sequence[Tuple[str, str, Optional[str]]]) -> str:
    """빈 표 틀을 읽어 첫 줄 주석을 바꾸고 지정한 문자열만 바꾼다."""
    text = (TEMPLATE_DIR / (name + ".in")).read_text(encoding="utf-8")
    text = FILLED_HEADER + "\n" + text.split("\n", 1)[1]
    for old, new, after in edits:
        text = replace_once(text, old, new, after)
    return text


def query_pooled(data: Dict[str, Any]) -> float:
    """질의를 모두 합친 Recall@1을 시퀀스별 결과에서 다시 계산해 확인한다."""
    per, counts = data["per_sequence"], data["n_queries"]
    pooled = sum(per[s] * counts[s] for s in per) / sum(counts.values())
    if abs(pooled - data["query_weighted"]) > 1e-12:
        raise ValueError("질의 합산 Recall@1이 요약 값과 다르다")
    return checked_number(pooled)


def results_table(offline: Dict[str, Any]) -> str:
    """부록 tab:offline_results에 NSD·NSD-H의 데이터셋별 시퀀스 평균만 채운다.

    대표 평균(Mean)의 묶음·가중은 정하지 않았으므로 비워 둔다.
    """
    edits = [(
        r"\caption{\cmedit{\textbf{Offline place retrieval.} Dataset-level results and seed variability are unmeasured, and the grouping and weighting for the overall mean remain unspecified.",
        r"\caption{\cmedit{\textbf{Offline place retrieval.} Recall@1 within 5\,m. NSD and NSD-H use seed 1 from Table~\ref{tab:v2_offline_pilot}; each dataset column averages its test sequences with equal weight. Other methods and seed variability are unmeasured, and the grouping and weighting for the overall mean remain unspecified.",
        None,
    )]
    for name, key in OPERATING_POINTS:
        per = offline[key]["per_sequence"]
        cells, used = [], 0
        for _, prefix in DATASET_COLUMNS:
            values = [checked_number(v) for s, v in per.items() if s.startswith(prefix)]
            if not values:
                raise ValueError("데이터셋 열에 해당하는 시퀀스가 없다: " + prefix)
            used += len(values)
            cells.append(cell(statistics.mean(values)))
        if used != len(per):
            raise ValueError("데이터셋 열에 속하지 않는 시퀀스가 있다")
        label = r"\textbf{" + name + "}"
        edits.append((table_row(label, [DASH] * 7), table_row(label, cells + [DASH]), None))
    return fill_template("table1_strict.tex", edits)


def online_results_table(main: Dict[str, Any], baselines: Dict[str, Any],
                         official: Dict[str, Any], seeds: Dict[str, Any]) -> str:
    """공통 질의의 대칭 문맥 주 결과를 채운다. 지연 설명은 방법 절에 둔다.

    비교 방법은 공식 코드로 다시 돌린 시드 1 기록(official)만 쓴다. 아직 끝나지 않은
    방법은 대시로 둔다. baselines(이전 SC++·RING#-L 기록)는 검사만 하고 표에 쓰지 않는다."""
    ops = main["stages"]["test_selected"]["ops"]
    if not baselines["checks_passed"] or not official["checks_passed"]:
        raise ValueError("비교 방법 공통 질의 검사가 통과하지 않았다")
    metrics = CAUSAL_METRICS + ("R@1",)
    edits = [(
        r"\caption{\cmedit{\textbf{Place recognition with odometry.} AP, maximum F1, recall at 100\% precision, and revisit Recall@1 characterize the final decisions. Descriptor construction is specified in Sec.~\ref{sec:graph}. Dashes denote unmeasured results and seed variability.}}",
        r"\caption{\cmedit{\textbf{Place recognition with odometry.} KISS-ICP replay on a common query set from 46 test sequences, with six-city macro averages. NSD, NSD-H, and the learned comparison methods report mean and standard deviation over three training seeds (DeepRING: seeds 1, 3, and 4); SC++, RING, and RING++ are training-free." + (
            "" if all(k in official["methods"] for _, k in OFFICIAL_ROWS) else " Dashes denote comparisons still running.")
        + "}}",
        None,
    )]
    for label, key in OFFICIAL_ROWS:
        if key not in official["methods"]:
            continue
        data = official["methods"][key]
        if len(data["per_sequence"]) != 46 or data["n_queries"] != 195726:
            raise ValueError("주표의 공통 질의가 다르다")
        ms = seeds["methods"][key]
        if not seeds["checks_passed"] or ms["per_seed"][ms["seeds"][0]]["macro"] != data["macro"]:
            raise ValueError("시드 집계의 첫 실행이 주표 재집계와 다르다")
        if ms["training_free"]:
            values = [r"\cmedit{" + f"{checked_number(ms['mean_std'][m]['mean']):.4f}" + "}" for m in metrics]
        else:
            values = [r"\cmedit{$" + f"{checked_number(ms['mean_std'][m]['mean']):.4f}" + r"_{\pm"
                      + f"{ms['mean_std'][m]['std']:.4f}" + "}$}" for m in metrics]
        edits.append((table_row(label, [DASH] * 4), table_row(label, values), None))
    for name, key in OPERATING_POINTS:
        if sorted(ops[key]["per_seed"]) != ["1", "2", "3"]:
            raise ValueError("주표의 시드가 다르다")
        for run in ops[key]["per_seed"].values():
            arm = run["arms"]["symmetric"]
            assert len(arm["per_sequence"]) == 46
            assert sum(v["n_queries"] for v in arm["per_sequence"].values()) == 195726
            assert arm["excluded_no_positive"] == ["kitti/07"]
        label = r"\textbf{" + name + "}"
        values = [r"\cmedit{$" + f"{checked_number(ops[key]['mean_std']['symmetric'][m]['mean']):.4f}"
                  + r"_{\pm" + f"{ops[key]['mean_std']['symmetric'][m]['std']:.4f}" + "}$}"
                  for m in metrics]
        edits.append((table_row(label, [DASH] * 4), table_row(label, values), None))
    # 평균±표준편차 칸이 한 단 폭을 넘으므로 표를 단 폭에 맞춘다.
    edits.append((r"\begin{tabular}{lcccc}",
                  "\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lcccc}", None))
    edits.append((
        r"\end{tabular}",
        r"""\end{tabular}}
\par\smallskip
\begin{minipage}{\linewidth}\footnotesize
\cmedit{Descriptor construction follows Sec.~\ref{sec:graph}; all rows use the same 195,726 queries and candidates $j\leq q-30$. Comparison methods run the authors' public code with its default configuration (DeepRING: reimplemented from the paper), retrained on our training split; SeqOT uses the released centred windows of raw scans. KITTI~07 has no positive query in this set and is excluded from all rows' AP/recall means; its negative queries are retained separately. The means cover 45 sequences grouped into six cities. Max F1 and R@100P summarize test threshold sweeps. Revisit R@1 measures the final top-1 match on positive queries, ignoring the acceptance score.}
\end{minipage}""",
        None,
    ))
    table = fill_template("table1_online.tex", edits)
    return table.replace(
        FILLED_HEADER,
        FILLED_HEADER + "\n% 출처: " + MAIN_PATH + ", " + OFFICIAL_BASELINE_PATH
        + "; source_manifest.json에 해시를 기록한다.",
        1,
    )


def _cell(value: float) -> str:
    return r"\cmedit{" + f"{checked_number(value):.3f}" + "}"


def _cell_ms(stats: Dict[str, float]) -> str:
    return (r"\cmedit{$" + f"{checked_number(stats['mean']):.3f}" + r"_{\pm" + f"{stats['std']:.3f}" + "}$}")


def baseline_analysis_table(app: Dict[str, Any]) -> str:
    """부록: 주표 비교의 방향별·10 m·NSD 판정 통계 이식 AP(_pdv2_appendix_baselines.py)."""
    if not app["checks_passed"] or app["cspec_recompute_max_abs_err"] >= 1e-5:
        raise ValueError("부록 분석 검사가 통과하지 않았다")
    variants = ("all", "same", "opposite", "r10")
    rows = []
    for label in ("NSD", "NSD-H"):
        m = app["methods"][label]
        rows.append(table_row(r"\textbf{" + label + "}", [_cell_ms(m[v]["AP"]) for v in variants] + [DASH]))
    for label, key in OFFICIAL_ROWS:
        m = app["methods"].get(key)
        cells = ([(_cell_ms(m[v]["AP"]) if isinstance(m[v]["AP"], dict) else _cell(m[v]["AP"]))
                  for v in variants + ("cspec",)] if m else [DASH] * 5)
        rows.append(table_row(label, cells))
    return "\n".join([
        FILLED_HEADER + "\n% 출처: " + APPENDIX_PATH + "; source_manifest.json에 해시를 기록한다.",
        r"\begin{table*}[!htbp]",
        r"\caption{\cmedit{\textbf{AP breakdown of the main comparison.} Same 195,726 queries, candidates $j\leq q-30$, and six-city macro averages as Table~\ref{tab:results}. \emph{Same}/\emph{Opposite}: only positive queries whose correct past place exists in the same travel direction, or only in the opposite direction, are kept (other positives excluded, negatives unchanged). \emph{10\,m}: the returned match counts as correct within 10\,m, with positives still defined within 5\,m. \emph{NSD $c_{\mathrm{spec}}$}: each method's own top-1 match is scored by NSD's training-free spectral statistic, applied unchanged without selection; Revisit Recall@1 is therefore unchanged. Learned methods: mean and standard deviation over three training seeds (DeepRING: seeds 1, 3, and 4); SC++, RING, and RING++ are training-free." + (
            "" if all(k in app["methods"] for _, k in OFFICIAL_ROWS) else " Dashes denote comparisons still running.")
        + "}}",
        r"\label{tab:baseline_analysis}",
        r"\centering\footnotesize\color{blue}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & All & Same direction & Opposite direction & 10\,m & NSD $c_{\mathrm{spec}}$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])


def paired_bootstrap_table(boot: Dict[str, Any]) -> str:
    """부록: NSD·NSD-H − 비교 방법의 짝지은 부트스트랩 구간(_pdv2_paired_bootstrap.py)."""
    if not boot["checks_passed"]:
        raise ValueError("부트스트랩 검사가 통과하지 않았다")
    comp = boot["comparisons"]
    present = [(label, key) for label, key in OFFICIAL_ROWS if key in comp["NSD"]["AP"]]
    rows_of = lambda key: [comp[op][m][key] for op in comp for m in comp[op]]
    ahead = [key for _, key in present if all(r["p_holm"] <= 0.05 for r in rows_of(key))]
    p_max = max(r["p_holm"] for key in ahead for r in rows_of(key))
    metric_names = {"AP": "AP", "max_F1": "maximum F1", "R@100P": "R@100P", "R@1": "Revisit Recall@1"}
    short = lambda label: label.split("~")[0]
    others = []
    for label, key in present:
        if key in ahead:
            continue
        status = {}
        for m, name in metric_names.items():
            lo = min(comp[op][m][key]["ci95"][0] for op in comp)
            hi = max(comp[op][m][key]["ci95"][1] for op in comp)
            status.setdefault("lie below zero" if hi < 0 else "lie above zero" if lo > 0 else "include zero",
                              []).append(name)
        others.append("For " + short(label) + ", the intervals " + " and ".join(
            k + " for " + (v[0] if len(v) == 1 else ", ".join(v[:-1]) + (", and " if len(v) > 2 else " and ") + v[-1])
            for k, v in status.items()) + ".")
    if others:
        verdict = (r" Against every comparison method except " + ", ".join(
            short(l) for l, k in present if k not in ahead) + r", each one-sided Holm-adjusted $p$ for NSD and NSD-H being higher in AP, maximum F1, R@100P, and Revisit Recall@1 is at most "
            + f"{p_max:.3f}" + ". " + " ".join(others))
    else:
        verdict = (r" For AP, maximum F1, R@100P, and Revisit Recall@1, every one-sided Holm-adjusted $p$ is at most "
                   + f"{p_max:.3f}" + ".")
    dashes = "" if len(present) == len(OFFICIAL_ROWS) else " Dashes denote comparisons still running."

    def diff(op: str, metric: str, key: str) -> str:
        r = comp[op][metric].get(key)
        if r is None:
            return DASH
        return (r"\cmedit{$" + f"{r['diff']:+.3f}" + r"$ [$" + f"{r['ci95'][0]:+.3f}" + r"$, $"
                + f"{r['ci95'][1]:+.3f}" + r"$]}")

    rows = [table_row(label, [diff("NSD", "AP", key), diff("NSD-H", "AP", key), diff("NSD", "R@1", key)])
            for label, key in OFFICIAL_ROWS]
    design = boot["design"]
    return "\n".join([
        FILLED_HEADER + "\n% 출처: " + BOOTSTRAP_PATH + "; source_manifest.json에 해시를 기록한다.",
        r"\begin{table*}[!htbp]",
        r"\caption{\cmedit{\textbf{Paired bootstrap of differences to comparison methods.} Entries are differences of six-city macro metrics on the common queries of Table~\ref{tab:results} (NSD or NSD-H minus the comparison method) with 95\% percentile intervals. Each of "
        + f"{design['replicates']:,}" + r" draws resamples the sequences within each city and then " + str(design["block_queries"])
        + r"-query time blocks within each sequence, shared by all methods; decision scores and query labels are fixed, and every learned method averages its three seeds on each draw."
        + verdict + r" Intervals condition on the evaluated training runs and do not resample seed variation." + dashes + "}}",
        r"\label{tab:paired_bootstrap}",
        r"\centering\footnotesize\color{blue}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & $\Delta$AP (NSD) & $\Delta$AP (NSD-H) & $\Delta$Revisit R@1 (NSD) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])


def main_context_table(main: Dict[str, Any], controls: Optional[Dict[str, Any]] = None) -> str:
    """주표의 대칭 문맥과 동일 설정 문맥 제거 결과를 기여표에 채운다."""
    ops = main["stages"]["test_selected"]["ops"]
    path = TEMPLATE_DIR / "table_online_components.tex.in"
    text = path.read_text()
    start = text.index(r"\caption{")
    end = text.index("\n", start)
    text = text[:start] + r"\caption{\cmedit{\textbf{Component comparisons.} Three-seed means on the common queries of Table~\ref{tab:results}. The no-context row removes only the context block and reuses each full model's validation-selected fusion and confidence rules. Candidates are regenerated. The full-reference rows repeat the same system; dashes mark unmeasured controls.}}" + text[end:]
    for label, arm in (("Invariant key (zero context; 416D)", "no_context"),
                       ("Full symmetric context (416D; reference)", "symmetric"),
                       ("Even union ($100+100$; reference)", "symmetric"),
                       ("On (reference)", "symmetric")):
        values = [r"\cmedit{" + f"{ops[key]['mean_std'][arm][metric]['mean']:.4f}" + "}"
                  for _, key in OPERATING_POINTS for metric in CAUSAL_METRICS]
        text = replace_once(text, table_row(label, [DASH] * 6), table_row(label, values))
    if controls is not None:
        labels = (
            ("Graph-free learned projection (416D)", "graph_free"),
            ("Key index only ($200+0$)", "key_only"),
            ("Auxiliary index only ($0+200$)", "aux_only"),
            ("Off (no alignment ranking or confidence)", "no_alignment"),
        )
        for label, variant in labels:
            values = []
            for _, key in OPERATING_POINTS:
                result = controls["ops"][key][variant]
                if sorted(result["per_seed"]) != ["1", "2", "3"]:
                    raise ValueError("표 4 대조군의 세 시드가 완전하지 않다")
                for run in result["per_seed"].values():
                    per = run["per_sequence"]
                    if len(per) != 46 or sum(r["n_queries"] for r in per.values()) != 195726:
                        raise ValueError("표 4 대조군의 시험 질의가 다르다")
                for metric in CAUSAL_METRICS:
                    mean = statistics.mean(r["macro"][metric] for r in result["per_seed"].values())
                    if abs(mean - result["mean_std"][metric]["mean"]) > 1e-12:
                        raise ValueError("표 4 대조군의 시드 평균이 다르다")
                    values.append(r"\cmedit{" + f"{checked_number(mean):.4f}" + "}")
            text = replace_once(text, table_row(label, [DASH] * 6), table_row(label, values))
        start = text.index(r"\caption{")
        end = text.index("\n", start)
        caption = r"\caption{\cmedit{\textbf{Component comparisons.} Three-seed means of final matching and decision metrics on the common queries of Table~\ref{tab:results}. Controls retain the full model's validation-selected settings, except alignment-off confidence, selected on validation using only non-alignment scores. The no-context row zeros the context block; the graph-free projection is retrained. Candidates are regenerated for key and allocation controls and fixed for alignment controls. Full-reference rows repeat the same system.}}"
        text = text[:start] + caption + text[end:]
    source = MAIN_PATH + (", " + TABLE4_PATH if controls is not None else "")
    return FILLED_HEADER + "\n% 출처: " + source + "\n" + text[text.index(r"\begin{table*}"):]


def context_systems_table(common: Dict[str, Any]) -> str:
    """동일 질의의 과거·대칭 최종 시스템 비교를 별도 부록 표로 만든다."""
    rows = []
    for name, op in OPERATING_POINTS:
        for label, field in (("past-only", "common"), ("symmetric", "symmetric")):
            values = [r"\cmedit{" + f"{common['ops'][op]['rows'][field][m]:.4f}" + "}"
                      for m in (*CAUSAL_METRICS, "R@1")]
            rows.append(table_row(name + " (" + label + ")", values))
    return FILLED_HEADER + r"""
\begin{table}[t]
\caption{\cmedit{\textbf{Temporal-context systems on common queries.} Three-seed means, with the query set and aggregation of Table~\ref{tab:results}. Each system retains its own validation-selected scoring settings.}}
\label{tab:context_systems}
\centering\scriptsize\color{blue}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
System & AP & Max F1 & R@100P & R@1 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def causal_pr_table(sources: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]) -> str:
    """tab:causal_pr에 주행 추정 조건의 결과가 있는 행만 채운다.

    NSD·NSD-H는 별도 과거 전용 학습 시드 세 개의 평균이다.
    """
    runs, baselines = sources["odom"]
    seed_runs(runs, "odom")
    edits = [
        (
            r"\caption{\cmedit{\textbf{Past-only loop-closure decisions.} Dashes denote unmeasured decision metrics and seed variability. The evaluation protocol, validation selection, and prefix-invariance checks remain to be established.}}",
            r"\caption{\cmedit{\textbf{Past-only loop-closure decisions: separate diagnostic.} KISS-ICP odometry on the complete query set of 46 test sequences, with six-city macro averages. Learned methods report mean and standard deviation over three training seeds (DeepRING: seeds 1, 3, and 4); the trailing-window SeqOT run has one seed, and SC++, RING, and RING++ are training-free. NSD and NSD-H are separately trained past-only models with their own validation selections. Comparison methods are the official seed-1 runs of Table~\ref{tab:results}: single-scan methods use only the query keyframe, and SeqOT runs the released code with trailing windows of raw scans instead of centred ones. Dashes denote unavailable results, including F1 at $\tau^{*}$. A truncation check on three validation sequences reproduced past-only NSD embeddings within float32 rounding.}}",
            None,
        ),
        (
            r"\multicolumn{6}{l}{\cmedit{Operating point at a validation-selected threshold (unmeasured):}} \\",
            r"\multicolumn{6}{l}{\cmedit{Operating point at the validation-selected threshold:}} \\",
            None,
        ),
    ]
    rows = (
        (r"\textbf{NSD} \cmedit{(causal refiner)}", "pbev128_main"),
        (r"\textbf{NSD-H} \cmedit{(causal refiner)}", "hspec12"),
        *((label, key) for label, key in OFFICIAL_ROWS if key != "seqot"),
        ("SeqOT (trailing window)", "seqot"),
    )
    for label, key in rows:
        if key in ("pbev128_main", "hspec12"):
            if any(len(data[key]["per_sequence"]) != 46 for data in runs):
                raise ValueError("인과 시퀀스별 결과가 완전하지 않다")
            values = [cell_pm(seed_values(runs, key, m)) for m in CAUSAL_PR_METRICS]
        else:
            if key not in baselines:
                continue
            b = baselines[key]
            if any(len(r["per_sequence"]) != 46 for r in b["per_seed"].values()):
                raise ValueError("인과 시퀀스별 결과가 완전하지 않다")
            if b["city_macro6_std"] is None:
                values = [cell(b["city_macro6"][m]) for m in CAUSAL_PR_METRICS]
            else:
                values = [r"\cmedit{$" + "{:.3f}".format(checked_number(b["city_macro6"][m])) + r"_{\pm"
                          + "{:.3f}".format(b["city_macro6_std"][m]) + "}$}" for m in CAUSAL_PR_METRICS]
        edits.append((table_row(label, [DASH] * 5), table_row(label, values), None))
    for name, key in OPERATING_POINTS:
        prefix = name + r" \cmedit{(causal refiner)} @$\tau^{*}$ & \multicolumn{5}{l}{"
        edits.append((
            prefix + r"P = \cmedit{---}, R = \cmedit{---}, F1 = \cmedit{---}} \\",
            prefix + "P = " + cell(statistics.mean(seed_values(runs, key, "P_at_tau")))
            + ", R = " + cell(statistics.mean(seed_values(runs, key, "R_at_tau")))
            + r", F1 = \cmedit{---}} \\",
            None,
        ))
    return fill_template("appdx_causal_pr.tex", edits)


def uncertainty_table(offline: Dict[str, Any]) -> str:
    """tab:uncertainty에 단일 시드 NSD·NSD-H 값과 짝지은 차이만 채운다."""
    nsd, nsdh = offline["pbev128_main"], offline["hspec12"]
    if nsd["n_queries"] != nsdh["n_queries"]:
        raise ValueError("NSD와 NSD-H의 질의 목록이 다르다")
    n_queries = sum(nsd["n_queries"].values())
    edits = [(
        r"Measurements are unavailable, and the resampling and multiplicity protocol is unspecified.}}",
        r"NSD and NSD-H use seed 1 from Table~\ref{tab:v2_offline_pilot}. One trained run does not provide seed SD, and no interval is available for the paired difference. Additional measurements are unavailable, and the resampling and multiplicity protocol is unspecified.}}",
        None,
    )]
    for name, key in OPERATING_POINTS:
        label = r"\textbf{" + name + "}"
        mean = r"\cmedit{" + "{:.1f}".format(100 * query_pooled(offline[key])) + r" $\pm$ n/a}"
        edits.append((table_row(label, [DASH] * 2),
                      table_row(label, [r"\cmedit{1}", mean]), None))
    delta = 100 * (query_pooled(nsdh) - query_pooled(nsd))
    edits.append((
        table_row(r"NSD-H $-$ NSD", [DASH] * 4),
        table_row(r"NSD-H $-$ NSD", [
            r"\cmedit{" + "{:,}".format(n_queries).replace(",", "{,}") + "}",
            r"\cmedit{" + "{:+.2f}".format(delta) + "}", DASH, DASH]),
        None,
    ))
    return fill_template("table_uncertainty.tex", edits)


def final_pipeline_table(offline: Dict[str, Any], name: str, row_label: str,
                         empty_cells: Sequence[str], caption_new: str) -> str:
    """전체 문맥·100+100 합집합 행만 채운다. 이 행에서 sel.과 fix.는 같은 조합이다."""
    edits = [(r"Dashes denote unmeasured results.}}", caption_new, None)]
    headers = (
        (r"\textbf{NSD: BEV auxiliary magnitude index}", "pbev128_main"),
        (r"\textbf{NSD-H: height auxiliary magnitude index}", "hspec12"),
    )
    for header, key in headers:
        value = cell(query_pooled(offline[key]))
        old = table_row(row_label, list(empty_cells) + [DASH, DASH])
        new = table_row(row_label, list(empty_cells) + [value, value])
        edits.append((old, new, header))
    return fill_template(name, edits)


def generate(output_dir: Path) -> List[Path]:
    """표와 원천 파일 목록을 지정한 폴더에 쓴다."""
    offline, offline_record = read_source(OFFLINE_PATH)
    source_records = [offline_record]
    main, main_record = read_source(MAIN_PATH)
    main_baselines, baseline_record = read_source(MAIN_BASELINE_PATH)
    official_baselines, official_record = read_source(OFFICIAL_BASELINE_PATH)
    seeds, seeds_record = read_source(SEEDS_PATH)
    source_records.append(seeds_record)
    appendix, appendix_record = read_source(APPENDIX_PATH)
    bootstrap, bootstrap_record = read_source(BOOTSTRAP_PATH)
    source_records.extend((appendix_record, bootstrap_record))
    common, common_record = read_source(PAST_COMMON_PATH)
    source_records.extend((main_record, baseline_record, official_record, common_record))
    controls = None
    if (REPO / TABLE4_PATH).exists():
        controls, control_record = read_source(TABLE4_PATH)
        if (controls.get("final_system_metrics") is not True
                or controls.get("base_summary_sha256") != main_record["sha256"]):
            raise ValueError("표 4 대조군이 현재 주표의 최종 판정 결과가 아니다")
        source_records.append(control_record)
    causal_sources = {}
    for condition, paths in NSD_PATHS.items():
        runs = []
        for path in paths:
            data, record = read_source(path)
            runs.append(data)
            source_records.append(record)
        causal_sources[condition] = (runs, None)
    causal_official, record = read_source(CAUSAL_OFFICIAL_PATH)
    source_records.append(record)
    if not causal_official["_meta"]["checks_passed"]:
        raise ValueError("과거 전용 비교 방법 재집계 검사가 통과하지 않았다")
    causal_sources["odom"] = (causal_sources["odom"][0], causal_official)
    odom_paths = list(NSD_PATHS["odom"]) + [CAUSAL_OFFICIAL_PATH]
    causal_paths = [p for c in NSD_PATHS for p in NSD_PATHS[c]] + [CAUSAL_OFFICIAL_PATH]
    ring_selection, ring_record = read_source(RING_SELECTION_PATH)
    source_records.append(ring_record)
    if not ring_selection["rule"].startswith("highest split-v2 validation"):
        raise ValueError("RING#-L 체크포인트가 분할 v2 검증으로 선택되지 않았다")

    # 모든 원천 검사를 마친 뒤 산출물을 쓴다. 예비 표 두 개가 원천 검사를 먼저 한다.
    tables = {
        "pilot_offline.tex": offline_table(offline),
        "pilot_causal.tex": causal_table(causal_sources, ring_selection),
        "table1_online.tex": online_results_table(main, main_baselines, official_baselines, seeds),
        "table_online_components.tex": main_context_table(main, controls),
        "table_context_systems.tex": context_systems_table(common),
        "table1_strict.tex": results_table(offline),
        "appdx_causal_pr.tex": causal_pr_table(causal_sources),
        "appdx_baseline_analysis.tex": baseline_analysis_table(appendix),
        "appdx_paired_bootstrap.tex": paired_bootstrap_table(bootstrap),
        "table_uncertainty.tex": uncertainty_table(offline),
        "table_fixed_budget_final.tex": final_pipeline_table(
            offline, "table_fixed_budget_final.tex", "Even union & 100 & 100",
            [r"\cmedit{---/---}"],
            r"The even-union entries use seed 1 from Table~\ref{tab:v2_offline_pilot}; sel.\ and fix.\ use the same tuple in this row by definition. Its unique-pool size was not recorded; dashes denote unavailable measurements.}}"),
        "table_context_final.tex": final_pipeline_table(
            offline, "table_context_final.tex", r"Full context, $\mathbf f$ & 416 & Yes", [],
            r"The full-context entries use seed 1 and coincide with the even-union row of Table~\ref{tab:fixed_budget_final}. Dashes denote unmeasured comparisons.}}"),
    }
    ring_scope = ["odom"]
    filled = {
        "source_paths": [OFFLINE_PATH],
        "seed": 1,
        "preregistration": False,
        "template_dir": str(TEMPLATE_DIR.relative_to(REPO)),
    }
    manifest = {
        "schema": "pdv2-paper-tables-v1",
        "generator": "scripts/_paper_tables_pdv2.py",
        "split": "training_place_disjoint_v2",
        "seeds": {"online_odom": list(ONLINE_SEEDS), "online_gt": [1], "offline": [1]},
        "n_test_sequences": 46,
        "preregistration": False,
        "status": "주평가는 공통 질의·6도시·3시드 집계다. 오프라인 대표 집계는 미확정이며 시험 장소의 개발 노출이 있다.",
        "nsd_bev_auxiliary": "BEV 최대 높이 격자를 센서 높이(max(0, z))에서 자른다 "
                             "(scripts/_bevclip_spec.py, 2026-10-10)",
        "main_table": "table1_online.tex",
        "main_evaluation_basis": "대칭 문맥·15키프레임 지연, KISS-ICP, 공통 질의의 최종 판정 결과",
        "main_decision_statistic": {
            op: [main["stages"]["test_selected"]["ops"][op]["per_seed"][str(seed)]["selection"]["decision"]
                 for seed in ONLINE_SEEDS] for _, op in OPERATING_POINTS},
        "nsd_decision_statistic": {
            condition: {op: [run[op]["statistic"] for run in causal_sources[condition][0]]
                        for op in ("pbev128_main", "hspec12")}
            for condition in NSD_PATHS},
        "source_paths": [record["path"] for record in source_records],
        "sources": source_records,
        "tables": {
            "pilot_offline.tex": {
                "source_paths": [OFFLINE_PATH],
                "metric": "5 m Recall@1",
                "json_fields": list(OFFLINE_METRICS),
                "seed": 1,
                "preregistration": False,
                "protocol": "기준 자세와 완성 궤적을 쓰는 오프라인 양방향 검색",
            },
            "pilot_causal.tex": {
                "source_paths": causal_paths,
                "metric": list(CAUSAL_METRICS),
                "json_field": "city_macro6",
                "seed": 1,
                "nsd_rows": "두 조건 모두 시드 1",
                "preregistration": False,
                "conditions": ["odom", "gt"],
                "protocol": "과거 전용 판정. 각 조건에서 검증 9개로 선택을 따로 한다.",
                "training_poses": "GT",
                "refiner_checkpoint_selection_poses": "GT",
                "seqot_checkpoint": "공식 코드 과거 창 실행, 같은 창으로 검증 선택(시드 1)",
                "ringsharp_checkpoint": "분할 v2 재학습, 검증으로 고른 " + ring_selection["selected_epoch"],
                "ringsharp_conditions": ring_scope,
                "baseline_scope": "주표와 같은 공식 시드 1 실행의 SC++, RING#-L, SeqOT(과거 창); 기준 자세 조건은 다시 돌리지 않아 빈칸",
            },
            "table1_online.tex": {
                "label": "tab:results",
                "role": "main_online_results",
                "source_paths": [MAIN_PATH, OFFICIAL_BASELINE_PATH, SEEDS_PATH],
                "metric": list(CAUSAL_METRICS) + ["R@1"],
                "metric_note": "R@1_revisit는 2026-10-09 시험 결과를 본 뒤 추가한 진단 지표",
                "json_field": "stages.test_selected.ops.*.mean_std.symmetric / methods.*.macro",
                "seeds": list(ONLINE_SEEDS),
                "nsd_cells": "시드 평균과 표본 표준편차",
                "preregistration": False,
                "condition": "odom",
                "protocol": "KISS-ICP 대칭 문맥 판정 재생, 30 <= q < N-15의 공통 질의",
                "candidate_rule": "j <= q - 30",
                "training_poses": "GT",
                "refiner_checkpoint_selection_poses": "GT",
                "selection_poses": "KISS-ICP odometry",
                "selection_sequences": 9,
                "template_dir": str(TEMPLATE_DIR.relative_to(REPO)),
                "filled": ("NSD·NSD-H(시드 3개)와 공식 코드 비교 방법의 시드 평균·표본 표준편차(학습 방법 시드 3개, "
                           "DeepRING 1·3·4; 학습 없는 SC++·RING·RING++는 실행 하나): "
                           + ", ".join(sorted(seeds["methods"]))),
                "baseline_scope": "공식 코드 비교 방법 10개, " + SEEDS_PATH,
                "delayed_query_frames": 15,
                "n_queries": 195726,
                "n_positive": 61682,
                "empty_positive_excluded_from_means": ["kitti/07"],
                "latency_disclosure": "숫자·정보 접근 범위는 Methodology에 적고 표 2에는 중복하지 않는다.",
            },
            "table_online_components.tex": {
                "source_paths": [MAIN_PATH] + ([TABLE4_PATH] if controls is not None else []),
                "label": "tab:online_components",
                "seeds": list(ONLINE_SEEDS), "metric": list(CAUSAL_METRICS),
                "filled": ("모든 행의 최종 매칭·판정 성능. full-reference는 같은 시스템의 반복." if controls is not None
                           else "동일 설정 문맥 제거와 대칭 전체 모델. 나머지 full-reference는 같은 시스템의 반복."),
                "selection": "전체 모델의 검증 선택값 유지. 추가 정렬 제거 대조군의 판정 점수는 비정렬 채널만으로 검증 선택한다.",
            },
            "table_context_systems.tex": {
                "source_paths": [PAST_COMMON_PATH], "label": "tab:context_systems",
                "seeds": list(ONLINE_SEEDS), "metric": list(CAUSAL_METRICS) + ["R@1"],
                "protocol": "공통 질의, 각 과거/대칭 시스템의 검증 선택값 유지",
            },
            "table1_strict.tex": dict(filled, label="tab:offline_results",
                                      role="supplementary_offline_results",
                                      metric="5 m Recall@1, 데이터셋 안 시퀀스 평균",
                                      filled="NSD·NSD-H 데이터셋 열. Mean은 집계 미확정이라 비움"),
            "appdx_causal_pr.tex": dict(
                {k: v for k, v in filled.items() if k != "seed"}, label="tab:causal_pr",
                source_paths=odom_paths, json_field="city_macro6", seeds=list(ONLINE_SEEDS),
                filled="주행 추정 조건의 NSD, NSD-H(시드 3개 평균)와 공식 시드 1 비교 방법 10개(SeqOT은 과거 창), "
                       "τ*의 P·R(시드 평균). τ*의 F1은 결과 파일에 없어 비움"),
            "table_uncertainty.tex": dict(filled, label="tab:uncertainty",
                                          metric="질의 합산 5 m Recall@1 (%)",
                                          filled="NSD·NSD-H의 S=1과 값, NSD-H − NSD의 n_q와 Δ"),
            "table_fixed_budget_final.tex": dict(filled, label="tab:fixed_budget_final",
                                                 metric="질의 합산 5 m Recall@1",
                                                 filled="100+100 합집합 행의 sel.·fix."),
            "table_context_final.tex": dict(filled, label="tab:context_final",
                                            metric="질의 합산 5 m Recall@1",
                                            filled="전체 문맥 행의 sel.·fix."),
        },
        "excluded_json_blocks": ["reference_pd2026_s1_same_sequences"],
        "output_sha256": {
            name: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for name, value in tables.items()
        },
    }
    apply_current_main(tables, manifest)
    from _paper_power_nsd import apply_power_main
    apply_power_main(tables, manifest)
    from _paper_efficiency import apply_efficiency
    apply_efficiency(tables, manifest)
    from _paper_captions import TABLE_CAPTIONS, highlight_column_ranks, replace_captions
    tables = {name: replace_captions(value, TABLE_CAPTIONS)
              for name, value in tables.items()}
    for name, higher_is_better in (("table1_online.tex", True), ("table3_efficiency.tex", False)):
        tables[name] = highlight_column_ranks(tables[name], higher_is_better)
    manifest["caption_policy"] = "scripts/_paper_captions.py"
    manifest["rank_highlight_policy"] = {
        "table1_online.tex": "열별 표시 평균값 내림차순; 1위 굵게, 2위 밑줄; SD는 그대로",
        "table3_efficiency.tex": "열별 표시 평균값 오름차순; 1위 굵게, 2위 밑줄",
        "ties": "같은 값은 같은 순위로 표시하며 미측정 값은 제외",
    }
    manifest["output_sha256"] = {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for name, value in tables.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, value in tables.items():
        path = output_dir / name
        path.write_text(value, encoding="utf-8")
        outputs.append(path)
    manifest_path = output_dir / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    outputs.append(manifest_path)
    return outputs


def main() -> None:
    """명령줄 인자를 읽고 표를 생성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="표와 출처 목록을 저장할 폴더")
    args = parser.parse_args()
    for path in generate(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
