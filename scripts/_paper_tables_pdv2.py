#!/usr/bin/env python3
"""분할 v2 예비 결과 JSON에서 원고 표와 출처 목록을 만든다.

표에는 분할 v2 결과(pdv2_bevclip, pdv2_c3_bevclip_s*, pdv2_c3의 비교 방법)만
사용한다. 이전 분할 비교 블록은 읽지
않으며, 표를 원고에 포함할지는 원고 진입 파일에서 따로 정한다.
예비 결과 표 두 개와 함께, tables_rev/templates의 빈 표 틀 여섯 개에
같은 결과로 채울 수 있는 칸만 채운 사본을 만든다. 틀 원본은 고치지 않는다.
호스트의 파이썬 표준 라이브러리만으로 실행할 수 있다.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "docs/paper/body_v2.2/tables_v2"
TEMPLATE_DIR = REPO / "docs/paper/body_v2.2/tables_rev/templates"
# NSD's BEV auxiliary grid is clipped at the sensor plane (scripts/_bevclip_spec.py).
OFFLINE_PATH = "results/pdv2_bevclip/summary.json"
RING_SELECTION_PATH = "results/pdv2_c3/ringsharp_pdv2_ckpt_selection.json"
ONLINE_SEEDS = (1, 2, 3)
# NSD/NSD-H: selection refit per seed, decision statistic selected over ALIGN_STATS
# on validation (scripts/_pdv2_bevclip.py); the original ω is kept. The
# reference-pose diagnostic has seed 1 only.
NSD_PATHS = {
    "odom": tuple("results/pdv2_c3_bevclip_s{}/odom/pr_summary_align.json".format(s)
                  for s in ONLINE_SEEDS),
    "gt": ("results/pdv2_c3_bevclip_s1/gt/pr_summary_align.json",),
}
BASELINE_PATHS = {condition: "results/pdv2_c3/{}/pr_summary_baselines.json".format(condition)
                  for condition in ("odom", "gt")}
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
% 출처: results/pdv2/summary.json; source_manifest.json에 해시를 기록한다.
\begin{table*}[t]
\color{blue}
\caption{\cmedit{Preliminary offline retrieval on split v2: seed 1, 46 test sequences, Recall@1 within 5\,m. The test split was evaluated before preregistration, and the grouping rule remains undecided; all three aggregate views are reported.}}
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
\cmedit{Each aggregate averages sequences within a group and weights groups equally. The six cities are Karlsruhe, Seoul, Daejeon, Ann Arbor (NCLT), Stockholm, and Hamburg. The six dataset groups are KITTI with KITTI-360, HeLiPR Town, HeLiPR Daejeon, MulRan Daejeon, NCLT, and MCD (both cities). Retrieval uses completed trajectories, reference poses, and a union of two top-100 candidate lists. These are single-seed pilot results, with no uncertainty estimate or complete baseline comparison.}
\end{minipage}
\end{table*}
"""


def causal_table(sources: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]],
                 ring_selection: Dict[str, Any]) -> str:
    """주행 추정·기준 자세 조건의 인과 평가 표를 만든다.

    RING#-L은 집계가 끝난 조건만 채우고 나머지 조건은 빈칸으로 둔다.
    """
    methods = (
        ("SC++", "scpp", 1), ("NSD", "pbev128_main", 0),
        ("NSD-H", "hspec12", 0), (r"RING\#-L", "ringsharp", 1),
        ("SeqOT (trailing window)", "seqot", 1),
    )
    ring_conditions = [c for c in ("odom", "gt") if "ringsharp" in sources[c][1]]
    if not ring_conditions:
        raise ValueError("RING#-L 집계 결과가 어느 조건에도 없다")
    for condition, (runs, baselines) in sources.items():
        seed_runs(runs, condition)
        validate_roster(baselines, condition)
        if baselines["_meta"].get("seqot") != "retrained on v2 train, seed 1, last weights":
            raise ValueError("SeqOT 가중치의 분할·선택 규칙이 달라졌다")
    rows = []
    for name, key, source_index in methods:
        cells = []
        for condition in ("odom", "gt"):
            if key == "ringsharp" and condition not in ring_conditions:
                cells.extend(["---"] * len(CAUSAL_METRICS))
                continue
            # NSD/NSD-H: seed 1 in both conditions (the GT diagnostic has only seed 1).
            source = sources[condition][source_index]
            data = (source[0] if source_index == 0 else source)[key]
            if len(data["per_sequence"]) != 46:
                raise ValueError("인과 시퀀스별 결과가 완전하지 않다")
            cells.extend("{:.3f}".format(checked_number(data["city_macro6"][m]))
                         for m in CAUSAL_METRICS)
        rows.append(name + " & " + " & ".join(cells) + r" \\")
    ring_note = (r" RING\#-L is retrained on split v2 with its official trainer for five"
                 r" epochs; epoch {} is selected by split-v2 validation Recall@1 pooled"
                 r" within sensor families.").format(int(ring_selection["selected_epoch"][1:]))
    if "gt" not in ring_conditions:
        ring_note += " Its reference-pose run is pending."
    return r"""% scripts/_paper_tables_pdv2.py가 생성한다. 숫자를 직접 고치지 않는다.
% 출처: results/pdv2_c3_bevclip_s1/{odom,gt}/pr_summary_align.json, results/pdv2_c3/{odom,gt}/pr_summary_baselines.json.
\begin{table*}[t]
\color{blue}
\caption{\cmedit{Preliminary causal decision evaluation on split v2: 46 test sequences, one seed for learned methods (for NSD and NSD-H, seed 1 of the three-seed study in Table~\ref{tab:results}), and six-city macro averages. SC++ has no learned checkpoint. The pilot was not preregistered and includes only a partial baseline comparison.}}
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
\cmedit{The causal refiner and edge selector are trained with reference poses, and refiner checkpoint selection also uses reference-pose validation caches. At inference, the odometry condition uses KISS-ICP poses for keyframe selection and temporal-edge attributes; ground truth supplies evaluation labels. Fusion tuples, confidence statistics, and decision thresholds are selected separately for each condition on the nine validation sequences, so this is not a pose-only substitution with a shared scorer. SeqOT uses final split-v2 checkpoints and trailing-window inference.""" + ring_note + r""" Grouping is exploratory. Each condition uses its own common keyframe roster across methods. This past-only decision protocol differs from the offline retrieval pilot; its AP and F1 values are not offline Recall@1.}
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
        r"\caption{\cmedit{\textbf{Offline place retrieval: split-v2 evaluation template.} Dataset columns reserve the comparison layout; values, seed summaries, and the grouping and weighting of the final mean are pending.",
        r"\caption{\cmedit{\textbf{Offline place retrieval on split v2.} Recall@1 within 5\,m. The NSD and NSD-H rows are the preliminary seed-1 pilot of Table~\ref{tab:v2_offline_pilot}; each dataset column averages its test sequences with equal weight. The other rows, seed summaries, and the grouping and weighting of the final mean are pending.",
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


def online_results_table(
    sources: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]],
) -> str:
    """메인 tab:results에 완료된 주행 추정 조건의 과거 전용 결과만 채운다.

    NSD·NSD-H는 학습 시드 세 개의 평균 ± 표준편차, 비교 방법은 단일 실행 값이다.
    """
    runs, baselines = sources["odom"]
    seed_runs(runs, "odom")
    if len(runs) != len(ONLINE_SEEDS):
        raise ValueError("온라인 NSD 결과의 시드 수가 다르다")
    validate_roster(baselines, "odom")
    metrics = CAUSAL_METRICS + ("R@1_revisit",)
    edits = [(
        r"\caption{\cmedit{\textbf{Online place recognition: split-v2 evaluation template.} AP, maximum F1, recall at 100\% precision, and revisit Recall@1 reserve the past-only decision comparison. Results and seed summaries are pending.}}",
        r"\caption{\cmedit{\textbf{Online place recognition on split v2.} The completed past-only KISS-ICP replay covers 46 test sequences with six-city macro averages. NSD and NSD-H entries give the mean and standard deviation over three training seeds; RING\#-L uses one seed. Filled entries form a partial baseline comparison; dashes are pending results. This preliminary study was not preregistered.}}",
        None,
    )]
    for label, key in (("SC++", "scpp"), (r"RING\#-L", "ringsharp")):
        if key not in baselines:
            continue
        if len(baselines[key]["per_sequence"]) != 46:
            raise ValueError("온라인 시퀀스별 결과가 완전하지 않다")
        values = [cell(baselines[key]["city_macro6"][metric]) for metric in metrics]
        edits.append((table_row(label, [DASH] * 4), table_row(label, values), None))
    for name, key in OPERATING_POINTS:
        if any(len(data[key]["per_sequence"]) != 46 for data in runs):
            raise ValueError("온라인 시퀀스별 결과가 완전하지 않다")
        label = r"\textbf{" + name + "}"
        values = [cell_pm(seed_values(runs, key, metric)) for metric in metrics]
        edits.append((table_row(label, [DASH] * 4), table_row(label, values), None))
    # 평균±표준편차 칸이 한 단 폭을 넘으므로 표를 단 폭에 맞춘다.
    edits.append((r"\begin{tabular}{lcccc}",
                  "\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lcccc}", None))
    edits.append((
        r"\end{tabular}",
        r"""\end{tabular}}
\par\smallskip
\begin{minipage}{\linewidth}\footnotesize
\cmedit{KISS-ICP supplies keyframe selection and temporal-edge attributes; ground truth supplies scoring labels. NSD refiner and edge-selector training and refiner checkpoint selection use reference poses. Fusion, confidence, and thresholds are selected for each seed on the nine odometry-validation sequences. SC++ is training-free; RING\#-L uses a split-v2-trained, validation-selected checkpoint. Candidates satisfy $j\leq q-30$. Max F1 and R@100P summarize test precision--recall curves. Revisit R@1 is the fraction of revisit queries (a past frame within 5\,m) whose top-1 past match lies within 5\,m; it ignores the acceptance score.}
\end{minipage}""",
        None,
    ))
    table = fill_template("table1_online.tex", edits)
    return table.replace(
        FILLED_HEADER,
        FILLED_HEADER + "\n% 출처: results/pdv2_c3_bevclip_s{1,2,3}/odom/pr_summary_align.json, "
        "results/pdv2_c3/odom/pr_summary_baselines.json; source_manifest.json에 해시를 기록한다.",
        1,
    )


def causal_pr_table(sources: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]) -> str:
    """tab:causal_pr에 주행 추정 조건의 결과가 있는 행만 채운다.

    NSD·NSD-H는 학습 시드 세 개의 평균이다(표준편차는 tab:results).
    """
    runs, baselines = sources["odom"]
    seed_runs(runs, "odom")
    edits = [
        (
            r"\caption{\cmedit{\textbf{Causal loop-closure decisions: split-v2 evaluation template.} The causal roster, context construction, confidence and threshold selection, seed summaries, and decision metrics are pending. This empty table makes no claim of completed pre-registration or prefix-invariance verification.}}",
            r"\caption{\cmedit{\textbf{Causal loop-closure decisions on split v2.} Filled rows are the KISS-ICP odometry condition: 46 test sequences and six-city macro averages. NSD and NSD-H entries average three training seeds (Table~\ref{tab:results} gives standard deviations); the other learned rows use one seed. Refiner and edge-selector training and refiner checkpoint selection use reference poses; fusion, confidence, and $\tau^{*}$ are selected for each seed on odometry-keyframe validation caches. The other rows, controls, and F1 at $\tau^{*}$ are pending. The study was not preregistered. Re-running the NSD pipeline on truncated prefixes of three validation sequences reproduces the full-sequence embeddings up to float32 rounding, confirming prefix invariance.}}",
            None,
        ),
        (
            r"\multicolumn{6}{l}{\cmedit{Operating-point results pending validation selection:}} \\",
            r"\multicolumn{6}{l}{\cmedit{Operating point at the validation-selected threshold:}} \\",
            None,
        ),
    ]
    rows = (
        ("SC++", "scpp"),
        (r"\textbf{NSD} (causal-retrained)", "pbev128_main"),
        (r"\textbf{NSD-H} (causal-retrained)", "hspec12"),
        (r"RING\#-L", "ringsharp"),
        ("SeqOT (trailing window)", "seqot"),
    )
    for label, key in rows:
        if key in ("pbev128_main", "hspec12"):
            if any(len(data[key]["per_sequence"]) != 46 for data in runs):
                raise ValueError("인과 시퀀스별 결과가 완전하지 않다")
            values = [cell(statistics.mean(seed_values(runs, key, m))) for m in CAUSAL_PR_METRICS]
        else:
            if key not in baselines:
                continue
            if len(baselines[key]["per_sequence"]) != 46:
                raise ValueError("인과 시퀀스별 결과가 완전하지 않다")
            values = [cell(baselines[key]["city_macro6"][m]) for m in CAUSAL_PR_METRICS]
        edits.append((table_row(label, [DASH] * 5), table_row(label, values), None))
    for name, key in OPERATING_POINTS:
        prefix = name + r" (causal-retrained) @$\tau^{*}$ & \multicolumn{5}{l}{"
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
        r"Seeds, measurements, and the resampling and multiplicity protocol are pending.}}",
        r"NSD and NSD-H entries are the preliminary seed-1 pilot (Table~\ref{tab:v2_offline_pilot}): seed SD is unavailable for a single trained run, and the paired difference has no interval yet. Other seeds and measurements and the resampling and multiplicity protocol are pending.}}",
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
    edits = [(r"Split-v2 results are pending.}}", caption_new, None)]
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
    causal_sources = {}
    for condition, paths in NSD_PATHS.items():
        runs = []
        for path in paths:
            data, record = read_source(path)
            runs.append(data)
            source_records.append(record)
        baselines, record = read_source(BASELINE_PATHS[condition])
        source_records.append(record)
        causal_sources[condition] = (runs, baselines)
    odom_paths = list(NSD_PATHS["odom"]) + [BASELINE_PATHS["odom"]]
    causal_paths = [p for c in NSD_PATHS for p in list(NSD_PATHS[c]) + [BASELINE_PATHS[c]]]
    ring_selection, ring_record = read_source(RING_SELECTION_PATH)
    source_records.append(ring_record)
    if not ring_selection["rule"].startswith("highest v2-validation"):
        raise ValueError("RING#-L 체크포인트가 분할 v2 검증으로 선택되지 않았다")

    # 모든 원천 검사를 마친 뒤 산출물을 쓴다. 예비 표 두 개가 원천 검사를 먼저 한다.
    tables = {
        "pilot_offline.tex": offline_table(offline),
        "pilot_causal.tex": causal_table(causal_sources, ring_selection),
        "table1_online.tex": online_results_table(causal_sources),
        "table1_strict.tex": results_table(offline),
        "appdx_causal_pr.tex": causal_pr_table(causal_sources),
        "table_uncertainty.tex": uncertainty_table(offline),
        "table_fixed_budget_final.tex": final_pipeline_table(
            offline, "table_fixed_budget_final.tex", "Even union & 100 & 100",
            [r"\cmedit{---/---}"],
            r"The even-union entries are the preliminary seed-1 pilot (Table~\ref{tab:v2_offline_pilot}); in this row, sel.\ and fix.\ use the same tuple by definition. Its unique-pool size was not recorded, and the other split-v2 results are pending.}}"),
        "table_context_final.tex": final_pipeline_table(
            offline, "table_context_final.tex", r"Full context, $\mathbf f$ & 416 & Yes", [],
            r"The full-context entries are the preliminary seed-1 pilot and coincide with the even-union row of Table~\ref{tab:fixed_budget_final}. The other split-v2 results are pending.}}"),
    }
    ring_scope = [c for c in ("odom", "gt") if "ringsharp" in causal_sources[c][1]]
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
        "status": "예비 실행. 집계 방식은 미확정이며 시험 결과에 이미 노출됐다.",
        "nsd_bev_auxiliary": "BEV 최대 높이 격자를 센서 높이(max(0, z))에서 자른다 "
                             "(scripts/_bevclip_spec.py, 2026-10-10)",
        "main_table": "table1_online.tex",
        "main_evaluation_basis": "pdv2_c3_bevclip_s{1,2,3}/odom 과거 전용 KISS-ICP 판정 결과",
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
                "seqot_checkpoint": "분할 v2 재학습 시드 1의 마지막 가중치",
                "ringsharp_checkpoint": "분할 v2 재학습, 검증으로 고른 " + ring_selection["selected_epoch"],
                "ringsharp_conditions": ring_scope,
                "baseline_scope": "SC++, SeqOT, RING#-L(집계된 조건만)을 포함한 부분 비교",
            },
            "table1_online.tex": {
                "label": "tab:results",
                "role": "main_online_results",
                "source_paths": odom_paths + [RING_SELECTION_PATH],
                "metric": list(CAUSAL_METRICS) + ["R@1_revisit"],
                "metric_note": "R@1_revisit는 2026-10-09 시험 결과를 본 뒤 추가한 진단 지표",
                "json_field": "city_macro6",
                "seeds": list(ONLINE_SEEDS),
                "nsd_cells": "시드 평균과 표본 표준편차",
                "preregistration": False,
                "condition": "odom",
                "protocol": "KISS-ICP 키프레임의 과거 전용 온라인 판정 재생 평가",
                "candidate_rule": "j <= q - 30",
                "training_poses": "GT",
                "refiner_checkpoint_selection_poses": "GT",
                "selection_poses": "KISS-ICP odometry",
                "selection_sequences": 9,
                "template_dir": str(TEMPLATE_DIR.relative_to(REPO)),
                "filled": "NSD·NSD-H·SC++·RING#-L의 완료된 주행 추정 조건 값",
                "baseline_scope": "부분 비교. SeqOT는 부록의 trailing-window 결과로 유지",
                "delayed_query_frames": 0,
            },
            "table1_strict.tex": dict(filled, label="tab:offline_results",
                                      role="supplementary_offline_results",
                                      metric="5 m Recall@1, 데이터셋 안 시퀀스 평균",
                                      filled="NSD·NSD-H 데이터셋 열. Mean은 집계 미확정이라 비움"),
            "appdx_causal_pr.tex": dict(
                {k: v for k, v in filled.items() if k != "seed"}, label="tab:causal_pr",
                source_paths=odom_paths, json_field="city_macro6", seeds=list(ONLINE_SEEDS),
                filled="주행 추정 조건의 SC++, NSD, NSD-H(시드 3개 평균), RING#-L, SeqOT와 "
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
