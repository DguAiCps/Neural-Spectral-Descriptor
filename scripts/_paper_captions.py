"""표 재생성 때 유지할 짧은 캡션과 열별 순위 강조 도구."""

import re
from decimal import Decimal
from typing import Dict


TABLE_CAPTIONS = {
    "tab:results": (
        r"\textbf{Place recognition with odometry.} Six-city macro means; "
        r"learned methods report mean $\pm$ SD over three seeds. "
        r"Bold/underlined: best/second best (higher is better)."),
    "tab:efficiency": (
        r"\textbf{Storage and computation on Sejong01.} Gallery: 10,000--10,095 "
        r"frames. Times are means; resident state is per final gallery frame. "
        r"Bold/underlined: best/second best (lower is better)."),
    "tab:online_components": (
        r"\textbf{NSD ablations.} Three-seed, six-city means under the "
        r"Table~\ref{tab:results} protocol."),
    "tab:offline_results": (
        r"\textbf{Offline retrieval.} Seed-1 Recall@1 within 5\,m, averaged "
        r"equally over sequences within each dataset. Dashes denote unavailable results."),
    "tab:v2_offline_pilot": (
        r"Offline Recall@1 within 5\,m: three aggregation views of the "
        r"seed-1 evaluation on 46 test sequences."),
    "tab:uncertainty": (
        r"\textbf{Offline retrieval variability and paired differences.} "
        r"Recall@1 is in percent; differences are percentage points. "
        r"Only seed-1 results are available."),
    "tab:fixed_budget_final": (
        r"\textbf{Offline retrieval at a fixed budget of 200 candidates.} "
        r"Final query-pooled Recall@1 with row-selected (sel.) or shared "
        r"(fix.) fusion. Dashes denote unavailable measurements."),
    "tab:context_final": (
        r"\textbf{Context controls for offline retrieval.} Final query-pooled "
        r"Recall@1 with the fusion rules of Table~\ref{tab:fixed_budget_final}. "
        r"Dashes denote unmeasured controls."),
    "tab:context_systems": (
        r"\textbf{Temporal-context systems in the separate two-pass configuration.} "
        r"Three-seed means on the common queries of Table~\ref{tab:legacy_results}."),
    "tab:causal_pr": (
        r"\textbf{Past-only decision diagnostic with odometry.} Six-city macro "
        r"averages; NSD/NSD-H use three seeds, learned comparisons one. "
        r"Dashes denote unavailable results."),
    "tab:v2_causal_pilot": (
        r"Past-only decision diagnostic with odometry and reference poses. "
        r"Six-city macro averages, seed 1. Comparison methods have no reference-pose runs."),
    "tab:legacy_results": (
        r"\textbf{Separate two-pass configuration: place recognition.} "
        r"Six-city macro means; learned methods report mean $\pm$ SD over "
        r"three seeds. No temporal confidence or whitening."),
    "tab:legacy_components": (
        r"\textbf{Separate two-pass configuration: component ablations.} "
        r"Three-seed final-system means on the common queries of "
        r"Table~\ref{tab:legacy_results}."),
    "tab:baseline_analysis": (
        r"\textbf{Separate two-pass configuration: AP diagnostics.} "
        r"Revisit direction, match radius, and spectral rescoring of the "
        r"Table~\ref{tab:legacy_results} predictions."),
    "tab:paired_bootstrap": (
        r"\textbf{Separate two-pass configuration: paired metric differences.} "
        r"NSD or NSD-H minus each comparison method, with 95\% bootstrap "
        r"intervals conditional on the trained runs."),
    "tab:cost_accounting": (
        r"\textbf{Resource accounting across initial gallery sizes $N$.} "
        r"Memory: decimal MB; timing: mean/p95 in ms. Resident state excludes "
        r"shared input; RSS includes it."),
    "tab:e2e_latency": (
        r"\textbf{Processing time across initial gallery sizes.} "
        r"Mean/p95 in ms under the Table~\ref{tab:efficiency} protocol. "
        r"Total includes encoding, update, and query."),
}


def highlight_column_ranks(tex: str, higher_is_better: bool) -> str:
    """표시 평균값의 열별 1·2위를 강조하고 표준편차와 수치는 보존한다."""
    header, start, rest = tex.partition("\\midrule\n")
    body, end, footer = rest.partition("\\bottomrule")
    if not start or not end:
        raise ValueError("순위를 표시할 표의 데이터 행 경계가 없다")
    rows = []
    for line in body.splitlines():
        if not line.endswith(r" \\"):
            raise ValueError("예상하지 못한 표 데이터 행")
        rows.append(line[:-3].split(" & "))
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("순위를 표시할 열 수가 일정하지 않다")
    for column in range(1, len(rows[0])):
        cells = []
        for index, row in enumerate(rows):
            if row[column] == r"\cmedit{---}":
                continue
            match = re.fullmatch(r"(\\cmedit\{\$?)(\d+(?:\.\d+)?)(.*)", row[column])
            if match is None:
                raise ValueError("순위를 표시할 숫자 셀의 형식이 다르다")
            cells.append((index, match, Decimal(match.group(2))))
        levels = sorted({value for _, _, value in cells}, reverse=higher_is_better)[:2]
        for index, match, value in cells:
            if value not in levels:
                continue
            prefix, number, suffix = match.groups()
            if value == levels[0]:
                command = r"\mathbf" if prefix.endswith("$") else r"\textbf"
            else:
                command = r"\underline"
            rows[index][column] = prefix + command + "{" + number + "}" + suffix
    body = "\n".join(" & ".join(row) + r" \\" for row in rows) + "\n"
    return header + start + body + end + footer


def replace_captions(tex: str, captions: Dict[str, str]) -> str:
    """라벨로 캡션 본문만 바꾸고 수치·표 구조는 보존한다."""
    edits = []
    for match in re.finditer(r"\\caption\{", tex):
        start = match.end()
        depth = 1
        end = start
        while end < len(tex) and depth:
            if tex[end] in "{}":
                back = end - 1
                while back >= 0 and tex[back] == "\\":
                    back -= 1
                if (end - back - 1) % 2 == 0:
                    depth += 1 if tex[end] == "{" else -1
            end += 1
        if depth:
            raise ValueError("닫히지 않은 LaTeX 캡션")
        label = re.search(r"\\label\{([^}]+)\}", tex[end:])
        if label and label.group(1) in captions:
            value = r"\cmedit{" + captions[label.group(1)] + "}"
            edits.append((start, end - 1, value))
    for start, end, value in reversed(edits):
        tex = tex[:start] + value + tex[end:]
    return tex
