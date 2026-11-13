"""시간 전용·상관 보정 NSD의 원천 검증과 원고 표 전환을 담당한다."""

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict


REPO = Path(__file__).resolve().parents[1]
ROOT = "results/pdv2_spectral_whitening_20261026/fixed_test"
CURRENT_PATH = ROOT + "/table2_summary.json"
METRICS = ("AP", "max_F1", "R@100P", "R@1")
DASH = r"\cmedit{---}"


def _read(path: str, manifest: Dict[str, Any]) -> dict:
    raw = (REPO / path).read_bytes()
    manifest["sources"].append({"path": path, "sha256": hashlib.sha256(raw).hexdigest(),
                                "bytes": len(raw)})
    manifest["source_paths"].append(path)
    return json.loads(raw)


def _close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError("주결과 집계 불일치: {} != {}".format(actual, expected))


def _cells(stats: dict, sd: bool = True) -> str:
    cells = []
    for metric in METRICS:
        value = stats[metric]
        if not (0 <= value["mean"] <= 1 and math.isfinite(value["std"])):
            raise ValueError("유효하지 않은 지표")
        cell = (r"\cmedit{$%.4f_{\pm%.4f}$}" % (value["mean"], value["std"])
                if sd else r"\cmedit{%.4f}" % value["mean"])
        cells.append(cell)
    return " & ".join(cells)


def apply_current_main(tables: Dict[str, str], manifest: Dict[str, Any]) -> None:
    """확인된 새 주결과를 연결하고 별도 두 통과 결과는 부록용으로 보존한다."""
    data = _read(CURRENT_PATH, manifest)
    freeze = _read(ROOT + "/freeze.json", manifest)
    completed = _read(ROOT + "/completion_validation.json", manifest)
    if (not data["checks_passed"] or completed["status"] != "passed"
            or data["test_used_for_selection"] or freeze["fraction"] != 0.25
            or sorted(data["per_seed"]) != ["1", "2", "3"]
            or data["query_counts"] != {str(s): 195726 for s in (1, 2, 3)}):
        raise ValueError("새 주결과의 동결·완료·질의 조건이 다르다")
    summary_hash = hashlib.sha256((REPO / CURRENT_PATH).read_bytes()).hexdigest()
    freeze_hash = hashlib.sha256((REPO / ROOT / "freeze.json").read_bytes()).hexdigest()
    if completed["summary_sha256"] != summary_hash or data["freeze_sha256"] != freeze_hash:
        raise ValueError("완료 검사와 원천 해시가 다르다")
    for seed in ("1", "2", "3"):
        per = data["per_sequence"][seed]
        if set(per) != set(freeze["test"]) or len(per) != 46:
            raise ValueError("시험 시퀀스 목록 불일치")
        if sum(v["n_pos"] for v in per.values()) != 61682:
            raise ValueError("양성 질의 수 불일치")
        if [k for k, v in per.items() if v["n_pos"] == 0] != ["kitti/07"]:
            raise ValueError("양성 없는 시퀀스가 다르다")
        for metric in METRICS:
            cities = []
            for city, roster in freeze["cities"].items():
                value = statistics.mean(per[k][metric] for k in roster if per[k]["n_pos"])
                _close(value, data["per_seed"][seed]["cities"][city][metric])
                cities.append(value)
            _close(statistics.mean(cities), data["per_seed"][seed]["macro"][metric])
    stats = data["mean_std"]["whitening25"]
    for metric in METRICS:
        values = [data["per_seed"][s]["macro"][metric] for s in ("1", "2", "3")]
        _close(statistics.mean(values), stats[metric]["mean"])
        _close(statistics.stdev(values), stats[metric]["std"])
    _close(stats["R@1"]["mean"], data["mean_std"]["octave288"]["R@1"]["mean"])

    # 별도 구성의 표와 통계를 새 모델의 기여나 신뢰구간으로 재사용하지 않는다.
    for old, new, label in (
        ("table1_online.tex", "appdx_legacy_online.tex", "tab:legacy_results"),
        ("table_online_components.tex", "appdx_legacy_components.tex", "tab:legacy_components"),
    ):
        old_label = "tab:results" if old == "table1_online.tex" else "tab:online_components"
        text = tables[old].replace(old_label, label).replace("tab:results", "tab:legacy_results")
        text = text.replace(r"\caption{\cmedit{", r"\caption{\cmedit{Separate two-pass configuration (without temporal confidence or whitening). ", 1)
        text = text.replace(r"Descriptor construction follows Sec.~\ref{sec:graph};", "The two-pass configuration uses similarity-edge re-selection;")
        tables[new] = text
        manifest["tables"][new] = dict(manifest["tables"][old], label=label,
                                        role="separate_two_pass_configuration")
    for name in ("appdx_baseline_analysis.tex", "appdx_paired_bootstrap.tex", "table_context_systems.tex"):
        tables[name] = tables[name].replace("tab:results", "tab:legacy_results")
        tables[name] = tables[name].replace(r"\caption{\cmedit{", r"\caption{\cmedit{Separate two-pass configuration; these are not the current NSD results. ", 1)

    lines = tables["table1_online.tex"].splitlines()
    current = []
    for line in lines:
        if line.startswith(r"\textbf{NSD-H} &"):
            continue
        if line.startswith(r"\textbf{NSD} &"):
            line = r"\textbf{NSD} & " + _cells(stats) + r" \\"
        if line.startswith("% 출처:"):
            line = "% 출처: " + CURRENT_PATH + "; 비교 방법: results/pdv2_baseline_seeds/summary.json"
        line = line.replace("NSD, NSD-H, and the learned comparison methods", "NSD and the learned comparison methods")
        line = line.replace("its negative queries are retained separately.", "its negative queries are retained in the evaluation records.")
        current.append(line)
    # 주표의 평가 조건과 지표 정의는 본문의 Protocol and metrics에 둔다.
    main_table = "\n".join(current) + "\n"
    note_start = "\\par\\smallskip\n\\begin{minipage}{\\linewidth}\\footnotesize\n"
    table_body, marker, note = main_table.partition(note_start)
    if not marker or not note.startswith(r"\cmedit{Descriptor construction follows"):
        raise ValueError("주표의 평가 주석 경계가 다르다")
    _, marker, table_end = note.partition("\\end{minipage}\n")
    if not marker:
        raise ValueError("주표의 평가 주석이 닫히지 않았다")
    tables["table1_online.tex"] = table_body + table_end
    table4 = [
        "% scripts/_paper_current_nsd.py가 원천 결과로 생성한다. 숫자를 직접 고치지 않는다.",
        "% 출처: " + CURRENT_PATH,
        r"\begin{table}[t]",
        r"\caption{\cmedit{\textbf{NSD ablations.} Three-seed, six-city means under the Table~\ref{tab:results} protocol.}}",
        r"\label{tab:online_components}",
        r"\centering\footnotesize\color{blue}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Variant & AP & Max F1 & R@100P & Revisit R@1 \\", r"\midrule",
        r"\textbf{NSD} & " + _cells(stats, False) + r" \\",
        r"w/o correlation correction & " + _cells(data["mean_std"]["octave288"], False) + r" \\",
    ]
    from _paper_table4_current import verified_rows
    ablation_rows, ablation_sources = verified_rows(REPO, _read, manifest)
    for arm, label in (("graph_free", "Graph-free learned projection"),
                       ("standard", "Standard temporal attention"),
                       ("key_only", "Key index only"),
                       ("aux_only", "Auxiliary index only")):
        cells = _cells(ablation_rows[arm], False) if arm in ablation_rows else " & ".join([DASH] * 4)
        table4.append(label + " & " + cells + r" \\")
    table4 += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    tables["table_online_components.tex"] = "\n".join(table4)

    manifest["schema"] = "pdv2-paper-tables-v2"
    manifest["main_configuration"] = "octave288_temporal_diff_geometry_interaction_arc_unique_whitening25"
    manifest["main_decision_statistic"] = {
        "fraction": 0.25, "temporal_score": "arc_unique_b100",
        "channels": {s: freeze["confidence"][s]["decision"] for s in ("1", "2", "3")},
        "fixed_threshold_calibrated_for_mixture": False,
    }
    manifest["tables"]["table1_online.tex"].update({
        "source_paths": [CURRENT_PATH, ROOT + "/freeze.json", ROOT + "/completion_validation.json",
                         "results/pdv2_main_official_baselines/baselines_common.json",
                         "results/pdv2_baseline_seeds/summary.json"],
        "json_field": "mean_std.whitening25 / methods.*.mean_std",
        "refiner_checkpoint_selection_poses": "KISS-ICP odometry",
        "checkpoint_selection_metric": "final validation AP",
        "filled": "NSD 시간 전용·25% 상관 보정 및 비교 방법 10종. NSD-H는 별도 구성 부록에 둔다.",
    })
    manifest["tables"]["table_online_components.tex"].update({
        "source_paths": [CURRENT_PATH, *ablation_sources],
        "filled": "주모델·상관 보정 제거 및 검산을 마친 현재 구성 대조: " + ", ".join(sorted(ablation_rows)),
        "selection": "보정 제거는 동일 승자. 나머지는 현재 구성의 학습·검증 동결 뒤 3시드 최종 시험 원장만 사용.",
        "completed_ablation_rows": sorted(ablation_rows),
        "metric": list(METRICS),
    })
    manifest["output_sha256"] = {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest() for name, value in tables.items()
    }
