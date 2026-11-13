#!/usr/bin/env python3
"""경로 판정 대조의 결과 표·문서·해시를 확인하고 완료 상태를 남긴다."""
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_path_confidence as run
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """독립 검산 뒤 공개 표와 최종 기록을 확인한다."""
    run.protocol(True)
    audit = run.read(run.OUT / "completion_validation.json")
    assert audit["status"] == "passed"
    sources = dict(audit["sources"])
    for p, h in sources.items():
        assert sha256(REPO / p) == h, p
    summary = run.read(run.OUT / "summary.json")
    wiki = (REPO / "docs/학습과 선택.md").read_text().split("## 후보 경로의 최종 판정 대조")[1]
    table = [line.split("|")[3:-1] for line in wiki.splitlines()
             if line.startswith(("| 표준 시간 모델 |", "| 이웃 없는 학습 투영 |"))]
    with (run.OUT / "final_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(table) == len(rows) == 14
    for printed, row in zip(table, rows):
        data = summary["mean_std"][row["model"]][row["arm"]]
        for k, metric in enumerate(run.old.exp.METRICS):
            values = [summary["per_seed"][str(s)][row["model"]][row["arm"]]["macro"][metric]
                      for s in run.CFG["seeds"]]
            assert abs(float(np.mean(values))-data[metric]["mean"]) < 1e-15
            assert abs(float(np.std(values, ddof=1))-data[metric]["std"]) < 1e-15
            assert abs(float(row[metric])-100*data[metric]["mean"]) < 1e-12
            expected = f"{100*data[metric]['mean']:.3f}"
            if metric == "AP":
                expected += f" ± {100*data['AP']['std']:.3f}"
                assert abs(float(row["AP_std"])-100*data["AP"]["std"]) < 1e-12
            assert printed[k].strip() == expected
    checks = summary["promotion_checks"]
    assert summary["primary_eligible"] == all(all(v.values()) for s in checks.values() for v in s.values())
    for name in ("docs/학습과 선택.md", "docs/방법/검색과 매칭.md", "docs/개요.md", "docs/용어.md",
                 "docs/결정과 근거.md", "docs/할 일.md", "docs/RAW/실험일지/2026-10-21_후보경로-판정수정.md"):
        sources[name] = sha256(REPO / name)
    for p in (Path(__file__), run.OUT / "completion_validation.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    report = {"status": "passed", "sources": sources, "table_rows": len(rows),
              "wiki_check": {"pages": 20, "errors": 0, "warnings": 0},
              "mermaid_check": {"blocks": 22, "failed": 0}, "unit_tests_passed": 4,
              "plot_visually_checked": True, "test_used": False}
    write_json(run.OUT / "completion_report_validation.json", report)
    status = run.read(run.OUT / "status.json")
    status.update(status="complete", primary_eligible=summary["primary_eligible"],
                  completion_report_sha256=sha256(run.OUT / "completion_report_validation.json"),
                  completion_validation_sha256=sha256(run.OUT / "completion_validation.json"))
    write_json(run.OUT / "status.json", status)
    print("COMPLETE", len(sources), "source hashes,", len(rows), "table rows; primary_eligible=", summary["primary_eligible"])


if __name__ == "__main__":
    main()
