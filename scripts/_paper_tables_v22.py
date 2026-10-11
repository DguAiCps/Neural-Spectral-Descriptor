#!/usr/bin/env python3
"""분할 v2 결과를 기다리는 CVPR v2.2 표 틀과 출처 목록을 생성한다.

이 생성기는 tables_rev/templates의 빈 표만 복사한다. 이전 분할이나
분할 v2 예비 실행의 결과 JSON은 읽지 않으며, v2.1 생성기와 독립적이다.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "docs/paper/body_v2.2/tables_rev"
TEMPLATES = DEFAULT_OUTPUT / "templates"
TABLE_LABELS = {
    "table1_online.tex": "tab:results",
    "table1_strict.tex": "tab:offline_results",
    "table2_xsession.tex": "tab:xsession",
    "table3_efficiency.tex": "tab:efficiency",
    "table_online_components.tex": "tab:online_components",
    "table_fixed_budget_final.tex": "tab:fixed_budget_final",
    "table_context_final.tex": "tab:context_final",
    "table_cost_accounting.tex": "tab:cost_accounting",
    "table_uncertainty.tex": "tab:uncertainty",
    "appdx_excluded.tex": "tab:excluded",
    "appdx_sensitivity.tex": "tab:roster_sensitivity",
    "appdx_xs_pairs.tex": "tab:xs_pairs_R_1_5",
    "appdx_xs_pairs20.tex": "tab:xs_pairs_R_1_20",
    "appdx_xs_thresholds.tex": "tab:xs_thresholds",
    "appdx_coverage.tex": "tab:xs_coverage",
    "appdx_causal_pr.tex": "tab:causal_pr",
}


def generate_tables(output_dir: Path) -> None:
    """빈 표를 복사하고 상태·라벨·틀의 해시를 기록한다."""
    contents: Dict[str, str] = {}
    records = []
    for filename, label in TABLE_LABELS.items():
        source = TEMPLATES / (filename + ".in")
        content = source.read_text(encoding="utf-8")
        if "\\label{" + label + "}" not in content:
            raise ValueError("표 틀의 라벨이 일치하지 않는다: " + str(source))
        contents[filename] = content
        records.append({
            "file": filename,
            "label": label,
            "status": "pending_split_v2",
            "source_template": str(source.relative_to(REPO)),
            "template_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "result_sources": [],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in contents.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "pending_split_v2",
        "generator": "scripts/_paper_tables_v22.py",
        "source_policy": {
            "description": "빈 표 틀만 생성하며 이전 분할·구조의 실험 결과를 재삽입하지 않는다.",
            "read_result_files": False,
            "pilot_results": "분할 v2 예비 결과는 별도 tables_v2에서만 관리하며 이 표에 승계하지 않는다.",
            "aggregation": "데이터셋별 열은 표의 틀이며 대표 집계의 묶음·가중은 미확정이다.",
            "placeholder": "\\cmedit{---}",
        },
        "tables": records,
    }
    (output_dir / "table_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """출력 폴더를 받아 분할 v2용 빈 표를 생성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="표와 출처 목록을 저장할 폴더",
    )
    output_dir = parser.parse_args().output_dir.resolve()
    generate_tables(output_dir)
    print("분할 v2 대기 표 {}개 생성: {}".format(len(TABLE_LABELS), output_dir))


if __name__ == "__main__":
    main()
