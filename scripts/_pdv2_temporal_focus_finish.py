#!/usr/bin/env python3
"""완료된 시간 문맥 비교의 동결 자료와 보고서 산술을 재검산한다."""
import ast
import csv
import hashlib
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_temporal_focus_20261018"
CAP = REPO / "results/pdv2_temporal_capacity_20261018"
REPORTS = OUT / "reports"


def read(path: Path) -> dict:
    """JSON 산출물을 읽는다."""
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    """큰 파일도 나누어 읽어 SHA-256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """동결 해시·같은 부분 보고서·최종 평균·AP 몫·CSV를 검사한다."""
    hashes, manifests = {}, []
    for root in (OUT, CAP):
        assert read(root / "status.json")["status"] == "complete"
        assert read(root / "completion_validation.json")["status"] == "passed"
        protocol = read(root / "protocol.json")
        assert protocol["test_used"] is False
        manifests.extend(protocol[k] for k in ("runtime_sha256", "input_sha256"))
    full = read(REPORTS / "post_analysis.json")
    primary = read(REPORTS / "primary_post_analysis.json")
    comparison = read(CAP / "comparison.json")
    for document in (full, primary, comparison):
        assert document["test_used"] is False
        manifests.append(document["sources"])
    for manifest in manifests:
        for name, expected in manifest.items():
            if name not in hashes:
                hashes[name] = sha(REPO / name)
            assert hashes[name] == expected, name
    for name in primary["mean_std"]:
        assert primary["mean_std"][name] == full["mean_std"][name]
    for seed in ("1", "2", "3"):
        for key in ("per_seed", "ap_decomposition", "fixed_epoch_effects_pp"):
            for name, value in primary[key][seed].items():
                assert value == full[key][seed][name], (key, seed, name)
    checked = 0
    for method, operators in full["mean_std"].items():
        for op, metrics in operators.items():
            for metric, stats in metrics.items():
                values = [full["per_seed"][s][method][op][metric] for s in ("1", "2", "3")]
                assert abs(stats["mean"] - statistics.mean(values)) < 1e-12
                assert abs(stats["std"] - statistics.stdev(values)) < 1e-12
                checked += 1
    for seed, changes in full["ap_decomposition"].items():
        for change, operators in changes.items():
            for op, row in operators.items():
                for values in (row["macro"], *row["families"].values()):
                    parts = sum(values[k] for k in ("recovered", "regressed", "both_correct", "both_wrong", "negative"))
                    assert abs(parts - (values["after"] - values["before"])) < 1e-12
                if change == "capacity_point_to_standard":
                    for metric, expected in comparison["rows"][seed][op]["delta_pp"].items():
                        values = full["per_seed"][seed]
                        actual = (values["standard"][op][metric] - values["capacity_point"][op][metric]) * 100
                        assert abs(actual - expected) < 1e-12
    table = list(csv.DictReader((REPORTS / "all_three_seed_metrics.csv").open()))
    assert len(table) == 24
    for row in table:
        expected = full["per_seed"][row["seed"]][row["method"]][row["op"]]
        for metric, value in expected.items():
            assert abs(float(row[metric]) - value * 100) < 1e-12
    for path in REPO.glob("scripts/*temporal_focus*.py"):
        ast.parse(path.read_text(), filename=str(path))
    artifact_paths = [*REPORTS.glob("*.json"), *REPORTS.glob("*.csv"), *REPORTS.glob("*.png"), *REPORTS.glob("*.pdf")]
    target = REPORTS / "completion_report_validation.json"
    artifacts = {str(p.relative_to(REPO)): sha(p) for p in sorted(artifact_paths) if p != target}
    result = {"status": "passed", "unique_frozen_and_source_files": len(hashes),
              "mean_std_pairs": checked, "csv_rows": len(table), "primary_full_overlap_equal": True,
              "test_used": False, "script_sha256": sha(Path(__file__)), "artifacts": artifacts}
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print({k: v for k, v in result.items() if k != "artifacts"})


if __name__ == "__main__":
    main()
