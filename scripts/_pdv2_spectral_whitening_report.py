#!/usr/bin/env python3
"""완료된 표2 수치와 도시별 손익을 검산해 검토용 표로 내보낸다."""
import csv
from pathlib import Path

import numpy as np

import _pdv2_spectral_whitening_test as run
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """새 평가 없이 고정 시험 결과에서 표와 손익 보고서를 생성한다."""
    new = run.run.read(run.OUT / "table2_summary.json")
    old = run.run.read(run.run.PRIOR / "table2_summary.json")
    audit = run.run.read(run.OUT / "completion_validation.json")
    assert (
        audit["status"] == "passed"
        and sha256(run.OUT / "table2_summary.json") == audit["summary_sha256"]
    )
    cities, sequences, seeds = {}, {}, {}
    for city in new["per_seed"]["1"]["cities"]:
        cities[city] = {
            m: 100
            * float(
                np.mean(
                    [
                        new["per_seed"][str(s)]["cities"][city][m]
                        - old["per_seed"]["octave288"][str(s)]["cities"][city][m]
                        for s in (1, 2, 3)
                    ]
                )
            )
            for m in run.METRICS
        }
    for seed in (1, 2, 3):
        seeds[str(seed)] = {
            m: 100
            * (
                new["per_seed"][str(seed)]["macro"][m]
                - old["per_seed"]["octave288"][str(seed)]["macro"][m]
            )
            for m in run.METRICS
        }
    for sequence, row in new["per_sequence"]["1"].items():
        if row["n_pos"]:
            sequences[sequence] = {
                m: 100
                * float(
                    np.mean(
                        [
                            new["per_sequence"][str(s)][sequence][m]
                            - old["per_sequence"]["octave288"][str(s)][sequence][m]
                            for s in (1, 2, 3)
                        ]
                    )
                )
                for m in run.METRICS
            }
    with (run.OUT / "table2.csv").open() as handle:
        for row in csv.DictReader(handle):
            expected = new["mean_std"][row["method"]]
            for metric in run.METRICS:
                for key, suffix in (("mean", "mean_pct"), ("std", "std_pp")):
                    np.testing.assert_allclose(
                        float(row[metric + "_" + suffix]),
                        100 * expected[metric][key],
                        atol=1e-12,
                        rtol=0,
                    )
    labels = {
        "octave288": "NSD 연구 기준",
        "whitening25": "NSD · 상관 보정 25%",
        "bevplace": "BEVPlace++",
    }
    texlabels = {
        "octave288": "NSD research reference",
        "whitening25": r"NSD + whitening 25\%",
        "bevplace": "BEVPlace++",
    }
    lines = [
        "| 조건 | AP (%) | 최대 F1 (%) | R@100P (%) | 최종 R@1 (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    tex = []
    for method in labels:
        values = new["mean_std"][method]
        lines.append(
            "| "
            + labels[method]
            + " | "
            + " | ".join(
                f"{100*values[m]['mean']:.3f} ± {100*values[m]['std']:.3f}"
                for m in run.METRICS
            )
            + " |"
        )
        tex.append(
            texlabels[method]
            + " & "
            + " & ".join(
                f"${values[m]['mean']:.4f} \\pm {values[m]['std']:.4f}$"
                for m in run.METRICS
            )
            + r" \\"
        )
    (run.OUT / "table2.txt").write_text("\n".join(lines) + "\n")
    (run.OUT / "table2_rows.tex").write_text("\n".join(tex) + "\n")
    with (run.OUT / "city_delta.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["city"] + [m + "_delta_pp" for m in run.METRICS])
        for city, row in cities.items():
            writer.writerow([city] + [row[m] for m in run.METRICS])
    write_json(
        run.OUT / "report.json",
        {
            "city_delta_pp": cities,
            "sequence_delta_pp": sequences,
            "seed_delta_pp": seeds,
            "AP_cities_improved": sum(v["AP"] > 0 for v in cities.values()),
            "AP_sequences_improved": sum(v["AP"] > 0 for v in sequences.values()),
            "AP_sequences_lower": sum(v["AP"] < 0 for v in sequences.values()),
            "csv_values_checked": 24,
            "source_sha256": sha256(Path(__file__)),
            "summary_sha256": audit["summary_sha256"],
            "artifacts": {
                name: sha256(run.OUT / name)
                for name in (
                    "table2.csv",
                    "table2.txt",
                    "table2_rows.tex",
                    "city_delta.csv",
                )
            },
        },
    )
    print("REPORT_PASSED", len(cities), len(sequences), flush=True)


if __name__ == "__main__":
    main()
