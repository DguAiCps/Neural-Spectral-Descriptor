#!/usr/bin/env python3
"""세 주파수 키와 BEVPlace++의 동일 표2 지표를 검산하고 표로 저장한다."""
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_keys as run
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_no_context_ablation import sha256, write_json


def macro(rows: dict, cities: dict) -> dict:
    """양성이 있는 시퀀스→도시 균등 평균을 계산한다."""
    groups = {}
    for city, sequences in cities.items():
        items = [rows[s] for s in sequences if rows[s]["n_pos"]]
        assert items
        groups[city] = {m: float(np.mean([r[m] for r in items])) for m in run.METRICS}
    return {
        "cities": groups,
        "macro": {
            m: float(np.mean([v[m] for v in groups.values()])) for m in run.METRICS
        },
    }


def stats(rows: dict) -> dict:
    """시드별 도시 평균의 평균과 표본 표준편차를 계산한다."""
    result = {}
    for m in run.METRICS:
        values = [rows[str(s)]["macro"][m] for s in run.SEEDS]
        result[m] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
    return result


def main() -> None:
    """원장 전수 검산 후 3시드 최종 성능과 도시별 차이를 쓴다."""
    spec = run.protocol(True)
    frozen = run.read(run.OUT / "test_freeze.json")
    assert frozen["protocol_sha256"] == sha256(run.OUT / "protocol.json")
    assert frozen["validation_sha256"] == sha256(run.OUT / "validation_summary.json")
    for name, digest in frozen["inputs"].items():
        assert sha256(REPO / name) == digest, name
    baseline = run.read(REPO / frozen["baseline_summary"])
    assert baseline["checks_passed"]
    bev = baseline["methods"]["bevplace"]
    per_sequence, per_seed, sources = {}, {}, {}
    metric_count, query_count = 0, {}
    references = {}
    for arm in (*run.ARMS, "bevplace"):
        per_sequence[arm], per_seed[arm], query_count[arm] = {}, {}, {}
        for seed in run.SEEDS:
            rows, total = {}, 0
            for sequence in spec["test"]:
                name = sequence.replace("/", "_") + ".npz"
                if arm == "bevplace":
                    path = (
                        REPO
                        / bev["per_seed"][str(seed)]["source"]
                        / ("dump_bevplace_" + name)
                    )
                    full = run.base.exp.arrays(path)
                    q = references[(seed, sequence)]["query_idx"]
                    ids = np.searchsorted(full["query_idx"], q)
                    np.testing.assert_array_equal(full["query_idx"][ids], q)
                    row = {
                        k: v[ids]
                        for k, v in full.items()
                        if v.ndim and len(v) == len(full["query_idx"])
                    }
                    np.testing.assert_array_equal(
                        row["min_past_gt_dist"],
                        references[(seed, sequence)]["min_past_gt_dist"],
                    )
                    value = row[bev["score"]]
                else:
                    path = run.root(seed, arm) / "test" / name
                    info = run.read(path.with_suffix(".json"))
                    assert sha256(path) == info["sha256"]
                    for prefix, field in (
                        ("keys_", "keys_sha256"),
                        ("coverage_", "coverage_sha256"),
                    ):
                        assert sha256(path.with_name(prefix + name)) == info[field]
                    assert all(
                        p["winner_equal"] and p["score_error"] == 0
                        for p in info["prefixes"]
                    )
                    chosen = frozen["models"][str(seed)][arm]
                    assert info["checkpoint_sha256"] == chosen["sha256"]
                    row = run.base.exp.arrays(path)
                    value = row[run.METHOD]
                    references[(seed, sequence)] = row
                assert np.isfinite(value).all()
                result = independent_metrics(row, value)
                if arm != "bevplace":
                    expected = info["metrics"][run.METHOD]
                    for m in (*run.METRICS, "n_pos"):
                        if expected[m] is None:
                            assert result[m] is None
                        else:
                            np.testing.assert_allclose(
                                result[m], expected[m], atol=1e-12, rtol=0
                            )
                        metric_count += 1
                rows[sequence] = result
                total += len(value)
                sources[str(path.relative_to(REPO))] = sha256(path)
            assert total == 195726, (arm, seed, total)
            assert sum(r["n_pos"] > 0 for r in rows.values()) == 45
            query_count[arm][str(seed)] = total
            per_sequence[arm][str(seed)] = rows
            per_seed[arm][str(seed)] = macro(rows, frozen["cities"])
            if arm == "bevplace":
                for m in run.METRICS:
                    np.testing.assert_allclose(
                        per_seed[arm][str(seed)]["macro"][m],
                        bev["per_seed"][str(seed)]["macro"][m],
                        atol=1e-12,
                        rtol=0,
                    )
    mean_std = {arm: stats(rows) for arm, rows in per_seed.items()}
    delta = {
        arm: {
            m: 100 * (mean_std[arm][m]["mean"] - mean_std["bevplace"][m]["mean"])
            for m in run.METRICS
        }
        for arm in run.ARMS
    }
    city_delta = {
        arm: {
            city: {
                m: 100
                * np.mean(
                    [
                        per_seed[arm][str(seed)]["cities"][city][m]
                        - per_seed["bevplace"][str(seed)]["cities"][city][m]
                        for seed in run.SEEDS
                    ]
                )
                for m in run.METRICS
            }
            for city in frozen["cities"]
        }
        for arm in run.ARMS
    }
    summary = {
        "per_sequence": per_sequence,
        "per_seed": per_seed,
        "mean_std": mean_std,
        "minus_bevplace_pp": delta,
        "city_minus_bevplace_pp": city_delta,
        "protocol_sha256": sha256(run.OUT / "protocol.json"),
        "freeze_sha256": sha256(run.OUT / "test_freeze.json"),
        "test_used_for_selection": False,
        "query_counts": query_count,
        "checks_passed": True,
        "metric_checks": metric_count,
        "sources": sources,
        "paper_changed": False,
        "scope": "모든새조건은동일기하상호작용DiffAttnConv·최종판정이며원고기존NSD행과구별. "
        "각조건검증선택후동결한3시드평가. BEVPlace++공식재학습3시드공통원장재집계. "
        "기존시험개발노출이있으며새독립확증으로표현하지않음.",
    }
    write_json(run.OUT / "table2_summary.json", summary)
    labels = {
        "octave288": "NSD 집계 288D",
        "trunc288": "NSD 절단 288D",
        "trunc352": "NSD 절단 352D",
        "bevplace": "BEVPlace++",
    }
    with (run.OUT / "table2.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "method",
                *[
                    m + suffix
                    for m in run.METRICS
                    for suffix in ("_mean_pct", "_std_pct")
                ],
            ]
        )
        for arm in (*run.ARMS, "bevplace"):
            writer.writerow(
                [
                    labels[arm],
                    *[
                        100 * mean_std[arm][m][s]
                        for m in run.METRICS
                        for s in ("mean", "std")
                    ],
                ]
            )
    lines = [
        "| 방법 | AP (%) | 최대 F1 (%) | R@100P (%) | R@1 (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    tex = []
    for arm in (*run.ARMS, "bevplace"):
        cells = [
            f"{100*mean_std[arm][m]['mean']:.4f} ± {100*mean_std[arm][m]['std']:.4f}"
            for m in run.METRICS
        ]
        lines.append("| " + labels[arm] + " | " + " | ".join(cells) + " |")
        tex_cells = [
            f"{mean_std[arm][m]['mean']:.4f} $\\pm$ {mean_std[arm][m]['std']:.4f}"
            for m in run.METRICS
        ]
        tex.append(arm + " & " + " & ".join(tex_cells) + " \\\\")
    (run.OUT / "table2.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run.OUT / "table2_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("MINUS_BEVPLACE_PP", delta, flush=True)


if __name__ == "__main__":
    main()
