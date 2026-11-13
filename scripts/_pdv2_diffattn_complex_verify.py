#!/usr/bin/env python3
"""복소 차이 전달의 원장·학습 반복·출처와 그림 보존을 검산한다."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _pdv2_diffattn_complex_difference as e


def verify(learned: bool) -> dict:
    """각 원장의 최종 지표를 다시 계산하고 합성·접두 검사와 합친다."""
    if learned:
        import _pdv2_diffattn_complex_learn as learning

        spec = learning.prepare()
        root = learning.OUT
    else:
        spec = e.prepare()
        root = e.OUT
    summary = e.run.READ(root / "summary.json")
    stats = dict(
        rows=0,
        metric_sets=0,
        prefix_checks=0,
        maximum_metric_error=0.0,
        maximum_prefix_error=0.0,
        maximum_identity_error=0.0,
    )
    all_metrics = {}
    for s in spec["val"]:
        path = root / "val" / (s.replace("/", "_") + ".npz")
        record = e.run.READ(path.with_suffix(".json"))
        assert e.run.HASH(path) == record["ledger_sha256"]
        data = e.run.ARR(path)
        original = e.run.ARR(e.PARENT / "pools/val" / s.replace("/", "_") / "pool.npz")
        row = {
            k: data[k]
            for k in (
                "query_idx",
                "j_star",
                "min_past_gt_dist",
                "top1_geo_dist",
                "group",
            )
        }
        for k, v in row.items():
            np.testing.assert_array_equal(v, original[k])
        assert np.all(row["j_star"] <= row["query_idx"] - 30)
        stats["rows"] += len(row["query_idx"])
        for tag, m in record["metrics"].items():
            assert np.isfinite(data[tag]).all()
            measured = e.run.base.independent_metrics(row, data[tag])
            other = e.run.white.prior.BASE.exp.metrics(data[tag], row)
            for k in e.run.base.METRICS:
                if m[k] is not None:
                    error = max(abs(m[k] - measured[k]), abs(m[k] - other[k]))
                    assert error < 1e-12
                    stats["maximum_metric_error"] = max(
                        stats["maximum_metric_error"], error
                    )
            all_metrics.setdefault(tag, {})[s] = measured
            stats["metric_sets"] += 1
        for check in record["prefix_checks"]:
            assert check["coefficient_error"] < 4e-6 and check["score_error"] <= 8e-6
            stats["maximum_prefix_error"] = max(
                stats["maximum_prefix_error"], check["score_error"]
            )
            stats["prefix_checks"] += 1
        stats["maximum_identity_error"] = max(
            stats["maximum_identity_error"], record.get("identity_error", 0)
        )
    for tag, rows in all_metrics.items():
        recalculated = e.run.base.macro(rows)
        for k, v in recalculated["macro"].items():
            assert abs(v - summary["results"][tag]["macro"][k]) < 1e-12
    if learned:
        common = None
        for arm in spec["arms"]:
            rec = e.run.READ(root / arm / "training_complete.json")
            assert rec["updates"] == 512
            orders = [v["batch_hashes"] for v in rec["records"]]
            if common is None:
                common = orders
            assert orders == common
            for name, h in rec["checkpoints"].items():
                assert e.run.HASH(root / arm / name) == h
        assert e.run.READ(root / "repetition.json")["exact_weights"]
        assert e.run.READ(root / "preflight.json")["status"] == "passed"
        stats.update(
            training_conditions=len(spec["arms"]),
            updates=512 * len(spec["arms"]),
            additional_repeat_updates=512,
            deterministic_repetition=True,
            equal_batches=True,
        )
    figures = e.run.READ(
        e.ROOT / "results/pdv2_learning_plan_20261104/restoration.json"
    )["figure_sha256"]
    for name, h in figures.items():
        assert e.run.HASH(e.ROOT / "docs/paper/body_v2.2" / name) == h
    stats.update(
        status="passed",
        unchanged_figures=len(figures),
        test_used=False,
        verifier_sha256=e.run.HASH(Path(__file__)),
    )
    e.run.WRITE(root / "integrated_validation.json", stats)
    print(stats)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned", action="store_true")
    args = parser.parse_args()
    e.run.configure()
    verify(args.learned)
