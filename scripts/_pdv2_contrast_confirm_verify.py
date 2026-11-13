#!/usr/bin/env python3
"""세 시드와 갱신 범위 대조의 출처·최종 원장·기여를 함께 검산한다."""
from pathlib import Path

import numpy as np
import torch

import _pdv2_diffattn_contrast_seeds as e


def main() -> None:
    """같은 검증 질의에서 신경망 학습 범위의 세 시드 차이를 계산한다."""
    root = e.CAMPAIGN
    scope = e.e.ROOT / "results/pdv2_diffattn_contrast_scope_20261104"
    paths = [root / f"s{s}" for s in (1, 2, 3)] + [scope, scope / "s2", scope / "s3"]
    stats = dict(
        rows=0,
        metric_sets=0,
        prefix_checks=0,
        maximum_metric_error=0.0,
        maximum_prefix_error=0.0,
    )
    orders = None
    for p in paths:
        protocol = e.e.run.READ(p / "protocol.json")
        for name, h in protocol["sources"].items():
            assert e.e.run.HASH(e.e.ROOT / name) == h, name
        for arm in protocol["arms"]:
            train = e.e.run.READ(p / arm / "training_complete.json")
            assert train["updates"] == 512
            batches = [v["batch_hashes"] for v in train["records"]]
            if orders is None:
                orders = batches
            assert orders == batches
            if arm == "convs_only":
                assert all(n.startswith("convs.") for n in train["changed_parameters"])
            if arm == "other_only":
                assert all(
                    not n.startswith("convs.") for n in train["changed_parameters"]
                )
            for name, h in train["checkpoints"].items():
                assert e.e.run.HASH(p / arm / name) == h
        for s in protocol["val"]:
            name = s.replace("/", "_")
            meta = e.e.run.READ(p / "val" / (name + ".json"))
            path = p / "val" / (name + ".npz")
            assert e.e.run.HASH(path) == meta["ledger_sha256"]
            data = e.e.run.ARR(path)
            original = e.e.run.ARR(e.e.prior.PARENT / "pools/val" / name / "pool.npz")
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
            stats["rows"] += len(row["query_idx"])
            for tag, m in meta["metrics"].items():
                assert np.isfinite(data[tag]).all()
                measured = e.e.run.base.independent_metrics(row, data[tag])
                other = e.e.run.white.prior.BASE.exp.metrics(data[tag], row)
                for k in e.e.run.base.METRICS:
                    if m[k] is not None:
                        error = max(abs(m[k] - measured[k]), abs(m[k] - other[k]))
                        assert error < 1e-12
                        stats["maximum_metric_error"] = max(
                            stats["maximum_metric_error"], error
                        )
                stats["metric_sets"] += 1
            for check in meta["prefix_checks"]:
                assert (
                    check["coefficient_error"] < 4e-6 and check["score_error"] <= 8e-6
                )
                stats["maximum_prefix_error"] = max(
                    stats["maximum_prefix_error"], check["score_error"]
                )
                stats["prefix_checks"] += 1
    stages = {}
    for name, tag in (
        ("frozen", "transport_-0.5"),
        ("convs_only", "convs_only_8"),
        ("other_only", "other_only_8"),
        ("full", "full_8"),
    ):
        values = [
            e.e.run.READ(p / "summary.json")["results"][tag]["macro"] for p in paths[3:]
        ]
        stages[name] = dict(
            per_seed=values,
            mean={
                m: float(np.mean([v[m] for v in values])) for m in e.e.run.base.METRICS
            },
        )
    contrasts = {}
    for a, b in (
        ("full", "frozen"),
        ("convs_only", "frozen"),
        ("other_only", "frozen"),
        ("full", "other_only"),
    ):
        per = [
            {m: 100 * (x[m] - y[m]) for m in e.e.run.base.METRICS}
            for x, y in zip(stages[a]["per_seed"], stages[b]["per_seed"])
        ]
        contrasts[a + "-" + b] = dict(
            per_seed=per,
            mean={m: float(np.mean([v[m] for v in per])) for m in e.e.run.base.METRICS},
        )
    for seed in (1, 2, 3):
        assert e.e.run.READ(root / f"s{seed}/repetition.json")["exact_weights"]
    assert e.e.run.READ(root / "cache_validation.json")["maximum_score_error"] == 0
    figures = e.e.run.READ(
        e.e.ROOT / "results/pdv2_learning_plan_20261104/restoration.json"
    )["figure_sha256"]
    for name, h in figures.items():
        assert e.e.run.HASH(e.e.ROOT / "docs/paper/body_v2.2" / name) == h
    stats.update(
        status="passed",
        equal_batches=True,
        scope_update_masks_verified=True,
        unchanged_figures=len(figures),
        test_used=False,
        source_sha256=e.e.run.HASH(Path(__file__)),
    )
    e.e.run.WRITE(root / "integrated_validation.json", stats)
    e.e.run.WRITE(
        scope / "three_seed_summary.json",
        dict(
            stages=stages,
            contrasts_pp=contrasts,
            source_sha256=e.e.run.HASH(Path(__file__)),
            test_used=False,
        ),
    )
    e.e.run.WRITE(scope / "integrated_validation.json", stats)
    print(stats, stages, contrasts, flush=True)


if __name__ == "__main__":
    e.e.configure()
    main()
