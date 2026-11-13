#!/usr/bin/env python3
"""실제 검색 연결의 원장·과거 후보·지연·최종 지표를 독립 검산한다."""
from pathlib import Path

import numpy as np

import _pdv2_contrast_pipeline as pipeline

run = pipeline.run
ROOT = pipeline.ROOT


def verify(root: Path, role: str) -> dict:
    """동결 출처와 원장을 읽어 지표를 재계산한다."""
    spec = run.READ(root / "protocol.json")
    for name, digest in spec["sources"].items():
        assert run.HASH(ROOT / name) == digest, name
    stats = dict(
        ledgers=0,
        rows=0,
        queries=0,
        prefix_checks=0,
        maximum_metric_error=0.0,
        maximum_prefix_error=0.0,
        reference_reproductions=0,
    )
    for sequence in spec[role]:
        folder = root / role / sequence.replace("/", "_")
        meta = run.READ(folder / "summary.json")
        poolpath = folder / "candidate_pool.npz"
        assert run.HASH(poolpath) == meta["candidate_pool_sha256"]
        pool = run.ARR(poolpath)
        q = pool["query"]
        assert np.all(np.diff(q) == 1) and q[0] == 30
        assert len(pool["offset"]) == len(q) + 1
        assert pool["offset"][0] == 0 and pool["offset"][-1] == len(pool["ids"])
        sizes = np.diff(pool["offset"])
        assert (sizes > 0).all() and (sizes <= 200).all()
        assert (pool["ids"] <= np.repeat(q - 30, sizes)).all()
        assert np.isfinite(pool["static"]).all()
        stats["queries"] += len(q)
        reference = None
        for tag, case in spec["cases"].items():
            path = folder / (tag + ".npz")
            record = run.READ(path.with_suffix(".json"))
            assert run.HASH(path) == record["sha256"]
            assert record["checkpoint_sha256"] == case.get("checkpoint_sha256")
            row = run.ARR(path)
            np.testing.assert_array_equal(row["query_idx"], q)
            assert (row["j_star"] <= q - 30).all()
            assert np.isfinite(row["score"]).all()
            if reference is None:
                reference = row
            else:
                for field in ("query_idx", "min_past_gt_dist", "group"):
                    np.testing.assert_array_equal(row[field], reference[field])
            for n, j in enumerate(row["j_star"]):
                lo, hi = pool["offset"][n : n + 2]
                assert j in pool["ids"][lo:hi]
            actual = run.base.independent_metrics(row, row["score"])
            independent = run.white.prior.BASE.exp.metrics(row["score"], row)
            for m in run.base.METRICS:
                if actual[m] is not None:
                    error = max(
                        abs(actual[m] - independent[m]),
                        abs(actual[m] - record["metrics"][m]),
                        abs(actual[m] - meta["metrics"][tag][m]),
                    )
                    assert error < 1e-12
                    stats["maximum_metric_error"] = max(
                        stats["maximum_metric_error"], error
                    )
            assert len(record["prefix_checks"]) == 2
            for c in record["prefix_checks"]:
                assert c["score_error"] < 8e-6
                i = np.searchsorted(q, c["query"])
                assert q[i] == c["query"] and row["j_star"][i] == c["winner"]
                stats["maximum_prefix_error"] = max(
                    stats["maximum_prefix_error"], c["score_error"]
                )
                stats["prefix_checks"] += 1
            if "reproduction" in record:
                assert record["reproduction"]["winners_exact"]
                assert record["reproduction"]["score_error"] < 8e-6
                stats["reference_reproductions"] += 1
            stats["ledgers"] += 1
            stats["rows"] += len(q)
    figures = run.READ(ROOT / "results/pdv2_learning_plan_20261104/restoration.json")[
        "figure_sha256"
    ]
    for name, digest in figures.items():
        assert run.HASH(ROOT / "docs/paper/body_v2.2" / name) == digest, name
    stats.update(
        status="passed",
        unchanged_figures=len(figures),
        protocol_sha256=run.HASH(root / "protocol.json"),
        verifier_sha256=run.HASH(Path(__file__)),
        test_used=(role == "test"),
    )
    run.WRITE(root / "integrated_validation.json", stats)
    print(stats, flush=True)
    return stats


if __name__ == "__main__":
    pipeline.e.configure()
    verify(pipeline.OUT, "val")
