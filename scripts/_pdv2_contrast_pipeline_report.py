#!/usr/bin/env python3
"""고정 시험 결과의 학습 기여·도시 차이·짝지은 불확실성을 보고한다."""
import argparse
import time
from pathlib import Path

import numpy as np

import _pdv2_contrast_pipeline_test as experiment

run, ROOT, OUT = experiment.run, experiment.ROOT, experiment.OUT
PAIRS = (
    ("diff", "fixed"),
    ("convs_only", "fixed"),
    ("diff", "other_only"),
    ("diff", "abs_kv"),
    ("diff", "point_gate"),
    ("diff", "bev_native"),
    ("diff", "bev_temporal"),
)


def prepare() -> dict:
    """결과 집계 전에 비교 쌍과 재표집 절차를 고정한다."""
    target = OUT / "analysis_plan.json"
    spec = dict(
        created_unix=time.time(),
        pairs=PAIRS,
        replicates=10000,
        seed=20261104,
        sampling="고정한세모델의차이를시퀀스별평균후도시안에서시퀀스수를유지하며복원추출. 도시6개동일비중. 양성없는KITTI07은제외.",
        scope="시퀀스를단위로삼아질의독립을가정하지않는탐색적조건부95%구간. 고정3모델의학습불확실성,도시밖일반화,시험개발노출,다중비교를해결하지않음. 비유의차이를동등성으로해석하지않음.",
        protocol_sha256=run.HASH(OUT / "protocol.json"),
        source_sha256=run.HASH(Path(__file__)),
    )
    if target.exists():
        old = run.READ(target)
        assert old["source_sha256"] == spec["source_sha256"]
        assert old["protocol_sha256"] == spec["protocol_sha256"]
        return old
    run.WRITE(target, spec)
    return spec


def report() -> None:
    """좋고 나쁜 도시와 모든 고정 대조를 함께 계산한다."""
    plan = prepare()
    summary = run.READ(OUT / "summary.json")
    assert run.READ(OUT / "integrated_validation.json")["status"] == "passed"
    per = summary["per_case"]

    def tag(kind, seed):
        return kind if kind in ("identity", "fixed") else f"{kind}_s{seed}"

    result = {}
    for a, b in PAIRS:
        seed_delta = {
            str(s): {
                m: 100 * (per[tag(a, s)]["macro"][m] - per[tag(b, s)]["macro"][m])
                for m in run.base.METRICS
            }
            for s in (1, 2, 3)
        }
        city_delta = {}
        city_draws = []
        rng = np.random.default_rng(plan["seed"])
        for city, sequences in run.current.CITY.items():
            available = [
                q for q in sequences if per[tag(a, 1)]["sequences"][q]["n_pos"]
            ]
            matrix = np.array(
                [
                    [
                        np.mean(
                            [
                                100
                                * (
                                    per[tag(a, s)]["sequences"][q][m]
                                    - per[tag(b, s)]["sequences"][q][m]
                                )
                                for s in (1, 2, 3)
                            ]
                        )
                        for m in run.base.METRICS
                    ]
                    for q in available
                ]
            )
            city_delta[city] = dict(zip(run.base.METRICS, matrix.mean(0).tolist()))
            index = rng.integers(
                0, len(available), size=(plan["replicates"], len(available))
            )
            city_draws.append(matrix[index].mean(1))
        boot = np.stack(city_draws).mean(0)
        intervals = {
            m: np.quantile(boot[:, i], [0.025, 0.975]).tolist()
            for i, m in enumerate(run.base.METRICS)
        }
        result[a + "-" + b] = dict(
            per_seed_pp=seed_delta,
            city_delta_pp=city_delta,
            conditional_95_pp=intervals,
        )
    report = dict(
        comparisons=result,
        mean=summary["mean"],
        learning_checks=summary["learning_checks"],
        working_comparability=summary["working_comparability"],
        interval_scope=plan["scope"],
        test_used_for_selection=False,
        sources={
            str(p.relative_to(ROOT)): run.HASH(p)
            for p in (
                OUT / "summary.json",
                OUT / "integrated_validation.json",
                OUT / "analysis_plan.json",
                Path(__file__),
            )
        },
    )
    run.WRITE(OUT / "integrated_report.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "report"))
    args = parser.parse_args()
    globals()[args.action]()
