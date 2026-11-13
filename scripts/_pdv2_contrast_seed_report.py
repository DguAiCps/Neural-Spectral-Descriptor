#!/usr/bin/env python3
"""첫 시드의 고정 설정에 대한 세 시작 시드의 최종 검증을 합친다."""
from pathlib import Path

import numpy as np
import torch

import _pdv2_diffattn_contrast_seeds as e


def main() -> None:
    """설정 재선택 없이 평균·변동·쌍 차이와 후속 기준을 계산한다."""
    frozen = e.freeze()
    root = e.CAMPAIGN
    rows = {
        seed: e.e.run.READ(root / f"s{seed}" / "summary.json") for seed in (1, 2, 3)
    }
    tags = dict(frozen["selected_tags"])
    tags.update(
        {
            "fixed": "transport_-0.5",
            "fixed_plain": "plain_-0.5",
            "identity": "identity",
            "point_same": "point_gate_lr4_8",
            "gate_same": "gate_only_lr4_8",
        }
    )
    stats = {}
    for kind, tag in tags.items():
        per = {str(seed): row["results"][tag]["macro"] for seed, row in rows.items()}
        stats[kind] = dict(
            per_seed=per,
            mean={
                m: float(np.mean([v[m] for v in per.values()]))
                for m in e.e.run.base.METRICS
            },
            std={
                m: float(np.std([v[m] for v in per.values()], ddof=1))
                for m in e.e.run.base.METRICS
            },
        )
    comparisons = {}
    for a, b in (
        ("diff", "fixed"),
        ("diff", "abs_kv"),
        ("abs_k", "abs_kv"),
        ("diff", "abs_k"),
        ("diff", "point_gate"),
        ("diff", "gate_only"),
        ("diff", "global"),
    ):
        per = {
            str(seed): {
                m: 100
                * (
                    stats[a]["per_seed"][str(seed)][m]
                    - stats[b]["per_seed"][str(seed)][m]
                )
                for m in e.e.run.base.METRICS
            }
            for seed in (1, 2, 3)
        }
        comparisons[a + "-" + b] = dict(
            per_seed=per,
            mean={
                m: float(np.mean([v[m] for v in per.values()]))
                for m in e.e.run.base.METRICS
            },
        )
    controls = [
        "fixed",
        "fixed_plain",
        "identity",
        "abs_kv",
        "point_gate",
        "gate_only",
        "global",
        "point_same",
        "gate_same",
    ]
    checks = {
        f"{c}_{m}": stats["diff"]["mean"][m] >= stats[c]["mean"][m] + margin
        for c in controls
        for m, margin in (("AP", 0.001), ("max_F1", -0.003), ("R@100P", -0.01))
    }
    checks["positive_learning_at_least_two_seeds"] = (
        sum(v["AP"] > 0 for v in comparisons["diff-fixed"]["per_seed"].values()) >= 2
    )
    value = comparisons["abs_k-abs_kv"]
    value_checks = dict(
        mean_AP=value["mean"]["AP"] > 0,
        mean_R100=value["mean"]["R@100P"] > 0,
        at_least_two_AP=sum(v["AP"] > 0 for v in value["per_seed"].values()) >= 2,
    )
    originals = None
    for seed in (1, 2, 3):
        state = torch.load(
            root / f"s{seed}/global_lr4/epoch8.pth",
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"]
        if originals is None:
            originals = state
        assert all(torch.equal(v, originals[k]) for k, v in state.items())
    out = dict(
        stats=stats,
        comparisons_pp=comparisons,
        main_checks=checks,
        matching_candidate_passed=all(checks.values()),
        difference_value_checks=value_checks,
        difference_value_confirmed=all(value_checks.values()),
        frozen_config=frozen["selected_tags"],
        global_reproduction=True,
        test_used=False,
        comparable_not_confirmed=True,
        source_sha256=e.e.run.HASH(Path(__file__)),
    )
    e.e.run.WRITE(root / "summary.json", out)
    print(
        "MEAN",
        {k: v["mean"] for k, v in stats.items()},
        "MATCHING",
        out["matching_candidate_passed"],
        "VALUE",
        out["difference_value_confirmed"],
        flush=True,
    )


if __name__ == "__main__":
    main()
