#!/usr/bin/env python3
"""공통 성분 감산의 손실 교정과 K/V 차이 기여를 분리해 보고한다."""
from pathlib import Path

import _pdv2_diffattn_contrast_recalibrated as e


def main() -> None:
    """원 선택 기준을 유지하며 사전 고정한 추가 기전 비교를 계산한다."""
    spec = e.prepare()
    root = e.OUT
    data = e.e.run.READ(root / "summary.json")
    plan = e.e.run.READ(root / "mechanism_analysis_plan.json")
    assert plan["validation_ledgers_present"] == 0
    result = data["results"]

    def delta(a, b):
        return {
            m: 100 * (result[a]["macro"][m] - result[b]["macro"][m])
            for m in e.e.run.base.METRICS
        }

    comparisons = {}
    for suffix in plan["fixed_budget"]:
        comparisons[suffix] = dict(
            difference_key=delta("diff_" + suffix, "abs_k_" + suffix),
            difference_value=delta("abs_k_" + suffix, "abs_kv_" + suffix),
            full_vs_point=delta("diff_" + suffix, "point_gate_" + suffix),
            full_vs_gate=delta("diff_" + suffix, "gate_only_" + suffix),
            full_vs_initial=delta("diff_" + suffix, "transport_-0.5"),
        )
    # 이 대조는 원래의 '모든 대조보다 우위' 선택 판정을 바꾸지 않는다.
    controls = [
        data["best"][k] for k in ("abs_kv", "point_gate", "gate_only", "global")
    ]
    controls += [
        tag
        for tag in result
        if tag == "identity" or tag.startswith(("transport_", "plain_"))
    ]
    tests = {}
    for kind in ("diff", "abs_k"):
        chosen = data["best"][kind]
        tests[kind] = {
            f"{control}_{m}": result[chosen]["macro"][m]
            >= result[control]["macro"][m] + gain
            for control in controls
            for m, gain in (("AP", 0.001), ("max_F1", -0.003), ("R@100P", -0.01))
        }
    old = e.e.run.READ(e.PARENT / "summary.json")
    recalibration = {
        kind: {
            m: 100
            * (
                result[data["best"][kind]]["macro"][m]
                - old["results"][old["best"][kind]]["macro"][m]
            )
            for m in e.e.run.base.METRICS
        }
        for kind in data["best"]
    }
    out = dict(
        primary_selected=data["selected_matching_candidate"],
        fixed_budget=comparisons,
        difference_value_followup_checks=tests,
        difference_value_followup_candidates=[
            k for k, v in tests.items() if all(v.values())
        ],
        recalibration_best_delta_pp=recalibration,
        caveat="추가기전비교는원선택기준을대체하지않음. 단일시드의반복검증선택이며고정확증필요. R1과BEV동일시간규약의격차도유지.",
        test_used=False,
        protocol_sha256=e.e.run.HASH(root / "protocol.json"),
        analysis_plan_sha256=e.e.run.HASH(root / "mechanism_analysis_plan.json"),
        source_sha256=e.e.run.HASH(Path(__file__)),
    )
    e.e.run.WRITE(root / "learning_report.json", out)
    print(out)


if __name__ == "__main__":
    main()
