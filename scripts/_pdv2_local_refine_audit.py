#!/usr/bin/env python3
"""선택된 국소 보정의 최종 원장·회복 거리·시드별 차이를 집계한다."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import _pdv2_local_refine as local


def diagnose(role: str, sequences: list, selected: str) -> dict:
    """정답 라벨은 결정 이후의 회복·손실 분류에만 사용한다."""
    out, ledgers, prefixes, max_error = {}, 0, 0, 0.
    for seed in (1, 2, 3):
        total = {k: 0 for k in ("queries", "positive", "gained", "lost", "changed",
                                "gained_old_5_to_6m", "gained_old_6_to_10m", "gained_old_over_10m")}
        per_sequence = {}
        for sequence in sequences:
            name = sequence.replace("/", "_")
            root = local.OUT / role / f"s{seed}"
            info = local.read(root / (name + ".json"))
            a = local.arrays(root / f"{name}_baseline.npz")
            b = local.arrays(root / f"{name}_{selected}.npz")
            for arm in ("baseline", selected):
                item = info["arms"][arm]
                assert local.sha256(local.REPO / item["path"]) == item["sha256"]
                ledgers += 1
            for key in ("query_idx", "min_past_gt_dist", "group"):
                np.testing.assert_array_equal(a[key], b[key])
            assert np.isfinite(b["score"]).all()
            assert ((b["j_star"] >= 0) & (b["j_star"] <= b["query_idx"] - 30)).all()
            assert (abs(b["j_star"] - a["j_star"]) <= 1).all()
            positive = a["min_past_gt_dist"] <= 5
            old = positive & (a["top1_geo_dist"] <= 5)
            new = positive & (b["top1_geo_dist"] <= 5)
            gain = new & ~old
            distance = a["top1_geo_dist"][gain]
            row = {"queries": len(positive), "positive": int(positive.sum()),
                   "gained": int(gain.sum()), "lost": int((old & ~new).sum()),
                   "changed": int((a["j_star"] != b["j_star"]).sum()),
                   "gained_old_5_to_6m": int((distance <= 6).sum()),
                   "gained_old_6_to_10m": int(((distance > 6) & (distance <= 10)).sum()),
                   "gained_old_over_10m": int((distance > 10).sum())}
            per_sequence[sequence] = row
            for k, v in row.items():
                total[k] += v
            prefixes += len(info["prefix_checks"])
            max_error = max(max_error, info["baseline_score_max_error"])
        out[str(seed)] = {"total": total, "sequences": per_sequence}
    return {"per_seed": out, "checked_ledgers": ledgers, "prefix_checks": prefixes,
            "baseline_score_max_error": max_error,
            "scope": "같은질의의3모델반복집계. GT는최종오류진단에만사용하고온라인선택에는사용하지않음."}


def main() -> None:
    """두 분할의 검증을 완료된 시험 요약과 연결한다."""
    spec = local.prepare()
    frozen = local.read(local.OUT / "test_freeze.json")
    selected = frozen["selected"]
    assert selected == "radius1"
    summary = local.read(local.OUT / "table2_summary.json")
    records = {role: diagnose(role, roster, selected)
               for role, roster in (("val", spec["val"]), ("test", frozen["test"]))}
    records["test_seed_delta_pp"] = {s: {m: 100 * (summary["per_seed"][selected][s]["macro"][m]
        - summary["per_seed"]["baseline"][s]["macro"][m]) for m in local.METRICS} for s in ("1", "2", "3")}
    city_gains = summary["city_delta_pp"]
    records["cities_improved"] = {m: sum(row[m] > 0 for row in city_gains.values()) for m in local.METRICS}
    records["status"] = "passed"
    records["source_sha256"] = local.sha256(Path(__file__))
    records["summary_sha256"] = local.sha256(local.OUT / "table2_summary.json")
    local.write_json(local.OUT / "audit.json", records)
    print("AUDIT", records["test_seed_delta_pp"], records["cities_improved"], flush=True)


if __name__ == "__main__":
    main()
