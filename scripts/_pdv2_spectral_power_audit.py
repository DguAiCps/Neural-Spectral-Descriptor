#!/usr/bin/env python3
"""고정 시험의 질의·정답 수·도시 및 시드 손익을 독립 검산한다."""
from pathlib import Path

import numpy as np

import _pdv2_spectral_power_test as run


def main() -> None:
    """후보 선택에는 쓰지 않는 최종 시험 감사 보고서를 남긴다."""
    frozen = run.freeze()
    selected = frozen["selected"]
    summary = run.read(run.OUT / "table2_summary.json")
    seeds, prefix, score_error = {}, 0, 0.
    for seed in frozen["seeds"]:
        total = {k: 0 for k in ("queries", "positive", "changed", "gained_correct", "lost_correct")}
        seqs = {}
        for sequence in frozen["test"]:
            name = sequence.replace("/", "_")
            info = run.read(run.OUT / "test" / f"s{seed}" / (name + ".json"))
            data = {}
            for arm in ("baseline", selected):
                record = info["arms"][arm]
                path = run.REPO / record["path"]
                assert run.sha256(path) == record["sha256"]
                data[arm] = run.arrays(path)
                assert run.independent_metrics(data[arm], data[arm]["score"]) == record["metrics"]
            a, b = data["baseline"], data[selected]
            for k in ("query_idx", "min_past_gt_dist", "group"):
                np.testing.assert_array_equal(a[k], b[k])
            assert np.isfinite(b["score"]).all()
            assert ((b["j_star"] >= 0) & (b["j_star"] <= b["query_idx"] - 30)).all()
            positive = b["min_past_gt_dist"] <= 5
            old = positive & (a["top1_geo_dist"] <= 5)
            new = positive & (b["top1_geo_dist"] <= 5)
            row = {"queries": len(positive), "positive": int(positive.sum()),
                   "changed": int((a["j_star"] != b["j_star"]).sum()),
                   "gained_correct": int((new & ~old).sum()), "lost_correct": int((old & ~new).sum())}
            for k in ("gained_correct", "lost_correct"):
                assert row[k] == info[k]
            for k in total:
                total[k] += row[k]
            seqs[sequence] = row
            prefix += len(info["prefix_checks"])
            score_error = max(score_error, info["baseline_score_max_error"])
        assert total["queries"] == 195726 and total["positive"] == 61682
        seeds[str(seed)] = {"total": total, "sequences": seqs}
    delta = {s: {m: 100 * (summary["per_seed"][selected][s]["macro"][m]
                         - summary["per_seed"]["baseline"][s]["macro"][m]) for m in run.METRICS}
             for s in seeds}
    wins = {m: sum(row[m] > 0 for row in summary["city_delta_pp"].values()) for m in run.METRICS}
    result = {"status": "passed", "selected": selected, "per_seed": seeds,
              "seed_delta_pp": delta, "cities_improved": wins,
              "prefix_checks": prefix, "baseline_score_max_error": score_error,
              "source_sha256": run.sha256(Path(__file__)),
              "summary_sha256": run.sha256(run.OUT / "table2_summary.json"),
              "scope": "시험 설정 재선택 없이 최종 결과와 손익만 검산. 시드별 같은 질의를 반복하므로 합계는 독립 질의 수가 아니다."}
    run.write_json(run.OUT / "audit.json", result)
    print("AUDIT", delta, wins, flush=True)


if __name__ == "__main__":
    main()
