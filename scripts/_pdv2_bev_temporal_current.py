#!/usr/bin/env python3
"""현재 NSD와 원고 제외 대조 'BEVPlace++ + 같은 시간 판정'의 짝지은 구간을 반박 대비용으로 계산한다.

BEVPlace++ 원장(`results/pdv2_review_response_20261104/bev/test/s{1,2,3}`)의 자체 점수(score_negl2)와
NSD와 같은 거리 대응 시간 판정 점수(score)를, 현재 NSD(`diff`) 시험 원장과 같은 질의·지표 구현·구간 절차
(`_pdv2_macro_ablation.intervals`)로 비교한다. 새 학습·재선택은 없다. 원고에는 싣지 않는다.
"""
import time

import numpy as np

import _pdv2_contrast_pipeline_test as T
import _pdv2_macro_ablation as M

run, ROOT = M.run, M.ROOT
BEV = ROOT / "results/pdv2_review_response_20261104/bev/test"
OUT = ROOT / "results/pdv2_bev_temporal_current_20261106"


def main() -> None:
    spec = T.activate()
    plan = dict(replicates=10000, seed=20261104)
    per = {f"{c}_s{s}": {} for c in ("diff", "bevt", "bev") for s in M.SEEDS}
    count = 0
    for sequence in spec["test"]:
        name = sequence.replace("/", "_")
        source = run.READ(M.SOURCE / "test" / name / "summary.json")
        for s in M.SEEDS:
            per[f"diff_s{s}"][sequence] = source["metrics"][f"diff_s{s}"]
            ref = run.ARR(M.SOURCE / "test" / name / f"diff_s{s}.npz")
            meta = run.READ(BEV / f"s{s}" / f"{name}.json")
            assert meta["status"] == "passed" and run.HASH(BEV / f"s{s}" / f"{name}.npz") == meta["sha256"]
            led = run.ARR(BEV / f"s{s}" / f"{name}.npz")
            for field in ("query_idx", "min_past_gt_dist"):
                np.testing.assert_array_equal(led[field], ref[field])
            # group은 질의 속성이며 모든 조건이 같은 기준 원장의 값을 쓴다(_pdv2_contrast_pipeline_test.py:111).
            row = dict(query_idx=led["query_idx"], j_star=led["j_star"], min_past_gt_dist=led["min_past_gt_dist"],
                       top1_geo_dist=led["top1_geo_dist"], group=ref["group"])
            per[f"bevt_s{s}"][sequence] = M.metrics(row, led["score"])
            per[f"bev_s{s}"][sequence] = M.metrics(row, led["score_negl2"])
        count += len(ref["query_idx"])
    assert count == 195726, count
    results = {tag: dict(**run.current.adoption.previous.macro(rows, run.current.CITY), sequences=rows)
               for tag, rows in per.items()}
    means = {k: {m: float(np.mean([results[f"{k}_s{s}"]["macro"][m] for s in M.SEEDS])) for m in run.base.METRICS}
             for k in ("diff", "bevt", "bev")}
    comparisons = {f"diff-{b}": dict(delta_pp={m: 100 * (means["diff"][m] - means[b][m]) for m in run.base.METRICS},
                                     **M.intervals(results, "diff", b, plan)) for b in ("bevt", "bev")}
    comparisons["bevt-bev"] = dict(delta_pp={m: 100 * (means["bevt"][m] - means["bev"][m]) for m in run.base.METRICS},
                                   **M.intervals(results, "bevt", "bev", plan))
    OUT.mkdir(parents=True, exist_ok=True)
    run.WRITE(OUT / "summary.json", dict(
        status="passed", queries_per_case=count, mean=means, comparisons=comparisons,
        intervals=dict(**plan, method="도시 안 양성 시퀀스 복원추출, 세 시작점 평균 차이, 도시 동일 비중(_pdv2_macro_ablation.intervals)"),
        scope="반박 대비용. 원고 제외 대조이며 시험 선택에 쓰지 않는다.", test_used_for_selection=False,
        sources={str(p.relative_to(ROOT)): run.HASH(p) for p in (ROOT / "scripts/_pdv2_bev_temporal_current.py",)},
        completed_unix=time.time()))
    for k, v in comparisons.items():
        print(k, {m: (round(v["delta_pp"][m], 3), [round(x, 3) for x in v["conditional_95_pp"][m]]) for m in run.base.METRICS})
    print({k: {m: round(100 * v, 2) for m, v in d.items()} for k, d in means.items()})


if __name__ == "__main__":
    main()
