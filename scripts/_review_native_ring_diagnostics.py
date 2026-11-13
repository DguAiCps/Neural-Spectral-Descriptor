#!/usr/bin/env python3
"""기존 RING 원장의 검색 성공과 수락 점수 분리를 선택 없이 진단한다."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_main_baselines_official import COMMON, METHODS, aggregate, sha256
from _pdv2_pilot import CITY, TEST

OUT = REPO / "results/pdv2_review_native_20261105"


def pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """양성 점수가 음성보다 높은 쌍의 비율을 동점 절반으로 계산한다."""
    if not len(positive) or not len(negative):
        return None
    neg = np.sort(negative)
    below = np.searchsorted(neg, positive, side="left")
    equal_or_below = np.searchsorted(neg, positive, side="right")
    return float(np.mean((below + equal_or_below) / (2.0 * len(neg))))


def main() -> None:
    """현재 주표를 재검산하고 별도 진단 보고서를 저장한다."""
    sources = {}
    def track(path: Path) -> Path:
        sources[str(path.relative_to(REPO))] = sha256(path)
        return path
    base = {}
    for row in METHODS:
        if row[0] in ("ring", "ringpp"):
            base[row[0]] = aggregate(*row[1:], track)
    per_sequence = {}
    for sensor, seq in TEST:
        rows = {}
        with np.load(COMMON / f"s1/test_selected_symmetric_pbev128_main_{sensor}_{seq}.npz") as ref:
            common_queries = ref["query_idx"]
        for variant in ("ring", "ringpp"):
            with np.load(REPO / f"results/pdv2_ring_official/odom/dump_{variant}_{sensor}_{seq}.npz") as z:
                ix = np.searchsorted(z["query_idx"], common_queries)
                rows[variant] = {k: z[k][ix] for k in ("score_corr", "top1_geo_dist", "min_past_gt_dist")}
        pos = rows["ring"]["min_past_gt_dist"] <= 5
        neg = rows["ring"]["min_past_gt_dist"] > 25
        both_correct = pos & (rows["ring"]["top1_geo_dist"] <= 5) & (rows["ringpp"]["top1_geo_dist"] <= 5)
        report = {"both_correct_positive_queries": int(both_correct.sum()), "negative_queries": int(neg.sum())}
        for variant in ("ring", "ringpp"):
            score = rows[variant]["score_corr"]
            report[variant] = {"fixed_common_correct_vs_negative_score_AUC": pairwise_auc(score[both_correct], score[neg]),
                               "correct_score_quantiles": np.quantile(score[both_correct], [0, .5, 1]).tolist() if both_correct.any() else [],
                               "negative_score_quantiles": np.quantile(score[neg], [0, .5, 1]).tolist() if neg.any() else []}
        per_sequence[f"{sensor}/{seq}"] = report
    groups = {}
    for city, seqs in CITY.items():
        groups[city] = {variant: float(np.mean([
            per_sequence[s][variant]["fixed_common_correct_vs_negative_score_AUC"]
            for s in seqs if per_sequence[s][variant]["fixed_common_correct_vs_negative_score_AUC"] is not None]))
            for variant in ("ring", "ringpp")}
    p = base["ring"]["per_sequence"]
    q = base["ringpp"]["per_sequence"]
    result = {"scope": "현재 주표 195726 공통 질의 재검산 및 관찰 진단; 설정 변경 없음",
              "note": "양쪽 모두 5m 이내를 찾은 같은 양성 질의와 같은 음성 질의의 점수만 비교하여 검색 실패 효과를 제거한다. 평균 AUC는 주표 AP와 다른 진단 지표다.",
              "current_protocol": base, "score_separation": per_sequence,
              "score_separation_city_macro": groups,
              "ringpp_better_R1_but_worse_AP_sequences": [s for s in p if p[s]["n_positive"] and q[s]["R@1"] > p[s]["R@1"] and q[s]["AP"] < p[s]["AP"]],
              "input_sha256": sources}
    (OUT / "cause_diagnostics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"groups": groups, "R1_up_AP_down": result["ringpp_better_R1_but_worse_AP_sequences"]}, indent=2))


if __name__ == "__main__":
    main()
