#!/usr/bin/env python3
"""공통46 완료 뒤 검색 성공을 고정하고 세 RING 점수의 분리 능력을 진단한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata

from _pdv2_main_baselines_official import COMMON
from _pdv2_pilot import CITY, TEST
from _review_native_ring_diagnostics import pairwise_auc

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_ringpp_maxfix_20261105"
NATIVE = REPO / "results/pdv2_review_native_20261105"
METHODS = {
    "ring": ("pdv2_ring_official", "ring"),
    "ringpp_original": ("pdv2_ring_official", "ringpp"),
    "ringpp_maxfix": ("pdv2_ringpp_maxfix_20261105", "ringpp"),
}


def sha256(path: Path) -> str:
    """검산한 원장의 전체 파일 해시를 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """전체 완료 원장만 사용하며 부분 평균·설정 선택을 하지 않는다."""
    summary = json.loads((OUT / "summary_verified.json").read_text())
    geometry = json.loads((OUT / "geometry_independent_verification.json").read_text())
    assert summary["checks_passed"] and geometry["passed"] and geometry["complete"]
    assert geometry["sequences"] == 46 and geometry["common_queries"] == 195726
    sources, sequence_rows = {}, {}
    for sensor, seq in TEST:
        reference = COMMON / f"s1/test_selected_symmetric_pbev128_main_{sensor}_{seq}.npz"
        with np.load(reference) as z:
            query = z["query_idx"]
        sources[str(reference.relative_to(REPO))] = sha256(reference)
        rows = {}
        for name, (folder, prefix) in METHODS.items():
            path = REPO / "results" / folder / "odom" / f"dump_{prefix}_{sensor}_{seq}.npz"
            with np.load(path) as z:
                index = np.searchsorted(z["query_idx"], query)
                np.testing.assert_array_equal(z["query_idx"][index], query)
                rows[name] = {k: z[k][index] for k in ["score_corr", "min_past_gt_dist", "top1_geo_dist"]}
            sources[str(path.relative_to(REPO))] = sha256(path)
            assert np.isfinite(rows[name]["score_corr"]).all()
        distance = rows["ring"]["min_past_gt_dist"]
        for row in rows.values():
            np.testing.assert_array_equal(row["min_past_gt_dist"], distance)
        positive, negative = distance <= 5, distance > 25
        all_correct = positive.copy()
        for row in rows.values():
            all_correct &= row["top1_geo_dist"] <= 5
        metrics = {}
        for name, row in rows.items():
            p, n = row["score_corr"][all_correct], row["score_corr"][negative]
            auc = pairwise_auc(p, n)
            rank_auc = None
            if len(p) and len(n):
                ranks = rankdata(np.r_[p, n], method="average")
                rank_auc = float((ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))
                assert abs(rank_auc - auc) < 1e-12
            metrics[name] = {"conditional_score_AUC": auc, "independent_rank_AUC": rank_auc,
                             "positive_quantiles": np.quantile(p, [0, .5, 1]).tolist() if len(p) else [],
                             "negative_quantiles": np.quantile(n, [0, .5, 1]).tolist() if len(n) else []}
        sequence_rows[f"{sensor}/{seq}"] = {"common_queries": len(query),
            "all_three_correct_positive_queries": int(all_correct.sum()),
            "same_negative_queries": int(negative.sum()), "metrics": metrics}
    assert sum(r["common_queries"] for r in sequence_rows.values()) == 195726
    cities = {}
    for city, seqs in CITY.items():
        eligible = [s for s in seqs if sequence_rows[s]["metrics"]["ring"]["conditional_score_AUC"] is not None]
        assert eligible
        cities[city] = {"included_sequences": eligible, "mean_AUC": {
            method: float(np.mean([sequence_rows[s]["metrics"][method]["conditional_score_AUC"] for s in eligible]))
            for method in METHODS}}
    macro = {method: float(np.mean([c["mean_AUC"][method] for c in cities.values()])) for method in METHODS}
    report = {"passed": True, "created_unix": time.time(), "sequences": 46,
              "common_queries": 195726, "city_means": cities, "city_macro_conditional_AUC": macro,
              "sequence_rows": sequence_rows, "input_sha256": sources,
              "implementation_sha256": sha256(Path(__file__)),
              "scope": "세 방법 모두 맞힌 같은5m양성질의와 같은>25m음성질의만 사용한 관찰 진단. 실패양성을 제외하므로 주표AP/AUC가 아니고 단독인과효과도 아니다. 정답·후보·점수식·체크포인트 변경 없음."}
    (OUT / "score_separation_diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    text = "\\paragraph{Conditional discrimination after retrieval succeeds.}\n"
    text += "On the common benchmark, we retain exactly the positive queries correctly retrieved by all three RING conditions and the same negative queries. The six-city mean pairwise score AUC on this fixed subset is "
    text += f"{100*macro['ring']:.2f}\\% for RING, {100*macro['ringpp_original']:.2f}\\% for retained RING++, and {100*macro['ringpp_maxfix']:.2f}\\% for exact-max RING++. "
    text += "Ties receive half credit, and independent rank-sum recomputation agrees within $10^{-12}$. This conditional diagnostic separates score discrimination from retrieval failures on the chosen subset; it is not the main AP metric or a new calibration rule.\n"
    (NATIVE / "paper_proposal/ring_score_separation_addendum.tex").write_text(text)
    print(json.dumps({"passed": True, "city_macro_conditional_AUC": macro}))


if __name__ == "__main__":
    main()
