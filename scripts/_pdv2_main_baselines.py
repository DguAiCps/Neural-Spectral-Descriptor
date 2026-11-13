#!/usr/bin/env python3
"""주표 비교 방법의 고정 기록을 대칭 문맥과 같은 질의로 재집계한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_pilot import CITY, TEST
from _pr_eval_lib import AGG_METRICS, classify_queries, pr_sweep

OUT = REPO / "results/pdv2_main_lag15_20261014"
SOURCE = REPO / "results/pdv2_c3/odom"
COMMON = REPO / "results/pdv2_symmetric3_20261013"


def main() -> None:
    """원래 지표 재현과 질의 일치를 확인하고 동일 모집단의 지표를 쓴다."""
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}

    def track(path: Path) -> Path:
        sources[str(path.relative_to(REPO))] = sha256(path)
        return path

    track(Path(__file__))
    for name in ("_pr_eval_lib.py", "_pdv2_pilot.py", "_pdv2_no_context_ablation.py"):
        track(REPO / "scripts" / name)
    original = json.loads(track(SOURCE / "pr_summary_baselines.json").read_text())
    report = {"date": "2026-10-14", "scope": "고정된 SC++·RING#-L 시험 기록 재집계; 새 선택·추론 없음",
              "queries": "30 <= q < N-15", "candidates": "j <= q-30",
              "information_access": "비교 방법은 기존 현재·과거 입력을 유지한다. NSD의 앞보기 관측을 추가하지 않는다.",
              "methods": {}}
    metrics = (*AGG_METRICS, "R@1")
    for method, score_name in (("scpp", "score_scpp"), ("ringsharp", "score_trans")):
        per = {}
        for sensor, seq in TEST:
            ref = track(COMMON / f"s1/test_selected_symmetric_pbev128_main_{sensor}_{seq}.npz")
            path = track(SOURCE / f"dump_{method}_{sensor}_{seq}.npz")
            with np.load(path) as z, np.load(ref) as n:
                qc = classify_queries(z["min_past_gt_dist"])
                sw = pr_sweep(z[score_name], z["top1_geo_dist"], qc)
                old = original[method]["per_sequence"][f"{sensor}/{seq}"]
                for k in AGG_METRICS:
                    assert abs(sw[k] - old[k]) < 1e-12
                ix = np.searchsorted(z["query_idx"], n["query_idx"])
                np.testing.assert_array_equal(z["query_idx"][ix], n["query_idx"])
                np.testing.assert_array_equal(z["min_past_gt_dist"][ix], n["min_past_gt_dist"])
                assert len(ix) == len(z["query_idx"]) - 15
                assert np.all(z["j_star"][ix] <= n["query_idx"] - 30)
                qc = classify_queries(z["min_past_gt_dist"][ix])
                row = {"n_queries": len(ix), "n_positive": qc["n_pos"],
                       "n_negative": qc["n_neg"], "n_ambiguous": qc["n_ambiguous"]}
                if qc["n_pos"]:
                    sw = pr_sweep(z[score_name][ix], z["top1_geo_dist"][ix], qc)
                    row.update({k: float(sw[k]) for k in AGG_METRICS})
                    row["R@1"] = float((z["top1_geo_dist"][ix][qc["positive"]] <= 5).mean())
                else:
                    row.update({k: None for k in metrics})
                    row["negative_score_quantiles"] = np.quantile(
                        z[score_name][ix][qc["negative"]], [0, .5, .95, 1]).tolist()
                per[f"{sensor}/{seq}"] = row
        groups = {city: {k: float(np.mean([per[s][k] for s in names if per[s]["n_positive"]]))
                         for k in metrics} for city, names in CITY.items()}
        macro = {k: float(np.mean([g[k] for g in groups.values()])) for k in metrics}
        report["methods"][method] = {
            "per_sequence": per, "city_groups": groups, "macro": macro,
            "n_queries": sum(r["n_queries"] for r in per.values()),
            "n_positive": sum(r["n_positive"] for r in per.values()),
            "excluded_no_positive": [s for s, r in per.items() if not r["n_positive"]]}
        assert report["methods"][method]["n_queries"] == 195726
        assert report["methods"][method]["n_positive"] == 61682
        assert report["methods"][method]["excluded_no_positive"] == ["kitti/07"]
        print(method, macro, flush=True)
    report["input_sha256"] = sources
    report["checks_passed"] = True
    write_json(OUT / "baselines_common.json", report)


if __name__ == "__main__":
    main()
