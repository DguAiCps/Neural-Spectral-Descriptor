#!/usr/bin/env python3
"""원고 표 2의 고정 결과를 대칭 문맥과 같은 시험 질의로 재집계한다.

호스트에서 python3 scripts/_pdv2_lag15_paper_compare.py로 실행한다.
추론·학습·설정 선택 없이 저장된 최종 검색 기록만 읽는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_pilot import CITY, TEST
from _pr_eval_lib import classify_queries, pr_sweep

SOURCE = REPO / "results/pdv2_symmetric3_20261013"
OUT = REPO / "results/pdv2_paper2_common_queries_20261014"
METRICS = ("AP", "max_F1", "R@100P", "R@1")


def main() -> None:
    """원래 집계의 재현과 공통 질의의 일치를 검사해 비교를 저장한다."""
    # 원천 자료를 유지하고 재현 가능한 별도 집계 파일을 만든다.
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}

    def track(path: Path) -> Path:
        key = str(path.relative_to(REPO))
        if key not in sources:
            sources[key] = sha256(path)
        return path

    track(Path(__file__))
    for filename in ("_pr_eval_lib.py", "_pdv2_pilot.py", "_pdv2_no_context_ablation.py"):
        track(REPO / "scripts" / filename)
    new = json.loads(track(SOURCE / "summary.json").read_text())["stages"]["test_selected"]["ops"]
    result = {
        "date": "2026-10-14", "scope": "저장된 시험 결과의 사후 재집계. 새 추론·학습·설정 선택 없음.",
        "queries": "대칭 문맥 기록과 동일한 30 <= q < N-15; 후보 j <= q-30",
        "aggregation": "양성 없는 시퀀스 공통 제외, 시퀀스→6도시→3시드 평균. KITTI07 오수락은 별도 유지.",
        "interpretation": "과거 전용과 대칭 시스템은 각자의 검증 선택 설정을 유지한다. 문맥만의 기여 분리와 구분한다.",
        "ops": {},
    }
    for op in ("pbev128_main", "hspec12"):
        per_seed = {}
        for seed in (1, 2, 3):
            old = REPO / f"results/pdv2_c3_bevclip_s{seed}/odom"
            summary = json.loads(track(old / "pr_summary_align.json").read_text())[op]
            stat = summary["statistic"]
            per = {}
            counts = dict(old_queries=0, common_queries=0, old_positive=0, common_positive=0)
            excluded = {}
            for sensor, seq in TEST:
                name = f"{sensor}/{seq}"
                stem = f"pdv2_causal_{op}_s{seed}_{sensor}_{seq}.npz"
                p = track(old / f"dump_{stem}")
                ap = track(old / f"align_dump_{stem}")
                npth = track(SOURCE / f"s{seed}/test_selected_symmetric_{op}_{sensor}_{seq}.npz")
                with np.load(p) as z, np.load(ap) as a, np.load(npth) as n:
                    np.testing.assert_array_equal(z["query_idx"], a["query_idx"])
                    np.testing.assert_array_equal(z["j_star"], a["j_star"])
                    ix = np.searchsorted(z["query_idx"], n["query_idx"])
                    np.testing.assert_array_equal(z["query_idx"][ix], n["query_idx"])
                    np.testing.assert_array_equal(z["min_past_gt_dist"][ix], n["min_past_gt_dist"])
                    assert len(ix) == len(z["query_idx"]) - 15
                    assert np.all(z["j_star"] <= z["query_idx"] - 30)
                    qc = classify_queries(z["min_past_gt_dist"])
                    score = a[stat] if stat in ("c_spec", "c_col") else z[
                        {"absolute": "score_abs", "margin": "score_margin"}[stat]]
                    original = pr_sweep(score, z["top1_geo_dist"], qc)
                    for metric in METRICS[:3]:
                        assert abs(original[metric] - summary["per_sequence"][name][metric]) < 1e-12
                    original_r1 = float((z["top1_geo_dist"][qc["positive"]] <= 5).mean())
                    assert abs(original_r1 - summary["per_sequence"][name]["R@1_revisit"]) < 1e-12
                    counts["old_queries"] += len(z["query_idx"])
                    counts["common_queries"] += len(ix)
                    counts["old_positive"] += qc["n_pos"]
                    qc = classify_queries(z["min_past_gt_dist"][ix])
                    counts["common_positive"] += qc["n_pos"]
                    if not qc["n_pos"]:
                        excluded[name] = {
                            "original_counts": summary["per_sequence"][name]["counts"],
                            "common_negative": qc["n_neg"],
                            "common_ambiguous": qc["n_ambiguous"],
                            "negative_false_accepts": int((qc["negative"] &
                                (score[ix] >= summary["tau_star"])).sum()),
                        }
                        continue
                    sweep = pr_sweep(score[ix], z["top1_geo_dist"][ix], qc)
                    per[name] = {k: float(sweep[k]) for k in METRICS[:3]}
                    per[name]["R@1"] = float((z["top1_geo_dist"][ix][qc["positive"]] <= 5).mean())
            macro = {k: float(np.mean([
                np.mean([per[name][k] for name in names if name in per])
                for names in CITY.values()])) for k in METRICS}
            assert set(excluded) == {"kitti/07"}
            per_seed[str(seed)] = {
                "original": {k: summary["city_macro6"]["R@1_revisit" if k == "R@1" else k] for k in METRICS},
                "common": macro, "counts": counts, "excluded": excluded,
                "per_sequence_common": per, "decision": stat, "tau_star": summary["tau_star"],
            }
        rows = {kind: {k: float(np.mean([r[kind][k] for r in per_seed.values()]))
                       for k in METRICS} for kind in ("original", "common")}
        rows["symmetric"] = {k: new[op]["mean_std"]["symmetric"][k]["mean"] for k in METRICS}
        rows["symmetric_minus_past_common_pp"] = {
            k: 100 * (rows["symmetric"][k] - rows["common"][k]) for k in METRICS}
        result["ops"][op] = dict(rows=rows, per_seed=per_seed)
        print(op, json.dumps(rows), flush=True)
    result["checks"] = {"original_metrics_reproduced": True,
                        "query_indices_and_gt_labels_equal": True,
                        "past_candidate_constraint": True}
    result["input_sha256"] = sources
    write_json(OUT / "summary.json", result)


if __name__ == "__main__":
    main()
