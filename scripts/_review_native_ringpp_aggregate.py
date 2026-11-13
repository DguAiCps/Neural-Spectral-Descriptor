#!/usr/bin/env python3
"""새 RING++ 원장을 공통 질의로 집계하고 독립 산식으로 검산한다."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _review_native_pdv2_ringpp import OUT, check_protocol, sha256, write_json
from _pdv2_main_baselines_official import COMMON, METRICS, aggregate
from _pdv2_pilot import CITY, TEST
from _pr_eval_lib import AGG_METRICS, classify_queries, pr_sweep


def independent(scores: np.ndarray, distances: np.ndarray, minimum: np.ndarray) -> dict:
    """점수별 정수 빈도에서 AP·F1·고정정밀도 재현율을 별도로 계산한다."""
    positive = minimum <= 5
    eligible = positive | (minimum > 25)
    good = distances <= 5
    finite = eligible & np.isfinite(scores)
    values, group = np.unique(scores[finite], return_inverse=True)
    total = np.bincount(group, minlength=len(values))[::-1]
    correct = np.bincount(group, weights=(positive & good)[finite].astype(np.int64), minlength=len(values))[::-1]
    accepted = np.r_[0, total.cumsum()]
    true = np.r_[0, correct.cumsum()]
    precision = np.divide(true, accepted, out=np.ones(len(true)), where=accepted != 0)
    recall = true / positive.sum()
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(true)), where=precision + recall != 0)
    return {"AP": float(np.dot(np.diff(recall), envelope[1:])), "max_F1": float(f1.max()),
            **{name: float(recall[precision >= p].max()) for name, p in
               [("R@100P", 1), ("R@99P", .99), ("R@95P", .95)]},
            "R@1": float(good[positive].mean())}


def main() -> None:
    """완료 시퀀스의 검증 기록을 남기고46개일 때만 최종 macro를 만든다."""
    check_protocol(aggregation=True)
    per, full, provenance = {}, {}, {}
    original_summary = json.loads((REPO / "results/pdv2_main_official_baselines/baselines_common.json").read_text())
    original = original_summary["methods"]["ringpp"]["per_sequence"]
    for sensor, seq in TEST:
        name = f"{sensor}/{seq}"
        path = OUT / "odom" / f"dump_ringpp_{sensor}_{seq}.npz"
        if not path.exists():
            continue
        ref = COMMON / f"s1/test_selected_symmetric_pbev128_main_{sensor}_{seq}.npz"
        with np.load(path) as z, np.load(ref) as n:
            qc = classify_queries(z["min_past_gt_dist"])
            full[name] = {"n_queries": len(z["query_idx"]), "n_positive": qc["n_pos"]}
            if qc["n_pos"]:
                sw = pr_sweep(z["score_corr"], z["top1_geo_dist"], qc)
                full[name].update({k: float(sw[k]) for k in AGG_METRICS})
            ix = np.searchsorted(z["query_idx"], n["query_idx"])
            np.testing.assert_array_equal(z["query_idx"][ix], n["query_idx"])
            np.testing.assert_array_equal(z["min_past_gt_dist"][ix], n["min_past_gt_dist"])
            assert len(ix) == len(z["query_idx"]) - 15
            assert np.all(z["j_star"][ix] <= n["query_idx"] - 30)
            assert np.all(np.isfinite(z["score_corr"][ix]))
            qc = classify_queries(z["min_past_gt_dist"][ix])
            row = {"n_queries": len(ix), "n_positive": qc["n_pos"], "n_negative": qc["n_neg"],
                   "n_ambiguous": qc["n_ambiguous"], "ledger_sha256": sha256(path)}
            if qc["n_pos"]:
                sw = pr_sweep(z["score_corr"][ix], z["top1_geo_dist"][ix], qc)
                metrics = {k: float(sw[k]) for k in AGG_METRICS}
                metrics["R@1"] = float((z["top1_geo_dist"][ix][qc["positive"]] <= 5).mean())
                separate = independent(z["score_corr"][ix], z["top1_geo_dist"][ix], z["min_past_gt_dist"][ix])
                for k in METRICS:
                    assert abs(metrics[k] - separate[k]) < 1e-12, (name, k)
                row.update(metrics)
                row["independent"] = separate
                row["delta_previous_pp"] = {k: 100 * (metrics[k] - original[name][k]) for k in METRICS}
            else:
                row.update({k: None for k in METRICS})
            row["checks_passed"] = True
            per[name] = row
            provenance[str(ref.relative_to(REPO))] = sha256(ref)
    progress = {"completed_sequences": len(per), "total_sequences": 46,
                "completed_common_queries": sum(v["n_queries"] for v in per.values()),
                "completed_common_positives": sum(v["n_positive"] for v in per.values()),
                "per_sequence": per, "complete": len(per) == 46,
                "interpretation": "완료 부분집합 진행 기록; 전체46 macro는 전부 완료할 때만 계산",
                "protocol_sha256": sha256(OUT / "protocol.json")}
    write_json(OUT / "partial_verified.json", progress)
    write_json(OUT / "odom/pr_summary_ringpp_maxfix.json", {"ringpp": {"per_sequence": full}})
    if len(per) == 46:
        def track(p: Path) -> Path:
            provenance[str(p.relative_to(REPO))] = sha256(p)
            return p
        entry = aggregate(str((OUT / "odom").relative_to(REPO)), "dump_ringpp", "score_corr",
                          "pr_summary_ringpp_maxfix.json", "ringpp", track)
        independent_groups = {city: {k: float(np.mean([per[s][k] for s in names if per[s]["n_positive"]]))
                                    for k in METRICS} for city, names in CITY.items()}
        for key in METRICS:
            assert abs(entry["macro"][key] - np.mean([g[key] for g in independent_groups.values()])) < 1e-12
        write_json(OUT / "summary_verified.json", {"ringpp_maxfix": entry, "checks_passed": True,
                   "previous_ringpp": original_summary["methods"]["ringpp"]["macro"],
                   "delta_previous_pp": {k: 100 * (entry["macro"][k] - original_summary["methods"]["ringpp"]["macro"][k]) for k in METRICS},
                   "input_sha256": provenance, "protocol_sha256": sha256(OUT / "protocol.json")})
    print(json.dumps({k: v for k, v in progress.items() if k != "per_sequence"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
