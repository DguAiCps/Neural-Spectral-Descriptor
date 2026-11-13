#!/usr/bin/env python3
"""완료된 개선 후보와 두 투영 대조의 최종 성능·AP 경로를 비교한다."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal_focus as exp
from _pdv2_temporal_focus_diagnose import ap_parts, cohort_masks, CATEGORIES, load
from _pdv2_no_context_ablation import sha256, write_json

CAP = REPO / "results/pdv2_temporal_capacity_20261018"


def main(primary_only: bool = False) -> None:
    """선택 후 재조정 없이 계열 평균·시드 차이·AP 분해를 계산한다."""
    exp.protocol(True)
    assert exp.read(exp.OUT / "completion_validation.json")["status"] == "passed"
    if not primary_only:
        assert exp.read(CAP / "completion_validation.json")["status"] == "passed"
    summary = exp.read(exp.OUT / "summary.json")
    old = exp.read(exp.old.OUT / "summary.json")
    capacity = None if primary_only else exp.read(CAP / "summary.json")
    winner = summary["expansion"]["winner"]
    assert winner == "standard_original", "현재 후속 대조의 고정 선택 조건을 확인한다"
    data, metrics, sources, records, fixed = {}, {}, {}, [], {}
    for seed in (1, 2, 3):
        models = {
            "diff": (exp.old.OUT, "temporal", old["per_seed"][str(seed)]["temporal"]),
            "point": (exp.old.OUT, "point", old["per_seed"][str(seed)]["point"]),
            "standard": (exp.OUT, winner, summary["per_seed"][str(seed)][winner]),
        }
        if capacity is not None:
            models["capacity_point"] = (CAP, "point_original", capacity["per_seed"][str(seed)]["point_original"])
        conf = exp.read(exp.base.BASE / f"s{seed}/confidence_selection.json")["ops"]
        metrics[str(seed)], data[str(seed)] = {}, {}
        fixed[str(seed)] = {}
        for name, (root, arm, chosen) in models.items():
            metrics[str(seed)][name] = {op: chosen["ops"][op]["macro"] for op in exp.base.OPS}
            for op in exp.base.OPS:
                records.append({"seed": seed, "method": name, "epoch": chosen["epoch"], "op": op,
                    **{m: chosen["ops"][op]["macro"][m]*100 for m in exp.base.PRIMARY}})
        for before in (name for name in models if name != "standard"):
            data[str(seed)][before + "_to_standard"] = {}
            fixed[str(seed)][before + "_to_standard"] = {}
            for ep in (10, 20):
                a = next(c for c in models[before][2]["candidates"] if c["epoch"] == ep)
                b = next(c for c in models["standard"][2]["candidates"] if c["epoch"] == ep)
                fixed[str(seed)][before + "_to_standard"][str(ep)] = {
                    op: {m: (b["ops"][op]["macro"][m]-a["ops"][op]["macro"][m])*100
                         for m in exp.base.PRIMARY} for op in exp.base.OPS}
            for op in exp.base.OPS:
                rows, counts = {}, {k: 0 for k in CATEGORIES}
                false_accepts = {"before": 0, "after": 0, "negative_queries": 0}
                directions = {str(g): {"n": 0, "before_correct": 0, "after_correct": 0} for g in (0, 1, 2)}
                for sensor, seq in exp.base.VAL:
                    ledgers = []
                    for name in (before, "standard"):
                        root, arm, chosen = models[name]
                        path = root / f"s{seed}/{arm}/epoch{chosen['epoch']}/val_{op}_{sensor}_{seq}.npz"
                        ledgers.append(load(path))
                        sources[str(path.relative_to(REPO))] = sha256(path)
                    a, b = ledgers
                    for k in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(a[k], b[k])
                    for k, mask in zip(CATEGORIES, cohort_masks(a, b)):
                        counts[k] += int(mask.sum())
                    negative = a["min_past_gt_dist"] > 25
                    false_accepts["negative_queries"] += int(negative.sum())
                    for field, ledger in (("before", a), ("after", b)):
                        false_accepts[field] += int((negative & (ledger[conf[op]["decision"]] >= conf[op]["tau_star"])).sum())
                    for group in (0, 1, 2):
                        mask = (a["min_past_gt_dist"] <= 5) & (a["group"] == group)
                        directions[str(group)]["n"] += int(mask.sum())
                        directions[str(group)]["before_correct"] += int((mask & (a["top1_geo_dist"] <= 5)).sum())
                        directions[str(group)]["after_correct"] += int((mask & (b["top1_geo_dist"] <= 5)).sum())
                    r = ap_parts(a, b, conf[op]["decision"])
                    r["family"] = "kitti" if sensor == "kitti360" else sensor
                    rows[sensor + "/" + seq] = r
                families = {f: {k: float(np.mean([r[k] for r in rows.values() if r["n_pos"] and r["family"] == f]))
                               for k in ("before", "after", *CATEGORIES)} for f in ("kitti", "helipr", "mulran")}
                macro = {k: float(np.mean([v[k] for v in families.values()])) for k in ("before", "after", *CATEGORIES)}
                for name, field in ((before, "before"), ("standard", "after")):
                    assert abs(macro[field]-metrics[str(seed)][name][op]["AP"]) < 1e-12
                data[str(seed)][before + "_to_standard"][op] = {"macro": macro, "families": families, "counts": counts,
                    "fixed_tau_false_accepts": false_accepts, "directions": directions}
    means = {name: {op: {m: {"mean": float(np.mean([metrics[str(s)][name][op][m] for s in (1, 2, 3)])),
                              "std": float(np.std([metrics[str(s)][name][op][m] for s in (1, 2, 3)], ddof=1))}
                        for m in exp.base.PRIMARY} for op in exp.base.OPS} for name in models}
    table_name = "primary_three_seed_metrics.csv" if primary_only else "all_three_seed_metrics.csv"
    with (exp.OUT / "reports" / table_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    used = [Path(__file__), exp.OUT / "summary.json", exp.old.OUT / "summary.json"]
    if not primary_only:
        used.append(CAP / "summary.json")
    sources.update({str(p.relative_to(REPO)): sha256(p) for p in used})
    name = "primary_post_analysis.json" if primary_only else "post_analysis.json"
    write_json(exp.OUT / "reports" / name, {"mean_std": means, "per_seed": metrics,
        "ap_decomposition": data, "fixed_epoch_effects_pp": fixed, "sources": sources, "test_used": False,
        "scope": "동일 기존 표본·3시드·검증9의 최종 판정 비교. AP 몫은 회계 분해. 반복노출 검증이며 독립시험·유의성확증 아님."})
    print("POST_ANALYSIS", means, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-only", action="store_true")
    main(parser.parse_args().primary_only)
