#!/usr/bin/env python3
"""다방면 문맥 실험의 최종 성능과 요인별 효과를 한 파일로 정리한다."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_context_repair as fixed
import _pdv2_context_repair_train as learned
from _pdv2_no_context_ablation import sha256, write_json


def training_transitions() -> dict:
    """자료 계열·방향별 최종 정답 변화와 오수락을 같은 질의에서 센다."""
    selection = fixed.base.read(fixed.base.old.BASE / "s1/confidence_selection.json")["ops"]
    results = {}
    for arm in learned.ARMS:
        after = learned.OUT / (arm + "_all")
        references = {"initial_temporal": fixed.root("temporal"),
                      "own_temporal": learned.OUT / (arm + "_temporal")}
        if arm != "control":
            references["trained_control"] = learned.OUT / "control_all"
        for reference, before in references.items():
            comparison = results.setdefault(arm + "_vs_" + reference, {})
            for op in fixed.OPS:
                families = comparison.setdefault(op, {})
                sel = selection[op]
                for s, q in fixed.VAL:
                    a, _ = fixed.base.old.read_ledger(before / f"val_{op}_{s}_{q}.npz")
                    b, _ = fixed.base.old.read_ledger(after / f"val_{op}_{s}_{q}.npz")
                    for key in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(a[key], b[key])
                    family = "kitti" if s.startswith("kitti") else s
                    groups = families.setdefault(family, {})
                    correct_a, correct_b = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
                    for name, mask in (("positive", a["min_past_gt_dist"] <= 5),
                                       ("same", a["group"] == 0),
                                       ("opposite", a["group"] == 1)):
                        lost = mask & correct_a & ~correct_b
                        gained = mask & ~correct_a & correct_b
                        stats = {"queries": int(mask.sum()),
                                 "before_correct": int((mask & correct_a).sum()),
                                 "after_correct": int((mask & correct_b).sum()),
                                 "recovered": int(gained.sum()), "regressed": int(lost.sum()),
                                 "regressed_union_missing": int((lost & ~b["union_hit"]).sum())}
                        assert stats["after_correct"] - stats["before_correct"] == stats["recovered"] - stats["regressed"]
                        target = groups.setdefault(name, {k: 0 for k in stats})
                        for key, value in stats.items():
                            target[key] += value
                    negative = a["min_past_gt_dist"] > 25
                    stats = {"queries": int(negative.sum()),
                             "before_false_accepts": int((negative & (a[sel["decision"]] >= sel["tau_star"])).sum()),
                             "after_false_accepts": int((negative & (b[sel["decision"]] >= sel["tau_star"])).sum()),
                             "changed_candidate": int((negative & (a["j_star"] != b["j_star"])).sum())}
                    target = groups.setdefault("negative", {k: 0 for k in stats})
                    for key, value in stats.items():
                        target[key] += value
    return results


def report() -> None:
    """설정을 선택하지 않고 모든 조건·요인 효과를 보고한다."""
    fs = fixed.base.read(fixed.OUT / "summary.json")
    ts = fixed.base.read(learned.OUT / "summary.json")
    audits = {"fixed": fixed.base.read(fixed.OUT / "completion_validation.json"),
              "training": fixed.base.read(learned.OUT / "completion_validation.json")}
    for x in audits.values():
        assert x["status"] == "passed_with_inherited_gpu_reproducibility_limit"
    scores = {**fs["scores"], **ts["scores"]}
    effects = {}
    for op in fixed.OPS:
        effects[op] = {}
        for metric in fixed.base.PRIMARY:
            value = {a: ts["scores"][a + "_all"][op]["macro"][metric] for a in learned.ARMS}
            effects[op][metric] = {
                "opposite_minus_control": value["opposite"] - value["control"],
                "dropout_minus_control": value["dropout"] - value["control"],
                "combined_minus_control": value["combined"] - value["control"],
                "interaction": value["combined"] - value["opposite"] - value["dropout"] + value["control"],
                "similarity_by_model": {a: value[a] - ts["scores"][a + "_temporal"][op]["macro"][metric] for a in learned.ARMS}}
    quality = {}
    for op in fixed.OPS:
        quality[op] = {}
        for arm in fixed.GRAPH_ARMS:
            groups = {name: {"queries": 0, "connected": 0, "true_connected": 0, "edges": 0, "true_edges": 0, "far_edges": 0}
                      for name in ("same", "opposite", "other", "negative", "ambiguous")}
            for s, q in fixed.VAL:
                with np.load(fixed.base.OUT / "main" / f"edges_{s}_{q}.npz") as z:
                    query, distance, valid, group = z["query"], z["distance_3d"], z["valid"], z["group"]
                with np.load(fixed.OUT / f"selected_{s}_{q}.npz") as z:
                    ids = z[op + "__" + arm]
                for i, name in enumerate(groups):
                    mask = valid & (group == i)
                    keep = ids[mask[query[ids]]]
                    g = groups[name]
                    g["queries"] += int(mask.sum())
                    g["connected"] += len(np.unique(query[keep]))
                    g["true_connected"] += len(np.unique(query[keep][distance[keep] <= 5]))
                    g["edges"] += len(keep)
                    g["true_edges"] += int((distance[keep] <= 5).sum())
                    g["far_edges"] += int((distance[keep] > 25).sum())
            quality[op][arm] = groups
    paths = [fixed.OUT / "summary.json", learned.OUT / "summary.json",
             fixed.OUT / "completion_validation.json", learned.OUT / "completion_validation.json",
             fixed.OUT / "fusion_pool_validation.json"]
    out = {"scores": scores, "training_factorial_effects": effects,
           "training_transitions": training_transitions(),
           "fixed_edge_quality": quality, "test_used": False,
           "scope": "단일 시드 검증9. 모든 사전고정 조건 보고. 새로운 모델 선택·시험 적용 없음.",
           "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in paths},
           "script_sha256": sha256(Path(__file__))}
    write_json(fixed.OUT / "combined_summary.json", out)
    with (fixed.OUT / "combined_comparison.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["arm", "op", *fixed.base.PRIMARY, "AP_vs_initial_temporal_pp",
                    "AP_kitti", "AP_helipr", "AP_mulran"])
        for arm, ops in scores.items():
            for op, value in ops.items():
                m = value["macro"]
                w.writerow([arm, op, *[m[k] for k in fixed.base.PRIMARY],
                            100 * (m["AP"] - scores["temporal"][op]["macro"]["AP"]),
                            *[value["by_group"][g]["AP"] for g in ("kitti", "helipr", "mulran")]])
    plot(out)
    print("COMBINED", len(scores), flush=True)


def plot(result: dict) -> None:
    """현재 시간 간선 기준과 각 추가학습 모델의 간선 효과를 그린다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    labels = {"A": "A: 3D classifier", "C": "C: alignment features", "C_top1": "C: top 1",
              "C_top3": "C: top 3", "C_cut": "C: train threshold", "C_message": "C: message scaling",
              "C_half_key": "C: half key weight", "C_zero_key": "C: zero key weight",
              "control_all": "Fine-tune control", "opposite_all": "Opposite pairs",
              "dropout_all": "Similarity dropout", "combined_all": "Opposite + dropout"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    for ax, op, title in zip(axes, fixed.OPS, ("NSD", "NSD-H")):
        s = result["scores"]
        ref = s["temporal"][op]["macro"]["AP"]
        delta = [100 * (s[a][op]["macro"]["AP"] - ref) for a in labels]
        bars = ax.barh(np.arange(len(labels)), delta, color=["#4b8eaf" if not a.endswith("_all") else "#bc8241" for a in labels])
        ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=8)
        ax.set_yticks(np.arange(len(labels)), list(labels.values()))
        ax.axvline(0, color="black", lw=.7)
        ax.grid(axis="x", alpha=.2)
        ax.set_title(title)
        ax.set_xlabel("Final AP change vs initial temporal-only (pp)")
        ax.margins(x=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Validation 9 sequences, seed 1, sequence-to-family macro")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fixed.OUT / ("combined_ap." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    names = ["Fine-tune control", "Opposite pairs", "Similarity dropout", "Opposite + dropout"]
    for ax, op, title in zip(axes, fixed.OPS, ("NSD", "NSD-H")):
        effects = result["training_factorial_effects"][op]["AP"]["similarity_by_model"]
        values = [100 * effects[a] for a in learned.ARMS]
        bars = ax.barh(np.arange(len(names)), values,
                       color=["#36866b" if x > 0 else "#b96754" for x in values])
        ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=9)
        ax.set_yticks(np.arange(len(names)), names)
        ax.axvline(0, color="black", lw=.7)
        ax.grid(axis="x", alpha=.2)
        ax.set_title(title)
        ax.set_xlabel("Final AP: similarity + temporal minus temporal only (pp)")
        ax.margins(x=.35)
    axes[0].invert_yaxis()
    fig.suptitle("Same checkpoint and retrieval pipeline within each pair; validation, seed 1")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fixed.OUT / ("training_similarity_effect." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    report()
