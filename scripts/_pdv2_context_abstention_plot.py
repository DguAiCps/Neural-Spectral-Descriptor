#!/usr/bin/env python3
"""연결 거절의 자료별 효과·방향 보존과 최종 성능을 그림으로 남긴다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_context_abstention as exp
from _pdv2_no_context_ablation import sha256, write_json


def calibration_pooling() -> dict:
    """합산95% 보존이 소수 자료를 모두 버릴 수 있는지 학습에서만 진단한다."""
    result = {}
    for model in exp.MODELS:
        meta = exp.base.read(exp.OUT / (model + "_thresholds.json"))
        values = {op: {name: [] for name in ("classifier", "alignment")} for op in exp.OPS}
        for source in meta["sources"]:
            path = REPO / source["path"]
            assert sha256(path) == source["sha256"]
            with np.load(path) as z:
                query, prob, eligible = z["query"], z["probability"], z["eligible"]
                for op in exp.OPS:
                    align = z["extra_" + op][:, exp.CFG["alignment_channel"][op]]
                    _, a, b = exp.query_signals(query, prob, align, len(eligible))
                    values[op]["classifier"].append(a[eligible])
                    values[op]["alignment"].append(b[eligible])
        result[model] = {}
        for op, signals in values.items():
            result[model][op] = {}
            for name, arrays in signals.items():
                pooled = exp.retention_threshold(np.concatenate(arrays), .95)
                applied = meta["thresholds"][op][name]
                result[model][op][name] = {"pooled_threshold_diagnostic_only": pooled,
                    "applied_threshold": applied, "per_sequence": [
                        {"sequence": sequence, "positive_queries": len(a),
                         "pooled_retained": int((a >= pooled).sum()),
                         "applied_retained": int((a >= applied).sum())}
                        for sequence, a in zip(exp.CFG["calibration_sequences"], arrays)]}
    return result


def summarize() -> dict:
    """기존 전체 간선 대비 손익을 원장의 같은 질의에서 계산한다."""
    s = exp.base.read(exp.OUT / "summary.json")
    assert exp.base.read(exp.OUT / "completion_validation.json")["summary_sha256"] == sha256(exp.OUT / "summary.json")
    pooled, transitions, quality_checks = {}, {}, []
    for model in exp.MODELS:
        for policy in exp.POLICIES:
            label = model + "__" + policy
            pooled[label], transitions[label] = {}, {}
            for op in exp.OPS:
                family_rows, family_transitions = {}, {}
                for sensor, sequence in exp.VAL:
                    family = "kitti" if sensor.startswith("kitti") else sensor
                    target = family_rows.setdefault(family, {})
                    name = f"val_{op}_{sensor}_{sequence}.npz"
                    a, _ = exp.base.old.read_ledger(exp.folder(model, "all") / name)
                    b, _ = exp.base.old.read_ledger(exp.folder(model, policy) / name)
                    np.testing.assert_array_equal(a["query_idx"], b["query_idx"])
                    # 원장의 group은 양성 방향만 분류한다. 비재방문·모호군은
                    # 최종 원장의 GT 거리로 별도 작성해야 하며 추론 입력은 아니다.
                    with np.load(exp.OUT / f"{model}_edges_{sensor}_{sequence}.npz") as z:
                        query, distance = z["query"], z["distance"]
                        valid, group = z["valid"], z["group"].copy()
                        ids = z["selected__" + op + "__" + policy]
                        all_ids = z["selected__" + op + "__all"]
                    qi, md = b["query_idx"], b["min_past_gt_dist"]
                    connected = np.zeros(len(valid), bool)
                    connected[query[ids]] = True
                    before_has_true = np.zeros(len(valid), bool)
                    before_has_true[query[all_ids[distance[all_ids] <= 5]]] = True
                    group[qi[md > 25]] = 3
                    group[qi[(md > 5) & (md <= 25)]] = 4
                    np.testing.assert_array_equal(np.flatnonzero(valid), qi)
                    assert np.isin(group[valid], np.arange(5)).all()
                    corrected = {}
                    for i, name_group in enumerate(("same", "opposite", "other", "negative", "ambiguous")):
                        mask = valid & (group == i)
                        ii = ids[mask[query[ids]]]
                        d = distance[ii]
                        corrected[name_group] = {"queries": int(mask.sum()),
                            "connected": len(np.unique(query[ii])),
                            "true_connected": len(np.unique(query[ii][d <= 5])),
                            "edges": len(ii), "true_edges": int((d <= 5).sum()),
                            "far_edges": int((d > 25).sum())}
                        row_check = corrected[name_group]
                        assert row_check["true_connected"] <= row_check["connected"] <= row_check["queries"]
                        if name_group == "negative":
                            assert row_check["true_edges"] == 0 and row_check["far_edges"] == row_check["edges"]
                        if name_group == "ambiguous":
                            assert row_check["true_edges"] == 0
                    assert sum(x["queries"] for x in corrected.values()) == len(qi)
                    assert corrected["negative"]["queries"] == int((md > 25).sum())
                    assert corrected["ambiguous"]["queries"] == int(((md > 5) & (md <= 25)).sum())
                    s["edge_quality"][label][op][sensor + "/" + sequence] = corrected
                    quality_checks.append({"model": model, "policy": policy, "op": op,
                        "sequence": sensor + "/" + sequence, "queries": len(qi)})
                    for group_name, values in corrected.items():
                        row = target.setdefault(group_name, {k: 0 for k in values})
                        for key, value in values.items():
                            row[key] += value
                    t = family_transitions.setdefault(family, {})
                    for group, mask in (("positive", a["min_past_gt_dist"] <= 5),
                                        ("same", a["group"] == 0), ("opposite", a["group"] == 1)):
                        x, y = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
                        lost, gained = mask & x & ~y, mask & ~x & y
                        values = {"queries": int(mask.sum()), "recovered": int(gained.sum()),
                                  "regressed": int(lost.sum()), "correct_before": int((mask & x).sum()),
                                  "correct_after": int((mask & y).sum()),
                                  "lost_union_missing": int((lost & ~b["union_hit"]).sum()),
                                  "regressed_own_query_disconnected": int((lost & ~connected[qi]).sum()),
                                  "regressed_lost_true_connection": int((lost & ~connected[qi] & before_has_true[qi]).sum())}
                        row = t.setdefault(group, {k: 0 for k in values})
                        for key, value in values.items():
                            row[key] += value
                pooled[label][op], transitions[label][op] = family_rows, family_transitions
    correction = {"reason": "원장의 양성 방향 group을 그대로 복사한 초기 연결품질 집계에서 비재방문·모호군이 누락됐다. 최종 원장 GT 거리로 이 두 군만 재분류했다. 정책·하한·키·최종지표에는 영향이 없다.",
                  "scope": "summary.json 원본 보존. 현재 연결품질은 final_summary.json/analysis.json을 사용한다.",
                  "source_sha256": sha256(exp.OUT / "summary.json"),
                  "script_sha256": sha256(Path(__file__)), "checks": quality_checks}
    s["diagnostic_correction"] = correction
    s["calibration_pooling_diagnostic"] = calibration_pooling()
    write_json(exp.OUT / "final_summary.json", s)
    write_json(exp.OUT / "quality_correction.json", correction)
    result = {"pooled_edge_quality": pooled, "transitions_vs_all": transitions,
              "source_sha256": sha256(exp.OUT / "summary.json"), "script_sha256": sha256(Path(__file__)),
              "final_summary_sha256": sha256(exp.OUT / "final_summary.json"),
              "test_used": False}
    write_json(exp.OUT / "analysis.json", result)
    return s


def plot(s: dict) -> None:
    """최종 AP와 같은 모델의 유사 간선 효과를 내보낸다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    names = {"all": "A: top 10", "top1": "A: top 1", "classifier_gate": "Classifier gate",
             "alignment_gate": "Alignment gate", "alignment_gate_top1": "Alignment gate + top 1"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for row, model in enumerate(exp.MODELS):
        for col, op in enumerate(exp.OPS):
            ax = axes[row, col]
            ref = s["scores"][model + "__temporal"][op]["macro"]["AP"]
            delta = [100 * (s["scores"][model + "__" + p][op]["macro"]["AP"] - ref) for p in names]
            bars = ax.barh(np.arange(len(names)), delta,
                           color=["#36866b" if v > 0 else "#b96754" for v in delta])
            ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=9)
            ax.set_yticks(np.arange(len(names)), list(names.values()))
            ax.invert_yaxis()
            ax.axvline(0, color="black", lw=.7)
            ax.grid(axis="x", alpha=.2)
            ax.margins(x=.4)
            ax.set_title(("Control" if model == "control" else "Opposite pairs") + " / " + ("NSD" if col == 0 else "NSD-H"))
            ax.set_xlabel("Final AP change vs same-model temporal-only (pp)")
    fig.suptitle("Validation 9 sequences, seed 1; thresholds fixed on training sequences")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(exp.OUT / ("final_ap." + extension), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    shared_limit = max(abs(100 * (s["scores"]["opposite__" + p][op]["by_group"][g]["AP"] - s["scores"]["opposite__temporal"][op]["by_group"][g]["AP"]))
                       for op in exp.OPS for p in names for g in ("kitti", "helipr", "mulran"))
    for ax, op, title in zip(axes, exp.OPS, ("NSD", "NSD-H")):
        values = np.array([[100 * (s["scores"]["opposite__" + p][op]["by_group"][g]["AP"] - s["scores"]["opposite__temporal"][op]["by_group"][g]["AP"])
                            for g in ("kitti", "helipr", "mulran")] for p in names])
        lim = max(shared_limit, .01)
        ax.imshow(values, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
        for i, j in np.ndindex(values.shape):
            ax.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(values[i, j]) > .75 * lim else "black")
        ax.set_xticks(range(3), ["KITTI family", "HeLiPR", "MulRan"])
        ax.set_yticks(range(len(names)), list(names.values()))
        ax.set_title(title)
    fig.suptitle("Opposite-pair model: final AP change vs temporal-only by dataset (pp)")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(exp.OUT / ("dataset_ap." + extension), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot(summarize())
