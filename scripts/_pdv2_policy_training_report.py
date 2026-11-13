#!/usr/bin/env python3
"""학습·추론 교차 비교의 최종 수치와 자료별 효과를 그림으로 남긴다."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_policy_training as exp
from _pdv2_no_context_ablation import sha256, write_json


def verify() -> dict:
    """표의 평균·교차효과와 진단 출처를 독립 재검산한다."""
    p = exp.protocol(True)
    s = exp.base.read(exp.OUT / "summary.json")
    complete = exp.base.read(exp.OUT / "completion_validation.json")
    assert complete["summary_sha256"] == sha256(exp.OUT / "summary.json")
    assert len(s["scores"]) == 12 and len(complete["prefix"]["cases"]) == 96
    count = 0
    for ops in s["scores"].values():
        for op, score in ops.items():
            for metric, mean in score["macro"].items():
                groups = {}
                for sequence, row in score["per_sequence"].items():
                    fam = "kitti" if sequence.startswith("kitti") else sequence.split("/")[0]
                    if row[metric] is not None:
                        groups.setdefault(fam, []).append(row[metric])
                for fam, vals in groups.items():
                    assert abs(float(np.mean(vals)) - score["by_group"][fam][metric]) < 1e-12
                assert abs(float(np.mean([np.mean(v) for v in groups.values()])) - mean) < 1e-12
                count += 1
    with (exp.OUT / "comparison.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 24
    for row in rows:
        a, b, op = row["train_policy"], row["eval_policy"], row["op"]
        value = s["scores"][a + "__" + b][op]
        for metric in exp.base.PRIMARY:
            assert abs(float(row[metric]) - value["macro"][metric]) < 1e-12
        delta = value["macro"]["AP"] - s["scores"][a + "__temporal"][op]["macro"]["AP"]
        assert abs(float(row["AP_minus_own_temporal_pp"]) - 100 * delta) < 1e-12
        for fam in ("kitti", "helipr", "mulran"):
            assert abs(float(row["AP_" + fam]) - value["by_group"][fam]["AP"]) < 1e-12
    interactions = 0
    for arm, ops in s["policy_effects"].items():
        for op, values in ops.items():
            for metric, row in values.items():
                m = lambda a, b: s["scores"][a + "__" + b][op]["macro"][metric]
                expected = (m(arm, arm) - m(arm, "all")) - (m("all", arm) - m("all", "all"))
                assert abs(expected - row["interaction"]) < 1e-12
                assert abs(m(arm, arm) - m("all", arm) - row["matched_minus_control_same_inference"]) < 1e-12
                interactions += 1
    ranking = exp.base.read(exp.OUT / "ranking_diagnosis.json")
    assert ranking["script_sha256"] == sha256(REPO / "scripts/_pdv2_policy_training_diagnose.py")
    for name, digest in ranking["source_sha256"].items():
        assert sha256(REPO / name) == digest
    with np.load(exp.OUT / "all/graph_epoch1.npz") as z:
        first = {k: z[k] for k in ("query", "neighbor", "probability")}
    stride = int(first["query"].max()) + 1
    ids0 = first["query"] * stride + first["neighbor"]
    first_graph_drift = {}
    for arm in exp.ARMS[1:]:
        with np.load(exp.OUT / arm / "graph_epoch1.npz") as z:
            other = {k: z[k] for k in first}
        ids1 = other["query"] * stride + other["neighbor"]
        shared, i, j = np.intersect1d(ids0, ids1, return_indices=True)
        delta = np.abs(first["probability"][i] - other["probability"][j])
        first_graph_drift[arm] = {"source_edges": len(ids0), "common_source_edges": len(shared),
            "same_order": bool(np.array_equal(ids0, ids1)), "probability_max_error": float(delta.max()),
            "probability_median_error": float(np.median(delta)),
            "scope": "같은초기가중치의GPU첫학습그래프 재생성차이. 기존GPU수치변동을해결한것으로간주하지않는다."}
    negative_counts = {}
    for label, ops in s["scores"].items():
        negative_counts[label] = {}
        for op, score in ops.items():
            families = {}
            for sequence, row in score["per_sequence"].items():
                family = "kitti" if sequence.startswith("kitti") else sequence.split("/")[0]
                dest = families.setdefault(family, {"queries": 0, "false_accepts": 0})
                dest["queries"] += row["n_negative"]
                dest["false_accepts"] += row["negative_false_accepts"]
            negative_counts[label][op] = families
    write_json(exp.OUT / "negative_false_accepts.json", {"counts": negative_counts,
        "scope": "고정수락기준의비재방문오수락수. AP는전체신뢰도순서에의존하므로이수만으로AP변화를설명하지않는다.",
        "summary_sha256": sha256(exp.OUT / "summary.json"), "test_used": False})
    initial_path = exp.previous.OUT / "final_summary.json"
    initial = exp.base.read(initial_path)["scores"]
    older_path = exp.learned.OUT / "summary.json"
    older = exp.base.read(older_path)["scores"]
    delta_initial = {}
    for arm in exp.ARMS:
        for policy in exp.POLICIES:
            label = arm + "__" + policy
            delta_initial[label] = {op: {metric: s["scores"][label][op]["macro"][metric]
                - initial["opposite__" + policy][op]["macro"][metric] for metric in exp.base.PRIMARY}
                for op in exp.OPS}
    write_json(exp.OUT / "initial_model_reference.json", {"initial_scores": {
        policy: initial["opposite__" + policy] for policy in exp.POLICIES},
        "delta_vs_initial_same_inference": delta_initial,
        "prior_combined_temporal": older["combined_temporal"],
        "source_sha256": {str(initial_path.relative_to(REPO)): sha256(initial_path),
                          str(older_path.relative_to(REPO)): sha256(older_path)},
        "scope": "같은검증자료의절대성능참고. 초기모델보다6에포크추가학습했으므로학습정책효과의주대조는이번실험의all학습모델이다.",
        "test_used": False})
    write_json(exp.OUT / "report_validation.json", {"status": "passed", "macro_metrics": count,
        "csv_rows": len(rows), "policy_interactions": interactions, "ranking_endpoint_checks": ranking["endpoint_checks"],
        "runtime_files": len(p["runtime_sha256"]), "input_files": len(p["input_sha256"]),
        "summary_sha256": sha256(exp.OUT / "summary.json"), "ranking_sha256": sha256(exp.OUT / "ranking_diagnosis.json"),
        "script_sha256": sha256(Path(__file__)), "test_used": False, "gpu_numerical_limit_resolved": False,
        "first_epoch_gpu_source_graph_differences": first_graph_drift})
    return s


def plot(s: dict) -> None:
    """절대 AP와 계열별 학습 효과를 공통 색 범위로 그린다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    train_labels = ["Train: top 10", "Train: top 1", "Train: reject"]
    eval_labels = ["Top 10", "Temporal", "Top 1", "Reject"]
    absolute, change = [], []
    for op in exp.OPS:
        matrix = np.array([[100 * s["scores"][a + "__" + b][op]["macro"]["AP"]
                            for b in exp.POLICIES] for a in exp.ARMS])
        absolute.append(matrix)
        change.append(matrix - matrix[0:1])
    scale = max(.01, max(float(np.abs(m).max()) for m in change))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for i, ax in enumerate(axes):
        im = ax.imshow(change[i], cmap="RdYlGn", vmin=-scale, vmax=scale, aspect="auto")
        ax.set_xticks(np.arange(4), eval_labels)
        ax.set_yticks(np.arange(3), train_labels)
        ax.set_title(("NSD", "NSD-H")[i])
        ax.set_xlabel("Inference policy")
        for y in range(3):
            for x in range(4):
                ax.text(x, y, f"{absolute[i][y, x]:.3f}\n({change[i][y, x]:+.3f})", ha="center", va="center",
                        color="white" if abs(change[i][y, x]) > .72 * scale else "black")
    fig.colorbar(im, ax=axes, label="AP change vs top-10 training, same inference (pp)", shrink=.8)
    fig.suptitle("Final AP (%) and change; validation, one seed")
    for ext in ("png", "pdf"):
        fig.savefig(exp.OUT / ("policy_matrix." + ext), dpi=180)
    plt.close(fig)
    families = ("kitti", "helipr", "mulran")
    matrices = []
    for op in exp.OPS:
        matrices.append(np.array([[100 * (s["scores"][a + "__" + a][op]["by_group"][fam]["AP"]
            - s["scores"]["all__" + a][op]["by_group"][fam]["AP"])
            for fam in families] for a in exp.ARMS[1:]]))
    scale = max(.01, max(float(np.abs(m).max()) for m in matrices))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for i, ax in enumerate(axes):
        im = ax.imshow(matrices[i], cmap="RdYlGn", vmin=-scale, vmax=scale, aspect="auto")
        ax.set_xticks(np.arange(3), ["KITTI family", "HeLiPR", "MulRan"])
        ax.set_yticks(np.arange(2), ["Matched top 1", "Matched rejection"])
        ax.set_title(("NSD", "NSD-H")[i])
        for y in range(2):
            for x in range(3):
                ax.text(x, y, f"{matrices[i][y, x]:+.3f}", ha="center", va="center",
                        color="white" if abs(matrices[i][y, x]) > .72 * scale else "black")
    fig.colorbar(im, ax=axes, label="AP change (pp)", shrink=.8)
    fig.suptitle("Matched-policy training vs top-10 training; same inference")
    for ext in ("png", "pdf"):
        fig.savefig(exp.OUT / ("training_by_dataset." + ext), dpi=180)
    plt.close(fig)
    write_json(exp.OUT / "report_artifacts.json", {"sha256": {name: sha256(exp.OUT / name) for name in
        ("summary.json", "comparison.csv", "ranking_diagnosis.json", "negative_false_accepts.json", "initial_model_reference.json", "report_validation.json", "policy_matrix.png",
         "policy_matrix.pdf", "training_by_dataset.png", "training_by_dataset.pdf")}, "test_used": False})


if __name__ == "__main__":
    plot(verify())
