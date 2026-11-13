#!/usr/bin/env python3
"""고정된 간선 진단을 방향·비재방문별로 설명한다. 설정 선택에 쓰지 않는다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_edge_quality_20261016"
POLICIES = ("p000", "p025", "p050", "p075", "p090")


def main() -> None:
    """집계 결과의 분모를 분리해 그림과 재현 가능한 비율을 저장한다."""
    source = OUT / "quality_summary.json"
    quality = json.loads(source.read_text())
    result = {
        "scope": "사후 설명용 집계. 학습·설정 선택·시험 평가에 쓰지 않는다.",
        "quality_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "models": {},
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    colors = ("#2676b8", "#d47522")
    for row, model in enumerate(("current", "aligned")):
        record = {"validation_true_neighbor_coverage": {},
                  "no_loop_retaining_edges": {}}
        val = quality[model]["val"]
        ax = axes[row, 0]
        for group, label, color in zip(
                ("same_available", "opposite_only"),
                ("Same direction available", "Opposite direction only"), colors):
            counts = [val["groups"][group]["candidates30"],
                      *[val["policy"][p][group] for p in POLICIES]]
            fractions = [c["nodes_with_true_edge"] / c["nodes"] for c in counts]
            record["validation_true_neighbor_coverage"][group] = {
                "nodes": counts[0]["nodes"],
                "stages": dict(zip(("candidates30", *POLICIES), fractions)),
            }
            ax.plot(np.arange(len(counts)), np.array(fractions) * 100,
                    "o-", color=color, label=label)
        ax.set_xticks(range(6), ["Candidate\n30", "Selected\n10", "p >= .25",
                                "p >= .50", "p >= .75", "p >= .90"], fontsize=8)
        ax.set_title(model.capitalize() + ": validation true-neighbor coverage")
        ax.set_ylabel("Queries with a neighbor within 5 m (%)")
        ax = axes[row, 1]
        for role, label, color in zip(("train", "val"), ("Train", "Validation"), colors):
            counts = [quality[model][role]["policy"][p]["negative"] for p in POLICIES]
            fractions = [c["nodes_with_edge"] / c["nodes"] for c in counts]
            record["no_loop_retaining_edges"][role] = {
                "nodes": counts[0]["nodes"], "policies": dict(zip(POLICIES, fractions)),
            }
            ax.plot(np.arange(len(counts)), np.array(fractions) * 100,
                    "o-", color=color, label=label)
        ax.set_xticks(range(5), ["Selected 10", "p >= .25", "p >= .50", "p >= .75", "p >= .90"], fontsize=8)
        ax.set_title(model.capitalize() + ": no-loop queries retaining edges")
        ax.set_ylabel("No-loop queries with any similarity edge (%)")
        for ax in axes[row]:
            ax.set_ylim(0, 102)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        result["models"][model] = record
    fig.suptitle("Fixed checkpoints, seed 1. Edge diagnostics; these are not final recognition metrics.", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / ("edge_direction_quality." + extension), dpi=180)
    plt.close(fig)
    (OUT / "stratified_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
