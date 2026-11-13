#!/usr/bin/env python3
"""최종 판정 변화와 간선 확률별 품질을 내보낼 수 있는 그림으로 만든다."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_edge_quality as exp


def main() -> None:
    """단일 시드 검증의 최종 지표와 특권 진단을 구분해 그린다."""
    scores = exp.read(exp.OUT / "summary.json")["scores"]
    quality = exp.read(exp.OUT / "quality_summary.json")
    policies = exp.POLICIES[1:]
    labels = ["p>=.25", "p>=.50", "p>=.75", "p>=.90", "Random\n(same count)", "No sim\nedges", "GT filter\n(oracle)"]
    colors = ("#2676b8", "#d47522")
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for row, name in enumerate(exp.MODELS):
        for col, metric in enumerate(("AP", "R@100P")):
            ax = axes[row, col]
            for i, op in enumerate(exp.OPS):
                reference = scores[name]["p000"]["ops"][op]["macro"][metric]
                values = [100 * (scores[name][p]["ops"][op]["macro"][metric] - reference) for p in policies]
                ax.bar(np.arange(len(policies)) + (i - .5) * .34, values, width=.34,
                       color=colors[i], label=("NSD", "NSD-H")[i])
            ax.axhline(0, color="black", linewidth=.7)
            ax.axvspan(5.5, 6.5, color="grey", alpha=.12)
            ax.set_xticks(range(len(labels)), labels, fontsize=8)
            ax.set_title(f"{name.capitalize()} training: final {metric}")
            ax.set_ylabel("Change from unfiltered graph (percentage points)", fontsize=9)
            ax.grid(axis="y", alpha=.18)
            ax.legend(fontsize=8)
    fig.suptitle("Fixed checkpoint, validation only (seed 1). GT filter is a privileged diagnostic.")
    for suffix in ("png", "pdf"):
        fig.savefig(exp.OUT / ("comparison." + suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, name in zip(axes, exp.MODELS):
        for role, color in (("train", colors[0]), ("val", colors[1])):
            rows = quality[name][role]["probability_bins"]["selected10"]
            x = [r["mean_probability"] if r["edges"] else (r["lo"] + r["hi"]) / 2 for r in rows]
            y = [r["positive_3d_fraction"] if r["edges"] else np.nan for r in rows]
            ax.plot(x, y, "o-", color=color, label=role)
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean classifier score in bin",
               ylabel="Fraction of selected edges within 5 m (3D)", title=name.capitalize() + " training")
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle("Selected top-10 edges. Training and validation use different places and pose inputs.", fontsize=10)
    for suffix in ("png", "pdf"):
        fig.savefig(exp.OUT / ("edge_quality." + suffix), dpi=180)
    with (exp.OUT / "comparison.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "policy", "op", "AP", "max_F1", "R@100P", "R@1"])
        for name in exp.MODELS:
            for p in exp.POLICIES:
                for op in exp.OPS:
                    writer.writerow([name, p, op, *[scores[name][p]["ops"][op]["macro"][m] for m in exp.PRIMARY]])
    print("PLOTTED", flush=True)


if __name__ == "__main__":
    main()
