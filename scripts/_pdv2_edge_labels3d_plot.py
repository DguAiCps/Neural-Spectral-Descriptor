#!/usr/bin/env python3
"""라벨 수정의 최종 성능과 간선 품질을 표·그림으로 저장한다."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_edge_labels3d as exp


def main() -> None:
    """최종 판정과 간선 진단의 지표를 분리해 내보낸다."""
    summary = exp.read(exp.OUT / "summary.json")
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    for row, name in enumerate(exp.MODELS):
        for col, metric in enumerate(("AP", "R@100P")):
            ax = axes[row, col]
            for i, op in enumerate(exp.OPS):
                baseline = summary["scores"][name]["original"]["ops"][op]["macro"][metric]
                values = [100 * (summary["scores"][name][a]["ops"][op]["macro"][metric] - baseline) for a in exp.ARMS[1:]]
                ax.bar(np.arange(3) + (i - .5) * .35, values, .35, label=("NSD", "NSD-H")[i], color=("#2676b8", "#d47522")[i])
            ax.axhline(0, color="black", linewidth=.8)
            ax.set_xticks(range(3), ["XY retraining", "3D retraining", "No similarity edges"])
            ax.set_title(name.capitalize() + ": final " + metric)
            ax.set_ylabel("Change vs original classifier (pp)")
            ax.grid(axis="y", alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("Frozen GNN, validation only (seed 1). Classifier recipe and top-10 budget fixed.")
    for suffix in ("png", "pdf"):
        fig.savefig(exp.OUT / ("comparison." + suffix), dpi=180)
    plt.close(fig)
    with (exp.OUT / "comparison.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "classifier", "operating_point", *exp.PRIMARY])
        for name in exp.MODELS:
            for arm in exp.ARMS:
                for op in exp.OPS:
                    writer.writerow([name, arm, op, *[summary["scores"][name][arm]["ops"][op]["macro"][m] for m in exp.PRIMARY]])
    print("PLOTTED", flush=True)


if __name__ == "__main__":
    main()
