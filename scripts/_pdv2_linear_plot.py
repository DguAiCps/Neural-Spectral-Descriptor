#!/usr/bin/env python3
"""최종 검증 지표와 단순 가중합·MLP 비교 그림을 저장한다."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results/pdv2_linear_fusion_20261015"


def main() -> None:
    """선택된 결과만 표와 그림에 넣는다."""
    result = json.loads((OUT / "summary.json").read_text())
    rows = []
    for kind, record in result["initial_fixed_pipeline"].items():
        for op, value in record["ops"].items():
            rows.append({"method": "original", "arm": kind, "epoch": record["epoch"],
                         "op": op, **value["macro"]})
    for method, records in (("mlp", result["mlp_reference"]), ("linear", result["arms"])):
        for arm, record in records.items():
            for op, metrics in record["ops"].items():
                rows.append({"method": method, "arm": arm, "epoch": record["epoch"],
                             "op": op, **metrics})
    with (OUT / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    labels = ["Original scoring", "Joint + MLP", "Joint + linear"]
    for col, (op, name) in enumerate((("pbev128_main", "NSD"), ("hspec12", "NSD-H"))):
        for row, (metric, title) in enumerate((("AP", "Average precision"), ("R@100P", "Recall at 100% precision"))):
            ax = axes[row, col]
            for kind, offset, color in (("gnn", -.18, "#225ea8"), ("point", .18, "#d95f0e")):
                values = [result["initial_fixed_pipeline"][kind]["ops"][op]["macro"][metric],
                          result["mlp_reference"][kind + "_joint"]["ops"][op][metric],
                          result["arms"][kind + "_joint"]["ops"][op][metric]]
                bars = ax.bar(np.arange(3) + offset, values, width=.34, color=color,
                              label="GNN" if kind == "gnn" else "Pointwise")
                ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
            ax.set_xticks([0, 1, 2], labels)
            ax.set_ylim(0, 1)
            ax.set_title(name + " / " + title, fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    axes[0, 0].legend(loc="lower left", frameon=False)
    fig.suptitle("Final validation: four raw scores with five learned constants\n"
                 "One seed; three-family macro; MLP uses different features, scaling and learning rate", fontsize=11)
    fig.savefig(OUT / "comparison.png", dpi=180)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
