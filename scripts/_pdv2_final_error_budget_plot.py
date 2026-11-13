#!/usr/bin/env python3
"""완료된 AP 손실 분해를 잘리지 않는 독립 그림으로 내보낸다."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pdv2_key_metric_alignment_20261026/error_budget"


def main() -> None:
    """모델 대조와 계열 차이를 같은 단위의 가로 막대로 표시한다."""
    report = json.loads((OUT / "summary.json").read_text())
    keys = ("retrieval_miss", "nonrevisit_false_accept", "wrong_place_false_accept")
    colors = ("#6e8ca0", "#de9450", "#aa6878")
    names = (
        "Missed correct place",
        "Non-revisit false acceptance",
        "Wrong-place false acceptance",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    panels = [
        (
            ["Original", "Metric-aligned"],
            [report["mean"][k] for k in ("control", "aligned")],
            "Three-seed mean",
        ),
        (
            ["KITTI", "HeLiPR", "MulRan"],
            [
                {
                    k: np.mean(
                        [
                            r["families"][f][k]
                            for r in report["results"]["control"].values()
                        ]
                    )
                    for k in keys
                }
                for f in ("kitti", "helipr", "mulran")
            ],
            "Original model by validation family",
        ),
    ]
    for ax, (labels, rows, title) in zip(axes, panels):
        left = np.zeros(len(labels))
        for k, name, color in zip(keys, names, colors):
            value = 100 * np.asarray([r[k] for r in rows])
            ax.barh(labels, value, left=left, label=name, color=color, height=0.6)
            for i, v in enumerate(value):
                if v >= 1:
                    ax.text(
                        left[i] + v / 2,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
            left += value
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("100 - final validation AP (percentage points)", fontsize=9)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        "Diagnostic error decomposition using ground-truth labels", fontsize=12
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"final_error_budget_v2.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
