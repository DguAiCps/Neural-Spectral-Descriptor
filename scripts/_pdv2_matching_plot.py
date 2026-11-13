#!/usr/bin/env python3
"""완료된 네 조건 검증 비교의 표와 그림을 만든다."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_matching_mining_20261014"


def main() -> None:
    """선택된 검증 최종 AP와 같은 방향 최종 정답률을 표시한다."""
    data = json.loads((OUT / "pilot_summary.json").read_text())
    rows = data["arms"]
    font = REPO / "results/pdv2_symmetric3_20261013/UnDotum.ttf"
    fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": FontProperties(fname=str(font)).get_name(),
                         "font.size": 10, "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, op, title in zip(axes, ("pbev128_main", "hspec12"), ("NSD", "NSD-H")):
        for model, color, offset in (("gnn", "#2266bb", -0.03), ("point", "#d97722", 0.03)):
            values = [rows[f"{model}_{policy}"]["ops"][op]["macro"]["AP"]
                      for policy in ("original", "matching")]
            x = [offset, 1 + offset]
            axis.plot(x, values, "o-", color=color, linewidth=2, markersize=6)
            for index, value in enumerate(values):
                axis.annotate(f"{value:.4f}", (x[index], value), xytext=(0, 9 if model == "gnn" else -17),
                              textcoords="offset points", ha="center", color=color)
        axis.set_xticks([0, 1], ["기존 오답", "최종 매칭 오답 혼합"])
        axis.set(title=title, ylabel="검증 최종 AP", xlim=(-.25, 1.3))
        axis.margins(y=.3)
        axis.grid(axis="y", alpha=.2)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.legend(handles=[Line2D([], [], marker="o", color="#2266bb", label="대칭 문맥 GNN"),
                        Line2D([], [], marker="o", color="#d97722", label="이웃 없는 학습 투영")],
               loc="lower center", ncol=2, frameon=False)
    fig.suptitle("같은 학습량의 네 조건 비교 · 검증 1시드", y=1.0)
    fig.tight_layout(rect=(0, .08, 1, .97))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pilot_comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    table = []
    for arm, row in rows.items():
        for op, value in row["ops"].items():
            table.append({"arm": arm, "epoch": row["epoch"], "op": op, **value["macro"],
                          "same_direction_R1": value["directions"]["same_available"]["R@1"],
                          "opposite_direction_R1": value["directions"]["opposite_only"]["R@1"]})
    with (OUT / "pilot_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
