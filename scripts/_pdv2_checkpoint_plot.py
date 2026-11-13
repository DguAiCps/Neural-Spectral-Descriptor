#!/usr/bin/env python3
"""최종 검증 AP에 따른 체크포인트 선택 곡선과 비교 CSV를 만든다."""
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
OUT = REPO / "results/pdv2_checkpoint_select_20261014"


def main() -> None:
    """완료된 검증 선택의 값만 시각화하며 선택 규칙을 바꾸지 않는다."""
    result = json.loads((OUT / "validation_summary.json").read_text())
    font = REPO / "results/pdv2_symmetric3_20261013/UnDotum.ttf"
    fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "font.size": 10})
    colors = ("#2266bb", "#d97722", "#26916c")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharex=True)
    for ax, op, title in zip(axes, ("pbev128_main", "hspec12"), ("NSD", "NSD-H")):
        for seed, color in zip(("1", "2", "3"), colors):
            rows = sorted(result["per_seed"][seed], key=lambda row: row["epoch"])
            ax.plot([r["epoch"] for r in rows], [r["ops"][op]["macro"]["AP"] for r in rows],
                    "o-", color=color, markersize=4, linewidth=1.3)
            for row in rows:
                value = row["ops"][op]["macro"]["AP"]
                if row["reference"]:
                    ax.scatter(row["epoch"], value, marker="D", s=85, facecolors="none",
                               edgecolors=color, linewidths=1.6, zorder=4)
                if row["id"] == result["selected"][seed]["id"]:
                    ax.scatter(row["epoch"], value, marker="*", s=160, color=color,
                               edgecolors="white", linewidths=0.5, zorder=5)
        ax.set(title=title, xlabel="학습 에포크", ylabel="검증 최종 AP")
        ax.grid(alpha=0.2)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    handles = [Line2D([], [], color=c, marker="o", label=f"시드 {s}")
               for s, c in zip((1, 2, 3), colors)]
    handles += [Line2D([], [], color="#333333", marker="D", markerfacecolor="none",
                       linestyle="none", label="기존 키 R@1 선택"),
                Line2D([], [], color="#333333", marker="*", markersize=12,
                       linestyle="none", label="최종 AP 선택")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("같은 최종 파이프라인에서 학습 시점 비교", y=1.0)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"validation_checkpoint_curve.{suffix}", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    rows = []
    for seed, entries in result["per_seed"].items():
        for entry in entries:
            rows.append({"seed": seed, "checkpoint": entry["id"], "epoch": entry["epoch"],
                         "reference": entry["reference"],
                         "selected": entry["id"] == result["selected"][seed]["id"],
                         "NSD_AP": entry["ops"]["pbev128_main"]["macro"]["AP"],
                         "NSD_H_AP": entry["ops"]["hspec12"]["macro"]["AP"],
                         "joint_AP": entry["joint_val_AP"]})
    with (OUT / "validation_checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
