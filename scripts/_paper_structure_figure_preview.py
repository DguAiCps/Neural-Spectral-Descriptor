#!/usr/bin/env python3
"""실제 스캔·정렬 응답·방향별 최종 결과로 한 단 검토 그림을 만든다."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _paper_pair_diagnostics import points_for


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/paper_structure_figure_review_20261031"
GREEN, GREY, INK = "#196B24", "#7B8790", "#0E2841"


def draw_cloud(ax: plt.Axes, points: np.ndarray, title: str) -> None:
    """같은 80 m 센서 좌표 범위에 실제 점군을 그린다."""
    ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap="cividis",
               vmin=-2, vmax=8, s=.16, rasterized=True, linewidths=0)
    ax.plot(0, 0, "+", color="#B34E19", ms=3, mew=.7)
    ax.set(xlim=(-40, 40), ylim=(-40, 40), aspect="equal", xticks=[], yticks=[])
    ax.set_title(title, fontsize=7, pad=2)
    for spine in ax.spines.values():
        spine.set_color("#D4DEE5")


def render() -> None:
    """고정 수치의 전체 범위를 보존하는 그림 후보를 저장한다."""
    report = json.loads((OUT / "summary.json").read_text())
    example = json.loads((OUT / "example_profile.json").read_text())
    plt.rcParams.update({"font.family": "Carlito", "font.size": 7,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "text.color": INK, "axes.labelcolor": INK,
                         "axes.edgecolor": "#6D7780", "axes.linewidth": .6})
    fig = plt.figure(figsize=(3.35, 4.95))
    fig.text(.52, .995, "(a) Same place, opposite headings", fontsize=8,
             ha="center", va="top", weight="bold")
    for left, name in ((.17, "query"), (.56, "reference")):
        ax = fig.add_axes([left, .775, .30, .175])
        title = "Query" if name == "query" else "Past reference"
        draw_cloud(ax, points_for(example["sequence"], example[name]), title)
    fig.text(.52, .755, f"NCLT · {example['distance_m']:.2f} m apart · "
             f"heading gap {example['heading_deg']:.1f}°", ha="center", fontsize=6.8)

    ax = fig.add_axes([.17, .475, .785, .215])
    angles = np.r_[example["angles_deg"], 360.]
    scores = np.r_[example["profile"], example["profile"][0]]
    ax.plot(angles, scores, color=GREEN, lw=1.35)
    peak, value = example["peak_shift_deg"], example["maximum_score"]
    ax.axvline(peak, color=GREEN, lw=.7, ls="--", alpha=.6)
    ax.scatter([0, peak], [scores[0], value], s=17, c=[GREY, GREEN],
               edgecolors="white", linewidths=.4, zorder=5)
    ax.annotate(f"Max: {value:.2f}", (peak, value), xytext=(225, .55),
                color=GREEN, fontsize=6.9, weight="bold",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=.6))
    ax.annotate(f"0°: {scores[0]:.2f}", (0, scores[0]), xytext=(34, -.64),
                color=GREY, fontsize=6.9,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=.6))
    ax.set(xlim=(0, 360), ylim=(-1, 1.05), xticks=[0, 90, 180, 270, 360],
           yticks=[-1, 0, 1], xlabel="Circular shift (°)", ylabel="Spectral score")
    ax.set_title("(b) Alignment response for this scan pair", fontsize=7.5, pad=6)
    ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    ax.grid(color="#ECEFF1", lw=.45)

    ax = fig.add_axes([.17, .125, .785, .225])
    bins = report["bins"]
    centers = [(r["low_deg"]+r["high_deg"])/2 for r in bins]
    for stage, label, colour, style in (("full", "NSD", GREEN, "-"),
                                        ("zero_shift", "Fixed 0° alignment", GREY, "--")):
        means = np.array([r["stages"][stage]["mean"] for r in bins])*100
        sd = np.array([np.std(r["stages"][stage]["per_seed"], ddof=1) for r in bins])*100
        ax.errorbar(centers, means, yerr=sd, color=colour, ls=style, lw=1.4,
                    marker="o", ms=3, capsize=2, label=label)
        ax.annotate(f"{means[-1]:.1f}%", (centers[-1], means[-1]),
                    xytext=(-4, 6 if stage == "full" else 7),
                    textcoords="offset points", ha="right", fontsize=7,
                    color=colour, weight="bold")
    ax.set(xlim=(0, 180), ylim=(0, 100), xticks=centers, yticks=[0, 50, 100],
           xlabel="GT heading gap (°)", ylabel="Final top-1 correct (%)")
    ax.set_xticklabels([f"{int(r['low_deg'])}–{int(r['high_deg'])}\nn={r['queries']:,}" for r in bins], fontsize=5.8)
    ax.set_title("(c) Final matching across heading gaps", fontsize=7.5, pad=6)
    ax.tick_params(axis="y", labelsize=6.5, length=2, pad=1.5)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.grid(color="#ECEFF1", lw=.45)
    ax.legend(loc="upper center", bbox_to_anchor=(.51, -.39), ncol=2,
              frameon=False, fontsize=6.7, handlelength=2)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / ("alignment_mechanism_candidate."+suffix), dpi=260,
                    bbox_inches="tight", pad_inches=.04, facecolor="white")
    plt.close(fig)
    print(OUT / "alignment_mechanism_candidate.pdf")


if __name__ == "__main__":
    render()
