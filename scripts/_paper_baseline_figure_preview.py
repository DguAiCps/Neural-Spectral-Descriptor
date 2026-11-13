#!/usr/bin/env python3
"""최종 검색 결과의 정밀도 요구 수준별 비교 시안을 만든다."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from _paper_baseline_figure_review import LABELS, OUT


COLORS = dict(scpp="#B34E19", ring="#879BAF", ringpp="#546578", ringsharp="#156082",
              deepring="#379991", disco="#C29169", helios="#7B586F", seqot="#94791C",
              imlpr="#AB5874", bevplace="#753A85", nsd="#196B24")
STYLES = dict(scpp="--", ring=":", ringpp="-.", ringsharp="-", deepring="--",
              disco=":", helios="-.", seqot="--", imlpr=":", bevplace="-", nsd="-")


def render(layout: dict = None) -> None:
    """전체 범위와 명시한 확대 범위에서 같은 최종 수치를 그린다.

    layout을 주면 그림 크기·두 축 위치·범례 배치만 바꾼다. 기본값은 기존 배치다.
    colors(방법별 색 덮어쓰기)·fonts(글자 크기)·lw_base(비교 방법 선 굵기)도 layout으로 바꿀 수 있다.
    """
    layout = layout or {}
    colors = {**COLORS, **layout.get("colors", {})}
    fonts = {**dict(tick=6.5, legend=6.5, label=7, title=7.7, endpoint=7, title_pad=6), **layout.get("fonts", {})}
    lw_base = layout.get("lw_base", .9)
    data = json.loads((OUT / "curves.json").read_text())
    x = np.asarray(data["precision_grid"]) * 100
    plt.rcParams.update({"font.family": "Carlito", "font.size": fonts["label"],
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "text.color": "#0E2841", "axes.labelcolor": "#0E2841",
                         "axes.edgecolor": "#6D7780", "axes.linewidth": .6})
    fig = plt.figure(figsize=layout.get("figsize", (3.35, 4.2)))
    overview = fig.add_axes(layout.get("overview", [.155, .625, .81, .305]))
    detail = fig.add_axes(layout.get("detail", [.155, .235, .81, .285]))
    for method, row in data["methods"].items():
        mean = np.asarray(row["mean"]) * 100
        lw = 1.8 if method == "nsd" else 1.35 if method == "bevplace" else lw_base
        zorder = 5 if method == "nsd" else 4 if method == "bevplace" else 2
        for ax in (overview, detail):
            ax.plot(x, mean, color=colors[method], ls=STYLES[method], lw=lw, zorder=zorder)
        if method in ("nsd", "bevplace"):
            sd = np.asarray(row["std"]) * 100
            detail.fill_between(x, mean-sd, mean+sd, color=colors[method], alpha=.13,
                                edgecolor="none", zorder=1)
            detail.plot(100, mean[-1], "o", ms=3.5, c=colors[method], zorder=7,
                        markeredgecolor="white", markeredgewidth=.4, clip_on=False)
            # 끝점 라벨의 글자 위치(포인트)와 지시선 시작점은 확대 범위에 따라 바꿀 수 있다.
            offset = layout.get("endpoint_offsets", {}).get(method, (-5, 9 if method == "nsd" else -10))
            arrow = dict(arrowstyle="-", color=colors[method], lw=.6)
            if method in layout.get("endpoint_relpos", {}):
                arrow["relpos"] = layout["endpoint_relpos"][method]
            detail.annotate(f"{LABELS[method]} {mean[-1]:.1f}%", (100, mean[-1]),
                            xytext=offset, textcoords="offset points", ha="right",
                            fontsize=fonts["endpoint"], color=colors[method], weight="bold", arrowprops=arrow)
    x0, x1 = layout.get("zoom_x", (95, 100))
    y0, y1 = layout.get("zoom_y", (53, 85))
    overview.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor="#3B4650",
                                lw=.8, zorder=8))
    overview.set(xlim=(0, 100), ylim=(0, 100), xticks=[0, 25, 50, 75, 100],
                 yticks=[0, 25, 50, 75, 100])
    detail.set(xlim=(x0, x1), ylim=(y0, y1), xticks=layout.get("zoom_xticks", [95, 96, 97, 98, 99, 100]),
               yticks=layout.get("zoom_yticks", [55, 65, 75, 85]))
    for ax, title in ((overview, f"(a) Final detection: NSD and {len(data['methods']) - 1} baselines"),
                      (detail, "(b) High-precision detail")):
        ax.set(xlabel="Required precision (%)", ylabel="Recall (%)")
        ax.set_title(title, fontsize=fonts["title"], weight="bold", pad=fonts["title_pad"])
        ax.tick_params(labelsize=fonts["tick"], length=2, pad=2)
        ax.grid(color="#ECEFF1", lw=.45)
        ax.set_axisbelow(True)
    order = ["nsd", "bevplace"] + [m for m in LABELS if m in data["methods"] and m not in ("nsd", "bevplace")]
    handles = [Line2D([0], [0], color=colors[m], ls=STYLES[m],
                      lw=1.8 if m == "nsd" else max(1, lw_base), label=LABELS[m]) for m in order]
    legend = fig.legend(handles=handles, frameon=False, fontsize=fonts["legend"],
                        handlelength=layout.get("handlelength", 2),
                        handletextpad=.5, labelspacing=.35,
                        **layout.get("legend", dict(loc="lower center", bbox_to_anchor=(.53, -.005),
                                                    ncol=3, columnspacing=.9)))
    legend.get_texts()[0].set_weight("bold")
    legend.get_texts()[0].set_color(colors["nsd"])
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUT / ("baseline_precision_candidate." + suffix), dpi=300,
                    bbox_inches="tight", pad_inches=.04, facecolor="white")
    plt.close(fig)
    (OUT / "caption.txt").write_text(
        "Final-detection recall at required precision levels. "
        "The boxed region in (a) is enlarged in (b); shading denotes one seed SD.\n")
    (OUT / "body_note.txt").write_text(
        "We sweep the final acceptance score separately for each sequence and compute "
        "the maximum recall satisfying each precision requirement. We then average "
        "over sequences within each city, over the six cities, and over training seeds, "
        "as in Table 2. Training-free methods use one run. At 100% required precision, "
        "the curve equals the reported R@100P. These are threshold-sweep statistics, "
        "rather than results at a single deployment threshold. NSD reaches 65.69% "
        "recall at 99% precision and 62.51% recall at 100% precision, compared with "
        "64.60% and 57.94% for BEVPlace++. BEVPlace++ has higher recall at 95% "
        "precision (76.76% versus 73.47%).\n")
    # 검토자가 표의 숫자와 직접 대조할 수 있는 작은 자료도 남긴다.
    rows = ["method,precision,recall_mean,recall_seed_sd"]
    for method, row in data["methods"].items():
        for requirement in (90, 95, 99, 99.5, 100):
            i = int(np.argmin(abs(x-requirement)))
            sd = "" if row["std"] is None else f"{row['std'][i]:.12f}"
            rows.append(f"{LABELS[method]},{requirement / 100},{row['mean'][i]:.12f},{sd}")
    (OUT / "comparison.csv").write_text("\n".join(rows) + "\n")
    print(OUT / "baseline_precision_candidate.pdf")


if __name__ == "__main__":
    render()
