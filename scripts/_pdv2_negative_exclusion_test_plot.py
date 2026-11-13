#!/usr/bin/env python3
"""고정 시험의 최종 성능과 같은 구조 내 학습 수정 효과를 그린다."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "results/pdv2_negative_exclusion_test_20261021/summary.json"


def main(output: Path) -> None:
    """시드 1의 최종 판정 지표만 독립 그림 파일로 저장한다."""
    summary = json.loads(SOURCE.read_text())
    font = "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf"
    if Path(font).exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.unicode_minus": False})
    method = "arc_unique_b100"
    models = [summary["aggregate"][a][method]["macro"]
              for a in ("diff_baseline", "diff_mask25")]
    models += [summary["references"]["standard_seed1"]["macro"],
               summary["references"]["bevplace_seed1"]]
    labels = ["차이+tanh", "차이+tanh + 25m 제외", "기존 시간 NSD (표준 주의)", "공식 BEVPlace++"]
    colors = ["#637386", "#c07924", "#26756e", "#634c91"]
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    grid = fig.add_gridspec(2, 4, height_ratios=[1.3, 1])
    for index, metric in enumerate(("AP", "max_F1", "R@100P", "R@1")):
        ax = fig.add_subplot(grid[0, index])
        values = np.array([row[metric] * 100 for row in models])
        ax.scatter(values, np.arange(4), c=colors, s=65, zorder=3)
        for y, value in enumerate(values):
            ax.annotate(f"{value:.3f}", (value, y), xytext=(0, -17),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_yticks(np.arange(4), labels if index == 0 else [""] * 4)
        ax.set_ylim(3.6, -.5)
        margin = max(.15, np.ptp(values) * .18)
        ax.set_xlim(values.min() - margin, values.max() + margin)
        ax.set_title({"max_F1": "최대 F1", "R@1": "최종 재방문 R@1"}.get(metric, metric))
        ax.set_xlabel("최종 성능 (%)")
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[1, :])
    city_names = ["Karlsruhe", "Seoul", "Daejeon", "Ann Arbor", "Stockholm", "Hamburg"]
    deltas = [row["AP"] for row in summary["city_mask25_minus_baseline_pp"].values()]
    bars = ax.bar(city_names, deltas, color=["#26756e" if d >= 0 else "#b34b40" for d in deltas])
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.bar_label(bars, labels=[f"{d:+.3f}" for d in deltas], padding=4)
    ax.set_ylim(-.085, .16)
    ax.set_ylabel("최종 AP 차이 (%p)")
    ax.set_title("25m 제외 - 같은 차이 구조의 기존 손실")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("시험 46시퀀스 · 공통 195,726질의 · 시드 1\n"
                 "45개 양성 시퀀스 → 6개 도시 동일 비중 평균; 25m 제외는 학습 손실에만 적용",
                 fontsize=13)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"negative_exclusion_test.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
