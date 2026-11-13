#!/usr/bin/env python3
"""고정 삼중쌍 구분력과 실제 손실 사례를 구분해서 그린다."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def main() -> None:
    """논문용 벡터 PDF와 확인용 PNG를 함께 만든다."""
    root = Path(__file__).resolve().parents[1]/"results/pdv2_center_information_20261021"
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=root)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    font = FontProperties(fname="/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                         "font.size": 11, "pdf.fonttype": 42})
    summary = json.loads((root/"summary.json").read_text())
    names = ("l2_center", "l2_average", "l2_difference", "l2_post", "context", "key")
    labels = ("중심 C", "이웃 평균 A", "차이 D", "잔차 후", "최종 문맥", "검색 키")
    left = summary["val"]["diff_tanh"]["all"]["macro"]
    right = summary["cases"]["diff_tanh"]["diff_tanh"]["pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    colors = ["#72859a", "#268d99", "#ca6754", "#8c83ae", "#3c8059", "#3971a3"]
    for ax, data, title in zip(axes, (left, right),
            ("고정 어려운 오답과 비교 · 양성 질의 10,036개\n시퀀스→3계열 등비중 평균", 
             "실제 놓친 91개 · 두 모델의 최종 후보 비교\n91개 사례의 통합 개수")):
        values = [100*data["features"][name]["accuracy"] for name in names]
        bars = ax.bar(np.arange(len(names)), values, color=colors, width=.7)
        for bar, value in zip(bars, values):
            text = f"{value:.1f}%" if ax is axes[0] else f"{round(value*.91)}/91"
            ax.text(bar.get_x()+bar.get_width()/2, value+1.6, text, ha="center", fontsize=10)
        ax.set_xticks(np.arange(len(names)), labels, rotation=15)
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_ylim(0, 104)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("정답 후보의 코사인이 더 높은 비율 (%)")
    fig.suptitle("차이+tanh · 마지막 층의 중심항 진단 (학습 시드 1)", fontsize=15, y=1.02)
    fig.text(.5, -.03, "중간 특징의 쌍별 순위 진단이며 최종 AP가 아니다. 이웃 평균은 같은 주의 가중치에서 중심항을 복원한 값이다.", ha="center", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out/("center_information."+ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
