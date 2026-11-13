#!/usr/bin/env python3
"""가까운 오답 제외 대조의 선택된 최종 판정 지표를 그린다."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def main() -> None:
    """단일 시드 검증 결과를 PDF와 PNG로 보존한다."""
    root = Path(__file__).resolve().parents[1]/"results/pdv2_negative_exclusion_20261021"
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=root)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root/"summary.json").read_text())
    models = {**summary["per_seed"]["1"], **summary["references"]}
    names = ("center_zero", "diff_mask5", "diff_mask25", "center_full", "full_mask25", "standard_tanh", "standard_mask25")
    labels = ("차이 · 기존 손실", "차이 · 5m 제외", "차이 · 25m 제외", "전체 보존 · 기존 손실", "전체 보존 · 25m 제외", "표준 · 기존 손실", "표준 · 25m 제외")
    colors = ("#7a8697", "#267f8c", "#12616d", "#7a8697", "#3b865f", "#7a8697", "#bc7955")
    font = FontProperties(fname="/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
    for ax, metric in zip(axes, ("AP", "max_F1", "R@100P")):
        values = [100*models[a]["selected"]["arc_unique_b100"]["macro"][metric] for a in names]
        span = max(max(values)-min(values), .04)
        for i, (value, color) in enumerate(zip(values, colors)):
            ax.scatter(value, i, color=color, s=75, zorder=3)
            ax.text(value+.045*span, i-.14, f"{value:.3f}", color=color, fontsize=10)
        ax.set_xlim(min(values)-.18*span, max(values)+.5*span)
        ax.set_title({"AP": "최종 AP", "max_F1": "최대 F1", "R@100P": "R@100P"}[metric])
        ax.set_xlabel("%")
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(6.4, -.5)
    fig.suptitle("가까운 학습 오답 제외의 최종 성능", fontsize=15, y=1.04)
    fig.text(.5, -.06, "검증 9시퀀스 · 시드 1 · 각 조건의 10/20에포크 중 최종 AP로 선택 · 시퀀스→3계열 평균\n시간 간선만 사용 · 모든 모델의 키에 tanh 유지 · 기존 손실은 동결한 동일 예산 대조.", ha="center", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out/("negative_exclusion."+ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
