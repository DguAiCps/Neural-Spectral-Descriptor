#!/usr/bin/env python3
"""완료된 선택별 학습 쌍 결과를 겹치지 않는 수치 라벨로 그린다."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main() -> None:
    """최종 결과 JSON만 읽어 별도 그림 파일을 저장한다."""
    root = Path(__file__).resolve().parents[1] / "results/pdv2_native_pairs_20261020"
    summary = json.loads((root / "summary.json").read_text())
    font = "/tmp/nsd_context_accept_font.ttf"
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    for col, (method, title) in enumerate((
        ("baseline", "기존 시간 판정"), ("context_score", "고정 문맥 가중합")
    )):
        for row, metric in enumerate(("AP", "R@100P")):
            ax = axes[row, col]
            series = [
                [100*summary["aggregate"][arm][method]["macro"][metric] for arm in arms]
                for arms in (
                    ("common_gt", "gt_small", "gt_large"),
                    ("common_kiss", "kiss_small", "kiss_large"),
                )
            ]
            for index, (name, color) in enumerate((("GT", "#426b98"), ("KISS", "#b16430"))):
                values = series[index]
                ax.plot(range(3), values, "o-", label=name, color=color)
                for x, value in enumerate(values):
                    above = value >= series[1-index][x]
                    ax.annotate(f"{value:.3f}", (x, value),
                                xytext=(0, 10 if above else -16),
                                textcoords="offset points", ha="center", fontsize=8,
                                color=color, bbox={"facecolor": "white", "edgecolor": "none", "pad": .5})
            ref = 100*summary["aggregate"]["reference"][method]["macro"][metric]
            ax.axhline(ref, color="#777777", linestyle="--",
                       label=f"기존 전체 GT 900회 ({ref:.3f})")
            ax.set_xticks(range(3), ["공통 쌍\n720회", "선택별 쌍\n720회", "선택별 쌍\n900회"])
            ax.set_title(title+" · "+metric+" (%)", pad=32)
            ax.margins(x=.15, y=.3)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(.5, 1.005),
                      ncol=3, frameon=False)
    fig.suptitle("선택별 학습 쌍과 예산 대조 · 최종 검증 9 · 시드 1")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(root / ("native_pairs_final."+suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
