#!/usr/bin/env python3
"""검산된 후보 경로 수치의 그림과 그림 출처만 갱신한다."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_no_context_ablation import sha256, write_json
import json

OUT = REPO / "results/pdv2_candidate_paths_20261020"
ARMS = ("baseline", "unary_null", "joint_forced", "joint_null")
LABELS = ("기존 선택", "단일 선택·거절", "경로·중심 강제", "경로·거절 허용")


def main() -> None:
    """평균값의 라벨과 범례가 겹치지 않는 최종 비교 그림을 만든다."""
    audit = json.loads((OUT / "completion_validation.json").read_text())
    assert audit["status"] == "passed"
    rows = list(csv.DictReader((OUT / "final_metrics.csv").open()))
    font_path = REPO / "results/pdv2_sequence_report_20261018/UnDotum.ttf"
    font = FontProperties(fname=str(font_path))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric in zip(axes, ("AP", "R@100P")):
        for model, color, offset, label in (
                ("standard", "#24788c", -.18, "표준 시간 모델"),
                ("capacity_point", "#e2a143", .18, "이웃 없는 학습 투영")):
            values = [np.mean([float(r[metric]) for r in rows if r["model"] == model and r["arm"] == arm])
                      for arm in ARMS]
            bars = ax.bar(np.arange(4)+offset, values, width=.34, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set_xticks(np.arange(4), LABELS, fontproperties=font, fontsize=9)
        ax.set_ylabel(metric+" (%)")
        ax.set_ylim(0, 100 if metric == "AP" else 60)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(prop=font, loc="upper right")
    fig.suptitle("후보 대응 시간 제약: 최종 검증 지표, 3시드", fontproperties=font)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(OUT / ("candidate_paths."+extension), dpi=180, bbox_inches="tight")
    plt.close(fig)
    for p in (Path(__file__), font_path, OUT / "candidate_paths.png", OUT / "candidate_paths.pdf"):
        audit["sources"][str(p.relative_to(REPO))] = sha256(p)
    audit["plot_note"] = "검산 뒤 범례와 수치 겹침만 수정했다. 원장·지표·추론·검산 결과는 그대로다."
    write_json(OUT / "completion_validation.json", audit)


if __name__ == "__main__":
    main()
