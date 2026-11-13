#!/usr/bin/env python3
"""검산된 시험 결과의 도표를 한글·수학 기호 글꼴로 저장한다."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_temporal_test_control as run
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """기존 검산 산출물을 보존하고 글꼴을 수정한 별도 그림을 만든다."""
    audit = run.exp.read(run.OUT / "completion_validation.json")
    p = run.OUT / "summary.json"
    assert sha256(p) == audit["sources"][str(p.relative_to(REPO))]
    summary = run.exp.read(p)
    font_manager.fontManager.addfont(str(REPO / "results/pdv2_sequence_report_20261018/UnDotum.ttf"))
    plt.rcParams.update({"font.family": ["UnDotum", "DejaVu Sans"], "axes.unicode_minus": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, metric in zip(axes[:2], ("AP", "R@100P")):
        vals = [summary["mean_std"][m][run.METHOD][metric] for m in ("capacity_point", "standard")]
        for i, (row, color) in enumerate(zip(vals, ("#888888", "#316ba8"))):
            ax.errorbar(i, row["mean"] * 100, yerr=row["std"] * 100,
                        fmt="o", color=color, capsize=5)
            ax.annotate(f'{row["mean"]*100:.3f}', (i, (row["mean"]+row["std"])*100),
                        xytext=(0, 10), textcoords="offset points", ha="center")
        ax.set_xticks([0, 1], ["시간 간선 없음", "시간 간선 사용"])
        ax.set_xlim(-.5, 1.5)
        ax.margins(y=.4)
        ax.set_title(f"최종 {metric} (%)")
        ax.grid(axis="y", alpha=.2)
    names = {"karlsruhe": "카를스루에", "seoul": "서울", "daejeon": "대전",
             "ann_arbor": "앤아버", "stockholm": "스톡홀름", "hamburg": "함부르크"}
    cities = summary["city_temporal_minus_point_pp"]
    values = [cities[c]["AP"] for c in cities]
    bars = axes[2].barh([names[c] for c in cities], values,
                       color=["#316ba8" if v >= 0 else "#be703c" for v in values])
    axes[2].bar_label(bars, fmt="%+.3f", padding=4)
    axes[2].axvline(0, color="black", linewidth=.7)
    axes[2].margins(x=.35)
    axes[2].set_title("도시별 AP 차이 (%p)\n시간 간선 사용 - 없음")
    axes[2].grid(axis="x", alpha=.2)
    fig.suptitle("시험 46시퀀스 · 같은 최종 판정 · 3시드 평균±표본 표준편차")
    fig.tight_layout(rect=(0, 0, 1, .92))
    paths = []
    for ext in ("png", "pdf"):
        path = run.OUT / f"temporal_test_control_final.{ext}"
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    write_json(run.OUT / "plot_validation.json", {
        "summary_sha256": sha256(p), "script_sha256": sha256(Path(__file__)),
        "artifacts": {str(p.relative_to(REPO)): sha256(p) for p in paths},
        "reason": "수치·구성은 유지하고 제목의 지원되지 않는 빼기 기호를 수정한다.",
    })


if __name__ == "__main__":
    main()
