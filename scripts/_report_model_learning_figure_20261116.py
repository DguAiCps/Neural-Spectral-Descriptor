#!/usr/bin/env python3
"""학습 기여 그림(fig_learning_gain.pdf).

NSD(시드 4·5·6) − 균등 감산의 짝지은 차이를 지표 4개 × 부분집합 3개(전체·순방향 전용·역방향 전용)로 보이는 forest plot.
점: 세 학습 시드의 평균 차이, 막대: 모델 고정·도시 안 시퀀스 재표집 95% 구간(10,000회), 작은 표식: 시드별 차이.
원천: results/pdv2_report_model_20261116/analysis/main/{direction_summary,direction_sequence_summary}.json
(_report_model_main_tables_20261116.py 산출). 양식(글꼴·색·폭)은 기존 _paper_learning_figure_v23.py와 맞춘다.
출력: analysis/figures/fig_learning_gain.{pdf,png}, analysis/learning/learning_figure.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_model_common_20261116 as C  # noqa: E402

NAVY, EDGE, GRID = "#0E2841", "#6D7780", "#ECEFF1"
SUBSETS = (("all", "All revisits", NAVY, "D"), ("forward_only", "Forward-only", "#546578", "o"), ("reverse_only", "Reverse-only", "#156082", "s"))
METRIC_ROWS = (("AP", "AP"), ("max_F1", "Max F1"), ("R@100P", "R@100P"), ("R@1", "Revisit R@1"))


def collect():
    ds = json.loads((C.OUT / "main/direction_summary.json").read_text())
    iv = json.loads((C.OUT / "main/direction_sequence_summary.json").read_text())["comparisons"]["diff-fixed"]
    uni = ds["methods"]["fixed"]["per_seed"]["0"]
    out = {}
    for g, *_ in SUBSETS:
        out[g] = {}
        for k, _ in METRIC_ROWS:
            runs = [100 * (v[g]["macro"][k] - uni[g]["macro"][k]) for v in ds["methods"]["diff"]["per_seed"].values()]
            o = iv[g]["overall"][k]
            assert abs(np.mean(runs) - o["delta_pp"]) < 1e-9
            out[g][k] = dict(mean_pp=o["delta_pp"], interval=o["ci95_pp"], per_run=runs, positive_runs=int(sum(r > 0 for r in runs)))
    return out


def draw(d, stem):
    plt.rcParams.update({"font.family": "Carlito", "font.size": 7.9, "pdf.fonttype": 42, "text.color": NAVY, "axes.labelcolor": NAVY,
                         "axes.edgecolor": EDGE, "axes.linewidth": .6, "xtick.color": NAVY, "ytick.color": NAVY})
    fig = plt.figure(figsize=(3.25, 1.95))
    a = fig.add_axes([.205, .20, .70, .70])
    a.axvline(0, color=EDGE, lw=.7, ls=(0, (3, 2)), zorder=2)
    step, gap = 1.0, .27
    yt, yl = [], []
    for r, (k, lab) in enumerate(METRIC_ROWS[::-1]):
        y0 = r * step
        yt.append(y0); yl.append(lab)
        if r:
            a.axhline(y0 - step / 2, color=GRID, lw=.6, zorder=0)
        for j, (g, glab, col, mk) in enumerate(SUBSETS):
            y = y0 + (1 - j) * gap
            m = d[g][k]; lo, hi = m["interval"]
            a.plot([lo, hi], [y, y], color=col, lw=1.0, zorder=3, solid_capstyle="butt")
            for x in (lo, hi):
                a.plot([x, x], [y - .06, y + .06], color=col, lw=1.0, zorder=3)
            a.scatter(m["per_run"], np.full(3, y + .11), s=5, marker="v", facecolor=col, edgecolor="none", zorder=4, alpha=.8)
            a.scatter([m["mean_pp"]], [y], s=20 if mk != "D" else 22, marker=mk, color=col, edgecolor="white", linewidth=.4, zorder=5,
                      label=glab if r == 0 else None)
            a.text(1.015, y, f"{m['mean_pp']:+.2f}".replace("-", "−"), transform=a.get_yaxis_transform(),
                   ha="left", va="center", fontsize=6.5, color=col)
    a.set(yticks=yt, yticklabels=yl, xlim=(-1.3, 7.3), ylim=(-.5, len(METRIC_ROWS) - .5), xticks=[0, 2, 4, 6],
          xlabel="NSD minus uniform subtraction (points)")
    a.tick_params(labelsize=7.4, length=2, pad=2)
    a.tick_params(axis="y", length=0)
    a.grid(axis="x", color=GRID, lw=.45); a.set_axisbelow(True)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.9, loc="upper center", bbox_to_anchor=(.55, 1.01), ncol=3, handletextpad=.2, columnspacing=.9)
    a.text(1.015, len(METRIC_ROWS) - .5, "mean", transform=a.get_yaxis_transform(), ha="left", va="bottom", fontsize=6.2, color=EDGE)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", default=None, help="그림 출력 폴더(기본 analysis/figures). 예: docs/paper/body_v2.4/figs/pair_analysis")
    args = ap.parse_args()
    fig_dir = Path(args.fig_dir) if args.fig_dir else C.OUT / "figures"
    d = collect()
    C.write(C.OUT / "learning" / "learning_figure.json", dict(
        data=d, seeds=list(C.NEW_SEEDS),
        definition="NSD(시드 4·5·6 평균) − 균등 감산(재선택 설정). 구간: 모델 고정, 도시 안 시퀀스 재표집 10,000회(seed 20261104), 6도시 동일 비중."))
    fig_dir.mkdir(parents=True, exist_ok=True)
    draw(d, fig_dir / "fig_learning_gain")
    for g in d:
        print(g, {k: (round(v["mean_pp"], 2), [round(x, 2) for x in v["interval"]], v["positive_runs"]) for k, v in d[g].items()})


if __name__ == "__main__":
    main()
