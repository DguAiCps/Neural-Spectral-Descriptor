#!/usr/bin/env python3
"""옛 fig_geodesc의 현재 구조판: 재방문 쿼리의 이중 색인 후보(최대 200개)마다 지리 거리 대 점수.
위: 색인 점수(정렬 없음), 아래: 최종 선택 점수(정제 스펙트럼의 순환 정렬 포함). 열: KITTI 00, HeLiPR Town03.
점수는 쿼리 안 min-max 항의 가중 평균(0~1). 각 패널에 1위 후보가 25 m 밖인 쿼리 비율을 적는다.
원천: results/pdv2_geodesc_v24_20261117/*.npz (_geodesc_v24_export_20261117.py)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/pdv2_geodesc_v24_20261117"
OK, MID, FAR, INK = "#009E73", "#C4C8CC", "#D55E00", "#1A1A1A"
SEQS = (("kitti_00", "KITTI 00"), ("helipr_Town03", "HeLiPR Town03"))

plt.rcParams.update({"font.family": "Nimbus Roman", "font.size": 7, "pdf.fonttype": 42, "text.color": INK,
                     "axes.edgecolor": "#8C9096", "axes.linewidth": .5, "xtick.major.width": .4, "ytick.major.width": .4,
                     "xtick.major.size": 2, "ytick.major.size": 2, "xtick.color": INK, "ytick.color": INK})
fig, axes = plt.subplots(2, 2, figsize=(3.3, 2.55), sharex=True, sharey=True)
fig.subplots_adjust(left=.13, right=.99, top=.92, bottom=.15, wspace=.06, hspace=.12)
rng = np.random.default_rng(0)
for c, (name, title) in enumerate(SEQS):
    d = np.load(SRC / f"{name}.npz")
    g = np.maximum(d["geo"], .1)
    for r, key in enumerate(("index", "final")):
        a = axes[r, c]
        s = d[key]
        order = rng.permutation(len(g))
        col = np.where(g[order] <= 5, OK, np.where(g[order] >= 25, FAR, MID))
        a.scatter(g[order], s[order], s=.25, c=col, lw=0, alpha=.35, rasterized=True)
        # 쿼리별 1위 후보
        qs = d["query"]
        top = np.array([np.flatnonzero(qs == t)[np.argmax(s[qs == t])] for t in np.unique(qs)])
        far = np.mean(g[top] >= 25) * 100
        a.text(.97, .05, f"top-1 ≥ 25 m: {far:.1f}%", transform=a.transAxes, ha="right", va="bottom", fontsize=6.4,
               bbox=dict(boxstyle="square,pad=.15", fc="white", ec="none", alpha=.85))
        a.axvline(5, color=INK, lw=.4, ls=":"); a.axvline(25, color=INK, lw=.4, ls=":")
        a.set_xscale("log"); a.set_xlim(.1, 2000); a.set_ylim(-.02, 1.02)
        if r == 0:
            a.set_title(title, fontsize=7.4, pad=2)
        if c == 0:
            a.set_ylabel(("Index score" if r == 0 else "Final score"), fontsize=7)
axes[0, 0].text(5, 1.0, "5 m", fontsize=5.8, ha="right", va="top")
axes[0, 0].text(25, 1.0, " 25 m", fontsize=5.8, ha="left", va="top")
fig.supxlabel("Geographic distance to candidate (m)", fontsize=7, y=.02)
out = ROOT / "results/pdv2_loop_map_20261117/claude/fig_geodesc_v24"
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=400)
print(out)
