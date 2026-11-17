#!/usr/bin/env python3
"""학습된 DiffAttnConv 주의 가중치 그림 후보(fig_attention_weights.pdf).

(a) 이웃 오프셋(−5…+5)별 평균 주의 가중치(헤드 평균, 시험 내부 키프레임·시드 4·5·6 평균, 막대 = 시드 SD), 층 1·2, 균등 0.1 점선.
(b) 키프레임별 층 1의 −5·+5 가중치 결합 분포(육각 빈 로그 색, ×=균등 감산, ○=평균). raw/*.npz 사용.
원천: results/pdv2_report_model_20261116/analysis/attention/summary.json(원장과 비트 단위 일치 검사 checks.json).
양식은 _report_model_learning_figure_20261116.py와 같다.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/pdv2_report_model_20261116/analysis/attention/summary.json"
OUT = ROOT / "results/pdv2_report_model_20261116/analysis/figures/fig_attention_weights"
NAVY, EDGE, GRID = "#0E2841", "#6D7780", "#ECEFF1"
L1, L2 = "#156082", "#9AA8B5"

d = json.loads(SRC.read_text())
plt.rcParams.update({"font.family": "Carlito", "font.size": 7.9, "pdf.fonttype": 42, "text.color": NAVY, "axes.labelcolor": NAVY,
                     "axes.edgecolor": EDGE, "axes.linewidth": .6, "xtick.color": NAVY, "ytick.color": NAVY})
fig = plt.figure(figsize=(3.25, 1.55))
a = fig.add_axes([.125, .25, .365, .64])
b = fig.add_axes([.665, .25, .30, .64])

off = np.array(d["offsets"])
x = np.arange(len(off))
w = .38
for i, (L, col, lab) in enumerate((("layer1", L1, "Layer 1"), ("layer2", L2, "Layer 2"))):
    p = d["offset_profile"][L]["head_mean"]
    a.bar(x + (i - .5) * w, p["mean"], w, yerr=p["seed_sd"], color=col, label=lab, error_kw=dict(lw=.5, capsize=0, ecolor=NAVY), zorder=3)
a.axhline(d["uniform"], color=EDGE, lw=.7, ls=(0, (3, 2)), zorder=4)
a.text(4.5, d["uniform"] + .012, "uniform", ha="center", va="bottom", fontsize=6.3, color=EDGE)
a.set_xticks(x, [f"{o:+d}".replace("-", "−") for o in off], fontsize=6.3)
a.set_xlabel("Neighbor offset (keyframes)")
a.set_ylabel("Attention weight", labelpad=2)
a.set_ylim(0, .34)
a.grid(axis="y", color=GRID, lw=.45); a.set_axisbelow(True)
a.set_title("(a) By offset", fontsize=7.6, pad=3)

import glob
pm, pp = [], []
for f in sorted(glob.glob(str(SRC.parent / "raw" / "*.npz"))):
    z = np.load(f)
    inter = z["mask"].sum(1) == 10  # 이웃 10개가 모두 있는 키프레임
    w1 = z["att"][:, 0].mean(-1).mean(0)[inter]  # 층 1, 헤드·시드 평균
    pm.append(w1[:, 0]); pp.append(w1[:, 10])
pm, pp = np.concatenate(pm), np.concatenate(pp)
b.hexbin(pm, pp, gridsize=36, extent=(0, .7, 0, .7), bins="log", cmap="Blues", mincnt=1, linewidths=0)
b.plot([.1], [.1], marker="x", color=EDGE, ms=4, mew=1)
b.text(.13, .07, "uniform", fontsize=6.3, color=EDGE)
b.plot([pm.mean()], [pp.mean()], marker="o", mfc="none", mec=NAVY, ms=4.5, mew=.8)
b.text(pm.mean() + .035, pp.mean() + .035, "mean", fontsize=6.3, color=NAVY)
b.set_xlim(0, .7); b.set_ylim(0, .7)
b.set_xticks([0, .2, .4, .6]); b.set_yticks([0, .2, .4, .6])
b.set_xlabel("Weight on offset −5", labelpad=1)
b.set_ylabel("Weight on offset +5", labelpad=2)
b.set_title("(b) Per keyframe, layer 1", fontsize=7.6, pad=3)
print("keyframes", len(pm), "corr", round(float(np.corrcoef(pm, pp)[0, 1]), 3))

h, l = a.get_legend_handles_labels()
a.legend(h, l, frameon=False, fontsize=6.3, loc="upper center", ncol=1, handlelength=1.0, handletextpad=.3, borderaxespad=.2)
OUT.parent.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300)
print(OUT)
