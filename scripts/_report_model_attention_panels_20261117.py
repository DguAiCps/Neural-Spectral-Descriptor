#!/usr/bin/env python3
"""주의 가중치 그림 (b) 패널: B1, B2.
B1: 키프레임별 가장 먼 이웃(±5) 가중치 합의 분포(층 1·2, 시드 평균·헤드 평균, 이웃 10개가 모두 있는 키프레임). 균등 감산은 0.2.
B2: KITTI 00 첫 200개 내부 키프레임의 층 1 가중치(오프셋 × 키프레임, 시드·헤드 평균). 구간은 그리기 전에 규칙으로 고정했다.
원천: results/pdv2_report_model_20261116/analysis/attention/raw/*.npz
"""
import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/pdv2_report_model_20261116/analysis/attention/raw"
OUT = ROOT / "results/pdv2_report_model_20261116/analysis/figures"
NAVY, EDGE, GRID = "#0E2841", "#6D7780", "#ECEFF1"
L1, L2 = "#156082", "#9AA8B5"
plt.rcParams.update({"font.family": "Carlito", "font.size": 7.9, "pdf.fonttype": 42, "text.color": NAVY, "axes.labelcolor": NAVY,
                     "axes.edgecolor": EDGE, "axes.linewidth": .6, "xtick.color": NAVY, "ytick.color": NAVY})

far = {0: [], 1: []}
for f in sorted(glob.glob(str(RAW / "*.npz"))):
    z = np.load(f)
    inter = z["mask"].sum(1) == 10
    for L in (0, 1):
        a = z["att"][:, L].mean(-1).mean(0)[inter]
        far[L].append(a[:, [0, 10]].sum(1))
far = {L: np.concatenate(v) for L, v in far.items()}

fig, b = plt.subplots(figsize=(1.6, 1.4))
bins = np.linspace(0, 1, 41)
for L, col, lab in ((0, L1, "Layer 1"), (1, L2, "Layer 2")):
    h, e = np.histogram(far[L], bins)
    b.stairs(h / h.sum(), e, color=col, lw=1.0, label=lab, fill=False)
b.axvline(.2, color=EDGE, lw=.7, ls=(0, (3, 2)))
b.text(.21, b.get_ylim()[1] * .92, "uniform", fontsize=6.3, color=EDGE)
b.set_xlabel("Weight on ±5 neighbors"); b.set_ylabel("Fraction of keyframes")
b.legend(frameon=False, fontsize=6.3)
fig.tight_layout(pad=.3)
fig.savefig(OUT / "panel_b1_far_share.png", dpi=300)

z = np.load(RAW / "kitti_00.npz")
inter = np.where(z["mask"].sum(1) == 10)[0][:200]
a = z["att"][:, 0].mean(-1).mean(0)[inter]
keep = z["mask"][inter[0]] > 0
fig, b = plt.subplots(figsize=(1.6, 1.4))
im = b.imshow(a[:, keep].T, aspect="auto", cmap="Blues", vmin=0, vmax=.5, origin="lower", extent=[0, len(inter), -.5, 9.5])
b.set_yticks(range(10), ["−5", "−4", "−3", "−2", "−1", "+1", "+2", "+3", "+4", "+5"], fontsize=5.8)
b.set_xlabel("Keyframe (KITTI 00)"); b.set_ylabel("Offset")
fig.colorbar(im, ax=b, fraction=.05, pad=.02).ax.tick_params(labelsize=5.8)
fig.tight_layout(pad=.3)
fig.savefig(OUT / "panel_b2_example.png", dpi=300)
print({L: np.quantile(v, [.05, .5, .95]).round(3).tolist() for L, v in far.items()})

# B3: 키프레임별 층 1의 −5·+5 가중치 결합 분포(고정 커널이면 한 점, 균등 감산은 (0.1, 0.1)).
pm, pp = [], []
for f in sorted(glob.glob(str(RAW / "*.npz"))):
    z = np.load(f)
    inter = z["mask"].sum(1) == 10
    a = z["att"][:, 0].mean(-1).mean(0)[inter]
    pm.append(a[:, 0]); pp.append(a[:, 10])
pm, pp = np.concatenate(pm), np.concatenate(pp)
fig, b = plt.subplots(figsize=(1.6, 1.4))
b.hexbin(pm, pp, gridsize=40, extent=(0, .7, 0, .7), bins="log", cmap="Blues", mincnt=1, linewidths=0)
b.plot([.1], [.1], marker="x", color=EDGE, ms=4, mew=1)
b.text(.12, .06, "uniform", fontsize=6.3, color=EDGE)
b.plot([pm.mean()], [pp.mean()], marker="o", mfc="none", mec=NAVY, ms=4, mew=.8)
b.text(pm.mean() + .03, pp.mean() + .03, "mean", fontsize=6.3, color=NAVY)
b.set_xlim(0, .7); b.set_ylim(0, .7); b.set_aspect("equal")
b.set_xlabel("Weight on −5"); b.set_ylabel("Weight on +5")
fig.tight_layout(pad=.3)
fig.savefig(OUT / "panel_b3_past_future.png", dpi=300)
print("corr", np.corrcoef(pm, pp)[0, 1])
