#!/usr/bin/env python3
"""앨리어싱 기울기 그림 시안: 고르는 단계 먼 곳 반환율 → 재현율 50%에서 수락된 루프 중 먼 곳 비율(11방법).
원천: results/pdv2_aliasing_accept_20261117/summary.json (scripts/_aliasing_accept_20261117.py)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/pdv2_aliasing_accept_20261117"
d = json.loads((R / "summary.json").read_text())["methods"]
LABEL = {"scpp": "SC++", "fresco": "FreSCo", "ring": "RING", "ringpp": "RING++", "disco": "DiSCO", "deepring": "DeepRING",
         "ringsharp": "RING#-L", "helios": "HeLiOS", "seqot": "SeqOT", "bevplace": "BEVPlace++", "nsd": "NSD"}
INK, GRAY, NSD, BEV = "#1A1A1A", "#B5BAC0", "#C0392B", "#2E6DA4"
plt.rcParams.update({"font.family": "Nimbus Roman", "font.size": 7.5, "pdf.fonttype": 42, "axes.edgecolor": "#8C8C8C", "axes.linewidth": .5})
fig, a = plt.subplots(figsize=(3.25, 2.1))
for m, v in d.items():
    y0, y1 = v["mean"]["far"], v["mean"]["alias@0.5"]
    col, lw, z = (NSD, 1.6, 4) if m == "nsd" else (BEV, 1.2, 3) if m == "bevplace" else (GRAY, .8, 1)
    a.plot([0, 1], [y0, y1], color=col, lw=lw, marker="o", ms=3, zorder=z)
    if m in ("nsd", "bevplace"):
        a.text(1.04, y1, LABEL[m], color=col, va="center", fontsize=7.2, fontweight="bold" if m == "nsd" else "normal")
a.set_yscale("log"); a.set_ylim(0.4, 50)
a.set_yticks([0.5, 1, 2, 5, 10, 20, 40]); a.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "40"])
a.set_xticks([0, 1], ["Selection\n(revisit queries)", "Acceptance\n(accepted loops, 50% recall)"])
a.set_xlim(-0.25, 1.35); a.set_ylabel("Far returns (>25 m, %)")
a.grid(axis="y", color="#ECEFF1", lw=.5); a.set_axisbelow(True)
a.plot([], [], color=GRAY, lw=.8, label="Other baselines"); a.legend(frameon=False, fontsize=6.6, loc="upper right")
fig.tight_layout(pad=.3)
for ext in ("pdf", "png"):
    fig.savefig(R / f"figures/fig_aliasing_slope.{ext}", dpi=300)
print("ok")
