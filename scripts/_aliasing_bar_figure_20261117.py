#!/usr/bin/env python3
"""앨리어싱 막대 그림 시안: 방법별로 고르는 단계 먼 곳 반환율과 재현율 50%에서 수락된 루프 중 먼 곳 비율을 나란히(11방법).
원천: results/pdv2_aliasing_accept_20261117/summary.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/pdv2_aliasing_accept_20261117"
d = json.loads((R / "summary.json").read_text())["methods"]
LABEL = {"scpp": "SC++", "fresco": "FreSCo", "ring": "RING", "ringpp": "RING++", "disco": "DiSCO", "deepring": "DeepRING",
         "ringsharp": "RING#-L", "helios": "HeLiOS", "seqot": "SeqOT", "bevplace": "BEVPlace++", "nsd": "NSD"}
order = sorted(d, key=lambda m: d[m]["mean"]["alias@0.5"])
sel = np.array([d[m]["mean"]["far"] for m in order]); acc = np.array([d[m]["mean"]["alias@0.5"] for m in order])
INK, LIGHT, DARK, HL = "#1A1A1A", "#C9D3DD", "#3B5B7A", "#C0392B"
plt.rcParams.update({"font.family": "Nimbus Roman", "font.size": 7.5, "pdf.fonttype": 42, "axes.edgecolor": "#8C8C8C", "axes.linewidth": .5})
fig, a = plt.subplots(figsize=(3.25, 2.5))
y = np.arange(len(order))[::-1]; h = .38
a.barh(y + h / 2, sel, h, color=LIGHT, label="Selection: revisit queries")
a.barh(y - h / 2, acc, h, color=[HL if m == "nsd" else DARK for m in order], label="Acceptance: accepted loops at 50% recall")
for yi, s_, c_ in zip(y, sel, acc):
    a.text(c_ + .4, yi - h / 2, f"{c_:.1f}", va="center", fontsize=6.2, color=INK)
a.set_yticks(y, [LABEL[m] for m in order])
for t, m in zip(a.get_yticklabels(), order):
    if m == "nsd": t.set_fontweight("bold"); t.set_color(HL)
a.set_xlabel("Aliased matches (%)")
a.set_xlim(0, 40); a.grid(axis="x", color="#ECEFF1", lw=.5); a.set_axisbelow(True)
for sp in ("top", "right"): a.spines[sp].set_visible(False)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=LIGHT, label="Place selection (revisit queries)"), Patch(color=DARK, label="Accepted loops (50% recall)")], frameon=False, fontsize=6.4, loc="upper center", ncol=2, bbox_to_anchor=(.55, 1.0), handlelength=1.2, columnspacing=1.0)
fig.tight_layout(pad=.3, rect=(0, 0, 1, .93))
for ext in ("pdf", "png"):
    fig.savefig(R / f"figures/fig_aliasing_bars.{ext}", dpi=300)
print([LABEL[m] for m in order])
