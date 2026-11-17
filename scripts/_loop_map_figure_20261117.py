#!/usr/bin/env python3
"""궤적 루프 폐합 지도 시안: 한 시험 시퀀스에서 NSD(시드 4)와 BEVPlace++(시드 1)가 R@100P 동작점에서 수락하는 재방문.

선택 규칙(그리기 전에 고정): 순방향 전용·역방향 전용 재방문 수의 작은 쪽이 가장 큰 시험 시퀀스 = HeLiPR Town03.
동작점: 시퀀스별 R@100P 기준값 = max(음성 점수, 틀린 장소를 돌려준 양성 점수). 양성 = 과거 5 m 안, 음성 = 25 m 밖.
위치는 정답 자세(표시에만 사용). 원천: results/pdv2_loop_map_20261117/codex/data/ (원장 사본).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/pdv2_loop_map_20261117/codex/data"
OUT = ROOT / "results/pdv2_loop_map_20261117/claude"
NAVY, EDGE, TRACK = "#0E2841", "#6D7780", "#D5DBE0"
OK, MISS = "#1F7A4D", "#E07B39"


def load(seq):
    pos = np.load(DATA / f"positions_{seq}.npz")["pos"][:, :2]
    d = np.load(DATA / f"direction_{seq}.npz")
    q, grp = d["query_idx"], d["group"]
    out = {}
    for name, f, key in (("BEVPlace++", f"bevplace_{seq}.npz", "score_negl2"), ("NSD", f"nsd_{seq}.npz", "score")):
        z = np.load(DATA / f)
        ix = np.searchsorted(z["query_idx"], q)
        assert np.array_equal(z["query_idx"][ix], q)
        s, top1, gt = z[key][ix].astype(float), z["top1_geo_dist"][ix], z["min_past_gt_dist"][ix]
        pos_q, neg = gt <= 5, gt > 25
        thr = max(s[neg].max(initial=-np.inf), s[pos_q & (top1 > 5)].max(initial=-np.inf))
        acc = pos_q & (top1 <= 5) & (s > thr)
        out[name] = dict(acc=acc, pos=pos_q, j=z["j_star"][ix])
        print(seq, name, "R@100P", round(100 * acc.sum() / pos_q.sum(), 2),
              {g: f"{(acc & (grp == k)).sum()}/{(pos_q & (grp == k)).sum()}" for g, k in (("fwd", 0), ("rev", 1))})
    return pos, q, grp, out


def draw(seq, stem):
    pos, q, grp, res = load(seq)
    plt.rcParams.update({"font.family": "Carlito", "font.size": 7.6, "pdf.fonttype": 42, "text.color": NAVY,
                         "axes.edgecolor": EDGE, "axes.linewidth": .5})
    p0 = pos - pos.min(0)
    fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.9))
    for a, name in zip(axes, ("BEVPlace++", "NSD")):
        r = res[name]
        a.plot(p0[:, 0], p0[:, 1], color=TRACK, lw=.6, zorder=1)
        for k, mk in ((0, "o"), (1, "s")):
            m = r["pos"] & (grp == k)
            a.scatter(p0[q[m & ~r["acc"]], 0], p0[q[m & ~r["acc"]], 1], s=1.3, marker=mk, color=MISS, lw=0, zorder=2)
            a.scatter(p0[q[m & r["acc"]], 0], p0[q[m & r["acc"]], 1], s=1.3, marker=mk, color=OK, lw=0, zorder=3)
        fw = f"{(r['acc'] & (grp == 0)).sum() / (r['pos'] & (grp == 0)).sum() * 100:.0f}"
        rv = f"{(r['acc'] & (grp == 1)).sum() / (r['pos'] & (grp == 1)).sum() * 100:.0f}"
        a.set_title(f"{name}\nforward {fw}%, reverse {rv}% accepted", fontsize=7.2, pad=2)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    span = p0.max(0)
    axes[0].plot([span[0] * .05, span[0] * .05 + 200], [-span[1] * .04] * 2, color=NAVY, lw=1)
    axes[0].text(span[0] * .05 + 100, -span[1] * .04 - span[1] * .02, "200 m", ha="center", va="top", fontsize=6.3)
    h = [plt.Line2D([], [], ls="", marker="o", ms=3, color=OK, label="accepted"),
         plt.Line2D([], [], ls="", marker="o", ms=3, color=MISS, label="missed")]
    fig.legend(handles=h, frameon=False, fontsize=6.6, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.01), handletextpad=.1)
    fig.subplots_adjust(left=.01, right=.99, top=.84, bottom=.1, wspace=.04)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)


if __name__ == "__main__":
    for seq in (sys.argv[1:] or ["helipr_Town03"]):
        draw(seq, f"loop_map_{seq}")
