#!/usr/bin/env python3
"""궤적 루프 폐합 지도(11방법): 사전 고정 시퀀스 HeLiPR Town03에서 각 방법이 R@100P 동작점에 수락하는 순방향·역방향 재방문.

선택 규칙(그리기 전에 고정): 순방향 전용·역방향 전용 재방문 수의 작은 쪽이 가장 큰 시험 시퀀스 = HeLiPR Town03.
실행: 학습 방법은 각 방법의 첫 시드(NSD 시드 4, 비교 방법 시드 1), 학습 없는 방법은 1회.
동작점: 시퀀스별 R@100P 기준값 = max(음성 점수, 틀린 장소를 돌려준 양성 점수). 양성 = 과거 5 m 안, 음성 = 25 m 밖.
검산: 수락 수/양성 수를 같은 원장의 pr_sweep R@100P(_report_model_common_20261116.seq_metric)와 대조한다.
구성은 Codex 시안(results/pdv2_loop_map_20261117/codex/out/make_loop_map.py)의 방향별 색 구분을 따른다.
출력: results/pdv2_loop_map_20261117/claude/loop_map_all_Town03.{pdf,png}, counts_Town03.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _report_model_common_20261116 as C  # noqa: E402

SEQ = "helipr/Town03"
NAME = SEQ.replace("/", "_")
OUT = ROOT / "results/pdv2_loop_map_20261117/claude"
SUMMARY = ROOT / "results/pdv2_deepring_scaleinit_integration_20261110/baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json"
ORDER = ["scpp", "fresco", "ring", "ringpp", "disco", "deepring", "ringsharp", "helios", "seqot", "bevplace", "nsd"]
LABEL = {"scpp": "SC++", "fresco": "FreSCo", "ring": "RING", "ringpp": "RING++", "disco": "DiSCO", "deepring": "DeepRING",
         "ringsharp": "RING#-L", "helios": "HeLiOS", "seqot": "SeqOT", "bevplace": "BEVPlace++", "nsd": "NSD"}
PREFIX = {"scpp": "dump_scpp", "fresco": "dump_fresco", "ring": "dump_ring", "ringpp": "dump_ringpp_i2", "disco": "dump_disco",
          "deepring": "dump_deepring", "ringsharp": "dump_ringsharp", "helios": "dump_helios", "seqot": "dump_seqotl2",
          "bevplace": "dump_bevplace"}
FWD, REV, MISS, TRACK, INK, EDGE = "#0072B2", "#D55E00", "#9AA1A8", "#E6E8EB", "#1A1A1A", "#8C8C8C"


def ledgers():
    s = json.loads(SUMMARY.read_text())
    out = {}
    for m in ORDER:
        if m == "nsd":
            d = C.load("new:4", SEQ)
            out[m] = dict(q=d["q"], score=d["score"], top1=d["top1"], gt=d["gt"], seed="4")
            continue
        e = s["methods"][m]
        seed = list(e["per_seed"])[0]
        with np.load(ROOT / e["per_seed"][seed]["source"] / f"{PREFIX[m]}_{NAME}.npz") as z:
            out[m] = dict(q=z["query_idx"], score=z[e["score"]].astype(float), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"], seed=seed)
    return out


def main():
    pos = np.load(ROOT / f"results/pdv2_loop_map_20261117/positions/{NAME}.npz")["pos"][:, :2]
    with np.load(C.DIRS / f"{NAME}.npz") as d:
        q, grp = d["query_idx"], d["group"]
    led = ledgers()
    res, counts = {}, {}
    for m, d in led.items():
        ix = np.searchsorted(d["q"], q)
        assert np.array_equal(d["q"][ix], q), m
        s, top1, gt = d["score"][ix], d["top1"][ix], d["gt"][ix]
        p, n = gt <= 5, gt > 25
        thr = max(s[n].max(initial=-np.inf), s[p & (top1 > 5)].max(initial=-np.inf))
        acc = p & (top1 <= 5) & (s > thr)
        ref = C.seq_metric(dict(score=s, top1=top1, gt=gt))["R@100P"]
        assert abs(acc.sum() / p.sum() - ref) < 1e-12, (m, acc.sum() / p.sum(), ref)
        res[m] = (p, acc)
        counts[m] = dict(seed=d["seed"], R100P=100 * ref,
                         forward=[int((acc & (grp == 0)).sum()), int((p & (grp == 0)).sum())],
                         reverse=[int((acc & (grp == 1)).sum()), int((p & (grp == 1)).sum())])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "counts_Town03.json").write_text(json.dumps(dict(sequence=SEQ, operating_point="per-sequence R@100P", methods=counts), indent=1))

    plt.rcParams.update({"font.family": "Nimbus Roman", "font.size": 7, "pdf.fonttype": 42, "text.color": INK,
                         "axes.edgecolor": "#B5BAC0", "axes.linewidth": .5})
    xy = (pos - pos.min(0))[:, ::-1] * np.array([1, -1])  # Codex 단일 열 판과 같은 90° 회전(가로로 긴 배치)
    xy -= xy.min(0)
    span = xy.max(0)
    W, ncol, nrow, gap, mx = 6.9, 6, 2, .035, .04  # 인치: 그림 폭, 칸 사이, 칸 안 여백 비율
    cw = (W - gap * (ncol - 1)) / ncol
    ch = cw * span[1] / span[0] + .24  # 지도 + 위(이름)·아래(수치) 글줄
    H = nrow * ch + gap * (nrow - 1)
    fig = plt.figure(figsize=(W, H))
    axes = []
    for i in range(nrow * ncol):
        r, c = divmod(i, ncol)
        axes.append(fig.add_axes([(c * (cw + gap)) / W, 1 - ((r + 1) * ch + r * gap) / H, cw / W, ch / H]))
    pad_x = span[0] * mx
    sx = (span[0] + 2 * pad_x)  # 칸 폭에 맞춘 데이터 폭
    sy = sx * ch / cw  # 같은 축척의 칸 높이
    for a, m in zip(axes, ORDER):
        p, acc = res[m]
        a.set_xlim(-pad_x, span[0] + pad_x)
        y_mid = span[1] / 2
        a.set_ylim(y_mid - sy / 2 - .02 * sy, y_mid + sy / 2 - .02 * sy)
        a.plot(xy[:, 0], xy[:, 1], color=TRACK, lw=.5, zorder=1)
        for k, col in ((0, FWD), (1, REV)):
            sel = p & (grp == k)
            a.scatter(*xy[q[sel & ~acc]].T, s=.4, color=MISS, lw=0, zorder=2, rasterized=True)
            a.scatter(*xy[q[sel & acc]].T, s=.6, color=col, lw=0, zorder=3, rasterized=True)
        cnt = counts[m]
        f, rv = 100 * cnt["forward"][0] / cnt["forward"][1], 100 * cnt["reverse"][0] / cnt["reverse"][1]
        a.text(.03, .955, LABEL[m], transform=a.transAxes, ha="left", va="top", fontsize=7.0, fontweight="bold" if m == "nsd" else "normal")
        a.text(.97, .035, f"F {f:.1f}  R {rv:.1f}", transform=a.transAxes, ha="right", va="bottom", fontsize=6.0)
        a.set_xticks([]); a.set_yticks([])
    leg = axes[-1]
    for sp in leg.spines.values():
        sp.set_visible(False)
    leg.set_xticks([]); leg.set_yticks([])
    leg.set_xlim(axes[0].get_xlim()); leg.set_ylim(axes[0].get_ylim())
    h = [plt.Line2D([], [], ls="", marker="o", ms=2.5, color=FWD, label="Accepted (F)"),
         plt.Line2D([], [], ls="", marker="o", ms=2.5, color=REV, label="Accepted (R)"),
         plt.Line2D([], [], ls="", marker="o", ms=2.5, color=MISS, label="Missed"),
         plt.Line2D([], [], color=TRACK, lw=1.2, label="Trajectory")]
    leg.legend(handles=h, frameon=False, fontsize=6.0, loc="upper left", handletextpad=.2, borderaxespad=.1, labelspacing=.3)
    x0, y0 = span[0] * .05, leg.get_ylim()[0] + .12 * sy
    leg.plot([x0, x0 + 200], [y0, y0], color=INK, lw=.8)
    leg.text(x0 + 220, y0, "200 m", ha="left", va="center", fontsize=6.8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"loop_map_all_Town03.{ext}", dpi=400)
    print("size", round(W, 2), round(H, 2))
    for m in ORDER:
        print(m, counts[m])


if __name__ == "__main__":
    main()
