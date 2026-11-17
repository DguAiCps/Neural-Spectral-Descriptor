#!/usr/bin/env python3
"""scripts/_rawmag_probe_20261114.py 결과를 주파수 대역별로 집계하고 그림을 그린다.

변화 비율 = 같은 장소 조건의 평균 제곱차 / 다른 장소(20 m 초과) 쌍의 평균 제곱차.
0이면 변화 없음, 1이면 다른 장소만큼 달라진 것이다. 호스트에서 돈다.

    python3 scripts/_rawmag_probe_report_20261114.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results/rawmag_probe_20261114"
SEQS = ["kitti360_0004", "helipr_Roundabout01", "mulran_Sejong03"]
REPS = {"range": "Range image 16x360, azimuth |DFT|",
        "polar": "Polar occupancy 79x360, azimuth |DFT|",
        "cart": "Cartesian occupancy 160x160, 2-D |DFT|"}
BANDS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 180)]
CONDS = ["yaw_0.5", "yaw_5", "yaw_30", "fwd_0.5", "fwd_1.0", "fwd_2.0", "fwd_4.0",
         "lat_0.5", "lat_1.0", "lat_2.0", "lat_4.0", "se3_real", "real", "adjacent"]
LINES = [("lat_0.5", "#86b6ef", "-", 2, "shift 0.5 m"),
         ("lat_1.0", "#5598e7", "-", 2, "shift 1 m"),
         ("lat_2.0", "#256abf", "-", 2, "shift 2 m"),
         ("lat_4.0", "#0d366b", "-", 2, "shift 4 m"),
         ("yaw_0.5", "#eb6834", "-", 2, "yaw 0.5 deg"),
         ("yaw_5", "#eb6834", "--", 2, "yaw 5 deg"),
         ("se3_real", "#1baf7a", "-", 2, "same points, revisit pose"),
         ("real", "#0b0b0b", "-", 3, "real revisit")]


_k = np.fft.fftfreq(160) * 160  # scripts/_rawmag_probe_20261114.py의 RADIAL과 같다
RADIAL = np.minimum(np.rint(np.hypot(_k[:, None], _k[None, :])).astype(int), 80)


def bands(rep):
    return [b for b in BANDS if b[0] <= (80 if rep == "cart" else 180)]


def summarize(name):
    z = np.load(OUT / f"{name}.npz")
    ok = z["meta"][:, 5] >= 0.6
    xyz = z["xyz"]
    far = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) > 20.0
    res = {"n_pairs": int(ok.sum()), "reps": {}}
    for rep in REPS:
        base = z[f"base_{rep}"].astype(np.float64)
        flat = base.reshape(len(base), -1)
        # 다른 장소 쌍의 주파수별 제곱차
        ii, jj = np.nonzero(np.triu(far, 1))
        pick = np.random.default_rng(0).choice(len(ii), min(4000, len(ii)), replace=False)
        diff2 = (base[ii[pick]] - base[jj[pick]]) ** 2
        if rep == "cart":
            between = np.stack([np.bincount(RADIAL.ravel(), weights=d.ravel(), minlength=81)
                                for d in diff2]).mean(0)
        else:
            between = diff2.sum(1).mean(0)
        # 전체 벡터(DC 제외 없음, 정규화 없음)의 유클리드 거리로 본 혼동률
        g = flat @ flat.T
        n2 = np.diag(g)
        negd = n2[:, None] + n2[None] - 2 * g
        r = {"between_per_band": [], "conds": {}}
        bs = bands(rep)
        r["bands"] = bs
        r["between_per_band"] = [float(between[a:b + 1].sum()) for a, b in bs]
        for c in CONDS:
            sq = z[f"sq_{rep}_{c}"].astype(np.float64)
            w = sq[ok].mean(0)
            tot = sq.sum(1)
            conf = [(negd[i][far[i]] < tot[i]).mean() for i in np.flatnonzero(ok)
                    if far[i].any()]
            r["conds"][c] = {
                "ratio_per_band": [float(w[a:b + 1].sum() / between[a:b + 1].sum())
                                   for a, b in bs],
                "ratio_dc": float(w[0] / between[0]),
                "ratio_all_no_dc": float(w[1:].sum() / between[1:].sum()),
                "confusion_rate": float(np.mean(conf)),
            }
        res["reps"][rep] = r
    return res


def main():
    summary = {s: summarize(s) for s in SEQS}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": "#52514e", "xtick.color": "#898781",
                         "ytick.color": "#898781", "text.color": "#0b0b0b"})
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharey=True, facecolor="#fcfcfb")
    for i, s in enumerate(SEQS):
        for k, rep in enumerate(REPS):
            ax = axes[i, k]
            ax.set_facecolor("#fcfcfb")
            r = summary[s]["reps"][rep]
            x = [np.sqrt(a * b) for a, b in r["bands"]]
            ax.axhline(1.0, color="#898781", lw=1, ls=":")
            for c, col, ls, lw, lab in LINES:
                if c == "yaw_5" and rep != "cart":
                    continue
                y = np.maximum(r["conds"][c]["ratio_per_band"], 1e-3)
                ax.plot(x, y, color=col, ls=ls, lw=lw, marker="o", ms=3.5, label=lab)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 3)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{a}" if a == b else f"{a}-{b}" for a, b in r["bands"]],
                               fontsize=7)
            ax.grid(True, color="#e1e0d9", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(REPS[rep], fontsize=10, loc="left")
            if k == 0:
                ax.set_ylabel(f"{s}\nchange / different-place change")
            if i == 2:
                ax.set_xlabel("radial frequency band (cycles per 160 m)" if rep == "cart"
                              else "azimuth frequency band (cycles per turn)")
    axes[0, 2].text(0.98, 0.93, "1 = as different as another place",
                    transform=axes[0, 2].transAxes, ha="right", fontsize=8, color="#52514e")
    h, l = axes[0, 2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, frameon=False, fontsize=9)
    fig.suptitle("Raw magnitude spectrum: change under a transform of the same scan "
                 "vs. a real revisit", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / "rawmag_change_by_band.png", dpi=130)

    for s in SEQS:
        print(f"\n== {s} (n={summary[s]['n_pairs']})")
        for rep in REPS:
            r = summary[s]["reps"][rep]
            print(f" [{rep}] bands {r['bands']}")
            for c in CONDS:
                v = r["conds"][c]
                print(f"  {c:9s} DC {v['ratio_dc']:.3f} | "
                      + " ".join(f"{t:.3f}" for t in v["ratio_per_band"])
                      + f" | noDC {v['ratio_all_no_dc']:.3f} conf {100*v['confusion_rate']:.1f}%")


if __name__ == "__main__":
    main()
