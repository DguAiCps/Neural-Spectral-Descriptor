#!/usr/bin/env python3
"""scripts/_rawmag_fm_probe_20261114.py 결과를 반지름 대역별로 집계하고 그림을 그린다.

직교 격자의 같은 칸 비교(summary.json의 cart)와 나란히 놓는다. 변화 비율과 혼동률의
정의는 scripts/_rawmag_probe_report_20261114.py와 같다. fm_corr는 다른 장소 쌍에도
같은 원형 상관 정렬을 적용한다. 결과 폴더가 root 소유라 컨테이너에서 돈다.

    docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
      python3 scripts/_rawmag_fm_report_20261114.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results/rawmag_probe_20261114"
SEQS = ["kitti360_0004", "helipr_Roundabout01", "mulran_Sejong03"]
BANDS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 80)]
CONDS = ["yaw_0.5", "yaw_5", "yaw_30", "fwd_0.5", "fwd_1.0", "fwd_2.0", "fwd_4.0",
         "lat_0.5", "lat_1.0", "lat_2.0", "lat_4.0", "se3_real", "real", "adjacent"]
TITLES = {"cart": "2-D |DFT|, same-cell comparison",
          "fm_mag": "polar-resampled, angular |DFT|",
          "fm_corr": "polar-resampled, circular correlation"}
LINES = [("lat_0.5", "#86b6ef", "-", 2, "shift 0.5 m"),
         ("lat_4.0", "#0d366b", "-", 2, "shift 4 m"),
         ("yaw_5", "#eb6834", "-", 2, "yaw 5 deg"),
         ("yaw_30", "#eb6834", "--", 2, "yaw 30 deg"),
         ("adjacent", "#898781", "-", 2, "next keyframe"),
         ("se3_real", "#1baf7a", "-", 2, "same points, revisit pose"),
         ("real", "#0b0b0b", "-", 3, "real revisit")]


def band(v):
    return np.array([v[..., a - 1:b].sum(-1) for a, b in BANDS]).T  # 반지름 1 = 색인 0


def summarize(name):
    z = np.load(OUT / f"fm_{name}.npz")
    ok = z["meta"][:, 3] >= 0.6
    far = np.linalg.norm(z["xyz"][:, None] - z["xyz"][None], axis=-1) > 20.0
    F = np.fft.rfft(z["base"].astype(np.float64), axis=2)           # (n, 80, 91)
    n = len(F)
    # 다른 장소 쌍의 반지름별 제곱차 (n, n, 80)
    M = np.abs(F)
    e = (z["base"].astype(np.float64) ** 2).sum(2)                  # (n, 80)
    neg = {"fm_mag": np.empty((n, n, 80)), "fm_corr": np.empty((n, n, 80))}
    for i in range(n):
        neg["fm_mag"][i] = ((M[i][None] - M) ** 2).sum(2)
        c = np.fft.irfft(F[i][None] * np.conj(F), n=180, axis=2)    # (n, 80, 180)
        s = c.sum(1).argmax(1)
        neg["fm_corr"][i] = e[i][None] + e - 2.0 * c[np.arange(n), :, s]
    res = {"n_pairs": int(ok.sum()), "bands": BANDS, "reps": {}}
    for rep in ("fm_mag", "fm_corr"):
        between = band(neg[rep][far].mean(0))
        negtot = neg[rep].sum(2)
        r = {}
        for c in CONDS:
            sq = z[f"sq_{rep}_{c}"].astype(np.float64)
            tot = sq.sum(1)
            conf = [(negtot[i][far[i]] < tot[i]).mean() for i in np.flatnonzero(ok)
                    if far[i].any()]
            r[c] = {"ratio_per_band": (band(sq[ok].mean(0)) / between).tolist(),
                    "ratio_all": float(sq[ok].mean(0).sum() / neg[rep][far].mean(0).sum()),
                    "confusion_rate": float(np.mean(conf))}
        res["reps"][rep] = r
    return res


def main():
    cart = json.loads((OUT / "summary.json").read_text())
    summary = {s: summarize(s) for s in SEQS}
    for s in SEQS:
        c = cart[s]["reps"]["cart"]["conds"]
        summary[s]["reps"]["cart"] = {
            k: {"ratio_per_band": v["ratio_per_band"], "ratio_all": v["ratio_all_no_dc"],
                "confusion_rate": v["confusion_rate"]} for k, v in c.items()}
    (OUT / "fm_summary.json").write_text(json.dumps(summary, indent=1))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": "#52514e", "xtick.color": "#898781",
                         "ytick.color": "#898781", "text.color": "#0b0b0b"})
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharey=True, facecolor="#fcfcfb")
    x = [np.sqrt(a * b) for a, b in BANDS]
    for i, s in enumerate(SEQS):
        for k, rep in enumerate(TITLES):
            ax = axes[i, k]
            ax.set_facecolor("#fcfcfb")
            ax.axhline(1.0, color="#898781", lw=1, ls=":")
            for c, col, ls, lw, lab in LINES:
                y = np.maximum(summary[s]["reps"][rep][c]["ratio_per_band"], 1e-3)
                ax.plot(x, y, color=col, ls=ls, lw=lw, marker="o", ms=3.5, label=lab)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 3)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{a}" if a == b else f"{a}-{b}" for a, b in BANDS],
                               fontsize=7)
            ax.grid(True, color="#e1e0d9", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(TITLES[rep], fontsize=10, loc="left")
            if k == 0:
                ax.set_ylabel(f"{s}\nchange / different-place change")
            if i == 2:
                ax.set_xlabel("radial frequency band (cycles per 160 m)")
    axes[0, 2].text(0.98, 0.93, "1 = as different as another place",
                    transform=axes[0, 2].transAxes, ha="right", fontsize=8, color="#52514e")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, fontsize=9)
    fig.suptitle("Cartesian occupancy magnitude spectrum: removing rotation after translation",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT / "fm_change_by_band.png", dpi=130)

    for s in SEQS:
        print(f"\n== {s} (n={summary[s]['n_pairs']})  bands {BANDS}")
        for rep in TITLES:
            print(f" [{rep}]")
            for c in CONDS:
                v = summary[s]["reps"][rep][c]
                print(f"  {c:9s} " + " ".join(f"{t:.3f}" for t in v["ratio_per_band"])
                      + f" | all {v['ratio_all']:.3f} conf {100*v['confusion_rate']:.1f}%")


if __name__ == "__main__":
    main()
