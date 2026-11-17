#!/usr/bin/env python3
"""scattering 설명 그림: 실제 스캔 한 장(KITTI-360 0004)으로 단계별 중간 결과를 그린다.

    docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
      python3 scripts/_scatter_explain_fig_20261114.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _magspec_invariance_probe_20261114 import KEYS, apply, shift_mat, yaw_mat  # noqa: E402
from _pd_build_test_caches import _make_loader  # noqa: E402
from _scatter_probe_20261114 import J, L, N, bank, morlet, occupancy  # noqa: E402

OUT = REPO / "results/rawmag_probe_20261114"
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#86b6ef", "#0d366b"])
INK, MUTED = "#0b0b0b", "#898781"

cache = np.load(next(KEYS.glob("cache_*_kitti360_val_0004.npz")))
pts = _make_loader("kitti360", "0004")[int(cache["scan_ids"][2000])]["points"]
F = bank()
JJ = 2                                                   # 파장 10.7 m


def u1(p):
    return np.abs(np.fft.ifft2(np.fft.fft2(occupancy(p))[None, None] * F))  # (J, L, N, N)


U = u1(pts)
S = {"original": U.mean((-1, -2)),
     "rotated 30 deg": u1(apply(yaw_mat(30.0), pts)).mean((-1, -2)),
     "shifted 3 m": u1(apply(shift_mat(0.0, 3.0), pts)).mean((-1, -2))}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "text.color": INK,
                     "axes.labelcolor": "#52514e", "xtick.color": MUTED,
                     "ytick.color": MUTED})
fig, ax = plt.subplots(2, 4, figsize=(15, 7.6), facecolor="#fcfcfb")
ext = [-80, 80, -80, 80]


def show(a, img, title, cmap=SEQ, **kw):
    a.imshow(img.T, origin="lower", extent=ext, cmap=cmap, **kw)
    a.set_title(title, loc="left", fontsize=10)
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")


show(ax[0, 0], occupancy(pts), "1  Input: top-down occupancy grid (1 m cells)", vmax=1.2)

tiles = np.block([[morlet(JJ, 0).real[40:120, 40:120], morlet(JJ, 4).real[40:120, 40:120]],
                  [morlet(JJ, 8).real[40:120, 40:120], morlet(JJ, 12).real[40:120, 40:120]]])
m = np.abs(tiles).max()
ax[0, 1].imshow(tiles.T, origin="lower", cmap=DIV, vmin=-m, vmax=m)
ax[0, 1].set_title("2  Wavelets: 10.7 m ripple at 0, 45, 90, 135 deg", loc="left", fontsize=10)
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])
ax[0, 1].axhline(79.5, color="#c3c2b7", lw=1)
ax[0, 1].axvline(79.5, color="#c3c2b7", lw=1)

vm = np.percentile(U[JJ], 99.7)
show(ax[0, 2], U[JJ, 0], "3  |scan * wavelet|, ripple along x (0 deg)", vmax=vm)
show(ax[0, 3], U[JJ, 8], "3  |scan * wavelet|, ripple along y (90 deg)", vmax=vm)

deg = np.arange(L) * 180 / L
for (k, v), c, ls, lw in zip(S.items(), [INK, "#eb6834", "#2a78d6"], ["-", "-", "--"],
                             [3, 2, 2]):
    ax[1, 0].plot(deg, v[JJ], color=c, ls=ls, lw=lw, marker="o", ms=4, label=k)
ax[1, 0].set_title("4  Average over space: one number per direction", loc="left", fontsize=10)
ax[1, 0].set_xlabel("wavelet direction (deg)")
ax[1, 0].set_ylabel("mean response (10.7 m scale)")
ax[1, 0].set_ylim(0, None)
ax[1, 0].legend(frameon=False, fontsize=8, loc="lower left")
ax[1, 0].grid(True, color="#e1e0d9", lw=0.6)
for sp in ("top", "right"):
    ax[1, 0].spines[sp].set_visible(False)

cover = np.fft.fftshift(np.abs(F[:, ::4]).max(1), axes=(-1, -2))   # 스케일별, 방향 4개
k = np.arange(N) - N // 2
cols = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
for j in range(J):
    ax[1, 1].contour(k, k, cover[j].T, levels=[0.5 * cover[j].max()], colors=cols[::-1][j],
                     linewidths=1.6)
ax[1, 1].set_xlim(-80, 80)
ax[1, 1].set_ylim(-80, 80)
ax[1, 1].set_aspect("equal")
ax[1, 1].set_xticks(np.arange(-60, 61, 10), minor=True)
ax[1, 1].set_yticks(np.arange(-60, 61, 10), minor=True)
ax[1, 1].grid(True, which="both", color="#e1e0d9", lw=0.5)
ax[1, 1].set_title("5  Wavelet bands widen with frequency",
                   loc="left", fontsize=10)
ax[1, 1].set_xlabel("frequency along x (cycles per 160 m)")
ax[1, 1].set_ylabel("frequency along y")
ax[1, 1].text(0, 72, "DFT: one cell per frequency (grid drawn every 10)",
              ha="center", fontsize=7.5, color="#52514e")

U2 = np.abs(np.fft.ifft2(np.fft.fft2(U[1, 0]) * F[3, 8]))
show(ax[1, 2], U2, "6  Second order: 5 m structure, 21 m spacing",
     vmax=np.percentile(U2, 99.7))

both = np.vstack([S["original"], np.full((1, L), np.nan), S["rotated 30 deg"]])
ax[1, 3].imshow(both / np.nanmax(both), cmap=SEQ, aspect="auto",
                extent=[-180 / L / 2, 180 - 180 / L / 2, 2 * J + 1, 0])
ax[1, 3].set_yticks([J / 2, J + 1 + J / 2])
ax[1, 3].set_yticklabels(["original\n(5 scales)", "rotated\n30 deg"], fontsize=8)
ax[1, 3].set_xlabel("wavelet direction (deg)")
ax[1, 3].set_title("7  Rotation = sideways shift of the table", loc="left", fontsize=10)

fig.suptitle("Wavelet scattering on one LiDAR scan (KITTI-360 0004, keyframe 2000)",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "scatter_explain.png", dpi=120)
print("ok")
