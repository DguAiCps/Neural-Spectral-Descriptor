#!/usr/bin/env python3
"""이동·회전 모두에 불변인 네 번째 격자(스케일 없는 Fourier-Mellin)를 잰다.

  1. 직교 점유 160x160(1 m)의 2차원 |DFT|, 2배 영 채움(320x320)   -> 이동 불변
  2. 그 크기 스펙트럼을 극좌표로 표본화: 반지름 주파수 1~80, 각도 0~179도(1도)
     (크기 스펙트럼은 원점 대칭이라 반 바퀴면 충분하고, 회전은 각도 축 순환 이동이 된다)
  3. 각도 축 처리 두 가지
       fm_mag   반지름마다 각도 방향 |DFT|                    (크기를 두 번 취함)
       fm_corr  모든 반지름을 함께 쓰는 원형 상관으로 각도를 맞춘 뒤 비교 (RING 방식)

쌍·ICP·변환 조건·척도는 scripts/_rawmag_probe_20261114.py와 같다. 제곱차는 반지름 주파수별로
저장한다.

    docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
      python3 scripts/_rawmag_fm_probe_20261114.py --seq kitti360/0004
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _magspec_invariance_probe_20261114 import (  # noqa: E402
    KEYS, apply, register, shift_mat, yaw_mat)
from _pd_build_test_caches import _make_loader  # noqa: E402

OUT = REPO / "results/rawmag_probe_20261114"
SHIFTS = (0.5, 1.0, 2.0, 4.0)
N, PAD = 160, 2
_r = np.arange(1, 81, dtype=np.float64)[:, None] * PAD
_t = np.deg2rad(np.arange(180))[None, :]
COORDS = np.stack([N * PAD // 2 + _r * np.cos(_t), N * PAD // 2 + _r * np.sin(_t)])


def polar_spectrum(pts):
    """직교 점유 -> 2배 영 채움 2차원 |DFT| -> 극좌표 (80, 180)."""
    x, y = pts[:, 0], pts[:, 1]
    g = np.zeros((N, N), np.float32)
    ix, iy = np.floor(x + 80.0).astype(int), np.floor(y + 80.0).astype(int)
    ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (np.hypot(x, y) >= 1.0)
    g[ix[ok], iy[ok]] = 1.0
    mag = np.fft.fftshift(np.abs(np.fft.fft2(g, s=(N * PAD, N * PAD))))
    return map_coordinates(mag, COORDS, order=1, mode="nearest").astype(np.float32)


def sq_mag(PA, PC):
    """각도 방향 |DFT|끼리의 반지름별 제곱차."""
    return ((np.abs(np.fft.rfft(PA, axis=1)) - np.abs(np.fft.rfft(PC, axis=1))) ** 2).sum(1)


def sq_corr(PA, PC):
    """원형 상관이 가장 큰 각도 이동에서의 반지름별 제곱차."""
    c = np.fft.irfft(np.fft.rfft(PA, axis=1) * np.conj(np.fft.rfft(PC, axis=1)),
                     n=180, axis=1)
    s = int(c.sum(0).argmax())
    return (PA ** 2).sum(1) + (PC ** 2).sum(1) - 2.0 * c[:, s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    sensor, seq = a.seq.split("/")
    cache = np.load(next(KEYS.glob(f"cache_*_{sensor}_val_{seq}.npz")))
    gt = cache["poses_gt"].astype(np.float64)
    xyz = gt[:, :3, 3]
    pairs = []
    for q in range(30, len(xyz) - 1):
        d = np.linalg.norm(xyz[: q - 29] - xyz[q], axis=1)
        j = int(d.argmin())
        if d[j] <= 5.0 and (d > 20.0).any():
            pairs.append((q, j, float(d[j])))
    if len(pairs) > a.n:
        pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, a.n).astype(int)]
    print(f"[{a.seq}] revisit queries used: {len(pairs)}", flush=True)

    loader = _make_loader(sensor, seq)
    base, meta = [], []
    sq = {"fm_mag": {}, "fm_corr": {}}
    for n, (q, j, dist) in enumerate(pairs):
        if n % 50 == 0:
            print(f"  pair {n}/{len(pairs)}", flush=True)
        A = loader[int(cache["scan_ids"][q])]["points"]
        B = loader[int(cache["scan_ids"][j])]["points"]
        rel, fit, rmse = register(A, B, np.linalg.inv(gt[j]) @ gt[q])
        conds = {
            "yaw_0.5": apply(yaw_mat(0.5), A),
            "yaw_5": apply(yaw_mat(5.0), A),
            "yaw_30": apply(yaw_mat(30.0), A),
            "se3_real": apply(rel, A),
            "real": B,
            "adjacent": loader[int(cache["scan_ids"][q + 1])]["points"],
        }
        for s in SHIFTS:
            conds[f"fwd_{s}"] = apply(shift_mat(s, 0.0), A)
            conds[f"lat_{s}"] = apply(shift_mat(0.0, s), A)
        PA = polar_spectrum(A)
        base.append(PA)
        for name, pts in conds.items():
            PC = polar_spectrum(pts).astype(np.float64)
            sq["fm_mag"].setdefault(name, []).append(sq_mag(PA.astype(np.float64), PC))
            sq["fm_corr"].setdefault(name, []).append(sq_corr(PA.astype(np.float64), PC))
        meta.append((q, j, dist, fit, rmse))

    out = {"meta": np.asarray(meta), "xyz": xyz[[p[0] for p in pairs]],
           "base": np.asarray(base, np.float32)}
    for r, d in sq.items():
        for name, v in d.items():
            out[f"sq_{r}_{name}"] = np.asarray(v, np.float32)
    path = OUT / f"fm_{sensor}_{seq}.npz"
    np.savez_compressed(path, **out)
    print(f"OK {path}", flush=True)


if __name__ == "__main__":
    main()
