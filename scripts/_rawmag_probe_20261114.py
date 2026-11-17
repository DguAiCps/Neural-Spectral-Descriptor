#!/usr/bin/env python3
"""순수 크기 스펙트럼이 변환과 실제 재방문에서 주파수별로 얼마나 변하는지 잰다.

구간 묶기·재계량·줄 평균·정규화 없이, 격자 세 종류의 DFT 크기를 그대로 본다.

  range   거리 영상 16x360(가장 가까운 거리, 빈 칸 순환 보간)의 줄별 방위 |DFT|  (16x181)
  polar   극좌표 점유 격자 79고리x360부채꼴(1 m, 1도, 0/1)의 고리별 방위 |DFT|   (79x181)
  cart    직교 점유 격자 160x160(1 m, +-80 m, 0/1)의 2차원 |DFT|                (160x160)

range·polar의 크기는 칸 정수배 회전에 불변이고, cart의 크기는 (격자 경계를 빼면)
이동에 불변이다. 쌍·ICP 상대 자세·변환 조건은
scripts/_magspec_invariance_probe_20261114.py와 같다.

쌍마다 조건별로 저장하는 것
  sq[rep][cond]   주파수별 제곱차 (range·polar: 줄 합, cart: 반지름 주파수 고리 합)
  스캔 A의 크기 스펙트럼 전체 (다른 장소 쌍의 제곱차와 순위 계산용)

    docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
      python3 scripts/_rawmag_probe_20261114.py --seq kitti360/0004
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _magspec_invariance_probe_20261114 import (  # noqa: E402
    KEYS, apply, register, shift_mat, yaw_mat)
from _pd_build_test_caches import (  # noqa: E402
    _effective_config, _make_desc_encoder, _make_loader)
from encoding.range_image import interpolate_range_image  # noqa: E402

OUT = REPO / "results/rawmag_probe_20261114"
SHIFTS = (0.5, 1.0, 2.0, 4.0)
N = 160
_ky = np.fft.fftfreq(N) * N
RADIAL = np.minimum(np.rint(np.hypot(_ky[:, None], _ky[None, :])).astype(int), 80)


def spectra(pts, projector):
    """세 격자의 순수 크기 스펙트럼."""
    img, _ = projector.project(pts, keep_intensity=False)
    rng = np.abs(np.fft.rfft(interpolate_range_image(img, method="linear"), axis=1))
    x, y = pts[:, 0], pts[:, 1]
    rho = np.hypot(x, y)
    keep = (rho >= 1.0) & (rho < 80.0)
    polar = np.zeros((79, 360), np.float32)
    az = np.floor(np.rad2deg(np.arctan2(y[keep], x[keep])) % 360.0).astype(int) % 360
    polar[(rho[keep] - 1.0).astype(int), az] = 1.0
    cart = np.zeros((N, N), np.float32)
    ix, iy = np.floor(x + 80.0).astype(int), np.floor(y + 80.0).astype(int)
    ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (rho >= 1.0)
    cart[ix[ok], iy[ok]] = 1.0
    return {"range": rng.astype(np.float32),
            "polar": np.abs(np.fft.rfft(polar, axis=1)).astype(np.float32),
            "cart": np.abs(np.fft.fft2(cart)).astype(np.float32)}


def per_freq(rep, sq):
    """제곱차를 주파수 축으로 합친다."""
    if rep == "cart":
        return np.bincount(RADIAL.ravel(), weights=sq.ravel(), minlength=81)
    return sq.sum(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    sensor, seq = a.seq.split("/")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    projector = _make_desc_encoder(_effective_config(), sensor, device).projector
    reps = ("range", "polar", "cart")
    base = {r: [] for r in reps}
    sq = {r: {} for r in reps}
    meta = []
    for n, (q, j, dist) in enumerate(pairs):
        if n % 50 == 0:
            print(f"  pair {n}/{len(pairs)}", flush=True)
        A = loader[int(cache["scan_ids"][q])]["points"]
        B = loader[int(cache["scan_ids"][j])]["points"]
        rel, fit, rmse = register(A, B, np.linalg.inv(gt[j]) @ gt[q])
        yaw = float(np.rad2deg(np.arctan2(rel[1, 0], rel[0, 0])))
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
        sA = spectra(A, projector)
        for r in reps:
            base[r].append(sA[r])
        for name, pts in conds.items():
            sC = spectra(pts, projector)
            for r in reps:
                sq[r].setdefault(name, []).append(per_freq(r, (sA[r] - sC[r]) ** 2))
        meta.append((q, j, dist, yaw, float(np.hypot(rel[0, 3], rel[1, 3])), fit, rmse,
                     float(np.linalg.norm(xyz[q + 1] - xyz[q]))))

    OUT.mkdir(parents=True, exist_ok=True)
    out = {"meta": np.asarray(meta), "xyz": xyz[[p[0] for p in pairs]],
           "meta_cols": np.asarray(["q", "j", "gt_dist_m", "icp_yaw_deg", "icp_shift_m",
                                    "icp_fitness", "icp_rmse_m", "adjacent_dist_m"])}
    for r in reps:
        out[f"base_{r}"] = np.asarray(base[r], np.float32)
        for name, v in sq[r].items():
            out[f"sq_{r}_{name}"] = np.asarray(v, np.float32)
    path = OUT / f"{sensor}_{seq}.npz"
    np.savez_compressed(path, **out)
    print(f"OK {path}", flush=True)


if __name__ == "__main__":
    main()
