#!/usr/bin/env python3
"""직교 점유 격자의 2차 웨이블릿 scattering이 변환과 실제 재방문에서 얼마나 변하는지 잰다.

roto-translation scattering의 간단판이다(학습 없음).
  U1[j,t]      = |x * psi_{j,t}|                    Morlet, 스케일 j=0..4, 방향 t=0..15 (180도/16)
  S1[j,t]      = 공간 전체 평균 U1                    -> 이동 불변
  S2[j1,j2,t1,d] = 공간 전체 평균 | U1[j1,t1] * psi_{j2,t1+d} |,  j2 > j1
회전은 t(S2에서는 t1)의 순환 이동이 되고 상대 방향 d는 그대로다. 비교할 때 모든 행을 함께 쓰는
원형 상관으로 방향을 맞춘다(방향 축을 128표본으로 푸리에 보간한 뒤). 정식 판은 방향 축에도
웨이블릿을 걸지만 여기서는 걸지 않는다.

쌍·ICP·척도는 scripts/_rawmag_fm_probe_20261114.py와 같다. 끝에 집계까지 한다.

    docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
      python3 scripts/_scatter_probe_20261114.py --seq kitti360/0004
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "results/rawmag_probe_20261114"
N, J, L, UP = 160, 5, 16, 128
PAIRS = [(a, b) for a in range(J) for b in range(a + 1, J)]
CONDS = ["yaw_5", "yaw_30", "lat_0.5", "lat_2.3", "se3_real", "real", "adjacent"]


def morlet(j, t):
    """공간 영역 Morlet (N, N) 복소. 파장 2.67 * 2^j 칸, 방향 t * 180/L 도."""
    sigma, xi, slant = 0.8 * 2 ** j, 0.75 * np.pi / 2 ** j, 0.5
    g = np.arange(N) - N // 2
    x, y = np.meshgrid(g, g, indexing="ij")
    th = np.pi * t / L
    u, v = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
    env = np.exp(-(u ** 2 + (slant * v) ** 2) / (2 * sigma ** 2))
    wave = env * np.exp(1j * xi * u)
    return (wave - env * wave.sum() / env.sum()) * slant / (2 * np.pi * sigma ** 2)


def bank():
    return np.stack([[np.fft.fft2(np.fft.ifftshift(morlet(j, t))) for t in range(L)]
                     for j in range(J)]).astype(np.complex64)      # (J, L, N, N)


def occupancy(pts):
    x, y = pts[:, 0], pts[:, 1]
    g = np.zeros((N, N), np.float32)
    ix, iy = np.floor(x + 80.0).astype(int), np.floor(y + 80.0).astype(int)
    ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (np.hypot(x, y) >= 1.0)
    g[ix[ok], iy[ok]] = 1.0
    return g


def scatter(grid, F):
    """(J + len(PAIRS)*L, L) 행렬: 앞 J행은 S1, 나머지는 S2의 (쌍, 상대 방향 d)별 행."""
    import torch
    x = torch.from_numpy(grid).to(F.device)
    U1 = torch.fft.ifft2(torch.fft.fft2(x)[None, None] * F).abs()   # (J, L, N, N)
    rows = [U1.mean((-1, -2))]
    idx = (torch.arange(L, device=F.device)[:, None]
           + torch.arange(L, device=F.device)[None, :]) % L         # [t1, d] -> t2
    for j1 in range(J - 1):
        U1f = torch.fft.fft2(U1[j1])                                # (L, N, N)
        for j2 in range(j1 + 1, J):
            S2 = torch.fft.ifft2(U1f[:, None] * F[j2][None]).abs().mean((-1, -2))  # [t1, t2]
            rows.append(torch.gather(S2, 1, idx).T)                 # [d, t1]
    return torch.cat(rows).double().cpu().numpy()


def upsample(D):
    return np.fft.irfft(np.fft.rfft(D, axis=-1), n=UP, axis=-1) * (UP / L)


def sq_corr(A, C):
    """방향 축 원형 상관이 가장 큰 이동에서의 행별 제곱차. A (R, UP), C (..., R, UP)."""
    c = np.fft.irfft(np.fft.rfft(A, axis=-1) * np.conj(np.fft.rfft(C, axis=-1)),
                     n=UP, axis=-1)
    s = c.sum(-2).argmax(-1)
    best = np.take_along_axis(c, s[..., None, None], axis=-1)[..., 0]
    return (A ** 2).sum(-1) + (C ** 2).sum(-1) - 2.0 * best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    sensor, seq = a.seq.split("/")
    import torch
    from _magspec_invariance_probe_20261114 import (
        KEYS, apply, register, shift_mat, yaw_mat)
    from _pd_build_test_caches import _make_loader
    F = torch.from_numpy(bank()).cuda()

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
    base, fit = [], []
    sq = {c: [] for c in CONDS}
    for n, (q, j, dist) in enumerate(pairs):
        if n % 50 == 0:
            print(f"  pair {n}/{len(pairs)}", flush=True)
        A = loader[int(cache["scan_ids"][q])]["points"]
        B = loader[int(cache["scan_ids"][j])]["points"]
        rel, f, _ = register(A, B, np.linalg.inv(gt[j]) @ gt[q])
        conds = {"yaw_5": apply(yaw_mat(5.0), A), "yaw_30": apply(yaw_mat(30.0), A),
                 "lat_0.5": apply(shift_mat(0.0, 0.5), A),
                 "lat_2.3": apply(shift_mat(0.0, 2.3), A),
                 "se3_real": apply(rel, A), "real": B,
                 "adjacent": loader[int(cache["scan_ids"][q + 1])]["points"]}
        DA = upsample(scatter(occupancy(A), F))
        base.append(DA)
        fit.append(f)
        for c, pts in conds.items():
            sq[c].append(sq_corr(DA, upsample(scatter(occupancy(pts), F))))

    base = np.asarray(base)
    ok = np.asarray(fit) >= 0.6
    P = xyz[[p[0] for p in pairs]]
    far = np.linalg.norm(P[:, None] - P[None], axis=-1) > 20.0
    neg = np.stack([sq_corr(base[i], base) for i in range(len(base))])    # (n, n, R)
    groups = {f"S1 scale {j} ({2.67 * 2 ** j:.1f} m)": slice(j, j + 1) for j in range(J)}
    groups["S1 all"] = slice(0, J)
    groups["S2 all"] = slice(J, None)
    groups["S1+S2"] = slice(0, None)
    res = {"n_pairs": int(ok.sum()), "groups": {}}
    print(f"\n== {a.seq} (n={ok.sum()})   change ratio | confusion")
    for g, sl in groups.items():
        between = neg[far][:, sl].sum(1).mean()
        negtot = neg[:, :, sl].sum(2)
        r = {}
        for c in CONDS:
            s = np.asarray(sq[c])[:, sl].sum(1)
            conf = [(negtot[i][far[i]] < s[i]).mean() for i in np.flatnonzero(ok)
                    if far[i].any()]
            r[c] = {"ratio": float(s[ok].mean() / between), "confusion": float(np.mean(conf))}
        res["groups"][g] = r
        print(f" {g:22s} " + "  ".join(
            f"{c} {r[c]['ratio']:.3f}|{100 * r[c]['confusion']:.1f}%" for c in CONDS))
    (OUT / f"scatter_{sensor}_{seq}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / f"scatter_{sensor}_{seq}.npz", base=base.astype(np.float32),
                        xyz=P, fit=np.asarray(fit),
                        **{f"sq_{c}": np.asarray(v, np.float32) for c, v in sq.items()})


if __name__ == "__main__":
    main()
