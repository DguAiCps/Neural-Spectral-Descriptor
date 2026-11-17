#!/usr/bin/env python3
"""scattering의 공간 전체 평균을 대신하는 두 방식을 전체 지도 검색과 PR로 잰다.

공통: 직교 점유 160x160(1 m)에 Morlet(스케일 5, 방향 16) -> U1[j,t] = |x * psi_{j,t}|.

  loc   부분 평균. U1을 센서 중심 극좌표 구역(고리 4개: 1-10, 10-20, 20-40, 40-80 m x 부채꼴 32개)
        별로 평균한다. 필터 방향을 구역의 방위 기준 상대 방향 t' = t - a 로 적으면 회전은 부채꼴 축
        순환 이동이 된다. 부채꼴 축은 조화 성분 0~KMAX만 남긴다(매끄럽게).
        행 (j, 고리, t') 320개 x 부채꼴 32  -> loc_k4 (KMAX 4), loc_k8 (KMAX 8)
  spec  평균 대신 스펙트럼. U1[j,t]마다 2차원 |DFT|(2배 영 채움)를 반지름 주파수 10개
        (1,2,3,4,6,8,12,16,24,32) x 필터 방향 기준 상대 각도 16개에서 읽는다(각도 구간 안 4점 평균).
        이동에 불변이고 회전은 t축 순환 이동이 된다. S1(평균) 5행을 더해
        행 (j, 반지름, 상대 각도) 805개 x 방향 16.

비교 방식·질의·회계는 scripts/_fullmap_probe_20261114.py의 (1), (3)과 같다(코사인, 원형 상관 정렬,
scripts/_pr_eval_lib.py). R@100P는 정답 반경 5 m와 10 m 둘 다 적는다.

    docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
      python3 scripts/_poolalt_probe_20261114.py --seq kitti360/0004
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _magspec_invariance_probe_20261114 import KEYS, apply, shift_mat, yaw_mat  # noqa: E402
from _pd_build_test_caches import _make_loader  # noqa: E402
from _pr_eval_lib import classify_queries, pr_sweep  # noqa: E402
from _scatter_probe_20261114 import J, L, N, bank, occupancy  # noqa: E402

OUT = REPO / "results/rawmag_probe_20261114"
DEV = "cuda"
RINGS = (1.0, 10.0, 20.0, 40.0, 80.0)
NSEC = 32
RADII = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
PAD = 2
NFFT = {"loc_k4": 128, "loc_k8": 128, "spec": 128}

# loc: 칸 -> (고리, 부채꼴)
_g = torch.arange(N, dtype=torch.float32) - N // 2 + 0.5
_x, _y = torch.meshgrid(_g, _g, indexing="ij")
_rho = torch.hypot(_x, _y)
_ring = torch.bucketize(_rho, torch.tensor(RINGS)) - 1
_sec = torch.floor(torch.rad2deg(torch.atan2(_y, _x)) % 360.0 / (360.0 / NSEC)).long() % NSEC
_valid = ((_ring >= 0) & (_ring < len(RINGS) - 1)).flatten()
CELL = (_ring * NSEC + _sec).flatten()[_valid].to(DEV)
NCELL = (len(RINGS) - 1) * NSEC
COUNT = torch.zeros(NCELL, device=DEV).index_add_(0, CELL, torch.ones_like(CELL, dtype=torch.float32))
VALID = _valid.to(DEV)
REL = ((torch.arange(L)[:, None] + torch.arange(NSEC)[None, :]) % L).to(DEV)   # [t', a] -> t

# spec: (t, 반지름, 상대 각도, 부표본) -> 영 채움 스펙트럼의 표본 좌표
_sub = torch.tensor([-0.375, -0.125, 0.125, 0.375])
_ang = (torch.arange(L)[:, None, None] + torch.arange(L)[None, :, None] + _sub[None, None, :]
        ) * (np.pi / L)                                                 # [t, t', sub]
_rad = torch.tensor(RADII, dtype=torch.float32)[None, :, None, None] * PAD
_c = N * PAD // 2
_ix = _c + _rad * torch.cos(_ang)[:, None]                              # [t, r, t', sub]
_iy = _c + _rad * torch.sin(_ang)[:, None]
_norm = lambda v: v / (N * PAD - 1) * 2 - 1                             # noqa: E731
GRID = torch.stack([_norm(_iy), _norm(_ix)], -1).reshape(L, -1, 1, 2).to(DEV)


def describe(grid, F):
    """U1에서 loc (320, 32)과 spec (805, 16)을 만든다."""
    x = torch.from_numpy(grid).to(DEV)
    U1 = torch.fft.ifft2(torch.fft.fft2(x)[None, None] * F).abs()       # (J, L, N, N)
    # loc
    P = torch.zeros(J, L, NCELL, device=DEV).index_add_(
        2, CELL, U1.flatten(2)[:, :, VALID]) / COUNT                    # (J, L, 고리*부채꼴)
    P = P.reshape(J, L, len(RINGS) - 1, NSEC).permute(0, 2, 1, 3)       # (J, 고리, t, a)
    loc = torch.gather(P, 2, REL[None, None].expand(J, len(RINGS) - 1, L, NSEC))
    # spec
    M = torch.fft.fftshift(torch.fft.fft2(U1, s=(N * PAD, N * PAD)).abs(), dim=(-1, -2))
    S = torch.nn.functional.grid_sample(
        M.transpose(0, 1).reshape(L, J, N * PAD, N * PAD), GRID, mode="bilinear",
        align_corners=True)                                             # (L, J, r*t'*sub, 1)
    S = S.reshape(L, J, len(RADII), L, 4).mean(-1).permute(1, 2, 3, 0)  # (J, r, t', t)
    spec = torch.cat([U1.mean((-1, -2)), S.reshape(-1, L)])
    return loc.reshape(-1, NSEC).cpu().numpy(), spec.cpu().numpy()


def to_freq(D, kmax=None):
    D = torch.as_tensor(np.asarray(D), dtype=torch.float32, device=DEV)
    Fd = torch.fft.rfft(D, dim=-1)
    if kmax is not None:
        Fd = Fd[..., : kmax + 1]
    nrm = torch.sqrt((Fd.abs() ** 2 * torch.tensor(
        [1.0] + [2.0] * (Fd.shape[-1] - 1), device=DEV)).sum((-1, -2)))
    return Fd.movedim(-1, 0).contiguous(), nrm


def aligned_sim(Fq, nq, Fdb, ndb, nfft):
    C = torch.matmul(Fq, Fdb.conj().transpose(1, 2))
    c = torch.fft.irfft(C.movedim(0, -1), n=nfft, dim=-1).amax(-1)
    return c * nfft / (nq[:, None] * ndb[None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sensor, seq = a.seq.split("/")
    cache = np.load(next(KEYS.glob(f"cache_*_{sensor}_val_{seq}.npz")))
    n = a.limit or len(cache["scan_ids"])
    ids = cache["scan_ids"][:n]
    xyz = torch.as_tensor(cache["poses_gt"][:n, :3, 3], dtype=torch.float32, device=DEV)
    loader = _make_loader(sensor, seq)
    F = torch.from_numpy(bank()).to(DEV)

    loc = np.empty((n, J * (len(RINGS) - 1) * L, NSEC), np.float32)
    spec = np.empty((n, J + J * len(RADII) * L, L), np.float32)
    for i, sid in enumerate(ids):
        if i % 2000 == 0:
            print(f"  map {i}/{n}", flush=True)
        loc[i], spec[i] = describe(occupancy(loader[int(sid)]["points"]), F)
    db = {"loc_k4": to_freq(loc, 4), "loc_k8": to_freq(loc, 8), "spec": to_freq(spec)}
    kmax = {"loc_k4": 4, "loc_k8": 8, "spec": None}

    # 점검: 회전 30도, 옆 2.3 m 이동한 스캔과 원본의 유사도, 임의의 다른 스캔과의 유사도
    rng = np.random.default_rng(0)
    pick = rng.choice(n, 20, replace=False)
    chk = {k: {"yaw_30": [], "lat_2.3": [], "other": []} for k in db}
    for i in pick:
        pts = loader[int(ids[i])]["points"]
        for name, T in (("yaw_30", yaw_mat(30.0)), ("lat_2.3", shift_mat(0.0, 2.3))):
            dl, ds = describe(occupancy(apply(T, pts)), F)
            for k, d in (("loc_k4", dl), ("loc_k8", dl), ("spec", ds)):
                Fq, nq = to_freq(d[None], kmax[k])
                chk[k][name].append(float(aligned_sim(
                    Fq, nq, db[k][0][:, i:i + 1], db[k][1][i:i + 1], NFFT[k])))
        o = int(rng.integers(n))
        for k in db:
            chk[k]["other"].append(float(aligned_sim(
                db[k][0][:, i:i + 1], db[k][1][i:i + 1], db[k][0][:, o:o + 1],
                db[k][1][o:o + 1], NFFT[k])))
    print(f"\n== {a.seq}: map {n}")
    for k, v in chk.items():
        print(f" check {k:7s} " + "  ".join(f"{c} {np.median(x):.4f}" for c, x in v.items()))

    idx = torch.arange(n, device=DEV)
    qs = torch.arange(30, n, device=DEV)
    res = {"seq": a.seq, "n_map": n, "check": {k: {c: float(np.median(x)) for c, x in v.items()}
                                               for k, v in chk.items()}, "reps": {}}
    for rep, (Fdb, ndb) in db.items():
        score, top1d, dmin, hit100 = [], [], [], []
        for s in range(0, len(qs), 16):
            q = qs[s:s + 16]
            past = idx[None] <= (q[:, None] - 30)
            D = (xyz[q][:, None] - xyz[None]).norm(dim=2).masked_fill(~past, torch.inf)
            S = aligned_sim(Fdb[:, q], ndb[q], Fdb, ndb, NFFT[rep]).masked_fill(
                ~past, -torch.inf)
            v, t = S.topk(100, dim=1)
            ok = t <= (q[:, None] - 30)
            score.append(v[:, 0])
            top1d.append(D.gather(1, t[:, :1])[:, 0])
            dmin.append(D.min(dim=1).values)
            hit100.append(((D.gather(1, t) <= 5.0) & ok).any(1))
        score, top1d, dmin = (torch.cat(v).double().cpu().numpy() for v in (score, top1d, dmin))
        hit100 = torch.cat(hit100).cpu().numpy()
        cls = classify_queries(dmin)
        pos = cls["positive"]
        pr = pr_sweep(score, top1d, cls)
        elig = pos | cls["negative"]
        fp10 = elig & (top1d > 10.0)
        r100p_10 = float((pos & (top1d <= 10.0) & (score > score[fp10].max())).sum() / pos.sum())
        res["reps"][rep] = {"R@1": float((top1d[pos] <= 5.0).mean()),
                            "R@100": float(hit100[pos].mean()),
                            "AP": float(pr["AP"]), "max_F1": float(pr["max_F1"]),
                            "R@100P": float(pr["R@100P"]), "R@100P_10m": r100p_10}
        res["counts"] = {k: cls[k] for k in ("n_pos", "n_neg", "n_ambiguous")}
        print(f" {rep:7s} " + "  ".join(f"{k} {100 * v:.1f}" for k, v in res["reps"][rep].items()))
    suffix = f"_limit{a.limit}" if a.limit else ""
    (OUT / f"poolalt_{sensor}_{seq}{suffix}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
