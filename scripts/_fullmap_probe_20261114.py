#!/usr/bin/env python3
"""표현별 전체 지도 검색과, 변형만 적용한 합성 조건에서의 1위 유지율을 잰다.

표현 (모두 코사인 유사도, 정렬 표현은 방향 축 원형 상관의 최댓값)
  r      재계량 원키 288D (캐시)
  m      BEV 크기 128D (캐시)
  fm     직교 점유 2차원 |DFT| -> 극좌표 80x180, 원형 상관 정렬 (14,400개)
  scat   2차 웨이블릿 scattering 165x16, 원형 상관 정렬 (2,640개)
  scat1  scat의 1차 계수만 5x16 (80개)

(1) 검색: 시퀀스의 모든 키프레임을 지도에 넣는다. 질의는 과거(j <= q-30)에 GT 5 m 안
    키프레임이 있는 q 전부. 후보는 j <= q-30 전체. R@1, R@100을 잰다.
(2) 변형: 고르게 뽑은 질의 300개의 스캔 A에 변형을 하나씩 적용해 다시 인코딩하고, 지도 전체에
    검색한다. 1위 유지율 = 변형한 A와 원래 A의 유사도가 20 m 넘게 떨어진 모든 키프레임과의
    유사도보다 큰 비율.
      occl_30/occl_90  임의 방위 30도/90도 부채꼴의 점을 모두 지움 (가림)
      crop_50/crop_30  수평 거리 50 m/30 m 밖의 점을 지움 (범위 잘림)
      drop_50          점의 절반을 임의로 지움 (밀도)
      cars_10          5~30 m 안에 4.5x2 m 상자 10개를 추가 (새 물체)
      warp_0.5/warp_1  파장 60 m의 매끄러운 비강체 변위, 진폭 0.5 m/1 m (찌그러짐)
      stretch_3        수평으로 3% 늘림
    참고로 real(실제 재방문 키프레임), adjacent(다음 키프레임)도 같은 방식으로 잰다.

    docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
      python3 scripts/_fullmap_probe_20261114.py --seq kitti360/0004
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

from _magspec_invariance_probe_20261114 import (  # noqa: E402
    BP, KEYS, OMEGA, SPEC, _sketch, unit)
from _pd_build_test_caches import (  # noqa: E402
    _effective_config, _make_desc_encoder, _make_loader)
from _rawmag_fm_probe_20261114 import polar_spectrum  # noqa: E402
from _scatter_probe_20261114 import J, L, bank, occupancy, scatter  # noqa: E402

OUT = REPO / "results/rawmag_probe_20261114"
DEV = "cuda"
NFFT = {"fm": 180, "scat": 128, "scat1": 128}
REPS = ("r", "m", "fm", "scat", "scat1")
CONDS = ("occl_30", "occl_90", "crop_50", "crop_30", "drop_50", "cars_10",
         "warp_0.5", "warp_1", "stretch_3")


def deform(pts, name, rng):
    x, y = pts[:, 0], pts[:, 1]
    if name.startswith("occl"):
        w = float(name.split("_")[1])
        a = (np.rad2deg(np.arctan2(y, x)) - rng.uniform(0, 360)) % 360
        return pts[a > w]
    if name.startswith("crop"):
        return pts[np.hypot(x, y) < float(name.split("_")[1])]
    if name == "drop_50":
        return pts[rng.random(len(pts)) < 0.5]
    if name == "cars_10":
        gx, gy, gz = np.meshgrid(np.arange(-2.25, 2.26, 0.25), np.arange(-1.0, 1.01, 0.25),
                                 np.arange(-1.2, -0.19, 0.25), indexing="ij")
        box = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
        add = []
        for _ in range(10):
            rad, az, yaw = rng.uniform(5, 30), rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi)
            c, s = np.cos(yaw), np.sin(yaw)
            b = box.copy()
            b[:, 0], b[:, 1] = (c * box[:, 0] - s * box[:, 1] + rad * np.cos(az),
                                s * box[:, 0] + c * box[:, 1] + rad * np.sin(az))
            add.append(b)
        new = np.zeros((sum(len(b) for b in add), pts.shape[1]), pts.dtype)
        new[:, :3] = np.concatenate(add)
        return np.concatenate([pts, new])
    out = pts.copy()
    if name.startswith("warp"):
        amp = float(name.split("_")[1])
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        out[:, 0] = x + amp * np.sin(2 * np.pi * y / 60.0 + p1)
        out[:, 1] = y + amp * np.sin(2 * np.pi * x / 60.0 + p2)
        return out
    if name == "stretch_3":
        out[:, :2] *= 1.03
        return out
    raise ValueError(name)


def to_freq(D):
    """(..., R, T) 실수 -> 방향 축 rfft (K, ..., R) 복소와 노름."""
    D = torch.as_tensor(np.asarray(D), dtype=torch.float32, device=DEV)
    return torch.fft.rfft(D, dim=-1).movedim(-1, 0).contiguous(), D.flatten(-2).norm(dim=-1)


def aligned_sim(Fq, nq, Fdb, ndb, nfft):
    """Fq (K, b, R), Fdb (K, n, R) -> (b, n) 방향 정렬 뒤 코사인."""
    C = torch.matmul(Fq, Fdb.conj().transpose(1, 2))                 # (K, b, n)
    c = torch.fft.irfft(C.movedim(0, -1), n=nfft, dim=-1).amax(-1)
    return c * (nfft / (2 * (Fq.shape[0] - 1))) / (nq[:, None] * ndb[None])


def pr_eval(a, sensor, seq, n, xyz, idx, sims):
    """(3) 키 하나의 1위 유사도를 수락 점수로 쓴 AP·최대 F1·R@100P.

    회계는 scripts/_pr_eval_lib.py 그대로다: 질의는 q >= 30 전부, 과거 최근접 5 m 이하면 양성,
    25 m 초과면 음성, 그 사이는 제외. 수락한 1위가 5 m 밖이면 질의 종류와 무관하게 오검출.
    """
    from _pr_eval_lib import classify_queries, pr_sweep
    qs = torch.arange(30, n, device=DEV)
    out = {"seq": a.seq, "n_map": n, "pr": {}}
    dump = {}
    print(f"\n== {a.seq}: PR over all queries q >= 30 ({len(qs)})")
    for rep in REPS:
        score, top1d, dmin, top1 = [], [], [], []
        for s in range(0, len(qs), 16):
            q = qs[s:s + 16]
            past = idx[None] <= (q[:, None] - 30)
            D = (xyz[q][:, None] - xyz[None]).norm(dim=2).masked_fill(~past, torch.inf)
            S = sims(rep, q_idx=q).masked_fill(~past, -torch.inf)
            v, t = S.max(dim=1)
            score.append(v)
            top1.append(t)
            top1d.append(D.gather(1, t[:, None])[:, 0])
            dmin.append(D.min(dim=1).values)
        cls = classify_queries(torch.cat(dmin).cpu().numpy())
        r = pr_sweep(torch.cat(score).double().cpu().numpy(),
                     torch.cat(top1d).double().cpu().numpy(), cls)
        out["pr"][rep] = {k: float(r[k]) for k in ("AP", "max_F1", "R@100P", "R@99P", "R@95P")}
        dump[f"{rep}_score"] = torch.cat(score).cpu().numpy()
        dump[f"{rep}_top1"] = torch.cat(top1).cpu().numpy()
        dump[f"{rep}_top1_dist"] = torch.cat(top1d).cpu().numpy()
        dump["dmin"] = torch.cat(dmin).cpu().numpy()
        out["counts"] = {k: cls[k] for k in ("n_pos", "n_neg", "n_ambiguous")}
        print(f" {rep:6s} " + "  ".join(f"{k} {100 * v:.1f}" for k, v in out["pr"][rep].items()))
    print(f" counts {out['counts']}")
    (OUT / f"pr_{sensor}_{seq}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(OUT / f"pr_{sensor}_{seq}_queries.npz", q=qs.cpu().numpy(),
                        xyz=xyz.cpu().numpy(), **dump)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--limit", type=int, default=0, help="앞쪽 키프레임만 (시험용)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--pr-only", action="store_true", help="(3)만 실행")
    a = ap.parse_args()
    sensor, seq = a.seq.split("/")
    cache = np.load(next(KEYS.glob(f"cache_*_{sensor}_val_{seq}.npz")))
    spec = np.load(SPEC / f"valspec_{sensor}_{seq}.npz")
    n = a.limit or len(cache["scan_ids"])
    ids = cache["scan_ids"][:n]
    xyz = torch.as_tensor(cache["poses_gt"][:n, :3, 3], dtype=torch.float32, device=DEV)
    r_bank = unit(cache["descriptors"][:n].astype(np.float32) * OMEGA[None])
    z = (spec["sk_re"] + 1j * spec["sk_im"])[:n, :, :8]
    m_bank = unit(np.abs(z).reshape(n, -1).astype(np.float32))

    loader = _make_loader(sensor, seq)
    enc = _make_desc_encoder(_effective_config(), sensor, DEV)
    F = torch.from_numpy(bank()).to(DEV)

    def encode(pts):
        with torch.no_grad():
            d = enc.encode_points(pts).detach().cpu().numpy().astype(np.float32)
        bev, _ = BP.project(pts, keep_intensity=False)
        zz = _sketch(np.maximum(bev, 0.0))[:, :8]
        sc = scatter(occupancy(pts), F)
        return {"r": unit(d * OMEGA), "m": unit(np.abs(zz).reshape(-1).astype(np.float32)),
                "fm": polar_spectrum(pts), "scat": sc, "scat1": sc[:J]}

    fm = np.empty((n, 80, 180), np.float32)
    sc = np.empty((n, J + (J * (J - 1) // 2) * L, L), np.float32)
    for i, sid in enumerate(ids):
        if i % 2000 == 0:
            print(f"  map {i}/{n}", flush=True)
        pts = loader[int(sid)]["points"]
        fm[i] = polar_spectrum(pts)
        sc[i] = scatter(occupancy(pts), F)

    db = {"r": torch.as_tensor(r_bank, device=DEV), "m": torch.as_tensor(m_bank, device=DEV)}
    for k, D in (("fm", fm), ("scat", sc), ("scat1", sc[:, :J])):
        db[k] = to_freq(D)

    def sims(rep, q_desc=None, q_idx=None):
        """질의(지도 색인 또는 새 서술자 묶음)와 지도 전체의 유사도 (b, n)."""
        if rep in ("r", "m"):
            Q = db[rep][q_idx] if q_desc is None else torch.as_tensor(
                np.stack(q_desc), device=DEV)
            return Q @ db[rep].T
        Fdb, ndb = db[rep]
        if q_desc is None:
            Fq, nq = Fdb[:, q_idx], ndb[q_idx]
        else:
            Fq, nq = to_freq(np.stack(q_desc))
        return aligned_sim(Fq, nq, Fdb, ndb, NFFT[rep])

    idx = torch.arange(n, device=DEV)
    if a.pr_only:
        pr_eval(a, sensor, seq, n, xyz, idx, sims)
        return

    # ---- (1) 검색 ----
    queries = [q for q in range(30, n)
               if (xyz[: q - 29] - xyz[q]).norm(dim=1).min() <= 5.0]
    res = {"seq": a.seq, "n_map": n, "n_queries": len(queries), "retrieval": {}}
    top = {}
    bs = 16
    for rep in REPS:
        hit1, hit100, keep = [], [], []
        for s in range(0, len(queries), bs):
            q = torch.as_tensor(queries[s:s + bs], device=DEV)
            S = sims(rep, q_idx=q)
            S[idx[None] > (q[:, None] - 30)] = -torch.inf
            t = S.topk(100, dim=1).indices
            ok = ((xyz[t] - xyz[q][:, None]).norm(dim=2) <= 5.0) & (t <= q[:, None] - 30)
            hit1.append(ok[:, 0])
            hit100.append(ok.any(1))
            keep.append(ok.any(1))
        top[rep] = torch.cat(keep)
        res["retrieval"][rep] = {"R@1": float(torch.cat(hit1).float().mean()),
                                 "R@100": float(torch.cat(hit100).float().mean())}
    res["retrieval"]["r+m union"] = {"R@100+100": float((top["r"] | top["m"]).float().mean())}
    res["retrieval"]["r+scat union"] = {
        "R@100+100": float((top["r"] | top["scat"]).float().mean())}
    print(f"\n== {a.seq}: map {n}, revisit queries {len(queries)}")
    for k, v in res["retrieval"].items():
        print(f" {k:14s} " + "  ".join(f"{m} {100 * x:.1f}%" for m, x in v.items()))

    # ---- (2) 변형 조건의 1위 유지율 ----
    pairs = []
    for q in range(30, n - 1):
        d = (xyz[: q - 29] - xyz[q]).norm(dim=1)
        j = int(d.argmin())
        if d[j] <= 5.0 and (d > 20.0).any():
            pairs.append((q, j))
    if len(pairs) > a.n:
        pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, a.n).astype(int)]
    rng = np.random.default_rng(0)
    keep = {rep: {c: [] for c in CONDS + ("real", "adjacent")} for rep in REPS}
    for k, (q, j) in enumerate(pairs):
        if k % 50 == 0:
            print(f"  deform {k}/{len(pairs)}", flush=True)
        A = loader[int(ids[q])]["points"]
        descs = [encode(deform(A, c, rng)) for c in CONDS]
        neg = (xyz - xyz[q]).norm(dim=1) > 20.0
        for rep in REPS:
            S = sims(rep, q_desc=[d[rep] for d in descs])              # (conds, n)
            for ci, c in enumerate(CONDS):
                keep[rep][c].append(bool(S[ci, q] > S[ci][neg].max()))
            S0 = sims(rep, q_idx=torch.as_tensor([q], device=DEV))[0]
            keep[rep]["real"].append(bool(S0[j] > S0[neg].max()))
            keep[rep]["adjacent"].append(bool(S0[q + 1] > S0[neg].max()))
    res["deform"] = {rep: {c: float(np.mean(v)) for c, v in d.items()}
                     for rep, d in keep.items()}
    res["n_deform_queries"] = len(pairs)
    print(f"\n top-1 retention over {len(pairs)} queries")
    print(" " * 7 + "".join(f"{c:>10s}" for c in CONDS + ("real", "adjacent")))
    for rep in REPS:
        print(f" {rep:6s}" + "".join(f"{100 * res['deform'][rep][c]:9.1f}%"
                                     for c in CONDS + ("real", "adjacent")))
    suffix = f"_limit{a.limit}" if a.limit else ""
    (OUT / f"fullmap_{sensor}_{seq}{suffix}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
