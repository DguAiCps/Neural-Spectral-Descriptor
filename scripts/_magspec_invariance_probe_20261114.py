#!/usr/bin/env python3
"""크기 스펙트럼이 같은 장소에서 왜 달라지는지 원인을 나누는 진단.

검증 시퀀스의 실제 재방문 쌍 (A=질의, B=과거 5 m 안 최근접 키프레임)마다
같은 스캔 A에 변환만 적용한 경우와 실제 재방문 B를 같은 척도로 비교한다.

  yaw90       A를 90도 회전            (칸 정수배, 0이어야 함: 점검용)
  yaw_half    A를 0.5도 / 3도 회전     (거리 영상 / BEV의 반 칸)
  yaw_real    A를 쌍의 (GT 초기값 + ICP) 상대 yaw만큼 회전
  trans_real  A를 쌍의 (GT 초기값 + ICP) 상대 평면 이동만큼 이동
  se2_real    yaw + 평면 이동
  se3_real    ICP로 맞춘 상대 자세 전체 (B 위치에서 본 A의 점들. 가림·동적 물체 변화 없음)
  real        실제 재방문 스캔 B
  adjacent    같은 주행의 다음 키프레임 (참고)
  fwd/lat     A를 앞/옆으로 0.5, 1, 2, 4 m 이동 (용량-반응)

거리는 검색에 쓰는 두 키의 1-코사인이다: 재계량 원키 r(288D), BEV 크기 m(128D).
척도 기준으로 질의마다 가장 가까운 음성(과거, GT 20 m 초과)까지의 거리를 함께 적는다.

    docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
      python3 scripts/_magspec_invariance_probe_20261114.py --seq kitti360/0004
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

from _bevclip_spec import BP, _sketch  # noqa: E402
from _pd_build_test_caches import (  # noqa: E402
    _effective_config, _family, _make_desc_encoder, _make_loader)

OUT = REPO / "results/magspec_invariance_probe_20261114"
KEYS = REPO / "data/preprocessed_v2_odom"
SPEC = REPO / "data/preprocessed_v2_odom_bevclip"
OMEGA = np.load(REPO / "results/pdv2_spectral_keys_20261025/remetric_octave288.npy"
                ).astype(np.float32)
SHIFTS = (0.5, 1.0, 2.0, 4.0)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def yaw_mat(deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def shift_mat(x, y):
    T = np.eye(4)
    T[0, 3], T[1, 3] = x, y
    return T


def apply(T, pts):
    out = pts.copy()
    out[:, :3] = pts[:, :3] @ T[:3, :3].T.astype(pts.dtype) + T[:3, 3].astype(pts.dtype)
    return out


def register(A, B, init):
    """GT 상대 자세를 초기값으로 A를 B에 점-평면 ICP로 맞춘다.

    GT 자세가 센서 좌표가 아닌 데이터셋(KITTI-360은 IMU, z 아래)이 있어
    z·y축을 뒤집은 초기값도 시험하고 더 잘 맞는 쪽을 쓴다.
    반환: 변환, 0.5 m 대응 비율, 대응 RMSE.
    """
    import open3d as o3d

    def cloud(p):
        c = o3d.geometry.PointCloud()
        c.points = o3d.utility.Vector3dVector(
            p[np.linalg.norm(p[:, :3], axis=1) < 50, :3].astype(np.float64))
        c = c.voxel_down_sample(0.4)
        c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
        return c

    a, b = cloud(A), cloud(B)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    icp = o3d.pipelines.registration
    best = None
    for T in (init, flip @ init @ flip):
        for dist in (2.0, 1.0, 0.5):
            res = icp.registration_icp(
                a, b, dist, T, icp.TransformationEstimationPointToPlane())
            T = res.transformation
        if best is None or res.fitness > best.fitness:
            best = res
    return np.asarray(best.transformation), float(best.fitness), float(best.inlier_rmse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    sensor, seq = a.seq.split("/")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = np.load(next(KEYS.glob(f"cache_*_{sensor}_val_{seq}.npz")))
    spec = np.load(SPEC / f"valspec_{sensor}_{seq}.npz")
    assert np.array_equal(cache["scan_ids"], spec["scan_ids"])
    r_bank = unit(cache["descriptors"].astype(np.float32) * OMEGA[None])
    z_bank = (spec["sk_re"] + 1j * spec["sk_im"])[:, :, :8]
    m_bank = unit(np.abs(z_bank).reshape(len(z_bank), -1).astype(np.float32))
    gt = cache["poses_gt"].astype(np.float64)
    xyz = gt[:, :3, 3]

    # 재방문 쌍: 질의 q, 과거 j <= q-30 중 GT 5 m 안 최근접. 고르게 n개.
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
    enc = _make_desc_encoder(_effective_config(), sensor, device)

    def encode(pts):
        with torch.no_grad():
            d = enc.encode_points(pts).detach().cpu().numpy().astype(np.float32)
        bev, _ = BP.project(pts, keep_intensity=False)
        z = _sketch(np.maximum(bev, 0.0))[:, :8]
        return unit(d * OMEGA), unit(np.abs(z).reshape(-1).astype(np.float32))

    rows = []
    for n, (q, j, dist) in enumerate(pairs):
        if n % 50 == 0:
            print(f"  pair {n}/{len(pairs)}", flush=True)
        A = loader[int(cache["scan_ids"][q])]["points"]
        B = loader[int(cache["scan_ids"][j])]["points"]
        rel, fit, rmse = register(A, B, np.linalg.inv(gt[j]) @ gt[q])  # A -> B 좌표
        yaw = float(np.rad2deg(np.arctan2(rel[1, 0], rel[0, 0])))
        tx, ty = rel[0, 3], rel[1, 3]
        rA, mA = encode(A)
        conds = {
            "yaw90": apply(yaw_mat(90.0), A),
            "yaw_half_range": apply(yaw_mat(0.5), A),
            "yaw_half_bev": apply(yaw_mat(3.0), A),
            "yaw_real": apply(yaw_mat(yaw), A),
            "trans_real": apply(shift_mat(tx, ty), A),
            "se2_real": apply(shift_mat(tx, ty) @ yaw_mat(yaw), A),
            "se3_real": apply(rel, A),
            "real": B,
            "adjacent": loader[int(cache["scan_ids"][q + 1])]["points"],
        }
        for s in SHIFTS:
            conds[f"fwd_{s}"] = apply(shift_mat(s, 0.0), A)
            conds[f"lat_{s}"] = apply(shift_mat(0.0, s), A)
        row = {"q": q, "j": j, "gt_dist_m": dist, "gt_yaw_deg": yaw,
               "adjacent_dist_m": float(np.linalg.norm(xyz[q + 1] - xyz[q])),
               "cache_check_r": float(1 - rA @ r_bank[q]),
               "cache_check_m": float(1 - mA @ m_bank[q])}
        for name, pts in conds.items():
            r, m = encode(pts)
            row[f"r_{name}"] = float(1 - rA @ r)
            row[f"m_{name}"] = float(1 - mA @ m)
        neg = np.flatnonzero(np.linalg.norm(xyz[: q - 29] - xyz[q], axis=1) > 20.0)
        row["r_hardneg"] = float(1 - (r_bank[neg] @ rA).max())
        row["m_hardneg"] = float(1 - (m_bank[neg] @ mA).max())
        row["icp_fitness"], row["icp_rmse_m"] = fit, rmse
        rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{sensor}_{seq}.json"
    path.write_text(json.dumps({"seq": a.seq, "n_pairs": len(rows), "rows": rows}))
    print(f"OK {path}", flush=True)


if __name__ == "__main__":
    main()
