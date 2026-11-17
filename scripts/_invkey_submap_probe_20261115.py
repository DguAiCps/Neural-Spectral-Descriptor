#!/usr/bin/env python3
"""invkey 진단: 학습 없이 스캔을 누적하면 반대 방향 재방문과 R@100P가 얼마나 오르는지 상한을 잰다 (2026-11-15).

표현은 바꾸지 않는다. 점유 격자 occupancy()·bank()(scripts/_scatter_probe_20261114.py)와 describe()의 spec
(scripts/_poolalt_probe_20261114.py)를 그대로 부른다. 키는 raw spec과 선형 16(seed 0,
results/invkey_study_20261114/linear/run_final_K16_tau0.05_lr0.003_s0_e40.npz의 s·W를 누적 spec에도 그대로 씀).
검색·PR 회계는 scripts/_invkey_train_linear_20261114.py eval_seq와 같다(질의 q >= 30, 양성 GT <= 5 m, 거리는 float64
직접 차, TF32 끔). 다른 점은 후보 조건에 주행 거리 제외 간격을 더할 수 있다는 것뿐이다:
  후보 j: j <= q-30 그리고 s[q] - s[j] > gap   (s = 키프레임 GT 위치의 누적 주행 거리)

라이다 좌표 자세: T_world_lidar = T_gt @ X (X = extrinsic(sensor))
  kitti360  poses.txt는 IMU(GPS/IMU) 자세 -> X = calib_cam_to_pose(image_00) @ inv(calib_cam_to_velo)
  mulran    global_pose.csv는 차체(base) 자세 -> X = MulRan 공개 보정 base->Ouster
            (x 1.7042, y -0.021, z 1.8047 m, roll 0.0001, pitch 0.0003, yaw 179.6654 도)
  helipr    LiDAR_GT/Velodyne_gt.txt는 Velodyne 좌표 자세 -> X = I
  `align`이 인접·10 m·회전 구간 쌍에서 GT(+보정)로 옮긴 뒤의 잔차와 ICP 보정량으로 이를 확인한다.

누적: 키프레임 i의 라이다 좌표로, 주행 거리 창 [s_i - W, s_i + W_f] 안의 모든 키프레임 스캔을 GT 자세로 옮겨 합친다.
  창 past{W} = [-W, 0], both{W} = [-W, +W], W in {10, 20, 40}. 지도와 질의 모두 같은 방식.
  격자 크기·중심(현재 센서)·1 m 이진·1 m 안 제외는 그대로. 다른 스캔은 자기 좌표 수평 2.5 m 안 점(차체·리그 반사)을 뺀다.
  점은 솎지 않는다(GPU에서 그대로 옮김). 누적 spec은 저장하지 않고 메모리에서 평가만 한다(float16, 저장 특징과 같은 형).
제외 간격: W마다 gap_W = 2W + 10 m (30, 50, 90 m). both 창끼리도 스캔을 공유하지 않는 최소 조건(2W)에 여유 10 m.
  단일 스캔 기준을 gap 0(원래), 30, 50, 90 m에서 같이 잰다. 비대칭 대조 asym20 = 질의 past20, 지도 단일 스캔 (gap 50).
  참고로 모든 조건을 gap 90 m(창끼리 같은 질의 집합)과 200 m(제외 간격 바로 바깥의 같은 길 후보를 더 뺌)에서도 잰다.
  틀린 1위가 제외 간격 바로 바깥(주행 거리 차 < gap + 50 m)인지 따로 센다(wrong_top1).

실행 (리포 루트, 컨테이너):
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_invkey_submap_probe_20261115.py align
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_invkey_submap_probe_20261115.py diagA
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_invkey_submap_probe_20261115.py run --seq kitti360/0004
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_invkey_submap_probe_20261115.py figs --seq kitti360/0004
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_invkey_submap_probe_20261115.py report
출력: results/invkey_study_20261114/submap/ (align.json, diagA.json, run_<sensor>_<seq>.json, metrics.json, *.png)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _invkey_data_survey_20261114 import DIRS, classify, headings  # noqa: E402
from _invkey_train_linear_20261114 import (  # noqa: E402
    DEV, NFFT, OUTD, SURVEY, chmod_write, encode, load_seq, to_freq_t)

OUT = REPO / "results/invkey_study_20261114/submap"
RUN = OUTD / "run_final_K16_tau0.05_lr0.003_s0_e40.npz"
DATA = Path("/workspace/data")
N = 160
SELF_R = 2.5
WINDOWS = {"past10": (10.0, 0.0), "both10": (10.0, 10.0), "past20": (20.0, 0.0),
           "both20": (20.0, 20.0), "past40": (40.0, 0.0), "both40": (40.0, 40.0)}
GAPS = {10: 30.0, 20: 50.0, 40: 90.0}
BINS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf)
BIN_NAMES = ("0-1", "1-2", "2-3", "3-4", "4-5", ">5")
DIAG_A = {"kitti360": ["kitti360/0004", "kitti360/0006"],
          "helipr_roundabout": ["helipr/Roundabout01", "helipr/Roundabout02", "helipr/Roundabout03"],
          "mulran_sejong": ["mulran/Sejong01", "mulran/Sejong02", "mulran/Sejong03"],
          "helipr_dcc(dev)": ["helipr/DCC04", "helipr/DCC05", "helipr/DCC06"],
          "mulran_dcc(dev)": ["mulran/DCC01", "mulran/DCC02", "mulran/DCC03"]}
ALIGN_SEQS = ("kitti360/0004", "helipr/Roundabout01", "helipr/DCC04", "mulran/Sejong03",
              "mulran/DCC02", "mulran/DCC03")


# ---------------------------------------------------------------- 자세
def _rt4(vals):
    T = np.eye(4)
    T[:3, :] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
    return T


def extrinsic(sensor, variant="calib"):
    """GT 자세 좌표 -> 라이다 좌표 변환 X (T_world_lidar = T_gt @ X). variant 'none'은 단위 행렬."""
    if variant == "none" or sensor == "helipr":
        return np.eye(4)
    if sensor == "kitti360":
        cal = DATA / "kitti360/calibration"
        line = [ln for ln in open(cal / "calib_cam_to_pose.txt") if ln.startswith("image_00:")][0]
        c2p = _rt4(line.split()[1:13])
        c2v = _rt4(open(cal / "calib_cam_to_velo.txt").read().split()[:12])
        return c2p @ np.linalg.inv(c2v)
    if sensor == "mulran":
        from scipy.spatial.transform import Rotation
        X = np.eye(4)
        X[:3, :3] = Rotation.from_euler("xyz", [0.0001, 0.0003, 179.6654], degrees=True).as_matrix()
        if variant == "calib_rot_only":
            return X
        X[:3, 3] = [1.7042, -0.021, 1.8047]
        return X
    raise ValueError(sensor)


def load_poses(key, n, xyz):
    """특징 파일 행과 같은 순서의 GT 자세 (n,4,4) float64. 위치가 특징 xyz와 같은지 확인한다."""
    d = json.load(open(SURVEY))
    rec = {s["key"]: s for s in d["sequences"]}[key]
    P = np.asarray(np.load(REPO / rec["cache"])[rec["pose_field"]], dtype=np.float64)[:n]
    assert np.array_equal(P[:, :3, 3], xyz), key
    return P


def path_length(xyz):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]


def seq_data(key):
    spec, xyz, ids = load_seq(key)
    P = load_poses(key, len(spec), xyz)
    return spec, xyz, ids, P


# ---------------------------------------------------------------- 평가
def evaluate(Dq, Dm, xyz64, s, gaps, chunk):
    """eval_seq와 같은 회계. Dq(질의 키), Dm(지도 키) (n, R, 16) GPU. gaps (m) 목록마다 결과.
    반환 {gap: (지표, 질의별 배열)}. gap 0은 원래 조건(j <= q-30)만."""
    import torch
    from _poolalt_probe_20261114 import aligned_sim
    from _pr_eval_lib import classify_queries, pr_sweep
    Fq, nq = to_freq_t(Dq)
    Fm, nm = (Fq, nq) if Dm is Dq else to_freq_t(Dm)
    assert bool((nq > 0).all()) and bool((nm > 0).all()), "zero-norm key"
    n = Dq.shape[0]
    xyz = torch.as_tensor(xyz64, dtype=torch.float64, device=DEV)
    st = torch.as_tensor(s, dtype=torch.float64, device=DEV)
    idx = torch.arange(n, device=DEV)
    qs = torch.arange(30, n, device=DEV)
    acc = {g: {k: [] for k in ("score", "top1d", "dmin", "jnn", "hit100", "t1")} for g in gaps}
    for c in range(0, len(qs), chunk):
        q = qs[c:c + chunk]
        S0 = aligned_sim(Fq[:, q], nq[q], Fm, nm, NFFT)
        Dist = (xyz[q][:, None] - xyz[None]).norm(dim=2)
        base = idx[None] <= (q[:, None] - 30)
        for g in gaps:
            past = base & ((st[q][:, None] - st[None]) > g) if g > 0 else base
            Dm_ = Dist.masked_fill(~past, torch.inf)
            S = S0.masked_fill(~past, -torch.inf)
            v, t = S.topk(min(100, n), dim=1)
            ok = past.gather(1, t)
            a = acc[g]
            a["score"].append(v[:, 0])
            a["t1"].append(t[:, 0])
            a["top1d"].append(Dm_.gather(1, t[:, :1])[:, 0])
            dm, jm = Dm_.min(dim=1)
            a["dmin"].append(dm)
            a["jnn"].append(jm)
            a["hit100"].append(((Dm_.gather(1, t) <= 5.0) & ok).any(1))
    out = {}
    for g in gaps:
        a = acc[g]
        score = torch.cat(a["score"]).double().cpu().numpy()
        top1d, dmin = (torch.cat(a[k]).cpu().numpy() for k in ("top1d", "dmin"))
        jnn, hit100 = torch.cat(a["jnn"]).cpu().numpy(), torch.cat(a["hit100"]).cpu().numpy()
        t1 = torch.cat(a["t1"]).cpu().numpy()
        cls = classify_queries(dmin)
        pos = cls["positive"]
        pr = pr_sweep(score, top1d, cls)
        elig = pos | cls["negative"]
        fp10 = elig & (top1d > 10.0)
        thr = score[fp10].max() if fp10.any() else -np.inf
        r100p_10 = float((pos & (top1d <= 10.0) & (score > thr)).sum() / pos.sum()) if pos.any() else 0.0
        hit1 = top1d <= 5.0
        m = {"R@1": float(hit1[pos].mean()) if pos.any() else None,
             "R@100": float(hit100[pos].mean()) if pos.any() else None,
             "AP": float(pr["AP"]), "max_F1": float(pr["max_F1"]),
             "R@100P": float(pr["R@100P"]), "R@100P_10m": r100p_10,
             "n_pos": cls["n_pos"], "n_neg": cls["n_neg"], "n_ambiguous": cls["n_ambiguous"],
             "n_hit1": int(hit1[pos].sum()), "n_map": int(n), "gap_m": g}
        # 틀린 1위가 주행 경로상 어디인가: 질의와의 주행 거리 차(제외 간격 바로 바깥인지, 다른 주행인지)와 직선 거리
        qa = np.arange(30, n)
        wrong = pos & ~hit1
        pg = s[qa[wrong]] - s[t1[wrong]]
        m["wrong_top1"] = {"n": int(wrong.sum()),
                           "path_gap_lt_gap+50": int((pg < g + 50).sum()),
                           "path_gap_gap+50_to_gap+200": int(((pg >= g + 50) & (pg < g + 200)).sum()),
                           "path_gap_ge_gap+200": int((pg >= g + 200).sum()),
                           "dist_med_m": float(np.median(top1d[wrong])) if wrong.any() else None}
        out[g] = (m, {"pos": pos, "hit1": hit1, "q": qa, "jnn": jnn, "t1": t1})
    return out


def offsets(xyz64, PL, q, j):
    """질의 진행 방향(조사 스크립트 headings, ±3 키프레임) 기준 (옆, 진행) 어긋남 |.| (m). PL: 라이다 위치."""
    u, moving = headings(np.asarray(xyz64, dtype=np.float64))
    uq = u[q][:, :2]
    nrm = np.linalg.norm(uq, axis=1)
    ok = moving[q] & (nrm > 0)
    uq = np.where(ok[:, None], uq / np.maximum(nrm, 1e-12)[:, None], np.nan)
    d = (PL[j] - PL[q])[:, :2]
    lon = np.abs((d * uq).sum(1))
    lat = np.abs(d[:, 0] * uq[:, 1] - d[:, 1] * uq[:, 0])
    return lat, lon


def breakdown(xyz64, PL, arrs):
    """양성 질의의 방향 × (옆/진행 어긋남 구간)별 질의 수와 1위 정답 수."""
    pos = arrs["pos"]
    q, j, h = arrs["q"][pos], arrs["jnn"][pos], arrs["hit1"][pos]
    u, moving = headings(np.asarray(xyz64, dtype=np.float64))
    dirs = classify(u, moving, q, j)
    res = {"dir_n": {d: int((dirs == i).sum()) for i, d in enumerate(DIRS)},
           "dir_hit1": {d: int(h[dirs == i].sum()) for i, d in enumerate(DIRS)}}
    for pname, P in (("lidar", PL), ("gt", np.asarray(xyz64))):
        lat, lon = offsets(xyz64, P, q, j)
        for oname, v in (("lat", lat), ("lon", lon)):
            b = np.digitize(v, BINS[1:-1])           # 0..5, nan -> 6
            b[~np.isfinite(v)] = -1
            tab = {}
            for di, d in enumerate(DIRS[:3]):
                tab[d] = {"n": [int(((dirs == di) & (b == k)).sum()) for k in range(len(BIN_NAMES))],
                          "hit": [int((h & (dirs == di) & (b == k)).sum()) for k in range(len(BIN_NAMES))]}
            res[f"{oname}_{pname}"] = tab
    return res


def load_lin16():
    import torch
    z = np.load(RUN)
    return torch.as_tensor(z["s"], device=DEV), torch.as_tensor(z["W"], device=DEV)


# ---------------------------------------------------------------- 진단 A
def diag_a(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = json.load(open(OUTD / "metrics.json"))
    s16, W16 = load_lin16()
    res = {"groups": DIAG_A, "bins": BIN_NAMES, "per_seq": {}}
    t0 = time.time()
    for g, keys in DIAG_A.items():
        for key in keys:
            spec, xyz, ids, P = seq_data(key)
            PL = (P @ extrinsic(key.split("/")[0]))[:, :3, 3]
            res["per_seq"][key] = {}
            for rep, (s, W) in (("raw", (None, None)), ("lin16", (s16, W16))):
                with torch.no_grad():
                    D = encode(spec, s, W)
                    m, arrs = evaluate(D, D, xyz, path_length(xyz), [0.0], 16 if W is None else 64)[0.0]
                del D
                torch.cuda.empty_cache()
                m.update(breakdown(xyz, PL, arrs))
                # 기존 수치와 대조 (held-out은 metrics.json heldout_per_seq, dev raw는 dev_controls)
                rname = "raw" if rep == "raw" else "lin16_s0"
                if key in ref["heldout_per_seq"]:
                    r = ref["heldout_per_seq"][key][rname]
                elif rep == "raw":
                    r = ref["dev_controls"]["reps"]["raw"]["per_seq"][key]
                else:
                    r = None
                if r is not None:
                    m["ref_check"] = {k: (m[k], r[k]) for k in ("R@1", "AP", "R@100P") if k in r}
                    m["ref_ok"] = all(abs(x - y) < 1e-9 for x, y in m["ref_check"].values())
                res["per_seq"][key][rep] = m
            print(f"{key:22s} raw R@1 {100 * res['per_seq'][key]['raw']['R@1']:.1f}"
                  f" lin16 {100 * res['per_seq'][key]['lin16']['R@1']:.1f}"
                  f"  ref_ok {[res['per_seq'][key][r].get('ref_ok') for r in ('raw', 'lin16')]}"
                  f"  {time.time() - t0:.0f}s", flush=True)
    chmod_write(OUT / "diagA.json", text=json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------- 정합 검증
def _cloud(p, voxel=0.3):
    import open3d as o3d
    c = o3d.geometry.PointCloud()
    r = np.hypot(p[:, 0], p[:, 1])
    c.points = o3d.utility.Vector3dVector(p[(r > SELF_R) & (r < 60), :3].astype(np.float64))
    c = c.voxel_down_sample(voxel)
    c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    return c


def icp(src, tgt, init):
    import open3d as o3d
    reg = o3d.pipelines.registration
    a, b = _cloud(src), _cloud(tgt)
    T = init
    for dist in (1.0, 0.5, 0.25):
        r = reg.registration_icp(a, b, dist, T, reg.TransformationEstimationPointToPlane())
        T = r.transformation
    return np.asarray(T), float(r.fitness)


def structure(p, g):
    """비지면 구조 점: 수평 3-40 m, 지면(g) + 0.5 ~ + 4 m."""
    r = np.hypot(p[:, 0], p[:, 1])
    return p[(r >= 3) & (r <= 40) & (p[:, 2] > g + 0.5) & (p[:, 2] < g + 4.0), :3]


def ground_z(p):
    r = np.hypot(p[:, 0], p[:, 1])
    return float(np.percentile(p[(r >= 3) & (r <= 15), 2], 10))


def nn_xy(p, tgt_struct):
    from scipy.spatial import cKDTree
    return cKDTree(tgt_struct[:, :2]).query(p[:, :2])[0]


def residual(d, ov):
    """수평 최근접 거리 d: 전체 중앙값, 0.2 m 안 비율, 겹침 부분집합(ov: ICP 정합 뒤 0.5 m 안인 점)의 중앙값."""
    return {"med": float(np.median(d)), "f02": float((d < 0.2).mean()),
            "med_ov": float(np.median(d[ov])) if ov.any() else None, "n": int(len(d)), "n_ov": int(ov.sum())}


def apply_T(T, p):
    return p[:, :3] @ T[:3, :3].T + T[:3, 3]


def yaw_of(T):
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def align(a):
    from _pd_build_test_caches import _make_loader
    rng = np.random.default_rng(0)
    out = {}
    for key in a.seqs:
        sensor, seq = key.split("/")
        spec, xyz, ids, P = seq_data(key)
        del spec
        s = path_length(xyz)
        loader = _make_loader(sensor, seq)
        variants = ["none", "calib"] + (["calib_rot_only"] if sensor == "mulran" else [])
        if sensor == "helipr":
            variants = ["calib"]          # GT가 이미 Velodyne 좌표 (X = I)
        n = len(xyz)
        b10 = np.searchsorted(s, s + 10.0)
        ok10 = np.flatnonzero(b10 < n)
        pairs = [("adjacent", int(i), int(i) + 1) for i in rng.choice(n - 1, a.n, replace=False)]
        pairs += [("10m", int(i), int(b10[i])) for i in rng.choice(ok10, a.n, replace=False)]
        # 회전 구간: 10 m 쌍 중 상대 yaw가 큰 것 (서로 50 키프레임 이상 떨어지게)
        yaws = np.array([abs(yaw_of(np.linalg.inv(P[i]) @ P[b10[i]])) for i in ok10])
        chosen = []
        for i in ok10[np.argsort(-yaws)]:
            if all(abs(i - c) > 50 for c in chosen):
                chosen.append(int(i))
            if len(chosen) >= a.n_turn:
                break
        pairs += [("turn10m", i, int(b10[i])) for i in chosen]
        rows = []
        for kind, i, j in pairs:
            A = loader[int(ids[i])]["points"]
            B = loader[int(ids[j])]["points"]
            sa = structure(A, ground_z(A))
            sb = structure(B, ground_z(B))          # B 좌표에서 고른 구조 점 (변환마다 같은 점 집합)
            row = {"kind": kind, "i": i, "j": j, "dist": float(np.linalg.norm(xyz[j] - xyz[i])),
                   "yaw_deg": yaw_of(np.linalg.inv(P[i]) @ P[j])}
            Xc = extrinsic(sensor, "calib")
            Tc = np.linalg.inv(P[i] @ Xc) @ (P[j] @ Xc)        # B 라이다 좌표 -> A 라이다 좌표
            Ti, fit = icp(B, A, Tc)
            dT = Ti @ np.linalg.inv(Tc)
            row["icp_dt_m"] = float(np.linalg.norm(dT[:2, 3]))
            row["icp_dz_m"] = float(abs(dT[2, 3]))
            row["icp_drot_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1))))
            row["icp_fitness"] = fit
            di = nn_xy(apply_T(Ti, sb), sa)
            ov = di < 0.5
            row["res_icp"] = residual(di, ov)
            for v in variants:
                X = extrinsic(sensor, v)
                T = np.linalg.inv(P[i] @ X) @ (P[j] @ X)
                row[f"res_{v}"] = residual(nn_xy(apply_T(T, sb), sa), ov)
            rows.append(row)
        summ = {}
        for kind in ("adjacent", "10m", "turn10m"):
            rs = [r for r in rows if r["kind"] == kind]
            e = {"n": len(rs), "dist_med": float(np.median([r["dist"] for r in rs])),
                 "abs_yaw_med": float(np.median([abs(r["yaw_deg"]) for r in rs]))}
            for v in variants + ["icp"]:
                e[f"res_{v}_med"] = float(np.median([r[f"res_{v}"]["med"] for r in rs]))
                e[f"res_{v}_f02"] = float(np.median([r[f"res_{v}"]["f02"] for r in rs]))
                e[f"res_{v}_med_ov"] = float(np.median([r[f"res_{v}"]["med_ov"] for r in rs
                                                        if r[f"res_{v}"]["med_ov"] is not None]))
            for k in ("icp_dt_m", "icp_dz_m", "icp_drot_deg"):
                v = np.array([r[k] for r in rs])
                e[k] = {"med": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                        "max": float(v.max())}
            summ[kind] = e
        out[key] = {"variants": variants, "summary": summ, "rows": rows}
        print(f"\n== {key}")
        for kind, e in summ.items():
            print(f" {kind:9s} n {e['n']} d {e['dist_med']:.1f} |yaw| {e['abs_yaw_med']:.1f}  "
                  + "  ".join(f"res_{v} {e[f'res_{v}_med']:.3f}/{e[f'res_{v}_f02']:.2f}/ov {e[f'res_{v}_med_ov']:.3f}"
                              for v in variants + ["icp"])
                  + f"  icp dt {e['icp_dt_m']['med']:.3f}/{e['icp_dt_m']['p90']:.3f}"
                    f" drot {e['icp_drot_deg']['med']:.2f}/{e['icp_drot_deg']['p90']:.2f}", flush=True)
    path = OUT / ("align.json" if not a.tag else f"align_{a.tag}.json")
    chmod_write(path, text=json.dumps(out, indent=1))
    return 0


# ---------------------------------------------------------------- 누적
class Accumulator:
    """창 안 키프레임 스캔을 GPU 고정 칸(slot) 버퍼에 들고, 키프레임 i 좌표의 칸 집합을 스캔별로 한 번에 만든다.

    스캔 k는 칸 k % K에 들어간다(K = 가장 큰 창의 키프레임 수). 키프레임마다 모든 칸을 batched 행렬곱 하나로 옮긴다.
    현재 스캔은 단위 변환(정확히 I)이라 occupancy()와 같은 칸이 나온다(run에서 확인). 읽기는 스레드로 미리 한다."""

    def __init__(self, loader, ids, TL, s, wmax):
        from concurrent.futures import ThreadPoolExecutor
        self.loader, self.ids, self.TL, self.s = loader, ids, TL, s
        self.n = len(ids)
        self.lo = np.searchsorted(s, s - wmax, side="left")
        self.hi = np.searchsorted(s, s + wmax, side="right")
        self.K = int((self.hi - self.lo).max())
        self.slot_kf = np.full(self.K, -1, dtype=np.int64)
        self.cap = 0
        self.P = self.valid = self.far = None
        self.pool = ThreadPoolExecutor(4)
        self.fut = {}
        self.next_submit = self.next_put = 0

    def _fetch(self, k):
        return np.ascontiguousarray(self.loader[int(self.ids[k])]["points"][:, :3], dtype=np.float32)

    def _grow(self, m):
        import torch
        cap = int(m * 1.2) + 1
        P = torch.zeros(self.K, cap, 3, device=DEV)
        valid = torch.zeros(self.K, cap, dtype=torch.bool, device=DEV)
        far = torch.zeros(self.K, cap, dtype=torch.bool, device=DEV)
        if self.P is not None:
            P[:, :self.cap], valid[:, :self.cap], far[:, :self.cap] = self.P, self.valid, self.far
        self.P, self.valid, self.far, self.cap = P, valid, far, cap

    def _put(self, k, pts):
        import torch
        m = len(pts)
        if m > self.cap:
            self._grow(m)
        sl = k % self.K
        p = torch.from_numpy(pts).to(DEV)
        self.P[sl].zero_()
        self.P[sl, :m] = p
        self.valid[sl] = False
        self.valid[sl, :m] = True
        self.far[sl] = False
        self.far[sl, :m] = torch.hypot(p[:, 0], p[:, 1]) >= SELF_R
        self.slot_kf[sl] = k

    def cells(self, i):
        """(칸별 키프레임 번호 (K,), 창 [lo_i, hi_i) 밖 칸은 -1) 와 (K, N*N) bool 칸 표."""
        import torch
        lo, hi = int(self.lo[i]), int(self.hi[i])
        for k in range(self.next_submit, min(self.n, hi + 32)):
            self.fut[k] = self.pool.submit(self._fetch, k)
        self.next_submit = max(self.next_submit, min(self.n, hi + 32))
        for k in range(self.next_put, hi):
            self._put(k, self.fut.pop(k).result())
        self.next_put = max(self.next_put, hi)
        kf = np.where((self.slot_kf >= lo) & (self.slot_kf < hi), self.slot_kf, -1)
        Ti = np.linalg.inv(self.TL[i])
        T = np.tile(np.eye(4), (self.K, 1, 1))
        for sl in np.flatnonzero(kf >= 0):
            if kf[sl] != i:
                T[sl] = Ti @ self.TL[kf[sl]]
        Tg = torch.as_tensor(T[:, :3, :], dtype=torch.float32, device=DEV)
        q = torch.baddbmm(Tg[:, None, :, 3], self.P, Tg[:, :, :3].transpose(1, 2))   # (K, cap, 3)
        x, y = q[..., 0], q[..., 1]
        ix, iy = torch.floor(x + 80.0).long(), torch.floor(y + 80.0).long()
        inwin = torch.as_tensor(kf >= 0, device=DEV)
        center = torch.as_tensor(kf == i, device=DEV)
        ok = (self.valid & (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (torch.hypot(x, y) >= 1.0)
              & inwin[:, None] & (self.far | center[:, None]))
        flat = (torch.arange(self.K, device=DEV)[:, None] * (N * N) + ix * N + iy)[ok]
        occ = torch.zeros(self.K * N * N, dtype=torch.bool, device=DEV)
        occ[flat] = True
        return kf, occ.view(self.K, N * N)


def run(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from _pd_build_test_caches import _make_loader
    from _poolalt_probe_20261114 import describe
    from _scatter_probe_20261114 import bank, occupancy
    key = a.seq
    sensor, seq = key.split("/")
    spec1, xyz, ids, P = seq_data(key)
    n = len(spec1) if not a.limit else a.limit
    spec1, xyz, ids, P = spec1[:n], xyz[:n], ids[:n], P[:n]
    TL = P @ extrinsic(sensor)
    PL = TL[:, :3, 3]
    s = path_length(xyz)
    windows = {c: WINDOWS[c] for c in a.conds}
    wmax = max(max(w) for w in windows.values())
    loader = _make_loader(sensor, seq)
    F = torch.from_numpy(bank()).to(DEV)
    acc = Accumulator(loader, ids, TL, s, wmax)
    specs = {c: np.empty((n, 805, 16), np.float16) for c in windows}
    ksz = {c: [] for c in windows}
    chk = {"grid_equal": 0, "max_rel_err_center": 0.0, "n": 0}
    t0 = time.time()
    for i in range(n):
        if i % 1000 == 0:
            el = time.time() - t0
            print(f"  [{key}] {i}/{n}  {el:.0f}s  {i / el if el else 0:.1f} kf/s", flush=True)
        kf, occ = acc.cells(i)
        sk = np.where(kf >= 0, s[np.maximum(kf, 0)], np.nan)
        if i % 500 == 0:   # 점검: 현재 스캔만의 칸 표 == occupancy(), describe == 저장 spec
            g0 = occupancy(loader[int(ids[i])]["points"])
            gc = occ[int(np.flatnonzero(kf == i)[0])].reshape(N, N).float().cpu().numpy()
            chk["grid_equal"] += int(np.array_equal(g0, gc))
            d0 = describe(gc, F)[1]
            ref = spec1[i].astype(np.float32)
            chk["max_rel_err_center"] = max(chk["max_rel_err_center"],
                                            float(np.linalg.norm(d0 - ref) / np.linalg.norm(ref)))
            chk["n"] += 1
        for c, (wb, wf) in windows.items():
            m = torch.as_tensor((kf >= 0) & (sk >= s[i] - wb) & (sk <= s[i] + wf), device=DEV)
            g = occ[m].any(0).reshape(N, N).float().cpu().numpy()
            specs[c][i] = describe(g, F)[1]
            ksz[c].append(int(m.sum()))
    t_acc = time.time() - t0
    print(f"  [{key}] accumulated {n} kf x {len(windows)} windows in {t_acc:.0f}s  check {chk}",
          flush=True)
    acc.pool.shutdown()
    del acc
    torch.cuda.empty_cache()

    s16, W16 = load_lin16()
    # 주 비교는 창마다 gap_W. 창끼리 같은 질의 집합으로 보려고 90 m(가장 큰 gap_W), 다른 주행만 남기는 200 m도 같이 잰다.
    conds = {"single": ("single", "single", [0.0] + sorted(GAPS.values()) + [200.0])}
    for c in windows:
        conds[c] = (c, c, sorted({GAPS[int(windows[c][0])], 90.0, 200.0}))
    if "past20" in windows:
        conds["asym20"] = ("past20", "single", [GAPS[20], 90.0, 200.0])
    res = {"seq": key, "n": n, "windows": {c: list(w) for c, w in windows.items()},
           "gaps_m": GAPS, "self_filter_m": SELF_R, "accumulate_s": round(t_acc, 1), "center_check": chk,
           "scans_per_window": {c: {"med": float(np.median(v)), "min": int(np.min(v)), "max": int(np.max(v))}
                                for c, v in ksz.items()},
           "lin16_note": "s, W from single-scan training applied unchanged to accumulated spec",
           "eval": {}}
    src = dict(specs, single=spec1)
    for rep, (sv, Wv) in (("raw", (None, None)), ("lin16", (s16, W16))):
        res["eval"][rep] = {}
        for cname, (cq, cm, gaps) in conds.items():
            t1 = time.time()
            with torch.no_grad():
                Dq = encode(src[cq], sv, Wv)
                Dm = Dq if cm == cq else encode(src[cm], sv, Wv)
                ev = evaluate(Dq, Dm, xyz, s, gaps, 16 if Wv is None else 64)
            del Dq, Dm
            torch.cuda.empty_cache()
            res["eval"][rep][cname] = {}
            for g, (m, arrs) in ev.items():
                m.update(breakdown(xyz, PL, arrs))
                res["eval"][rep][cname][str(int(g))] = m
                rv = m["dir_hit1"]["reverse"] / m["dir_n"]["reverse"] if m["dir_n"]["reverse"] else float("nan")
                pc = lambda x: "-" if x is None else f"{100 * x:.1f}"  # noqa: E731
                print(f"  {rep:5s} {cname:7s} gap {g:4.0f}  pos {m['n_pos']:5d}  R@1 {pc(m['R@1'])}"
                      f"  R@100 {pc(m['R@100'])}  AP {100 * m['AP']:.1f}  R@100P {100 * m['R@100P']:.1f}"
                      f"  R@100P10 {100 * m['R@100P_10m']:.1f}  rev {100 * rv:.1f} (n {m['dir_n']['reverse']})"
                      f"  {time.time() - t1:.0f}s", flush=True)
    sfx = f"_limit{a.limit}" if a.limit else ""
    chmod_write(OUT / f"run_{sensor}_{seq}{sfx}.json", text=json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------- 추가 점검: 사각 부채꼴 대칭 지우기 (요청 밖)
# 라이다 좌표 방위(도)에서 센서 리그가 가리는 부채꼴 중심과 반폭. mulran: 차량 뒤쪽(라이다 0도) 330-30도에 점이 없음.
# helipr: 차량 앞쪽(라이다 0도) 340-20도의 점 밀도가 평균의 0.2-0.5.
BLIND = {"mulran": (0.0, 40.0), "helipr": (0.0, 30.0)}


def sector(a):
    """단일 스캔에서 사각 부채꼴과 그 180도 맞은편을 함께 지워(mirror) 반대 방향 재방문의 보이는 방위를 맞춘다.
    대조(perp): 같은 폭의 부채꼴 두 개를 90도 돌린 위치에서 지운다. 같은 occupancy -> describe, gap 0 평가."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from concurrent.futures import ThreadPoolExecutor
    from _pd_build_test_caches import _make_loader
    from _poolalt_probe_20261114 import describe
    from _scatter_probe_20261114 import bank, occupancy
    key = a.seq
    sensor, seq = key.split("/")
    c0, hw = BLIND[sensor]
    spec1, xyz, ids, P = seq_data(key)
    n = len(spec1)
    PL = (P @ extrinsic(sensor))[:, :3, 3]
    loader = _make_loader(sensor, seq)
    F = torch.from_numpy(bank()).to(DEV)
    variants = {"mirror": (c0, c0 + 180.0), "perp": (c0 + 90.0, c0 + 270.0)}
    specs = {v: np.empty((n, 805, 16), np.float16) for v in variants}
    kept = {v: [] for v in variants}
    pool = ThreadPoolExecutor(4)
    futs = {}
    t0 = time.time()
    for i in range(n):
        for k in range(i, min(n, i + 32)):
            if k not in futs:
                futs[k] = pool.submit(lambda k: loader[int(ids[k])]["points"], k)
        pts = futs.pop(i).result()
        az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        for v, cs in variants.items():
            drop = np.zeros(len(pts), bool)
            for c in cs:
                drop |= np.abs((az - c + 180.0) % 360.0 - 180.0) < hw
            specs[v][i] = describe(occupancy(pts[~drop]), F)[1]
            kept[v].append(float((~drop).mean()))
        if i % 1000 == 0:
            print(f"  [{key}] sector {i}/{n} {time.time() - t0:.0f}s", flush=True)
    pool.shutdown()
    s16, W16 = load_lin16()
    s = path_length(xyz)
    res = {"seq": key, "n": n, "blind_center_deg": c0, "half_width_deg": hw,
           "variants": {v: list(c) for v, c in variants.items()},
           "kept_point_fraction": {v: float(np.mean(k)) for v, k in kept.items()}, "eval": {}}
    for rep, (sv, Wv) in (("raw", (None, None)), ("lin16", (s16, W16))):
        res["eval"][rep] = {}
        for v in ["single"] + list(variants):
            X = spec1 if v == "single" else specs[v]
            with torch.no_grad():
                D = encode(X, sv, Wv)
                m, arrs = evaluate(D, D, xyz, s, [0.0], 16 if Wv is None else 64)[0.0]
            del D
            torch.cuda.empty_cache()
            m.update(breakdown(xyz, PL, arrs))
            res["eval"][rep][v] = m
            dn, dh = m["dir_n"], m["dir_hit1"]
            print(f"  {rep:5s} {v:6s} R@1 {100 * m['R@1']:.1f} AP {100 * m['AP']:.1f} R@100P {100 * m['R@100P']:.1f}"
                  f"  same {100 * dh['same'] / max(dn['same'], 1):.1f} rev {100 * dh['reverse'] / max(dn['reverse'], 1):.1f}"
                  f" (n {dn['reverse']}) cross {100 * dh['crossing'] / max(dn['crossing'], 1):.1f}", flush=True)
    chmod_write(OUT / f"sector_{sensor}_{seq}.json", text=json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------- 그림
def figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _pd_build_test_caches import _make_loader
    key = a.seq
    sensor, seq = key.split("/")
    spec, xyz, ids, P = seq_data(key)
    del spec
    TL = P @ extrinsic(sensor)
    s = path_length(xyz)
    n = len(xyz)
    loader = _make_loader(sensor, seq)
    # 고르는 규칙(고정): 회전이 가장 큰 곳 1개(±20 m 창 안 yaw 변화), 직선 1개(같은 창 yaw 변화 최소), 시퀀스 중간 1개
    lo = np.searchsorted(s, s - 20.0)
    hi = np.minimum(np.searchsorted(s, s + 20.0, side="right") - 1, n - 1)
    yaw = np.unwrap(np.arctan2(P[:, 1, 0], P[:, 0, 0]))
    dyaw = np.degrees(np.abs(yaw[hi] - yaw[lo]))
    valid = np.flatnonzero((s - s[0] > 45) & (s[-1] - s > 45))
    picks = {"turn": int(valid[np.argmax(dyaw[valid])]), "straight": int(valid[np.argmin(dyaw[valid])]),
             "middle": int(valid[len(valid) // 2])}
    for name, i in picks.items():
        ks = np.flatnonzero((s >= s[i] - 20.0) & (s <= s[i] + 20.0))
        Ti = np.linalg.inv(TL[i])
        pts = []
        for k in ks:
            p = loader[int(ids[k])]["points"][:, :3].astype(np.float64)
            if k != i:
                p = p[np.hypot(p[:, 0], p[:, 1]) >= SELF_R]
                p = apply_T(Ti @ TL[k], p)
            pts.append((k == i, p))
        single = [p for c, p in pts if c][0]
        allp = np.concatenate([p for _, p in pts])
        g = ground_z(single)

        def raster(p, half, res):
            p = p[(p[:, 2] > g + 0.5) & (p[:, 2] < g + 4.0) & (np.hypot(p[:, 0], p[:, 1]) >= SELF_R)]
            m = int(2 * half / res)
            ix = np.floor((p[:, 0] + half) / res).astype(int)
            iy = np.floor((p[:, 1] + half) / res).astype(int)
            ok = (ix >= 0) & (ix < m) & (iy >= 0) & (iy < m)
            img = np.zeros((m, m))
            np.add.at(img, (ix[ok], iy[ok]), 1)
            return np.log1p(img).T     # x 오른쪽, y 위

        def occ(p):
            x, y = p[:, 0], p[:, 1]
            o = np.zeros((N, N))
            ix, iy = np.floor(x + 80).astype(int), np.floor(y + 80).astype(int)
            ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (np.hypot(x, y) >= 1)
            o[ix[ok], iy[ok]] = 1
            return o.T

        fig, ax = plt.subplots(2, 2, figsize=(13, 13))
        for c, (lab, p) in enumerate((("single scan", single), (f"both 20 m ({len(ks)} scans)", allp))):
            im = raster(p, 25.0, 0.1)
            ax[0, c].imshow(im, origin="lower", extent=(-25, 25, -25, 25), cmap="magma",
                            vmax=max(float(np.percentile(im[im > 0], 99)), 1e-6) if (im > 0).any() else 1.0)
            ax[0, c].set_title(f"{lab}: non-ground points, 0.1 m, +-25 m")
            ax[1, c].imshow(occ(p), origin="lower", extent=(-80, 80, -80, 80), cmap="gray_r")
            ax[1, c].set_title(f"{lab}: occupancy 1 m, 160 m (descriptor input)")
        fig.suptitle(f"{key} keyframe {i} ({name}, yaw change +-20 m: {dyaw[i]:.0f} deg) - lidar frame")
        fig.tight_layout()
        path = OUT / f"fig_{sensor}_{seq}_{name}_{i}.png"
        fig.savefig(path, dpi=80)
        plt.close(fig)
        path.chmod(0o666)
        print(f"saved {path}", flush=True)
    return 0


# ---------------------------------------------------------------- 모으기
def report(a):
    out = {"date": "2026-11-15", "script": "scripts/_invkey_submap_probe_20261115.py",
           "windows": WINDOWS, "gaps_m": GAPS, "bins_m": BIN_NAMES}
    for f in ("align.json", "diagA.json"):
        if (OUT / f).is_file():
            out[f[:-5]] = json.load(open(OUT / f))
    out["sector_extra"] = {json.load(open(p))["seq"]: json.load(open(p)) for p in sorted(OUT.glob("sector_*.json"))}
    out["runs"] = {}
    for p in sorted(OUT.glob("run_*.json")):
        if "_limit" in p.name:
            continue
        d = json.load(open(p))
        out["runs"][d["seq"]] = d
    chmod_write(OUT / "metrics.json", text=json.dumps(out, indent=1))
    print(f"metrics.json: runs {list(out['runs'])}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("align")
    p.add_argument("--seqs", nargs="*", default=list(ALIGN_SEQS))
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--n-turn", type=int, default=10)
    p.add_argument("--tag", default="")
    sub.add_parser("diagA")
    p = sub.add_parser("run")
    p.add_argument("--seq", required=True)
    p.add_argument("--conds", nargs="*", default=list(WINDOWS))
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("figs")
    p.add_argument("--seq", required=True)
    p = sub.add_parser("sector")
    p.add_argument("--seq", required=True)
    sub.add_parser("report")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"align": align, "diagA": diag_a, "run": run, "figs": figs, "sector": sector,
            "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
