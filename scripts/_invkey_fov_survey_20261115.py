#!/usr/bin/env python3
"""invkey 진단: 데이터셋·센서별 방위 사각(리그 가림) 조사와 방향별 R@1 (2026-11-15). 학습 없음.

1. 방위 사각 (scan, ouster)
  시퀀스마다 키프레임(특징 파일 scan_ids)에서 고르게 200스캔. 점마다 줄(아래 표)과 센서 좌표 방위(5도 칸 72개).
    세계 반환  수평 거리 >= 2.5 m          자기 반환  3차원 거리 < 1 m (반환 있음 = 3차원 > 0.05 m)
    중간       나머지(1 m <= 3차원, 수평 < 2.5 m; 차체·리그 가장자리나 아주 가까운 물체)
  반환율 = 반환 수 / 칸당 발사 수. 발사 수: Ouster(mulran, mcd, HeLiPR OS2-128)는 열 1024개 / 72칸,
    VLP-16은 1800발 / 72칸(10 Hz), HDL-64E·HDL-32E는 줄별 발사 수를 몰라 줄마다 (반환 있는 점 수의 방위 칸
    95% 분위)를 쓰되 줄 중앙값의 절반 아래로는 내리지 않는다(죽은 줄이 0이 되게).
  사각 판정(한 가지로 고정, 모든 데이터셋 같음):
    시퀀스 안에서 줄 r의 기준 M_r = 그 시퀀스 표본 전체로 모은 칸별 세계 반환율의 줄 최대값.
    칸 (r, b)는 표본 스캔의 95% 이상에서 그 스캔의 칸 세계 반환율 < theta * M_r 이면 그 시퀀스에서 사각.
    단위(같은 리그로 보는 묶음)에서는 단위의 모든 시퀀스(장소·날짜가 다름)에서 사각일 때만 사각.
    theta = 0.05 고정. 민감도는 theta 0.025, 0.10 (그리고 지속 비율 0.90, 0.99)로 따로 잰다.
    M_r < 0.05(절대)인 줄은 '빈 줄'(죽었거나 거의 반환 없음)로 따로 세고 사각 지표에서 뺀다.
  줄 (원래 번호; 표와 그림은 고도각 내림차순)
    kitti, kitti360  점 순서로 센 줄(_invkey_augment_20261115.estimate_rings). 64줄이 아니면 그 스캔은 뺀다.
    mulran           색인 % 64          mcd  색인 // 1024     (65536점이 아니면 뺀다)
    nclt             원시 파일의 레이저 번호(0-31). 좌표는 로더와 같게(z 부호 뒤집기) 읽고 광선 원점 (0, 0, 1.05) m 기준.
    helipr           원시 파일 ring(VLP-16 0-15)      helipr-os2  원시 파일 ring(OS2-128 0-127)
  진행 방향의 센서 좌표 방위: 키프레임 GT 자세(라이다 좌표 = GT @ X)에서 이웃 키프레임 이동의 센서 좌표 방위.
    X: kitti = calib.txt Tr(velo -> cam0), kitti360·mulran = _invkey_submap_probe_20261115.extrinsic, helipr·nclt = I,
    mcd = 원본 gt/<seq>/pose_inW.csv(몸체 자세, 쿼터니언) @ calibration/hhs_calib.yaml body.os_sensor
    (추출 캐시에는 회전이 없어 원본을 읽는다). helipr-os2 = LiDAR_GT/Ouster_gt.txt.
    점검: 키프레임 (i, i+2) 쌍 15개를 ICP(항등 초기값)로 맞춰 얻은 이동 방위와 GT 방위를 비교한다.

2. 방향별 R@1 (recall)
  특징 파일의 단일 스캔 spec, raw와 선형 16(seed 0). 검색 회계는 _invkey_train_linear_20261114.eval_seq,
  방향은 direction_of(질의와 과거 최근접 양성의 진행 방향 = GT 위치 P[i+3]-P[i-3]; 같은 < 30도, 반대 > 150도).
  MCD는 진행 방향 분류를 원본 자세 회전(센서 yaw 차)과 대조한다(mcdrot).

명령 (컨테이너 ring_baseline, 리포 루트)
  python3 scripts/_invkey_fov_survey_20261115.py scan --seqs kitti/00 kitti/05
  python3 scripts/_invkey_fov_survey_20261115.py ouster        # HeLiPR Sample OS2-128 (tar.xz를 풀지 않고 읽음)
  python3 scripts/_invkey_fov_survey_20261115.py recall        # GPU
  python3 scripts/_invkey_fov_survey_20261115.py mcdrot
  python3 scripts/_invkey_fov_survey_20261115.py report        # metrics.json, 그림
출력: results/invkey_study_20261114/fov_survey/ (seq_*.json, recall.json, mcd_rotation.json, metrics.json, fig_*.png)
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

OUT = REPO / "results/invkey_study_20261114/fov_survey"
SURVEY = REPO / "results/invkey_study_20261114/data_survey.json"
LINMET = REPO / "results/invkey_study_20261114/linear/metrics.json"
DATA = Path("/workspace/data")
FEAT = DATA / "invkey_study_20261114/features"
AZB = 72
AZW = 360.0 / AZB
R_SELF, R_WORLD, VALID_R = 1.0, 2.5, 0.05
THETA = 0.05
THETAS = (0.025, 0.05, 0.10)
PERSIST = 0.95
PERSISTS = (0.90, 0.95, 0.99)
EMPTY_M = 0.05
NCLT_ORIGIN = np.array([0.0, 0.0, 1.05])
NCLT_DT = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("intensity", "u1"), ("laser", "u1")])
VLP_DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("ring", "<u2"), ("time", "<f4")])
OS2_DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("t", "<u4"),
                   ("reflectivity", "<u2"), ("ring", "<u2"), ("ambient", "<u2")])
NRING = {"kitti": 64, "kitti360": 64, "mulran": 64, "mcd": 64, "nclt": 32, "helipr": 16, "helipr-os2": 128}
SHOTS = {"mulran": 1024 / AZB, "mcd": 1024 / AZB, "helipr": 1800 / AZB, "helipr-os2": 1024 / AZB}
OS2_TAR = DATA / "helipr/Sample/LiDAR/Ouster.tar.xz"
OS2_GT = DATA / "helipr/Sample/LiDAR_GT/Ouster_gt.txt"

# 사각 조사 단위 -> 시퀀스 (단위 = 같은 리그로 보는 묶음; NCLT는 9세션 모두 재고 report에서 초기·후기로 묶는다)
FOV_UNITS = {
    "kitti": ["kitti/00", "kitti/05", "kitti/08"],
    "kitti360": ["kitti360/0000", "kitti360/0004", "kitti360/0006"],
    "nclt_early": ["nclt/2012-01-08", "nclt/2012-02-04", "nclt/2012-03-17", "nclt/2012-05-11", "nclt/2012-08-04"],
    "nclt_late": ["nclt/2012-11-04", "nclt/2012-11-16", "nclt/2013-01-10", "nclt/2013-02-23"],
    "mulran_KAIST": ["mulran/KAIST01", "mulran/KAIST03"],
    "mulran_DCC": ["mulran/DCC01", "mulran/DCC02"],
    "mulran_Riverside": ["mulran/Riverside01", "mulran/Riverside02"],
    "mulran_Sejong": ["mulran/Sejong01", "mulran/Sejong03"],
    "helipr_Bridge": ["helipr/Bridge01", "helipr/Bridge04"],
    "helipr_Roundabout": ["helipr/Roundabout01", "helipr/Roundabout03"],
    "helipr_Town": ["helipr/Town01", "helipr/Town03"],
    "helipr_KAIST": ["helipr/KAIST04", "helipr/KAIST06"],
    "helipr_DCC": ["helipr/DCC04", "helipr/DCC06"],
    "helipr_Riverside": ["helipr/Riverside04", "helipr/Riverside06"],
    "mcd_kth": ["mcd/kth_day_06", "mcd/kth_day_09", "mcd/kth_night_01"],
    "mcd_tuhh": ["mcd/tuhh_day_02", "mcd/tuhh_day_03", "mcd/tuhh_night_07"],
}
DATASETS = {"kitti": ["kitti"], "kitti360": ["kitti360"], "nclt": ["nclt_early", "nclt_late"],
            "mulran": ["mulran_KAIST", "mulran_DCC", "mulran_Riverside", "mulran_Sejong"],
            "helipr": ["helipr_Bridge", "helipr_Roundabout", "helipr_Town", "helipr_KAIST", "helipr_DCC",
                       "helipr_Riverside"],
            "mcd": ["mcd_kth", "mcd_tuhh"]}


def chmod_write(path, text):
    import os
    path = Path(path)
    path.write_text(text)
    os.chmod(path, 0o666)


def wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def circ_mean(deg):
    r = np.radians(np.asarray(deg, dtype=np.float64))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def unit_of(key):
    """방향별 R@1 표의 단위 (63시퀀스 전부)."""
    sensor, seq = key.split("/")
    if sensor in ("kitti", "kitti360"):
        return sensor
    if sensor == "nclt":
        return "nclt_early" if seq < "2012-09" else "nclt_late"
    if sensor == "mcd":
        return "mcd_" + seq.split("_")[0]
    site = "".join(c for c in seq if not c.isdigit())
    return f"{sensor}_{site}"


# ---------------------------------------------------------------- 자세와 진행 방향
def _rt4(vals):
    T = np.eye(4)
    T[:3, :] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
    return T


def kitti_tr(seq):
    for ln in open(DATA / f"kitti/dataset/sequences/{seq}/calib.txt"):
        if ln.startswith("Tr:"):
            return _rt4(ln.split()[1:13])
    raise ValueError(seq)


def mcd_sensor_poses(seq, ts_ns):
    """MCD 원본 pose_inW.csv(몸체, ~10 Hz)를 스캔 시각에 가장 가까운 행으로 맞춰 센서 자세 (n,4,4)와 시각 차(s)."""
    from scipy.spatial.transform import Rotation
    import yaml
    rows = np.loadtxt(DATA / f"mcd/{seq}/gt/{seq}/pose_inW.csv", delimiter=",", skiprows=1)
    t, p, q = rows[:, 1], rows[:, 2:5], rows[:, 5:9]
    cal = yaml.safe_load(open(DATA / "mcd/calibration/hhs_calib.yaml"))
    Tbs = np.asarray(cal["body"]["os_sensor"]["T"], dtype=np.float64)
    ts = np.asarray(ts_ns, dtype=np.float64) / 1e9
    k = np.clip(np.searchsorted(t, ts), 1, len(t) - 1)
    k = np.where(np.abs(t[k - 1] - ts) < np.abs(t[k] - ts), k - 1, k)
    T = np.tile(np.eye(4), (len(ts), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(q[k]).as_matrix()
    T[:, :3, 3] = p[k]
    return T @ Tbs, np.abs(t[k] - ts)


def lidar_poses(key, n, xyz, L):
    """키프레임 라이다 자세 (n,4,4)와 메모. 위치 기준은 특징 xyz(MCD 제외)."""
    from _invkey_submap_probe_20261115 import extrinsic, load_poses
    sensor, seq = key.split("/")
    if sensor == "mcd":
        return None, "mcd: see mcd_sensor_poses"
    P = load_poses(key, n, xyz)
    if sensor == "kitti":
        X = kitti_tr(seq)
    elif sensor in ("kitti360", "mulran"):
        X = extrinsic(sensor)
    else:
        X = np.eye(4)
    return P @ X, "ok"


def motion_azimuth(TL, idx=None):
    """이웃 키프레임 이동의 센서 좌표 방위(도)와 고도각. 0.5 m 넘게 움직인 것만."""
    d = np.diff(TL[:, :3, 3], axis=0)
    loc = np.einsum("nji,nj->ni", TL[:-1, :3, :3], d)
    nm = np.linalg.norm(loc, axis=1)
    ok = nm > 0.5
    if idx is not None:
        ok &= np.isin(np.arange(len(loc)), idx)
    az = np.degrees(np.arctan2(loc[ok, 1], loc[ok, 0]))
    el = np.degrees(np.arcsin(np.clip(loc[ok, 2] / nm[ok], -1, 1)))
    return az, el


def az_summary(az, el=None):
    if len(az) == 0:
        return None
    m = circ_mean(az)
    dev = np.abs(wrap(az - m))
    out = {"n": int(len(az)), "circ_mean_deg": round(m, 2), "abs_dev_median_deg": round(float(np.median(dev)), 2),
           "abs_dev_p90_deg": round(float(np.percentile(dev, 90)), 2),
           "frac_within_15deg": round(float((dev < 15).mean()), 4),
           "frac_within_45deg": round(float((dev < 45).mean()), 4),
           "frac_reversed_gt_135deg": round(float((dev > 135).mean()), 4)}
    if el is not None:
        out["elev_median_deg"] = round(float(np.median(el)), 2)
    return out


# ---------------------------------------------------------------- 스캔 읽기
def read_scan(sensor, L, sid):
    """(xyz 센서 원점 기준 float64 (N,3), 원래 줄 번호 int (N,), 메모). 줄을 모르면 None."""
    if sensor == "helipr":
        d = np.fromfile(L.scan_files[sid], dtype=VLP_DT)
        return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64), d["ring"].astype(np.int64)
    if sensor == "nclt":
        d = np.fromfile(L.frame_files[sid], dtype=NCLT_DT)
        x = d["x"].astype(np.float32) * 0.005 - 100.0
        y = d["y"].astype(np.float32) * 0.005 - 100.0
        z = -(d["z"].astype(np.float32) * 0.005 - 100.0)
        p = np.stack([x, y, z], 1).astype(np.float64)
        ok = (np.abs(p) < 200.0).all(1)
        return p[ok] - NCLT_ORIGIN, d["laser"][ok].astype(np.int64)
    pts = L[int(sid)]["points"]
    p = pts[:, :3].astype(np.float64)
    if sensor in ("mulran", "mcd"):
        if len(p) != 65536:
            return None, None
        ring = np.arange(len(p)) % 64 if sensor == "mulran" else np.arange(len(p)) // 1024
        return p, ring
    if sensor in ("kitti", "kitti360"):
        from _invkey_augment_20261115 import estimate_rings
        ring, rel = estimate_rings(pts, sensor, p, np.linalg.norm(p, axis=1) > 0.5)
        if int(np.isfinite(rel).sum()) != 64 or ring.max() != 63:
            return None, None
        return p, ring
    raise ValueError(sensor)


def scan_counts(p, ring, nr):
    """칸 (nr, AZB) 세계·자기·중간 반환 수와 줄별 세계 점 고도각 중앙값."""
    r3 = np.linalg.norm(p, axis=1)
    rxy = np.hypot(p[:, 0], p[:, 1])
    valid = (r3 > VALID_R) & (ring >= 0) & (ring < nr)
    az = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360.0
    b = np.minimum((az / AZW).astype(np.int64), AZB - 1)
    idx = ring * AZB + b
    w = valid & (rxy >= R_WORLD)
    s = valid & (r3 < R_SELF)
    m = valid & ~w & ~s
    cnt = [np.bincount(idx[k], minlength=nr * AZB)[:nr * AZB].reshape(nr, AZB) for k in (w, s, m)]
    el = np.degrees(np.arcsin(np.clip(p[w, 2] / np.maximum(r3[w], 1e-9), -1, 1)))
    rg = ring[w]
    elev = np.full(nr, np.nan)
    for k in np.unique(rg):
        elev[k] = np.median(el[rg == k])
    return cnt[0], cnt[1], cnt[2], elev


def shots_per_bin(sensor, allv, nscan):
    """칸당 발사 수 (nr,) (스캔 하나당)."""
    nr = allv.shape[0]
    if sensor in SHOTS:
        return np.full(nr, SHOTS[sensor])
    est = np.percentile(allv, 95, axis=1) / nscan
    return np.maximum(est, 0.5 * np.median(est))


def fov_stats(sensor, W, S, Mi, elevs, nkeep):
    """표본 스캔별 수 W, S, Mi (n, nr, AZB) -> 시퀀스 기록."""
    n = W.shape[0]
    allv = (W + S + Mi).sum(0)
    shots = shots_per_bin(sensor, allv, n)
    wr = W.sum(0) / (shots[:, None] * n)
    M = wr.max(1)
    rate_scan = W / shots[None, :, None]
    frac = {str(t): np.round((rate_scan < t * M[None, :, None]).mean(0), 3).tolist() for t in THETAS}
    return {
        "n_scans": int(n), "n_dropped": int(nkeep), "shots_per_bin_per_scan": np.round(shots, 3).tolist(),
        "ring_elev_deg": np.round(np.nanmedian(np.array(elevs), 0), 3).tolist(),
        "world_rate": np.round(wr, 4).tolist(),
        "self_rate": np.round(S.sum(0) / (shots[:, None] * n), 4).tolist(),
        "mid_rate": np.round(Mi.sum(0) / (shots[:, None] * n), 4).tolist(),
        "ring_max_world_rate": np.round(M, 4).tolist(),
        "frac_scans_below_theta_x_max": frac,
        "points_per_scan_median": float(np.median((W + S + Mi).sum((1, 2)))),
    }


def icp_pair(A, B):
    """B 점을 A 좌표로 옮기는 T(항등 초기값)와 fitness. A, B 센서 원점 기준 (N,3)."""
    import open3d as o3d
    from _invkey_submap_probe_20261115 import _cloud
    reg = o3d.pipelines.registration
    a, b = _cloud(A), _cloud(B)
    T = np.eye(4)
    for dist in (2.0, 1.0, 0.5, 0.25):
        r = reg.registration_icp(b, a, dist, T, reg.TransformationEstimationPointToPlane())
        T = r.transformation
    return np.asarray(T), float(r.fitness)


def icp_check(pairs_pts, TL_pairs):
    """[(A, B)]와 GT 상대 자세 목록 -> ICP 이동 방위 대 GT 이동 방위."""
    rows = []
    for (A, B), Tg in zip(pairs_pts, TL_pairs):
        T, fit = icp_pair(A, B)
        ti, tg = T[:3, 3], Tg[:3, 3]
        row = {"fitness": round(fit, 3), "icp_move_m": round(float(np.linalg.norm(ti[:2])), 3),
               "icp_az": round(float(np.degrees(np.arctan2(ti[1], ti[0]))), 2),
               "icp_yaw": round(float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))), 2)}
        if Tg is not None:
            row["gt_move_m"] = round(float(np.linalg.norm(tg[:2])), 3)
            row["gt_az"] = round(float(np.degrees(np.arctan2(tg[1], tg[0]))), 2)
            row["gt_yaw"] = round(float(np.degrees(np.arctan2(Tg[1, 0], Tg[0, 0]))), 2)
        rows.append(row)
    ok = [r for r in rows if r["fitness"] > 0.3 and r["icp_move_m"] > 0.5]
    out = {"n": len(rows), "n_ok(fit>0.3, move>0.5m)": len(ok),
           "icp_az_circ_mean": round(circ_mean([r["icp_az"] for r in ok]), 2) if ok else None,
           "rows": rows}
    if ok and "gt_az" in ok[0]:
        d = np.abs(wrap([r["icp_az"] - r["gt_az"] for r in ok]))
        out["abs_az_diff_icp_vs_gt_median"] = round(float(np.median(d)), 2)
        out["abs_az_diff_icp_vs_gt_max"] = round(float(d.max()), 2)
        dy = np.abs(wrap([r["icp_yaw"] - r["gt_yaw"] for r in ok]))
        out["abs_yaw_diff_icp_vs_gt_median"] = round(float(np.median(dy)), 2)
    return out


# ---------------------------------------------------------------- 명령: 시퀀스 조사
def cmd_scan(a):
    from _pd_build_test_caches import _make_loader
    for key in a.seqs:
        t0 = time.time()
        sensor, seq = key.split("/")
        nr = NRING[sensor]
        with np.load(FEAT / f"{sensor}_{seq}.npz") as z:
            ids, xyz = z["scan_ids"], z["xyz"].astype(np.float64)
        n = len(ids)
        L = _make_loader(sensor, seq)
        pick = np.linspace(0, n - 3, a.n).astype(int)
        # 진행 방향 (GT)
        if sensor == "mcd":
            TL, dtp = mcd_sensor_poses(seq, L.scan_timestamps_ns[ids])
            # 위치 점검: 원본 자세 위치(첫 스캔 기준)와 특징 xyz
            p0 = TL[:, :3, 3] - TL[0, :3, 3]
            pos_check = {"max_abs_m(sensor pos - feature xyz, both from first)": float(
                np.abs((p0 - (xyz - xyz[0]))).max()), "pose_time_diff_s_max": float(dtp.max()),
                "lever_arm_note": "sensor vs body origin differ by ~6 cm"}
        else:
            TL, _ = lidar_poses(key, n, xyz, L)
            pos_check = None
        az, el = motion_azimuth(TL)
        # 세계 위쪽: kitti 자세 세계 = 첫 cam0(y 아래), nclt = NED(z 아래), 나머지 z 위
        wup = {"kitti": (0.0, -1.0, 0.0), "nclt": (0.0, 0.0, -1.0)}.get(sensor, (0.0, 0.0, 1.0))
        up = np.einsum("nji,j->ni", TL[:, :3, :3], np.array(wup))
        W, S, Mi, elevs = [], [], [], []
        drop = 0
        icp_src = set(pick[np.linspace(0, len(pick) - 1, a.icp).astype(int)].tolist()) if a.icp else set()
        pts_cache, pairs, gtp = {}, [], []
        for i in pick:
            p, ring = read_scan(sensor, L, int(ids[i]))
            if p is None:
                drop += 1
                continue
            w, s, m, e = scan_counts(p, ring, nr)
            W.append(w)
            S.append(s)
            Mi.append(m)
            elevs.append(e)
            if int(i) in icp_src:
                q, _ = read_scan(sensor, L, int(ids[i + 2]))
                if q is None:
                    q = L[int(ids[i + 2])]["points"][:, :3].astype(np.float64)
                pairs.append((p, q))
                gtp.append(np.linalg.inv(TL[i]) @ TL[i + 2])
        W, S, Mi = (np.array(x, dtype=np.int32) for x in (W, S, Mi))
        rec = {"key": key, "sensor": sensor, "n_keyframes": int(n), "picked": pick.tolist()}
        rec.update(fov_stats(sensor, W, S, Mi, elevs, drop))
        rec["motion_azimuth_sensor_frame"] = az_summary(az, el)
        rec["world_up_in_sensor_median"] = np.round(np.median(up, 0), 4).tolist()
        rec["pose_check"] = pos_check
        rec["icp_check"] = icp_check(pairs, gtp) if pairs else None
        chmod_write(OUT / f"seq_{sensor}_{seq}.json", json.dumps(rec))
        ma = rec["motion_azimuth_sensor_frame"]
        ic = rec["icp_check"] or {}
        print(f"{key:22s} scans {rec['n_scans']} drop {drop} motion az {ma['circ_mean_deg']} "
              f"(dev med {ma['abs_dev_median_deg']}, <15deg {ma['frac_within_15deg']}) "
              f"icp az {ic.get('icp_az_circ_mean')} diff {ic.get('abs_az_diff_icp_vs_gt_median')} "
              f"{time.time() - t0:.0f}s", flush=True)
    return 0


def cmd_ouster(a):
    """HeLiPR Sample의 OS2-128 원시 스캔을 tar.xz에서 디스크에 풀지 않고 읽는다."""
    import tarfile
    from scipy.spatial.transform import Rotation
    t0 = time.time()
    gt = np.loadtxt(OS2_GT)
    gts = np.array([int(ln.split()[0]) for ln in open(OS2_GT) if ln.strip()], dtype=np.int64)  # float64는 19자리 시각을 잃는다
    assert len(gts) == len(gt)
    TL = np.tile(np.eye(4), (len(gt), 1, 1))
    TL[:, :3, :3] = Rotation.from_quat(gt[:, 4:8]).as_matrix()
    TL[:, :3, 3] = gt[:, 1:4]
    order = np.arange(len(gts))
    want = set(order[np.linspace(0, len(gts) - 3, a.n).astype(int)].tolist())
    icp_src = sorted(want)[::max(1, len(want) // a.icp)][:a.icp] if a.icp else []
    need = want | {i + 2 for i in icp_src}
    pos = {int(t): k for k, t in enumerate(gts)}
    scans, members, sizes = {}, 0, []
    with tarfile.open(OS2_TAR, "r|xz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            members += 1
            sizes.append(m.size)
            k = pos.get(int(Path(m.name).stem))
            if k is None or k not in need:
                continue
            d = np.frombuffer(tf.extractfile(m).read(), dtype=OS2_DT)
            scans[k] = (np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64), d["ring"].astype(np.int64))
    print(f"read {members} members, kept {len(scans)}  {time.time() - t0:.0f}s", flush=True)
    ring_max = max(int(r.max()) for _, r in scans.values())
    W, S, Mi, elevs = [], [], [], []
    for k in sorted(want):
        if k not in scans:
            continue
        w, s, mm, e = scan_counts(*scans[k], NRING["helipr-os2"])
        W.append(w)
        S.append(s)
        Mi.append(mm)
        elevs.append(e)
    W, S, Mi = (np.array(x, dtype=np.int32) for x in (W, S, Mi))
    rec = {"key": "helipr-os2/Sample", "sensor": "helipr-os2", "n_keyframes": int(len(gts)),
           "archive": {"path": str(OS2_TAR), "members": members, "member_bytes_median": float(np.median(sizes)),
                       "bytes_per_point": OS2_DT.itemsize, "ring_max": ring_max,
                       "note": "Sample = first ~60 s of Roundabout01 (timestamps); scans are all frames, not keyframes"}}
    rec.update(fov_stats("helipr-os2", W, S, Mi, elevs, len(want) - len(W)))
    az, el = motion_azimuth(TL)
    rec["motion_azimuth_sensor_frame"] = az_summary(az, el)
    rec["world_up_in_sensor_median"] = np.round(np.median(np.einsum("nji,j->ni", TL[:, :3, :3],
                                                                    np.array([0., 0., 1.])), 0), 4).tolist()
    pairs = [(scans[i][0], scans[i + 2][0]) for i in icp_src if i in scans and i + 2 in scans]
    gtp = [np.linalg.inv(TL[i]) @ TL[i + 2] for i in icp_src if i in scans and i + 2 in scans]
    rec["icp_check"] = icp_check(pairs, gtp) if pairs else None
    chmod_write(OUT / "seq_helipr-os2_Sample.json", json.dumps(rec))
    print(f"helipr-os2 Sample scans {rec['n_scans']} motion {rec['motion_azimuth_sensor_frame']} "
          f"icp {(rec['icp_check'] or {}).get('icp_az_circ_mean')} {time.time() - t0:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------- 명령: 방향별 R@1
def cmd_recall(a):
    import torch
    from _invkey_submap_probe_20261115 import load_lin16
    from _invkey_train_linear_20261114 import DIRS, direction_of, encode, eval_seq, load_seq
    torch.backends.cuda.matmul.allow_tf32 = False
    recs = json.load(open(SURVEY))["sequences"]
    keys = a.seqs or [r["key"] for r in recs]
    s16, W16 = load_lin16()
    path = OUT / "recall.json"
    out = json.load(open(path)) if path.exists() and a.seqs else {}
    t0 = time.time()
    for key in keys:
        spec, xyz, ids = load_seq(key)
        out[key] = {}
        for rep, (sv, Wv) in (("raw", (None, None)), ("lin16_s0", (s16, W16))):
            with torch.no_grad():
                D = encode(spec, sv, Wv)
                m, arr = eval_seq(D, xyz, 16 if Wv is None else 64)
            del D
            torch.cuda.empty_cache()
            pos = arr["pos"]
            dirs = direction_of(xyz, arr["q"][pos], arr["jnn"][pos])
            h = arr["hit1"][pos]
            m["dir_n"] = {d: int((dirs == i).sum()) for i, d in enumerate(DIRS)}
            m["dir_hit1"] = {d: int(h[dirs == i].sum()) for i, d in enumerate(DIRS)}
            if key.startswith("mcd/"):
                m["pos_q"] = arr["q"][pos].tolist()
                m["pos_jnn"] = arr["jnn"][pos].tolist()
                m["pos_hit1"] = h.astype(int).tolist()
                m["pos_dir"] = dirs.astype(int).tolist()
            out[key][rep] = m
        print(f"{key:22s} n {len(xyz):6d} raw R@1 {out[key]['raw']['R@1']} rev {out[key]['raw']['dir_hit1']['reverse']}"
              f"/{out[key]['raw']['dir_n']['reverse']} lin16 rev {out[key]['lin16_s0']['dir_hit1']['reverse']}"
              f"  {time.time() - t0:.0f}s", flush=True)
        chmod_write(path, json.dumps(out))
    return 0


def cmd_mcdrot(a):
    """MCD: 진행 방향 분류(위치)와 원본 자세 회전(센서 yaw 차) 대조, 질의별 센서 좌표 진행 방위."""
    from _pd_build_test_caches import _make_loader
    from _invkey_data_survey_20261114 import headings
    rc = json.load(open(OUT / "recall.json"))
    recs = [r["key"] for r in json.load(open(SURVEY))["sequences"] if r["key"].startswith("mcd/")]
    out = {}
    for key in recs:
        seq = key.split("/")[1]
        with np.load(FEAT / f"mcd_{seq}.npz") as z:
            ids, xyz = z["scan_ids"], z["xyz"].astype(np.float64)
        L = _make_loader("mcd", seq)
        TS, dt = mcd_sensor_poses(seq, L.scan_timestamps_ns[ids])
        yaw = np.degrees(np.arctan2(TS[:, 1, 0], TS[:, 0, 0]))
        u, moving = headings(xyz)
        # 센서 좌표 진행 방위(질의 시각, 진행 방향 = P[i+3]-P[i-3])
        loc = np.einsum("nji,nj->ni", TS[:, :3, :3], u)
        maz = np.degrees(np.arctan2(loc[:, 1], loc[:, 0]))
        m = rc[key]["raw"]
        q, j, dirs = (np.array(m[k], dtype=int) for k in ("pos_q", "pos_jnn", "pos_dir"))
        rel = np.abs(wrap(yaw[q] - yaw[j]))
        rot_cls = np.where(rel < 30, 0, np.where(rel > 150, 1, 2))
        known = dirs < 3
        conf = {f"travel_{d}": {f"rot_{e}": int(((dirs == di) & (rot_cls == ei)).sum())
                                for ei, e in enumerate(("same", "reverse", "crossing"))}
                for di, d in enumerate(("same", "reverse", "crossing"))}
        hits = {}
        for rep in ("raw", "lin16_s0"):
            h = np.array(rc[key][rep]["pos_hit1"], dtype=bool)
            hits[rep] = {f"rot_{e}": [int(h[rot_cls == ei].sum()), int((rot_cls == ei).sum())]
                         for ei, e in enumerate(("same", "reverse", "crossing"))}
        out[key] = {"n_pos": int(len(q)), "pose_time_diff_s_max": float(dt.max()),
                    "agreement_travel_vs_rotation": float((rot_cls[known] == dirs[known]).mean()) if known.any() else None,
                    "confusion": conf, "hit1_by_rotation_class[hit,n]": hits,
                    "motion_az_sensor_at_queries": az_summary(maz[q][moving[q]]),
                    "motion_az_sensor_all_keyframes": az_summary(maz[moving])}
        print(f"{key:22s} agree {out[key]['agreement_travel_vs_rotation']} motion az "
              f"{out[key]['motion_az_sensor_all_keyframes']}", flush=True)
    chmod_write(OUT / "mcd_rotation.json", json.dumps(out, indent=1))
    return 0


# ---------------------------------------------------------------- 명령: 모으기
def runs(mask):
    """원형 연속 참 구간 [(시작 칸, 길이)]."""
    if mask.all():
        return [(0, len(mask))]
    if not mask.any():
        return []
    k = int(np.argmin(mask))           # 거짓 칸에서 시작해 원형으로 돈다
    m = np.roll(mask, -k)
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            out.append(((i + k) % len(m), j - i))
            i = j
        else:
            i += 1
    return out


def side_of(rel):
    r = abs(float(wrap(rel)))
    return "front" if r <= 45 else ("back" if r >= 135 else "side")


def unit_fov(unit, recs, theta=THETA, persist=PERSIST):
    """단위 사각 표와 지표."""
    els = np.nanmedian(np.array([r["ring_elev_deg"] for r in recs], dtype=float), 0)
    order = np.argsort(-np.nan_to_num(els, nan=-999.0))       # 고도각 내림차순(빈 줄은 맨 아래)
    blind_each = np.array([np.array(r["frac_scans_below_theta_x_max"][str(theta)]) >= persist for r in recs])
    M = np.array([r["ring_max_world_rate"] for r in recs])
    empty = (M < EMPTY_M).any(0)
    blind = blind_each.all(0) & ~empty[:, None]
    some = blind_each.any(0) & ~blind & ~empty[:, None]
    fwd = [r["motion_azimuth_sensor_frame"]["circ_mean_deg"] for r in recs if r.get("motion_azimuth_sensor_frame")]
    fwd = circ_mean(fwd) if fwd else None
    width = blind.sum(1) * AZW
    live = ~empty
    best = None
    for rr in np.flatnonzero(live):
        for st, ln in runs(blind[rr]):
            if best is None or ln > best[2]:
                best = (rr, st, ln)
    out = {"unit": unit, "seqs": [r["key"] for r in recs], "n_scans": int(sum(r["n_scans"] for r in recs)),
           "n_rings": int(len(els)), "n_empty_rings": int(empty.sum()),
           "empty_ring_elev": [round(float(els[k]), 2) if np.isfinite(els[k]) else None for k in np.flatnonzero(empty)],
           "forward_az_sensor_deg": None if fwd is None else round(fwd, 1),
           "blind_frac_cells": round(float(blind[live].mean()), 4) if live.any() else None,
           "n_rings_blind_gt30deg": int((width[live] > 30).sum()),
           "n_rings_any_blind": int((width[live] > 0).sum()),
           "ring_blind_width_deg_by_elev": [[round(float(els[k]), 2) if np.isfinite(els[k]) else None,
                                             float(width[k])] for k in order],
           "inconsistent_cells_frac(blind in some seqs only)": round(float(some[live].mean()), 4) if live.any() else None}
    if best is not None:
        rr, st, ln = best
        c = (st + ln / 2.0) * AZW
        out["widest"] = {"ring_elev_deg": round(float(els[rr]), 2), "width_deg": float(ln * AZW),
                         "center_az_sensor_deg": round(float(c % 360), 1),
                         "from_deg": float(st * AZW), "to_deg": float(((st + ln) * AZW) % 360)}
        if fwd is not None:
            rel = float(wrap(c - fwd))
            out["widest"]["center_rel_forward_deg"] = round(rel, 1)
            out["widest"]["position"] = side_of(rel)
        # 가장 넓은 사각과 같은 방위 칸을 사각으로 갖는 줄 수
        cols = [(st + k) % AZB for k in range(ln)]
        out["widest"]["rings_blind_at_its_center_bin"] = int(blind[:, cols[len(cols) // 2]].sum())
    # 방위별 사각 줄 비율(살아 있는 줄 기준)
    prof = blind[live].mean(0) if live.any() else np.zeros(AZB)
    out["az_bins_with_ge50pct_rings_blind_deg"] = [float(b * AZW) for b in np.flatnonzero(prof >= 0.5)]
    # 방위 폭: 살아 있는 줄 전부가 사각인 방위(모든 거리에서 안 보임)와 절반 이상이 사각인 방위
    for name, mk in (("all_rings", prof >= 1.0 - 1e-9), ("ge50pct_rings", prof >= 0.5)):
        e = {"width_deg": float(mk.sum() * AZW)}
        rs = runs(mk)
        if rs:
            st, ln = max(rs, key=lambda x: x[1])
            c = (st + ln / 2.0) * AZW
            e.update(largest_run_deg=float(ln * AZW), center_az_sensor_deg=round(float(c % 360), 1))
            if fwd is not None:
                e.update(center_rel_forward_deg=round(float(wrap(c - fwd)), 1), position=side_of(c - fwd))
        out[f"az_blind_{name}"] = e
    return out, {"els": els, "order": order, "blind": blind, "empty": empty, "fwd": fwd}


def pooled(recs, field):
    """단위 풀링 반환율 (스캔 수 가중)."""
    n = np.array([r["n_scans"] for r in recs], float)
    return np.tensordot(n / n.sum(), np.array([r[field] for r in recs], float), 1)


def draw(dataset, units, out_path, recs_by_unit, info_by_unit, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nu = len(units)
    fig, axes = plt.subplots(nu, 2, figsize=(15, 2.6 + 2.5 * nu), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1]})
    for ui, unit in enumerate(units):
        recs, inf = recs_by_unit[unit], info_by_unit[unit]
        o = inf["order"]
        els = inf["els"][o]
        wr, sr = pooled(recs, "world_rate")[o], pooled(recs, "self_rate")[o]
        blind = inf["blind"][o]
        nr = len(o)
        for ci, (arr, title, vmax, cmap) in enumerate(((wr, "world return rate (horiz >= 2.5 m)", 1.0, "viridis"),
                                                       (sr, "self return rate (3D < 1 m)", 0.5, "magma"))):
            ax = axes[ui, ci]
            im = ax.imshow(arr, aspect="auto", origin="upper", extent=(0, 360, nr, 0), vmin=0, vmax=vmax,
                           cmap=cmap, interpolation="nearest")
            if ci == 0 and blind.any():
                yy, xx = np.nonzero(blind)
                ax.scatter((xx + 0.5) * AZW, yy + 0.5, s=6 if nr <= 32 else 3, c="red", marker="s", lw=0,
                           label=f"blind (theta {THETA}, persist {PERSIST}, all seqs)")
                ax.legend(loc="lower right", fontsize=7, framealpha=0.8)
            if inf["fwd"] is not None:
                for dv, ls in ((0, "-"), (180, "--")):
                    ax.axvline((inf["fwd"] + dv) % 360, color="w" if ci == 0 else "c", ls=ls, lw=1.2)
            step = max(1, nr // 8)
            ax.set_yticks(np.arange(0, nr, step) + 0.5)
            ax.set_yticklabels([f"{e:.1f}" if np.isfinite(e) else "dead" for e in els[::step]], fontsize=7)
            ax.set_xticks(np.arange(0, 361, 45))
            ax.set_ylabel("ring elev (deg)", fontsize=8)
            s = summ[unit]
            fw = "n/a" if inf["fwd"] is None else f"{inf['fwd']:.0f}"
            head = (f"{unit}  ({', '.join(r['key'].split('/')[1] for r in recs)}; {s['n_scans']} scans)  "
                    f"fwd az {fw} (solid; dashed = back)")
            if ci == 0:
                ax.set_title(head + f"\nblind cells {100 * (s['blind_frac_cells'] or 0):.1f}%, "
                             f"rings >30deg {s['n_rings_blind_gt30deg']}/{nr - s['n_empty_rings']}", fontsize=8)
            else:
                ax.set_title(title, fontsize=8)
            if ui == nu - 1:
                ax.set_xlabel("sensor azimuth (deg, atan2(y, x))", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.suptitle(f"{dataset}: ring x azimuth (5 deg) return rate, sensor frame", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985 if nu > 2 else 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def cmd_report(a):
    import os
    lin = json.load(open(LINMET))
    role = {**{k: "train" for k in lin["train"]}, **{k: "dev" for k in lin["dev"]},
            **{k: "heldout" for k in lin["heldout"]}}
    seqrec = {}
    for p in sorted(OUT.glob("seq_*.json")):
        r = json.load(open(p))
        seqrec[r["key"]] = r
    res = {"date": "2026-11-15", "script": "scripts/_invkey_fov_survey_20261115.py",
           "definitions": {"world": f"horizontal range >= {R_WORLD} m", "self": f"3D range < {R_SELF} m",
                           "az_bin_deg": AZW, "theta": THETA, "persist": PERSIST,
                           "blind": "cell blind in a seq if >= persist of sampled scans have cell world rate < "
                                    "theta * ring max (pooled over that seq); blind in a unit if blind in all its seqs",
                           "empty_ring": f"ring max world rate < {EMPTY_M} in any seq of the unit (excluded)"},
           "units": {}, "sensitivity": {}, "per_seq_fov": {}, "recall": {}, "units_recall": {}}
    recs_by_unit, info_by_unit = {}, {}
    for unit, keys in FOV_UNITS.items():
        recs = [seqrec[k] for k in keys if k in seqrec]
        if len(recs) < 1:
            continue
        s, inf = unit_fov(unit, recs)
        res["units"][unit] = s
        recs_by_unit[unit], info_by_unit[unit] = recs, inf
        sens = {}
        for t in THETAS:
            for pp in PERSISTS:
                x, _ = unit_fov(unit, recs, t, pp)
                sens[f"theta{t}_persist{pp}"] = {k: x.get(k) for k in ("blind_frac_cells", "n_rings_blind_gt30deg",
                                                                       "n_rings_any_blind")}
                sens[f"theta{t}_persist{pp}"]["widest"] = {k: x["widest"][k] for k in (
                    "width_deg", "center_az_sensor_deg", "position", "ring_elev_deg") if k in x["widest"]} \
                    if "widest" in x else None
        res["sensitivity"][unit] = sens
        for r in recs:      # 시퀀스 하나만으로 본 사각(단위 판정과 비교용)
            x, _ = unit_fov(r["key"], [r])
            res["per_seq_fov"][r["key"]] = {k: x.get(k) for k in ("blind_frac_cells", "n_rings_blind_gt30deg",
                                                                  "n_empty_rings", "forward_az_sensor_deg")}
            res["per_seq_fov"][r["key"]]["widest"] = x.get("widest")
            res["per_seq_fov"][r["key"]]["motion_azimuth_sensor_frame"] = r.get("motion_azimuth_sensor_frame")
            res["per_seq_fov"][r["key"]]["icp_check"] = {k: v for k, v in (r.get("icp_check") or {}).items()
                                                         if k != "rows"}
            res["per_seq_fov"][r["key"]]["self_rate_mean"] = round(float(np.mean(r["self_rate"])), 4)
            res["per_seq_fov"][r["key"]]["mid_rate_mean"] = round(float(np.mean(r["mid_rate"])), 4)
            res["per_seq_fov"][r["key"]]["n_dropped"] = r["n_dropped"]
    if "helipr-os2/Sample" in seqrec:
        r = seqrec["helipr-os2/Sample"]
        s, inf = unit_fov("helipr_os2_Sample", [r])
        s["note"] = "single sequence (Sample = first ~60 s of Roundabout01); not 'all seqs' persistent"
        res["units"]["helipr_os2_Sample"] = s
        recs_by_unit["helipr_os2_Sample"], info_by_unit["helipr_os2_Sample"] = [r], inf
        res["sensitivity"]["helipr_os2_Sample"] = {f"theta{t}": {k: unit_fov("x", [r], t)[0].get(k) for k in (
            "blind_frac_cells", "n_rings_blind_gt30deg")} for t in THETAS}
        res["archive_os2"] = r["archive"]
        res["per_seq_fov"]["helipr-os2/Sample"] = {"icp_check": {k: v for k, v in (r.get("icp_check") or {}).items()
                                                                 if k != "rows"},
                                                   "motion_azimuth_sensor_frame": r["motion_azimuth_sensor_frame"]}
    # 그림
    figs = []
    for ds, units in list(DATASETS.items()) + [("helipr_os2", ["helipr_os2_Sample"])]:
        units = [u for u in units if u in recs_by_unit]
        if not units:
            continue
        fp = OUT / f"fig_fov_{ds}.png"
        draw(ds, units, fp, recs_by_unit, info_by_unit, res["units"])
        os.chmod(fp, 0o666)
        figs.append(fp.name)
    res["figures"] = figs
    # 방향별 R@1
    rp = OUT / "recall.json"
    if rp.exists():
        rc = json.load(open(rp))
        ref = {}
        for k, v in lin["heldout_per_seq"].items():
            ref[(k, "raw")], ref[(k, "lin16_s0")] = v["raw"], v["lin16_s0"]
        for k, v in lin["dev_controls"]["reps"]["raw"]["per_seq"].items():
            ref[(k, "raw")] = v
        checks = []
        for key, reps in rc.items():
            res["recall"][key] = {"unit": unit_of(key), "lin16_role": role.get(key)}
            for rep, m in reps.items():
                res["recall"][key][rep] = {k: m[k] for k in ("R@1", "n_pos", "dir_n", "dir_hit1", "AP", "R@100P")}
                if (key, rep) in ref:
                    r0 = ref[(key, rep)]
                    checks.append(abs((m["R@1"] or 0) - (r0["R@1"] or 0)) < 1e-9 and m["dir_hit1"] == r0["dir_hit1"])
        res["recall_ref_check"] = {"n_compared": len(checks), "all_equal": bool(all(checks))}
        groups = {}
        for key, v in res["recall"].items():
            for rep in ("raw", "lin16_s0"):
                for g in (rep == "raw" and ["all"]) or ["all", v["lin16_role"]]:
                    e = groups.setdefault((v["unit"], rep, g), {"seqs": [], **{f"n_{d}": 0 for d in (
                        "same", "reverse", "crossing", "unknown")}, **{f"hit_{d}": 0 for d in (
                        "same", "reverse", "crossing", "unknown")}, "n_pos": 0, "hit_all": 0})
                    m = v[rep]
                    e["seqs"].append(key)
                    e["n_pos"] += m["n_pos"]
                    e["hit_all"] += sum(m["dir_hit1"].values())
                    for d in ("same", "reverse", "crossing", "unknown"):
                        e[f"n_{d}"] += m["dir_n"][d]
                        e[f"hit_{d}"] += m["dir_hit1"][d]
        for (u, rep, g), e in sorted(groups.items()):
            for d in ("same", "reverse", "crossing"):
                e[f"R@1_{d}"] = round(e[f"hit_{d}"] / e[f"n_{d}"], 4) if e[f"n_{d}"] else None
            e["R@1_all"] = round(e["hit_all"] / e["n_pos"], 4) if e["n_pos"] else None
            res["units_recall"][f"{u}|{rep}|{g}"] = e
    mp = OUT / "mcd_rotation.json"
    if mp.exists():
        res["mcd_rotation"] = {k: {kk: vv for kk, vv in v.items()} for k, v in json.load(open(mp)).items()}
    chmod_write(OUT / "metrics.json", json.dumps(res, indent=1))
    print_tables(res)
    return 0


def print_tables(res):
    print("\n== FOV units (theta 0.05, persist 0.95)")
    for u, s in res["units"].items():
        w = s.get("widest") or {}
        print(f"{u:20s} rings {s['n_rings']:3d} empty {s['n_empty_rings']} blind {s['blind_frac_cells']} "
              f">30deg {s['n_rings_blind_gt30deg']} any {s['n_rings_any_blind']} fwd {s['forward_az_sensor_deg']} "
              f"widest {w.get('width_deg')} @ {w.get('center_az_sensor_deg')} ({w.get('position')}, rel "
              f"{w.get('center_rel_forward_deg')}, elev {w.get('ring_elev_deg')}) incons {s['inconsistent_cells_frac(blind in some seqs only)']}"
              f" | az all-rings {s['az_blind_all_rings']} | az >=50% {s['az_blind_ge50pct_rings']}")
    print("\n== sensitivity (blind frac / rings>30)")
    for u, d in res["sensitivity"].items():
        print(f"{u:20s} " + "  ".join(f"{k.replace('theta', 't').replace('_persist', '/p')}:"
                                      f"{v['blind_frac_cells']}/{v['n_rings_blind_gt30deg']}" for k, v in d.items()))
    if res["units_recall"]:
        print("\n== R@1 by direction (hit/n)")
        for k, e in res["units_recall"].items():
            print(f"{k:34s} same {e['hit_same']}/{e['n_same']} rev {e['hit_reverse']}/{e['n_reverse']} "
                  f"cross {e['hit_crossing']}/{e['n_crossing']} unk {e['n_unknown']}  "
                  f"R@1 same {e['R@1_same']} rev {e['R@1_reverse']} cross {e['R@1_crossing']}")
        print("ref check", res.get("recall_ref_check"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--seqs", nargs="+", required=True)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--icp", type=int, default=15)
    o = sub.add_parser("ouster")
    o.add_argument("--n", type=int, default=200)
    o.add_argument("--icp", type=int, default=15)
    r = sub.add_parser("recall")
    r.add_argument("--seqs", nargs="*")
    sub.add_parser("mcdrot")
    sub.add_parser("report")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"scan": cmd_scan, "ouster": cmd_ouster, "recall": cmd_recall, "mcdrot": cmd_mcdrot,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
