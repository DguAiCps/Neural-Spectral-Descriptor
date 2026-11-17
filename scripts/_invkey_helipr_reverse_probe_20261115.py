#!/usr/bin/env python3
"""invkey 진단: HeLiPR(VLP-16)에서 반대 방향 재방문이 실패하는 원인을 찾는다 (2026-11-15). 학습 없음.

표현은 그대로 쓴다: occupancy()·bank() (scripts/_scatter_probe_20261114.py), describe()의 spec
(scripts/_poolalt_probe_20261114.py). 검색·PR 회계와 방향·옆 어긋남 분류는 scripts/_invkey_submap_probe_20261115.py의
evaluate()·breakdown()을 그대로 부른다(질의 q >= 30, 후보 j <= q-30, 양성 GT <= 5 m, 제외 간격 0).
선형 16: results/invkey_study_20261114/linear/run_final_K16_tau0.05_lr0.003_s0_e40.npz의 s, W를 단일 스캔에서 배운
그대로 개입한 spec에도 쓴다(입력 분포가 달라지므로 선형 16 수치는 참고).

자세: helipr LiDAR_GT/Velodyne_gt.txt는 Velodyne 좌표(X = I), GT 시각 = 스캔 파일 시각(스캔 끝).
원시 점 파일(22바이트/점)에 ring(uint16)과 점별 time(float32, 스캔 시각 기준 초, 약 -0.0995 .. +0.0013)이 있다.

지면: _invkey_augment_20261115.ground_plane()(3-10 m)으로 시작해 수평 3-30 m, |높이| < 0.25 m 점으로 평면을 두 번
다시 맞춘다(기울기 8도 넘으면 앞 단계 유지). 지면 위 높이 h = z - (a x + b y + c). 바닥 점 = h <= 0.3 m.

개입(모든 키프레임 스캔, 지도·질의 같게; 이름 -> 남기는 점)
  base         전부 (저장된 spec·기존 수치 재현 확인용)
  nognd        h > 0.3                 gndonly     h <= 0.3
  nognd_far20  바닥 점 중 수평 20 m 밖만 뺌 (기울기 때문에 앞뒤가 비대칭인 먼 바닥 고리만 지움)
  deskew       점별 time과 GT 자세(이 스캔과 바로 앞 GT, 10 Hz) 보간으로 스캔 끝 좌표로 옮김
  level        스캔마다 지면 평면 법선이 z축이 되게 센서 원점 기준 회전
  r20, r40     수평 거리 <= 20, <= 40 m           out20   > 20 m
  h03_2        0.3 < h <= 2.0          h2up        h > 2.0
  fb30/45/60   센서 x(차량 앞) 기준 앞 0도와 뒤 180도 둘레 ±30/45/60도를 지움 (좌우 옆면만 남김)
  lr45         좌 90도, 우 270도 둘레 ±45도를 지움 (앞뒤만 남김; fb45의 대조)
  symvis       줄마다 사각(blind_table: 세계 반환 < 0.03 또는 자기 반환 > 0.1, 5도 칸)과 그 180도 맞은편을 지움
  symvis_perp  같은 사각 표를 90도 돌려 지움(대조; 줄 0-4는 사각이 약 170도라 symvis와 거의 같은 양을 지워 대조가 약함)
  drop_r0_4    줄 0-4(-15..-7도, 앞쪽 절반이 리그에 가림)를 뺌      drop_r11_15  줄 11-15(+7..+15도)를 뺌(대조)
  조합은 '+'로 잇는다(예: nognd+deskew). 점 선택은 교집합, 좌표 변환(deskew, level)은 선택 뒤에 적용.

명령 (컨테이너 ring_baseline, 리포 루트)
  python3 scripts/_invkey_helipr_reverse_probe_20261115.py sensor --seq helipr/Roundabout01
  python3 scripts/_invkey_helipr_reverse_probe_20261115.py pairs  --seq helipr/Roundabout01
  python3 scripts/_invkey_helipr_reverse_probe_20261115.py figs   --seq helipr/Roundabout01
  python3 scripts/_invkey_helipr_reverse_probe_20261115.py interv --seq helipr/Roundabout01 --variants base nognd ...
  python3 scripts/_invkey_helipr_reverse_probe_20261115.py report
출력: results/invkey_study_20261114/helipr_reverse/ (sensor_*.json, pairs_*.json/.npz, interv_*.json, metrics.json, *.png).
다시 계산한 spec은 저장하지 않는다(메모리에서 평가만).
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

from _invkey_augment_20261115 import ground_plane  # noqa: E402
from _invkey_data_survey_20261114 import DIRS, classify, headings, positive_pairs  # noqa: E402
from _invkey_submap_probe_20261115 import (  # noqa: E402
    apply_T, breakdown, evaluate, extrinsic, icp, load_lin16, offsets, path_length, seq_data)
from _invkey_train_linear_20261114 import DEV, NFFT, chmod_write, encode, to_freq_t  # noqa: E402

OUT = REPO / "results/invkey_study_20261114/helipr_reverse"
DIAGA = REPO / "results/invkey_study_20261114/submap/diagA.json"
N = 160
GND_H = 0.3
RAW_DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
                   ("ring", "<u2"), ("time", "<f4")])
RINGS = ((1.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 80.0))
SECTORS = ("front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right")
ATOMS = ("base", "nognd", "gndonly", "nognd_far20", "deskew", "level", "r20", "r40", "out20",
         "h03_2", "h2up", "fb30", "fb45", "fb60", "lr45", "symvis", "symvis_perp", "drop_r0_4", "drop_r11_15")
AZB = 72                      # 사각 표의 방위 칸 (5도)
BLIND = {"tab": None}         # 시퀀스마다 blind_table()로 채움 (16, AZB) bool


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def blind_table(sc, n, nsamp=150):
    """줄 x 방위(5도) 사각: 세계 반환(수평 >= 2.5 m) 비율 < 0.03 또는 자기 반환(3차원 < 1 m) 비율 > 0.1.
    비율 = 반환 수 / (VLP-16 10 Hz 한 바퀴 1800발 / 72칸 x 표본 수)."""
    nr = 16 if sc.sensor == "helipr" else 64
    world = np.zeros((nr, AZB))
    selfr = np.zeros((nr, AZB))
    for i in np.linspace(0, n - 1, nsamp).astype(int):
        d = sc.get(int(i))
        p = d["xyz"].astype(np.float64)
        ok = (d["ring"] >= 0) & (d["ring"] < nr)
        p, rg = p[ok], d["ring"][ok]
        r = np.hypot(p[:, 0], p[:, 1])
        b = np.floor(np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360.0 / (360.0 / AZB)).astype(int) % AZB
        np.add.at(world, (rg, b), (r >= 2.5).astype(float))
        np.add.at(selfr, (rg, b), (np.linalg.norm(p, axis=1) < 1.0).astype(float))
    if sc.sensor == "helipr":
        exp = 1800.0 / AZB * nsamp
    else:   # kitti360: 줄마다 방위 칸 반환 수의 95% 분위를 "다 돌아온" 수로 본다(줄별 발사 수를 모름)
        exp = np.maximum(np.percentile(world + selfr, 95, axis=1, keepdims=True), 1.0)
    w, s = world / exp, selfr / exp
    tab = (w < 0.03) | (s > 0.1)
    return tab, w, s


# ---------------------------------------------------------------- 스캔 읽기와 전처리
class Scans:
    """키프레임 i의 점. helipr는 원시 파일에서 ring·time까지 읽는다. 자세는 로더의 전체 GT(모든 스캔)."""

    def __init__(self, key, ids):
        from _pd_build_test_caches import _make_loader
        self.sensor, seq = key.split("/")
        self.L = _make_loader(self.sensor, seq)
        self.ids = ids

    def get(self, i):
        sid = int(self.ids[i])
        if self.sensor == "helipr":
            d = np.fromfile(self.L.scan_files[sid], dtype=RAW_DT)
            xyz = np.stack([d["x"], d["y"], d["z"]], 1)
            k = int(self.L.scan_pose_indices[sid])
            ts = self.L.timestamps_gt
            assert self.L.scan_timestamps[sid] == ts[k]
            prev = None
            if 0 < k < len(ts) - 1:
                prev = (self.L.poses[k], self.L.poses[k - 1], (ts[k] - ts[k - 1]) / 1e9,
                        self.L.poses[k + 1], (ts[k + 1] - ts[k]) / 1e9)
            return {"xyz": xyz, "ring": d["ring"].astype(np.int16), "time": d["time"].astype(np.float64),
                    "prev": prev}
        p = self.L[sid]["points"]
        xyz = np.ascontiguousarray(p[:, :3], dtype=np.float32)
        ring = None
        if self.sensor == "kitti360":      # 점 순서로 줄을 센다(_invkey_augment_20261115.estimate_rings, 0이 맨 위)
            from _invkey_augment_20261115 import estimate_rings
            q = xyz.astype(np.float64)
            ring, _ = estimate_rings(p, "kitti360", q, np.linalg.norm(q, axis=1) > 0.5)
        return {"xyz": xyz, "ring": ring, "time": None, "prev": None}


def ground_fit(xyz):
    """(a, b, c, info). z = a x + b y + c (센서 원점 좌표)."""
    p = xyz.astype(np.float64)
    valid = np.linalg.norm(p, axis=1) > 0.5
    g = ground_plane(p, valid)
    if not g["ok"]:
        return 0.0, 0.0, float(np.percentile(p[valid, 2], 5)), {"stage": "none"}
    a, b, c = g["a"], g["b"], g["c"]
    rxy = np.hypot(p[:, 0], p[:, 1])
    stage = "aug" if g["fallback"] is None else "aug_fallback"
    for _ in range(2):
        h = p[:, 2] - (a * p[:, 0] + b * p[:, 1] + c)
        sel = valid & (rxy >= 3.0) & (rxy <= 30.0) & (np.abs(h) < 0.25)
        if sel.sum() < 100:
            break
        A = np.c_[p[sel, 0], p[sel, 1], np.ones(sel.sum())]
        (a2, b2, c2), *_ = np.linalg.lstsq(A, p[sel, 2], rcond=None)
        if np.degrees(np.arctan(np.hypot(a2, b2))) > 8.0:
            break
        a, b, c = float(a2), float(b2), float(c2)
        stage = "wide"
    return a, b, c, {"stage": stage, "tilt_deg": float(np.degrees(np.arctan(np.hypot(a, b))))}


def deskew_xyz(xyz, t, prev):
    """점별 시각 t(스캔 시각 = GT 시각 기준 초)로 스캔 시각의 센서 좌표로 옮긴다.
    prev = (T_k, T_{k-1}, dt_prev, T_{k+1}, dt_next). t < 0은 앞 GT, t > 0은 다음 GT와 보간한다.
    (Roundabout01은 t가 약 -0.1..0(스캔 시각 = 끝), DCC04는 0..+0.1(스캔 시각 = 시작).)"""
    from scipy.spatial.transform import Rotation
    Tk, Tkm1, dtp, Tkp1, dtn = prev
    p = xyz.astype(np.float64)
    out = np.empty_like(p)
    for m, Tn, dt, sign in ((t < 0, Tkm1, dtp, -1.0), (t >= 0, Tkp1, dtn, 1.0)):
        if not m.any():
            continue
        D = np.linalg.inv(Tk) @ Tn                   # 이웃 GT 자세를 지금 좌표로
        rv = Rotation.from_matrix(D[:3, :3]).as_rotvec()
        alpha = sign * t[m] / dt                     # 0: 스캔 시각, 1: 이웃 GT 시각
        R = Rotation.from_rotvec(alpha[:, None] * rv[None]).as_matrix()
        out[m] = np.einsum("nij,nj->ni", R, p[m]) + alpha[:, None] * D[:3, 3][None]
    return out.astype(np.float32)


def level_xyz(xyz, a, b):
    """지면 평면 법선 (-a, -b, 1)을 z축으로 돌린다(센서 원점 기준)."""
    from scipy.spatial.transform import Rotation
    n = np.array([-a, -b, 1.0])
    n /= np.linalg.norm(n)
    axis = np.cross(n, [0.0, 0.0, 1.0])
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return xyz
    R = Rotation.from_rotvec(axis / s * np.arcsin(min(s, 1.0))).as_matrix()
    return (xyz.astype(np.float64) @ R.T).astype(np.float32)


def prep(scan):
    """개입에 필요한 점별 양: 높이 h, 수평 거리 r, 방위 az(도), 평면."""
    xyz = scan["xyz"]
    a, b, c, info = ground_fit(xyz)
    x, y, z = (xyz[:, k].astype(np.float64) for k in range(3))
    scan.update(h=z - (a * x + b * y + c), r=np.hypot(x, y), az=np.degrees(np.arctan2(y, x)),
                plane=(a, b, c), plane_info=info)
    return scan


def select(scan, atom):
    """atom의 점 선택 마스크(좌표 변환 atom은 전부 True)."""
    h, r, az = scan["h"], scan["r"], scan["az"]
    gnd = h <= GND_H
    if atom in ("base", "deskew", "level"):
        return np.ones(len(h), bool)
    if atom == "nognd":
        return ~gnd
    if atom == "gndonly":
        return gnd
    if atom == "nognd_far20":
        return ~(gnd & (r > 20.0))
    if atom == "r20":
        return r <= 20.0
    if atom == "r40":
        return r <= 40.0
    if atom == "out20":
        return r > 20.0
    if atom == "h03_2":
        return (h > GND_H) & (h <= 2.0)
    if atom == "h2up":
        return h > 2.0
    if atom.startswith("fb"):
        hw = float(atom[2:])
        return ~((np.abs(wrap(az)) < hw) | (np.abs(wrap(az - 180.0)) < hw))
    if atom.startswith("lr"):
        hw = float(atom[2:])
        return ~((np.abs(wrap(az - 90.0)) < hw) | (np.abs(wrap(az + 90.0)) < hw))
    if atom in ("symvis", "symvis_perp"):
        # 줄마다 사각 방위와 그 180도 맞은편을 함께 지운다. 대조(perp): 같은 칸 수를 90도 돌린 위치(사각+90, +270)에서 지운다.
        tab = BLIND["tab"]
        b = np.floor((az % 360.0) / (360.0 / AZB)).astype(int) % AZB
        rg = scan["ring"].astype(int)
        okr = (rg >= 0) & (rg < len(tab))              # 줄을 모르는 점은 그대로 둔다
        rc = np.clip(rg, 0, len(tab) - 1)
        if atom == "symvis":
            drop = tab[rc, b] | tab[rc, (b + AZB // 2) % AZB]
        else:
            drop = tab[rc, (b + AZB // 4) % AZB] | tab[rc, (b + 3 * AZB // 4) % AZB]
        return ~(drop & okr)
    if atom == "drop_r0_4":
        return scan["ring"] > 4
    if atom == "drop_r11_15":       # drop_r0_4의 대조: 앞뒤 대칭으로 보는 위쪽 다섯 줄(+7..+15도)을 뺀다
        return scan["ring"] < 11
    raise ValueError(atom)


def variant_points(scan, name):
    """'+'로 이은 개입을 적용한 점 (M, 3) float32."""
    atoms = name.split("+")
    m = np.ones(len(scan["h"]), bool)
    for at in atoms:
        assert at in ATOMS, at
        m &= select(scan, at)
    p = scan["xyz"]
    if "deskew" in atoms:
        p = deskew_xyz(p, scan["time"], scan["prev"]) if scan["prev"] is not None else p
    if "level" in atoms:
        p = level_xyz(p, *scan["plane"][:2])
    return p[m]


def raster(xy):
    """occupancy()와 같은 칸 규칙(1 m 안 제외는 호출 쪽에서) bool (N, N)."""
    g = np.zeros((N, N), bool)
    ix, iy = np.floor(xy[:, 0] + 80.0).astype(int), np.floor(xy[:, 1] + 80.0).astype(int)
    ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
    g[ix[ok], iy[ok]] = True
    return g


def own(p):
    return p[np.hypot(p[:, 0], p[:, 1]) >= 1.0]


# ---------------------------------------------------------------- 유사도
class Sim:
    """spec 두 개(또는 묶음)의 원형 상관 정렬 코사인 (raw와 선형 16)."""

    def __init__(self):
        import torch
        from _scatter_probe_20261114 import bank
        torch.backends.cuda.matmul.allow_tf32 = False
        self.F = torch.from_numpy(bank()).to(DEV)
        self.s16, self.W16 = load_lin16()

    def spec(self, grid):
        from _poolalt_probe_20261114 import describe
        return describe(grid.astype(np.float32), self.F)[1].astype(np.float16)

    def pair(self, A, B):
        """A, B (805, 16) float16 -> (raw, lin16)."""
        import torch
        from _poolalt_probe_20261114 import aligned_sim
        out = []
        X = np.stack([A, B])
        for s, W in ((None, None), (self.s16, self.W16)):
            with torch.no_grad():
                D = encode(X, s, W)
                Fq, nq = to_freq_t(D)
                out.append(float(aligned_sim(Fq[:, :1], nq[:1], Fq[:, 1:], nq[1:], NFFT)[0, 0]))
        return tuple(out)


# ---------------------------------------------------------------- 공통 시퀀스 정보
def seq_info(key):
    spec, xyz, ids, P = seq_data(key)
    sensor = key.split("/")[0]
    TL = P @ extrinsic(sensor)
    return spec, xyz, ids, TL


def base_retrieval(spec, xyz, TL):
    """저장 spec의 검색 결과(raw, lin16): 질의별 양성·1위·1위 점수·과거 최근접과 방향."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    s16, W16 = load_lin16()
    s = path_length(xyz)
    res = {}
    for rep, (sv, Wv) in (("raw", (None, None)), ("lin16", (s16, W16))):
        with torch.no_grad():
            D = encode(spec, sv, Wv)
            ev = evaluate(D, D, xyz, s, [0.0], 16 if Wv is None else 64)
        m, arrs = ev[0.0]
        del D
        torch.cuda.empty_cache()
        res[rep] = (m, arrs)
    return res


# ---------------------------------------------------------------- 센서 특성
def cmd_sensor(a):
    key = a.seq
    spec, xyz, ids, TL = seq_info(key)
    del spec
    n = len(ids)
    sc = Scans(key, ids)
    out = {"seq": key, "n_keyframes": n}
    # GT 자세에서: 센서 좌표의 세계 위쪽 벡터, 진행 방향의 센서 좌표 고도
    up = TL[:, :3, :3].transpose(0, 2, 1) @ np.array([0.0, 0.0, 1.0])
    mv = np.diff(TL[:, :3, 3], axis=0)
    loc = np.einsum("nji,nj->ni", TL[:-1, :3, :3], mv)
    nm = np.linalg.norm(loc, axis=1)
    u = loc[nm > 0.5] / nm[nm > 0.5, None]
    out["gt_tilt"] = {
        "world_up_in_sensor_median": [float(np.median(up[:, k])) for k in range(3)],
        "pitch_deg_from_up_x(+: sensor x points up)": float(np.degrees(np.arcsin(np.median(up[:, 0])))),
        "roll_deg_from_up_y(+: sensor y points up)": float(np.degrees(np.arcsin(np.median(up[:, 1])))),
        "motion_elev_in_sensor_deg_median": float(np.degrees(np.arcsin(np.median(u[:, 2])))),
        "motion_azimuth_in_sensor_deg_median": float(np.degrees(np.median(np.arctan2(u[:, 1], u[:, 0])))),
        "motion_abs_cos_x_gt_0.9": float((np.abs(u[:, 0]) > 0.9).mean())}
    # 속도: 키프레임과 전체 GT
    if sc.sensor == "helipr":
        ts = sc.L.timestamps_gt / 1e9
        gp = sc.L.poses[:, :3, 3]
        v = np.linalg.norm(np.diff(gp, axis=0), axis=1) / np.diff(ts)
        yawr = np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(sc.L.poses[:, 1, 0], sc.L.poses[:, 0, 0]))))) / np.diff(ts)
        out["speed_mps_allscans"] = {f"p{q}": float(np.percentile(v, q)) for q in (5, 25, 50, 75, 95, 99)}
        out["yaw_rate_dps_allscans"] = {f"p{q}": float(np.percentile(yawr, q)) for q in (50, 90, 99)}
        kv = np.zeros(n)
        kidx = np.asarray(sc.L.scan_pose_indices)[ids]
        kv = v[np.clip(kidx - 1, 0, len(v) - 1)]
        out["speed_mps_keyframes"] = {f"p{q}": float(np.percentile(kv, q)) for q in (5, 25, 50, 75, 95)}
    # 표본 스캔
    pick = np.linspace(0, n - 1, a.n).astype(int)
    AZB = 72
    hist = np.zeros((16, AZB))
    rng_med = []
    planes, stages, el_ring, trange, seam = [], [], [], [], []
    gr_front, gr_back, gr_left, gr_right = ([] for _ in range(4))
    for i in pick:
        scan = prep(sc.get(int(i)))
        p = scan["xyz"].astype(np.float64)
        a_, b_, c_ = scan["plane"]
        planes.append((a_, b_, c_))
        stages.append(scan["plane_info"]["stage"])
        azb = np.floor((scan["az"] % 360.0) / (360.0 / AZB)).astype(int) % AZB
        if scan["ring"] is not None:
            np.add.at(hist, (scan["ring"], azb), 1)
            r3 = np.linalg.norm(p, axis=1)
            el = np.degrees(np.arcsin(np.clip(p[:, 2] / np.maximum(r3, 1e-9), -1, 1)))
            el_ring.append([float(np.median(el[scan["ring"] == k])) if (scan["ring"] == k).any() else np.nan
                            for k in range(16)])
            t = scan["time"]
            trange.append((float(t.min()), float(t.max())))
            seam.append(float(scan["az"][np.argmin(t)]))
            # 바닥 고리 반경(방위 부채꼴 ±20도, |h| < 0.2): 줄 0..7
            gnd = np.abs(scan["h"]) < 0.2
            for lst, c0 in ((gr_front, 0.0), (gr_back, 180.0), (gr_left, 90.0), (gr_right, -90.0)):
                m = gnd & (np.abs(wrap(scan["az"] - c0)) < 20.0)
                lst.append([float(np.median(scan["r"][m & (scan["ring"] == k)]))
                            if (m & (scan["ring"] == k)).sum() > 5 else np.nan for k in range(8)])
        rng_med.append([float(np.median(scan["r"][azb == k])) if (azb == k).any() else np.nan for k in range(AZB)])
    planes = np.array(planes)
    tilt = np.degrees(np.arctan(np.hypot(planes[:, 0], planes[:, 1])))
    # 평면 기울기: a = dz/dx. 센서 x가 위로 들리면 앞쪽 지면이 센서 좌표에서 내려가 a < 0.
    out["ground_plane"] = {
        "n_scans": len(pick), "stage_counts": {s: stages.count(s) for s in set(stages)},
        "a_dzdx_median": float(np.median(planes[:, 0])), "b_dzdy_median": float(np.median(planes[:, 1])),
        "pitch_deg_from_plane(+: sensor x up)": float(np.degrees(np.arctan(-np.median(planes[:, 0])))),
        "roll_deg_from_plane(+: sensor y up)": float(np.degrees(np.arctan(-np.median(planes[:, 1])))),
        "tilt_deg": {"p10": float(np.percentile(tilt, 10)), "median": float(np.median(tilt)),
                     "p90": float(np.percentile(tilt, 90))},
        "c_sensor_height_m_median": float(-np.median(planes[:, 2])),
        "tilt_direction_deg_median(azimuth of downhill in sensor frame)": float(np.degrees(np.arctan2(
            -np.median(planes[:, 1]), -np.median(planes[:, 0]))))}
    out["range_median_by_azimuth_5deg"] = np.nanmedian(np.array(rng_med), 0).round(2).tolist()
    if scan["ring"] is not None:
        expect = 1800.0 / AZB * len(pick)      # VLP-16 10 Hz 약 0.2도 간격
        rate = hist / expect
        out["return_rate_ring_x_az5"] = rate.round(3).tolist()
        out["return_rate_by_az5_allrings"] = rate.mean(0).round(3).tolist()
        out["return_rate_by_ring"] = rate.mean(1).round(3).tolist()
        dens = hist.sum(0)
        out["density_by_az5_rel_mean"] = (dens / dens.mean()).round(3).tolist()
        out["ring_elev_deg_median"] = np.nanmedian(np.array(el_ring), 0).round(2).tolist()
        tr = np.array(trange)
        out["point_time_s"] = {"min_med": float(np.median(tr[:, 0])), "max_med": float(np.median(tr[:, 1])),
                               "note": "relative to scan timestamp (= GT timestamp)"}
        sm = np.array(seam)
        out["seam_azimuth_deg"] = {"values_sample": sm[:20].round(1).tolist(),
                                   "hist_45deg": np.histogram(sm % 360.0, bins=8, range=(0, 360))[0].tolist()}
        out["ground_ring_range_m_by_sector(ring0..7, |h|<0.2, +-20deg)"] = {
            nm_: np.nanmedian(np.array(v), 0).round(2).tolist()
            for nm_, v in (("front", gr_front), ("back", gr_back), ("left", gr_left), ("right", gr_right))}
    # deskew 확인: 움직이는 구간의 10 m 쌍 30개, 비지면 구조 점의 GT 정합 잔차(수평 최근접 중앙값) 원본 대 보정
    if sc.sensor == "helipr":
        from scipy.spatial import cKDTree
        s = path_length(xyz)
        b10 = np.searchsorted(s, s + 10.0)
        cand = np.flatnonzero((b10 < n) & (kv > 6.0))
        rng = np.random.default_rng(0)
        rows = []
        for i in rng.choice(cand, min(30, len(cand)), replace=False):
            j = int(b10[i])
            A, B = prep(sc.get(int(i))), prep(sc.get(j))
            T = np.linalg.inv(TL[i]) @ TL[j]
            r = {}
            for nm_, fa, fb in (("raw", A["xyz"], B["xyz"]),
                                ("deskew", deskew_xyz(A["xyz"], A["time"], A["prev"]),
                                 deskew_xyz(B["xyz"], B["time"], B["prev"]))):
                sa = fa[(A["h"] > 0.5) & (A["r"] > 3) & (A["r"] < 40)]
                sb = apply_T(T, fb[(B["h"] > 0.5) & (B["r"] > 3) & (B["r"] < 40)])
                d = cKDTree(sa[:, :2]).query(sb[:, :2])[0]
                r[nm_] = (float(np.median(d)), float((d < 0.2).mean()))
            rows.append(r)
        out["deskew_check_10m_pairs_moving"] = {
            k: {"res_med_m": float(np.median([r[k][0] for r in rows])),
                "frac_lt_0.2m": float(np.median([r[k][1] for r in rows]))} for k in ("raw", "deskew")}
        out["deskew_check_10m_pairs_moving"]["n"] = len(rows)
        out["deskew_check_10m_pairs_moving"]["deskew_better_frac"] = float(
            np.mean([r["deskew"][0] < r["raw"][0] for r in rows]))
    path = OUT / f"sensor_{key.replace('/', '_')}.json"
    chmod_write(path, text=json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("return_rate_ring")
                      and k not in ("range_median_by_azimuth_5deg",)}, indent=1)[:6000], flush=True)
    return 0


# ---------------------------------------------------------------- 1단계: 쌍 겹침과 공통 영역 상한
CX, CY = np.meshgrid(np.arange(N) - 80 + 0.5, np.arange(N) - 80 + 0.5, indexing="ij")
CR = np.hypot(CX, CY)
CAZ = np.degrees(np.arctan2(CY, CX))
SEC_IDX = (np.floor((CAZ % 360.0 + 22.5) / 45.0).astype(int)) % 8


def vis_map(p):
    """수평 1도 방위 칸마다 가장 먼 반환의 수평 거리(이웃 칸 포함 최대). p는 자기 좌표."""
    r = np.hypot(p[:, 0], p[:, 1])
    ok = r >= 1.0
    b = np.floor(np.degrees(np.arctan2(p[ok, 1], p[ok, 0])) % 360.0).astype(int) % 360
    m = np.zeros(360)
    np.maximum.at(m, b, r[ok])
    return np.maximum(m, np.maximum(np.roll(m, 1), np.roll(m, -1)))


def visible(vm, x, y):
    b = np.floor(np.degrees(np.arctan2(y, x)) % 360.0).astype(int) % 360
    return np.hypot(x, y) <= vm[b] + 1.0


def to_frame(T, x, y):
    """2차원 점 (x, y)를 4x4 T로 옮긴다(z = 0)."""
    return T[0, 0] * x + T[0, 1] * y + T[0, 3], T[1, 0] * x + T[1, 1] * y + T[1, 3]


def cell_index(xy):
    ix, iy = np.floor(xy[:, 0] + 80.0).astype(int), np.floor(xy[:, 1] + 80.0).astype(int)
    ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
    return ix, iy, ok


def overlap_stats(A, B, T):
    """A(질의) 좌표에서 두 스캔의 1 m 점유 겹침. T: B 좌표 -> A 좌표. 반환 dict(영역 -> (교, 합, |A|, |B|, A in dil B, B in dil A))."""
    from scipy.ndimage import binary_dilation
    pa = own(A["xyz"])
    ma = np.hypot(A["xyz"][:, 0], A["xyz"][:, 1]) >= 1.0
    mb = np.hypot(B["xyz"][:, 0], B["xyz"][:, 1]) >= 1.0
    pb = apply_T(T, B["xyz"][mb].astype(np.float64))
    ga_g, ga_n = raster(pa[A["h"][ma] <= GND_H]), raster(pa[A["h"][ma] > GND_H])
    gb_g, gb_n = raster(pb[B["h"][mb] <= GND_H]), raster(pb[B["h"][mb] > GND_H])
    # B 격자가 덮는 칸만
    bx, by = to_frame(np.linalg.inv(T), CX, CY)
    cover = (np.abs(bx) < 80.0) & (np.abs(by) < 80.0) & (CR >= 1.0)
    k3 = np.ones((3, 3), bool)
    out = {}
    for lay, ga, gb in (("all", ga_g | ga_n, gb_g | gb_n), ("ground", ga_g, gb_g), ("nonground", ga_n, gb_n)):
        da, db = binary_dilation(ga, k3), binary_dilation(gb, k3)
        regions = {"all": cover}
        for lo, hi in RINGS:
            regions[f"r{int(lo)}-{int(hi)}"] = cover & (CR >= lo) & (CR < hi)
        for si, sn in enumerate(SECTORS):
            regions[f"sec_{sn}"] = cover & (SEC_IDX == si)
        for rn, m in regions.items():
            out[f"{lay}|{rn}"] = (int((ga & gb & m).sum()), int(((ga | gb) & m).sum()), int((ga & m).sum()),
                                  int((gb & m).sum()), int((ga & db & m).sum()), int((gb & da & m).sum()))
    return out


def struct_fit(A, B, T):
    """B의 비지면 구조 점(수평 3-40 m, h 0.5-4 m)을 T로 A 좌표에 옮겼을 때 A 구조 점 0.3 m(수평) 안에 드는 비율."""
    from scipy.spatial import cKDTree
    sa = A["xyz"][(A["h"] > 0.5) & (A["h"] < 4.0) & (A["r"] >= 3) & (A["r"] <= 40)]
    sb = B["xyz"][(B["h"] > 0.5) & (B["h"] < 4.0) & (B["r"] >= 3) & (B["r"] <= 40)]
    if len(sa) < 10 or len(sb) < 10:
        return 0.0
    d = cKDTree(sa[:, :2]).query(apply_T(T, sb.astype(np.float64))[:, :2])[0]
    return float((d < 0.3).mean())


def masked_grids(A, B, T):
    """공통 영역 상한용 격자 쌍(각자 자기 좌표). T: B -> A."""
    from scipy.ndimage import binary_dilation
    Ti = np.linalg.inv(T)
    pa, pb = own(A["xyz"]).astype(np.float64), own(B["xyz"]).astype(np.float64)
    GA, GB = raster(pa), raster(pb)
    pb_a = apply_T(T, pb)
    GBa = raster(pb_a)
    out = {"orig": (GA, GB)}
    # g1 정확: 두 스캔 모두 점유된 칸(A 좌표). B 쪽은 그 칸에 떨어지는 B 점을 B 좌표에서 다시 칸으로.
    C = GA & GBa
    ix, iy, ok = cell_index(pb_a)
    inC = np.zeros(len(pb), bool)
    inC[ok] = C[ix[ok], iy[ok]]
    out["common_exact"] = (C, raster(pb[inC]))
    # g1 허용: A 칸 중 B 칸 3x3 이웃에 있는 것, B 점 중 A 칸 3x3 이웃에 떨어지는 것
    k3 = np.ones((3, 3), bool)
    dA = binary_dilation(GA, k3)
    inDA = np.zeros(len(pb), bool)
    inDA[ok] = dA[ix[ok], iy[ok]]
    out["common_tol"] = (GA & binary_dilation(GBa, k3), raster(pb[inDA]))
    # g2 보이는 영역: 두 센서 모두에서 방위별 최대 반환 거리 안인 칸
    vA, vB = vis_map(pa), vis_map(pb)
    bx, by = to_frame(Ti, CX, CY)
    VA = visible(vA, CX, CY) & visible(vB, bx, by)
    ax_, ay_ = to_frame(T, CX, CY)
    VB = visible(vB, CX, CY) & visible(vA, ax_, ay_)
    out["common_vis"] = (GA & VA, GB & VB)
    # g3 같은 점: A의 점을 B 자세에서 본 것으로 옮긴 격자
    out["same_points"] = (GA, raster(apply_T(Ti, pa)))
    return out


def cmd_pairs(a):
    import torch
    key = a.seq
    sensor = key.split("/")[0]
    spec, xyz, ids, TL = seq_info(key)
    n = len(ids)
    PL = TL[:, :3, 3]
    s = path_length(xyz)
    t0 = time.time()
    ret = base_retrieval(spec, xyz, TL)
    print(f"[{key}] base retrieval {time.time() - t0:.0f}s  raw R@1 {ret['raw'][0]['R@1']:.4f}"
          f" lin16 {ret['lin16'][0]['R@1']:.4f}", flush=True)
    m_raw, ar = ret["raw"]
    m_lin, al = ret["lin16"]
    u, moving = headings(np.asarray(xyz, dtype=np.float64))
    pos = ar["pos"]
    q_all, j_all = ar["q"][pos], ar["jnn"][pos]
    dirs = classify(u, moving, q_all, j_all)
    hit_raw, hit_lin = ar["hit1"][pos], al["hit1"][pos]
    t1_raw, t1_lin = ar["t1"][pos], al["t1"][pos]
    lat, lon = offsets(xyz, PL, q_all, j_all)
    rng = np.random.default_rng(0)
    # 반대 방향: 검색의 과거 최근접 쌍 전부(많으면 고르게 a.n개)
    rev = np.flatnonzero(dirs == 1)
    if len(rev) > a.n:
        rev = rev[np.linspace(0, len(rev) - 1, a.n).astype(int)]
    # 같은 방향(옆 어긋남 맞춤): 5 m 안 모든 양성 쌍(q - j >= 30) 중 같은 방향, 옆 어긋남 0.5 m 칸별로 반대 쪽 개수만큼
    qa, ja, _ = positive_pairs(np.asarray(xyz, dtype=np.float64))
    da = classify(u, moving, qa, ja)
    sq, sj = qa[da == 0], ja[da == 0]
    slat, slon = offsets(xyz, PL, sq, sj)
    okl = np.isfinite(slat)
    sq, sj, slat, slon = sq[okl], sj[okl], slat[okl], slon[okl]
    LB = 0.5
    rb = np.floor(lat[rev] / LB).astype(int)
    sb = np.floor(slat / LB).astype(int)
    same_sel, short = [], 0
    for b in np.unique(rb):
        need = int((rb == b).sum())
        pool = np.flatnonzero(sb == b)
        if len(pool) == 0:
            short += need
            continue
        take = rng.choice(pool, need, replace=len(pool) < need)
        same_sel.extend(take.tolist())
    same_sel = np.array(same_sel, int)
    # 같은 방향(자연): 검색의 과거 최근접 쌍 중 같은 방향
    nat = np.flatnonzero(dirs == 0)
    nat = nat[np.linspace(0, len(nat) - 1, min(a.n_nat, len(nat))).astype(int)]
    pairs = ([("reverse", int(q_all[k]), int(j_all[k]), int(k)) for k in rev]
             + [("same_latmatch", int(sq[k]), int(sj[k]), -1) for k in same_sel]
             + [("same_nat", int(q_all[k]), int(j_all[k]), int(k)) for k in nat])
    print(f"[{key}] pairs: reverse {len(rev)} same_latmatch {len(same_sel)} (bins short {short})"
          f" same_nat {len(nat)}", flush=True)

    sim = Sim()
    sc = Scans(key, ids)
    cache = {}

    def get(i):
        if i not in cache:
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
            cache[i] = prep(sc.get(i))
        return cache[i]

    # 음성 대조: 질의와 50 m 넘게 떨어진 무작위 키프레임(질의별 고정)
    rows = []
    t0 = time.time()
    for pi, (kind, q, j, k) in enumerate(pairs):
        if pi % 100 == 0:
            print(f"  pair {pi}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
        A, B = get(q), get(j)
        Tc = np.linalg.inv(TL[q]) @ TL[j]
        Ti, fit = icp(B["xyz"], A["xyz"], Tc)
        # GT와 ICP 중 비지면 구조 점(수평 3-40 m, h 0.5-4 m)이 0.3 m 안에 더 많이 겹치는 쪽을 쓴다(모든 쌍 같은 규칙)
        f_gt, f_icp = struct_fit(A, B, Tc), struct_fit(A, B, Ti)
        use_icp = f_icp >= f_gt
        Ti = Ti if use_icp else Tc
        dT = Ti @ np.linalg.inv(Tc)
        row = {"kind": kind, "q": q, "j": j, "dist": float(np.linalg.norm(xyz[q] - xyz[j])),
               "icp_fit": fit, "icp_dt": float(np.linalg.norm(dT[:2, 3])),
               "icp_drot": float(np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))),
               "yaw_rel": float(np.degrees(np.arctan2(Ti[1, 0], Ti[0, 0]))),
               "struct_f03_gt": f_gt, "struct_f03_icp": f_icp, "use_icp": bool(use_icp)}
        la, lo = offsets(xyz, PL, np.array([q]), np.array([j]))
        row["lat"], row["lon"] = float(la[0]), float(lo[0])
        if k >= 0:
            row.update(hit_raw=bool(hit_raw[k]), hit_lin=bool(hit_lin[k]), t1_raw=int(t1_raw[k]),
                       t1_lin=int(t1_lin[k]))
        row["ov"] = overlap_stats(A, B, Ti)
        # 왜곡 보정 후 전체 겹침(참고)
        if A["prev"] is not None and B["prev"] is not None:
            Ad = dict(A, xyz=deskew_xyz(A["xyz"], A["time"], A["prev"]))
            Bd = dict(B, xyz=deskew_xyz(B["xyz"], B["time"], B["prev"]))
            Td, _ = icp(Bd["xyz"], Ad["xyz"], Tc)
            fd_gt, fd_icp = struct_fit(Ad, Bd, Tc), struct_fit(Ad, Bd, Td)
            Td = Td if fd_icp >= fd_gt else Tc
            dTd = Td @ np.linalg.inv(Tc)
            row["deskew_struct_f03_gt"], row["deskew_struct_f03_icp"] = fd_gt, fd_icp
            row["deskew_icp_dt"] = float(np.linalg.norm(dTd[:2, 3]))
            o = overlap_stats(Ad, Bd, Td)
            row["ov_deskew_all"] = o["all|all"]
            row["ov_deskew_nonground"] = o["nonground|all"]
        # 공통 영역 상한 (반대 방향과 같은 방향 모두; 반대 방향은 음성 대조도)
        if kind != "same_nat" or pi % 2 == 0:
            g = masked_grids(A, B, Ti)
            row["bound"] = {}
            for gname, (ga, gb) in g.items():
                sa = sim.spec(ga) if gname != "orig" else spec[q]
                sb = sim.spec(gb) if gname != "orig" else spec[j]
                row["bound"][gname] = sim.pair(sa, sb) + (int(ga.sum()), int(gb.sum()))
            if kind == "reverse":
                far = np.flatnonzero(np.linalg.norm(xyz - xyz[q], axis=1) > 50.0)
                nn = int(rng.choice(far))
                Nn = get(nn)
                gN = masked_grids(A, Nn, Ti)
                gN.pop("same_points")       # B를 쓰지 않으므로 음성 대조가 없다
                row["neg"] = {"idx": nn}
                for gname, (ga, gb) in gN.items():
                    sa = sim.spec(ga) if gname != "orig" else spec[q]
                    sb = sim.spec(gb) if gname != "orig" else spec[nn]
                    row["neg"][gname] = sim.pair(sa, sb) + (int(ga.sum()), int(gb.sum()))
                # 질의의 1위 점수(저장 spec, raw와 lin16)
                row["top1_score"] = (sim.pair(spec[q], spec[int(t1_raw[k])])[0],
                                     sim.pair(spec[q], spec[int(t1_lin[k])])[1])
        rows.append(row)
    print(f"[{key}] pairs done {time.time() - t0:.0f}s", flush=True)

    # 틀린 1위의 위치(반대 방향 실패 질의 전부)
    wrong = {}
    for rep, hit, t1 in (("raw", hit_raw, t1_raw), ("lin16", hit_lin, t1_lin)):
        for dname, di in (("reverse", 1), ("same", 0)):
            sel = np.flatnonzero((dirs == di) & ~hit)
            if len(sel) == 0:
                continue
            qq, tt, jj = q_all[sel], t1[sel], j_all[sel]
            d = np.linalg.norm(xyz[qq] - xyz[tt], axis=1)
            dj = np.linalg.norm(xyz[jj] - xyz[tt], axis=1)
            pg = s[qq] - s[tt]
            tdir = classify(u, moving, qq, tt)
            wrong[f"{rep}|{dname}"] = {
                "n": int(len(sel)),
                "dist_to_query_m": {"5-10": int(((d > 5) & (d <= 10)).sum()), "10-20": int(((d > 10) & (d <= 20)).sum()),
                                    "20-50": int(((d > 20) & (d <= 50)).sum()), "50-200": int(((d > 50) & (d <= 200)).sum()),
                                    ">200": int((d > 200).sum()), "median": float(np.median(d))},
                "dist_to_true_match_m_median": float(np.median(dj)),
                "top1_heading_vs_query": {DIRS[c]: int((tdir == c).sum()) for c in range(4)},
                "path_gap_m": {"<100": int((pg < 100).sum()), "100-500": int(((pg >= 100) & (pg < 500)).sum()),
                               ">=500": int((pg >= 500).sum())},
                "top1_is_query_recent_pass(path_gap<100 & dist<100)": int(((pg < 100) & (d < 100)).sum())}
    res = {"seq": key, "n": n, "base": {"raw": m_raw, "lin16": m_lin},
           "dir_n": {DIRS[c]: int((dirs == c).sum()) for c in range(4)},
           "pair_counts": {"reverse": len(rev), "same_latmatch": len(same_sel), "same_nat": len(nat),
                           "latmatch_short": short},
           "wrong_top1": wrong, "rows": rows}
    chmod_write(OUT / f"pairs_{sensor}_{key.split('/')[1]}.json", text=json.dumps(res))
    summarize_pairs(res)
    return 0


def summarize_pairs(res, quiet=False):
    """겹침 표와 공통 영역 상한 요약."""
    rows = res["rows"]
    groups = {"reverse": [r for r in rows if r["kind"] == "reverse"],
              "reverse_fail_raw": [r for r in rows if r["kind"] == "reverse" and not r["hit_raw"]],
              "reverse_hit_raw": [r for r in rows if r["kind"] == "reverse" and r["hit_raw"]],
              "same_latmatch": [r for r in rows if r["kind"] == "same_latmatch"],
              "same_nat": [r for r in rows if r["kind"] == "same_nat"]}
    regions = sorted({k for r in rows for k in r["ov"]}, key=lambda x: (x.split("|")[0], x))
    tab = {}
    for g, rs in groups.items():
        if not rs:
            continue
        e = {"n": len(rs), "lat_med": float(np.median([r["lat"] for r in rs])),
             "icp_fit_med": float(np.median([r["icp_fit"] for r in rs])),
             "icp_dt_med": float(np.median([r["icp_dt"] for r in rs])),
             "use_icp_frac": float(np.mean([r["use_icp"] for r in rs])),
             "struct_f03_gt_med": float(np.median([r["struct_f03_gt"] for r in rs])),
             "struct_f03_icp_med": float(np.median([r["struct_f03_icp"] for r in rs]))}
        if all("deskew_struct_f03_gt" in r for r in rs):
            e["deskew_struct_f03_gt_med"] = float(np.median([r["deskew_struct_f03_gt"] for r in rs]))
            e["deskew_struct_f03_icp_med"] = float(np.median([r["deskew_struct_f03_icp"] for r in rs]))
            e["deskew_icp_dt_med"] = float(np.median([r["deskew_icp_dt"] for r in rs]))
        for reg in regions:
            v = np.array([r["ov"][reg] for r in rs], float)       # (n, 6)
            iou = v[:, 0] / np.maximum(v[:, 1], 1)
            okr = v[:, 1] > 0
            # 허용 겹침: A 칸 중 B 이웃이 있는 비율과 그 반대의 평균
            tol = 0.5 * (v[:, 4] / np.maximum(v[:, 2], 1) + v[:, 5] / np.maximum(v[:, 3], 1))
            okt = (v[:, 2] > 0) & (v[:, 3] > 0)
            e[reg] = {"iou_mean": float(iou[okr].mean()) if okr.any() else None,
                      "iou_pooled": float(v[:, 0].sum() / max(v[:, 1].sum(), 1)),
                      "tol_mean": float(tol[okt].mean()) if okt.any() else None,
                      "nA_mean": float(v[:, 2].mean()), "nB_mean": float(v[:, 3].mean())}
        dsk = [r for r in rs if "ov_deskew_all" in r]
        if dsk:
            for nm_, kk in (("deskew_all", "ov_deskew_all"), ("deskew_nonground", "ov_deskew_nonground")):
                v = np.array([r[kk] for r in dsk], float)
                e[nm_] = {"iou_mean": float((v[:, 0] / np.maximum(v[:, 1], 1)).mean()),
                          "tol_mean": float((0.5 * (v[:, 4] / np.maximum(v[:, 2], 1)
                                                    + v[:, 5] / np.maximum(v[:, 3], 1))).mean())}
        bd = [r for r in rs if "bound" in r]
        if bd:
            e["bound_sim_median"] = {gn: [float(np.median([r["bound"][gn][c] for r in bd])) for c in (0, 1)]
                                     for gn in bd[0]["bound"]}
            e["bound_cells_median"] = {gn: [float(np.median([r["bound"][gn][c] for r in bd])) for c in (2, 3)]
                                       for gn in bd[0]["bound"]}
        ng = [r for r in rs if "neg" in r]
        if ng:
            e["neg_sim_median"] = {gn: [float(np.median([r["neg"][gn][c] for r in ng])) for c in (0, 1)]
                                   for gn in ng[0]["neg"] if gn != "idx"}
            # 실패 질의에서: 정답 쌍(가린 것)과 음성(같은 가림)이 1위 점수를 넘는 비율
            beats = {}
            for ci, rep, hk in ((0, "raw", "hit_raw"), (1, "lin16", "hit_lin")):
                fl = [r for r in ng if not r[hk]]
                if not fl:
                    continue
                for gn in ng[0]["bound"]:
                    beats[f"{rep}|{gn}|pair>top1"] = float(np.mean([r["bound"][gn][ci] > r["top1_score"][ci] for r in fl]))
                    if gn in ng[0]["neg"]:
                        beats[f"{rep}|{gn}|neg>top1"] = float(np.mean([r["neg"][gn][ci] > r["top1_score"][ci] for r in fl]))
                        beats[f"{rep}|{gn}|pair>neg"] = float(np.mean([r["bound"][gn][ci] > r["neg"][gn][ci] for r in fl]))
                beats[f"{rep}|n_fail"] = len(fl)
                beats[f"{rep}|top1_score_median"] = float(np.median([r["top1_score"][ci] for r in fl]))
            e["fail_beats"] = beats
        tab[g] = e
    res["summary"] = tab
    if not quiet:
        for g, e in tab.items():
            print(f"\n== {g} n {e['n']} lat_med {e['lat_med']:.2f} icp_fit {e['icp_fit_med']:.2f} dt {e['icp_dt_med']:.2f}"
                  f" use_icp {e['use_icp_frac']:.2f} struct f03 gt {e['struct_f03_gt_med']:.2f} icp {e['struct_f03_icp_med']:.2f}"
                  f" | deskew gt {e.get('deskew_struct_f03_gt_med', float('nan')):.2f}"
                  f" icp {e.get('deskew_struct_f03_icp_med', float('nan')):.2f} dt {e.get('deskew_icp_dt_med', float('nan')):.2f}")
            for reg in regions:
                if reg.split("|")[1] in ("all", "r1-10", "r10-20", "r20-40", "r40-80") or reg.startswith("all|sec"):
                    x = e[reg]
                    print(f"   {reg:24s} IoU {x['iou_mean'] if x['iou_mean'] is None else round(x['iou_mean'], 3)}"
                          f"  tol {x['tol_mean'] if x['tol_mean'] is None else round(x['tol_mean'], 3)}"
                          f"  nA {x['nA_mean']:.0f} nB {x['nB_mean']:.0f}")
            for k in ("deskew_all", "deskew_nonground", "bound_sim_median", "neg_sim_median", "fail_beats"):
                if k in e:
                    print(f"   {k}: {e[k]}")
    return tab


# ---------------------------------------------------------------- 1단계 그림
def cmd_figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    key = a.seq
    sensor, seq = key.split("/")
    res = json.load(open(OUT / f"pairs_{sensor}_{seq}.json"))
    spec, xyz, ids, TL = seq_info(key)
    del spec
    sc = Scans(key, ids)
    rev = [r for r in res["rows"] if r["kind"] == "reverse"]
    # 고정 규칙: raw 실패/성공 반대 방향 쌍을 질의 번호 순으로 고르게 a.k개씩
    picks = []
    for lab, rs in (("fail", [r for r in rev if not r["hit_raw"]]), ("hit", [r for r in rev if r["hit_raw"]])):
        rs = sorted(rs, key=lambda r: r["q"])
        for k in np.linspace(0, len(rs) - 1, min(a.k, len(rs))).astype(int):
            picks.append((lab, rs[k]))
    cmap = ListedColormap(["white", "#d62728", "#1f77b4", "#2ca02c"])
    for lab, r in picks:
        q, j = r["q"], r["j"]
        A, B = prep(sc.get(q)), prep(sc.get(j))
        Tc = np.linalg.inv(TL[q]) @ TL[j]
        Ti, fit = icp(B["xyz"], A["xyz"], Tc)
        pa = own(A["xyz"]).astype(np.float64)
        mb = np.hypot(B["xyz"][:, 0], B["xyz"][:, 1]) >= 1.0
        pb = apply_T(Ti, B["xyz"][mb].astype(np.float64))
        ha = A["h"][np.hypot(A["xyz"][:, 0], A["xyz"][:, 1]) >= 1.0]
        hb = B["h"][mb]
        fig, ax = plt.subplots(1, 3, figsize=(21, 7.4))
        for c, (gsel, title) in enumerate(((None, "points, all"), ("ng", "points, non-ground (h>0.3)"))):
            sa = slice(None) if gsel is None else ha > GND_H
            sb = slice(None) if gsel is None else hb > GND_H
            ax[c].scatter(pa[sa, 0], pa[sa, 1], s=0.3, c="#d62728", lw=0, label=f"query {q}")
            ax[c].scatter(pb[sb, 0], pb[sb, 1], s=0.3, c="#1f77b4", lw=0, label=f"map {j} (ICP aligned)")
            ax[c].set_xlim(-45, 45)
            ax[c].set_ylim(-45, 45)
            ax[c].set_aspect("equal")
            ax[c].arrow(0, 0, 6, 0, width=0.4, color="k")
            ax[c].set_title(f"{title}, +-45 m (arrow: query forward)")
            if c == 0:
                ax[c].legend(loc="upper right", markerscale=20, fontsize=8)
        ga, gb = raster(pa), raster(pb)
        img = ga.astype(int) + 2 * gb.astype(int)
        ax[2].imshow(img.T, origin="lower", extent=(-80, 80, -80, 80), cmap=cmap, vmin=0, vmax=3,
                     interpolation="nearest")
        iou = (ga & gb).sum() / max((ga | gb).sum(), 1)
        ax[2].set_title(f"1 m occupancy: red query only, blue map only, green both (IoU {iou:.2f})")
        t1 = r["t1_raw"]
        fig.suptitle(f"{key} reverse pair ({lab} raw; lin16 {'hit' if r['hit_lin'] else 'fail'})  q {q} j {j}"
                     f"  dist {r['dist']:.1f} m  lat {r['lat']:.1f} m  ICP fit {fit:.2f}"
                     f"  | raw top1 {t1} at {np.linalg.norm(xyz[t1] - xyz[q]):.0f} m"
                     f"  sim pair {r['bound']['orig'][0]:.3f} top1 {r['top1_score'][0]:.3f}", fontsize=11)
        fig.tight_layout()
        path = OUT / f"fig_rev_{lab}_{seq}_q{q}.png"
        fig.savefig(path, dpi=70)
        plt.close(fig)
        path.chmod(0o666)
        print(f"saved {path}", flush=True)
    # 개관: 궤적과 반대 방향 실패 질의 -> 틀린 1위
    fig, ax = plt.subplots(1, 1, figsize=(11, 11))
    ax.plot(xyz[:, 0], xyz[:, 1], "-", c="0.75", lw=0.8)
    fl = [r for r in rev if not r["hit_raw"]]
    for r in fl:
        q, t1 = r["q"], r["t1_raw"]
        ax.plot([xyz[q, 0], xyz[t1, 0]], [xyz[q, 1], xyz[t1, 1]], "-", c="#d62728", lw=0.3, alpha=0.5)
    ax.scatter([xyz[r["q"], 0] for r in fl], [xyz[r["q"], 1] for r in fl], s=4, c="#d62728", label="reverse query, raw fail")
    ax.scatter([xyz[r["q"], 0] for r in rev if r["hit_raw"]], [xyz[r["q"], 1] for r in rev if r["hit_raw"]], s=4,
               c="#2ca02c", label="reverse query, raw hit")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"{key}: reverse queries; red lines = query -> wrong raw top-1")
    path = OUT / f"fig_overview_{seq}.png"
    fig.savefig(path, dpi=70)
    plt.close(fig)
    path.chmod(0o666)
    print(f"saved {path}", flush=True)
    return 0


# ---------------------------------------------------------------- 2단계: 개입
def cmd_interv(a):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from _scatter_probe_20261114 import occupancy
    torch.backends.cuda.matmul.allow_tf32 = False
    key = a.seq
    sensor, seq = key.split("/")
    spec1, xyz, ids, TL = seq_info(key)
    n = len(ids) if not a.limit else a.limit
    spec1, xyz, ids, TL = spec1[:n], xyz[:n], ids[:n], TL[:n]
    PL = TL[:, :3, 3]
    s = path_length(xyz)
    variants = list(a.variants)
    sim = Sim()
    sc = Scans(key, ids)
    blind_info = None
    if sc.sensor in ("helipr", "kitti360"):
        BLIND["tab"], w_, s_ = blind_table(sc, n)
        blind_info = {"rule": "world(r_xy>=2.5) rate < 0.03 or self(r3<1) rate > 0.1, 5 deg bins",
                      "blind_az_deg_by_ring": [[int(b * 360 / AZB) for b in np.flatnonzero(BLIND["tab"][k])]
                                               for k in range(len(BLIND["tab"]))],
                      "symvis_drop_frac_by_ring": [float((BLIND["tab"][k] | np.roll(BLIND["tab"][k], -AZB // 2)).mean())
                                                   for k in range(len(BLIND["tab"]))]}
        print(f"  blind bins per ring {BLIND['tab'].sum(1).tolist()}", flush=True)
    specs = {v: np.empty((n, 805, 16), np.float16) for v in variants}
    kept = {v: np.zeros(n, np.float32) for v in variants}
    chk = {"base_bitwise_equal": 0, "base_max_rel_err": 0.0, "n_check": 0}
    planes = np.zeros((n, 3))
    pool = ThreadPoolExecutor(6)
    futs = {}
    t0 = time.time()
    for i in range(n):
        for k in range(i, min(n, i + 24)):
            if k not in futs:
                futs[k] = pool.submit(lambda k: prep(sc.get(k)), k)
        scan = futs.pop(i).result()
        planes[i] = scan["plane"]
        for v in variants:
            p = variant_points(scan, v)
            kept[v][i] = len(p) / max(len(scan["xyz"]), 1)
            specs[v][i] = sim.spec(occupancy(p))
            if v == "base":
                ref = spec1[i]
                chk["base_bitwise_equal"] += int(np.array_equal(specs[v][i], ref))
                chk["base_max_rel_err"] = max(chk["base_max_rel_err"], float(
                    np.linalg.norm(specs[v][i].astype(np.float32) - ref.astype(np.float32))
                    / np.linalg.norm(ref.astype(np.float32))))
                chk["n_check"] += 1
        if i % 1000 == 0:
            el = time.time() - t0
            print(f"  [{key}] {i}/{n} {el:.0f}s", flush=True)
    pool.shutdown()
    t_spec = time.time() - t0
    s16, W16 = load_lin16()
    ref = json.load(open(DIAGA))["per_seq"].get(key) if DIAGA.is_file() else None
    res = {"seq": key, "n": n, "variants": variants, "spec_s": round(t_spec, 1), "base_check": chk,
           "kept_point_fraction": {v: float(kept[v].mean()) for v in variants},
           "plane_tilt_deg_median": float(np.degrees(np.median(np.arctan(np.hypot(planes[:, 0], planes[:, 1]))))),
           "blind": blind_info,
           "lin16_note": "s, W from single-scan training applied unchanged to intervened spec", "eval": {}}
    hits = {}
    u_h, mv_h = headings(np.asarray(xyz, dtype=np.float64))
    for rep, (sv, Wv) in (("raw", (None, None)), ("lin16", (s16, W16))):
        res["eval"][rep] = {}
        for v in variants:
            with torch.no_grad():
                D = encode(specs[v], sv, Wv)
                m, arrs = evaluate(D, D, xyz, s, [0.0], 16 if Wv is None else 64)[0.0]
            del D
            torch.cuda.empty_cache()
            m.update(breakdown(xyz, PL, arrs))
            dn, dh = m["dir_n"], m["dir_hit1"]
            m["dir_R@1"] = {d: (dh[d] / dn[d] if dn[d] else None) for d in DIRS[:3]}
            # 1위의 질의 진행 방향 기준 부호 있는 앞뒤 어긋남(음수 = 질의 뒤쪽), 방향별 실패 질의와 전체
            pq, pj, pt, ph = (arrs[k][arrs["pos"]] for k in ("q", "jnn", "t1", "hit1"))
            dcl = classify(u_h, mv_h, pq, pj)
            uq = u_h[pq][:, :2] / np.maximum(np.linalg.norm(u_h[pq][:, :2], axis=1, keepdims=True), 1e-12)
            lt = ((xyz[pt] - xyz[pq])[:, :2] * uq).sum(1)
            dtop = np.linalg.norm(xyz[pt] - xyz[pq], axis=1)
            m["top1_signed_lon"] = {}
            for dname, di in (("same", 0), ("reverse", 1)):
                for gname, gm in (("fail", ~ph), ("all", np.ones_like(ph))):
                    sel = (dcl == di) & gm
                    m["top1_signed_lon"][f"{dname}_{gname}"] = (
                        {"n": int(sel.sum()), "median_m": float(np.median(lt[sel])), "frac_neg": float((lt[sel] < 0).mean()),
                         "frac_top1_within_10m": float((dtop[sel] <= 10.0).mean())}
                        if sel.any() else {"n": 0})
            if v == "base" and ref is not None and rep in ref and not a.limit:
                m["ref_check"] = {k: (m[k], ref[rep][k]) for k in ("R@1", "R@100", "AP", "R@100P")}
                m["ref_ok"] = all(abs(x - y) < 1e-9 for x, y in m["ref_check"].values())
            res["eval"][rep][v] = m
            hits[f"{rep}|{v}"] = np.packbits(arrs["hit1"][arrs["pos"]])
            pc = lambda x: "-" if x is None else f"{100 * x:.1f}"  # noqa: E731
            print(f"  {rep:5s} {v:22s} R@1 {pc(m['R@1'])} R@100 {pc(m['R@100'])} AP {pc(m['AP'])}"
                  f" R@100P {pc(m['R@100P'])} | same {pc(m['dir_R@1']['same'])} rev {pc(m['dir_R@1']['reverse'])}"
                  f" cross {pc(m['dir_R@1']['crossing'])}"
                  f" | rev fail top1 lon {m['top1_signed_lon']['reverse_fail'].get('median_m', float('nan')):.1f} m"
                  f" neg {m['top1_signed_lon']['reverse_fail'].get('frac_neg', float('nan')):.2f}"
                  + (f"  ref_ok {m.get('ref_ok')}" if v == "base" else ""),
                  flush=True)
    sfx = f"_{a.tag}" if a.tag else ""
    sfx += f"_limit{a.limit}" if a.limit else ""
    chmod_write(OUT / f"interv_{sensor}_{seq}{sfx}.json", text=json.dumps(res, indent=1))
    np.savez_compressed(OUT / f"interv_hits_{sensor}_{seq}{sfx}.npz", **{k.replace("|", "__"): v for k, v in hits.items()})
    (OUT / f"interv_hits_{sensor}_{seq}{sfx}.npz").chmod(0o666)
    return 0


# ---------------------------------------------------------------- 모으기
def cmd_report(a):
    out = {"date": "2026-11-15", "script": "scripts/_invkey_helipr_reverse_probe_20261115.py",
           "ground_h_m": GND_H, "sensor": {}, "pairs": {}, "interv": {}}
    for p in sorted(OUT.glob("sensor_*.json")):
        d = json.load(open(p))
        out["sensor"][d["seq"]] = d
    for p in sorted(OUT.glob("pairs_*.json")):
        d = json.load(open(p))
        summ = summarize_pairs(d, quiet=True)
        out["pairs"][d["seq"]] = {k: d[k] for k in ("n", "dir_n", "pair_counts", "wrong_top1")}
        out["pairs"][d["seq"]]["base"] = {r: {k: d["base"][r][k] for k in ("R@1", "R@100", "AP", "R@100P", "n_pos")}
                                          for r in d["base"]}
        out["pairs"][d["seq"]]["summary"] = summ
    for p in sorted(OUT.glob("interv_*.json")):
        if "_limit" in p.name:
            continue
        d = json.load(open(p))
        slim = {}
        for rep, vs in d["eval"].items():
            slim[rep] = {v: {k: m[k] for k in ("R@1", "R@100", "AP", "R@100P", "R@100P_10m", "dir_n", "dir_hit1",
                                               "dir_R@1", "lat_lidar", "top1_signed_lon") if k in m}
                         | ({"ref_ok": m["ref_ok"]} if "ref_ok" in m else {}) for v, m in vs.items()}
        out["interv"].setdefault(d["seq"], {})[p.stem] = {
            "variants": d["variants"], "base_check": d["base_check"], "kept_point_fraction": d["kept_point_fraction"],
            "eval": slim}
    chmod_write(OUT / "metrics.json", text=json.dumps(out, indent=1))
    print(f"metrics.json: sensor {list(out['sensor'])} pairs {list(out['pairs'])} interv "
          f"{ {k: list(v) for k, v in out['interv'].items()} }", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sensor")
    p.add_argument("--seq", required=True)
    p.add_argument("--n", type=int, default=200)
    p = sub.add_parser("pairs")
    p.add_argument("--seq", required=True)
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--n-nat", type=int, default=200)
    p = sub.add_parser("figs")
    p.add_argument("--seq", required=True)
    p.add_argument("--k", type=int, default=7)
    p = sub.add_parser("interv")
    p.add_argument("--seq", required=True)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--limit", type=int, default=0)
    sub.add_parser("report")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"sensor": cmd_sensor, "pairs": cmd_pairs, "figs": cmd_figs, "interv": cmd_interv,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
