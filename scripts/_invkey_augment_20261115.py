#!/usr/bin/env python3
"""invkey 진단 연구용 광선 기준 가림·잘림 증강 (2026-11-15).

스캔의 각 점을 센서 원점에서 나간 광선 하나가 돌아온 것으로 본다.

  (a) 물체 삽입(가림): 지면 위에 방향 있는 상자를 놓고, 광선이 상자에 원래 점보다 먼저 닿으면
      그 점을 상자 표면의 교차점으로 바꾼다. 원래 반환이 없던 방향(하늘 등)은 줄 x 방위 0.4도 칸
      격자에서 빈 칸마다 광선을 만들어 상자에 닿는 것만 점으로 더한다.
        승용차 4.5x1.8x1.5, 승합·SUV 5.0x2.0x2.0, 버스·트럭 12x2.5x3.2 (m, 길이x폭x높이),
        치수마다 x U(0.9, 1.1), 비율 0.60/0.25/0.15.
        무작위 물체: 0-8개(균등), 중심 수평거리 U(2, 30) m, 방위·yaw 균등.
          센서에서 상자 바닥 사각형까지 수평 거리가 1.5 m 미만이면 위치·yaw를 다시 뽑는다.
        바로 옆 큰 차량: 버스·트럭 크기, 긴 면까지 U(2, 4) m, 진행 방향 기준 yaw U(-15, 15)도,
          좌우 무작위, 차량 길이 방향 위치 U(-L/2, L/2) (센서가 긴 면 옆에 오도록).
      지면: 센서 원점 기준 수평 3-10 m 점(50개 미만이면 3-20 m)의 z 5% 분위수에서 시작해
        평면 z = a x + b y + c를 반복 적합(띠 0.3 m -> 0.2 m, 3회)한다. 상자는 상자 중심의 평면 높이에
        바닥을 두고 평면 법선 방향으로 세운다. 기울기 8도 초과이거나 띠 안 점이 30개 미만이면
        5% 분위수 높이의 수평면을 쓴다(info.ground.fallback).
  (b) 거리별 탈락(잘림): 점을 남길 확률 exp(-(r/r0)^2), r은 센서 원점까지 3차원 거리.
  (c) 줄 수 줄이기: 고도 순위로 센 줄 번호에서 ring % f == offset인 줄만 남긴다.
      64줄 f in {2, 4}, 32줄 f = 2, 16줄 적용 안 함. offset은 U{0..f-1}.

  적용 순서: (a) 바꾸기·더하기 -> (c) -> (b). 더한 점도 (c)·(b)를 거친다.

센서별 사실 (데이터로 확인, 2026-11-15; 확인 방법은 `check` 명령)
  kitti, kitti360  HDL-64E. 로더에 줄 번호가 없다. .bin의 점 순서가 줄 단위(위에서 아래)이고 줄마다
                   방위가 한 바퀴 돈다. 첫 점 방위 기준 방위가 180도 넘게 줄어드는 곳을 줄 경계로 센다.
                   고도각 묶기는 쓰지 않는다(한 줄 안 고도각 표준편차 약 0.4-0.5도 > 줄 간격 약 0.35도).
  mulran           OS1-64. 65536점 = 열 1024 x 줄 64, 줄 = 색인 % 64 (0이 맨 위). 반환 없음은 (0,0,0).
  mcd              OS1-64. 65536점, 줄 = 색인 // 1024 (0이 맨 위). 손에 든 센서라 전방 축이 없다.
  nclt             HDL-32E. 로더 좌표의 원점은 센서가 아니라 지면 근처다. 원시 파일의 레이저 번호로
                   레이저별 고도각 분산이 가장 작은 원점을 찾으면 (0, 0, 1.05) m이고 기울기는 0.03도
                   미만이다(3개 세션 18스캔). 그 원점 기준 고도각을 아래 표의 가장 가까운 값에 묶는다.
  helipr           VLP-16. 고도각이 -15..15도 2도 간격 그대로다. 16줄이라 (c)는 적용하지 않는다.
  전방 축: kitti360·mulran·nclt·helipr는 키프레임 사이 이동이 센서 x축 방향(|cos| > 0.9가 100%).
          kitti 자세는 카메라 좌표라 카메라 z = Velodyne x(calib Tr 셋째 행 ≈ (1, 0, 0)).

이름 있는 조건 (평가용 고정 조건 포함)
  all         학습용 전체 조합: 물체 0-8개, 옆 큰 차량 확률 0.25, 탈락 확률 0.5 (r0 U(30, 80)),
              줄 줄이기 확률 0.3
  obj_mix     물체 0-8개만
  near_bus    바로 옆 큰 차량 하나만
  range_r40   r0 = 40 탈락만
  beams_half  줄 1/2만 (16줄 센서는 그대로)

    from _invkey_augment_20261115 import augment
    pts_aug, info = augment(pts, np.random.default_rng(seed), "kitti360", "all")

명령 (컨테이너 ring_baseline, 리포 루트)
    python3 scripts/_invkey_augment_20261115.py check   # 줄·지면·전방 축 점검
    python3 scripts/_invkey_augment_20261115.py figs    # 증강 전후 그림
    python3 scripts/_invkey_augment_20261115.py stats   # 기준 3시퀀스 수치 점검 -> aug/aug_stats.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "results/invkey_study_20261114"
AUG_DIR = STUDY / "aug"

# ---------------------------------------------------------------- 상수
OBJECTS = {"car": (4.5, 1.8, 1.5), "van": (5.0, 2.0, 2.0), "bus": (12.0, 2.5, 3.2)}
OBJ_TYPES = ("car", "van", "bus")
OBJ_P = (0.60, 0.25, 0.15)
SIZE_JITTER = 0.10
MIN_FACE = 1.5                 # 센서에서 상자 바닥 사각형까지 최소 수평 거리 (m)
AZ_BIN = 0.4                   # 반환 없는 광선 격자의 방위 칸 (도)
VALID_R = 0.5                  # 이보다 가까운 점은 반환 없음으로 본다 (MulRan (0,0,0) 등)
MIN_SEG = 20                   # kitti 점 순서로 줄을 셀 때 줄로 인정하는 최소 점 수

NCLT_EL = np.array([-10.88, -9.51, -8.17, -6.82, -5.46, -4.12, -2.77, -1.44, -0.10, 1.23,
                    2.56, 3.91, 5.23, 6.56, 7.90, 9.23, 10.57, 11.89, 13.22, 14.55, 15.88,
                    17.20, 18.55, 19.87, 21.20, 22.52, 23.85, 25.17, 26.50, 27.83, 29.14, 30.46])
SENSORS = {
    # beams, 줄 추정 방식, 광선 원점(로더 좌표), 진행 방향 yaw(도, None이면 없음)
    "kitti": dict(beams=64, ring="order", origin=(0.0, 0.0, 0.0), forward=0.0),
    "kitti360": dict(beams=64, ring="order", origin=(0.0, 0.0, 0.0), forward=0.0),
    "mulran": dict(beams=64, ring="mod64", origin=(0.0, 0.0, 0.0), forward=0.0),
    "mcd": dict(beams=64, ring="div1024", origin=(0.0, 0.0, 0.0), forward=None),
    "nclt": dict(beams=32, ring="table", table=NCLT_EL, origin=(0.0, 0.0, 1.05), forward=0.0),
    "helipr": dict(beams=16, ring="table", table=np.arange(-15.0, 15.1, 2.0),
                   origin=(0.0, 0.0, 0.0), forward=0.0),
}
THIN = {64: (2, 4), 32: (2,), 16: ()}

CONDITIONS = {
    "all": dict(n_obj=(0, 8), p_near=0.25, p_range=0.5, r0=(30.0, 80.0), p_thin=0.3, thin=None),
    "obj_mix": dict(n_obj=(0, 8), p_near=0.0, p_range=0.0, r0=None, p_thin=0.0, thin=None),
    "near_bus": dict(n_obj=(0, 0), p_near=1.0, p_range=0.0, r0=None, p_thin=0.0, thin=None),
    "range_r40": dict(n_obj=(0, 0), p_near=0.0, p_range=1.0, r0=(40.0, 40.0), p_thin=0.0,
                      thin=None),
    "beams_half": dict(n_obj=(0, 0), p_near=0.0, p_range=0.0, r0=None, p_thin=1.0, thin=(2,)),
}


def seed_for(key, scan_id, tag="aug1"):
    """시퀀스 이름·scan_id·태그에서 정해지는 시드."""
    h = hashlib.sha256(f"{tag}|{key}|{int(scan_id)}".encode()).digest()
    return int.from_bytes(h[:8], "little")


# ---------------------------------------------------------------- 줄 추정
def estimate_rings(pts, sensor, p, valid):
    """줄 번호(고도 순위, 0이 맨 위; 모르면 -1)와 줄별 고도각 중앙값(도).

    pts: 로더 점 (N, C), p: 센서 원점 기준 xyz (N, 3), valid: 반환 있는 점.
    """
    cfg = SENSORS[sensor]
    n = len(p)
    ring = np.full(n, -1, np.int64)
    if n == 0:
        return ring, np.zeros(0)
    r = np.linalg.norm(p, axis=1)
    el = np.degrees(np.arcsin(np.clip(p[:, 2] / np.maximum(r, 1e-9), -1, 1)))
    mode = cfg["ring"]
    if mode == "order":
        az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        idx = np.flatnonzero(valid)
        if len(idx):
            a = (az[idx] - az[idx[0]]) % 360.0
            b = np.flatnonzero(np.diff(a) < -180.0) + 1          # 새 줄이 시작하는 위치
            # 이음매 근처 방위 흔들림이 만드는 MIN_SEG점 미만 조각은 다음 줄에 붙인다
            starts = np.concatenate([[0], b])
            ends = np.concatenate([b, [len(idx)]])
            okb = (ends - starts)[:-1] >= MIN_SEG
            if len(b) and ends[-1] - starts[-1] < MIN_SEG:
                okb[-1] = False                                  # 마지막 조각은 앞 줄에 붙인다
            mark = np.zeros(len(idx), np.int64)
            mark[b[okb]] = 1
            ring[idx] = np.cumsum(mark)
    elif mode == "mod64":
        if n == 65536:
            ring[:] = np.arange(n) % 64
    elif mode == "div1024":
        if n == 65536:
            ring[:] = np.arange(n) // 1024
    elif mode == "table":
        t = cfg["table"]
        ring[valid] = np.abs(el[valid, None] - t[None]).argmin(1)
        ring[valid] = len(t) - 1 - ring[valid]          # 0이 맨 위
    nr =int(ring.max()) + 1 if (ring >= 0).any() else 0
    ring_el = np.full(nr, np.nan)
    if nr:
        m = valid & (ring >= 0)
        cnt = np.bincount(ring[m], minlength=nr)
        order = np.argsort(ring[m], kind="stable")
        els = el[m][order]
        starts = np.concatenate([[0], np.cumsum(cnt)])
        for k in range(nr):
            if cnt[k]:
                ring_el[k] = float(np.median(els[starts[k]:starts[k + 1]]))
    return ring, ring_el


# ---------------------------------------------------------------- 지면
def ground_plane(p, valid):
    """센서 원점 기준 좌표의 지면 평면 (a, b, c): z = a x + b y + c."""
    rxy = np.hypot(p[:, 0], p[:, 1])
    m = valid & (rxy >= 3.0) & (rxy <= 10.0)
    window = "3-10"
    if m.sum() < 50:
        m = valid & (rxy >= 3.0) & (rxy <= 20.0)
        window = "3-20"
    out = {"window": window, "n_window": int(m.sum())}
    if m.sum() < 50:
        out.update(ok=False, fallback="no_points")
        return out
    z0 = float(np.percentile(p[m, 2], 5))
    a = b = 0.0
    c = z0
    sel = m & (np.abs(p[:, 2] - z0) < 0.3)
    ok_fit = False
    for _ in range(3):
        if sel.sum() < 30:
            break
        A = np.c_[p[sel, 0], p[sel, 1], np.ones(sel.sum())]
        (a, b, c), *_ = np.linalg.lstsq(A, p[sel, 2], rcond=None)
        res = p[:, 2] - (a * p[:, 0] + b * p[:, 1] + c)
        sel = m & (np.abs(res) < 0.2)
        ok_fit = True
    tilt = float(np.degrees(np.arctan(np.hypot(a, b))))
    out.update(ok=True, z_q05=z0, n_inlier=int(sel.sum()), tilt_deg=tilt)
    if not ok_fit or tilt > 8.0 or sel.sum() < 30:
        out.update(a=0.0, b=0.0, c=z0, fallback="horizontal_q05")
    else:
        out.update(a=float(a), b=float(b), c=float(c), fallback=None)
    return out


# ---------------------------------------------------------------- 상자
def rect_dist(cx, cy, yaw, L, W):
    """원점에서 (cx, cy) 중심, yaw 방향 L x W 사각형까지 수평 거리 (안이면 0)."""
    c, s = np.cos(yaw), np.sin(yaw)
    qx, qy = -(c * cx + s * cy), -(-s * cx + c * cy)
    return float(np.hypot(max(abs(qx) - L / 2, 0.0), max(abs(qy) - W / 2, 0.0)))


def make_box(kind, L, W, H, cx, cy, yaw, g, near=False):
    """바닥 중심을 지면 평면에 두고 평면 법선으로 세운 상자."""
    a, b, c = g["a"], g["b"], g["c"]
    n = np.array([-a, -b, 1.0])
    n /= np.linalg.norm(n)
    h = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    u = h - (h @ n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    c0 = np.array([cx, cy, a * cx + b * cy + c])
    return {"type": kind, "L": float(L), "W": float(W), "H": float(H), "cx": float(cx),
            "cy": float(cy), "cz": float(c0[2]), "yaw_deg": float(np.degrees(yaw) % 360),
            "near": bool(near), "gap": rect_dist(cx, cy, yaw, L, W),
            "_R": np.stack([u, v, n], 1), "_c0": c0}


def corners(box):
    R, c0 = box["_R"], box["_c0"]
    L, W, H = box["L"], box["W"], box["H"]
    q = np.array([[sx * L / 2, sy * W / 2, sz * H] for sx in (-1, 1) for sy in (-1, 1)
                  for sz in (0, 1)])
    return c0 + q @ R.T


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def cast(D, az, boxes):
    """단위 방향 D (N, 3) 광선(원점 0)이 상자에 처음 닿는 거리 (N,) (없으면 inf)와 상자 번호.

    az: D의 방위(라디안). 상자 모서리 방위 범위 밖의 광선은 계산하지 않는다(센서는 상자 밖).
    """
    t = np.full(len(D), np.inf)
    who = np.full(len(D), -1, np.int64)
    for k, box in enumerate(boxes):
        cr = corners(box)
        a_c = np.arctan2(box["cy"], box["cx"])
        span = np.abs(_wrap(np.arctan2(cr[:, 1], cr[:, 0]) - a_c)).max() + 1e-3
        cand = np.flatnonzero(np.abs(_wrap(az - a_c)) <= span)
        if not len(cand):
            continue
        R, c0 = box["_R"], box["_c0"]
        ob = -c0 @ R                                   # 원점의 상자 좌표
        db = D[cand] @ R
        lo = np.array([-box["L"] / 2, -box["W"] / 2, 0.0])
        hi = np.array([box["L"] / 2, box["W"] / 2, box["H"]])
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / db
            t1, t2 = (lo - ob) * inv, (hi - ob) * inv
            tmin = np.minimum(t1, t2).max(1)
            tmax = np.maximum(t1, t2).min(1)
        hit = (tmax >= tmin) & (tmin > 0)
        better = hit & (tmin < t[cand])
        t[cand[better]] = tmin[better]
        who[cand[better]] = k
    return t, who


def sample_boxes(rng, cfg, n_obj, near, g):
    boxes = []
    for _ in range(n_obj):
        kind = OBJ_TYPES[int(rng.choice(3, p=OBJ_P))]
        L, W, H = np.asarray(OBJECTS[kind]) * rng.uniform(1 - SIZE_JITTER, 1 + SIZE_JITTER, 3)
        for _ in range(100):
            rad, az, yaw = rng.uniform(2.0, 30.0), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
            cx, cy = rad * np.cos(az), rad * np.sin(az)
            if rect_dist(cx, cy, yaw, L, W) >= MIN_FACE:
                boxes.append(make_box(kind, L, W, H, cx, cy, yaw, g))
                break
    if near:
        L, W, H = np.asarray(OBJECTS["bus"]) * rng.uniform(1 - SIZE_JITTER, 1 + SIZE_JITTER, 3)
        gap = rng.uniform(2.0, 4.0)
        side = 1.0 if rng.random() < 0.5 else -1.0
        if cfg["forward"] is None:
            yaw = rng.uniform(0, 2 * np.pi)
        else:
            yaw = np.radians(cfg["forward"] + rng.uniform(-15.0, 15.0))
        along = rng.uniform(-L / 2, L / 2)
        lat = side * (gap + W / 2)
        cx = along * np.cos(yaw) - lat * np.sin(yaw)
        cy = along * np.sin(yaw) + lat * np.cos(yaw)
        boxes.append(make_box("bus", L, W, H, cx, cy, yaw, g, near=True))
    return boxes


# ---------------------------------------------------------------- 증강
def augment(pts, rng, sensor, spec=None):
    """pts (N, C) 로더 점 -> (증강한 점 (M, C), info). 무작위는 rng에서만 나온다.

    spec: CONDITIONS의 이름 또는 같은 형식의 dict. None이면 "all".
    """
    name = "all" if spec is None else (spec if isinstance(spec, str) else "custom")
    S = CONDITIONS[spec] if isinstance(spec, str) else (CONDITIONS["all"] if spec is None else spec)
    cfg = SENSORS[sensor]
    beams = cfg["beams"]
    o = np.asarray(cfg["origin"], np.float64)

    # 결정은 늘 같은 순서로 뽑는다
    n_obj = int(rng.integers(S["n_obj"][0], S["n_obj"][1] + 1))
    near = bool(rng.random() < S["p_near"])
    use_range = bool(rng.random() < S["p_range"])
    r0 = float(rng.uniform(*S["r0"])) if use_range else None
    use_thin = bool(rng.random() < S["p_thin"])
    choices = THIN[beams] if S["thin"] is None else tuple(f for f in S["thin"] if THIN[beams])
    thin = int(choices[int(rng.integers(len(choices)))]) if (use_thin and choices) else 1
    thin_off = int(rng.integers(thin)) if thin > 1 else 0

    pts = np.asarray(pts)
    p = pts[:, :3].astype(np.float64) - o
    r = np.linalg.norm(p, axis=1)
    valid = r > VALID_R
    n_valid = int(valid.sum())
    ring, ring_el = estimate_rings(pts, sensor, p, valid)
    info = {"sensor": sensor, "cond": name, "n_in": int(len(pts)), "n_valid": n_valid,
            "n_rings": int(np.isfinite(ring_el).sum()), "n_obj": n_obj, "near_bus": near,
            "r0": r0, "thin": thin, "thin_offset": thin_off, "objects": [], "ground": None}

    newp = p.copy()
    rr = r.copy()
    replaced = np.zeros(len(p), bool)
    add_p = np.zeros((0, 3))
    add_ring = np.zeros(0, np.int64)
    boxes = []
    if n_obj or near:
        g = ground_plane(p, valid)
        info["ground"] = g
        if g["ok"]:
            boxes = sample_boxes(rng, cfg, n_obj, near, g)
    if boxes:
        vi = np.flatnonzero(valid)
        D = p[vi] / r[vi, None]
        az = np.arctan2(D[:, 1], D[:, 0])
        t, who = cast(D, az, boxes)
        rep = t < r[vi] - 1e-6
        replaced[vi[rep]] = True
        newp[vi[rep]] = D[rep] * t[rep, None]
        rr[vi[rep]] = t[rep]
        # 반환 없는 광선: 줄 x 방위 칸 격자의 빈 칸
        nr = len(ring_el)
        nb = int(round(360.0 / AZ_BIN))
        if nr:
            occ = np.zeros((nr, nb), bool)
            m = valid & (ring >= 0)
            ab = (np.floor((np.degrees(np.arctan2(p[m, 1], p[m, 0])) % 360.0) / AZ_BIN)
                  .astype(np.int64) % nb)
            occ[ring[m], ab] = True
            okr = np.isfinite(ring_el)
            occ[~okr] = True                           # 고도각을 모르는 줄은 건너뛴다
            ek, bk = np.nonzero(~occ)
            if len(ek):
                e = np.radians(ring_el[ek])
                a = np.radians((bk + 0.5) * AZ_BIN)
                Ds = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], 1)
                ts, _ = cast(Ds, _wrap(a), boxes)
                h = np.isfinite(ts)
                add_p = Ds[h] * ts[h, None]
                add_ring = ek[h]
        for bx in boxes:
            info["objects"].append({k: v for k, v in bx.items() if not k.startswith("_")})
        info["n_hit_by_box"] = np.bincount(who[who >= 0], minlength=len(boxes)).tolist()

    # (c) 줄 줄이기 -> (b) 거리 탈락
    keep = np.ones(len(p), bool)
    keep_add = np.ones(len(add_p), bool)
    n_thin = n_range = 0
    if thin > 1:
        if (ring[valid] >= 0).all() and info["n_rings"]:
            k_ring = (ring % thin) == thin_off
            keep &= ~valid | k_ring
            keep_add &= (add_ring % thin) == thin_off
            n_thin = int((valid & ~keep).sum())
        else:                                          # 줄 번호를 모름 (예: MulRan 65536점 아닌 파일)
            info["thin_skipped"] = thin
            info["thin"] = 1
    if r0 is not None:
        pk = np.exp(-(rr / r0) ** 2)
        u = rng.random(len(p))
        drop = valid & keep & (u >= pk)
        keep &= ~drop
        n_range = int(drop.sum())
        if len(add_p):
            ua = rng.random(len(add_p))
            keep_add &= ua < np.exp(-(np.linalg.norm(add_p, axis=1) / r0) ** 2)

    out = pts.copy()
    out[:, :3] = (newp + o).astype(pts.dtype)
    out = out[keep]
    if keep_add.any():
        extra = np.zeros((int(keep_add.sum()), pts.shape[1]), pts.dtype)
        extra[:, :3] = (add_p[keep_add] + o).astype(pts.dtype)
        out = np.concatenate([out, extra])
    removed = valid & ~keep
    nv = max(n_valid, 1)
    info.update(
        n_replaced=int((replaced & keep).sum()), n_removed_thin=n_thin, n_removed_range=n_range,
        n_added=int(keep_add.sum()), n_added_before_drop=int(len(add_p)),
        frac_replaced=float((replaced & keep).sum() / nv),
        frac_removed=float(removed.sum() / nv),
        frac_affected=float((replaced | removed)[valid].sum() / nv),
        frac_added=float(keep_add.sum() / nv))
    return out, info


# ================================================================ 명령: 점검·그림·수치
def _setup_paths():
    for s in ("", "src", "scripts"):
        sys.path.insert(0, str(REPO / s) if s else str(REPO))


def _survey():
    d = json.load(open(STUDY / "data_survey.json"))
    return {s["key"]: s for s in d["sequences"]}


def _loader(key):
    from _pd_build_test_caches import _make_loader
    s, q = key.split("/")
    return _make_loader(s, q)


def _clean(key):
    s, q = key.split("/")
    with np.load(Path("/workspace/data/invkey_study_20261114/features") / f"{s}_{q}.npz") as z:
        return {k: z[k] for k in ("spec", "scan_ids", "xyz")}


def cmd_check(a):
    """줄 수, 지면 추정, 전방 축을 시퀀스마다 점검한다."""
    _setup_paths()
    by = _survey()
    m = json.load(open(STUDY / "linear/metrics.json"))
    keys = a.seqs or (m["train"] + ["kitti360/0004", "helipr/Roundabout01", "mulran/Sejong03",
                                    "mcd/kth_day_06", "mcd/tuhh_day_02"])
    out = {}
    for key in keys:
        sensor = key.split("/")[0]
        rec = by[key]
        z = np.load(REPO / rec["cache"])
        ids, P = z["scan_ids"], np.asarray(z[rec["pose_field"]], np.float64)
        dt = P[1:, :3, 3] - P[:-1, :3, 3]
        loc = np.einsum("nji,nj->ni", P[:-1, :3, :3], dt)
        nm = np.linalg.norm(loc, axis=1)
        u = loc[nm > 0.5] / nm[nm > 0.5, None]
        fwd = {ax: float((np.abs(u[:, i]) > 0.9).mean()) for i, ax in enumerate("xyz")}
        L = _loader(key)
        sel = ids[np.linspace(0, len(ids) - 1, a.n).astype(int)]
        nr, gz, tilt, fb, nwin = [], [], [], [], []
        for sid in sel:
            pts = L[int(sid)]["points"]
            o = np.asarray(SENSORS[sensor]["origin"])
            p = pts[:, :3].astype(np.float64) - o
            valid = np.linalg.norm(p, axis=1) > VALID_R
            ring, rel = estimate_rings(pts, sensor, p, valid)
            nr.append(int(np.isfinite(rel).sum()))
            g = ground_plane(p, valid)
            nwin.append(g["n_window"])
            fb.append(g.get("fallback"))
            if g["ok"]:
                gz.append(g["c"])
                tilt.append(g["tilt_deg"])
        gz, tilt = np.array(gz), np.array(tilt)
        # 이웃 키프레임 대비 지면 높이 변화 (센서 높이는 그대로이므로 큰 변화는 추정 불안정 또는 지형)
        jump = np.abs(np.diff(gz)) if len(gz) > 1 else np.zeros(0)
        out[key] = {
            "n_scans": len(sel), "forward_frac_abs_cos_gt_0.9": fwd,
            "rings": {"expected": SENSORS[sensor]["beams"], "min": int(min(nr)),
                      "median": float(np.median(nr)), "max": int(max(nr)),
                      "frac_eq_expected": float(np.mean(np.array(nr) == SENSORS[sensor]["beams"]))},
            "ground_c": {"median": float(np.median(gz)), "p05": float(np.percentile(gz, 5)),
                         "p95": float(np.percentile(gz, 95)),
                         "frac_dev_gt_0.3_from_median": float(np.mean(np.abs(gz - np.median(gz)) > 0.3))},
            "tilt_deg": {"median": float(np.median(tilt)), "p90": float(np.percentile(tilt, 90)),
                         "max": float(tilt.max())},
            "fallback": {str(k): int(sum(1 for f in fb if f == k)) for k in set(fb)},
            "n_window_min": int(min(nwin)),
            "neighbor_jump_gt_0.3": float(np.mean(jump > 0.3)) if len(jump) else None,
        }
        print(f"{key:22s} fwd {fwd} rings {out[key]['rings']} ground {out[key]['ground_c']} "
              f"tilt {out[key]['tilt_deg']} fb {out[key]['fallback']} nwin_min {min(nwin)}",
              flush=True)
    path = AUG_DIR / (a.out or "aug_check.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


# ---------------------------------------------------------------- 그림
def _draw(key, sid, pts, aug, info, cond, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sensor = key.split("/")[0]
    o = np.asarray(SENSORS[sensor]["origin"])
    p0 = pts[:, :3] - o
    p1 = aug[:, :3] - o
    v0 = np.linalg.norm(p0, axis=1) > VALID_R
    v1 = np.linalg.norm(p1, axis=1) > VALID_R
    p0, p1 = p0[v0], p1[v1]
    objs = info["objects"]
    # 상자 표면 점(바뀌었거나 더한 점): 어느 상자 안쪽 0.05 m 이내
    onbox = np.zeros(len(p1), bool)
    for b in objs:
        box = make_box(b["type"], b["L"], b["W"], b["H"], b["cx"], b["cy"], np.radians(b["yaw_deg"]),
                       {"a": info["ground"]["a"], "b": info["ground"]["b"], "c": info["ground"]["c"]})
        q = (p1 - box["_c0"]) @ box["_R"]
        onbox |= ((np.abs(q[:, 0]) <= b["L"] / 2 + 0.05) & (np.abs(q[:, 1]) <= b["W"] / 2 + 0.05)
                  & (q[:, 2] >= -0.05) & (q[:, 2] <= b["H"] + 0.05))
    lim = 25.0 if cond == "near_bus" else 40.0
    fig, ax = plt.subplots(2, 2, figsize=(13, 11.5))
    for j, (P, title) in enumerate(((p0, "before"), (p1, "after"))):
        A = ax[0, j]
        m = (np.abs(P[:, 0]) < lim) & (np.abs(P[:, 1]) < lim)
        A.scatter(P[m, 0], P[m, 1], s=0.3, c=P[m, 2], cmap="viridis", vmin=-3, vmax=3, lw=0)
        if j == 1:
            mb = m & onbox
            A.scatter(P[mb, 0], P[mb, 1], s=1.5, c="red", lw=0)
        for b in objs:
            box = make_box(b["type"], b["L"], b["W"], b["H"], b["cx"], b["cy"],
                           np.radians(b["yaw_deg"]), info["ground"])
            cr = corners(box)[[0, 2, 6, 4, 0]]
            A.plot(cr[:, 0], cr[:, 1], color="red" if j else "orange", lw=0.8)
        A.plot(0, 0, "k^", ms=6)
        A.set_xlim(-lim, lim)
        A.set_ylim(-lim, lim)
        A.set_aspect("equal")
        A.set_title(f"top view, {title}  ({len(P)} pts)")
        A.set_xlabel("x (m)")
        A.set_ylabel("y (m)")
    # 옆에서 본 그림: 가장 큰(또는 옆 차량) 상자 방향으로 돌려 그 상자 폭 띠 안의 점
    tgt = next((b for b in objs if b["near"]), None) or max(objs, key=lambda b: b["L"] * b["H"])
    ang = np.arctan2(tgt["cy"], tgt["cx"])
    c, s = np.cos(-ang), np.sin(-ang)
    half = max(tgt["L"], tgt["W"]) / 2 + 1.5
    for j, P in enumerate((p0, p1)):
        A = ax[1, j]
        X, Y = c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1]
        m = (np.abs(Y) < half) & (X > -2) & (X < 60)
        A.scatter(X[m], P[m, 2], s=0.6, c="gray", lw=0)
        if j == 1:
            mb = m & onbox
            A.scatter(X[mb], P[mb, 2], s=2.0, c="red", lw=0)
        g = info["ground"]
        xs = np.linspace(0, 60, 50)
        gx, gy = xs * np.cos(ang), xs * np.sin(ang)
        A.plot(xs, g["a"] * gx + g["b"] * gy + g["c"], "b--", lw=0.8, label="ground plane")
        A.plot(0, 0, "k^", ms=6)
        d = np.hypot(tgt["cx"], tgt["cy"])
        A.axvline(d, color="orange", lw=0.6, ls=":")
        A.set_xlim(-2, 60)
        A.set_ylim(-4, 8)
        A.set_title(f"side view toward {'near bus' if tgt['near'] else tgt['type']} "
                    f"(center {d:.1f} m, strip |y| < {half:.1f} m), {'after' if j else 'before'}")
        A.set_xlabel("distance along box direction (m)")
        A.set_ylabel("z (m, sensor origin)")
        A.legend(loc="upper right", fontsize=8)
    ob = ", ".join(f"{b['type']}{'(near)' if b['near'] else ''} {np.hypot(b['cx'], b['cy']):.0f}m"
                   for b in objs)
    fig.suptitle(f"{key} scan {sid}  cond {cond}  r0 {info['r0']}  thin 1/{info['thin']}  "
                 f"replaced {100 * info['frac_replaced']:.1f}%  removed {100 * info['frac_removed']:.1f}%"
                 f"  added {100 * info['frac_added']:.1f}%\nground c {info['ground']['c']:.2f} m tilt "
                 f"{info['ground']['tilt_deg']:.1f} deg  objects: {ob}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cmd_figs(a):
    _setup_paths()
    by = _survey()
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    keys = a.seqs or ["kitti360/0004", "helipr/Roundabout01", "mulran/Sejong03", "nclt/2012-01-08",
                      "kitti/05"]
    for key in keys:
        rec = by[key]
        z = np.load(REPO / rec["cache"])
        ids = z["scan_ids"]
        L = _loader(key)
        sensor = key.split("/")[0]
        for n, i in enumerate(np.linspace(len(ids) * 0.2, len(ids) * 0.8, a.k).astype(int)):
            sid = int(ids[i])
            pts = L[sid]["points"]
            for cond in (("near_bus", "all") if n % 2 == 0 else ("obj_mix", "all")):
                for t in range(20):        # 물체가 있는 사례가 나올 때까지 시드를 바꾼다
                    aug, info = augment(pts, np.random.default_rng(seed_for(key, sid, f"fig{cond}{t}")),
                                        sensor, cond)
                    if info["objects"]:
                        break
                if not info["objects"]:
                    continue
                path = AUG_DIR / f"fig_{sensor}_{key.split('/')[1]}_{sid}_{cond}.png"
                _draw(key, sid, pts, aug, info, cond, path)
                print(f"  {path.name}: objs {len(info['objects'])} replaced "
                      f"{info['frac_replaced']:.3f} added {info['frac_added']:.3f} ground "
                      f"{info['ground']['c']:.2f} tilt {info['ground']['tilt_deg']:.1f}", flush=True)


# ---------------------------------------------------------------- 수치 점검
REFS = ("kitti360/0004", "helipr/Roundabout01", "mulran/Sejong03")
AUG_CONDS = ("obj_mix", "near_bus", "range_r40", "beams_half", "all")
DEFORM_CONDS = ("occl_30", "occl_90", "crop_50")


def _q(x):
    x = np.asarray(x, np.float64)
    return {"n": int(len(x)), "mean": float(x.mean()), "p05": float(np.percentile(x, 5)),
            "p25": float(np.percentile(x, 25)), "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)), "p95": float(np.percentile(x, 95))}


def cmd_stats(a):
    _setup_paths()
    import time
    import torch
    from _fullmap_probe_20261114 import deform
    from _poolalt_probe_20261114 import NFFT, aligned_sim, describe, to_freq
    from _scatter_probe_20261114 import bank, occupancy
    F = torch.from_numpy(bank()).to("cuda")
    res = {"date": "2026-11-15", "n_per_seq": a.n, "seqs": list(REFS),
           "note": "sim = aligned cosine (to_freq, aligned_sim, nfft 128 of _poolalt_probe_20261114) "
                   "between stored clean float16 spec and float32 spec of the changed scan. "
                   "frac_affected = share of valid returns (range > 0.5 m) replaced or removed. "
                   "ref_adjacent: next keyframe's clean spec; ref_random: a keyframe > 20 m away.",
           "per_seq": {}, "pooled": {}}
    pooled = {}
    for key in REFS:
        sensor = key.split("/")[0]
        cl = _clean(key)
        ids, xyz, spec = cl["scan_ids"], cl["xyz"], cl["spec"]

        def row(k):
            return to_freq(spec[k:k + 1].astype(np.float32))
        L = _loader(key)
        pick = np.linspace(0, len(ids) - 2, a.n).astype(int)
        rng0 = np.random.default_rng(0)
        rows = {c: {"sim": [], "frac_affected": [], "frac_replaced": [], "frac_removed": [],
                    "frac_added": []} for c in AUG_CONDS + DEFORM_CONDS}
        rows["recomputed_clean"] = {"sim": []}
        rows["ref_adjacent"] = {"sim": []}
        rows["ref_random"] = {"sim": []}
        t0 = time.time()
        aug_time = []
        for n, i in enumerate(pick):
            sid = int(ids[i])
            pts = L[sid]["points"]
            Fi, ni = row(i)

            def sim(pp):
                _, s = describe(occupancy(pp), F)
                Fq, nq = to_freq(s[None])
                return float(aligned_sim(Fq, nq, Fi, ni, NFFT["spec"]))
            rows["recomputed_clean"]["sim"].append(sim(pts))
            for c in AUG_CONDS:
                ta = time.time()
                aug, info = augment(pts, np.random.default_rng(seed_for(key, sid, f"stats_{c}")),
                                    sensor, c)
                aug_time.append(time.time() - ta)
                rows[c]["sim"].append(sim(aug))
                for k in ("frac_affected", "frac_replaced", "frac_removed", "frac_added"):
                    rows[c][k].append(info[k])
            valid = np.linalg.norm(pts[:, :3] - np.asarray(SENSORS[sensor]["origin"]), axis=1) > VALID_R
            nv = max(int(valid.sum()), 1)
            for c in DEFORM_CONDS:
                d = deform(pts[valid], c, np.random.default_rng(seed_for(key, sid, f"stats_{c}")))
                rows[c]["sim"].append(sim(d))
                fr = 1.0 - len(d) / nv
                rows[c]["frac_affected"].append(fr)
                rows[c]["frac_removed"].append(fr)
                rows[c]["frac_replaced"].append(0.0)
                rows[c]["frac_added"].append(0.0)
            rows["ref_adjacent"]["sim"].append(float(aligned_sim(*row(i + 1), Fi, ni, NFFT["spec"])))
            far = np.flatnonzero(np.linalg.norm(xyz - xyz[i], axis=1) > 20.0)
            k = int(rng0.choice(far))
            rows["ref_random"]["sim"].append(float(aligned_sim(*row(k), Fi, ni, NFFT["spec"])))
            if n % 50 == 0:
                print(f"  [{key}] {n}/{len(pick)} {time.time() - t0:.0f}s", flush=True)
        res["per_seq"][key] = {c: {k: _q(v) for k, v in d.items()} for c, d in rows.items()}
        res["per_seq"][key]["augment_time_s"] = _q(aug_time)
        for c, d in rows.items():
            for k, v in d.items():
                pooled.setdefault(c, {}).setdefault(k, []).extend(v)
        print(f"== {key}")
        for c, d in res["per_seq"][key].items():
            if c == "augment_time_s":
                continue
            print(f"  {c:16s} sim med {d['sim']['median']:.4f} [p05 {d['sim']['p05']:.4f}, p95 "
                  f"{d['sim']['p95']:.4f}]" + (f"  affected med {d['frac_affected']['median']:.3f} "
                                               f"[p05 {d['frac_affected']['p05']:.3f}, p95 "
                                               f"{d['frac_affected']['p95']:.3f}]"
                                               if "frac_affected" in d else ""), flush=True)
    res["pooled"] = {c: {k: _q(v) for k, v in d.items()} for c, d in pooled.items()}
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    (AUG_DIR / "aug_stats.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {AUG_DIR / 'aug_stats.json'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--seqs", nargs="*")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out")
    p = sub.add_parser("figs")
    p.add_argument("--seqs", nargs="*")
    p.add_argument("--k", type=int, default=2)
    p = sub.add_parser("stats")
    p.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    return {"check": cmd_check, "figs": cmd_figs, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
