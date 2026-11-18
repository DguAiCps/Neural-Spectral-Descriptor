#!/usr/bin/env python3
"""B6 점 단위 교란 평가(학습 없음): 차이 K/V(NSD) 대 절대 K/V(abskv).

원 점군(컨테이너 /workspace/data, 읽기만)을 스캔마다 흔들고 평가가 읽는 두 입력을 다시 만든다:
  디스크립터(288D)·spectral_entropies : encoder.encode_points(points, return_entropy=True)
      encoder = _pd_build_test_caches._make_desc_encoder(cfg, sensor, "cuda"), cfg = _effective_config()
      (NCLT는 _nclt_fov_fix.fixed_config(): 시험 캐시 data/nclt_fov/odom이 이 설정으로 다시 만든 것).
      cache key(compute_cache_key)가 시험 캐시 파일 이름의 키와 같은지 검사한다.
      process_dataset(=_pdv2_c3.cmd_caches의 _build_operating_cache)이 키프레임마다 부르는 것과 같은 호출이다.
  bevclip 스펙트럼 sk : _bevclip_spec과 같은 식 sk = phase_sketch(pool_rows(max(0, BP.project(points)), 16), 12),
      저장 규약대로 실수·허수부를 float32로 둔다(평가는 앞 8주파수를 complex64로 읽음).
교란 위치: 로더 __getitem__이 돌려주는 points((N,4) x,y,z,intensity, 센서 좌표)에 행 마스크를 건다. 두 입력이
같은 교란 점군에서 나온다. 키프레임은 기존 캐시 scan_ids로 고정, poses(KISS-ICP)·timestamps·poses_gt 등 다른 배열은
기존 캐시 그대로(한계: 점 교란이 오도메트리·키프레임 선택에는 반영되지 않는다. 오도메트리 열화는 B1에서 따로 봄).
그 뒤 travel·bank·후보 풀·그래프는 _kveval_subsample_20261118(B7)의 메모리 경로로 다시 만들어 평가한다.

-------------------------------------------------------------------- 조건(결과 보기 전에 고정)
  b6_drop50  스캔마다 점을 독립 50% 확률로 버린다.
  b6_range50 3차원 거리 <= 40 m 점만 남긴다(디스크립터 거리 영상 max_range 80 m의 50%; BEV 투영도 max_range 80).
  b6_beam2   빔을 2개 중 1개만: HeLiPR(VLP-16)·NCLT(HDL-32E)는 원 파일의 링 번호(HeLiPR ring, NCLT laser id)를
             스캔 안 링별 고도각 중앙값으로 순위 매겨 짝수 순위만 남긴다(NCLT laser id는 고도 순서가 아니라서 순위로;
             HeLiPR ring의 고도 단조 여부는 진단에 기록). 링 번호가 없는 KITTI·KITTI-360(HDL-64E)·MulRan·MCD(OS1-64)는
             고도각 atan2(z, r_xy)를 센서 위아래 범위(인코더 sensor_elevation_ranges: KITTI [-24.8, 2.0],
             MulRan [-16.6, 16.6], MCD [-22.5, 22.5])에서 64칸 균등 양자화해 짝수 칸만 남긴다.
  b6_occ90   스캔마다 방위 90도 부채꼴(중심 방위 U[0, 360) 난수)의 점을 지운다.
  b6_none    검산용(교란 없음): 기존 캐시·스펙트럼과 같아야 하고 평가가 perturb/base와 같아야 한다.
난수: default_rng([crc32(seq), 20261118, 6, 조건 번호, scan_id]) — 두 모델·main/ref가 같은 교란 캐시를 쓴다.

-------------------------------------------------------------------- 실행(컨테이너 안)
  ... build --cond b6_drop50 --sequence kitti/07      중간 캐시 tmp/b6_<조건>/<seq>.npz
  ... eval  --cond b6_drop50 --sequence kitti/07 --models all12   (중간 캐시가 없으면 먼저 build)
  ... campaign --cond b6_drop50 --models all12 --workers 2
  집계: python3 scripts/_kveval_subsample_20261118.py summary  (perturb/summary3.json)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _kveval_perturb_20261118 as H  # noqa: E402
import _kveval_subsample_20261118 as B7  # noqa: E402

OUT = H.OUT
ROOT = H.ROOT
TMP = ROOT / "results/kv_eval_conditions_20261118/tmp"
CODE = {"b6_none": 0, "b6_drop50": 1, "b6_range50": 2, "b6_beam2": 3, "b6_occ90": 4}
DESC = {"b6_drop50": "B6 점 무작위 50% 버림(스캔마다 독립)",
        "b6_range50": "B6 3차원 거리 <= 40 m(디스크립터 max_range 80 m의 50%) 점만",
        "b6_beam2": "B6 빔 2개 중 1개(HeLiPR·NCLT 링 번호 고도 순위 짝수, 그 밖은 고도각 64칸 양자화 짝수 칸)",
        "b6_occ90": "B6 방위 90도 부채꼴 가림(중심 방위 스캔마다 균등 난수)",
        "b6_none": "B6 검산: 교란 없이 같은 경로로 다시 만든 캐시"}
MAX_RANGE = 80.0
BEAMS = {"kitti": 64, "kitti360": 64, "mulran": 64, "mcd": 64}
RING_SENSORS = ("helipr", "nclt")


# ----------------------------------------------------------------- 로더·인코더
def encoder_and_config(sensor: str, cache_path: Path):
    from _pd_build_test_caches import _effective_config, _make_desc_encoder, _family
    from train_multi_dataset import compute_cache_key, get_elevation_range
    if sensor == "nclt":
        from _nclt_fov_fix import fixed_config
        cfg = fixed_config()
    else:
        cfg = _effective_config()
    key = compute_cache_key(cfg, dataset_type=_family(sensor))
    assert f"cache_{key}_" in cache_path.name, (key, cache_path.name)
    return _make_desc_encoder(cfg, sensor, "cuda"), tuple(get_elevation_range(cfg, _family(sensor))), key


def ring_of(sensor: str, loader, sid: int, points: np.ndarray) -> np.ndarray:
    """원 파일에서 링 번호를 다시 읽어 loader 점과 행 정렬을 검사한다."""
    if sensor == "helipr":
        dt = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32), ("intensity", np.float32),
                       ("ring", np.uint16), ("time", np.float32)])
        d = np.fromfile(loader.scan_files[sid], dtype=dt)
        xyz = np.stack([d["x"], d["y"], d["z"]], -1).astype(np.float32)
        ring = d["ring"].astype(np.int64)
    else:  # nclt: 로더와 같은 변환·유효 마스크
        dt = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("intensity", "u1"), ("laser", "u1")])
        d = np.fromfile(loader.frame_files[sid], dtype=dt)
        x = d["x"].astype(np.float32) * 0.005 - 100.0
        y = d["y"].astype(np.float32) * 0.005 - 100.0
        z = -(d["z"].astype(np.float32) * 0.005 - 100.0)
        xyz = np.column_stack([x, y, z])
        ok = np.isfinite(xyz).all(1) & (np.abs(x) < 200.0) & (np.abs(y) < 200.0) & (np.abs(z) < 200.0)
        xyz, ring = xyz[ok], d["laser"][ok].astype(np.int64)
    assert np.array_equal(xyz, points[:, :3]), (sensor, sid, "링 정렬 불일치")
    return ring


def elevation(points: np.ndarray) -> np.ndarray:
    p = points[:, :3].astype(np.float64)
    return np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1])))


def keep_mask(cond: str, seq: str, sid: int, sensor: str, points: np.ndarray, loader, elev_range) -> tuple:
    n = len(points)
    info = {}
    if cond == "b6_none":
        return np.ones(n, bool), info
    rng = np.random.default_rng([zlib.crc32(seq.encode()), 20261118, 6, CODE[cond], int(sid)])
    if cond == "b6_drop50":
        return rng.random(n) >= 0.5, info
    if cond == "b6_range50":
        r = np.linalg.norm(points[:, :3].astype(np.float64), axis=1)
        return r <= 0.5 * MAX_RANGE, info
    if cond == "b6_occ90":
        center = rng.uniform(0.0, 360.0)
        az = np.degrees(np.arctan2(points[:, 1].astype(np.float64), points[:, 0].astype(np.float64)))
        dev = np.abs((az - center + 180.0) % 360.0 - 180.0)
        info["center_deg"] = float(center)
        return dev > 45.0, info
    if cond == "b6_beam2":
        el = elevation(points)
        if sensor in RING_SENSORS:
            ring = ring_of(sensor, loader, sid, points)
            ids = np.unique(ring)
            med = np.array([np.median(el[ring == i]) for i in ids])
            order = ids[np.argsort(med, kind="stable")]
            rank = np.empty(int(ring.max()) + 1, np.int64)
            rank[order] = np.arange(len(order))
            info.update(rings_present=int(len(ids)), ring_index_elev_monotone=bool(np.all(np.diff(med) > 0)))
            return rank[ring] % 2 == 0, info
        lo, hi = elev_range
        b = BEAMS[sensor]
        k = np.clip(np.floor((el - lo) / (hi - lo) * b).astype(np.int64), 0, b - 1)
        info["outside_range_frac"] = float(np.mean((el < lo) | (el > hi)))
        return k % 2 == 0, info
    raise KeyError(cond)


# ----------------------------------------------------------------- 중간 캐시(조건 하나·시퀀스 하나)
def tmp_path(cond: str, seq: str) -> Path:
    return TMP / cond / (seq.replace("/", "_") + ".npz")


def build(cond: str, seq: str) -> Path:
    target = tmp_path(cond, seq)
    if target.exists():
        return target
    c = H.ctx()
    torch, run = c["torch"], c["run"]
    from _pd_build_test_caches import _make_loader
    from _pbev_branch_fusion import BP
    from encoding.phase_features import _phase_sketch, _pool_rows
    tick = time.time()
    tup0 = c["P"].inputs("test", seq)
    cache0, (cp, sp) = tup0[0], tup0[7]
    sensor, q = seq.split("/")
    encoder, elev_range, key = encoder_and_config(sensor, Path(cp))
    loader = _make_loader(sensor, q)
    sids = np.asarray(cache0["scan_ids"])
    desc, ent, sk = [], [], []
    kept, npts, infos = [], [], []
    t_load = t_enc = t_bev = 0.0
    with torch.no_grad():
        for sid in sids:
            t0 = time.time()
            pts = loader[int(sid)]["points"]
            t_load += time.time() - t0
            m, info = keep_mask(cond, seq, int(sid), sensor, pts, loader, elev_range)
            p2 = pts[m]
            kept.append(float(m.mean()) if len(m) else 1.0)
            npts.append(len(pts))
            if info:
                infos.append(info)
            t0 = time.time()
            d, e = encoder.encode_points(p2, return_entropy=True)
            desc.append(d.detach().cpu().numpy())
            ent.append(float(e))
            t_enc += time.time() - t0
            t0 = time.time()
            bev, _ = BP.project(p2, keep_intensity=False)
            sk.append(_phase_sketch(_pool_rows(np.maximum(bev, 0.0), 16), 12))
            t_bev += time.time() - t0
    sk = np.stack(sk)
    desc = np.stack(desc).astype(cache0["descriptors"].dtype)
    ent = np.asarray(ent, dtype=np.asarray(cache0["spectral_entropies"]).dtype)
    diag = dict(cond=cond, seq=seq, sensor=sensor, cache_key=key, elevation_range=list(elev_range), n_keyframes=int(len(sids)),
                kept_frac_mean=float(np.mean(kept)), kept_frac_min=float(np.min(kept)), points_mean=float(np.mean(npts)),
                seconds=dict(total=time.time() - tick, load=t_load, encode=t_enc, bev=t_bev),
                beam_method=("ring_rank" if sensor in RING_SENSORS else f"elevation_quantized_{BEAMS.get(sensor)}") if cond == "b6_beam2" else None)
    if infos and "ring_index_elev_monotone" in infos[0]:
        diag["ring_index_elev_monotone_frac"] = float(np.mean([i["ring_index_elev_monotone"] for i in infos]))
        diag["rings_present_median"] = float(np.median([i["rings_present"] for i in infos]))
    if infos and "outside_range_frac" in infos[0]:
        diag["outside_elev_range_frac_mean"] = float(np.mean([i["outside_range_frac"] for i in infos]))
    if cond == "b6_none":  # 기존 캐시·스펙트럼과 대조
        sp0 = run.ARR(Path(sp))
        d0 = np.asarray(cache0["descriptors"])
        e0 = np.asarray(cache0["spectral_entropies"])
        diag["check"] = dict(
            descriptors_bitwise=bool(np.array_equal(desc, d0)), descriptors_max_abs=float(np.abs(desc - d0).max()),
            descriptors_max_rel=float((np.abs(desc - d0) / np.maximum(np.abs(d0), 1e-30)).max()),
            descriptors_scale=float(np.abs(d0).max()),
            entropy_bitwise=bool(np.array_equal(ent, e0)), entropy_max_abs=float(np.abs(ent - e0).max()),
            sk_re_bitwise=bool(np.array_equal(sk.real.astype(np.float32), sp0["sk_re"])),
            sk_im_bitwise=bool(np.array_equal(sk.imag.astype(np.float32), sp0["sk_im"])),
            sk_max_abs=float(max(np.abs(sk.real.astype(np.float32) - sp0["sk_re"]).max(), np.abs(sk.imag.astype(np.float32) - sp0["sk_im"]).max())),
            scan_ids_equal=bool(np.array_equal(sp0["scan_ids"], sids)))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + ".part.npz")
    np.savez(tmp, scan_ids=sids, descriptors=desc, spectral_entropies=ent, sk_re=sk.real.astype(np.float32),
             sk_im=sk.imag.astype(np.float32), kept_frac=np.asarray(kept, np.float32), diag=json.dumps(diag, ensure_ascii=False))
    tmp.rename(target)
    print("BUILD", cond, seq, len(sids), json.dumps(diag["seconds"]), flush=True)
    return target


# ----------------------------------------------------------------- 평가
def evaluate(cond: str, seq: str, models: list, root: Path = OUT) -> None:
    c = H.ctx()
    pipeline = c["pipeline"]
    name = seq.replace("/", "_")
    folder = root / cond / "ledgers" / "test" / name
    if all((folder / (m + ".json")).exists() for m in models):
        return
    path = build(cond, seq)
    tick = time.time()
    with np.load(path) as z:
        t = {k: z[k] for k in z.files}
    bdiag = json.loads(str(t["diag"]))
    tup0 = c["P"].inputs("test", seq)
    cache0 = tup0[0]
    assert np.array_equal(t["scan_ids"], cache0["scan_ids"])
    cache = {k: np.asarray(v) for k, v in cache0.items()}  # 새 dict(원본 배열은 바꾸지 않음)
    cache["descriptors"], cache["spectral_entropies"] = t["descriptors"], t["spectral_entropies"]
    spectrum = t["sk_re"][:, :, :8].astype(np.float32) + 1j * t["sk_im"][:, :, :8].astype(np.float32)
    tup = B7.build_tuple(cache, spectrum, tup0[7])
    q = np.arange(30, len(spectrum) - 15)
    assert np.array_equal(tup[2][q], tup0[2][q]) and np.array_equal(tup[3], tup0[3])  # 정답·travel 그대로
    graph = B7.build_graph(tup, spectrum)
    raw = pipeline.raw_key(cache["descriptors"])
    pool = B7.build_pool(raw, tup[4])
    base = c["run"].ARR(c["C"].P.TEST / "test" / name / "candidate_pool.npz")
    sizes = np.diff(pool["offset"])
    diag = dict(build=bdiag, n_queries=int(len(pool["query"])), pool_equal_disk=B7.disk_pool_check(seq, pool),
                pool_size_mean=float(sizes.mean()), pool_size_mean_base=float(np.diff(base["offset"]).mean()))
    prep = time.time() - tick
    B7.evaluate_tuple(cond, seq, models, root, tup, graph, pool, diag, prep, check_base=(cond == "b6_none"))
    print("EVAL", cond, seq, round(time.time() - tick, 1), flush=True)


def campaign(cond: str, group: str, workers: int) -> None:
    assert cond != "b6_none"
    B7.campaign(cond, group, workers, script=__file__, desc=DESC,
                params=dict(family="B6", cond=cond, max_range_m=MAX_RANGE, beams=BEAMS, ring_sensors=list(RING_SENSORS),
                            keyframes="existing scan_ids", odometry="existing KISS-ICP poses (not re-run)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("build", "eval", "campaign"))
    ap.add_argument("--cond", default="b6_none")
    ap.add_argument("--sequence")
    ap.add_argument("--models", default="all12")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--root", default=str(OUT))
    a = ap.parse_args()
    assert a.cond in CODE
    c = H.ctx()
    c["P"].e.configure()
    c["torch"].set_num_threads(4)
    if a.action == "build":
        build(a.cond, a.sequence)
    elif a.action == "eval":
        evaluate(a.cond, a.sequence, H.model_names(a.models), Path(a.root))
    else:
        campaign(a.cond, a.models, a.workers)
