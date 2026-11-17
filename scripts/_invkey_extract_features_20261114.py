#!/usr/bin/env python3
"""invkey 진단 연구용 고정 서술자 spec(805x16)을 후보 분할 A의 63시퀀스 키프레임마다 뽑아 저장한다 (2026-11-14).

서술자 정의는 바꾸지 않는다. scripts/_scatter_probe_20261114.py의 occupancy()·bank()와
scripts/_poolalt_probe_20261114.py의 describe()를 그대로 불러 쓰고, describe()가 함께 만드는 loc은 버린다
(loc을 건너뛰려면 spec 계산을 복사해야 해서 함수를 그대로 부른다).

입력: results/invkey_study_20261114/data_survey.json의 candidates["A"] (train 42, heldout 21).
  시퀀스마다 survey의 `cache`에서 키프레임 scan_ids·timestamps를, `pose_field`(poses 또는 poses_gt)의
  평행 이동에서 GT 위치를 읽는다. 스캔은 scripts/_pd_build_test_caches.py의 _make_loader로 읽는다.
출력:
  /mnt/d/NSD_datasets/invkey_study_20261114/features/<sensor>_<seq>.npz (컨테이너 /workspace/data/...)
    spec (n, 805, 16) float16, scan_ids (n,) int64, xyz (n, 3) float64, timestamps (n,) float64.
    np.savez(비압축). 임시 이름(.npz.tmp)으로 쓴 뒤 이름을 바꾼다. 이미 있고 읽히는 파일은 건너뛴다.
    float32 값이 float16 범위를 넘거나 NaN/Inf가 있으면 저장하지 않고 실패로 끝낸다(척도 조정 없음).
  results/invkey_study_20261114/features_logs/  시퀀스별 표준 출력, 실행 기록 (호스트에서 씀)
  results/invkey_study_20261114/features_manifest.json  목록·검사 결과 (호스트에서 씀)

실행 (리포 루트):
  # 호스트: 시퀀스마다 컨테이너 프로세스 하나, 동시 최대 3개. 띄우기 전마다 D 실제 여유
  #   (Windows Get-PSDrive)에서 진행 중·새 시퀀스 예상 크기를 빼고 15 GiB 밑이면 멈춘다.
  #   한 시퀀스가 실패하면(로더 오류 등) 기록하고 나머지는 계속한다. float16 범위 초과는 전체를 멈춘다.
  python3 scripts/_invkey_extract_features_20261114.py run --jobs 3 [--seqs kitti360/0004 ...]
  # 컨테이너: 한 시퀀스 (run이 부른다)
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_extract_features_20261114.py extract --seq kitti360/0004
  # 컨테이너: 저장한 float16으로 기준 3시퀀스 R@1·R@100 재현(--part ref),
  #   표본 스캔을 다시 계산해 float32와 대조(--part sample)
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_extract_features_20261114.py verify --part ref
  # 호스트: 개수·위치(재방문 질의 수)·무결성 검사와 목록 작성
  python3 scripts/_invkey_extract_features_20261114.py manifest
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SURVEY = REPO / "results/invkey_study_20261114/data_survey.json"
STUDY = REPO / "results/invkey_study_20261114"
LOGS = STUDY / "features_logs"
MANIFEST = STUDY / "features_manifest.json"
HOST_FEAT = Path("/mnt/d/NSD_datasets/invkey_study_20261114/features")
CONT_FEAT = Path("/workspace/data/invkey_study_20261114/features")
CONTAINER = "ring_baseline"
CONT_REPO = "/workspace/Neural-Spectral-Codec"
SHAPE = (805, 16)
F16_MAX = 65504.0
F16_TINY = 6.103515625e-05          # float16 최소 정규수
FLOOR_GIB = 15.0                    # D 실제 여유 하한
REFS = ("kitti360/0004", "helipr/Roundabout01", "mulran/Sejong03")
SAMPLE_SEQS = ("kitti/00", "kitti360/0004", "helipr/Roundabout01", "mulran/Sejong03",
               "nclt/2012-01-08", "mcd/kth_day_06")
DESCRIPTOR_SCRIPTS = ("scripts/_poolalt_probe_20261114.py", "scripts/_scatter_probe_20261114.py")


def survey_split():
    """candidates["A"]의 시퀀스 레코드와 분할 쪽(train/heldout)."""
    d = json.load(open(SURVEY))
    by = {s["key"]: s for s in d["sequences"]}
    A = d["candidates"]["A"]
    side = {k: "train" for k in A["train"]}
    side.update({k: "heldout" for k in A["heldout"]})
    assert len(side) == 63, len(side)
    return {k: by[k] for k in side}, side


def fname(key):
    sensor, seq = key.split("/")
    return f"{sensor}_{seq}.npz"


def loads_ok(path, n):
    if not path.is_file():
        return False
    try:
        with np.load(path) as z:
            s = z["spec"]
            return (s.shape == (n, *SHAPE) and s.dtype == np.float16
                    and len(z["scan_ids"]) == n and z["xyz"].shape == (n, 3)
                    and len(z["timestamps"]) == n)
    except Exception:
        return False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- 컨테이너: 한 시퀀스 추출
def extract(key):
    import torch
    from _pd_build_test_caches import _make_loader
    from _poolalt_probe_20261114 import describe
    from _scatter_probe_20261114 import bank, occupancy

    recs, side = survey_split()
    rec = recs[key]
    n = int(rec["n_keyframes"])
    out = CONT_FEAT / fname(key)
    if loads_ok(out, n):
        print(f"[{key}] exists and loads, skip", flush=True)
        print("RESULT " + json.dumps({"key": key, "status": "skipped"}), flush=True)
        return 0
    z = np.load(REPO / rec["cache"])
    ids = np.asarray(z["scan_ids"], dtype=np.int64)
    xyz = np.ascontiguousarray(np.asarray(z[rec["pose_field"]], dtype=np.float64)[:, :3, 3])
    ts = np.asarray(z["timestamps"], dtype=np.float64)
    if len(ids) != n or len(xyz) != n or len(ts) != n:
        print(f"STOP [{key}] cache length {len(ids)}/{len(xyz)}/{len(ts)} != survey {n}", flush=True)
        return 2
    sensor, seq = key.split("/")
    loader = _make_loader(sensor, seq)
    F = torch.from_numpy(bank()).to("cuda")

    spec = np.empty((n, *SHAPE), np.float16)
    st = dict(nonfinite_f32=0, nonfinite_f16=0, max_abs_f32=0.0, min_f32=np.inf, zeros_f32=0,
              below_f16_normal=0, max_rel_err_normal=0.0, max_abs_err_below_normal=0.0,
              max_desc_rel_err=0.0)
    t0 = time.time()
    print(f"[{key}] side {side[key]} n {n} cache {rec['cache']} pose {rec['pose_field']}", flush=True)
    for i, sid in enumerate(ids):
        if i % 2000 == 0:
            el = time.time() - t0
            print(f"  {i}/{n}  {el:.0f}s  {i / el if el else 0:.1f} scans/s", flush=True)
        _, s = describe(occupancy(loader[int(sid)]["points"]), F)
        if s.shape != SHAPE or s.dtype != np.float32:
            print(f"STOP [{key}] describe returned {s.shape} {s.dtype}", flush=True)
            return 2
        h = s.astype(np.float16)
        spec[i] = h
        a = np.abs(s)
        err = np.abs(h.astype(np.float32) - s)
        normal = a >= F16_TINY
        st["nonfinite_f32"] += int((~np.isfinite(s)).sum())
        st["nonfinite_f16"] += int((~np.isfinite(h)).sum())
        st["max_abs_f32"] = max(st["max_abs_f32"], float(a.max()))
        st["min_f32"] = min(st["min_f32"], float(s.min()))
        st["zeros_f32"] += int((s == 0).sum())
        st["below_f16_normal"] += int((~normal & (s != 0)).sum())
        if normal.any():
            st["max_rel_err_normal"] = max(st["max_rel_err_normal"],
                                           float((err[normal] / a[normal]).max()))
        if (~normal).any():
            st["max_abs_err_below_normal"] = max(st["max_abs_err_below_normal"],
                                                 float(err[~normal].max()))
        st["max_desc_rel_err"] = max(st["max_desc_rel_err"],
                                     float(np.linalg.norm(err) / np.linalg.norm(s)))
    dt = time.time() - t0
    res = {"key": key, "n": n, "extract_s": round(dt, 1), "scans_per_s": round(n / dt, 1), **st}
    if st["nonfinite_f32"] or st["nonfinite_f16"] or st["max_abs_f32"] > F16_MAX:
        res["status"] = "failed_range"
        print(f"STOP [{key}] non-finite or float16 overflow; not saved", flush=True)
        print("RESULT " + json.dumps(res), flush=True)
        return 3
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, spec=spec, scan_ids=ids, xyz=xyz, timestamps=ts)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    if not loads_ok(out, n):
        res["status"] = "failed_reload"
        print("RESULT " + json.dumps(res), flush=True)
        return 4
    res["status"] = "done"
    res["save_s"] = round(time.time() - t0 - dt, 1)
    print(f"[{key}] saved {out} ({out.stat().st_size} B) in {time.time() - t0:.0f}s", flush=True)
    print("RESULT " + json.dumps(res), flush=True)
    return 0


# ---------------------------------------------------------------- 컨테이너: 검증
def verify_ref():
    """scripts/_poolalt_probe_20261114.py main()의 spec 검색 부분(R@1, R@100)을 저장한 float16으로 다시 잰다."""
    import torch
    from _poolalt_probe_20261114 import DEV, NFFT, aligned_sim, to_freq
    from _pr_eval_lib import classify_queries

    out = {}
    for key in REFS:
        sensor, seq = key.split("/")
        exp = json.load(open(REPO / f"results/rawmag_probe_20261114/poolalt_{sensor}_{seq}.json"))
        with np.load(CONT_FEAT / fname(key)) as z:
            spec, xyz_np = z["spec"], z["xyz"]
        n = len(spec)
        xyz = torch.as_tensor(xyz_np, dtype=torch.float32, device=DEV)
        Fdb, ndb = to_freq(spec)
        idx = torch.arange(n, device=DEV)
        qs = torch.arange(30, n, device=DEV)
        top1d, dmin, hit100 = [], [], []
        for s in range(0, len(qs), 16):
            q = qs[s:s + 16]
            past = idx[None] <= (q[:, None] - 30)
            D = (xyz[q][:, None] - xyz[None]).norm(dim=2).masked_fill(~past, torch.inf)
            S = aligned_sim(Fdb[:, q], ndb[q], Fdb, ndb, NFFT["spec"]).masked_fill(
                ~past, -torch.inf)
            v, t = S.topk(100, dim=1)
            ok = t <= (q[:, None] - 30)
            top1d.append(D.gather(1, t[:, :1])[:, 0])
            dmin.append(D.min(dim=1).values)
            hit100.append(((D.gather(1, t) <= 5.0) & ok).any(1))
        top1d, dmin = (torch.cat(v).double().cpu().numpy() for v in (top1d, dmin))
        hit100 = torch.cat(hit100).cpu().numpy()
        cls = classify_queries(dmin)
        pos = cls["positive"]
        r = {"n_map": n, "n_pos": cls["n_pos"],
             "R@1": 100 * float((top1d[pos] <= 5.0).mean()),
             "R@100": 100 * float(hit100[pos].mean()),
             "expected_n_map": exp["n_map"], "expected_n_pos": exp["counts"]["n_pos"],
             "expected_R@1": 100 * exp["reps"]["spec"]["R@1"],
             "expected_R@100": 100 * exp["reps"]["spec"]["R@100"]}
        r["diff_R@1"] = r["R@1"] - r["expected_R@1"]
        r["diff_R@100"] = r["R@100"] - r["expected_R@100"]
        r["ok"] = bool(abs(r["diff_R@1"]) <= 0.2 and abs(r["diff_R@100"]) <= 0.2
                       and n == exp["n_map"] and cls["n_pos"] == exp["counts"]["n_pos"])
        out[key] = r
        print(f"{key:22s} n {n} pos {cls['n_pos']}  R@1 {r['R@1']:.2f} (exp {r['expected_R@1']:.2f})"
              f"  R@100 {r['R@100']:.2f} (exp {r['expected_R@100']:.2f})  ok {r['ok']}", flush=True)
        del Fdb, ndb
        torch.cuda.empty_cache()
    print("VERIFY_REF " + json.dumps(out), flush=True)
    return 0 if all(r["ok"] for r in out.values()) else 5


def verify_sample(seqs, k):
    """시퀀스마다 무작위 키프레임 k개를 다시 계산해(float32) 저장한 float16과 대조한다."""
    import torch
    from _pd_build_test_caches import _make_loader
    from _poolalt_probe_20261114 import describe
    from _scatter_probe_20261114 import bank, occupancy

    F = torch.from_numpy(bank()).to("cuda")
    rng = np.random.default_rng(0)
    out = {}
    for key in seqs:
        p = CONT_FEAT / fname(key)
        if not p.is_file():
            print(f"{key}: no file, skip", flush=True)
            continue
        with np.load(p) as z:
            spec, ids = z["spec"], z["scan_ids"]
        sensor, seq = key.split("/")
        loader = _make_loader(sensor, seq)
        pick = np.sort(rng.choice(len(ids), k, replace=False))
        rel, desc_rel, absmax = 0.0, 0.0, 0.0
        for i in pick:
            _, s = describe(occupancy(loader[int(ids[i])]["points"]), F)
            h = spec[i].astype(np.float32)
            err = np.abs(h - s)
            nm = np.abs(s) >= F16_TINY
            rel = max(rel, float((err[nm] / np.abs(s[nm])).max()))
            desc_rel = max(desc_rel, float(np.linalg.norm(err) / np.linalg.norm(s)))
            absmax = max(absmax, float(np.abs(s).max()))
        out[key] = {"k": int(k), "max_rel_err_normal": rel, "max_desc_rel_err": desc_rel,
                    "max_abs_f32": absmax, "ok": bool(rel <= 2.0 ** -11 * 1.01)}
        print(f"{key:22s} k {k}  max elem rel err {rel:.3e}  max desc rel err {desc_rel:.3e}"
              f"  max |x| {absmax:.1f}  ok {out[key]['ok']}", flush=True)
    print("VERIFY_SAMPLE " + json.dumps(out), flush=True)
    return 0 if all(r["ok"] for r in out.values()) else 6


# ---------------------------------------------------------------- 호스트: 실행 감독
def d_free_gib():
    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", "(Get-PSDrive D).Free/1GB"],
                       capture_output=True, text=True, timeout=120)
    return float(r.stdout.strip())


def expected_gib(n):
    return n * SHAPE[0] * SHAPE[1] * 2 / 2 ** 30


def run(jobs, seqs):
    recs, side = survey_split()
    if seqs:
        order = list(seqs)
    else:   # 기준 3개 먼저, 나머지는 큰 것부터
        rest = sorted((k for k in recs if k not in REFS), key=lambda k: -recs[k]["n_keyframes"])
        order = list(REFS) + rest
    LOGS.mkdir(parents=True, exist_ok=True)
    HOST_FEAT.mkdir(parents=True, exist_ok=True)
    pending = [k for k in order if not loads_ok(HOST_FEAT / fname(k), recs[k]["n_keyframes"])]
    stamp = time.strftime("%Y%m%dT%H%M%S")
    free0 = d_free_gib()
    t_start = time.time()
    print(f"run {stamp}: {len(order)} requested, {len(pending)} pending, D free {free0:.2f} GiB",
          flush=True)
    running, done, failed, stopped, frees = {}, [], [], None, [free0]
    range_fail = False
    while pending or running:
        for k, (p, fh, t0) in list(running.items()):
            if p.poll() is None:
                continue
            fh.close()
            ok = p.returncode == 0 and loads_ok(HOST_FEAT / fname(k), recs[k]["n_keyframes"])
            (done if ok else failed).append(k)
            range_fail |= p.returncode == 3
            fr = d_free_gib()
            frees.append(fr)
            print(f"  {'done' if ok else 'FAILED'} {k} rc {p.returncode} {time.time() - t0:.0f}s"
                  f"  D free {fr:.2f} GiB", flush=True)
            del running[k]
        # 로더 오류 등 한 시퀀스의 실패는 목록에 남기고 나머지는 계속한다.
        # float16 범위 초과·NaN/Inf(rc 3)는 척도 문제라 전체를 멈춘다.
        if range_fail and pending:
            print(f"STOP: float16 range failure in {failed}; not launching {len(pending)} more",
                  flush=True)
            stopped = {"reason": "float16 range", "failed": failed}
            pending = []
        if pending and len(running) < jobs:
            k = pending[0]
            free = d_free_gib()
            need = sum(expected_gib(recs[r]["n_keyframes"]) for r in running) + \
                expected_gib(recs[k]["n_keyframes"])
            if free - need < FLOOR_GIB:
                stopped = {"before": k, "free_gib": free, "need_gib": need}
                print(f"STOP disk guard: free {free:.2f} - need {need:.2f} < {FLOOR_GIB} GiB",
                      flush=True)
                pending = []
                continue
            pending.pop(0)
            fh = open(LOGS / f"{fname(k)[:-4]}.log", "w")
            cmd = ["docker", "exec", "-w", CONT_REPO, CONTAINER, "python3",
                   "scripts/_invkey_extract_features_20261114.py", "extract", "--seq", k]
            running[k] = (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh, time.time())
            print(f"  launch {k} (n {recs[k]['n_keyframes']}, need {need:.2f} GiB,"
                  f" free {free:.2f} GiB)", flush=True)
            continue
        time.sleep(5)
    rec = {"stamp": stamp, "jobs": jobs, "requested": order, "done": done, "failed": failed,
           "stopped": stopped, "wall_s": round(time.time() - t_start, 1),
           "d_free_gib_start": free0, "d_free_gib_end": d_free_gib(), "d_free_gib_min": min(frees)}
    (LOGS / f"run_{stamp}.json").write_text(json.dumps(rec, indent=1))
    print("RUN " + json.dumps(rec), flush=True)
    return 0 if not failed and not stopped else 7


# ---------------------------------------------------------------- 호스트: 검사와 목록
def manifest():
    from _invkey_data_survey_20261114 import brute_counts
    recs, side = survey_split()
    seqs, problems = {}, []
    for key in recs:
        rec = recs[key]
        n = int(rec["n_keyframes"])
        p = HOST_FEAT / fname(key)
        if not p.is_file():
            problems.append(f"{key}: missing")
            continue
        z = np.load(REPO / rec["cache"])
        with np.load(p) as f:
            spec, ids, xyz, ts = f["spec"], f["scan_ids"], f["xyz"], f["timestamps"]
        nonfinite = 0
        amax = 0.0
        zero_rows = []
        for s in range(0, len(spec), 2000):
            c = spec[s:s + 2000]
            nonfinite += int((~np.isfinite(c)).sum())
            amax = max(amax, float(np.abs(c).max()))
            zero_rows += (s + np.flatnonzero(~c.reshape(len(c), -1).any(1))).tolist()
        nq, _ = brute_counts(xyz)
        chk = {
            "shape_ok": spec.shape == (n, *SHAPE) and spec.dtype == np.float16,
            "scan_ids_eq_cache": bool(np.array_equal(ids, z["scan_ids"])),
            "xyz_eq_cache": bool(np.array_equal(xyz, np.asarray(z[rec["pose_field"]], np.float64)[:, :3, 3])),
            "timestamps_eq_cache": bool(np.array_equal(ts, z["timestamps"])),
            "nonfinite": nonfinite, "max_abs_f16": amax,
            "all_zero_rows": zero_rows, "all_zero_scan_ids": [int(ids[i]) for i in zero_rows],
            "revisit_queries": nq, "revisit_queries_survey": rec["revisit_queries"],
        }
        bad = [c for c in ("shape_ok", "scan_ids_eq_cache", "xyz_eq_cache", "timestamps_eq_cache")
               if not chk[c]]
        if nonfinite or amax > F16_MAX:
            bad.append("range")
        if nq != rec["revisit_queries"]:
            bad.append("revisit_queries")
        if bad:
            problems.append(f"{key}: {bad}")
        ext = None
        log = LOGS / f"{fname(key)[:-4]}.log"
        if log.is_file():
            for line in log.read_text().splitlines():
                if line.startswith("RESULT "):
                    ext = json.loads(line[7:])
        seqs[key] = {"side": side[key], "n_keyframes": n, "file": str(p), "size_bytes": p.stat().st_size,
                     "sha256": sha256(p), "cache": rec["cache"], "pose_field": rec["pose_field"],
                     "checks": chk, "extract": ext}
        print(f"{key:24s} {side[key]:7s} n {n:6d}  RQ {nq}/{rec['revisit_queries']}  "
              f"max {amax:.1f}  bad {bad}", flush=True)
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    out = {
        "date": "2026-11-14", "script": "scripts/_invkey_extract_features_20261114.py",
        "script_sha256": sha256(Path(__file__)),
        "git_commit": git,
        "git_note": "descriptor scripts are untracked at this commit; use their sha256",
        "descriptor_scripts_sha256": {s: sha256(REPO / s) for s in DESCRIPTOR_SCRIPTS},
        "survey": str(SURVEY.relative_to(REPO)), "split": "candidates.A",
        "dtype_decision": [
            "spec은 float16으로 저장한다(2026-11-14 주 세션 결정). float32 전체는 17.7 GiB라 D 여유 15 GiB 하한을 지킬 수 없다.",
            "기준 재현: kitti360/0004 95.77/99.95, helipr/Roundabout01 64.77/97.24(기대 64.72), mulran/Sejong03 93.05/100.0(기대 92.79/100.0).",
            "Sejong03 진단: float32로 다시 계산하면 92.79/100.0으로 기대와 같고 저장값은 float32의 float16 변환과 전부 같다. "
            "차이는 양성 791개 중 2개(q 19017, 19467)로, float32 1·2위 코사인 차가 8.5e-6·2.7e-6인 거의 동점이 float16 반올림(서술자 상대 오차 약 2.2e-4)으로 뒤바뀐 것이다.",
        ],
        "data_notes": [
            "spec이 전부 0인 키프레임은 시퀀스별 checks.all_zero_rows에 있다. 2026-11-14 추출에서는 mulran/Sejong02의 "
            "마지막 키프레임(행 19068, scan_id 21168) 하나뿐이다. 원본 Ouster/1566279795718079314.bin이 0바이트라 로더가 점 0개를 준다.",
            "이 키프레임은 재방문 질의(과거 최근접 0.62 m)이고 노름이 0이라 코사인·정렬 유사도가 0/0이 된다. 쓰는 쪽에서 제외하거나 따로 처리해야 한다.",
        ],
        "reference_check": next((json.loads(line[11:]) for line in
                                 (LOGS / "verify_ref.log").read_text().splitlines()
                                 if line.startswith("VERIFY_REF ")), None),
        "format": {"container": "np.savez (uncompressed)", "spec": "float16 (n, 805, 16)",
                   "scan_ids": "int64 (n,)", "xyz": "float64 (n, 3) GT translation of pose_field",
                   "timestamps": "float64 (n,)"},
        "feature_dir_host": str(HOST_FEAT), "feature_dir_container": str(CONT_FEAT),
        "n_sequences": len(seqs), "n_keyframes": sum(v["n_keyframes"] for v in seqs.values()),
        "total_bytes": sum(v["size_bytes"] for v in seqs.values()),
        "problems": problems,
        "runs": [json.loads(p.read_text()) for p in sorted(LOGS.glob("run_*.json"))],
        "sequences": seqs,
    }
    tmp = MANIFEST.with_name(MANIFEST.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, MANIFEST)
    print(f"manifest: {len(seqs)} sequences, {out['n_keyframes']} keyframes, "
          f"{out['total_bytes'] / 2 ** 30:.2f} GiB, problems {problems}", flush=True)
    return 0 if not problems and len(seqs) == 63 else 8


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--seq", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--part", choices=("ref", "sample"), required=True)
    p.add_argument("--seqs", nargs="*", default=list(SAMPLE_SEQS))
    p.add_argument("--k", type=int, default=20)
    p = sub.add_parser("run")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--seqs", nargs="*")
    sub.add_parser("manifest")
    a = ap.parse_args()
    if a.cmd == "extract":
        return extract(a.seq)
    if a.cmd == "verify":
        return verify_ref() if a.part == "ref" else verify_sample(a.seqs, a.k)
    if a.cmd == "run":
        return run(a.jobs, a.seqs)
    return manifest()


if __name__ == "__main__":
    sys.exit(main())
