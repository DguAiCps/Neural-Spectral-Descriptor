#!/usr/bin/env python3
"""invkey 학습 33시퀀스의 키프레임마다 광선 기준 증강(조건 all)을 1회 적용한 spec(805x16)을 저장한다 (2026-11-15).

증강은 scripts/_invkey_augment_20261115.py의 augment(), 서술자는 깨끗한 계수와 같은
occupancy()·bank()·describe()다. 시드는 seed_for(시퀀스, scan_id, "aug1")라 다시 돌리면 같은 결과가 나온다.

입력: results/invkey_study_20261114/linear/metrics.json의 train 목록(33개),
  시퀀스 레코드(캐시·자세 필드·키프레임 수)는 scripts/_invkey_extract_features_20261114.py의 survey_split(),
  키프레임 scan_ids는 깨끗한 계수 파일(features/<sensor>_<seq>.npz)과 캐시 둘 다에서 읽어 같은지 단언한다.
출력:
  /mnt/d/NSD_datasets/invkey_study_20261114/features_aug1/<sensor>_<seq>.npz (컨테이너 /workspace/data/...)
    spec (n, 805, 16) float16, scan_ids (n,) int64 (깨끗한 파일과 같은 행 순서), 스캔별 증강 요약:
    n_obj int8, near_bus bool, r0 float32 (안 하면 nan), thin int8 (1이면 안 함), thin_offset int8,
    frac_replaced·frac_removed·frac_added·frac_affected float32 (반환 있는 점 대비),
    n_boxes_hit int8 (점이 하나라도 닿은 상자 수), ground_c·ground_tilt float32 (지면 추정 안 했으면 nan),
    ground_fallback int8 (-1 추정 안 함, 0 평면, 1 수평 5% 분위수, 2 점 부족), n_valid int32, n_rings int16.
    np.savez(비압축), 임시 이름(.npz.tmp)으로 쓴 뒤 이름을 바꾼다. 이미 있고 검사를 통과하면 건너뛴다.
  results/invkey_study_20261114/features_aug1_logs/  시퀀스별 출력, 실행 기록 (호스트)
  results/invkey_study_20261114/features_aug1_manifest.json  (호스트)

실행 (리포 루트):
  # 호스트: 시퀀스마다 컨테이너 프로세스 하나, 동시 최대 3개. 시작 전과 띄우기 전마다 df /mnt/d의 여유에서
  #   진행 중·새 시퀀스 예상 크기를 빼고 12 GiB 밑이면 멈춘다.
  python3 scripts/_invkey_extract_aug_20261115.py run --jobs 3
  # 컨테이너: 한 시퀀스 (run이 부른다)
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_extract_aug_20261115.py extract --seq kitti/01
  # 컨테이너: 저장한 행 일부를 다시 계산해 같은지 확인
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_invkey_extract_aug_20261115.py verify --seqs kitti/01 nclt/2013-01-10
  # 호스트: 검사와 목록
  python3 scripts/_invkey_extract_aug_20261115.py manifest
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

from _invkey_augment_20261115 import augment, seed_for  # noqa: E402
from _invkey_extract_features_20261114 import F16_MAX, SHAPE, fname, survey_split  # noqa: E402

STUDY = REPO / "results/invkey_study_20261114"
METRICS = STUDY / "linear/metrics.json"
LOGS = STUDY / "features_aug1_logs"
MANIFEST = STUDY / "features_aug1_manifest.json"
HOST_CLEAN = Path("/mnt/d/NSD_datasets/invkey_study_20261114/features")
CONT_CLEAN = Path("/workspace/data/invkey_study_20261114/features")
HOST_AUG = Path("/mnt/d/NSD_datasets/invkey_study_20261114/features_aug1")
CONT_AUG = Path("/workspace/data/invkey_study_20261114/features_aug1")
CONTAINER = "ring_baseline"
CONT_REPO = "/workspace/Neural-Spectral-Codec"
TAG = "aug1"
COND = "all"
FLOOR_GIB = 12.0
FALLBACK = {None: 0, "horizontal_q05": 1, "no_points": 2}
SUMMARY = {"n_obj": np.int8, "near_bus": np.bool_, "r0": np.float32, "thin": np.int8,
           "thin_offset": np.int8, "frac_replaced": np.float32, "frac_removed": np.float32,
           "frac_added": np.float32, "frac_affected": np.float32, "n_boxes_hit": np.int8,
           "ground_c": np.float32, "ground_tilt": np.float32, "ground_fallback": np.int8,
           "n_valid": np.int32, "n_rings": np.int16}


def train_keys():
    return list(json.load(open(METRICS))["train"])


def summarize(info):
    g = info["ground"]
    return {"n_obj": info["n_obj"], "near_bus": info["near_bus"],
            "r0": np.nan if info["r0"] is None else info["r0"], "thin": info["thin"],
            "thin_offset": info["thin_offset"], "frac_replaced": info["frac_replaced"],
            "frac_removed": info["frac_removed"], "frac_added": info["frac_added"],
            "frac_affected": info["frac_affected"],
            "n_boxes_hit": int(sum(1 for c in info.get("n_hit_by_box", []) if c > 0)),
            "ground_c": np.nan if (g is None or not g["ok"]) else g["c"],
            "ground_tilt": np.nan if (g is None or not g["ok"]) else g["tilt_deg"],
            "ground_fallback": -1 if g is None else FALLBACK[g["fallback"]],
            "n_valid": info["n_valid"], "n_rings": info["n_rings"]}


def check_file(path, ids):
    """저장 파일이 깨끗한 파일과 같은 행(scan_ids)·모양인지."""
    if not path.is_file():
        return False
    try:
        with np.load(path) as z:
            ok = (z["spec"].shape == (len(ids), *SHAPE) and z["spec"].dtype == np.float16
                  and np.array_equal(z["scan_ids"], ids)
                  and all(len(z[k]) == len(ids) for k in SUMMARY))
        return bool(ok)
    except Exception:
        return False


def clean_ids(key, root):
    with np.load(root / fname(key)) as z:
        return np.asarray(z["scan_ids"], np.int64)


# ---------------------------------------------------------------- 컨테이너: 한 시퀀스
def extract(key):
    import torch
    from _pd_build_test_caches import _make_loader
    from _poolalt_probe_20261114 import describe
    from _scatter_probe_20261114 import bank, occupancy

    assert key in train_keys(), key
    recs, _ = survey_split()
    rec = recs[key]
    ids = clean_ids(key, CONT_CLEAN)
    cache_ids = np.asarray(np.load(REPO / rec["cache"])["scan_ids"], np.int64)
    assert np.array_equal(ids, cache_ids), f"{key}: clean scan_ids != cache scan_ids"
    n = len(ids)
    assert n == int(rec["n_keyframes"]), (n, rec["n_keyframes"])
    out = CONT_AUG / fname(key)
    if check_file(out, ids):
        print(f"[{key}] exists and checks, skip", flush=True)
        print("RESULT " + json.dumps({"key": key, "status": "skipped"}), flush=True)
        return 0
    sensor, seq = key.split("/")
    loader = _make_loader(sensor, seq)
    F = torch.from_numpy(bank()).to("cuda")
    spec = np.empty((n, *SHAPE), np.float16)
    summ = {k: np.empty(n, t) for k, t in SUMMARY.items()}
    nonfinite, amax = 0, 0.0
    t_aug = 0.0
    t0 = time.time()
    print(f"[{key}] n {n} cache {rec['cache']} cond {COND} tag {TAG}", flush=True)
    for i, sid in enumerate(ids):
        if i % 2000 == 0:
            el = time.time() - t0
            print(f"  {i}/{n}  {el:.0f}s  {i / el if el else 0:.1f} scans/s", flush=True)
        pts = loader[int(sid)]["points"]
        ta = time.time()
        aug, info = augment(pts, np.random.default_rng(seed_for(key, sid, TAG)), sensor, COND)
        t_aug += time.time() - ta
        _, s = describe(occupancy(aug), F)
        if s.shape != SHAPE or s.dtype != np.float32:
            print(f"STOP [{key}] describe returned {s.shape} {s.dtype}", flush=True)
            return 2
        nonfinite += int((~np.isfinite(s)).sum())
        amax = max(amax, float(np.abs(s).max()))
        spec[i] = s.astype(np.float16)
        for k, v in summarize(info).items():
            summ[k][i] = v
    dt = time.time() - t0
    res = {"key": key, "n": n, "extract_s": round(dt, 1), "augment_s": round(t_aug, 1),
           "scans_per_s": round(n / dt, 1), "nonfinite_f32": nonfinite, "max_abs_f32": amax}
    if nonfinite or amax > F16_MAX:
        res["status"] = "failed_range"
        print(f"STOP [{key}] non-finite or float16 overflow; not saved", flush=True)
        print("RESULT " + json.dumps(res), flush=True)
        return 3
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, spec=spec, scan_ids=ids, **summ)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    if not check_file(out, ids):
        res["status"] = "failed_reload"
        print("RESULT " + json.dumps(res), flush=True)
        return 4
    res["status"] = "done"
    res["bytes"] = out.stat().st_size
    print(f"[{key}] saved {out} ({res['bytes']} B) in {time.time() - t0:.0f}s", flush=True)
    print("RESULT " + json.dumps(res), flush=True)
    return 0


def verify(seqs, k):
    """저장한 행 k개를 다시 계산해(같은 시드) float16 값과 요약이 같은지 본다."""
    import torch
    from _pd_build_test_caches import _make_loader
    from _poolalt_probe_20261114 import describe
    from _scatter_probe_20261114 import bank, occupancy
    F = torch.from_numpy(bank()).to("cuda")
    out = {}
    for key in seqs:
        with np.load(CONT_AUG / fname(key)) as z:
            spec, ids = z["spec"], z["scan_ids"]
            summ = {s: z[s] for s in SUMMARY}
        sensor, seq = key.split("/")
        loader = _make_loader(sensor, seq)
        pick = np.linspace(0, len(ids) - 1, k).astype(int)
        same_spec, same_summ = 0, 0
        for i in pick:
            sid = int(ids[i])
            aug, info = augment(loader[sid]["points"], np.random.default_rng(seed_for(key, sid, TAG)),
                                sensor, COND)
            _, s = describe(occupancy(aug), F)
            same_spec += int(np.array_equal(s.astype(np.float16), spec[i]))
            sm = summarize(info)
            same_summ += int(all(np.array_equal(np.asarray(sm[c], SUMMARY[c]), summ[c][i], equal_nan=True)
                                 if SUMMARY[c] != np.bool_ else sm[c] == summ[c][i] for c in SUMMARY))
        out[key] = {"k": int(k), "spec_equal": same_spec, "summary_equal": same_summ}
        print(f"{key:22s} spec equal {same_spec}/{k}  summary equal {same_summ}/{k}", flush=True)
    print("VERIFY " + json.dumps(out), flush=True)
    return 0 if all(v["spec_equal"] == v["k"] and v["summary_equal"] == v["k"]
                    for v in out.values()) else 6


# ---------------------------------------------------------------- 호스트: 실행 감독
def d_free_gib():
    r = subprocess.run(["df", "-B1", "--output=avail", "/mnt/d"], capture_output=True, text=True,
                       timeout=60)
    return int(r.stdout.split()[-1]) / 2 ** 30


def expected_gib(n):
    return n * SHAPE[0] * SHAPE[1] * 2 / 2 ** 30


def run(jobs, seqs):
    recs, _ = survey_split()
    order = list(seqs) if seqs else sorted(train_keys(), key=lambda k: -recs[k]["n_keyframes"])
    assert all(k in train_keys() for k in order)
    LOGS.mkdir(parents=True, exist_ok=True)
    HOST_AUG.mkdir(parents=True, exist_ok=True)
    ids = {k: clean_ids(k, HOST_CLEAN) for k in order}
    pending = [k for k in order if not check_file(HOST_AUG / fname(k), ids[k])]
    stamp = time.strftime("%Y%m%dT%H%M%S")
    free0 = d_free_gib()
    need_all = sum(expected_gib(len(ids[k])) for k in pending)
    t_start = time.time()
    print(f"run {stamp}: {len(order)} requested, {len(pending)} pending, D free {free0:.2f} GiB, "
          f"pending need {need_all:.2f} GiB, floor {FLOOR_GIB} GiB", flush=True)
    running, done, failed, stopped, frees = {}, [], [], None, [free0]
    range_fail = False
    while pending or running:
        for k, (p, fh, t0) in list(running.items()):
            if p.poll() is None:
                continue
            fh.close()
            ok = p.returncode == 0 and check_file(HOST_AUG / fname(k), ids[k])
            (done if ok else failed).append(k)
            range_fail |= p.returncode == 3
            fr = d_free_gib()
            frees.append(fr)
            print(f"  {'done' if ok else 'FAILED'} {k} rc {p.returncode} {time.time() - t0:.0f}s"
                  f"  D free {fr:.2f} GiB", flush=True)
            del running[k]
        if range_fail and pending:
            print(f"STOP: float16 range failure in {failed}", flush=True)
            stopped = {"reason": "float16 range", "failed": failed}
            pending = []
        if pending and len(running) < jobs:
            k = pending[0]
            free = d_free_gib()
            need = sum(expected_gib(len(ids[r])) for r in running) + expected_gib(len(ids[k]))
            if free - need < FLOOR_GIB:
                stopped = {"before": k, "free_gib": free, "need_gib": need}
                print(f"STOP disk guard: free {free:.2f} - need {need:.2f} < {FLOOR_GIB} GiB",
                      flush=True)
                pending = []
                continue
            pending.pop(0)
            fh = open(LOGS / f"{fname(k)[:-4]}.log", "w")
            cmd = ["docker", "exec", "-w", CONT_REPO, CONTAINER, "python3",
                   "scripts/_invkey_extract_aug_20261115.py", "extract", "--seq", k]
            running[k] = (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh, time.time())
            print(f"  launch {k} (n {len(ids[k])}, need {need:.2f} GiB, free {free:.2f} GiB)",
                  flush=True)
            continue
        time.sleep(5)
    rec = {"stamp": stamp, "jobs": jobs, "requested": order, "done": done, "failed": failed,
           "stopped": stopped, "wall_s": round(time.time() - t_start, 1),
           "d_free_gib_start": free0, "d_free_gib_end": d_free_gib(), "d_free_gib_min": min(frees)}
    (LOGS / f"run_{stamp}.json").write_text(json.dumps(rec, indent=1))
    print("RUN " + json.dumps(rec), flush=True)
    return 0 if not failed and not stopped else 7


# ---------------------------------------------------------------- 호스트: 검사와 목록
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def manifest():
    keys = train_keys()
    seqs, problems = {}, []
    pooled = {k: [] for k in SUMMARY}
    for key in keys:
        p = HOST_AUG / fname(key)
        ids = clean_ids(key, HOST_CLEAN)
        if not check_file(p, ids):
            problems.append(f"{key}: missing or failed check")
            continue
        with np.load(p) as z:
            spec = z["spec"]
            summ = {k: z[k] for k in SUMMARY}
        nonfinite, amax, zero_rows = 0, 0.0, 0
        for s in range(0, len(spec), 2000):
            c = spec[s:s + 2000]
            nonfinite += int((~np.isfinite(c)).sum())
            amax = max(amax, float(np.abs(c).max()))
            zero_rows += int((~c.reshape(len(c), -1).any(1)).sum())
        if nonfinite or amax > F16_MAX:
            problems.append(f"{key}: range")
        ext = None
        log = LOGS / f"{fname(key)[:-4]}.log"
        if log.is_file():
            for line in log.read_text().splitlines():
                if line.startswith("RESULT "):
                    ext = json.loads(line[7:])
        for k in SUMMARY:
            pooled[k].append(summ[k])
        r0 = summ["r0"]
        seqs[key] = {
            "n_keyframes": len(ids), "file": str(p), "size_bytes": p.stat().st_size,
            "sha256": sha256(p), "scan_ids_eq_clean": True, "nonfinite": nonfinite,
            "max_abs_f16": amax, "all_zero_rows": zero_rows,
            "extract_s": None if ext is None else ext.get("extract_s"),
            "augment_s": None if ext is None else ext.get("augment_s"),
            "aug": {
                "n_obj_mean": float(summ["n_obj"].mean()), "near_bus_rate": float(summ["near_bus"].mean()),
                "range_rate": float(np.isfinite(r0).mean()),
                "thin_rate": {str(f): float((summ["thin"] == f).mean()) for f in (1, 2, 4)},
                "frac_affected_median": float(np.median(summ["frac_affected"])),
                "frac_affected_p95": float(np.percentile(summ["frac_affected"], 95)),
                "frac_replaced_mean": float(summ["frac_replaced"].mean()),
                "frac_removed_mean": float(summ["frac_removed"].mean()),
                "frac_added_mean": float(summ["frac_added"].mean()),
                "ground_fallback_counts": {str(v): int((summ["ground_fallback"] == v).sum())
                                           for v in (-1, 0, 1, 2)},
                "ground_c_median": float(np.nanmedian(summ["ground_c"])) if np.isfinite(summ["ground_c"]).any() else None,
                "ground_tilt_p95": float(np.nanpercentile(summ["ground_tilt"], 95)) if np.isfinite(summ["ground_tilt"]).any() else None,
            }}
        print(f"{key:24s} n {len(ids):6d} {p.stat().st_size / 2 ** 20:7.1f} MiB  "
              f"aff med {seqs[key]['aug']['frac_affected_median']:.3f}  max {amax:.1f}", flush=True)
    P = {k: np.concatenate(v) for k, v in pooled.items()} if all(pooled.values()) else None
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    out = {
        "date": "2026-11-15", "script": "scripts/_invkey_extract_aug_20261115.py",
        "script_sha256": sha256(Path(__file__)),
        "augment_script_sha256": sha256(REPO / "scripts/_invkey_augment_20261115.py"),
        "git_commit": git, "git_note": "scripts are untracked at this commit; use their sha256",
        "cond": COND, "seed": "seed_for(key, scan_id, 'aug1') in scripts/_invkey_augment_20261115.py",
        "split": "train list of results/invkey_study_20261114/linear/metrics.json",
        "feature_dir_host": str(HOST_AUG), "feature_dir_container": str(CONT_AUG),
        "format": {"container": "np.savez (uncompressed)", "spec": "float16 (n, 805, 16)",
                   "scan_ids": "int64 (n,), equal to features/<sensor>_<seq>.npz scan_ids",
                   **{k: str(np.dtype(t)) for k, t in SUMMARY.items()}},
        "n_sequences": len(seqs), "n_keyframes": sum(v["n_keyframes"] for v in seqs.values()),
        "total_bytes": sum(v["size_bytes"] for v in seqs.values()),
        "problems": problems,
        "pooled_aug": None if P is None else {
            "n_obj_hist": np.bincount(P["n_obj"], minlength=9).tolist(),
            "near_bus_rate": float(P["near_bus"].mean()),
            "range_rate": float(np.isfinite(P["r0"]).mean()),
            "thin_counts": {str(f): int((P["thin"] == f).sum()) for f in (1, 2, 4)},
            "frac_affected_quantiles": {q: float(np.percentile(P["frac_affected"], q))
                                        for q in (5, 25, 50, 75, 95)},
            "frac_replaced_mean": float(P["frac_replaced"].mean()),
            "frac_removed_mean": float(P["frac_removed"].mean()),
            "frac_added_mean": float(P["frac_added"].mean()),
            "ground_fallback_counts": {str(v): int((P["ground_fallback"] == v).sum())
                                       for v in (-1, 0, 1, 2)},
        },
        "runs": [json.loads(p.read_text()) for p in sorted(LOGS.glob("run_*.json"))],
        "verify": next((json.loads(line[7:]) for line in
                        (LOGS / "verify.log").read_text().splitlines()
                        if line.startswith("VERIFY ")), None) if (LOGS / "verify.log").is_file() else None,
        "sequences": seqs,
    }
    tmp = MANIFEST.with_name(MANIFEST.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, MANIFEST)
    print(f"manifest: {len(seqs)} sequences, {out['n_keyframes']} keyframes, "
          f"{out['total_bytes'] / 2 ** 30:.2f} GiB, problems {problems}", flush=True)
    return 0 if not problems and len(seqs) == len(keys) else 8


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--seq", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--seqs", nargs="+", required=True)
    p.add_argument("--k", type=int, default=10)
    p = sub.add_parser("run")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--seqs", nargs="*")
    sub.add_parser("manifest")
    a = ap.parse_args()
    if a.cmd == "extract":
        return extract(a.seq)
    if a.cmd == "verify":
        return verify(a.seqs, a.k)
    if a.cmd == "run":
        return run(a.jobs, a.seqs)
    return manifest()


if __name__ == "__main__":
    sys.exit(main())
