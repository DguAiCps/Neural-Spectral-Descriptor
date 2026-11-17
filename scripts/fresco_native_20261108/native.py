#!/usr/bin/env python3
"""FreSCo 원 논문 조건 대표 재현: KITTI 08 (2026-11-08).

논문 표I(ICARCV 2022) KITTI 08 최대F1 0.8594를 공개 코드와 공개 전처리 그대로의 원 조건
(GT 2 m 키프레임, 2D 10 m, 과거 키프레임 1..q-100, 후보 20, 50쌍 임계, 공식 PR 산식)으로
다시 낸다. 공통 표2 규약 실행(results/pdv2_fresco_official_20261107)과 분리한다.

  select      (로컬) poses/08.txt -> 공식 KITTI 선택기 규칙의 키프레임 목록·keyframe_pose.csv
  produce     (로컬) 키프레임 원시 bin -> 무손실 스트림(stdout), codec.pack 재사용
  recv        (서버) 스트림 -> 공식 전처리 드라이버(fresco_prep_kitti) -> NAS bev.u8, 이어받기
  server_run  (서버) 앞부분 공식 바이너리 대조 -> 공식 MATLAB 함수 무수정 실행 -> 어댑터 매칭
  analyze     (로컬) 서버 산출 -> 논문 대조·조건 차이 계량·검산 -> summary.json
  controller  (로컬, detach) 위 단계를 순서대로 이어서 실행, 재개 가능, 끝나면 root 알림 등록

공식 원천은 읽기 전용으로만 쓴다: _external/fresco_official_20261107(FreSCo 2bf5cee, 전처리
d94040e), 기존 NAS build/의 공식 바이너리·드라이버·fresco_adapter.m, scripts/fresco_official/codec.py.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "fresco_official"))
import codec  # noqa: E402  (읽기 전용 재사용)

SEQ = "08"
FAMILY = "kitti_hdl64"
SERVER = "joms-gpu"
SERVER_REPO = "/home/qkrrlans00/Neural-Spectral-Codec"
OLD_NAS = Path("/mnt/aicps-nas/qkrrlans00/nsd_fresco_official_20261107")
NAS = Path(os.environ.get("FRESCO_NATIVE_NAS", "/mnt/aicps-nas/qkrrlans00/nsd_fresco_native_20261108"))
NAS_BUDGET = 8 * 2 ** 30
WORK = NAS / f"kitti_{SEQ}"
MATLAB = "/mnt/aicps-nas/qkrrlans00/matlab/R2026a/bin/matlab"
OFFICIAL_M = Path(SERVER_REPO) / "_external/fresco_official_20261107/FreSCo/fresco-matlab"
GUI_SHIM = Path(SERVER_REPO) / "scripts/fresco_official/matlab_gui_shim"
ADAPTER_DIR = OLD_NAS / "build/matlab"
DRIVER = OLD_NAS / "build/fresco_prep_kitti"
OFF_BIN = OLD_NAS / "build/official"

LOCAL_KITTI = Path("/mnt/d/NSD_datasets/kitti/dataset")
OUT = Path(os.environ.get("FRESCO_NATIVE_OUT",
                          "/mnt/d/NSD_local/results/pdv2_overnight_repro_20261108/fresco_native"))
BEV_BYTES = 224 * 224
PREFIX_SCANS = 150          # 공식 선택기·BEV 실행 파일 대조에 쓰는 앞부분 스캔 수(사전 고정)
KEYFRAME_DIST = np.float32(2.0)
COMMIT_EVERY = 200

# KittiPointCloudSelect.cpp:399-403 (공식 하드코딩, KITTI 08 calib.txt의 Tr와 다르다)
LIDAR_WRT_CAM = np.array([[7.967514e-03, -9.999679e-01, -8.462264e-04, -1.377769e-02],
                          [-2.771053e-03, 8.241710e-04, -9.999958e-01, -5.542117e-02],
                          [9.999644e-01, 7.969825e-03, -2.764397e-03, -2.918589e-01],
                          [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def rjson(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------ select (local)
def _euler012(r: np.ndarray) -> tuple[float, float, float]:
    """Eigen 3.4.0 MatrixBase::eulerAngles(0,1,2) (EulerAngles.h Tait-Bryan 분기), libm 스칼라."""
    i, j, k = 0, 1, 2   # a0=0, a1=1 -> odd = 0
    r0 = math.atan2(r[j, k], r[k, k])
    c2 = math.sqrt(r[i, i] * r[i, i] + r[i, j] * r[i, j])   # Vector2::norm()
    if r0 > 0.0:        # (!odd) && res[0] > 0
        r0 -= math.pi
        r1 = math.atan2(-r[i, k], -c2)
    else:
        r1 = math.atan2(-r[i, k], c2)
    s1, c1 = math.sin(r0), math.cos(r0)
    r2 = math.atan2(s1 * r[k, i] - c1 * r[j, i], c1 * r[j, j] - s1 * r[k, j])
    return -r0, -r1, -r2   # !odd -> res = -res


def select_keyframes(pose_txt: Path):
    """KittiPointCloudSelect.cpp:262-303(자세 읽기)과 :440-466(2 m 선택)을 옮긴다."""
    cam_wrt_lidar = np.linalg.inv(LIDAR_WRT_CAM)
    cam_inv = np.linalg.inv(cam_wrt_lidar)
    rows, poses = [], []
    for ln in pose_txt.read_text().splitlines():
        f = ln.split()
        if len(f) < 12:
            break
        cam = np.eye(4)
        cam[:3, :4] = np.array([float(x) for x in f[:12]]).reshape(3, 4)
        lid = (cam_wrt_lidar @ cam) @ cam_inv
        poses.append(lid)
    xyz32 = np.array([[np.float32(p[0, 3]), np.float32(p[1, 3]), np.float32(p[2, 3])] for p in poses],
                     dtype=np.float32)
    last = np.array([-1e10, -1e10, 0.0], dtype=np.float32)
    sel, dists = [], []
    for c in range(len(poses)):
        d = xyz32[c] - last                                   # float 뺄셈
        dist = np.sqrt(np.float32(d[0] * d[0] + d[1] * d[1]) + np.float32(d[2] * d[2]))
        dists.append(float(dist))
        if dist < KEYFRAME_DIST:
            continue
        sel.append(c)
        last = xyz32[c]
    lines = []
    for kf, c in enumerate(sel):
        p = poses[c]
        rm = p[:3, :3]
        e = _euler012(rm)
        vals = [float(xyz32[c, 0]), float(xyz32[c, 1]), float(xyz32[c, 2]),
                float(np.float32(e[0])), float(np.float32(e[1])), float(np.float32(e[2]))]
        vals += [float(v) for v in rm.reshape(-1)]
        lines.append(f"{c:06d}," + ",".join(f"{v:.6f}" for v in vals) + "\n")
    # 2 m 경계 근처(선택이 float 반올림에 민감할 수 있는) 비교 횟수
    d = np.array(dists)
    near = int(np.sum(np.abs(d - 2.0) < 1e-5))
    return sel, "".join(lines), xyz32, near, len(poses)


def cmd_select():
    pose_txt = LOCAL_KITTI / "poses" / f"{SEQ}.txt"
    times = LOCAL_KITTI / "sequences" / SEQ / "times.txt"
    velo = LOCAL_KITTI / "sequences" / SEQ / "velodyne"
    sel, csv, xyz32, near, n_pose = select_keyframes(pose_txt)
    n_times = len(times.read_text().split())
    n_bin = len(list(velo.glob("*.bin")))
    assert n_pose == n_times == n_bin, (n_pose, n_times, n_bin)   # 공식 :427-430 개수 일치 조건
    d = OUT / "keyframes"
    d.mkdir(parents=True, exist_ok=True)
    (d / "keyframe_pose.csv").write_text(csv)
    path_len = float(np.linalg.norm(np.diff(xyz32.astype(np.float64), axis=0), axis=1).sum())
    rec = {"seq": f"kitti/{SEQ}", "n_scans": n_pose, "n_keyframes": len(sel), "scan_ids": sel,
           "prefix_scans": PREFIX_SCANS,
           "n_keyframes_in_prefix": int(sum(1 for s in sel if s < PREFIX_SCANS)),
           "near_threshold_comparisons_1e-5m": near, "trajectory_length_m": round(path_len, 1),
           "inputs": {"poses": str(pose_txt), "poses_sha256": sha256(pose_txt),
                      "times": str(times), "times_sha256": sha256(times), "velodyne": str(velo)},
           "keyframe_pose_csv_sha256": hashlib.sha256(csv.encode()).hexdigest(),
           "rule": "KittiPointCloudSelect.cpp:262-303,399-403,440-466 (float 3D 거리 < 2.0f면 건너뜀, 첫 프레임 포함)",
           "created_utc": now()}
    write_json(d / "keyframes.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "scan_ids"}, ensure_ascii=False))


# ------------------------------------------------------------------ produce (local)
def cmd_produce(start: int):
    kf = rjson(OUT / "keyframes" / "keyframes.json")
    sids = kf["scan_ids"]
    velo = LOCAL_KITTI / "sequences" / SEQ / "velodyne"
    out = sys.stdout.buffer
    hdr = json.dumps({"seq": f"kitti/{SEQ}", "family": FAMILY, "n": len(sids), "start": start,
                      "keyframe_pose_csv_sha256": kf["keyframe_pose_csv_sha256"]}).encode()
    out.write(len(hdr).to_bytes(4, "little") + hdr)
    for i in range(start, len(sids)):
        out.write(codec.pack(i, int(sids[i]), FAMILY, (velo / f"{sids[i]:06d}.bin").read_bytes()))
    out.write(codec.REC.pack(-1, len(sids), 0, 0, b"\0" * 32))
    out.flush()


# ------------------------------------------------------------------ recv (server)
def _read(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"stream truncated ({len(b)}/{n})")
    return b


def du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def cmd_recv(start: int):
    WORK.mkdir(parents=True, exist_ok=True)
    prog_p = WORK / "progress.json"
    prog = rjson(prog_p) or {"committed": 0}
    if (WORK / "stream_done.json").exists():
        sys.stdin.buffer.read()
        print("ALREADY_DONE", file=sys.stderr)
        return 0
    if start != prog["committed"]:
        print(f"RESUME_MISMATCH start={start} committed={prog['committed']}", file=sys.stderr)
        return 76
    f = sys.stdin.buffer
    hdr = json.loads(_read(f, int.from_bytes(_read(f, 4), "little")))
    kf = rjson(WORK / "keyframes.json")
    assert hdr["keyframe_pose_csv_sha256"] == kf["keyframe_pose_csv_sha256"], "keyframe list differs"
    sids = kf["scan_ids"]
    for p, unit in (("bev.u8", BEV_BYTES), ("stats.i32", 16)):
        with open(WORK / p, "ab") as fh:
            fh.truncate(prog["committed"] * unit)
    velo_fix = WORK / "prefix_fixture" / "velodyne"     # 앞부분 키프레임은 공식 선택기 대조용으로 남긴다
    velo_fix.mkdir(parents=True, exist_ok=True)
    shm = Path(f"/dev/shm/fresco_native_{os.getpid()}")
    shm.mkdir(parents=True, exist_ok=True)
    drv = subprocess.Popen([str(DRIVER), str(shm)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=open(WORK / "driver_stderr.log", "ab"))
    fb, fs = open(WORK / "bev.u8", "ab"), open(WORK / "stats.i32", "ab")
    done, t0 = prog["committed"], time.time()
    bb, sb = [], []

    def commit():
        nonlocal bb, sb
        fb.write(b"".join(bb)); fs.write(b"".join(sb))
        fb.flush(); fs.flush(); os.fsync(fb.fileno()); os.fsync(fs.fileno())
        bb, sb = [], []
        write_json(prog_p, {"committed": done, "n": len(sids), "updated_utc": now()})
    try:
        while True:
            idx, sid, plen, olen, dig = codec.REC.unpack(_read(f, codec.REC.size))
            if idx == -1:
                assert sid == len(sids) and done == len(sids), (sid, done)
                break
            assert idx == done and sid == sids[idx], (idx, done, sid)
            official = codec.decode(FAMILY, zlib.decompress(_read(f, plen)))
            assert len(official) == olen and codec.sha(official) == dig, f"sha mismatch at {idx}"
            if sid < PREFIX_SCANS:
                (velo_fix / f"{sid:06d}.bin").write_bytes(official)
            drv.stdin.write(len(official).to_bytes(4, "little") + official)
            drv.stdin.flush()
            o = drv.stdout.read(BEV_BYTES + 16)
            assert len(o) == BEV_BYTES + 16, "driver died"
            bb.append(o[:BEV_BYTES]); sb.append(o[BEV_BYTES:])
            done += 1
            if done % COMMIT_EVERY == 0:
                commit()
                assert du(NAS) < NAS_BUDGET, "NAS budget"
        commit()
    finally:
        try:
            drv.stdin.write((0).to_bytes(4, "little")); drv.stdin.close()
        except Exception:
            pass
        drv.wait(timeout=60)
        fb.close(); fs.close()
        shutil.rmtree(shm, ignore_errors=True)   # 이 실행이 만든 RAM 작업 폴더
    st = np.fromfile(WORK / "stats.i32", dtype="<i4").reshape(-1, 4)
    assert len(st) == done == len(sids) and os.path.getsize(WORK / "bev.u8") == done * BEV_BYTES
    write_json(WORK / "stream_done.json", {
        "n": done, "bev_sha256": sha256(WORK / "bev.u8"), "driver": str(DRIVER),
        "driver_sha256": sha256(DRIVER), "empty_bev_frames": int((st[:, 3] == 0).sum()),
        "points_mean": float(st[:, 0].mean()), "bev_nonzero_mean": float(st[:, 3].mean()),
        "sec_this_run": round(time.time() - t0, 1), "finished_utc": now()})
    print(f"RECV_OK {done}", file=sys.stderr)
    return 0


# ------------------------------------------------------------------ server_run (server)
def _matlab(script: str, cwd: Path, log: Path, timeout: int) -> int:
    env = {k: v for k, v in os.environ.items() if k != "TMPDIR"}   # MathWorks IPC는 로컬 /tmp 필요
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with open(log, "a") as lf:
        lf.write(f"\n=== {now()}\n"); lf.flush()
        try:
            r = subprocess.run(["nice", "-n", "10", "ionice", "-c3", MATLAB, "-nodisplay", "-singleCompThread",
                                "-batch", script], stdin=subprocess.DEVNULL, stdout=lf,
                               stderr=subprocess.STDOUT, env=env, cwd=str(cwd), timeout=timeout)
        except subprocess.TimeoutExpired:
            return -999
    return r.returncode


def stage_prefix(st: dict):
    """E1·E2: 앞 PREFIX_SCANS 스캔에 공식 선택기·batch_multi_bev_gen을 무수정 실행해 대조."""
    out = WORK / "prefix_check.json"
    if out.exists():
        return rjson(out)
    fx = WORK / "prefix_fixture"
    kf = rjson(WORK / "keyframes.json")
    pre = [s for s in kf["scan_ids"] if s < PREFIX_SCANS]
    poses = (WORK / "inputs" / "poses.txt").read_text().splitlines(keepends=True)[:PREFIX_SCANS]
    times = (WORK / "inputs" / "times.txt").read_text().splitlines(keepends=True)[:PREFIX_SCANS]
    (fx / "global_pose.txt").write_text("".join(poses))
    (fx / "times.txt").write_text("".join(times))
    log = open(WORK / "logs" / "prefix_official.log", "a")
    r1 = subprocess.run(["nice", "-n", "10", str(OFF_BIN / "kitti_point_cloud_select"), str(fx), "2"],
                        stdout=log, stderr=log)
    kfd = fx / "selected_keyframes_2.00m"
    r2 = subprocess.run(["nice", "-n", "10", str(OFF_BIN / "batch_multi_bev_gen"), str(kfd), "HDL_64E"],
                        stdout=log, stderr=log) if r1.returncode == 0 else None
    mine = (WORK / "keyframe_pose.csv").read_text().splitlines(keepends=True)[:len(pre)]
    off = (kfd / "keyframe_pose.csv").read_text().splitlines(keepends=True) if r1.returncode == 0 else []
    xy_eq = [a.split(",")[:3] == b.split(",")[:3] for a, b in zip(mine, off)]
    csvs = sorted((kfd / "output_single_bev/csv").glob("*.csv")) if r2 and r2.returncode == 0 else []
    bev = np.fromfile(WORK / "bev.u8", dtype=np.uint8, count=len(pre) * BEV_BYTES).reshape(-1, 224, 224)
    mism = [int((np.loadtxt(c, delimiter=",").astype(np.int64) != bev[i]).sum()) for i, c in enumerate(csvs)]
    rec = {"prefix_scans": PREFIX_SCANS, "n_keyframes_port": len(pre), "n_keyframes_official": len(off),
           "selector_rc": r1.returncode, "bevgen_rc": None if r2 is None else r2.returncode,
           "E1_keyframe_pose_csv_text_equal": off == mine,
           "E1_idx_xyz_fields_equal": len(off) == len(mine) and all(xy_eq),
           "E2_n_csv": len(csvs), "E2_bev_bit_equal": len(csvs) == len(pre) and sum(mism) == 0,
           "E2_mismatch_cells": int(sum(mism)),
           "binaries_sha256": {p.name: sha256(p) for p in (OFF_BIN / "kitti_point_cloud_select",
                                                          OFF_BIN / "batch_multi_bev_gen")},
           "note": "공식 선택기 입력 bin은 codec official bytes(intensity 0). KITTI 선택기는 intensity를 -1로 덮어쓴다(:237-238).",
           "finished_utc": now()}
    if not rec["E1_keyframe_pose_csv_text_equal"]:
        rec["E1_first_diff"] = next(([i, a, b] for i, (a, b) in enumerate(zip(mine, off)) if a != b), None)
    write_json(out, rec)
    return rec


def stage_official(st: dict):
    """공식 createDescsAndRetrieve.m 무수정 실행(waitbar·visualizeMatchPairs만 그리지 않는 shim)."""
    done = WORK / "official" / "official_done.json"
    if done.exists():
        return rjson(done)
    od = WORK / "official"
    n = rjson(WORK / "stream_done.json")["n"]
    csvd = od / "kf/output_single_bev/csv"
    csvd.mkdir(parents=True, exist_ok=True)
    bev = np.fromfile(WORK / "bev.u8", dtype=np.uint8).reshape(n, 224, 224)
    for i in range(n):   # cv::format FMT_CSV와 같은 정수 텍스트(기존 C2 대조와 같은 방식)
        p = csvd / f"{i:06d}.csv"
        if not p.exists():
            tmp = p.with_suffix(".tmp")
            np.savetxt(tmp, bev[i], fmt="%d", delimiter=", ")
            os.replace(tmp, p)
    shutil.copyfile(WORK / "keyframe_pose.csv", od / "kf/keyframe_pose.csv")
    for sub in ("cached_descriptors", "pr_result", "export"):
        if (od / sub).exists() and sub != "export":
            shutil.rmtree(od / sub)        # 이 단계가 만든 산출물만 다시 만든다(재개 시)
    (od / "export").mkdir(exist_ok=True)
    ex = od / "export"
    script = (f"addpath('{OFFICIAL_M}'); addpath('{GUI_SHIM}'); cd('{od}'); "
              f"createDescsAndRetrieve('{csvd}/', '{od}/kf/keyframe_pose.csv', 'KITTI08', [70;0.000700000000000000], false); "
              f"k = load('./cached_descriptors/KITTI08/cell_keys.mat'); f = load('./cached_descriptors/KITTI08/cell_frescos.mat'); "
              f"K = cell2mat(cellfun(@(x) x', k.cell_keys, 'UniformOutput', false)); "
              f"F = zeros(120, 20, numel(f.cell_fresco)); for ii = 1:numel(f.cell_fresco), F(:,:,ii) = f.cell_fresco{{ii}}'; end; "
              f"fid = fopen('{ex}/keys.f64', 'w'); fwrite(fid, K', 'double'); fclose(fid); "
              f"fid = fopen('{ex}/fresco.f64', 'w'); fwrite(fid, F, 'double'); fclose(fid); "
              f"C = []; for nm = {{'num_true_pos','num_false_pos','num_true_neg','num_false_neg'}}, s = load(['./pr_result/KITTI08/' nm{{1}} '.mat']); C = [C; s.(nm{{1}})]; end; "
              f"fid = fopen('{ex}/counts.f64', 'w'); fwrite(fid, C', 'double'); fclose(fid); disp('OFFICIAL_EXPORT_OK');")
    t0 = time.time()
    rc = _matlab(script, od, WORK / "logs" / "official_matlab.log", 6 * 3600)
    rec = {"rc": rc, "n": n, "sec": round(time.time() - t0, 1),
           "official_sha256": {p: sha256(OFFICIAL_M / p) for p in ("createDescsAndRetrieve.m", "ImToPolar.m",
                                                                   "drawPresicionRecallCurve.m")},
           "gui_shim_sha256": {p.name: sha256(p) for p in sorted(GUI_SHIM.glob("*.m"))},
           "matlab": MATLAB, "finished_utc": now()}
    if rc != 0:
        raise RuntimeError(f"official MATLAB rc={rc}")
    rec["counts_sha256"] = sha256(ex / "counts.f64")
    # 공식 PR 함수(drawPresicionRecallCurve.m) 무수정 실행: 그림 출력이 막혀도 계산은 위 카운트로 같다
    script2 = (f"addpath('{OFFICIAL_M}'); cd('{od}'); set(0,'DefaultFigureVisible','off'); "
               f"try, drawPresicionRecallCurve('KITTI08'); disp('OFFICIAL_PR_OK'); "
               f"catch e, disp(['OFFICIAL_PR_ERR ' e.message]); end; "
               f"F1 = load('./pr_result/KITTI08/F1scores.mat'); fid = fopen('{ex}/f1_official.f64','w'); "
               f"fwrite(fid, F1.F1scores, 'double'); fclose(fid);")
    rc2 = _matlab(script2, od, WORK / "logs" / "official_pr.log", 1800)
    rec["pr_rc"] = rc2
    txt = (WORK / "logs" / "official_pr.log").read_text()
    rec["pr_log_max_f1"] = [ln for ln in txt.splitlines() if ln.startswith("max_f1score")][-1:] or None
    write_json(done, rec)
    return rec


def stage_adapter(st: dict, gap: int):
    """질의별 승자·L1·cos_row(공식 함수는 내보내지 않음): 기존 fresco_adapter.m(공식 줄 해시 확인본)."""
    d = WORK / f"adapter_gap{gap}"
    done = d / "match_done.json"
    if done.exists():
        return rjson(done)
    d.mkdir(parents=True, exist_ok=True)
    n = rjson(WORK / "stream_done.json")["n"]
    if gap == 100:
        if not (d / "bev.u8").exists():
            os.symlink(WORK / "bev.u8", d / "bev.u8")
        mode = "all"
    else:
        for f in ("keys.f64", "fresco.f64"):
            if not (d / f).exists():
                os.symlink(WORK / "adapter_gap100" / f, d / f)
        mode = "match"
    t0 = time.time()
    rc = _matlab(f"addpath('{ADAPTER_DIR}'); fresco_adapter('{mode}', '{d}', {n}, {gap});", d,
                 d / "matlab.log", 6 * 3600)
    if rc != 0:
        raise RuntimeError(f"adapter gap{gap} rc={rc}")
    for f, size in (("keys.f64", n * 320), ("fresco.f64", n * 19200), ("match.f64", n * 48), ("cand.i32", n * 80)):
        assert os.path.getsize(d / f) == size, (f, size)
    rec = {"gap": gap, "n": n, "mode": mode, "sec": round(time.time() - t0, 1),
           "adapter_sha256": sha256(ADAPTER_DIR / "fresco_adapter.m"),
           "match_sha256": sha256(d / "match.f64"), "finished_utc": now()}
    write_json(done, rec)
    return rec


def cmd_server_run():
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    lockf = open(WORK / "server_run.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("server_run already running", file=sys.stderr)
        return 0
    (WORK / "server_run.pid").write_text(str(os.getpid()))
    status = WORK / "server_status.json"
    st = {"pid": os.getpid(), "started_utc": now()}
    stages = [("prefix_check", stage_prefix), ("official", stage_official),
              ("adapter_gap100", lambda s: stage_adapter(s, 100)),
              ("adapter_gap30", lambda s: stage_adapter(s, 30))]
    try:
        assert (WORK / "stream_done.json").exists(), "stream not done"
        for name, fn in stages:
            st.update(stage=name, updated_utc=now()); write_json(status, st)
            st[name] = fn(st)
        st.update(stage="SERVER_DONE", nas_bytes=du(NAS), updated_utc=now())
        assert st["nas_bytes"] < NAS_BUDGET, "NAS budget"
        write_json(status, st)
        write_json(WORK / "SERVER_DONE.json", st)
    except Exception as e:   # noqa: BLE001
        st.update(stage="SERVER_FAILED", error=f"{type(e).__name__}: {e}", updated_utc=now())
        write_json(status, st)
        write_json(WORK / "SERVER_FAILED.json", st)
        return 1
    return 0


def cmd_server_status():
    p = rjson(WORK / "progress.json")
    print(json.dumps({"committed": p.get("committed", 0),
                      "stream_done": (WORK / "stream_done.json").exists(),
                      "server_done": (WORK / "SERVER_DONE.json").exists(),
                      "server_failed": (WORK / "SERVER_FAILED.json").exists(),
                      "status": rjson(WORK / "server_status.json")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    for c in ("produce", "recv"):
        sub.add_parser(c).add_argument("--start", type=int, default=0)
    sub.add_parser("server_run")
    sub.add_parser("server_status")
    a = ap.parse_args()
    if a.cmd == "select":
        cmd_select()
    elif a.cmd == "produce":
        cmd_produce(a.start)
    elif a.cmd == "recv":
        sys.exit(cmd_recv(a.start))
    elif a.cmd == "server_run":
        sys.exit(cmd_server_run())
    elif a.cmd == "server_status":
        cmd_server_status()
