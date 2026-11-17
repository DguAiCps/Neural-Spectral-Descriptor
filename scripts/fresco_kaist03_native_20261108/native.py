#!/usr/bin/env python3
"""FreSCo 원 논문 조건 두 번째 재현: MulRan KAIST03 (2026-11-08).

논문 표I(ICARCV 2022) KAIST03 최대F1 0.9866을 공개 코드와 공개 전처리 그대로의 원 조건
(GT 2 m 키프레임, 2D 10 m, 과거 키프레임 1..q-100, 후보 20, 50쌍 임계, 공식 PR 산식)으로
다시 낸다. KITTI08(scripts/fresco_native_20261108)과 공통 표2 규약 실행과 분리한다.

  select      (서버) 공식 mulran_point_cloud_select로 전체 keyframe_pose.csv(자리표시 점군)
  produce     (로컬) 키프레임 원시 bin -> 무손실 스트림(stdout), codec.pack 재사용
  recv        (서버) 스트림 -> 공식 전처리 드라이버(fresco_prep_mulran) -> NAS bev.u8, 이어받기
  server_run  (서버) 앞 150스캔 공식 실행 파일 대조 -> 공식 MATLAB 함수 무수정 실행 -> 어댑터 매칭
  controller  (로컬, controller.py) 위 단계를 이어서 실행, 재개 가능, 끝나면 완료/실패 알림 등록

공식 원천은 읽기 전용으로만 쓴다: _external/fresco_official_20261107(FreSCo 2bf5cee, 전처리
d94040e), 기존 NAS build/의 공식 실행 파일·드라이버·fresco_adapter.m, scripts/fresco_official/codec.py.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
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

SEQ = "KAIST03"
FAMILY = "mulran_os1"
SERVER = "joms-gpu"
SERVER_REPO = "/home/qkrrlans00/Neural-Spectral-Codec"
OLD_NAS = Path("/mnt/aicps-nas/qkrrlans00/nsd_fresco_official_20261107")
NAS = Path(os.environ.get("FRESCO_K03_NAS", "/mnt/aicps-nas/qkrrlans00/nsd_fresco_kaist03_native_20261108"))
NAS_BUDGET = 8 * 2 ** 30
WORK = NAS / "mulran_KAIST03"
MATLAB = "/mnt/aicps-nas/qkrrlans00/matlab/R2026a/bin/matlab"
OFFICIAL_M = Path(SERVER_REPO) / "_external/fresco_official_20261107/FreSCo/fresco-matlab"
GUI_SHIM = Path(SERVER_REPO) / "scripts/fresco_official/matlab_gui_shim"
ADAPTER_DIR = OLD_NAS / "build/matlab"
DRIVER = OLD_NAS / "build/fresco_prep_mulran"
OFF_BIN = OLD_NAS / "build/official"

LOCAL_SEQ = Path("/mnt/d/NSD_datasets/mulran/KAIST03")
OUT = Path(os.environ.get("FRESCO_K03_OUT",
                          "/mnt/d/NSD_local/results/pdv2_overnight_repro_20261108/fresco_kaist03"))
BEV_BYTES = 224 * 224
PREFIX_SCANS = 150          # 공식 선택기·BEV 실행 파일 대조에 쓰는 앞부분 스캔 수(사전 고정)
COMMIT_EVERY = 200
BEST_THRESHOLD = "[50;0.000500000000000000]"   # test4KASIT03.m:4 (시각화용, PR 무관)


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


def du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def sorted_stamps(path: Path) -> list[int]:
    """MulranPointCloudSelect.cpp:202-231과 같이 정수로 읽어 오름차순 정렬."""
    return sorted(int(x) for x in path.read_text().split())


# ------------------------------------------------------------------ select (server)
def cmd_select():
    """공식 선택기를 전체 시각·자세에 실행한다. 점군 파일 대신 16 B 자리표시를 링크한다."""
    done = WORK / "select_done.json"
    if done.exists():
        print(json.dumps(rjson(done)))
        return 0
    full = WORK / "full"
    stamps_p = full / "sensor_data" / "ouster_front_stamp.csv"
    stamps = sorted_stamps(stamps_p)
    ph = WORK / "placeholder_point.bin"
    ph.write_bytes(b"\0" * 16)
    od = full / "sensor_data" / "Ouster"
    od.mkdir(parents=True, exist_ok=True)
    for s in stamps:
        p = od / f"{s:010d}.bin"
        if not p.is_symlink():
            os.symlink(ph, p)
    log = open(WORK / "logs" / "select_official.log", "w")
    t0 = time.time()
    r = subprocess.run(["nice", "-n", "10", "ionice", "-c3", str(OFF_BIN / "mulran_point_cloud_select"), str(full), "2"],
                       stdout=log, stderr=log)
    kfd = full / "selected_keyframes_2.00m"
    csv = (kfd / "keyframe_pose.csv").read_text()
    lines = csv.splitlines()
    cidx = [int(ln.split(",")[0]) for ln in lines]
    assert all(b > a for a, b in zip(cidx, cidx[1:])), "cloud_idx not increasing"
    (WORK / "keyframe_pose.csv").write_text(csv)
    txt = (WORK / "logs" / "select_official.log").read_text()
    rec = {"seq": f"mulran/{SEQ}", "selector_rc": r.returncode, "sec": round(time.time() - t0, 1),
           "n_scans": len(stamps), "n_keyframes": len(lines), "cloud_idx": cidx,
           "scan_stamps": [stamps[c] for c in cidx],
           "n_could_not_find_pose": txt.count("Could not find pose"),
           "keyframe_pose_csv_sha256": hashlib.sha256(csv.encode()).hexdigest(),
           "inputs_sha256": {"global_pose.csv": sha256(full / "global_pose.csv"),
                             "ouster_front_stamp.csv": sha256(stamps_p)},
           "selector_sha256": sha256(OFF_BIN / "mulran_point_cloud_select"),
           "placeholder": "Ouster/<stamp>.bin -> 16 B 0점 1개(선택은 자세·시각만 사용, E1에서 실제 원시로 대조)",
           "finished_utc": now()}
    assert r.returncode == 0, f"selector rc={r.returncode}"
    write_json(WORK / "keyframes.json", rec)
    write_json(done, {k: v for k, v in rec.items() if k not in ("cloud_idx", "scan_stamps")})
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cloud_idx", "scan_stamps")}))
    return 0


# ------------------------------------------------------------------ produce (local)
def cmd_produce(start: int):
    kf = rjson(OUT / "keyframes" / "keyframes.json")
    stamps = kf["scan_stamps"]
    velo = LOCAL_SEQ / "Ouster"
    out = sys.stdout.buffer
    hdr = json.dumps({"seq": f"mulran/{SEQ}", "family": FAMILY, "n": len(stamps), "start": start,
                      "keyframe_pose_csv_sha256": kf["keyframe_pose_csv_sha256"]}).encode()
    out.write(len(hdr).to_bytes(4, "little") + hdr)
    for i in range(start, len(stamps)):
        out.write(codec.pack(i, int(kf["cloud_idx"][i]), FAMILY, (velo / f"{stamps[i]:010d}.bin").read_bytes()))
    out.write(codec.REC.pack(-1, len(stamps), 0, 0, b"\0" * 32))
    out.flush()


# ------------------------------------------------------------------ recv (server)
def _read(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"stream truncated ({len(b)}/{n})")
    return b


def cmd_recv(start: int):
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
    cidx = kf["cloud_idx"]
    for p, unit in (("bev.u8", BEV_BYTES), ("stats.i32", 16)):
        with open(WORK / p, "ab") as fh:
            fh.truncate(prog["committed"] * unit)
    shm = Path(f"/dev/shm/fresco_k03_{os.getpid()}")
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
        write_json(prog_p, {"committed": done, "n": len(cidx), "updated_utc": now()})
    try:
        while True:
            idx, sid, plen, olen, dig = codec.REC.unpack(_read(f, codec.REC.size))
            if idx == -1:
                assert sid == len(cidx) and done == len(cidx), (sid, done)
                break
            assert idx == done and sid == cidx[idx], (idx, done, sid)
            official = codec.decode(FAMILY, zlib.decompress(_read(f, plen)))
            assert len(official) == olen and codec.sha(official) == dig, f"sha mismatch at {idx}"
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
    assert len(st) == done == len(cidx) and os.path.getsize(WORK / "bev.u8") == done * BEV_BYTES
    write_json(WORK / "stream_done.json", {
        "n": done, "bev_sha256": sha256(WORK / "bev.u8"), "driver": str(DRIVER),
        "driver_sha256": sha256(DRIVER), "empty_bev_frames": int((st[:, 3] == 0).sum()),
        "points_mean": float(st[:, 0].mean()), "bev_nonzero_mean": float(st[:, 3].mean()),
        "sec_this_run": round(time.time() - t0, 1), "finished_utc": now()})
    print(f"RECV_OK {done}", file=sys.stderr)
    return 0


# ------------------------------------------------------------------ server_run (server)
def _env():
    env = {k: v for k, v in os.environ.items() if k != "TMPDIR"}   # MathWorks IPC는 로컬 /tmp 필요
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return env


def _matlab_cmd(script: str):
    return ["nice", "-n", "10", "ionice", "-c3", MATLAB, "-nodisplay", "-singleCompThread", "-batch", script]


def _matlab(script: str, cwd: Path, log: Path, timeout: int) -> int:
    with open(log, "a") as lf:
        lf.write(f"\n=== {now()}\n"); lf.flush()
        try:
            r = subprocess.run(_matlab_cmd(script), stdin=subprocess.DEVNULL, stdout=lf,
                               stderr=subprocess.STDOUT, env=_env(), cwd=str(cwd), timeout=timeout)
        except subprocess.TimeoutExpired:
            return -999
    return r.returncode


def _driver_bev(raws: list[bytes]) -> bytes:
    """같은 원시를 codec official bytes(intensity 0)로 바꿔 스트리밍 드라이버에 넣는다(recv와 같은 경로)."""
    shm = Path(f"/dev/shm/fresco_k03_prefix_{os.getpid()}")
    shm.mkdir(parents=True, exist_ok=True)
    drv = subprocess.Popen([str(DRIVER), str(shm)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=open(WORK / "logs" / "prefix_driver_stderr.log", "ab"))
    out = []
    try:
        for raw in raws:
            _, official = codec.encode(FAMILY, raw)
            drv.stdin.write(len(official).to_bytes(4, "little") + official)
            drv.stdin.flush()
            o = drv.stdout.read(BEV_BYTES + 16)
            assert len(o) == BEV_BYTES + 16, "driver died"
            out.append(o[:BEV_BYTES])
    finally:
        try:
            drv.stdin.write((0).to_bytes(4, "little")); drv.stdin.close()
        except Exception:
            pass
        drv.wait(timeout=60)
        shutil.rmtree(shm, ignore_errors=True)
    return b"".join(out)


def cmd_prefix():
    """E1·E2(전체 실행 전 관문): 앞 PREFIX_SCANS 스캔 실제 원시에 공식 선택기·batch_multi_bev_gen을
    무수정 실행하고, 같은 원시를 스트리밍 드라이버에 넣은 BEV와 비트 대조한다."""
    out = WORK / "prefix_check.json"
    if out.exists():
        rec = rjson(out)
        print(json.dumps(rec))
        return 0 if rec.get("gate_pass") else 1
    fx = WORK / "prefix_fixture"
    stamps = sorted_stamps(WORK / "full" / "sensor_data" / "ouster_front_stamp.csv")[:PREFIX_SCANS]
    (fx / "sensor_data").mkdir(parents=True, exist_ok=True)
    (fx / "sensor_data" / "ouster_front_stamp.csv").write_text("".join(f"{s}\n" for s in stamps))
    shutil.copyfile(WORK / "full" / "global_pose.csv", fx / "global_pose.csv")
    raw = fx / "sensor_data" / "Ouster"
    missing = [s for s in stamps if not (raw / f"{s:010d}.bin").exists()]
    assert not missing, f"prefix raw missing {len(missing)}"
    log = open(WORK / "logs" / "prefix_official.log", "a")
    r1 = subprocess.run(["nice", "-n", "10", str(OFF_BIN / "mulran_point_cloud_select"), str(fx), "2"],
                        stdout=log, stderr=log)
    kfd = fx / "selected_keyframes_2.00m"
    r2 = subprocess.run(["nice", "-n", "10", str(OFF_BIN / "batch_multi_bev_gen"), str(kfd), "OS1_64"],
                        stdout=log, stderr=log) if r1.returncode == 0 else None
    off = (kfd / "keyframe_pose.csv").read_text().splitlines(keepends=True) if r1.returncode == 0 else []
    full = (WORK / "keyframe_pose.csv").read_text().splitlines(keepends=True)[:len(off)]
    csvs = sorted((kfd / "output_single_bev/csv").glob("*.csv")) if r2 and r2.returncode == 0 else []
    pidx = [int(ln.split(",")[0]) for ln in off]
    raws = [(raw / f"{stamps[c]:010d}.bin").read_bytes() for c in pidx]
    b = _driver_bev(raws)
    (WORK / "prefix_driver_bev.u8").write_bytes(b)
    bev = np.frombuffer(b, dtype=np.uint8).reshape(-1, 224, 224)
    mism = [int((np.loadtxt(c, delimiter=",").astype(np.int64) != bev[i]).sum()) for i, c in enumerate(csvs)]
    nfull = rjson(WORK / "keyframes.json")["cloud_idx"]
    rec = {"prefix_scans": PREFIX_SCANS, "n_keyframes_prefix_official": len(off),
           "n_keyframes_full_in_prefix": int(sum(1 for c in nfull if c < PREFIX_SCANS)),
           "prefix_cloud_idx": pidx,
           "selector_rc": r1.returncode, "bevgen_rc": None if r2 is None else r2.returncode,
           "E1_prefix_csv_text_equal_full_head": len(off) > 0 and off == full,
           "E2_n_csv": len(csvs), "E2_bev_bit_equal": len(csvs) == len(off) > 0 and sum(mism) == 0,
           "E2_mismatch_cells": int(sum(mism)), "E2_nonzero_cells_mean": float((bev > 0).sum((1, 2)).mean()) if len(bev) else None,
           "prefix_driver_bev_sha256": hashlib.sha256(b).hexdigest(),
           "binaries_sha256": {p.name: sha256(p) for p in (OFF_BIN / "mulran_point_cloud_select",
                                                          OFF_BIN / "batch_multi_bev_gen", DRIVER)},
           "note": "공식 실행은 원래 intensity를 가진 실제 원시 bin, 드라이버는 같은 원시의 codec official bytes(intensity 0).",
           "finished_utc": now()}
    rec["E1_prefix_count_equal"] = rec["n_keyframes_prefix_official"] == rec["n_keyframes_full_in_prefix"]
    if not rec["E1_prefix_csv_text_equal_full_head"]:
        rec["E1_first_diff"] = next(([i, a, b] for i, (a, b) in enumerate(zip(off, full)) if a != b), None)
    rec["gate_pass"] = bool(rec["E1_prefix_csv_text_equal_full_head"] and rec["E1_prefix_count_equal"]
                            and rec["E2_bev_bit_equal"])
    write_json(out, rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "prefix_cloud_idx"}))
    return 0 if rec["gate_pass"] else 1


def stage_stream_head(st: dict):
    """E2b: 전체 스트림 bev.u8 앞 키프레임 == 관문에서 같은 드라이버로 만든 앞부분 BEV(비트)."""
    pre = rjson(WORK / "prefix_check.json")
    k = pre["n_keyframes_prefix_official"]
    a = np.fromfile(WORK / "bev.u8", dtype=np.uint8, count=k * BEV_BYTES)
    b = np.fromfile(WORK / "prefix_driver_bev.u8", dtype=np.uint8)
    return {"n": k, "E2b_stream_head_bit_equal": bool(len(a) == len(b) == k * BEV_BYTES and np.array_equal(a, b))}


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
    for i in range(n):   # cv::format FMT_CSV와 같은 정수 텍스트(KITTI08과 같은 방식)
        p = csvd / f"{i:06d}.csv"
        if not p.exists():
            tmp = p.with_suffix(".tmp")
            np.savetxt(tmp, bev[i], fmt="%d", delimiter=", ")
            os.replace(tmp, p)
    shutil.copyfile(WORK / "keyframe_pose.csv", od / "kf/keyframe_pose.csv")
    for sub in ("cached_descriptors", "pr_result"):
        if (od / sub).exists():
            shutil.rmtree(od / sub)        # 이 단계가 만든 산출물만 다시 만든다(재개 시)
    ex = od / "export"
    ex.mkdir(exist_ok=True)
    script = (f"addpath('{OFFICIAL_M}'); addpath('{GUI_SHIM}'); cd('{od}'); "
              f"createDescsAndRetrieve('{csvd}/', '{od}/kf/keyframe_pose.csv', '{SEQ}', {BEST_THRESHOLD}, false); "
              f"k = load('./cached_descriptors/{SEQ}/cell_keys.mat'); f = load('./cached_descriptors/{SEQ}/cell_frescos.mat'); "
              f"K = cell2mat(cellfun(@(x) x', k.cell_keys, 'UniformOutput', false)); "
              f"F = zeros(120, 20, numel(f.cell_fresco)); for ii = 1:numel(f.cell_fresco), F(:,:,ii) = f.cell_fresco{{ii}}'; end; "
              f"fid = fopen('{ex}/keys.f64', 'w'); fwrite(fid, K', 'double'); fclose(fid); "
              f"fid = fopen('{ex}/fresco.f64', 'w'); fwrite(fid, F, 'double'); fclose(fid); "
              f"C = []; for nm = {{'num_true_pos','num_false_pos','num_true_neg','num_false_neg'}}, s = load(['./pr_result/{SEQ}/' nm{{1}} '.mat']); C = [C; s.(nm{{1}})]; end; "
              f"fid = fopen('{ex}/counts.f64', 'w'); fwrite(fid, C', 'double'); fclose(fid); disp('OFFICIAL_EXPORT_OK');")
    t0 = time.time()
    rc = _matlab(script, od, WORK / "logs" / "official_matlab.log", 6 * 3600)
    rec = {"rc": rc, "n": n, "sec": round(time.time() - t0, 1),
           "official_sha256": {p: sha256(OFFICIAL_M / p) for p in ("createDescsAndRetrieve.m", "ImToPolar.m",
                                                                   "drawPresicionRecallCurve.m", "test4KASIT03.m")},
           "gui_shim_sha256": {p.name: sha256(p) for p in sorted(GUI_SHIM.glob("*.m"))},
           "matlab": MATLAB, "finished_utc": now()}
    if rc != 0:
        raise RuntimeError(f"official MATLAB rc={rc}")
    rec["counts_sha256"] = sha256(ex / "counts.f64")
    write_json(done, rec)
    return rec


def stage_official_pr(st: dict):
    """공식 drawPresicionRecallCurve.m 무수정 실행. 그림 출력(print -dpdf)은 KITTI08에서 DISPLAY 없이 멈췄다.
    F1/P/R .mat 저장(:72-74)과 max_f1score 출력이 확인된 뒤에도 120 s 안에 끝나지 않으면 이 MATLAB만 종료한다."""
    done = WORK / "official" / "official_pr_done.json"
    if done.exists():
        return rjson(done)
    od = WORK / "official"
    prd = od / "pr_result" / SEQ
    for nm in ("F1scores.mat", "Precisions.mat", "Recalls.mat"):
        if (prd / nm).exists():
            (prd / nm).unlink()            # 이 단계가 만드는 파일만(재개 시)
    log = WORK / "logs" / "official_pr.log"
    script = (f"addpath('{OFFICIAL_M}'); cd('{od}'); set(0,'DefaultFigureVisible','off'); "
              f"try, drawPresicionRecallCurve('{SEQ}'); disp('OFFICIAL_PR_OK'); "
              f"catch e, disp(['OFFICIAL_PR_ERR ' e.message]); end;")
    t0 = time.time()
    with open(log, "a") as lf:
        lf.write(f"\n=== {now()}\n"); lf.flush()
        p = subprocess.Popen(_matlab_cmd(script), stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                             env=_env(), cwd=str(od), start_new_session=True)
    seen, terminated = None, False
    while p.poll() is None and time.time() - t0 < 3600:
        time.sleep(5)
        ok = all((prd / nm).exists() for nm in ("F1scores.mat", "Precisions.mat", "Recalls.mat")) \
            and "max_f1score" in log.read_text()
        if ok and seen is None:
            seen = time.time()
        if seen is not None and time.time() - seen > 120:
            os.killpg(p.pid, 15)           # 이 단계가 띄운 MATLAB 세션만
            terminated = True
            p.wait(timeout=60)
            break
    if p.poll() is None:
        os.killpg(p.pid, 15); p.wait(timeout=60)
        raise RuntimeError("official PR timed out before .mat")
    txt = log.read_text()
    rec = {"rc": p.returncode, "figure_stage_terminated_after_mat": terminated, "sec": round(time.time() - t0, 1),
           "pr_log_max_f1": [ln for ln in txt.splitlines() if ln.startswith("max_f1score")][-1:] or None,
           "mat_sha256": {nm: sha256(prd / nm) for nm in ("F1scores.mat", "Precisions.mat", "Recalls.mat")
                          if (prd / nm).exists()},
           "finished_utc": now()}
    if len(rec["mat_sha256"]) != 3:
        raise RuntimeError("official PR .mat missing")
    write_json(done, rec)
    return rec


def stage_adapter(st: dict, gap: int = 100):
    """질의별 승자·L1·cos_row(공식 함수는 내보내지 않음): 기존 fresco_adapter.m(공식 줄 해시 확인본)."""
    d = WORK / f"adapter_gap{gap}"
    done = d / "match_done.json"
    if done.exists():
        return rjson(done)
    d.mkdir(parents=True, exist_ok=True)
    n = rjson(WORK / "stream_done.json")["n"]
    if not (d / "bev.u8").exists():
        os.symlink(WORK / "bev.u8", d / "bev.u8")
    t0 = time.time()
    rc = _matlab(f"addpath('{ADAPTER_DIR}'); fresco_adapter('all', '{d}', {n}, {gap});", d, d / "matlab.log", 6 * 3600)
    if rc != 0:
        raise RuntimeError(f"adapter gap{gap} rc={rc}")
    for f, size in (("keys.f64", n * 320), ("fresco.f64", n * 19200), ("match.f64", n * 48), ("cand.i32", n * 80)):
        assert os.path.getsize(d / f) == size, (f, size)
    rec = {"gap": gap, "n": n, "mode": "all", "sec": round(time.time() - t0, 1),
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
    stages = [("stream_head_check", stage_stream_head), ("official", stage_official),
              ("official_pr", stage_official_pr), ("adapter_gap100", stage_adapter)]
    try:
        assert (WORK / "stream_done.json").exists(), "stream not done"
        for name, fn in stages:
            st.update(stage=name, updated_utc=now()); write_json(status, st)
            st[name] = fn(st)
            assert du(NAS) < NAS_BUDGET, "NAS budget"
        st.update(stage="SERVER_DONE", nas_bytes=du(NAS), updated_utc=now())
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
                      "select_done": (WORK / "select_done.json").exists(),
                      "stream_done": (WORK / "stream_done.json").exists(),
                      "server_done": (WORK / "SERVER_DONE.json").exists(),
                      "server_failed": (WORK / "SERVER_FAILED.json").exists(),
                      "status": rjson(WORK / "server_status.json")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    sub.add_parser("prefix")
    for c in ("produce", "recv"):
        sub.add_parser(c).add_argument("--start", type=int, default=0)
    sub.add_parser("server_run")
    sub.add_parser("server_status")
    a = ap.parse_args()
    if a.cmd == "select":
        sys.exit(cmd_select())
    elif a.cmd == "prefix":
        sys.exit(cmd_prefix())
    elif a.cmd == "produce":
        cmd_produce(a.start)
    elif a.cmd == "recv":
        sys.exit(cmd_recv(a.start))
    elif a.cmd == "server_run":
        sys.exit(cmd_server_run())
    elif a.cmd == "server_status":
        cmd_server_status()
