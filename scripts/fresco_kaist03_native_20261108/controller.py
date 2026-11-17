#!/usr/bin/env python3
"""FreSCo MulRan KAIST03 원 조건 재현의 로컬 controller와 분석 (2026-11-08).

  python3 controller.py run        로컬 D에서 detach 실행(nohup setsid). flock으로 하나만 돈다.
                                   설치 -> 서버 공식 선택기 -> 앞 150스캔 원시 복사 -> 스트리밍(이어받기)
                                   -> 서버 server_run(서버에서 detach) -> 감시 -> 작은 산출 미러
                                   -> analyze -> done.json -> 완료/실패 알림
  python3 controller.py analyze    미러된 산출만으로 지표·검산을 다시 계산(summary.json)

서버 계산은 서버에서 setsid로 떠 있어 WSL이 멈춰도 이어진다. controller를 다시 띄우면
완료 표시(select_done, stream_done, *_done.json, SERVER_DONE)를 보고 남은 단계만 이어간다.
알림: 계산 완료(+로컬 검산 통과) 또는 실패 때 completion_bridge.py arm(--root 절대 D 경로) 성공 뒤
sentinel(notified_<name>)을 쓰고 watch를 띄운다. 이 이름은 root의 API 종료 감시와 다르다.
분석 함수는 scripts/fresco_native_20261108/controller.py(KITTI08)의 같은 산식을 옮겼다(원본 무수정).
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native as N  # noqa: E402

OUT = N.OUT
ROOT = OUT.parent
BRIDGE = ROOT / "completion_bridge.py"
MIRROR = OUT / "server_mirror"
SRV_CODE = f"{N.SERVER_REPO}/scripts/fresco_kaist03_native_20261108"
PAPER_F1, PAPER_P, PAPER_R = 0.9866, 0.9866, 0.9866
REF_COMMON = Path("/home/pkm/Neural-Spectral-Codec/results/pdv2_fresco_official_20261107/pipeline")
NAME_OK = "ovn_fresco_kaist03_compute_20261108"
NAME_FAIL = "ovn_fresco_kaist03_fail_20261108"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg):
    with open(OUT / "controller.log", "a") as f:
        f.write(f"{now()} {msg}\n")


def status(**kw):
    p = OUT / "worker_status.json"
    d = N.rjson(p)
    d.update(role="fresco_kaist03", updated_utc=now(), **kw)
    N.write_json(p, d)


def ssh(cmd, timeout=600, **kw):
    return subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                           "-o", "ServerAliveInterval=30", N.SERVER, cmd], timeout=timeout, **kw)


def srv_status():
    try:
        r = ssh(f"cd {N.SERVER_REPO} && python3 {SRV_CODE}/native.py server_status", timeout=120,
                capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return None
    return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None


# ------------------------------------------------------------------ stages
def setup():
    w = str(N.WORK)
    ssh(f"mkdir -p {SRV_CODE} {w}/full/sensor_data {w}/prefix_fixture/sensor_data/Ouster {w}/logs", check=True)
    for f in ("native.py", "controller.py"):
        subprocess.run(["scp", "-q", str(HERE / f), f"{N.SERVER}:{SRV_CODE}/{f}"], check=True)
    st = srv_status() or {}
    if not st.get("select_done"):
        for src, dst in ((N.LOCAL_SEQ / "global_pose.csv", "full/global_pose.csv"),
                         (N.LOCAL_SEQ / "ouster_front_stamp.csv", "full/sensor_data/ouster_front_stamp.csv")):
            subprocess.run(["scp", "-q", str(src), f"{N.SERVER}:{w}/{dst}"], check=True)
    log("setup ok")


def select():
    st = srv_status() or {}
    if not st.get("select_done"):
        status(stage="server official selector (full sequence)", running_processes=["local controller", "server selector"])
        r = ssh(f"cd {N.SERVER_REPO} && python3 {SRV_CODE}/native.py select", timeout=3600,
                capture_output=True, text=True)
        log(f"select rc={r.returncode} {r.stdout.strip()[-400:]} {r.stderr.strip()[-400:]}")
        if r.returncode != 0:
            return False
    kd = OUT / "keyframes"
    kd.mkdir(exist_ok=True)
    for f in ("keyframes.json", "keyframe_pose.csv", "select_done.json"):
        subprocess.run(["scp", "-q", f"{N.SERVER}:{N.WORK}/{f}", str(kd / f)], check=True)
    kf = N.rjson(kd / "keyframes.json")
    import hashlib
    assert hashlib.sha256((kd / "keyframe_pose.csv").read_bytes()).hexdigest() == kf["keyframe_pose_csv_sha256"]
    log(f"select ok n_keyframes={kf['n_keyframes']}")
    return True


def upload_prefix():
    """앞 PREFIX_SCANS 스캔의 실제 원시 bin(원래 intensity 포함)을 그대로 복사(rsync 1스트림)."""
    stamps = N.sorted_stamps(N.LOCAL_SEQ / "ouster_front_stamp.csv")[:N.PREFIX_SCANS]
    lst = OUT / "prefix_files.txt"
    lst.write_text("".join(f"{s:010d}.bin\n" for s in stamps))
    subprocess.run(["nice", "-n", "10", "rsync", "-a", "--checksum", f"--files-from={lst}", f"{N.LOCAL_SEQ}/Ouster/",
                    f"{N.SERVER}:{N.WORK}/prefix_fixture/sensor_data/Ouster/"], check=True, timeout=3600)
    log("prefix raw upload ok")


def prefix_gate():
    """전체 스트리밍 전 관문: 공식 선택기·BEV 실행 파일 vs 자리표시 전체 선택·드라이버(E1·E2)."""
    status(stage="prefix gate (official selector/bevgen vs driver, first 150 scans)",
           running_processes=["local controller", "server prefix check"])
    r = ssh(f"cd {N.SERVER_REPO} && nice -n 10 ionice -c3 python3 {SRV_CODE}/native.py prefix", timeout=3600,
            capture_output=True, text=True)
    log(f"prefix gate rc={r.returncode} {r.stdout.strip()[-600:]} {r.stderr.strip()[-400:]}")
    subprocess.run(["scp", "-q", f"{N.SERVER}:{N.WORK}/prefix_check.json", str(OUT / "prefix_check.json")])
    return r.returncode == 0


def stream():
    for attempt in range(6):
        st = srv_status()
        if st is None:
            log("server status failed; retry in 60 s"); time.sleep(60); continue
        if st["stream_done"]:
            return True
        c = st["committed"]
        status(stage=f"streaming from keyframe {c}", running_processes=["local controller+producer", "server recv+driver"])
        log(f"stream start={c} attempt={attempt}")
        prod = subprocess.Popen(["nice", "-n", "10", sys.executable, str(HERE / "native.py"), "produce",
                                 "--start", str(c)], stdout=subprocess.PIPE, stderr=open(OUT / "produce.log", "a"))
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", N.SERVER,
                            f"cd {N.SERVER_REPO} && nice -n 10 ionice -c3 python3 {SRV_CODE}/native.py recv --start {c}"],
                           stdin=prod.stdout, stderr=open(OUT / "recv.log", "a"))
        prod.stdout.close(); prod.wait()
        log(f"stream rc={r.returncode} producer rc={prod.returncode}")
        time.sleep(5)
    st = srv_status()
    return bool(st and st["stream_done"])


def server_run():
    relaunched = 0
    while True:
        st = srv_status()
        if st is None:
            time.sleep(60); continue
        if st["server_done"]:
            return True
        if st["server_failed"]:
            return False
        try:
            alive = ssh(f"test -f {N.WORK}/server_run.pid && kill -0 $(cat {N.WORK}/server_run.pid) 2>/dev/null",
                        timeout=60).returncode == 0
        except subprocess.TimeoutExpired:
            time.sleep(60); continue
        if not alive:
            if relaunched >= 3:
                return False
            try:   # 실행 호출의 timeout은 실패가 아니다(다음 주기에 서버 PID로 판정)
                ssh(f"cd {N.SERVER_REPO} && nohup setsid nice -n 10 ionice -c3 python3 {SRV_CODE}/native.py "
                    f"server_run >> {N.WORK}/logs/server_run.out 2>&1 < /dev/null &",
                    timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                log("server_run launch call timed out; checking PID next cycle")
            relaunched += 1
            log(f"server_run launched ({relaunched})")
            time.sleep(20)
            continue
        status(stage=f"server {st['status'].get('stage')}", running_processes=["local controller"],
               server_pids={"running": [st["status"].get("pid")]})
        time.sleep(30)


def mirror():
    MIRROR.mkdir(exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude=bev.u8", "--exclude=*.csv", "--exclude=prefix_fixture/",
                    "--exclude=full/", "--exclude=cached_descriptors/", "--exclude=*.pcd",
                    "--exclude=placeholder_point.bin",
                    f"{N.SERVER}:{N.WORK}/", f"{MIRROR}/"], check=True, timeout=1800)
    return True


# ------------------------------------------------------------------ analysis (local, mirrored files only)
def official_pr(tp, fp, tn, fn):
    """drawPresicionRecallCurve.m:42-59, 77 (MATLAB max는 NaN을 무시하고 첫 최대 위치)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
    i = int(np.nanargmax(f1))
    return {"max_F1": float(f1[i]), "idx_1based": i + 1, "precision": float(p[i]), "recall": float(r[i]),
            "F1_all": [None if not np.isfinite(x) else float(x) for x in f1],
            "n_ties_at_max": int(np.sum(f1 == f1[i]))}


def thresholds50():
    f = np.concatenate([np.linspace(0, 200, 41), np.linspace(200, 1000, 9)])
    c = np.concatenate([np.linspace(0, 0.002, 41), np.linspace(0.002, 0.010, 9)])
    return f, c


def native_counts(m, xy, gap=100, radius=10.0):
    """createDescsAndRetrieve.m:186-246 (1부터 번호), 50쌍 임계."""
    n = len(xy)
    f, c = thresholds50()
    tp, fp, tn, fn = (np.zeros(len(f), np.int64) for _ in range(4))
    for qi in range(1, n + 1):
        hist = xy[0:max(1, qi - gap)]
        is_rev = np.linalg.norm(hist - xy[qi - 1], axis=1).min() <= radius
        if qi - gap <= 0:
            best, l1, cr = -1, 1e10, 1e10
        else:
            best, l1, cr = int(m[qi - 1, 0]), m[qi - 1, 1], m[qi - 1, 2]
        acc = (l1 <= f) & (cr <= c)
        good = best > 0 and np.linalg.norm(xy[best - 1] - xy[qi - 1]) <= radius
        tp += acc & good; fp += acc & ~good
        fn += ~acc & is_rev; tn += ~acc & ~is_rev
    return tp, fp, tn, fn


def native_continuous(m, xy, gap=100, radius=10.0):
    """N1: 같은 산식, 임계만 스칼라 점수 s=-max(L1,1e5 cos_row)의 모든 고유값."""
    n = len(xy)
    q = np.arange(1, n + 1)
    has = q - gap > 0
    rev = np.array([np.linalg.norm(xy[0:max(1, k - gap)] - xy[k - 1], axis=1).min() <= radius for k in q])
    best = np.where(has, m[:, 0], -1).astype(int)
    s = np.where(has, -np.maximum(m[:, 1], 1e5 * m[:, 2]), -np.inf)
    good = np.zeros(n, bool)
    good[has] = np.linalg.norm(xy[best[has] - 1] - xy[has], axis=1) <= radius
    best_f1 = (-1.0, None)
    for t in np.unique(s[np.isfinite(s)])[::-1]:
        acc = s >= t
        tp = int((acc & good).sum()); fp = int((acc & ~good).sum()); fn = int((~acc & rev).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        p, r = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if f1 > best_f1[0]:
            best_f1 = (f1, {"tau": float(t), "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn})
    return {"max_F1": best_f1[0], **(best_f1[1] or {}), "n_queries": n, "n_revisit": int(rev.sum()),
            "n_with_candidates": int(has.sum())}


def analyze():
    import scipy.io as sio
    M = MIRROR
    kfcsv = np.loadtxt(OUT / "keyframes" / "keyframe_pose.csv", delimiter=",")
    n = len(kfcsv)
    xy = kfcsv[:, 1:3]
    ver = {}
    pre = N.rjson(M / "prefix_check.json")
    for k in ("E1_prefix_csv_text_equal_full_head", "E2_bev_bit_equal", "E2_mismatch_cells",
              "n_keyframes_prefix_official", "n_keyframes_full_in_prefix"):
        ver[k] = pre.get(k)
    ver["E1_prefix_count_equal"] = pre.get("n_keyframes_prefix_official") == pre.get("n_keyframes_full_in_prefix")
    ver["E2b_stream_head_bit_equal"] = N.rjson(M / "SERVER_DONE.json").get("stream_head_check", {}).get(
        "E2b_stream_head_bit_equal")
    sd = N.rjson(M / "stream_done.json")
    ver["stream_n_equal_keyframes"] = sd.get("n") == n
    ex = M / "official" / "export"
    C = np.fromfile(ex / "counts.f64", "<f8").reshape(4, 50).astype(np.int64)
    tp, fp, tn, fn = C
    ver["counts_sum_equal_keyframes"] = bool(np.all(tp + fp + tn + fn == n))
    n0 = official_pr(tp.astype(float), fp.astype(float), tn.astype(float), fn.astype(float))
    prd = M / "official" / "pr_result" / N.SEQ
    f1m = sio.loadmat(prd / "F1scores.mat")["F1scores"].ravel()
    ver["E6_official_F1scores_mat_equal_port"] = bool(np.array_equal(
        np.nan_to_num(f1m, nan=-1), np.nan_to_num(np.array([np.nan if v is None else v for v in n0["F1_all"]]), nan=-1)))
    pm = sio.loadmat(prd / "Precisions.mat")["Precisions"].ravel()
    rm = sio.loadmat(prd / "Recalls.mat")["Recalls"].ravel()
    i0 = n0["idx_1based"] - 1
    ver["official_mat_P_R_at_best_equal"] = bool(pm[i0] == n0["precision"] and rm[i0] == n0["recall"])
    ad = M / "adapter_gap100"
    ka = np.fromfile(ad / "keys.f64", "<f8").reshape(n, 40)
    ko = np.fromfile(ex / "keys.f64", "<f8").reshape(n, 40)
    ver["E3_keys_bit_equal"] = bool(np.array_equal(ka, ko, equal_nan=True))
    ver["E3_fresco_bit_equal"] = bool(np.array_equal(np.fromfile(ad / "fresco.f64", "<f8"),
                                                     np.fromfile(ex / "fresco.f64", "<f8"), equal_nan=True))
    ver["nonfinite_keys"] = int((~np.isfinite(ka)).any(1).sum())
    m100 = np.fromfile(ad / "match.f64", "<f8").reshape(n, 6)
    mine = native_counts(m100, xy)
    ver["E4_counts_equal"] = bool(all(np.array_equal(a, b) for a, b in zip(mine, (tp, fp, tn, fn))))
    n1 = native_continuous(m100, xy)
    ref = N.rjson(REF_COMMON / "summary.json")["scores"]["score_fresco"]["per_sequence"]["mulran/KAIST03"]
    d = n0["max_F1"] - PAPER_F1
    verdict = ("matches_reported_4dp" if abs(d) <= 0.00005 else
               "reproduced_within_tolerance" if abs(d) <= 0.01 else "not_reproduced")
    checks_ok = all(ver[k] is True for k in ("E1_prefix_csv_text_equal_full_head", "E1_prefix_count_equal",
                                               "E2_bev_bit_equal", "E2b_stream_head_bit_equal",
                                               "stream_n_equal_keyframes",
                                               "counts_sum_equal_keyframes", "E6_official_F1scores_mat_equal_port",
                                               "official_mat_P_R_at_best_equal", "E3_keys_bit_equal",
                                               "E3_fresco_bit_equal", "E4_counts_equal"))
    b = n0["idx_1based"] - 1
    summ = {"created_utc": now(), "seq": f"mulran/{N.SEQ}", "n_keyframes": n,
            "n_scans": N.rjson(OUT / "keyframes" / "keyframes.json").get("n_scans"),
            "paper": {"max_F1": PAPER_F1, "precision": PAPER_P, "recall": PAPER_R,
                      "source": "arXiv 2206.12628v2 PDF p.5 Table I"},
            "N0_native_official": {k: v for k, v in n0.items() if k != "F1_all"} | {
                "counts_at_best": {"tp": int(tp[b]), "fp": int(fp[b]), "tn": int(tn[b]), "fn": int(fn[b])},
                "thresholds_at_best": [float(thresholds50()[0][b]), float(thresholds50()[1][b])],
                "F1_all_50": n0["F1_all"],
                "TP_plus_FN_all_50": [int(x) for x in tp + fn]},
            "rounded": {"F1_4dp": round(n0["max_F1"], 4), "P_pct_2dp": round(100 * n0["precision"], 2),
                        "R_pct_2dp": round(100 * n0["recall"], 2)},
            "delta_vs_paper": {"max_F1": d, "precision": n0["precision"] - PAPER_P, "recall": n0["recall"] - PAPER_R},
            "verdict": verdict,
            "secondary": {"N1_native_all_thresholds": n1, "REF_common_table_KAIST03_quote_only": ref},
            "verification": ver, "verification_all_pass": bool(checks_ok)}
    N.write_json(OUT / "summary.json", summ)
    return summ


# ------------------------------------------------------------------ notify / main
def notify(name):
    if (OUT / f"notified_{name}").exists():
        return True
    state = ROOT / "completion_monitor" / f"watch_{name}.json"
    if not state.exists():
        with open(OUT / "controller.log", "a") as f:
            r = subprocess.run([sys.executable, str(BRIDGE), "arm", "--root", str(ROOT), "--name", name,
                                "--workers", OUT.name, "--live", "--max-hours", "24"], stdout=f, stderr=f)
        if r.returncode != 0 or not state.exists():
            log(f"arm {name} failed rc={r.returncode}; retry")
            return False
    (OUT / f"notified_{name}").write_text(now())
    log(f"armed {name}; watch")
    with open(OUT / "controller.log", "a") as f:
        p = subprocess.Popen([sys.executable, str(BRIDGE), "watch", "--root", str(ROOT), "--name", name,
                              "--interval", "60"], stdout=f, stderr=f, start_new_session=True)
    (OUT / f"watch_{name}.pid").write_text(str(p.pid))
    return True


def finish(ok, why, summ=None):
    name = NAME_OK if ok else NAME_FAIL
    N.write_json(OUT / "compute_done.json", {
        "compute_complete": ok, "reason": why, "utc": now(),
        "summary": "summary.json" if summ else None,
        "verdict": summ.get("verdict") if summ else None,
        "verification_all_pass": summ.get("verification_all_pass") if summ else None,
        "note": "계산 완료 표시(controller). 작업자 최종 done.json은 보고서 작성 뒤 따로 쓴다."})
    status(stage=("compute complete: server compute + local analysis done" if ok else f"FAILED: {why}"),
           running_processes=[], server_pids={"running": []},
           next_action="report_ko.txt·verification.json·done.json 작성(작업자) / root 검산")
    for _ in range(48):
        if notify(name):
            return
        time.sleep(300)


def run():
    lockf = open(OUT / "controller.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("controller already running"); return 0
    (OUT / "controller.pid").write_text(str(os.getpid()))
    log(f"controller start pid={os.getpid()}")
    try:
        setup()
        if not select():
            finish(False, "official selector failed"); return 1
        upload_prefix()
        if not prefix_gate():
            finish(False, "prefix gate failed (see prefix_check.json); full run not started"); return 1
        if not stream():
            finish(False, "stream failed"); return 1
        status(stage="stream done; server_run", running_processes=["local controller"])
        if not server_run():
            mirror(); finish(False, "server_run failed (see server_mirror/SERVER_FAILED.json)"); return 1
        mirror()
        summ = analyze()
        log(f"analyze verdict={summ['verdict']} F1={summ['N0_native_official']['max_F1']:.6f} "
            f"checks={summ['verification_all_pass']}")
        finish(bool(summ["verification_all_pass"]),
               "complete + local checks pass" if summ["verification_all_pass"] else "complete but checks failed", summ)
    except Exception as e:  # noqa: BLE001
        log(f"controller error {type(e).__name__}: {e}")
        finish(False, f"controller error {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        sys.exit(run())
    elif cmd == "analyze":
        print(json.dumps({k: v for k, v in analyze().items() if k != "secondary"}, indent=1, default=str))
