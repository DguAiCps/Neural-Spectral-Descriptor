#!/usr/bin/env python3
"""seqot_stable_gem 서버 계산의 로컬(D) 감시기. 서버 계산은 바꾸지 않는다.

10분마다 서버 server_status.json, COMPUTE_DONE/FAILED, supervisor 생존을 읽고 작은 결과
(records, flags, logs, stage2 checkpoint; 시험 서술자 desc 제외)를 D의 server_mirror/로 받는다.
- ssh·rsync 실패는 기록하고 다음 주기에 다시 시도한다(통신 실패만으로 실패 판정하지 않는다).
- supervisor가 완료 파일 없이 사라지면 같은 supervisor를 최대 3번 다시 띄운다(단계는 멱등이라 이어서 한다).
- COMPUTE_DONE.json + 그 주기 rsync 성공 + 로컬 검산(원장 해시, 공통 질의 재집계) 통과: 완료 신호 1회.
- COMPUTE_FAILED.json, 로컬 검산 불일치, 재시작 한도 초과, 감시 기한 초과: 실패 신호 1회.
신호는 completion_bridge arm 성공 뒤 sentinel(notified_<이름>)을 쓰고 watch를 띄운다.
중복 실행은 flock으로 막는다.
"""
from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOCAL_REPO = Path("/home/pkm/Neural-Spectral-Codec")
OVN = Path("/mnt/d/NSD_local/results/pdv2_overnight_repro_20261108")
WORKER = "seqot_stable_gem"
HERE = OVN / WORKER
RES = Path("/mnt/d/NSD_local/results/pdv2_seqot_stable_gem_20261108")
MIRROR = RES / "server_mirror"
SRV = "joms-gpu"
SREPO = "/home/qkrrlans00/Neural-Spectral-Codec"
SW = f"{SREPO}/_external/seqot_stable_gem_20261108"
BRIDGE = OVN / "completion_bridge.py"
INTERVAL = 600
DEADLINE_H = 40
MAX_RELAUNCH = 3
OK_NAME, FAIL_NAME = "ovn_seqot_stable_compute_20261108", "ovn_seqot_stable_fail_20261108"
START = (f"cd {SREPO} && nohup setsid bash _external/seqot_stable_gem_20261108/code/"
         f"_seqot_stable_gem_20261108_supervisor.sh >> {SW}/logs/supervisor.log 2>&1 < /dev/null & echo $!")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg):
    with open(HERE / "local_watch.log", "a") as f:
        f.write(f"{now()} {msg}\n")


def rjson(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


def update(path, **kw):
    d = rjson(path)
    d.update(updated_utc=now(), **kw)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    tmp.replace(path)


def ssh(cmd, timeout=120):
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", SRV, cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return 255, ""


def mirror():
    try:
        r = subprocess.run(["rsync", "-a", "--exclude=desc/", "--exclude=*.tmp", "--exclude=supervisor.lock",
                            f"{SRV}:{SW}/", str(MIRROR) + "/"], capture_output=True, timeout=900)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def local_check():
    """미러의 원장 해시와 공통 질의 재집계를 로컬 파이썬(numpy)으로 다시 확인한다."""
    import hashlib
    import importlib.util
    import numpy as np
    rep = {"utc": now()}
    ver, summ = rjson(MIRROR / "records/verification.json"), rjson(MIRROR / "records/summary.json")
    rep["server_verification_passed"] = bool(ver.get("passed"))
    hashes_ok = True
    for seed, lst in (ver.get("ledger_sha256") or {}).items():
        for name, h in lst.items():
            p = MIRROR / f"records/s{seed}/odom/{name}"
            hashes_ok &= p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == h
    rep["ledger_hashes_match"] = bool(hashes_ok and len(ver.get("ledger_sha256") or {}) == 3)
    sys.path[:0] = [str(LOCAL_REPO), str(LOCAL_REPO / "scripts"), str(LOCAL_REPO / "src")]
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("_agg_local", LOCAL_REPO / "scripts/_pdv2_main_baselines_official.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    M.REPO = MIRROR
    M.COMMON = LOCAL_REPO / "results/pdv2_symmetric3_20261013"
    diffs, counts = {}, {}
    for seed in ("1", "2", "3"):
        e = M.aggregate(f"records/s{seed}/odom", "dump_seqotl2", "score_l2", "pr_summary_baselines.json",
                        "seqotl2", lambda p: Path(p))
        counts[seed] = [e["n_queries"], e["n_positive"]]
        diffs[seed] = max(abs(e["macro"][k] - summ["per_seed"][seed]["macro"][k]) for k in M.METRICS)
    vals = np.array([[summ["per_seed"][s]["macro"][k] for k in M.METRICS] for s in ("1", "2", "3")])
    mean_diff = max(abs(vals[:, i].mean() - summ["mean_std"][k]["mean"]) for i, k in enumerate(M.METRICS))
    rep.update(local_macro_max_abs_diff=diffs, counts=counts, mean_max_abs_diff=float(mean_diff))
    rep["passed"] = bool(rep["server_verification_passed"] and rep["ledger_hashes_match"]
                         and all(v < 1e-12 for v in diffs.values()) and mean_diff < 1e-12
                         and all(c == [195726, 61682] for c in counts.values()))
    (RES / "local_check.json").write_text(json.dumps(rep, indent=1))
    return rep["passed"]


def notify(name):
    if (HERE / f"notified_{name}").exists():
        return True
    state = OVN / "completion_monitor" / f"watch_{name}.json"
    if not state.exists():
        with open(HERE / "local_watch.log", "a") as f:
            r = subprocess.run([sys.executable, str(BRIDGE), "arm", "--root", str(OVN), "--name", name,
                                "--workers", WORKER, "--live", "--max-hours", "24"], stdout=f, stderr=f)
        if r.returncode != 0 or not state.exists():
            log(f"arm {name} failed rc={r.returncode}; retry next cycle")
            return False
    (HERE / f"notified_{name}").write_text(now())
    log(f"armed {name}; watch")
    with open(HERE / "local_watch.log", "a") as f:
        subprocess.Popen([sys.executable, str(BRIDGE), "watch", "--root", str(OVN), "--name", name,
                          "--interval", "60"], stdout=f, stderr=f, start_new_session=True)
    return True


def finish(ok, why):
    name = OK_NAME if ok else FAIL_NAME
    st = rjson(MIRROR / "server_status.json")
    update(HERE / "done.json", compute_complete=ok, compute_state="complete" if ok else "failed",
           compute_seen_utc=now(), reason=why, mirror=str(MIRROR), server_stage=st.get("stage"),
           local_check=str(RES / "local_check.json"))
    update(HERE / "worker_status.json", running=False, running_processes=[],
           stage=("server compute complete + local check pass (COMPUTE_DONE)" if ok
                  else f"server compute FAILED: {why}"),
           remaining_dependencies=["root verification", "experiment log + wiki/manuscript integration (root)"],
           next_action=f"root: {MIRROR}/{{COMPUTE_DONE,COMPUTE_FAILED}}.json, records/{{summary,verification}}.json, "
                       f"{RES}/local_check.json 검산")
    log(f"finish ok={ok}: {why}")
    while not notify(name):
        time.sleep(300)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    MIRROR.mkdir(parents=True, exist_ok=True)
    lock = open(HERE / "local_watch.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("local watcher already running")
        return 0
    (HERE / "local_watch.pid").write_text(f"{__import__('os').getpid()}\n")
    deadline = datetime.now(timezone.utc) + timedelta(hours=DEADLINE_H)
    misses = relaunch = dead = 0
    log("local watcher start")
    while True:
        rc, out = ssh(f"cat {SW}/server_status.json 2>/dev/null; echo '@@'; "
                      f"test -f {SW}/COMPUTE_DONE.json && echo DONE; "
                      f"test -f {SW}/COMPUTE_FAILED.json && echo FAILED; "
                      f"test -d /proc/$(cat {SW}/supervisor.pid 2>/dev/null || echo 0) && echo ALIVE; "
                      f"df -BG --output=avail / | tail -1")
        if rc != 0 and not out:
            misses += 1
            log(f"ssh failed ({misses}); retry")
            (HERE / "local_watch.heartbeat").write_text(f"{now()} ssh_fail={misses}\n")
            if datetime.now(timezone.utc) > deadline:
                finish(False, f"watch deadline {DEADLINE_H} h passed (ssh failing)")
                return 0
            time.sleep(INTERVAL)
            continue
        misses = 0
        tail = out.split("@@")[-1].split()
        synced = mirror()
        st = rjson(MIRROR / "server_status.json")
        (HERE / "local_watch.heartbeat").write_text(
            f"{now()} {' '.join(tail)} stage={st.get('stage')} status={st.get('status')} note={st.get('note')} "
            f"mirror={'ok' if synced else 'fail'}\n")
        if "DONE" in tail and synced:
            try:
                ok = local_check()
            except Exception as e:  # noqa: BLE001
                log(f"local check error {e!r}")
                ok = False
            finish(ok, "COMPUTE_DONE.json and local check pass" if ok else "COMPUTE_DONE.json but local check failed")
            return 0
        if "FAILED" in tail and synced:
            finish(False, "COMPUTE_FAILED.json (see server_mirror/COMPUTE_FAILED.json)")
            return 0
        if {"ALIVE", "DONE", "FAILED"} & set(tail):
            dead = 0
        else:
            dead += 1
            if dead >= 2:
                if relaunch >= MAX_RELAUNCH:
                    finish(False, f"supervisor gone without COMPUTE_DONE/FAILED after {relaunch} relaunches")
                    return 0
                rc2, pid = ssh(START)
                relaunch += 1
                dead = 0
                log(f"supervisor gone; relaunch {relaunch} rc={rc2} pid={pid.strip()}")
        if datetime.now(timezone.utc) > deadline:
            finish(False, f"watch deadline {DEADLINE_H} h passed; last stage {st.get('stage')}")
            return 0
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
