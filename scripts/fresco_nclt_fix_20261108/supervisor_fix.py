#!/usr/bin/env python3
"""NCLT9 수정 재평가 로컬 감시기 (로컬 호스트, 2026-11-08). 기존 fresco_official/supervisor.py와 helios local_watch의
규칙을 따르되 후속 Opus를 띄우지 않는다(자식 agent 금지).

  setsid nohup python3 scripts/fresco_nclt_fix_20261108/supervisor_fix.py [--interval 600] &
중복 방지: OUT/supervisor.lock(flock). --interval초마다
  * 로컬 스트림 controller(stream_fix.py controller)가 /proc에 있는지 보고, 스트림이 남았는데 없으면 다시 띄운다(최대 5회).
  * ssh로 서버 master_fix.py status를 읽고, 끝 표시 없이 죽었으면 다시 띄운다(최대 3회). 통신 오류를 성공으로 보지 않는다.
  * 새 NAS의 작은 산출(results/ 원장·상태 json, test/*/*.json, logs/)만 rsync로 OUT/mirror/에 받는다(삭제 없음).
  * 스트림 확정 프레임·원장 수가 6회 연속 늘지 않으면 stalled로 기록한다(프로세스는 건드리지 않는다).
끝(한 번):
  done    서버 ALL_LEDGERS_DONE + 서버 원장 sha256 = mirror sha256 + aggregate_fix.py 통과 -> OUT/COMPUTE_DONE.json
  failed  서버 FAILED, 재시작 한도 초과, 집계 검산 실패, ssh 연속 12회 실패 -> OUT/COMPUTE_FAILED.json(원인)
알림: OUT/notify_name.json(root가 정한 이름)이 있을 때만 completion_bridge arm/watch를 그 이름으로 1회 한다.
없으면 표지 파일만 쓰고 root 감시에 맡긴다(새 감시 이름을 스스로 만들지 않는다).
상태: OUT/supervisor_status.json, OUT/supervisor.heartbeat, OUT/supervisor.log
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixenv  # noqa: E402

OUT = fixenv.LOCAL_OUT
SERVER = "joms-gpu"
SREPO = "/home/qkrrlans00/Neural-Spectral-Codec"
SPY = "scripts/fresco_nclt_fix_20261108"
NAS = str(fixenv.NEW_NAS)
MIRROR = OUT / "mirror"
STATUS = OUT / "supervisor_status.json"
BRIDGE = OUT.parent / "completion_bridge.py"
D_ENV = {"TMPDIR": "/mnt/d/NSD_local/tmp", "TMP": "/mnt/d/NSD_local/tmp", "TEMP": "/mnt/d/NSD_local/tmp",
         "XDG_CACHE_HOME": "/mnt/d/NSD_local/cache", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
         "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg):
    with open(OUT / "supervisor.log", "a") as f:
        f.write(f"{now()} {msg}\n")


def ssh(cmd, timeout=180):
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", SERVER, cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 255, "", "timeout"


def controller_pids():
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(c.endswith(b"stream_fix.py") for c in cmd) and b"controller" in cmd:
            out.append(int(p.name))
    return out


def start_controller():
    lf = open(OUT / "controller.log", "a")
    env = dict(os.environ, **D_ENV)
    p = subprocess.Popen(["nice", "-n", "10", sys.executable, str(fixenv.HERE / "stream_fix.py"), "controller"],
                         stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                         cwd=str(OUT), start_new_session=True)
    return p.pid


def start_master():
    return ssh(f"cd {SREPO} && PYTHONDONTWRITEBYTECODE=1 setsid nohup nice -n 10 ionice -c3 python3 {SPY}/master_fix.py run "
               f">> {NAS}/logs/master.log 2>&1 < /dev/null & echo started")


def rsync_small():
    MIRROR.mkdir(parents=True, exist_ok=True)
    cmd = ["timeout", "900", "rsync", "-a", "-m", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20",
           "--include=*/", "--include=results/**", "--include=logs/**", "--include=test/*/*.json",
           "--include=test/*/*.log", "--include=test_*", "--include=build/build_fix.json", "--exclude=*",
           f"{SERVER}:{NAS}/", str(MIRROR) + "/"]
    return subprocess.run(cmd, capture_output=True).returncode


def notify(kind):
    p = OUT / "notify_name.json"
    if not p.exists():
        log(f"{kind}: notify_name.json 없음 -> 표지 파일만 기록(root 감시)")
        return True
    name = json.loads(p.read_text())[kind]
    sent = OUT / f"notified_{name}"
    if sent.exists():
        return True
    state = OUT.parent / "completion_monitor" / f"watch_{name}.json"
    if not state.exists():
        with open(OUT / "supervisor.log", "a") as f:
            r = subprocess.run([sys.executable, str(BRIDGE), "arm", "--root", str(OUT.parent), "--name", name,
                                "--workers", "fresco_fix_pipeline", "--live", "--max-hours", "24"], stdout=f, stderr=f)
        if r.returncode != 0 or not state.exists():
            log(f"arm {name} failed rc={r.returncode}")
            return False
    sent.write_text(now())
    with open(OUT / "supervisor.log", "a") as f:
        subprocess.Popen([sys.executable, str(BRIDGE), "watch", "--root", str(OUT.parent), "--name", name,
                          "--interval", "60"], stdout=f, stderr=f, start_new_session=True)
    log(f"armed {name}")
    return True


def finish(state, ok, why, extra=None):
    rec = {"utc": now(), "ok": ok, "reason": why, **(extra or {})}
    (OUT / ("COMPUTE_DONE.json" if ok else "COMPUTE_FAILED.json")).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    state.update(stage="compute_done_verified" if ok else "compute_failed", finished=rec)
    fixenv_write(STATUS, state)
    ws = OUT / "worker_status.json"
    try:
        w = json.loads(ws.read_text())
    except (OSError, ValueError):
        w = {}
    w.update(updated_utc=now(), server_compute_running=False,
             stage=("4_compute_done_local_aggregate_passed (보고서 작성 전)" if ok else f"4_compute_FAILED: {why}"),
             compute_marker=str(OUT / ("COMPUTE_DONE.json" if ok else "COMPUTE_FAILED.json")))
    fixenv_write(ws, w)
    log(f"finish ok={ok}: {why}")
    for _ in range(12):
        if notify("done" if ok else "failed"):
            break
        time.sleep(300)


def fixenv_write(p: Path, obj):
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, p)


def check_done(state):
    rc, out, err = ssh(f"cd {NAS}/results/ledgers/test && sha256sum dump_fresco_nclt_*.npz")
    if rc != 0:
        return None
    remote = {ln.split()[1]: ln.split()[0] for ln in out.strip().splitlines()}
    if len(remote) != 9:
        return False, f"server ledgers {len(remote)} != 9", {}
    if rsync_small() != 0:
        return None
    fixenv_write(MIRROR / "ledger_sha256_server.json", remote)
    local = {fn: fixenv.sha256_file(MIRROR / "results/ledgers/test" / fn) for fn in remote}
    if local != remote:
        return False, "mirror ledger sha256 != server", {"remote": remote, "local": local}
    env = dict(os.environ, **D_ENV)
    r = subprocess.run(["nice", "-n", "10", sys.executable, str(fixenv.HERE / "aggregate_fix.py")],
                       capture_output=True, text=True, env=env, cwd=str(OUT))
    (OUT / "aggregate_fix.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        return False, f"aggregate_fix rc={r.returncode}", {}
    fs = json.loads((OUT / "final/fix_summary.json").read_text())
    return bool(fs["passed"]), "server 9 ledgers + mirror sha256 + aggregate_fix passed", {
        "ledger_sha256": remote, "fix_summary": str(OUT / "final/fix_summary.json"),
        "nclt9_score_fresco_percent": fs["scores"]["score_fresco"]["nclt9"]["ann_arbor_mean_percent"],
        "all46_score_fresco_macro_percent": fs["scores"]["score_fresco"]["all46_macro_percent"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600)
    a = ap.parse_args()
    lock = open(OUT / "supervisor.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another supervisor_fix holds the lock", file=sys.stderr)
        return 3
    if (OUT / "COMPUTE_DONE.json").exists() or (OUT / "COMPUTE_FAILED.json").exists():
        print("terminal marker exists; not restarting", file=sys.stderr)
        return 0
    state = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    state.update(pid=os.getpid(), started_utc=now(), interval=a.interval)
    state.setdefault("restarts", {"controller": 0, "master": 0})
    state.setdefault("log", [])
    ssh_fail, still, last_prog = 0, 0, None
    log(f"supervisor_fix start pid={os.getpid()}")
    while True:
        rc, out, err = ssh(f"cd {SREPO} && PYTHONDONTWRITEBYTECODE=1 python3 {SPY}/master_fix.py status")
        if rc != 0 or not out.strip():
            ssh_fail += 1
            log(f"ssh status failed rc={rc} ({ssh_fail}) {err[-200:]}")
            if ssh_fail >= 12:
                finish(state, False, f"server unreachable {ssh_fail} times")
                return 1
            time.sleep(a.interval)
            continue
        ssh_fail = 0
        st = json.loads(out.strip().splitlines()[-1])
        streams = st["streams"]
        n_streams_done = sum(v["done"] for v in streams.values())
        n_ledgers = sum(v["ledger"] for v in streams.values())
        committed = sum(v["committed"] for v in streams.values())
        cpids = controller_pids()
        state.update(updated_utc=now(), server=st, controller_pids=cpids, streams_done=n_streams_done,
                     ledgers_done=n_ledgers, committed_frames=committed, frames_total=49529)
        if st["markers"]["FAILED"]:
            rsync_small()
            finish(state, False, "server FAILED marker", {"server_failed": st.get("failed")})
            return 1
        if st["markers"]["ALL_LEDGERS_DONE"]:
            res = check_done(state)
            if res is None:
                log("check_done transient failure; retry")
            else:
                ok, why, extra = res
                finish(state, ok, why, extra)
                return 0 if ok else 1
        else:
            if n_streams_done < 9 and not cpids:
                if state["restarts"]["controller"] >= 5:
                    finish(state, False, "controller restart limit"); return 1
                pid = start_controller()
                state["restarts"]["controller"] += 1
                log(f"controller started pid={pid} (restart {state['restarts']['controller']})")
            if not st["alive"]:
                if state["restarts"]["master"] >= 3:
                    finish(state, False, "master restart limit"); return 1
                r = start_master()
                state["restarts"]["master"] += 1
                log(f"master start rc={r[0]} {r[1].strip()[-80:]} {r[2].strip()[-200:]} (restart {state['restarts']['master']})")
        prog = (committed, n_ledgers)
        still = still + 1 if prog == last_prog else 0
        last_prog = prog
        state["stalled"] = still >= 6
        state["rsync_rc"] = rsync_small()
        fixenv_write(STATUS, state)
        (OUT / "supervisor.heartbeat").write_text(
            f"{now()} streams_done={n_streams_done}/9 ledgers={n_ledgers}/9 frames={committed}/49529 "
            f"master_alive={st['alive']} controller={cpids} stalled={state['stalled']}\n")
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
