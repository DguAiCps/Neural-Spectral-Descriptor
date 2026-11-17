#!/usr/bin/env python3
"""공식 FreSCo 시험 46 실행의 로컬 감시기 (로컬 호스트, 2026-11-07).

  setsid nohup python3 scripts/fresco_official/supervisor.py [--interval 600] &
--interval초(기본 600)마다:
  * 로컬 스트림 controller(stream.py controller --role test)가 살아 있는지 /proc로 확인, 스트림이 남았는데
    죽었으면 다시 띄운다(최대 5회).
  * 제한된 SSH로 서버 master(server_master.py status)와 스트림 진행을 읽고, master가 끝 표시 없이 죽었으면
    다시 띄운다(최대 3회). 통신 오류를 성공으로 보지 않는다.
  * 서버의 작은 산출(원장 npz, 상태 json)만 rsync로 받는다(삭제 없음).
  * 진행(스트림 확정 프레임·원장 수)이 6회 연속 늘지 않으면 stalled로 기록한다.
끝 상태(한 번씩만):
  test_done  서버 ALL_LEDGERS_DONE + 로컬 aggregate.py 검산 통과 -> pipeline/TEST_DONE.json,
             확인·위키 반영용 후속 Opus 1회 실행 + _review_completion_monitor.py arm/watch(--workers pipeline)
  failed     서버 FAILED, 재시작 한도 초과, 집계 검산 실패, SSH 연속 실패 -> pipeline/FAILED.json(원인),
             승인 범위 안의 복구 Opus(최대 2회) + 같은 감시기 연결
후속 Opus 종료는 기존 완료 감시기가 root 대화에 알린다(이 감시기는 계산 완료/실패만 다룬다).
상태: results/pdv2_fresco_official_20261107/pipeline/supervisor_status.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, RESULTS, SERVER, SERVER_REPO, write_json  # noqa: E402

ROOTDIR = RESULTS.parent
STATUS = RESULTS / "supervisor_status.json"
CLI = "/mnt/d/NSD_tools/claude/2.1.280/claude"
MODEL = "claude-opus-5-5"
SPY = "scripts/fresco_official"
SRES = f"{SERVER_REPO}/results/pdv2_fresco_official_20261107/pipeline"
# 로컬 실행 위치·임시·캐시는 D(AGENTS.md 실행 위치 규칙). 결과 폴더는 D로 옮기고 원래 경로에 symlink를 둔다.
LOCAL_D = Path("/mnt/d/NSD_local")
D_ENV = {"TMPDIR": str(LOCAL_D / "tmp"), "TMP": str(LOCAL_D / "tmp"), "TEMP": str(LOCAL_D / "tmp"),
         "XDG_CACHE_HOME": str(LOCAL_D / "cache"), "PYTHONDONTWRITEBYTECODE": "1"}
FOLLOWUP_CWD = LOCAL_D / "work/fresco_followup"

PROMPT_COMMON = f"""repo 루트는 {REPO}이고 아래 상대 경로는 모두 이 repo 기준이다. 네 cwd는 D 작업 폴더({FOLLOWUP_CWD})이며
로컬 실행·임시·캐시·로그·생성물은 D에 둔다(C의 /home·/tmp·도커 볼륨에 새 큰 파일 금지, 무거운 계산은 서버).
""" + """너는 공식 FreSCo 시험46 평가 파이프라인의 후속 Opus 5.5 작업자다. 문서·설명은 한국어, git commit/push 금지.
기존 파일을 삭제하거나 실행 중인 다른 실험(SeqOT 등, 보류 docker 1015929/1976838)을 건드리지 마라. 원고·그림·Dropbox는 건드리지 않는다.
시험셋으로 설정을 고르지 않는다. 먼저 docs/AGENTS.md와 results/pdv2_fresco_official_20261107/pipeline/{protocol.json,
worker_status.json,supervisor_status.json,report_ko.txt}를 읽어라. 코드: scripts/fresco_official/ (stream.py, evaluate.py,
server_master.py, aggregate.py, supervisor.py). 서버 ssh joms-gpu, 서버 repo /home/qkrrlans00/Neural-Spectral-Codec,
NAS 루트 /mnt/aicps-nas/qkrrlans00/nsd_fresco_official_20261107. 서버 CPU는 nice 10, 최대 2코어.
worker_status.json(results/pdv2_fresco_official_20261107/pipeline/)에 stage·evidence·running_processes·
remaining_dependencies·next_step·blockers·updated_utc를 갱신한다.
"""
PROMPT_DONE = """상황: supervisor가 TEST_DONE을 기록했다(서버 46원장 + 로컬 aggregate.py 검산 통과).
할 일: (1) summary.json·verification.json·원장 46개를 독립 재검산(해시, 195,726질의, 61,682양성, kitti/07 포함·평균 제외,
j<=q-30, 유한 점수, 매크로 재계산). (2) report_ko.txt 최종본, storage.json(NAS 사용량 실측), pilot.json 갱신.
(3) docs/RAW/실험일지/2026-11-07_fresco-공식-평가.md 작성, 위키 최소 수정(docs/평가/비교 방법.md의 FreSCo 문장·
docs/할 일.md·docs/실행 환경.md에 확인된 값과 경로만), python3 docs/도구/check_wiki.py docs 통과.
공식 재현(원 논문 조건)과 우리 과제 평가(표2 조건)의 차이를 명시하고, 서버 시간은 진단용 하드웨어 표시로만 적는다.
root 검산 전 원고 주표·Dropbox 반영 금지. 끝나면 worker_status.json stage를 "TEST_DONE_verified..."로 바꾼다.
"""
PROMPT_FAIL = """상황: supervisor가 FAILED를 기록했다. 원인: {why}
할 일: pipeline/FAILED.json·supervisor_status.json·서버 NAS/test_master_status.json과 로그를 읽고 원인을 찾아,
이미 승인된 범위(같은 protocol.json 설정, 같은 코드 경로의 버그 수정, 재시작) 안에서만 복구하라. 설정을 바꾸거나
시험 결과를 보고 조정하지 않는다. 복구 후 server_master.py run / stream.py controller / supervisor.py를 재기동하고
실제 진행 증가를 확인한 뒤 worker_status.json에 기록하라. 복구가 불가능하면 원인과 필요한 사용자 결정을 기록하고 끝낸다.
"""


def utc():
    return datetime.now(timezone.utc)


def stamp():
    return utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def ssh(cmd, timeout=120):
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", SERVER, cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def controller_pids() -> list[int]:
    """/proc에서 본 실행 controller(--role test, --limit 없음)를 찾는다(PID 파일에 의존하지 않음)."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            c = (d / "cmdline").read_bytes()
        except OSError:
            continue
        if b"stream.py\x00controller\x00--role\x00test" in c and b"--limit" not in c:
            out.append(int(d.name))
    return out


def proc_alive(pid, needle: bytes) -> bool:
    try:
        return needle in Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, TypeError):
        return False


def start_controller():
    log = open(RESULTS / "logs/controller_test.log", "ab")
    p = subprocess.Popen([sys.executable, str(REPO / SPY / "stream.py"), "controller", "--role", "test"],
                         cwd=str(LOCAL_D / "work"), env={**os.environ, **D_ENV},
                         stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    return p.pid


def start_master():
    cmd = (f"cd {SERVER_REPO} && mkdir -p /mnt/aicps-nas/qkrrlans00/nsd_fresco_official_20261107/logs && "
           f"(setsid nohup nice -n 10 python3 {SPY}/server_master.py run --role test "
           f">> /mnt/aicps-nas/qkrrlans00/nsd_fresco_official_20261107/logs/server_master_test.log 2>&1 "
           f"< /dev/null & ) ; sleep 2; python3 {SPY}/server_master.py status --role test")
    return ssh(cmd)


def rsync_small():
    r = subprocess.run(["timeout", "900", "rsync", "-a", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20",
                        "--include", "*/", "--include", "*.npz", "--include", "*.json", "--exclude", "*",
                        f"{SERVER}:{SRES}/", str(RESULTS / "server") + "/"], capture_output=True, text=True)
    return r.returncode


def launch_followup(state, kind, why):
    if len([f for f in state.get("followups", []) if f["kind"] == kind]) >= (1 if kind == "test_done" else 2):
        return None
    t = utc().strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS
    prompt = out / f"prompt_followup_{t}.txt"
    events = out / f"events_{t}.jsonl"  # 완료 감시기 capture가 events_<영숫자>.jsonl만 run ID로 받는다
    stderr = out / f"stderr_followup_{t}.log"
    prompt.write_text(PROMPT_COMMON + (PROMPT_DONE if kind == "test_done" else PROMPT_FAIL.format(why=why)))
    args = [CLI, "--safe-mode", "--model", MODEL, "--effort", "high", "--permission-mode", "dontAsk",
            "--tools", "Read,Glob,Grep,Write,Edit,Bash,WebFetch,WebSearch",
            "--allowedTools", "Read,Glob,Grep,Write,Edit,Bash,WebFetch,WebSearch",
            "--disallowedTools", "Bash(rm *)", "Bash(sudo *)", "Bash(git commit *)", "Bash(git push *)",
            "--no-session-persistence", "--verbose", "--output-format", "stream-json", "-p", ""]
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}, **D_ENV}
    FOLLOWUP_CWD.mkdir(parents=True, exist_ok=True)
    with open(prompt, "rb") as fi, open(events, "wb") as fo, open(stderr, "wb") as fe:
        p = subprocess.Popen(args, stdin=fi, stdout=fo, stderr=fe, cwd=str(FOLLOWUP_CWD), env=env,
                             start_new_session=True)
    old = json.loads((out / "launcher.json").read_text())
    write_json(out / f"launcher_before_{t}.json", old)
    write_json(out / "launcher.json", {
        "worker": "pipeline", "requested_model": MODEL, "pid": p.pid, "started_utc": utc().isoformat(),
        "prompt": str(prompt.relative_to(REPO)), "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "events": str(events.relative_to(REPO)), "stderr": str(stderr.relative_to(REPO)), "cli": CLI,
        "supersedes": f"launcher_before_{t}.json", "reason": f"fresco supervisor follow-up: {kind} ({why})"})
    name = f"fresco_followup_{t}"
    mon = str(REPO / "scripts/_review_completion_monitor.py")
    arm = subprocess.run([sys.executable, mon, "arm", "--root", str(ROOTDIR.relative_to(REPO)), "--name", name,
                          "--workers", "pipeline", "--live", "--max-hours", "24"], cwd=str(LOCAL_D / "work"),
                         capture_output=True, text=True)
    watch_pid = None
    if arm.returncode == 0:
        log = open(ROOTDIR / "completion_monitor" / f"{name}.log", "ab")
        w = subprocess.Popen([sys.executable, mon, "watch", "--root", str(ROOTDIR.relative_to(REPO)), "--name", name],
                             cwd=str(LOCAL_D / "work"), stdout=log, stderr=log, start_new_session=True)
        watch_pid = w.pid
    rec = {"kind": kind, "why": why, "utc": stamp(), "pid": p.pid, "events": str(events.relative_to(REPO)),
           "monitor": name, "arm_rc": arm.returncode, "arm_out": (arm.stdout + arm.stderr)[-600:],
           "watch_pid": watch_pid, "armed": arm.returncode == 0 and watch_pid is not None}
    state.setdefault("followups", []).append(rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--max-hours", type=float, default=36)
    ap.add_argument("--no-followup", action="store_true")
    a = ap.parse_args()
    state = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    state.update({"pid": os.getpid(), "started_utc": stamp(), "interval_s": a.interval,
                  "semantics": "계산 완료/실패 감시. 작업자 종료 알림은 root 감시기가 별도로 맡는다. "
                               "TEST_DONE은 서버 46원장 + 로컬 aggregate.py 검산 통과 뒤에만."})
    state.setdefault("restarts", {"controller": 0, "master": 0})
    state.setdefault("log", [])
    t_end = time.time() + a.max_hours * 3600
    ssh_fail, still, last_prog = 0, 0, None

    def log(msg):
        state["log"] = (state["log"] + [f"{stamp()} {msg}"])[-300:]

    def terminal(kind, why):
        state["terminal"] = {"kind": kind, "why": why, "utc": stamp()}
        write_json(RESULTS / ("TEST_DONE.json" if kind == "test_done" else "FAILED.json"), state["terminal"])
        log(f"TERMINAL {kind}: {why}")
        if not a.no_followup:
            rec = launch_followup(state, kind, why)
            log(f"followup {rec}")
            if rec and not rec["armed"]:
                log(f"FOLLOWUP_MONITOR_NOT_ARMED arm_rc={rec['arm_rc']} (완료 알림 없음, root 확인 필요)")
        state["updated_utc"] = stamp()
        write_json(STATUS, state)

    while time.time() < t_end:
        state["updated_utc"] = stamp()
        # local controller
        tsp = RESULTS / "transfer_status_test.json"
        ts = json.loads(tsp.read_text()) if tsp.exists() else {}
        pids = controller_pids()
        ctrl_alive = bool(pids)
        ctrl_failed = bool(ts.get("log") and "FAILED:" in ts["log"][-1])
        state["controller"] = {"pids": pids, "alive": ctrl_alive, "last": (ts.get("log") or [""])[-1]}
        # server
        rc, out, err = ssh(f"cd {SERVER_REPO} && python3 {SPY}/server_master.py status --role test && "
                           f"python3 {SPY}/stream.py status --role test")
        if rc != 0:
            ssh_fail += 1
            log(f"ssh failed rc={rc} ({ssh_fail}) {err[-200:]}")
            if ssh_fail >= 12:
                terminal("failed", f"server unreachable {ssh_fail} times")
                return 1
            write_json(STATUS, state)
            time.sleep(a.interval)
            continue
        ssh_fail = 0
        lines = out.strip().splitlines()
        master, streams = json.loads(lines[-2]), json.loads(lines[-1])
        committed = sum(v["committed"] for v in streams["seqs"].values())
        n_done = sum(v["done"] for v in streams["seqs"].values())
        state["server"] = {"master": master, "stream_frames_committed": committed, "streams_done": n_done,
                           "guard_ok": streams["guard_ok"],
                           "root_free_gib": round(streams["root_free"] / 2 ** 30, 1),
                           "nas_free_gib": round(streams["nas_free"] / 2 ** 30, 1)}
        ctrl_done = n_done == 46
        state["controller"]["done"] = ctrl_done
        prog = (committed, master.get("ledgers_done"))
        still = still + 1 if prog == last_prog else 0
        last_prog = prog
        state["stalled_polls"] = still
        if still >= 6:
            log(f"no progress for {still} polls {prog}")
        if not ctrl_alive and not ctrl_done and n_done < 46:
            if ctrl_failed or state["restarts"]["controller"] >= 5:
                terminal("failed", f"stream controller stopped: {state['controller']['last']}")
                return 1
            state["restarts"]["controller"] += 1
            log(f"restart controller pid={start_controller()}")
        if master.get("markers", {}).get("FAILED"):
            rsync_small()
            terminal("failed", f"server master FAILED: {master.get('failed')}")
            return 1
        if not master.get("alive") and not master.get("markers", {}).get("ALL_LEDGERS_DONE"):
            if state["restarts"]["master"] >= 3:
                terminal("failed", "server master died repeatedly")
                return 1
            state["restarts"]["master"] += 1
            r = start_master()
            log(f"restart master rc={r[0]} {r[1][-300:]}")
        state["rsync_rc"] = rsync_small()
        if master.get("markers", {}).get("ALL_LEDGERS_DONE"):
            src = RESULTS / "server/ledgers/test"
            dst = RESULTS / "ledgers/test"
            dst.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.glob("dump_fresco_*.npz")):
                if not (dst / f.name).exists() or (dst / f.name).read_bytes() != f.read_bytes():
                    (dst / f.name).write_bytes(f.read_bytes())
            agg = subprocess.run([sys.executable, str(REPO / SPY / "aggregate.py")], cwd=str(LOCAL_D / "work"), capture_output=True,
                                 text=True)
            state["aggregate"] = {"rc": agg.returncode, "out": agg.stdout[-1500:], "err": agg.stderr[-800:]}
            if agg.returncode == 0:
                terminal("test_done", "46 ledgers + aggregate verification passed")
                return 0
            terminal("failed", f"aggregate verification failed rc={agg.returncode}")
            return 1
        write_json(STATUS, state)
        time.sleep(a.interval)
    terminal("failed", f"supervisor max hours {a.max_hours} reached")
    return 1


if __name__ == "__main__":
    sys.exit(main())
