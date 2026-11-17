#!/usr/bin/env python3
"""Light local supervisor of the SeqOT paper-recipe run (2026-11-07).

Every --interval seconds (default 600) it checks, with bounded SSH, the
server chain markers (_external/seqot_paper_recipe_20261107/chain/*), the
server master PID and the local test-input controller, and writes
results/pdv2_baseline_revision_20261107/seqot/supervisor_status.json.
Communication errors are never treated as success.

Terminal states (exactly one follow-up launch per state, at most 3 in total):
  test_done    chain/TEST_DONE      -> rsync the server records (no deletion),
                                       launch the verification Opus worker
  failed       chain/FAILED, test_inputs/FAILED, master dead without TEST_DONE,
               controller dead without READY -> launch a recovery Opus worker
  unreachable  > --max-ssh-failures consecutive SSH failures -> recovery worker
Launch: same CLI / flags as the root launch (claude-opus-5-5), prompt on
stdin, stream-json events to seqot/events_<UTC>.jsonl; launcher.json is
updated with the new PID / events, then
  scripts/_review_completion_monitor.py arm --root <abs repo>/results/pdv2_baseline_revision_20261107
      --name seqot_followup_<UTC> --workers seqot --live
and its watch runs detached, so root is told when the follow-up ends. Arm is retried
(ARM_TRIES); if it still fails, or the follow-up budget is used up, seqot/root_attention.json
gets one entry per key. At --max-hours one read-only "expired" check worker is launched
(same budget), otherwise a root_attention entry.
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

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/pdv2_baseline_revision_20261107"
OUT = ROOT / "seqot"
STATUS = OUT / "supervisor_status.json"
SERVER = "joms-gpu"
SREPO = "/home/qkrrlans00/Neural-Spectral-Codec"
SW = f"{SREPO}/_external/seqot_paper_recipe_20261107"
CLI = "/mnt/d/NSD_tools/claude/2.1.280/claude"
MODEL = "claude-opus-5-5"
CONTROLLER_MATCH = b"_review_seqot_paper_recipe_20261107_testinputs.py\x00controller"
# local run location / temp / cache on D (AGENTS.md); the result folder is on D behind a symlink
LOCAL_D = Path("/mnt/d/NSD_local")
D_ENV = {"TMPDIR": str(LOCAL_D / "tmp"), "TMP": str(LOCAL_D / "tmp"), "TEMP": str(LOCAL_D / "tmp"),
         "XDG_CACHE_HOME": str(LOCAL_D / "cache"), "PYTHONDONTWRITEBYTECODE": "1"}
FOLLOWUP_CWD = LOCAL_D / "work/seqot_followup"


def utc():
    return datetime.now(timezone.utc)


def stamp():
    return utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def ssh(cmd, timeout=120):
    try:
        r = subprocess.run(["timeout", str(timeout), "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                            SERVER, cmd], capture_output=True, text=True)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:  # noqa: BLE001
        return 255, "", repr(e)


PROBE = f"""cd {SREPO}; C={SW}/chain; python3 - <<'EOF'
import json, os, glob, re
C = "{SW}/chain"; W = "{SW}"
def rd(p):
    try: return open(p).read().strip()
    except OSError: return None
pid = rd(C + "/master.pid")
alive = False
if pid:
    try: alive = "_review_seqot_paper_recipe_20261107_chain.sh" in open(f"/proc/{{pid}}/cmdline").read()
    except OSError: alive = False
seeds = {{}}
for s in (1, 2, 3):
    d = f"{{W}}/s{{s}}"
    seeds[s] = {{"markers": sorted(os.path.basename(x) for x in glob.glob(f"{{C}}/s{{s}}/*")),
                "stage1_ckpts": len(glob.glob(f"{{d}}/runs/stage1/amodel_seqot*.pth.tar")),
                "stage2_ckpts": len(glob.glob(f"{{d}}/centered/runs/stage2/amodel_gem*.pth.tar"))}}
    sel = f"{SREPO}/results/pdv2_baseline_revision_20261107/seqot/s{{s}}/train_selection.json"
    try:
        j = json.load(open(sel)); seeds[s]["scored"] = [len(j["stage1"]["epochs"]), len(j["stage2"]["epochs"])]
        seeds[s]["chosen"] = [j[k].get("chosen", {{}}).get("epoch") for k in ("stage1", "stage2")]
    except Exception: pass
    try:
        lines = open(f"{{d}}/runs/stage1/log.jsonl").read().splitlines()
        if lines: seeds[s]["last_epoch_log"] = json.loads(lines[-1])
    except OSError: pass
ti = f"{{W}}/test_inputs"
done = len(glob.glob(ti + "/*/done.json"))
st = os.statvfs("{SREPO}")
print(json.dumps({{"master_pid": pid, "master_alive": alive, "TRAIN_DONE": os.path.exists(C + "/TRAIN_DONE"),
                  "TEST_DONE": os.path.exists(C + "/TEST_DONE"), "FAILED": rd(C + "/FAILED"),
                  "test_inputs_done": done, "READY": os.path.exists(ti + "/READY"),
                  "test_inputs_FAILED": rd(ti + "/FAILED"), "free_gib": round(st.f_bavail * st.f_frsize / 2**30, 1),
                  "chain_log_tail": (rd("{SW}/logs/chain.log") or "").splitlines()[-4:], "seeds": seeds}}))
EOF"""


def controller_alive():
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                if CONTROLLER_MATCH in (p / "cmdline").read_bytes():
                    return int(p.name)
            except OSError:
                pass
    return None


PROMPT_COMMON = f"""repo 루트는 {REPO}이고 아래 상대 경로는 모두 이 repo 기준이다. 당신의 cwd는 D 작업 폴더({FOLLOWUP_CWD})이며
로컬 실행·임시·캐시·로그·생성물은 D에 둔다(C의 /home·/tmp·도커 볼륨에 새 큰 파일 금지, 무거운 계산은 서버).
""" + """당신은 SeqOT 담당(seqot) Opus 5.5 후속 작업자다. 실제 model은 claude-opus-5-5다. 한국어로 보고.
먼저 docs/AGENTS.md, docs/개요.md, results/pdv2_baseline_revision_20261107/seqot/{protocol.json,worker_status.json,supervisor_status.json,report_ko.txt(있으면)}와
scripts/_review_seqot_paper_recipe_20261107*.{py,sh}를 읽는다. docs/RAW/는 현재 근거가 아니다.
규칙: 원본 결과/캐시/가중치/대형 파일 삭제 금지, git commit/push 금지, 다른 작업자(ringpp, manuscript)의 파일·프로세스·GPU 사용 금지,
원고·공통 표생성기·위키 직접 수정 금지(반영안은 seqot 폴더의 manuscript_integration.json으로). 서버 joms-gpu docker ring_baseline GPU0만,
GPU1 금지, 보존 PID 340070·1083011 및 타인 프로세스 건드리지 않음. SSH는 BatchMode·timeout. 시험셋으로 설정·시드·에폭 선택 금지,
결과가 낮아도 시험 기반 튜닝 금지. 새 실험 추가 금지(승인 범위는 이 SeqOT 논문 레시피 3시드 재학습·검증선택·46시험·검산과 그 오류 복구뿐).
worker_status.json을 즉시 갱신(단계·실제 PID·장소·산출물·ETA·남은 작업). root 알림은 supervisor가 이미 감시기를 등록했으므로 직접 중복 등록하지 않는다.
"""

PROMPT_DONE = """상황: supervisor가 서버 chain/TEST_DONE을 확인하고 서버 기록을 로컬 seqot 폴더로 복사했다(supervisor_status.json 참조).
할 일(실제 실행·검산 증거 필수):
1. 서버와 로컬의 summary.json, verification.json, test_input_reproduction.json, s{1,2,3}/train_selection.json, s{1,2,3}/odom 원장 해시를 대조하고,
   독립 검산(예: 로컬에서 원장으로 _pdv2_main_baselines_official.aggregate 재실행 또는 별도 구현으로 AP/maxF1/R@100P/R@1 재계산, 195,726질의·61,682양성·KITTI07 제외 확인,
   j*<=q-30, 선택 기록이 시험 원장보다 먼저 쓰였는지, 체크포인트 해시, 에폭 수 20/20과 finished 표식, 실제 LR·6/6·Eq.13 메타)을 수행해 verification_followup.json에 기록.
2. 최종 deliverables 작성/보완: report_ko.txt(논문 명시값과 해석값 구분, 3시드 선택 에폭, 학습 시간, 주표 대비 변화, 한계),
   summary.json 확인, manuscript_integration.json(바뀔 주표 SeqOT 행 값 mean±SD, 부록 설명 문안, 비용 영향: 모델 구조 동일하므로 비용 재측정 필요 여부 판단 — 로컬 GPU는 ringpp 전용이라 요청만 기록),
   source/checkpoint hash 목록. 3. 실험 일지 docs/RAW/실험일지/2026-11-07_seqot-논문레시피-재학습.md 작성 가능(위키 본문은 수정하지 않음).
4. worker_status.json을 stage=complete_verified(또는 검산 실패 시 구체 원인)로 갱신하고 종료.
"""

PROMPT_FAIL = """상황: supervisor가 비정상 상태를 감지했다: {why}. supervisor_status.json, 서버 _external/seqot_paper_recipe_20261107/{{chain,logs}}와
로컬 seqot/logs, test_inputs_transfer.json을 읽고 원인을 진단한다. 승인 범위 안의 오류 복구만 한다(같은 프로토콜로 재시작 가능한 단계 재개:
chain master 재실행은 'setsid nohup bash scripts/_review_seqot_paper_recipe_20261107_chain.sh master'로, 송신 controller 재실행은
'setsid nohup python3 scripts/_review_seqot_paper_recipe_20261107_testinputs.py controller'). 프로토콜·레시피·선택 규칙 변경 금지.
통신 오류만이면 bounded retry로 확인 후 계산이 계속 중인지 판단한다. 재개했다면 새 supervisor를 같은 스크립트로 다시 띄우고
(--followups-used 를 현재 사용 횟수로) worker_status에 PID를 기록한다. 복구 불가하면 구체적 blocker와 최소 요청을 worker_status와 report_ko.txt에 쓰고 종료.
"""

PROMPT_EXPIRED = """상황: supervisor가 --max-hours 기한에 닿았지만 끝 상태(TEST_DONE/FAILED/불통)를 보지 못했다: {why}.
읽기 전용 확인만 한다: supervisor_status.json, 서버 _external/seqot_paper_recipe_20261107/{{chain,logs}} 표식과 master 생존(BatchMode·timeout SSH).
서버 계산·학습·평가를 시작·재시작·중지하지 않는다. 계산이 계속 중이면 같은 supervisor 스크립트만 D cwd에서 같은 --interval과
--followups-used 현재 사용 횟수로 다시 띄운다(살아 있는 supervisor가 있으면 띄우지 않는다). 이미 끝났거나 멈췄으면 진단과 root 판단 요청만
worker_status.json과 report_ko.txt에 쓰고 종료한다.
"""

ARM_TRIES = 3
ARM_WAIT_S = 30


def root_attention(key, reason, extra=None):
    """Durable note for root when the live signal is not armed; one entry per key."""
    path = OUT / "root_attention.json"
    try:
        items = json.loads(path.read_text())
    except (OSError, ValueError):
        items = []
    if any(i.get("key") == key for i in items):
        return
    items.append({"key": key, "utc": stamp(), "supervisor_pid": os.getpid(), "reason": reason, **(extra or {})})
    write_json(path, items)


def launch_followup(kind, why, state):
    t = utc().strftime("%Y%m%dT%H%M%SZ")
    prompt = OUT / f"prompt_{t}.txt"
    events = OUT / f"events_{t}.jsonl"
    stderr = OUT / f"stderr_{t}.log"
    tail = {"test_done": PROMPT_DONE, "expired": PROMPT_EXPIRED}.get(kind, PROMPT_FAIL)
    body = PROMPT_COMMON + (tail if kind == "test_done" else tail.format(why=why))
    prompt.write_text(body)
    args = [CLI, "--safe-mode", "--model", MODEL, "--effort", "high", "--permission-mode", "dontAsk",
            "--tools", "Read,Glob,Grep,Write,Edit,Bash,WebFetch,WebSearch",
            "--allowedTools", "Read,Glob,Grep,Write,Edit,Bash,WebFetch,WebSearch",
            "--disallowedTools", "Bash(rm *)", "Bash(sudo *)", "Bash(git commit *)", "Bash(git push *)",
            "--no-session-persistence", "--verbose", "--output-format", "stream-json", "-p", ""]
    # same environment as the root launch: no inherited Claude Code session variables
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}, **D_ENV}
    FOLLOWUP_CWD.mkdir(parents=True, exist_ok=True)
    with open(prompt, "rb") as fi, open(events, "wb") as fo, open(stderr, "wb") as fe:
        p = subprocess.Popen(args, stdin=fi, stdout=fo, stderr=fe, cwd=str(FOLLOWUP_CWD), env=env,
                             start_new_session=True)
    old = json.loads((OUT / "launcher.json").read_text())
    hist = OUT / f"launcher_before_{t}.json"
    write_json(hist, old)
    launcher = {"worker": "seqot", "requested_model": MODEL, "pid": p.pid, "started_utc": utc().isoformat(),
                "prompt": str(prompt.relative_to(REPO)),
                "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
                "events": str(events.relative_to(REPO)), "stderr": str(stderr.relative_to(REPO)),
                "cli": CLI, "supersedes": hist.name, "reason": f"supervisor follow-up: {kind} ({why})"}
    write_json(OUT / "launcher.json", launcher)
    name = f"seqot_followup_{t}"
    mon = str(REPO / "scripts/_review_completion_monitor.py")
    # absolute --root: the monitor resolves a relative root against its cwd (D), not the repo
    statef = ROOT / "completion_monitor" / f"watch_{name}.json"
    tries = []
    for i in range(ARM_TRIES):
        try:
            arm = subprocess.run([sys.executable, mon, "arm", "--root", str(ROOT), "--name", name,
                                  "--workers", "seqot", "--live", "--max-hours", "24"], cwd=str(LOCAL_D / "work"),
                                 capture_output=True, text=True, timeout=300)
            tries.append({"rc": arm.returncode, "out": (arm.stdout + arm.stderr)[-400:]})
        except (OSError, subprocess.SubprocessError) as e:
            tries.append({"rc": None, "out": repr(e)[-400:]})
        if statef.exists():           # armed; the same name is never armed twice
            break
        if i + 1 < ARM_TRIES:
            time.sleep(ARM_WAIT_S)
    watch_pid = None
    if statef.exists():
        try:
            log = open(ROOT / "completion_monitor" / f"{name}.log", "ab")
            w = subprocess.Popen([sys.executable, mon, "watch", "--root", str(ROOT), "--name", name],
                                 cwd=str(LOCAL_D / "work"), stdout=log, stderr=log, start_new_session=True)
            watch_pid = w.pid
        except OSError as e:
            tries.append({"watch_error": repr(e)[-400:]})
        try:
            mons = json.loads((ROOT / "monitors.json").read_text())
            mons.append({"role": "seqot", "name": name, "pid": watch_pid,
                         "state_file": f"completion_monitor/watch_{name}.json", "armed_by": "seqot supervisor"})
            write_json(ROOT / "monitors.json", mons)
        except (OSError, ValueError):
            pass
    if watch_pid is None:
        root_attention(f"monitor:{name}", "follow-up launched but completion monitor not armed/watching",
                       {"kind": kind, "why": why, "followup_pid": p.pid, "events": launcher["events"],
                        "monitor": name, "tries": tries})
    rec = {"kind": kind, "why": why, "utc": stamp(), "pid": p.pid, "events": launcher["events"],
           "monitor": name, "arm_rc": [x["rc"] for x in tries if "rc" in x][-1],"arm_tries": tries, "armed": statef.exists(),
           "watch_pid": watch_pid}
    state.setdefault("followups", []).append(rec)
    return rec


def rsync_records():
    r = subprocess.run(["timeout", "1800", "rsync", "-a", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20",
                        f"{SERVER}:{SREPO}/results/pdv2_baseline_revision_20261107/seqot/", str(OUT) + "/"],
                       capture_output=True, text=True)
    r2 = subprocess.run(["timeout", "600", "rsync", "-a", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20",
                         f"{SERVER}:{SW}/logs/", str(OUT / "server_logs") + "/"], capture_output=True, text=True)
    return {"records_rc": r.returncode, "logs_rc": r2.returncode, "err": (r.stderr + r2.stderr)[-500:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=600)
    ap.add_argument("--max-ssh-failures", type=int, default=24)
    ap.add_argument("--max-hours", type=float, default=60)
    ap.add_argument("--followups-used", type=int, default=0)
    a = ap.parse_args()
    state = {"pid": os.getpid(), "started_utc": stamp(), "interval_s": a.interval, "polls": 0,
             "ssh_failures": 0, "followups": [], "followups_used_before": a.followups_used,
             "contract": "terminal: TEST_DONE -> rsync + verification Opus; FAILED/dead/unreachable -> recovery Opus; "
                         "max 3 follow-ups; communication errors are not success"}
    t_end = time.time() + a.max_hours * 3600
    handled = set()
    ctrl_dead_polls = 0
    while time.time() < t_end:
        rc, out, err = ssh(PROBE)
        state["polls"] += 1
        state["updated_utc"] = stamp()
        state["controller_pid"] = controller_alive()
        try:
            tr = json.loads((OUT / "test_inputs_transfer.json").read_text())
            state["transfer"] = {"done": len(tr.get("done", [])), "last": tr.get("log", [])[-2:],
                                 "finished_utc": tr.get("finished_utc")}
        except (OSError, ValueError):
            pass
        why = None
        if rc != 0:
            state["ssh_failures"] += 1
            state["last_ssh_error"] = (err or "")[-300:]
            if state["ssh_failures"] > a.max_ssh_failures:
                why = f"server unreachable for {state['ssh_failures']} polls"
        else:
            state["ssh_failures"] = 0
            try:
                srv = json.loads(out.strip().splitlines()[-1])
            except (ValueError, IndexError):
                srv = None
                state["probe_parse_error"] = out[-300:]
            if srv is not None:
                state["server"] = srv
                if srv["TEST_DONE"]:
                    why = "test_done"
                elif srv["FAILED"]:
                    why = f"chain FAILED: {srv['FAILED']}"
                elif srv["test_inputs_FAILED"]:
                    why = f"test inputs FAILED: {srv['test_inputs_FAILED']}"
                elif not srv["master_alive"]:
                    why = "server chain master not alive without TEST_DONE"
                if why is None and not srv["READY"] and state["controller_pid"] is None:
                    ctrl_dead_polls += 1      # two polls in a row: avoid the finalize race
                    if ctrl_dead_polls >= 2:
                        why = "local test-input controller dead before READY"
                else:
                    ctrl_dead_polls = 0
        if why:
            kind = "test_done" if why == "test_done" else "failed"
            key = "unreachable" if rc != 0 else (kind if kind == "test_done" else why)
            if key not in handled and len(state["followups"]) + a.followups_used < 3:
                handled.add(key)
                if kind == "test_done":
                    state["rsync"] = rsync_records()
                rec = launch_followup(kind, why, state)
                state["last_terminal"] = rec
                write_json(STATUS, state)
                break
            if key not in handled:
                handled.add(key)
                root_attention(f"budget:{key}", "terminal state seen but follow-up budget (3) used up",
                               {"why": why, "followups_used": len(state["followups"]) + a.followups_used})
        write_json(STATUS, state)
        try:
            ws = json.loads((OUT / "worker_status.json").read_text())
            ws["live_progress"] = {"from": "supervisor", "utc": state["updated_utc"],
                                   "server": state.get("server", {}).get("seeds"),
                                   "test_inputs_done": state.get("server", {}).get("test_inputs_done"),
                                   "ssh_failures": state["ssh_failures"]}
            write_json(OUT / "worker_status.json", ws)
        except (OSError, ValueError):
            pass
        time.sleep(a.interval)
    else:
        # do not end silently: hand over to one read-only check worker (no new experiment rights), or tell root
        state["expired"] = True
        write_json(STATUS, state)
        why = f"max-hours {a.max_hours} reached after {state['polls']} polls without a terminal state"
        if len(state["followups"]) + a.followups_used < 3:
            state["last_terminal"] = launch_followup("expired", why, state)
        else:
            root_attention("expired", why, {"followups_used": len(state["followups"]) + a.followups_used})
        write_json(STATUS, state)


if __name__ == "__main__":
    main()
