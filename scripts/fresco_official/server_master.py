#!/usr/bin/env python3
"""서버 master: 스트림이 끝난 시험 시퀀스마다 공식 MATLAB 단계와 원장을 만든다 (2026-11-07).

  setsid nohup python3 scripts/fresco_official/server_master.py run [--role test] &   (서버 호스트)
  python3 scripts/fresco_official/server_master.py status
중복 방지: /tmp/fresco_official_20261107_<role>_master.lock(flock). 재시작하면 match_done.json·원장이
있는 시퀀스는 건너뛴다. MATLAB은 -singleCompThread, nice 10.
시작·시퀀스마다 root >= 30 GiB, NAS >= 200 GiB 가드(부족하면 30분 대기, 진행 기록 유지).
시퀀스 실패는 2회까지 재시도, 세 번 실패하면 FAILED 표시(원인 기록) 후 다른 시퀀스를 계속한다.
MATLAB은 스트림 수신 중 1개, 46 스트림이 모두 끝난 뒤 2개까지 병렬(receiver 포함 <= 2코어).
상태: NAS/<role>_master_status.json 과 results/.../pipeline/server_master_status_<role>.json.
끝: 모든 원장이 있으면 NAS/<role>_ALL_LEDGERS_DONE, 실패가 있으면 NAS/<role>_FAILED.
최종 TEST_DONE은 로컬 supervisor가 공통 집계·검산(aggregate.py)을 통과한 뒤에만 쓴다.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NAS_ROOT, REPO, RESULTS, ROLES, free_bytes, key, name, write_json  # noqa: E402
from common import MIN_NAS_FREE, MIN_ROOT_FREE  # noqa: E402

PY = sys.executable
EVAL = str(Path(__file__).resolve().parent / "evaluate.py")


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ledger_path(role, sensor, seq):
    return RESULTS / "ledgers" / role / f"dump_fresco_{name(sensor, seq)}.npz"


def run(role: str, poll: int):
    lock = open(f"/tmp/fresco_official_20261107_{role}_master.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another master holds the lock", file=sys.stderr)
        return 3
    seqs = ROLES[role][0]
    st_paths = [NAS_ROOT / f"{role}_master_status.json", RESULTS / f"server_master_status_{role}.json"]
    for marker in ("ALL_LEDGERS_DONE", "FAILED"):
        p = NAS_ROOT / f"{role}_{marker}"
        if p.exists():
            p.rename(p.with_name(p.name + f".before_{int(time.time())}"))
    st = {"pid": os.getpid(), "role": role, "started_utc": stamp(), "failures": {}, "failed": {},
          "timings": {}, "log": []}

    def save(stage, **kw):
        st.update(kw, stage=stage, updated_utc=stamp())
        done = [key(s, q) for s, q in seqs if ledger_path(role, s, q).exists()]
        st["ledgers_done"] = len(done)
        st["n_sequences"] = len(seqs)
        for p in st_paths:
            write_json(p, st)

    def log(msg):
        st["log"] = (st["log"] + [f"{stamp()} {msg}"])[-300:]
        print(msg, flush=True)

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    running = {}   # seq -> (Popen, t0, log path)
    save("started")
    while True:
        for k, (p, t0, lp) in list(running.items()):   # 끝난 작업 회수
            if p.poll() is None:
                continue
            del running[k]
            tail = lp.read_text().strip().splitlines()[-4:] if lp.exists() else []
            s, q = k.split("/")
            if p.returncode == 0 and ledger_path(role, s, q).exists():
                try:
                    st["timings"][k] = {"wall_sec": round(time.time() - t0, 1), **json.loads(tail[-1])}
                except (ValueError, IndexError):
                    st["timings"][k] = {"wall_sec": round(time.time() - t0, 1)}
                log(f"ledger {k} in {time.time() - t0:.0f} s")
            else:
                st["failures"][k] = st["failures"].get(k, 0) + 1
                log(f"FAILED {k} rc={p.returncode} ({st['failures'][k]}): {' | '.join(tail)}")
                if st["failures"][k] >= 3:
                    st["failed"][k] = {"rc": p.returncode, "tail": tail, "log": str(lp), "utc": stamp()}
        root, nas = free_bytes(Path("/")), free_bytes(NAS_ROOT)
        st["guard"] = {"root_free_gib": round(root / 2 ** 30, 1), "nas_free_gib": round(nas / 2 ** 30, 1)}
        if root < MIN_ROOT_FREE or nas < MIN_NAS_FREE:
            log(f"DISK_PAUSE {st['guard']}")
            save("paused_disk_guard", current=sorted(running))
            time.sleep(1800)
            continue
        todo, waiting = [], []
        for s, q in seqs:
            k = key(s, q)
            if ledger_path(role, s, q).exists() or k in st["failed"] or k in running:
                continue
            (todo if (NAS_ROOT / role / name(s, q) / "stream_done.json").exists() else waiting).append(k)
        if not todo and not waiting and not running:
            if st["failed"]:
                (NAS_ROOT / f"{role}_FAILED").write_text(json.dumps(st["failed"], indent=1))
                save("FAILED", current=[])
                return 1
            (NAS_ROOT / f"{role}_ALL_LEDGERS_DONE").write_text(stamp() + "\n")
            save("ALL_LEDGERS_DONE", current=[])
            return 0
        # 스트림 수신기가 도는 동안 MATLAB 1개, 스트림이 모두 끝나면 2개(<= 2코어)
        max_par = 1 if waiting else 2
        while todo and len(running) < max_par:
            k = todo.pop(0)
            lp = NAS_ROOT / "logs" / f"eval_{role}_{k.replace('/', '_')}.log"
            lp.parent.mkdir(parents=True, exist_ok=True)
            cmd = (f"{PY} {EVAL} matlab --role {role} --seq {k} && "
                   f"{PY} {EVAL} ledger --role {role} --seq {k}")
            p = subprocess.Popen(["bash", "-c", cmd], env=env, cwd=str(REPO), stdout=open(lp, "a"),
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            running[k] = (p, time.time(), lp)
            log(f"start {k} (parallel {len(running)}/{max_par})")
        save("running" if running else "waiting_for_streams", current=sorted(running),
             waiting=len(waiting), max_parallel=max_par)
        time.sleep(poll if not running else 30)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--role", default="test", choices=list(ROLES))
    p.add_argument("--poll", type=int, default=120)
    p = sub.add_parser("status")
    p.add_argument("--role", default="test")
    a = ap.parse_args()
    if a.cmd == "run":
        sys.exit(run(a.role, a.poll))
    p = NAS_ROOT / f"{a.role}_master_status.json"
    st = json.loads(p.read_text()) if p.exists() else {}
    alive = False
    if st.get("pid"):
        try:
            alive = Path(f"/proc/{st['pid']}/cmdline").read_bytes().find(b"server_master.py") >= 0
        except OSError:
            alive = False
    markers = {m: (NAS_ROOT / f"{a.role}_{m}").exists() for m in ("ALL_LEDGERS_DONE", "FAILED")}
    print(json.dumps({"alive": alive, "markers": markers, **{k: st.get(k) for k in (
        "pid", "stage", "current", "ledgers_done", "n_sequences", "waiting", "failed", "guard", "updated_utc")}}))


if __name__ == "__main__":
    main()
