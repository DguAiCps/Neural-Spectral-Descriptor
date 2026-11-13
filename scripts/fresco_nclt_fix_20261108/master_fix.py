#!/usr/bin/env python3
"""NCLT9 수정 재평가 서버 master (서버 호스트, 2026-11-08). 기존 server_master.run과 같은 규칙을 NCLT9에만 쓴다.

  setsid nohup python3 scripts/fresco_nclt_fix_20261108/master_fix.py run &
  python3 scripts/fresco_nclt_fix_20261108/master_fix.py status
기존과 다른 점: 시퀀스 = NCLT9, 잠금 /tmp/fresco_nclt_fix_20261108_test_master.lock, MATLAB·원장 = evaluate_fix.py,
상태·원장·표지 = 새 NAS 루트, 가드 = 서버 root >= 35 GiB·NAS 여유 >= 200 GiB·새 NAS 루트 사용량 <= 8 GiB.
시작 전 gate 확인: 새 NAS results/gate_ok.json(로컬이 gate.json 해시 대조 뒤 쓴다)이 없으면 시작하지 않는다.
재시도 2회, 세 번 실패면 FAILED 기록. MATLAB은 스트림 수신 중 1개, 9 스트림이 끝나면 2개(수신 포함 <= 2코어).
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
import fixenv  # noqa: E402

common, codec_fix, stream, evaluate = fixenv.setup()
NAS = fixenv.NEW_NAS
RES = fixenv.NEW_RESULTS
PY = sys.executable
EVAL = str(Path(__file__).resolve().parent / "evaluate_fix.py")
ROLE = "test"
LOCK = f"/tmp/{fixenv.TAG}_{ROLE}_master.lock"
STATUS = NAS / f"{ROLE}_master_status.json"


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ledger_path(k):
    return RES / "ledgers" / ROLE / f"dump_fresco_{k.replace('/', '_')}.npz"


def nas_used() -> int:
    tot = 0
    for dp, _, fs in os.walk(NAS):
        for f in fs:
            try:
                tot += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return tot


def run(poll: int):
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another master holds the lock", file=sys.stderr)
        return 3
    if not (RES / "gate_ok.json").exists():
        print("gate_ok.json missing: correctness gate not passed", file=sys.stderr)
        return 4
    (NAS / "tmp").mkdir(parents=True, exist_ok=True)
    seqs = list(fixenv.NCLT9)
    for marker in ("ALL_LEDGERS_DONE", "FAILED"):
        p = NAS / f"{ROLE}_{marker}"
        if p.exists():
            p.rename(p.with_name(p.name + f".before_{int(time.time())}"))
    st = {"pid": os.getpid(), "role": ROLE, "started_utc": stamp(), "failures": {}, "failed": {},
          "timings": {}, "log": [], "gate_ok": json.loads((RES / "gate_ok.json").read_text())}

    def save(stage, **kw):
        st.update(kw, stage=stage, updated_utc=stamp())
        st["ledgers_done"] = sum(ledger_path(k).exists() for k in seqs)
        st["n_sequences"] = len(seqs)
        common.write_json(STATUS, st)

    def log(msg):
        st["log"] = (st["log"] + [f"{stamp()} {msg}"])[-300:]
        print(msg, flush=True)

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    running = {}
    save("started")
    while True:
        for k, (p, t0, lp) in list(running.items()):
            if p.poll() is None:
                continue
            del running[k]
            tail = lp.read_text().strip().splitlines()[-4:] if lp.exists() else []
            if p.returncode == 0 and ledger_path(k).exists():
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
        root, nas_free, used = common.free_bytes(Path("/")), common.free_bytes(NAS), nas_used()
        st["guard"] = {"root_free_gib": round(root / 2 ** 30, 1), "nas_free_gib": round(nas_free / 2 ** 30, 1),
                       "new_nas_used_gib": round(used / 2 ** 30, 3)}
        if used > fixenv.NAS_BUDGET:
            log(f"NAS_BUDGET_EXCEEDED {st['guard']}")
            st["failed"]["_budget"] = {"utc": stamp(), **st["guard"]}
        elif root < fixenv.MIN_ROOT_FREE or nas_free < common.MIN_NAS_FREE:
            log(f"DISK_PAUSE {st['guard']}")
            save("paused_disk_guard", current=sorted(running))
            time.sleep(1800)
            continue
        todo, waiting = [], []
        for k in seqs:
            if ledger_path(k).exists() or k in st["failed"] or k in running or "_budget" in st["failed"]:
                continue
            (todo if (NAS / ROLE / k.replace("/", "_") / "stream_done.json").exists() else waiting).append(k)
        if not todo and not waiting and not running:
            if st["failed"]:
                (NAS / f"{ROLE}_FAILED").write_text(json.dumps(st["failed"], indent=1))
                save("FAILED", current=[])
                return 1
            (NAS / f"{ROLE}_ALL_LEDGERS_DONE").write_text(stamp() + "\n")
            save("ALL_LEDGERS_DONE", current=[])
            return 0
        max_par = 1 if waiting else 2
        while todo and len(running) < max_par:
            k = todo.pop(0)
            lp = NAS / "logs" / f"eval_{ROLE}_{k.replace('/', '_')}.log"
            lp.parent.mkdir(parents=True, exist_ok=True)
            cmd = (f"nice -n 10 ionice -c3 {PY} {EVAL} matlab --seq {k} && "
                   f"nice -n 10 ionice -c3 {PY} {EVAL} ledger --seq {k}")
            p = subprocess.Popen(["bash", "-c", cmd], env=env, cwd=str(fixenv.REPO), stdout=open(lp, "a"),
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
    p.add_argument("--poll", type=int, default=120)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "run":
        sys.exit(run(a.poll))
    st = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    alive = False
    if st.get("pid"):
        try:
            alive = Path(f"/proc/{st['pid']}/cmdline").read_bytes().find(b"master_fix.py") >= 0
        except OSError:
            alive = False
    streams = {}
    for k in fixenv.NCLT9:
        d = NAS / ROLE / k.replace("/", "_")
        pp = d / "progress.json"
        streams[k] = {"committed": json.loads(pp.read_text())["committed"] if pp.exists() else 0,
                      "done": (d / "stream_done.json").exists(), "ledger": ledger_path(k).exists()}
    markers = {m: (NAS / f"{ROLE}_{m}").exists() for m in ("ALL_LEDGERS_DONE", "FAILED")}
    print(json.dumps({"alive": alive, "markers": markers, "streams": streams, **{k: st.get(k) for k in (
        "pid", "stage", "current", "ledgers_done", "n_sequences", "waiting", "failed", "guard", "updated_utc")}}))


if __name__ == "__main__":
    main()
