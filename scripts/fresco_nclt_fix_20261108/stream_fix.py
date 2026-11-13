#!/usr/bin/env python3
"""NCLT9 수정 재평가의 키프레임 스트림 (2026-11-08). 기존 scripts/fresco_official/stream.py를 patch해서 쓴다.

  produce --seq S --start K      로컬 컨테이너(ring_baseline): 기존 cmd_produce, 가족 nclt_flu_hdl32_inverted
  recv    --seq S --start K      서버 호스트: 기존 cmd_recv, 새 NAS 루트, 드라이버 fresco_prep_ncltflu, root 35 GiB 가드
  status                         서버 호스트: 기존 cmd_status(새 NAS 루트)
  controller [--seqs ...]        로컬 호스트: 시퀀스마다 produce | ssh recv (기존 controller와 같은 재개·재시도 규칙)
patch 내용은 fixenv.PATCHES. role은 test만 쓴다. 기존 NAS·기존 결과 폴더에는 쓰지 않는다.
원시 점은 로컬 D(/mnt/d/NSD_datasets)에서 키프레임 1개씩 읽어 보내고, 서버에는 /dev/shm 작업 파일 1개만 둔다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixenv  # noqa: E402

common, codec_fix, stream, evaluate = fixenv.setup()
ROLE = "test"
STATUS = fixenv.LOCAL_OUT / "transfer_status_nclt9.json"
SPY = "scripts/fresco_nclt_fix_20261108"


def _ssh(cmd, **kw):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", common.SERVER,
                           cmd], **kw)


def cmd_controller(seqs, retry_wait):
    seqs = seqs or list(fixenv.NCLT9)
    assert all(k in fixenv.NCLT9 for k in seqs), seqs
    status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    status.update({"pid": os.getpid(), "role": ROLE, "started_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seqs": seqs})
    status.setdefault("log", [])

    def log(msg):
        status["log"] = (status["log"] + [time.strftime("%H:%M:%SZ ", time.gmtime()) + msg])[-200:]
        status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        common.write_json(STATUS, status)
        print(msg, flush=True)
    remote = f"cd {common.SERVER_REPO} && "
    env = ("env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "
           f"TMPDIR={fixenv.NEW_NAS}/tmp nice -n 10 ionice -c3 python3 {SPY}/stream_fix.py")
    for k in seqs:
        fails = 0
        while True:
            st = _ssh(remote + f"{env} status", capture_output=True, text=True)
            if st.returncode != 0:
                fails += 1
                log(f"status ssh failed rc={st.returncode} ({fails})")
                if fails > 20:
                    log("FAILED: ssh status repeatedly failing"); return 1
                time.sleep(retry_wait); continue
            info = json.loads(st.stdout.strip().splitlines()[-1])
            s = info["seqs"][f"{ROLE}:{k}"]
            if s["done"]:
                status.setdefault("done", [])
                if k not in status["done"]:
                    status["done"].append(k)
                log(f"done {k}"); break
            if not info["guard_ok"]:
                log(f"DISK_PAUSE {k}: guard root={info['root_free']/2**30:.1f}GiB "
                    f"nas={info['nas_free']/2**30:.0f}GiB; retry in 30 min")
                time.sleep(1800); continue
            start = s["committed"]
            log(f"stream {k} from {start}")
            t0 = time.time()
            pipe = ("set -o pipefail; docker exec -w /workspace/Neural-Spectral-Codec ring_baseline "
                    "nice -n 10 taskset -c 13 env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "
                    f"python3 {SPY}/stream_fix.py produce --seq {k} --start {start} "
                    f"| ssh -o BatchMode=yes -o ServerAliveInterval=30 {common.SERVER} "
                    f"'{remote}{env} recv --seq {k} --start {start}'")
            r = subprocess.run(["bash", "-c", pipe], capture_output=True, text=True)
            tail = (r.stderr or "").strip().splitlines()[-3:]
            log(f"stream {k} rc={r.returncode} in {time.time() - t0:.0f} s: {' | '.join(tail)}")
            if r.returncode == 0:
                continue
            fails += 1
            if fails > 10:
                log(f"FAILED: {k} failed {fails} times"); return 1
            time.sleep(retry_wait)
    log("CONTROLLER_DONE")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("produce", "recv"):
        p = sub.add_parser(c)
        p.add_argument("--seq", required=True, choices=fixenv.NCLT9)
        p.add_argument("--start", type=int, default=0)
        if c == "produce":
            p.add_argument("--limit", type=int, default=0)
    sub.add_parser("status")
    p = sub.add_parser("controller")
    p.add_argument("--seqs", nargs="*", default=None)
    p.add_argument("--retry-wait", type=int, default=120)
    a = ap.parse_args()
    if a.cmd == "produce":
        stream.cmd_produce(ROLE, a.seq, a.start, a.limit)
    elif a.cmd == "recv":
        (fixenv.NEW_NAS / "tmp").mkdir(parents=True, exist_ok=True)
        sys.exit(stream.cmd_recv(ROLE, a.seq, a.start))
    elif a.cmd == "status":
        stream.cmd_status(ROLE)
    else:
        sys.exit(cmd_controller(a.seqs, a.retry_wait))


if __name__ == "__main__":
    main()
