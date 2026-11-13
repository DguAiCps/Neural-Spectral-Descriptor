#!/usr/bin/env python3
"""키프레임 원시 스캔 로컬 -> 서버 스트리밍과 서버 공식 전처리(단층 BEV) (2026-11-07).

서버에는 시험·검증 원시 자료가 없다(inventory.json). 로컬은 키프레임 원시 파일만 읽어 무손실
페이로드로 보내고(codec.py, 1 스레드), 서버 receiver가 공식 C++ 드라이버(fresco_prep_*)로
단층 BEV(uint8 224x224)를 만들어 NAS에만 쓴다. 원시 점은 서버 디스크에 남기지 않는다
(드라이버 작업 파일 1개를 /dev/shm에서 덮어쓴다).

  produce --role R --seq S --start K [--limit N]   로컬 컨테이너: stdout 스트림
  recv    --role R --seq S --start K               서버 호스트: stdin -> NAS/<role>/<name>/
  status  [--role R]                                서버 호스트: JSON 진행 상황
  controller --role R [--seqs ...] [--limit N]      로컬 호스트: 시퀀스마다 produce | ssh recv,
                                                    재개·재시도·디스크 가드 대기
NAS/<role>/<name>/: bev.u8 (n x 224 x 224), stats.i32 (n x 4: 점 수, 정렬 점 수, label0 수,
BEV 비0 칸 수), progress.json, stream_done.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FAMILY, MIN_NAS_FREE, MIN_ROOT_FREE, NAS_ROOT, RESULTS, ROLES, SERVER,  # noqa
                    SERVER_REPO, cache_path, free_bytes, key, name, sha256, write_json)
import codec  # noqa: E402

BEV_BYTES = 224 * 224
COMMIT_EVERY = 200
RC_DISK = 75
RC_RESUME = 76


def seq_dir(role, sensor, seq) -> Path:
    return NAS_ROOT / role / name(sensor, seq)


# ------------------------------------------------------------------ producer (local)
def cmd_produce(role, k, start, limit):
    import numpy as np
    from common import raw_files
    sensor, seq = k.split("/")
    fam = FAMILY[sensor]
    cp = cache_path(role, sensor, seq)
    with np.load(cp) as z:
        sids = z["scan_ids"].astype(np.int64)
    files = raw_files(sensor, seq)
    n = len(sids) if not limit else min(limit, len(sids))
    out = sys.stdout.buffer
    hdr = json.dumps({"role": role, "seq": k, "family": fam, "n": int(n),
                      "n_total": int(len(sids)), "start": int(start),
                      "cache": cp.name, "cache_sha256": sha256(cp)}).encode()
    out.write(len(hdr).to_bytes(4, "little") + hdr)
    for i in range(start, n):
        raw = Path(files[int(sids[i])]).read_bytes()
        out.write(codec.pack(i, int(sids[i]), fam, raw))
    out.write(codec.REC.pack(-1, n, 0, 0, b"\0" * 32))
    out.flush()


# ------------------------------------------------------------------ receiver (server)
def _read(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"stream truncated ({len(b)}/{n})")
    return b


def _guard():
    root, nas = free_bytes(Path("/")), free_bytes(NAS_ROOT)
    return root >= MIN_ROOT_FREE and nas >= MIN_NAS_FREE, {"root_free": root, "nas_free": nas}


def cmd_recv(role, k, start):
    import numpy as np
    sensor, seq = k.split("/")
    fam = FAMILY[sensor]
    d = seq_dir(role, sensor, seq)
    d.mkdir(parents=True, exist_ok=True)
    prog_p = d / "progress.json"
    prog = json.loads(prog_p.read_text()) if prog_p.exists() else {"committed": 0}
    if (d / "stream_done.json").exists():
        print(f"ALREADY_DONE {k}", file=sys.stderr)
        sys.stdin.buffer.read()
        return 0
    if start != prog["committed"]:
        print(f"RESUME_MISMATCH {k} start={start} committed={prog['committed']}", file=sys.stderr)
        return RC_RESUME
    ok, fr = _guard()
    if not ok:
        print(f"DISK_GUARD {k} {fr}", file=sys.stderr)
        return RC_DISK
    f = sys.stdin.buffer
    hdr = json.loads(_read(f, int.from_bytes(_read(f, 4), "little")))
    cp = cache_path(role, sensor, seq)
    assert hdr["seq"] == k and hdr["family"] == fam and hdr["start"] == start, hdr
    assert hdr["cache_sha256"] == sha256(cp), "local and server caches differ"
    with np.load(cp) as z:
        n_cache = len(z["scan_ids"])
    assert hdr["n_total"] == n_cache, (hdr, n_cache)
    for p, unit in (("bev.u8", BEV_BYTES), ("stats.i32", 16)):   # 확정분까지 자른다
        with open(d / p, "ab") as fh:
            fh.truncate(prog["committed"] * unit)
    shm = Path(f"/dev/shm/fresco_{name(sensor, seq)}_{os.getpid()}")
    shm.mkdir(parents=True, exist_ok=True)
    exe = NAS_ROOT / "build" / f"fresco_prep_{codec.DRIVER[fam]}"
    drv = subprocess.Popen([str(exe), str(shm)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=open(d / "driver_stderr.log", "ab"))
    fb, fs = open(d / "bev.u8", "ab"), open(d / "stats.i32", "ab")
    t0, done, expect = time.time(), prog["committed"], start
    bev_buf, st_buf = [], []

    def commit():
        nonlocal bev_buf, st_buf
        fb.write(b"".join(bev_buf)); fs.write(b"".join(st_buf))
        fb.flush(); fs.flush(); os.fsync(fb.fileno()); os.fsync(fs.fileno())
        bev_buf, st_buf = [], []
        write_json(prog_p, {"committed": done, "n": hdr["n"], "n_total": hdr["n_total"],
                            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "sec_this_run": round(time.time() - t0, 1)})
    rc = 0
    try:
        while True:
            idx, sid, plen, olen, dig = codec.REC.unpack(_read(f, codec.REC.size))
            if idx == -1:
                assert sid == hdr["n"] and done == hdr["n"], (sid, done, hdr["n"])
                break
            assert idx == expect, (idx, expect)
            official = codec.decode(fam, zlib.decompress(_read(f, plen)))
            assert len(official) == olen and codec.sha(official) == dig, f"sha mismatch at {idx}"
            drv.stdin.write(len(official).to_bytes(4, "little") + official)
            drv.stdin.flush()
            out = drv.stdout.read(BEV_BYTES + 16)
            assert len(out) == BEV_BYTES + 16, "driver died"
            bev_buf.append(out[:BEV_BYTES]); st_buf.append(out[BEV_BYTES:])
            done += 1; expect += 1
            if done % COMMIT_EVERY == 0:
                commit()
                ok, fr = _guard()
                if not ok:
                    print(f"DISK_GUARD {k} at {done} {fr}", file=sys.stderr)
                    rc = RC_DISK
                    break
        commit()
    finally:
        try:
            drv.stdin.write((0).to_bytes(4, "little")); drv.stdin.close()
        except Exception:
            pass
        drv.wait(timeout=60)
        fb.close(); fs.close()
        for p in shm.glob("**/*"):
            if p.is_file():
                p.unlink()
        for p in sorted(shm.glob("**/*"), reverse=True):
            p.rmdir()
        shm.rmdir()
    if rc:
        return rc
    if done == hdr["n_total"]:
        st = np.fromfile(d / "stats.i32", dtype="<i4").reshape(-1, 4)
        assert len(st) == done and os.path.getsize(d / "bev.u8") == done * BEV_BYTES
        write_json(d / "stream_done.json", {
            "seq": k, "role": role, "family": fam, "n": done, "cache": cp.name,
            "cache_sha256": hdr["cache_sha256"], "bev_sha256": sha256(d / "bev.u8"),
            "driver": exe.name, "driver_sha256": sha256(exe),
            "empty_bev_frames": int((st[:, 3] == 0).sum()),
            "points_mean": float(st[:, 0].mean()), "bev_nonzero_mean": float(st[:, 3].mean()),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"RECV_OK {k} committed={done}/{hdr['n_total']} sec={time.time() - t0:.0f}",
          file=sys.stderr)
    return 0


def cmd_status(role):
    out = {}
    for r, (seqs, _, _) in ROLES.items():
        if role and r != role:
            continue
        for sensor, seq in seqs:
            d = seq_dir(r, sensor, seq)
            p = d / "progress.json"
            out[f"{r}:{key(sensor, seq)}"] = {
                "committed": json.loads(p.read_text())["committed"] if p.exists() else 0,
                "done": (d / "stream_done.json").exists()}
    ok, fr = _guard()
    print(json.dumps({"seqs": out, "guard_ok": ok, **fr}))


# ------------------------------------------------------------------ controller (local host)
def _ssh(cmd, **kw):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", SERVER,
                           cmd], **kw)


def cmd_controller(role, seqs, limit, retry_wait):
    seqs = seqs or [key(s, q) for s, q in ROLES[role][0]]
    LOCAL_STATUS = RESULTS / f"transfer_status_{role}.json"   # 검증 kitti/09 시범 기록: transfer_status.json
    status = json.loads(LOCAL_STATUS.read_text()) if LOCAL_STATUS.exists() else {}
    status.update({"pid": os.getpid(), "role": role, "started_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seqs": seqs})
    status.setdefault("log", [])

    def log(msg):
        status["log"] = (status["log"] + [time.strftime("%H:%M:%SZ ", time.gmtime()) + msg])[-200:]
        status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(LOCAL_STATUS, status)
        print(msg, flush=True)
    remote = f"cd {SERVER_REPO} && "
    env = ("env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
           f"TMPDIR={NAS_ROOT}/tmp nice -n 10 ionice -c3 python3 scripts/fresco_official/stream.py")
    for k in seqs:
        fails = 0
        while True:
            st = _ssh(remote + f"{env} status --role {role}", capture_output=True, text=True)
            if st.returncode != 0:
                fails += 1
                log(f"status ssh failed rc={st.returncode} ({fails})")
                if fails > 20:
                    log("FAILED: ssh status repeatedly failing"); return 1
                time.sleep(retry_wait); continue
            info = json.loads(st.stdout.strip().splitlines()[-1])
            s = info["seqs"][f"{role}:{k}"]
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
            lim = f" --limit {limit}" if limit else ""
            pipe = ("set -o pipefail; docker exec -w /workspace/Neural-Spectral-Codec ring_baseline "
                    "nice -n 10 taskset -c 13 env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "
                    f"python3 scripts/fresco_official/stream.py produce --role {role} --seq {k} "
                    f"--start {start}{lim} | ssh -o BatchMode=yes -o ServerAliveInterval=30 {SERVER} "
                    f"'{remote}{env} recv --role {role} --seq {k} --start {start}'")
            r = subprocess.run(["bash", "-c", pipe], capture_output=True, text=True)
            tail = (r.stderr or "").strip().splitlines()[-3:]
            log(f"stream {k} rc={r.returncode} in {time.time() - t0:.0f} s: {' | '.join(tail)}")
            if r.returncode == 0:
                if limit:
                    log(f"limit {limit} reached for {k}"); break
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
        p.add_argument("--role", required=True, choices=list(ROLES))
        p.add_argument("--seq", required=True)
        p.add_argument("--start", type=int, default=0)
        if c == "produce":
            p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("status"); p.add_argument("--role", default=None)
    p = sub.add_parser("controller")
    p.add_argument("--role", required=True, choices=list(ROLES))
    p.add_argument("--seqs", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--retry-wait", type=int, default=120)
    a = ap.parse_args()
    if a.cmd == "produce":
        cmd_produce(a.role, a.seq, a.start, a.limit)
    elif a.cmd == "recv":
        sys.exit(cmd_recv(a.role, a.seq, a.start))
    elif a.cmd == "status":
        cmd_status(a.role)
    else:
        sys.exit(cmd_controller(a.role, a.seqs, a.limit, a.retry_wait))


if __name__ == "__main__":
    main()
