#!/usr/bin/env python3
"""SeqOT test inputs for the paper-recipe revision (2026-11-07): bounded
streaming of the loader points local -> server, official range projection on
the server, only the final range images stored.

Why: the server has no raw test data and the old test range stores were
removed; projecting locally for hours is not allowed.  The local side only
READS the needed raw scans with the original loaders and streams their
coordinates (1 pinned CPU thread, no intermediate file); the server projects
them with _seqot_official_prep.range_image (official range_projection, same
FOV / parameters) and stores the rows in a new store.

Needed rows per TEST sequence: _seqot_official_prep.centered_raw_ids(kf,
n_raw) = raw scans [s-11, s+10] of every KISS-ICP keyframe s (official
centred windows of the main table), kf from data/preprocessed_v2_odom.

Payload (lossless for the projection):
  kind 0: loader points[:, :3] float32 (official_points uses [:, :3] -> float64)
  kind 1 (NCLT): u = rint((p64 * [1, 1, -1] + 100) / 0.005) as uint16 with
          the 1e-3 grid assertion of nclt_official_coords; the server
          rebuilds xyz = u * 0.005 - 100.0 (float64) @ _R_FLIP exactly as
          nclt_official_coords does.
Stream: [u32 len][header json] then per row [i32 raw][i32 n][u8 kind][payload],
trailer [i32 -1][i32 rows][u8 0][32-byte sha256 of all payloads].

Store: _external/seqot_paper_recipe_20261107/test_inputs/<sensor>_<seq>/
  range.npy (rows of need, float32 (32, 900)), progress.json, index.npz, done.json;
  test_inputs/READY when all 46 are done (finalize).

Subcommands
  produce --seq S --start K      local container: framed stream on stdout
  recv --seq S --start K         server container: stdin -> store
  status [--seq S]               server container: JSON progress
  finalize                       server container: READY marker
  controller                     local host: per sequence produce | ssh recv,
                                 resumable, bounded retry with backoff
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
sys.dont_write_bytecode = True

TAG = "seqot_paper_recipe_20261107"
STORE = REPO / "_external" / TAG / "test_inputs"
CHAIN_RUNS = REPO / "_external" / TAG
MIN_FREE = 20 * 2 ** 30
CKPT_BYTES = 142 * 2 ** 20
REC = struct.Struct("<iiB")
H, W = 32, 900
LOCAL_STATUS = REPO / "results/pdv2_baseline_revision_20261107/seqot/test_inputs_transfer.json"
SERVER = "joms-gpu"
SERVER_REPO = "/home/qkrrlans00/Neural-Spectral-Codec"
SCRIPT = "scripts/_review_seqot_paper_recipe_20261107_testinputs.py"


def P():
    import _seqot_official_prep as prep
    return prep


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def needed(sensor, seq, n_raw):
    p = P()
    with np.load(p.cache_path("test_odom", sensor, seq)) as z:
        kf = z["scan_ids"].astype(np.int64)
        poses = z["poses_gt"]
    return kf, poses, p.centered_raw_ids(kf, n_raw)


def make_loader(sensor, seq):
    """scripts/_pd_build_test_caches._make_loader and
    scripts/_pd_build_val_layouts._make_loader (same constructors, same
    NSD_DATA_ROOT), without the torch-importing module chain."""
    root = Path(os.environ.get("NSD_DATA_ROOT", "/workspace/data"))
    if sensor == "kitti360":
        from data.kitti360_loader import Kitti360Loader
        return Kitti360Loader(str(root / "kitti360"), seq, lazy_load=True)
    if sensor == "mcd":
        from data.mcd_loader import MCDLoader
        return MCDLoader(str(root / "mcd_extracted"), seq, lazy_load=True)
    if sensor == "nclt":
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(str(root / "nclt"), seq, lazy_load=True)
    if sensor == "helipr":
        from data.helipr_loader import HeLiPRLoader
        seq_path = root / "helipr" / seq
        if not (seq_path / "LiDAR").is_dir():
            seq_path = seq_path / seq
        return HeLiPRLoader(str(seq_path), lazy_load=True)
    if sensor == "mulran":
        from data.mulran_loader import MulRanLoader
        return MulRanLoader(str(root / "mulran"), seq, lazy_load=True)
    if sensor == "kitti":
        from data.kitti_loader import KITTILoader
        return KITTILoader(str(root / "kitti/dataset"), seq, lazy_load=True)
    raise ValueError(sensor)


def nclt_grid(points):
    """nclt_official_coords up to u (same arithmetic, same assertion)."""
    raw = np.asarray(points, dtype=np.float64)[:, :3].copy()
    raw[:, 2] = -raw[:, 2]
    u = np.rint((raw + 100.0) / 0.005)
    assert np.all(np.abs(raw - (u * 0.005 - 100.0)) < 1e-3), "not on the NCLT 5 mm grid"
    assert u.min() >= 0 and u.max() <= 65535
    return u.astype(np.uint16)


def readn(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"stream ended ({len(b)}/{n} bytes)")
    return b


# ------------------------------------------------------------ local producer
def cmd_produce(key, start):
    sensor, seq = key.split("/")
    L = make_loader(sensor, seq)
    n_raw = len(L)
    kf, _, need = needed(sensor, seq, n_raw)
    hdr = {"seq": key, "n_raw": int(n_raw), "n_need": int(len(need)), "start": int(start),
           "need_sha256": hashlib.sha256(need.tobytes()).hexdigest(),
           "kf_sha256": hashlib.sha256(kf.tobytes()).hexdigest(),
           "kind": 1 if sensor == "nclt" else 0}
    out = sys.stdout.buffer
    hb = json.dumps(hdr).encode()
    out.write(struct.pack("<I", len(hb)) + hb)
    h = hashlib.sha256()
    for k in range(start, len(need)):
        r = int(need[k])
        pts = np.asarray(L[r]["points"])
        assert pts.dtype == np.float32 and pts.ndim == 2 and pts.shape[1] >= 3, (r, pts.dtype, pts.shape)
        if sensor == "nclt":
            pay = np.ascontiguousarray(nclt_grid(pts)).tobytes()
        else:
            pay = np.ascontiguousarray(pts[:, :3]).tobytes()
        h.update(pay)
        out.write(REC.pack(r, len(pts), hdr["kind"]) + pay)
    out.write(REC.pack(-1, len(need) - start, 0) + h.digest())
    out.flush()


# ------------------------------------------------------------ server receiver
def sdir(key):
    return STORE / key.replace("/", "_")


def progress(key):
    d = sdir(key)
    if (d / "done.json").exists():
        return {"seq": key, "done": True, **json.loads((d / "done.json").read_text())}
    if (d / "progress.json").exists():
        return {"seq": key, "done": False, **json.loads((d / "progress.json").read_text())}
    return {"seq": key, "done": False, "rows_done": 0}


def ckpt_reserve():
    have = len(list(CHAIN_RUNS.glob("s*/runs/stage1/amodel_seqot*.pth.tar")))
    return max(0, 60 - have) * CKPT_BYTES + 2 * 2 ** 30


def cmd_recv(key, start):
    p = P()
    sensor, seq = key.split("/")
    f = sys.stdin.buffer
    (n,) = struct.unpack("<I", readn(f, 4))
    hdr = json.loads(readn(f, n))
    assert hdr["seq"] == key and hdr["start"] == start, hdr
    kf, poses, need = needed(sensor, seq, hdr["n_raw"])
    assert hashlib.sha256(kf.tobytes()).hexdigest() == hdr["kf_sha256"], "keyframes differ local/server"
    assert hashlib.sha256(need.tobytes()).hexdigest() == hdr["need_sha256"], "needed rows differ"
    d = sdir(key)
    d.mkdir(parents=True, exist_ok=True)
    prog = progress(key)
    assert not prog["done"], f"{key} already done"
    assert prog["rows_done"] == start, (prog["rows_done"], start)
    path = d / "range.npy"
    shape = (len(need), H, W)
    unwritten = (len(need) - start) * H * W * 4
    free = shutil.disk_usage(d).free
    if free - unwritten - ckpt_reserve() < MIN_FREE:
        write_json(d / "disk_pause.json", {"free": free, "unwritten": unwritten, "reserve": ckpt_reserve(),
                                           "min_free": MIN_FREE, "utc": time.time()})
        print(f"DISK_PAUSE {key}", file=sys.stderr, flush=True)
        sys.exit(75)
    mm = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float32, shape=shape)
    up, down = p.FOV[sensor]
    rp = p.official_utils().range_projection
    h = hashlib.sha256()
    sessions = prog.get("sessions", [])
    t0 = time.time()
    k = start
    while True:
        r, npts, kind = REC.unpack(readn(f, REC.size))
        if r == -1:
            assert npts == k - start, (npts, k, start)
            dig = readn(f, 32)
            assert dig == h.digest(), "payload sha256 mismatch"
            break
        assert k < len(need) and r == int(need[k]) and kind == hdr["kind"], (k, r, kind)
        if kind == 1:
            u = np.frombuffer(readn(f, npts * 6), dtype=np.uint16).reshape(npts, 3)
            h.update(u.tobytes())
            xyz = u.astype(np.float64) * 0.005 - 100.0
            img = rp(np.matmul(xyz, p._R_FLIP), fov_up=up, fov_down=down, **p.PROJ)[0]
        else:
            pts = np.frombuffer(readn(f, npts * 12), dtype=np.float32).reshape(npts, 3)
            h.update(pts.tobytes())
            img = p.range_image(sensor, pts)
        assert img.shape == (H, W)
        mm[k] = img
        k += 1
        if k % 256 == 0:
            mm.flush()
            write_json(d / "progress.json", {"rows_done": k, "n_need": len(need), "sessions": sessions,
                                             "s": round(time.time() - t0, 1)})
    mm.flush()
    del mm
    sessions.append({"start": start, "end": k, "payload_sha256": h.hexdigest(), "s": round(time.time() - t0, 1)})
    assert k == len(need), (k, len(need))
    np.savez(d / "index.npz", raw_ids=need, n_raw=np.int64(hdr["n_raw"]), kf_scan_ids=kf, poses_gt=poses)
    hs = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            hs.update(b)
    write_json(d / "done.json", {"rows_done": k, "n_need": len(need), "n_raw": hdr["n_raw"],
                                 "n_keyframes": int(len(kf)), "sessions": sessions,
                                 "range_sha256": hs.hexdigest(),
                                 "cache": str(p.cache_path("test_odom", sensor, seq).relative_to(REPO)),
                                 "projection": "_seqot_official_prep.range_image (official range_projection)",
                                 "payload_kind": hdr["kind"], "utc": time.time()})
    print(f"OK {key} {k} rows", file=sys.stderr, flush=True)


def cmd_status(key):
    keys = [key] if key else [f"{s}/{q}" for s, q in P().TEST]
    out = [progress(k) for k in keys]
    for o in out:
        o.pop("sessions", None)
    print(json.dumps({"store": str(STORE), "free": shutil.disk_usage(REPO).free,
                      "ready": (STORE / "READY").exists(), "seqs": out}))


def cmd_finalize():
    keys = [f"{s}/{q}" for s, q in P().TEST]
    assert len(keys) == 46
    st = {k: progress(k) for k in keys}
    bad = [k for k, v in st.items() if not v["done"]]
    assert not bad, bad
    write_json(STORE / "READY.json", {"sequences": len(keys), "rows": int(sum(v["rows_done"] for v in st.values())),
                                      "range_sha256": {k: v["range_sha256"] for k, v in st.items()}})
    (STORE / "READY").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print("READY", flush=True)


# ------------------------------------------------------------ local controller
def ssh(cmd, timeout=120):
    return subprocess.run(["timeout", str(timeout), "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                           "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", SERVER, cmd],
                          capture_output=True, text=True)


def server_py(args):
    return (f"docker exec -i -w /workspace/Neural-Spectral-Codec -e OMP_NUM_THREADS=1 ring_baseline "
            f"python3 {SCRIPT} {args}")


def cmd_controller():
    import _pdv2_pilot
    keys = [f"{s}/{q}" for s, q in _pdv2_pilot.TEST]
    state = {"pid": os.getpid(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "keys": len(keys), "done": [], "log": []}

    def log(msg):
        state["log"] = (state["log"] + [f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}"])[-60:]
        state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(LOCAL_STATUS, state)
        print(msg, flush=True)

    fails = 0
    i = 0
    while i < len(keys):
        key = keys[i]
        r = ssh(server_py(f"status --seq {key}"))
        if r.returncode != 0:
            fails += 1
            log(f"status {key} ssh rc={r.returncode} {r.stderr[-200:]!r} (fail {fails})")
            if fails > 12:
                log("FAILED: server unreachable")
                ssh(f"mkdir -p {SERVER_REPO}/_external/{TAG}/test_inputs && "
                    f"echo controller_unreachable > {SERVER_REPO}/_external/{TAG}/test_inputs/FAILED", 60)
                sys.exit(1)
            time.sleep(min(600, 30 * 2 ** min(fails, 5)))
            continue
        st = json.loads(r.stdout.strip().splitlines()[-1])["seqs"][0]
        if st["done"]:
            if key not in state["done"]:
                state["done"].append(key)
            log(f"done {key} ({len(state['done'])}/{len(keys)})")
            i += 1
            fails = 0
            continue
        start = int(st["rows_done"])
        log(f"stream {key} from row {start}")
        t0 = time.time()
        prod = subprocess.Popen(
            ["docker", "exec", "-w", "/workspace/Neural-Spectral-Codec", "-e", "OMP_NUM_THREADS=1",
             "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "ring_baseline",
             "taskset", "-c", "15", "nice", "-n", "19", "ionice", "-c", "3",
             "python3", SCRIPT, "produce", "--seq", key, "--start", str(start)],
            stdout=subprocess.PIPE)
        cons = subprocess.Popen(
            ["taskset", "-c", "14", "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
             "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6", SERVER,
             server_py(f"recv --seq {key} --start {start}")],
            stdin=prod.stdout)
        prod.stdout.close()
        rc_c = cons.wait()
        rc_p = prod.wait()
        dt = time.time() - t0
        if rc_c == 75:
            log(f"DISK_PAUSE {key}: server free space guard (20 GiB + checkpoint reserve); retry in 30 min")
            time.sleep(1800)
            continue
        if rc_c != 0 or rc_p != 0:
            fails += 1
            log(f"stream {key} failed rc_prod={rc_p} rc_recv={rc_c} after {dt:.0f} s (fail {fails})")
            if fails > 12:
                log("FAILED: repeated stream failures")
                ssh(f"echo stream_failures_{key.replace('/', '_')} > {SERVER_REPO}/_external/{TAG}/test_inputs/FAILED", 60)
                sys.exit(1)
            time.sleep(min(600, 30 * 2 ** min(fails, 5)))
            continue
        fails = 0
        log(f"streamed {key} in {dt:.0f} s")
    r = ssh(server_py("finalize"), 600)
    log(f"finalize rc={r.returncode} {r.stdout[-200:]!r} {r.stderr[-300:]!r}")
    if r.returncode != 0:
        ssh(f"echo finalize_failed > {SERVER_REPO}/_external/{TAG}/test_inputs/FAILED", 60)
        sys.exit(1)
    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log("TEST_INPUTS_READY")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("produce", "recv"):
        p = sub.add_parser(c)
        p.add_argument("--seq", required=True)
        p.add_argument("--start", type=int, default=0)
    p = sub.add_parser("status")
    p.add_argument("--seq", default=None)
    sub.add_parser("finalize")
    sub.add_parser("controller")
    a = ap.parse_args()
    if a.cmd == "produce":
        cmd_produce(a.seq, a.start)
    elif a.cmd == "recv":
        cmd_recv(a.seq, a.start)
    elif a.cmd == "status":
        cmd_status(a.seq)
    elif a.cmd == "finalize":
        cmd_finalize()
    elif a.cmd == "controller":
        cmd_controller()


if __name__ == "__main__":
    main()
