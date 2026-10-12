#!/usr/bin/env python3
"""HeLiOS official pipeline on split v2 -- part 1: scans (2026-10-11).

Turns split-v2 keyframes into the OFFICIAL HeLiOS training/evaluation input:
one HeLiPR-Pointcloud-Toolbox output scan per keyframe (official HeLiOS
config _external/HeLiOS/datasets/preprocess/helipr_toolbox_config.yaml:
accumulatedSize 1 = single scan, crop 100 m sphere, downsample to exactly
8192 points, normalise x/100), stored as the toolbox writes it
(float32 x, y, z, intensity per point, .bin) and read later by the official
datasets/pnv_raw.PNVPointCloudLoader.

Official code executed (compiled unchanged into
_external/helios_env/build/libhelios_toolbox.so, see build_reference.sh):
  toolbox src/utility.cpp  down_sampling_voxel (voxel centroids),
                           normalizePointCloud (x/cropSize; intensity scaling)
Transcribed (each checked against the official code in `selftest`):
  * the scan read + accumulate loop of PointCloudProcessor.cpp (readBinFile
    `while (!file.eof())` appends one more point = a copy of the last one;
    accumulateScans iterates from the last point down to, and excluding,
    point 0; its per-scan transform q0^-1((q0 p + t0) - t0) is the identity
    except -0.0 -> +0.0; crop `x*x + y*y + z*z > cropSize*cropSize` in float);
  * the voxel-size search of downsample_to_target_size (utility.cpp:59-85,
    in the ctypes wrapper).  The official function then fills the missing
    points with `rand() % n` after `srand(time(nullptr))` (not reproducible);
    we draw the same kind of fill (uniform, with replacement, from the cropped
    scan) from numpy default_rng([1, crc32(sensor/seq), scan_id]).
Motion compensation (official toolbox: undistortFlag True, B-spline over the
LiDAR trajectory and per-point times).  Per-point times exist only in the
HeLiPR files, so `index --deskew helipr` (default) undistorts HeLiPR scans with
the OFFICIAL toolbox binary itself (downsampling / normalising / cropping off
there, then the adapter below): TRAIN / VAL with LiDAR_GT/Velodyne_gt.txt (the
official trajectory), TEST with the KISS-ICP odometry of every scan
(results/pdv2_c3/odometry, GT may not be used at test; the B-spline reads the
odometry of the next one or two scans).  KITTI, KITTI-360, MulRan, NCLT and MCD
scans carry no per-point times in our loaders and are used as they are.
`--deskew none` turns it off everywhere.  `deskew-effect` measures it.
Keyframe spacing (official code value; project rule 2026-10-11: the public
code's actual behaviour, not paper values): the toolbox keeps a scan when it
is more than distanceThreshold = 3 m (helipr_toolbox_config.yaml:23) from the
last kept one (the paper says 5 m; the released weights carry no evidence
either way).  `index --train-spacing` applies that rule to the split-v2 TRAIN
keyframe stream (default 3; 0 = all keyframes).  VAL and TEST keep every
keyframe; VAL Roundabout01 (official validation sequence) also gets the 3 m
mask for the validation pickle.

Layout (official dataset folder; rel paths go into the pickles):
  _external/helios_official_pdv2/data/{role}/{sensor}_{seq}/LiDAR/{scan_id:06d}.bin
  _external/helios_official_pdv2/data/{role}/{sensor}_{seq}/{index.npz,meta.json}
roles: train (v2 TRAIN, GT keyframe caches), val (v2 VAL, GT keyframe caches),
test_odom (v2 TEST, KISS-ICP keyframe caches; poses_gt for scoring only).

Subcommands (inside ring_baseline):
  index [--train-spacing 3] [--deskew helipr|none]   index.npz + meta.json
  jobs [--chunk 2000]         "role seq shard nshards" generation jobs
  gen --role R --seq S [--shard k --nshards n]   write .bin scans (resumable)
  status                      count .bin files against the index
  selftest [--n-scans 3]      adapter == official toolbox binary (6 sensors,
                              HeLiPR also with undistortion on)
  deskew-effect [--n-scans 4] official toolbox with/without undistortion
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

from _pdv2_pilot import TEST, TRAIN, VAL  # noqa: E402  (split-v2 lists)

HELIOS = REPO / "_external/HeLiOS"
HELIOS_COMMIT = "6915b75"          # resolved to the full hash at run time
TOOLBOX = REPO / "_external/HeLiPR-Pointcloud-Toolbox"
TOOLBOX_COMMIT = "9c2fbf8a825671365474261ad3ba31bd2dba849c"
HENV = REPO / "_external/helios_env"
LIB = HENV / "build/libhelios_toolbox.so"
TOOL_BIN = HENV / "build/toolbox/helipr_tool"
OVERLAP_BIN = HENV / "build/overlap/overlap"
TOOLBOX_CFG = HELIOS / "datasets/preprocess/helipr_toolbox_config.yaml"
WORK = REPO / "_external/helios_official_pdv2"
DATA = WORK / "data"                 # official `dataset_folder`
V2 = REPO / "data/preprocessed_v2"
ODOM = REPO / "data/preprocessed_v2_odom"
# seed of the reproducible random fill (fill_rng); FIXED at 1 and not read from
# the environment: the prepared scans are shared by every training seed
FILL_SEED = 1
LOG_DIR = REPO / "logs/baseline_official_20261011"
# role -> (sequence list, cache dir, cache-name middle, pose key in cache)
ROLES = {
    "train": (TRAIN, V2, "", "poses"),
    "val": (VAL, V2, "val_", "poses"),
    "test_odom": (TEST, ODOM, "test_", "poses_gt"),
}
# official validation set (generate_training_sets.py:87 validation_set =
# ["Roundabout01"]); Roundabout01 is a split-v2 VAL sequence
OFFICIAL_VAL = [("helipr", "Roundabout01")]
# sequences whose cache poses share one frame (results/split_audit/
# geo_overlap.json frames + cache check: Bridge01-04 = local ENU of
# Velodyne_gt.txt); every other sequence is its own frame
FRAME_GROUP = {f"helipr/Bridge0{i}": "helipr_bridge" for i in (1, 2, 3, 4)}
# official keyframe interval: distanceThreshold: 3
# (_external/HeLiOS/datasets/preprocess/helipr_toolbox_config.yaml:23,
# asserted in official_toolbox_config); the paper's 5 m is NOT used
TOOLBOX_SPACING_M = 3.0


def seq_name(sensor: str, seq: str) -> str:
    return f"{sensor}_{seq}"


def seq_dir(role: str, sensor: str, seq: str) -> Path:
    return DATA / role / seq_name(sensor, seq)


def rel_scan_path(role: str, sensor: str, seq: str, sid: int) -> str:
    """Path relative to the official dataset_folder (pickle rel_scan_filepath)."""
    return f"{role}/{seq_name(sensor, seq)}/LiDAR/{int(sid):06d}.bin"


def frame_group(sensor: str, seq: str) -> str:
    return FRAME_GROUP.get(f"{sensor}/{seq}", f"{sensor}_{seq}")


def cache_path(role: str, sensor: str, seq: str) -> Path:
    _, cdir, mid, _ = ROLES[role]
    hits = sorted(glob.glob(str(cdir / f"cache_????????_{sensor}_{mid}{seq}.npz")))
    assert len(hits) == 1, (role, sensor, seq, hits)
    return Path(hits[0])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git_head(repo: Path) -> str:
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repo), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def official_toolbox_config() -> dict:
    """The official HeLiOS toolbox config; asserts the values this pipeline
    implements (anything else would need code changes here)."""
    import yaml
    cfg = yaml.safe_load(TOOLBOX_CFG.read_text())
    s, u = cfg["Save"], cfg["Undistort"]
    assert s["downSampleFlag"] is True and s["downsamplePointSize"] == 8192, s
    assert s["normalizeFlag"] is True and s["cropFlag"] is True and s["cropSize"] == 100, s
    assert s["accumulatedSize"] == 1 and s["accumulatedStep"] == 1, s
    assert s["saveAs"] == "bin" and s["distanceThreshold"] == 3, s
    assert u["undistortFlag"] is True and u["numIntervals"] == 250, u
    return cfg


# ------------------------------------------------------------ official toolbox (ctypes)
class Toolbox:
    """Official toolbox utility.cpp through libhelios_toolbox.so."""

    def __init__(self):
        assert LIB.exists(), f"{LIB} missing: bash _external/helios_env/build_reference.sh"
        lib = ctypes.CDLL(str(LIB))
        F = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
        i, d = ctypes.c_int, ctypes.c_double
        lib.helios_target_centroids.argtypes = [F, i, i, F, i, ctypes.POINTER(d)]
        lib.helios_official_downsample_to_target_size.argtypes = [F, i, i, F, i]
        lib.helios_down_sampling_voxel.argtypes = [F, i, d, F, i]
        lib.helios_normalize.argtypes = [F, i, d]
        lib.helios_normalize.restype = None
        self.lib = lib

    def target_centroids(self, acc: np.ndarray, target: int):
        out = np.zeros((len(acc) + 1, 4), np.float32)
        scale = ctypes.c_double()
        n = self.lib.helios_target_centroids(acc, len(acc), target, out, len(out),
                                             ctypes.byref(scale))
        assert n >= 0, n
        return out[:n].copy(), scale.value

    def official_downsample(self, acc: np.ndarray, target: int) -> np.ndarray:
        out = np.zeros((len(acc) + target, 4), np.float32)
        n = self.lib.helios_official_downsample_to_target_size(acc, len(acc), target, out, len(out))
        assert n >= 0, n
        return out[:n].copy()

    def normalize(self, pts: np.ndarray, crop: float) -> np.ndarray:
        pts = np.ascontiguousarray(pts, dtype=np.float32).copy()
        self.lib.helios_normalize(pts, len(pts), float(crop))
        return pts


def toolbox_order(points: np.ndarray) -> np.ndarray:
    """PointCloudProcessor read + accumulateScans for one scan (accumulatedSize
    1, undistortion off): [copy of the last point] + points[n-1 .. 1] (point 0
    is skipped by the loop), -0.0 -> +0.0."""
    a = np.ascontiguousarray(np.asarray(points)[:, :4], dtype=np.float32)
    if a.shape[1] < 4:
        a = np.concatenate([a, np.zeros((len(a), 4 - a.shape[1]), np.float32)], 1)
    acc = np.concatenate([a[-1:], a[:0:-1]])
    # accumulateScans always applies q0^-1 ((q0 p + t0) - t0) (identity for one
    # scan, exact in double for float32 inputs) -- except that it turns -0.0
    # into +0.0; adding +0.0 does exactly that and nothing else.
    acc[:, :3] += np.float32(0.0)
    return acc


def toolbox_crop(acc: np.ndarray, crop: float = 100.0) -> np.ndarray:
    """accumulateScans: drop x*x + y*y + z*z > cropSize*cropSize (float)."""
    r2 = (acc[:, 0] * acc[:, 0] + acc[:, 1] * acc[:, 1]) + acc[:, 2] * acc[:, 2]
    c = np.float32(crop)
    return np.ascontiguousarray(acc[~(r2 > c * c)])


def toolbox_accumulate(points: np.ndarray, crop: float = 100.0) -> np.ndarray:
    return toolbox_crop(toolbox_order(points), crop)


def fill_rng(sensor: str, seq: str, sid: int) -> np.random.Generator:
    return np.random.default_rng([FILL_SEED, zlib.crc32(f"{sensor}/{seq}".encode()), int(sid)])


def process_accumulated(tb: Toolbox, acc: np.ndarray, rng: np.random.Generator,
                        target: int = 8192, crop: float = 100.0):
    """Toolbox-ordered scan (metres) -> crop, downsample_to_target_size,
    normalizePointCloud -> (target x 4) float32.

    A scan with no point left after the crop (e.g. an empty .bin file) has no
    official output: downsample_to_target_size would compute `rand() % 0`
    (utility.cpp:92).  Run-only fix, like DiSCO's empty image: such a scan
    becomes `target` points at the origin with intensity 0, i.e. exactly what
    normalizePointCloud returns for an all-origin cloud (origin points are also
    what the official RemoveRandomPoints augmentation writes).  It is listed by
    cmd_gen in empty_after_crop_<shard>_of_<n>.txt.  Non-empty scans are
    untouched by this branch."""
    acc = toolbox_crop(acc, crop)
    if len(acc) == 0:
        return (np.zeros((target, 4), np.float32),
                {"n_crop": 0, "n_vox": 0, "voxel_m": None, "empty_after_crop": True})
    cent, scale = tb.target_centroids(acc, target)
    k = target - len(cent)
    fill = acc[rng.integers(0, len(acc), k)] if k > 0 else acc[:0]
    out = tb.normalize(np.concatenate([cent, fill]), crop)
    return out, {"n_crop": int(len(acc)), "n_vox": int(len(cent)), "voxel_m": round(float(scale), 3)}


def process_scan(tb: Toolbox, points: np.ndarray, rng: np.random.Generator,
                 target: int = 8192, crop: float = 100.0):
    """One keyframe (sensor frame, metres) -> official toolbox output."""
    out, info = process_accumulated(tb, toolbox_order(points), rng, target, crop)
    return out, {"n_raw": int(len(points)), **info}


# ------------------------------------------------------------ poses
def _kitti_tr(seq: str) -> np.ndarray:
    from _pd_build_test_caches import DATA_ROOT
    for line in (DATA_ROOT / "kitti/dataset/sequences" / seq / "calib.txt").read_text().splitlines():
        if line.startswith("Tr:"):
            T = np.eye(4)
            T[:3, :] = np.array(line.split()[1:], dtype=np.float64).reshape(3, 4)
            return T
    raise ValueError(seq)


def lidar_poses(sensor: str, seq: str, cache_poses: np.ndarray):
    """LiDAR poses in the cache frame, for the overlap transform, or None.
    HeLiPR caches hold LiDAR_GT/Velodyne_gt.txt rows (LiDAR poses); KITTI caches
    hold camera-0 poses (poses/XX.txt) -> T_world_velo = P_cam0 @ Tr (calib.txt).
    Other datasets are only needed for val R@1 / test scoring (no overlap)."""
    if sensor == "helipr":
        return cache_poses.copy()
    if sensor == "kitti":
        return cache_poses @ _kitti_tr(seq)
    return None


def horizontal(sensor: str, xyz: np.ndarray) -> np.ndarray:
    """(northing, easting)-like horizontal axes: KITTI camera world frame has y
    down, so its horizontal plane is x-z; every other cache is z-up."""
    return xyz[:, [0, 2]] if sensor == "kitti" else xyz[:, :2]


def official_keyframes(positions: np.ndarray, threshold: float) -> np.ndarray:
    """accumulateScans keyframe rule: keep a scan when euclidean_distance(last
    kept, current) > distanceThreshold (Point3D floats; lastPoint starts at
    (-999, -999, -999)).  threshold 0 keeps every keyframe."""
    keep = np.zeros(len(positions), bool)
    p = positions.astype(np.float32)
    last = np.full(3, -999.0, np.float32)
    thr = np.float32(threshold)
    for i in range(len(p)):
        d = p[i] - last
        if np.sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) > thr:
            keep[i] = True
            last = p[i]
    return keep


# ------------------------------------------------------------ index / gen
def cmd_index(roles, train_spacing: float, deskew: str):
    from _pd_build_test_caches import _make_loader
    cfg = official_toolbox_config()
    head = git_head(HELIOS)
    assert head.startswith(HELIOS_COMMIT), head
    for role in roles:
        seqs, _, _, pose_key = ROLES[role]
        for sensor, seq in seqs:
            cp = cache_path(role, sensor, seq)
            with np.load(cp) as z:
                sids = z["scan_ids"].astype(np.int64)
                poses = z[pose_key].astype(np.float64)
                stamps = z["timestamps"].astype(np.float64)
            assert len(sids) == len(poses) and np.all(np.diff(sids) > 0), (role, sensor, seq)
            assert sids.max() < 10 ** 6
            lp = lidar_poses(sensor, seq, poses) if role != "test_odom" else None
            if role == "train":
                spacing = train_spacing
            elif role == "val" and (sensor, seq) in OFFICIAL_VAL:
                spacing = float(cfg["Save"]["distanceThreshold"])
            else:
                spacing = None
            keep = (official_keyframes(lp[:, :3, 3], spacing) if spacing
                    else np.ones(len(sids), bool))
            gen = keep if role == "train" else np.ones(len(sids), bool)
            d = seq_dir(role, sensor, seq)
            dsk = (deskew_trajectory(role, sensor, seq, _make_loader(sensor, seq))
                   if deskew == "helipr" and sensor == "helipr" else None)
            if (d / "meta.json").exists() and any((d / "LiDAR").glob("*.bin")):
                old = json.loads((d / "meta.json").read_text())
                same = (old.get("deskew") or {}).get("kind") == (dsk or {}).get("kind") \
                    and old.get("keyframe_spacing_m") == spacing
                assert same, f"{d}: scans exist with another deskew/spacing setting; remove them first"
            (d / "LiDAR").mkdir(parents=True, exist_ok=True)
            arrays = dict(scan_ids=sids, timestamps=stamps, positions=poses[:, :3, 3],
                          keep=keep, generate=gen)
            if lp is not None:
                arrays["lidar_poses"] = lp
            np.savez(d / "index.npz", **arrays)
            meta = {"role": role, "sensor": sensor, "sequence": seq,
                    "n_frames": int(len(sids)), "n_keep": int(keep.sum()),
                    "n_generate": int(gen.sum()),
                    "keyframe_spacing_m": spacing,
                    "keyframe_rule": "official accumulateScans distanceThreshold rule on "
                                     "the split-v2 keyframe stream (LiDAR positions)"
                                     if spacing else "every split-v2 keyframe",
                    "frame_group": frame_group(sensor, seq),
                    "lidar_poses": None if lp is None else
                    ("cache poses (LiDAR_GT/Velodyne_gt.txt)" if sensor == "helipr"
                     else "cache camera-0 poses @ calib.txt Tr"),
                    "cache": str(cp.relative_to(REPO)), "cache_sha256": sha256(cp),
                    "positions_from": pose_key,
                    "deskew": dsk,
                    "helios_commit": head, "toolbox_commit": TOOLBOX_COMMIT,
                    "loader": "scripts/_pd_build_test_caches._make_loader"}
            (d / "meta.json").write_text(json.dumps(meta, indent=1))
            print(f"OK index {role}/{sensor}/{seq}: {len(sids)} keyframes, "
                  f"keep {keep.sum()}, generate {gen.sum()}", flush=True)


def _index(role, sensor, seq):
    return np.load(seq_dir(role, sensor, seq) / "index.npz")


def cmd_jobs(chunk: int):
    for role, (seqs, _, _, _) in ROLES.items():
        for sensor, seq in seqs:
            n_gen = int(_index(role, sensor, seq)["generate"].sum())
            n = max(1, -(-n_gen // chunk))
            for k in range(n):
                print(role, f"{sensor}/{seq}", k, n)


def cmd_gen(role: str, key: str, shard: int, nshards: int, chunk: int = 400):
    from _pd_build_test_caches import _make_loader
    sensor, seq = key.split("/")
    assert (sensor, seq) in ROLES[role][0], (role, key)
    cfg = official_toolbox_config()["Save"]
    target, crop = int(cfg["downsamplePointSize"]), float(cfg["cropSize"])
    z = _index(role, sensor, seq)
    meta = json.loads((seq_dir(role, sensor, seq) / "meta.json").read_text())
    dsk = meta.get("deskew")
    sids, poses_pos, gen = z["scan_ids"], z["positions"], z["generate"]
    tb = Toolbox()
    loader = _make_loader(sensor, seq)
    d = seq_dir(role, sensor, seq)
    todo = [int(i) for i in np.flatnonzero(gen)[shard::nshards]
            if not (d / "LiDAR" / f"{int(sids[i]):06d}.bin").exists()]
    t0, done, max_dt, nvox, n_fallback, n_empty = time.time(), 0, 0.0, [], 0, 0
    for c in range(0, len(todo), chunk):
        part = todo[c:c + chunk]
        und = {}
        if dsk:   # official toolbox undistortion, one run per chunk
            with tempfile.TemporaryDirectory(dir=str(WORK)) as td:
                und = toolbox_undistort(Path(td), [loader.scan_files[int(sids[i])] for i in part],
                                        dsk["trajectory"])
        for i in part:
            sid = int(sids[i])
            out_p = d / "LiDAR" / f"{sid:06d}.bin"
            stem = Path(loader.scan_files[sid]).stem if dsk else None
            if dsk:   # HeLiPR loader: index space == cache index space
                dt = float(np.abs(loader.scan_poses[sid][:3, 3] - poses_pos[i]).max())
                max_dt = max(max_dt, dt)
                assert dt < 1e-2, (key, sid, dt)
            if dsk and stem in und:
                pts, info = process_accumulated(tb, und[stem], fill_rng(sensor, seq, sid), target, crop)
            else:
                if dsk:
                    # the toolbox drops scans whose B-spline pose is undefined (first /
                    # last scans of the trajectory); the protocol keeps every keyframe,
                    # so such a scan is used without undistortion and listed
                    with open(d / f"deskew_fallback_{shard}_of_{nshards}.txt", "a") as fb:
                        fb.write(f"{sid}\n")
                    n_fallback += 1
                item = loader[sid]
                if item.get("pose") is not None:   # loader index space == cache index space
                    dt = float(np.abs(np.asarray(item["pose"])[:3, 3] - poses_pos[i]).max())
                    max_dt = max(max_dt, dt)
                    assert dt < 1e-2, (key, sid, dt)
                pts, info = process_scan(tb, item["points"], fill_rng(sensor, seq, sid), target, crop)
            assert pts.shape == (target, 4)
            if info.get("empty_after_crop"):
                with open(d / f"empty_after_crop_{shard}_of_{nshards}.txt", "a") as fe:
                    fe.write(f"{sid}\n")
                n_empty += 1
            tmp = out_p.with_name(f".{sid:06d}_{shard}.tmp")
            pts.astype("<f4").tofile(tmp)
            os.replace(tmp, out_p)
            nvox.append(info["n_vox"])
            done += 1
            if done % 500 == 0:
                print(f"  [{role}/{key} {shard}/{nshards}] {done}/{len(todo)} "
                      f"({(time.time() - t0) / done:.3f} s/scan)", flush=True)
    print(f"OK gen {role}/{key} shard {shard}/{nshards}: {done} new in "
          f"{time.time() - t0:.0f} s, deskew {dsk['kind'] if dsk else None} "
          f"(fallback without undistortion: {n_fallback}), empty after crop (origin "
          f"placeholder): {n_empty}, median centroids "
          f"{np.median(nvox) if nvox else -1:.0f}, max |pose - cache| {max_dt:.2e} m", flush=True)


def cmd_status() -> bool:
    ok = True
    for role, (seqs, _, _, _) in ROLES.items():
        n_all = n_have = 0
        empties = []
        for sensor, seq in seqs:
            d = seq_dir(role, sensor, seq)
            for f in sorted(d.glob("empty_after_crop_*.txt")):
                empties += [f"{sensor}/{seq}:{s}" for s in f.read_text().split()]
            if not (d / "index.npz").exists():
                print(f"  {role}/{sensor}/{seq}: not indexed")
                ok = False
                continue
            z = _index(role, sensor, seq)
            want = {f"{int(s):06d}.bin" for s in z["scan_ids"][z["generate"]]}
            have = {f for f in os.listdir(d / "LiDAR") if f.endswith(".bin")}
            n_all += len(want)
            n_have += len(want & have)
            if want - have:
                ok = False
                print(f"  {role}/{sensor}/{seq}: {len(want & have)}/{len(want)} scans")
        print(f"{role}: {n_have}/{n_all} scans"
              + (f", {len(empties)} empty-after-crop placeholders ({' '.join(empties)})" if empties else ""))
    print("SCANS_COMPLETE" if ok else "SCANS_INCOMPLETE")
    return ok


# ------------------------------------------------------------ official toolbox binary
VEL_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
                      ("ring", "<u2"), ("time", "<f4")])   # readBinFile(PointXYZIRT), 22 B


def write_velodyne_bin(path: Path, points: np.ndarray):
    a = np.zeros(len(points), VEL_DTYPE)
    p = np.asarray(points, np.float32)
    a["x"], a["y"], a["z"] = p[:, 0], p[:, 1], p[:, 2]
    if p.shape[1] > 3:
        a["intensity"] = p[:, 3]
    a.tofile(path)


def write_synthetic_trajectory(path: Path, stamps_ns):
    """TUM rows (ns x y z qx qy qz qw) every 50 ms from 1 s before the first to
    1 s after the last scan; identity rotation, x = 1 m per second, so every
    scan has a distinct position (distanceThreshold 0 keeps all) and the
    undistortion-off transform q^-1((q p + t) - t) is exact."""
    t0, t1 = min(stamps_ns) - 10 ** 9, max(stamps_ns) + 10 ** 9
    ts = np.arange(t0, t1 + 1, 50_000_000, dtype=np.int64)
    with open(path, "w") as f:
        for t in ts:
            f.write(f"{t} {(t - t0) / 1e9:.9f} 0 0 0 0 0 1\n")


def run_toolbox(work: Path, bin_dir: Path, traj: Path, undistort: bool,
                undistort_only: bool = False) -> Path:
    """Run the official toolbox binary with the official HeLiOS config, only
    Path / LiDAR type (1 = Velodyne reader) / distanceThreshold 0 (we choose
    keyframes) / undistortFlag overridden.  undistort_only additionally turns
    off cropping, downsampling and normalising, so the output is the
    undistorted, accumulated scan in metres (finished by process_accumulated,
    whose crop is the same filter applied after the same transform)."""
    import yaml
    cfg = official_toolbox_config()
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    cfg["Path"] = {"binPath": str(bin_dir) + "/", "trajPath": str(traj), "savePath": str(out) + "/"}
    cfg["Undistort"]["undistortFlag"] = bool(undistort)
    cfg["Save"]["LiDAR"] = 1
    cfg["Save"]["distanceThreshold"] = 0
    if undistort_only:
        cfg["Save"].update(downSampleFlag=False, normalizeFlag=False, cropFlag=False)
    (work / "config").mkdir(parents=True, exist_ok=True)
    (work / "config/config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (work / "build").mkdir(exist_ok=True)
    r = subprocess.run([str(TOOL_BIN)], cwd=work / "build", capture_output=True, text=True,
                       env={**os.environ, "TERM": "dumb"})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return out


def deskew_trajectory(role: str, sensor: str, seq: str, loader=None) -> dict | None:
    """Trajectory for the toolbox undistortion of a HeLiPR sequence.
    TRAIN / VAL: LiDAR_GT/Velodyne_gt.txt (official trajPath = LiDAR_GT/<sensor>_gt.txt).
    TEST: KISS-ICP odometry pose of every scan (results/pdv2_c3/odometry), written
    in the same TUM format (ns x y z qx qy qz qw) under _external/helios_official_pdv2/traj/."""
    if sensor != "helipr":
        return None
    if loader is None:
        from _pd_build_test_caches import _make_loader
        loader = _make_loader(sensor, seq)
    if role != "test_odom":
        return {"kind": "gt", "trajectory": str(loader.gt_file)}
    from scipy.spatial.transform import Rotation
    z = np.load(REPO / f"results/pdv2_c3/odometry/{sensor}_{seq}.npz")
    od = z["odom_poses"]
    stamps = np.asarray(loader.scan_timestamps, dtype=np.int64)
    assert len(od) == len(stamps), (seq, len(od), len(stamps))
    assert np.allclose(z["timestamps"], stamps / 1e9, atol=1e-3)
    q = Rotation.from_matrix(od[:, :3, :3]).as_quat()
    out = WORK / "traj" / f"{sensor}_{seq}_kissicp.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for t, T, qq in zip(stamps, od, q):
            f.write(f"{t} {T[0, 3]:.9f} {T[1, 3]:.9f} {T[2, 3]:.9f} "
                    f"{qq[0]:.12f} {qq[1]:.12f} {qq[2]:.12f} {qq[3]:.12f}\n")
    return {"kind": "kiss-icp odometry", "trajectory": str(out),
            "source": f"results/pdv2_c3/odometry/{sensor}_{seq}.npz"}


def toolbox_undistort(work: Path, files, traj: str) -> dict:
    """Official toolbox undistortion of raw HeLiPR files -> {stem: (n, 4) float32}."""
    (work / "bin").mkdir(parents=True, exist_ok=True)
    for f in files:
        os.symlink(f, work / "bin" / Path(f).name)
    out = run_toolbox(work, work / "bin", Path(traj), undistort=True, undistort_only=True)
    res = {}
    for f in files:
        o = out / f"{Path(f).stem}.bin"
        if o.exists():
            res[Path(f).stem] = np.fromfile(o, dtype="<f4").reshape(-1, 4)
    return res


def _rows(a):
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


SELFTEST_SEQS = [("train", "kitti/05"), ("train", "helipr/Bridge01"),
                 ("val", "kitti360/0004"), ("val", "mulran/Sejong01"),
                 ("test_odom", "nclt/2012-01-08"), ("test_odom", "mcd/kth_day_06")]


def cmd_selftest(n_scans: int, out_json: Path | None):
    """Adapter vs the official toolbox binary run on the same points:
    loader points written in the toolbox's Velodyne format (time 0) with a
    synthetic trajectory, undistortion off; HeLiPR additionally on the raw
    LiDAR/Velodyne/*.bin files themselves.  Checked: the x, y, z of the
    voxel-centroid rows (all but the random fill) are byte-identical (the
    intensity column depends on the fill and is unused); the fill rows are points of
    the cropped scan; 8192 rows; plus our transcribed size search == the
    official downsample_to_target_size in-process."""
    from _pd_build_test_caches import _make_loader
    tb = Toolbox()
    cfgs = official_toolbox_config()["Save"]
    target, crop = int(cfgs["downsamplePointSize"]), float(cfgs["cropSize"])
    rep = {"toolbox_commit": git_head(TOOLBOX), "helios_commit": git_head(HELIOS),
           "toolbox_binary_sha256": sha256(TOOL_BIN), "lib_sha256": sha256(LIB), "cases": []}
    ok_all = True
    with tempfile.TemporaryDirectory(dir=str(WORK)) as td:
        td = Path(td)
        for role, key in SELFTEST_SEQS:
            sensor, seq = key.split("/")
            sids = np.load(cache_path(role, sensor, seq))["scan_ids"]
            pick = [int(sids[int(i)]) for i in
                    np.linspace(len(sids) * 0.1, len(sids) * 0.9, n_scans).round().astype(int)]
            loader = _make_loader(sensor, seq)
            variants = [("loader_points", None)]
            if sensor == "helipr":
                variants.append(("raw_helipr_bin", loader))
            for vname, raw in variants:
                w = td / f"{sensor}_{seq}_{vname}"
                (w / "bin").mkdir(parents=True)
                stamps, pts_by = {}, {}
                for k, sid in enumerate(pick):
                    pts = loader[sid]["points"]
                    pts_by[sid] = pts
                    if raw is not None:
                        f = Path(raw.scan_files[sid])
                        stamps[sid] = int(f.stem)
                        os.symlink(f, w / "bin" / f.name)
                    else:
                        stamps[sid] = 1_600_000_000_000_000_000 + k * 100_000_000
                        write_velodyne_bin(w / "bin" / f"{stamps[sid]}.bin", pts)
                write_synthetic_trajectory(w / "traj.txt", list(stamps.values()))
                out = run_toolbox(w, w / "bin", w / "traj.txt", undistort=False)
                for sid in pick:
                    off = np.fromfile(out / f"{stamps[sid]}.bin", dtype="<f4").reshape(-1, 4)
                    t0 = time.perf_counter()
                    ours, info = process_scan(tb, pts_by[sid], fill_rng(sensor, seq, sid),
                                              target, crop)
                    dt = time.perf_counter() - t0
                    nv = info["n_vox"]
                    acc = toolbox_accumulate(pts_by[sid], crop)
                    acc_n = tb.normalize(acc, crop)[:, :3]
                    # xyz bytes: the intensity column is scaled by the minimum over all
                    # 8192 rows incl. the random fill (normalizePointCloud) and is never
                    # read by HeLiOS (PNVPointCloudLoader keeps [:, :3])
                    cent_equal = bool(np.array_equal(_rows(np.ascontiguousarray(off[:nv, :3])),
                                                     _rows(np.ascontiguousarray(ours[:nv, :3]))))
                    int_equal = bool(np.array_equal(off[:nv, 3], ours[:nv, 3]))
                    fill_ok = bool(np.isin(_rows(np.ascontiguousarray(off[nv:, :3])),
                                           _rows(np.ascontiguousarray(acc_n))).all())
                    off_cent = tb.official_downsample(acc, target)[:nv]
                    inproc = bool(np.array_equal(_rows(off_cent), _rows(tb.target_centroids(acc, target)[0])))
                    ok = cent_equal and fill_ok and inproc and off.shape == (target, 4)
                    ok_all &= ok
                    case = {"role": role, "seq": key, "scan_id": sid, "variant": vname,
                            **info, "official_rows": int(len(off)),
                            "centroid_rows_xyz_byte_equal": cent_equal,
                            "centroid_rows_intensity_equal": int_equal,
                            "fill_rows_in_cropped_scan": fill_ok,
                            "search_loop_equals_official_function": inproc,
                            "adapter_s": round(dt, 3)}
                    rep["cases"].append(case)
                    print(json.dumps(case), flush=True)
        ok_all &= _selftest_deskew(tb, td, n_scans, rep, target, crop)
    rep["all_ok"] = bool(ok_all)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rep, indent=1))
    print("PREP_SELFTEST_OK" if ok_all else "PREP_SELFTEST_FAILED", flush=True)
    if not ok_all:
        sys.exit(1)


def _occ(p):
    return set(map(tuple, np.floor(p[:, :3] / 0.01).astype(np.int64)))


def _jac(a, b):
    return len(a & b) / max(1, len(a | b))


def _selftest_deskew(tb, td: Path, n_scans: int, rep: dict, target: int, crop: float) -> bool:
    """HeLiPR with undistortion: (1) Bridge01 (TRAIN, GT trajectory): full
    official toolbox (undistort + crop + downsample + normalise) vs our path
    (toolbox undistortion only -> process_accumulated): centroid rows
    byte-identical, fill rows from the cropped undistorted scan.  (2)
    Roundabout01 (VAL): the deskew_trajectory files (GT for val; the KISS-ICP
    odometry file that TEST uses, here built for a VAL sequence) -> 1 m
    occupancy agreement of the two undistortions."""
    from _pd_build_test_caches import _make_loader
    ok = True
    loader = _make_loader("helipr", "Bridge01")
    sids = np.load(cache_path("train", "helipr", "Bridge01"))["scan_ids"]
    pick = [int(sids[int(i)]) for i in np.linspace(len(sids) * 0.2, len(sids) * 0.8, n_scans).round().astype(int)]
    files = [loader.scan_files[s] for s in pick]
    traj = deskew_trajectory("train", "helipr", "Bridge01", loader)["trajectory"]
    w = td / "deskew_full"
    (w / "bin").mkdir(parents=True)
    for f in files:
        os.symlink(f, w / "bin" / Path(f).name)
    full = run_toolbox(w, w / "bin", Path(traj), undistort=True)
    und = toolbox_undistort(td / "deskew_only", files, traj)
    for sid, f in zip(pick, files):
        stem = Path(f).stem
        off = np.fromfile(full / f"{stem}.bin", dtype="<f4").reshape(-1, 4)
        ours, info = process_accumulated(tb, und[stem], fill_rng("helipr", "Bridge01", sid), target, crop)
        nv = info["n_vox"]
        acc_n = tb.normalize(toolbox_crop(und[stem], crop), crop)[:, :3]
        cent = bool(np.array_equal(_rows(np.ascontiguousarray(off[:nv, :3])),
                                   _rows(np.ascontiguousarray(ours[:nv, :3]))))
        fill = bool(np.isin(_rows(np.ascontiguousarray(off[nv:, :3])), _rows(np.ascontiguousarray(acc_n))).all())
        nodsk, _ = process_scan(tb, loader[sid]["points"], fill_rng("helipr", "Bridge01", sid), target, crop)
        case = {"seq": "helipr/Bridge01", "scan_id": sid, "variant": "undistort_gt_vs_full_official",
                **info, "centroid_rows_xyz_byte_equal": cent, "fill_rows_in_cropped_scan": fill,
                "jaccard_1m_vs_no_deskew": round(_jac(_occ(ours), _occ(nodsk)), 4)}
        ok &= cent and fill and off.shape == (target, 4)
        rep["cases"].append(case)
        print(json.dumps(case), flush=True)
    loader = _make_loader("helipr", "Roundabout01")
    sids = np.load(cache_path("val", "helipr", "Roundabout01"))["scan_ids"]
    pick = [int(sids[int(i)]) for i in np.linspace(len(sids) * 0.2, len(sids) * 0.8, n_scans).round().astype(int)]
    files = [loader.scan_files[s] for s in pick]
    gt = deskew_trajectory("val", "helipr", "Roundabout01", loader)["trajectory"]
    od = deskew_trajectory("test_odom", "helipr", "Roundabout01", loader)
    a = toolbox_undistort(td / "ra_gt", files, gt)
    b = toolbox_undistort(td / "ra_odom", files, od["trajectory"])
    Path(od["trajectory"]).unlink()            # VAL sequence: not a production file
    for sid, f in zip(pick, files):
        stem = Path(f).stem
        rng = lambda: fill_rng("helipr", "Roundabout01", sid)  # noqa: E731
        pa, _ = process_accumulated(tb, a[stem], rng(), target, crop)
        pb, _ = process_accumulated(tb, b[stem], rng(), target, crop)
        pn, _ = process_scan(tb, loader[sid]["points"], rng(), target, crop)
        case = {"seq": "helipr/Roundabout01", "scan_id": sid, "variant": "undistort_gt_vs_kissicp",
                "jaccard_1m_gt_vs_odometry": round(_jac(_occ(pa), _occ(pb)), 4),
                "jaccard_1m_gt_vs_no_deskew": round(_jac(_occ(pa), _occ(pn)), 4),
                "max_abs_diff_m_gt_vs_odometry": float(np.abs(a[stem][:, :3] - b[stem][:, :3]).max())}
        rep["cases"].append(case)
        print(json.dumps(case), flush=True)
    return ok


def cmd_deskew_effect(n_scans: int, out_json: Path | None):
    """Official toolbox on raw HeLiPR VLP-16 files with the GT trajectory
    (LiDAR_GT/Velodyne_gt.txt), undistortFlag True vs False.  Reported: the
    Jaccard index of the 1 m model-input voxels (floor(x / 0.01) of the
    normalised points = CartesianQuantizer(0.01)) and, as the noise floor, the
    same index between the undistortion-off output and our adapter (they
    differ only by the random fill).  TRAIN / VAL sequences only."""
    from _pd_build_test_caches import DATA_ROOT, _make_loader
    tb = Toolbox()
    rep = {"what": "official toolbox, undistortFlag True vs False, GT trajectory", "cases": []}

    def occ(p):
        return set(map(tuple, np.floor(p[:, :3] / 0.01).astype(np.int64)))

    def jac(a, b):
        return len(a & b) / max(1, len(a | b))

    with tempfile.TemporaryDirectory(dir=str(WORK)) as td:
        td = Path(td)
        for role, sensor, seq in (("train", "helipr", "Bridge01"), ("val", "helipr", "Roundabout01")):
            loader = _make_loader(sensor, seq)
            sids = np.load(cache_path(role, sensor, seq))["scan_ids"]
            pick = [int(sids[int(i)]) for i in
                    np.linspace(len(sids) * 0.1, len(sids) * 0.9, n_scans).round().astype(int)]
            w = td / seq
            (w / "bin").mkdir(parents=True)
            for sid in pick:
                f = Path(loader.scan_files[sid])
                os.symlink(f, w / "bin" / f.name)
            gt = DATA_ROOT / "helipr" / seq / "LiDAR_GT/Velodyne_gt.txt"
            outs = {}
            for und in (True, False):
                ww = w / f"und_{int(und)}"
                ww.mkdir()
                outs[und] = run_toolbox(ww, w / "bin", gt, undistort=und)
            for sid in pick:
                stem = Path(loader.scan_files[sid]).stem
                a = np.fromfile(outs[True] / f"{stem}.bin", dtype="<f4").reshape(-1, 4)
                b = np.fromfile(outs[False] / f"{stem}.bin", dtype="<f4").reshape(-1, 4)
                ours, _ = process_scan(tb, loader[sid]["points"], fill_rng(sensor, seq, sid))
                v = np.asarray(loader[sid]["points"])
                speed = None
                if sid + 1 < len(loader):
                    speed = float(np.linalg.norm(loader[sid + 1]["pose"][:3, 3] - loader[sid]["pose"][:3, 3])
                                  / max(1e-6, loader[sid + 1]["timestamp"] - loader[sid]["timestamp"]))
                case = {"seq": f"{sensor}/{seq}", "scan_id": sid, "n_raw": int(len(v)),
                        "speed_mps": None if speed is None else round(speed, 2),
                        "jaccard_1m_undistort_vs_not": round(jac(occ(a), occ(b)), 4),
                        "jaccard_1m_noise_floor_fill_only": round(jac(occ(b), occ(ours)), 4)}
                rep["cases"].append(case)
                print(json.dumps(case), flush=True)
    j = [c["jaccard_1m_undistort_vs_not"] for c in rep["cases"]]
    n = [c["jaccard_1m_noise_floor_fill_only"] for c in rep["cases"]]
    rep["summary"] = {"jaccard_undistort_mean": float(np.mean(j)), "jaccard_undistort_min": float(np.min(j)),
                      "noise_floor_mean": float(np.mean(n))}
    print(json.dumps(rep["summary"]), flush=True)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index")
    p.add_argument("--role", choices=tuple(ROLES), nargs="+", default=list(ROLES))
    p.add_argument("--train-spacing", type=float, default=TOOLBOX_SPACING_M,
                   help="official toolbox distanceThreshold on the TRAIN keyframes (0 = all)")
    p.add_argument("--deskew", choices=("helipr", "none"), default="helipr",
                   help="official toolbox undistortion for HeLiPR (the only per-point times)")
    p = sub.add_parser("jobs")
    p.add_argument("--chunk", type=int, default=2000)
    p = sub.add_parser("gen")
    p.add_argument("--role", choices=tuple(ROLES), required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    sub.add_parser("status")
    p = sub.add_parser("selftest")
    p.add_argument("--n-scans", type=int, default=3)
    p.add_argument("--out-json", type=Path, default=LOG_DIR / "helios_selftest_prep.json")
    p = sub.add_parser("deskew-effect")
    p.add_argument("--n-scans", type=int, default=4)
    p.add_argument("--out-json", type=Path, default=LOG_DIR / "helios_deskew_effect.json")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.cmd == "index":
        cmd_index(a.role, a.train_spacing, a.deskew)
    elif a.cmd == "jobs":
        cmd_jobs(a.chunk)
    elif a.cmd == "gen":
        cmd_gen(a.role, a.seq, a.shard, a.nshards)
    elif a.cmd == "status":
        sys.exit(0 if cmd_status() else 1)
    elif a.cmd == "selftest":
        cmd_selftest(a.n_scans, a.out_json)
    elif a.cmd == "deskew-effect":
        cmd_deskew_effect(a.n_scans, a.out_json)


if __name__ == "__main__":
    main()
