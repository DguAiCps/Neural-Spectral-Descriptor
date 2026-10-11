#!/usr/bin/env python3
"""DiSCO official pipeline on split v2 -- part 1: shared pieces and the
multi-layer polar occupancy inputs (2026-10-11).

Official code: _external/DiSCO = MaverickPeter/DiSCO-pytorch (README: "Code
for DiSCO: Differentiable Scan Context with Orientation", RA-L 2021), commit
OFFICIAL_COMMIT, unchanged.  The only compiled piece, the polar-occupancy CUDA
extension `gputransform` (multi-layer-polar-cython/cython), is built from the
unchanged sources for sm_120 into _external/disco_env/pylib by
_external/disco_env/build_gputransform.sh (only -arch=sm_30 -> sm_120 in a
copy of setup.py).

Official input (generating_queries/nclt/generate_training_tuples_baseline_with_pose.py,
the script that wrote every training / evaluation .npy, latest filter commit
0e598a8):
  load_lidar_file_nclt: keep |x| < 70, |y| < 70, -20 < z < -2 (NCLT raw z
      points down, i.e. 2..20 m ABOVE the sensor), not(|x| < 1 and |y| < 1);
      hits = [x/70, y/70, z/20], then hits[:, 2] = -hits[:, 2]
  load_pc_file: gputransform.GPUTransformer(hits.T.flatten() as float32, N,
      max_length 1, max_height 1, num_ring 40, num_sector 120, num_height 20,
      enough_large 1) -> point_t.reshape(-1, 3)[..., 2].reshape(20, 40, 120):
      binary occupancy, ring = 1.75 m (70 m / 40), sector = 3 deg,
      height layer = 2 m of the [-20, 20] m sensor-frame z range (z/20 in
      steps of 0.1; NCLT fills layers 11..19).
Our adapter `official_hits` applies exactly these lines to the points of our
loaders (scripts/_pd_build_test_caches._make_loader, sensor frame, z up; NCLT
already z-flipped by the loader, so z_raw = -z_up as in the official reader).
NCLT points are mapped back to the official float64 coordinates (our loader
converts the uint16 fields in float32; the 5 mm grid makes the inversion
exact, checked against the official reader on the raw files in `selftest`).
`official_polar` then calls the official CUDA extension exactly as
load_pc_file does.

Height band (user decision 2026-10-11, logs/baseline_official_20261011/disco.md
section 9): the official code has one setting (NCLT, 2..20 m above the
sensor; its generic ROS path load_pc_infer uses the same band).  It is kept
for HeLiPR, MulRan, NCLT and MCD (Z_SHIFT 0).  For the KITTI family
(HDL-64E, +2 deg upper FOV) that band is nearly empty, so the cloud is first
translated up by Z_SHIFT = +2 m: the official filter then keeps 0..18 m
above the sensor (same 18 m width, same 2 m layers relative to the shifted
origin).  This shift is OUR rule, not an official setting.

Paper over code (user rule 2026-10-11: where the RA-L paper and the code
differ, the paper value is used and the overridden code line is disclosed):
  RANGE_M = 80   paper Sec. IV-A "within 80m far"; code generator:109-110
                 |x|, |y| < 70 and x/70, y/70 (1.75 m rings) -> 80, 2 m rings
  1 m layers     paper Sec. IV-B "discretizes the height 20m with 20 layers";
                 code generator:110 z/20 into the kernel's 20 layers over
                 [-20, 20] m = 2 m layers (kernel.cu:51, 70) -> z_up/10 - 1, so
                 the unchanged kernel puts layer k = floor(z_up / 1 m), 0..19
                 (z_up after Z_SHIFT; kept band 2 < z_up < 20 as in the code)
Both are applied in `official_filter` (our adapter); the official files are
unchanged.  `official_filter(..., mode="code")` reproduces the code exactly
and is what the selftest compares with the official function.

Layout: one packed store per role and sequence
  _external/disco_official_pdv2/polar/{role}/{sensor}_{seq}/
      index.npz   scan_ids, poses (role pose key), heading (train only)
      bits.npy    (N, 12000) uint8 = np.packbits of the (20, 40, 120) occupancy
      meta.json   cache sha256, rule, extension sha256, empty scans, complete
roles: train (v2 TRAIN, GT keyframe caches), val (v2 VAL, GT keyframe caches),
test_odom (v2 TEST, KISS-ICP keyframe caches; poses = poses_gt, scoring only).
The official training reads one float32 .npy per scan (load_pc_file); the
training driver decodes these rows to the identical float32 array instead
(the occupancy is exactly 0/1), 12 kB instead of 384 kB per scan.

Subcommands (inside ring_baseline):
  index [--role R ...]          index.npz + meta for every sequence
  jobs                          print "role sensor/seq" generation jobs
  gen --role R --seq S          polar images via the official CUDA extension
  status                        completeness of every store
  selftest [--n-scans 2]        CPU only (see cmd_selftest)
  gpu-check [--n-scans 3]       GPU: official extension == CPU transcription
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
sys.dont_write_bytecode = True      # keep the official clone free of __pycache__

from _pdv2_pilot import TEST, TRAIN, VAL  # noqa: E402  (split-v2 lists)

OFFICIAL = REPO / "_external/DiSCO"
OFFICIAL_COMMIT = "7087fd2f7b752f4096cabf1db71e8cbe6a13f9fe"
OFFICIAL_REPO = "https://github.com/MaverickPeter/DiSCO-pytorch"
PYLIB = REPO / "_external/disco_env/pylib"      # gputransform built for sm_120
GENERATOR = OFFICIAL / "generating_queries/nclt/generate_training_tuples_baseline_with_pose.py"
WORK = REPO / "_external/disco_official_pdv2"
POLAR = WORK / "polar"
V2 = REPO / "data/preprocessed_v2"
ODOM = REPO / "data/preprocessed_v2_odom"
RESULTS = REPO / "results/pdv2_disco_official"
ROLES = {   # role -> (sequence list, cache dir, cache-name middle, pose key)
    "train": (TRAIN, V2, "", "poses"),
    "val": (VAL, V2, "val_", "poses"),
    "test_odom": (TEST, ODOM, "test_", "poses_gt"),
}
SHAPE = (20, 40, 120)                 # official cfg num_height, num_ring, num_sector
NBYTES = int(np.prod(SHAPE)) // 8     # 12000

# Literal official rule for every sensor (see module docstring).
Z_SHIFT = {"kitti": 2.0, "kitti360": 2.0, "helipr": 0.0, "mulran": 0.0,
           "nclt": 0.0, "mcd": 0.0}
RANGE_M = 80.0     # paper; code 70 (generator:109-110)
CODE_RANGE_M = 70.0
RULE = ("generate_training_tuples_baseline_with_pose.load_lidar_file_nclt with the "
        "paper values: |x|<80, |y|<80 (code 70), 2 < z_up < 20 (official "
        "-20 < z_raw < -2), not(|x|<1 and |y|<1); [x/80, y/80, z_up/10 - 1] "
        "(code [x/70, y/70, z_up/20]) -> 2 m rings, 1 m layers over 0-20 m (code "
        "1.75 m rings, 2 m layers over -20..20 m); load_pc_file: gputransform "
        "20x40x120 occupancy; cloud translated by Z_SHIFT before the filter: +2 m "
        "for kitti/kitti360 (0 < z_up < 18, our rule), 0 otherwise")

# Heading (radians) of the LiDAR about the world vertical axis, only for the
# TRAIN sequences: train_DiSCO.py rotates each positive by
# (heading_query - heading_positive) (rotation_on_SCI) before the yaw loss.
# (world up vector, LiDAR forward axis in the pose's body frame, world
# reference horizontal axis), all in the cache pose convention:
#   kitti : KITTI odometry poses.txt = camera-0 poses (x right, y down,
#           z forward); the velodyne x axis is the camera z axis
#   helipr: Velodyne_gt.txt = LiDAR poses, z up
# Positive = counter-clockwise seen from above; the official NCLT heading is
# the NED yaw (clockwise) with NCLT x/y read in the z-down raw frame, so in
# both cases a yaw increase moves scene azimuths the same way (checked
# empirically in selftest `heading`).
HEADING_AXES = {
    "kitti": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
    "helipr": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
}


def heading(sensor: str, poses: np.ndarray) -> np.ndarray:
    up, fwd, e1 = (np.asarray(v, dtype=np.float64) for v in HEADING_AXES[sensor])
    f = poses[:, :3, :3] @ fwd
    e2 = np.cross(up, e1)
    return np.arctan2(f @ e2, f @ e1)


def seq_name(sensor: str, seq: str) -> str:
    return f"{sensor}_{seq}"


def cache_path(role: str, sensor: str, seq: str) -> Path:
    _, cdir, mid, _ = ROLES[role]
    hits = sorted(glob.glob(str(cdir / f"cache_????????_{sensor}_{mid}{seq}.npz")))
    assert len(hits) == 1, (role, sensor, seq, hits)
    return Path(hits[0])


def store_dir(role: str, sensor: str, seq: str) -> Path:
    return POLAR / role / seq_name(sensor, seq)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------ official code
def official_env(cpu_only: bool = False):
    """Make the official modules importable (no official file is changed).

      * sys.path: PYLIB (gputransform for sm_120) and the clone in front;
      * torch.rfft / torch.ifft: removed after torch 1.7 (README requires
        torch <= 1.6); re-added with the torch-1.4 semantics on torch.fft
        (rfft: full spectrum, 1/sqrt(n) when normalized; ifft: 1/n), see
        `_rfft_compat` / `_ifft_compat` and selftest `fft`;
      * cpu_only: torch.cuda.is_available() -> False BEFORE the official
        modules choose their device (train_DiSCO.py / evaluate.py set
        CUDA_VISIBLE_DEVICES=0 at import time).
    """
    import torch
    if cpu_only:
        torch.cuda.is_available = lambda: False
    for p in (str(PYLIB), str(OFFICIAL)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    if not hasattr(torch, "rfft"):
        torch.rfft = _rfft_compat
    if not hasattr(torch, "ifft"):
        torch.ifft = _ifft_compat
    return torch


def _rfft_compat(input, signal_ndim, normalized=False, onesided=True):
    """torch<=1.6 torch.rfft (real-to-complex, last signal_ndim dims, output
    (..., 2)); normalized=True -> 1/sqrt(n)."""
    import torch
    dims = tuple(range(-signal_ndim, 0))
    norm = "ortho" if normalized else "backward"
    out = (torch.fft.rfftn(input, dim=dims, norm=norm) if onesided
           else torch.fft.fftn(input, dim=dims, norm=norm))
    return torch.view_as_real(out)


def _ifft_compat(input, signal_ndim, normalized=False):
    """torch<=1.6 torch.ifft (complex (..., 2) -> complex (..., 2), last
    signal_ndim complex dims); normalized=False -> 1/n."""
    import torch
    dims = tuple(range(-signal_ndim, 0))
    z = torch.view_as_complex(input.contiguous())
    out = torch.fft.ifftn(z, dim=dims, norm="ortho" if normalized else "backward")
    return torch.view_as_real(out)


def import_official(cpu_only: bool = False, fake_gputransform=None):
    """Import config, loading_pointclouds, models.DiSCO, loss.loss_function and
    evaluate from the clone.  fake_gputransform (CPU checks only) replaces the
    CUDA extension module."""
    official_env(cpu_only)
    if fake_gputransform is not None:
        sys.modules["gputransform"] = fake_gputransform
    import config as cfg
    import loading_pointclouds as lp
    import models.DiSCO as SC
    import loss.loss_function as LF
    import evaluate as EV
    for m in (cfg, lp, SC, LF, EV):
        assert Path(m.__file__).resolve().is_relative_to(OFFICIAL.resolve()), m.__file__
    if fake_gputransform is None:
        import gputransform
        assert Path(gputransform.__file__).resolve().parent == PYLIB.resolve(), \
            gputransform.__file__
    return cfg, lp, SC, LF, EV


def generator_functions(gputransform_mod):
    """convert / load_lidar_file_nclt / load_pc_file from the official
    generator, executed from its unchanged source.  The file runs a
    hard-coded NCLT walk at import time, so only these FunctionDefs are
    compiled (with the generator's own config module)."""
    spec = importlib.util.spec_from_file_location(
        "disco_gen_config", GENERATOR.parent / "config.py")
    gcfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gcfg)
    tree = ast.parse(GENERATOR.read_text(), str(GENERATOR))
    want = {"convert", "load_lidar_file_nclt", "load_pc_file"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert {n.name for n in body} == want
    ns = {"np": np, "struct": struct, "os": os, "cfg": gcfg,
          "gputransform": gputransform_mod}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(GENERATOR), "exec"), ns)
    return ns, gcfg


# ------------------------------------------------------------ adapter
def _nclt_official_coords(p: np.ndarray) -> np.ndarray:
    """Our NCLT loader: v32 = float32(u) * 0.005 - 100 (float32), z flipped.
    Official convert(): u * 0.005 + -100.0 in float64.  u is recovered
    exactly (float32 error << 5 mm grid)."""
    raw = p.copy()
    raw[:, 2] = -raw[:, 2]
    u = np.rint((raw + 100.0) / 0.005)
    assert np.all(np.abs(raw - (u * 0.005 - 100.0)) < 1e-3), "not on the NCLT 5 mm grid"
    off = u * 0.005 + -100.0
    off[:, 2] = -off[:, 2]
    return off


def official_points(sensor: str, points: np.ndarray) -> np.ndarray:
    """Our loader points -> float64 (x, y, z_up) fed to the official filter:
    NCLT grid inversion, then the vertical Z_SHIFT (KITTI family only)."""
    p = np.asarray(points)[:, :3].astype(np.float64)
    if sensor == "nclt":
        p = _nclt_official_coords(p)
    if Z_SHIFT[sensor]:
        p[:, 2] = p[:, 2] + Z_SHIFT[sensor]
    return p


def official_hits(sensor: str, points: np.ndarray) -> np.ndarray:
    """Our loader points (x, y, z_up) -> the (N, 3) float64 `hits` of the
    official load_lidar_file_nclt (same comparisons, same arithmetic order)."""
    return official_filter(official_points(sensor, points))


def official_filter(p: np.ndarray, mode: str = "paper") -> np.ndarray:
    """load_lidar_file_nclt lines 105-114 on float64 (x, y, z_up).
    mode "code": the official lines exactly (70 m, z/20 -> 2 m layers);
    mode "paper": RANGE_M = 80 m and z_up/10 - 1 (1 m layers over 0-20 m)."""
    assert mode in ("paper", "code"), mode
    r = RANGE_M if mode == "paper" else CODE_RANGE_M
    x, y, z = p[:, 0], p[:, 1], -p[:, 2]          # official raw frame: z down
    keep = (np.abs(x) < r) & (z > -20.) & (z < -2.) & (np.abs(y) < r) & \
        ~((np.abs(x) < 1.) & (np.abs(y) < 1.))
    hits = np.stack([x[keep] / r, y[keep] / r, z[keep] / 20.], axis=1)
    hits[:, 2] = -hits[:, 2]
    if mode == "paper":
        hits[:, 2] = -z[keep] / 10. - 1.           # z_up in (2, 20) -> (-0.8, 1)
    return hits


def official_polar(hits: np.ndarray, gputransform_mod, cfg) -> np.ndarray:
    """generator load_pc_file lines 138-147 on `hits`.  An empty `hits` (the
    official reader would raise IndexError on it) gives the empty image; the
    extension is not called with zero points."""
    if len(hits) == 0:
        return np.zeros(SHAPE, dtype=np.float32)
    size = hits.shape[0]
    pc = hits.transpose().flatten().astype(np.float32)
    transer = gputransform_mod.GPUTransformer(pc, size, cfg.max_length, cfg.max_height,
                                              cfg.num_ring, cfg.num_sector,
                                              cfg.num_height, 1)
    transer.transform()
    point_t = transer.retreive()
    point_t = point_t.reshape(-1, 3)
    point_t = point_t[..., 2]
    return point_t.reshape(cfg.num_height, cfg.num_ring, cfg.num_sector)


def pack(img: np.ndarray) -> np.ndarray:
    assert img.shape == SHAPE and img.dtype == np.float32
    occ = img == 1.0
    assert np.all(occ | (img == 0.0)), "official occupancy must be 0/1"
    return np.packbits(occ.ravel())


def unpack(row: np.ndarray) -> np.ndarray:
    """== np.load of the official float32 .npy for that scan."""
    return np.unpackbits(row).reshape(SHAPE).astype(np.float32)


# ------------------------------------------------------------ CPU transcription
def polar_numpy(pc_flat: np.ndarray, size: int, enough_large: int = 1) -> np.ndarray:
    """CHECK ONLY: float32 numpy transcription of kernel.cu point2gridmap +
    manager.cu retreive (max_length = max_height = 1, 40 x 120 x 20).
    Returns the extension's retreive() buffer (grid_size * 3 float32).
    atan/sqrt are numpy's float32 routines, so cells can differ from the GPU
    for points within ~1e-5 deg of a sector edge (gpu-check measures it)."""
    assert enough_large == 1
    f32 = np.float32
    nr, ns, nh = 40, 120, 20
    p = np.asarray(pc_flat, dtype=f32).reshape(3, size)
    x, y, z = p[0].copy(), p[1].copy(), p[2].copy()
    x[x == 0] = f32(0.0001)
    y[y == 0] = f32(0.0001)
    z[z == 0] = f32(0.0001)
    deg = 180.0 / np.pi                       # (180/M_PI), double
    th = np.empty(size, dtype=f32)
    q = [(x >= 0) & (y >= 0), (x < 0) & (y >= 0), (x < 0) & (y < 0), (x >= 0) & (y < 0)]
    th[q[0]] = (deg * np.arctan(y[q[0]] / x[q[0]]).astype(np.float64)).astype(f32)
    th[q[1]] = (180 - deg * np.arctan(y[q[1]] / (-x[q[1]])).astype(np.float64)).astype(f32)
    th[q[2]] = (180 + deg * np.arctan(y[q[2]] / x[q[2]]).astype(np.float64)).astype(f32)
    th[q[3]] = (360 - deg * np.arctan((-y[q[3]]) / x[q[3]]).astype(np.float64)).astype(f32)
    gap_ring = f32(1) / f32(nr)
    gap_sector = f32(360.0 / float(f32(ns)))
    gap_height = f32(2.0 * float(f32(1)) / float(f32(nh)))
    far = np.sqrt(x * x + y * y)
    ir = np.floor(far / gap_ring).astype(np.int64)
    isec = np.floor(th / gap_sector).astype(np.int64)
    ih = np.floor((z + f32(1)) / gap_height).astype(np.int64)
    ir[ir >= nr] = nr - 1
    flat = isec + ir * ns + ih * ns * nr
    ok = (flat >= 0) & (flat < ns * nr * nh)  # outside = out-of-bounds write in CUDA
    out = np.zeros(ns * nr * nh * 3, dtype=f32)
    out[3 * flat[ok] + 2] = 1
    return out


class FakeGPUTransformer:
    """CHECK ONLY: drop-in for gputransform.GPUTransformer backed by polar_numpy."""

    def __init__(self, point, size, max_length, max_height, num_ring, num_sector,
                 num_height, enough_large):
        assert (max_length, max_height, num_ring, num_sector, num_height) == (1, 1, 40, 120, 20)
        self.point, self.size, self.enough_large = point, size, enough_large

    def transform(self):
        pass

    def retreive(self):
        return polar_numpy(self.point, self.size, self.enough_large)


def fake_gputransform_module():
    m = types.ModuleType("gputransform")
    m.GPUTransformer = FakeGPUTransformer
    m.__file__ = "<cpu transcription>"
    return m


# ------------------------------------------------------------ index / gen
def cmd_index(role: str):
    seqs, _, _, pose_key = ROLES[role]
    for sensor, seq in seqs:
        cp = cache_path(role, sensor, seq)
        with np.load(cp) as z:
            sids = z["scan_ids"].astype(np.int64)
            poses = z[pose_key].astype(np.float64)
        assert len(sids) == len(poses) and np.all(np.diff(sids) > 0), (sensor, seq)
        d = store_dir(role, sensor, seq)
        if (d / "meta.json").exists():
            prev = json.loads((d / "meta.json").read_text())
            assert prev["cache_sha256"] == sha256(cp), f"{d}: cache changed since indexing"
            if prev["rule"] == RULE and prev["z_shift"] == Z_SHIFT[sensor]:
                print(f"[skip] index {role}/{sensor}/{seq} exists", flush=True)
                continue
            assert not prev.get("complete") and not (d / "bits.npy").exists(), \
                f"{d}: images were made with another band rule; delete the store first"
            print(f"[reindex] {role}/{sensor}/{seq}: band rule changed", flush=True)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {"scan_ids": sids, "poses": poses}
        if role == "train":
            arrays["heading"] = heading(sensor, poses)
        np.savez(d / "index.npz", **arrays)
        meta = {"role": role, "sensor": sensor, "sequence": seq, "n_frames": int(len(sids)),
                "cache": str(cp.relative_to(REPO)), "cache_sha256": sha256(cp),
                "poses_from": pose_key, "rule": RULE, "z_shift": Z_SHIFT[sensor],
                "official": f"{OFFICIAL_REPO}@{OFFICIAL_COMMIT}",
                "loader": "scripts/_pd_build_test_caches._make_loader",
                "complete": False}
        if role == "train":
            meta["heading_axes"] = HEADING_AXES[sensor]
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"OK index {role}/{sensor}/{seq}: {len(sids)} keyframes", flush=True)


def cmd_jobs():
    for role, (seqs, _, _, _) in ROLES.items():
        for sensor, seq in seqs:
            print(role, f"{sensor}/{seq}")


def cmd_gen(role: str, key: str, flush_every: int = 500):
    from _pd_build_test_caches import _make_loader
    sensor, seq = key.split("/")
    assert (sensor, seq) in ROLES[role][0], (role, key)
    d = store_dir(role, sensor, seq)
    meta = json.loads((d / "meta.json").read_text())
    assert meta["rule"] == RULE and meta["z_shift"] == Z_SHIFT[sensor], \
        f"{d}: indexed with another band rule; run `index` again"
    if meta.get("complete"):
        print(f"[skip] {role}/{key} complete", flush=True)
        return
    official_env()
    import gputransform
    import config as cfg
    assert Path(gputransform.__file__).resolve().parent == PYLIB.resolve()
    idx = np.load(d / "index.npz")
    sids, poses = idx["scan_ids"], idx["poses"]
    n = len(sids)
    bits_p, done_p = d / "bits.npy", d / "done.npy"
    bits = np.lib.format.open_memmap(bits_p, mode="r+" if bits_p.exists() else "w+",
                                     dtype=np.uint8, shape=(n, NBYTES))
    done = np.load(done_p) if done_p.exists() else np.zeros(n, dtype=bool)
    nhits = np.load(d / "nhits.npy") if (d / "nhits.npy").exists() else np.full(n, -1, np.int64)
    loader = _make_loader(sensor, seq)
    t0, new, max_dt = time.time(), 0, 0.0
    for i in np.flatnonzero(~done):
        item = loader[int(sids[i])]
        if item.get("pose") is not None:  # loader index space == cache index space
            dt = float(np.abs(np.asarray(item["pose"])[:3, 3] - poses[i][:3, 3]).max())
            max_dt = max(max_dt, dt)
            assert dt < 1e-2, (key, int(sids[i]), dt)
        hits = official_hits(sensor, item["points"])
        bits[i] = pack(official_polar(hits, gputransform, cfg))
        nhits[i] = len(hits)
        done[i] = True
        new += 1
        if new % flush_every == 0:
            bits.flush()
            np.save(done_p, done)
            np.save(d / "nhits.npy", nhits)
            print(f"  [{role}/{key}] {int(done.sum())}/{n} "
                  f"({(time.time() - t0) / new * 1e3:.1f} ms/scan)", flush=True)
    bits.flush()
    del bits
    np.save(d / "nhits.npy", nhits)
    assert done.all()
    meta.update({"complete": True, "bits_sha256": sha256(bits_p),
                 "n_empty": int((nhits == 0).sum()),
                 "hits_median": int(np.median(nhits)),
                 "gputransform_sha256": sha256(Path(gputransform.__file__)),
                 "max_abs_pose_check_m": max_dt})
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    done_p.unlink(missing_ok=True)
    print(f"OK gen {role}/{key}: {new} new in {time.time() - t0:.0f} s, "
          f"{meta['n_empty']} empty, median hits {meta['hits_median']}", flush=True)


def cmd_status() -> bool:
    ok = True
    for role, (seqs, _, _, _) in ROLES.items():
        n_all = n_done = n_empty = 0
        for sensor, seq in seqs:
            d = store_dir(role, sensor, seq)
            if not (d / "meta.json").exists():
                print(f"  {role}/{sensor}/{seq}: not indexed")
                ok = False
                continue
            meta = json.loads((d / "meta.json").read_text())
            n_all += meta["n_frames"]
            if meta.get("complete"):
                n_done += meta["n_frames"]
                n_empty += meta["n_empty"]
            else:
                ok = False
                k = int(np.load(d / "done.npy").sum()) if (d / "done.npy").exists() else 0
                print(f"  {role}/{sensor}/{seq}: {k}/{meta['n_frames']}")
        print(f"{role}: {n_done}/{n_all} scans complete, {n_empty} empty")
    print("POLAR_COMPLETE" if ok else "POLAR_INCOMPLETE")
    return ok


def load_store(role: str, sensor: str, seq: str, mmap: bool = True):
    d = store_dir(role, sensor, seq)
    meta = json.loads((d / "meta.json").read_text())
    assert meta.get("complete"), f"{d}: polar store incomplete"
    assert meta["rule"] == RULE and meta["z_shift"] == Z_SHIFT[sensor], \
        f"{d}: made with another band rule"
    idx = np.load(d / "index.npz")
    bits = np.load(d / "bits.npy", mmap_mode="r" if mmap else None)
    return bits, idx, meta


# ------------------------------------------------------------ checks
SELFTEST_SEQS = [("train", "kitti", "05"), ("train", "helipr", "Bridge02"),
                 ("val", "kitti360", "0004"), ("val", "mulran", "Sejong01"),
                 ("test_odom", "nclt", "2012-01-08"), ("test_odom", "mcd", "kth_day_06")]


def _write_nclt_bin(path: Path, xyz_up: np.ndarray):
    """Points -> official NCLT velodyne_sync format (uint16 x, y, z_raw on the
    5 mm grid, z_raw = -z_up), so the official reader can read any sensor."""
    raw = np.asarray(xyz_up, dtype=np.float64).copy()
    raw[:, 2] = -raw[:, 2]
    u = np.clip(np.rint((raw + 100.0) / 0.005), 0, 65535).astype("<u2")
    rec = np.zeros(len(u), dtype=[("x", "<u2"), ("y", "<u2"), ("z", "<u2"),
                                  ("i", "u1"), ("l", "u1")])
    rec["x"], rec["y"], rec["z"] = u[:, 0], u[:, 1], u[:, 2]
    rec.tofile(path)
    deq = u.astype(np.float64) * 0.005 + -100.0
    deq[:, 2] = -deq[:, 2]
    return deq


def _best_shift(a: np.ndarray, b: np.ndarray) -> int:
    """s maximizing sum a[..., c] * b[..., (c - s) mod 120]  (a ~ roll(b, s))."""
    A = np.fft.fft(a.reshape(-1, a.shape[-1]), axis=-1)   # every (layer, ring) row
    B = np.fft.fft(b.reshape(-1, b.shape[-1]), axis=-1)
    corr = np.fft.ifft((A * np.conj(B)).sum(0)).real
    return int(np.argmax(corr))


def _paper_layout_ok(p, hits, fake, cfg, tol=1e-3) -> bool:
    """Paper mode through the unchanged kernel (CPU transcription): occupied
    (layer, ring) pairs match {(floor(z_up / 1 m), min(floor(r / 2 m), 39))}
    of the kept points; points within `tol` m of a layer / ring edge may fall
    on either side (the kernel works in float32)."""
    x, y, z = p[:, 0], p[:, 1], -p[:, 2]
    keep = (np.abs(x) < RANGE_M) & (z > -20.) & (z < -2.) & (np.abs(y) < RANGE_M) & \
        ~((np.abs(x) < 1.) & (np.abs(y) < 1.))
    if len(hits) != int(keep.sum()):
        return False
    if len(hits) == 0:
        return True
    zu, r = p[keep, 2], np.hypot(x[keep], y[keep])
    gap = RANGE_M / 40
    ring_of = lambda v: np.minimum(np.floor(v / gap).astype(int), 39)  # noqa: E731
    l_lo, l_hi = np.floor(zu - tol).astype(int), np.floor(zu + tol).astype(int)
    r_lo, r_hi = ring_of(r - tol), ring_of(r + tol)
    sure = (l_lo == l_hi) & (r_lo == r_hi)
    possible = {(int(a), int(b)) for la, lb, ra, rb in zip(l_lo, l_hi, r_lo, r_hi)
                for a in (la, lb) for b in (ra, rb)}
    certain = {(int(a), int(b)) for a, b in zip(l_lo[sure], r_lo[sure])}
    got = {(int(a), int(b)) for a, b, _ in np.argwhere(official_polar(hits, fake, cfg) > 0)}
    return certain <= got <= possible and l_lo.min() >= 1 and l_hi.max() <= 19


def cmd_selftest(n_scans: int, out_json: Path | None):
    """CPU only.
    1. reader: for each of six sensors, n_scans keyframes (after Z_SHIFT)
       written in the official NCLT binary format and read by the official
       load_lidar_file_nclt == our filter on the same points (bit-equal), and the official load_pc_file (CPU transcription in place
       of the CUDA module) == our official_polar (bit-equal);
    2. nclt: official reader on the RAW NCLT file == official_hits on our
       NCLT loader's points (bit-equal hits and image);
    3. heading: relation between heading difference and the polar column
       shift (official NCLT heading vs ours on KITTI / HeLiPR train);
    4. fft: torch.rfft / torch.ifft shims == numpy definitions.
    Every reader case also records the kept points and occupied cells.
    """
    import torch  # noqa: F401
    from _pd_build_test_caches import _make_loader
    fake = fake_gputransform_module()
    ns, gcfg = generator_functions(fake)
    cfg, *_ = import_official(cpu_only=True, fake_gputransform=fake)
    for k in ("num_ring", "num_sector", "num_height", "max_length", "max_height"):
        assert getattr(cfg, k) == getattr(gcfg, k), k
    rep = {"official_commit": OFFICIAL_COMMIT, "rule": RULE, "reader": [], "band": {}}
    ok = True
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for role, sensor, seq in SELFTEST_SEQS:
            sids = np.load(cache_path(role, sensor, seq))["scan_ids"]
            loader = _make_loader(sensor, seq)
            for k in np.linspace(0, len(sids) - 1, n_scans).round().astype(int):
                sid = int(sids[k])
                pts = np.asarray(loader[sid]["points"])[:, :3]
                f = td / f"{sensor}_{sid}.bin"
                # the official reader sees the cloud after our Z_SHIFT (KITTI
                # family); the adapter's filter must then equal it exactly
                deq = _write_nclt_bin(f, official_points(sensor, pts))
                off = ns["load_lidar_file_nclt"](str(f))
                ours = official_filter(deq, mode="code")
                hp = official_hits(sensor, pts)
                paper_ok = _paper_layout_ok(official_points(sensor, pts), hp, fake, cfg)
                same = off.shape == ours.shape and np.array_equal(off, ours)
                img_off = ns["load_pc_file"](str(f))
                img_ours = official_polar(ours, fake, cfg)
                same_img = np.array_equal(img_off, img_ours)
                case = {"role": role, "seq": f"{sensor}/{seq}", "scan_id": sid,
                        "z_shift": Z_SHIFT[sensor], "hits_code": int(len(ours)),
                        "occupied_code": int(img_ours.sum()),
                        "hits_equal": bool(same), "image_equal": bool(same_img),
                        "hits_paper": int(len(hp)),
                        "occupied_paper": int(official_polar(hp, fake, cfg).sum()),
                        "paper_layout_ok": bool(paper_ok)}
                same &= paper_ok
                if sensor == "nclt":
                    raw = ns["load_lidar_file_nclt"](str(loader.frame_files[sid]))
                    h2 = official_filter(official_points("nclt", loader[sid]["points"]),
                                         mode="code")
                    case["nclt_raw_hits_equal"] = bool(raw.shape == h2.shape
                                                       and np.array_equal(raw, h2))
                    case["nclt_raw_image_equal"] = bool(np.array_equal(
                        ns["load_pc_file"](str(loader.frame_files[sid])),
                        official_polar(h2, fake, cfg)))
                    same &= case["nclt_raw_hits_equal"] and case["nclt_raw_image_equal"]
                ok &= same and same_img
                rep["reader"].append(case)
                print(json.dumps(case), flush=True)
    rep["heading"] = _heading_check()
    ok &= rep["heading"]["consistent"]
    rep["fft"] = _fft_check()
    ok &= rep["fft"]["ok"]
    rep["all_ok"] = bool(ok)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rep, indent=1))
    print("PREP_SELFTEST_OK" if ok else "PREP_SELFTEST_FAILED", flush=True)
    if not ok:
        sys.exit(1)


def _heading_check(max_pairs: int = 40):
    """Relation between d = heading_q - heading_p (deg) and the measured polar
    column shift s (q ~ roll(p, s)): sign of s / (d / 3).  Official: NCLT
    heading (GT yaw, as the official tuples) with official points (z-down raw
    x/y); ours: KITTI / HeLiPR train caches with `heading`.  Geometry and GT
    poses only (the NCLT sequence is a test sequence; no method output)."""
    from _pd_build_test_caches import _make_loader
    fake = fake_gputransform_module()
    out = {}
    sets = [("train", "kitti", "05", "ours"), ("train", "helipr", "Bridge02", "ours"),
            ("test_odom", "nclt", "2012-01-08", "official")]
    for role, sensor, seq, kind in sets:
        with np.load(cache_path(role, sensor, seq)) as z:
            sids = z["scan_ids"]
            P = z[ROLES[role][3]].astype(np.float64)
        if kind == "ours":
            h = heading(sensor, P)
        else:   # official NCLT tuples: 'yaw' column of the ground truth csv
            h = np.arctan2(P[:, 1, 0], P[:, 0, 0])
        X = P[:, :3, 3]
        pairs = []
        for i in range(len(P) - 3):
            for k in (1, 2, 3):
                d = np.degrees(np.angle(np.exp(1j * (h[i + k] - h[i]))))
                if np.linalg.norm(X[i + k] - X[i]) < 4.0 and 12.0 < abs(d) < 60.0:
                    pairs.append((i + k, i, d))
                    break
        pairs = pairs[::max(1, len(pairs) // max_pairs)][:max_pairs]
        loader = _make_loader(sensor, seq)
        cfg_ = types.SimpleNamespace(max_length=1, max_height=1, num_ring=40,
                                     num_sector=120, num_height=20)
        cache = {}

        def img(i):
            if i not in cache:
                cache[i] = official_polar(official_hits(sensor, loader[int(sids[i])]["points"]),
                                          fake, cfg_)
            return cache[i]
        votes = {"minus": 0, "plus": 0, "neither": 0}
        for q, p, d in pairs:
            s = _best_shift(img(q), img(p))
            s = (s + 60) % 120 - 60
            exp = d / 3.0
            if abs(s + exp) <= 2:
                votes["minus"] += 1
            elif abs(s - exp) <= 2:
                votes["plus"] += 1
            else:
                votes["neither"] += 1
        out[f"{sensor}/{seq} ({kind})"] = {"pairs": len(pairs), **votes}
        print(f"  heading {sensor}/{seq} ({kind}): {votes}", flush=True)
    sign = lambda v: "minus" if v["minus"] > v["plus"] else "plus"  # noqa: E731
    signs = {k: sign(v) for k, v in out.items()}
    out["relation"] = signs
    out["consistent"] = len(set(signs.values())) == 1 and all(
        v[sign(v)] >= 0.6 * v["pairs"] for k, v in out.items() if isinstance(v, dict)
        and "pairs" in v)
    return out


def _fft_check():
    import torch
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 1, 40, 120)).astype(np.float32)
    a = torch.rfft(torch.from_numpy(x), 2, normalized=True, onesided=False).numpy()
    ref = np.fft.fft2(x.astype(np.float64), norm="ortho")
    e1 = float(np.abs(a[..., 0] + 1j * a[..., 1] - ref).max())
    c = rng.standard_normal((3, 1, 40, 120, 2)).astype(np.float32)
    b = torch.ifft(torch.from_numpy(c), 2).numpy()
    ref2 = np.fft.ifft2(c[..., 0].astype(np.float64) + 1j * c[..., 1])
    e2 = float(np.abs(b[..., 0] + 1j * b[..., 1] - ref2).max())
    t = torch.from_numpy(x).requires_grad_(True)
    torch.rfft(t, 2, normalized=True, onesided=False).pow(2).sum().backward()
    # Parseval (ortho): sum |F|^2 = sum x^2 -> d/dx = 2x
    e3 = float((t.grad.numpy() - 2 * x).__abs__().max())
    ok = e1 < 1e-4 and e2 < 1e-5 and e3 < 1e-3
    print(f"  fft shims: rfft max err {e1:.2e}, ifft {e2:.2e}, grad {e3:.2e}", flush=True)
    return {"rfft_max_abs_err": e1, "ifft_max_abs_err": e2, "grad_max_abs_err": e3, "ok": ok}


def cmd_gpu_check(n_scans: int, out_json: Path | None):
    """GPU (run before `gen`): the official CUDA extension vs the CPU
    transcription on n_scans keyframes of six sensors; repeat-call
    determinism of the extension."""
    from _pd_build_test_caches import _make_loader
    official_env()
    import gputransform
    import config as cfg
    rep = {"gputransform": gputransform.__file__,
           "gputransform_sha256": sha256(Path(gputransform.__file__)), "cases": []}
    for role, sensor, seq in SELFTEST_SEQS:
        sids = np.load(cache_path(role, sensor, seq))["scan_ids"]
        loader = _make_loader(sensor, seq)
        for k in np.linspace(0, len(sids) - 1, n_scans).round().astype(int):
            hits = official_hits(sensor, loader[int(sids[k])]["points"])
            t0 = time.perf_counter()
            g1 = official_polar(hits, gputransform, cfg)
            dt = time.perf_counter() - t0
            g2 = official_polar(hits, gputransform, cfg)
            c = official_polar(hits, fake_gputransform_module(), cfg)
            case = {"seq": f"{sensor}/{seq}", "scan_id": int(sids[k]), "hits": int(len(hits)),
                    "occupied_gpu": int(g1.sum()), "cells_differ_cpu": int((g1 != c).sum()),
                    "repeat_equal": bool(np.array_equal(g1, g2)), "ms": round(dt * 1e3, 2)}
            rep["cases"].append(case)
            print(json.dumps(case), flush=True)
    rep["max_cells_differ"] = max(c["cells_differ_cpu"] for c in rep["cases"])
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rep, indent=1))
    print(f"GPU_CHECK_DONE max cells differing from the CPU transcription: "
          f"{rep['max_cells_differ']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index")
    p.add_argument("--role", choices=tuple(ROLES), nargs="+", default=list(ROLES))
    sub.add_parser("jobs")
    p = sub.add_parser("gen")
    p.add_argument("--role", choices=tuple(ROLES), required=True)
    p.add_argument("--seq", required=True)
    sub.add_parser("status")
    p = sub.add_parser("selftest")
    p.add_argument("--n-scans", type=int, default=2)
    p.add_argument("--out-json", type=Path, default=None)
    p = sub.add_parser("gpu-check")
    p.add_argument("--n-scans", type=int, default=3)
    p.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "index":
        for r in a.role:
            cmd_index(r)
    elif a.cmd == "jobs":
        cmd_jobs()
    elif a.cmd == "gen":
        cmd_gen(a.role, a.seq)
    elif a.cmd == "status":
        sys.exit(0 if cmd_status() else 1)
    elif a.cmd == "selftest":
        cmd_selftest(a.n_scans, a.out_json)
    elif a.cmd == "gpu-check":
        cmd_gpu_check(a.n_scans, a.out_json)


if __name__ == "__main__":
    main()
