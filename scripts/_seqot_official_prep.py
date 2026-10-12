#!/usr/bin/env python3
"""SeqOT official pipeline on split v2 -- part 1: range images and overlap
labels (2026-10-11).

Official repo: _external/SeqOT (github.com/BIT-MJY/SeqOT, OFFICIAL_COMMIT).
No official file is changed; the official functions are imported from the
unmodified files (tools/utils/utils.py, tools/utils/com_overlap_yaw.py) and
the official scripts are executed unmodified in --selftest.

Range images (official tools/utils/gen_depth_data.py:62-80):
  utils.range_projection(points, fov_up, fov_down, proj_H=32, proj_W=900,
  max_range=80, cut_z=False, low=0.1, high=6) on float64 (x, y, z) points,
  stored float32 (32, 900), -1 = no return.  Our points come from
  scripts/_pd_build_test_caches._make_loader (sensor frame, z up).
  Field of view per sensor (FOV below): the official file only has the NCLT
  HDL-32E window, which is applied AFTER its z flip (data2xyzi flip=True):
  fov_up 30.67, fov_down -10.67.  Our NCLT loader applies the same flip, so
  NCLT uses exactly the official window and the official float64 coordinates
  (5 mm grid integers recovered from the loader's float32 values).  The
  other sensors use their datasheet windows (our choice; the official code
  has no setting for them).

Overlap labels (official data_prepararion/gen_training_index.py, which calls
tools/utils/com_overlap_yaw.com_overlap_yaw):
  * every 5th raw scan of a TRAIN sequence (gen_training_index.py:30);
  * for each such frame i, the overlap with every frame j >= i
    (com_overlap_yaw on scan_paths[i:], frame_idx=0): both scans projected
    with the range projection above, the reference scan moved into the
    current frame with the GT poses (LiDAR frame), count =
    #(|r_ref - r_cur| < 1.2 on valid reference pixels) (com_overlap_yaw.py:
    57-104; the yaw column is not used by the index and is not computed);
    overlap = count / #valid current (com_overlap_yaw.py:102-103, the
    default 'code'); 'paper' = count / min(#valid current, #valid
    reprojected reference) (Eq. 13) exists but is not used;
  * EXACT shortcut (not an approximation): when the pose distance exceeds
    80 m + the largest point range of the reference scan (+1 m margin), no
    reference point can land inside the 80 m projection, the reference image
    is empty and the official overlap is exactly 0.0.  Those pairs are not
    projected and are kept as implicit zeros;
  * then the official "normalization" (gen_training_index.py:57-100), done
    per sequence with np.random.seed(SEED): raw indices = 5 * frame index,
    6-digit strings; bins [0.7,1.1) [0.5,0.7) [0.3,0.5) x 15000,
    [0.1,0.3) [0,0.1) x 30000, long-term (|raw i - raw j| > 1000 and
    overlap > 0.3) x 10000, all drawn with replacement by np.random.randint;
    output = the official training-index format (N x 3 '<U32' array:
    idx1, idx2, overlap).  An EMPTY bin makes the official script crash
    (randint(0, 0)); we skip that bin (documented in labels meta).
  Poses: KITTI loader poses are camera-0 poses -> LiDAR poses with the
  official utils.load_calib Tr: T_velo_cam . inv(P0) . P . T_cam_velo (the
  OverlapNet ground-truth convention of the same authors); HeLiPR loader
  poses (Velodyne_gt.txt) are LiDAR poses already.

Storage (SEQOT_RANGE_ROOT overrides the range root; default
_external/seqot_official_pdv2/range):
  range/{role}/{sensor}_{seq}/index.npz   raw_ids (rows of range.npy),
                                          kf_scan_ids, n_raw, poses
  range/{role}/{sensor}_{seq}/range.npy   float32 (len(raw_ids), 32, 900)
  range/{role}/{sensor}_{seq}/done_*.json shard completion markers
  labels/{sensor}_{seq}/points.f32        5th-scan points (float32, as loaded)
  labels/{sensor}_{seq}/frames.npz        their offsets, rmax, LiDAR poses
  labels/{sensor}_{seq}/ovc_*of*.npz      projected pairs (i, j, count, valid_cur, valid_ref)
  labels/{sensor}_{seq}/normalized_data.npy  official training index
  roles: train = all raw scans of the 8 TRAIN sequences (training windows
  read raw scans); val / test_odom = the raw scans [s - 21, s] behind each
  keyframe s (3-scan window x 20-scan window, trailing), keyframes from the
  split-v2 caches (val: GT keyframes, test: KISS-ICP keyframes).

Subcommands (inside ring_baseline):
  config                     our config copy (official HEAD values, our paths)
  index [--role R ...] [--seq S ...]   index.npz + empty range.npy per sequence
  jobs [--chunk 2000] [--role R ...] [--seq S ...]   "role seq shard nshards" range jobs
  gen --role R --seq S --shard k --nshards n
  frames --seq S             5th-scan point store for overlap labels (train)
  ovjobs [--nshards 16]      "seq shard nshards" overlap jobs
  overlap --seq S --shard k --nshards n
  labels [--seq S] [--scale official|density] [--force]
                             official normalization -> normalized_data.npy
  status                     completeness of range stores and labels
  selftest [--n-scans 2]     CPU: adapter == official functions / scripts
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
sys.dont_write_bytecode = True      # keep the official clone free of __pycache__

from _pdv2_pilot import SEED, TEST, TRAIN, VAL  # noqa: E402  (split-v2 lists, seed 1)

OFFICIAL = REPO / "_external/SeqOT"
OFFICIAL_COMMIT = "6e10697a4e6a423b5581e6a6ed6df9cfb386658b"
OFFICIAL_REPO = "https://github.com/BIT-MJY/SeqOT"
WORK = REPO / "_external/seqot_official_pdv2"
RANGE = Path(os.environ.get("SEQOT_RANGE_ROOT", WORK / "range"))
LABELS = WORK / "labels"
CONFIG_COPY = WORK / "config_pdv2.yml"
RESULTS = REPO / "results/pdv2_seqot_official"
V2 = REPO / "data/preprocessed_v2"
ODOM = REPO / "data/preprocessed_v2_odom"
DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", "/workspace/data"))
ROLES = {   # role -> (sequence list, cache dir, cache-name middle, pose key)
    "train": (TRAIN, V2, "", "poses"),
    "val": (VAL, V2, "val_", "poses"),
    "test_odom": (TEST, ODOM, "test_", "poses_gt"),
}

# (fov_up, fov_down) in degrees, on our z-up points.  nclt = the official
# gen_depth_data.py:66-67 window (after its z flip == our loader's flip).
FOV = {"nclt": (30.67, -10.67),
       "kitti": (2.0, -24.8), "kitti360": (2.0, -24.8),     # HDL-64E datasheet
       "helipr": (15.0, -15.0),                              # VLP-16
       "mulran": (16.6, -16.6),                              # OS1-64 (gen1)
       "mcd": (22.5, -22.5)}                                 # OS1-64 (gen2)
PROJ = dict(proj_H=32, proj_W=900, max_range=80, cut_z=False, low=0.1, high=6)
H, W, MAX_RANGE = 32, 900, 80.0
DS_STEP = 5            # gen_training_index.py:30
TAU = 1.2              # com_overlap_yaw.py:77
SEQL, GEM_SEQLEN = 3, 20   # config.yml training_seqot.seqlen / training_gem.seqlen
BACK = (SEQL - 1) + (GEM_SEQLEN - 1)   # raw scans behind a keyframe that its
#                                        trailing descriptor reads (21)
SHORTCUT_MARGIN = 1.0  # m, see module docstring
# gen_training_index.py:81-95 -- (lower, upper, count) in drawing order
BINS = [(0.7, 1.1, 15000), (0.5, 0.7, 15000), (0.3, 0.5, 15000),
        (0.1, 0.3, 30000), (0.0, 0.1, 30000)]
LONG_TERM = (1000, 0.3, 10000)   # |raw i - raw j| > 1000 and overlap > 0.3
# Overlap normalisation: the code (com_overlap_yaw.py:102-103) divides by
# valid(R_i) -- used (user rule 2026-10-11: official code behaviour); paper
# Eq. 13 divides by min(valid(R_i), valid(R'_j)) ('paper', not used).
OVERLAP_FORMULA = os.environ.get("SEQOT_OVERLAP_FORMULA", "code")
# every-5th frames of NCLT 2012-01-08 (28,127 scans), the sequence behind the
# official 115,000-row index; `labels --scale density` multiplies the drawn
# counts by M / OFFICIAL_M (same rows per frame as the official index)
OFFICIAL_M = 5626


def seq_name(sensor: str, seq: str) -> str:
    return f"{sensor}_{seq}"


def cache_path(role: str, sensor: str, seq: str) -> Path:
    _, cdir, mid, _ = ROLES[role]
    hits = sorted(glob.glob(str(cdir / f"cache_????????_{sensor}_{mid}{seq}.npz")))
    assert len(hits) == 1, (role, sensor, seq, hits)
    return Path(hits[0])


def store_dir(role: str, sensor: str, seq: str) -> Path:
    return RANGE / role / seq_name(sensor, seq)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _git(*args, check=True) -> str:
    """git on the official clone; -c safe.directory avoids the container's
    'dubious ownership' refusal without touching any git config."""
    return subprocess.run(["git", "-c", f"safe.directory={OFFICIAL}", "-C", str(OFFICIAL), *args],
                          capture_output=True, text=True, check=check).stdout


def _write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)   # e.g. results/pdv2_seqot_official/ on a fresh checkout
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# ------------------------------------------------------------ official code
def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_UTILS = None


def official_utils():
    """tools/utils/utils.py of the official clone (numpy only), loaded by file
    path: a regular `tools` package in site-packages and our src/utils shadow
    the official namespace packages `tools` / `utils`."""
    global _UTILS
    if _UTILS is None:
        _UTILS = _load_file("seqot_official_utils", OFFICIAL / "tools/utils/utils.py")
        assert Path(_UTILS.__file__).resolve() == (OFFICIAL / "tools/utils/utils.py").resolve()
    return _UTILS


class official_imports:
    """Context manager: make `tools`, `tools.utils`, `tools.utils.utils`,
    `modules` and (optionally) `utils` resolve to the official clone while the
    official files are imported, then restore sys.modules / sys.path.
    utils_as: None | 'package' (read_samples: from utils.utils import *) |
    'module' (gen_depth_data: from utils import *)."""

    KEYS = ("tools", "tools.utils", "tools.utils.utils", "tools.loss",
            "tools.read_samples", "tools.utils.com_overlap_yaw", "modules",
            "utils", "utils.utils")

    def __init__(self, utils_as=None, keep=()):
        self.utils_as, self.keep = utils_as, set(keep)

    def __enter__(self):
        self.saved = {k: sys.modules.get(k) for k in self.KEYS}
        self.path = list(sys.path)
        for k in self.KEYS:
            sys.modules.pop(k, None)
        tools = types.ModuleType("tools")
        tools.__path__ = [str(OFFICIAL / "tools")]
        sys.modules["tools"] = tools
        modules = types.ModuleType("modules")
        modules.__path__ = [str(OFFICIAL / "modules")]
        sys.modules["modules"] = modules
        import tools.utils            # noqa: F401  (official package __init__)
        import tools.utils.utils      # noqa: F401
        if self.utils_as == "package":
            sys.modules["utils"] = sys.modules["tools.utils"]
            sys.modules["utils.utils"] = sys.modules["tools.utils.utils"]
        elif self.utils_as == "module":
            sys.modules["utils"] = sys.modules["tools.utils.utils"]
        sys.path.insert(0, str(OFFICIAL))
        return self

    def __exit__(self, *exc):
        sys.path[:] = self.path
        for k, v in self.saved.items():
            if k in self.keep:
                continue
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        return False


# ------------------------------------------------------------ adapter
_R_FLIP = np.eye(3)
_R_FLIP[2, 2] = -1


def nclt_official_coords(p: np.ndarray) -> np.ndarray:
    """Our NCLT loader: float32(u) * 0.005 - 100 in float32, z negated.
    Official gen_depth_data.data2xyzi: u * 0.005 - 100.0 in float64, then
    np.matmul(xyz, diag(1, 1, -1)).  u is recovered exactly (float32 error
    << 5 mm) and the official float64 arithmetic is repeated."""
    raw = np.asarray(p, dtype=np.float64)[:, :3].copy()
    raw[:, 2] = -raw[:, 2]
    u = np.rint((raw + 100.0) / 0.005)
    assert np.all(np.abs(raw - (u * 0.005 - 100.0)) < 1e-3), "not on the NCLT 5 mm grid"
    xyz = u * 0.005 - 100.0
    return np.matmul(xyz, _R_FLIP)


def official_points(sensor: str, points: np.ndarray) -> np.ndarray:
    """Loader points -> the float64 (N, 3) array the official code projects."""
    if sensor == "nclt":
        return nclt_official_coords(points)
    return np.asarray(points)[:, :3].astype(np.float64)


def range_image(sensor: str, points: np.ndarray) -> np.ndarray:
    """gen_depth_data.py:66-80 with the sensor's window (FOV)."""
    up, down = FOV[sensor]
    img, _, _ = official_utils().range_projection(
        official_points(sensor, points), fov_up=up, fov_down=down, **PROJ)
    return img


def overlap_counts(cur_pts, cur_pose, refs, fov):
    """com_overlap_yaw.py:57-104 (overlap only) for one current frame and the
    reference frames `refs` = [(points float64 (N,3), pose 4x4), ...].
    Returns (valid_num of the current image, [(count, valid_ref), ...]) with
    count = #(|r_ref - r_cur| < 1.2) on valid reference pixels (the official
    numerator) and valid_ref = #valid pixels of the reprojected reference."""
    rp = official_utils().range_projection
    up, down = fov
    current_range, project_points, _ = rp(cur_pts, fov_up=up, fov_down=down, **PROJ)
    visible_points = project_points[current_range > 0]
    valid_num = len(visible_points)
    assert valid_num > 0, "empty current range image (official code divides by it)"
    current_pose = cur_pose
    tau_ = TAU
    out = []
    for reference_points_tmp, reference_pose in refs:
        reference_points = np.ones((reference_points_tmp.shape[0], reference_points_tmp.shape[1] + 1))
        reference_points[:, :-1] = reference_points_tmp
        reference_points_world = reference_pose.dot(reference_points.T).T
        reference_points_in_current = np.linalg.inv(current_pose).dot(reference_points_world.T).T
        reference_range, _, _ = rp(reference_points_in_current, fov_up=up, fov_down=down, **PROJ)
        count = np.count_nonzero(
            abs(reference_range[reference_range > 0] - current_range[reference_range > 0]) < tau_)
        out.append((count, int(np.count_nonzero(reference_range > 0))))
    return valid_num, out


def overlap_row(cur_pts, cur_pose, refs, fov):
    """Official overlap (com_overlap_yaw.py:102-103: count / valid_num, Python
    int / int as in the official code)."""
    valid_num, cv = overlap_counts(cur_pts, cur_pose, refs, fov)
    return [c / valid_num for c, _ in cv]


def overlap_values(count, valid_cur, valid_ref, formula):
    """formula 'code': count / valid(current)  (com_overlap_yaw.py:102-103);
    'paper': count / min(valid(current), valid(reprojected reference))
    (paper Eq. 13), 0 when the reprojected reference is empty (Eq. 13 is 0/0
    there; no reference pixel -> no overlap).  The paper's '<= delta' equals
    the code's '< 1.2' here: a float32 range difference is never exactly the
    float64 1.2.  float64 division of exact integers == Python int / int."""
    c = np.asarray(count, np.float64)
    vc = np.asarray(valid_cur, np.float64)
    if formula == "code":
        return c / vc
    vr = np.asarray(valid_ref, np.float64)
    den = np.minimum(vc, vr)
    return np.where(den > 0, c / np.where(den > 0, den, 1.0), 0.0)


def lidar_poses(sensor: str, seq: str, loader) -> np.ndarray:
    """GT poses of every loader scan in the LiDAR frame (see docstring)."""
    if sensor == "kitti":
        T_cam_velo = official_utils().load_calib(
            str(DATA_ROOT / "kitti/dataset/sequences" / seq / "calib.txt"))
        T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        T_velo_cam = np.linalg.inv(T_cam_velo)
        P = np.asarray(loader.poses, dtype=np.float64)
        pose0_inv = np.linalg.inv(P[0])
        return np.stack([T_velo_cam.dot(pose0_inv).dot(p).dot(T_cam_velo) for p in P])
    if sensor == "helipr":
        return np.asarray(loader.scan_poses, dtype=np.float64)
    raise ValueError(f"no training sequences for sensor {sensor}")


# ------------------------------------------------------------ config copy
def cmd_config():
    """Official config.yml at OFFICIAL_COMMIT (the clone's working copy has
    local path edits) with only the path entries pointed at our stores."""
    import yaml
    txt = _git("show", f"{OFFICIAL_COMMIT}:config/config.yml")
    cfg = yaml.safe_load(txt)
    ours = {
        "data_root": {"range_image_database_root": str(RANGE / "train") + "/<sensor>_<seq>/range.npy",
                      "range_image_query_root": str(RANGE / "test_odom") + "/<sensor>_<seq>/range.npy"},
        "training_seqot": {"traindata_file": str(LABELS) + "/<sensor>_<seq>/normalized_data.npy",
                           "poses_file": str(LABELS) + "/<sensor>_<seq>/frames.npz",
                           "weights": None},
        "gen_sub_descriptors": {"weights": "results/pdv2_seqot_official/train_selection.json:stage1.chosen"},
        "training_gem": {"traindata_file": str(LABELS) + "/<sensor>_<seq>/normalized_data.npy",
                         "poses_file": str(LABELS) + "/<sensor>_<seq>/frames.npz",
                         "descriptors_file": str(WORK) + "/subdesc/train/<sensor>_<seq>.npy",
                         "weights": None},
        "test_gem_prepare": {"sub_descriptors_database_file": None,
                             "sub_descriptors_query_file": None,
                             "weights": "results/pdv2_seqot_official/train_selection.json:stage2.chosen"},
        "test_seqot": {"ground_truth_file": "data/preprocessed_v2_odom (poses_gt, scoring only)"},
        "viz": {"poses_database": None, "poses_query": None},
        "gen_training_index": {"poses_database": str(LABELS) + "/<sensor>_<seq>/frames.npz",
                               "poses_query": None,
                               "scan_database_root": "scripts/_pd_build_test_caches._make_loader"},
    }
    changed = []
    for sec, kv in ours.items():
        for k, v in kv.items():
            assert k in cfg[sec], (sec, k)
            cfg[sec][k] = v
            changed.append(f"{sec}.{k}")
    WORK.mkdir(parents=True, exist_ok=True)
    head = (f"# SeqOT config.yml @ {OFFICIAL_COMMIT} with only the path entries "
            f"changed for split v2\n# changed: {', '.join(changed)}\n"
            "# every other value (height, width, seqlen, lr, resume) is the official one\n")
    CONFIG_COPY.write_text(head + yaml.safe_dump(cfg, sort_keys=False))
    print(f"OK config -> {CONFIG_COPY}", flush=True)


def official_config():
    """The official config.yml at OFFICIAL_COMMIT (code values; the copy in
    CONFIG_COPY documents the values actually used)."""
    import yaml
    return yaml.safe_load(_git("show", f"{OFFICIAL_COMMIT}:config/config.yml"))


# ------------------------------------------------------------ range stores
def needed_raw_ids(role: str, kf: np.ndarray, n_raw: int) -> np.ndarray:
    if role == "train":
        return np.arange(n_raw, dtype=np.int64)
    lo = np.maximum(kf - BACK, 0)
    mark = np.zeros(n_raw + 1, dtype=np.int64)
    np.add.at(mark, lo, 1)
    np.add.at(mark, kf + 1, -1)
    return np.flatnonzero(np.cumsum(mark)[:n_raw] > 0).astype(np.int64)


def cmd_index(role: str, only=None):
    from _pd_build_test_caches import _make_loader
    seqs, _, _, pose_key = ROLES[role]
    for sensor, seq in seqs:
        if only and f"{sensor}/{seq}" not in only:
            continue
        d = store_dir(role, sensor, seq)
        d.mkdir(parents=True, exist_ok=True)
        cp = cache_path(role, sensor, seq)
        with np.load(cp) as z:
            kf = z["scan_ids"].astype(np.int64)
            kf_poses = z[pose_key].astype(np.float64)
        loader = _make_loader(sensor, seq)
        n_raw = len(loader)
        assert np.all(np.diff(kf) > 0) and kf.max() < n_raw, (role, sensor, seq)
        raw_ids = needed_raw_ids(role, kf, n_raw)
        idx = d / "index.npz"
        if idx.exists():
            with np.load(idx) as z:
                assert np.array_equal(z["raw_ids"], raw_ids), f"{d}: index changed"
        else:
            np.savez(idx, raw_ids=raw_ids, kf_scan_ids=kf, kf_poses=kf_poses,
                     n_raw=np.int64(n_raw))
        rng = d / "range.npy"
        if not rng.exists():
            mm = np.lib.format.open_memmap(rng, mode="w+", dtype=np.float32,
                                           shape=(len(raw_ids), H, W))
            del mm
        meta = {"role": role, "sensor": sensor, "sequence": seq, "n_raw": int(n_raw),
                "n_rows": int(len(raw_ids)), "n_keyframes": int(len(kf)),
                "rows": "all raw scans" if role == "train" else
                        f"raw scans [s-{BACK}, s] of every keyframe s",
                "cache": str(cp.relative_to(REPO)), "cache_sha256": sha256(cp),
                "kf_poses_from": pose_key, "fov_up_down_deg": list(FOV[sensor]),
                "projection": {**PROJ, "fn": "tools/utils/utils.py range_projection"},
                "official": f"SeqOT@{OFFICIAL_COMMIT[:12]} gen_depth_data.py:62-80",
                "loader": "scripts/_pd_build_test_caches._make_loader"}
        _write_json(d / "meta.json", meta)
        print(f"OK index {role}/{seq_name(sensor, seq)}: {len(raw_ids)} rows "
              f"({n_raw} raw, {len(kf)} keyframes)", flush=True)
        del loader


def cmd_jobs(chunk: int, roles, only=None):
    for role in roles:
        for sensor, seq in ROLES[role][0]:
            if only and f"{sensor}/{seq}" not in only:
                continue
            with np.load(store_dir(role, sensor, seq) / "index.npz") as z:
                n = len(z["raw_ids"])
            k = max(1, -(-n // chunk))
            for s in range(k):
                print(role, f"{sensor}/{seq}", s, k)


def _shard_rows(n, shard, nshards):
    a = (n * shard) // nshards
    b = (n * (shard + 1)) // nshards
    return a, b


def cmd_gen(role: str, key: str, shard: int, nshards: int):
    from _pd_build_test_caches import _make_loader
    sensor, seq = key.split("/")
    assert (sensor, seq) in ROLES[role][0], (role, key)
    d = store_dir(role, sensor, seq)
    done = d / f"done_{shard}of{nshards}.json"
    if done.exists():
        print(f"SKIP {role}/{key} shard {shard}/{nshards}: done", flush=True)
        return
    with np.load(d / "index.npz") as z:
        raw_ids = z["raw_ids"]
    a, b = _shard_rows(len(raw_ids), shard, nshards)
    mm = np.load(d / "range.npy", mmap_mode="r+")
    loader = _make_loader(sensor, seq)
    t0 = time.time()
    for r in range(a, b):
        mm[r] = range_image(sensor, loader[int(raw_ids[r])]["points"])
        if (r - a + 1) % 1000 == 0:
            print(f"  [{role}/{key} {shard}/{nshards}] {r - a + 1}/{b - a} "
                  f"({(time.time() - t0) / (r - a + 1):.3f} s/scan)", flush=True)
    mm.flush()
    del mm
    _write_json(done, {"rows": [a, b], "s": round(time.time() - t0, 1)})
    print(f"OK gen {role}/{key} shard {shard}/{nshards}: rows {a}-{b} in "
          f"{time.time() - t0:.0f} s", flush=True)


def range_complete(role: str, sensor: str, seq: str) -> bool:
    d = store_dir(role, sensor, seq)
    if not (d / "index.npz").exists():
        return False
    with np.load(d / "index.npz") as z:
        n = len(z["raw_ids"])
    cover = np.zeros(n, dtype=bool)
    for f in d.glob("done_*of*.json"):
        a, b = json.loads(f.read_text())["rows"]
        cover[a:b] = True
    return bool(cover.all())


def open_store(role: str, sensor: str, seq: str):
    """(range memmap, raw_ids, kf_scan_ids, kf_poses, n_raw) of a complete store."""
    assert range_complete(role, sensor, seq), f"range store incomplete: {role}/{sensor}/{seq}"
    d = store_dir(role, sensor, seq)
    with np.load(d / "index.npz") as z:
        out = (z["raw_ids"], z["kf_scan_ids"], z["kf_poses"], int(z["n_raw"]))
    return (np.load(d / "range.npy", mmap_mode="r"),) + out


# ------------------------------------------------------------ overlap labels
def label_dir(sensor: str, seq: str) -> Path:
    return LABELS / seq_name(sensor, seq)


def cmd_frames(key: str):
    """Points (float32, as the loader returns them) and LiDAR poses of every
    5th raw scan of one TRAIN sequence, for the overlap workers."""
    from _pd_build_test_caches import _make_loader
    sensor, seq = key.split("/")
    assert (sensor, seq) in TRAIN, key
    d = label_dir(sensor, seq)
    d.mkdir(parents=True, exist_ok=True)
    out = d / "frames.npz"
    if out.exists():
        print(f"SKIP frames {key}: exists", flush=True)
        return
    loader = _make_loader(sensor, seq)
    n_raw = len(loader)
    poses_all = lidar_poses(sensor, seq, loader)
    assert len(poses_all) == n_raw
    ds = np.arange(0, n_raw, DS_STEP)
    off, rmax = [0], []
    t0 = time.time()
    with open(d / "points.f32", "wb") as f:      # (sum N_k, 3) float32, row-major
        for k, sid in enumerate(ds):
            p = np.ascontiguousarray(np.asarray(loader[int(sid)]["points"])[:, :3])
            assert p.dtype == np.float32, p.dtype     # float32 -> float64 is exact
            f.write(p.tobytes())
            off.append(off[-1] + len(p))
            rmax.append(float(np.linalg.norm(p.astype(np.float64), axis=1).max()))
            if (k + 1) % 1000 == 0:
                print(f"  [frames {key}] {k + 1}/{len(ds)}", flush=True)
    tmp = d / "frames.tmp.npz"
    np.savez(tmp, offsets=np.asarray(off, np.int64), rmax=np.asarray(rmax),
             poses=poses_all[ds], raw_ids=ds, n_raw=np.int64(n_raw))
    os.replace(tmp, out)
    print(f"OK frames {key}: {len(ds)} frames, {off[-1]} points, "
          f"{time.time() - t0:.0f} s", flush=True)


class Frames:
    def __init__(self, sensor, seq):
        d = label_dir(sensor, seq)
        with np.load(d / "frames.npz") as z:
            self.offsets, self.rmax = z["offsets"], z["rmax"]
            self.poses, self.raw_ids = z["poses"], z["raw_ids"]
        self.points = np.memmap(d / "points.f32", dtype=np.float32, mode="r").reshape(-1, 3)
        assert len(self.points) == self.offsets[-1]
        self.M = len(self.raw_ids)

    def pts(self, i):
        return self.points[self.offsets[i]:self.offsets[i + 1]].astype(np.float64)


def near_refs(F: Frames, i: int) -> np.ndarray:
    """Frames j >= i whose official overlap can be non-zero."""
    t = F.poses[:, :3, 3]
    d = np.linalg.norm(t[i:] - t[i], axis=1)
    return i + np.flatnonzero(d < MAX_RANGE + F.rmax[i:] + SHORTCUT_MARGIN)


def cmd_ovjobs(nshards: int):
    """Overlap jobs of the sequences without labels; a sequence whose
    existing shards used another count keeps that count."""
    for sensor, seq in TRAIN:
        d = label_dir(sensor, seq)
        if (d / "normalized_data.npy").exists():
            continue
        have = {int(f.name.split("of")[1].split(".")[0]) for f in d.glob("ovc_*of*.npz")}
        n = have.pop() if len(have) == 1 else nshards
        for k in range(n):
            print(f"{sensor}/{seq}", k, n)


def cmd_overlap(key: str, shard: int, nshards: int):
    sensor, seq = key.split("/")
    d = label_dir(sensor, seq)
    out = d / f"ovc_{shard}of{nshards}.npz"
    if out.exists():
        print(f"SKIP overlap {key} {shard}/{nshards}", flush=True)
        return
    F = Frames(sensor, seq)
    I, J, C, VC, VR = [], [], [], [], []
    t0, n_pairs = time.time(), 0
    anchors = range(shard, F.M, nshards)
    for c, i in enumerate(anchors):
        js = near_refs(F, i)
        vc, cv = overlap_counts(F.pts(i), F.poses[i], [(F.pts(j), F.poses[j]) for j in js], FOV[sensor])
        I.append(np.full(len(js), i, np.int64))
        J.append(js.astype(np.int64))
        C.append(np.array([c for c, _ in cv], np.int64))
        VR.append(np.array([v for _, v in cv], np.int64))
        VC.append(np.full(len(js), vc, np.int64))
        n_pairs += len(js)
        if (c + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  [overlap {key} {shard}/{nshards}] anchor {c + 1}/{len(anchors)} "
                  f"pairs {n_pairs} ({1e3 * el / max(n_pairs, 1):.2f} ms/pair)", flush=True)
    tmp = d / f"ovc_{shard}of{nshards}.tmp.npz"
    cat = lambda L: np.concatenate(L) if L else np.zeros(0, np.int64)  # noqa: E731
    np.savez(tmp, i=cat(I), j=cat(J), count=cat(C), valid_cur=cat(VC), valid_ref=cat(VR),
             seconds=np.float64(time.time() - t0))
    os.replace(tmp, out)
    print(f"OK overlap {key} {shard}/{nshards}: {n_pairs} pairs in "
          f"{time.time() - t0:.0f} s", flush=True)


def load_explicit(sensor, seq, formula):
    """Projected pairs (i, j, overlap by `formula`) in the official row order."""
    d = label_dir(sensor, seq)
    files = sorted(d.glob("ovc_*of*.npz"))
    assert files, f"no overlap shards in {d}"
    ns = {int(f.name.split("of")[1].split(".")[0]) for f in files}
    assert len(ns) == 1, f"{d}: mixed shard counts {ns}"
    n = ns.pop()
    assert len(files) == n, f"{d}: {len(files)}/{n} overlap shards"
    parts = [np.load(f) for f in files]
    i = np.concatenate([p["i"] for p in parts])
    j = np.concatenate([p["j"] for p in parts])
    o = overlap_values(*(np.concatenate([p[k] for p in parts])
                         for k in ("count", "valid_cur", "valid_ref")), formula)
    order = np.lexsort((j, i))
    return i[order], j[order], o[order]


class ImplicitBin:
    """The rows of one official bin (lo <= overlap < hi) among all pairs
    (i, j >= i) of M frames, in the official row order (i major, j minor),
    where only the explicit (projected) pairs carry an overlap and every
    other pair is an exact 0.0 -- without materialising the M(M+1)/2 rows."""

    def __init__(self, M, i, j, o, lo, hi):
        self.i, self.j, self.o, self.lo, self.hi = i, j, o, lo, hi
        self.start = np.searchsorted(i, np.arange(M + 1))
        self.zeros_in = lo <= 0.0 < hi
        m = (o >= lo) & (o < hi)
        cnt = np.bincount(i[m], minlength=M).astype(np.int64)
        if self.zeros_in:   # implicit zeros belong to this bin
            cnt += (M - np.arange(M)) - np.bincount(i, minlength=M)
        self.cum = np.cumsum(cnt)
        self.size = int(self.cum[-1]) if M else 0

    def pick(self, k):
        """(i, j, overlap) of the k-th row (0-based) of the bin."""
        a = int(np.searchsorted(self.cum, k, side="right"))
        r = int(k - (self.cum[a - 1] if a else 0))
        s, e = self.start[a], self.start[a + 1]
        js, os_ = self.j[s:e], self.o[s:e]
        inbin = (os_ >= self.lo) & (os_ < self.hi)
        if not self.zeros_in:
            return a, int(js[inbin][r]), float(os_[inbin][r])
        jj = a + r
        for x in js[~inbin]:          # r-th j in [a, M) that is not an explicit
            if x <= jj:               # pair outside the bin (sorted ascending)
                jj += 1
            else:
                break
        hit = np.flatnonzero(js == jj)
        return a, jj, (float(os_[hit[0]]) if len(hit) else 0.0)


def official_row(i, j, ov):
    """One row as gen_training_index.py builds it (lines 46-51, then 62-64)."""
    return [str(int(i) * DS_STEP).zfill(6), str(int(j) * DS_STEP).zfill(6), np.float64(ov)]


def normalize(M, i, j, o, seed, factor=1.0):
    """gen_training_index.py:57-100 on the implicit-zero pair set (same bins,
    same np.random.randint call order and sizes, same output format).
    factor != 1 scales the drawn counts (NOT official; see OFFICIAL_M).
    Returns (normalized '<U32' array, report)."""
    cnt = (lambda n: n) if factor == 1.0 else (lambda n: max(1, int(round(n * factor))))
    np.random.seed(seed)
    rep = {"seed": seed, "M": int(M), "n_pairs_total": int(M * (M + 1) // 2),
           "n_pairs_projected": int(len(i)), "count_factor": factor, "bins": [],
           "skipped_empty_bins": []}
    chosen_rows = []
    for lo, hi, n in BINS:
        n = cnt(n)
        B = ImplicitBin(M, i, j, o, lo, hi)
        rep["bins"].append({"range": [lo, hi], "size": B.size, "drawn": n if B.size else 0})
        if B.size == 0:
            rep["skipped_empty_bins"].append([lo, hi])
            continue
        ks = np.random.randint(0, B.size, n)
        chosen_rows += [official_row(*B.pick(int(k))) for k in ks]
    gap, thr, n = LONG_TERM
    n = cnt(n)
    lt = np.flatnonzero((np.abs(i - j) * DS_STEP > gap) & (o > thr))   # official row order
    rep["long_term"] = {"size": int(len(lt)), "drawn": n if len(lt) else 0}
    if len(lt):
        ks = np.random.randint(0, len(lt), n)
        chosen_rows += [official_row(i[lt[k]], j[lt[k]], o[lt[k]]) for k in ks]
    else:
        rep["skipped_empty_bins"].append("long_term")
    arr = np.array(chosen_rows)
    rep["n_rows"] = int(len(arr))
    rep["dtype"] = str(arr.dtype)
    return arr, rep


def cmd_labels(keys, scale="official", force=False, formula=None):
    formula = formula or OVERLAP_FORMULA
    for key in keys:
        sensor, seq = key.split("/")
        d = label_dir(sensor, seq)
        out = d / "normalized_data.npy"
        if out.exists() and not force:
            old = json.loads((d / "labels_meta.json").read_text())
            assert (old.get("scale"), old.get("overlap_formula")) == (scale, formula), \
                f"{key}: labels exist with {old.get('scale')}/{old.get('overlap_formula')}; use --force"
            print(f"SKIP labels {key}: exists", flush=True)
            continue
        with np.load(d / "frames.npz") as z:
            M = len(z["raw_ids"])
        i, j, o = load_explicit(sensor, seq, formula)
        assert np.all(j >= i) and np.all(np.isfinite(o)), key
        factor = 1.0 if scale == "official" else M / OFFICIAL_M
        arr, rep = normalize(M, i, j, o, SEED, factor)
        rep["scale"] = scale
        rep["overlap_formula"] = formula
        np.save(d / "normalized_data.tmp.npy", arr)
        os.replace(d / "normalized_data.tmp.npy", out)
        np.savez_compressed(d / "pairs_explicit.npz", i=i, j=j, overlap=o)
        rep.update({"sequence": key, "fov_up_down_deg": list(FOV[sensor]),
                    "official": f"SeqOT@{OFFICIAL_COMMIT[:12]} gen_training_index.py + "
                                "com_overlap_yaw.py (overlap)",
                    "shortcut": f"pairs with pose distance >= 80 m + rmax(ref) + "
                                f"{SHORTCUT_MARGIN} m are exact zeros (not projected)"})
        _write_json(d / "labels_meta.json", rep)
        print(f"OK labels {key}: {rep['n_rows']} rows, projected "
              f"{rep['n_pairs_projected']}/{rep['n_pairs_total']} pairs, "
              f"skipped {rep['skipped_empty_bins']}", flush=True)


def labels_complete() -> bool:
    return all((label_dir(s, q) / "normalized_data.npy").exists() for s, q in TRAIN)


def cmd_status() -> bool:
    ok = True
    for role, (seqs, _, _, _) in ROLES.items():
        n_ok = 0
        for sensor, seq in seqs:
            c = range_complete(role, sensor, seq)
            n_ok += c
            if not c:
                print(f"  range {role}/{sensor}/{seq}: incomplete")
        print(f"range {role}: {n_ok}/{len(seqs)} complete")
        ok &= n_ok == len(seqs) or role == "test_odom"
    nl = sum((label_dir(s, q) / "normalized_data.npy").exists() for s, q in TRAIN)
    print(f"labels: {nl}/{len(TRAIN)}")
    ok &= nl == len(TRAIN)
    print("PREP_COMPLETE" if ok else "PREP_INCOMPLETE")
    return ok


# ------------------------------------------------------------ CPU selftest
SELFTEST_SEQS = [("train", "kitti/05"), ("train", "helipr/Bridge02"),
                 ("val", "kitti360/0004"), ("val", "mulran/Sejong01"),
                 ("test_odom", "nclt/2012-01-08"), ("test_odom", "mcd/kth_day_06")]


def _official_gen_depth():
    with official_imports(utils_as="module"):
        return _load_file("seqot_official_gen_depth", OFFICIAL / "tools/utils/gen_depth_data.py")


def _official_com_overlap():
    with official_imports():
        import matplotlib
        matplotlib.use("Agg")
        return _load_file("seqot_official_com_overlap", OFFICIAL / "tools/utils/com_overlap_yaw.py")


def _write_nclt_bin(path: Path, xyz_up: np.ndarray) -> np.ndarray:
    """Points -> NCLT velodyne_sync records (uint16 x, y, z_raw on the 5 mm
    grid, z_raw = -z_up) so the official reader can read any sensor; returns
    what the official get_velo will return (float64)."""
    raw = np.asarray(xyz_up, dtype=np.float64).copy()
    raw[:, 2] = -raw[:, 2]
    u = np.rint((raw + 100.0) / 0.005)
    assert u.min() >= 0 and u.max() <= 65535
    u = u.astype("<u2")
    rec = np.zeros(len(u), dtype=[("x", "<u2"), ("y", "<u2"), ("z", "<u2"),
                                  ("i", "u1"), ("l", "u1")])
    rec["x"], rec["y"], rec["z"] = u[:, 0], u[:, 1], u[:, 2]
    rec.tofile(path)
    return np.matmul(u.astype(np.float64) * 0.005 - 100.0, _R_FLIP)


def _selftest_range(n_scans, report):
    """(1) NCLT: official gen_depth_data() on the RAW velodyne_sync files ==
    adapter on our loader points.  (2) KITTI: official load_vertex on the raw
    .bin + range_projection == adapter.  (3) every sensor: adapter == official
    range_projection on the same float64 points with the sensor window, plus
    row-usage diagnostics."""
    from _pd_build_test_caches import _make_loader
    gd = _official_gen_depth()
    U = official_utils()
    ok = True
    for role, key in SELFTEST_SEQS:
        sensor, seq = key.split("/")
        sids = np.load(cache_path(role, sensor, seq))["scan_ids"]
        pick = [int(sids[int(k)]) for k in np.linspace(0, len(sids) - 1, n_scans).round().astype(int)]
        loader = _make_loader(sensor, seq)
        for sid in pick:
            pts = loader[sid]["points"]
            t0 = time.perf_counter()
            ours = range_image(sensor, pts)
            dt = time.perf_counter() - t0
            up, down = FOV[sensor]
            ref, _, _ = U.range_projection(official_points(sensor, pts), fov_up=up, fov_down=down, **PROJ)
            case = {"role": role, "seq": key, "scan_id": sid, "adapter_s": round(dt, 4),
                    "equal_range_projection": bool(np.array_equal(ours, ref))}
            p = official_points(sensor, pts)
            dep = np.linalg.norm(p, axis=1)
            m = (dep > 0) & (dep < MAX_RANGE)
            pitch = np.degrees(np.arcsin(p[m, 2] / dep[m]))
            case["frac_points_in_window"] = round(float(np.mean((pitch >= down) & (pitch <= up))), 4)
            case["frac_above_window"] = round(float(np.mean(pitch > up)), 4)
            case["frac_below_window"] = round(float(np.mean(pitch < down)), 4)
            case["rows_used"] = int(((ours > 0).sum(axis=1) > 0).sum())
            case["valid_px"] = int((ours > 0).sum())
            if sensor == "kitti":
                vtx = U.load_vertex(str(loader.frame_files[sid]))
                off, _, _ = U.range_projection(vtx, fov_up=up, fov_down=down, **PROJ)
                case["equal_official_load_vertex"] = bool(np.array_equal(off, ours))
                ok &= case["equal_official_load_vertex"]
            if sensor == "nclt":
                with tempfile.TemporaryDirectory() as td:
                    src, dst = Path(td) / "velodyne_sync", Path(td) / "range_image"
                    src.mkdir(), dst.mkdir()
                    os.symlink(loader.frame_files[sid], src / "000000.bin")
                    cwd = os.getcwd()
                    os.chdir(td)
                    try:
                        gd.gen_depth_data(str(src), str(dst))   # official, unmodified
                    finally:
                        os.chdir(cwd)
                    off = np.load(dst / "depth" / "000000.npy")
                case["equal_official_gen_depth_data_raw_file"] = bool(
                    off.dtype == ours.dtype and np.array_equal(off, ours))
                ok &= case["equal_official_gen_depth_data_raw_file"]
            ok &= case["equal_range_projection"]
            report["range"].append(case)
            print(json.dumps(case), flush=True)
        del loader
    return ok


def _selftest_labels(report, n_frames=202, n_points=3000, n_full=40):
    """The official gen_training_index.py, executed unmodified (its own
    com_overlap_yaw, normalization and np.random draws after np.random.seed),
    on a small synthetic NCLT-format sequence vs our overlap rows (NCLT window,
    exact-zero shortcut) + implicit-zero sampler.  Sequence: KITTI 05 raw
    scans (real points with |coord| < 99 m, LiDAR poses) placed at raw indices
    0, 5, 10, ... (the frames the official script reads; the files in between
    are empty and never read): frames 0..n_full-1 = every 2nd raw scan at full
    density (overlaps fall from 1 to < 0.3, so every bin is populated), the
    rest every 10th raw scan subsampled to n_points (speed; the 1.7 km span
    makes the exact-zero shortcut skip far pairs), the last frame a copy of frame 0
    (scan and pose) so the long-term bin is not empty (the official script
    crashes on any empty bin)."""
    from _pd_build_test_caches import _make_loader
    loader = _make_loader("kitti", "05")
    poses_all = lidar_poses("kitti", "05", loader)
    n_raw = DS_STEP * n_frames
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        scans, cfgd, prep = td / "velodyne_sync", td / "config", td / "data_prepararion"
        for p in (scans, cfgd, prep):
            p.mkdir()
        poses = np.tile(np.eye(4), (n_raw, 1, 1))
        deq = {}
        for k in range(n_frames):
            src = 0 if k == n_frames - 1 else (2 * k if k < n_full else 2 * n_full + 10 * (k - n_full))
            pts = np.asarray(loader[src]["points"])[:, :3].astype(np.float64)
            pts = pts[np.all(np.abs(pts) < 99.0, axis=1)]
            if 0 < k < n_frames - 1 and k >= n_full:
                pts = pts[np.linspace(0, len(pts) - 1, n_points).round().astype(int)]
            deq[k] = _write_nclt_bin(scans / f"{k * DS_STEP:06d}.bin", pts)
            poses[k * DS_STEP] = poses_all[src]
        for r in range(n_raw):
            if r % DS_STEP:
                (scans / f"{r:06d}.bin").write_bytes(b"")    # listed, never read
        np.save(td / "poses_db.npy", poses)
        np.save(td / "poses_q.npy", poses[:5])
        import yaml
        (cfgd / "config.yml").write_text(yaml.safe_dump({"gen_training_index": {
            "poses_database": str(td / "poses_db.npy"), "poses_query": str(td / "poses_q.npy"),
            "scan_database_root": str(scans) + "/"}}))
        # ---- official script, unmodified
        src_file = OFFICIAL / "data_prepararion/gen_training_index.py"
        code = compile(src_file.read_text(), str(src_file), "exec")
        cwd = os.getcwd()
        t0 = time.time()
        with official_imports():
            import matplotlib
            matplotlib.use("Agg")
            os.chdir(prep)
            try:
                np.random.seed(SEED)
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(code, {"__name__": "__main__", "__file__": str(src_file)})
            finally:
                os.chdir(cwd)
        t_off = time.time() - t0
        off_raw = np.load(prep / "no_normalized_data.npy", allow_pickle=True)
        off_norm = np.load(prep / "normalized_data.npy", allow_pickle=True)
    # ---- ours
    t0 = time.time()
    P = np.stack([poses[k * DS_STEP] for k in range(n_frames)])
    rmax = np.array([np.linalg.norm(deq[k], axis=1).max() for k in range(n_frames)])
    I, J, C, VC, VR = [], [], [], [], []
    t = P[:, :3, 3]
    for i in range(n_frames):      # the stored-count path of cmd_overlap / load_explicit
        d = np.linalg.norm(t[i:] - t[i], axis=1)
        js = i + np.flatnonzero(d < MAX_RANGE + rmax[i:] + SHORTCUT_MARGIN)
        vc, cv = overlap_counts(deq[i], P[i], [(deq[j], P[j]) for j in js], FOV["nclt"])
        I += [i] * len(js)
        J += js.tolist()
        C += [c for c, _ in cv]
        VR += [v for _, v in cv]
        VC += [vc] * len(js)
    I, J = np.asarray(I), np.asarray(J)
    O = overlap_values(C, VC, VR, "code")
    ours_norm, rep = normalize(n_frames, I, J, O, SEED)
    t_ours = time.time() - t0
    # full official row list from our implicit representation
    full = np.zeros((n_frames, n_frames))
    full[I, J] = O
    exp_ov = np.array([full[i, j] for i in range(n_frames) for j in range(i, n_frames)])
    off_ov = off_raw[:, 2].astype(float)
    case = {"n_frames": n_frames, "n_full": n_full, "n_points": n_points, "pairs": int(len(off_raw)),
            "projected_pairs_ours": int(len(I)),
            "shortcut_zero_pairs": int(len(off_raw) - len(I)),
            "shortcut_pairs_official_overlap_zero": bool(np.all(exp_ov[~np.isin(
                np.arange(len(exp_ov)), _row_pos(I, J, n_frames))] == 0.0)),
            "overlap_all_pairs_equal": bool(np.array_equal(off_ov, exp_ov)),
            "max_abs_overlap_diff": float(np.abs(off_ov - exp_ov).max()),
            "nonzero_pairs": int((off_ov > 0).sum()),
            "normalized_shape": [list(off_norm.shape), list(ours_norm.shape)],
            "normalized_dtype": [str(off_norm.dtype), str(ours_norm.dtype)],
            "normalized_equal": bool(off_norm.shape == ours_norm.shape and np.array_equal(off_norm, ours_norm)),
            "bins": rep["bins"], "long_term": rep["long_term"],
            "official_s": round(t_off, 1), "ours_s": round(t_ours, 1)}
    # paper Eq. 13 normalisation on the same counts (no official reference):
    # numerator identical, denominator min(valid cur, valid ref) -> >= code value
    Op = overlap_values(C, VC, VR, "paper")
    _, rep_p = normalize(n_frames, I, J, Op, SEED)
    case["paper_formula"] = {"overlap_ge_code_all": bool(np.all(Op >= O)),
                             "pairs_changed": int((Op != O).sum()),
                             "bins": [b["size"] for b in rep_p["bins"]],
                             "long_term": rep_p["long_term"]["size"]}
    report["labels"] = case
    print(json.dumps(case), flush=True)
    return case["overlap_all_pairs_equal"] and case["normalized_equal"] and \
        case["paper_formula"]["overlap_ge_code_all"]


def _selftest_overlap_real(report, n_anchor=2, n_ref=6):
    """Official com_overlap_yaw (unmodified, NCLT reader + NCLT window) on
    NCLT-format files of real full HeLiPR Bridge02 scans vs overlap_row with
    the NCLT window; then cost of overlap_row per sensor window (ms/pair)."""
    co = _official_com_overlap()
    from _pd_build_test_caches import _make_loader
    out = {}
    ok = True
    for key in ("helipr/Bridge02", "kitti/05"):
        sensor, seq = key.split("/")
        loader = _make_loader(sensor, seq)
        poses_all = lidar_poses(sensor, seq, loader)
        base = 400 if sensor == "helipr" else 100
        idx = [base + DS_STEP * k for k in range(n_ref)]
        with tempfile.TemporaryDirectory() as td:
            paths, deq = [], []
            for r in idx:
                pts = np.asarray(loader[r]["points"])[:, :3].astype(np.float64)
                pts = pts[np.all(np.abs(pts) < 99.0, axis=1)]
                f = Path(td) / f"{r:06d}.bin"
                deq.append(_write_nclt_bin(f, pts))
                paths.append(str(f))
            P = poses_all[idx]
            eq = []
            for a in range(n_anchor):
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    gm = co.com_overlap_yaw(paths[a:], P[a:], frame_idx=0, leg_output_width=360)
                ours = overlap_row(deq[a], P[a], [(deq[j], P[j]) for j in range(a, n_ref)], FOV["nclt"])
                eq.append(bool(np.array_equal(gm[:, 2], np.asarray(ours))))
        # timing with the sensor's own window on full scans
        F_pts = [np.asarray(loader[r]["points"])[:, :3].astype(np.float64) for r in idx]
        t0 = time.perf_counter()
        ov = overlap_row(F_pts[0], P[0], [(F_pts[j], P[j]) for j in range(n_ref)], FOV[sensor])
        ms = 1e3 * (time.perf_counter() - t0) / n_ref
        out[key] = {"official_equal": eq, "ms_per_pair_full_scan": round(ms, 2),
                    "overlaps_sensor_window": [round(x, 4) for x in ov],
                    "points_per_scan": int(len(F_pts[0]))}
        ok &= all(eq)
        print(key, json.dumps(out[key]), flush=True)
        del loader
    report["overlap_real"] = out
    return ok


def _row_pos(I, J, M):
    """Official row position of pair (i, j >= i) among all pairs of M frames."""
    I, J = np.asarray(I), np.asarray(J)
    return I * M - I * (I - 1) // 2 + (J - I)


def cmd_selftest(n_scans: int, out_json: Path | None, parts):
    report = {"official_commit": OFFICIAL_COMMIT, "range": []}
    ok = True
    if "range" in parts:
        ok &= _selftest_range(n_scans, report)
    if "overlap" in parts:
        ok &= _selftest_overlap_real(report)
    if "labels" in parts:
        ok &= _selftest_labels(report)
    st = _git("status", "--porcelain", check=False)
    report["official_clone_status"] = st.splitlines()
    report["all_ok"] = bool(ok)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out_json, report)
    print("PREP_SELFTEST_OK" if ok else "PREP_SELFTEST_FAILED", flush=True)
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("config")
    p = sub.add_parser("index")
    p.add_argument("--role", choices=tuple(ROLES), nargs="+", default=list(ROLES))
    p.add_argument("--seq", nargs="+", default=None)
    p = sub.add_parser("jobs")
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--role", choices=tuple(ROLES), nargs="+", default=list(ROLES))
    p.add_argument("--seq", nargs="+", default=None)
    p = sub.add_parser("gen")
    p.add_argument("--role", choices=tuple(ROLES), required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p = sub.add_parser("frames")
    p.add_argument("--seq", required=True)
    p = sub.add_parser("ovjobs")
    p.add_argument("--nshards", type=int, default=16)
    p = sub.add_parser("overlap")
    p.add_argument("--seq", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p = sub.add_parser("labels")
    p.add_argument("--seq", nargs="+", default=None)
    p.add_argument("--scale", choices=("official", "density"),
                   default=os.environ.get("SEQOT_LABEL_SCALE", "official"),
                   help="official: the script's fixed counts per sequence; density: "
                        "counts x M / 5626 (not official, rows per frame of the official index)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--formula", choices=("paper", "code"), default=None,
                   help="overlap normalisation (default SEQOT_OVERLAP_FORMULA or code)")
    sub.add_parser("status")
    p = sub.add_parser("selftest")
    p.add_argument("--n-scans", type=int, default=2)
    p.add_argument("--parts", nargs="+", default=["range", "overlap", "labels"])
    p.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "config":
        cmd_config()
    elif a.cmd == "index":
        for r in a.role:
            cmd_index(r, a.seq)
    elif a.cmd == "jobs":
        cmd_jobs(a.chunk, a.role, a.seq)
    elif a.cmd == "gen":
        cmd_gen(a.role, a.seq, a.shard, a.nshards)
    elif a.cmd == "frames":
        cmd_frames(a.seq)
    elif a.cmd == "ovjobs":
        cmd_ovjobs(a.nshards)
    elif a.cmd == "overlap":
        cmd_overlap(a.seq, a.shard, a.nshards)
    elif a.cmd == "labels":
        cmd_labels(a.seq or [f"{s}/{q}" for s, q in TRAIN], a.scale, a.force, a.formula)
    elif a.cmd == "status":
        sys.exit(0 if cmd_status() else 1)
    elif a.cmd == "selftest":
        cmd_selftest(a.n_scans, a.out_json, a.parts)


if __name__ == "__main__":
    main()
