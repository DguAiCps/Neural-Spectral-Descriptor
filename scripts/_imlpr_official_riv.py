#!/usr/bin/env python3
"""ImLPR official pipeline on split v2 -- part 1: RIV images (2026-10-11).

Feeds our loaders' points into the OFFICIAL RIV generator of _external/ImLPR
(github.com/minwoo0611/ImLPR, commit OFFICIAL_COMMIT, unmodified):
  * preprocess/generate_riv_images.py is imported; `compute_v_indices`,
    `scale_log_to_255` are called as they are, and the per-frame recipe of
    `process_one_frame` (lines 358-395: normals -> reflectivity image ->
    range image -> normal image -> HxWx3 float32 stack) is followed;
  * the script's own parameters (lines 417-438) are used: max_neighbor_m =
    15.0, W = 1022, (H0, Hf) = (128, 126) for the angle / Ouster path and
    (64, 64) for the MulRan path, 3.0 m training subsample; the normal k is
    the PAPER value (App. A: 25 for MulRan / NCLT / VLP-16), not the script's
    15 (project rule 2026-10-11: paper over code where they differ);
  * three pieces are re-implemented for speed and checked against the
    official functions in `selftest` (same inputs -> same output):
      - fast_normals: the per-point SVD loop of compute_normals (lines 41-61)
        as one batched float32 np.linalg.svd over the same sklearn kNN;
      - fast_image / fast_range: the per-point Python max/min pooling loops
        of pointCloudToImage / pointCloudToRangeImage (lines 133-137,
        186-190) as np.maximum.at / np.minimum.at (order-free, so equal).

What is ours (official code has no setting for these sensors; SENSORS below):
  * which scans: the split-v2 keyframes (train: GT keyframes thinned with the
    official 3.0 m rule `select_frame_indices`; val: all GT keyframes;
    test_odom: all KISS-ICP keyframes);
  * points: scripts/_pd_build_test_caches._make_loader (sensor frame, z up;
    NCLT is flipped by the loader, so its window is [-10.67, +30.67] deg);
  * the official rule applied per sensor (paper App. A/D + code): 64-beam
    sensors single scan, VLP-16 / HDL-32E 5 s accumulated submaps
    (paper App. D.1/D.3) built from PAST scans only (online protocol);
  * intensity scaling for sensors without the Ouster reflectivity channel.

Layout (official README layout under one dataset_folder):
  _external/imlpr_official_pdv2/data/{training,validation,test_odom}/{sensor}_{seq}/
      index.npz  (scan_ids, cache_idx, poses_lidar, timestamps[, poses_gt])
      meta.json, poses.txt (training only: "t x y z qx qy qz qw", official format)
      {pos:06d}.npy  (Hf, 1022, 3) float32 0..255 [reflectivity, range, normal]
  training images are stored at 126 rows (the official TrainingDataset does
  not resize); validation / test images at the generator height Hf, resized
  to 126 rows at load time exactly as eval/evaluate.py:95-98 does.

Subcommands (run inside ring_baseline):
  index [--role R ...]        keyframe lists, poses, meta
  jobs [--chunk N] [--role R ...] [--seq S]   "role seq shard nshards" jobs
  gen --role R --seq S [--shard k --nshards n]   write RIVs (resumable)
  status [--role R ...]       count RIVs against the index
  selftest [--n-scans 2] [--out-json P]   CPU: adapter == official functions
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

from _pdv2_pilot import TEST, TRAIN, VAL  # noqa: E402  (split-v2 lists)

OFFICIAL = REPO / "_external/ImLPR"
OFFICIAL_COMMIT = "5b2a79ac4504722437b9b3f6d91d2e5a1743d2b6"
WORK = REPO / "_external/imlpr_official_pdv2"
DATA = WORK / "data"                       # official `dataset_folder`
V2 = REPO / "data/preprocessed_v2"
ODOM = REPO / "data/preprocessed_v2_odom"
ODOM_TRACKS = REPO / "results/pdv2_c3/odometry"   # KISS-ICP pose of every scan
DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", "/workspace/data"))

# role -> (sequence list, cache dir, cache-name middle, sub-folder of DATA)
ROLES = {
    "train": (TRAIN, V2, "", "training"),
    "val": (VAL, V2, "val_", "validation"),
    "test_odom": (TEST, ODOM, "test_", "test_odom"),
}

# generate_riv_images.py:428-438 (the released script's settings)
SCRIPT_K_NEIGHBORS = 15        # script value, overridden per sensor by the paper (SENSORS["k"])
MAX_NEIGHBOR_M = 15.0          # script value (the paper gives no radius)
# Paper App. A (p.12): "k = 8 nearest neighbors for the HeLiPR-O dataset and
# k = 25 for the MulRan, NCLT and HeLiPR-V datasets" -> project rule (paper
# over code): k = 25 for every sensor here (no OS2-128 in our data; KITTI
# HDL-64E and MCD OS1-64 are not in the paper and follow the non-OS2-128 value)
PAPER_K_NON_OUSTER128 = 25
IMAGE_W = 1022
TRAIN_H = 126                  # DINOv2 input rows (paper: H x W = 126 x 1022)
SUBSAMPLE_M = 3.0              # select_frame_indices dist_threshold
# Accumulation (paper App. D.1: "we aggregate 5 seconds of scan data to form
# a submap ... we exclude scans acquired during stationary periods"; the
# aggregation script itself is not released, README TODO).  Online protocol:
# only the keyframe scan and scans BEFORE it.  "Stationary" is ours: a past
# scan is skipped when it lies within STATIONARY_M of the last kept scan.
ACCUM_WINDOW_S = 5.0
STATIONARY_M = 0.10

# Per-sensor settings.  `src` says where each value comes from.
#   v_mode / H0 / Hf : compute_v_indices mode and image heights
#   fov_up / fov_down: angle-mode window (degrees, both positive numbers)
#   max_range        : range-channel scale (paper App. C "LiDAR's maximum range")
#   intensity        : how the loader's 4th column becomes 0..255
#   accumulate       : 5 s past submap
SENSORS = {
    # MulRan OS1-64: official MulRan path verbatim (ring = idx % 64, H0 = Hf =
    # 64, 120 m, reflectivity clamp 4096 -> /4096*255; read_bin_file:244-256,
    # script comments :434, :437); eval resizes 64 -> 126 (evaluate.py:95-98).
    "mulran": dict(k=PAPER_K_NON_OUSTER128, v_mode="ring", ring="idx%64", H0=64, Hf=64, fov_up=16.0, fov_down=16.0,
                   max_range=120.0, intensity="mulran4096", accumulate=False,
                   src="official MulRan settings"),
    # MCD OS1-64: same sensor model as MulRan -> MulRan rule.  Our MCD files
    # are row-major (beam = idx // 1024; idx % 64 does not follow the beams).
    "mcd": dict(k=PAPER_K_NON_OUSTER128, v_mode="ring", ring="idx//1024", H0=64, Hf=64, fov_up=22.5, fov_down=22.5,
                max_range=120.0, intensity="mulran4096", accumulate=False,
                src="MulRan rule applied to OS1-64 (row-major beam index)"),
    # KITTI / KITTI-360 HDL-64E: 64 beams like OS1-64 -> single scan, 64 rows,
    # no beam index in the files -> official 'angle' mode with the HDL-64E
    # window; 120 m (HDL-64E spec); loader reflectance 0..1 -> x255.
    "kitti": dict(k=PAPER_K_NON_OUSTER128, v_mode="angle", ring=None, H0=64, Hf=64, fov_up=2.0, fov_down=24.8,
                  max_range=120.0, intensity="x255", accumulate=False,
                  src="rule: 64-beam single scan, angle mode, sensor window"),
    "kitti360": dict(k=PAPER_K_NON_OUSTER128, v_mode="angle", ring=None, H0=64, Hf=64, fov_up=2.0, fov_down=24.8,
                     max_range=120.0, intensity="x255", accumulate=False,
                     src="rule: 64-beam single scan, angle mode, sensor window"),
    # HeLiPR VLP-16: paper App. D.1 (5 s submap, Roundabout-V) -> accumulated,
    # projected at the generator's default size (128 -> 126, INTER_AREA) in
    # angle mode with the VLP-16 window; 100 m (VLP-16 spec); intensity 0..255.
    "helipr": dict(k=PAPER_K_NON_OUSTER128, v_mode="angle", ring=None, H0=128, Hf=126, fov_up=15.0, fov_down=15.0,
                   max_range=100.0, intensity="clip255", accumulate=True,
                   src="paper App. D.1 (5 s submap) + generator default size"),
    # NCLT HDL-32E: paper App. D.3 (same accumulation as HeLiPR-V).  The
    # sensor is mounted upside down; the loader flips z, so the window after
    # the flip is [-10.67, +30.67] deg.  Paper: "we invert and rescale the
    # intensity distribution" (no formula) -> 255 - raw (decision D5).
    "nclt": dict(k=PAPER_K_NON_OUSTER128, v_mode="angle", ring=None, H0=128, Hf=126, fov_up=30.67, fov_down=10.67,
                 max_range=100.0, intensity=os.environ.get("IMLPR_NCLT_INTENSITY", "invert255"),
                 accumulate=True,
                 src="paper App. D.3 (5 s submap, inverted intensity) + flipped window"),
}


def seq_name(sensor: str, seq: str) -> str:
    return f"{sensor}_{seq}"


def role_dir(role: str, sensor: str, seq: str) -> Path:
    return DATA / ROLES[role][3] / seq_name(sensor, seq)


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


# ------------------------------------------------------------ official code
_GEN = None


def official_generator():
    """preprocess/generate_riv_images.py, imported unmodified (its __main__
    block is guarded)."""
    global _GEN
    if _GEN is None:
        spec = importlib.util.spec_from_file_location(
            "imlpr_generate_riv_images", OFFICIAL / "preprocess/generate_riv_images.py")
        _GEN = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_GEN)
        assert (_GEN.k_neighbors, _GEN.max_neighbor_m, _GEN.W_img) == \
            (SCRIPT_K_NEIGHBORS, MAX_NEIGHBOR_M, IMAGE_W), "official script settings changed"
    return _GEN


# ------------------------------------------------------------ poses
def kitti_tr(seq: str) -> np.ndarray:
    """Tr (velodyne -> cam0) of a KITTI odometry sequence (calib.txt)."""
    for line in open(DATA_ROOT / "kitti/dataset/sequences" / seq / "calib.txt"):
        if line.startswith("Tr:"):
            T = np.eye(4)
            T[:3, :] = np.array(line.split()[1:], dtype=np.float64).reshape(3, 4)
            return T
    raise ValueError(f"no Tr in calib of kitti {seq}")


def lidar_poses(sensor: str, seq: str, poses: np.ndarray) -> np.ndarray:
    """Cache poses -> poses of the LiDAR frame in a z-up world.  KITTI caches
    hold cam0 poses (world = cam0 of frame 0, y down): Tr^-1 T Tr gives the
    velodyne pose in the velodyne frame of frame 0 (x forward, z up).  Other
    sensors' cache poses are already LiDAR poses (HeLiPR Velodyne_gt,
    MulRan/MCD/KITTI-360 positions, KISS-ICP).  Needed for the official XY
    KD-tree and the Patch-InfoNCE relative transforms."""
    if sensor != "kitti":
        return poses
    tr = kitti_tr(seq)
    return np.einsum("ij,njk,kl->nil", np.linalg.inv(tr), poses, tr)


def select_frame_indices(positions: np.ndarray, dist_threshold: float) -> list[int]:
    """generate_riv_images.py:330-345 on an (N, 3) position array (the official
    function takes pose dicts; same rule: keep the first, then every pose
    farther than the threshold from the last kept one, 3-D Euclidean)."""
    gen = official_generator()
    poses = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in positions]
    return gen.select_frame_indices(poses, dist_threshold)


# ------------------------------------------------------------ loaders / tracks
def make_loader(sensor: str, seq: str):
    from _pd_build_test_caches import _make_loader
    return _make_loader(sensor, seq)


class Track:
    """Time stamp and pose of EVERY loader scan (for accumulation).  train /
    val: the dataset GT the keyframes were chosen with (HeLiPR Velodyne_gt,
    one pose per scan).  test_odom: the KISS-ICP pose of every scan
    (results/pdv2_c3/odometry), i.e. the poses the test keyframes were chosen
    with; GT is never used to build test inputs."""

    def __init__(self, role: str, sensor: str, seq: str, loader=None):
        self.loader = loader if loader is not None else make_loader(sensor, seq)
        n = len(self.loader)
        if role == "test_odom":
            z = np.load(ODOM_TRACKS / f"{sensor}_{seq}.npz")
            self.T, self.t = z["odom_poses"].astype(np.float64), z["timestamps"].astype(np.float64)
            self.source = str((ODOM_TRACKS / f"{sensor}_{seq}.npz").relative_to(REPO))
        elif sensor == "helipr":
            self.T = np.asarray(self.loader.scan_poses, dtype=np.float64)
            self.t = np.asarray(self.loader.scan_timestamps, dtype=np.int64) / 1e9
            self.source = "loader GT (LiDAR_GT/Velodyne_gt.txt, one pose per scan)"
        else:
            raise ValueError(f"no accumulation track for {role} {sensor}")
        assert len(self.T) == len(self.t) == n, (role, sensor, seq, len(self.T), n)
        assert np.all(np.diff(self.t) > 0), "scan times not increasing"

    def window(self, k: int) -> list[int]:
        """Scan k plus past scans j < k with t_k - t_j <= 5 s, newest first,
        skipping a scan within STATIONARY_M of the last kept one."""
        keep, last = [k], self.T[k][:3, 3]
        j = k - 1
        while j >= 0 and self.t[k] - self.t[j] <= ACCUM_WINDOW_S:
            p = self.T[j][:3, 3]
            if np.linalg.norm(p - last) > STATIONARY_M:
                keep.append(j)
                last = p
            j -= 1
        return keep

    def submap(self, k: int):
        """Points of window(k) in the frame of scan k: (N, 4) float32."""
        idx = self.window(k)
        Tk_inv = np.linalg.inv(self.T[k])
        out = []
        for j in idx:
            p = np.asarray(self.loader[j]["points"], dtype=np.float32)
            if j != k:
                rel = Tk_inv @ self.T[j]
                xyz = p[:, :3].astype(np.float64) @ rel[:3, :3].T + rel[:3, 3]
                p = np.concatenate([xyz.astype(np.float32), p[:, 3:4]], axis=1)
            out.append(p)
        return np.concatenate(out, axis=0), idx


# ------------------------------------------------------------ per-sensor inputs
def intensity_255(sensor: str, inten: np.ndarray) -> np.ndarray:
    mode = SENSORS[sensor]["intensity"]
    i = np.asarray(inten, dtype=np.float32)
    if mode == "x255":            # KITTI reflectance 0..1 ("normalized intensity")
        return np.clip(i * 255.0, 0.0, 255.0).astype(np.float32)
    if mode == "clip255":         # HeLiPR VLP-16 intensity is already 0..255
        return np.clip(i, 0.0, 255.0).astype(np.float32)
    if mode in ("x255raw", "invert255"):   # NCLT loader returns uint8 / 255
        raw = np.rint(i.astype(np.float64) * 255.0)
        return (raw if mode == "x255raw" else 255.0 - raw).astype(np.float32)
    if mode == "mulran4096":      # read_bin_file 'mulran' branch, per point
        r = np.minimum(i.astype(np.float64), 4096.0)
        return (r / 4096.0 * 255.0).astype(np.float32)
    raise ValueError(mode)


def ring_index(sensor: str, n: int) -> np.ndarray:
    rule = SENSORS[sensor]["ring"]
    if rule == "idx%64":
        return (np.arange(n) % 64).astype(np.int32)
    if rule == "idx//1024":
        assert n == 65536, n
        return (np.arange(n) // 1024).astype(np.int32)
    return np.zeros(n, dtype=np.int32)   # unused in angle mode


def frame_points(sensor: str, role: str, item_points, track: Track | None, k: int):
    """(xyz float32, 0..255 values float32, ring int32, used scan indices)."""
    if SENSORS[sensor]["accumulate"]:
        pts, used = track.submap(k)
    else:
        pts, used = np.asarray(item_points, dtype=np.float32), [k]
    xyz = np.ascontiguousarray(pts[:, :3], dtype=np.float32)
    return xyz, intensity_255(sensor, pts[:, 3]), ring_index(sensor, len(xyz)), used


# ------------------------------------------------------------ renderers
def official_frame(xyz, refl, ring, sensor, normal=None):
    """process_one_frame lines 361-395 with the official functions (slow
    reference used by the selftest; `normal` = compute_normals output if
    already computed)."""
    gen, c = official_generator(), SENSORS[sensor]
    kw = dict(image_width=IMAGE_W, image_height_initial=c["H0"], image_height_final=c["Hf"],
              v_mode=c["v_mode"], fov_up_deg=c["fov_up"], fov_down_deg=c["fov_down"],
              invert_vertical=False)
    if normal is None:
        normal = gen.compute_normals(xyz, k=c["k"], max_neighbor_m=MAX_NEIGHBOR_M)
    img_r = gen.pointCloudToImage(xyz, refl, ring, **kw)
    img_g = gen.pointCloudToRangeImage(xyz, ring, lidar_type=sensor, max_range_m=c["max_range"], **kw)
    img_n = gen.pointCloudToImage(xyz, normal, ring, **kw)
    return np.concatenate([img_r[:, :, np.newaxis], img_g[:, :, np.newaxis],
                           img_n[:, :, np.newaxis]], axis=2).astype(np.float32)


def fast_normals(points, k, max_neighbor_m=MAX_NEIGHBOR_M, chunk=200_000):
    """compute_normals (lines 41-61) without the per-point loop: same sklearn
    kNN call, same float32 centring, the same LAPACK float32 SVD batched over
    points, same ratio / far-neighbour / log-scale steps.

    Ouster files (MulRan ~40k, MCD ~9k per scan) mark missing returns as exact
    (0,0,0) points.  A kNN over thousands of identical points is slow and its
    result is fixed: a zero point's k neighbours are zeros (normal value of an
    all-zero neighbourhood), and any other point can take at most k zeros.
    So all but k+1 zero points are left out of the kNN and get the value of
    the kept ones; the neighbour lists of every point are unchanged
    (bit-equality with compute_normals is checked in selftest)."""
    points = np.asarray(points, dtype=np.float32)
    zero = np.flatnonzero(~points.any(axis=1))
    if len(zero) > k + 1:
        keep = np.ones(len(points), dtype=bool)
        keep[zero[k + 1:]] = False
        out = np.empty(len(points), dtype=np.float32)
        out[keep] = _normals_knn(points[keep], k, max_neighbor_m, chunk)
        kept_zero = np.flatnonzero(keep[zero])          # positions inside `zero`
        vals = out[zero[kept_zero]]
        assert np.all(vals == vals[0]), "zero points with different normal values"
        out[zero[k + 1:]] = vals[0]
        return out
    return _normals_knn(points, k, max_neighbor_m, chunk)


def _normals_knn(points, k, max_neighbor_m, chunk):
    from sklearn.neighbors import NearestNeighbors
    gen = official_generator()
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    ratio = np.empty(len(points), dtype=np.float64)
    for s in range(0, len(points), chunk):
        neigh = points[indices[s:s + chunk, 1:]]                  # (n, k, 3) float32
        neigh = neigh - np.mean(neigh, axis=1, keepdims=True)
        sv = np.linalg.svd(neigh, full_matrices=False, compute_uv=False)   # (n, 3) float32
        largest = np.max(sv, axis=1).astype(np.float64)
        smallest = (np.min(sv, axis=1) + 1e-5).astype(np.float64)
        ratio[s:s + chunk] = largest / smallest
    ratio[np.max(distances[:, 1:], axis=1) > max_neighbor_m] = 0.7
    return gen.scale_log_to_255(ratio)


def _uv(points, ring, c):
    gen = official_generator()
    azimuth = np.arctan2(points[:, 1], points[:, 0])                 # lines 124-126
    azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)
    u = (azimuth / (2 * np.pi) * (IMAGE_W - 1)).astype(np.int32)
    v = gen.compute_v_indices(points, ring, c["H0"], mode=c["v_mode"],
                              fov_up_deg=c["fov_up"], fov_down_deg=c["fov_down"],
                              invert_vertical=False)
    ok = (u >= 0) & (u < IMAGE_W) & (v >= 0) & (v < c["H0"])
    return u[ok], v[ok], ok


def _resize(image, c):
    import cv2
    if c["Hf"] is not None and c["Hf"] != c["H0"]:
        image = cv2.resize(image, (IMAGE_W, int(c["Hf"])), interpolation=cv2.INTER_AREA)
    return image


def fast_image(points, values, ring, c):
    """pointCloudToImage: max pooling (np.maximum.at == the loop, any order)."""
    u, v, ok = _uv(points, ring, c)
    image = np.zeros((c["H0"], IMAGE_W), dtype=np.float32)
    np.maximum.at(image, (v, u), np.asarray(values, dtype=np.float32)[ok])
    return _resize(image, c)


def fast_range(points, ring, c):
    """pointCloudToRangeImage: min pooling of the scaled range."""
    u, v, ok = _uv(points, ring, c)
    image = np.ones((c["H0"], IMAGE_W), dtype=np.float32) * 255.0
    range_point = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
    max_range_m = c["max_range"]
    range_scaled = np.clip(range_point, 0.0, float(max_range_m)) / float(max_range_m) * 255.0
    channel_vals = range_scaled.astype(np.float32)
    np.minimum.at(image, (v, u), channel_vals[ok])
    image = np.where(image == 255.0, 0.0, image)
    return _resize(image, c)


def fast_frame(xyz, refl, ring, sensor):
    c = SENSORS[sensor]
    normal = fast_normals(xyz, c["k"])
    img_r = fast_image(xyz, refl, ring, c)
    img_g = fast_range(xyz, ring, c)
    img_n = fast_image(xyz, normal, ring, c)
    return np.concatenate([img_r[:, :, np.newaxis], img_g[:, :, np.newaxis],
                           img_n[:, :, np.newaxis]], axis=2).astype(np.float32)


def to_train_height(img: np.ndarray) -> np.ndarray:
    """64-row images -> 126 rows with the official eval resize (evaluate.py:
    95-98, INTER_LINEAR).  The height-only resize treats every column alone,
    so resizing before the 28-column padding equals resizing after it
    (checked in selftest)."""
    import cv2
    if img.shape[0] == TRAIN_H:
        return img
    return cv2.resize(img, (img.shape[1], TRAIN_H), interpolation=cv2.INTER_LINEAR).astype(np.float32)


# ------------------------------------------------------------ index / gen
def cmd_index(role: str):
    seqs, _, _, _ = ROLES[role]
    for sensor, seq in seqs:
        cp = cache_path(role, sensor, seq)
        with np.load(cp) as z:
            sids = z["scan_ids"].astype(np.int64)
            poses = z["poses"].astype(np.float64)
            stamps = z["timestamps"].astype(np.float64)
            poses_gt = z["poses_gt"].astype(np.float64) if "poses_gt" in z.files else None
        assert len(sids) == len(poses) and np.all(np.diff(sids) > 0), (role, sensor, seq)
        # GT caches (train/val) -> LiDAR poses; odometry caches already are.
        lp = poses if role == "test_odom" else lidar_poses(sensor, seq, poses)
        keep = np.arange(len(sids))
        if role == "train":
            keep = np.asarray(select_frame_indices(lp[:, :3, 3], SUBSAMPLE_M), dtype=np.int64)
        d = role_dir(role, sensor, seq)
        d.mkdir(parents=True, exist_ok=True)
        arrays = dict(scan_ids=sids[keep], cache_idx=keep, poses_lidar=lp[keep],
                      poses_cache=poses[keep], timestamps=stamps[keep])
        if poses_gt is not None:
            arrays["poses_gt"] = poses_gt[keep]
        np.savez(d / "index.npz", **arrays)
        if role == "train":   # official poses.txt: "timestamp x y z qx qy qz qw"
            from scipy.spatial.transform import Rotation
            q = Rotation.from_matrix(lp[keep][:, :3, :3]).as_quat()
            with open(d / "poses.txt", "w") as f:
                for t, P, qq in zip(stamps[keep], lp[keep], q):
                    f.write(f"{float(t)!r} {float(P[0, 3])!r} {float(P[1, 3])!r} {float(P[2, 3])!r} "
                            f"{float(qq[0])!r} {float(qq[1])!r} {float(qq[2])!r} {float(qq[3])!r}\n")
        c = SENSORS[sensor]
        meta = {"role": role, "sensor": sensor, "sequence": seq,
                "n_frames": int(len(keep)), "n_cache_keyframes": int(len(sids)),
                "cache": str(cp.relative_to(REPO)), "cache_sha256": sha256(cp),
                "subsample": (f"official select_frame_indices, {SUBSAMPLE_M} m, on LiDAR "
                              "positions" if role == "train" else None),
                "stored_height": TRAIN_H if role == "train" else c["Hf"],
                "sensor_settings": {k: v for k, v in c.items()},
                "generator": {"official_commit": OFFICIAL_COMMIT,
                              "k_neighbors": c["k"], "k_source": "paper App. A (script: 15)",
                              "max_neighbor_m": MAX_NEIGHBOR_M, "image_width": IMAGE_W},
                "accumulation": ({"window_s": ACCUM_WINDOW_S, "past_only": True,
                                  "stationary_skip_m": STATIONARY_M,
                                  "poses": "KISS-ICP (results/pdv2_c3/odometry)"
                                  if role == "test_odom" else "dataset GT"}
                                 if c["accumulate"] else None),
                "loader": "scripts/_pd_build_test_caches._make_loader"}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"OK index {role}/{sensor}_{seq}: {len(keep)} frames "
              f"(of {len(sids)} keyframes)", flush=True)


def _index(role, sensor, seq):
    return np.load(role_dir(role, sensor, seq) / "index.npz")


def cmd_jobs(chunk: int, roles, only_seq=None):
    for role in roles:
        for sensor, seq in ROLES[role][0]:
            if only_seq and f"{sensor}/{seq}" != only_seq:
                continue
            n = len(_index(role, sensor, seq)["scan_ids"])
            if SENSORS[sensor]["accumulate"]:
                c = max(1, chunk // 8)       # ~10-25 s per accumulated frame
            else:
                c = chunk
            ns = max(1, -(-n // c))
            for k in range(ns):
                print(role, f"{sensor}/{seq}", k, ns)


def riv_for(role, sensor, seq, pos, sid, loader, track, check_pose=None):
    item = loader[int(sid)]
    if check_pose is not None and item.get("pose") is not None and role != "test_odom":
        dt = float(np.abs(np.asarray(item["pose"])[:3, 3] - check_pose[:3, 3]).max())
        assert dt < 1e-2, (role, sensor, seq, int(sid), dt)
    xyz, refl, ring, used = frame_points(sensor, role, item["points"], track, int(sid))
    img = fast_frame(xyz, refl, ring, sensor)
    if role == "train":
        img = to_train_height(img)
    return img, used, len(xyz)


def cmd_gen(role: str, key: str, shard: int, nshards: int, limit: int = 0, out_dir=None):
    import cv2
    cv2.setNumThreads(1)
    sensor, seq = key.split("/")
    assert (sensor, seq) in ROLES[role][0], (role, key)
    idx = _index(role, sensor, seq)
    sids, pc = idx["scan_ids"], idx["poses_cache"]
    d = Path(out_dir) if out_dir else role_dir(role, sensor, seq)
    d.mkdir(parents=True, exist_ok=True)
    loader = make_loader(sensor, seq)
    track = Track(role, sensor, seq, loader) if SENSORS[sensor]["accumulate"] else None
    n = len(sids) if not limit else min(limit, len(sids))
    t0, done, nscan, npts = time.time(), 0, 0, 0
    for pos in range(shard, n, nshards):
        out = d / f"{pos:06d}.npy"
        if out.exists():
            continue
        img, used, m = riv_for(role, sensor, seq, pos, sids[pos], loader, track, pc[pos])
        tmp = d / f".{pos:06d}.{shard}.tmp.npy"
        np.save(tmp, img)
        os.replace(tmp, out)
        done += 1
        nscan += len(used)
        npts += m
        if done % 200 == 0:
            print(f"  [{role}/{key} {shard}/{nshards}] {done} "
                  f"({(time.time() - t0) / done:.2f} s/frame, {nscan / done:.1f} scans, "
                  f"{npts / done / 1e3:.0f}k pts per frame)", flush=True)
    dt = time.time() - t0
    print(f"OK gen {role}/{key} shard {shard}/{nshards}: {done} new in {dt:.0f} s"
          + (f" ({dt / done:.2f} s/frame, {nscan / done:.1f} scans/frame)" if done else ""),
          flush=True)


def cmd_status(roles, quiet=False) -> bool:
    ok = True
    for role in roles:
        n_all = n_have = 0
        for sensor, seq in ROLES[role][0]:
            d = role_dir(role, sensor, seq)
            if not (d / "index.npz").exists():
                print(f"  {role}/{sensor}/{seq}: not indexed")
                ok = False
                continue
            n = len(np.load(d / "index.npz")["scan_ids"])
            have = sum(1 for p in range(n) if (d / f"{p:06d}.npy").exists())
            n_all += n
            n_have += have
            if have != n:
                ok = False
                if not quiet:
                    print(f"  {role}/{sensor}/{seq}: {have}/{n} RIVs")
        print(f"{role}: {n_have}/{n_all} RIVs")
    print("RIV_COMPLETE" if ok else "RIV_INCOMPLETE")
    return ok


def seq_complete(role, sensor, seq) -> bool:
    d = role_dir(role, sensor, seq)
    if not (d / "index.npz").exists():
        return False
    n = len(np.load(d / "index.npz")["scan_ids"])
    return all((d / f"{p:06d}.npy").exists() for p in range(n))


# ------------------------------------------------------------ CPU selftest
SELFTEST = [("train", "kitti/05"), ("val", "kitti360/0004"), ("val", "mulran/Sejong01"),
            ("test_odom", "mcd/kth_day_06"), ("train", "helipr/Bridge02"),
            ("val", "helipr/Roundabout02"), ("test_odom", "helipr/Town01"),
            ("test_odom", "nclt/2012-01-08")]


def _cmp(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"bit_equal": bool(np.array_equal(a, b)), "max_abs_diff": float(d.max()),
            "n_px_diff_gt_1e-3": int((d > 1e-3).sum())}


def cmd_selftest(n_scans: int, out_json: Path | None, accum_scans: int):
    """CPU only.  Per case: (1) official functions vs adapter on the same
    points (normals, three images, final stack); (2) MulRan additionally
    end-to-end: official process_one_frame on the raw MulRan .bin vs our
    loader + adapter; (3) accumulation: only the keyframe and earlier scans,
    within 5 s; (4) resize-then-pad == pad-then-resize (evaluate.py order)."""
    import cv2
    gen = official_generator()
    rep = {"official_commit": OFFICIAL_COMMIT, "numpy": np.__version__, "cv2": cv2.__version__,
           "cases": []}
    ok = True
    for role, key in SELFTEST:
        sensor, seq = key.split("/")
        cp = cache_path(role, sensor, seq)
        with np.load(cp) as z:
            sids_all = z["scan_ids"].astype(np.int64)
        loader = make_loader(sensor, seq)
        acc = SENSORS[sensor]["accumulate"]
        track = Track(role, sensor, seq, loader) if acc else None
        m = 1 if acc else n_scans
        picks = [int(sids_all[int(i)]) for i in
                 np.linspace(len(sids_all) // 3, len(sids_all) - 1, m).round().astype(int)]
        for sid in picks:
            item = loader[sid]
            if acc and accum_scans:   # bound the reference cost on CPU
                win = track.window(sid)[:accum_scans]
                track_w = track.window
                track.window = lambda k, _w=win: _w
                xyz, refl, ring, used = frame_points(sensor, role, item["points"], track, sid)
                track.window = track_w
            else:
                xyz, refl, ring, used = frame_points(sensor, role, item["points"], track, sid)
            t0 = time.perf_counter()
            fast = fast_frame(xyz, refl, ring, sensor)
            t_fast = time.perf_counter() - t0
            nf = fast_normals(xyz, SENSORS[sensor]["k"])
            t0 = time.perf_counter()
            no = gen.compute_normals(xyz, k=SENSORS[sensor]["k"], max_neighbor_m=MAX_NEIGHBOR_M)
            ref = official_frame(xyz, refl, ring, sensor, normal=no)
            t_off = time.perf_counter() - t0
            case = {"role": role, "seq": key, "scan_id": sid, "n_points": int(len(xyz)),
                    "scans_used": [int(u) for u in used], "shape": list(fast.shape),
                    "normals": _cmp(nf, no), "frame": _cmp(fast, ref),
                    "channels": {c: _cmp(fast[..., i], ref[..., i])
                                 for i, c in enumerate(("reflectivity", "range", "normal"))},
                    "s_fast": round(t_fast, 2), "s_official": round(t_off, 2),
                    "nonzero_px_frac": float((fast[..., 1] > 0).mean())}
            good = case["frame"]["bit_equal"]
            if acc:
                tt = track.t
                full = track.window(sid)
                case["accumulation"] = {
                    "full_window_scans": len(full),
                    "all_past_or_self": bool(all(j <= sid for j in full)),
                    "max_age_s": float(tt[sid] - tt[min(full)]),
                    "pose_source": track.source}
                good &= case["accumulation"]["all_past_or_self"] and \
                    case["accumulation"]["max_age_s"] <= ACCUM_WINDOW_S
            if sensor == "mulran":   # end-to-end official path on the raw .bin
                f = str(loader.scan_files[sid])
                pts_o, refl_o, ring_o, _ = gen.read_bin_file(f, "mulran")
                same_in = bool(np.array_equal(pts_o, xyz) and np.array_equal(refl_o, refl)
                               and np.array_equal(ring_o, ring))
                task = (0, f, {"time": "0", "x": 0, "y": 0, "z": 0, "qx": 0, "qy": 0,
                               "qz": 0, "qw": 1}, None, IMAGE_W, 64, 64, "ring", 16.0, 16.0,
                        False, "mulran", 120.0, SENSORS["mulran"]["k"], MAX_NEIGHBOR_M)
                import tempfile
                with tempfile.TemporaryDirectory() as td:
                    task = task[:3] + (td,) + task[4:]
                    gen.process_one_frame(task)
                    e2e = np.load(Path(td) / "000000.npy")
                case["mulran_end_to_end"] = {"inputs_equal": same_in, **_cmp(fast, e2e)}
                good &= same_in and case["mulran_end_to_end"]["bit_equal"]
            if fast.shape[0] != TRAIN_H:   # resize/pad commute (column-wise resize)
                pad = lambda a: np.concatenate([a[:, -28:], a, a[:, :28]], axis=1)  # noqa: E731
                a = cv2.resize(pad(fast), (IMAGE_W + 56, TRAIN_H), interpolation=cv2.INTER_LINEAR)
                b = pad(to_train_height(fast))
                case["resize_then_pad_equal"] = bool(np.array_equal(a, b))
                good &= case["resize_then_pad_equal"]
            case["ok"] = bool(good)
            ok &= good
            rep["cases"].append(case)
            print(json.dumps({k: case[k] for k in ("role", "seq", "scan_id", "n_points", "ok",
                                                  "frame", "normals", "s_fast", "s_official")}),
                  flush=True)
    rep["all_ok"] = bool(ok)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rep, indent=1))
    print("RIV_SELFTEST_OK" if ok else "RIV_SELFTEST_FAILED", flush=True)
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index")
    p.add_argument("--role", choices=tuple(ROLES), nargs="+", default=list(ROLES))
    p = sub.add_parser("jobs")
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--role", choices=tuple(ROLES), nargs="+", default=["train", "val"])
    p.add_argument("--seq", default=None)
    p = sub.add_parser("gen")
    p.add_argument("--role", choices=tuple(ROLES), required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--out-dir", default=None, help=argparse.SUPPRESS)
    p = sub.add_parser("status")
    p.add_argument("--role", choices=tuple(ROLES), nargs="+", default=["train", "val"])
    p = sub.add_parser("selftest")
    p.add_argument("--n-scans", type=int, default=2)
    p.add_argument("--accum-scans", type=int, default=0,
                   help="cap the accumulated window in the selftest (0 = full 5 s)")
    p.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "index":
        for r in a.role:
            cmd_index(r)
    elif a.cmd == "jobs":
        cmd_jobs(a.chunk, a.role, a.seq)
    elif a.cmd == "gen":
        cmd_gen(a.role, a.seq, a.shard, a.nshards, a.limit, a.out_dir)
    elif a.cmd == "status":
        sys.exit(0 if cmd_status(a.role) else 1)
    elif a.cmd == "selftest":
        cmd_selftest(a.n_scans, a.out_json, a.accum_scans)


if __name__ == "__main__":
    main()
