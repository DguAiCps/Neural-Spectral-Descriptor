#!/usr/bin/env python3
"""Official RING / RING++ pipeline pieces for our sensor-frame keyframes.

Which code is "official" (2026-10-11 decision, see
logs/baseline_official_20261011/ring_family.md):
  * RING (IROS'22) and RING++ (T-RO'23): the authors' repository
    lus6-Jenny/RING, clone _external/RING @ c79c585 (no local changes).
    Its README calls it the official implementation of both papers, and it
    is the only code with per-dataset settings for KITTI, MulRan and NCLT.
    RINGSharp (_external/RINGSharp) re-implements RING / RING++ inside
    glnet as baselines for the RING# paper (NCLT / Oxford configs only);
    its RING++ feature BEV differs (features on raw points, other cell
    aggregation), so it is NOT used for RING / RING++ here.
  * Every step below calls the RING repo unchanged, in the order of its own
    evaluator (evaluation/evaluate.py generate_representations, GPU path):
      pc      = <Dataset>PointCloudLoader.read_pc(...)   (zero-point removal,
                ground removal z > ground_plane_level, np.unique)
      pc_norm = <Dataset>PointCloudLoader.normalize_pc(pc)  (|x|,|y| < 70 m,
                ground_plane_level < z < zbound, scale to [-1, 1])
      bev     = utils.core.generate_bev_occ(pc_norm)        RING   (1x120x120)
                utils.core.generate_bev_feats(pc_norm)      RING++ (6x120x120)
      tiring  = utils.core.generate_RING(bev)[1]   (angle grid: paper value,
                see generate_ring_paper; overrides utils/core.py:153)
      score   = utils.core.batch_circorr  (dist = 1 - normalized circular
                correlation; exhaustive over the candidates)
    The only thing replaced is the class's raw-file reader
    (load_pc_kitti / load_pc_mulran / load_pc_nclt): it is swapped for the
    points of our src/data loader for the same scan (official_points), which
    return the same sensor-frame coordinates (KITTI / MulRan: the raw bin;
    NCLT: the loader applies the same z flip as load_pc_nclt, [x, y, -z]).

DATASET_TABLE is the single place that maps our six sensors onto the
official per-dataset settings (see its comment for the transfer rule used
for KITTI-360, HeLiPR and MCD, which have no official setting).

RING repo imports: the repo's `utils` and `datasets` folders are namespace
packages (no __init__.py) and lose to our src/utils and to the HuggingFace
`datasets` package; import_ring() pre-registers them in sys.modules so the
repo's own absolute imports (import utils.config as cfg, ...) resolve to it.
"""
from __future__ import annotations
import glob
import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RING_DIR = REPO / "_external" / "RING"
RING_COMMIT = "c79c585"
DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", "/workspace/data"))
CACHE_ROOT = REPO / "_external" / "ring_official_cache"
OUT_ROOT = REPO / "results" / "pdv2_ring_official"
CACHE_DIR = {"odom": REPO / "data/preprocessed_v2_odom",
             "gt": REPO / "data/preprocessed_v2"}

# ---------------------------------------------------------------------------
# Single dataset mapping table (frozen 2026-10-11).
#
# Official settings (_external/RING/datasets/*Dataset.py, PointCloudLoader
# defaults + normalize_pc zbound), sensor frame, z up:
#   KITTI  ground_plane_level -1.0  zbound 10.0  z/zbound            (:47, :70)
#   MulRan ground_plane_level -0.9  zbound 19.0  (z-gpl)/(zbound-gpl) (:37, :61)
#   NCLT   ground_plane_level  1.0  zbound 20.0  (z-gpl)/(zbound-gpl) (:51, :74)
#          (after load_pc_nclt's z flip; our NCLT loader flips the same way)
# The official code does not derive these from a formula; they are
# per-dataset constants.  For the two car settings the lower bound sits
# about 0.75 m above the ground (KITTI 1.748 - 1.0 = 0.75 m, MulRan
# 1.674 - 0.9 = 0.77 m, heights measured below).
#
# Transfer rule for datasets without an official setting: take the closest
# official dataset and translate its band by the measured difference in
# sensor height, so the band covers the same heights above the ground:
#   gpl_new = gpl_ref + (h_ref - h_new),  zbound_new = zbound_ref + (h_ref - h_new)
# (normalization formula and x/y bounds from the reference class; the band
# width zbound - gpl is unchanged).
# h = mean of the 'lowest' and 'mode' per-dataset medians in
# results/pdv2_ring_official/sensor_height.json
# (scripts/_ring_official_sensor_height.py, 30 scans per sequence, split-v2
# train/val sequences only; MCD has none, its rows are test scans, geometry
# only).  The same estimator runs on both datasets, so its bias cancels in
# h_ref - h_new.
#   kitti    h 1.748 (lowest 1.784 / mode 1.713; published KITTI 1.73)
#   kitti360 h 1.712 (1.735 / 1.689) -> dh = +0.036 m (estimators +0.024 /
#            +0.048) vs KITTI: same car and HDL-64E and |dh| < 0.05 m, so
#            the KITTI constants are kept unchanged
#   mulran   h 1.674 (1.688 / 1.659)
#   helipr   h 1.599 (1.586 / 1.613) -> MulRan band shifted by +0.075 m
#            (car; same Daejeon routes as MulRan; estimators +0.047 / +0.102)
#   mcd      h 1.793 (1.948 / 1.638) -> MulRan band shifted by -0.119 m
#            (Ouster OS1-64 like MulRan; handheld; estimators -0.260 /
#            +0.021, the least certain row)
#   nclt     not measured: the flipped HDL-32E sees the ground with almost
#            no beams; not needed, the official NCLT constants apply to the
#            identical frame
DATASET_TABLE = {
    "kitti":    {"official_class": "KITTI", "ground_plane_level": -1.0,
                 "zbound": 10.0, "basis": "official (KITTIDataset.py:47,70)"},
    "kitti360": {"official_class": "KITTI", "ground_plane_level": -1.0,
                 "zbound": 10.0,
                 "basis": "KITTI constants (same car/sensor; measured dh +0.036 m)"},
    "mulran":   {"official_class": "MulRan", "ground_plane_level": -0.9,
                 "zbound": 19.0, "basis": "official (MulRanDataset.py:37,61)"},
    "nclt":     {"official_class": "NCLT", "ground_plane_level": 1.0,
                 "zbound": 20.0,
                 "basis": "official (NCLTDataset.py:51,74); same z flip as load_pc_nclt"},
    "helipr":   {"official_class": "MulRan", "ground_plane_level": -0.825,
                 "zbound": 19.075,
                 "basis": "MulRan band shifted by measured dh = +0.075 m"},
    "mcd":      {"official_class": "MulRan", "ground_plane_level": -1.019,
                 "zbound": 18.881,
                 "basis": "MulRan band shifted by measured dh = -0.119 m"},
}


def ringsharp_bounds(sensor: str):
    """RING#-L BEV bounds for glnet generate_bev (user decision 2026-10-11:
    the same per-dataset height band as RING / RING++): x, y +-70 m as the
    official RING#-L configs, z from ground_plane_level to zbound of the
    DATASET_TABLE row (the 160x160x20 grid itself is unchanged)."""
    row = DATASET_TABLE[sensor]
    return (-70.0, 70.0, -70.0, 70.0,
            float(row["ground_plane_level"]), float(row["zbound"]))


# raw-file reader of each official class (swapped for our loader's points)
_READER = {"KITTI": ("datasets.KITTIDataset", "KITTIPointCloudLoader", "load_pc_kitti"),
           "MulRan": ("datasets.MulRanDataset", "MulRanPointCloudLoader", "load_pc_mulran"),
           "NCLT": ("datasets.NCLTDataset", "NCLTPointCloudLoader", "load_pc_nclt")}

_RING = {}


def import_ring():
    """Import the RING repo (utils.core, utils.config, dataset classes)."""
    if _RING:
        return _RING
    for name in ("utils", "datasets"):
        cur = sys.modules.get(name)
        if cur is not None and list(getattr(cur, "__path__", [])) == [str(RING_DIR / name)]:
            continue
        assert cur is None, (
            f"module '{name}' already imported from {getattr(cur, '__file__', cur)}; "
            "import _ring_official_lib before anything that imports it")
        m = types.ModuleType(name)
        m.__path__ = [str(RING_DIR / name)]
        m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        m.__spec__.submodule_search_locations = [str(RING_DIR / name)]
        sys.modules[name] = m
    import utils.config as cfg  # noqa: E402  (RING repo)
    _RING["cfg"] = cfg
    _RING["modules"] = {k: importlib.import_module(v[0]) for k, v in _READER.items()}
    _RING["classes"] = {k: getattr(_RING["modules"][k], v[1]) for k, v in _READER.items()}
    return _RING


def import_ring_core():
    """utils.core needs voxelocc / voxelfeat / torch_radon (container)."""
    import_ring()
    if "core" not in _RING:
        import utils.core as core  # noqa: E402  (RING repo)
        _RING["core"] = core
    return _RING["core"]


# Paper setting (project rule 2026-10-11: paper value wins over code, code
# line disclosed).  RING (IROS'22 Sec. III: theta in [0, 2pi), k = 120) and
# RING++ (T-RO'23 Sec. VI-E: rotation resolution 360/120 = 3 deg/pixel)
# sample 120 projection angles over [0, 2pi).  The repo's GPU path
# (utils/core.py:153 generate_RING) uses np.linspace(0, 2*np.pi, 120), i.e.
# 360/119 deg with both 0 and 360 deg; its own CPU path (core.py:164) and
# its yaw read-out (evaluation/evaluate.py, ang_res = 2*pi/num_ring) use the
# paper grid.
def ring_angles_paper(n: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, n, endpoint=False).astype(np.float32)


def generate_ring_paper(pc_bev):
    """utils/core.py generate_RING with the paper angle grid (only change)."""
    import torch
    from torch_radon import ParallelBeam
    core = import_ring_core()
    cfg = _RING["cfg"]
    radon = ParallelBeam(cfg.num_sector, torch.FloatTensor(ring_angles_paper(cfg.num_ring)))
    pc_RING = radon.forward(pc_bev)
    pc_TIRING, _ = core.forward_row_fft(pc_RING)
    return pc_RING, pc_TIRING


def official_points(sensor: str, pts: np.ndarray):
    """Our loader points -> (pc, pc_norm) through the official class.

    pc      = <Class>(ground_plane_level=gpl).read_pc(.)  with the class's raw
              reader returning `pts` (all other read_pc steps unchanged)
    pc_norm = the same object's normalize_pc(pc[:, :3], zbound=zbound)
    """
    R = import_ring()
    row = DATASET_TABLE[sensor]
    cls_name = row["official_class"]
    mod = R["modules"][cls_name]
    reader = _READER[cls_name][2]
    loader = R["classes"][cls_name](ground_plane_level=row["ground_plane_level"])
    xyz = np.asarray(pts)[:, :3]
    orig = getattr(mod, reader)
    setattr(mod, reader, lambda _path: xyz)
    try:
        pc = loader.read_pc("<points from src/data loader>")
    finally:
        setattr(mod, reader, orig)
    pc = pc[:, :3]                               # as evaluate.generate_representations
    pc_norm = loader.normalize_pc(pc, zbound=row["zbound"])
    return pc, pc_norm


def official_points_from_file(sensor: str, path: str):
    """Reference: the official class reading the raw file itself
    (KITTI / MulRan / NCLT only) — used by the equality check."""
    R = import_ring()
    row = DATASET_TABLE[sensor]
    loader = R["classes"][row["official_class"]](ground_plane_level=row["ground_plane_level"])
    pc = loader.read_pc(path)[:, :3]
    return pc, loader.normalize_pc(pc, zbound=row["zbound"])


def occupancy_numpy(pc_norm: np.ndarray, n: int = 120) -> np.ndarray:
    """CHECK ONLY: numpy transcription of the voxelocc CUDA kernel
    (utils/generate_bev_occ_cython/src/kernel.cu point2gridmap + manager.cu
    retreive with max_length = max_height = 1, num_height = 1) so the BEV
    effect of point differences can be measured without a GPU.  Production
    BEVs come from utils.core.generate_bev_occ."""
    p = np.asarray(pc_norm, dtype=np.float32).copy()
    for c in range(3):
        v = p[:, c]
        v[v == 0.0] = 0.0001
        v[v > 1.0] = 0.9999
        v[v < -1.0] = -0.9999
    gap = np.float32(2.0) / np.float32(n)
    ix = np.floor((p[:, 0] + np.float32(1.0)) / gap).astype(np.int64)
    iy = np.floor((p[:, 1] + np.float32(1.0)) / gap).astype(np.int64)
    bev = np.zeros((n, n), dtype=np.uint8)
    bev[ix, iy] = 1
    return bev


def make_loader(sensor: str, seq: str):
    """Same constructors as scripts/_pd_build_test_caches._make_loader."""
    sys.path.insert(0, str(REPO / "src"))
    if sensor == "kitti":
        from data.kitti_loader import KITTILoader
        return KITTILoader(str(DATA_ROOT / "kitti/dataset"), seq, lazy_load=True)
    if sensor == "nclt":
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(str(DATA_ROOT / "nclt"), seq, lazy_load=True)
    if sensor == "helipr":
        from data.helipr_loader import HeLiPRLoader
        p = DATA_ROOT / "helipr" / seq
        if not (p / "LiDAR").is_dir():
            p = p / seq  # legacy double-nested layout
        return HeLiPRLoader(str(p), lazy_load=True)
    if sensor == "mulran":
        from data.mulran_loader import MulRanLoader
        return MulRanLoader(str(DATA_ROOT / "mulran"), seq, lazy_load=True)
    if sensor == "kitti360":
        from data.kitti360_loader import Kitti360Loader
        return Kitti360Loader(str(DATA_ROOT / "kitti360"), seq, lazy_load=True)
    if sensor == "mcd":
        from data.mcd_loader import MCDLoader
        return MCDLoader(str(DATA_ROOT / "mcd_extracted"), seq, lazy_load=True)
    raise ValueError(sensor)


def raw_scan_path(loader, sensor: str, idx: int) -> str:
    """Raw file behind loader[idx] for the sensors with an official reader."""
    if sensor in ("kitti", "nclt"):
        return str(loader.frame_files[idx])
    if sensor == "mulran":
        return str(loader.scan_files[idx])
    raise ValueError(f"no official raw reader for {sensor}")


def split_cache(cond: str, role: str, sensor: str, seq: str) -> Path:
    """Split-v2 keyframe cache (odom: KISS-ICP keyframes + poses_gt)."""
    hits = sorted(glob.glob(str(CACHE_DIR[cond] / f"cache_*_{sensor}_{role}_{seq}.npz")))
    assert len(hits) == 1, (cond, role, sensor, seq, hits)
    return Path(hits[0])


def load_keyframes(cache: Path):
    """scan_ids, scoring poses (poses_gt if present, else poses)."""
    with np.load(cache, allow_pickle=True) as c:
        scan_ids = c["scan_ids"].astype(np.int64)
        poses = (c["poses_gt"] if "poses_gt" in c.files else c["poses"]).astype(np.float64)
    return scan_ids, poses


def min_past_dists(positions: np.ndarray, gap: int) -> np.ndarray:
    """min 3D distance from t to any frame <= t - gap (per-query prefix scan,
    as scripts/_causal_pr_dump.py)."""
    n = len(positions)
    mp = np.full(n, np.nan)
    for t in range(gap, n):
        mp[t] = np.linalg.norm(positions[:t - gap + 1] - positions[t], axis=1).min()
    return mp
