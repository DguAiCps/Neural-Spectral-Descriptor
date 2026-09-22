#!/usr/bin/env python3
"""ImLPR RIV-image library + per-sequence cache builder (pd2026 roster).

Zero-shot ImLPR baseline, stage 1 of 2 (stage 2 = scripts/_imlpr_desc.py):
raw keyframe point clouds -> 3-channel Range-Intensity-... panoramas exactly
per the official ImLPR preprocessing (_external/ImLPR/preprocess/
generate_riv_images.py). The official renderers pointCloudToImage /
pointCloudToRangeImage / compute_v_indices are IMPORTED and reused verbatim
(never copied); only compute_normals is replaced by a vectorized equivalent
(compute_normals_fast) because the official per-point Python SVD loop is
~30-60 s/frame. Numerical equivalence is enforced by
tests/test_imlpr_riv_equiv.py.

Recipe = the authors' own cross-dataset ZERO-SHOT branch ('mulran' in their
code, used for their official MulRan zero-shot eval): intensity linearly
scaled to 0..255, range linearly scaled [0, 120 m] -> [0, 255] with
min-pooling collisions and empty=0, reflectivity/normal channels max-pooled,
k=4 / max_neighbor_m=50 normals (the compute_normals signature defaults).
The HeLiPR-Ouster log-reflectivity path (scale_log_to_255 on OS2-128 uint16
reflectivity) is NOT used: none of our streams are OS2-128.

Sensor v-projections:
  mulran   official recipe verbatim: ring = idx % 64 (scan-order XYZI from
           MulRanLoader is ring-major 64-beam), v_mode='ring', H0=64
  kitti / kitti360  HDL-64E  angle mode H0=64, fov +2.0 / -24.8
  nclt     HDL-32E  angle mode H0=32, fov +10.67 / -30.67
  helipr   VLP-16   angle mode H0=16, fov +15.0 / -15.0
  mcd      OS1-64(gen2) angle mode H0=64, fov +22.5 / -22.5

Intensity pre-scale to 0..255 (per sensor family, BEFORE the image):
  kitti/kitti360  loader floats 0..1 -> x255
  nclt            raw is uint8 0..255 but NCLTLoader returns intensity/255
                  -> x255 restores the raw 0..255 values
  helipr          VLP-16 intensity already 0..255 -> clip
  mulran/mcd      raw float reflectivity -> official read_bin_file 'mulran'
                  scaling: clamp at 4096, /4096*255

FROZEN cache interface (consumed by scripts/_imlpr_desc.py):
  _external/imlpr_data_pd2026/{sensor}_{seq}/%06d.npy   (H0, 1022, 3) float32
      0..255, one per roster keyframe, in operating-cache scan_id order
  _external/imlpr_data_pd2026/{sensor}_{seq}/meta.json  sensor params,
      n_frames, scan_ids, train_overlap

Usage (inside docker ring_baseline, per-frame resumable):
    python3 scripts/_imlpr_riv_lib.py --seqs kitti/05 [mulran/DCC03 ...] \
        [--workers 8] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
# Reuse the sibling's loader factory (kitti/nclt/helipr incl. double-nested
# layout/mulran/kitti360/mcd routing) and its DATA_ROOT/MANIFEST conventions.
from _pdb_bevplace_eval import DATA_ROOT, MANIFEST, make_loader  # noqa: E402

_IMLPR_PRE = str(REPO / "_external" / "ImLPR" / "preprocess")
if _IMLPR_PRE not in sys.path:
    sys.path.insert(0, _IMLPR_PRE)
import generate_riv_images as riv_official  # noqa: E402  (official code, reused)

OUT_ROOT = REPO / "_external" / "imlpr_data_pd2026"

IMAGE_WIDTH = 1022
MAX_RANGE_M = 120.0      # official mulran zero-shot branch
K_NEIGHBORS = 4          # official compute_normals signature default
MAX_NEIGHBOR_M = 50.0    # official compute_normals signature default

# sensor -> v-projection + intensity pre-scaling (see module docstring)
SENSORS = {
    "kitti":    dict(H0=64, v_mode="angle", fov_up_deg=2.0, fov_down_deg=24.8,
                     intensity="x255"),
    "kitti360": dict(H0=64, v_mode="angle", fov_up_deg=2.0, fov_down_deg=24.8,
                     intensity="x255"),
    "nclt":     dict(H0=32, v_mode="angle", fov_up_deg=10.67, fov_down_deg=30.67,
                     intensity="x255"),
    "helipr":   dict(H0=16, v_mode="angle", fov_up_deg=15.0, fov_down_deg=15.0,
                     intensity="clip255"),
    "mulran":   dict(H0=64, v_mode="ring", fov_up_deg=16.0, fov_down_deg=16.0,
                     intensity="mulran4096"),
    "mcd":      dict(H0=64, v_mode="angle", fov_up_deg=22.5, fov_down_deg=22.5,
                     intensity="mulran4096"),
}

# ImLPR (CoRL'25) was trained on HeLiPR DCC04-06 / KAIST04-06 / Riverside04-06,
# which ARE pd2026 TEST sequences: results there are train-contaminated and
# every output meta carries this flag (aed302d lower-bound convention, same as
# scripts/_pdb_ringsharp_eval.py).
TRAIN_OVERLAP = {("helipr", s) for s in
                 ["DCC04", "DCC05", "DCC06", "KAIST04", "KAIST05", "KAIST06",
                  "Riverside04", "Riverside05", "Riverside06"]}


def intensity_to_255(intensity: np.ndarray, sensor: str) -> np.ndarray:
    """Per-sensor-family pre-scale of loader intensities to 0..255 float32."""
    mode = SENSORS[sensor]["intensity"]
    i = np.asarray(intensity, dtype=np.float32)
    if mode == "x255":
        return np.clip(i * 255.0, 0.0, 255.0)
    if mode == "clip255":
        return np.clip(i, 0.0, 255.0)
    if mode == "mulran4096":
        # official read_bin_file 'mulran' branch: clamp 4096, /4096*255
        return np.minimum(i, 4096.0) / 4096.0 * 255.0
    raise ValueError(mode)


def compute_normals_fast(points: np.ndarray, k: int = K_NEIGHBORS,
                         max_neighbor_m: float = MAX_NEIGHBOR_M) -> np.ndarray:
    """Vectorized equivalent of the official compute_normals.

    Same kNN (sklearn kd_tree, k+1 with self, self excluded), same float32
    centering; the per-point SVD is replaced by batched 3x3 scatter-matrix
    eigvalsh (singular values s_i of the centered (k,3) neighborhood satisfy
    s_i^2 = eigvals of X^T X), same ratio max_s / (min_s + 1e-5), same
    far-neighbor rule (max distance > max_neighbor_m -> 0.7), same
    scale_log_to_255. Equivalence: tests/test_imlpr_riv_equiv.py.
    """
    from sklearn.neighbors import NearestNeighbors

    points = np.asarray(points, dtype=np.float32)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(points)
    distances, indices = nbrs.kneighbors(points)

    neigh = points[indices[:, 1:]]                       # (N, k, 3) float32
    neigh = neigh - neigh.mean(axis=1, keepdims=True)    # float32, as official
    neigh64 = neigh.astype(np.float64)
    scatter = np.einsum("nki,nkj->nij", neigh64, neigh64)  # (N, 3, 3)
    eig = np.linalg.eigvalsh(scatter)                    # ascending eigvals
    eig = np.clip(eig, 0.0, None)
    s_min = np.sqrt(eig[:, 0])
    s_max = np.sqrt(eig[:, -1])
    ratio = s_max / (s_min + 1e-5)
    ratio[distances[:, 1:].max(axis=1) > max_neighbor_m] = 0.7
    return riv_official.scale_log_to_255(ratio)          # (N,) float32 0..255


def riv_for_points(points_xyzi: np.ndarray, sensor: str) -> np.ndarray:
    """(N,4) XYZI -> (H0, 1022, 3) float32 0..255 [reflectivity, range, normal].

    Rendering goes through the OFFICIAL pointCloudToImage /
    pointCloudToRangeImage (imported, unmodified); no vertical resize here
    (image_height_final=None keeps H0-native rows; the 126-row resize is the
    descriptor stage's job, as in the official mulran eval).
    """
    cfg = SENSORS[sensor]
    pts = np.ascontiguousarray(points_xyzi[:, :3], dtype=np.float32)
    inten = intensity_to_255(points_xyzi[:, 3], sensor)
    if cfg["v_mode"] == "ring":
        ring = (np.arange(pts.shape[0]) % 64).astype(np.int32)  # official mulran
    else:
        ring = np.zeros(pts.shape[0], dtype=np.int32)           # unused in angle mode

    normals = compute_normals_fast(pts)
    kw = dict(image_width=IMAGE_WIDTH, image_height_initial=cfg["H0"],
              image_height_final=None, v_mode=cfg["v_mode"],
              fov_up_deg=cfg["fov_up_deg"], fov_down_deg=cfg["fov_down_deg"],
              invert_vertical=False)
    img_reflect = riv_official.pointCloudToImage(pts, inten, ring, **kw)
    img_range = riv_official.pointCloudToRangeImage(
        pts, ring, lidar_type="mulran", max_range_m=MAX_RANGE_M, **kw)
    img_normal = riv_official.pointCloudToImage(pts, normals, ring, **kw)
    return np.stack([img_reflect, img_range, img_normal], axis=2).astype(np.float32)


# ----------------------------
# Roster / cache helpers (shared with scripts/_imlpr_desc.py)
# ----------------------------

def load_roster() -> list[dict]:
    return json.loads(Path(MANIFEST).read_text())["sequences"]


def resolve_seqs(seq_args: list[str] | None) -> list[dict]:
    roster = load_roster()
    if not seq_args:
        return roster
    wanted = list(dict.fromkeys(seq_args))
    have = {f"{e['sensor']}/{e['sequence']}": e for e in roster}
    missing = [s for s in wanted if s not in have]
    if missing:
        raise SystemExit(f"not in the frozen 42-seq roster: {missing}")
    return [have[s] for s in wanted]


def load_cache_scan_ids(entry: dict) -> np.ndarray:
    cache = REPO / entry["expected"]["operating_cache"]
    with np.load(cache, allow_pickle=False) as c:
        return c["scan_ids"].astype(np.int64)


def seq_dir(sensor: str, seq: str) -> Path:
    return OUT_ROOT / f"{sensor}_{seq}"


# ----------------------------
# Per-sequence cache builder (frame-level workers, per-frame resumable)
# ----------------------------

_worker_loader = None
_worker_key = None


def _worker_init():
    import cv2
    cv2.setNumThreads(0)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _build_frame(sensor: str, seq: str, scan_id: int, out_path: str) -> str:
    global _worker_loader, _worker_key
    if _worker_key != (sensor, seq):
        _worker_loader = make_loader(sensor, seq)
        _worker_key = (sensor, seq)
    pts = _worker_loader[int(scan_id)]["points"]
    img = riv_for_points(pts, sensor)
    tmp = out_path + ".tmp.npy"
    np.save(tmp, img)
    os.replace(tmp, out_path)
    return out_path


def build_sequence(entry: dict, workers: int, overwrite: bool) -> None:
    sensor, seq = entry["sensor"], entry["sequence"]
    scan_ids = load_cache_scan_ids(entry)
    out_dir = seq_dir(sensor, seq)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for pos, sid in enumerate(scan_ids):
        out_path = out_dir / f"{pos:06d}.npy"
        if overwrite or not out_path.is_file():
            todo.append((pos, int(sid), str(out_path)))
    print(f"[{sensor}/{seq}] {len(scan_ids)} keyframes, {len(todo)} to build "
          f"-> {out_dir}", flush=True)

    if todo:
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_worker_init) as ex:
            futs = [ex.submit(_build_frame, sensor, seq, sid, path)
                    for _, sid, path in todo]
            for f in as_completed(futs):
                f.result()  # re-raise worker errors
                done += 1
                if done % 200 == 0 or done == len(todo):
                    dt = time.time() - t0
                    print(f"  [{sensor}/{seq}] {done}/{len(todo)} "
                          f"({dt / done:.2f} s/frame eff.)", flush=True)

    n_ok = sum((out_dir / f"{pos:06d}.npy").is_file()
               for pos in range(len(scan_ids)))
    if n_ok != len(scan_ids):
        raise RuntimeError(f"{sensor}/{seq}: {n_ok}/{len(scan_ids)} frames present")

    cfg = SENSORS[sensor]
    meta = {
        "sensor": sensor,
        "sequence": seq,
        "n_frames": int(len(scan_ids)),
        "scan_ids": [int(s) for s in scan_ids],
        "operating_cache": entry["expected"]["operating_cache"],
        "params": {
            "image_width": IMAGE_WIDTH,
            "image_height_initial": cfg["H0"],
            "image_height_final": None,
            "v_mode": cfg["v_mode"],
            "fov_up_deg": cfg["fov_up_deg"],
            "fov_down_deg": cfg["fov_down_deg"],
            "invert_vertical": False,
            "max_range_m": MAX_RANGE_M,
            "k_neighbors": K_NEIGHBORS,
            "max_neighbor_m": MAX_NEIGHBOR_M,
            "intensity_scaling": cfg["intensity"],
            "channels": ["reflectivity", "range", "normal_ratio"],
            "recipe": ("official ImLPR generate_riv_images.py renderers, "
                       "authors' mulran zero-shot branch scaling; "
                       "compute_normals vectorized (equivalence-tested)"),
        },
        "train_overlap": (sensor, seq) in TRAIN_OVERLAP,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[{sensor}/{seq}] complete -> {meta_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="subset like kitti/05 mulran/DCC03 (default: all 42)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for entry in resolve_seqs(args.seqs):
        build_sequence(entry, workers=args.workers, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
