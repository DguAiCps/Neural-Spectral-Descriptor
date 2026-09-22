#!/usr/bin/env python3
"""Prepare ImLPR training data for a CLEAN pd2026 place-disjoint retrain.

Counterpart of scripts/_pdb_ringsharp_prepare.py for the OFFICIAL ImLPR
trainer (_external/ImLPR/training/train.py). Two resumable stages:

Stage riv  — RIV .npy directories under _external/imlpr_data_pd2026/:
  training/{sensor}_{seq}/%06d.npy    16 pd TRAIN sequences
      (configs/training_place_disjoint.yaml), cache keyframes SUBSAMPLED with
      the official 3.0 m greedy consecutive-distance rule
      (generate_riv_images.py:select_frame_indices, dist_threshold=3.0),
      rendered by the frozen _imlpr_riv_lib.riv_for_points recipe and resized
      to 126 rows (their trainer stacks batches, so all sensors must share
      H=126; INTER_LINEAR = the official eval-time resize our frozen zero-shot
      pipeline uses, chosen over the preprocess-side INTER_AREA because
      NCLT/HeLiPR are UPscaled 32/16 -> 126 where INTER_AREA degenerates).
      Values stay float32 0..255; TrainingDataset divides by 255 itself.
  validation/{sensor}_{seq}/%06d.npy  4 pd VAL sequences (kitti/09,
      nclt/2012-01-08, helipr/Roundabout03, mulran/Sejong03) at FULL cache
      roster, H0-native rows — the same frozen interface as the zero-shot RIV
      cache, so scripts/_imlpr_desc.py:prep_frame (pad 28 -> resize 126)
      applies unchanged for our post-training checkpoint selection
      (scripts/_imlpr_pd_select.py). NEVER consumed by their trainer.
  Each dir gets meta.json (scan_ids kept + cache indices + params).

Stage pickle — _external/imlpr_data_pd2026/training_pd.pickle in EXACTLY the
  per-anchor dict format of _external/ImLPR/preprocess/
  generate_training_tuples_baseline_img.py:construct_query_dict, the format
  datasets/base_datasets.py:TrainingDataset reads:
      {id, timestamp, rel_scan_filepath, positives, non_negatives, position}
  positives = same-group anchors within 10 m (XY, self excluded, sorted list),
  non_negatives = within 30 m (self INCLUDED, sorted list — their convention),
  position = [x, y, z, qx, qy, qz, qw] float32, timestamp = int(%06d stem).
  KD-trees are built PER GROUP (precedent _pdb_ringsharp_prepare.py: the 5
  NCLT train sessions form ONE co-registered campus group with cross-session
  positives; every other sequence is its own group — MulRan/HeLiPR cache
  poses are per-sequence local frames, cross-sequence pairs are meaningless).
  KITTI cache poses are cam0-convention and are conjugated into a z-up body
  frame first (same transform + rationale as _pdb_ringsharp_prepare.py);
  without it, cam0 "y" is height and the XY radius queries would be wrong.

Stage config — writes/validates _external/imlpr_data_pd2026/config_pd.txt
  (the training config for their trainer; see CONFIG_TEXT for the key-by-key
  rationale). Validation runs on every invocation: a hand-edited config is
  loudly flagged, never silently overwritten.

Checkpoint-selection rule (pre-declared, RING# precedent): highest
pd-validation sequence-macro 5 m Recall@1 over the saved epochs
(save_freq=10 -> epochs 10,20,...,100); earlier epoch on ties. Implemented in
scripts/_imlpr_pd_select.py; only after its selection JSON is written may the
selected checkpoint touch the test roster.

Usage (inside docker ring_baseline; CPU only, per-frame resumable):
    python3 scripts/_imlpr_pd_prepare.py --stage riv [--split train|val|both]
        [--seqs kitti/07 ...] [--workers 8] [--overwrite]
    python3 scripts/_imlpr_pd_prepare.py --stage pickle
    python3 scripts/_imlpr_pd_prepare.py --stage config
Smoke (single sequence + schema-checkable mini pickle):
    python3 scripts/_imlpr_pd_prepare.py --stage riv --split train --seqs kitti/07
    python3 scripts/_imlpr_pd_prepare.py --stage pickle --seqs kitti/07 \
        --pickle-out training_pd_smoke.pickle
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

OUT_ROOT = REPO / "_external" / "imlpr_data_pd2026"
TRAIN_ROOT = OUT_ROOT / "training"
VAL_ROOT = OUT_ROOT / "validation"
CACHE_DIR = REPO / "data" / "preprocessed_pd2026"
SPLIT_YAML = REPO / "configs" / "training_place_disjoint.yaml"
CONFIG_PATH = OUT_ROOT / "config_pd.txt"
PICKLE_DEFAULT = "training_pd.pickle"

CACHE_KEYS = {"kitti": "fecdaef7", "nclt": "aff53e59",
              "helipr": "ad22d735", "mulran": "50223b37"}

VAL_SEQS = [("kitti", "09"), ("nclt", "2012-01-08"),
            ("helipr", "Roundabout03"), ("mulran", "Sejong03")]

DIST_THRESHOLD_M = 3.0   # official ImLPR training subsample (generate_riv_images.py)
TRAIN_H = 126            # their trainer needs one shared image height (DINOv2 rows)
POS_M, NONNEG_M = 10.0, 30.0  # official construct_query_dict radii

# Training config for _external/ImLPR/training/train.py. Kept as a canonical
# constant so --stage config can write it and every invocation validates the
# on-disk copy against it. Key decisions (vs their config_helipr.txt):
#   load_weights EMPTY  -> params.load_weights = "" -> os.path.exists("") is
#       False -> trainer prints "previous weights not found" and keeps the
#       constructor init = pretrained DINOv2 backbone (torch.hub) + fresh
#       adapters/SALAD. The released ImLPR_default.pth is HeLiPR-Daejeon-
#       trained (9 pd TEST sequences) and would contaminate the clean retrain.
#       NOTE the key must be PRESENT-but-empty: if absent, params.load_weights
#       is None and os.path.exists(None) raises TypeError in do_train.
#   save_freq = 10      -> misc/utils.py TrainingParams reads [TRAIN] save_freq
#       (default 0 = no periodic saves); trainer writes
#       _external/ImLPR/weights/{model_name}_{epoch}.pth every 10 epochs plus
#       {model_name}_final.pth (identical to _100 when save_freq | epochs).
#   sensor_range = 120  -> our RIV range channel is scaled [0,120 m] -> [0,255]
#       (_imlpr_riv_lib MAX_RANGE_M, the authors' own mulran zero-shot branch;
#       their generate_riv_images.py comment: "200 for helipr, 120 for
#       mulran"). The collate_fn rescales channel 1 by sensor_range to meters
#       for the ICP/patch-positive machinery; 200 would scale every
#       backprojected cloud by 1.67x against the metric GT translations.
#   uFoV/dFoV kept at the helipr values: the pd train set mixes 4 sensors with
#       different FoVs and the config has a single global pair; positive pairs
#       are always same-sensor and yaw-dominant, so the elevation-grid error is
#       second-order for the patch-positive ICP (disclosed limitation).
#   Everything else mirrors config_helipr.txt verbatim (batch 2048/split 64,
#       lr 5e-5, epochs 100, TruncatedSmoothAP params, image 1022x126).
CONFIG_TEXT = """\
# ImLPR pd2026 clean-retrain config (written by scripts/_imlpr_pd_prepare.py
# --stage config; edit CONFIG_TEXT there, not this file).
# Semantics = _external/ImLPR/config/config_helipr.txt with:
#   train_file/dataset_folder -> the pd2026 staging dir (relative path:
#       training MUST run with CWD = _external/ImLPR, which the launcher
#       scripts/_imlpr_pd_train.py guarantees)
#   load_weights EMPTY -> from DINOv2 init (clean retrain; the released
#       ImLPR_default.pth is HeLiPR-Daejeon-trained = pd TEST contamination)
#   save_freq = 10 -> weights/ImLPR_pd2026_{10..100}.pth + _final.pth
#   sensor_range = 120 -> matches the [0,120 m] range scaling of our RIVs
[DEFAULT]
model_name = ImLPR_pd2026
dataset_folder = ../imlpr_data_pd2026/
load_weights =

[TRAIN]
save_freq = 10
num_workers = 0
batch_size = 2048
batch_split_size = 32
val_batch_size = 1

lr = 5e-5
epochs = 100

loss = TruncatedSmoothAP
tau1 = 0.01
positives_per_query = 4

num_positive_pairs = 216
num_pos = 192
num_hn_samples = 64
temperature = 0.20
vdist = 3
hdist = 20
circular_horizontal = True

image_H = 1022
image_W = 126
uFoV = 11.5
dFoV = -10.1
sensor_range = 120

similarity = euclidean

train_file = training_pd.pickle
"""

# KITTI caches store cam0-convention poses; conjugate into a z-up body frame
# (same transform + rationale as scripts/_pdb_ringsharp_prepare.py).
_R_BC = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
_M_BC = np.eye(4)
_M_BC[:3, :3] = _R_BC


def fix_pose_frame(sensor: str, poses: np.ndarray) -> np.ndarray:
    if sensor != "kitti":
        return poses
    return np.einsum("ij,njk,kl->nil", _M_BC, poses, np.linalg.inv(_M_BC))


def load_train_list() -> list[tuple[str, str]]:
    import yaml
    cfg = yaml.safe_load(open(SPLIT_YAML))
    train_list = [(ds["type"], str(seq))
                  for ds in cfg["data"]["datasets"]["train"]
                  for seq in ds["sequences"]]
    assert len(train_list) == 16, \
        f"expected 16 pd train sequences, got {len(train_list)}"
    return train_list


def cache_path(sensor: str, seq: str, split: str) -> Path:
    mid = f"{sensor}_val" if split == "val" else sensor
    return CACHE_DIR / f"cache_{CACHE_KEYS[sensor]}_{mid}_{seq}.npz"


def load_cache(sensor: str, seq: str, split: str):
    with np.load(cache_path(sensor, seq, split), allow_pickle=True) as c:
        return (c["scan_ids"].astype(np.int64),
                c["poses"].astype(np.float64),
                c["timestamps"].astype(np.float64))


def subsample_indices(positions: np.ndarray, dist_threshold: float) -> list[int]:
    """Official select_frame_indices rule: keep the first pose, then every
    pose whose 3D Euclidean distance from the LAST KEPT pose exceeds the
    threshold (generate_riv_images.py:330, dist_threshold=3.0)."""
    keep: list[int] = []
    last = None
    for i, p in enumerate(positions):
        if last is None or float(np.linalg.norm(p - last)) > dist_threshold:
            keep.append(i)
            last = p
    return keep


def split_dir(sensor: str, seq: str, split: str) -> Path:
    root = TRAIN_ROOT if split == "train" else VAL_ROOT
    return root / f"{sensor}_{seq}"


# ----------------------------
# Stage riv (frame-level workers, per-frame resumable, atomic writes)
# ----------------------------

_worker_loader = None
_worker_key = None


def _worker_init():
    import cv2
    cv2.setNumThreads(0)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _build_frame(sensor: str, seq: str, scan_id: int, out_path: str,
                 resize_h: int | None) -> str:
    global _worker_loader, _worker_key
    from _imlpr_riv_lib import riv_for_points
    if _worker_key != (sensor, seq):
        from _pdb_bevplace_eval import make_loader
        _worker_loader = make_loader(sensor, seq)
        _worker_key = (sensor, seq)
    pts = _worker_loader[int(scan_id)]["points"]
    img = riv_for_points(pts, sensor)          # (H0, 1022, 3) float32 0..255
    if resize_h is not None and img.shape[0] != resize_h:
        import cv2
        img = cv2.resize(img, (img.shape[1], resize_h),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32)
    tmp = out_path + ".tmp.npy"
    np.save(tmp, img)
    os.replace(tmp, out_path)
    return out_path


def build_sequence(sensor: str, seq: str, split: str, workers: int,
                   overwrite: bool) -> None:
    from _imlpr_riv_lib import (IMAGE_WIDTH, K_NEIGHBORS, MAX_NEIGHBOR_M,
                                MAX_RANGE_M, SENSORS)

    scan_ids, poses, timestamps = load_cache(sensor, seq, split)
    if split == "train":
        kept = subsample_indices(poses[:, :3, 3], DIST_THRESHOLD_M)
        resize_h = TRAIN_H
    else:
        kept = list(range(len(scan_ids)))      # full cache roster
        resize_h = None                        # H0-native (frozen zero-shot iface)
    out_dir = split_dir(sensor, seq, split)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for pos, ci in enumerate(kept):
        out_path = out_dir / f"{pos:06d}.npy"
        if overwrite or not out_path.is_file():
            todo.append((int(scan_ids[ci]), str(out_path)))
    print(f"[{split}:{sensor}/{seq}] {len(scan_ids)} cache kf -> {len(kept)} "
          f"kept ({'3.0 m subsample' if split == 'train' else 'full roster'}), "
          f"{len(todo)} to build -> {out_dir}", flush=True)

    if todo:
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_worker_init) as ex:
            futs = [ex.submit(_build_frame, sensor, seq, sid, path, resize_h)
                    for sid, path in todo]
            for f in as_completed(futs):
                f.result()  # re-raise worker errors
                done += 1
                if done % 200 == 0 or done == len(todo):
                    dt = time.time() - t0
                    print(f"  [{split}:{sensor}/{seq}] {done}/{len(todo)} "
                          f"({dt / done:.2f} s/frame eff.)", flush=True)

    n_ok = sum((out_dir / f"{pos:06d}.npy").is_file() for pos in range(len(kept)))
    if n_ok != len(kept):
        raise RuntimeError(f"{split}:{sensor}/{seq}: {n_ok}/{len(kept)} frames")

    cfg = SENSORS[sensor]
    meta = {
        "split": split,
        "sensor": sensor,
        "sequence": seq,
        "n_frames": int(len(kept)),
        "n_cache_keyframes": int(len(scan_ids)),
        "cache_indices": [int(i) for i in kept],
        "scan_ids": [int(scan_ids[i]) for i in kept],
        "cache": str(cache_path(sensor, seq, split).relative_to(REPO)),
        "subsample": ({"rule": "official select_frame_indices greedy "
                               "consecutive 3D distance on cache poses",
                       "dist_threshold_m": DIST_THRESHOLD_M}
                      if split == "train" else None),
        "params": {
            "image_width": IMAGE_WIDTH,
            "image_height_initial": cfg["H0"],
            "image_height_final": resize_h,
            "resize_interpolation": ("INTER_LINEAR (official eval-time resize; "
                                     "see module docstring)"
                                     if resize_h is not None else None),
            "v_mode": cfg["v_mode"],
            "fov_up_deg": cfg["fov_up_deg"],
            "fov_down_deg": cfg["fov_down_deg"],
            "max_range_m": MAX_RANGE_M,
            "k_neighbors": K_NEIGHBORS,
            "max_neighbor_m": MAX_NEIGHBOR_M,
            "intensity_scaling": cfg["intensity"],
            "channels": ["reflectivity", "range", "normal_ratio"],
            "recipe": "frozen _imlpr_riv_lib.riv_for_points (official "
                      "renderers, authors' mulran zero-shot branch scaling)",
        },
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = out_dir / "meta.json"
    tmp = out_dir / ".tmp_meta.json"
    tmp.write_text(json.dumps(meta, indent=1) + "\n")
    os.replace(tmp, meta_path)
    print(f"[{split}:{sensor}/{seq}] complete -> {meta_path}", flush=True)


# ----------------------------
# Stage pickle
# ----------------------------

def build_pickle(train_list: list[tuple[str, str]], pickle_out: str,
                 is_subset: bool) -> None:
    from scipy.spatial.transform import Rotation
    from sklearn.neighbors import KDTree  # their generator's tree

    seqs = {}
    for sensor, seq in train_list:
        out_dir = split_dir(sensor, seq, "train")
        meta_path = out_dir / "meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"[{sensor}/{seq}] RIV stage incomplete: no "
                             f"{meta_path} (run --stage riv first)")
        meta = json.loads(meta_path.read_text())
        scan_ids, poses, _ = load_cache(sensor, seq, "train")
        kept = np.asarray(meta["cache_indices"], dtype=np.int64)
        assert np.array_equal(scan_ids[kept],
                              np.asarray(meta["scan_ids"], dtype=np.int64)), \
            f"[{sensor}/{seq}] meta scan_ids drifted from cache"
        for pos in range(len(kept)):
            assert (out_dir / f"{pos:06d}.npy").is_file(), \
                f"[{sensor}/{seq}] missing frame {pos:06d}.npy"
        poses = fix_pose_frame(sensor, poses[kept])
        seqs[(sensor, seq)] = dict(poses=poses, n=len(kept))
        print(f"[{sensor}/{seq}] {len(kept)} kept frames")

    def group_of(sensor, seq):
        # NCLT sessions are co-registered on one campus -> cross-session
        # positives; all other cache pose frames are per-sequence local.
        return "nclt_campus" if sensor == "nclt" else f"{sensor}_{seq}"

    offset = 0
    for key in train_list:
        seqs[key]["gid0"] = offset
        offset += seqs[key]["n"]
    n_total = offset
    print(f"total {n_total} kept train frames")

    groups: dict[str, list] = {}
    for key in train_list:
        s = seqs[key]
        gids = np.arange(s["gid0"], s["gid0"] + s["n"])
        xy = s["poses"][:, :2, 3]
        groups.setdefault(group_of(*key), [[], []])
        groups[group_of(*key)][0].append(gids)
        groups[group_of(*key)][1].append(xy)
    trees = {}
    for g, (gid_l, xy_l) in groups.items():
        gids = np.concatenate(gid_l)
        xy = np.concatenate(xy_l)
        tree = KDTree(xy)
        # their construct_query_dict: query_radius r=10 (positives),
        # r=30 (non-negatives, self included)
        trees[g] = (gids, tree.query_radius(xy, r=POS_M),
                    tree.query_radius(xy, r=NONNEG_M))

    queries = {}
    n_no_pos = 0
    for sensor, seq in train_list:
        s = seqs[(sensor, seq)]
        gids, ind_nn, ind_r = trees[group_of(sensor, seq)]
        local0 = int(np.searchsorted(gids, s["gid0"]))  # gids sorted, contiguous
        quats = Rotation.from_matrix(s["poses"][:, :3, :3]).as_quat()  # qx qy qz qw
        for i in range(s["n"]):
            gid = s["gid0"] + i
            li = local0 + i
            assert gids[li] == gid
            positives = np.setdiff1d(gids[ind_nn[li]], [gid]).astype(int)
            non_negatives = np.sort(gids[ind_r[li]]).astype(int)
            if len(positives) == 0:
                n_no_pos += 1
            t = s["poses"][i, :3, 3]
            position = np.array([t[0], t[1], t[2], *quats[i]], dtype=np.float32)
            queries[gid] = {
                "id": int(gid),
                "timestamp": i,  # generator convention: int(%06d stem)
                "rel_scan_filepath": f"training/{sensor}_{seq}/{i:06d}.npy",
                "positives": np.sort(positives).tolist(),
                "non_negatives": non_negatives.tolist(),
                "position": position,
            }

    assert len(queries) == n_total
    print(f"tuples: {len(queries)}, anchors without positives: {n_no_pos}")

    out = OUT_ROOT / pickle_out
    tmp = OUT_ROOT / f".tmp_{pickle_out}"
    with open(tmp, "wb") as f:
        pickle.dump(queries, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    manifest = {
        "split": "pd2026 place-disjoint (configs/training_place_disjoint.yaml)",
        "smoke_subset": is_subset,
        "train_sequences": [f"{d}/{s}" for d, s in train_list],
        "n_frames_per_sequence": {f"{d}/{s}": seqs[(d, s)]["n"]
                                  for d, s in train_list},
        "n_tuples": len(queries),
        "n_anchors_without_positives": n_no_pos,
        "subsample_dist_threshold_m": DIST_THRESHOLD_M,
        "pos_m": POS_M, "nonneg_m": NONNEG_M,
        "non_negatives_include_self": True,
        "grouping": "nclt sessions merged (co-registered campus); others "
                    "per-sequence (local pose frames)",
        "kitti_pose_conjugation": "cam0 -> z-up body (_pdb_ringsharp_prepare "
                                  "precedent)",
        "schema": "generate_training_tuples_baseline_img.construct_query_dict",
        "cache_keys": CACHE_KEYS,
        "checkpoint_selection_rule": (
            "highest pd-validation sequence-macro 5 m Recall@1 over the saved "
            "epochs (save_freq=10: epochs 10,20,...,100); earlier epoch on "
            "ties; weight sha256 frozen before any test contact "
            "(scripts/_imlpr_pd_select.py)"),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    man_path = OUT_ROOT / f"{pickle_out}.manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {man_path}")


# ----------------------------
# Stage config
# ----------------------------

def ensure_config(write: bool) -> bool:
    """Write config_pd.txt if missing; validate byte-identity if present.
    Returns True when the on-disk config matches CONFIG_TEXT."""
    if not CONFIG_PATH.is_file():
        if write:
            OUT_ROOT.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(CONFIG_TEXT)
            print(f"wrote {CONFIG_PATH}")
            return True
        print(f"WARNING: {CONFIG_PATH} missing (run --stage config)")
        return False
    if CONFIG_PATH.read_text() != CONFIG_TEXT:
        print(f"WARNING: {CONFIG_PATH} differs from the canonical CONFIG_TEXT "
              f"in {__file__} — hand-edited? NOT overwritten; reconcile before "
              f"training.")
        return False
    return True


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=["riv", "pickle", "config"])
    ap.add_argument("--split", default="both", choices=["train", "val", "both"],
                    help="riv stage: which split(s) to build")
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="subset like kitti/07 (default: full roster)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pickle-out", default=PICKLE_DEFAULT,
                    help="pickle stage output name (a non-default name is "
                         "required for subset/smoke builds)")
    args = ap.parse_args()

    if args.stage == "config":
        ok = ensure_config(write=True)
        raise SystemExit(0 if ok else 1)

    ensure_config(write=False)  # validate only; never blocks data generation

    train_list = load_train_list()

    if args.stage == "riv":
        rosters = []
        if args.split in ("train", "both"):
            rosters.append(("train", train_list))
        if args.split in ("val", "both"):
            rosters.append(("val", VAL_SEQS))
        jobs = []
        if args.seqs:
            # lenient across rosters: a seq only needs to match ONE of the
            # selected splits (e.g. --split both --seqs kitti/07 kitti/09)
            wanted = list(dict.fromkeys(args.seqs))
            matched = set()
            for split, pairs in rosters:
                have = {f"{d}/{s}": (d, s) for d, s in pairs}
                for w in wanted:
                    if w in have:
                        jobs.append((*have[w], split))
                        matched.add(w)
            unmatched = [w for w in wanted if w not in matched]
            if unmatched:
                raise SystemExit(
                    f"not in the selected split roster(s): {unmatched}")
        else:
            jobs = [(d, s, split) for split, pairs in rosters for d, s in pairs]
        for sensor, seq, split in jobs:
            build_sequence(sensor, seq, split, args.workers, args.overwrite)
        return

    # stage pickle
    if args.seqs:
        wanted = list(dict.fromkeys(args.seqs))
        have = {f"{d}/{s}": (d, s) for d, s in train_list}
        missing = [w for w in wanted if w not in have]
        if missing:
            raise SystemExit(f"not in the 16-seq pd train roster: {missing}")
        subset = [have[w] for w in wanted]
    else:
        subset = train_list
    is_subset = len(subset) != len(train_list)
    if is_subset and args.pickle_out == PICKLE_DEFAULT:
        raise SystemExit("refusing to write the canonical training_pd.pickle "
                         "from a sequence subset — pass --pickle-out NAME for "
                         "smoke builds")
    build_pickle(subset, args.pickle_out, is_subset)


if __name__ == "__main__":
    main()
