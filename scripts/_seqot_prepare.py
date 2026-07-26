#!/usr/bin/env python3
"""Prepare SeqOT-format range-image data from OUR keyframe split.

For each requested `<dtype>/<sequence>` this script:
  1. loads keyframe scan_ids + poses from the training cache
     (data/preprocessed/cache_<key>_<dtype>[_val]_<seq>.npz,
      key 056e0a02 for kitti/nclt/helipr, 33919e6e for mulran),
  2. fetches the raw point clouds at those scan_ids via src/data loaders
     (loader factory copied from baselines/evaluate_baselines.py:create_loader),
  3. projects each scan to a (32, 900) range image with SeqOT's own
     range_projection (_external/SeqOT/tools/utils/utils.py) and writes
     per-keyframe NNNNNN.npy files under _external/seqot_data/<dtype>_<seq>/
     (SeqOT's native file layout), plus poses.npy and meta.json,
  4. for TRAIN-split sequences, writes train_index.npy with rows
     (anchor_idx, other_idx, overlap) — see surrogate note below.

Deviations from the official SeqOT pipeline (documented on purpose):
  * Overlap surrogate: the official training index is built from overlap
    reprojection (tools/utils/com_overlap_yaw.py), which is far too expensive
    for 24 sequences. We use a pose-distance surrogate instead:
    overlap=1.0 (positive) if 3D pose distance < 10 m, overlap=0.0 (negative)
    if > 50 m; pairs in the 10–50 m band are excluded. Positives include
    temporal neighbors, as in the official index (which is dominated by
    short-range pairs from the same drive-by).
  * Per-sensor FOV: the official gen_depth_data.py hard-codes NCLT HDL-32E
    FOV and NCLT binary parsing. We keep the native 32x900 resolution (their
    column-CNN stride requires height 32) but project with each sensor's
    elevation range from configs/training_multi_dataset.yaml:
    KITTI HDL-64E [-24.8, 2.0], NCLT HDL-32E [-30.67, 10.67],
    HeLiPR VLP-16 [-15, 15], MulRan OS1-64 [-16.6, 16.6]; max_range 80.
  * NCLT z handling: official SeqOT flips z (gen_depth_data.py data2xyzi
    flip=True) and projects with fov_up=30.67/fov_down=-10.67. We keep our
    loader's points unflipped and project with the true sensor FOV
    [-30.67, 10.67] — verified empirically (scan0 of 2012-05-11: 96.7% of
    points in-band unflipped vs 80.6% flipped). This is the exact vertical
    mirror of the official projection (same in-FOV content; row order is
    immaterial for a CNN trained from scratch).
  * Sequences are OUR keyframe sequences (0.8 m / 20 deg / 30 s selector),
    not raw 10 Hz scan streams, so seqL/seqlen windows span keyframes.

Run inside the docker container (needs src/data loaders + dataset mounts):
  python scripts/_seqot_prepare.py --all-train --all-val          # everything
  python scripts/_seqot_prepare.py kitti/02 nclt/2012-05-11 --max-frames 300
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
SEQOT = REPO / "_external" / "SeqOT"
sys.path.insert(0, str(SEQOT))          # for tools.utils.utils.range_projection
sys.path.insert(1, str(REPO / "src"))   # for data.* loaders

# SeqOT's tools/ is a namespace package; a regular `tools` package in the
# docker image's site-packages shadows it, so load by file path (unchanged).
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "_seqot_utils", SEQOT / "tools" / "utils" / "utils.py")
_seqot_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_seqot_utils)
range_projection = _seqot_utils.range_projection  # SeqOT original, unchanged

CACHE_KEYS = {"kitti": "056e0a02", "nclt": "056e0a02",
              "helipr": "056e0a02", "mulran": "33919e6e"}

# (fov_up, fov_down) in degrees, applied AFTER the optional z-flip.
SENSOR_PROJ = {
    "kitti":  {"fov_up": 2.0,   "fov_down": -24.8,  "flip_z": False},
    "nclt":   {"fov_up": 10.67, "fov_down": -30.67, "flip_z": False},
    "helipr": {"fov_up": 15.0,  "fov_down": -15.0,  "flip_z": False},
    "mulran": {"fov_up": 16.6,  "fov_down": -16.6,  "flip_z": False},
}
PROJ_H, PROJ_W, MAX_RANGE = 32, 900, 80.0
POS_MAX_M, NEG_MIN_M = 10.0, 50.0
N_POS_PER_ANCHOR, N_NEG_PER_ANCHOR = 8, 8


def create_loader(dataset_type, root, sequence):
    """Copied from baselines/evaluate_baselines.py (loader factory)."""
    if dataset_type == 'kitti':
        from data.kitti_loader import KITTILoader
        return KITTILoader(root, sequence, lazy_load=True)
    elif dataset_type == 'nclt':
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(root, sequence, lazy_load=True)
    elif dataset_type == 'helipr':
        from data.helipr_loader import HeLiPRLoader
        seq_path = os.path.join(root, sequence, sequence)
        return HeLiPRLoader(seq_path, lazy_load=True)
    elif dataset_type == 'mulran':
        from data.mulran_loader import MulRanLoader
        return MulRanLoader(root, sequence, lazy_load=True)
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_split_table(config):
    """Return {(dtype, seq): (split, root)} from the training config."""
    table = {}
    for split in ("train", "val"):
        for ds in config["data"]["datasets"].get(split, []):
            for seq in ds["sequences"]:
                table[(ds["type"], str(seq))] = (split, ds["root"])
    return table


# Three train sequences are absent under the primary keys; compatible caches
# (same keyframe-selector density, 544D) exist under these fallback keys:
# kitti_01 N=1101 (de201724), nclt_2012-11-16 / 2013-02-23 N=5460 (76c42cd7).
CACHE_KEY_FALLBACK = {("kitti", "01"): "de201724",
                      ("nclt", "2012-11-16"): "76c42cd7",
                      ("nclt", "2013-02-23"): "76c42cd7"}


def cache_path_for(dtype, seq, split):
    key = CACHE_KEYS[dtype]
    name = f"cache_{key}_{dtype}_val_{seq}.npz" if split == "val" \
        else f"cache_{key}_{dtype}_{seq}.npz"
    p = REPO / "data" / "preprocessed" / name
    if not p.exists() and split != "val" and (dtype, str(seq)) in CACHE_KEY_FALLBACK:
        fb = CACHE_KEY_FALLBACK[(dtype, str(seq))]
        p = REPO / "data" / "preprocessed" / f"cache_{fb}_{dtype}_{seq}.npz"
    return p


def build_train_index(positions, rng):
    """Pose-distance surrogate index: rows (anchor, other, overlap in {1, 0})."""
    from scipy.spatial import cKDTree
    n = len(positions)
    tree = cKDTree(positions)
    rows = []
    n_anchor = 0
    for a in range(n):
        pos_c = [i for i in tree.query_ball_point(positions[a], POS_MAX_M)
                 if i != a]
        if not pos_c:
            continue
        dists = np.linalg.norm(positions - positions[a], axis=1)
        neg_c = np.where(dists > NEG_MIN_M)[0]
        if neg_c.size == 0:
            continue
        pos_s = rng.choice(pos_c, size=min(N_POS_PER_ANCHOR, len(pos_c)),
                           replace=False)
        neg_s = rng.choice(neg_c, size=min(N_NEG_PER_ANCHOR, neg_c.size),
                           replace=False)
        rows.extend((a, p, 1.0) for p in pos_s)
        rows.extend((a, m, 0.0) for m in neg_s)
        n_anchor += 1
    return np.asarray(rows, dtype=np.float32), n_anchor


def prepare_sequence(dtype, seq, split, root, out_root, max_frames, rng):
    proj = SENSOR_PROJ[dtype]
    cpath = cache_path_for(dtype, seq, split)
    if not cpath.exists():
        print(f"  SKIP {dtype}/{seq}: cache missing: {cpath}")
        return False
    with np.load(cpath) as c:
        scan_ids = c["scan_ids"]
        poses = c["poses"]
    if max_frames and len(scan_ids) > max_frames:
        scan_ids, poses = scan_ids[:max_frames], poses[:max_frames]

    out_dir = out_root / f"{dtype}_{seq}"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = create_loader(dtype, root, seq)
    for i, sid in enumerate(scan_ids):
        pts = loader[int(sid)]["points"][:, :3].astype(np.float64)
        if proj["flip_z"]:
            pts = pts.copy()
            pts[:, 2] = -pts[:, 2]
        if i == 0:  # FOV sanity diagnostic on the first scan
            depth = np.linalg.norm(pts, axis=1)
            ok = (depth > 0) & (depth < MAX_RANGE)
            pitch = np.degrees(np.arcsin(pts[ok, 2] / depth[ok]))
            in_band = np.mean((pitch >= proj["fov_down"]) &
                              (pitch <= proj["fov_up"]))
            print(f"  [{dtype}_{seq}] scan0 pitch p1/p99 = "
                  f"{np.percentile(pitch, 1):.2f}/{np.percentile(pitch, 99):.2f} deg, "
                  f"in-FOV fraction = {in_band:.3f} "
                  f"(fov [{proj['fov_down']}, {proj['fov_up']}], "
                  f"flip_z={proj['flip_z']})")
        proj_range, _, _ = range_projection(
            pts, fov_up=proj["fov_up"], fov_down=proj["fov_down"],
            proj_H=PROJ_H, proj_W=PROJ_W, max_range=MAX_RANGE, cut_z=False)
        np.save(out_dir / f"{i:06d}.npy", proj_range.astype(np.float32))
        if (i + 1) % 200 == 0:
            print(f"    [{dtype}_{seq}] {i + 1}/{len(scan_ids)} projected")
    del loader

    np.save(out_dir / "poses.npy", poses.astype(np.float64))

    n_index = 0
    if split == "train":
        index, n_anchor = build_train_index(poses[:, :3, 3].astype(np.float64),
                                            rng)
        np.save(out_dir / "train_index.npy", index)
        n_index = len(index)
        print(f"  [{dtype}_{seq}] train_index: {n_index} rows, "
              f"{n_anchor}/{len(poses)} usable anchors")

    meta = {
        "dtype": dtype, "sequence": seq, "split": split,
        "n_frames": int(len(scan_ids)),
        "fov_up": proj["fov_up"], "fov_down": proj["fov_down"],
        "flip_z": proj["flip_z"], "max_range": MAX_RANGE,
        "proj_H": PROJ_H, "proj_W": PROJ_W,
        "cache": str(cpath.name),
        "truncated": bool(max_frames and len(scan_ids) == max_frames),
        "index_pos_max_m": POS_MAX_M, "index_neg_min_m": NEG_MIN_M,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  [{dtype}_{seq}] done: {len(scan_ids)} frames -> {out_dir}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sequences", nargs="*",
                    help="dtype/sequence pairs, e.g. kitti/02 nclt/2012-05-11")
    ap.add_argument("--all-train", action="store_true",
                    help="prepare every train-split sequence from the config")
    ap.add_argument("--all-val", action="store_true",
                    help="prepare every val-split sequence from the config")
    ap.add_argument("--config", default=str(REPO / "configs/training_multi_dataset.yaml"))
    ap.add_argument("--out-root", default=str(REPO / "_external/seqot_data"))
    ap.add_argument("--max-frames", type=int, default=0,
                    help="truncate each sequence to the first N keyframes (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    table = load_split_table(config)
    rng = np.random.default_rng(args.seed)

    targets = []
    if args.all_train:
        targets += [k for k, v in table.items() if v[0] == "train"]
    if args.all_val:
        targets += [k for k, v in table.items() if v[0] == "val"]
    for s in args.sequences:
        dtype, seq = s.split("/", 1)
        if (dtype, seq) not in table:
            ap.error(f"{s} not in config train/val lists")
        targets.append((dtype, seq))
    if not targets:
        ap.error("nothing to do: give dtype/seq pairs or --all-train/--all-val")

    out_root = Path(args.out_root)
    ok = 0
    for dtype, seq in targets:
        split, root = table[(dtype, seq)]
        print(f"Preparing {dtype}/{seq} [{split}] ...")
        ok += bool(prepare_sequence(dtype, seq, split, root, out_root,
                                    args.max_frames, rng))
    print(f"\nPrepared {ok}/{len(targets)} sequences under {out_root}")


if __name__ == "__main__":
    main()
