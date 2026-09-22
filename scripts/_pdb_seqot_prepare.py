#!/usr/bin/env python3
"""Prepare SeqOT-format range images for the pd2026 place-disjoint splits.

Same recipe as scripts/_seqot_prepare.py (official SeqOT range_projection,
32x900, per-sensor FOV, pose-distance surrogate train index), re-pointed at
the pd2026 keyframe rosters:

  * TRAIN/VAL sequences come from configs/training_place_disjoint.yaml
    (train: kitti 01/02/06/07, helipr Bridge01-03+Roundabout01-02,
     mulran Sejong01/02, nclt 2012-05-11/2012-08-04/2012-11-04/2012-11-16/
     2013-02-23; val: kitti 09, nclt 2012-01-08, helipr Roundabout03,
     mulran Sejong03), scan_ids/poses from the committed pd2026 operating
    caches (data/preprocessed_pd2026/cache_<key>_<sensor>[_val]_<seq>.npz).
  * TEST sequences are the frozen 42-seq roster from
    scripts/_pd_test_cache_manifest.json (adds kitti360 + mcd).
  * train_index.npy (pose-distance surrogate: positive < 10 m, negative
    > 50 m, 8+8 per anchor — identical to _seqot_prepare.py) is written ONLY
    for pd TRAIN sequences, so scripts/_seqot_train.py::find_train_seqs can
    never pick up a test sequence.

New sensors (additive FOV entries; everything else per _seqot_prepare.py):
  kitti360  HDL-64E, same FOV as kitti  [-24.8, 2.0]
  mcd       Ouster OS1-64 gen2          [-22.5, 22.5]

Output layout (SeqOT native, one dir per sequence) under
_external/seqot_data_pd2026/<sensor>_<seq>/: NNNNNN.npy range images,
poses.npy, meta.json, train_index.npy (train only).

Run inside the ring_baseline container, ONE HEAVY SEQUENCE PER PROCESS:
  python scripts/_pdb_seqot_prepare.py kitti/02 --max-frames 300   # smoke
  python scripts/_pdb_seqot_prepare.py mcd/tuhh_night_09
  python scripts/_pdb_seqot_prepare.py --list                      # roster+splits
"""
from __future__ import annotations
import argparse
import glob as globmod
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = os.environ.get("NSD_DATA_ROOT", "/workspace/data")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(1, str(REPO / "src"))

import _seqot_prepare as base  # noqa: E402  (range_projection, build_train_index)

PD_CACHE = REPO / "data" / "preprocessed_pd2026"
PD_CONFIG = REPO / "configs" / "training_place_disjoint.yaml"
MANIFEST = REPO / "scripts" / "_pd_test_cache_manifest.json"
OUT_ROOT_DEFAULT = REPO / "_external" / "seqot_data_pd2026"

# (fov_up, fov_down) in degrees; kitti/nclt/helipr/mulran identical to
# _seqot_prepare.SENSOR_PROJ, kitti360/mcd additive.
SENSOR_PROJ = dict(base.SENSOR_PROJ)
SENSOR_PROJ["kitti360"] = {"fov_up": 2.0, "fov_down": -24.8, "flip_z": False}
SENSOR_PROJ["mcd"] = {"fov_up": 22.5, "fov_down": -22.5, "flip_z": False}


def load_splits():
    """{(sensor, seq): split} for train/val (pd config) + test (manifest)."""
    table = {}
    cfg = yaml.safe_load(open(PD_CONFIG))
    for split in ("train", "val"):
        for ds in cfg["data"]["datasets"].get(split, []):
            for seq in ds["sequences"]:
                table[(ds["type"], str(seq))] = split
    man = json.loads(MANIFEST.read_text())
    test_caches = {}
    for e in man["sequences"]:
        key = (e["sensor"], e["sequence"])
        assert key not in table, f"test sequence also in train/val: {key}"
        table[key] = "test"
        test_caches[key] = REPO / e["expected"]["operating_cache"]
    return table, test_caches


def cache_path_for(sensor, seq, split, test_caches):
    if split == "test":
        return test_caches[(sensor, seq)]
    pat = f"cache_*_{sensor}_val_{seq}.npz" if split == "val" \
        else f"cache_*_{sensor}_{seq}.npz"
    hits = sorted(globmod.glob(str(PD_CACHE / pat)))
    assert len(hits) == 1, f"expected exactly one pd2026 cache for {pat}: {hits}"
    return Path(hits[0])


def make_pd_loader(sensor: str, seq: str):
    if sensor == "kitti360":
        from data.kitti360_loader import Kitti360Loader
        return Kitti360Loader(os.path.join(DATA_ROOT, "kitti360"), seq, lazy_load=True)
    if sensor == "mcd":
        from data.mcd_loader import MCDLoader
        return MCDLoader(os.path.join(DATA_ROOT, "mcd_extracted"), seq, lazy_load=True)
    root = {"kitti": os.path.join(DATA_ROOT, "kitti", "dataset"),
            "nclt": os.path.join(DATA_ROOT, "nclt"),
            "helipr": os.path.join(DATA_ROOT, "helipr"),
            "mulran": os.path.join(DATA_ROOT, "mulran")}[sensor]
    return base.create_loader(sensor, root, seq)


def prepare_sequence(sensor, seq, split, cache_path, out_root, max_frames, rng):
    proj = SENSOR_PROJ[sensor]
    with np.load(cache_path) as c:
        scan_ids = c["scan_ids"]
        poses = c["poses"]
    if max_frames and len(scan_ids) > max_frames:
        scan_ids, poses = scan_ids[:max_frames], poses[:max_frames]

    out_dir = out_root / f"{sensor}_{seq}"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = make_pd_loader(sensor, seq)
    t0 = time.time()
    for i, sid in enumerate(scan_ids):
        pts = loader[int(sid)]["points"][:, :3].astype(np.float64)
        if proj["flip_z"]:
            pts = pts.copy()
            pts[:, 2] = -pts[:, 2]
        if i == 0:  # FOV sanity diagnostic on the first scan
            depth = np.linalg.norm(pts, axis=1)
            ok = (depth > 0) & (depth < base.MAX_RANGE)
            pitch = np.degrees(np.arcsin(pts[ok, 2] / depth[ok]))
            in_band = np.mean((pitch >= proj["fov_down"]) & (pitch <= proj["fov_up"]))
            print(f"  [{sensor}_{seq}] scan0 pitch p1/p99 = "
                  f"{np.percentile(pitch, 1):.2f}/{np.percentile(pitch, 99):.2f} deg, "
                  f"in-FOV fraction = {in_band:.3f} "
                  f"(fov [{proj['fov_down']}, {proj['fov_up']}])", flush=True)
        proj_range, _, _ = base.range_projection(
            pts, fov_up=proj["fov_up"], fov_down=proj["fov_down"],
            proj_H=base.PROJ_H, proj_W=base.PROJ_W,
            max_range=base.MAX_RANGE, cut_z=False)
        np.save(out_dir / f"{i:06d}.npy", proj_range.astype(np.float32))
        if (i + 1) % 500 == 0:
            print(f"    [{sensor}_{seq}] {i + 1}/{len(scan_ids)} projected "
                  f"({(time.time() - t0) / (i + 1):.3f} s/kf)", flush=True)
    del loader
    dt = time.time() - t0
    np.save(out_dir / "poses.npy", poses.astype(np.float64))

    n_index = 0
    if split == "train":
        index, n_anchor = base.build_train_index(
            poses[:, :3, 3].astype(np.float64), rng)
        np.save(out_dir / "train_index.npy", index)
        n_index = len(index)
        print(f"  [{sensor}_{seq}] train_index: {n_index} rows, "
              f"{n_anchor}/{len(poses)} usable anchors", flush=True)

    meta = {
        "dtype": sensor, "sequence": seq, "split": f"pd2026_{split}",
        "n_frames": int(len(scan_ids)),
        "fov_up": proj["fov_up"], "fov_down": proj["fov_down"],
        "flip_z": proj["flip_z"], "max_range": base.MAX_RANGE,
        "proj_H": base.PROJ_H, "proj_W": base.PROJ_W,
        "cache": cache_path.name,
        "truncated": bool(max_frames and len(scan_ids) == max_frames),
        "index_pos_max_m": base.POS_MAX_M, "index_neg_min_m": base.NEG_MIN_M,
        "prepare_s_per_kf": round(dt / max(len(scan_ids), 1), 4),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  [{sensor}_{seq}] done [{split}]: {len(scan_ids)} frames in "
          f"{dt:.0f}s ({dt / max(len(scan_ids), 1):.3f} s/kf) -> {out_dir}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sequences", nargs="*",
                    help="sensor/seq pairs (one heavy sequence per process)")
    ap.add_argument("--all-train", action="store_true")
    ap.add_argument("--all-val", action="store_true")
    ap.add_argument("--all-test", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="print every (sensor, seq, split) and exit")
    ap.add_argument("--out-root", default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--max-frames", type=int, default=0, help="smoke only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    table, test_caches = load_splits()
    if args.list:
        for (sensor, seq), split in sorted(table.items(), key=lambda kv: (kv[1], kv[0])):
            print(f"{split:5s} {sensor}/{seq}")
        return

    targets = []
    for want, flag in (("train", args.all_train), ("val", args.all_val),
                       ("test", args.all_test)):
        if flag:
            targets += [k for k, v in table.items() if v == want]
    for s in args.sequences:
        sensor, seq = s.split("/", 1)
        if (sensor, seq) not in table:
            ap.error(f"{s} not in the pd2026 train/val/test rosters")
        targets.append((sensor, seq))
    if not targets:
        ap.error("nothing to do: give sensor/seq pairs or --all-train/val/test")

    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out_root)
    for sensor, seq in targets:
        split = table[(sensor, seq)]
        cache = cache_path_for(sensor, seq, split, test_caches)
        print(f"Preparing {sensor}/{seq} [{split}] from {cache.name} ...", flush=True)
        prepare_sequence(sensor, seq, split, cache, out_root, args.max_frames, rng)
    print("PDB_SEQOT_PREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
