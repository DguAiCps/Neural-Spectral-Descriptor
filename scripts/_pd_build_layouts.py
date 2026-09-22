#!/usr/bin/env python3
"""Build nsd_layouts (16x60) for pd-protocol TRAIN keyframes that lack a
current-loader operating cache.

Needed because the edge classifier's phase-alignment features consume
nsd_layouts, but (a) helipr Bridge02/Bridge03/Roundabout02 have no operating
cache at all, and (b) the on-disk nclt operating caches were built with the
pre-z-fix loader and no longer match the pd2026 descriptors (per-frame cosine
~0.83). Keyframe selection is unchanged (scan_ids/keyframe_ids identical), so
we only re-project the known keyframe scans with the current loader/encoder.

Writes data/preprocessed_pd2026/layouts_<sensor>_<seq>_layout60.npz
(nsd_layouts + scan_ids + keyframe_ids). TRAIN sequences only.

--selftest re-projects the first 100 kitti 07 keyframes and compares against
the committed operating cache (loader unchanged for kitti -> must match).
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_kitti_operating_point import _load_config, _make_encoder, _project_nsd_layout

DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", "/workspace/data"))
PD_CACHE = REPO / "data/preprocessed_pd2026"
CONFIG = REPO / "configs/training_place_disjoint.yaml"

# pd TRAIN sequences without a pd-consistent operating cache
SEQS = [
    ("nclt", "2012-05-11"), ("nclt", "2012-08-04"), ("nclt", "2012-11-04"),
    ("nclt", "2012-11-16"), ("nclt", "2013-02-23"),
    ("helipr", "Bridge02"), ("helipr", "Bridge03"), ("helipr", "Roundabout02"),
]


def _pd_cache(sensor: str, seq: str) -> np.lib.npyio.NpzFile:
    hits = sorted(glob.glob(str(PD_CACHE / f"cache_*_{sensor}_{seq}.npz")))
    assert len(hits) == 1, (sensor, seq, hits)
    return np.load(hits[0])


def _make_loader(sensor: str, seq: str):
    if sensor == "nclt":
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(str(DATA_ROOT / "nclt"), seq, lazy_load=True)
    if sensor == "helipr":
        from data.helipr_loader import HeLiPRLoader
        seq_path = DATA_ROOT / "helipr" / seq
        if not (seq_path / "LiDAR").is_dir():
            seq_path = seq_path / seq  # legacy double-nested layout
        return HeLiPRLoader(str(seq_path), lazy_load=True)
    if sensor == "kitti":  # selftest only
        from data.kitti_loader import KITTILoader
        return KITTILoader(str(DATA_ROOT / "kitti/dataset"), seq, lazy_load=True)
    raise ValueError(sensor)


def _build(encoder, sensor: str, seq: str, scan_ids, poses, limit=None):
    loader = _make_loader(sensor, seq)
    layouts, pose_mismatch = [], 0
    ids = scan_ids if limit is None else scan_ids[:limit]
    for j, sid in enumerate(ids):
        if j % 500 == 0:
            print(f"  [{sensor}/{seq}] keyframe {j}/{len(ids)}", flush=True)
        item = loader[int(sid)]
        if not np.allclose(item["pose"], poses[j], atol=1e-5):
            pose_mismatch += 1
        layouts.append(_project_nsd_layout(encoder, item["points"], n_layout_sectors=60))
    if pose_mismatch:
        print(f"  WARNING {sensor}/{seq}: {pose_mismatch}/{len(ids)} pose mismatches "
              "between loader and pd2026 cache", flush=True)
    return np.asarray(layouts, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    encoder = _make_encoder(_load_config(CONFIG), "cpu")

    if args.selftest:
        cache = _pd_cache("kitti", "07")
        op = np.load(REPO / "data/preprocessed_cross_sensor_operating/"
                     "kitti_operating_07_layout60_stride1.npz")
        lay = _build(encoder, "kitti", "07", cache["scan_ids"], cache["poses"], limit=100)
        diff = np.abs(lay - op["nsd_layouts"][:100]).max()
        print(f"SELFTEST kitti/07 first 100 keyframes: max|layout diff| = {diff:.6f}")
        return

    for sensor, seq in SEQS:
        out = PD_CACHE / f"layouts_{sensor}_{seq}_layout60.npz"
        if out.exists():
            print("SKIP", out.name, flush=True)
            continue
        cache = _pd_cache(sensor, seq)
        lay = _build(encoder, sensor, seq, cache["scan_ids"], cache["poses"])
        np.savez_compressed(out, nsd_layouts=lay, scan_ids=cache["scan_ids"],
                            keyframe_ids=cache["keyframe_ids"])
        print(f"OK {out.name}: {lay.shape}", flush=True)


if __name__ == "__main__":
    main()
