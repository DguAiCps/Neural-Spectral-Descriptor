#!/usr/bin/env python3
"""Build nsd_layouts (16x60) + bev_layouts (79x60) for the pd-protocol NEW VAL
sequences (kitti 09, nclt 2012-01-08, helipr Roundabout03, mulran Sejong03).

Companion to scripts/_pd_build_layouts.py (TRAIN). The pd2026 val caches hold
descriptors/poses/keyframes but no layouts; the committed nclt operating caches
(plain and .zup) are not pd-consistent (per-frame descriptor cosine 0.84/0.75),
and kitti 09 / helipr Roundabout03 / mulran Sejong03 have no operating or BEV
caches at all. Keyframe selection is taken as-is from the pd2026 caches
(scan_ids), so we only re-project the known keyframe scans with the current
loaders/encoder. BEV parameters match the committed rerank caches
(s60_max_r1-80_z-3-5_h8).

Writes data/preprocessed_pd2026/vallayouts_<sensor>_<seq>.npz
(nsd_layouts + bev_layouts + scan_ids + keyframe_ids). VAL sequences only —
never test.

--selftest re-projects the first 50 kitti 07 TRAIN keyframes and compares both
layout kinds against the committed caches (loader unchanged for kitti -> must
match).
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_kitti_operating_point import _load_config, _make_encoder, _project_nsd_layout
from encoding.bev_image import BEVProjector, interpolate_bev_image

DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", "/workspace/data"))
PD_CACHE = REPO / "data/preprocessed_pd2026"
CONFIG = REPO / "configs/training_place_disjoint.yaml"

# pd NEW VAL sequences (pd2026 cache basenames carry a val_ marker)
SEQS = [
    ("kitti", "09"), ("nclt", "2012-01-08"),
    ("helipr", "Roundabout03"), ("mulran", "Sejong03"),
]


def _pd_cache(sensor: str, seq: str, val: bool) -> np.lib.npyio.NpzFile:
    tag = "val_" if val else ""
    hits = sorted(glob.glob(str(PD_CACHE / f"cache_*_{sensor}_{tag}{seq}.npz")))
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
    if sensor == "mulran":
        from data.mulran_loader import MulRanLoader
        return MulRanLoader(str(DATA_ROOT / "mulran"), seq, lazy_load=True)
    if sensor == "kitti":
        from data.kitti_loader import KITTILoader
        return KITTILoader(str(DATA_ROOT / "kitti/dataset"), seq, lazy_load=True)
    raise ValueError(sensor)


def _make_bev_projector() -> BEVProjector:
    # match committed rerank caches: *_s60_max_r1-80_z-3-5_h8*
    return BEVProjector(n_sectors=60, max_range=80.0, min_range=1.0,
                        z_min=-3.0, height_encoding="max",
                        n_height_layers=8, z_max=5.0)


def _build(encoder, bev_proj, sensor: str, seq: str, scan_ids, poses, limit=None):
    loader = _make_loader(sensor, seq)
    nsd, bev, pose_mismatch = [], [], 0
    ids = scan_ids if limit is None else scan_ids[:limit]
    for j, sid in enumerate(ids):
        if j % 500 == 0:
            print(f"  [{sensor}/{seq}] keyframe {j}/{len(ids)}", flush=True)
        item = loader[int(sid)]
        if not np.allclose(item["pose"], poses[j], atol=1e-5):
            pose_mismatch += 1
        nsd.append(_project_nsd_layout(encoder, item["points"], n_layout_sectors=60))
        img, _ = bev_proj.project(item["points"], keep_intensity=False)
        bev.append(interpolate_bev_image(img, method="linear", n_channels=1).astype(np.float32))
    if pose_mismatch:
        print(f"  WARNING {sensor}/{seq}: {pose_mismatch}/{len(ids)} pose mismatches "
              "between loader and pd2026 cache", flush=True)
    return (np.asarray(nsd, dtype=np.float32), np.asarray(bev, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    encoder = _make_encoder(_load_config(CONFIG), "cpu")
    bev_proj = _make_bev_projector()

    if args.selftest:
        cache = _pd_cache("kitti", "07", val=False)
        op = np.load(REPO / "data/preprocessed_cross_sensor_operating/"
                     "kitti_operating_07_layout60_stride1.npz")
        bev_ref = np.load(REPO / "data/preprocessed_kitti_bev_layout/"
                          "kitti_bev_layout_07_s60_max_r1-80_z-3-5_h8.npz")["bev_layouts"]
        nsd, bev = _build(encoder, bev_proj, "kitti", "07",
                          cache["scan_ids"], cache["poses"], limit=50)
        print(f"SELFTEST kitti/07 first 50: max|nsd diff| = "
              f"{np.abs(nsd - op['nsd_layouts'][:50]).max():.6f}  "
              f"max|bev diff| = {np.abs(bev - bev_ref[:50]).max():.6f}  "
              f"bev shape = {bev.shape}")
        return

    for sensor, seq in SEQS:
        out = PD_CACHE / f"vallayouts_{sensor}_{seq}.npz"
        if out.exists():
            print("SKIP", out.name, flush=True)
            continue
        cache = _pd_cache(sensor, seq, val=True)
        nsd, bev = _build(encoder, bev_proj, sensor, seq, cache["scan_ids"], cache["poses"])
        np.savez_compressed(out, nsd_layouts=nsd, bev_layouts=bev,
                            scan_ids=cache["scan_ids"], keyframe_ids=cache["keyframe_ids"])
        print(f"OK {out.name}: nsd {nsd.shape} bev {bev.shape}", flush=True)
    print("PD_VAL_LAYOUTS_DONE", flush=True)


if __name__ == "__main__":
    main()
