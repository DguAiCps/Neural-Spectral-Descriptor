#!/usr/bin/env python3
"""Build the pd-protocol cache artifacts for the FROZEN TEST sequences.

Per test sequence this produces the same three artifacts the pd NEW VAL stack
consumes (scripts/_pd_branch_fusion.py), with a test_ naming scheme:

  cache_<key>_<sensor>_test_<seq>.npz
      operating cache (descriptors 288D, poses, timestamps, scan_ids,
      keyframe_ids, spectral_entropies), built exactly like
      train_multi_dataset.process_dataset + save_keyframes_cache under the
      pd2026 config (configs/training_place_disjoint.yaml + no_interdiff
      preset + per-sensor elevation range). <key> is
      train_multi_dataset.compute_cache_key for the sequence's sensor family,
      so kitti/nclt/helipr/mulran keys equal the committed pd2026 val keys.
  testlayouts_<sensor>_<seq>.npz
      nsd_layouts (16x60) + bev_layouts (79x60) + scan_ids + keyframe_ids,
      schema and transforms of scripts/_pd_build_val_layouts.py (raw-config
      CPU encoder, s60_max_r1-80_z-3-5_h8 BEV projector).
  testspec_<sensor>_<seq>.npz
      scan_ids + sk_re + sk_im (16x12 BEV phase sketch) + raw_height
      (20x120 float16), schema and transforms of
      scripts/_pd_build_val_spec_caches.py.

New sensors relative to the val stack (checkpoint-free preprocessing only):
  kitti360  src/data/kitti360_loader.py, sensor family 'kitti' (HDL-64E)
  mcd       src/data/mcd_loader.py, NEW sensor family 'mcd' with elevation
            range [-22.5, 22.5] (Ouster OS1-64 gen2), wired additively into
            the config copy this script uses.

One heavy sequence per process: drive with --seqs sensor/seq and shell out
per sequence (never loop heavy sequences in-process). --manifest writes
scripts/_pd_test_cache_manifest.json listing every required artifact for the
full roster. --selftest checks the pd2026 cache keys and byte-compares a
rebuilt kitti val 09 prefix against the committed pd2026 val cache.
Never touches frozen artifacts/checkpoints/results.
"""
from __future__ import annotations
import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # train_multi_dataset lives at the repo root
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

PD_CACHE = REPO / "data/preprocessed_pd2026"
CONFIG = REPO / "configs/training_place_disjoint.yaml"
MANIFEST = REPO / "scripts/_pd_test_cache_manifest.json"
DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", "/workspace/data"))

# Frozen TEST roster (configs/training_place_disjoint.yaml test split +
# kitti360 place-disjoint drives + the 12 extracted MCD sequences).
TEST_SEQS = [
    ("kitti", s) for s in ["00", "05", "08"]
] + [
    ("helipr", s) for s in [
        "Town01", "Town02", "Town03", "Bridge04",
        "KAIST04", "KAIST05", "KAIST06", "DCC04", "DCC05", "DCC06",
        "Riverside04", "Riverside05", "Riverside06"]
] + [
    ("mulran", s) for s in [
        "DCC01", "DCC02", "DCC03", "KAIST01", "KAIST02", "KAIST03",
        "Riverside01", "Riverside02", "Riverside03"]
] + [
    ("nclt", s) for s in ["2012-02-04", "2012-03-17"]
] + [
    ("kitti360", s) for s in ["0000", "0004", "0006"]
] + [
    ("mcd", s) for s in [
        "kth_day_06", "kth_day_09", "kth_day_10",
        "kth_night_01", "kth_night_04", "kth_night_05",
        "tuhh_day_02", "tuhh_day_03", "tuhh_day_04",
        "tuhh_night_07", "tuhh_night_08", "tuhh_night_09"]
]

# Sensor family used for elevation range + cache key (kitti360 is an HDL-64E
# like KITTI odometry; everything else is its own family).
SENSOR_FAMILY = {"kitti360": "kitti"}
MCD_ELEVATION = [-22.5, 22.5]  # Ouster OS1-64 gen2

# Committed pd2026 cache keys (val stack) — the seen-sensor test caches must
# reuse them, or downstream would silently mix encoder configurations.
EXPECTED_KEYS = {"kitti": "fecdaef7", "nclt": "aff53e59",
                 "helipr": "ad22d735", "mulran": "50223b37"}


def _family(sensor: str) -> str:
    return SENSOR_FAMILY.get(sensor, sensor)


def _effective_config():
    """pd2026 training config + no_interdiff preset + additive mcd sensor."""
    import yaml
    from train_multi_dataset import apply_encoder_preset
    cfg = apply_encoder_preset(yaml.safe_load(open(CONFIG)), "no_interdiff")
    # Additive: new 'mcd' sensor entry, resolved wherever sensor ranges are
    # looked up (get_elevation_range + compute_cache_key). Existing sensors
    # untouched.
    cfg["encoding"].setdefault("sensor_elevation_ranges", {}).setdefault(
        "mcd", list(MCD_ELEVATION))
    return cfg


def _paths(cfg, sensor: str, seq: str):
    from train_multi_dataset import compute_cache_key
    key = compute_cache_key(cfg, dataset_type=_family(sensor))
    return {
        "operating_cache": PD_CACHE / f"cache_{key}_{sensor}_test_{seq}.npz",
        "testlayouts": PD_CACHE / f"testlayouts_{sensor}_{seq}.npz",
        "testspec": PD_CACHE / f"testspec_{sensor}_{seq}.npz",
    }


def _make_loader(sensor: str, seq: str):
    if sensor == "kitti360":
        from data.kitti360_loader import Kitti360Loader
        return Kitti360Loader(str(DATA_ROOT / "kitti360"), seq, lazy_load=True)
    if sensor == "mcd":
        from data.mcd_loader import MCDLoader
        return MCDLoader(str(DATA_ROOT / "mcd_extracted"), seq, lazy_load=True)
    from _pd_build_val_layouts import _make_loader as _val_loader
    return _val_loader(sensor, seq)


def _make_desc_encoder(cfg, sensor: str, device: str):
    """Byte-mirror of train_multi_dataset main() stage-1 encoder creation,
    plus the per-sensor elevation range set before each dataset."""
    from encoding.spectral_encoder import SpectralEncoder
    from train_multi_dataset import get_elevation_range
    enc = cfg["encoding"]
    bev_config = enc.get("bev", {})
    cross_cfg = enc.get("cross_spectrum", {})
    encoder = SpectralEncoder(
        n_elevation=enc["n_elevation"],
        n_azimuth=enc["n_azimuth"],
        n_bins=enc["n_bins"],
        alpha=enc["alpha"],
        learnable_alpha=enc["learnable_alpha"],
        target_elevation_bins=enc["target_elevation_bins"],
        elevation_range=tuple(enc["elevation_range"]),
        bin_statistics=enc.get("bin_statistics", ["sum"]),
        inter_bin_statistics=enc.get("inter_bin_statistics", []),
        projection_type=enc.get("projection_type", "range_image"),
        max_range=enc.get("max_range", 80.0),
        min_range=enc.get("min_range", 1.0),
        z_min=bev_config.get("z_min", -3.0),
        height_encoding=bev_config.get("height_encoding", "max"),
        n_height_layers=bev_config.get("n_height_layers", 8),
        z_max=bev_config.get("z_max", 5.0),
        zero_center=enc.get("zero_center", False),
        log_magnitude=enc.get("log_magnitude", False),
        binning_strategy=enc.get("binning_strategy", "exponential"),
        normalize_channels=enc.get("normalize_channels", True),
        cross_spectrum_enabled=cross_cfg.get("enabled", False),
        cross_spectrum_n_freqs=cross_cfg.get("n_freqs", 0),
    ).to(device)
    encoder.set_elevation_range(get_elevation_range(cfg, _family(sensor)))
    return encoder


def _build_operating_cache(cfg, sensor: str, seq: str, loader, out: Path,
                           device: str, max_scans=None):
    from keyframe.selector import KeyframeSelector
    from train_multi_dataset import process_dataset, save_keyframes_cache
    encoder = _make_desc_encoder(cfg, sensor, device)
    selector = KeyframeSelector(
        distance_threshold=cfg["keyframe"]["distance_threshold"],
        rotation_threshold=cfg["keyframe"]["rotation_threshold"],
        overlap_threshold=cfg["keyframe"]["overlap_threshold"],
        temporal_threshold=cfg["keyframe"]["temporal_threshold"],
    )
    selector.reset()
    keyframes = process_dataset(
        loader, encoder, selector, device, max_scans=max_scans,
        dataset_name=f"{sensor}_test_{seq}",
        compute_fft_magnitudes=False, compute_phase_features=False,
    )
    if out is not None:
        assert keyframes, f"no keyframes selected for {sensor}/{seq}"
        save_keyframes_cache(out, keyframes)
        print(f"OK {out.name}: {len(keyframes)} keyframes", flush=True)
    return keyframes


def _build_layouts_and_spec(sensor: str, seq: str, loader, cache, paths):
    from run_kitti_operating_point import _load_config, _make_encoder, \
        _project_nsd_layout
    from encoding.bev_image import interpolate_bev_image
    from _pd_build_val_layouts import _make_bev_projector
    from _pd_build_val_spec_caches import spec_for_points

    need_lay = not paths["testlayouts"].exists()
    need_spec = not paths["testspec"].exists()
    if not (need_lay or need_spec):
        return
    # Mirror _pd_build_val_layouts: raw config, CPU, no per-sensor elevation
    # override (only the projector/target bins are used by the layout path).
    layout_encoder = _make_encoder(_load_config(CONFIG), "cpu") if need_lay else None
    bev_proj = _make_bev_projector() if need_lay else None

    scan_ids = cache["scan_ids"]
    nsd, bev, res, ims, raws, pose_mismatch = [], [], [], [], [], 0
    for j, sid in enumerate(scan_ids):
        if j % 500 == 0:
            print(f"  [{sensor}/{seq}] keyframe {j}/{len(scan_ids)}", flush=True)
        item = loader[int(sid)]
        if not np.allclose(item["pose"], cache["poses"][j], atol=1e-5):
            pose_mismatch += 1
        if need_lay:
            nsd.append(_project_nsd_layout(layout_encoder, item["points"],
                                           n_layout_sectors=60))
            img, _ = bev_proj.project(item["points"], keep_intensity=False)
            bev.append(interpolate_bev_image(
                img, method="linear", n_channels=1).astype(np.float32))
        if need_spec:
            re, im, raw = spec_for_points(item["points"])
            res.append(re)
            ims.append(im)
            raws.append(raw)
    if pose_mismatch:
        print(f"  WARNING {sensor}/{seq}: {pose_mismatch}/{len(scan_ids)} pose "
              "mismatches between loader and operating cache", flush=True)
    if need_lay:
        nsd = np.asarray(nsd, dtype=np.float32)
        bev = np.asarray(bev, dtype=np.float32)
        np.savez_compressed(paths["testlayouts"], nsd_layouts=nsd,
                            bev_layouts=bev, scan_ids=scan_ids,
                            keyframe_ids=cache["keyframe_ids"])
        print(f"OK {paths['testlayouts'].name}: nsd {nsd.shape} bev {bev.shape}",
              flush=True)
    if need_spec:
        sk_re = np.asarray(res, np.float32)
        sk_im = np.asarray(ims, np.float32)
        raw_height = np.asarray(raws, np.float16)
        np.savez_compressed(paths["testspec"], scan_ids=scan_ids, sk_re=sk_re,
                            sk_im=sk_im, raw_height=raw_height)
        print(f"OK {paths['testspec'].name}: sk {sk_re.shape} "
              f"raw_height {raw_height.shape}", flush=True)


def build_sequence(sensor: str, seq: str, device: str):
    cfg = _effective_config()
    paths = _paths(cfg, sensor, seq)
    if all(p.exists() for p in paths.values()):
        print(f"SKIP {sensor}/{seq}: all three artifacts exist", flush=True)
        return
    loader = _make_loader(sensor, seq)
    if not paths["operating_cache"].exists():
        _build_operating_cache(cfg, sensor, seq, loader,
                               paths["operating_cache"], device)
    cache = np.load(paths["operating_cache"])
    _build_layouts_and_spec(sensor, seq, loader, cache, paths)


def write_manifest():
    cfg = _effective_config()
    entries = []
    for sensor, seq in TEST_SEQS:
        paths = _paths(cfg, sensor, seq)
        entries.append({
            "sensor": sensor,
            "sequence": seq,
            "sensor_family": _family(sensor),
            "expected": {k: str(p.relative_to(REPO)) for k, p in paths.items()},
        })
    payload = {
        "_meta": {
            "generated_by": "scripts/_pd_build_test_caches.py --manifest",
            "config": str(CONFIG.relative_to(REPO)),
            "encoder_preset": "no_interdiff",
            "mcd_elevation_range": MCD_ELEVATION,
            "n_sequences": len(entries),
            "note": "one heavy sequence per process: drive with "
                    "--seqs sensor/seq, one subprocess each",
        },
        "sequences": entries,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {MANIFEST} ({len(entries)} sequences)", flush=True)


def selftest(device: str):
    from train_multi_dataset import compute_cache_key
    cfg = _effective_config()
    keys = {s: compute_cache_key(cfg, dataset_type=_family(s))
            for s in ("kitti", "nclt", "helipr", "mulran", "kitti360", "mcd")}
    print("cache keys:", keys, flush=True)
    for s, want in EXPECTED_KEYS.items():
        assert keys[s] == want, (s, keys[s], want)
    assert keys["kitti360"] == keys["kitti"]
    assert keys["mcd"] not in EXPECTED_KEYS.values()
    print("SELFTEST cache keys match committed pd2026 val caches", flush=True)

    hits = sorted(glob.glob(str(PD_CACHE / "cache_*_kitti_val_09.npz")))
    assert len(hits) == 1, hits
    ref = np.load(hits[0])
    loader = _make_loader("kitti", "09")
    kfs = _build_operating_cache(cfg, "kitti", "09", loader, None, device,
                                 max_scans=300)
    n = len(kfs)
    scan_ids = np.array([kf.scan_id for kf in kfs])
    desc = np.array([kf.descriptor for kf in kfs])
    ent = np.array([kf.spectral_entropy for kf in kfs])
    same_ids = bool((scan_ids == ref["scan_ids"][:n]).all())
    d_desc = float(np.abs(desc - ref["descriptors"][:n]).max())
    d_ent = float(np.abs(ent - ref["spectral_entropies"][:n]).max())
    print(f"SELFTEST kitti/09 first 300 scans -> {n} keyframes vs committed "
          f"pd2026 val cache: scan_ids match={same_ids} "
          f"max|desc diff|={d_desc:.6g} max|entropy diff|={d_ent:.6g}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*", default=None,
                    help="sequences like mcd/tuhh_night_09 kitti360/0006 "
                         "(one heavy sequence per process!)")
    ap.add_argument("--manifest", action="store_true",
                    help="write scripts/_pd_test_cache_manifest.json and exit")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default=None,
                    help="descriptor-encoder device (default cuda if "
                         "available, else cpu; layouts always cpu)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.manifest:
        write_manifest()
        return
    if args.selftest:
        selftest(args.device)
        return
    if not args.seqs:
        ap.error("--seqs is required (or use --manifest/--selftest)")
    todo = [tuple(s.split("/", 1)) for s in args.seqs]
    known = set(TEST_SEQS)
    for sensor, seq in todo:
        assert (sensor, seq) in known, f"not in the frozen test roster: {sensor}/{seq}"
    for sensor, seq in todo:
        print(f"=== {sensor}/{seq} ===", flush=True)
        build_sequence(sensor, seq, args.device)
    print("PD_TEST_CACHES_DONE", flush=True)


if __name__ == "__main__":
    main()
