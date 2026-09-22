#!/usr/bin/env python3
"""HeLiOS descriptor extraction on the pd2026 42-seq roster (single stage).

Zero-shot HeLiOS (ICRA'25) baseline: raw keyframe point clouds -> official
HeLiOS sparse-voxel pipeline + model, per the working smoke contract of
scripts/_helios_infer_smoke.py (model built via their config/model_helios.txt
+ misc.utils.ModelParams + models.model_factory; weights
_external/HeLiOS/weights/HeLiOS.pth, torch.load weights_only=True, strict;
forward returns dict with output['global'] = (1, 8448) L2-normalized).

Input pipeline (their official eval contract), one cloud per forward:

    raw scan via our loaders (the same make_loader routing
    scripts/_imlpr_riv_lib.py uses — imported from it) -> XYZ only float32
    (intensity is NOT used; features are all-ones) -> radial crop
    0.5 m < r < 100 m (their HeLiPR toolbox crop radius) -> scale x0.01
    (coords in [-1, 1]) -> params.quantizer = CartesianQuantizer(0.01)
    (ME.utils.sparse_quantize) -> ME.utils.batched_coordinates ->
    model -> desc (8448,) float32.

DOMAIN NOTE (disclosed zero-shot convention): descriptors are computed on
single-keyframe scans, NOT the HeLiPR-toolbox accumulated submaps HeLiOS was
trained on. Input is in the SENSOR frame: their eval pickles reference the
untransformed /LiDAR/ scans — the LiDAR_transformed folder is
training-overlap-only (see _external/HeLiOS/datasets/preprocess/
generate_training_sets.py:47, pointcloud_fols = "/LiDAR/").

FROZEN descriptor interface (per sequence; same schema as the ImLPR
desc_{sensor}_{seq}.npz produced by scripts/_imlpr_desc.py):
  _external/imlpr_data_pd2026/desc_helios_{sensor}_{seq}.npz
      desc      (n, 8448) float16, L2-normalized before the fp16 cast
      scan_ids  (n,) int64 == the pd2026 operating-cache scan_ids
  _external/imlpr_data_pd2026/desc_helios_{sensor}_{seq}.meta.json
      weights sha256, input-pipeline/domain notes, train_overlap flag
      (HeLiOS was TRAINED on HeLiPR DCC04-06/KAIST04-06/Riverside04-06
      = 9 pd TEST sequences; aed302d lower-bound convention, as in
      scripts/_imlpr_desc.py).

One heavy sequence per worker: the parent process spawns one subprocess per
sequence (model + GPU state fully released between sequences); resumable —
sequences whose npz exists are skipped unless --overwrite.

Usage (inside docker ring_baseline):
    python3 scripts/_helios_desc.py --seqs kitti/05 [--overwrite]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from _imlpr_riv_lib import (  # noqa: E402
    OUT_ROOT, TRAIN_OVERLAP, load_cache_scan_ids, make_loader, resolve_seqs)

HELIOS = REPO / "_external" / "HeLiOS"
DEFAULT_WEIGHTS = HELIOS / "weights" / "HeLiOS.pth"
MODEL_CONFIG = HELIOS / "config" / "model_helios.txt"
DESC_DIM = 8448          # num_clusters 64 * cluster_dim 128 + token 256
CROP_RADIUS_M = 100.0    # HeLiPR toolbox crop; coords scaled by 1/100
MIN_RANGE_M = 0.5        # near-return filter (scripts/_helios_infer_smoke.py)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(weights: Path, device: str):
    """Official HeLiOS construction, exactly as scripts/_helios_infer_smoke.py:
    config/model_helios.txt -> ModelParams -> model_factory, strict load."""
    import torch

    for p in (str(HELIOS),
              str(HELIOS / "docker" / "thirdparty" / "SparseTransformer")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from misc.utils import ModelParams
    from models.model_factory import model_factory

    torch.manual_seed(0)
    params = ModelParams(str(MODEL_CONFIG))
    model = model_factory(params)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), params


def crop_scale(points_xyzi: np.ndarray) -> np.ndarray:
    """(N,>=3) -> XYZ float32, radial crop 0.5..100 m, scaled to [-1, 1]."""
    xyz = np.ascontiguousarray(points_xyzi[:, :3], dtype=np.float32)
    rng = np.linalg.norm(xyz, axis=1)
    return xyz[(rng > MIN_RANGE_M) & (rng < CROP_RADIUS_M)] / CROP_RADIUS_M


def encode_sequence(entry: dict, args) -> None:
    import MinkowskiEngine as ME
    import torch
    import torch.nn.functional as F

    sensor, seq = entry["sensor"], entry["sequence"]
    scan_ids = load_cache_scan_ids(entry)
    loader = make_loader(sensor, seq)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Path(args.weights)
    model, params = load_model(weights, device)

    n = len(scan_ids)
    desc = np.zeros((n, DESC_DIM), dtype=np.float32)
    t0 = time.time()
    for pos, sid in enumerate(scan_ids):
        xyz = crop_scale(loader[int(sid)]["points"])
        coords, _ = params.quantizer(torch.from_numpy(xyz))
        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        batch = {"coords": bcoords.to(device), "features": feats.to(device)}
        with torch.no_grad():
            out = model(batch)
        desc[pos] = F.normalize(out["global"], p=2, dim=1)[0].cpu().numpy()
        if (pos + 1) % 200 == 0:
            print(f"  [{sensor}/{seq}] {pos + 1}/{n} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    t_total = time.time() - t0

    norms = np.linalg.norm(desc, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (norms.min(), norms.max())
    desc16 = (desc / norms[:, None]).astype(np.float16)

    out_npz = OUT_ROOT / f"desc_helios_{sensor}_{seq}.npz"
    tmp = OUT_ROOT / f".tmp_desc_helios_{sensor}_{seq}.npz"
    np.savez(tmp, desc=desc16, scan_ids=scan_ids)
    os.replace(tmp, out_npz)  # atomic: a killed worker never leaves a truncated npz
    meta = {
        "method": "helios",
        "sensor": sensor,
        "sequence": seq,
        "n_frames": int(n),
        "desc_dim": DESC_DIM,
        "weights": str(weights.relative_to(REPO)),
        "weights_sha256": sha256_file(weights),
        "model_config": str(MODEL_CONFIG.relative_to(REPO)),
        "input_pipeline": (
            "raw keyframe XYZ float32 (intensity unused, features all-ones) "
            f"-> radial crop {MIN_RANGE_M} m < r < {CROP_RADIUS_M:.0f} m "
            "-> scale x0.01 (coords in [-1,1]) -> CartesianQuantizer(0.01) "
            "(ME.sparse_quantize) -> ME.batched_coordinates, one cloud per "
            "forward (scripts/_helios_infer_smoke.py contract)"),
        "input_domain_note": (
            "single-keyframe scans, NOT HeLiPR-toolbox accumulated submaps; "
            "sensor-frame input (their eval pickles reference the "
            "untransformed /LiDAR/ scans — LiDAR_transformed is "
            "training-overlap-only; _external/HeLiOS/datasets/preprocess/"
            "generate_training_sets.py:47 pointcloud_fols='/LiDAR/'); "
            "disclosed zero-shot domain note"),
        "operating_cache": entry["expected"]["operating_cache"],
        "weight_convention": (
            "official ICRA'25 HeLiOS.pth used ZERO-SHOT (no pd2026 "
            "retraining); HeLiOS was trained on HeLiPR DCC04-06/KAIST04-06/"
            "Riverside04-06, which overlap 9 pd TEST sequences — on those, "
            "recall is train-contaminated (optimistic upper bound), flagged "
            "per sequence (aed302d lower-bound convention)"),
        "train_overlap": (sensor, seq) in TRAIN_OVERLAP,
        "zero_shot_wrt_pd_train": True,
        "encode_s_per_frame": round(t_total / n, 4),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_meta = OUT_ROOT / f"desc_helios_{sensor}_{seq}.meta.json"
    out_meta.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[{sensor}/{seq}] desc {desc16.shape} -> {out_npz} "
          f"({t_total / n * 1e3:.0f} ms/frame)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="subset like kitti/05 (default: all 42)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker is not None:
        # subprocess mode: exactly one sequence, model lives and dies here
        [entry] = resolve_seqs([args.worker])
        encode_sequence(entry, args)
        return

    for entry in resolve_seqs(args.seqs):
        tag = f"{entry['sensor']}/{entry['sequence']}"
        out_npz = OUT_ROOT / f"desc_helios_{entry['sensor']}_{entry['sequence']}.npz"
        if out_npz.is_file() and not args.overwrite:
            print(f"[skip] {out_npz} exists", flush=True)
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", tag,
               "--weights", args.weights]
        print(f"[spawn] {tag}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"worker failed for {tag} (rc={r.returncode})")


if __name__ == "__main__":
    main()
