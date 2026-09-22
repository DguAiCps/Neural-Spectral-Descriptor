#!/usr/bin/env python3
"""ImLPR descriptor extraction on the pd2026 RIV caches (stage 2 of 2).

Consumes the frozen RIV cache written by scripts/_imlpr_riv_lib.py
(_external/imlpr_data_pd2026/{sensor}_{seq}/%06d.npy, (H0,1022,3) float32
0..255) and runs the OFFICIAL ImLPR input pipeline + model
(_external/ImLPR/eval/evaluate.py:70-118 semantics, replicated exactly):

    npy -> cylindrical pad 28 cols each side (right block prepended, left
    appended) -> if H != 126: cv2.resize (W stays, H->126, INTER_LINEAR)
    -> CHW float32 /255.0 -> ImLPR forward -> tuple[1] = L2-normalized
    (B, 8448) global descriptor.

Model = official ImLPR(params) construction with the verified shim params
(dinov2_vits14, num_trainable_blocks=2, adapter_frequency=3, num_channels=384,
num_clusters=64, cluster_dim=128, token_dim=256); weights
_external/ImLPR/weights/ImLPR_default.pth, 'module.' prefixes stripped,
load_state_dict strict=False asserted to have 0 missing / 0 unexpected keys.

FROZEN descriptor interface (per sequence):
  _external/imlpr_data_pd2026/desc_{sensor}_{seq}.npz
      desc      (n, 8448) float16, L2-normalized before the fp16 cast
      scan_ids  (n,) int64 == the pd2026 operating-cache scan_ids
  _external/imlpr_data_pd2026/desc_{sensor}_{seq}.meta.json
      weights sha256, model params, train_overlap flag (ImLPR was TRAINED on
      HeLiPR DCC04-06/KAIST04-06/Riverside04-06 = 9 pd TEST sequences;
      aed302d lower-bound convention, as in scripts/_pdb_ringsharp_eval.py).

One heavy sequence per worker: the parent process spawns one subprocess per
sequence (model + GPU state fully released between sequences).

Usage (inside docker ring_baseline):
    python3 scripts/_imlpr_desc.py --seqs kitti/05 [--overwrite] [--batch 16]
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
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from _imlpr_riv_lib import (  # noqa: E402
    OUT_ROOT, TRAIN_OVERLAP, load_cache_scan_ids, resolve_seqs, seq_dir)

DEFAULT_WEIGHTS = REPO / "_external" / "ImLPR" / "weights" / "ImLPR_default.pth"
MODEL_PARAMS = dict(model_name="dinov2_vits14", num_trainable_blocks=2,
                    adapter_frequency=3, num_channels=384, num_clusters=64,
                    cluster_dim=128, token_dim=256)
DESC_DIM = 8448  # token_dim 256 + num_clusters 64 * cluster_dim 128
PAD_COLS = 28    # official cylindrical padding
TARGET_H = 126   # official DINOv2 input rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(weights: Path, device: str):
    import torch

    imlpr_root = str(REPO / "_external" / "ImLPR")
    if imlpr_root not in sys.path:
        sys.path.insert(0, imlpr_root)
    from models.ImLPR import ImLPR

    model = ImLPR(SimpleNamespace(**MODEL_PARAMS))
    state = torch.load(str(weights), map_location="cpu")
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    res = model.load_state_dict(state, strict=False)
    assert not res.missing_keys and not res.unexpected_keys, (
        f"weight mismatch: missing={res.missing_keys} "
        f"unexpected={res.unexpected_keys}")
    return model.to(device).eval()


def prep_frame(npy_path: Path) -> np.ndarray:
    """Official eval pipeline: pad 28 cyl -> resize H->126 -> CHW /255."""
    import cv2

    img = np.load(npy_path)  # (H0, 1022, 3) float32 0..255
    left, right = img[:, :PAD_COLS], img[:, -PAD_COLS:]
    img = np.concatenate([right, img, left], axis=1)  # (H0, 1078, 3)
    if img.shape[0] != TARGET_H:
        img = cv2.resize(img, (img.shape[1], TARGET_H),
                         interpolation=cv2.INTER_LINEAR)
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


def encode_sequence(entry: dict, args) -> None:
    import torch
    import torch.nn.functional as F

    sensor, seq = entry["sensor"], entry["sequence"]
    riv_dir = seq_dir(sensor, seq)
    meta_path = riv_dir / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"[{sensor}/{seq}] RIV cache incomplete: no {meta_path} "
                         f"(run scripts/_imlpr_riv_lib.py first)")
    riv_meta = json.loads(meta_path.read_text())
    scan_ids = load_cache_scan_ids(entry)
    assert riv_meta["n_frames"] == len(scan_ids), (riv_meta["n_frames"], len(scan_ids))
    assert np.array_equal(np.asarray(riv_meta["scan_ids"], dtype=np.int64), scan_ids), \
        f"[{sensor}/{seq}] RIV meta scan_ids != operating-cache scan_ids"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Path(args.weights)
    model = load_model(weights, device)

    n = len(scan_ids)
    desc = np.zeros((n, DESC_DIM), dtype=np.float32)
    t0 = time.time()
    for i0 in range(0, n, args.batch):
        idx = range(i0, min(i0 + args.batch, n))
        batch = np.stack([prep_frame(riv_dir / f"{i:06d}.npy") for i in idx])
        x = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            _, d = model(x)  # (B, 8448) L2-normalized
        desc[idx.start:idx.stop] = F.normalize(d, p=2, dim=1).cpu().numpy()
        if (i0 // args.batch) % 20 == 0:
            print(f"  [{sensor}/{seq}] {idx.stop}/{n} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    t_total = time.time() - t0

    norms = np.linalg.norm(desc, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (norms.min(), norms.max())
    desc16 = (desc / norms[:, None]).astype(np.float16)

    out_npz = OUT_ROOT / f"{args.prefix}{sensor}_{seq}.npz"
    tmp = OUT_ROOT / f".tmp_{args.prefix}{sensor}_{seq}.npz"
    np.savez(tmp, desc=desc16, scan_ids=scan_ids)
    os.replace(tmp, out_npz)  # atomic: a killed worker never leaves a truncated npz
    meta = {
        "method": "imlpr",
        "sensor": sensor,
        "sequence": seq,
        "n_frames": int(n),
        "desc_dim": DESC_DIM,
        "weights": str(weights.relative_to(REPO)),
        "weights_sha256": sha256_file(weights),
        "model_params": MODEL_PARAMS,
        "input_pipeline": ("RIV npy (H0,1022,3) 0..255 -> cyl pad 28 both sides "
                           "-> resize H->126 INTER_LINEAR -> CHW /255 "
                           "(official eval/evaluate.py)"),
        "riv_meta": str(meta_path.relative_to(REPO)),
        "operating_cache": entry["expected"]["operating_cache"],
        "weight_convention": (
            "official CoRL'25 ImLPR_default.pth used ZERO-SHOT (no pd2026 "
            "retraining); ImLPR was trained on HeLiPR DCC04-06/KAIST04-06/"
            "Riverside04-06, which overlap 9 pd TEST sequences — on those, "
            "recall is train-contaminated (optimistic upper bound), flagged "
            "per sequence (aed302d lower-bound convention)"),
        "train_overlap": (sensor, seq) in TRAIN_OVERLAP,
        "zero_shot_wrt_pd_train": True,
        "encode_s_per_frame": round(t_total / n, 4),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_meta = OUT_ROOT / f"{args.prefix}{sensor}_{seq}.meta.json"
    out_meta.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[{sensor}/{seq}] desc {desc16.shape} -> {out_npz} "
          f"({t_total / n * 1e3:.0f} ms/frame)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="subset like kitti/05 (default: all 42)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--prefix", default="desc_",
                    help="output npz filename prefix (e.g. desc_imlpr_pd_)")
    ap.add_argument("--batch", type=int, default=16)
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
        out_npz = OUT_ROOT / f"{args.prefix}{entry['sensor']}_{entry['sequence']}.npz"
        if out_npz.is_file() and not args.overwrite:
            print(f"[skip] {out_npz} exists", flush=True)
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", tag,
               "--weights", args.weights, "--batch", str(args.batch),
               "--prefix", args.prefix]
        print(f"[spawn] {tag}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"worker failed for {tag} (rc={r.returncode})")


if __name__ == "__main__":
    main()
