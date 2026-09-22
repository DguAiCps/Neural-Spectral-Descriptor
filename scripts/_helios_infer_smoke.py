#!/usr/bin/env python3
"""Smoke test: official HeLiOS (ICRA'25) inference on torch 2.10 / CUDA 13 / sm_120.

Loads _external/HeLiOS/weights/HeLiOS.pth, runs one forward pass on a real KITTI
scan, prints the descriptor shape and norm.

Run inside the `ring_baseline` docker container:

    docker exec ring_baseline python \
        /workspace/Neural-Spectral-Codec/scripts/_helios_infer_smoke.py

Prerequisites inside the container (scripts/_helios_port_setup.sh reproduces
the full recipe):
  - sptr_cuda      built from _external/HeLiOS/docker/thirdparty/SparseTransformer
  - torch_scatter  built from source (github.com/rusty1s/pytorch_scatter)
  - torch_cluster  built from source (github.com/rusty1s/pytorch_cluster)
  - MinkowskiEngine 0.5.4 built from the vendored copy + 3 CUDA-13 patches
  - torch_geometric, timm (stock NGC container already has them)

Input convention expected by HeLiOS (matches their HeLiPR preprocessing):
  - XYZ only, float32; intensity is NOT used (features are all-ones).
  - Points cropped to a 100 m radius and scaled by 1/100 -> coords in [-1, 1].
  - CartesianQuantizer, step 0.01 (= 1 m voxels at real scale) dedupes points.
"""
import os
import sys

import numpy as np
import torch

REPO = "/workspace/Neural-Spectral-Codec"
if not os.path.isdir(REPO):  # fallback when run outside the container
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELIOS = os.path.join(REPO, "_external", "HeLiOS")

sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HELIOS)
# The `sptr` python package is used straight from the source tree (upstream
# installs only the sptr_cuda extension), same as HeLiOS eval/evaluate.py does.
sys.path.insert(0, os.path.join(HELIOS, "docker", "thirdparty", "SparseTransformer"))

import MinkowskiEngine as ME  # noqa: E402

from misc.utils import ModelParams  # noqa: E402  (HeLiOS repo)
from models.model_factory import model_factory  # noqa: E402  (HeLiOS repo)
from data.kitti_loader import KITTILoader  # noqa: E402  (our repo, src/)

CROP_RADIUS_M = 100.0  # HeLiOS scales clouds by 1/100 ("0.01 is from 1/crop_size")
KITTI_ROOT = "/workspace/data/kitti/dataset"
SEQ, FRAME = "05", 100


def main():
    device = "cuda"
    torch.manual_seed(0)

    params = ModelParams(os.path.join(HELIOS, "config", "model_helios.txt"))
    model = model_factory(params)
    weights = os.path.join(HELIOS, "weights", "HeLiOS.pth")
    state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: HeLiOS ({n_params / 1e6:.2f} M params), weights: {weights}")

    # Real scan through our loader: KITTI seq 05, frame 100.
    loader = KITTILoader(KITTI_ROOT, SEQ)
    pc = loader.get_point_cloud(FRAME)  # (N, 4) x, y, z, intensity
    xyz = pc[:, :3].astype(np.float32)
    rng = np.linalg.norm(xyz, axis=1)
    xyz = xyz[(rng > 0.5) & (rng < CROP_RADIUS_M)] / CROP_RADIUS_M
    pc_t = torch.tensor(xyz, dtype=torch.float32)

    coords, _ = params.quantizer(pc_t)  # CartesianQuantizer(0.01)
    bcoords = ME.utils.batched_coordinates([coords])
    feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
    batch = {"coords": bcoords.to(device), "features": feats.to(device)}

    with torch.no_grad():
        out = model(batch)
    g = out["global"]

    print(f"input: KITTI {SEQ}/{FRAME:06d}: {pc.shape[0]} pts -> "
          f"{xyz.shape[0]} in crop -> {coords.shape[0]} voxels")
    print(f"descriptor shape: {tuple(g.shape)}")
    print(f"descriptor L2 norm: {g.norm().item():.6f}")
    print(f"descriptor abs-max: {g.abs().max().item():.6f}")
    assert torch.isfinite(g).all(), "descriptor contains non-finite values"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
