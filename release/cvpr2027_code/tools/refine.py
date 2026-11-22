#!/usr/bin/env python3
"""Add the refined coefficients z_refined to encoded sequence files.

    python tools/refine.py --inputs encoded/ --weights assets/weights/s4.pth --out refined_s4/ [--device cuda]

Each input '<dataset>_<sequence>.npz' (written by tools/encode_sequence.py) is copied to the output
folder with one extra field, z_refined complex64 [N, 16, 8], computed by the trained DiffAttnConv
refiner (nsd.model) from d, z and the odometry poses. Test-time poses are LiDAR odometry poses, so no
extrinsic is needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nsd import deterministic  # noqa: E402
from nsd.model import load_refiner, refine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, type=Path, help="folder of encoded .npz files")
    ap.add_argument("--weights", required=True, type=Path, help="released refiner weights (.pth)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    deterministic()
    net = load_refiner(args.weights, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.inputs.glob("*.npz")):
        with np.load(path) as s:
            data = {k: s[k] for k in s.files}
        data["z_refined"] = refine(net, data["d"], data["z"], data["poses"])
        np.savez_compressed(args.out / path.name, **data)
        print(path.stem, len(data["d"]), "keyframes", flush=True)


if __name__ == "__main__":
    main()
