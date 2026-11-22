#!/usr/bin/env python3
"""Train the DiffAttnConv refiner from per-sequence training files.

    python3 tools/train.py --inputs TRAIN_DIR --seed 4 --out weights_s4.pth

TRAIN_DIR holds one ``<name>.npz`` per training sequence in the shared per-sequence format (``nsd/__init__.py``):
d, z, poses, poses_gt and, for sequences whose poses are not LiDAR poses, ``lidar_to_pose`` (4x4; the KITTI
calibration ``Tr`` for KITTI camera poses). The reported model used the dataset reference poses of the training
sequences as ``poses`` (so ``poses`` = ``poses_gt``); labels always come from ``poses_gt``.

The sequence order matters (it fixes the sample order); the default is the order of the reported runs.
Fixed training pairs are cached next to the inputs as ``<name>.pairs.npz`` unless --no-cache.
The reported runs used a deterministic CUDA setup (deterministic algorithms, TF32 off); training is bitwise
reproducible on the same GPU type.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nsd.model import build_graph  # noqa: E402
from nsd.train import UPDATES, build_pool, train  # noqa: E402

ORDER = ["kitti_01", "kitti_02", "kitti_05", "kitti_06", "helipr_Bridge01", "helipr_Bridge02", "helipr_Bridge03",
         "helipr_Bridge04"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--sequences", nargs="*", default=ORDER)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--updates", type=int, default=UPDATES)
    ap.add_argument("--assets", type=Path, default=ROOT / "assets")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)

    matching = json.loads((a.assets / "matching.json").read_text())
    loss_calibration = json.loads((a.assets / "loss_calibration.json").read_text())
    omega = np.load(a.assets / "remetric_weights.npy")
    whitening = torch.from_numpy(np.load(a.assets / "whitening_matrix.npy"))
    fw = matching["fusion_weights"]
    fusion = np.array([fw["key"], fw["aux"], fw["spectral_alignment"], fw["column_alignment"]], np.float32)

    graphs, pools = [], []
    for name in a.sequences:
        data = dict(np.load(a.inputs / f"{name}.npz"))
        graphs.append(build_graph(data["d"], data["z"], data["poses"], data.get("lidar_to_pose"), a.device))
        cache = a.inputs / f"{name}.pairs.npz"
        if cache.exists() and not a.no_cache:
            pools.append(dict(np.load(cache)))
        else:
            pools.append(build_pool(data["d"], data["z"], data["poses"], data["poses_gt"][:, :3, 3], omega, fusion))
            if not a.no_cache:
                np.savez(cache, **pools[-1])
        print(f"{name}: {len(data['d'])} keyframes, {int(pools[-1]['eligible'].sum())} pairs "
              f"({int(pools[-1]['correct'].sum())} positive)", flush=True)

    tick = time.time()

    def log(step: int, loss: float, grad: float) -> None:
        if step % 64 == 0:
            print(f"update {step}: loss {loss:.5f}  grad-norm {grad:.5f}  ({time.time() - tick:.1f} s)", flush=True)

    net = train(graphs, pools, a.seed, whitening, matching["whitened_mix"], matching["score_calibration"],
                loss_calibration, a.updates, log)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
