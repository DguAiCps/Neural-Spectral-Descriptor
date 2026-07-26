#!/usr/bin/env python3
"""Point-cloud yaw stability of the deployed 288D key, integer vs non-integer bins.

The paper's Table (empirical yaw stability) rotates single scans by 15-degree
offsets, all integer multiples of the 1-degree azimuth bin, so 1.000 stability
restates the DFT shift theorem. This script measures what Prop. 1(ii) actually
bounds: descriptor drift under *non-integer-bin* yaw of the raw point cloud,
and places it on the scale of real revisit spread.

Protocol (KITTI 00, HDL-64E):
  - sample scans at a fixed stride; rotate raw points about z by Delta-phi
  - integer-bin angles {1, 15, 90, 178} deg and
    non-integer   {0.5, 7.3, 45.5, 177.5} deg
  - encode both with the deployed 288D no_interdiff encoder (encode_points path)
  - report cosine similarity + L2 drift of the unit-normalized key
  - reference scale: same-place key distances of genuine revisit pairs (<5 m,
    >=30-frame gap) encoded through the identical path

Run: docker run --rm --gpus all -v $REPO:/ws -v /mnt/d/NSD_datasets:/data -w /ws \
       nvcr.io/nvidia/pyg:26.01-py3 python scripts/_verify_noninteger_yaw.py
Output: results/_alias_source/noninteger_yaw.json
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset  # noqa: E402
from run_kitti_operating_point import _make_encoder  # noqa: E402

VELO = Path("/data/kitti/dataset/sequences/00/velodyne")
POSES = Path("/data/kitti/dataset/poses/00.txt")
CALIB = Path("/data/kitti/dataset/sequences/00/calib.txt")
OUT = REPO / "results/_alias_source"
STRIDE = 30
ANGLES_INT = [1.0, 15.0, 90.0, 178.0]
ANGLES_NONINT = [0.5, 7.3, 45.5, 177.5]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_scan(idx):
    pts = np.fromfile(VELO / f"{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    return pts


def rot_z(pts, deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    out = pts.copy()
    out[:, 0] = c * pts[:, 0] - s * pts[:, 1]
    out[:, 1] = s * pts[:, 0] + c * pts[:, 1]
    return out


def unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml"))
    cfg288 = _apply_encoder_preset(copy.deepcopy(cfg), "no_interdiff")
    enc = _make_encoder(cfg288, DEVICE)

    def encode(pts):
        with torch.no_grad():
            h = enc.encode_points(pts)
        return unit(h.detach().cpu().numpy().astype(np.float64))

    n_scans = len(list(VELO.glob("*.bin")))
    idxs = list(range(0, n_scans, STRIDE))
    print(f"{len(idxs)} scans, device={DEVICE}", flush=True)

    res = {str(a): {"cos": [], "l2": []} for a in ANGLES_INT + ANGLES_NONINT}
    for i in idxs:
        pts = load_scan(i)
        d0 = encode(pts)
        for a in ANGLES_INT + ANGLES_NONINT:
            da = encode(rot_z(pts, a))
            res[str(a)]["cos"].append(float(d0 @ da))
            res[str(a)]["l2"].append(float(np.linalg.norm(d0 - da)))

    # reference: genuine revisit spread through the same encode path
    P = np.loadtxt(POSES).reshape(-1, 3, 4)[:, :, 3]
    # velodyne->cam translation is irrelevant at 5 m threshold scale; use cam poses
    pairs = []
    for j in range(30, len(P), 5):
        g = np.linalg.norm(P[: j - 30 + 1] - P[j], axis=1)
        h = np.where(g < 5.0)[0]
        if len(h):
            pairs.append((j, int(h[0])))
    pairs = pairs[::max(1, len(pairs) // 150)]
    rev_l2, rev_cos = [], []
    for a, b in pairs:
        da, db = encode(load_scan(a)), encode(load_scan(b))
        rev_l2.append(float(np.linalg.norm(da - db)))
        rev_cos.append(float(da @ db))

    summary = {
        "n_scans": len(idxs), "stride": STRIDE,
        "angles": {
            a: {
                "kind": "integer" if float(a) == int(float(a)) else "non-integer",
                "cos_mean": float(np.mean(v["cos"])),
                "cos_min": float(np.min(v["cos"])),
                "l2_mean": float(np.mean(v["l2"])),
                "l2_p95": float(np.quantile(v["l2"], 0.95)),
            } for a, v in res.items()
        },
        "revisit_reference": {
            "n_pairs": len(pairs),
            "l2_mean": float(np.mean(rev_l2)), "l2_median": float(np.median(rev_l2)),
            "cos_mean": float(np.mean(rev_cos)),
        },
    }
    json.dump(summary, open(OUT / "noninteger_yaw.json", "w"), indent=1)
    for a, v in summary["angles"].items():
        print(f"  {a:>6} deg ({v['kind']:<11}) cos={v['cos_mean']:.6f} "
              f"l2={v['l2_mean']:.5f} p95={v['l2_p95']:.5f}", flush=True)
    print(f"  revisit ref: l2 mean {summary['revisit_reference']['l2_mean']:.4f} "
          f"median {summary['revisit_reference']['l2_median']:.4f}")
    print("wrote", OUT / "noninteger_yaw.json")


if __name__ == "__main__":
    main()
