"""Compute AliasRate, including yaw-conditioned splits, on cached descriptors.

Definition (paper Eq. 16):
    AliasRate_{eps,gamma} = Pr( ||d_i - d_j||_2 <= eps  |  ||p_i - p_j||_2 >= gamma )

Yaw-conditioned splits (this script's contribution to v0.7+):
    AliasRate^forward = AliasRate restricted to |Delta yaw_ij| <= 30 deg
    AliasRate^reverse = AliasRate restricted to |Delta yaw_ij| > 90 deg

Inputs
------
- Per-dataset .npz cache (default produced by train_multi_dataset.py):
    keys = {descriptors (N, D), poses (N, 4, 4), ...}
- The default cache key (range_image v0.6/v0.7) for KITTI/NCLT/HeLiPR is
  `056e0a02`; MulRan val cache uses `b714414b`. Both are queried automatically
  by sequence name unless --cache-key overrides.
- For refined (NSD+GNN) descriptors, point --refined-dir at a directory of
  per-dataset .npz files containing key 'embeddings'.

Output
------
results/aliasrate_yaw_split.json with structure:
    { "<dataset>": {"raw": {"all": x, "forward": y, "reverse": z, "n_far": ...},
                    "gnn": {"all": x, ...}|null} }

Usage (Docker)
--------------
    python3 scripts/compute_aliasrate.py \
        --output results/aliasrate_yaw_split.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


SEQUENCES = [
    "KITTI_00", "KITTI_05", "KITTI_08",
    "NCLT_2012-01-08", "NCLT_2013-01-10",
    "HeLiPR_Town01",
    "MulRan_DCC03", "MulRan_KAIST03", "MulRan_Riverside03",
]

# Mapping: display name -> (cache_dir prefix slug, default cache_key).
# Slugs match how train_multi_dataset.py names cache files
# (e.g. cache_<key>_kitti_val_00.npz, cache_<key>_helipr_val_Town01.npz, ...).
CACHE_SLUG = {
    "KITTI_00": "kitti_val_00",
    "KITTI_05": "kitti_val_05",
    "KITTI_08": "kitti_val_08",
    "NCLT_2012-01-08": "nclt_val_2012-01-08",
    "NCLT_2013-01-10": "nclt_val_2013-01-10",
    "HeLiPR_Town01": "helipr_val_Town01",
    "MulRan_DCC03": "mulran_val_DCC03",
    "MulRan_KAIST03": "mulran_val_KAIST03",
    "MulRan_Riverside03": "mulran_val_Riverside03",
}
DEFAULT_CACHE_KEYS = {
    "KITTI_00": "056e0a02", "KITTI_05": "056e0a02", "KITTI_08": "056e0a02",
    "NCLT_2012-01-08": "056e0a02", "NCLT_2013-01-10": "056e0a02",
    "HeLiPR_Town01": "056e0a02",
    "MulRan_DCC03": "b714414b", "MulRan_KAIST03": "b714414b",
    "MulRan_Riverside03": "b714414b",
}


def yaw_from_pose(R: np.ndarray) -> np.ndarray:
    """Extract yaw (rotation about z) from (N, 4, 4) or (N, 3, 3) SE3/SO3 stack."""
    if R.shape[-2:] == (4, 4):
        R = R[..., :3, :3]
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def load_cache(cache_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(cache_path)
    return {"descriptors": data["descriptors"], "poses": data["poses"]}


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


def sample_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> tuple:
    """Uniformly sample n_pairs distinct (i, j) pairs with i != j."""
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    # Sample with replacement and reject self-pairs; budget is generous.
    target = n_pairs
    out_i = []
    out_j = []
    n_have = 0
    while n_have < target:
        batch = max(target - n_have, 1024)
        i = rng.integers(0, n, size=batch)
        j = rng.integers(0, n, size=batch)
        keep = i != j
        out_i.append(i[keep])
        out_j.append(j[keep])
        n_have += int(keep.sum())
        if n_have == 0 and n == 1:
            break
    i = np.concatenate(out_i)[:target]
    j = np.concatenate(out_j)[:target]
    return i, j


def aliasrate_with_yaw(
    descriptors: np.ndarray,
    poses: np.ndarray,
    eps: float,
    gamma: float,
    n_pairs: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Compute AliasRate (overall + forward + reverse yaw bins) on sampled pairs."""
    d = l2_normalize_rows(descriptors.astype(np.float64))
    pos = poses[:, :3, 3]  # (N, 3)
    # Use 2D xy distance (consistent with paper Eq. 16 in driving setting).
    pos_xy = pos[:, :2]
    yaw = yaw_from_pose(poses)

    n = d.shape[0]
    i, j = sample_pairs(n, n_pairs, rng)
    if i.size == 0:
        return {"all": float("nan"), "forward": float("nan"), "reverse": float("nan"),
                "n_far": 0, "n_far_forward": 0, "n_far_reverse": 0,
                "n_collide": 0, "n_collide_forward": 0, "n_collide_reverse": 0}

    geo = np.linalg.norm(pos_xy[i] - pos_xy[j], axis=1)
    desc = np.linalg.norm(d[i] - d[j], axis=1)
    dyaw = np.abs(np.degrees(np.arctan2(np.sin(yaw[i] - yaw[j]),
                                        np.cos(yaw[i] - yaw[j]))))

    far = geo >= gamma  # geographically distant pairs
    far_fwd = far & (dyaw <= 30.0)
    far_rev = far & (dyaw > 90.0)

    collide = (desc <= eps) & far
    collide_fwd = collide & far_fwd
    collide_rev = collide & far_rev

    def _rate(num: int, den: int) -> float:
        return float(num / den) if den > 0 else float("nan")

    return {
        "all": _rate(int(collide.sum()), int(far.sum())),
        "forward": _rate(int(collide_fwd.sum()), int(far_fwd.sum())),
        "reverse": _rate(int(collide_rev.sum()), int(far_rev.sum())),
        "n_far": int(far.sum()),
        "n_far_forward": int(far_fwd.sum()),
        "n_far_reverse": int(far_rev.sum()),
        "n_collide": int(collide.sum()),
        "n_collide_forward": int(collide_fwd.sum()),
        "n_collide_reverse": int(collide_rev.sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("data/preprocessed"))
    ap.add_argument("--cache-key", type=str, default=None,
                    help="Override default per-dataset cache keys with one key.")
    ap.add_argument("--refined-dir", type=Path, default=None,
                    help="Directory of <dataset>.npz files with key 'embeddings' "
                         "for NSD+GNN refined descriptors. If omitted, gnn row is null.")
    ap.add_argument("--eps", type=float, default=0.1,
                    help="Descriptor-space collision threshold (paper default 0.1).")
    ap.add_argument("--gamma", type=float, default=25.0,
                    help="Geographic far-pair threshold in meters (paper default 25).")
    ap.add_argument("--n-pairs", type=int, default=200_000,
                    help="Number of (i, j) pairs to sample per dataset.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out: Dict[str, Dict[str, Optional[Dict]]] = {}

    for dataset in SEQUENCES:
        slug = CACHE_SLUG[dataset]
        key = args.cache_key or DEFAULT_CACHE_KEYS[dataset]
        cache_path = args.cache_dir / f"cache_{key}_{slug}.npz"
        if not cache_path.exists():
            print(f"[skip] {dataset}: cache missing ({cache_path})")
            continue

        cache = load_cache(cache_path)
        raw = aliasrate_with_yaw(cache["descriptors"], cache["poses"],
                                 args.eps, args.gamma, args.n_pairs, rng)

        gnn = None
        if args.refined_dir is not None:
            ref_path = args.refined_dir / f"{dataset}.npz"
            if ref_path.exists():
                refined = np.load(ref_path)
                gnn = aliasrate_with_yaw(refined["embeddings"], cache["poses"],
                                         args.eps, args.gamma, args.n_pairs, rng)
            else:
                print(f"[warn] {dataset}: refined missing ({ref_path}); gnn=null")

        out[dataset] = {"raw": raw, "gnn": gnn}
        print(f"[ok] {dataset}  raw all={raw['all']:.4f}  fwd={raw['forward']:.4f}  rev={raw['reverse']:.4f}"
              + (f"   gnn all={gnn['all']:.4f}" if gnn else ""))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"eps": args.eps, "gamma": args.gamma, "n_pairs": args.n_pairs,
                   "datasets": out}, f, indent=2)
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
