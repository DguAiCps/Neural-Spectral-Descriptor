#!/usr/bin/env python3
"""Official RING++ / RING (learning-free) evaluation on OUR val split.

Uses the official TRO'25 RINGSharp implementation of RING++ end to end
(glnet generate_feats LPD feature BEV -> torch-radon sinogram -> row-FFT
magnitude TIRING spectrum -> circular-correlation similarity, estimate_yaw),
scored under our protocol (query = revisit < 5 m with >= 30-keyframe gap,
Recall@1/5/10 via baselines/eval_utils). Ranking is EXHAUSTIVE circular
correlation over the whole intra-sequence database (their official PR
ranking; no coarse pre-filter).

Variants:
  --variant ringpp  (default) 6-channel LPD feature BEV  = RING++
  --variant ring    1-channel occupancy BEV              = RING

Sensor adaptation (documented): point clouds come from our loaders (z-up,
sensor origin); BEV bounds are +-70 m with the official above-sensor z-band
[1, 20] m (RING repo utils/config.py z_bound [1,20], NCLTPointCloudLoader
ground_plane_level=1 / zbound=20), grid 120x120 (official RING/RING++
resolution), applied uniformly to all four sensors.

Feature BEVs are cached per sequence under _external/ringsharp_data/ringpp_cache/.

Writes results/ringpp_eval.json (or ring_eval.json). Run inside docker:
  python scripts/_ringpp_eval.py --workers 5
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RINGSHARP = REPO / "_external" / "RINGSharp"
sys.path.insert(0, str(RINGSHARP))
sys.path.insert(1, str(REPO))
sys.path.insert(2, str(REPO / "src"))

from baselines.eval_utils import compute_recall_cosine_then_rerank  # noqa: E402

K_VALUES = [1, 5, 10]
CACHE_KEYS = {"kitti": "056e0a02", "nclt": "056e0a02",
              "helipr": "056e0a02", "mulran": "33919e6e"}
VAL_SEQS = [("kitti", "00"), ("kitti", "05"), ("kitti", "08"),
            ("nclt", "2012-01-08"), ("nclt", "2013-01-10"),
            ("helipr", "Town01"),
            ("mulran", "DCC03"), ("mulran", "KAIST03"), ("mulran", "Riverside03")]
BOUNDS = (-70.0, 70.0, -70.0, 70.0, 1.0, 20.0)   # official z-band, z-up
GRID = 120                                        # official RING/RING++ grid
CACHE_DIR = REPO / "_external" / "ringsharp_data" / "ringpp_cache"


def load_cache(dtype, seq):
    p = REPO / "data" / "preprocessed" / f"cache_{CACHE_KEYS[dtype]}_{dtype}_val_{seq}.npz"
    with np.load(p, allow_pickle=True) as c:
        return c["scan_ids"].astype(np.int64), c["poses"].astype(np.float64)


def _worker_init(dtype, seq):
    global _W_LOADER, _W_VARIANT
    from glnet.datasets.nsc.nsc_dataset import make_nsc_loader
    _W_LOADER = make_nsc_loader(dtype, seq, "/workspace/data")


def _bev_one(args):
    idx, scan_id, variant = args
    import torch as _t
    from glnet.utils.data_utils.point_clouds import generate_feats, generate_bev
    pts = np.ascontiguousarray(_W_LOADER[int(scan_id)]["points"][:, :3],
                               dtype=np.float32)
    with _t.no_grad():
        if variant == "ringpp":
            bev = generate_feats(pts, Z=1, Y=GRID, X=GRID, bounds=BOUNDS)
            bev = bev.reshape(-1, GRID, GRID)          # (6,120,120)
        else:
            bev = generate_bev(pts, Z=1, Y=GRID, X=GRID, bounds=BOUNDS)
            bev = bev.reshape(-1, GRID, GRID)          # (1,120,120)
    return idx, bev.numpy().astype(np.float16)


def compute_bevs(dtype, seq, scan_ids, variant, workers):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cpath = CACHE_DIR / f"{variant}_{dtype}_{seq}.npz"
    if cpath.exists():
        with np.load(cpath) as c:
            if len(c["bevs"]) == len(scan_ids):
                return c["bevs"]
    import multiprocessing as mp
    n = len(scan_ids)
    C = 6 if variant == "ringpp" else 1
    bevs = np.zeros((n, C, GRID, GRID), dtype=np.float16)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_worker_init,
                                      initargs=(dtype, seq)) as pool:
        args = [(i, scan_ids[i], variant) for i in range(n)]
        for m, (idx, bev) in enumerate(pool.imap_unordered(_bev_one, args,
                                                           chunksize=8)):
            bevs[idx] = bev
            if (m + 1) % 500 == 0:
                rate = (m + 1) / (time.time() - t0)
                print(f"    [{dtype}_{seq}] {m+1}/{n} BEVs "
                      f"({rate:.1f}/s, ETA {(n-m-1)/rate/60:.0f} min)", flush=True)
    np.savez_compressed(cpath, bevs=bevs)
    print(f"    cached {cpath} ({cpath.stat().st_size/1e6:.0f} MB)")
    return bevs


def compute_specs(bevs, device, batch=64):
    """Official TIRING: radon sinogram -> row-FFT magnitude (RINGPlusPlus.compute_spec)."""
    from torch_radon import ParallelBeam
    n_ang = bevs.shape[-2]
    angles = torch.FloatTensor(np.linspace(0, 2 * np.pi, n_ang).astype(np.float32))
    radon = ParallelBeam(bevs.shape[-1], angles)
    specs = torch.zeros(bevs.shape, dtype=torch.float16)
    for i0 in range(0, len(bevs), batch):
        x = torch.from_numpy(bevs[i0:i0 + batch].astype(np.float32)).to(device)
        with torch.no_grad():
            sino = radon.forward(x)
            m = torch.fft.fft2(sino, dim=-1, norm="ortho")
            spec = torch.sqrt(m.real ** 2 + m.imag ** 2 + 1e-15)
        specs[i0:i0 + batch] = spec.half().cpu()
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sequences", nargs="*", help="dtype/seq (default: all 9 val)")
    ap.add_argument("--variant", choices=["ringpp", "ring"], default="ringpp")
    ap.add_argument("--bev-only", action="store_true",
                    help="only build/refresh the per-sequence BEV caches (CPU), "
                         "skip the GPU spectrum/correlation scoring stage")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from glnet.models.utils import estimate_yaw   # official correlation

    targets = VAL_SEQS if not args.sequences else \
        [tuple(s.split("/")) for s in args.sequences]
    out_path = Path(args.out) if args.out else \
        REPO / "results" / f"{'ringpp' if args.variant == 'ringpp' else 'ring'}_eval.json"

    results, total_q = {}, 0
    for dtype, seq in targets:
        scan_ids, poses = load_cache(dtype, seq)
        if args.max_frames:
            scan_ids, poses = scan_ids[:args.max_frames], poses[:args.max_frames]
        print(f"[{dtype}_{seq}] {len(scan_ids)} keyframes ({args.variant})")
        bevs = compute_bevs(dtype, seq, scan_ids, args.variant, args.workers)
        if args.bev_only:
            continue
        specs = compute_specs(bevs, device)
        n = len(specs)
        # coarse descriptor only used to hand ALL candidates to rerank_fn
        glob = specs.float().sum(dim=-2).reshape(n, -1).numpy()
        glob /= (np.linalg.norm(glob, axis=1, keepdims=True) + 1e-12)

        def rerank_fn(qi, cand):
            cand = np.asarray(cand)
            qspec = specs[qi].float().to(device)
            dists = np.empty(len(cand), dtype=np.float64)
            bs = 256
            with torch.no_grad():
                for j0 in range(0, len(cand), bs):
                    js = cand[j0:j0 + bs]
                    cspec = specs[js].float().to(device)
                    _, scores, _ = estimate_yaw(
                        qspec.unsqueeze(0).expand(len(js), -1, -1, -1), cspec)
                    dists[j0:j0 + bs] = (1 - scores).cpu().numpy().reshape(-1)
            return cand[np.argsort(dists)]

        t0 = time.time()
        recalls, n_q = compute_recall_cosine_then_rerank(
            glob, rerank_fn, poses, k_values=K_VALUES, n_coarse=n)
        dt = time.time() - t0
        print(f"  [{dtype}_{seq}] queries={n_q} "
              + " ".join(f"R@{k}={recalls[k]:.4f}" for k in K_VALUES)
              + f" ({dt:.0f}s)")
        results[f"{dtype}_{seq}"] = {"n_queries": n_q,
                                     **{f"recall@{k}": recalls[k] for k in K_VALUES}}
        total_q += n_q
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    if total_q:
        mean = {f"recall@{k}": sum(r[f"recall@{k}"] * r["n_queries"]
                                   for r in results.values()
                                   if "n_queries" in r) / total_q
                for k in K_VALUES}
        results["_query_weighted_mean"] = {"n_queries": total_q, **mean}
        print("query-weighted mean: "
              + " ".join(f"R@{k}={mean[f'recall@{k}']:.4f}" for k in K_VALUES))
    results["_meta"] = {"variant": args.variant, "grid": GRID, "bounds": BOUNDS,
                        "ranking": "exhaustive official circular correlation",
                        "protocol": "revisit<5m, skip 30 frames, intra-sequence"}
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
