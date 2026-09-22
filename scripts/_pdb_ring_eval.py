#!/usr/bin/env python3
"""RING / RING++ (official TRO'25 RINGSharp impl) on the frozen pd 42-seq TEST roster.

pd-baseline (_pdb_) driver, additive to scripts/_ringpp_eval.py: same official
pipeline (glnet generate_feats LPD feature BEV / generate_bev occupancy BEV ->
torch-radon sinogram -> row-FFT magnitude TIRING spectrum -> exhaustive
circular-correlation ranking via estimate_yaw), same protocol (revisit < 5 m,
>= 30-keyframe gap, R@1/5/10 via baselines/eval_utils), same sensor adaptation
(z-up +-70 m, official above-sensor z-band [1, 20] m, 120x120 grid, uniform
across all six sensors).

Differences from scripts/_ringpp_eval.py (all additive):
  * roster + keyframes/poses come from scripts/_pd_test_cache_manifest.json
    (pd2026 operating caches), incl. kitti360 (src/data/kitti360_loader.py)
    and mcd (src/data/mcd_loader.py) loaders;
  * ONE worker pass builds BOTH variants' BEVs per keyframe (the 6-ch LPD
    feature BEV dominates at ~2.4-2.8 s/kf; the 1-ch occupancy BEV is nearly
    free), so ring and ringpp share a single per-sequence BEV cache;
  * resume-safe cache layout under _external/ringsharp_data/pdb_ring_cache/:
      chunks_{sensor}_{seq}/chunk_{i0:06d}.npz   1000-kf shards (scan_ids,
                                                 bev6, bev1), written tmp+rename
      bev_{sensor}_{seq}.npz                     consolidated (scan_ids,
                                                 bevs_ringpp, bevs_ring);
                                                 chunks removed afterwards
    A killed build resumes at chunk granularity; a finished sequence is a
    single npz keyed by exact scan_ids match. Projected size for the full
    197,977-kf roster: ~4 GB compressed (~40 GB raw fp16 upper bound).
  * per-sequence per-variant results:
      results/pd2026_baselines/{ring,ringpp}_{sensor}_{seq}.json
    (with --max-frames N a _mfN suffix keeps partial runs out of the
    canonical names).

One heavy sequence per process (memory cgroup): drive with one sensor/seq
argument per invocation. Run inside docker ring_baseline:
  python3 scripts/_pdb_ring_eval.py mcd/tuhh_night_09 --variant both --workers 5
  python3 scripts/_pdb_ring_eval.py --roster    # print the 42 sensor/seq ids
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = os.environ.get("NSD_DATA_ROOT", "/workspace/data")
RINGSHARP = REPO / "_external" / "RINGSharp"
sys.path.insert(0, str(RINGSHARP))
sys.path.insert(1, str(REPO))
sys.path.insert(2, str(REPO / "src"))

from baselines.eval_utils import compute_recall_cosine_then_rerank  # noqa: E402

K_VALUES = [1, 5, 10]
BOUNDS = (-70.0, 70.0, -70.0, 70.0, 1.0, 20.0)   # official z-band, z-up
GRID = 120                                        # official RING/RING++ grid
CHUNK = 1000                                      # BEV shard size (keyframes)
CACHE_DIR = REPO / "_external" / "ringsharp_data" / "pdb_ring_cache"
MANIFEST = REPO / "scripts" / "_pd_test_cache_manifest.json"
OUT_DIR = REPO / "results" / "pd2026_baselines"


def load_manifest():
    m = json.loads(MANIFEST.read_text())
    return {(s["sensor"], s["sequence"]): s["expected"]["operating_cache"]
            for s in m["sequences"]}


def load_cache(rel_cache_path):
    with np.load(REPO / rel_cache_path, allow_pickle=True) as c:
        return c["scan_ids"].astype(np.int64), c["poses"].astype(np.float64)


def make_loader(sensor, seq):
    """All six pd test sensors (mirrors scripts/_pd_build_test_caches.py)."""
    if sensor == "kitti":
        from data.kitti_loader import KITTILoader
        return KITTILoader(os.path.join(DATA_ROOT, "kitti", "dataset"), seq,
                           lazy_load=True)
    if sensor == "nclt":
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(os.path.join(DATA_ROOT, "nclt"), seq, lazy_load=True)
    if sensor == "helipr":
        from data.helipr_loader import HeLiPRLoader
        seq_path = os.path.join(DATA_ROOT, "helipr", seq)
        if not os.path.isdir(os.path.join(seq_path, "LiDAR")):
            seq_path = os.path.join(seq_path, seq)  # legacy double-nested layout
        return HeLiPRLoader(seq_path, lazy_load=True)
    if sensor == "mulran":
        from data.mulran_loader import MulRanLoader
        return MulRanLoader(os.path.join(DATA_ROOT, "mulran"), seq,
                            lazy_load=True)
    if sensor == "kitti360":
        from data.kitti360_loader import Kitti360Loader
        return Kitti360Loader(os.path.join(DATA_ROOT, "kitti360"), seq,
                              lazy_load=True)
    if sensor == "mcd":
        from data.mcd_loader import MCDLoader
        return MCDLoader(os.path.join(DATA_ROOT, "mcd_extracted"), seq,
                         lazy_load=True)
    raise ValueError(f"Unknown sensor: {sensor}")


def _worker_init(sensor, seq):
    global _W_LOADER
    # get_pointfeat prints per-scan timing lines; silence workers.
    sys.stdout = open(os.devnull, "w")
    _W_LOADER = make_loader(sensor, seq)


def _bev_one(args):
    """One keyframe -> (idx, 6-ch LPD feature BEV, 1-ch occupancy BEV)."""
    idx, scan_id = args
    import torch as _t
    from glnet.utils.data_utils.point_clouds import generate_feats, generate_bev
    pts = np.ascontiguousarray(_W_LOADER[int(scan_id)]["points"][:, :3],
                               dtype=np.float32)
    with _t.no_grad():
        f6 = generate_feats(pts, Z=1, Y=GRID, X=GRID, bounds=BOUNDS)
        f6 = f6.reshape(-1, GRID, GRID)            # (6,120,120)
        o1 = generate_bev(pts, Z=1, Y=GRID, X=GRID, bounds=BOUNDS)
        o1 = o1.reshape(-1, GRID, GRID)            # (1,120,120)
    return idx, f6.numpy().astype(np.float16), o1.numpy().astype(np.float16)


def compute_bevs(sensor, seq, scan_ids, workers, tag=""):
    """Shared ring+ringpp BEVs, resume-safe (chunk shards -> one npz)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    final = CACHE_DIR / f"bev_{sensor}_{seq}{tag}.npz"
    if final.exists():
        with np.load(final) as c:
            if np.array_equal(c["scan_ids"], scan_ids):
                print(f"    BEV cache hit: {final.name}")
                return c["bevs_ringpp"], c["bevs_ring"], (0.0, 0)
    n = len(scan_ids)
    bev6 = np.zeros((n, 6, GRID, GRID), dtype=np.float16)
    bev1 = np.zeros((n, 1, GRID, GRID), dtype=np.float16)
    chunk_dir = CACHE_DIR / f"chunks_{sensor}_{seq}{tag}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for c0 in range(0, n, CHUNK):
        c1 = min(c0 + CHUNK, n)
        cpath = chunk_dir / f"chunk_{c0:06d}.npz"
        if cpath.exists():
            try:
                with np.load(cpath) as c:
                    if np.array_equal(c["scan_ids"], scan_ids[c0:c1]):
                        bev6[c0:c1] = c["bev6"]
                        bev1[c0:c1] = c["bev1"]
                        continue
            except Exception:
                pass  # corrupt shard -> rebuild it
        todo.append((c0, c1))
    n_resumed = n - sum(c1 - c0 for c0, c1 in todo)
    if n_resumed:
        print(f"    resumed {n_resumed}/{n} keyframes from shards")
    t_build, n_built = 0.0, 0
    if todo:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers, initializer=_worker_init,
                                          initargs=(sensor, seq)) as pool:
            for c0, c1 in todo:
                t0 = time.time()
                args = [(i, int(scan_ids[i])) for i in range(c0, c1)]
                for idx, f6, o1 in pool.imap_unordered(_bev_one, args,
                                                       chunksize=4):
                    bev6[idx] = f6
                    bev1[idx] = o1
                tmp = chunk_dir / f".tmp_chunk_{c0:06d}.npz"
                np.savez_compressed(tmp, scan_ids=scan_ids[c0:c1],
                                    bev6=bev6[c0:c1], bev1=bev1[c0:c1])
                os.replace(tmp, chunk_dir / f"chunk_{c0:06d}.npz")
                dt = time.time() - t0
                t_build += dt
                n_built += c1 - c0
                rate = n_built / t_build
                eta = (n - n_resumed - n_built) / rate / 60
                print(f"    [{sensor}_{seq}] BEV {n_resumed + n_built}/{n} "
                      f"({rate:.2f} kf/s wall, {workers} workers, "
                      f"ETA {eta:.0f} min)", flush=True)
    tmp = CACHE_DIR / f".tmp_bev_{sensor}_{seq}{tag}.npz"
    np.savez_compressed(tmp, scan_ids=scan_ids, bevs_ringpp=bev6,
                        bevs_ring=bev1)
    os.replace(tmp, final)
    shutil.rmtree(chunk_dir, ignore_errors=True)
    print(f"    cached {final} ({final.stat().st_size / 1e6:.0f} MB)")
    return bev6, bev1, (t_build, n_built)


def compute_specs(bevs, device, batch=64):
    """Official TIRING: radon sinogram -> row-FFT magnitude (as _ringpp_eval)."""
    from torch_radon import ParallelBeam
    n_ang = bevs.shape[-2]
    angles = torch.FloatTensor(np.linspace(0, 2 * np.pi, n_ang).astype(np.float32))
    radon = ParallelBeam(bevs.shape[-1], angles)
    specs = torch.zeros(bevs.shape, dtype=torch.float16)
    for i0 in range(0, len(bevs), batch):
        x = torch.from_numpy(np.asarray(bevs[i0:i0 + batch],
                                        dtype=np.float32)).to(device)
        with torch.no_grad():
            sino = radon.forward(x)
            m = torch.fft.fft2(sino, dim=-1, norm="ortho")
            spec = torch.sqrt(m.real ** 2 + m.imag ** 2 + 1e-15)
        specs[i0:i0 + batch] = spec.half().cpu()
    return specs


def eval_variant(variant, bevs, poses, device, timing):
    from glnet.models.utils import estimate_yaw   # official correlation
    t0 = time.time()
    specs = compute_specs(bevs, device)
    timing["spec_s"] = round(time.time() - t0, 1)
    n = len(specs)
    # coarse descriptor only used to hand ALL candidates to rerank_fn
    glob = specs.float().sum(dim=-2).reshape(n, -1).numpy()
    glob /= (np.linalg.norm(glob, axis=1, keepdims=True) + 1e-12)

    rr = {"s": 0.0, "n": 0}

    def rerank_fn(qi, cand):
        tq = time.time()
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
        rr["s"] += time.time() - tq
        rr["n"] += 1
        return cand[np.argsort(dists)]

    recalls, n_q = compute_recall_cosine_then_rerank(
        glob, rerank_fn, poses, k_values=K_VALUES, n_coarse=n)
    timing["rerank_ms_per_query"] = round(1e3 * rr["s"] / max(rr["n"], 1), 1)
    return recalls, n_q


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sequences", nargs="*", help="sensor/seq (run ONE per process)")
    ap.add_argument("--variant", choices=["both", "ringpp", "ring"],
                    default="both")
    ap.add_argument("--bev-only", action="store_true",
                    help="only build the shared per-sequence BEV cache, "
                         "skip the GPU spectrum/correlation scoring stage")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="truncate the sequence (results get a _mfN suffix)")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the result json already exists")
    ap.add_argument("--roster", action="store_true",
                    help="print the 42 sensor/seq ids and exit")
    args = ap.parse_args()

    manifest = load_manifest()
    if args.roster or not args.sequences:
        for sensor, seq in manifest:
            print(f"{sensor}/{seq}")
        if not args.sequences:
            print("\nno sequences given; pass ONE sensor/seq per process",
                  file=sys.stderr)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    variants = ["ringpp", "ring"] if args.variant == "both" else [args.variant]
    tag = f"_mf{args.max_frames}" if args.max_frames else ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for spec_arg in args.sequences:
        sensor, seq = spec_arg.split("/")
        if (sensor, seq) not in manifest:
            sys.exit(f"{sensor}/{seq} not in the frozen 42-sequence roster")
        out_paths = {v: OUT_DIR / f"{v}_{sensor}_{seq}{tag}.json"
                     for v in variants}
        if not args.force and not args.bev_only \
                and all(p.exists() for p in out_paths.values()):
            print(f"[{sensor}_{seq}] all result files exist, skipping "
                  f"(--force to redo)")
            continue
        scan_ids, poses = load_cache(manifest[(sensor, seq)])
        if args.max_frames:
            scan_ids = scan_ids[:args.max_frames]
            poses = poses[:args.max_frames]
        print(f"[{sensor}_{seq}] {len(scan_ids)} keyframes "
              f"(variants: {','.join(variants)})")
        bev6, bev1, (t_build, n_built) = compute_bevs(
            sensor, seq, scan_ids, args.workers, tag)
        bev_meta = {
            "bev_wall_s_per_kf": round(t_build / n_built, 3) if n_built else None,
            "bev_workers": args.workers, "bev_kf_built": n_built,
        }
        if args.bev_only:
            continue
        for variant in variants:
            if not args.force and out_paths[variant].exists():
                print(f"  [{sensor}_{seq}] {variant} result exists, skipping")
                continue
            bevs = bev6 if variant == "ringpp" else bev1
            timing = dict(bev_meta)
            t0 = time.time()
            recalls, n_q = eval_variant(variant, bevs, poses, device, timing)
            print(f"  [{sensor}_{seq}] {variant} queries={n_q} "
                  + " ".join(f"R@{k}={recalls[k]:.4f}" for k in K_VALUES)
                  + f" ({time.time() - t0:.0f}s, "
                    f"{timing['rerank_ms_per_query']}ms/q)")
            result = {
                "sensor": sensor, "sequence": seq, "variant": variant,
                "n_db": len(scan_ids), "n_queries": n_q,
                **{f"recall@{k}": recalls[k] for k in K_VALUES},
                "timing": timing,
                "_meta": {
                    "grid": GRID, "bounds": BOUNDS,
                    "ranking": "exhaustive official circular correlation",
                    "protocol": "revisit<5m, skip 30 frames, intra-sequence",
                    "keyframes": str(manifest[(sensor, seq)]),
                    "max_frames": args.max_frames or None,
                },
            }
            out_paths[variant].write_text(json.dumps(result, indent=2))
            print(f"  wrote {out_paths[variant]}")


if __name__ == "__main__":
    main()
