#!/usr/bin/env python3
"""Latency + peak-memory benchmark for SC++ / BEVPlace++ (ms-ft) / SeqOT,
directly comparable to scripts/_bench_latency_memory.py (NSD) and
scripts/_bench_ring_latency.py (RING family): same four maps
(KITTI_05 N=2081, KITTI_00 N=3506, MulRan_KAIST03 N=5701,
NCLT_2012-01-08 N=7467) from the same val caches
(cache_056e0a02_* / mulran cache_33919e6e_*), real point clouds through the
same src/data loaders, ENCODE_SAMPLE=150 evenly spaced keyframes for indexing
timing, the harness revisit-query set (baselines/eval_utils.
_find_revisit_queries, 5 m / 30 frames), 10 warmup + 100 timed queries,
p50/p95, torch.cuda.synchronize around GPU sections, per-stage
torch.cuda.max_memory_allocated, host VmRSS/VmHWM.

Methods (each mirrors its deployed eval path exactly):
  scpp        Scan Context++ (baselines/scan_context.py, CPU/numpy).
              INDEXING per keyframe: pointload_io + encode_with_aux
              (SC matrix 20x60 + ring key). QUERY = the exact
              compute_recalls path (eval_utils.compute_recall_cosine_then_
              rerank): faiss IndexFlatIP(20) ring-key cosine with
              search_k=min(n_coarse+2*skip, N), temporal mask, first
              n_coarse=200 candidates -> column-shift cosine rerank
              (utils.cyclic_shift_distance) over the candidates + argsort.
  bevplace    BEVPlace++ ms-ft (baselines/bevplace.py + vendored REIN,
              checkpoint baselines/weights/bevplace_finetune.pth).
              INDEXING per keyframe: pointload_io + _bev_image (CPU) +
              REIN forward batch 1 (8-rotation warp/unwarp/max inside the
              model) + D2H + renorm, exactly BEVPlaceBaseline.encode.
              QUERY = compute_recall_multi_k path: faiss IndexFlatIP(8192)
              (CPU, as the harness), search_k=min(max_k+2*skip, N)=70,
              temporal mask, top-10.
  seqot       SeqOT trained on our split (scripts/_seqot_eval.py plumbing,
              checkpoints _external/seqot_runs/full/{seqot,gem}_latest).
              INDEXING per keyframe (honest amortized cost of one new
              keyframe; sub-descriptors of past keyframes cached):
              pointload_io + range_projection 32x900 (SeqOT original code,
              per-sensor FOV of scripts/_seqot_prepare.py) + seqL=3
              featureExtracter forward batch 1 + GeM over the seqlen=20
              cached sub-descriptor window batch 1. QUERY =
              compute_recall_multi_k path: faiss IndexFlatIP(256),
              search_k=70, temporal mask, top-10.

Database prep is untimed, streamed through the loaders (SC matrices /
8192D descriptors / sub+final descriptors for all N keyframes).

Writes/merges results/baseline_latency_bench.json incrementally (per map)
so the CPU-only scpp run and the later GPU run compose into one file.

Run (CPU-only, safe next to GPU training):
  docker run --rm --cpus 4 --memory 10g -e OMP_NUM_THREADS=4 ... \
    nvcr.io/nvidia/pyg:26.01-py3 nice -n 19 ionice -c 3 \
    python scripts/_bench_baseline_latency.py --methods scpp
Run (GPU part, via scripts/_bench_baseline_latency.sh when GPU is free):
  python scripts/_bench_baseline_latency.py --methods bevplace seqot
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))            # baselines.*
sys.path.insert(1, str(REPO / "src"))    # data.*, utils.*
sys.path.insert(2, str(REPO / "scripts"))  # _seqot_prepare / _seqot_eval

SEQS = [("kitti", "05", "KITTI_05"), ("kitti", "00", "KITTI_00"),
        ("mulran", "KAIST03", "MulRan_KAIST03"), ("nclt", "2012-01-08", "NCLT_2012-01-08")]
CACHE_KEYS = {"kitti": "056e0a02", "nclt": "056e0a02", "mulran": "33919e6e"}
DATA_ROOTS = {"kitti": "/workspace/data/kitti/dataset",
              "nclt": "/workspace/data/nclt",
              "mulran": "/workspace/data/mulran"}
ENCODE_SAMPLE = 150
QUERY_SAMPLE = 100
QUERY_WARMUP = 10
DTH, SKIP = 5.0, 30
MAX_K = 10                                  # harness k_values=[1,5,10]
SC_N_COARSE = 200                           # ScanContextPP default n_coarse
BEVPLACE_WEIGHTS = REPO / "baselines" / "weights" / "bevplace_finetune.pth"
SEQOT_CKPT = REPO / "_external" / "seqot_runs" / "full" / "seqot_latest.pth.tar"
GEM_CKPT = REPO / "_external" / "seqot_runs" / "full" / "gem_latest.pth.tar"
SEQOT_DATA = REPO / "_external" / "seqot_data"
SEQOT_INFER_BATCH = 16                      # _seqot_eval defaults (untimed DB prep)
SEQOT_GEM_BATCH = 512
OUT = REPO / "results" / "baseline_latency_bench.json"
DEVICE = "cuda"


def host_rss():
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM")):
            k, v = line.split(":")
            out[k] = round(int(v.strip().split()[0]) / 1024.0, 1)
    return out


def load_cache_meta(dtype, seq):
    p = REPO / "data" / "preprocessed" / f"cache_{CACHE_KEYS[dtype]}_{dtype}_val_{seq}.npz"
    with np.load(p, allow_pickle=True) as c:
        return c["scan_ids"].astype(np.int64), c["poses"].astype(np.float64)


def find_queries(poses):
    from baselines.eval_utils import _find_revisit_queries
    return _find_revisit_queries(poses[:, :3, 3], DTH, SKIP)


def sample_ids(scan_ids, n):
    step = max(1, len(scan_ids) // n)
    return scan_ids[::step][:n]


def make_loader(dtype, seq):
    if dtype == "kitti":
        from data.kitti_loader import KITTILoader
        return KITTILoader(DATA_ROOTS["kitti"], seq, lazy_load=True)
    if dtype == "nclt":
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(DATA_ROOTS["nclt"], seq, lazy_load=True)
    if dtype == "mulran":
        from data.mulran_loader import MulRanLoader
        return MulRanLoader(DATA_ROOTS["mulran"], seq, lazy_load=True)
    raise ValueError(dtype)


def cap_faiss_threads(faiss):
    n = os.environ.get("OMP_NUM_THREADS")
    if n:
        faiss.omp_set_num_threads(int(n))


def pXX(lat_ms):
    lat = np.asarray(lat_ms)
    return (round(float(lat.mean()), 3),
            round(float(np.percentile(lat, 50)), 3),
            round(float(np.percentile(lat, 95)), 3))


def save_incremental(results, method, method_entry):
    results["methods"][method] = method_entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))


# ------------------------------------------------------------------- SC++
def bench_scpp(results):
    import faiss
    from baselines.scan_context import ScanContextPP

    cap_faiss_threads(faiss)
    enc = ScanContextPP()
    ment = results["methods"].get("scpp", {})
    ment["env"] = {"impl": "baselines/scan_context.py (CPU numpy)",
                   "n_rings": enc.n_rings, "n_sectors": enc.n_sectors,
                   "max_range": enc.max_range, "z_min": enc.z_min,
                   "n_coarse": enc.n_coarse,
                   "faiss": faiss.__version__,
                   "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    ment.setdefault("maps", {})

    for dtype, seq, name in SEQS:
        scan_ids, poses = load_cache_meta(dtype, seq)
        queries = find_queries(poses)
        n = len(scan_ids)
        print(f"[scpp][{name}] N={n} queries={len(queries)}", flush=True)
        loader = make_loader(dtype, seq)

        # ---- indexing timing on ENCODE_SAMPLE real scans ----
        ids = sample_ids(scan_ids, ENCODE_SAMPLE)
        for sid in ids[:3]:                       # warmup
            enc.encode_with_aux(loader[int(sid)]["points"])
        t_io = t_enc = 0.0
        npts = []
        for sid in ids:
            t0 = time.perf_counter()
            pts = loader[int(sid)]["points"]
            t1 = time.perf_counter()
            enc.encode_with_aux(pts)
            t2 = time.perf_counter()
            t_io += t1 - t0
            t_enc += t2 - t1
            npts.append(len(pts))
        k = len(ids)
        indexing = {"n_sampled": k, "mean_points_per_scan": int(np.mean(npts)),
                    "ms_per_kf_pointload_io": round(t_io / k * 1e3, 3),
                    "ms_per_kf_sc_encode": round(t_enc / k * 1e3, 3),
                    "ms_per_kf_total": round(t_enc / k * 1e3, 3)}

        # ---- DB prep (untimed): SC matrices + ring keys for all N ----
        rk = np.zeros((n, enc.n_rings), dtype=np.float32)
        sc_matrices = np.zeros((n, enc.n_rings, enc.n_sectors), dtype=np.float32)
        for i, sid in enumerate(scan_ids):
            d, aux = enc.encode_with_aux(loader[int(sid)]["points"])
            rk[i] = d
            sc_matrices[i] = aux["sc_matrix"]
            if (i + 1) % 1000 == 0:
                print(f"  [scpp][{name}] db {i + 1}/{n}", flush=True)
        del loader

        # ---- query: exact compute_recall_cosine_then_rerank path ----
        from utils.cyclic_shift_distance import cyclic_column_cosine_distance
        coarse_f32 = rk.astype(np.float32).copy()
        faiss.normalize_L2(coarse_f32)
        index = faiss.IndexFlatIP(coarse_f32.shape[1])
        index.add(coarse_f32)
        search_k = min(SC_N_COARSE + 2 * SKIP, n)
        positions = poses[:, :3, 3].astype(np.float64)

        def one_query(qi):
            t0 = time.perf_counter()
            query_emb = coarse_f32[qi:qi + 1]
            _, indices = index.search(query_emb, search_k)
            valid_mask = np.abs(indices[0] - qi) > SKIP
            candidates = indices[0][valid_mask][:SC_N_COARSE]
            t1 = time.perf_counter()
            sc_q = sc_matrices[qi]
            distances = np.empty(len(candidates), dtype=np.float32)
            for i, c in enumerate(candidates):
                distances[i] = cyclic_column_cosine_distance(sc_q, sc_matrices[c])
            reranked = candidates[np.argsort(distances)]
            t2 = time.perf_counter()
            return (t1 - t0) * 1e3, (t2 - t1) * 1e3, candidates, reranked

        qs = [q for q, _ in queries][:QUERY_SAMPLE + QUERY_WARMUP]
        for qi in qs[:QUERY_WARMUP]:
            one_query(qi)
        rows, ncand = [], []
        hit_coarse = hit_rerank = 0
        for qi in qs[QUERY_WARMUP:]:
            pre, rr, candidates, reranked = one_query(qi)
            rows.append((pre, rr))
            ncand.append(len(candidates))
            hit_coarse += int(np.linalg.norm(positions[candidates[0]] - positions[qi]) < DTH)
            hit_rerank += int(np.linalg.norm(positions[reranked[0]] - positions[qi]) < DTH)
        rows = np.array(rows)
        tot_mean, p50, p95 = pXX(rows.sum(axis=1))
        query = {"n_timed": len(rows), "n_coarse": SC_N_COARSE,
                 "search_k": int(search_k),
                 "avg_candidates": round(float(np.mean(ncand)), 1),
                 "prefilter_ms": round(float(rows[:, 0].mean()), 3),
                 "rerank_ms": round(float(rows[:, 1].mean()), 3),
                 "total_ms_per_query": tot_mean, "p50_ms": p50, "p95_ms": p95,
                 "db_ringkey_mb": round(rk.nbytes / 1e6, 1),
                 "db_sc_matrices_mb": round(sc_matrices.nbytes / 1e6, 1)}
        sanity = {"coarse_R@1": round(hit_coarse / len(rows), 4),
                  "rerank_R@1": round(hit_rerank / len(rows), 4)}
        ment["maps"][name] = {"N": int(n), "n_queries": int(len(queries)),
                              "indexing": indexing, "query": query,
                              "sanity": sanity, "host_rss_mb": host_rss()}
        save_incremental(results, "scpp", ment)
        print(f"[scpp][{name}] idx {indexing['ms_per_kf_total']} ms/kf, "
              f"query {query['total_ms_per_query']} ms (pre {query['prefilter_ms']}"
              f" + rerank {query['rerank_ms']}), rerank R@1 {sanity['rerank_R@1']}",
              flush=True)


# ------------------------------------------------------------- BEVPlace++
def bench_bevplace(results):
    os.environ["NSD_BEVPLACE_WEIGHTS"] = str(BEVPLACE_WEIGHTS)
    import faiss
    import torch
    from baselines.bevplace import BEVPlaceBaseline, _bev_image, _REINModule

    cap_faiss_threads(faiss)
    enc = BEVPlaceBaseline()
    assert enc.is_available(), f"BEVPlace++ unavailable (weights={BEVPLACE_WEIGHTS})"
    model = _REINModule.get(DEVICE)
    ment = results["methods"].get("bevplace_msft", {})
    ment["env"] = {"impl": "baselines/bevplace.py + vendored REIN (8 rotations)",
                   "weights": str(BEVPLACE_WEIGHTS.relative_to(REPO)),
                   "img_size": enc.img_size, "max_range": enc.max_range,
                   "descriptor_dim": enc.descriptor_dim,
                   "gpu": torch.cuda.get_device_name(0),
                   "torch": torch.__version__, "faiss": faiss.__version__,
                   "faiss_index": "IndexFlatIP (CPU, as harness)",
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    ment.setdefault("maps", {})

    for dtype, seq, name in SEQS:
        scan_ids, poses = load_cache_meta(dtype, seq)
        queries = find_queries(poses)
        n = len(scan_ids)
        print(f"[bevplace][{name}] N={n} queries={len(queries)}", flush=True)
        loader = make_loader(dtype, seq)

        # ---- indexing timing (exact BEVPlaceBaseline.encode, staged) ----
        ids = sample_ids(scan_ids, ENCODE_SAMPLE)
        for sid in ids[:3]:                       # warmup
            enc.encode(loader[int(sid)]["points"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t_io = t_bev = t_fwd = 0.0
        npts = []
        for sid in ids:
            t0 = time.perf_counter()
            pts = loader[int(sid)]["points"]
            t1 = time.perf_counter()
            bev = _bev_image(pts, enc.img_size, enc.max_range)
            t2 = time.perf_counter()
            with torch.no_grad():
                x = torch.from_numpy(bev).unsqueeze(0).to(DEVICE)
                _, _, global_desc = model(x)
                emb = global_desc.squeeze(0).cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            t_io += t1 - t0
            t_bev += t2 - t1
            t_fwd += t3 - t2
            npts.append(len(pts))
        k = len(ids)
        indexing = {"n_sampled": k, "mean_points_per_scan": int(np.mean(npts)),
                    "ms_per_kf_pointload_io": round(t_io / k * 1e3, 3),
                    "ms_per_kf_bev": round(t_bev / k * 1e3, 3),
                    "ms_per_kf_model_forward": round(t_fwd / k * 1e3, 3),
                    "ms_per_kf_total": round((t_bev + t_fwd) / k * 1e3, 3),
                    "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)}

        # ---- DB prep (untimed): 8192D descriptors for all N ----
        desc = np.zeros((n, enc.descriptor_dim), dtype=np.float32)
        for i, sid in enumerate(scan_ids):
            desc[i] = enc.encode(loader[int(sid)]["points"])
            if (i + 1) % 500 == 0:
                print(f"  [bevplace][{name}] db {i + 1}/{n}", flush=True)
        del loader

        # ---- query: exact compute_recall_multi_k path ----
        emb32 = desc.astype(np.float32).copy()
        faiss.normalize_L2(emb32)
        index = faiss.IndexFlatIP(emb32.shape[1])
        index.add(emb32)
        search_k = min(MAX_K + 2 * SKIP, n)
        positions = poses[:, :3, 3].astype(np.float64)

        def one_query(qi):
            t0 = time.perf_counter()
            query_emb = emb32[qi:qi + 1]
            _, indices = index.search(query_emb, search_k)
            valid_mask = np.abs(indices[0] - qi) > SKIP
            ranked = indices[0][valid_mask][:MAX_K]
            t1 = time.perf_counter()
            return (t1 - t0) * 1e3, ranked

        qs = [q for q, _ in queries][:QUERY_SAMPLE + QUERY_WARMUP]
        for qi in qs[:QUERY_WARMUP]:
            one_query(qi)
        lat = []
        hit = 0
        for qi in qs[QUERY_WARMUP:]:
            ms, ranked = one_query(qi)
            lat.append(ms)
            hit += int(np.linalg.norm(positions[ranked[0]] - positions[qi]) < DTH)
        tot_mean, p50, p95 = pXX(lat)
        query = {"n_timed": len(lat), "search_k": int(search_k),
                 "search_ms": tot_mean,
                 "total_ms_per_query": tot_mean, "p50_ms": p50, "p95_ms": p95,
                 "db_desc_host_mb": round(desc.nbytes / 1e6, 1)}
        sanity = {"R@1": round(hit / len(lat), 4)}
        ment["maps"][name] = {"N": int(n), "n_queries": int(len(queries)),
                              "indexing": indexing, "query": query,
                              "sanity": sanity, "host_rss_mb": host_rss()}
        save_incremental(results, "bevplace_msft", ment)
        torch.cuda.empty_cache()
        print(f"[bevplace][{name}] idx {indexing['ms_per_kf_total']} ms/kf "
              f"(bev {indexing['ms_per_kf_bev']} + fwd "
              f"{indexing['ms_per_kf_model_forward']}), "
              f"query {query['total_ms_per_query']} ms, R@1 {sanity['R@1']}",
              flush=True)


# ------------------------------------------------------------------ SeqOT
def bench_seqot(results):
    import faiss
    import torch
    # _seqot_eval inserts _external/SeqOT into sys.path and re-exports the
    # unchanged official modules (featureExtracter, GeM).
    from _seqot_eval import (featureExtracter, GeM, load_range_stack, window,
                             SEQL, GEM_SEQLEN)
    from _seqot_prepare import SENSOR_PROJ, PROJ_H, PROJ_W, MAX_RANGE, range_projection

    cap_faiss_threads(faiss)
    fe = featureExtracter(seqL=SEQL).to(DEVICE)
    fe.load_state_dict(torch.load(SEQOT_CKPT, map_location=DEVICE,
                                  weights_only=False)["state_dict"])
    fe.eval()
    gem = GeM().to(DEVICE)
    gem.load_state_dict(torch.load(GEM_CKPT, map_location=DEVICE,
                                   weights_only=False)["state_dict"])
    gem.eval()
    ment = results["methods"].get("seqot", {})
    ment["env"] = {"impl": "scripts/_seqot_eval.py plumbing (seqL=3, seqlen=20)",
                   "seqot_checkpoint": str(SEQOT_CKPT.relative_to(REPO)),
                   "gem_checkpoint": str(GEM_CKPT.relative_to(REPO)),
                   "proj": f"{PROJ_H}x{PROJ_W} range image, max_range {MAX_RANGE}",
                   "descriptor_dim": 256,
                   "gpu": torch.cuda.get_device_name(0),
                   "torch": torch.__version__, "faiss": faiss.__version__,
                   "faiss_index": "IndexFlatIP (CPU, as harness)",
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    ment.setdefault("maps", {})

    for dtype, seq, name in SEQS:
        scan_ids, poses = load_cache_meta(dtype, seq)
        queries = find_queries(poses)
        n = len(scan_ids)
        seq_dir = SEQOT_DATA / f"{dtype}_{seq}"
        stack = load_range_stack(seq_dir, 0)
        assert len(stack) == n, f"{name}: stack {len(stack)} != cache {n}"
        print(f"[seqot][{name}] N={n} queries={len(queries)}", flush=True)

        # ---- DB prep (untimed): sub-descriptors + final descriptors, batched
        # exactly as _seqot_eval.compute_final_descriptors ----
        sub = np.zeros((n, 256), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n, SEQOT_INFER_BATCH):
                idx = np.arange(i, min(i + SEQOT_INFER_BATCH, n))
                w = np.stack([window(j, SEQL, n) for j in idx])
                sub[idx] = fe(torch.from_numpy(stack[w]).to(DEVICE)).cpu().numpy()
            final = np.zeros((n, 256), dtype=np.float32)
            for i in range(0, n, SEQOT_GEM_BATCH):
                idx = np.arange(i, min(i + SEQOT_GEM_BATCH, n))
                w = np.stack([window(j, GEM_SEQLEN, n) for j in idx])
                out = gem(torch.from_numpy(sub[w]).to(DEVICE)).squeeze(1)
                final[idx] = out.cpu().numpy()

        # ---- indexing timing: per new keyframe = range proj + seqL=3
        # sub-descriptor forward + seqlen=20 GeM (sub of neighbors cached) ----
        proj = SENSOR_PROJ[dtype]
        loader = make_loader(dtype, seq)
        pos_sample = sample_ids(np.arange(n), ENCODE_SAMPLE)
        for i in pos_sample[:3]:                  # warmup
            pts = loader[int(scan_ids[i])]["points"][:, :3].astype(np.float64)
            range_projection(pts, fov_up=proj["fov_up"], fov_down=proj["fov_down"],
                             proj_H=PROJ_H, proj_W=PROJ_W, max_range=MAX_RANGE,
                             cut_z=False)
            with torch.no_grad():
                fe(torch.from_numpy(stack[window(i, SEQL, n)][None]).to(DEVICE))
                gem(torch.from_numpy(sub[window(i, GEM_SEQLEN, n)][None]).to(DEVICE))
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t_io = t_proj = t_sub = t_gem = 0.0
        npts = []
        for i in pos_sample:
            t0 = time.perf_counter()
            pts = loader[int(scan_ids[i])]["points"][:, :3].astype(np.float64)
            t1 = time.perf_counter()
            range_projection(pts, fov_up=proj["fov_up"], fov_down=proj["fov_down"],
                             proj_H=PROJ_H, proj_W=PROJ_W, max_range=MAX_RANGE,
                             cut_z=False)
            t2 = time.perf_counter()
            with torch.no_grad():
                fe(torch.from_numpy(stack[window(i, SEQL, n)][None]).to(DEVICE))
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            with torch.no_grad():
                gem(torch.from_numpy(sub[window(i, GEM_SEQLEN, n)][None]).to(DEVICE))
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            t_io += t1 - t0
            t_proj += t2 - t1
            t_sub += t3 - t2
            t_gem += t4 - t3
            npts.append(len(pts))
        k = len(pos_sample)
        del loader
        indexing = {"n_sampled": k, "mean_points_per_scan": int(np.mean(npts)),
                    "ms_per_kf_pointload_io": round(t_io / k * 1e3, 3),
                    "ms_per_kf_range_proj": round(t_proj / k * 1e3, 3),
                    "ms_per_kf_subdesc_forward": round(t_sub / k * 1e3, 3),
                    "ms_per_kf_gem_window20": round(t_gem / k * 1e3, 3),
                    "ms_per_kf_total": round((t_proj + t_sub + t_gem) / k * 1e3, 3),
                    "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)}

        # ---- query: exact compute_recall_multi_k path ----
        emb32 = final.astype(np.float32).copy()
        faiss.normalize_L2(emb32)
        index = faiss.IndexFlatIP(emb32.shape[1])
        index.add(emb32)
        search_k = min(MAX_K + 2 * SKIP, n)
        positions = poses[:, :3, 3].astype(np.float64)

        def one_query(qi):
            t0 = time.perf_counter()
            query_emb = emb32[qi:qi + 1]
            _, indices = index.search(query_emb, search_k)
            valid_mask = np.abs(indices[0] - qi) > SKIP
            ranked = indices[0][valid_mask][:MAX_K]
            t1 = time.perf_counter()
            return (t1 - t0) * 1e3, ranked

        qs = [q for q, _ in queries][:QUERY_SAMPLE + QUERY_WARMUP]
        for qi in qs[:QUERY_WARMUP]:
            one_query(qi)
        lat = []
        hit = 0
        for qi in qs[QUERY_WARMUP:]:
            ms, ranked = one_query(qi)
            lat.append(ms)
            hit += int(np.linalg.norm(positions[ranked[0]] - positions[qi]) < DTH)
        tot_mean, p50, p95 = pXX(lat)
        query = {"n_timed": len(lat), "search_k": int(search_k),
                 "search_ms": tot_mean,
                 "total_ms_per_query": tot_mean, "p50_ms": p50, "p95_ms": p95,
                 "db_desc_host_mb": round(final.nbytes / 1e6, 1)}
        sanity = {"R@1": round(hit / len(lat), 4)}
        ment["maps"][name] = {"N": int(n), "n_queries": int(len(queries)),
                              "indexing": indexing, "query": query,
                              "sanity": sanity, "host_rss_mb": host_rss()}
        save_incremental(results, "seqot", ment)
        del stack
        torch.cuda.empty_cache()
        print(f"[seqot][{name}] idx {indexing['ms_per_kf_total']} ms/kf "
              f"(proj {indexing['ms_per_kf_range_proj']} + sub "
              f"{indexing['ms_per_kf_subdesc_forward']} + gem "
              f"{indexing['ms_per_kf_gem_window20']}), "
              f"query {query['total_ms_per_query']} ms, R@1 {sanity['R@1']}",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", nargs="+", default=["scpp", "bevplace", "seqot"],
                    choices=["scpp", "bevplace", "seqot"])
    args = ap.parse_args()

    if OUT.exists():
        results = json.loads(OUT.read_text())
    else:
        results = {"env": {}, "methods": {}}
    results["env"].update({
        "cache_keys": CACHE_KEYS,
        "encode_sample_per_seq": ENCODE_SAMPLE,
        "query_sample": QUERY_SAMPLE, "query_warmup": QUERY_WARMUP,
        "distance_threshold_m": DTH, "skip_frames": SKIP, "max_k": MAX_K,
        "image": "nvcr.io/nvidia/pyg:26.01-py3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})

    for m in args.methods:
        {"scpp": bench_scpp, "bevplace": bench_bevplace,
         "seqot": bench_seqot}[m](results)

    print(f"\nwrote {OUT}\n")
    print(f"{'method':<15}{'map':<18}{'N':>6}{'idx ms/kf':>11}{'query ms':>10}"
          f"{'p50 ms':>9}{'p95 ms':>9}")
    for m, ment in results["methods"].items():
        for name, e in ment.get("maps", {}).items():
            print(f"{m:<15}{name:<18}{e['N']:>6}"
                  f"{e['indexing']['ms_per_kf_total']:>11}"
                  f"{e['query']['total_ms_per_query']:>10}"
                  f"{e['query']['p50_ms']:>9}"
                  f"{e['query']['p95_ms']:>9}")


if __name__ == "__main__":
    main()
