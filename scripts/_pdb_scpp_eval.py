#!/usr/bin/env python3
"""SC++ (Scan Context++) baseline on the frozen pd2026 42-sequence TEST roster.

Protocol = the paper SC++ path (baselines/scan_context.py scored through
baselines/eval_utils.py): per-keyframe Scan Context matrix (20 rings x 60
sectors, max_range 80 m, z_min -3 m), ring-key cosine FAISS prefilter
(n_coarse=200), cyclic column-shift cosine rerank
(utils.cyclic_shift_distance.cyclic_column_cosine_distance), revisit queries
< 5 m with > 30-keyframe gap, R@1/5/10. Query selection and recall scoring are
imported from baselines.eval_utils, so they are byte-identical to the paper
evaluation. NCLT z-handling comes from the fixed loader (z-down -> z-up flip
inside data/nclt_loader.py), matching scripts/_scpp_nclt_fix8.py.

Keyframes: scan_ids + poses from the frozen pd2026 TEST operating caches
listed in scripts/_pd_test_cache_manifest.json — identical rosters to the NSD
pd evaluation. Raw clouds are re-loaded at those scan_ids via the repo
loaders (kitti / nclt / helipr / mulran / kitti360 / mcd).

One heavy sequence per process (drive with --seqs sensor/seq, one subprocess
each). Scan-context building and query rerank are parallelized with the
spawn-Pool + worker-initializer pattern of scripts/_ringpp_eval.py. CPU-only
(numpy + faiss). Writes results/pd2026_baselines/scpp_<sensor>_<seq>.json.

Run inside docker ring_baseline:
  docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_pdb_scpp_eval.py --seqs mulran/DCC03 --workers 8
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))          # baselines/
sys.path.insert(0, str(REPO / "src"))  # data loaders + utils

DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", "/workspace/data"))
MANIFEST = REPO / "scripts" / "_pd_test_cache_manifest.json"

# Paper SC++ parameters (baselines/scan_context.py defaults)
SC_PARAMS = dict(n_rings=20, n_sectors=60, max_range=80.0, z_min=-3.0)
N_COARSE = 200
K_VALUES = [1, 5, 10]
DIST_TH = 5.0
SKIP = 30


def _make_loader(sensor: str, seq: str):
    """Mirror of scripts/_pd_build_test_caches.py loader wiring (all 6 sensors)."""
    if sensor == "kitti":
        from data.kitti_loader import KITTILoader
        return KITTILoader(str(DATA_ROOT / "kitti/dataset"), seq, lazy_load=True)
    if sensor == "nclt":
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(str(DATA_ROOT / "nclt"), seq, lazy_load=True)
    if sensor == "helipr":
        from data.helipr_loader import HeLiPRLoader
        seq_path = DATA_ROOT / "helipr" / seq
        if not (seq_path / "LiDAR").is_dir():
            seq_path = seq_path / seq  # legacy double-nested layout
        return HeLiPRLoader(str(seq_path), lazy_load=True)
    if sensor == "mulran":
        from data.mulran_loader import MulRanLoader
        return MulRanLoader(str(DATA_ROOT / "mulran"), seq, lazy_load=True)
    if sensor == "kitti360":
        from data.kitti360_loader import Kitti360Loader
        return Kitti360Loader(str(DATA_ROOT / "kitti360"), seq, lazy_load=True)
    if sensor == "mcd":
        from data.mcd_loader import MCDLoader
        return MCDLoader(str(DATA_ROOT / "mcd_extracted"), seq, lazy_load=True)
    raise ValueError(sensor)


# ── stage 1: scan-context encoding (Pool worker per loader) ─────────────────
def _enc_init(sensor, seq):
    global _W_LOADER
    _W_LOADER = _make_loader(sensor, seq)


def _enc_one(args):
    idx, scan_id = args
    from baselines.scan_context import build_scan_context
    t0 = time.perf_counter()
    pts = _W_LOADER[int(scan_id)]["points"]
    t1 = time.perf_counter()
    sc = build_scan_context(pts, **SC_PARAMS)
    t2 = time.perf_counter()
    return idx, sc.astype(np.float32), (t1 - t0) * 1e3, (t2 - t1) * 1e3


# ── stage 2: cyclic column-shift rerank (Pool worker over shared SC file) ───
def _rr_init(sc_path):
    global _W_SC
    _W_SC = np.load(sc_path)


def _rr_one(args):
    qi, cand = args
    from utils.cyclic_shift_distance import cyclic_column_cosine_distance
    t0 = time.perf_counter()
    d = np.empty(len(cand), dtype=np.float32)
    for i, c in enumerate(cand):
        d[i] = cyclic_column_cosine_distance(_W_SC[qi], _W_SC[int(c)])
    return qi, cand[np.argsort(d)], (time.perf_counter() - t0) * 1e3


def _resolve_cache(sensor: str, seq: str) -> Path:
    man = json.loads(MANIFEST.read_text())
    for e in man["sequences"]:
        if e["sensor"] == sensor and e["sequence"] == seq:
            return REPO / e["expected"]["operating_cache"]
    raise SystemExit(f"not in the frozen 42-sequence test roster: {sensor}/{seq}")


def run_sequence(sensor: str, seq: str, workers: int, out_dir: Path,
                 limit: int = 0):
    import multiprocessing as mp
    import faiss
    from baselines.eval_utils import (_find_revisit_queries,
                                      _score_recalls_from_ranked)

    cache_path = _resolve_cache(sensor, seq)
    with np.load(cache_path) as c:
        scan_ids = c["scan_ids"].astype(np.int64)
        poses = c["poses"].astype(np.float64)
    if limit:
        scan_ids, poses = scan_ids[:limit], poses[:limit]
    n = len(scan_ids)
    print(f"[{sensor}/{seq}] {n} keyframes ({cache_path.name})", flush=True)

    # stage 1: scan contexts
    scs = np.zeros((n, SC_PARAMS["n_rings"], SC_PARAMS["n_sectors"]),
                   dtype=np.float32)
    load_ms, enc_ms = [], []
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_enc_init,
                                      initargs=(sensor, seq)) as pool:
        args = [(i, scan_ids[i]) for i in range(n)]
        for m, (idx, sc, lm, em) in enumerate(
                pool.imap_unordered(_enc_one, args, chunksize=8)):
            scs[idx] = sc
            load_ms.append(lm)
            enc_ms.append(em)
            if (m + 1) % 1000 == 0:
                rate = (m + 1) / (time.time() - t0)
                print(f"  [{sensor}/{seq}] sc {m+1}/{n} ({rate:.1f}/s, "
                      f"ETA {(n-m-1)/rate/60:.1f} min)", flush=True)
    encode_wall = time.time() - t0

    # ring keys (paper Eq.5) -> cosine prefilter, exactly eval_utils' path
    rks = scs.mean(axis=2).astype(np.float32).copy()
    faiss.normalize_L2(rks)
    index = faiss.IndexFlatIP(rks.shape[1])
    index.add(rks)

    positions = poses[:, :3, 3]
    queries = _find_revisit_queries(positions, DIST_TH, SKIP)
    print(f"[{sensor}/{seq}] {len(queries)} queries", flush=True)

    search_k = min(N_COARSE + 2 * SKIP, n)
    tasks = []
    for query_idx, _ in queries:
        _, indices = index.search(rks[query_idx:query_idx + 1], search_k)
        valid_mask = np.abs(indices[0] - query_idx) > SKIP
        cand = indices[0][valid_mask][:N_COARSE].astype(np.int64)
        tasks.append((int(query_idx), cand))

    # stage 2: parallel rerank (SC matrices handed to workers via temp .npy)
    ranked_by_q, rr_ms = {}, []
    t1 = time.time()
    if queries:
        fd, sc_path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(sc_path, scs)
        try:
            with mp.get_context("spawn").Pool(workers, initializer=_rr_init,
                                              initargs=(sc_path,)) as pool:
                for m, (qi, ranked, ms) in enumerate(
                        pool.imap_unordered(_rr_one, tasks, chunksize=8)):
                    ranked_by_q[qi] = ranked
                    rr_ms.append(ms)
                    if (m + 1) % 500 == 0:
                        print(f"  [{sensor}/{seq}] rerank {m+1}/{len(tasks)}",
                              flush=True)
        finally:
            os.unlink(sc_path)
    rerank_wall = time.time() - t1

    ranked_lists = [ranked_by_q[qi] if len(cand) else np.array([], np.int64)
                    for qi, cand in tasks]
    recalls = _score_recalls_from_ranked(
        queries, ranked_lists, positions, K_VALUES, DIST_TH, poses=poses)

    n_q = len(queries)
    result = {
        "method": "sc++",
        "sensor": sensor,
        "sequence": seq,
        "n_keyframes": n,
        "n_queries": n_q,
        **{f"recall@{k}": recalls[k] for k in K_VALUES},
        "protocol": {
            "distance_threshold_m": DIST_TH,
            "skip_frames": SKIP,
            "n_coarse": N_COARSE,
            "sc_params": SC_PARAMS,
            "keyframes": str(cache_path.relative_to(REPO)),
            "limit": limit,
        },
        "timing": {
            "workers": workers,
            "encode_wall_s": round(encode_wall, 2),
            "encode_wall_ms_per_kf": round(encode_wall * 1e3 / n, 3),
            "worker_load_ms_per_kf": round(float(np.mean(load_ms)), 3),
            "worker_encode_ms_per_kf": round(float(np.mean(enc_ms)), 3),
            "rerank_wall_s": round(rerank_wall, 2),
            "rerank_wall_ms_per_query": round(rerank_wall * 1e3 / max(n_q, 1), 3),
            "worker_rerank_ms_per_query": round(float(np.mean(rr_ms)), 3) if rr_ms else 0.0,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"scpp_{sensor}_{seq}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[{sensor}/{seq}] q={n_q} "
          + " ".join(f"R@{k}={recalls[k]:.4f}" for k in K_VALUES)
          + f" -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seqs", nargs="+", required=True,
                    help="sensor/seq like mulran/DCC03 (one heavy sequence "
                         "per process!)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=str(REPO / "results/pd2026_baselines"))
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N keyframes (smoke test)")
    args = ap.parse_args()

    for s in args.seqs:
        sensor, seq = s.split("/", 1)
        run_sequence(sensor, seq, args.workers, Path(args.out_dir),
                     limit=args.limit)
    print("PDB_SCPP_DONE", flush=True)


if __name__ == "__main__":
    main()
