#!/usr/bin/env python3
"""Matched-condition efficiency benchmark: one sequence, one DB size, one protocol.

Feeds the paper table  Method | descriptor dim | storage per frame | build ms |
query ms  with numbers measured under IDENTICAL conditions for every method.

MATCHED CONDITIONS (identical for all methods):
  * workload: kitti/00, keyframe roster = data/preprocessed_cross_sensor_operating/
    kitti_operating_00_layout60_stride1.npz scan_ids (verified at runtime to be
    identical to the cache_056e0a02 val roster used by _bench_baseline_latency.py
    and _bench_ring_latency.py over the frames used here);
  * DB = the first --n-db (default 1000) roster frames; queries = the next
    --n-queries (default 200) frames, matched against the DB only. Queries are
    disjoint from the DB, so NO temporal/self exclusion is applied anywhere;
  * batch 1, sequential queries;
  * disk I/O excluded: all raw point clouds are preloaded into RAM before any
    timed region; per-method dtype conversion is also untimed (the source
    benches counted loader access + conversion in an untimed pointload stage);
  * warmup: the first 10 DB frames (build) / first 10 queries (query) run once
    untimed, then EVERY DB frame and EVERY query is timed
    (build n = n_db, query n = n_queries);
  * torch.cuda.synchronize() before/after every timed region that touches GPU;
  * mean/p50/p95 ms per frame (build) and per query (query) are reported.

Each method's compute stack is the exact deployed path of the existing benches
(imports where possible, verbatim lifts with file:line provenance comments):
  scpp/bevplace/seqot  scripts/_bench_baseline_latency.py
  ring/ringpp          scripts/_bench_ring_latency.py
  ringsharp            build = official BEV20 + RING#-L forward + trans_cnn;
                       stored set and query rerank = scripts/_pdb_ringsharp_eval.py
                       (DEFAULT_WEIGHT checkpoint, n_coarse=200, n_trans=20;
                       timing is weight-independent);
  nsd/nsdh             build = the deployed B3 two-pass pipeline of
                       scripts/_bench_latency_memory.py (per-frame encode stages;
                       graph/edge/classifier/bank stages are batch and amortized
                       per keyframe) with the deployed auxiliary spectra
                       (_pbev_branch_fusion BEV 16x8 / _hspec_branch_fusion
                       height-polar 20x12); query = the two-index union +
                       per-query min-max 4-channel fusion + cyclic alignment of
                       scripts/_headline_latency.py, tuples NSD (0.5,1,0,0.5),
                       NSD-H (1,1,1,2).

STORAGE DEFINITION (measured, never hardcoded): storage_bytes_per_frame = sum
of the actual bytes of EVERY array the method must persist per frame for
retrieval to work (ALL retrieval-time stored state, not just the global
descriptor), measured from the arrays this run actually builds for the DB and
divided by n_db:
  scpp        normalized 20D ring key + 20x60 scan-context matrix (rerank input)
  bevplace    8192D final descriptor
  seqot       256D final descriptor
  ring/ringpp TIRING correlation spectrum the DB holds (fp16, C x 120 x 120)
  ringsharp   160D coarse global + rotation-branch spec (fp16) + trans-branch
              feature bts (fp16). The packed occupancy BEV is NOT counted:
              _pdb_ringsharp_eval.rerank_fn indexes only specs and bts.
  nsd         416D key + stored complex spectrum (cyl 16x4 + BEV 16x8, c64)
  nsdh        416D key + stored complex spectrum (cyl 16x4 + height 20x12, c64)

Output: results/matched_bench_n1000.json, one entry per method. Resumable:
existing entries are skipped unless --overwrite. env per entry = {gpu, torch}
only (no timestamps).

Run inside the ring_baseline container with the GPU free (raw KITTI at
$NSD_DATA_ROOT/kitti/dataset, default /workspace/data):
  python scripts/_bench_matched_n1000.py
  python scripts/_bench_matched_n1000.py --methods scpp nsd --overwrite
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))              # baselines.*
sys.path.insert(1, str(REPO / "src"))      # data.*, utils.*, encoding.*
sys.path.insert(2, str(REPO / "scripts"))  # bench/eval sibling modules
RINGSHARP = REPO / "_external" / "RINGSharp"

DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", "/workspace/data"))
ROSTER_CACHE = REPO / ("data/preprocessed_cross_sensor_operating/"
                       "kitti_operating_00_layout60_stride1.npz")
VAL_CACHE = REPO / "data/preprocessed/cache_056e0a02_kitti_val_00.npz"
OUT = REPO / "results" / "matched_bench_n1000.json"
DEVICE = "cuda"
WARMUP = 10
DTH = 5.0                       # 5 m sanity threshold (all source benches)
SC_N_COARSE = 200               # ScanContextPP deployed (_bench_baseline_latency.py:82)
MAX_K = 10                      # harness top-k (_bench_baseline_latency.py:81)
RS_N_COARSE, RS_N_TRANS = 200, 20  # deployed (_pdb_ringsharp_eval.py:204-208)
CORR_BATCH = 256                # _bench_ring_latency.py:72
GRID = 120                      # official RING/RING++ (_bench_ring_latency.py:69)
BOUNDS = (-70.0, 70.0, -70.0, 70.0, 1.0, 20.0)  # _bench_ring_latency.py:70
K_NSD = 100                     # two-index top-K (_headline_latency.py:34)
BEVPLACE_WEIGHTS = REPO / "baselines" / "weights" / "bevplace_finetune.pth"
SEQOT_CKPT = REPO / "_external" / "seqot_runs" / "full" / "seqot_latest.pth.tar"
GEM_CKPT = REPO / "_external" / "seqot_runs" / "full" / "gem_latest.pth.tar"
SEQOT_DATA = REPO / "_external" / "seqot_data"
METHODS = ("scpp", "bevplace", "seqot", "ring", "ringpp", "ringsharp", "nsd", "nsdh")


def stats(lat_ms):
    lat = np.asarray(lat_ms, dtype=np.float64)
    return {"mean": round(float(lat.mean()), 3),
            "p50": round(float(np.percentile(lat, 50)), 3),
            "p95": round(float(np.percentile(lat, 95)), 3),
            "n": int(len(lat))}


def env_info():
    import torch
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "torch": torch.__version__}


def cap_faiss_threads(faiss):   # _bench_baseline_latency.py:131-134
    n = os.environ.get("OMP_NUM_THREADS")
    if n:
        faiss.omp_set_num_threads(int(n))


def make_entry(dim, breakdown, build_lat, query_lat, hit, notes, extra=None):
    breakdown = {k: round(float(v), 1) for k, v in breakdown.items()}
    entry = {"descriptor_dim": dim,
             "storage_bytes_per_frame": round(sum(breakdown.values()), 1),
             "storage_breakdown": breakdown,
             "build_ms": stats(build_lat),
             "query_ms": stats(query_lat),
             "sanity_top1_within_5m": round(float(hit), 4),
             "notes": notes,
             "env": env_info()}
    if extra:
        entry.update(extra)
    return entry


class Workload:
    """Shared roster + preloaded point clouds (disk I/O excluded from timing)."""

    def __init__(self, n_db, n_q):
        self.n_db, self.n_q = n_db, n_q
        self.n_total = n_db + n_q
        npz = np.load(ROSTER_CACHE)
        assert len(npz["scan_ids"]) >= self.n_total, \
            f"roster has {len(npz['scan_ids'])} frames < {self.n_total}"
        self.scan_ids = npz["scan_ids"][: self.n_total].astype(np.int64)
        self.poses = npz["poses"][: self.n_total]
        self.positions = self.poses[:, :3, 3].astype(np.float64)
        if VAL_CACHE.exists():  # roster identity with the baseline/RING benches
            with np.load(VAL_CACHE, allow_pickle=True) as v:
                assert np.array_equal(v["scan_ids"][: self.n_total].astype(np.int64),
                                      self.scan_ids), "operating vs 056e0a02 roster mismatch"
        self.q_idx = np.arange(n_db, self.n_total)
        from data.kitti_loader import KITTILoader
        loader = KITTILoader(str(DATA_ROOT / "kitti" / "dataset"), "00", lazy_load=True)
        print(f"[preload] {self.n_total} raw kitti/00 scans into RAM ...", flush=True)
        self.points = [loader[int(s)]["points"] for s in self.scan_ids]
        del loader


class SubCache:
    """NpzFile stand-in truncated to the first n frames, arrays materialized so
    repeated access never re-decompresses (cf. _bench_latency_memory.CacheShim)."""

    def __init__(self, npz, n):
        full = len(npz["scan_ids"])
        self.files = list(npz.files)
        self._d = {}
        for k in npz.files:
            a = npz[k]
            self._d[k] = a[:n] if getattr(a, "ndim", 0) >= 1 and a.shape[0] == full else a

    def __getitem__(self, k):
        return self._d[k]


def timed_build(W, step, prep=None, use_cuda=False):
    """Warmup on the first WARMUP frames (untimed), then time every frame.
    Query frames (i >= n_db) are also encoded (their representations are needed
    at query time) but only DB frames enter the build statistics."""
    if use_cuda:
        import torch
    get = prep if prep is not None else (lambda i: W.points[i])
    for i in range(min(WARMUP, W.n_db)):
        step(i, get(i))
    lat = np.zeros(W.n_total)
    for i in range(W.n_total):
        x = get(i)                       # dtype conversion untimed (see docstring)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step(i, x)
        if use_cuda:
            torch.cuda.synchronize()
        lat[i] = (time.perf_counter() - t0) * 1e3
    return lat[: W.n_db]


def timed_query(W, one_query, use_cuda=False):
    if use_cuda:
        import torch
    for q in W.q_idx[:WARMUP]:
        one_query(int(q))
    lat = np.zeros(W.n_q)
    top1 = np.zeros(W.n_q, np.int64)
    for j, q in enumerate(W.q_idx):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        top1[j] = one_query(int(q))
        if use_cuda:
            torch.cuda.synchronize()
        lat[j] = (time.perf_counter() - t0) * 1e3
    hit = np.mean(np.linalg.norm(W.positions[top1] - W.positions[W.q_idx], axis=1) < DTH)
    return lat, hit


# ------------------------------------------------------------------- SC++
def bench_scpp(W):
    import faiss
    from baselines.scan_context import ScanContextPP
    from utils.cyclic_shift_distance import cyclic_column_cosine_distance
    cap_faiss_threads(faiss)
    enc = ScanContextPP()
    rk = np.zeros((W.n_total, enc.n_rings), np.float32)
    sc = np.zeros((W.n_total, enc.n_rings, enc.n_sectors), np.float32)

    def step(i, pts):   # _bench_baseline_latency.py:184,199-201
        d, aux = enc.encode_with_aux(pts)
        rk[i] = d
        sc[i] = aux["sc_matrix"]

    build_lat = timed_build(W, step)

    db = rk[: W.n_db].copy()            # _bench_baseline_latency.py:209-212
    faiss.normalize_L2(db)
    index = faiss.IndexFlatIP(db.shape[1])
    index.add(db)
    qk = rk[W.n_db:].copy()
    faiss.normalize_L2(qk)
    search_k = min(SC_N_COARSE, W.n_db)

    def one_query(q):   # _bench_baseline_latency.py:215-228 minus temporal mask
        _, ind = index.search(qk[q - W.n_db: q - W.n_db + 1], search_k)
        cand = ind[0]
        sq = sc[q]
        dist = np.empty(len(cand), np.float32)
        for i, c in enumerate(cand):
            dist[i] = cyclic_column_cosine_distance(sq, sc[c])
        return int(cand[np.argsort(dist)][0])

    query_lat, hit = timed_query(W, one_query)
    breakdown = {"ring_key_f32": db.nbytes / W.n_db,
                 "sc_matrix_f32": sc[: W.n_db].nbytes / W.n_db}
    return make_entry(
        enc.n_rings, breakdown, build_lat, query_lat, hit,
        "CPU numpy encode (baselines/scan_context.py); query = faiss IndexFlatIP "
        f"ring-key cosine top-{search_k} + column-shift cosine rerank "
        "(utils/cyclic_shift_distance). Storage = ring key + full SC matrix "
        "(the rerank reads the matrix at query time).")


# ------------------------------------------------------------- BEVPlace++
def bench_bevplace(W):
    os.environ["NSD_BEVPLACE_WEIGHTS"] = str(BEVPLACE_WEIGHTS)  # _bench_baseline_latency.py:265
    import faiss
    import torch
    from baselines.bevplace import BEVPlaceBaseline, _bev_image, _REINModule
    cap_faiss_threads(faiss)
    enc = BEVPlaceBaseline()
    assert enc.is_available(), f"BEVPlace++ weights missing: {BEVPLACE_WEIGHTS}"
    model = _REINModule.get(DEVICE)
    desc = np.zeros((W.n_total, enc.descriptor_dim), np.float32)

    def step(i, pts):   # exact BEVPlaceBaseline.encode staged, _bench_baseline_latency.py:304-313
        bev = _bev_image(pts, enc.img_size, enc.max_range)
        with torch.no_grad():
            x = torch.from_numpy(bev).unsqueeze(0).to(DEVICE)
            _, _, g = model(x)
            emb = g.squeeze(0).cpu().numpy().astype(np.float32)
        nrm = np.linalg.norm(emb)
        desc[i] = emb / nrm if nrm > 1e-8 else emb

    build_lat = timed_build(W, step, use_cuda=True)

    db = desc[: W.n_db].copy()          # _bench_baseline_latency.py:336-339
    faiss.normalize_L2(db)
    index = faiss.IndexFlatIP(db.shape[1])
    index.add(db)
    qd = desc[W.n_db:].copy()
    faiss.normalize_L2(qd)

    def one_query(q):   # _bench_baseline_latency.py:343-350 minus temporal mask
        _, ind = index.search(qd[q - W.n_db: q - W.n_db + 1], MAX_K)
        return int(ind[0][0])

    query_lat, hit = timed_query(W, one_query)
    breakdown = {"descriptor_f32": db.nbytes / W.n_db}
    entry = make_entry(
        enc.descriptor_dim, breakdown, build_lat, query_lat, hit,
        "BEVPlace++ ms-ft (vendored REIN, 8 rotations inside the model, batch 1); "
        "query = faiss IndexFlatIP(8192) CPU top-10 as the harness.")
    del model
    torch.cuda.empty_cache()
    return entry


# ------------------------------------------------------------------ SeqOT
def bench_seqot(W):
    import faiss
    import torch
    from _seqot_eval import featureExtracter, GeM, load_range_stack, window, SEQL, GEM_SEQLEN
    from _seqot_prepare import SENSOR_PROJ, PROJ_H, PROJ_W, MAX_RANGE, range_projection
    cap_faiss_threads(faiss)
    fe = featureExtracter(seqL=SEQL).to(DEVICE)     # _bench_baseline_latency.py:390-397
    fe.load_state_dict(torch.load(SEQOT_CKPT, map_location=DEVICE,
                                  weights_only=False)["state_dict"])
    fe.eval()
    gem = GeM().to(DEVICE)
    gem.load_state_dict(torch.load(GEM_CKPT, map_location=DEVICE,
                                   weights_only=False)["state_dict"])
    gem.eval()
    n = W.n_total
    stack = load_range_stack(SEQOT_DATA / "kitti_00", 0)  # _bench_baseline_latency.py:415
    assert len(stack) >= n, f"seqot range stack too short: {len(stack)} < {n}"
    stack = stack[:n]

    # Untimed DB prep: sub + final descriptors over the n-frame subset
    # (windows clamp at the subset edge), _bench_baseline_latency.py:421-431.
    sub = np.zeros((n, 256), np.float32)
    final = np.zeros((n, 256), np.float32)
    with torch.no_grad():
        for i in range(0, n, 16):
            idx = np.arange(i, min(i + 16, n))
            w = np.stack([window(j, SEQL, n) for j in idx])
            sub[idx] = fe(torch.from_numpy(stack[w]).to(DEVICE)).cpu().numpy()
        for i in range(0, n, 512):
            idx = np.arange(i, min(i + 512, n))
            w = np.stack([window(j, GEM_SEQLEN, n) for j in idx])
            final[idx] = gem(torch.from_numpy(sub[w]).to(DEVICE)).squeeze(1).cpu().numpy()
    proj = SENSOR_PROJ["kitti"]

    def prep(i):
        return np.ascontiguousarray(W.points[i][:, :3], np.float64)

    def step(i, pts):   # amortized per-new-keyframe cost, _bench_baseline_latency.py:451-465
        range_projection(pts, fov_up=proj["fov_up"], fov_down=proj["fov_down"],
                         proj_H=PROJ_H, proj_W=PROJ_W, max_range=MAX_RANGE, cut_z=False)
        with torch.no_grad():
            fe(torch.from_numpy(stack[window(i, SEQL, n)][None]).to(DEVICE))
            gem(torch.from_numpy(sub[window(i, GEM_SEQLEN, n)][None]).to(DEVICE))

    build_lat = timed_build(W, step, prep=prep, use_cuda=True)

    db = final[: W.n_db].copy()
    faiss.normalize_L2(db)
    index = faiss.IndexFlatIP(db.shape[1])
    index.add(db)
    qd = final[W.n_db:].copy()
    faiss.normalize_L2(qd)

    def one_query(q):   # _bench_baseline_latency.py:490-497 minus temporal mask
        _, ind = index.search(qd[q - W.n_db: q - W.n_db + 1], MAX_K)
        return int(ind[0][0])

    query_lat, hit = timed_query(W, one_query)
    breakdown = {"descriptor_f32": db.nbytes / W.n_db}
    entry = make_entry(
        256, breakdown, build_lat, query_lat, hit,
        "SeqOT (seqL=3 sub-descriptor forward + seqlen=20 GeM, batch 1, past "
        "sub-descriptors cached as in the source bench); query = faiss "
        "IndexFlatIP(256) CPU top-10. Storage = final 256D descriptor (the "
        "retrieval index); the sub-descriptor window cache is build-time state.")
    del fe, gem, stack
    torch.cuda.empty_cache()
    return entry


# ------------------------------------------------------------- RING/RING++
def bench_ring_family(W, variant):
    import torch
    if str(RINGSHARP) not in sys.path:
        sys.path.insert(0, str(RINGSHARP))  # _bench_ring_latency.py:58
    from glnet.utils.data_utils.point_clouds import generate_feats, generate_bev
    from glnet.models.utils import estimate_yaw
    from torch_radon import ParallelBeam
    angles = torch.FloatTensor(np.linspace(0, 2 * np.pi, GRID).astype(np.float32))
    radon = ParallelBeam(GRID, angles)      # _bench_ring_latency.py:124-125
    n_ch = 6 if variant == "ringpp" else 1
    specs = torch.zeros((W.n_total, n_ch, GRID, GRID), dtype=torch.float16, device=DEVICE)

    def make_bev(pts):  # _bench_ring_latency.py:128-130
        if variant == "ringpp":
            return generate_feats(pts, Z=1, Y=GRID, X=GRID,
                                  bounds=BOUNDS).reshape(-1, GRID, GRID)
        return generate_bev(pts, Z=1, Y=GRID, X=GRID, bounds=BOUNDS).reshape(-1, GRID, GRID)

    def prep(i):
        return np.ascontiguousarray(W.points[i][:, :3], dtype=np.float32)

    def step(i, pts):   # _bench_ring_latency.py:146-159 + fp16 write of the stored spec
        bev = make_bev(pts).float().to(DEVICE).unsqueeze(0)
        sino = radon.forward(bev)
        m = torch.fft.fft2(sino, dim=-1, norm="ortho")
        specs[i] = torch.sqrt(m.real ** 2 + m.imag ** 2 + 1e-15)[0].half()

    build_lat = timed_build(W, step, prep=prep, use_cuda=True)
    db = specs[: W.n_db]

    def one_query(q):   # official exhaustive correlation, _bench_ring_latency.py:218-225
        qspec = specs[q].float()
        dists = torch.empty(W.n_db, device=DEVICE)
        for j0 in range(0, W.n_db, CORR_BATCH):
            c = db[j0: j0 + CORR_BATCH].float()
            _, scores, _ = estimate_yaw(
                qspec.unsqueeze(0).expand(len(c), -1, -1, -1), c)
            dists[j0: j0 + len(c)] = 1 - scores.reshape(-1)
        return int(torch.argsort(dists)[0])

    query_lat, hit = timed_query(W, one_query, use_cuda=True)
    breakdown = {"tiring_spec_fp16": db.element_size() * db.nelement() / W.n_db}
    entry = make_entry(
        n_ch * GRID * GRID, breakdown, build_lat, query_lat, hit,
        f"Official {'RING++ (6-ch LPD feats)' if variant == 'ringpp' else 'RING (occupancy)'}; "
        "no global descriptor: the DB holds the correlation-ready TIRING spectrum "
        "(descriptor_dim = its element count) and queries rank the FULL DB by "
        "circular correlation (O(N)). Spec fp16 on GPU as the source bench.")
    del specs, db, radon
    torch.cuda.empty_cache()
    return entry


# ---------------------------------------------------------------- RING#-L
def bench_ringsharp(W):
    import faiss
    import torch
    if str(RINGSHARP) not in sys.path:
        sys.path.insert(0, str(RINGSHARP))
    import _ringsharp_eval as rs                      # official glnet wiring + e2cnn load
    from _pdb_ringsharp_eval import DEFAULT_WEIGHT    # paper-era default checkpoint
    cap_faiss_threads(faiss)
    model, trans_cnn = rs.load_model(str(DEFAULT_WEIGHT), DEVICE)
    n = W.n_total
    glob = np.zeros((n, 160), np.float32)

    def prep(i):
        return np.ascontiguousarray(W.points[i][:, :3], dtype=np.float64)

    with torch.no_grad():   # untimed probe for stored-array shapes
        bev0 = rs.generate_bev(prep(0), Z=rs.NSC_Z, Y=rs.NSC_Y, X=rs.NSC_X,
                               bounds=rs.NSC_BOUNDS)
        out0 = model({"pc": bev0.unsqueeze(0).to(DEVICE), "img": None})
        bt0 = trans_cnn(out0["bev"])
    specs = torch.zeros((n, *out0["spec"].shape[1:]), dtype=torch.float16)  # cpu, as _pdb
    bts = torch.zeros((n, *bt0.shape[1:]), dtype=torch.float16)             # cpu, as _pdb

    def step(i, pts):
        # build = official BEV20 (_bench_ring_latency.py:284-291) + forward +
        # trans_cnn; stored set glob/spec/bts as _pdb_ringsharp_eval.py:118-129.
        bev = rs.generate_bev(pts, Z=rs.NSC_Z, Y=rs.NSC_Y, X=rs.NSC_X,
                              bounds=rs.NSC_BOUNDS)
        with torch.no_grad():
            out = model({"pc": bev.unsqueeze(0).to(DEVICE), "img": None})
            bts[i] = trans_cnn(out["bev"])[0].half().cpu()
        glob[i] = out["global"].float().cpu().numpy()[0]
        specs[i] = out["spec"][0].half().cpu()

    build_lat = timed_build(W, step, prep=prep, use_cuda=True)

    coarse = glob[: W.n_db].copy()  # coarse stage of compute_recall_cosine_then_rerank
    faiss.normalize_L2(coarse)      # (baselines/eval_utils.py:219-231)
    index = faiss.IndexFlatIP(coarse.shape[1])
    index.add(coarse)
    qg = glob[W.n_db:].copy()
    faiss.normalize_L2(qg)
    n_coarse = min(RS_N_COARSE, W.n_db)

    def one_query(q):
        _, ind = index.search(qg[q - W.n_db: q - W.n_db + 1], n_coarse)
        cand = ind[0]
        # rerank verbatim from _pdb_ringsharp_eval.py:151-176 (rerank_fn),
        # q = query-frame row, cand = DB rows.
        qspec = specs[q].float().to(DEVICE)
        cspec = specs[cand].float().to(DEVICE)
        with torch.no_grad():
            _, scores, angles = rs.estimate_yaw(
                qspec.unsqueeze(0).expand(len(cand), -1, -1, -1), cspec)
        rot_dist = (1 - scores).cpu().numpy().reshape(-1)
        angles = (angles * 2 * np.pi / specs.shape[-2]).cpu().numpy().reshape(-1)
        order = np.argsort(rot_dist)
        m = min(RS_N_TRANS, len(cand))
        top = order[:m]
        with torch.no_grad():
            bt = bts[np.concatenate(([q], cand[top]))].to(DEVICE).float()
            qbt, cbt = bt[:1], bt[1:]
            ang = torch.from_numpy(angles[top]).float().to(DEVICE)
            qrot = rs.rotate_bev_batch(qbt, ang)
            qrot_e = rs.rotate_bev_batch(qbt, ang - np.pi)
            _, _, e1, _ = rs.solve_translation(qrot, cbt)
            _, _, e2, _ = rs.solve_translation(qrot_e, cbt)
            errs = np.minimum(e1.cpu().numpy().reshape(-1),
                              e2.cpu().numpy().reshape(-1))
        order[:m] = top[np.argsort(errs)]
        return int(cand[order][0])

    query_lat, hit = timed_query(W, one_query, use_cuda=True)
    db_specs, db_bts = specs[: W.n_db], bts[: W.n_db]
    breakdown = {
        "global_f32": coarse.nbytes / W.n_db,
        "rot_spec_fp16": db_specs.element_size() * db_specs.nelement() / W.n_db,
        "trans_bt_fp16": db_bts.element_size() * db_bts.nelement() / W.n_db,
    }
    entry = make_entry(
        160, breakdown, build_lat, query_lat, hit,
        f"RING#-L, weight = {Path(str(DEFAULT_WEIGHT)).name} (timing is "
        f"weight-independent); n_coarse={n_coarse}, n_trans={RS_N_TRANS}. Query = "
        "faiss 160D cosine -> estimate_yaw rotation rerank -> cached-bts "
        "translation rerank (the _pdb_ringsharp_eval deployed path). Packed "
        "occupancy BEV excluded from storage: rerank_fn indexes only spec+bts. "
        "spec/bts stored on host fp16; per-query H2D copies are timed.")
    del model, trans_cnn, specs, bts
    torch.cuda.empty_cache()
    return entry


# ----------------------------------------------------------------- NSD(-H)
def bench_nsd(W, variant):
    """Deployed B3 two-pass build (_bench_latency_memory.py stages) + two-index
    union query (_headline_latency.py). variant: 'nsd' (aux BEV 16x8 spectrum,
    128D magnitude index) or 'nsdh' (height-polar 20x12, 240D index + align)."""
    import copy as _copy
    import torch
    from _height_candidate_fusion import (make_model, make_height_grids, minmax,
                                          column_align_scores, align_scores)
    from _pbev_branch_fusion import BP, variant_inputs
    from _hspec_branch_fusion import hspec_inputs
    from evaluate_kitti_checkpoint import (_build_eval_graph, _cache_to_keyframes,
                                           _phase_sketch)
    from run_kitti_operating_point import (_make_encoder, _normalize,
                                           _project_nsd_layout)
    from build_and_train_edge_classifier import edge_features, range_phase_flat
    from encoding.phase_features import (_phase_sketch as pf_phase_sketch,
                                         _pool_rows as pf_pool_rows)

    n, keep_k = W.n_total, 10
    cfg, model, clf, blob = make_model(DEVICE, 1)   # _height_candidate_fusion.py:210-233
    mu = blob["mu"].astype(np.float32)
    sd = blob["sd"].astype(np.float32)
    r = np.load(REPO / "artifacts/key_remetrize_r.npy").astype(np.float32)
    cache = SubCache(np.load(ROSTER_CACHE), n)

    # ---- per-frame encode (staged as _bench_latency_memory.py:128-168) ----
    enc_cfg = _copy.deepcopy(cfg)
    ranges = enc_cfg["encoding"].get("sensor_elevation_ranges", {})
    if "kitti" in ranges:
        enc_cfg["encoding"]["elevation_range"] = ranges["kitti"]
    encoder = _make_encoder(enc_cfg, DEVICE)
    cyl = np.zeros((n, 16, 4), np.complex64)        # shared cylindrical block
    sk = np.zeros((n, 16, 8), np.complex64) if variant == "nsd" else None
    raw_height = np.zeros((n, 20, 120), np.float32) if variant == "nsdh" else None
    hcoef = np.zeros((n, 20, 12), np.complex64) if variant == "nsdh" else None
    st = {k: 0.0 for k in ("key288", "fft_magnitudes", "range_layout", "aux_spectrum")}

    def encode_one(i, count):
        pts = W.points[i]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        encoder.encode_points(pts).detach().cpu().numpy()   # 288D key
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        encoder.compute_fft_magnitudes(pts)                 # cached as graph x_fft
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        layout = _project_nsd_layout(encoder, pts, 60)      # 16x60 range layout
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        # stored cylindrical spectrum 16x4 (_bench_latency_memory.py:324)
        cyl[i] = _phase_sketch(np.asarray(layout, np.float32)[None], 4)[0]
        if variant == "nsd":
            bev, _ = BP.project(pts, keep_intensity=False)  # _pbev_branch_fusion.py:61-62
            sk[i] = pf_phase_sketch(pf_pool_rows(bev, 16), 8)  # deployed 8 of the 12-freq cache
        else:
            _h, h_raw, _d, _o = make_height_grids(pts)      # _height_candidate_fusion.py:124-148
            raw_height[i] = h_raw
            hcoef[i] = np.fft.rfft(h_raw, axis=-1, norm="ortho")[:, 1:13]  # _hspec_branch_fusion.py:39-40
        t4 = time.perf_counter()
        if count:
            st["key288"] += t1 - t0
            st["fft_magnitudes"] += t2 - t1
            st["range_layout"] += t3 - t2
            st["aux_spectrum"] += t4 - t3
        return (t4 - t0) * 1e3

    for i in range(WARMUP):
        encode_one(i, count=False)
    enc_lat = np.array([encode_one(i, count=(i < W.n_db)) for i in range(n)])

    # ---- batch stages over the joint (DB+query) session graph, amortized ----
    d_raw = cache["descriptors"].astype(np.float32)
    poses = cache["poses"]

    def fwd3(g):    # 3-rep median forward + D2H (_bench_latency_memory.py:212-227)
        with torch.no_grad():
            model(g.to(DEVICE))         # warmup
        reps, emb = [], None
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                emb_t = model(g.to(DEVICE))
            torch.cuda.synchronize()
            reps.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        emb = emb_t.cpu().numpy()
        return emb, float(np.median(reps)) + (time.perf_counter() - t0)

    # graph build + DAC pass-1 (_bench_latency_memory.py:201-232)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    graph = _build_eval_graph(
        keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
        cache=cache, config=cfg, device=DEVICE, temporal_edge_mode="bidirectional",
        temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
        sensor_key="kitti")
    torch.cuda.synchronize()
    t_g1 = time.perf_counter() - t0
    embA, t_f1 = fwd3(graph)

    # edge selection (_bench_latency_memory.py:236-262)
    with torch.no_grad():   # classifier warmup
        clf(torch.from_numpy((np.zeros((32, mu.shape[0]), np.float32) - mu) / sd)
            .float().to(DEVICE))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    d_rm = _normalize(d_raw * r[None, :])
    x_phase = range_phase_flat(cache["nsd_layouts"])
    src, dst, feats, _ = edge_features(embA, d_rm, x_phase, poses, want_labels=False)
    with torch.no_grad():
        logits = clf(torch.from_numpy((feats - mu) / sd).float().to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
    torch.cuda.synchronize()
    probs2 = probs.reshape(n, -1)
    pick = np.argsort(-probs2, axis=1)[:, :keep_k]
    src2 = np.repeat(np.arange(n), keep_k)
    dst2 = dst.reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
    cos2 = feats[:, 0].reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
    prob2 = probs2[np.arange(n)[:, None], pick].reshape(-1)
    l2_2 = np.linalg.norm(embA[src2] - embA[dst2], axis=1)
    t_edge = time.perf_counter() - t0

    # graph rebuild + DAC pass-2 (_bench_latency_memory.py:269-311)
    cfg_t = _copy.deepcopy(cfg)
    cfg_t["keyframe"].setdefault("graph", {})["similarity_threshold"] = 2.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g = _build_eval_graph(
        keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
        cache=cache, config=cfg_t, device=DEVICE, temporal_edge_mode="bidirectional",
        temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
        sensor_key="kitti")
    dev = g.edge_index.device
    ei_new = torch.from_numpy(np.stack([src2, dst2])).long().to(dev)
    attr_new = torch.from_numpy(np.stack([
        np.zeros_like(cos2), np.zeros_like(cos2), cos2,
        np.log1p(l2_2) / 5.0, prob2], axis=1).astype(np.float32)).to(dev)
    g.edge_index = torch.cat([g.edge_index, ei_new], dim=1)
    g.edge_attr = torch.cat([g.edge_attr, attr_new], dim=0)
    g.edge_type = torch.cat(
        [g.edge_type, torch.ones(ei_new.shape[1], dtype=torch.long, device=dev)])
    torch.cuda.synchronize()
    t_g2 = time.perf_counter() - t0
    embB3, t_f2 = fwd3(g)

    # retrieval-side bank derivation from the stored state (index build)
    t0 = time.perf_counter()
    normed = _normalize(embB3)
    if variant == "nsd":
        hk, hfft, hnorm, colbank = variant_inputs(sk, 8)        # _pbev_branch_fusion.py:75-85
    else:
        hk, hfft, hnorm, colbank = hspec_inputs(raw_height, 12)  # _hspec_branch_fusion.py:36-48
    t_bank = time.perf_counter() - t0

    amort_parts = {"graph_pass1": t_g1 + t_f1, "edge_select": t_edge,
                   "graph_pass2": t_g2 + t_f2, "bank_prep": t_bank}
    amort_ms = sum(amort_parts.values()) / n * 1e3
    build_lat = enc_lat[: W.n_db] + amort_ms

    # ---- query: _headline_latency.py:88-104 (disjoint queries: no exclusion) ----
    F = torch.from_numpy(normed).to(DEVICE)
    H = torch.from_numpy(np.ascontiguousarray(hk)).to(DEVICE)
    F_db, H_db = F[: W.n_db], H[: W.n_db]
    k = min(K_NSD, W.n_db)
    wts = (0.5, 1.0, 0.0, 0.5) if variant == "nsd" else (1.0, 1.0, 1.0, 2.0)

    def one_query(q):
        sf_all = F_db @ F[q]
        sh_all = H_db @ H[q]
        cf = torch.topk(sf_all, k).indices
        ch = torch.topk(sh_all, k).indices
        cand = torch.unique(torch.cat([cf, ch])).cpu().numpy()
        s = (wts[0] * minmax(sf_all[cand].cpu().numpy())
             + wts[1] * minmax(sh_all[cand].cpu().numpy())
             + wts[3] * minmax(column_align_scores(colbank, int(q), cand)))
        if variant == "nsdh":
            s = s + wts[2] * minmax(align_scores(hfft, hnorm, int(q), cand))
        return int(cand[int(np.argmax(s))])

    query_lat, hit = timed_query(W, one_query, use_cuda=True)

    aux = sk if variant == "nsd" else hcoef
    breakdown = {
        "key416_f32": normed[: W.n_db].nbytes / W.n_db,
        "cyl_spectrum_16x4_c64": cyl[: W.n_db].nbytes / W.n_db,
        ("bev_spectrum_16x8_c64" if variant == "nsd"
         else "height_spectrum_20x12_c64"): aux[: W.n_db].nbytes / W.n_db,
    }
    entry = make_entry(
        416, breakdown, build_lat, query_lat, hit,
        f"{'NSD (800 stored floats)' if variant == 'nsd' else 'NSD-H (1024 stored floats)'}; "
        "build = per-frame encode + batch B3 stages (two-pass DAC + learned edge "
        "selection + bank derivation) run once over the joint "
        f"{n}-node session graph and amortized per keyframe (each build sample = "
        "encode ms + amortized constant). Query = two top-"
        f"{k} cosine indexes (416D key + {hk.shape[1]}D aux magnitude index derived "
        "from the stored spectrum) -> union -> per-query min-max fusion "
        f"{wts} + cyclic alignment. Alignment banks are index-side derivations of "
        "the stored coefficients (counted in bank_prep, not extra storage).",
        extra={"build_breakdown": {
            "per_frame_encode_ms": {kk: round(v / W.n_db * 1e3, 3) for kk, v in st.items()},
            "amortized_batch_ms_per_kf": {kk: round(v / n * 1e3, 4) for kk, v in amort_parts.items()},
            "n_graph_nodes": n}})
    del model, clf, encoder, graph, g, F, H, F_db, H_db
    torch.cuda.empty_cache()
    return entry


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n-db", type=int, default=1000)
    ap.add_argument("--n-queries", type=int, default=200)
    args = ap.parse_args()

    results = json.loads(OUT.read_text()) if OUT.exists() else {"workload": {}, "methods": {}}
    todo = [m for m in args.methods if args.overwrite or m not in results["methods"]]
    skipped = [m for m in args.methods if m not in todo]
    if skipped:
        print(f"[skip] existing entries (--overwrite to redo): {skipped}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return
    old = results.get("workload") or {}
    if results["methods"] and old and (old.get("n_db") != args.n_db
                                       or old.get("n_queries") != args.n_queries):
        sys.exit(f"{OUT} holds entries measured at n_db={old.get('n_db')}/"
                 f"n_queries={old.get('n_queries')}; matched conditions forbid mixing "
                 "-- rerun with matching sizes or delete the file")

    W = Workload(args.n_db, args.n_queries)
    results["workload"] = {
        "sequence": "kitti/00",
        "roster_cache": str(ROSTER_CACHE.relative_to(REPO)),
        "roster_matches_val_cache_056e0a02": VAL_CACHE.exists(),
        "n_db": W.n_db, "n_queries": W.n_q,
        "db_frames": [0, W.n_db], "query_frames": [W.n_db, W.n_total],
        "warmup": WARMUP, "batch": 1,
        "protocol": ("point clouds preloaded to RAM (disk I/O excluded); queries "
                     "disjoint from DB (no self/temporal exclusion); "
                     "torch.cuda.synchronize around every GPU timed region; "
                     "first 10 build frames / queries warmed untimed"),
    }
    bench = {"scpp": bench_scpp, "bevplace": bench_bevplace, "seqot": bench_seqot,
             "ring": lambda w: bench_ring_family(w, "ring"),
             "ringpp": lambda w: bench_ring_family(w, "ringpp"),
             "ringsharp": bench_ringsharp,
             "nsd": lambda w: bench_nsd(w, "nsd"),
             "nsdh": lambda w: bench_nsd(w, "nsdh")}
    for m in todo:
        print(f"\n[bench] {m} ...", flush=True)
        entry = bench[m](W)
        results["methods"][m] = entry
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1))   # incremental, resumable
        print(f"[bench] {m}: build {entry['build_ms']['mean']} ms/frame, "
              f"query {entry['query_ms']['mean']} ms, "
              f"{entry['storage_bytes_per_frame']:.0f} B/frame, "
              f"top1<5m {entry['sanity_top1_within_5m']}", flush=True)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    print(f"\nwrote {OUT}\n")
    print(f"{'method':<11}{'dim':>8}{'B/frame':>10}{'build ms':>10}{'query ms':>10}{'q p95':>9}")
    for m, e in results["methods"].items():
        print(f"{m:<11}{e['descriptor_dim']:>8}{e['storage_bytes_per_frame']:>10.0f}"
              f"{e['build_ms']['mean']:>10.2f}{e['query_ms']['mean']:>10.2f}"
              f"{e['query_ms']['p95']:>9.2f}")


if __name__ == "__main__":
    main()
