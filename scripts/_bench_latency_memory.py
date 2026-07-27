#!/usr/bin/env python3
"""Full end-to-end latency + peak-memory benchmark of the deployed B3 pipeline.

Pipeline instrumented = exactly scripts/eval_rank_k.py / _dump_b3_yaw_perquery.py.

INDEXING stages (per keyframe = stage total / N):
  encode        encoder.encode_points (288D key) + encoder.compute_fft_magnitudes
                (cached as graph.x_fft; NOT read by the deployed no_interdiff
                model) + _project_nsd_layout (16x60 range phase layout) +
                BEVProjector.project + interpolate_bev_image (60-sector BEV
                layout, max/h8/r1-80/z-3-5). Timed on the sequence's own raw
                scans (src/data loaders) over an evenly spaced sample of the
                cache's keyframe scan_ids. Point-load disk I/O timed separately.
  graph_pass1   _cache_to_keyframes + _build_eval_graph (temporal bidirectional
                + cosine-threshold similarity edges) then DAC pass-1 forward.
  edge_select   d_rm re-metrization + range_phase_flat + edge_features
                (30 candidates/node, incl. NxN cosine + phase-alignment feats)
                + EdgeMLP scoring + top-10 selection.
  graph_pass2   temporal-only graph rebuild (similarity_threshold=2.0) +
                classifier-edge append + DAC pass-2 forward.
  sketch        _pool_rows(bev,16,max) + _phase_sketch(range,4)/(bev,8) +
                magnitude keys (the stored 384D phase sketch + search keys).

QUERY stages (per query, all _find_queries(poses, 5.0, 30) queries):
  search        three top-800 cosine searches: 416D final key, 64D range-sketch
                key, 128D BEV-sketch key
  union         np.unique over the pooled candidate lists
  rerank        416D cosine distances + _phase_sketch_distances_fft (BEV+range)
                + fused argsort at the deployed grid point (w_b, w_r) = (0.5, 0)

MEMORY: torch.cuda.reset_peak_memory_stats()/max_memory_allocated() around each
GPU stage; host VmRSS/VmHWM (/proc/self/status) + getrusage ru_maxrss.

Run inside nvcr.io/nvidia/pyg:26.01-py3 with the repo at /workspace/...:
  python scripts/_bench_latency_memory.py
Writes results/latency_memory_bench.json and prints a table.
"""
from __future__ import annotations
import copy, json, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import (_apply_encoder_preset, _make_model, _build_eval_graph,
    _cache_to_keyframes, _pool_rows, _phase_sketch, _phase_sketch_keys,
    _phase_sketch_distances_fft)
from run_kitti_operating_point import (_find_queries, _topk_cosine, _normalize,
    _make_encoder, _project_nsd_layout)
from build_and_train_edge_classifier import edge_features, range_phase_flat, EdgeMLP, N_FEAT
from encoding.bev_image import BEVProjector, interpolate_bev_image

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
RVEC = REPO / "artifacts/key_remetrize_r.npy"
CLS = REPO / "artifacts/edge_classifier.pt"
CKPT = REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth"
OUT = None  # set below once N_COARSE is known
DATA_ROOTS = {"kitti": "/workspace/data/kitti/dataset", "nclt": "/workspace/data/nclt",
              "mulran": "/workspace/data/mulran"}
BEV_DIRS = {"kitti": "preprocessed_kitti_bev_layout", "nclt": "preprocessed_nclt_bev_layout",
            "helipr": "preprocessed_helipr_bev_layout", "mulran": "preprocessed_mulran_bev_layout"}
SEQS = [("kitti", "05", "KITTI_05"), ("kitti", "00", "KITTI_00"),
        ("mulran", "KAIST03", "MulRan_KAIST03"), ("nclt", "2012-01-08", "NCLT_2012-01-08")]
DTH, SKIP, KEEP_K = 5.0, 30, 10
N_COARSE = int(os.environ.get("NSC_N_COARSE", "800"))  # tab:ksweep timing sweep
WB, WR = 0.5, 0.0  # deployed grid point
PRIMARY_ONLY = os.environ.get("NSC_PRIMARY_ONLY", "0") == "1"  # tab:ksweep 416D-only row
OUT = REPO / ("results/latency_memory_bench.json" if N_COARSE == 800 and not PRIMARY_ONLY else f"results/latency_memory_bench_K{N_COARSE}" + ("_primary" if PRIMARY_ONLY else "") + ".json")
ENCODE_SAMPLE = int(os.environ.get("NSC_ENCODE_SAMPLE", "150"))
DEVICE = "cuda"


class CacheShim:
    """NpzFile stand-in with materialized arrays so repeated cache[...] access
    inside _build_eval_graph does not re-decompress from disk mid-timing."""

    def __init__(self, npz):
        self.files = list(npz.files)
        self._d = {k: npz[k] for k in npz.files}

    def __getitem__(self, k):
        return self._d[k]


def _rss():
    vm = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM")):
            k, v = line.split(":", 1)
            vm[k] = round(int(v.split()[0]) / 1024.0, 1)  # MB
    vm["ru_maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    return vm


def _gpu_reset():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _gpu_peak_mb():
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def _make_loader(sensor, seq):
    if sensor == "kitti":
        from data.kitti_loader import KITTILoader
        return KITTILoader(DATA_ROOTS["kitti"], seq, lazy_load=True)
    if sensor == "nclt":
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(DATA_ROOTS["nclt"], seq, lazy_load=True)
    if sensor == "mulran":
        from data.mulran_loader import MulRanLoader
        return MulRanLoader(DATA_ROOTS["mulran"], seq, lazy_load=True)
    raise ValueError(sensor)


def bench_encode(sensor, seq, cfg, scan_ids, n_sample=ENCODE_SAMPLE):
    """Time the cache-generation encode calls on real raw scans."""
    try:
        loader = _make_loader(sensor, seq)
    except Exception as e:  # dataset missing -> report, do not fake numbers
        return {"error": f"loader failed: {type(e).__name__}: {e}"}
    enc_cfg = copy.deepcopy(cfg)
    ranges = enc_cfg["encoding"].get("sensor_elevation_ranges", {})
    if sensor in ranges:
        enc_cfg["encoding"]["elevation_range"] = ranges[sensor]
    encoder = _make_encoder(enc_cfg, DEVICE)
    projector = BEVProjector(n_sectors=60, max_range=80.0, min_range=1.0, z_min=-3.0,
                             height_encoding="max", n_height_layers=8, z_max=5.0)
    ids = [int(s) for s in scan_ids if 0 <= int(s) < len(loader)]
    n_oob = len(scan_ids) - len(ids)
    step = max(1, len(ids) // n_sample)
    sample = ids[::step][:n_sample]

    for sid in sample[:3]:  # warmup
        pts = loader[sid]["points"]
        encoder.encode_points(pts).detach().cpu().numpy()
        encoder.compute_fft_magnitudes(pts)
        _project_nsd_layout(encoder, pts, n_layout_sectors=60)
        bev, _ = projector.project(pts, keep_intensity=False)
        interpolate_bev_image(bev, method="linear", n_channels=1)
    _gpu_reset()

    t_io = t_key = t_fft = t_rl = t_bl = 0.0
    n_pts = []
    for sid in sample:
        t0 = time.perf_counter()
        pts = loader[sid]["points"]
        t1 = time.perf_counter()
        encoder.encode_points(pts).detach().cpu().numpy()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        encoder.compute_fft_magnitudes(pts)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        _project_nsd_layout(encoder, pts, n_layout_sectors=60)
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        bev, _ = projector.project(pts, keep_intensity=False)
        interpolate_bev_image(bev, method="linear", n_channels=1)
        t5 = time.perf_counter()
        t_io += t1 - t0; t_key += t2 - t1; t_fft += t3 - t2
        t_rl += t4 - t3; t_bl += t5 - t4
        n_pts.append(len(pts))
    k = len(sample)
    ms = lambda t: round(t / k * 1e3, 3)
    return {"n_sampled": k, "scan_ids_out_of_range": n_oob,
            "mean_points_per_scan": int(np.mean(n_pts)),
            "ms_per_kf_pointload_io": ms(t_io),
            "ms_per_kf_key288": ms(t_key),
            "ms_per_kf_fft_magnitudes": ms(t_fft),
            "ms_per_kf_range_layout": ms(t_rl),
            "ms_per_kf_bev_layout": ms(t_bl),
            "ms_per_kf_total": ms(t_key + t_fft + t_rl + t_bl),
            "gpu_peak_mb": _gpu_peak_mb()}


def bench_sequence(model, clf, mu, sd, r, rm_cfg, sensor, seq, name,
                   encode_sample=ENCODE_SAMPLE, max_queries=None, skip_encode=False):
    res = {"name": name}
    npz = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
    cache = CacheShim(npz)
    d_raw = cache["descriptors"].astype(np.float32)
    poses = cache["poses"]
    n = len(d_raw)
    res["N"] = n
    rss = {}

    # ---------------- indexing 1: encode ----------------
    if skip_encode:
        res["encode"] = {"skipped": True}
    else:
        res["encode"] = bench_encode(sensor, seq, rm_cfg, cache["scan_ids"], encode_sample)
    rss["encode"] = _rss()

    # ---------------- indexing 2: graph build + DAC pass-1 ----------------
    _gpu_reset()
    t0 = time.perf_counter()
    graph = _build_eval_graph(
        keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
        cache=cache, config=rm_cfg, device=DEVICE, temporal_edge_mode="bidirectional",
        temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
        sensor_key=sensor)
    torch.cuda.synchronize()
    t_build1 = time.perf_counter() - t0
    m_build1 = _gpu_peak_mb()
    with torch.no_grad():  # warmup
        model(graph.to(DEVICE))
    _gpu_reset()
    reps = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            embA_t = model(graph.to(DEVICE))
        torch.cuda.synchronize()
        reps.append(time.perf_counter() - t0)
    t_fwd1 = float(np.median(reps))
    m_fwd1 = _gpu_peak_mb()
    t0 = time.perf_counter()
    embA = embA_t.cpu().numpy()
    t_d2h1 = time.perf_counter() - t0
    res["graph_pass1"] = {"build_s": round(t_build1, 4), "forward_s": round(t_fwd1, 4),
                          "d2h_s": round(t_d2h1, 4),
                          "n_edges": int(graph.edge_index.shape[1]),
                          "ms_per_kf": round((t_build1 + t_fwd1 + t_d2h1) / n * 1e3, 4),
                          "gpu_peak_mb_build": m_build1, "gpu_peak_mb_forward": m_fwd1}
    rss["graph_pass1"] = _rss()

    # ---------------- indexing 3: edge selection ----------------
    t0 = time.perf_counter()
    d_rm = _normalize(d_raw * r[None, :])
    x_phase = range_phase_flat(cache["nsd_layouts"])
    t_prep = time.perf_counter() - t0
    t0 = time.perf_counter()
    src, dst, feats, _ = edge_features(embA, d_rm, x_phase, poses, want_labels=False)
    t_feat = time.perf_counter() - t0
    with torch.no_grad():  # warmup classifier
        clf(torch.from_numpy((feats[:1024] - mu) / sd).float().to(DEVICE))
    _gpu_reset()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = clf(torch.from_numpy((feats - mu) / sd).float().to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
    torch.cuda.synchronize()
    t_clf = time.perf_counter() - t0
    m_clf = _gpu_peak_mb()
    t0 = time.perf_counter()
    probs2 = probs.reshape(n, -1)
    pick = np.argsort(-probs2, axis=1)[:, :KEEP_K]
    src2 = np.repeat(np.arange(n), KEEP_K)
    dst2 = dst.reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
    cos2 = feats[:, 0].reshape(n, -1)[np.arange(n)[:, None], pick].reshape(-1)
    prob2 = probs2[np.arange(n)[:, None], pick].reshape(-1)
    l2_2 = np.linalg.norm(embA[src2] - embA[dst2], axis=1)
    t_select = time.perf_counter() - t0
    t_edge = t_prep + t_feat + t_clf + t_select
    res["edge_select"] = {"prep_s": round(t_prep, 4), "edge_features_s": round(t_feat, 4),
                          "classifier_s": round(t_clf, 4), "topk_select_s": round(t_select, 4),
                          "n_scored_edges": int(len(feats)),
                          "ms_per_kf": round(t_edge / n * 1e3, 4), "gpu_peak_mb": m_clf}
    rss["edge_select"] = _rss()

    # ---------------- indexing 4: graph rebuild + DAC pass-2 ----------------
    cfg_t = copy.deepcopy(rm_cfg)
    cfg_t["keyframe"].setdefault("graph", {})["similarity_threshold"] = 2.0
    _gpu_reset()
    t0 = time.perf_counter()
    g = _build_eval_graph(
        keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
        cache=cache, config=cfg_t, device=DEVICE, temporal_edge_mode="bidirectional",
        temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
        sensor_key=sensor)
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
    t_build2 = time.perf_counter() - t0
    m_build2 = _gpu_peak_mb()
    with torch.no_grad():  # warmup
        model(g.to(DEVICE))
    _gpu_reset()
    reps = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            embB3_t = model(g.to(DEVICE))
        torch.cuda.synchronize()
        reps.append(time.perf_counter() - t0)
    t_fwd2 = float(np.median(reps))
    m_fwd2 = _gpu_peak_mb()
    t0 = time.perf_counter()
    embB3 = embB3_t.cpu().numpy()
    t_d2h2 = time.perf_counter() - t0
    res["graph_pass2"] = {"build_s": round(t_build2, 4), "forward_s": round(t_fwd2, 4),
                          "d2h_s": round(t_d2h2, 4),
                          "n_edges": int(g.edge_index.shape[1]),
                          "ms_per_kf": round((t_build2 + t_fwd2 + t_d2h2) / n * 1e3, 4),
                          "gpu_peak_mb_build": m_build2, "gpu_peak_mb_forward": m_fwd2}
    rss["graph_pass2"] = _rss()

    # ---------------- indexing 5: sketch precompute ----------------
    stride = "_stride1" if sensor == "nclt" else ""
    bev_raw = np.load(REPO / "data" / BEV_DIRS[sensor] /
        f"{sensor}_bev_layout_{seq}_s60_max_r1-80_z-3-5_h8{stride}.npz")["bev_layouts"].astype(np.float32)
    range_layouts = cache["nsd_layouts"].astype(np.float32)
    for _ in range(2):  # 1st iteration = warmup, 2nd timed
        t0 = time.perf_counter()
        normed = _normalize(embB3)
        bev = _pool_rows(bev_raw, 16, "max")
        range_sketch = _phase_sketch(range_layouts, 4)
        bev_sketch = _phase_sketch(bev, 8)
        range_keys = _phase_sketch_keys(range_sketch)
        bev_keys = _phase_sketch_keys(bev_sketch)
        t_sketch = time.perf_counter() - t0
    res["sketch"] = {"total_s": round(t_sketch, 4), "ms_per_kf": round(t_sketch / n * 1e3, 4)}
    rss["sketch"] = _rss()

    res["indexing_total_ms_per_kf"] = round(
        (0.0 if skip_encode or "error" in res["encode"] else res["encode"]["ms_per_kf_total"])
        + res["graph_pass1"]["ms_per_kf"] + res["edge_select"]["ms_per_kf"]
        + res["graph_pass2"]["ms_per_kf"] + res["sketch"]["ms_per_kf"], 3)

    # ---------------- query stages ----------------
    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    if max_queries:
        queries = queries[:max_queries]
    nq = len(queries)
    res["n_queries"] = nq
    for qi, _mi in queries[:5]:  # warmup
        ce = _topk_cosine(normed, qi, N_COARSE, SKIP)
        cand = np.unique(np.concatenate([ce,
            _topk_cosine(range_keys, qi, N_COARSE, SKIP),
            _topk_cosine(bev_keys, qi, N_COARSE, SKIP)]))
        ed = 1.0 - (normed[cand] @ normed[qi])
        bd = _phase_sketch_distances_fft(bev_sketch[qi], bev_sketch[cand], n_sectors=60)
        rd = _phase_sketch_distances_fft(range_sketch[qi], range_sketch[cand], n_sectors=60)
        cand[np.argsort(ed + WB * bd + WR * rd)]

    acc = {k: 0.0 for k in ("s_emb", "s_rng", "s_bev", "union", "r_embd", "r_bev",
                            "r_rng", "r_sort")}
    n_cand = []
    hit_coarse = hit_rerank = 0
    per_query_total = []
    for qi, _mi in queries:
        t0 = time.perf_counter()
        cand_emb = _topk_cosine(normed, qi, N_COARSE, SKIP)
        t1 = time.perf_counter()
        cand_rng = cand_emb[:0] if PRIMARY_ONLY else _topk_cosine(range_keys, qi, N_COARSE, SKIP)
        t2 = time.perf_counter()
        cand_bev = cand_emb[:0] if PRIMARY_ONLY else _topk_cosine(bev_keys, qi, N_COARSE, SKIP)
        t3 = time.perf_counter()
        candidates = np.unique(np.concatenate([cand_emb, cand_rng, cand_bev]))
        t4 = time.perf_counter()
        emb_d = 1.0 - (normed[candidates] @ normed[qi])
        t5 = time.perf_counter()
        bev_d = _phase_sketch_distances_fft(bev_sketch[qi], bev_sketch[candidates], n_sectors=60)
        t6 = time.perf_counter()
        rng_d = _phase_sketch_distances_fft(range_sketch[qi], range_sketch[candidates], n_sectors=60)
        t7 = time.perf_counter()
        fused = emb_d + WB * bev_d + WR * rng_d
        ranked = candidates[np.argsort(fused)]
        t8 = time.perf_counter()
        acc["s_emb"] += t1 - t0; acc["s_rng"] += t2 - t1; acc["s_bev"] += t3 - t2
        acc["union"] += t4 - t3; acc["r_embd"] += t5 - t4; acc["r_bev"] += t6 - t5
        acc["r_rng"] += t7 - t6; acc["r_sort"] += t8 - t7
        n_cand.append(len(candidates))
        per_query_total.append(t8 - t0)
        hit_coarse += int(np.linalg.norm(positions[cand_emb[0]] - positions[qi]) < DTH)
        hit_rerank += int(np.linalg.norm(positions[ranked[0]] - positions[qi]) < DTH)
    ms = lambda t: round(t / nq * 1e3, 4)
    search_ms = ms(acc["s_emb"] + acc["s_rng"] + acc["s_bev"])
    rerank_ms = ms(acc["r_embd"] + acc["r_bev"] + acc["r_rng"] + acc["r_sort"])
    res["query"] = {
        "search": {"emb416_ms": ms(acc["s_emb"]), "range_key_ms": ms(acc["s_rng"]),
                   "bev_key_ms": ms(acc["s_bev"]), "ms_per_query": search_ms},
        "union": {"ms_per_query": ms(acc["union"]), "avg_candidates": round(float(np.mean(n_cand)), 1)},
        "rerank": {"emb_dist_ms": ms(acc["r_embd"]), "bev_fft_ms": ms(acc["r_bev"]),
                   "range_fft_ms": ms(acc["r_rng"]), "sort_ms": ms(acc["r_sort"]),
                   "ms_per_query": rerank_ms},
        "total_ms_per_query": round(search_ms + ms(acc["union"]) + rerank_ms, 4),
        "p50_ms": round(float(np.percentile(per_query_total, 50)) * 1e3, 4),
        "p95_ms": round(float(np.percentile(per_query_total, 95)) * 1e3, 4)}
    rss["query"] = _rss()
    res["sanity"] = {"coarse_R@1": round(hit_coarse / nq, 4),
                     "rerank_R@1": round(hit_rerank / nq, 4)}
    res["host_rss_mb"] = rss
    gpu_peaks = [res["graph_pass1"]["gpu_peak_mb_build"], res["graph_pass1"]["gpu_peak_mb_forward"],
                 res["edge_select"]["gpu_peak_mb"], res["graph_pass2"]["gpu_peak_mb_build"],
                 res["graph_pass2"]["gpu_peak_mb_forward"]]
    if isinstance(res["encode"], dict) and "gpu_peak_mb" in res["encode"]:
        gpu_peaks.append(res["encode"]["gpu_peak_mb"])
    res["gpu_peak_mb_overall"] = max(gpu_peaks)
    return res


def main():
    torch.backends.cudnn.benchmark = False
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    rm_cfg = copy.deepcopy(cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(RVEC)}
    r = np.load(RVEC).astype(np.float32)
    model = _make_model(rm_cfg, CKPT, DEVICE)
    blob = torch.load(CLS, map_location="cpu", weights_only=False)
    clf = EdgeMLP(N_FEAT)
    clf.load_state_dict(blob["state_dict"])
    clf.eval().to(DEVICE)
    mu, sd = blob["mu"].astype(np.float32), blob["sd"].astype(np.float32)
    model_resident_mb = round(torch.cuda.memory_allocated() / 2**20, 1)

    print("[warmup] full pipeline pass on kitti/05 (untimed)", flush=True)
    bench_sequence(model, clf, mu, sd, r, rm_cfg, "kitti", "05", "warmup",
                   encode_sample=10, max_queries=50)

    report = {"env": {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda": torch.version.cuda, "image": "nvcr.io/nvidia/pyg:26.01-py3",
        "checkpoint": str(CKPT.relative_to(REPO)), "seed": "seed1",
        "n_coarse": N_COARSE, "keep_k": KEEP_K, "skip_frames": SKIP,
        "distance_threshold_m": DTH, "fusion_weights_bev_range": [WB, WR],
        "fusion_norm": "raw", "encode_sample_per_seq": ENCODE_SAMPLE,
        "model_resident_gpu_mb": model_resident_mb,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, "sequences": {}}
    for sensor, seq, name in SEQS:
        print(f"[bench] {name} ...", flush=True)
        res = bench_sequence(model, clf, mu, sd, r, rm_cfg, sensor, seq, name)
        report["sequences"][name] = res
        print(f"[bench] {name}: idx {res['indexing_total_ms_per_kf']} ms/kf, "
              f"query {res['query']['total_ms_per_query']} ms/q, "
              f"R@1 coarse/rerank {res['sanity']['coarse_R@1']}/{res['sanity']['rerank_R@1']}",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=1)
    print("saved", OUT, flush=True)

    # ---------------- table ----------------
    S = report["sequences"]
    print("\n=== INDEXING (ms per keyframe; encode timed on raw scans) ===")
    hdr = (f"{'sequence':<17}{'N':>6} {'encode':>8} {'graph1':>8} {'fwd1':>7} "
           f"{'edgesel':>8} {'graph2':>8} {'fwd2':>7} {'sketch':>7} {'TOTAL':>8}")
    print(hdr)
    for name, s in S.items():
        enc = s["encode"].get("ms_per_kf_total", float("nan")) if isinstance(s["encode"], dict) else float("nan")
        g1 = s["graph_pass1"]; g2 = s["graph_pass2"]
        print(f"{name:<17}{s['N']:>6} {enc:>8.2f} "
              f"{(g1['build_s'] / s['N'] * 1e3):>8.3f} {(g1['forward_s'] + g1['d2h_s']) / s['N'] * 1e3:>7.3f} "
              f"{s['edge_select']['ms_per_kf']:>8.3f} "
              f"{(g2['build_s'] / s['N'] * 1e3):>8.3f} {(g2['forward_s'] + g2['d2h_s']) / s['N'] * 1e3:>7.3f} "
              f"{s['sketch']['ms_per_kf']:>7.3f} {s['indexing_total_ms_per_kf']:>8.2f}")
    print("\n=== QUERY (ms per query) ===")
    print(f"{'sequence':<17}{'n_q':>6} {'search':>8} {'union':>7} {'rerank':>8} "
          f"{'TOTAL':>8} {'p95':>8} {'cands':>7}")
    for name, s in S.items():
        q = s["query"]
        print(f"{name:<17}{s['n_queries']:>6} {q['search']['ms_per_query']:>8.3f} "
              f"{q['union']['ms_per_query']:>7.3f} {q['rerank']['ms_per_query']:>8.3f} "
              f"{q['total_ms_per_query']:>8.3f} {q['p95_ms']:>8.3f} "
              f"{q['union']['avg_candidates']:>7.0f}")
    print("\n=== MEMORY ===")
    print(f"{'sequence':<17}{'gpu_peak_MB':>12} {'fwd1_MB':>9} {'fwd2_MB':>9} {'host_HWM_MB':>12}")
    for name, s in S.items():
        print(f"{name:<17}{s['gpu_peak_mb_overall']:>12.0f} "
              f"{s['graph_pass1']['gpu_peak_mb_forward']:>9.0f} "
              f"{s['graph_pass2']['gpu_peak_mb_forward']:>9.0f} "
              f"{s['host_rss_mb']['query']['VmHWM']:>12.0f}")
    print("\n=== SANITY (must match deployed B3 seed1 numbers) ===")
    for name, s in S.items():
        print(f"{name:<17} coarse R@1 {s['sanity']['coarse_R@1']:.4f}   "
              f"rerank(0.5,0) R@1 {s['sanity']['rerank_R@1']:.4f}")


if __name__ == "__main__":
    main()
