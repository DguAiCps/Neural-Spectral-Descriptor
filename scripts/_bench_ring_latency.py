#!/usr/bin/env python3
"""End-to-end latency + peak-memory benchmark for the RING family, directly
comparable to the NSD benchmark (scripts/_bench_latency_memory.py /
results/latency_memory_bench.json): same four maps (KITTI_05 N=2081,
KITTI_00 N=3506, MulRan_KAIST03 N=5701, NCLT_2012-01-08 N=7467), real point
clouds through the same loaders, per-stage breakdown, warmups,
torch.cuda.synchronize around GPU sections, per-stage
torch.cuda.max_memory_allocated, host VmRSS/VmHWM.

Variants:
  ringpp      official RING++ (6-channel LPD feature BEV, TRO'25 RINGSharp code)
  ring        official RING (1-channel occupancy BEV; identical stages, cheaper
              features and correlation channels)
    INDEXING per keyframe (ENCODE_SAMPLE evenly spaced real scans):
      pointload_io  loader scan fetch (disk I/O, host)
      bev           official generate_feats (ringpp, CPU LPD KNN + voxelize)
                    or generate_bev Z=1 (ring, CPU)
      radon         torch-radon ParallelBeam sinogram (GPU, batch 1)
      tiring_fft    row-FFT magnitude -> TIRING spectrum (GPU, batch 1)
    QUERY (official exhaustive circular-correlation ranking, estimate_yaw
    against the FULL database — O(N) by construction; ms/query over
    QUERY_SAMPLE revisit queries of our protocol, 10 warmup):
      corr_rank     batched circorr over all N + argsort

  ring_sharp_l  RING#-L with the newest available checkpoint (timing is
                weights-independent). INDEXING: pointload_io, bev20 (official
                generate_bev Z=20, CPU), model_forward (batch 1, eval, GPU).
                QUERY at the deployed scripts/_ringsharp_eval.py settings
                (n_coarse=200, n_trans=20):
      coarse        160D cosine matmul + top-200 (GPU)
      rot_rerank    official estimate_yaw spec correlation on 200 candidates
      trans_rerank  occupancy unpack + model forward on [query]+top-20 +
                    trans_cnn + rotate_bev_batch + solve_translation x2

Database side (untimed prep): TIRING specs / global+spec+packed-occ built once
per sequence; RING/RING++ BEVs come from the _ringpp_eval.py caches when
present (falls back to computing occupancy BEVs on the fly for `ring`).

Run inside the container with the GPU EXCLUSIVE (no training):
  python scripts/_bench_ring_latency.py            # all variants
  python scripts/_bench_ring_latency.py --variants ringpp ring
Writes results/ring_latency_bench.json and prints a table.
"""
from __future__ import annotations
import argparse
import json
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

SEQS = [("kitti", "05", "KITTI_05"), ("kitti", "00", "KITTI_00"),
        ("mulran", "KAIST03", "MulRan_KAIST03"), ("nclt", "2012-01-08", "NCLT_2012-01-08")]
CACHE_KEYS = {"kitti": "056e0a02", "nclt": "056e0a02", "mulran": "33919e6e"}
ENCODE_SAMPLE = 60   # reduced from 150 (user time-priority amendment)
QUERY_SAMPLE = 30    # reduced from 100 (user time-priority amendment)
QUERY_WARMUP = 10
DTH, SKIP = 5.0, 30
GRID = 120                                 # official RING/RING++ resolution
BOUNDS = (-70.0, 70.0, -70.0, 70.0, 1.0, 20.0)
N_COARSE, N_TRANS = 200, 20                # deployed _ringsharp_eval settings
CORR_BATCH = 256
BEV_CACHE = REPO / "_external" / "ringsharp_data" / "ringpp_cache"
WEIGHTS_DIR = RINGSHARP / "tools" / "results" / "weights" / "ring_sharp_l_nsc_full"
OUT = REPO / "results" / "ring_latency_bench.json"
DEVICE = "cuda"


def host_rss():
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM")):
            k, v = line.split(":")
            out[k] = round(int(v.strip().split()[0]) / 1024.0, 1)
    return out


def sync():
    torch.cuda.synchronize()


def gpu_peak_reset():
    torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb():
    return round(torch.cuda.max_memory_allocated() / 1e6, 1)


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


# ---------------------------------------------------------------- RING/RING++
def bench_ring_indexing(variant, dtype, seq, scan_ids):
    from glnet.datasets.nsc.nsc_dataset import make_nsc_loader
    from glnet.utils.data_utils.point_clouds import generate_feats, generate_bev
    from torch_radon import ParallelBeam

    loader = make_nsc_loader(dtype, seq, "/workspace/data")
    ids = sample_ids(scan_ids, ENCODE_SAMPLE)
    angles = torch.FloatTensor(np.linspace(0, 2 * np.pi, GRID).astype(np.float32))
    radon = ParallelBeam(GRID, angles)

    def make_bev(pts):
        if variant == "ringpp":
            return generate_feats(pts, Z=1, Y=GRID, X=GRID, bounds=BOUNDS).reshape(-1, GRID, GRID)
        return generate_bev(pts, Z=1, Y=GRID, X=GRID, bounds=BOUNDS).reshape(-1, GRID, GRID)

    # warmup (3 scans, untimed)
    for sid in ids[:3]:
        pts = np.ascontiguousarray(loader[int(sid)]["points"][:, :3], dtype=np.float32)
        bev = make_bev(pts).float().to(DEVICE)
        sync()
        s = radon.forward(bev.unsqueeze(0))
        m = torch.fft.fft2(s, dim=-1, norm="ortho")
        _ = torch.sqrt(m.real ** 2 + m.imag ** 2 + 1e-15)
        sync()

    t_io = t_bev = t_radon = t_fft = 0.0
    npts = []
    gpu_peak_reset()
    for sid in ids:
        t0 = time.perf_counter()
        pts = np.ascontiguousarray(loader[int(sid)]["points"][:, :3], dtype=np.float32)
        t1 = time.perf_counter()
        bev = make_bev(pts)
        t2 = time.perf_counter()
        bev = bev.float().to(DEVICE).unsqueeze(0)
        sync()
        t3 = time.perf_counter()
        sino = radon.forward(bev)
        sync()
        t4 = time.perf_counter()
        m = torch.fft.fft2(sino, dim=-1, norm="ortho")
        spec = torch.sqrt(m.real ** 2 + m.imag ** 2 + 1e-15)
        sync()
        t5 = time.perf_counter()
        t_io += t1 - t0
        t_bev += t2 - t1
        t_radon += t4 - t3
        t_fft += t5 - t4
        npts.append(len(pts))
    k = len(ids)
    del loader
    stage_ms = {"ms_per_kf_pointload_io": round(t_io / k * 1e3, 3),
                "ms_per_kf_bev": round(t_bev / k * 1e3, 3),
                "ms_per_kf_radon": round(t_radon / k * 1e3, 3),
                "ms_per_kf_tiring_fft": round(t_fft / k * 1e3, 3)}
    return {"n_sampled": k, "mean_points_per_scan": int(np.mean(npts)),
            **stage_ms,
            "ms_per_kf_total": round(sum(v for kk, v in stage_ms.items()
                                         if kk != "ms_per_kf_pointload_io"), 3),
            "gpu_peak_mb": gpu_peak_mb()}


def build_specs_from_cache(variant, dtype, seq, scan_ids):
    """DB specs (untimed prep). Uses the _ringpp_eval BEV cache when present."""
    from glnet.utils.data_utils.point_clouds import generate_bev
    from glnet.datasets.nsc.nsc_dataset import make_nsc_loader
    from torch_radon import ParallelBeam
    cpath = BEV_CACHE / f"{variant}_{dtype}_{seq}.npz"
    if cpath.exists():
        with np.load(cpath) as c:
            bevs = c["bevs"]
        assert len(bevs) == len(scan_ids), f"cache mismatch {cpath}"
    else:
        assert variant == "ring", f"missing BEV cache for {variant}: {cpath}"
        loader = make_nsc_loader(dtype, seq, "/workspace/data")
        bevs = np.zeros((len(scan_ids), 1, GRID, GRID), dtype=np.float16)
        for i, sid in enumerate(scan_ids):
            pts = np.ascontiguousarray(loader[int(sid)]["points"][:, :3], dtype=np.float32)
            bevs[i] = generate_bev(pts, Z=1, Y=GRID, X=GRID,
                                   bounds=BOUNDS).reshape(-1, GRID, GRID).numpy()
        del loader
    angles = torch.FloatTensor(np.linspace(0, 2 * np.pi, GRID).astype(np.float32))
    radon = ParallelBeam(GRID, angles)
    n = len(bevs)
    specs = torch.zeros(bevs.shape, dtype=torch.float16, device=DEVICE)
    for i0 in range(0, n, 64):
        x = torch.from_numpy(bevs[i0:i0 + 64].astype(np.float32)).to(DEVICE)
        s = radon.forward(x)
        m = torch.fft.fft2(s, dim=-1, norm="ortho")
        specs[i0:i0 + 64] = torch.sqrt(m.real ** 2 + m.imag ** 2 + 1e-15).half()
    sync()
    return specs   # (N, C, 120, 120) fp16 on GPU


def bench_ring_query(specs, queries):
    """Official exhaustive circular-correlation ranking vs the full DB."""
    from glnet.models.utils import estimate_yaw
    n = specs.shape[0]

    def one_query(qi):
        qspec = specs[qi].float()
        dists = torch.empty(n, device=DEVICE)
        for j0 in range(0, n, CORR_BATCH):
            c = specs[j0:j0 + CORR_BATCH].float()
            _, scores, _ = estimate_yaw(
                qspec.unsqueeze(0).expand(len(c), -1, -1, -1), c)
            dists[j0:j0 + len(c)] = 1 - scores.reshape(-1)
        order = torch.argsort(dists)
        return order

    qs = [q for q, _ in queries][:QUERY_SAMPLE + QUERY_WARMUP]
    for qi in qs[:QUERY_WARMUP]:
        one_query(qi)
    sync()
    gpu_peak_reset()
    lat = []
    for qi in qs[QUERY_WARMUP:]:
        t0 = time.perf_counter()
        one_query(qi)
        sync()
        lat.append((time.perf_counter() - t0) * 1e3)
    lat = np.array(lat)
    db_mb = specs.element_size() * specs.nelement() / 1e6
    return {"n_timed": len(lat), "corr_batch": CORR_BATCH,
            "total_ms_per_query": round(float(lat.mean()), 3),
            "p50_ms": round(float(np.percentile(lat, 50)), 3),
            "p95_ms": round(float(np.percentile(lat, 95)), 3),
            "gpu_peak_mb": gpu_peak_mb(),
            "db_specs_gpu_mb": round(db_mb, 1)}


# ---------------------------------------------------------------- RING#-L
def load_ring_sharp(weight):
    from glnet.utils.params import ModelParams
    from glnet.models.model_factory import model_factory
    from glnet.models.backbones_2d.unet import last_conv_block
    mp = ModelParams(str(RINGSHARP / "glnet/config/ring_sharp_l_nsc.txt"), "nsc", "/tmp")
    model = model_factory(mp).to(DEVICE)
    trans_cnn = last_conv_block(mp.feature_dim, mp.feature_dim, bn=False).to(DEVICE)
    ckpt = torch.load(weight, map_location=DEVICE, weights_only=False)
    state = {k.replace("module.", "", 1): v for k, v in ckpt["model"].items()}
    # e2cnn: train() drops the init-registered expanded-filter buffers so the
    # strict load matches a train-mode checkpoint; eval() re-expands them from
    # the loaded weights (see scripts/_ringsharp_eval.py load_model).
    model.train()
    model.load_state_dict(state, strict=True)
    trans_cnn.load_state_dict(ckpt["trans_cnn"], strict=True)
    model.eval()
    trans_cnn.eval()
    return model, trans_cnn


def bench_rsl_indexing(model, dtype, seq, scan_ids):
    from glnet.datasets.nsc.nsc_dataset import make_nsc_loader, NSC_BOUNDS, NSC_X, NSC_Y, NSC_Z
    from glnet.utils.data_utils.point_clouds import generate_bev
    loader = make_nsc_loader(dtype, seq, "/workspace/data")
    ids = sample_ids(scan_ids, ENCODE_SAMPLE)
    for sid in ids[:3]:   # warmup
        pts = np.ascontiguousarray(loader[int(sid)]["points"][:, :3], dtype=np.float64)
        bev = generate_bev(pts, Z=NSC_Z, Y=NSC_Y, X=NSC_X, bounds=NSC_BOUNDS)
        with torch.no_grad():
            model({"pc": bev.unsqueeze(0).to(DEVICE), "img": None})
        sync()
    t_io = t_bev = t_fwd = 0.0
    gpu_peak_reset()
    for sid in ids:
        t0 = time.perf_counter()
        pts = np.ascontiguousarray(loader[int(sid)]["points"][:, :3], dtype=np.float64)
        t1 = time.perf_counter()
        bev = generate_bev(pts, Z=NSC_Z, Y=NSC_Y, X=NSC_X, bounds=NSC_BOUNDS)
        t2 = time.perf_counter()
        x = bev.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            model({"pc": x, "img": None})
        sync()
        t3 = time.perf_counter()
        t_io += t1 - t0
        t_bev += t2 - t1
        t_fwd += t3 - t2
    k = len(ids)
    del loader
    return {"n_sampled": k,
            "ms_per_kf_pointload_io": round(t_io / k * 1e3, 3),
            "ms_per_kf_bev20": round(t_bev / k * 1e3, 3),
            "ms_per_kf_model_forward": round(t_fwd / k * 1e3, 3),
            "ms_per_kf_total": round((t_bev + t_fwd) / k * 1e3, 3),
            "gpu_peak_mb": gpu_peak_mb()}


def build_rsl_db(model, dtype, seq, scan_ids, batch=8):
    """Untimed DB prep: global desc (GPU), specs fp16 (GPU), packed occ (CPU)."""
    from glnet.datasets.nsc.nsc_dataset import make_nsc_loader, NSC_BOUNDS, NSC_X, NSC_Y, NSC_Z
    from glnet.utils.data_utils.point_clouds import generate_bev
    loader = make_nsc_loader(dtype, seq, "/workspace/data")
    n = len(scan_ids)
    glob = torch.zeros(n, 160, device=DEVICE)
    specs = None
    packed = []
    for i0 in range(0, n, batch):
        bevs = []
        for i in range(i0, min(i0 + batch, n)):
            pts = np.ascontiguousarray(loader[int(scan_ids[i])]["points"][:, :3],
                                       dtype=np.float64)
            bevs.append(generate_bev(pts, Z=NSC_Z, Y=NSC_Y, X=NSC_X, bounds=NSC_BOUNDS))
        x = torch.stack(bevs).to(DEVICE)
        with torch.no_grad():
            out = model({"pc": x, "img": None})
        if specs is None:
            specs = torch.zeros(n, *out["spec"].shape[1:], dtype=torch.float16,
                                device=DEVICE)
        glob[i0:i0 + len(bevs)] = out["global"].float()
        specs[i0:i0 + len(bevs)] = out["spec"].half()
        packed.append(np.packbits((x.cpu().numpy() > 0).reshape(len(bevs), -1), axis=1))
    del loader
    glob = torch.nn.functional.normalize(glob, dim=1)
    return glob, specs, np.concatenate(packed)


def bench_rsl_query(model, trans_cnn, glob, specs, packed, queries):
    from glnet.models.utils import estimate_yaw, rotate_bev_batch, solve_translation
    from glnet.datasets.nsc.nsc_dataset import NSC_X, NSC_Y, NSC_Z
    n = glob.shape[0]
    shape_bev = (NSC_Z, NSC_Y, NSC_X)

    def unpack(idxs):
        b = np.unpackbits(packed[idxs], axis=1)[:, :NSC_Z * NSC_Y * NSC_X]
        return torch.from_numpy(b.reshape(len(idxs), *shape_bev)).float().to(DEVICE)

    def one_query(qi):
        t0 = time.perf_counter()
        sims = glob @ glob[qi]
        sims[qi] = -2.0
        cand = torch.topk(sims, min(N_COARSE, n)).indices
        sync()
        t1 = time.perf_counter()
        qspec = specs[qi].float()
        cspec = specs[cand].float()
        with torch.no_grad():
            _, scores, angles = estimate_yaw(
                qspec.unsqueeze(0).expand(len(cand), -1, -1, -1), cspec)
        rot_order = torch.argsort(1 - scores.reshape(-1))
        sync()
        t2 = time.perf_counter()
        m = min(N_TRANS, len(cand))
        top = cand[rot_order[:m]].cpu().numpy()
        ang = (angles.reshape(-1)[rot_order[:m]] * 2 * np.pi / specs.shape[-2]).float()
        with torch.no_grad():
            occ = unpack(np.concatenate(([qi], top)))
            feat = model({"pc": occ, "img": None})["bev"]
            bt = trans_cnn(feat)
            qbt, cbt = bt[:1], bt[1:]
            qrot = rotate_bev_batch(qbt.expand(m, -1, -1, -1), ang)
            qrot_e = rotate_bev_batch(qbt.expand(m, -1, -1, -1), ang - np.pi)
            _, _, e1, _ = solve_translation(qrot, cbt)
            _, _, e2, _ = solve_translation(qrot_e, cbt)
            _ = np.minimum(e1.cpu().numpy().reshape(-1), e2.cpu().numpy().reshape(-1))
        sync()
        t3 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3

    qs = [q for q, _ in queries][:QUERY_SAMPLE + QUERY_WARMUP]
    for qi in qs[:QUERY_WARMUP]:
        one_query(qi)
    gpu_peak_reset()
    rows = []
    for qi in qs[QUERY_WARMUP:]:
        rows.append(one_query(qi))
    rows = np.array(rows)
    tot = rows.sum(axis=1)
    db_mb = (glob.element_size() * glob.nelement()
             + specs.element_size() * specs.nelement()) / 1e6
    return {"n_timed": len(rows), "n_coarse": N_COARSE, "n_trans": N_TRANS,
            "coarse_ms": round(float(rows[:, 0].mean()), 3),
            "rot_rerank_ms": round(float(rows[:, 1].mean()), 3),
            "trans_rerank_ms": round(float(rows[:, 2].mean()), 3),
            "total_ms_per_query": round(float(tot.mean()), 3),
            "p50_ms": round(float(np.percentile(tot, 50)), 3),
            "p95_ms": round(float(np.percentile(tot, 95)), 3),
            "gpu_peak_mb": gpu_peak_mb(),
            "db_gpu_mb": round(db_mb, 1),
            "db_packed_occ_host_mb": round(packed.nbytes / 1e6, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", nargs="+",
                    default=["ringpp", "ring", "ring_sharp_l"],
                    choices=["ringpp", "ring", "ring_sharp_l"])
    ap.add_argument("--weight", default="",
                    help="RING#-L checkpoint (default: newest in weights dir)")
    args = ap.parse_args()

    results = {"env": {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "grid": GRID, "bounds": BOUNDS,
        "encode_sample_per_seq": ENCODE_SAMPLE,
        "query_sample": QUERY_SAMPLE, "query_warmup": QUERY_WARMUP,
        "sample_note": "reduced samples (encode 60, query 30) per user time-priority; NSD bench used 150/all-queries",
        "distance_threshold_m": DTH, "skip_frames": SKIP,
        "n_coarse": N_COARSE, "n_trans": N_TRANS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, "variants": {}}

    rsl = None
    if "ring_sharp_l" in args.variants:
        weight = args.weight or str(sorted(WEIGHTS_DIR.glob("model_*.pth"),
                                           key=lambda p: p.stat().st_mtime)[-1])
        results["env"]["ring_sharp_l_weight"] = weight
        rsl = load_ring_sharp(weight)

    for variant in args.variants:
        vres = {}
        for dtype, seq, name in SEQS:
            scan_ids, poses = load_cache_meta(dtype, seq)
            queries = find_queries(poses)
            print(f"[{variant}][{name}] N={len(scan_ids)} queries={len(queries)}",
                  flush=True)
            entry = {"N": int(len(scan_ids)), "n_queries": int(len(queries))}
            if variant in ("ringpp", "ring"):
                entry["indexing"] = bench_ring_indexing(variant, dtype, seq, scan_ids)
                specs = build_specs_from_cache(variant, dtype, seq, scan_ids)
                entry["query"] = bench_ring_query(specs, queries)
                del specs
            else:
                model, trans_cnn = rsl
                entry["indexing"] = bench_rsl_indexing(model, dtype, seq, scan_ids)
                glob, specs, packed = build_rsl_db(model, dtype, seq, scan_ids)
                entry["query"] = bench_rsl_query(model, trans_cnn, glob, specs,
                                                 packed, queries)
                del glob, specs, packed
            torch.cuda.empty_cache()
            entry["host_rss_mb"] = host_rss()
            vres[name] = entry
            OUT.write_text(json.dumps(results | {"variants": {**results["variants"],
                                                              variant: vres}}, indent=1))
        results["variants"][variant] = vres

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT}\n")
    print(f"{'variant':<14}{'map':<18}{'N':>6}{'idx ms/kf':>11}{'query ms':>10}"
          f"{'p95 ms':>9}{'q peak MB':>11}")
    for variant, vres in results["variants"].items():
        for name, e in vres.items():
            print(f"{variant:<14}{name:<18}{e['N']:>6}"
                  f"{e['indexing']['ms_per_kf_total']:>11}"
                  f"{e['query']['total_ms_per_query']:>10}"
                  f"{e['query']['p95_ms']:>9}"
                  f"{e['query']['gpu_peak_mb']:>11}")


if __name__ == "__main__":
    main()
