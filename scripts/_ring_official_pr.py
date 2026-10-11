#!/usr/bin/env python3
"""Official RING / RING++ on split v2: past-only all-frames PR ledgers.

Pipeline (scripts/_ring_official_lib.py docstring has the provenance):
  bev   one test sequence -> per-keyframe official BEVs
        (read_pc -> normalize_pc with the DATASET_TABLE row -> RING repo
        generate_bev_occ (RING, 1x120x120) / generate_bev_feats (RING++,
        6x120x120)), 1000-keyframe shards under
        _external/ring_official_cache/{cond}/{sensor}_{seq}/
  dump  one test sequence -> TIRING per keyframe (RING repo generate_RING
        with the paper angle grid, _ring_official_lib.generate_ring_paper:
        120 angles over [0, 2pi) instead of utils/core.py:153's 0..2pi
        inclusive) and the causal ledger: for every query t >= 30 the
        normalized circular correlation (RING repo batch_circorr, dist =
        1 - score) against EVERY keyframe j <= t - 30; top-1 = argmin dist.
        Scoring uses the exact per-frame hoist of batch_circorr
        (scripts/_causal_ring_pr.precompute_fft_bank / correlate_prefix_fast:
        InstanceNorm + FFT done once per frame instead of once per pair) and
        re-checks it against the official batch_circorr on three queries of
        every sequence (asserted, recorded in meta); --official-scorer runs
        batch_circorr for every query instead (slow).
        Output results/pdv2_ring_official/{cond}/dump_{ring,ringpp}_{s}_{q}.npz,
        schema "pr-dump-v1" (same arrays as scripts/_causal_ring_pr.py):
        query_idx / j_star / score_corr (= corr_max - 1 = -dist) /
        top1_geo_dist / min_past_gt_dist / n_candidates + meta.
  aggregate  PR metrics per sequence and city / dataset groups for every
        method ledger present in the condition folder (ring, ringpp,
        ringsharp from scripts/_ringsharp_official.py).
  check      CPU-only checks (preprocessing == official class on raw files,
        fast scorer == batch_circorr, ledger schema, CPU timing).
  gpu-check  GPU smoke: official BEVs on a few scans vs the CPU transcription
        and the fast scorer vs batch_circorr on real descriptors.

Conditions: odom = KISS-ICP keyframes (data/preprocessed_v2_odom; poses_gt
only for scoring), gt = ground-truth keyframes (data/preprocessed_v2).
Roster: the 46 split-v2 test sequences (scripts/_pdv2_pilot.TEST).
One sequence per process (memory); the shell runner loops.

Run inside docker ring_baseline from the repo root, with `ulimit -s 81920`
(RING README: the voxelocc / voxelfeat kernels keep per-point arrays on the
stack):
  python3 scripts/_ring_official_pr.py bev  --cond odom --seq kitti/00
  python3 scripts/_ring_official_pr.py dump --cond odom --seq kitti/00
  python3 scripts/_ring_official_pr.py aggregate --cond odom
  CUDA_VISIBLE_DEVICES= python3 scripts/_ring_official_pr.py check
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import _ring_official_lib as rol  # noqa: E402  (before anything importing 'utils')
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402  (torch-free)
from _pdv2_pilot import CITY, DATASET, TEST  # noqa: E402  (torch-free lists)

VARIANTS = ("ring", "ringpp")
SHARD = 1000
MIN_FEAT_POINTS = 30      # k of build_neighbors_NN inside generate_bev_feats
FAST_CHUNK = 1024
N_XCHECK = 3              # queries per sequence re-scored with batch_circorr
DEVICE = "cuda"           # the RING repo kernels (voxelocc, voxelfeat, torch_radon) need CUDA
SCORE_COLS = {"ring": ["score_corr"], "ringpp": ["score_corr"],
              "ringsharp": ["score_trans", "score_rot", "score_rotbranch"]}


def _split(seq):
    s, q = seq.split("/")
    assert (s, q) in TEST, f"{seq} is not a split-v2 test sequence"
    return s, q


def bev_dir(cond, sensor, seq):
    return rol.CACHE_ROOT / cond / f"{sensor}_{seq}"


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ------------------------------------------------------------------ bev stage
def cmd_bev(cond, seq):
    """Both variants' BEVs in one pass (shared cache)."""
    import torch
    sensor, q = _split(seq)
    assert DEVICE != "cuda" or torch.cuda.is_available(), "voxelocc / voxelfeat are CUDA kernels"
    core = rol.import_ring_core()
    cache = rol.split_cache(cond, "test", sensor, q)
    scan_ids, _ = rol.load_keyframes(cache)
    d = bev_dir(cond, sensor, q)
    done = d / "done.json"
    if done.exists() and json.loads(done.read_text())["scan_ids_sha256"] == _sha(scan_ids):
        print(f"SKIP {cond} {seq}: BEV cache complete", flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    loader = rol.make_loader(sensor, q)
    n = len(scan_ids)
    t_all, n_built = 0.0, 0
    for c0 in range(0, n, SHARD):
        c1 = min(c0 + SHARD, n)
        shard = d / f"shard_{c0:06d}.npz"
        if shard.exists():
            with np.load(shard) as z:
                if np.array_equal(z["scan_ids"], scan_ids[c0:c1]):
                    continue
        m = c1 - c0
        occ = np.zeros((m, 1, 120, 120), np.uint8)
        feat = np.zeros((m, 6, 120, 120), np.float32)
        n_pts = np.zeros(m, np.int64)
        t0 = time.time()
        for k in range(m):
            pts = loader[int(scan_ids[c0 + k])]["points"]
            _, pc_norm = rol.official_points(sensor, pts)
            n_pts[k] = len(pc_norm)
            with torch.no_grad():
                if len(pc_norm) > 0:
                    occ[k] = core.generate_bev_occ(pc_norm).cpu().numpy()
                if len(pc_norm) >= MIN_FEAT_POINTS:
                    feat[k] = core.generate_bev_feats(pc_norm).cpu().numpy()
        tmp = d / f".tmp_shard_{c0:06d}.npz"
        np.savez_compressed(tmp, scan_ids=scan_ids[c0:c1], bev_ring=occ,
                            bev_ringpp=feat, n_points=n_pts)
        os.replace(tmp, shard)
        dt = time.time() - t0
        t_all += dt
        n_built += m
        print(f"  [{cond} {seq}] BEV {c1}/{n} ({dt / m:.3f} s/kf, ETA "
              f"{(n - c1) * t_all / n_built / 60:.0f} min)", flush=True)
    n_pts = np.concatenate([np.load(d / f"shard_{c0:06d}.npz")["n_points"]
                            for c0 in range(0, n, SHARD)])
    done.write_text(json.dumps({
        "sensor": sensor, "sequence": q, "cond": cond, "cache": str(cache),
        "n_frames": n, "scan_ids_sha256": _sha(scan_ids),
        "table_row": rol.DATASET_TABLE[sensor], "ring_commit": rol.RING_COMMIT,
        "n_frames_no_points": int((n_pts == 0).sum()),
        "n_frames_lt30_points_zero_feat_bev": int((n_pts < MIN_FEAT_POINTS).sum()),
        "points_per_frame_median": float(np.median(n_pts)),
        "bev_s_per_kf_this_run": round(t_all / n_built, 4) if n_built else None,
    }, indent=2))
    print(f"BEV_DONE {cond} {seq}: {n} keyframes", flush=True)


def bev_shards(cond, sensor, seq, variant, scan_ids):
    """(info, generator of per-shard BEV arrays) — shard by shard, so the
    full float32 RING++ BEV stack is never held in memory."""
    d = bev_dir(cond, sensor, seq)
    info = json.loads((d / "done.json").read_text())
    assert info["scan_ids_sha256"] == _sha(scan_ids), "BEV cache / keyframe mismatch"

    def gen():
        for c0 in range(0, len(scan_ids), SHARD):
            with np.load(d / f"shard_{c0:06d}.npz") as z:
                yield z[f"bev_{variant}"]
    return info, gen()


# ----------------------------------------------------------------- dump stage
def tiring_bank(core, shards, n, c):
    """generate_RING per keyframe with the paper angle grid
    (_ring_official_lib.generate_ring_paper) -> (n, C, 120, 120) float32 CPU."""
    import torch
    bank = torch.empty((n, c, 120, 120), dtype=torch.float32)
    i = 0
    with torch.no_grad():
        for bevs in shards:
            for b in bevs:
                x = torch.from_numpy(b.astype(np.float32)).to(DEVICE)
                _, ti = rol.generate_ring_paper(x)
                bank[i] = ti.reshape(c, 120, 120).float().cpu()
                i += 1
    assert i == n, (i, n)
    return bank


def official_scores(core, bank, t, n_prefix):
    """batch_circorr (RING repo, batches of cfg.batch_size as evaluate.py)
    of query t against prefix rows [0, n_prefix); returns 1 - dist."""
    import torch
    bs = rol.import_ring()["cfg"].batch_size
    out = np.empty(n_prefix, dtype=np.float64)
    q = bank[t].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        for j0 in range(0, n_prefix, bs):
            j1 = min(j0 + bs, n_prefix)
            dist, _ = core.batch_circorr(q.repeat(j1 - j0, 1, 1, 1),
                                         bank[j0:j1].to(DEVICE))
            out[j0:j1] = 1.0 - np.asarray(dist, dtype=np.float64).reshape(-1)
    return out


def cmd_dump(cond, seq, variants, official_scorer, overwrite):
    import torch
    from _causal_ring_pr import (correlate_prefix_fast, dump_variant,
                                 precompute_fft_bank)
    sensor, q = _split(seq)
    assert DEVICE != "cuda" or torch.cuda.is_available(), "torch_radon needs CUDA"
    core = rol.import_ring_core()
    cache = rol.split_cache(cond, "test", sensor, q)
    scan_ids, poses = rol.load_keyframes(cache)
    positions = poses[:, :3, 3]
    min_past = rol.min_past_dists(positions, MIN_INDEX_GAP)
    out_dir = rol.OUT_ROOT / cond
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(scan_ids)
    for v in variants:
        out = out_dir / f"dump_{v}_{sensor}_{q}.npz"
        if out.exists() and not overwrite:
            print(f"SKIP {out.name}", flush=True)
            continue
        t0 = time.time()
        info, shards = bev_shards(cond, sensor, q, v, scan_ids)
        bank = tiring_bank(core, shards, n, 1 if v == "ring" else 6)
        spec_s = time.time() - t0
        fft_bank, on_gpu = precompute_fft_bank(bank, DEVICE)

        def fast(t, n_prefix):
            return correlate_prefix_fast(fft_bank, on_gpu, t, n_prefix,
                                         FAST_CHUNK, DEVICE)

        # exactness re-check of the hoisted scorer on real descriptors
        xcheck = []
        for t in sorted({n // 4, n // 2, n - 1})[:N_XCHECK]:
            if t < MIN_INDEX_GAP:
                continue
            npf = t - MIN_INDEX_GAP + 1
            s_off, s_fast = official_scores(core, bank, t, npf), fast(t, npf)
            diff = float(np.max(np.abs(s_off - s_fast)))
            same = int(np.argmax(s_off)) == int(np.argmax(s_fast))
            gap = float(np.sort(s_off)[-1] - np.sort(s_off)[-2]) if npf > 1 else 0.0
            xcheck.append({"t": int(t), "n_prefix": npf, "max_abs_diff": diff,
                           "same_argmax": same, "official_top_gap": gap})
            assert diff < 1e-4 and (same or gap < 1e-5), xcheck[-1]
        scorer = ((lambda t, npf: official_scores(core, bank, t, npf))
                  if official_scorer else fast)
        t1 = time.time()
        arrays, n_corr = dump_variant(None, positions, min_past, 0, DEVICE, None,
                                      tag=f"[{cond} {seq}] {v}", scorer=scorer,
                                      n_frames=n)
        wall = time.time() - t1
        del fft_bank, bank
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        nq = len(arrays["query_idx"])
        meta = {
            "schema": "pr-dump-v1", "method": v, "sensor": sensor,
            "sequence": q, "condition": cond, "min_index_gap": MIN_INDEX_GAP,
            "n_frames": n, "n_queries": nq,
            "pipeline": ("official RING repo (lus6-Jenny/RING @ "
                         f"{rol.RING_COMMIT}): read_pc -> normalize_pc -> "
                         + ("generate_bev_occ" if v == "ring" else "generate_bev_feats")
                         + " -> generate_RING with the paper angle grid (120 angles "
                           "over [0, 2pi), 3 deg; overrides utils/core.py:153 "
                           "linspace(0, 2pi, 120)) -> batch_circorr (dist = 1 - "
                           "score), exhaustive over the prefix j <= t - 30"),
            "dataset_table_row": rol.DATASET_TABLE[sensor],
            "scorer": ("official batch_circorr" if official_scorer else
                       "exact per-frame hoist of batch_circorr "
                       "(_causal_ring_pr.correlate_prefix_fast)"),
            "scorer_crosscheck_vs_batch_circorr": xcheck,
            "score_convention": ("similarity, higher = accept; score_corr = "
                                 "corr_max - 1 = -(official dist) of the "
                                 "prefix argmin(dist)"),
            "poses_for_scoring": "poses_gt" if cond == "odom" else "poses (GT)",
            "budget": {"candidates": "exhaustive over prefix j <= t - 30",
                       "n_correlations": int(n_corr), "wall_s": round(wall, 1),
                       "ms_per_query": round(1e3 * wall / max(nq, 1), 1),
                       "tiring_s": round(spec_s, 1)},
            "bev_cache": info,
            "artifacts": {"keyframe_cache": str(cache)},
        }
        np.savez_compressed(out, meta=json.dumps(meta), **arrays)
        print(f"DUMP_DONE {out.name}: {nq} queries, {wall:.0f} s "
              f"(+{spec_s:.0f} s TIRING), xcheck {xcheck}", flush=True)


# ------------------------------------------------------------------ aggregate
def cmd_aggregate(cond):
    from _pr_eval_lib import AGG_METRICS, MATCH_RADIUS_M, classify_queries, pr_sweep
    out_dir = rol.OUT_ROOT / cond
    summary = {"_meta": {"condition": cond, "n_test_sequences": len(TEST),
                         "score_columns": SCORE_COLS,
                         "groups": "sequence mean within a group, groups "
                                   "weighted equally (as pr_summary_baselines)"}}
    for m, cols in SCORE_COLS.items():
        missing = [f"{s}/{q}" for s, q in TEST
                   if not (out_dir / f"dump_{m}_{s}_{q}.npz").exists()]
        if missing:
            print(f"-- {cond} {m}: {len(missing)} ledgers missing, skipped", flush=True)
            continue
        for col in cols:
            dist_col = "top1_geo_dist_rot" if col == "score_rotbranch" else "top1_geo_dist"
            per = {}
            for s, q in TEST:
                z = np.load(out_dir / f"dump_{m}_{s}_{q}.npz")
                qc = classify_queries(z["min_past_gt_dist"])
                sw = pr_sweep(z[col], z[dist_col], qc)
                d = np.asarray(z[dist_col], dtype=np.float64)[qc["positive"]]
                per[f"{s}/{q}"] = {**{k: float(sw[k]) for k in AGG_METRICS},
                                   "R@1_revisit": float((np.isfinite(d) & (d <= MATCH_RADIUS_M)).mean()),
                                   "counts": sw["counts"]}
            metrics = list(AGG_METRICS) + ["R@1_revisit"]

            def group_means(G):
                return {g: {k: statistics.mean(per[x][k] for x in ks) for k in metrics}
                        for g, ks in G.items()}

            city, dset = group_means(CITY), group_means(DATASET)
            key = m if len(cols) == 1 else f"{m}:{col}"
            summary[key] = {
                "city_groups": city,
                "city_macro6": {k: statistics.mean(v[k] for v in city.values()) for k in metrics},
                "city_macro5_without_nclt": {
                    k: statistics.mean(v[k] for c, v in city.items() if c != "ann_arbor")
                    for k in metrics},
                "dataset_macro6": {k: statistics.mean(v[k] for v in dset.values()) for k in metrics},
                "per_sequence": per}
            c6 = summary[key]["city_macro6"]
            print(f"== {cond} {key}: city6 AP {c6['AP']:.4f} maxF1 {c6['max_F1']:.4f} "
                  f"R@100P {c6['R@100P']:.4f} R@1rev {c6['R@1_revisit']:.4f}", flush=True)
    p = out_dir / "pr_summary_ring_official.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2))
    print(f"wrote {p}", flush=True)


# ----------------------------------------------------------------- CPU checks
CHECK_SEQS = [("kitti", "00"), ("mulran", "DCC01"), ("nclt", "2012-01-08"),
              ("kitti360", "0000"), ("helipr", "Town01"), ("mcd", "kth_day_06")]
OLD_BAND = (1.0, 20.0)     # previous pipeline: z in [1, 20] m for every sensor


def _check_preprocessing(rep):
    """Our loader points through the official class == the official class
    reading the raw file (KITTI / MulRan / NCLT); table rows for the rest."""
    rows = []
    for sensor, q in CHECK_SEQS:
        scan_ids, _ = rol.load_keyframes(rol.split_cache("odom", "test", sensor, q))
        loader = rol.make_loader(sensor, q)
        for i in (0, len(scan_ids) // 2, len(scan_ids) - 1):
            sid = int(scan_ids[i])
            t0 = time.time()
            pts = np.asarray(loader[sid]["points"])
            t_read = time.time() - t0
            t0 = time.time()
            pc, pcn = rol.official_points(sensor, pts)
            t_off = time.time() - t0
            r = {"sensor": sensor, "seq": q, "scan_id": sid, "raw_points": int(len(pts)),
                 "after_read_pc": int(len(pc)), "after_normalize": int(len(pcn)),
                 "z_norm_range": [float(pcn[:, 2].min()), float(pcn[:, 2].max())] if len(pcn) else None,
                 "t_loader_s": round(t_read, 3), "t_official_points_s": round(t_off, 3)}
            occ_new = rol.occupancy_numpy(pcn)
            r["occ_cells"] = int(occ_new.sum())
            # previous pipeline band for comparison (same 120x120 +-70 m grid)
            xyz = pts[:, :3].astype(np.float32)
            m = ((np.abs(xyz[:, 0]) < 70) & (np.abs(xyz[:, 1]) < 70)
                 & (xyz[:, 2] > OLD_BAND[0]) & (xyz[:, 2] < OLD_BAND[1]))
            old = xyz[m] / np.float32(70.0)
            r["occ_cells_old_band_1_20"] = int(rol.occupancy_numpy(old).sum())
            if sensor in ("kitti", "mulran", "nclt"):
                pc_f, pcn_f = rol.official_points_from_file(
                    sensor, rol.raw_scan_path(loader, sensor, sid))
                r["equal_shape_to_official_file_read"] = pcn_f.shape == pcn.shape
                r["bit_equal_to_official_file_read"] = bool(
                    pcn_f.shape == pcn.shape and np.array_equal(pcn_f, pcn))
                r["max_abs_diff_normalized"] = (float(np.abs(pcn_f - pcn).max())
                                                if pcn_f.shape == pcn.shape else None)
                r["occ_cells_differing_vs_file_read"] = int(
                    (rol.occupancy_numpy(pcn_f) != occ_new).sum())
            rows.append(r)
            print("  prep", json.dumps(r), flush=True)
        del loader
    rep["preprocessing"] = rows
    file_rows = [r for r in rows if "bit_equal_to_official_file_read" in r]
    assert all(r["equal_shape_to_official_file_read"] for r in file_rows)
    assert all(r["occ_cells_differing_vs_file_read"] == 0 for r in file_rows)


def _check_scorer(rep):
    """Hoisted scorer == RING repo batch_circorr (CPU, synthetic TIRINGs)."""
    import torch
    core = rol.import_ring_core()
    from _causal_ring_pr import correlate_prefix_fast, precompute_fft_bank
    rng = np.random.default_rng(0)
    out = {}
    for c in (1, 6):
        bank = torch.from_numpy(np.abs(rng.standard_normal((70, c, 120, 120))).astype(np.float32))
        bank[60] = torch.roll(bank[7], shifts=17, dims=-2)  # rotated revisit
        fb, on_gpu = precompute_fft_bank(bank, "cpu")
        worst, same = 0.0, True
        for t in (40, 60, 69):
            npf = t - MIN_INDEX_GAP + 1
            s_fast = correlate_prefix_fast(fb, on_gpu, t, npf, 16, "cpu")
            dist, _ = core.batch_circorr(bank[t].unsqueeze(0).repeat(npf, 1, 1, 1), bank[:npf])
            s_off = 1.0 - np.asarray(dist, dtype=np.float64).reshape(-1)
            worst = max(worst, float(np.abs(s_off - s_fast).max()))
            same &= int(np.argmax(s_off)) == int(np.argmax(s_fast))
        out[f"C{c}"] = {"max_abs_diff": worst, "same_argmax": bool(same)}
        assert worst < 1e-5 and same, out
    print("  scorer", out, flush=True)
    rep["scorer_hoist_vs_batch_circorr_cpu"] = out


def _check_ledger_schema(rep):
    """Synthetic ledger from the shared loop vs an existing pr-dump-v1 file."""
    import torch
    from _causal_ring_pr import _toy_yaw, dump_variant
    rng = np.random.default_rng(1)
    specs = torch.from_numpy(rng.standard_normal((50, 1, 8, 6)).astype(np.float32))
    pos = rng.standard_normal((50, 3)) * 50
    mp = rol.min_past_dists(pos, MIN_INDEX_GAP)
    arrays, _ = dump_variant(specs, pos, mp, 8, "cpu", _toy_yaw, tag="[schema]")
    ref = REPO / "results/pd2026_pr/dump_ring_kitti_05.npz"
    z = np.load(ref)
    mine = {k: str(v.dtype) for k, v in arrays.items()}
    theirs = {k: str(z[k].dtype) for k in z.files if k != "meta"}
    rep["ledger_schema"] = {"reference": str(ref.relative_to(REPO)),
                            "same_arrays_and_dtypes": mine == theirs,
                            "mine": mine, "reference_arrays": theirs}
    print("  ledger", rep["ledger_schema"], flush=True)
    assert mine == theirs


def _check_timing(rep):
    """CPU part of generate_bev_feats (build_neighbors_NN, k = 30) per sensor."""
    rol.import_ring()
    from utils.extract_local_descriptor import build_neighbors_NN  # RING repo
    tim = {}
    for sensor, q in CHECK_SEQS:
        scan_ids, _ = rol.load_keyframes(rol.split_cache("odom", "test", sensor, q))
        loader = rol.make_loader(sensor, q)
        ts = []
        for i in (len(scan_ids) // 3, 2 * len(scan_ids) // 3):
            _, pcn = rol.official_points(sensor, loader[int(scan_ids[i])]["points"])
            t0 = time.time()
            build_neighbors_NN(pcn, 30)
            ts.append(time.time() - t0)
        tim[sensor] = {"knn_eig_s_per_kf": round(float(np.mean(ts)), 3),
                       "threads": os.cpu_count()}
        del loader
    print("  timing", tim, flush=True)
    rep["cpu_timing_ringpp_features"] = tim


def _check_angles(rep):
    """Paper angle grid vs the code line it overrides (no radon on CPU)."""
    paper = np.degrees(rol.ring_angles_paper(120).astype(np.float64))
    code = np.degrees(np.linspace(0, 2 * np.pi, 120).astype(np.float32).astype(np.float64))
    out = {"paper_step_deg": float(np.diff(paper).mean()), "paper_last_deg": float(paper[-1]),
           "code_core_py_153_step_deg": float(np.diff(code).mean()),
           "code_core_py_153_last_deg": float(code[-1])}
    assert abs(out["paper_step_deg"] - 3.0) < 1e-4 and abs(out["paper_last_deg"] - 357.0) < 1e-3
    print("  angles", out, flush=True)
    rep["angle_grid"] = out


def cmd_check():
    rep = {"_meta": {"date": time.strftime("%Y-%m-%d %H:%M"),
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "dataset_table": rol.DATASET_TABLE}}
    for f in (_check_angles, _check_preprocessing, _check_scorer, _check_ledger_schema,
              _check_timing):
        print(f"[check] {f.__name__}", flush=True)
        f(rep)
    p = rol.OUT_ROOT / "check_ring_official_cpu.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep, indent=2, default=str))
    print(f"CHECK_OK -> {p}", flush=True)


def cmd_gpu_check():
    """GPU smoke (seconds): official BEVs on two scans per sensor vs the CPU
    transcription, and generate_bev_feats shape / finiteness."""
    import torch
    core = rol.import_ring_core()
    res = []
    for sensor, q in CHECK_SEQS:
        scan_ids, _ = rol.load_keyframes(rol.split_cache("odom", "test", sensor, q))
        loader = rol.make_loader(sensor, q)
        for i in (0, len(scan_ids) // 2):
            _, pcn = rol.official_points(sensor, loader[int(scan_ids[i])]["points"])
            with torch.no_grad():
                occ = core.generate_bev_occ(pcn).cpu().numpy()[0]
                feat = core.generate_bev_feats(pcn).cpu().numpy()
                _, ti = rol.generate_ring_paper(torch.from_numpy(occ[None].astype(np.float32)).cuda())
            r = {"sensor": sensor, "seq": q, "occ_equal_cpu_transcription":
                 bool(np.array_equal(occ.astype(np.uint8), rol.occupancy_numpy(pcn))),
                 "feat_shape": list(feat.shape), "feat_finite": bool(np.isfinite(feat).all()),
                 "tiring_shape": list(ti.shape)}
            res.append(r)
            print(" ", r, flush=True)
    assert all(r["occ_equal_cpu_transcription"] for r in res)
    print("GPU_CHECK_OK", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bev", "dump"):
        p = sub.add_parser(name)
        p.add_argument("--cond", choices=("odom", "gt"), default="odom")
        p.add_argument("--seq", required=True)
        if name == "dump":
            p.add_argument("--variant", choices=("both",) + VARIANTS, default="both")
            p.add_argument("--official-scorer", action="store_true",
                           help="batch_circorr for every query (slow)")
            p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("aggregate")
    p.add_argument("--cond", choices=("odom", "gt"), default="odom")
    sub.add_parser("check")
    sub.add_parser("gpu-check")
    sub.add_parser("roster")
    a = ap.parse_args()
    variants = VARIANTS if getattr(a, "variant", "both") == "both" else (a.variant,)
    if a.cmd == "bev":
        cmd_bev(a.cond, a.seq)
    elif a.cmd == "dump":
        cmd_dump(a.cond, a.seq, variants, a.official_scorer, a.overwrite)
    elif a.cmd == "aggregate":
        cmd_aggregate(a.cond)
    elif a.cmd == "check":
        cmd_check()
    elif a.cmd == "gpu-check":
        cmd_gpu_check()
    elif a.cmd == "roster":
        print(" ".join(f"{s}/{q}" for s, q in TEST))


if __name__ == "__main__":
    main()
