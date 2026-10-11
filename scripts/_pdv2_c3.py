#!/usr/bin/env python3
"""C3 run on split v2 (2026-10-07): past-only NSD / NSD-H whose keyframes and
graph inputs come from LiDAR odometry instead of dataset ground-truth poses.

What changes against the causal protocol (docs/평가/인과 평가.md):
  * poses the SYSTEM sees -- keyframe selection (0.8 m / 20 deg / 30 s) and the
    temporal-edge distance/rotation inputs of the trajectory graph -- come
    from KISS-ICP odometry (deskew off, 1-100 m) run over every scan;
  * ground-truth poses are used ONLY to score (5 m correctness, query
    classes), stored next to them as poses_gt (the patched readers in
    scripts/_causal_pr_dump.py / _causal_val_select.py prefer poses_gt);
  * training (scripts/_pdv2_run-style train_multi_dataset run on
    configs/training_place_disjoint_v2_causal.yaml) and the causal edge
    classifier use the GT train caches -- supervision, not inference.
Selection (fusion tuple, confidence statistic, tau*) is redone on the v2 val
set with the split-v2 rule: sensor families (kitti incl. kitti360, helipr,
mulran) weighted equally. Two conditions share every step after the caches:
  odom  data/preprocessed_v2_odom (odometry keyframes)   -> results/pdv2_c3/odom
  gt    data/preprocessed_v2      (GT keyframes, pilot)  -> results/pdv2_c3/gt
The pd/causal scripts are imported unchanged apart from the poses_gt reader;
this driver only swaps lists, paths and the variant entry.

Subcommands (one heavy sequence per process):
  odometry --seq S              KISS-ICP over all scans of a val/test sequence
  odom-report                   drift of odometry vs GT -> odometry_report.json
  caches --role val|test --seq S  odometry-keyframe caches (+ poses_gt)
  clf [--only S | --train-only] causal edge classifier on v2 train (GT)
  val-tuples --cond C --tag T   tuple grid on one val sequence
  select-tuple --cond C         family-equal tuple selection
  val-conf --cond C --tag T     all-frames val ledger with the chosen tuple
  conf-select --cond C          confidence statistic + tau* (family-equal)
  test-dump --cond C --seq S    all-frames test ledger
  aggregate --cond C            per-sequence PR metrics + groups
"""
from __future__ import annotations
import argparse
import ast
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _pdv2_pilot import (  # noqa: E402  (lists and groups of split v2)
    CITY, DATASET, FAMILY, SEED, TEST, TRAIN, V2, VAL, VAL_TAGS, _keys,
)

ODOM_CACHE = REPO / "data/preprocessed_v2_odom"
ODOM_DIR = REPO / "results/pdv2_c3/odometry"
OUT_ROOT = REPO / "results/pdv2_c3"
CONFIG = REPO / "configs/training_place_disjoint_v2_causal.yaml"
RVEC = REPO / "artifacts/key_remetrize_r_pdv2.npy"
CLS = REPO / "artifacts/edge_classifier_pdv2_causal.pt"
VARIANT = "pdv2_causal"
OPS = ("pbev128_main", "hspec12")
CACHE_OF = {"odom": ODOM_CACHE, "gt": V2}
EDGE_HOLDOUT = {"kitti/06", "helipr/Bridge01"}
TIE_BREAK_DELTA = 0.005


def _split(seq):
    s, q = seq.split("/")
    return s, q


def _family(sensor):
    return FAMILY.get(sensor, sensor)


# ---------------------------------------------------------------- odometry
def cmd_odometry(seq):
    from kiss_icp.config import load_config
    from kiss_icp.kiss_icp import KissICP
    from _pd_build_test_caches import _make_loader
    sensor, q = _split(seq)
    assert (sensor, q) in VAL + TEST, seq
    out = ODOM_DIR / f"{sensor}_{q}.npz"
    if out.exists():
        print(f"SKIP {out.name}", flush=True)
        return
    cfg = load_config(None)
    cfg.data.deskew = False          # loaders give no per-point times
    cfg.data.min_range = 1.0         # drop the platform / operator returns
    cfg.data.max_range = 100.0
    cfg.registration.max_num_threads = 2
    odom = KissICP(cfg)
    loader = _make_loader(sensor, q)
    n = len(loader)
    poses = np.empty((n, 4, 4))
    gt = np.empty((n, 4, 4))
    stamps = np.empty(n)
    t0 = time.time()
    no_ts = np.empty(0)
    for i in range(n):
        item = loader[i]
        pts = np.asarray(item["points"])[:, :3].astype(np.float64)
        odom.register_frame(pts, no_ts)
        poses[i] = odom.last_pose
        gt[i] = item["pose"]
        stamps[i] = float(item["timestamp"])
        if i % 2000 == 0:
            print(f"  [{seq}] scan {i}/{n} ({(i + 1) / (time.time() - t0):.1f} Hz)",
                  flush=True)
    ODOM_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, odom_poses=poses, gt_poses=gt, timestamps=stamps,
                        meta=json.dumps({"sensor": sensor, "sequence": q,
                                         "n_scans": n,
                                         "runtime_s": time.time() - t0,
                                         "kiss_icp": "1.3.0",
                                         "config": cfg.model_dump()}))
    print(f"OK {out.name}: {n} scans in {time.time() - t0:.0f} s", flush=True)


def _rigid_rmse(a, b):
    """RMSE of positions a (odometry, lidar frame) after the best rigid
    alignment onto b (GT world frame; Umeyama without scale). GT and
    odometry use different body frames (e.g. KITTI GT is the camera), so
    only frame-invariant comparisons are made; the lever arm is < 0.5 m."""
    ma, mb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ma).T @ (b - mb))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    r = vt.T @ d @ u.T
    return float(np.sqrt(np.mean(np.sum((a @ r.T + (mb - r @ ma) - b) ** 2, 1))))


def _rot_angle(rel):
    c = np.clip((np.trace(rel[:3, :3], axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def _odometry_errors(odom, gt, seg_m=100.0, stride=50):
    po, pg = odom[:, :3, 3], gt[:, :3, 3]
    dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pg, axis=0), axis=1))]
    ate = _rigid_rmse(po[::5], pg[::5])
    win = []
    for i in range(0, len(gt), stride):
        j = int(np.searchsorted(dist, dist[i] + seg_m))
        if j >= len(gt):
            break
        win.append(_rigid_rmse(po[i:j + 1], pg[i:j + 1]))
    # scan-to-scan step: translation length and rotation angle are invariant
    # to the body frame up to the lever arm -- these are what keyframe
    # selection (0.8 m / 20 deg) and the temporal-edge inputs consume.
    rel_o = np.einsum("nij,njk->nik", np.linalg.inv(odom[:-1]), odom[1:])
    rel_g = np.einsum("nij,njk->nik", np.linalg.inv(gt[:-1]), gt[1:])
    dt = np.abs(np.linalg.norm(rel_o[:, :3, 3], axis=1)
                - np.linalg.norm(rel_g[:, :3, 3], axis=1))
    dr = np.abs(_rot_angle(rel_o) - _rot_angle(rel_g))
    return {"length_m": float(dist[-1]), "ate_rmse_m": ate,
            "ate_pct_of_length": 100.0 * ate / max(dist[-1], 1e-9),
            "win100_rmse_m_mean": float(np.mean(win)) if win else None,
            "win100_rmse_m_p95": float(np.percentile(win, 95)) if win else None,
            "step_dt_m_median": float(np.median(dt)),
            "step_dt_m_p99": float(np.percentile(dt, 99)),
            "step_drot_deg_median": float(np.median(dr)),
            "step_drot_deg_p99": float(np.percentile(dr, 99))}


def cmd_odom_report():
    rep = {}
    for sensor, q in VAL + TEST:
        f = ODOM_DIR / f"{sensor}_{q}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        r = _odometry_errors(z["odom_poses"], z["gt_poses"])
        r.update(n_scans=meta["n_scans"], runtime_s=meta["runtime_s"])
        rep[f"{sensor}/{q}"] = r
        w = r["win100_rmse_m_mean"]
        print(f"{sensor}/{q:14s} {r['length_m'] / 1000:6.2f} km  ATE "
              f"{r['ate_rmse_m']:7.2f} m ({r['ate_pct_of_length']:5.2f}%)  100m-win "
              f"{w if w is not None else float('nan'):5.2f} m  step |dt| "
              f"{r['step_dt_m_median']:.3f}/{r['step_dt_m_p99']:.3f} m  |drot| "
              f"{r['step_drot_deg_median']:.3f}/{r['step_drot_deg_p99']:.3f} deg",
              flush=True)
    (OUT_ROOT / "odometry_report.json").write_text(json.dumps(rep, indent=2))


# ---------------------------------------------------------------- caches
class _OdomLoader:
    """The dataset loader with 'pose' replaced by the odometry pose."""

    def __init__(self, loader, odom_poses):
        assert len(loader) == len(odom_poses), "scan count drifted"
        self.loader, self.odom = loader, odom_poses

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, i):
        item = dict(self.loader[i])
        item["pose"] = self.odom[i]
        return item


def cmd_caches(role, seq, device):
    from _pd_build_test_caches import (
        _effective_config, _make_loader, _build_operating_cache,
        _build_layouts_and_spec)
    sensor, q = _split(seq)
    assert (sensor, q) in (VAL if role == "val" else TEST), seq
    key = _keys()[sensor]
    paths = {
        "operating_cache": ODOM_CACHE / f"cache_{key}_{sensor}_{role}_{q}.npz",
        "testlayouts": ODOM_CACHE / f"{role}layouts_{sensor}_{q}.npz",
        "testspec": ODOM_CACHE / f"{role}spec_{sensor}_{q}.npz",
    }
    if all(p.exists() for p in paths.values()):
        print(f"SKIP {seq}: all three artifacts exist", flush=True)
        return
    ODOM_CACHE.mkdir(parents=True, exist_ok=True)
    z = np.load(ODOM_DIR / f"{sensor}_{q}.npz")
    loader = _OdomLoader(_make_loader(sensor, q), z["odom_poses"])
    if not paths["operating_cache"].exists():
        from train_multi_dataset import save_keyframes_cache
        kfs = _build_operating_cache(_effective_config(), sensor, q, loader,
                                     None, device)
        assert kfs, f"no keyframes for {seq}"
        tmp = paths["operating_cache"].with_suffix(".tmp.npz")
        save_keyframes_cache(tmp, kfs)
        d = dict(np.load(tmp))
        d["poses_gt"] = z["gt_poses"][d["scan_ids"]]
        np.savez(paths["operating_cache"], **d)
        tmp.unlink()
        print(f"OK {paths['operating_cache'].name}: {len(kfs)} keyframes "
              "(odometry poses + poses_gt)", flush=True)
    cache = np.load(paths["operating_cache"])
    _build_layouts_and_spec(sensor, q, loader, cache, paths)
    print("C3_CACHES_DONE", flush=True)


# ---------------------------------------------------------------- variant
def _register_variant():
    import _causal_branch_fusion as cbf
    cbf.VARIANTS[VARIANT] = dict(ckpt_dir="pdv2_causal_s{seed}", rvec=RVEC,
                                 cls=CLS, config=CONFIG)
    return cbf


def cmd_clf(only, train_only):
    import _pd_build_edge_classifier as pdec
    import _causal_build_edge_classifier as cec
    pdec.PD_CACHE = V2                 # GT train caches (supervision)
    train_seqs = {}
    for s, q in TRAIN:
        train_seqs.setdefault(s, []).append(q)
    cec.PD_TRAIN_SEQS = train_seqs
    cec.PD_HOLDOUT = EDGE_HOLDOUT
    cec.VARIANTS[VARIANT] = dict(ckpt_dir="pdv2_causal_s{seed}", rvec=RVEC,
                                 config=CONFIG, out=CLS,
                                 feat_dir=REPO / "results/_edge_cls_pdv2_causal")
    sys.argv = [cec.__file__, "--variant", VARIANT, "--seed", str(SEED)] + \
        (["--only", only] if only else []) + (["--train-only"] if train_only else [])
    cec.main()


# ---------------------------------------------------------------- track A
def _val_module(cond):
    import _causal_val_select as vs
    _register_variant()
    out = OUT_ROOT / cond
    out.mkdir(parents=True, exist_ok=True)
    vs.PD_CACHE = CACHE_OF[cond]
    vs.OUT = out
    vs.SELECTION = out / "causal_tuple_selection.json"
    vs.VAL_SEQS = tuple(VAL)
    vs.SEQ_KEYS = [f"{s}/{q}" for s, q in VAL]
    vs.VARIANT_CHOICES = (VARIANT,)
    vs.SEEDS = (SEED,)
    return vs


def _worker_args(device="cuda"):
    return argparse.Namespace(variant=VARIANT, seed=SEED, ops=list(OPS),
                              device=device, pool_k=100, overwrite=False)


def cmd_val_tuples(cond, tag, device):
    vs = _val_module(cond)
    sensor, q = VAL_TAGS[tag]
    vs.run_tuples_worker(_worker_args(device), sensor, q)


def _groups(keys):
    g = {}
    for k in keys:
        g.setdefault(_family(k.split("/")[0]), []).append(k)
    return g


def cmd_select_tuple(cond):
    vs = _val_module(cond)
    if vs.SELECTION.exists():
        print(f"{vs.SELECTION} exists; frozen", flush=True)
        return
    keys = vs.SEQ_KEYS
    groups = _groups(keys)
    selections = {}
    for op in OPS:
        rep = {k: json.loads(vs.val_tuples_path(VARIANT, op, SEED,
                                                *k.split("/")).read_text())
               for k in keys}
        ws = [w for w in rep[keys[0]]["tuples"] if w.startswith("(")]

        def pooled(w, g):
            return (sum(rep[k]["tuples"][w]["r1"] * rep[k]["n_queries"] for k in g)
                    / sum(rep[k]["n_queries"] for k in g))

        def family_macro(w):
            return statistics.mean(pooled(w, g) for g in groups.values())

        w_sel = max(ws, key=family_macro)       # ties -> first in WEIGHTS order
        w_q = max(ws, key=lambda w: pooled(w, keys))
        selections[f"{VARIANT}/{op}"] = {
            "selected_tuple": w_sel,
            "tuple": [float(x) for x in ast.literal_eval(w_sel)],
            "family_macro": family_macro(w_sel),
            "per_family": {f: pooled(w_sel, g) for f, g in groups.items()},
            "query_weighted": pooled(w_sel, keys),
            "per_sequence": {k: rep[k]["tuples"][w_sel]["r1"] for k in keys},
            "n_queries": {k: rep[k]["n_queries"] for k in keys},
            "reference_query_weighted_choice": {
                "tuple": w_q, "query_weighted": pooled(w_q, keys),
                "family_macro": family_macro(w_q)},
        }
        print(f"{op}: tuple {w_sel} family-macro {family_macro(w_sel):.4f} "
              f"(query-weighted choice {w_q})", flush=True)
    vs.SELECTION.write_text(json.dumps({
        "schema": "causal-tuple-selection-v1",
        "rule": "argmax over WEIGHTS of the mean over sensor families of the "
                "query-pooled causal R@1 within each family (split v2 rule; "
                f"seed {SEED})",
        "condition": cond, "selection_pool": keys,
        "selections": selections}, indent=2))
    print("C3_SELECT_TUPLE_DONE", flush=True)


def cmd_val_conf(cond, tag, device):
    vs = _val_module(cond)
    sensor, q = VAL_TAGS[tag]
    vs.run_confidence_worker(_worker_args(device), sensor, q)


def _family_macro_tau_star(seq_sweeps, groups):
    """vs._macro_f1_tau_star with families weighted equally: per-sequence F1
    at every candidate tau, mean within a family, mean over families."""
    import _causal_val_select as vs
    keys = list(seq_sweeps)
    cand = np.unique(np.concatenate(
        [s["thresholds"][np.isfinite(s["thresholds"])] for s in seq_sweeps.values()]))
    per = {}
    for k, s in seq_sweeps.items():
        asc_t = s["thresholds"][::-1]
        f1_asc = vs._f1_curve(s)[::-1]
        per[k] = f1_asc[np.searchsorted(asc_t, cand, side="left")]
    fam = np.stack([np.mean([per[k] for k in g], axis=0) for g in groups.values()])
    macro = fam.mean(axis=0)
    i = len(macro) - 1 - int(np.argmax(macro[::-1]))   # ties -> highest tau
    return float(cand[i]), float(macro[i]), {k: float(per[k][i]) for k in keys}


def cmd_conf_select(cond):
    from _pr_eval_lib import classify_queries, pr_sweep
    vs = _val_module(cond)
    out = vs.OUT / "confidence_selection.json"
    if out.exists():
        print(f"{out} exists; frozen", flush=True)
        return
    keys = vs.SEQ_KEYS
    groups = _groups(keys)
    res = {}
    for op in OPS:
        w, w_str = vs.selected_tuple(VARIANT, op)
        sweeps = {}
        for k in keys:
            z = np.load(vs.valdump_path(VARIANT, op, SEED, *k.split("/")))
            assert [float(x) for x in json.loads(str(z["meta"]))["tuple"]] == list(w)
            qc = classify_queries(z["min_past_gt_dist"])
            sweeps[k] = {stat: pr_sweep(z[col], z["top1_geo_dist"], qc)
                         for stat, col in vs.STATS.items()}
        ap = {stat: statistics.mean(
                  statistics.mean(sweeps[k][stat]["AP"] for k in g)
                  for g in groups.values())
              for stat in vs.STATS}
        gap = abs(ap["absolute"] - ap["margin"])
        decision = "absolute" if (gap < TIE_BREAK_DELTA
                                  or ap["absolute"] >= ap["margin"]) else "margin"
        tau, mf1, per_f1 = _family_macro_tau_star(
            {k: sweeps[k][decision] for k in keys}, groups)
        res[op] = {"tuple": list(w), "tuple_str": w_str,
                   "family_macro_AP": ap, "decision": decision,
                   "tau_star": tau, "val_family_macro_F1": mf1,
                   "per_sequence_F1_at_tau": per_f1,
                   "per_sequence_AP": {k: {s: sweeps[k][s]["AP"] for s in vs.STATS}
                                       for k in keys},
                   "counts": {k: sweeps[k]["absolute"]["counts"] for k in keys}}
        print(f"{op}: AP abs {ap['absolute']:.4f} margin {ap['margin']:.4f} -> "
              f"{decision}; tau* {tau:.6f} val F1 {mf1:.4f}", flush=True)
    out.write_text(json.dumps({
        "rule": "statistic = argmax family-macro val AP (|gap| < 0.005 -> "
                "absolute); tau* = argmax family-macro val F1 (ties -> highest "
                "tau); families weighted equally (split v2 rule)",
        "condition": cond, "selection_pool": keys, "ops": res}, indent=2))
    print("C3_CONF_SELECT_DONE", flush=True)


# ---------------------------------------------------------------- track B
def cmd_test_dump(cond, seq, device):
    import _pd_apply_frozen as af
    import _causal_pr_dump as pr
    _register_variant()
    af.PD_CACHE = CACHE_OF[cond]      # resolve_caches: test_ > val_ > unmarked
    out = OUT_ROOT / cond
    pr.OUT = out
    sensor, q = _split(seq)
    assert (sensor, q) in TEST, seq
    sel = json.loads((out / "causal_tuple_selection.json").read_text())["selections"]
    assert (out / "confidence_selection.json").exists(), "run conf-select first"
    args = argparse.Namespace(
        variant=VARIANT, seed=SEED, ops=list(OPS), device=device, pool_k=100,
        overwrite=False, tuple=None,
        tuple_pbev=sel[f"{VARIANT}/pbev128_main"]["tuple"],
        tuple_hspec=sel[f"{VARIANT}/hspec12"]["tuple"])
    pr.run_worker(args, sensor, q)


def _revisit_r1(top1_dists, qc):
    """Fraction of positive (revisit) queries whose returned top-1 past match
    lies within the match radius -- the pr_sweep correctness rule, with the
    acceptance score ignored (diagnostic added 2026-10-09)."""
    from _pr_eval_lib import MATCH_RADIUS_M
    d = np.asarray(top1_dists, dtype=np.float64)[qc["positive"]]
    return float((np.isfinite(d) & (d <= MATCH_RADIUS_M)).mean())


def cmd_prefix_check(cond, seq, ks, device):
    """Truncation test of the past-only embedding on one val sequence: for each
    k, the pipeline run on keyframes [0, k] must reproduce rows 0..k of the
    full-sequence embedding (prefix invariance; a ledger row t only reads
    embedding rows <= t, so this covers j*, scores and candidates)."""
    import _pd_apply_frozen as af
    from _lag_branch_fusion import TruncatedCache
    cbf = _register_variant()
    sensor, q = _split(seq)
    assert (sensor, q) in VAL, "prefix check runs on validation sequences"
    af.PD_CACHE = CACHE_OF[cond]
    cache_p, lay_p, _, _ = af.resolve_caches(sensor, q)
    cache, lay = np.load(cache_p), np.load(lay_p)
    assert (lay["scan_ids"] == cache["scan_ids"]).all()
    d288 = cache["descriptors"].astype(np.float32)
    layouts = lay["nsd_layouts"].astype(np.float32)
    cfg, model, clf, blob, r, ckpt = cbf.make_model_causal(VARIANT, SEED, device)
    full = cbf.nsd_embedding_causal(cache, sensor, d288, layouts, cfg, model, clf,
                                    blob, r, device)
    rows = []
    for k in [k for k in ks if k < len(d288)]:
        n = k + 1
        e = cbf.nsd_embedding_causal(TruncatedCache(cache, n), sensor, d288[:n],
                                     layouts[:n], cfg, model, clf, blob, r, device)
        rows.append({"k": int(k), "row_k": float(np.abs(e[k] - full[k]).max()),
                     "rows_le_k": float(np.abs(e - full[:n]).max())})
        print(f"k={k:6d}  |prefix-full| row k {rows[-1]['row_k']:.3e}  "
              f"rows<=k {rows[-1]['rows_le_k']:.3e}", flush=True)
    out = OUT_ROOT / cond / f"prefix_check_{VARIANT}_{sensor}_{q}.json"
    out.write_text(json.dumps({"seq": seq, "condition": cond, "checkpoint": str(ckpt),
                               "device": device, "n_frames": int(len(d288)),
                               "rows": rows}, indent=2))
    print(f"wrote {out}", flush=True)


# --------------------------------------------- alignment confidence (2026-10-09)
# Confidence candidates widened AFTER the test-set false-positive diagnosis
# (docs/RAW/실험일지/2026-10-09_fp_investigation.md): the two §1.5 statistics
# plus the raw values, at the returned match j*, of the two cyclic-alignment
# fusion channels. j* and every other step are unchanged.
ALIGN_STATS = ("absolute", "margin", "c_spec", "c_col")


def _host_path(path):
    """Ledger metadata stores container paths; map them onto this checkout."""
    return REPO / str(path).split("/Neural-Spectral-Codec/", 1)[1]


def _align_path(dump_path):
    return dump_path.with_name("align_" + dump_path.name)


def _stat_scores(dump_path, z, stat):
    """Per-query decision score of one candidate statistic for a ledger."""
    import _causal_val_select as vs
    if stat in vs.STATS:
        return z[vs.STATS[stat]]
    a = np.load(_align_path(dump_path))
    assert (a["query_idx"] == z["query_idx"]).all(), dump_path
    assert (a["j_star"] == z["j_star"]).all(), dump_path
    return a[stat]


def cmd_align_channels(cond):
    """c_spec (_xs_eval_lib.align_scores) and c_col (column_align_scores) at
    (t, j*) for every val and test ledger of this condition: the pre-min-max
    values of fusion channels 3 and 4 of _causal_pr_dump.dump_op, computed
    from the spec cache recorded in the ledger. CPU only."""
    from _xs_eval_lib import (align_scores, column_align_scores, hspec_inputs,
                              variant_inputs)
    out = OUT_ROOT / cond
    dumps = (sorted(out.glob(f"valdump_{VARIANT}_*_s{SEED}_*.npz"))
             + sorted(out.glob(f"dump_{VARIANT}_*_s{SEED}_*.npz")))
    assert len(dumps) == 2 * (len(VAL) + len(TEST)), (cond, len(dumps))
    for p in dumps:
        dst = _align_path(p)
        if dst.exists():
            continue
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        spec_path = _host_path(meta["caches"]["spec"])
        spec = np.load(spec_path)
        if meta["op"] == "pbev128_main":
            sk = spec["sk_re"].astype(np.float32) + 1j * spec["sk_im"].astype(np.float32)
            _, hfft, hnorm, column_bank = variant_inputs(sk, 8)
        else:
            _, hfft, hnorm, column_bank = hspec_inputs(
                spec["raw_height"].astype(np.float32), 12)
        assert len(hnorm) == meta["n_frames"], (p.name, "spec length")
        c_spec = np.full(len(z["query_idx"]), np.nan)
        c_col = np.full(len(z["query_idx"]), np.nan)
        for i, (t, j) in enumerate(zip(z["query_idx"], z["j_star"])):
            if j < 0:
                continue
            cand = np.array([int(j)])
            c_spec[i] = align_scores(hfft, hnorm, int(t), cand)[0]
            c_col[i] = column_align_scores(column_bank, int(t), cand)[0]
        np.savez_compressed(dst, query_idx=z["query_idx"], j_star=z["j_star"],
                            c_spec=c_spec, c_col=c_col, meta=json.dumps({
                                "source_ledger": p.name, "op": meta["op"],
                                "spec": str(spec_path.relative_to(REPO)),
                                "definition": "raw fusion channels 3 (align_scores) "
                                              "and 4 (column_align_scores) at j*"}))
        print(f"OK {dst.name}", flush=True)
    print("C3_ALIGN_CHANNELS_DONE", flush=True)


def cmd_conf_select_align(cond):
    """cmd_conf_select over ALIGN_STATS: argmax family-macro val AP, absolute
    kept when within TIE_BREAK_DELTA of the best; tau* as before."""
    from _pr_eval_lib import classify_queries, pr_sweep
    vs = _val_module(cond)
    out = vs.OUT / "confidence_selection_align.json"
    if out.exists():
        print(f"{out} exists; frozen", flush=True)
        return
    keys = vs.SEQ_KEYS
    groups = _groups(keys)
    res = {}
    for op in OPS:
        w, w_str = vs.selected_tuple(VARIANT, op)
        sweeps = {}
        for k in keys:
            p = vs.valdump_path(VARIANT, op, SEED, *k.split("/"))
            z = np.load(p)
            assert [float(x) for x in json.loads(str(z["meta"]))["tuple"]] == list(w)
            qc = classify_queries(z["min_past_gt_dist"])
            sweeps[k] = {stat: pr_sweep(_stat_scores(p, z, stat), z["top1_geo_dist"], qc)
                         for stat in ALIGN_STATS}
        ap = {stat: statistics.mean(
                  statistics.mean(sweeps[k][stat]["AP"] for k in g)
                  for g in groups.values())
              for stat in ALIGN_STATS}
        best = max(ALIGN_STATS, key=lambda s: ap[s])
        decision = "absolute" if ap["absolute"] >= ap[best] - TIE_BREAK_DELTA else best
        tau, mf1, per_f1 = _family_macro_tau_star(
            {k: sweeps[k][decision] for k in keys}, groups)
        res[op] = {"tuple": list(w), "tuple_str": w_str,
                   "family_macro_AP": ap, "decision": decision,
                   "tau_star": tau, "val_family_macro_F1": mf1,
                   "per_sequence_F1_at_tau": per_f1,
                   "per_sequence_AP": {k: {s: sweeps[k][s]["AP"] for s in ALIGN_STATS}
                                       for k in keys}}
        print(f"{op}: " + " ".join(f"{s} {ap[s]:.4f}" for s in ALIGN_STATS)
              + f" -> {decision}; tau* {tau:.6f} val F1 {mf1:.4f}", flush=True)
    out.write_text(json.dumps({
        "rule": "statistic = argmax family-macro val AP over ALIGN_STATS "
                "(absolute kept if within 0.005 of the best); tau* = argmax "
                "family-macro val F1 (ties -> highest tau); families weighted "
                "equally (split v2 rule)",
        "candidates": list(ALIGN_STATS),
        "note": "candidates widened after the test-set false-positive diagnosis "
                "of 2026-10-09; selection itself uses the nine val sequences only",
        "condition": cond, "selection_pool": keys, "ops": res}, indent=2))
    print("C3_CONF_SELECT_ALIGN_DONE", flush=True)


def cmd_aggregate(cond, align=False):
    from _pr_eval_lib import AGG_METRICS, classify_queries, operating_point, pr_sweep
    out = OUT_ROOT / cond
    sel_name = "confidence_selection_align.json" if align else "confidence_selection.json"
    conf = json.loads((out / sel_name).read_text())["ops"]
    summary = {"_meta": {"condition": cond, "seed": SEED,
                         "n_test_sequences": len(TEST),
                         "grouping": "city and dataset groups (sequence mean "
                                     "within a group, groups weighted equally)"}}
    if align:
        summary["_meta"]["confidence_selection"] = sel_name
    for op in OPS:
        stat = conf[op]["decision"]
        tau = conf[op]["tau_star"]
        per = {}
        for s, q in TEST:
            p = out / f"dump_{VARIANT}_{op}_s{SEED}_{s}_{q}.npz"
            z = np.load(p)
            score = _stat_scores(p, z, stat)
            qc = classify_queries(z["min_past_gt_dist"])
            sw = pr_sweep(score, z["top1_geo_dist"], qc)
            opp = operating_point(score, z["top1_geo_dist"], qc, tau)
            per[f"{s}/{q}"] = {**{m: float(sw[m]) for m in AGG_METRICS},
                               "P_at_tau": opp["precision"],
                               "R_at_tau": opp["recall"],
                               "R@1_revisit": _revisit_r1(z["top1_geo_dist"], qc),
                               "counts": sw["counts"]}
        metrics = list(AGG_METRICS) + ["P_at_tau", "R_at_tau", "R@1_revisit"]

        def group_means(G):
            return {g: {m: statistics.mean(per[k][m] for k in ks) for m in metrics}
                    for g, ks in G.items()}

        city = group_means(CITY)
        dset = group_means(DATASET)
        summary[op] = {
            "statistic": stat, "tau_star": tau,
            "city_groups": city,
            "city_macro6": {m: statistics.mean(v[m] for v in city.values())
                            for m in metrics},
            "city_macro5_without_nclt": {
                m: statistics.mean(v[m] for c, v in city.items() if c != "ann_arbor")
                for m in metrics},
            "dataset_macro6": {m: statistics.mean(v[m] for v in dset.values())
                               for m in metrics},
            "per_sequence": per}
        c6 = summary[op]["city_macro6"]
        print(f"== {cond} {op} [{stat}]: city6 AP {c6['AP']:.4f} maxF1 {c6['max_F1']:.4f} "
              f"R@100P {c6['R@100P']:.4f} | P/R@tau* {c6['P_at_tau']:.3f}/"
              f"{c6['R_at_tau']:.3f}", flush=True)
    name = "pr_summary_align.json" if align else "pr_summary.json"
    (out / name).write_text(json.dumps(summary, indent=2))
    print("C3_AGGREGATE_DONE", flush=True)


# ---------------------------------------------------------------- baselines
# SC++ (training-free) and SeqOT (retrained on v2 train, seed 1, last
# weights -- the pd2026 recipe) on the same keyframes and scoring as NSD.
SEQOT_PD_DATA = REPO / "_external/seqot_data_pd2026"
SEQOT_TRAIN_DATA = REPO / "_external/seqot_data_pdv2_train"
SEQOT_RUN = REPO / "_external/seqot_runs_pdv2/s1"
SEQOT_TEST_DATA = {c: REPO / f"_external/seqot_data_pdv2_{c}" for c in CACHE_OF}
BASELINES = {"scpp": "score_scpp", "seqot": "score_cos",
             "bevplace": "score_cos", "ringsharp": "score_trans"}
BEV_WEIGHTS = REPO / "baselines/weights/bevplace_pdv2_s1.pth"
RING_DATA = REPO / "_external/ringsharp_data_pdv2"
RING_TOOLS = REPO / "_external/RINGSharp/tools"
RING_EXP = "ring_sharp_l_nsc_pdv2"
RING_WDIR = RING_TOOLS / "results/weights" / RING_EXP
RING_VAL = OUT_ROOT / "ringsharp_val"
RING_SEL = OUT_ROOT / "ringsharp_pdv2_ckpt_selection.json"


def _test_cache(cond, sensor, q):
    p = CACHE_OF[cond] / f"cache_{_keys()[sensor]}_{sensor}_test_{q}.npz"
    assert p.exists(), p
    return p


def cmd_seqot_train_data():
    """v2 train range stacks: the pd2026 SeqOT stacks of the same keyframes
    (hard links), poses checked against the v2 GT train caches; train_index
    built (seed 42, as _pdb_seqot_prepare) where pd2026 had none."""
    import glob
    import os
    import shutil
    import _seqot_prepare as sp
    for s, q in TRAIN:
        src, dst = SEQOT_PD_DATA / f"{s}_{q}", SEQOT_TRAIN_DATA / f"{s}_{q}"
        cache = np.load(glob.glob(str(V2 / f"cache_*_{s}_{q}.npz"))[0])
        poses = np.load(src / "poses.npy")
        assert np.array_equal(poses, cache["poses"].astype(np.float64)), (s, q)
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("[0-9]" * 6 + ".npy"))
        assert len(files) == len(poses), (s, q, len(files), len(poses))
        for f in files + [src / "poses.npy"]:
            if not (dst / f.name).exists():
                os.link(f, dst / f.name)
        meta = json.loads((src / "meta.json").read_text())
        meta["split"] = "pdv2_train"
        (dst / "meta.json").write_text(json.dumps(meta, indent=2))
        if (src / "train_index.npy").exists():
            if not (dst / "train_index.npy").exists():
                shutil.copy2(src / "train_index.npy", dst / "train_index.npy")
        elif not (dst / "train_index.npy").exists():
            idx, n_anchor = sp.build_train_index(poses[:, :3, 3],
                                                 np.random.default_rng(42))
            np.save(dst / "train_index.npy", idx)
            print(f"  {s}_{q}: new train_index {len(idx)} rows, "
                  f"{n_anchor}/{len(poses)} anchors", flush=True)
        print(f"OK {dst.name}: {len(files)} frames", flush=True)


def cmd_seqot_train():
    import _seqot_train
    want = {f"{s}_{q}" for s, q in TRAIN}
    found = set(_seqot_train.find_train_seqs(SEQOT_TRAIN_DATA))
    assert found == want, f"PLACE-DISJOINT GUARD: {sorted(found ^ want)}"
    if (SEQOT_RUN / "gem_latest.pth.tar").exists():
        print(f"SKIP: {SEQOT_RUN} already trained", flush=True)
        return
    sys.argv = [_seqot_train.__file__, "--data-root", str(SEQOT_TRAIN_DATA),
                "--run-dir", str(SEQOT_RUN), "--seed", str(SEED)]
    _seqot_train.main()


def cmd_seqot_prep(cond, seq):
    import _pdb_seqot_prepare as pp
    sensor, q = _split(seq)
    assert (sensor, q) in TEST, seq
    if (SEQOT_TEST_DATA[cond] / f"{sensor}_{q}" / "meta.json").exists():
        print(f"SKIP {cond} {seq}", flush=True)
        return
    pp.prepare_sequence(sensor, q, "test", _test_cache(cond, sensor, q),
                        SEQOT_TEST_DATA[cond], 0, np.random.default_rng(42))


def cmd_seqot_dump(cond, seq, device):
    import _pd_apply_frozen as af
    import _causal_seqot_pr as cs
    af.PD_CACHE = CACHE_OF[cond]
    cs.OUT = OUT_ROOT / cond
    sensor, q = _split(seq)
    assert (sensor, q) in TEST, seq
    args = argparse.Namespace(
        seqot_checkpoint=str(SEQOT_RUN / "seqot_latest.pth.tar"),
        gem_checkpoint=str(SEQOT_RUN / "gem_latest.pth.tar"),
        data_root=str(SEQOT_TEST_DATA[cond]), infer_batch=16, device=device,
        max_frames=0, overwrite=False)
    cs.run_worker(args, sensor, q)


def cmd_scpp_dump(cond, seq):
    import _causal_scpp_pr as sc
    sensor, q = _split(seq)
    assert (sensor, q) in TEST, seq
    sc._resolve_cache = lambda s, qq: _test_cache(cond, s, qq)
    sc.run_sequence(sensor, q, 8, OUT_ROOT / cond, False)


def cmd_aggregate_baselines(cond):
    from _pr_eval_lib import AGG_METRICS, classify_queries, pr_sweep
    out = OUT_ROOT / cond
    summary = {"_meta": {"condition": cond, "n_test_sequences": len(TEST),
                         "score_keys": BASELINES,
                         "seqot": "retrained on v2 train, seed 1, last weights"}}
    for m, col in BASELINES.items():
        missing = [f"{s}/{q}" for s, q in TEST
                   if not (out / f"dump_{m}_{s}_{q}.npz").exists()]
        if missing:
            print(f"-- {cond} {m}: {len(missing)} dumps missing, skipped", flush=True)
            continue
        per = {}
        for s, q in TEST:
            z = np.load(out / f"dump_{m}_{s}_{q}.npz")
            qc = classify_queries(z["min_past_gt_dist"])
            sw = pr_sweep(z[col], z["top1_geo_dist"], qc)
            per[f"{s}/{q}"] = {**{k: float(sw[k]) for k in AGG_METRICS},
                               "R@1_revisit": _revisit_r1(z["top1_geo_dist"], qc),
                               "counts": sw["counts"]}
        metrics = list(AGG_METRICS) + ["R@1_revisit"]

        def group_means(G):
            return {g: {k: statistics.mean(per[x][k] for x in ks)
                        for k in metrics} for g, ks in G.items()}

        city, dset = group_means(CITY), group_means(DATASET)
        summary[m] = {
            "city_groups": city,
            "city_macro6": {k: statistics.mean(v[k] for v in city.values())
                            for k in metrics},
            "city_macro5_without_nclt": {
                k: statistics.mean(v[k] for c, v in city.items() if c != "ann_arbor")
                for k in metrics},
            "dataset_macro6": {k: statistics.mean(v[k] for v in dset.values())
                               for k in metrics},
            "per_sequence": per}
        c6 = summary[m]["city_macro6"]
        print(f"== {cond} {m}: city6 AP {c6['AP']:.4f} maxF1 {c6['max_F1']:.4f} "
              f"R@100P {c6['R@100P']:.4f}", flush=True)
    (out / "pr_summary_baselines.json").write_text(json.dumps(summary, indent=2))
    print("C3_AGGREGATE_BASELINES_DONE", flush=True)


def _family_pooled(per_seq):
    """{'sensor/seq' or 'sensor_seq': {'r1'|'recall@1', 'n_queries'}} ->
    mean over sensor families of the query-pooled R@1 (split v2 rule)."""
    groups = {}
    for name, v in per_seq.items():
        sensor = name.replace("/", "_").split("_", 1)[0]
        groups.setdefault(_family(sensor), []).append(v)
    r = lambda v: v["r1"] if "r1" in v else v["recall@1"]  # noqa: E731
    return statistics.mean(
        sum(r(v) * v["n_queries"] for v in g) / sum(v["n_queries"] for v in g)
        for g in groups.values() if sum(v["n_queries"] for v in g) > 0)


def cmd_bev_train():
    """BEVPlace++ on v2 train from ImageNet ResNet34 (author recipe: Adam
    lr 1e-4, 40 epochs; script's weight decay 1e-3 and same-sequence
    triplets kept), best epoch by the v2 val family-equal rule."""
    import _pdb_bevplace_train as bt
    from _pd_build_test_caches import _make_loader
    done = BEV_WEIGHTS.with_name(BEV_WEIGHTS.name + ".done")
    if done.exists():   # the best-epoch file appears after epoch 1 already
        print(f"SKIP: {done} exists", flush=True)
        return
    keys = _keys()
    bt.VAL_CACHES = {(s, q): str((V2 / f"cache_{keys[s]}_{s}_val_{q}.npz")
                                 .relative_to(REPO)) for s, q in VAL}
    orig_loader, orig_val = bt.create_loader, bt.val_recall_at_1
    bt.create_loader = (lambda t, root, seq: _make_loader(t, seq)
                        if t == "kitti360" else orig_loader(t, root, seq))

    def val_family(model, val_sets, device, batch_size=8):
        res = orig_val(model, val_sets, device, batch_size)
        res["seq_mean_r1"] = res["mean_r1"]
        res["mean_r1"] = _family_pooled(res["per_seq"])
        return res

    bt.val_recall_at_1 = val_family
    sys.argv = [bt.__file__, "--config", str(REPO / "configs/training_place_disjoint_v2.yaml"),
                "--output", str(BEV_WEIGHTS), "--init-weights", "imagenet",
                "--epochs", "40", "--lr", "1e-4", "--seed", str(SEED)]
    bt.main()
    done.write_text("training finished\n")


def cmd_bev_dump(cond, seq, device):
    import _causal_bevplace_pr as bc
    sensor, q = _split(seq)
    assert (sensor, q) in TEST, seq
    bc.OUT = OUT_ROOT / cond
    args = argparse.Namespace(weights=BEV_WEIGHTS, overwrite=False, device=device)
    bc.run_worker(args, sensor, q,
                  str(_test_cache(cond, sensor, q).relative_to(REPO)))


def cmd_ring_prep():
    import _pdb_ringsharp_prepare as rp
    rp.OUT = RING_DATA
    rp.CONFIG = REPO / "configs/training_place_disjoint_v2.yaml"
    rp.CACHE_DIR = V2
    rp.EXPECTED_TRAIN = len(TRAIN)
    if not (RING_DATA / "train_nsc.pickle").exists():
        rp.main()
    src = REPO / "_external/ringsharp_data_pd2026/config_nsc_pd2026.txt"
    txt = src.read_text().replace("ringsharp_data_pd2026", RING_DATA.name) \
        .replace("pd2026 place-disjoint 16-sequence split", "split v2 8-sequence split")
    (RING_DATA / "config_nsc_pdv2.txt").write_text(txt)
    print(f"wrote {RING_DATA / 'config_nsc_pdv2.txt'}", flush=True)


def cmd_ring_train():
    import os
    import subprocess
    if list(RING_WDIR.glob("model_*_5.pth")):
        print(f"SKIP: {RING_WDIR} has epoch-5 weights", flush=True)
        return
    env = dict(os.environ, NSC_MICRO_BATCH="2")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "_external/RINGSharp"), env.get("PYTHONPATH", "")])
    cmd = [sys.executable, "train.py",
           "--config", str(RING_DATA / "config_nsc_pdv2.txt"),
           "--model_config", str(REPO / "_external/RINGSharp/glnet/config/ring_sharp_l_nsc.txt"),
           "--exp_name", RING_EXP]
    print("cwd", RING_TOOLS, "cmd", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=RING_TOOLS, env=env).returncode
    assert rc == 0, f"RING#-L training failed rc={rc}"


def _ring_epoch_weight(e):
    hits = sorted(RING_WDIR.glob(f"model_*_{e}.pth"))
    assert len(hits) == 1, (e, hits)
    return hits[0]


def cmd_ring_val(epoch, seq):
    import _pdb_ringsharp_pd_eval as rpe
    sensor, q = _split(seq)
    assert (sensor, q) in VAL, seq
    rpe.PD_CACHE = V2
    RING_VAL.mkdir(parents=True, exist_ok=True)
    sys.argv = [rpe.__file__, seq, "--weight", str(_ring_epoch_weight(epoch)),
                "--val", "--tag", f"e{epoch}", "--out-dir", str(RING_VAL)]
    rpe.main()


def cmd_ring_select():
    import hashlib
    if RING_SEL.exists():
        print(f"{RING_SEL} exists; frozen", flush=True)
        return
    table = {}
    for e in range(1, 6):
        per = {}
        for s, q in VAL:
            d = json.loads((RING_VAL / f"_val_ringsharp_pd_e{e}_{s}_{q}.json").read_text())
            res = d[f"{s}_{q}"]
            per[f"{s}/{q}"] = {"recall@1": res["recall@1"], "n_queries": res["n_queries"]}
        table[f"e{e}"] = {"family_macro_r1": _family_pooled(per), "per_seq": per}
        print(f"e{e}: family-macro R@1 {table[f'e{e}']['family_macro_r1']:.4f}", flush=True)
    best = min(table, key=lambda t: (-table[t]["family_macro_r1"], int(t[1:])))
    w = _ring_epoch_weight(int(best[1:]))
    sha = hashlib.sha256(w.read_bytes()).hexdigest()
    RING_SEL.write_text(json.dumps({
        "rule": "highest v2-validation family-macro 5 m R@1 over epochs 1-5 "
                "(queries pooled within a sensor family, families weighted "
                "equally); ties -> earlier epoch",
        "selected_epoch": best, "selected_weight": str(w), "weight_sha256": sha,
        "epochs": table}, indent=2))
    print(f"selected {best}: {w.name}", flush=True)


def cmd_ring_dump(cond, seq):
    import _causal_ringsharp_pr as rc
    sensor, q = _split(seq)
    assert (sensor, q) in TEST, seq
    rc.OUT = OUT_ROOT / cond
    rc.CKPT_SELECTION = RING_SEL
    rc.load_roster = lambda: (None, {(sensor, q): _test_cache(cond, sensor, q)})
    args = argparse.Namespace(n_coarse=200, n_trans=20, batch=8, max_frames=0,
                              overwrite=False)
    rc.run_worker(args, sensor, q)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bev-train", "ring-prep", "ring-train", "ring-select"):
        sub.add_parser(name)
    p = sub.add_parser("ring-val")
    p.add_argument("--epoch", type=int, choices=range(1, 6), required=True)
    p.add_argument("--seq", required=True)
    for name in ("bev-dump", "ring-dump"):
        p = sub.add_parser(name)
        p.add_argument("--cond", choices=tuple(CACHE_OF), required=True)
        p.add_argument("--seq", required=True)
        p.add_argument("--device", default="cuda")
    sub.add_parser("seqot-train-data")
    sub.add_parser("seqot-train")
    for name in ("seqot-prep", "seqot-dump", "scpp-dump"):
        p = sub.add_parser(name)
        p.add_argument("--cond", choices=tuple(CACHE_OF), required=True)
        p.add_argument("--seq", required=True)
        p.add_argument("--device", default="cuda")
    p = sub.add_parser("aggregate-baselines")
    p.add_argument("--cond", choices=tuple(CACHE_OF), required=True)
    p = sub.add_parser("odometry"); p.add_argument("--seq", required=True)
    sub.add_parser("odom-report")
    p = sub.add_parser("caches")
    p.add_argument("--role", choices=("val", "test"), required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("clf")
    p.add_argument("--only", default=None)
    p.add_argument("--train-only", action="store_true")
    for name in ("val-tuples", "val-conf"):
        p = sub.add_parser(name)
        p.add_argument("--cond", choices=tuple(CACHE_OF), required=True)
        p.add_argument("--tag", choices=list(VAL_TAGS), required=True)
        p.add_argument("--device", default="cuda")
    for name in ("select-tuple", "conf-select", "aggregate", "align-channels",
                 "conf-select-align", "aggregate-align"):
        p = sub.add_parser(name)
        p.add_argument("--cond", choices=tuple(CACHE_OF), required=True)
    p = sub.add_parser("test-dump")
    p.add_argument("--cond", choices=tuple(CACHE_OF), required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("prefix-check")
    p.add_argument("--cond", choices=tuple(CACHE_OF), required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--ks", type=int, nargs="+", default=[40, 300, 900])
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("roster")
    p.add_argument("--role", choices=("val", "test"), required=True)
    a = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "bev-train":
        cmd_bev_train()
    elif a.cmd == "bev-dump":
        cmd_bev_dump(a.cond, a.seq, a.device)
    elif a.cmd == "ring-prep":
        cmd_ring_prep()
    elif a.cmd == "ring-train":
        cmd_ring_train()
    elif a.cmd == "ring-val":
        cmd_ring_val(a.epoch, a.seq)
    elif a.cmd == "ring-select":
        cmd_ring_select()
    elif a.cmd == "ring-dump":
        cmd_ring_dump(a.cond, a.seq)
    elif a.cmd == "seqot-train-data":
        cmd_seqot_train_data()
    elif a.cmd == "seqot-train":
        cmd_seqot_train()
    elif a.cmd == "seqot-prep":
        cmd_seqot_prep(a.cond, a.seq)
    elif a.cmd == "seqot-dump":
        cmd_seqot_dump(a.cond, a.seq, a.device)
    elif a.cmd == "scpp-dump":
        cmd_scpp_dump(a.cond, a.seq)
    elif a.cmd == "aggregate-baselines":
        cmd_aggregate_baselines(a.cond)
    elif a.cmd == "odometry":
        cmd_odometry(a.seq)
    elif a.cmd == "odom-report":
        cmd_odom_report()
    elif a.cmd == "caches":
        cmd_caches(a.role, a.seq, a.device)
    elif a.cmd == "clf":
        cmd_clf(a.only, a.train_only)
    elif a.cmd == "val-tuples":
        cmd_val_tuples(a.cond, a.tag, a.device)
    elif a.cmd == "select-tuple":
        cmd_select_tuple(a.cond)
    elif a.cmd == "val-conf":
        cmd_val_conf(a.cond, a.tag, a.device)
    elif a.cmd == "conf-select":
        cmd_conf_select(a.cond)
    elif a.cmd == "test-dump":
        cmd_test_dump(a.cond, a.seq, a.device)
    elif a.cmd == "aggregate":
        cmd_aggregate(a.cond)
    elif a.cmd == "align-channels":
        cmd_align_channels(a.cond)
    elif a.cmd == "conf-select-align":
        cmd_conf_select_align(a.cond)
    elif a.cmd == "aggregate-align":
        cmd_aggregate(a.cond, align=True)
    elif a.cmd == "prefix-check":
        cmd_prefix_check(a.cond, a.seq, a.ks, a.device)
    elif a.cmd == "roster":
        print(" ".join(f"{s}/{q}" for s, q in (VAL if a.role == "val" else TEST)))


if __name__ == "__main__":
    main()
