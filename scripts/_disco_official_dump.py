#!/usr/bin/env python3
"""DiSCO official pipeline on split v2 -- part 3: past-only test ledgers
(2026-10-11).

Per split-v2 TEST sequence (46), on the KISS-ICP keyframes of
data/preprocessed_v2_odom (scan_ids; poses_gt only for scoring):
  1. descriptors: official model loading of evaluate.py evaluate()
     (DiSCO(output_dim=1024), Corr2Softmax(200, 0), torch.load,
     load_state_dict, nn.DataParallel) and the official inference
     evaluate.get_latent_vectors (eval mode) on the polar stores of
     scripts/_disco_official_prep.py (role test_odom);
  2. ledger (pr-dump-v1, the arrays and dtypes of
     scripts/_causal_bevplace_pr.ledger): queries t >= 30, exhaustive top-1
     over the prefix j <= t - 30 with the official retrieval metric, the
     Euclidean distance between the raw 1024-D signatures (evaluate.get_recall
     builds a KDTree on them; the signatures are not normalised, so the
     cosine ledger of _causal_bevplace_pr cannot be reused).  score_disco =
     -distance (higher = accept).  The top-1 is exact in float64: a gemv
     shortlist of 8 over the prefix rows only, then the exact distance on the
     shortlist; meta.l2_check repeats 50 queries per sequence with sklearn
     KDTree (the official structure).

Subcommands (inside ring_baseline):
  dump [--seqs S ...] [--use chosen|official_final|val_best] [--checkpoint P]
                                (chosen = val_best, see _disco_official_train.py)
  aggregate                     PR summary via _pdv2_c3 (unchanged)
  selftest                      CPU, synthetic: causality, exhaustive
                                reference, prefix invariance
  dry-run-cpu --seq S --n N --out-dir D   CPU format check, untrained net
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

from _disco_official_prep import (  # noqa: E402
    OFFICIAL_COMMIT, OFFICIAL_REPO, RESULTS, RULE, SEED, cache_path, load_store, seq_name,
    sha256, store_dir)
from _pdv2_pilot import TEST  # noqa: E402
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402

OUT = RESULTS / "odom"
SELECTION = RESULTS / "train_selection.json"
METHOD = "disco"
SCORE = "score_disco"
SHORTLIST = 8


# ------------------------------------------------------------ ledger (torch-free)
def l2_top1_prefix(D, sq, t, n_prefix, k=SHORTLIST):
    """Exact float64 Euclidean nearest neighbour of row t among rows
    [0, n_prefix): gemv over the prefix rows only (rows >= n_prefix are never
    read), shortlist of k, exact distances, ties -> lower index."""
    d2 = sq[:n_prefix] - 2.0 * (D[:n_prefix] @ D[t]) + sq[t]
    k = min(k, n_prefix)
    cand = np.argpartition(d2, k - 1)[:k] if k < n_prefix else np.arange(n_prefix)
    ex = np.sqrt(((D[cand] - D[t]) ** 2).sum(axis=1))
    o = np.lexsort((cand, ex))[0]
    return int(cand[o]), float(ex[o])


def ledger(desc: np.ndarray, positions: np.ndarray) -> dict:
    """pr-dump-v1 arrays; min_past_gt_dist as _causal_bevplace_pr.ledger."""
    n = len(desc)
    assert len(positions) == n
    D = np.asarray(desc, dtype=np.float64)
    sq = np.einsum("ij,ij->i", D, D)
    min_past = np.full(n, np.nan)
    for t in range(MIN_INDEX_GAP, n):
        min_past[t] = np.linalg.norm(positions[:t - MIN_INDEX_GAP + 1] - positions[t],
                                     axis=1).min()
    queries = np.arange(MIN_INDEX_GAP, n, dtype=np.int64)
    j_star = np.full(len(queries), -1, dtype=np.int64)
    score = np.full(len(queries), np.nan)
    top1_geo = np.full(len(queries), np.nan)
    n_cand = np.zeros(len(queries), dtype=np.int64)
    for qi, t in enumerate(queries):
        t = int(t)
        n_prefix = t - MIN_INDEX_GAP + 1
        j, dist = l2_top1_prefix(D, sq, t, n_prefix)
        assert j <= t - MIN_INDEX_GAP
        j_star[qi], score[qi] = j, -dist
        top1_geo[qi] = float(np.linalg.norm(positions[j] - positions[t]))
        n_cand[qi] = n_prefix
        if qi and qi % 2000 == 0:
            print(f"  ledger queries {qi}/{len(queries)}", flush=True)
    return {"query_idx": queries, "j_star": j_star, SCORE: score,
            "top1_geo_dist": top1_geo, "min_past_gt_dist": min_past[queries],
            "n_candidates": n_cand}


def assert_descriptors_distinct(desc, name, min_fraction=0.5):
    """Guard against degenerate inputs (e.g. failed CUDA polar images, which
    made every descriptor of a sequence identical on 2026-10-12): at least
    `min_fraction` of the keyframes must have distinct descriptors."""
    n_distinct = len(np.unique(np.round(desc, 6), axis=0))
    frac = n_distinct / max(len(desc), 1)
    assert frac >= min_fraction, (f"{name}: only {n_distinct}/{len(desc)} distinct descriptors "
                                  "-- degenerate inputs; run the prep `verify`")
    return frac


def _l2_check(desc, arrays, n=50):
    """Official retrieval structure (sklearn KDTree, Euclidean) on the prefix."""
    from sklearn.neighbors import KDTree
    q, js = arrays["query_idx"], arrays["j_star"]
    pick = np.unique(np.linspace(0, len(q) - 1, min(n, len(q))).round().astype(int))
    agree = 0
    for k in pick:
        t = int(q[k])
        _, ind = KDTree(desc[:t - MIN_INDEX_GAP + 1]).query(desc[t:t + 1], k=1)
        agree += int(ind[0][0] == js[k])
    return {"n_checked": int(len(pick)), "kdtree_top1_equals_j_star": int(agree)}


# ------------------------------------------------------------ official model
def official_model(checkpoint: Path | None, cpu_only: bool = False, fake=None):
    """evaluate.py evaluate() model set-up (lines 33-46)."""
    from _disco_official_prep import import_official, official_env
    torch = official_env(cpu_only)
    if not cpu_only:   # initialise CUDA with the caller's CUDA_VISIBLE_DEVICES
        assert torch.cuda.is_available()   # before evaluate.py:7 sets it to 0
    cfg, lp, SC, LF, EV = import_official(cpu_only=cpu_only, fake_gputransform=fake)
    import torch.nn as nn
    model = SC.DiSCO(output_dim=cfg.FEATURE_OUTPUT_DIM)
    corr2soft = SC.Corr2Softmax(200., 0.)
    corr2soft = corr2soft.to(EV.device)
    model = model.to(EV.device)
    if checkpoint is not None:
        ck = torch.load(checkpoint, map_location=lambda storage, loc: storage,
                        weights_only=False)   # own file; torch>=2.6 default is True
        model.load_state_dict(ck["state_dict"])
        corr2soft.load_state_dict(ck["corr2soft"])
        print(f"=> loaded {checkpoint} (epoch {ck['epoch']}, iter {ck.get('iter')})", flush=True)
    model = nn.DataParallel(model)
    return cfg, lp, EV, model


def encode_and_dump(EV, model, bank, sensor, q, out_path, checkpoint, chosen,
                    limit=0, extra_meta=None):
    from _disco_official_train import descriptors
    name = seq_name(sensor, q)
    cp = cache_path("test_odom", sensor, q)
    with np.load(cp) as z:
        sids = z["scan_ids"].astype(np.int64)
        poses_gt = z["poses_gt"].astype(np.float64)
    smeta = json.loads((store_dir("test_odom", sensor, q) / "meta.json").read_text())
    idx = np.load(store_dir("test_odom", sensor, q) / "index.npz")
    assert np.array_equal(idx["scan_ids"], sids) and np.array_equal(idx["poses"], poses_gt), name
    n = limit or len(sids)
    print(f"[{sensor}/{q}] encoding {n} keyframes", flush=True)
    t0 = time.time()
    desc = descriptors(EV, model, bank, "test_odom", sensor, q, n)   # official inference
    t_enc = time.time() - t0
    distinct = assert_descriptors_distinct(desc, f"{sensor}/{q}")
    t0 = time.time()
    arrays = ledger(desc, poses_gt[:n, :3, 3])
    t_led = time.time() - t0
    l2 = _l2_check(desc, arrays)
    norms = np.linalg.norm(desc, axis=1)
    meta = {
        "schema": "pr-dump-v1",
        "method": METHOD,
        "variant": f"official DiSCO-pytorch recipe retrained on split-v2 train (seed {SEED})",
        "sensor": sensor, "sequence": q,
        "min_index_gap": MIN_INDEX_GAP,
        "n_frames": int(n),
        "n_queries": int(len(arrays["query_idx"])),
        "primary_score": SCORE,
        "score_direction": "higher = accept (negated distance)",
        "scores": {SCORE: "NEGATED Euclidean distance between the 1024-D DiSCO "
                          "signatures (centre 32x32 of the FFT magnitude, not "
                          "normalised) of the exhaustive prefix top-1; official "
                          "retrieval metric (evaluate.get_recall: sklearn KDTree, "
                          "Euclidean)"},
        "descriptor_norm": {"min": float(norms.min()), "max": float(norms.max()),
                            "median": float(np.median(norms))},
        "descriptor_distinct_fraction": round(float(distinct), 4),
        "l2_check": l2,
        "official": {"repo": OFFICIAL_REPO, "commit": OFFICIAL_COMMIT,
                     "code": "models/DiSCO.py, evaluate.py evaluate() loading and "
                             "get_latent_vectors, generator load_pc_file + "
                             "gputransform (polar occupancy)"},
        "artifacts": {"checkpoint": str(checkpoint) if checkpoint else None,
                      "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
                      "chosen": chosen, "selection": str(SELECTION.relative_to(REPO))},
        "caches": {"operating_cache": str(cp.relative_to(REPO)),
                   "operating_cache_sha256": smeta["cache_sha256"],
                   "polar_store": str(store_dir("test_odom", sensor, q).relative_to(REPO)),
                   "polar_rule": smeta["rule"], "polar_z_shift_m": smeta["z_shift"],
                   "polar_empty_scans": smeta.get("n_empty"),
                   "positions": "poses_gt (scoring only)"},
        "encoder": {"polar": "20 height x 40 ring x 120 sector occupancy (official "
                             "code values): 70 m, 1.75 m rings, 3 deg sectors, 2 m "
                             "layers over -20..20 m (KITTI family after +2 m shift)",
                    "descriptor_dim": int(desc.shape[1]), "normalized": False},
        "protocol": ("causal all-frames PR ledger: per-frame DiSCO signatures computed "
                     "once per sequence (each depends only on its own scan), then per "
                     "query an exhaustive Euclidean top-1 restricted to prefix rows "
                     "j <= t - 30; KISS-ICP keyframes, GT poses for scoring only"),
        "timing": {"encode_s": round(t_enc, 1),
                   "encode_ms_per_kf": round(t_enc / max(n, 1) * 1e3, 2),
                   "ledger_s": round(t_led, 1)},
    }
    meta.update(extra_meta or {})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    print(f"[{sensor}/{q}] n_queries={len(arrays['query_idx'])} kdtree-check "
          f"{l2['kdtree_top1_equals_j_star']}/{l2['n_checked']} ({t_enc:.0f} s encode, "
          f"{t_led:.0f} s ledger) -> {out_path}", flush=True)
    return out_path


def _checkpoint(use: str):
    sel = json.loads(SELECTION.read_text())
    assert sel.get("finished"), f"{SELECTION}: training not finished"
    key = sel["chosen"] if use == "chosen" else use
    ck = Path(sel[key]["checkpoint"])
    assert sha256(ck) == sel[key]["sha256"], f"{ck} changed after training"
    return ck, key


def cmd_dump(seqs, checkpoint, use, worker, overwrite):
    if checkpoint is None:
        checkpoint, chosen = _checkpoint(use)
    else:
        chosen = "explicit --checkpoint"
    todo = seqs or [f"{s}/{q}" for s, q in TEST]
    unknown = sorted(set(todo) - {f"{s}/{q}" for s, q in TEST})
    assert not unknown, f"not split-v2 test sequences: {unknown}"
    if worker:
        from _disco_official_train import PolarBank
        assert len(todo) == 1
        sensor, q = todo[0].split("/")
        from _disco_official_prep import check_store
        r = check_store("test_odom", sensor, q, n_sample=16)       # input integrity
        assert r["ok"], f"polar store failed verification: {r}"
        cfg, lp, EV, model = official_model(checkpoint)
        bank = PolarBank()
        lp.load_pc_file = bank.load
        bank.add("test_odom", sensor, q)
        encode_and_dump(EV, model, bank, sensor, q, OUT / f"dump_{METHOD}_{sensor}_{q}.npz",
                        checkpoint, chosen)
        return
    for key in todo:   # one sequence per subprocess (memory discipline)
        sensor, q = key.split("/")
        if (OUT / f"dump_{METHOD}_{sensor}_{q}.npz").exists() and not overwrite:
            print(f"[skip] {key}", flush=True)
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "dump", "--worker",
               "--seqs", key, "--checkpoint", str(checkpoint)]
        rc = subprocess.run(cmd).returncode
        if rc:
            print(f"WORKER_FAILED {key} rc={rc}", flush=True)
            sys.exit(rc)
    print("DISCO_OFFICIAL_DUMP_DONE", flush=True)


def cmd_aggregate():
    import _pdv2_c3 as c3
    c3.OUT_ROOT = RESULTS
    c3.BASELINES = {METHOD: SCORE}
    c3.cmd_aggregate_baselines("odom")
    p = OUT / "pr_summary_baselines.json"
    s = json.loads(p.read_text())
    s["_meta"].pop("seqot", None)   # hard-coded note of the reused aggregator
    sel = json.loads(SELECTION.read_text())
    s["_meta"][METHOD] = (f"official DiSCO-pytorch recipe retrained on split-v2 train, "
                          f"seed {SEED}, checkpoint '{sel.get('chosen')}' of "
                          f"{SELECTION.relative_to(REPO)}")
    p.write_text(json.dumps(s, indent=2))


# ------------------------------------------------------------ checks
def cmd_selftest():
    """CPU, synthetic, torch-free: causal, exhaustive-prefix reference,
    prefix invariance, KDTree agreement."""
    rng = np.random.default_rng(0)
    n = 150
    desc = (rng.standard_normal((n, 32)) * rng.uniform(0.5, 3, (n, 1))).astype(np.float32)
    desc[90] = desc[10]                      # an exact tie: lower index must win
    desc[95] = desc[10]
    pos = rng.uniform(0, 100, (n, 3))
    a = ledger(desc, pos)
    q = a["query_idx"]
    assert (q == np.arange(MIN_INDEX_GAP, n)).all()
    assert (a["j_star"] <= q - MIN_INDEX_GAP).all()
    assert (a["n_candidates"] == q - MIN_INDEX_GAP + 1).all()
    D = desc.astype(np.float64)
    for qi, t in enumerate(q):
        t = int(t)
        d = np.sqrt(((D[:t - MIN_INDEX_GAP + 1] - D[t]) ** 2).sum(1))
        j = int(np.flatnonzero(d == d.min())[0])
        assert a["j_star"][qi] == j and a[SCORE][qi] == -d[j], t
        assert a["min_past_gt_dist"][qi] == np.linalg.norm(
            pos[:t - MIN_INDEX_GAP + 1] - pos[t], axis=1).min()
    assert a["j_star"][95 - MIN_INDEX_GAP] == 10      # tie with 90 (both == row 10)
    for t in (30, 57, 95, n - 1):
        b = ledger(desc[:t + 1], pos[:t + 1])
        for k in a:
            x, y = a[k][t - MIN_INDEX_GAP], b[k][t - MIN_INDEX_GAP]
            assert x == y or (np.isnan(x) and np.isnan(y)), (t, k)
    chk = _l2_check(desc, a)
    assert chk["kdtree_top1_equals_j_star"] == chk["n_checked"], chk
    assert assert_descriptors_distinct(desc, "synthetic") > 0.9
    try:                      # the 2026-10-12 failure: identical descriptors
        assert_descriptors_distinct(np.tile(desc[:1], (n, 1)), "degenerate")
        raise RuntimeError("degenerate descriptors were not refused")
    except AssertionError:
        pass
    print(f"SELFTEST_OK: L2 ledger causal, exact, prefix-invariant; KDTree {chk}", flush=True)


def cmd_dry_run_cpu(key, n, out_dir):
    """CPU only: untrained official network (no checkpoint) on the first n
    keyframes, polar images from the CPU transcription (the CUDA store is
    not built yet); the ledger is compared with the SeqOT ledger of the same
    sequence in results/pdv2_c3/odom (same keyframes and GT positions ->
    identical query_idx / min_past_gt_dist / n_candidates on shared rows)."""
    import os
    import types
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from _disco_official_prep import (fake_gputransform_module, official_hits,
                                      official_polar, pack)
    from _disco_official_train import PolarBank
    from _pd_build_test_caches import _make_loader
    fake = fake_gputransform_module()
    cfg, lp, EV, model = official_model(None, cpu_only=True, fake=fake)
    import torch
    torch.set_num_threads(8)
    assert str(EV.device) == "cpu"
    sensor, q = key.split("/")
    sids = np.load(store_dir("test_odom", sensor, q) / "index.npz")["scan_ids"][:n]
    loader = _make_loader(sensor, q)
    cfg_ = types.SimpleNamespace(max_length=1, max_height=1, num_ring=40, num_sector=120,
                                 num_height=20)
    bits = np.stack([pack(official_polar(official_hits(sensor, loader[int(s)]["points"]),
                                         fake, cfg_)) for s in sids])
    bank = PolarBank()
    lp.load_pc_file = bank.load
    bank.add("test_odom", sensor, q, bits=bits)
    path = encode_and_dump(EV, model, bank, sensor, q,
                           Path(out_dir) / f"dump_{METHOD}_{sensor}_{q}.npz", None, None,
                           limit=n, extra_meta={"DRY_RUN": "CPU, untrained network, CPU "
                                                "transcription of the polar kernel; format "
                                                "check only"})
    z = np.load(path)
    ref = np.load(REPO / f"results/pdv2_c3/odom/dump_seqot_{sensor}_{q}.npz")
    rep = {"ours": {k: [str(z[k].dtype), list(z[k].shape)] for k in z.files},
           "ref_seqot": {k: [str(ref[k].dtype), list(ref[k].shape)] for k in ref.files}}
    m = len(z["query_idx"])
    for k in ("query_idx", "min_past_gt_dist", "n_candidates"):
        rep[f"{k}_equal_on_shared_rows"] = bool(np.array_equal(z[k], ref[k][:m]))
    zm, rm = json.loads(str(z["meta"])), json.loads(str(ref["meta"]))
    rep["meta_schema"] = [zm["schema"], rm["schema"]]
    rep["dtypes_match"] = all(str(z[k].dtype) == str(ref[k].dtype)
                              for k in ("query_idx", "j_star", "top1_geo_dist",
                                        "min_past_gt_dist", "n_candidates"))
    rep["score_dtype"] = [str(z[SCORE].dtype), str(ref["score_cos"].dtype)]
    rep["l2_check"] = zm["l2_check"]
    rep["descriptor_norm"] = zm["descriptor_norm"]
    rep["encode_ms_per_kf_cpu"] = zm["timing"]["encode_ms_per_kf"]
    ok = rep["dtypes_match"] and zm["schema"] == rm["schema"] == "pr-dump-v1" and all(
        rep[f"{k}_equal_on_shared_rows"] for k in ("query_idx", "min_past_gt_dist",
                                                    "n_candidates")) \
        and rep["score_dtype"][0] == "float64"
    print(json.dumps(rep, indent=1))
    print("DRY_RUN_OK" if ok else "DRY_RUN_MISMATCH", flush=True)
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--seqs", nargs="+", default=None)
    p.add_argument("--use", choices=("chosen", "official_final", "val_best"), default="chosen")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    sub.add_parser("aggregate")
    sub.add_parser("selftest")
    p = sub.add_parser("dry-run-cpu")
    p.add_argument("--seq", required=True)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "dump":
        cmd_dump(a.seqs, a.checkpoint, a.use, a.worker, a.overwrite)
    elif a.cmd == "aggregate":
        cmd_aggregate()
    elif a.cmd == "selftest":
        cmd_selftest()
    elif a.cmd == "dry-run-cpu":
        cmd_dry_run_cpu(a.seq, a.n, a.out_dir)


if __name__ == "__main__":
    main()
