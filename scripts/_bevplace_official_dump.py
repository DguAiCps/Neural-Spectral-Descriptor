#!/usr/bin/env python3
"""BEVPlace++ on split v2 -- part 3: past-only test ledgers (2026-10-11).

Per split-v2 TEST sequence (46), on the KISS-ICP keyframes of
data/preprocessed_v2_odom (scan_ids; poses_gt only for scoring):
  1. 8192-D descriptors: official test-mode loading (main.py:255-269:
     REIN, torch.load, strict load_state_dict) of model_best with the paper
     network overrides (_bevplace_official_bev.build_model), official
     main.infer over nclt_dataset.InferDataset (PNG /256, 3 channels) on the
     200x200 BEV images of role test_odom.
  2. PAPER OVERRIDE P8 (T-RO Sec. III-C: "In practice, we use PCA to reduce
     all the global descriptors into 512-dim"; main.py:177-180 `testPCA` is
     an empty TODO and main.py:58/kitti_dataset.py:55 search the raw 8192-D
     vectors): PCA to 512 dimensions, no whitening, fitted once on the
     descriptors of all split-v2 TRAIN keyframes (un-augmented stored
     images) with the selected checkpoint (`pca` subcommand).  The paper
     does not say what the PCA is fitted on; the training set is the only
     choice that touches neither validation nor test data.
  3. ledger (schema pr-dump-v1): queries t >= 30, candidates j <= t - 30,
     top-1 by L2 distance on the PCA-512 vectors (paper Eq. 3, arg min
     ||V_q - V_i||_2; faiss IndexFlatL2 as kitti_dataset.py:55), the index
     holding only the prefix rows.  score_negl2 = -L2 of the emitted top-1
     (native distance, negated: higher = accept) is the primary score;
     score_cos8192 = cosine of the 8192-D NetVLAD vectors of the same pair
     (the statistic named in the causal pre-registration §1.5, kept for
     comparison).

Subcommands (run inside ring_baseline):
  pca  [--checkpoint P]                  fit PCA-512 on the train descriptors
  dump [--seqs S ...] [--checkpoint P]   one worker subprocess per sequence
  aggregate                              PR summary via _pdv2_c3 (unchanged)
  selftest                               CPU: L2 prefix ledger checks
  dry-run-cpu --seq S --n N --out-dir D  CPU-only format check, untrained net
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

from _bevplace_official_bev import (  # noqa: E402
    BEV, IMG, OFFICIAL_COMMIT, cache_path, seq_name, sha256)
from _pdv2_pilot import TEST, TRAIN  # noqa: E402
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402

RESULTS = REPO / "results/pdv2_bevplace_official"
OUT = RESULTS / "odom"
SELECTION = RESULTS / "train_selection.json"
PCA_FILE = RESULTS / "pca512_train.npz"
PCA_DIM = 512
METHOD = "bevplace"


def _official_model(checkpoint: Path | None, threads: int, batch: int, device: str):
    """main.py test-mode model set-up (main.py:239-269) with the paper network."""
    from _bevplace_official_bev import build_model, load_official
    M, KD, ND, REIN_MOD = load_official()
    import torch
    argv = ["--mode", "test", "--threads", str(threads), "--cacheBatchSize", str(batch),
            "--seed", "1"]
    if checkpoint is not None:
        argv += ["--load_from", str(checkpoint.parent), "--ckpt", "best"]
    sys.argv = [M.__file__] + argv
    opt = M.get_args()
    M.opt = opt
    M.device = torch.device(device)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    model = build_model(REIN_MOD)
    if device == "cuda":
        model = model.cuda()
    if checkpoint is not None:
        ck = torch.load(checkpoint, map_location=lambda storage, loc: storage,
                        weights_only=False)   # own file; torch>=2.6 default is True
        model.load_state_dict(ck["state_dict"])
        model = model.to(M.device)
        print(f"=> loaded checkpoint '{checkpoint}' (epoch {ck['epoch']})", flush=True)
    M.model = model
    return M, ND, model


def ledger_l2(z: np.ndarray, positions: np.ndarray, desc_full: np.ndarray) -> dict:
    """Prefix L2 top-1 ledger (same rows / query set / min_past rule as
    scripts/_causal_bevplace_pr.ledger).  The faiss index holds rows
    [0, t - 30] when query t is searched, so later rows are never read and
    each row is identical under sequence truncation."""
    import faiss
    n = len(z)
    assert len(positions) == n == len(desc_full)
    z = np.ascontiguousarray(z, dtype=np.float32)
    min_past = np.full(n, np.nan)
    for t in range(MIN_INDEX_GAP, n):
        d = np.linalg.norm(positions[:t - MIN_INDEX_GAP + 1] - positions[t], axis=1)
        min_past[t] = d.min()
    queries = np.arange(MIN_INDEX_GAP, n, dtype=np.int64)
    j_star = np.full(len(queries), -1, dtype=np.int64)
    s_l2 = np.full(len(queries), np.nan)
    s_cos = np.full(len(queries), np.nan)
    top1_geo = np.full(len(queries), np.nan)
    n_cand = np.zeros(len(queries), dtype=np.int64)
    index = faiss.IndexFlatL2(z.shape[1])
    for qi, t in enumerate(queries):
        t = int(t)
        n_prefix = t - MIN_INDEX_GAP + 1
        if index.ntotal < n_prefix:
            index.add(z[index.ntotal:n_prefix])
        assert index.ntotal == n_prefix
        dist2, ind = index.search(z[t:t + 1], 1)   # nq = 1: exact per-row L2
        j = int(ind[0, 0])
        assert 0 <= j <= t - MIN_INDEX_GAP, "candidate beyond the causal prefix"
        j_star[qi] = j
        s_l2[qi] = -float(np.sqrt(max(float(dist2[0, 0]), 0.0)))
        s_cos[qi] = float(desc_full[j] @ desc_full[t])
        top1_geo[qi] = float(np.linalg.norm(positions[j] - positions[t]))
        n_cand[qi] = n_prefix
        if qi and qi % 2000 == 0:
            print(f"  ledger queries {qi}/{len(queries)}", flush=True)
    return dict(query_idx=queries, j_star=j_star, score_negl2=s_l2, score_cos8192=s_cos,
                top1_geo_dist=top1_geo, min_past_gt_dist=min_past[queries],
                n_candidates=n_cand)


def _pca_apply(desc, pca):
    return ((desc - pca["mean"]) @ pca["components"].T).astype(np.float32)


def encode_and_dump(M, ND, sensor, q, out_path, checkpoint, pca, limit=0, extra_meta=None):
    name = seq_name(sensor, q)
    cp = cache_path("test_odom", sensor, q)
    with np.load(cp) as z:
        sids = z["scan_ids"].astype(np.int64)
        poses_gt = z["poses_gt"].astype(np.float64)
    ds = ND.InferDataset(seq=name, dataset_path=str(BEV / "test_odom") + "/")
    assert [Path(p).name for p in ds.imgs_path] == [f"{s:06d}.png" for s in sids], \
        f"{name}: BEV images do not match the cache keyframes"
    assert np.array_equal(ds.poses, poses_gt[:, :3, :].reshape(-1, 12)), name
    if limit:
        ds.imgs_path, ds.poses = ds.imgs_path[:limit], ds.poses[:limit]
        sids, poses_gt = sids[:limit], poses_gt[:limit]
    print(f"[{sensor}/{q}] encoding {len(ds)} keyframes", flush=True)
    t0 = time.time()
    desc = M.infer(ds).astype(np.float32)           # official inference
    t_enc = time.time() - t0
    zq = _pca_apply(desc, pca)                       # P8
    t0 = time.time()
    arrays = ledger_l2(zq, poses_gt[:, :3, 3], desc)
    t_led = time.time() - t0
    bev_meta = json.loads((BEV / "test_odom" / name / "meta.json").read_text())
    norms = np.linalg.norm(desc, axis=1)
    meta = {
        "schema": "pr-dump-v1",
        "method": METHOD,
        "variant": "BEVPlace2 code with BEVPlace++ paper values, retrained on split-v2 train",
        "sensor": sensor, "sequence": q,
        "min_index_gap": MIN_INDEX_GAP,
        "n_frames": int(len(desc)),
        "n_queries": int(len(arrays["query_idx"])),
        "primary_score": "score_negl2",
        "scores": {"score_negl2": "-||V_q - V_j*||_2 on the PCA-512 descriptors of the "
                                  "emitted prefix top-1 (paper retrieval distance, "
                                  "negated: higher = accept)",
                   "score_cos8192": "cosine of the 8192-D NetVLAD vectors of the same "
                                    "pair (prereg §1.5 statistic, comparison only)"},
        "retrieval": "exhaustive L2 top-1 over prefix rows j <= t - 30, faiss "
                     "IndexFlatL2 holding only the prefix",
        "descriptor_norm_8192": {"min": float(norms.min()), "max": float(norms.max())},
        "pca": {"dim": int(zq.shape[1]), "whiten": False,
                "file": pca.get("file"), "sha256": pca.get("sha256"),
                "fit_on": pca.get("fit_on"),
                "explained_variance_ratio_sum": pca.get("evr_sum")},
        "official": {"repo": "https://github.com/zjuluolun/BEVPlace2",
                     "commit": OFFICIAL_COMMIT,
                     "code": "REIN.py (model), main.py infer/test-mode loading, "
                             "nclt_dataset.InferDataset, datasets/gen_bev_images.py "
                             "getBEV (BEV images)",
                     "paper_overrides": "200x200 image, bilinear up-sampling, NetVLAD "
                                        "bias, PCA-512 + L2 retrieval (see "
                                        "logs/baseline_official_20261011/bevplace.md)"},
        "artifacts": {"checkpoint": str(checkpoint) if checkpoint else None,
                      "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
                      "selection": str(SELECTION.relative_to(REPO))},
        "caches": {"operating_cache": str(cp.relative_to(REPO)),
                   "operating_cache_sha256": bev_meta["cache_sha256"],
                   "bev_dir": str((BEV / "test_odom" / name).relative_to(REPO)),
                   "bev_generator": bev_meta["generator"],
                   "positions": "poses_gt (scoring only)"},
        "encoder": {"img_size": IMG, "range_m": 40.0, "cell_m": 0.4,
                    "image": "point density (official generator)",
                    "descriptor_dim": 8192, "retrieval_dim": int(zq.shape[1]),
                    "infer_batch": M.opt.cacheBatchSize},
        "protocol": ("causal all-frames PR ledger: per-frame descriptors computed "
                     "once per sequence (each depends only on its own scan; PCA "
                     "fitted on train), per-query exhaustive L2 top-1 over prefix "
                     "rows j <= t - 30; KISS-ICP keyframes, GT poses for scoring only"),
        "timing": {"encode_s": round(t_enc, 1),
                   "encode_ms_per_kf": round(t_enc / max(len(desc), 1) * 1e3, 2),
                   "ledger_s": round(t_led, 1)},
    }
    meta.update(extra_meta or {})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    print(f"[{sensor}/{q}] n_queries={len(arrays['query_idx'])} "
          f"({t_enc:.0f} s encode, {t_led:.0f} s ledger) -> {out_path}", flush=True)
    return out_path


def _default_checkpoint() -> Path:
    sel = json.loads(SELECTION.read_text())
    assert sel.get("finished"), f"{SELECTION}: training not finished"
    ck = Path(sel["best"]["checkpoint"])
    assert sha256(ck) == sel["best"]["sha256"], "checkpoint changed after selection"
    return ck


def fit_pca(desc, n_components):
    from sklearn.decomposition import PCA
    p = PCA(n_components=n_components, whiten=False, svd_solver="covariance_eigh")
    p.fit(desc)
    return {"mean": p.mean_.astype(np.float32), "components": p.components_.astype(np.float32),
            "evr_sum": float(p.explained_variance_ratio_.sum())}


def cmd_pca(checkpoint, threads, batch):
    checkpoint = checkpoint or _default_checkpoint()
    M, ND, _ = _official_model(checkpoint, threads, batch, "cuda")
    descs, counts = [], {}
    for s, q in TRAIN:
        ds = ND.InferDataset(seq=seq_name(s, q), dataset_path=str(BEV / "train") + "/")
        descs.append(M.infer(ds).astype(np.float32))
        counts[f"{s}/{q}"] = len(ds)
        print(f"  train descriptors {s}/{q}: {len(ds)}", flush=True)
    X = np.concatenate(descs)
    t0 = time.time()
    pca = fit_pca(X, PCA_DIM)
    PCA_FILE.parent.mkdir(parents=True, exist_ok=True)
    meta = {"fit_on": "split-v2 TRAIN keyframes (stored, un-augmented BEV images)",
            "n_train": int(len(X)), "per_sequence": counts, "dim": PCA_DIM, "whiten": False,
            "solver": "sklearn PCA covariance_eigh", "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint), "evr_sum": pca["evr_sum"],
            "fit_s": round(time.time() - t0, 1)}
    np.savez(PCA_FILE, mean=pca["mean"], components=pca["components"], meta=json.dumps(meta))
    print(f"OK {PCA_FILE}: {len(X)} train descriptors, explained variance "
          f"{pca['evr_sum']:.4f}", flush=True)


def _load_pca(checkpoint):
    z = np.load(PCA_FILE)
    meta = json.loads(str(z["meta"]))
    assert meta["checkpoint_sha256"] == sha256(checkpoint), "PCA fitted on another checkpoint"
    return {"mean": z["mean"], "components": z["components"],
            "file": str(PCA_FILE.relative_to(REPO)), "sha256": sha256(PCA_FILE),
            "fit_on": meta["fit_on"], "evr_sum": meta["evr_sum"]}


def cmd_dump(seqs, checkpoint, threads, batch, worker, overwrite):
    checkpoint = checkpoint or _default_checkpoint()
    todo = seqs or [f"{s}/{q}" for s, q in TEST]
    unknown = sorted(set(todo) - {f"{s}/{q}" for s, q in TEST})
    assert not unknown, f"not split-v2 test sequences: {unknown}"
    if worker:
        assert len(todo) == 1
        sensor, q = todo[0].split("/")
        pca = _load_pca(checkpoint)
        M, ND, _ = _official_model(checkpoint, threads, batch, "cuda")
        encode_and_dump(M, ND, sensor, q, OUT / f"dump_{METHOD}_{sensor}_{q}.npz",
                        checkpoint, pca)
        return
    _load_pca(checkpoint)   # fail early
    for key in todo:   # one sequence per subprocess (memory discipline)
        sensor, q = key.split("/")
        if (OUT / f"dump_{METHOD}_{sensor}_{q}.npz").exists() and not overwrite:
            print(f"[skip] {key}", flush=True)
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "dump", "--worker",
               "--seqs", key, "--checkpoint", str(checkpoint),
               "--threads", str(threads), "--batch", str(batch)]
        rc = subprocess.run(cmd).returncode
        if rc:
            print(f"WORKER_FAILED {key} rc={rc}", flush=True)
            sys.exit(rc)
    print("BEVPLACE_OFFICIAL_DUMP_DONE", flush=True)


def cmd_aggregate():
    import _pdv2_c3 as c3
    c3.OUT_ROOT = RESULTS
    c3.BASELINES = {METHOD: "score_negl2"}
    c3.cmd_aggregate_baselines("odom")
    p = OUT / "pr_summary_baselines.json"
    s = json.loads(p.read_text())
    s["_meta"].pop("seqot", None)   # hard-coded note of the reused aggregator
    s["_meta"]["bevplace"] = ("BEVPlace2 code with BEVPlace++ paper values, retrained on "
                              "split-v2 train (scripts/_bevplace_official_train.py), "
                              "model_best by v2-val family-equal R@1, PCA-512 L2 "
                              "retrieval; " + str(SELECTION.relative_to(REPO)))
    p.write_text(json.dumps(s, indent=2))


def cmd_selftest():
    """CPU, no torch: L2 prefix ledger == brute-force numpy reference, causal,
    prefix-invariant; query set / min_past / n_candidates identical to
    scripts/_causal_bevplace_pr.ledger."""
    from _causal_bevplace_pr import ledger as cos_ledger
    rng = np.random.default_rng(0)
    n = 150
    z = rng.standard_normal((n, 512)).astype(np.float32)
    full = rng.standard_normal((n, 64)).astype(np.float32)
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    positions = rng.uniform(0.0, 100.0, size=(n, 3))
    a = ledger_l2(z, positions, full)
    q = a["query_idx"]
    assert (a["j_star"] <= q - MIN_INDEX_GAP).all()
    for qi, t in enumerate(q):
        t = int(t)
        d = np.sum((z[:t - MIN_INDEX_GAP + 1].astype(np.float64) - z[t]) ** 2, axis=1)
        assert a["j_star"][qi] == int(np.argmin(d)), t
        assert abs(-np.sqrt(d.min()) - a["score_negl2"][qi]) < 1e-3, t
    ref = cos_ledger(full, positions)
    for k in ("query_idx", "min_past_gt_dist", "n_candidates"):
        assert np.array_equal(a[k], ref[k]), k
    for t in (30, 77, 120, n - 1):
        b = ledger_l2(z[:t + 1], positions[:t + 1], full[:t + 1])
        qi = t - MIN_INDEX_GAP
        for k in ("j_star", "score_negl2", "score_cos8192", "top1_geo_dist",
                  "min_past_gt_dist", "n_candidates"):
            assert a[k][qi] == b[k][qi], (t, k)
    print("DUMP_SELFTEST_OK: L2 prefix ledger matches brute force, is causal and "
          "prefix-invariant; query rows match the cosine ledger", flush=True)


def cmd_dry_run_cpu(key, n, out_dir):
    """CPU-only format check: paper network (ImageNet ResNet34, untrained
    NetVLAD, no checkpoint) on the first n keyframes; PCA fitted on these n
    descriptors themselves (format only).  The ledger rows are compared with
    the SeqOT ledger of the same sequence in results/pdv2_c3/odom."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.Tensor.cuda = lambda self, *a, **k: self   # REIN.py hard-codes .cuda()
    M, ND, model = _official_model(None, 2, 8, "cpu")
    model.to = lambda *a, **k: model                  # main.infer calls model.to('cuda')
    sensor, q = key.split("/")
    ds = ND.InferDataset(seq=seq_name(sensor, q), dataset_path=str(BEV / "test_odom") + "/")
    ds.imgs_path, ds.poses = ds.imgs_path[:n], ds.poses[:n]
    pca = fit_pca(M.infer(ds).astype(np.float32), min(PCA_DIM, n))
    pca.update(file="DRY_RUN", sha256=None, fit_on="DRY_RUN: the n test descriptors")
    path = encode_and_dump(M, ND, sensor, q,
                           Path(out_dir) / f"dump_{METHOD}_{sensor}_{q}.npz", None, pca,
                           limit=n, extra_meta={"DRY_RUN": "CPU, untrained network, "
                                                "format check only"})
    z = np.load(path)
    ref = np.load(REPO / f"results/pdv2_c3/odom/dump_seqot_{sensor}_{q}.npz")
    m = len(z["query_idx"])
    zm, rm = json.loads(str(z["meta"])), json.loads(str(ref["meta"]))
    shared = ("query_idx", "j_star", "top1_geo_dist", "min_past_gt_dist", "n_candidates")
    rep = {"ours": {k: [str(z[k].dtype), list(z[k].shape)] for k in z.files if k != "meta"},
           "ref_seqot": {k: [str(ref[k].dtype), list(ref[k].shape)]
                         for k in ref.files if k != "meta"},
           "meta_schema": [zm["schema"], rm["schema"]],
           "primary_score": [zm["primary_score"], rm["primary_score"]],
           "dtypes_match": all(str(z[k].dtype) == str(ref[k].dtype) for k in shared)}
    for k in ("query_idx", "min_past_gt_dist", "n_candidates"):
        rep[f"{k}_equal_on_shared_rows"] = bool(np.array_equal(z[k], ref[k][:m]))
    rep["pca_dim"] = zm["pca"]["dim"]
    ok = rep["dtypes_match"] and zm["schema"] == rm["schema"] == "pr-dump-v1" and all(
        rep[f"{k}_equal_on_shared_rows"] for k in ("query_idx", "min_past_gt_dist",
                                                    "n_candidates"))
    print(json.dumps(rep, indent=1))
    print("DRY_RUN_OK" if ok else "DRY_RUN_MISMATCH", flush=True)
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pca", "dump"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", type=Path, default=None,
                       help="default: best checkpoint of train_selection.json")
        p.add_argument("--threads", type=int, default=24, help="official --threads")
        p.add_argument("--batch", type=int, default=128, help="official --cacheBatchSize")
        if name == "dump":
            p.add_argument("--seqs", nargs="+", default=None)
            p.add_argument("--worker", action="store_true")
            p.add_argument("--overwrite", action="store_true")
    sub.add_parser("aggregate")
    sub.add_parser("selftest")
    p = sub.add_parser("dry-run-cpu")
    p.add_argument("--seq", required=True)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "pca":
        cmd_pca(a.checkpoint, a.threads, a.batch)
    elif a.cmd == "dump":
        cmd_dump(a.seqs, a.checkpoint, a.threads, a.batch, a.worker, a.overwrite)
    elif a.cmd == "aggregate":
        cmd_aggregate()
    elif a.cmd == "selftest":
        cmd_selftest()
    elif a.cmd == "dry-run-cpu":
        cmd_dry_run_cpu(a.seq, a.n, a.out_dir)


if __name__ == "__main__":
    main()
