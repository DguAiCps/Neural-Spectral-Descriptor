#!/usr/bin/env python3
"""BEVPlace++ official pipeline on split v2 -- part 3: past-only test ledgers
(2026-10-11).

Per split-v2 TEST sequence (46), on the KISS-ICP keyframes of
data/preprocessed_v2_odom (scan_ids; poses_gt only for scoring):
  1. descriptors: official model loading of main.py test mode
     (REIN(), torch.load, load_state_dict on model_best.pth.tar) and official
     main.infer over the official nclt_dataset.InferDataset (PNG /256,
     3 channels; this class has no KITTI-specific split table) on the BEV
     images of scripts/_bevplace_official_bev.py (role test_odom);
  2. ledger: scripts/_causal_bevplace_pr.ledger, imported unchanged (schema
     pr-dump-v1; queries t >= 30, exhaustive top-1 over the prefix
     j <= t - 30, score_cos).  The official retrieval is faiss IndexFlatL2
     on the NetVLAD output, which is L2-normalised inside the model, so the
     L2 nearest neighbour is the cosine nearest neighbour and
     ||a - b||^2 = 2 - 2 cos is a monotone map of the score; the L2 top-1 is
     recomputed on 100 evenly spaced queries per sequence and the agreement
     is stored in meta.l2_check.

Subcommands (run inside ring_baseline):
  dump [--seqs S ...] [--checkpoint P]   one worker subprocess per sequence
  aggregate                              PR summary via _pdv2_c3 (unchanged)
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
    BEV, OFFICIAL_COMMIT, cache_path, seq_name, sha256)
from _causal_bevplace_pr import ledger  # noqa: E402  (torch-free, unchanged)
from _pdv2_pilot import TEST  # noqa: E402
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402

RESULTS = REPO / "results/pdv2_bevplace_official"
OUT = RESULTS / "odom"
SELECTION = RESULTS / "train_selection.json"
METHOD = "bevplace"


def _official_model(checkpoint: Path | None, threads: int, batch: int, device: str):
    """main.py test-mode model set-up (main.py:239-269)."""
    from _bevplace_official_bev import load_official
    M, KD, ND, REIN_MOD = load_official()
    import torch
    argv = ["--mode", "test", "--threads", str(threads), "--cacheBatchSize", str(batch)]
    if checkpoint is not None:
        argv += ["--load_from", str(checkpoint.parent), "--ckpt", "best"]
    sys.argv = [M.__file__] + argv
    opt = M.get_args()
    M.opt = opt
    M.device = torch.device(device)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    model = REIN_MOD.REIN()
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


def _l2_check(desc, arrays, n=100):
    q, js = arrays["query_idx"], arrays["j_star"]
    pick = np.unique(np.linspace(0, len(q) - 1, min(n, len(q))).round().astype(int))
    agree = 0
    for k in pick:
        t = int(q[k])
        d = np.sqrt(np.sum((desc[:t - MIN_INDEX_GAP + 1] - desc[t]) ** 2, axis=1))
        agree += int(np.argmin(d) == js[k])
    return {"n_checked": int(len(pick)), "l2_top1_equals_j_star": int(agree)}


def encode_and_dump(M, ND, sensor, q, out_path, checkpoint, limit=0, extra_meta=None):
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
    norms = np.linalg.norm(desc, axis=1)
    t0 = time.time()
    arrays = ledger(desc, poses_gt[:, :3, 3])
    t_led = time.time() - t0
    l2 = _l2_check(desc, arrays)
    bev_meta = json.loads((BEV / "test_odom" / name / "meta.json").read_text())
    meta = {
        "schema": "pr-dump-v1",
        "method": METHOD,
        "variant": "official BEVPlace2 recipe retrained on split-v2 train",
        "sensor": sensor, "sequence": q,
        "min_index_gap": MIN_INDEX_GAP,
        "n_frames": int(len(desc)),
        "n_queries": int(len(arrays["query_idx"])),
        "primary_score": "score_cos",
        "scores": {"score_cos": "8192D NetVLAD descriptor cosine of the exhaustive "
                                "prefix top-1; native similarity, higher = accept, "
                                "not negated (prereg §1.5 BEVPlace++ row). Official "
                                "retrieval is faiss IndexFlatL2 on the same unit "
                                "vectors: same top-1, L2^2 = 2 - 2 cos"},
        "descriptor_norm": {"min": float(norms.min()), "max": float(norms.max())},
        "l2_check": l2,
        "official": {"repo": "https://github.com/zjuluolun/BEVPlace2",
                     "commit": OFFICIAL_COMMIT,
                     "code": "REIN.py (model), main.py infer/test-mode loading, "
                             "nclt_dataset.InferDataset, datasets/gen_bev_images.py "
                             "getBEV (BEV images)"},
        "artifacts": {"checkpoint": str(checkpoint) if checkpoint else None,
                      "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
                      "selection": str(SELECTION.relative_to(REPO))},
        "caches": {"operating_cache": str(cp.relative_to(REPO)),
                   "operating_cache_sha256": bev_meta["cache_sha256"],
                   "bev_dir": str((BEV / "test_odom" / name).relative_to(REPO)),
                   "bev_generator": bev_meta["generator"],
                   "positions": "poses_gt (scoring only)"},
        "encoder": {"img_size": 201, "range_m": 40.0, "cell_m": 0.4,
                    "image": "point density (official)", "descriptor_dim": 8192,
                    "l2_normalized": "inside NetVLAD (official)",
                    "infer_batch": M.opt.cacheBatchSize},
        "protocol": ("causal all-frames PR ledger (Track C, bevplace): per-frame "
                     "8192D descriptors computed once per sequence (each depends "
                     "only on its own scan), then per-query gemv cosine search "
                     "restricted to prefix rows j <= t - 30 (exhaustive top-1; "
                     "scripts/_causal_bevplace_pr.ledger, unchanged); KISS-ICP "
                     "keyframes, GT poses for scoring only"),
        "timing": {"encode_s": round(t_enc, 1),
                   "encode_ms_per_kf": round(t_enc / max(len(desc), 1) * 1e3, 2),
                   "ledger_s": round(t_led, 1)},
    }
    meta.update(extra_meta or {})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    print(f"[{sensor}/{q}] n_queries={len(arrays['query_idx'])} "
          f"l2-check {l2['l2_top1_equals_j_star']}/{l2['n_checked']} "
          f"({t_enc:.0f} s encode, {t_led:.0f} s ledger) -> {out_path}", flush=True)
    return out_path


def _default_checkpoint() -> Path:
    sel = json.loads(SELECTION.read_text())
    assert sel.get("finished"), f"{SELECTION}: training not finished"
    ck = Path(sel["best"]["checkpoint"])
    assert sha256(ck) == sel["best"]["sha256"], "checkpoint changed after selection"
    return ck


def cmd_dump(seqs, checkpoint, threads, batch, worker, overwrite):
    checkpoint = checkpoint or _default_checkpoint()
    todo = seqs or [f"{s}/{q}" for s, q in TEST]
    unknown = sorted(set(todo) - {f"{s}/{q}" for s, q in TEST})
    assert not unknown, f"not split-v2 test sequences: {unknown}"
    if worker:
        assert len(todo) == 1
        sensor, q = todo[0].split("/")
        M, ND, _ = _official_model(checkpoint, threads, batch, "cuda")
        encode_and_dump(M, ND, sensor, q, OUT / f"dump_{METHOD}_{sensor}_{q}.npz",
                        checkpoint)
        return
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
    c3.BASELINES = {METHOD: "score_cos"}
    c3.cmd_aggregate_baselines("odom")
    p = OUT / "pr_summary_baselines.json"
    s = json.loads(p.read_text())
    s["_meta"].pop("seqot", None)   # hard-coded note of the reused aggregator
    s["_meta"]["bevplace"] = ("official BEVPlace2 recipe retrained on split-v2 train "
                              "(scripts/_bevplace_official_train.py), model_best by "
                              "v2-val family-equal R@1; " + str(SELECTION.relative_to(REPO)))
    p.write_text(json.dumps(s, indent=2))


def cmd_dry_run_cpu(key, n, out_dir):
    """CPU-only: official model (ImageNet ResNet34, untrained NetVLAD -- no
    checkpoint) on the first n keyframes; the ledger is compared with the
    SeqOT ledger of the same sequence in results/pdv2_c3/odom (same keyframes
    and GT positions -> identical query_idx / min_past_gt_dist /
    n_candidates on the shared rows)."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.Tensor.cuda = lambda self, *a, **k: self   # REIN.py hard-codes .cuda()
    M, ND, model = _official_model(None, 2, 8, "cpu")
    model.to = lambda *a, **k: model                  # main.infer calls model.to('cuda')
    sensor, q = key.split("/")
    path = encode_and_dump(M, ND, sensor, q,
                           Path(out_dir) / f"dump_{METHOD}_{sensor}_{q}.npz", None,
                           limit=n, extra_meta={"DRY_RUN": "CPU, untrained network, "
                                                "format check only"})
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
    rep["score_cos_dtype"] = [str(z["score_cos"].dtype), str(ref["score_cos"].dtype)]
    rep["l2_check"] = zm["l2_check"]
    rep["descriptor_norm"] = zm["descriptor_norm"]
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
    p = sub.add_parser("dump")
    p.add_argument("--seqs", nargs="+", default=None)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="default: best checkpoint of train_selection.json")
    p.add_argument("--threads", type=int, default=24, help="official --threads")
    p.add_argument("--batch", type=int, default=128, help="official --cacheBatchSize")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    sub.add_parser("aggregate")
    p = sub.add_parser("dry-run-cpu")
    p.add_argument("--seq", required=True)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "dump":
        cmd_dump(a.seqs, a.checkpoint, a.threads, a.batch, a.worker, a.overwrite)
    elif a.cmd == "aggregate":
        cmd_aggregate()
    elif a.cmd == "dry-run-cpu":
        cmd_dry_run_cpu(a.seq, a.n, a.out_dir)


if __name__ == "__main__":
    main()
