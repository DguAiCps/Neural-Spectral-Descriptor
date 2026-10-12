#!/usr/bin/env python3
"""ImLPR official pipeline on split v2 -- part 3: past-only test ledgers
(2026-10-11).

Per split-v2 TEST sequence (46), on the KISS-ICP keyframes of
data/preprocessed_v2_odom (scan_ids; poses_gt only for scoring):
  1. RIVs: scripts/_imlpr_official_riv.py, role test_odom (VLP-16 / HDL-32E
     submaps from the keyframe and the scans of the previous 5 s, placed with
     the KISS-ICP poses; no GT and no future scan is used);
  2. descriptors: official model loading of eval/evaluate.py:_load_model
     (ImLPR(params.model_params), 'module.' stripped, load_state_dict) and the
     official eval input (pad 28, rows -> 126 INTER_LINEAR, /255) and forward
     (`_, desc = model(x)`; SALAD output, L2-normalised inside the model);
  3. ledger: scripts/_causal_bevplace_pr.ledger, imported unchanged (schema
     pr-dump-v1; queries t >= 30, exhaustive top-1 over the prefix
     j <= t - 30, score_cos).  The official retrieval is a Euclidean KD-tree
     (evaluate.py:253) on unit vectors, so the L2 nearest neighbour is the
     cosine nearest neighbour; the L2 top-1 is recomputed on 100 queries per
     sequence and stored in meta.l2_check.

Subcommands (run inside ring_baseline):
  dump [--seqs S ...] [--checkpoint P]   one worker subprocess per sequence
  aggregate                              PR summary via _pdv2_c3 (unchanged)
  dry-run-cpu --seq S --n N --out-dir D  CPU-only format check (DINOv2 init)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

from _causal_bevplace_pr import ledger  # noqa: E402  (torch-free, unchanged)
from _imlpr_official_riv import (  # noqa: E402
    OFFICIAL_COMMIT, WORK, role_dir, sha256)
from _pdv2_pilot import TEST  # noqa: E402
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402

SEED = int(os.environ.get("SEED", "1"))      # seed 1: original locations
SFX = "" if SEED == 1 else f"_s{SEED}"
RESULTS = REPO / f"results/pdv2_imlpr_official{SFX}"
OUT = RESULTS / "odom"
SELECTION = RESULTS / "train_selection.json"
DESC = WORK / f"desc{SFX}"
METHOD = "imlpr"


def _official_model(checkpoint: Path | None, device: str):
    """eval/evaluate.py:_load_model (lines 393-417) on one device."""
    import torch
    from _imlpr_official_train import ensure_model_config, load_official
    off = load_official()
    mp = off.MU.ModelParams(str(ensure_model_config()))   # paper (m, l, e), see train module
    model = off.MI.ImLPR(mp).to(device)
    if checkpoint is not None:
        state = torch.load(str(checkpoint), map_location="cpu")
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        res = model.load_state_dict(state, strict=False)
        assert not res.missing_keys and not res.unexpected_keys, res
        print(f"[info] loaded weights: {checkpoint}", flush=True)
    return model.eval(), mp


def _l2_check(desc, arrays, n=100):
    q, js = arrays["query_idx"], arrays["j_star"]
    pick = np.unique(np.linspace(0, len(q) - 1, min(n, len(q))).round().astype(int))
    agree = 0
    for k in pick:
        t = int(q[k])
        d = np.sqrt(np.sum((desc[:t - MIN_INDEX_GAP + 1] - desc[t]) ** 2, axis=1))
        agree += int(np.argmin(d) == js[k])
    return {"n_checked": int(len(pick)), "l2_top1_equals_j_star": int(agree)}


def encode_and_dump(model, mp, sensor, q, out_path, checkpoint, device, batch, workers,
                    riv_dir=None, limit=0, extra_meta=None):
    from _imlpr_official_train import encode
    d = riv_dir or role_dir("test_odom", sensor, q)
    idx = np.load(role_dir("test_odom", sensor, q) / "index.npz")
    sids, poses_gt = idx["scan_ids"], idx["poses_gt"].astype(np.float64)
    if limit:
        sids, poses_gt = sids[:limit], poses_gt[:limit]
    paths = [str(Path(d) / f"{p:06d}.npy") for p in range(len(sids))]
    assert all(Path(p).exists() for p in paths), f"{sensor}/{q}: RIVs incomplete"
    print(f"[{sensor}/{q}] encoding {len(paths)} keyframes", flush=True)
    t0 = time.time()
    desc = encode(model, device, paths, batch, workers)
    t_enc = time.time() - t0
    norms = np.linalg.norm(desc, axis=1)
    if not limit:
        DESC.mkdir(parents=True, exist_ok=True)
        np.savez(DESC / f"desc_{sensor}_{q}.npz", desc=desc, scan_ids=sids)
    t0 = time.time()
    arrays = ledger(desc, poses_gt[:, :3, 3])
    t_led = time.time() - t0
    l2 = _l2_check(desc, arrays)
    riv_meta = json.loads((role_dir("test_odom", sensor, q) / "meta.json").read_text())
    meta = {
        "schema": "pr-dump-v1",
        "method": METHOD,
        "variant": "official ImLPR code/recipe retrained on split-v2 train",
        "sensor": sensor, "sequence": q,
        "min_index_gap": MIN_INDEX_GAP,
        "n_frames": int(len(desc)),
        "n_queries": int(len(arrays["query_idx"])),
        "primary_score": "score_cos",
        "scores": {"score_cos": "8448D SALAD descriptor cosine of the exhaustive prefix "
                                "top-1; native similarity, higher = accept, not negated. "
                                "Official retrieval is a Euclidean KD-tree on the same unit "
                                "vectors: same top-1, L2^2 = 2 - 2 cos"},
        "descriptor_norm": {"min": float(norms.min()), "max": float(norms.max())},
        "l2_check": l2,
        "official": {"repo": "https://github.com/minwoo0611/ImLPR", "commit": OFFICIAL_COMMIT,
                     "code": "models/ImLPR.py, eval/evaluate.py (_load_model, "
                             "_get_latent_vectors input), preprocess/generate_riv_images.py"},
        "artifacts": {"checkpoint": str(checkpoint) if checkpoint else None,
                      "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
                      "selection": str(SELECTION.relative_to(REPO))},
        "caches": {"operating_cache": riv_meta["cache"],
                   "operating_cache_sha256": riv_meta["cache_sha256"],
                   "riv_dir": str(Path(d).relative_to(REPO)) if Path(d).is_relative_to(REPO)
                   else str(d),
                   "sensor_settings": riv_meta["sensor_settings"],
                   "accumulation": riv_meta["accumulation"],
                   "positions": "poses_gt (scoring only)"},
        "encoder": {"model_params": vars(mp), "descriptor_dim": int(desc.shape[1]),
                    "input": "RIV (Hf,1022,3) -> pad 28 -> 126 rows INTER_LINEAR -> /255",
                    "batch": batch},
        "protocol": ("causal all-frames PR ledger (Track C, imlpr): per-keyframe 8448D "
                     "descriptors computed once per sequence (each depends only on its "
                     "own scan and the scans before it), then per-query gemv cosine "
                     "search restricted to prefix rows j <= t - 30 (exhaustive top-1; "
                     "scripts/_causal_bevplace_pr.ledger, unchanged); KISS-ICP keyframes "
                     "and poses, GT poses for scoring only"),
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


def cmd_dump(seqs, checkpoint, batch, workers, worker, overwrite):
    checkpoint = checkpoint or _default_checkpoint()
    todo = seqs or [f"{s}/{q}" for s, q in TEST]
    unknown = sorted(set(todo) - {f"{s}/{q}" for s, q in TEST})
    assert not unknown, f"not split-v2 test sequences: {unknown}"
    if worker:
        assert len(todo) == 1
        sensor, q = todo[0].split("/")
        model, mp = _official_model(checkpoint, "cuda")
        encode_and_dump(model, mp, sensor, q, OUT / f"dump_{METHOD}_{sensor}_{q}.npz",
                        checkpoint, "cuda", batch, workers)
        return
    for key in todo:   # one sequence per subprocess (memory discipline)
        sensor, q = key.split("/")
        if (OUT / f"dump_{METHOD}_{sensor}_{q}.npz").exists() and not overwrite:
            print(f"[skip] {key}", flush=True)
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "dump", "--worker",
               "--seqs", key, "--checkpoint", str(checkpoint), "--batch", str(batch),
               "--workers", str(workers)]
        rc = subprocess.run(cmd).returncode
        if rc:
            print(f"WORKER_FAILED {key} rc={rc}", flush=True)
            sys.exit(rc)
    print("IMLPR_OFFICIAL_DUMP_DONE", flush=True)


def cmd_aggregate():
    import _pdv2_c3 as c3
    c3.OUT_ROOT = RESULTS
    c3.BASELINES = {METHOD: "score_cos"}
    c3.cmd_aggregate_baselines("odom")
    p = OUT / "pr_summary_baselines.json"
    s = json.loads(p.read_text())
    s["_meta"].pop("seqot", None)   # hard-coded note of the reused aggregator
    s["_meta"]["imlpr"] = ("official ImLPR code/recipe retrained on split-v2 train "
                           "(scripts/_imlpr_official_train.py), `_best` by v2-val "
                           "family-equal R@1; " + str(SELECTION.relative_to(REPO)))
    p.write_text(json.dumps(s, indent=2))


def cmd_dry_run_cpu(key, n, out_dir):
    """CPU only: RIVs of the first n keyframes (written to out_dir), official
    model with DINOv2 weights and untrained adapters/SALAD (no checkpoint),
    ledger compared with the SeqOT ledger of the same sequence in
    results/pdv2_c3/odom (same keyframes and GT positions -> identical
    query_idx / min_past_gt_dist / n_candidates on the shared rows)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(4)
    from _imlpr_official_riv import cmd_gen
    sensor, q = key.split("/")
    rd = Path(out_dir) / "riv"
    t0 = time.time()
    cmd_gen("test_odom", key, 0, 1, limit=n, out_dir=rd)
    t_riv = time.time() - t0
    model, mp = _official_model(None, "cpu")
    path = encode_and_dump(model, mp, sensor, q,
                           Path(out_dir) / f"dump_{METHOD}_{sensor}_{q}.npz", None, "cpu",
                           4, 0, riv_dir=rd, limit=n,
                           extra_meta={"DRY_RUN": "CPU, DINOv2 + untrained adapters/SALAD, "
                                                  "format check only"})
    z = np.load(path)
    ref = np.load(REPO / f"results/pdv2_c3/odom/dump_seqot_{sensor}_{q}.npz")
    rep = {"ours": {k: [str(z[k].dtype), list(z[k].shape)] for k in z.files},
           "ref_seqot": {k: [str(ref[k].dtype), list(ref[k].shape)] for k in ref.files},
           "riv_s_per_frame": round(t_riv / n, 2)}
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
    rep["accumulation"] = zm["caches"]["accumulation"]
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
                   help="default: `_best` checkpoint of train_selection.json")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    sub.add_parser("aggregate")
    p = sub.add_parser("dry-run-cpu")
    p.add_argument("--seq", required=True)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "dump":
        cmd_dump(a.seqs, a.checkpoint, a.batch, a.workers, a.worker, a.overwrite)
    elif a.cmd == "aggregate":
        cmd_aggregate()
    elif a.cmd == "dry-run-cpu":
        cmd_dry_run_cpu(a.seq, a.n, a.out_dir)


if __name__ == "__main__":
    main()
