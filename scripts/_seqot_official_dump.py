#!/usr/bin/env python3
"""SeqOT official pipeline on split v2 -- part 3: past-only test ledgers
(2026-10-11).

Per split-v2 TEST sequence (46), on the KISS-ICP keyframes of
data/preprocessed_v2_odom (scan_ids; poses_gt only for scoring):
  1. descriptors: the selected phase-1 featureExtracter and phase-2 GeM
     (results/pdv2_seqot_official/train_selection.json), official modules,
     eval mode.  Protocol windows (past only): the sub-descriptor of raw scan
     s reads raw scans [s-2, s], the descriptor of a keyframe at raw scan s
     pools the sub-descriptors of raw scans [s-19, s] (the official code
     centres both windows: [s-1, s+1] and [s-10, s+9]; the paper describes
     past windows).  Official fallback: a window that leaves the sequence is
     filled with its last scan.  Raw scans = our loader's scan stream (10 Hz),
     range images from the role test_odom stores of _seqot_official_prep.py.
  2. ledgers (pr-dump-v1; queries t >= 30, exhaustive top-1 over the prefix
     j <= t - 30), two files per sequence:
       dump_seqot_*.npz    score_cos: cosine of the L2-normalised 256-D
                           descriptors (prereg §1.5 SeqOT row),
                           scripts/_causal_bevplace_pr.ledger unchanged;
       dump_seqotl2_*.npz  score_l2 = -Euclidean distance between the raw
                           GeM outputs (official test_gem_prepare.py faiss L2
                           retrieval), scripts/_disco_official_dump.ledger
                           unchanged (score column renamed).
     meta.agreement_cos_l2 = share of queries whose two top-1 agree.
     REPORTED ledger: seqotl2 (official L2; user decision 2026-10-11);
     seqot (cosine) is kept as a record.

Subcommands (inside ring_baseline):
  dump [--seqs S ...] [--worker]   one worker subprocess per sequence
  aggregate                        PR summary via _pdv2_c3 (unchanged)
  dry-run-cpu --seq S --n N --out-dir D   CPU format check, untrained net,
                                   range images computed on the fly
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
sys.dont_write_bytecode = True

from _seqot_official_prep import (  # noqa: E402
    BACK, FOV, GEM_SEQLEN, OFFICIAL_COMMIT, OFFICIAL_REPO, RESULTS, SEED, SEQL, TEST,
    WORK, cache_path, range_image, seq_name, sha256, store_dir)
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402
from _seqot_official_train import (  # noqa: E402  (torch-free at import)
    SEED_RESULTS, SEED_WORK, SELECTION, TRAIN_SEED)

OUT = SEED_RESULTS / "odom"                  # SEED = 1: results/pdv2_seqot_official/odom
DESC = SEED_WORK / "desc" / "test_odom"      # SEED = 1: _external/seqot_official_pdv2/desc/test_odom
METHODS = {"seqot": "score_cos", "seqotl2": "score_l2"}
REPORTED = "seqotl2"   # user decision 2026-10-11: official L2 retrieval


def ledgers(final, positions):
    """(cos arrays, l2 arrays) with the two existing ledger writers."""
    from _causal_bevplace_pr import ledger as ledger_cos      # torch-free, unchanged
    from _disco_official_dump import ledger as ledger_l2      # torch-free, unchanged
    fn = (final / np.linalg.norm(final, axis=1, keepdims=True)).astype(np.float32)
    a_cos = ledger_cos(fn, positions)
    a_l2 = ledger_l2(final, positions)
    a_l2["score_l2"] = a_l2.pop("score_disco")
    return a_cos, a_l2


def _meta(sensor, q, n, arrays, score, extra):
    desc_score = {
        "score_cos": "cosine of the L2-normalised 256-D SeqOT descriptors at the "
                     "exhaustive prefix top-1; higher = accept (prereg §1.5 SeqOT row)",
        "score_l2": "-Euclidean distance between the raw 256-D GeM outputs at the "
                    "exhaustive prefix top-1 (official retrieval metric: "
                    "test/test_gem_prepare.py faiss IndexIVFFlat(nlist=1) L2); "
                    "higher = accept"}[score]
    m = {"schema": "pr-dump-v1", "method": "seqot" if score == "score_cos" else "seqotl2",
         "variant": "official SeqOT code/recipe retrained on split-v2 train",
         "sensor": sensor, "sequence": q, "min_index_gap": MIN_INDEX_GAP,
         "n_frames": int(n), "n_queries": int(len(arrays["query_idx"])),
         "primary_score": score, "scores": {score: desc_score},
         "window": "trailing raw-scan windows ending at the keyframe: sub-descriptor "
                   f"[s-{SEQL - 1}, s], GeM [s-{GEM_SEQLEN - 1}, s]; official fallback "
                   "(window leaving the sequence -> filled with its last scan)",
         "deviation": "official code centres both windows ([s-1, s+1], [s-10, s+9]) and "
                      "training keeps them; the past-only protocol requires trailing windows "
                      "at evaluation (the paper describes m past scans). SeqOT reads up to "
                      f"{BACK} past raw scans behind each keyframe (other methods read the "
                      "keyframe scan only).",
         "official": {"repo": OFFICIAL_REPO, "commit": OFFICIAL_COMMIT,
                      "code": "modules/seqTransformerCat.featureExtracter, modules/gem.GeM, "
                              "tools/utils/utils.range_projection"},
         "fov_up_down_deg": list(FOV[sensor]), "seed": TRAIN_SEED,
         "reported": (score == METHODS[REPORTED]),
         "reported_note": f"{REPORTED} (official L2) is the reported SeqOT ledger; "
                          "seqot (cosine, prereg §1.5) is a record",
         "training": "official code defaults (lr 5e-7 / 99 ep, GeM 5e-6 / 99 ep, 3+3 / 6+6, "
                     "centred windows, code overlap); epochs chosen on v2 VAL, "
                     f"{SELECTION.relative_to(REPO)}"}
    m.update(extra)
    return m


def encode_sequence(fe, gem, dev, src, batch):
    from _seqot_official_train import trailing_final_descriptors
    t0 = time.time()
    final, _, need = trailing_final_descriptors(fe, gem, src, src.kf, dev, batch)
    return final, need, time.time() - t0


def write_dump(sensor, q, final, positions, out_dir, extra):
    t0 = time.time()
    a_cos, a_l2 = ledgers(final, positions)
    t_led = time.time() - t0
    agree = float(np.mean(a_cos["j_star"] == a_l2["j_star"])) if len(a_cos["j_star"]) else float("nan")
    from _disco_official_dump import _l2_check
    kd = _l2_check(final, a_l2)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for method, arrays in (("seqot", a_cos), ("seqotl2", a_l2)):
        score = METHODS[method]
        ex = dict(extra, agreement_cos_l2=agree, ledger_s=round(t_led, 1))
        if method == "seqotl2":
            ex["kdtree_check"] = kd
        meta = _meta(sensor, q, len(final), arrays, score, ex)
        p = out_dir / f"dump_{method}_{sensor}_{q}.npz"
        np.savez_compressed(p, meta=json.dumps(meta), **arrays)
        paths.append(p)
    print(f"[{sensor}/{q}] n_queries={len(a_cos['query_idx'])} top-1 cos==l2 {agree:.3f} "
          f"kdtree {kd['kdtree_top1_equals_j_star']}/{kd['n_checked']} -> {out_dir}", flush=True)
    return paths


def run_worker(sensor, q, batch):
    import torch
    from _seqot_official_train import StoreSource, _chosen, _load_fe, _load_gem, device
    dev = device()
    p1, c1 = _chosen("stage1")
    p2, c2 = _chosen("stage2")
    fe, gem = _load_fe(p1, dev), _load_gem(p2, dev)
    src = StoreSource("test_odom", sensor, q)
    cp = cache_path("test_odom", sensor, q)
    with np.load(cp) as z:
        assert np.array_equal(z["scan_ids"], src.kf), f"{sensor}/{q}: keyframes != cache"
        assert np.array_equal(z["poses_gt"], src.kf_poses), f"{sensor}/{q}: poses_gt != cache"
    final, need, t_enc = encode_sequence(fe, gem, dev, src, batch)
    DESC.mkdir(parents=True, exist_ok=True)
    np.savez(DESC / f"{seq_name(sensor, q)}.npz", final=final, kf_scan_ids=src.kf,
             stage1_sha256=c1["sha256"], stage2_sha256=c2["sha256"])
    rmeta = json.loads((store_dir("test_odom", sensor, q) / "meta.json").read_text())
    extra = {"artifacts": {"stage1": c1, "stage2": c2,
                           "selection": str(SELECTION.relative_to(REPO))},
             "caches": {"operating_cache": str(cp.relative_to(REPO)),
                        "operating_cache_sha256": rmeta["cache_sha256"],
                        "positions": "poses_gt (scoring only)",
                        "raw_scans_read": int(len(need)), "n_raw": int(src.n_raw)},
             "timing": {"encode_s": round(t_enc, 1),
                        "encode_ms_per_kf": round(1e3 * t_enc / max(len(final), 1), 2)},
             "device": str(dev), "torch": torch.__version__}
    write_dump(sensor, q, final, src.kf_poses[:, :3, 3], OUT, extra)


def cmd_dump(seqs, worker, overwrite, batch):
    todo = seqs or [f"{s}/{q}" for s, q in TEST]
    unknown = sorted(set(todo) - {f"{s}/{q}" for s, q in TEST})
    assert not unknown, f"not split-v2 test sequences: {unknown}"
    if worker:
        assert len(todo) == 1
        run_worker(*todo[0].split("/"), batch)
        return
    for key in todo:
        s, q = key.split("/")
        if all((OUT / f"dump_{m}_{s}_{q}.npz").exists() for m in METHODS) and not overwrite:
            print(f"[skip] {key}", flush=True)
            continue
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "dump", "--worker",
                             "--seqs", key, "--batch", str(batch)]).returncode
        if rc:
            print(f"WORKER_FAILED {key} rc={rc}", flush=True)
            sys.exit(rc)
    print("SEQOT_OFFICIAL_DUMP_DONE", flush=True)


def cmd_aggregate():
    import _pdv2_c3 as c3
    c3.OUT_ROOT = SEED_RESULTS
    c3.BASELINES = dict(METHODS)
    c3.cmd_aggregate_baselines("odom")
    p = OUT / "pr_summary_baselines.json"
    s = json.loads(p.read_text())
    s["_meta"]["seqot"] = ("official SeqOT code and default recipe, retrained on split-v2 train "
                           "(scripts/_seqot_official_train.py), epochs chosen on v2 VAL by "
                           "L2 R@1; REPORTED = seqotl2 (official L2 retrieval), seqot = "
                           "cosine record (prereg §1.5); "
                           f"{SELECTION.relative_to(REPO)}")
    s["_meta"]["reported"] = REPORTED
    if TRAIN_SEED != 1:     # SEED = 1 output unchanged
        s["_meta"]["train_seed"] = TRAIN_SEED
    p.write_text(json.dumps(s, indent=2))


def cmd_dry_run_cpu(key, n, out_dir):
    """CPU only: untrained featureExtracter (seed 1) + official GeM init on
    the first n keyframes; range images projected on the fly; ledgers checked
    against the existing SeqOT ledger of the same sequence (same keyframes and
    GT positions -> same query_idx / min_past_gt_dist / n_candidates on the
    shared rows, same arrays and dtypes)."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from _pd_build_test_caches import _make_loader
    from _seqot_official_train import DictSource, official, seed_all
    sensor, q = key.split("/")
    cp = cache_path("test_odom", sensor, q)
    with np.load(cp) as z:
        kf = z["scan_ids"][:n].astype(np.int64)
        pos = z["poses_gt"][:n, :3, 3].astype(np.float64)
    L = _make_loader(sensor, q)
    need = sorted({int(r) for s in kf for r in range(max(0, s - BACK), s + 1)})
    t0 = time.time()
    src = DictSource({r: range_image(sensor, L[r]["points"]) for r in need})
    src.n_raw, src.kf = len(L), kf
    t_proj = time.time() - t0
    seed_all(SEED)          # format check: fixed seed, independent of env SEED
    fe = official()["featureExtracter"](seqL=SEQL).eval()
    gem = official()["GeM"]().eval()
    final, need2, t_enc = encode_sequence(fe, gem, torch.device("cpu"), src, 16)
    paths = write_dump(sensor, q, final, pos, Path(out_dir),
                       {"DRY_RUN": "CPU, untrained network, format check only",
                        "timing": {"project_s": round(t_proj, 1), "encode_s": round(t_enc, 1),
                                   "raw_scans": len(need)}})
    ref = np.load(REPO / f"results/pdv2_c3/odom/dump_seqot_{sensor}_{q}.npz")
    rep = {"raw_scans_projected": len(need), "project_s_per_scan": round(t_proj / len(need), 4),
           "encode_s_per_raw_scan_cpu": round(t_enc / len(need2), 4)}
    ok = True
    for p in paths:
        z = np.load(p)
        m = len(z["query_idx"])
        r = {"arrays": sorted(z.files), "ref_arrays": sorted(ref.files),
             "dtypes_match": all(str(z[k].dtype) == str(ref[k].dtype)
                                 for k in ("query_idx", "j_star", "top1_geo_dist",
                                           "min_past_gt_dist", "n_candidates"))}
        for k in ("query_idx", "min_past_gt_dist", "n_candidates"):
            r[f"{k}_equal_on_shared_rows"] = bool(np.array_equal(z[k], ref[k][:m]))
        zm = json.loads(str(z["meta"]))
        r["schema"] = [zm["schema"], json.loads(str(ref["meta"]))["schema"]]
        r["score_col"] = zm["primary_score"]
        r["score_dtype"] = str(z[zm["primary_score"]].dtype)
        r["causal"] = bool(np.all(z["j_star"] <= z["query_idx"] - MIN_INDEX_GAP))
        ok &= r["dtypes_match"] and r["causal"] and r["schema"] == ["pr-dump-v1", "pr-dump-v1"] and \
            all(r[f"{k}_equal_on_shared_rows"] for k in ("query_idx", "min_past_gt_dist", "n_candidates"))
        rep[p.name] = r
        p.unlink()
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
    p.add_argument("--worker", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--batch", type=int, default=64)
    sub.add_parser("aggregate")
    p = sub.add_parser("dry-run-cpu")
    p.add_argument("--seq", required=True)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "dump":
        cmd_dump(a.seqs, a.worker, a.overwrite, a.batch)
    elif a.cmd == "aggregate":
        cmd_aggregate()
    elif a.cmd == "dry-run-cpu":
        cmd_dry_run_cpu(a.seq, a.n, a.out_dir)


if __name__ == "__main__":
    main()
