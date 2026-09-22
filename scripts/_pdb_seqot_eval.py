#!/usr/bin/env python3
"""Evaluate a SeqOT checkpoint pair on the pd2026 42-sequence TEST roster.

Pipeline per sequence — identical to scripts/_seqot_eval.py: 256D
sub-descriptor per keyframe (featureExtracter, seqL=3 clamped window) ->
trained GeM pooling over seqlen=20 sub-descriptor windows -> cosine
retrieval scored with baselines/eval_utils.compute_recall_multi_k
(revisit < 5 m, >= 30-keyframe gap, R@1/5/10). Same seqlen-vs-skip
protocol note as _seqot_eval.py applies.

Differences from scripts/_seqot_eval.py (left untouched):
  * sequences default to the frozen 42-seq roster
    (scripts/_pd_test_cache_manifest.json: kitti/helipr/mulran/nclt/
    kitti360/mcd), range images from _external/seqot_data_pd2026
    (scripts/_pdb_seqot_prepare.py);
  * one JSON per sequence: results/pd2026_baselines/seqot_<sensor>_<seq>.json,
    with per-sequence runtime (descriptor s/keyframe, retrieval ms/query);
  * intended checkpoints: the pd2026 retrain
    (scripts/_pdb_seqot_train.py, e.g. _external/seqot_runs_pd2026/full/
    {seqot,gem}_latest.pth.tar), trained on the 16 pd TRAIN sequences only.

Verification mode: point --data-root at the paper-era _external/seqot_data
and --out-dir somewhere disposable to reproduce committed
results/seqot_eval.json numbers with the paper-era checkpoints.

Examples (inside the ring_baseline container):
  python scripts/_pdb_seqot_eval.py mcd/tuhh_night_09 \
      --seqot-checkpoint _external/seqot_runs_pd2026/full/seqot_latest.pth.tar \
      --gem-checkpoint _external/seqot_runs_pd2026/full/gem_latest.pth.tar
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
sys.path.insert(0, str(REPO / "scripts"))

import _seqot_eval as base  # noqa: E402  (SeqOT modules + descriptor pipeline)
from baselines.eval_utils import compute_recall_multi_k  # noqa: E402

MANIFEST = REPO / "scripts" / "_pd_test_cache_manifest.json"
OUT_DIR_DEFAULT = REPO / "results" / "pd2026_baselines"


def roster():
    m = json.loads(MANIFEST.read_text())
    return [(e["sensor"], e["sequence"]) for e in m["sequences"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sequences", nargs="*",
                    help="sensor/seq (default: all 42 roster sequences; heavy "
                         "sequences are cheap here, but per-seq JSONs are "
                         "written incrementally either way)")
    ap.add_argument("--seqot-checkpoint", required=True)
    ap.add_argument("--gem-checkpoint", required=True)
    ap.add_argument("--data-root", default=str(REPO / "_external/seqot_data_pd2026"))
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--max-frames", type=int, default=0, help="smoke only")
    ap.add_argument("--infer-batch", type=int, default=16)
    args = ap.parse_args()

    targets = [tuple(s.split("/", 1)) for s in args.sequences] or roster()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = base.featureExtracter(seqL=base.SEQL).to(device)
    fe.load_state_dict(torch.load(args.seqot_checkpoint, map_location=device,
                                  weights_only=False)["state_dict"])
    fe.eval()
    gem = base.GeM().to(device)
    gem.load_state_dict(torch.load(args.gem_checkpoint, map_location=device,
                                   weights_only=False)["state_dict"])
    gem.eval()

    for sensor, seq in targets:
        seq_dir = data_root / f"{sensor}_{seq}"
        if not (seq_dir / "poses.npy").exists():
            print(f"SKIP {sensor}/{seq}: not prepared under {seq_dir}", flush=True)
            continue
        t0 = time.time()
        stack = base.load_range_stack(seq_dir, args.max_frames)
        poses = np.load(seq_dir / "poses.npy")[:len(stack)]
        t1 = time.time()
        final = base.compute_final_descriptors(fe, gem, stack, device,
                                               args.infer_batch)
        t_desc = time.time() - t1
        t2 = time.time()
        recalls, n_q = compute_recall_multi_k(
            final, poses, k_values=base.K_VALUES,
            distance_threshold=5.0, skip_frames=30)
        t_ret = time.time() - t2
        n = len(stack)
        print(f"{sensor}/{seq:>16}  R@1={recalls[1]:.4f} R@5={recalls[5]:.4f} "
              f"R@10={recalls[10]:.4f}  ({n_q} queries, {n} frames, "
              f"desc {t_desc:.1f}s = {t_desc / n:.4f} s/kf, "
              f"retrieval {1000 * t_ret / max(n_q, 1):.1f} ms/query)", flush=True)
        payload = {
            f"{sensor}_{seq}": {
                "n_frames": n,
                "n_queries": n_q,
                **{f"recall@{k}": recalls[k] for k in base.K_VALUES},
                "timing": {
                    "load_s": round(t1 - t0, 1),
                    "descriptor_s_total": round(t_desc, 1),
                    "descriptor_s_per_kf": round(t_desc / n, 4),
                    "retrieval_s_total": round(t_ret, 1),
                    "retrieval_ms_per_query": round(1000 * t_ret / max(n_q, 1), 2),
                },
            },
            "_meta": {
                "method": "SeqOT",
                "descriptor_dim": 256,
                "seqot_checkpoint": args.seqot_checkpoint,
                "gem_checkpoint": args.gem_checkpoint,
                "data_root": str(data_root),
                "max_frames": args.max_frames,
                "protocol": "revisit<5m, skip 30 frames, intra-sequence",
            },
        }
        out = out_dir / f"seqot_{sensor}_{seq}.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
