#!/usr/bin/env python3
"""Evaluate a trained SeqOT baseline on OUR val split (intra-sequence R@1/5/10).

Pipeline per val sequence (range images prepared by scripts/_seqot_prepare.py):
  1. 256D sub-descriptor per keyframe with the trained featureExtracter
     (seqL=3 window centered on the keyframe, clamped at sequence edges),
  2. final descriptor = trained GeM pooling over the seqlen=20 sub-descriptor
     window centered on the keyframe (clamped at edges) -> one 256D vector
     per keyframe,
  3. cosine retrieval scored with baselines/eval_utils.compute_recall_multi_k
     (query = revisit < 5 m with >= 30-keyframe gap — identical protocol to
     every other baseline in this repo).

Protocol note: the seqlen=20 temporal window means each final descriptor
mixes information from neighbor keyframes within +-10 keyframes of the query.
The 30-frame skip guard used by compute_recall_multi_k already exceeds that
window, so database entries whose windows overlap the query's window are
never counted as retrieval hits; residual leakage is limited to the query's
own descriptor being sequence-smoothed (inherent to sequence-based methods).

Prints per-sequence R@1/5/10 + query-weighted mean and writes
results/seqot_eval.json.

Example (inside docker):
  python scripts/_seqot_eval.py \
      --seqot-checkpoint _external/seqot_runs/full/seqot_latest.pth.tar \
      --gem-checkpoint _external/seqot_runs/full/gem_latest.pth.tar
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
SEQOT = REPO / "_external" / "SeqOT"
sys.path.insert(0, str(SEQOT))
sys.path.insert(1, str(REPO))

from modules.seqTransformerCat import featureExtracter  # noqa: E402 (unchanged)
from modules.gem import GeM                              # noqa: E402 (unchanged)
from baselines.eval_utils import compute_recall_multi_k  # noqa: E402

SEQL = 3
GEM_SEQLEN = 20
K_VALUES = [1, 5, 10]


def window(center, seqlen, n):
    lo = center - seqlen // 2
    return np.clip(np.arange(lo, lo + seqlen), 0, n - 1)


def load_range_stack(seq_dir, max_frames):
    files = sorted(seq_dir.glob("[0-9]" * 6 + ".npy"))
    if max_frames:
        files = files[:max_frames]
    return np.stack([np.load(f) for f in files]).astype(np.float32)


def compute_final_descriptors(fe, gem, stack, device, infer_batch):
    n = len(stack)
    sub = np.zeros((n, 256), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, infer_batch):
            idx = np.arange(i, min(i + infer_batch, n))
            w = np.stack([window(j, SEQL, n) for j in idx])
            sub[idx] = fe(torch.from_numpy(stack[w]).to(device)).cpu().numpy()
        final = np.zeros((n, 256), dtype=np.float32)
        gem_batch = 512
        for i in range(0, n, gem_batch):
            idx = np.arange(i, min(i + gem_batch, n))
            w = np.stack([window(j, GEM_SEQLEN, n) for j in idx])
            out = gem(torch.from_numpy(sub[w]).to(device)).squeeze(1)
            final[idx] = out.cpu().numpy()
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=str(REPO / "_external/seqot_data"))
    ap.add_argument("--seqot-checkpoint", required=True)
    ap.add_argument("--gem-checkpoint", required=True)
    ap.add_argument("--config", default=str(REPO / "configs/training_multi_dataset.yaml"))
    ap.add_argument("--sequences", nargs="+", default=None,
                    help="dtype/seq pairs (default: all 9 config val sequences)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="truncate each sequence to N keyframes (smoke only)")
    ap.add_argument("--infer-batch", type=int, default=16)
    ap.add_argument("--out", default=str(REPO / "results/seqot_eval.json"))
    args = ap.parse_args()

    if args.sequences is None:
        config = yaml.safe_load(open(args.config))
        args.sequences = [f"{ds['type']}/{seq}"
                          for ds in config["data"]["datasets"]["val"]
                          for seq in ds["sequences"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = featureExtracter(seqL=SEQL).to(device)
    fe.load_state_dict(torch.load(args.seqot_checkpoint, map_location=device,
                                  weights_only=False)["state_dict"])
    fe.eval()
    gem = GeM().to(device)
    gem.load_state_dict(torch.load(args.gem_checkpoint, map_location=device,
                                   weights_only=False)["state_dict"])
    gem.eval()

    data_root = Path(args.data_root)
    results = {}
    for s in args.sequences:
        dtype, seq = s.split("/", 1)
        seq_dir = data_root / f"{dtype}_{seq}"
        if not (seq_dir / "poses.npy").exists():
            print(f"SKIP {s}: not prepared under {seq_dir}")
            continue
        t0 = time.time()
        stack = load_range_stack(seq_dir, args.max_frames)
        poses = np.load(seq_dir / "poses.npy")[:len(stack)]
        final = compute_final_descriptors(fe, gem, stack, device,
                                          args.infer_batch)
        recalls, n_queries = compute_recall_multi_k(
            final, poses, k_values=K_VALUES,
            distance_threshold=5.0, skip_frames=30)
        results[s] = {"recalls": {str(k): recalls[k] for k in K_VALUES},
                      "n_queries": n_queries, "n_frames": len(stack)}
        print(f"{s:>24}  R@1={recalls[1]:.4f} R@5={recalls[5]:.4f} "
              f"R@10={recalls[10]:.4f}  ({n_queries} queries, "
              f"{len(stack)} frames, {time.time() - t0:.1f}s)")
        del stack

    total_q = sum(r["n_queries"] for r in results.values())
    weighted = {str(k): (sum(r["recalls"][str(k)] * r["n_queries"]
                             for r in results.values()) / total_q
                         if total_q else 0.0)
                for k in K_VALUES}
    print(f"{'query-weighted mean':>24}  R@1={weighted['1']:.4f} "
          f"R@5={weighted['5']:.4f} R@10={weighted['10']:.4f}  "
          f"({total_q} queries)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "method": "SeqOT (trained on our split)",
        "descriptor_dim": 256,
        "protocol": "intra-sequence R@K, revisit <5m, skip>30 keyframes",
        "seqot_checkpoint": str(args.seqot_checkpoint),
        "gem_checkpoint": str(args.gem_checkpoint),
        "max_frames": args.max_frames,
        "per_sequence": results,
        "weighted_mean": weighted,
        "total_queries": total_q,
    }, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
