#!/usr/bin/env python3
"""RING#-L pd-retrained evaluation wrapper (test roster + pd-val selection).

New wrapper so scripts/_pdb_ringsharp_eval.py stays byte-identical (its
_meta hardcodes the zero-shot/train-overlap convention of the paper-era
weights, which does not apply to the pd2026 retrain). Everything heavy is
imported unchanged from _pdb_ringsharp_eval (loader factory, feature
extraction, official rotation+translation rerank, protocol constants).

Two modes:
  * test (default): sequences from the frozen 42-seq pd2026 test roster ->
    results/pd2026_baselines/ringsharp_pd_{sensor}_{seq}.json.
    The pd2026 train split is place-disjoint from the test roster by
    construction, so train_overlap is False everywhere.
  * --val: sequences resolved against the pd2026 VAL caches
    (cache_*_{sensor}_val_{seq}.npz) for the pre-declared checkpoint
    selection rule ("highest pd-validation sequence-macro 5 m Recall@1 over
    epochs 1-5; earlier epoch on ties") ->
    results/pd2026_baselines/_val_ringsharp_pd_{tag}_{sensor}_{seq}.json.

The weight file's sha256 is recorded in every output _meta.

Examples (inside the ring_baseline container):
  python scripts/_pdb_ringsharp_pd_eval.py --weight W.pth --val --tag e3 kitti/09
  python scripts/_pdb_ringsharp_pd_eval.py --weight W.pth mcd/tuhh_night_09
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import _pdb_ringsharp_eval as pdrs                              # noqa: E402

PD_CACHE = REPO / "data" / "preprocessed_pd2026"
OUT_DIR_DEFAULT = REPO / "results" / "pd2026_baselines"


def val_cache(sensor: str, seq: str) -> Path:
    hits = sorted(glob.glob(str(PD_CACHE / f"cache_*_{sensor}_val_{seq}.npz")))
    assert len(hits) == 1, (sensor, seq, hits)
    return Path(hits[0])


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sequences", nargs="+", metavar="SENSOR/SEQ")
    ap.add_argument("--weight", required=True,
                    help="pd2026-retrained RING#-L checkpoint")
    ap.add_argument("--val", action="store_true",
                    help="resolve against pd2026 VAL caches (ckpt selection)")
    ap.add_argument("--tag", default="",
                    help="val-mode output tag (e.g. e1..e5); required with --val")
    ap.add_argument("--n-coarse", type=int, default=200)
    ap.add_argument("--n-trans", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0, help="smoke only")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.val:
        assert args.tag, "--val requires --tag (epoch label for the output name)"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roster = pdrs.load_manifest()
    targets = [tuple(s.split("/", 1)) for s in args.sequences]
    if not args.val:
        for t in targets:
            assert t in roster, f"not in the frozen 42-seq roster: {t[0]}/{t[1]}"

    weight_sha = sha256_of(args.weight)
    print(f"weight={args.weight}\nsha256={weight_sha}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, trans_cnn = pdrs.base.load_model(args.weight, device)

    for sensor, seq in targets:
        if args.val:
            cache_p = val_cache(sensor, seq)
            out = out_dir / f"_val_ringsharp_pd_{args.tag}_{sensor}_{seq}.json"
        else:
            cache_p = roster[(sensor, seq)]
            out = out_dir / f"ringsharp_pd_{sensor}_{seq}.json"
        if out.exists() and not args.overwrite:
            print(f"[{sensor}_{seq}] {out.name} exists, skipping", flush=True)
            continue
        with np.load(cache_p, allow_pickle=True) as c:
            scan_ids = c["scan_ids"].astype(np.int64)
            poses = c["poses"].astype(np.float64)
        if args.max_frames:
            scan_ids, poses = scan_ids[:args.max_frames], poses[:args.max_frames]
        print(f"[{sensor}_{seq}] {len(scan_ids)} keyframes "
              f"({'pd2026 val cache' if args.val else 'pd2026 test cache'})",
              flush=True)
        t0 = time.time()
        res = pdrs.evaluate_sequence(model, trans_cnn, sensor, seq, scan_ids,
                                     poses, device, args)
        payload = {
            f"{sensor}_{seq}": res,
            "_meta": {
                "weight": args.weight,
                "weight_sha256": weight_sha,
                "weight_convention": (
                    "pd2026 place-disjoint retrain (single seed, official "
                    "RINGSharp trainer, recipe inherited from the paper-era "
                    "run); checkpoint selected by the pre-declared rule: "
                    "highest pd-validation sequence-macro 5 m Recall@1 over "
                    "epochs 1-5, earlier epoch on ties"),
                "train_overlap": False,
                "mode": "val-selection" if args.val else "test",
                "cache": str(cache_p).replace(str(REPO) + "/", ""),
                "n_coarse": args.n_coarse,
                "n_trans": args.n_trans,
                "max_frames": args.max_frames,
                "protocol": "revisit<5m, skip 30 frames, intra-sequence",
                "wall_s": round(time.time() - t0, 1),
            },
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
