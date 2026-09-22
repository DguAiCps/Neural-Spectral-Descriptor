#!/usr/bin/env python3
"""Dump the frozen pd-protocol 416D retrieval keys of a sequence to an npz
that scripts/evaluate_cross_session.py can consume directly.

Per sequence, the deployed two-pass (B3) embedding is computed exactly as
scripts/_pd_branch_fusion.nsd_embedding_pd does (pass-1 graph -> edge
classifier -> classifier-selected edges on a temporal-only graph -> forward,
L2-normalized), with the frozen place-disjoint artifacts:

  --variant pd    checkpoints/pd_s{seed}   + artifacts/key_remetrize_r_pd2026.npy
                  + artifacts/edge_classifier_pd2026.pt
  --variant pdnn  checkpoints/pdnn_s{seed} + *_pdnn2026 artifacts

Inputs per sequence (data/preprocessed_pd2026):
  cache_*_{sensor}_[val_]{seq}.npz        descriptors/poses/scan_ids (required)
  vallayouts_{sensor}_{seq}.npz or
  layouts_{sensor}_{seq}_layout60.npz     nsd_layouts for the classifier (required)
  valspec/spec/testspec_{sensor}_{seq}.npz  optional; adds the branch magnitude
                                          keys pbev_mag (128D) / hspec_mag (240D)
                                          via the unchanged paper input builders

Writes results/pd2026/keys_{variant}_s{seed}_{sensor}_{seq}.npz with fields
emb416 + poses (auto-detected by evaluate_cross_session.py) + scan_ids +
keyframe_ids + meta. Existing outputs are skipped (delete to redo).

Memory discipline: with more than one --seqs entry the script re-execs itself
one sequence per subprocess; heavy sequences never share a process.
"""
from __future__ import annotations
import argparse
import glob
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

PD_CACHE = REPO / "data/preprocessed_pd2026"
OUT = REPO / "results/pd2026"
SENSOR_KEY = {"kitti360": "kitti"}  # kitti360 shares the HDL-64E sensor entry


def _find_one(patterns, what, sensor, seq):
    hits = []
    for pat in patterns:
        hits = sorted(glob.glob(str(PD_CACHE / pat)))
        if hits:
            break
    if len(hits) != 1:
        raise FileNotFoundError(
            f"{what} for {sensor}/{seq}: expected exactly one of "
            f"{patterns} in {PD_CACHE}, found {hits}")
    return Path(hits[0])


def dump_one(variant: str, seed: int, sensor: str, seq: str, device: str) -> Path:
    out = OUT / f"keys_{variant}_s{seed}_{sensor}_{seq}.npz"
    if out.exists():
        print(f"SKIP {out.name} (exists)", flush=True)
        return out

    # torch-side imports deferred so --help / multi-seq dispatch stay light
    from _pd_branch_fusion import make_model_pd, nsd_embedding_pd
    from _pbev_branch_fusion import variant_inputs
    from _hspec_branch_fusion import hspec_inputs

    cache_path = _find_one(
        [f"cache_*_{sensor}_val_{seq}.npz", f"cache_*_{sensor}_test_{seq}.npz",
         f"cache_*_{sensor}_{seq}.npz"],
        "pd2026 cache", sensor, seq)
    lay_path = _find_one(
        [f"vallayouts_{sensor}_{seq}.npz", f"layouts_{sensor}_{seq}_layout60.npz",
         f"testlayouts_{sensor}_{seq}.npz"],
        "nsd layouts", sensor, seq)
    cache = np.load(cache_path)
    lay = np.load(lay_path)
    assert (lay["scan_ids"] == cache["scan_ids"]).all(), (sensor, seq)
    d288 = cache["descriptors"].astype(np.float32)
    layouts = lay["nsd_layouts"].astype(np.float32)

    cfg, model, clf, clf_blob, r, ckpt = make_model_pd(variant, seed, device)
    emb = nsd_embedding_pd(cache, SENSOR_KEY.get(sensor, sensor), d288, layouts,
                           cfg, model, clf, clf_blob, r, device)

    fields = dict(
        emb416=emb.astype(np.float32),
        poses=cache["poses"],
        scan_ids=cache["scan_ids"],
        keyframe_ids=cache["keyframe_ids"],
    )
    spec_path = None
    for pat in (f"valspec_{sensor}_{seq}.npz", f"spec_{sensor}_{seq}.npz",
                f"testspec_{sensor}_{seq}.npz"):
        hits = sorted(glob.glob(str(PD_CACHE / pat)))
        if hits:
            spec_path = Path(hits[0])
            break
    if spec_path is not None:
        spec = np.load(spec_path)
        assert (spec["scan_ids"] == cache["scan_ids"]).all(), (sensor, seq)
        sk = spec["sk_re"].astype(np.float32) + 1j * spec["sk_im"].astype(np.float32)
        fields["pbev_mag"] = variant_inputs(sk, 8)[0].astype(np.float32)
        fields["hspec_mag"] = hspec_inputs(
            spec["raw_height"].astype(np.float32), 12)[0].astype(np.float32)
    else:
        print(f"NOTE {sensor}/{seq}: no spec cache -> emb416 only "
              "(no pbev_mag/hspec_mag branch fields)", flush=True)

    meta = json.dumps(dict(
        variant=variant, seed=seed, sensor=sensor, sequence=seq,
        stage="b3_two_pass", checkpoint=str(ckpt),
        cache=str(cache_path), layouts=str(lay_path),
        spec=str(spec_path) if spec_path else None,
        command=shlex.join(sys.argv),
    ))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **fields, meta=meta)
    extra = " +pbev_mag/hspec_mag" if spec_path else ""
    print(f"OK {out.name}: emb416 {emb.shape}{extra}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("pd", "pdnn"), required=True)
    ap.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--seqs", nargs="+", required=True,
                    help="sensor/sequence specs, e.g. kitti/09 mulran/DCC01")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if len(args.seqs) > 1:
        # one heavy sequence per process: re-exec self per sequence
        for spec in args.seqs:
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--variant", args.variant, "--seed", str(args.seed),
                   "--seqs", spec, "--device", args.device]
            print("SUBPROC " + shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        return

    sensor, seq = args.seqs[0].split("/", 1)
    dump_one(args.variant, args.seed, sensor, seq, args.device)


if __name__ == "__main__":
    main()
