#!/usr/bin/env python3
"""Frozen-tuple APPLY evaluation for the pd protocol.

Runs the full frozen place-disjoint pipeline -- variant checkpoint
(checkpoints/{pd,pdnn}_s{seed}) + omega (artifacts/key_remetrize_r_*.npy)
+ edge classifier (artifacts/edge_classifier_*.pt) + branch fusion -- on
arbitrary sequences whose pd2026 caches exist, scoring with the FIXED fusion
tuple recorded in results/pd2026/tuple_selection.json (never re-selected).

Both operating points are applied per sequence:
  pbev128_main -- NSD    (BEV 16x8 sketch,  scripts/_pbev_branch_fusion.variant_inputs)
  hspec12      -- NSD-H  (height-polar 20x12, scripts/_hspec_branch_fusion.hspec_inputs)

Model construction and the two-pass classifier-selected-edge embedding are
imported unchanged from scripts/_pd_branch_fusion.py; the per-query candidate
union (two top-100 cosine indexes), min-max 4-channel fusion, and ground truth
(<5 m revisit after a 30-frame skip) replicate
scripts/_height_candidate_fusion.evaluate exactly, restricted to the single
frozen tuple, plus R@5.

MEMORY DISCIPLINE: the CLI accepts a sequence list, but the driver processes
each sequence in its own subprocess (--worker); the heavy model/eval never
loops sequences in-process.

Caches are looked up in data/preprocessed_pd2026 with marker preference
test_ > val_ (> unmarked for the descriptor cache); the resolved paths and
marker are echoed into the output JSON:
  results/pd2026/test_apply_{variant}_{op}_{seed}_{sensor}_{seq}.json
"""
from __future__ import annotations
import argparse
import ast
import glob
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PD_CACHE = REPO / "data/preprocessed_pd2026"
OUT = REPO / "results/pd2026"
TUPLES = OUT / "tuple_selection.json"
OPS = ("pbev128_main", "hspec12")


def _find_one(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(str(PD_CACHE / pat)))
        if hits:
            assert len(hits) == 1, (pat, hits)
            return Path(hits[0])
    raise FileNotFoundError(f"no cache matching any of {patterns} in {PD_CACHE}")


def resolve_caches(sensor: str, seq: str):
    cache = _find_one([f"cache_*_{sensor}_test_{seq}.npz",
                       f"cache_*_{sensor}_val_{seq}.npz",
                       f"cache_*_{sensor}_{seq}.npz"])
    marker = ("test" if f"_{sensor}_test_{seq}" in cache.name
              else "val" if f"_{sensor}_val_{seq}" in cache.name else "none")
    lay = _find_one([f"testlayouts_{sensor}_{seq}.npz",
                     f"vallayouts_{sensor}_{seq}.npz"])
    spec = _find_one([f"testspec_{sensor}_{seq}.npz",
                      f"valspec_{sensor}_{seq}.npz"])
    return cache, lay, spec, marker


def frozen_tuple(variant: str, op: str):
    sel = json.loads(TUPLES.read_text())[f"{variant}/{op}"]["selected_tuple"]
    w = ast.literal_eval(sel)
    assert isinstance(w, tuple) and len(w) == 4, sel
    return tuple(float(x) for x in w), sel


def evaluate_fixed(poses, nsd, hkey, hfft, hnorm, column_bank, pool_k, w):
    """scripts/_height_candidate_fusion.evaluate restricted to one fixed
    tuple (identical query set, candidate union, min-max channels, and R@1
    argmax), extended with R@5 over the fused candidate ranking."""
    import numpy as np
    from _height_candidate_fusion import (
        DTH, SKIP, minmax, align_scores, column_align_scores)
    from run_kitti_operating_point import _find_queries, _topk_cosine

    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    r1 = r5 = 0
    for qi, (q, _) in enumerate(queries):
        cn = _topk_cosine(nsd, q, pool_k + 2 * SKIP, SKIP)[:pool_k]
        ch = _topk_cosine(hkey, q, pool_k + 2 * SKIP, SKIP)[:pool_k]
        candidates = np.unique(np.concatenate([cn, ch]))
        good = np.linalg.norm(positions[candidates] - positions[q], axis=1) < DTH
        sn = minmax(nsd[candidates] @ nsd[q])
        sh = minmax(hkey[candidates] @ hkey[q])
        sa = minmax(align_scores(hfft, hnorm, q, candidates))
        sd = minmax(column_align_scores(column_bank, q, candidates))
        score = w[0] * sn + w[1] * sh + w[2] * sa + w[3] * sd
        r1 += bool(good[int(np.argmax(score))])
        r5 += bool(good[np.argsort(-score, kind="stable")[:5]].any())
        if qi and qi % 500 == 0:
            print(f"  apply queries {qi}/{len(queries)}", flush=True)
    n = max(len(queries), 1)
    return float(r1 / n), float(r5 / n), len(queries)


def run_worker(args, sensor: str, seq: str):
    import numpy as np
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts"))
    from _pd_branch_fusion import VARIANTS, make_model_pd, nsd_embedding_pd
    from _pbev_branch_fusion import variant_inputs
    from _hspec_branch_fusion import hspec_inputs

    cache_p, lay_p, spec_p, marker = resolve_caches(sensor, seq)
    print(f"[{sensor}/{seq}] cache={cache_p.name} layouts={lay_p.name} "
          f"spec={spec_p.name} marker={marker}", flush=True)
    cache = np.load(cache_p)
    lay = np.load(lay_p)
    spec = np.load(spec_p)
    assert (lay["scan_ids"] == cache["scan_ids"]).all(), (sensor, seq, "layouts")
    assert (spec["scan_ids"] == cache["scan_ids"]).all(), (sensor, seq, "spec")
    d288 = cache["descriptors"].astype(np.float32)
    layouts = lay["nsd_layouts"].astype(np.float32)
    sk = spec["sk_re"].astype(np.float32) + 1j * spec["sk_im"].astype(np.float32)
    raw_height = spec["raw_height"].astype(np.float32)

    V = VARIANTS[args.variant]
    cfg, model, clf, clf_blob, r, ckpt = make_model_pd(
        args.variant, args.seed, args.device)
    nsd = nsd_embedding_pd(cache, sensor, d288, layouts,
                           cfg, model, clf, clf_blob, r, args.device)

    for op, (hkey, hfft, hnorm, column_bank) in (
            ("pbev128_main", variant_inputs(sk, 8)),
            ("hspec12", hspec_inputs(raw_height, 12))):
        out_path = OUT / (f"test_apply_{args.variant}_{op}_{args.seed}_"
                          f"{sensor}_{seq}.json")
        if out_path.exists() and not args.overwrite:
            print(f"[{sensor}/{seq}] {op}: {out_path.name} exists, skipping",
                  flush=True)
            continue
        w, sel_str = frozen_tuple(args.variant, op)
        print(f"[{sensor}/{seq}] op={op} tuple={sel_str}", flush=True)
        r1, r5, n_q = evaluate_fixed(cache["poses"], nsd, hkey, hfft, hnorm,
                                     column_bank, args.pool_k, w)
        result = {
            "variant": args.variant, "op": op, "seed": args.seed,
            "sensor": sensor, "sequence": seq,
            "r_at_1": r1, "r_at_5": r5, "n_queries": n_q,
            "tuple": list(w), "tuple_str": sel_str,
            "tuple_source": str(TUPLES),
            "pool_k": args.pool_k, "device": args.device,
            "artifacts": {
                "checkpoint": str(ckpt), "rvec": str(V["rvec"]),
                "classifier": str(V["cls"]), "config": str(V["config"]),
            },
            "caches": {"cache": str(cache_p), "layouts": str(lay_p),
                       "spec": str(spec_p), "marker": marker},
            "protocol": ("frozen-tuple apply: two-pass classifier-selected-edge "
                         "embedding, union of two top-100 cosine indexes, "
                         "per-query min-max 4-channel fusion at the FIXED tuple "
                         "(no re-selection), <5 m revisit after 30-frame skip"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[{sensor}/{seq}] {op}: R@1={r1:.6f} R@5={r5:.6f} "
              f"n={n_q} -> {out_path.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("pd", "pdnn"), required=True)
    ap.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--seqs", nargs="+", required=True,
                    metavar="SENSOR/SEQ", help="e.g. helipr/Roundabout03")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool-k", type=int, default=100)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--worker", action="store_true",
                    help="internal: run a single sequence in-process")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in args.seqs:
        assert s.count("/") == 1, f"--seqs entries must be sensor/seq, got {s!r}"

    if args.worker:
        assert len(args.seqs) == 1, "--worker takes exactly one sequence"
        sensor, seq = args.seqs[0].split("/")
        run_worker(args, sensor, seq)
        return

    # Driver: one heavy sequence per subprocess (memory discipline).
    for s in args.seqs:
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--variant", args.variant, "--seed", str(args.seed),
               "--seqs", s, "--device", args.device,
               "--pool-k", str(args.pool_k), "--worker"]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"=== spawning worker: {s} ===", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"WORKER_FAILED {s} rc={rc}", flush=True)
            sys.exit(rc)
    print("PD_APPLY_FROZEN_DONE", flush=True)


if __name__ == "__main__":
    main()
