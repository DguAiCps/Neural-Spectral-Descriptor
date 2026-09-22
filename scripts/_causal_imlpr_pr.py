#!/usr/bin/env python3
"""All-frames causal PR ledger for ImLPR (Track C baseline dump, Table 13).

Track C of docs/CAUSAL_PR_PREREGISTRATION.md: per sensor/seq of the Strict-v2
37-sequence roster (scripts/_pd_test_cache_manifest.json minus kitti/00,
kitti360/0004, helipr/Bridge04, nclt/2012-02-04, nclt/2012-03-17), dump one
row per query frame t >= 30 with everything scripts/_pr_eval_lib.py needs
downstream — no PR accounting happens here.

Protocol/ledger stage is scripts/_causal_bevplace_pr.ledger IMPORTED
UNCHANGED (per-query gemv cosine search restricted to prefix rows
j <= t - 30 via _causal_pr_dump.prefix_topk; exhaustive top-1 over the
prefix — never global-search-then-filter; O(N^2) dots per sequence, no
caps).  CAUSAL-SAFE precompute: each ImLPR descriptor depends only on its
own scan (prereg Track C: "per-frame methods reuse descriptors; only the
search is re-run").

Unlike the BEVPlace++ dump there is NO GPU encode stage: descriptors come
from the FROZEN interface produced by the ImLPR RIV/descriptor stage —
    _external/imlpr_data_pd2026/desc_{sensor}_{seq}.npz
(desc (n, 8448) float16, L2-normalized before the fp16 cast; scan_ids ==
the pd2026 operating-cache scan_ids, asserted here).  Rows are upcast
fp16 -> float32 on load and used as-is (norms within fp16 round-trip of 1;
recorded in meta).  This module imports NO torch, so the sibling's
--device / --worker-subprocess machinery is dropped: every sequence is a
cached-npz read + CPU-BLAS ledger and runs in-process (still resumable —
existing dumps are skipped unless --overwrite).

Native score (prereg §1.5 convention, per-frame descriptor row): 8448D
cosine of the prefix top-1.  Already a similarity (higher = accept) — NO
negation applied; recorded in meta.

Output, one npz per sequence (schema pr-dump-v1):
  results/pd2026_pr/dump_imlpr_{sensor}_{seq}.npz
  arrays query_idx / j_star / score_cos / top1_geo_dist / min_past_gt_dist /
  n_candidates + json-string meta (incl. the zero-shot weight convention and
  the per-sequence train_overlap flag: ImLPR trained on HeLiPR DCC04-06/
  KAIST04-06/Riverside04-06, 9 of which are Strict-v2 sequences).

--method helios serves the HeLiOS (ICRA'25) descriptors on the SAME frozen
interface (desc_helios_* from scripts/_helios_desc.py, also 8448D fp16,
same causal-safe per-frame precompute), writing dump_helios_*.npz; default
imlpr is unchanged.

Usage:
    python3 scripts/_causal_imlpr_pr.py --seqs mulran/DCC03
    python3 scripts/_causal_imlpr_pr.py            # all 37, resumable
    python3 scripts/_causal_imlpr_pr.py --selftest # synthetic, no data
    python3 scripts/_causal_imlpr_pr.py --method helios --seqs kitti/05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from _causal_bevplace_pr import (  # noqa: E402  (torch-free)
    STRICT_V2_EXCLUDED, ledger, selftest as _ledger_selftest,
    strict_v2_roster,
)
from _pdb_bevplace_eval import sha256_head  # noqa: E402  (torch-free)
from _pdb_imlpr_eval import (  # noqa: E402  (torch-free)
    METHODS, TRAIN_OVERLAP, load_desc, rel_to_repo,
)
from _pr_eval_lib import MIN_INDEX_GAP  # noqa: E402  (torch-free)

OUT = REPO / "results/pd2026_pr"


def load_sequence(sensor: str, seq: str, cache_rel: str,
                  prefix: str = "desc_"):
    """Stage 1 (torch-free): frozen ImLPR descriptors + operating-cache
    positions at the identical scan_ids (mirrors the cache loading of
    scripts/_causal_bevplace_pr.encode_sequence, encode replaced by read)."""
    cache_path = REPO / cache_rel
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        scan_ids = cache["scan_ids"].astype(np.int64)
        poses = cache["poses"].astype(np.float64)
    assert len(scan_ids) == len(poses), (sensor, seq, "scan_ids/poses length")

    desc, desc_sids, desc_path = load_desc(sensor, seq, prefix=prefix)
    assert np.array_equal(desc_sids, scan_ids), \
        (sensor, seq, "descriptor scan_ids != operating-cache scan_ids")
    return desc, poses[:, :3, 3], desc_path


def run_sequence(args, sensor: str, seq: str, cache_rel: str) -> None:
    spec = METHODS[args.method]
    out_path = OUT / f"dump_{args.method}_{sensor}_{seq}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path.name} exists", flush=True)
        return

    desc, positions, desc_path = load_sequence(sensor, seq, cache_rel,
                                               prefix=spec["prefix"])
    print(f"[{sensor}/{seq}] ledger over {len(desc)} frames "
          f"({desc_path.name})", flush=True)
    arrays = ledger(desc, positions)

    meta = {
        "schema": "pr-dump-v1",
        "method": args.method,
        "sensor": sensor, "sequence": seq,
        "min_index_gap": MIN_INDEX_GAP,
        "n_frames": int(len(desc)),
        "n_queries": int(len(arrays["query_idx"])),
        "artifacts": {
            "weights": str(args.weights),
            "weights_sha256_1mb": sha256_head(args.weights),
            "desc_cache": rel_to_repo(desc_path),
        },
        "caches": {"operating_cache": cache_rel},
        "encoder": dict(
            spec["encoder"],
            note="descriptors consumed from the frozen fp16 cache, upcast to "
                 "float32; L2-normalized before the fp16 cast, so row norms "
                 "are 1 within fp16 round-trip"),
        "weight_convention": spec["convention"],
        "train_overlap": (sensor, seq) in spec.get("overlap", TRAIN_OVERLAP),
        "zero_shot_wrt_pd_train": True,
        "primary_score": "score_cos",
        "scores": {"score_cos": "8448D descriptor cosine of the exhaustive "
                                "prefix top-1; native similarity, higher = "
                                "accept; NOT negated (prereg §1.5 per-frame "
                                "descriptor score is already a similarity)"},
        "protocol": (f"causal all-frames PR ledger (Track C, {args.method}): "
                     f"per-frame 8448D {spec['display']} descriptors "
                     "precomputed once per "
                     "sequence (each depends only on its own scan — "
                     "causal-safe precompute), then per-query gemv cosine "
                     "search restricted to prefix rows j <= t - 30 "
                     "(exhaustive top-1; rows beyond the prefix never read, "
                     "no budget caps); query set / accounting inputs per "
                     "docs/CAUSAL_PR_PREREGISTRATION.md §1.2/§1.3/§1.5"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    fin = np.isfinite(arrays["score_cos"])
    print(f"[{sensor}/{seq}] n_queries={len(arrays['query_idx'])} "
          f"finite={int(fin.sum())} train_overlap={meta['train_overlap']} "
          f"-> {out_path.name}", flush=True)


def selftest() -> None:
    """Synthetic, no data, no torch: (1) the imported shared ledger passes
    the sibling's full check suite (causality, exhaustive-prefix reference,
    prefix invariance); (2) the fp16 storage round-trip this driver feeds it
    (L2-normalize -> float16 -> float32) still matches the exhaustive-prefix
    reference computed on the same upcast bank."""
    _ledger_selftest()  # scripts/_causal_bevplace_pr.selftest on shared ledger

    rng = np.random.default_rng(1)
    n = 90
    desc = rng.standard_normal((n, 32)).astype(np.float32)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True)
    desc = desc.astype(np.float16).astype(np.float32)  # frozen-interface path
    positions = rng.uniform(0.0, 100.0, size=(n, 3))
    arrays = ledger(desc, positions)
    q = arrays["query_idx"]
    assert (q == np.arange(MIN_INDEX_GAP, n)).all(), "query set != all t >= 30"
    assert (arrays["j_star"] <= q - MIN_INDEX_GAP).all(), "causality violated"
    for qi, t in enumerate(q):
        t = int(t)
        sims = desc[:t - MIN_INDEX_GAP + 1] @ desc[t]
        j_ref = int(np.argmax(sims))
        assert arrays["j_star"][qi] == j_ref, (t, arrays["j_star"][qi], j_ref)
        assert arrays["score_cos"][qi] == float(desc[j_ref] @ desc[t]), t
    print("SELFTEST_OK: shared ledger checks pass and the fp16-upcast "
          "descriptor path matches the exhaustive-prefix reference",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=sorted(METHODS), default="imlpr",
                    help="frozen descriptor stack to dump")
    ap.add_argument("--weights", type=Path, default=None,
                    help="official checkpoint (sha-logged only; descriptors "
                         "are consumed precomputed; default: the method's "
                         "release checkpoint)")
    ap.add_argument("--seqs", nargs="+", default=None, metavar="SENSOR/SEQ",
                    help="subset of the Strict-v2 37, e.g. mulran/DCC03 "
                         "(default: all 37)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic ledger + fp16-path check (no data)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.weights is None:
        args.weights = METHODS[args.method]["weights"]
    roster = strict_v2_roster()
    if args.seqs:
        unknown = sorted(set(args.seqs) - set(roster))
        if unknown:
            raise SystemExit(
                f"not in the Strict-v2 37-seq roster: {unknown} "
                f"(excluded by prereg R5: {list(STRICT_V2_EXCLUDED)})")
        todo = [(s, roster[s]) for s in args.seqs]
    else:
        todo = list(roster.items())
    if not args.weights.is_file():
        raise SystemExit(f"missing {METHODS[args.method]['display']} "
                         f"checkpoint: {args.weights}")

    for key, cache_rel in todo:
        sensor, seq = key.split("/")
        run_sequence(args, sensor, seq, cache_rel)
    print(f"CAUSAL_{args.method.upper()}_PR_DONE", flush=True)


if __name__ == "__main__":
    main()
