#!/usr/bin/env python3
"""Track A (validation-phase) runner for the pre-registered causal PR campaign.

docs/CAUSAL_PR_PREREGISTRATION.md §3 track A: every selection made here --
fusion tuple (§2.3 "val 4시퀀스에서 재선택, 기존 grid·규칙"), confidence
statistic (§1.5), per-seed tau* (§1.4) -- is made on the pd VAL 4-sequence
pool ONLY and frozen before any test contact:

  kitti/09  nclt/2012-01-08  helipr/Roundabout03  mulran/Sejong03

Caches are resolved VAL-ONLY (cache_*_{sensor}_val_{seq}.npz +
vallayouts_/valspec_, the scripts/_pd_branch_fusion.py naming) -- deliberately
NOT via scripts/_pd_apply_frozen.resolve_caches, whose test_ > val_ marker
preference must never leak a test cache into the Track A selection.

Subcommands (all resumable: existing outputs are skipped unless --overwrite;
the two selection JSONs are freeze artifacts and are likewise never silently
rewritten):

  tuples --variant {pd_causal,pd} --seed {1,2,3}
      Per val sequence x op (pbev128_main = NSD, hspec12 = NSD-H): one causal
      two-pass embedding (scripts/_causal_branch_fusion.nsd_embedding_causal),
      causal revisit queries (run_kitti_operating_point._find_queries: a
      positive within 5 m at >= 30 frames EARLIER -- already past-only), and
      R@1 for EVERY tuple in scripts/_height_candidate_fusion.WEIGHTS with
      PAST-ONLY candidates: two per-query gemv prefix top-100 cosine pools
      over j <= t - 30 (scripts/_causal_pr_dump.prefix_topk) -> union ->
      per-query min-max 4-channel fusion (torch-free _xs_eval_lib helpers,
      the _causal_pr_dump code path).  Output per (variant, op, seed, seq):
      results/pd2026_pr/val_tuples_{variant}_{op}_s{seed}_{sensor}_{seq}.json

  select-tuple
      Aggregates all val_tuples JSONs; per (variant, op) picks the tuple
      maximizing the seed-summed query-weighted R@1 over the 4 val sequences
      -- EXACTLY the scripts/_pd_tuple_select.py rule:
        score(w) = sum_{seed} [ sum_seq n_q(seq) * r1(seq, w, seed)
                                / sum_seq n_q(seq) ],
      argmax via Python max first-wins => ties resolve to the first tuple in
      WEIGHTS order (the val_tuples JSON preserves WEIGHTS insertion order,
      as branch_fusion JSONs do for _pd_tuple_select).  Seeds enter only with
      all 4 sequences complete (the _pd_tuple_select "ok" rule).  Writes
      results/pd2026_pr/causal_tuple_selection.json (rule text + top-5 table
      + selected tuples).

  confidence --variant {pd_causal,pd} --seed {1,2,3}
      With the SELECTED tuples (causal_tuple_selection.json; fails loudly if
      absent): per val sequence x op the ALL-FRAMES PR ledger via
      scripts/_causal_pr_dump.dump_op (t >= 30, NaN-safe, §1.2) into
      results/pd2026_pr/valdump_{variant}_{op}_s{seed}_{sensor}_{seq}.npz,
      then (driver, torch-free) for BOTH pre-registered confidence statistics
        absolute = score_abs   (raw 416D key cosine at the fused argmax)
        margin   = score_margin (fused ranking S1 - S2)
      the per-sequence _pr_eval_lib.pr_sweep AP (classify_queries on
      min_past_gt_dist) and writes
      results/pd2026_pr/confidence_selection_{variant}.json: per-statistic
      sequence-macro AP (mean over the 4 val seqs, then over seeds present),
      the §1.5 decision (max val macro AP; |gap| < 0.005 -> absolute; one
      statistic common to all seeds), and per-seed tau* = argmax val macro F1
      of the winning statistic (§1.4; ties -> highest tau, the pr_sweep
      max_F1_tau convention).

MEMORY DISCIPLINE (scripts/_pd_apply_frozen.py convention): the driver runs
each embedding-heavy sequence in its own --worker subprocess; aggregation
stages are torch-free.  Workers need docker ring_baseline (torch + src/data
for the _height_candidate_fusion import); --help / select-tuple / the
confidence aggregation run on the host.
"""
from __future__ import annotations
import argparse
import ast
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _causal_pr_dump import OPS, dump_op, prefix_topk  # noqa: E402 (torch-free)
from _pr_eval_lib import (  # noqa: E402  (torch-free)
    MIN_INDEX_GAP, classify_queries, pr_sweep,
)
from _xs_eval_lib import (  # noqa: E402  (torch-free)
    align_scores, column_align_scores, hspec_inputs, minmax, variant_inputs,
)

PD_CACHE = REPO / "data/preprocessed_pd2026"
OUT = REPO / "results/pd2026_pr"
SELECTION = OUT / "causal_tuple_selection.json"

DTH = 5.0  # <5 m revisit; asserted == _height_candidate_fusion.DTH in workers
VAL_SEQS = (("kitti", "09"), ("nclt", "2012-01-08"),
            ("helipr", "Roundabout03"), ("mulran", "Sejong03"))
SEQ_KEYS = [f"{s}/{q}" for s, q in VAL_SEQS]
VARIANT_CHOICES = ("pd_causal", "pd")
SEEDS = (1, 2, 3)
STATS = {"absolute": "score_abs", "margin": "score_margin"}  # §1.5 candidates
TIE_BREAK_DELTA = 0.005

TUPLE_RULE = (
    "per (variant, op): argmax over the shared _height_candidate_fusion."
    "WEIGHTS grid of the seed-summed query-weighted causal R@1 on the 4 pd "
    "val sequences: score(w) = sum_seed [ sum_seq n_q*r1 / sum_seq n_q ]; "
    "seeds enter only with all 4 sequences complete; ties -> first tuple in "
    "WEIGHTS order (Python max first-wins over the JSON insertion order). "
    "Mirrors scripts/_pd_tuple_select.py on the causal Track A val runs.")

CONFIDENCE_RULE = (
    "prereg §1.5: statistic = argmax validation sequence-macro AP (per seed: "
    "mean AP over the 4 val sequences; then mean over seeds present); "
    "tie-break |AP_absolute - AP_margin| < 0.005 -> absolute; the chosen "
    "statistic is common to all seeds. §1.4: tau* is per seed = argmax over "
    "the union of that seed's val sweep thresholds of the sequence-macro F1 "
    "of the winning statistic; ties -> highest tau.")


# ------------------------------------------------------------------- paths

def val_tuples_path(variant: str, op: str, seed: int,
                    sensor: str, seq: str) -> Path:
    return OUT / f"val_tuples_{variant}_{op}_s{seed}_{sensor}_{seq}.json"


def valdump_path(variant: str, op: str, seed: int,
                 sensor: str, seq: str) -> Path:
    return OUT / f"valdump_{variant}_{op}_s{seed}_{sensor}_{seq}.npz"


def resolve_val_caches(sensor: str, seq: str):
    """VAL-ONLY cache resolution (_pd_branch_fusion naming; never the
    test_-first preference of _pd_apply_frozen.resolve_caches)."""
    hits = sorted(glob.glob(str(PD_CACHE / f"cache_*_{sensor}_val_{seq}.npz")))
    assert len(hits) == 1, (sensor, seq, hits)
    lay = PD_CACHE / f"vallayouts_{sensor}_{seq}.npz"
    spec = PD_CACHE / f"valspec_{sensor}_{seq}.npz"
    for p in (lay, spec):
        assert p.is_file(), p
    return Path(hits[0]), lay, spec


def parse_pairs(seqs):
    pairs = []
    for s in seqs:
        assert s.count("/") == 1, f"--seqs entries must be sensor/seq, got {s!r}"
        sensor, seq = s.split("/")
        assert (sensor, seq) in VAL_SEQS, \
            f"{s!r} is not a Track A val sequence (allowed: {SEQ_KEYS})"
        pairs.append((sensor, seq))
    return pairs


def selected_tuple(variant: str, op: str):
    """Frozen causal tuple from the select-tuple stage; fails loudly."""
    if not SELECTION.is_file():
        raise SystemExit(
            f"{SELECTION} not found -- run 'select-tuple' before 'confidence'")
    sel = json.loads(SELECTION.read_text()).get("selections", {})
    key = f"{variant}/{op}"
    if key not in sel:
        raise SystemExit(
            f"{key!r} missing from {SELECTION} -- tuples stage incomplete")
    w = ast.literal_eval(sel[key]["selected_tuple"])
    assert isinstance(w, tuple) and len(w) == 4, sel[key]["selected_tuple"]
    return tuple(float(x) for x in w), sel[key]["selected_tuple"]


# ------------------------------------------------------------ worker (heavy)

def _load_val_sequence(sensor: str, seq: str):
    cache_p, lay_p, spec_p = resolve_val_caches(sensor, seq)
    print(f"[{sensor}/{seq}] cache={cache_p.name} layouts={lay_p.name} "
          f"spec={spec_p.name} marker=val", flush=True)
    cache = np.load(cache_p)
    lay = np.load(lay_p)
    spec = np.load(spec_p)
    assert (lay["scan_ids"] == cache["scan_ids"]).all(), (sensor, seq, "layouts")
    assert (spec["scan_ids"] == cache["scan_ids"]).all(), (sensor, seq, "spec")
    d288 = cache["descriptors"].astype(np.float32)
    layouts = lay["nsd_layouts"].astype(np.float32)
    sk = spec["sk_re"].astype(np.float32) + 1j * spec["sk_im"].astype(np.float32)
    raw_height = spec["raw_height"].astype(np.float32)
    return cache, d288, layouts, sk, raw_height, (cache_p, lay_p, spec_p)


def _causal_embed(args, sensor: str, seq: str, cache, d288, layouts):
    sys.path.insert(0, str(REPO / "src"))
    from _causal_branch_fusion import (
        VARIANTS, make_model_causal, nsd_embedding_causal,
    )
    V = VARIANTS[args.variant]
    cfg, model, clf, clf_blob, r, ckpt = make_model_causal(
        args.variant, args.seed, args.device)
    emb = nsd_embedding_causal(cache, sensor, d288, layouts,
                               cfg, model, clf, clf_blob, r, args.device)
    assert len(emb) == len(cache["poses"]), (sensor, seq, "embedding/pose length")
    artifacts = {"checkpoint": str(ckpt), "rvec": str(V["rvec"]),
                 "classifier": str(V["cls"]), "config": str(V["config"])}
    return emb, artifacts


def op_banks(op: str, sk, raw_height):
    return (variant_inputs(sk, 8) if op == "pbev128_main"
            else hspec_inputs(raw_height, 12))


def evaluate_tuple_grid(positions, emb, hkey, hfft, hnorm, column_bank,
                        queries, weights, pool_k: int):
    """_height_candidate_fusion.evaluate restricted to the WEIGHTS R@1 sweep
    with the causal candidate replacement: the global-then-filter
    _topk_cosine pools become two per-query gemv prefix pools over
    j <= t - 30 (_causal_pr_dump.prefix_topk) -> union; same min-max
    4-channel fusion, same explicit per-tuple score expression and first-max
    argmax (bit-consistent with the dump_op ledger for the selected tuple)."""
    correct = {str(w): 0 for w in weights}
    for qi, (q, _) in enumerate(queries):
        q = int(q)
        n_prefix = q - MIN_INDEX_GAP + 1  # prefix rows [0, t - 30]
        cn = prefix_topk(emb, q, n_prefix, pool_k)
        ch = prefix_topk(hkey, q, n_prefix, pool_k)
        cand = np.unique(np.concatenate([cn, ch]))
        assert cand.size and cand.max() <= q - MIN_INDEX_GAP, \
            "candidate beyond the causal prefix"
        good = np.linalg.norm(positions[cand] - positions[q], axis=1) < DTH
        sn = minmax(emb[cand] @ emb[q])
        sh = minmax(hkey[cand] @ hkey[q])
        sa = minmax(align_scores(hfft, hnorm, q, cand))
        sc = minmax(column_align_scores(column_bank, q, cand))
        for w in weights:
            score = w[0] * sn + w[1] * sh + w[2] * sa + w[3] * sc
            correct[str(w)] += bool(good[int(np.argmax(score))])
        if qi and qi % 500 == 0:
            print(f"  grid queries {qi}/{len(queries)}", flush=True)
    n = max(len(queries), 1)
    return {k: float(v / n) for k, v in correct.items()}


def run_tuples_worker(args, sensor: str, seq: str):
    sys.path.insert(0, str(REPO / "src"))
    from _height_candidate_fusion import (
        DTH as HCF_DTH, SKIP as HCF_SKIP, WEIGHTS,
    )
    from run_kitti_operating_point import _find_queries
    assert float(HCF_DTH) == DTH and int(HCF_SKIP) == MIN_INDEX_GAP, \
        "query constants drifted from _height_candidate_fusion"

    todo = [op for op in args.ops if args.overwrite or not val_tuples_path(
        args.variant, op, args.seed, sensor, seq).exists()]
    if not todo:
        print(f"[{sensor}/{seq}] all tuple outputs exist, skipping", flush=True)
        return
    cache, d288, layouts, sk, raw_height, (cache_p, lay_p, spec_p) = \
        _load_val_sequence(sensor, seq)
    emb, artifacts = _causal_embed(args, sensor, seq, cache, d288, layouts)
    # Scoring uses ground-truth poses (poses_gt on odometry-keyframe caches).
    gt_poses = cache["poses_gt"] if "poses_gt" in cache.files else cache["poses"]
    positions = gt_poses[:, :3, 3]
    # Causal revisit queries: positive within 5 m at >= 30 frames EARLIER
    # (_find_queries scans prior = [0, t - 30] only -- already past-only).
    queries = _find_queries(gt_poses, DTH, MIN_INDEX_GAP)
    print(f"[{sensor}/{seq}] n_queries={len(queries)} "
          f"grid={len(WEIGHTS)} tuples", flush=True)

    for op in todo:
        out_path = val_tuples_path(args.variant, op, args.seed, sensor, seq)
        hkey, hfft, hnorm, column_bank = op_banks(op, sk, raw_height)
        assert len(hkey) == len(emb), (sensor, seq, op, "magnitude bank length")
        r1 = evaluate_tuple_grid(positions, emb, hkey, hfft, hnorm,
                                 column_bank, queries, WEIGHTS, args.pool_k)
        payload = {
            "schema": "causal-val-tuples-v1",
            "variant": args.variant, "op": op, "seed": args.seed,
            "sensor": sensor, "sequence": seq,
            "n_queries": len(queries),
            "pool_k": args.pool_k, "min_index_gap": MIN_INDEX_GAP,
            "dth_m": DTH, "device": args.device,
            "artifacts": artifacts,
            "caches": {"cache": str(cache_p), "layouts": str(lay_p),
                       "spec": str(spec_p), "marker": "val"},
            "protocol": ("Track A causal tuple grid: causal two-pass "
                         "classifier-selected-edge embedding, revisit queries "
                         "(<5 m positive at >= 30 frames earlier, past-only), "
                         "two per-query gemv prefix top-k cosine pools over "
                         "j <= t - 30 -> union -> per-query min-max 4-channel "
                         "fusion, R@1 per _height_candidate_fusion.WEIGHTS "
                         "tuple (docs/CAUSAL_PR_PREREGISTRATION.md §2.3/§3A)"),
            # WEIGHTS insertion order is preserved here and drives the
            # select-tuple first-wins tie-break, as in _pd_tuple_select.
            "tuples": {k: {"r1": v, "n_queries": len(queries)}
                       for k, v in r1.items()},
        }
        out_path.write_text(json.dumps(payload, indent=2))
        best = max(WEIGHTS, key=lambda w: r1[str(w)])
        print(f"[{sensor}/{seq}] {op}: best={r1[str(best)]:.4f} w={best} "
              f"q={len(queries)} -> {out_path.name}", flush=True)


def run_confidence_worker(args, sensor: str, seq: str):
    sel = {op: selected_tuple(args.variant, op) for op in args.ops}
    todo = [op for op in args.ops if args.overwrite or not valdump_path(
        args.variant, op, args.seed, sensor, seq).exists()]
    if not todo:
        print(f"[{sensor}/{seq}] all valdumps exist, skipping", flush=True)
        return
    cache, d288, layouts, sk, raw_height, (cache_p, lay_p, spec_p) = \
        _load_val_sequence(sensor, seq)
    emb, artifacts = _causal_embed(args, sensor, seq, cache, d288, layouts)
    n = len(emb)
    # Scoring uses ground-truth poses (poses_gt on odometry-keyframe caches).
    gt_poses = cache["poses_gt"] if "poses_gt" in cache.files else cache["poses"]
    positions = gt_poses[:, :3, 3]
    # min 3D distance from t to ANY frame <= t - 30 (query classification
    # input; per-query prefix scan -- _causal_pr_dump.run_worker verbatim).
    min_past = np.full(n, np.nan)
    for t in range(MIN_INDEX_GAP, n):
        d = np.linalg.norm(positions[:t - MIN_INDEX_GAP + 1] - positions[t],
                           axis=1)
        min_past[t] = d.min()

    for op in todo:
        out_path = valdump_path(args.variant, op, args.seed, sensor, seq)
        w, w_str = sel[op]
        print(f"[{sensor}/{seq}] op={op} tuple={w_str} "
              f"source={SELECTION.name}", flush=True)
        hkey, hfft, hnorm, column_bank = op_banks(op, sk, raw_height)
        assert len(hkey) == n, (sensor, seq, op, "magnitude bank length")
        arrays = dump_op(emb, hkey, hfft, hnorm, column_bank,
                         positions, min_past, w, args.pool_k)
        meta = {
            "schema": "pr-dump-v1",
            "stage": "trackA-val",
            "variant": args.variant, "op": op, "seed": args.seed,
            "sensor": sensor, "sequence": seq,
            "tuple": list(w), "tuple_str": w_str,
            "tuple_source": f"{SELECTION} [{args.variant}/{op}]",
            "pool_k": args.pool_k, "min_index_gap": MIN_INDEX_GAP,
            "n_frames": n, "n_queries": int(len(arrays["query_idx"])),
            "device": args.device,
            "artifacts": artifacts,
            "caches": {"cache": str(cache_p), "layouts": str(lay_p),
                       "spec": str(spec_p), "marker": "val"},
            "protocol": ("causal all-frames VAL PR ledger (track A): causal "
                         "two-pass classifier-selected-edge embedding, "
                         "per-query gemv prefix candidates (two top-k cosine "
                         "pools over j <= t - 30, union), per-query min-max "
                         "4-channel fusion at the FROZEN causal tuple; "
                         "scores/dists per docs/CAUSAL_PR_PREREGISTRATION.md "
                         "§1.2/§1.5"),
        }
        np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
        fin = np.isfinite(arrays["score_abs"])
        print(f"[{sensor}/{seq}] {op}: n_queries={len(arrays['query_idx'])} "
              f"finite={int(fin.sum())} -> {out_path.name}", flush=True)


# ------------------------------------------------------- select-tuple stage

def cmd_select_tuple(args):
    if SELECTION.exists() and not args.overwrite:
        keys = list(json.loads(SELECTION.read_text()).get("selections", {}))
        print(f"{SELECTION} exists (selections: {keys}); frozen -- "
              "pass --overwrite to re-select", flush=True)
        return
    selections = {}
    for variant in VARIANT_CHOICES:
        for op in OPS:
            seed_reps = {}
            for seed in SEEDS:
                paths = {f"{s}/{q}": val_tuples_path(variant, op, seed, s, q)
                         for s, q in VAL_SEQS}
                have = {k: p for k, p in paths.items() if p.exists()}
                if len(have) == len(paths):
                    seed_reps[seed] = {k: json.loads(p.read_text())
                                       for k, p in paths.items()}
                elif have:
                    print(f"--- {variant}/{op} seed{seed}: only "
                          f"{len(have)}/{len(paths)} sequences, seed excluded")
            if not seed_reps:
                print(f"--- {variant}/{op}: no complete seed, skipped")
                continue
            seeds_ok = sorted(seed_reps)
            ws = [k for k in seed_reps[seeds_ok[0]][SEQ_KEYS[0]]["tuples"]
                  if k.startswith("(")]
            nq = {}
            for k in SEQ_KEYS:
                vals = {s: seed_reps[s][k]["n_queries"] for s in seeds_ok}
                assert len(set(vals.values())) == 1, \
                    (k, vals, "n_queries differs across seeds")
                nq[k] = vals[seeds_ok[0]]
                for s in seeds_ok:
                    tk = [x for x in seed_reps[s][k]["tuples"]
                          if x.startswith("(")]
                    assert tk == ws, (variant, op, s, k, "tuple grid mismatch")

            def agg(rep, w):
                """_pd_tuple_select.agg: query-weighted pooled R@1, one seed."""
                n = d = 0.0
                for k in SEQ_KEYS:
                    x = rep[k]["tuples"][w]
                    n += x["n_queries"] * x["r1"]
                    d += x["n_queries"]
                return n / d

            score = {w: sum(agg(seed_reps[s], w) for s in seeds_ok)
                     for w in ws}
            # Python max is first-wins: ties -> first tuple in WEIGHTS order.
            w_sel = max(ws, key=lambda w: score[w])
            ranked = sorted(ws, key=lambda w: -score[w])  # stable: ties keep order
            rq_per_seed = [agg(seed_reps[s], w_sel) for s in seeds_ok]
            per_seq_mean = {
                k: float(np.mean([seed_reps[s][k]["tuples"][w_sel]["r1"]
                                  for s in seeds_ok]))
                for k in SEQ_KEYS}
            selections[f"{variant}/{op}"] = {
                "selected_tuple": w_sel,
                "tuple": [float(x) for x in ast.literal_eval(w_sel)],
                "n_seeds": len(seeds_ok), "seeds": seeds_ok,
                "selection_pool": SEQ_KEYS,
                "summed_Rq": score[w_sel],
                "Rq_per_seed": rq_per_seed,
                "Rq_mean": float(np.mean(rq_per_seed)),
                "per_sequence_mean": per_seq_mean,
                "n_queries": nq,
                "top5": [{"tuple": w, "summed_Rq": score[w],
                          "Rq_mean": score[w] / len(seeds_ok)}
                         for w in ranked[:5]],
            }
            print(f"--- {variant}/{op} ({len(seeds_ok)} seeds) w={w_sel}")
            print(f"  Rq_mean={float(np.mean(rq_per_seed)):.5f}  "
                  + "  ".join(f"{k}={per_seq_mean[k]:.4f}" for k in SEQ_KEYS))
    if not selections:
        raise SystemExit("no complete (variant, op) group -- "
                         "run the tuples stage first")
    payload = {"schema": "causal-tuple-selection-v1",
               "rule": TUPLE_RULE,
               "tie_break": "first tuple in WEIGHTS order (max first-wins)",
               "selection_pool": SEQ_KEYS,
               "selections": selections}
    SELECTION.write_text(json.dumps(payload, indent=2))
    print(f"wrote {SELECTION}")
    print("CAUSAL_TUPLE_SELECT_DONE", flush=True)


# --------------------------------------------------- confidence aggregation

def _f1_curve(sweep):
    p = np.asarray(sweep["precision"])
    r = np.asarray(sweep["recall"])
    s = p + r
    return np.where(s > 0, 2.0 * p * r / np.maximum(s, 1e-300), 0.0)


def _macro_f1_tau_star(seq_sweeps):
    """Per-seed tau* (§1.4): argmax over the union of the seed's val sweep
    thresholds of the sequence-macro F1; ties -> highest tau (the pr_sweep
    max_F1_tau convention).  Each sequence's F1 is a right-continuous step
    function of tau (accepted = score >= tau): F1_seq(tau) = F1 at the
    smallest sweep threshold >= tau, so the union of unique thresholds
    covers every achievable macro-F1 value."""
    cand = np.unique(np.concatenate(
        [s["thresholds"][np.isfinite(s["thresholds"])] for s in seq_sweeps]))
    assert cand.size, "no finite sweep thresholds on val (all-NaN scores?)"
    per = []
    for s in seq_sweeps:
        asc_t = s["thresholds"][::-1]  # strictly ascending, ends at +inf
        f1_asc = _f1_curve(s)[::-1]
        idx = np.searchsorted(asc_t, cand, side="left")
        per.append(f1_asc[idx])
    per = np.stack(per)
    macro = per.mean(axis=0)
    i = len(macro) - 1 - int(np.argmax(macro[::-1]))  # ties -> highest tau
    return float(cand[i]), float(macro[i]), [float(x) for x in per[:, i]]


def cmd_confidence_aggregate(args):
    sel_out = OUT / f"confidence_selection_{args.variant}.json"
    if sel_out.exists() and not args.overwrite:
        print(f"{sel_out} exists; frozen -- pass --overwrite to re-aggregate",
              flush=True)
        return
    ops_out = {}
    for op in OPS:
        try:
            w, w_str = selected_tuple(args.variant, op)
        except SystemExit as e:
            print(f"--- {args.variant}/{op}: {e}; skipped")
            continue
        seeds_present = []
        sweeps = {}
        for seed in SEEDS:
            paths = {f"{s}/{q}": valdump_path(args.variant, op, seed, s, q)
                     for s, q in VAL_SEQS}
            if not all(p.exists() for p in paths.values()):
                if any(p.exists() for p in paths.values()):
                    print(f"--- {args.variant}/{op} seed{seed}: partial "
                          "valdumps, seed excluded")
                continue
            seeds_present.append(seed)
            sweeps[seed] = {}
            for key, p in paths.items():
                z = np.load(p)
                meta = json.loads(str(z["meta"]))
                assert [float(x) for x in meta["tuple"]] == list(w), (
                    p.name, "stale valdump: tuple != current selection -- "
                    "re-run confidence with --overwrite")
                qc = classify_queries(z["min_past_gt_dist"])
                sweeps[seed][key] = {
                    stat: pr_sweep(z[col], z["top1_geo_dist"], qc)
                    for stat, col in STATS.items()}
        if not seeds_present:
            print(f"--- {args.variant}/{op}: no complete seed, skipped")
            continue

        counts = {}
        for key in SEQ_KEYS:
            c0 = sweeps[seeds_present[0]][key]["absolute"]["counts"]
            for seed in seeds_present[1:]:
                assert sweeps[seed][key]["absolute"]["counts"] == c0, \
                    (key, "query classification differs across seeds")
            counts[key] = c0

        per_stat = {}
        for stat in STATS:
            per_seq = {str(seed): {k: sweeps[seed][k][stat]["AP"]
                                   for k in SEQ_KEYS}
                       for seed in seeds_present}
            per_seed_macro = {
                s: float(np.mean(list(aps.values())))
                for s, aps in per_seq.items()}
            per_stat[stat] = {
                "per_sequence_AP": per_seq,
                "per_seed_macro_AP": per_seed_macro,
                "macro_AP": float(np.mean(list(per_seed_macro.values()))),
            }
        ap_abs = per_stat["absolute"]["macro_AP"]
        ap_mar = per_stat["margin"]["macro_AP"]
        gap = abs(ap_abs - ap_mar)
        tie = gap < TIE_BREAK_DELTA
        decision = "absolute" if (tie or ap_abs >= ap_mar) else "margin"

        tau_star = {}
        for seed in seeds_present:
            seq_sweeps = [sweeps[seed][k][decision] for k in SEQ_KEYS]
            tau, mf1, per_f1 = _macro_f1_tau_star(seq_sweeps)
            tau_star[str(seed)] = {"tau": tau, "val_macro_F1": mf1,
                                   "per_sequence_F1": dict(zip(SEQ_KEYS,
                                                               per_f1))}
        ops_out[op] = {
            "tuple": list(w), "tuple_str": w_str,
            "seeds": seeds_present,
            "per_statistic": per_stat,
            "decision": decision,
            "ap_absolute": ap_abs, "ap_margin": ap_mar,
            "ap_gap": gap, "tie_break_applied": tie,
            "tau_star": tau_star,
            "counts": counts,
        }
        print(f"--- {args.variant}/{op} ({len(seeds_present)} seeds): "
              f"AP abs={ap_abs:.4f} margin={ap_mar:.4f} gap={gap:.4f} "
              f"-> {decision}" + (" (tie-break)" if tie else ""))
        for s, t in tau_star.items():
            print(f"  seed{s}: tau*={t['tau']:.6f} "
                  f"val_macro_F1={t['val_macro_F1']:.4f}")
    if not ops_out:
        raise SystemExit("no complete (op, seed) valdump group -- "
                         "run the confidence dump stage first")
    payload = {
        "schema": "causal-confidence-selection-v1",
        "variant": args.variant,
        "rule": CONFIDENCE_RULE,
        "tie_break_delta": TIE_BREAK_DELTA,
        "selection_pool": SEQ_KEYS,
        "statistics": list(STATS),
        "ops": ops_out,
    }
    sel_out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {sel_out}")
    print("CAUSAL_CONFIDENCE_SELECT_DONE", flush=True)


# ------------------------------------------------------------------ driver

def _spawn_workers(args, pairs):
    """One embedding-heavy sequence per subprocess (memory discipline;
    _pd_apply_frozen --worker pattern)."""
    path_fn = val_tuples_path if args.cmd == "tuples" else valdump_path
    for sensor, seq in pairs:
        outs = [path_fn(args.variant, op, args.seed, sensor, seq)
                for op in args.ops]
        if not args.overwrite and all(p.exists() for p in outs):
            print(f"[{sensor}/{seq}] all outputs exist, skipping worker",
                  flush=True)
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), args.cmd,
               "--variant", args.variant, "--seed", str(args.seed),
               "--seqs", f"{sensor}/{seq}", "--ops", *args.ops,
               "--device", args.device, "--pool-k", str(args.pool_k),
               "--worker"]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"=== spawning worker: {sensor}/{seq} ===", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"WORKER_FAILED {sensor}/{seq} rc={rc}", flush=True)
            sys.exit(rc)


def main():
    ap = argparse.ArgumentParser(
        description="Track A (validation-phase) runner for the pre-registered "
                    "causal PR campaign (docs/CAUSAL_PR_PREREGISTRATION.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--variant", choices=VARIANT_CHOICES, required=True,
                       help="pd_causal = causal-retrained; pd = Frozen-GNN "
                            "causal adaptation")
        p.add_argument("--seed", type=int, choices=SEEDS, required=True)
        p.add_argument("--seqs", nargs="+", default=list(SEQ_KEYS),
                       metavar="SENSOR/SEQ",
                       help="subset of the 4 Track A val sequences "
                            "(default: all)")
        p.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
        p.add_argument("--device", default="cuda")
        p.add_argument("--pool-k", type=int, default=100)
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--worker", action="store_true",
                       help="internal: run a single sequence in-process")

    p_t = sub.add_parser(
        "tuples", help="per-sequence causal R@1 over the WEIGHTS tuple grid")
    add_common(p_t)
    p_s = sub.add_parser(
        "select-tuple", help="aggregate val_tuples JSONs -> frozen tuple "
                             "selection (_pd_tuple_select rule)")
    p_s.add_argument("--overwrite", action="store_true")
    p_c = sub.add_parser(
        "confidence", help="valdump ledgers at the frozen tuples + §1.5 "
                           "statistic selection + per-seed tau*")
    add_common(p_c)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.cmd == "select-tuple":
        cmd_select_tuple(args)
        return

    pairs = parse_pairs(args.seqs)
    if args.cmd == "confidence":
        for op in args.ops:  # fail loudly before any heavy work
            selected_tuple(args.variant, op)

    if args.worker:
        assert len(pairs) == 1, "--worker takes exactly one sequence"
        worker = (run_tuples_worker if args.cmd == "tuples"
                  else run_confidence_worker)
        worker(args, *pairs[0])
        return

    _spawn_workers(args, pairs)
    if args.cmd == "tuples":
        print("CAUSAL_VAL_TUPLES_DONE", flush=True)
    else:
        cmd_confidence_aggregate(args)
        print("CAUSAL_VAL_CONFIDENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
