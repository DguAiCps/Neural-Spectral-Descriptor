#!/usr/bin/env python3
"""Aggregate the causal PR campaign ledgers into the pre-registered tables.

Consumes results/pd2026_pr/dump_*.npz (NSD tracks + baselines) through
scripts/_pr_eval_lib.py under the frozen accounting
(docs/CAUSAL_PR_PREREGISTRATION.md §1.2-1.4, §1.6): query classification on
min_past_gt_dist (positive <=5 m / ambiguous (5,25] excluded / negative
>25 m), match correctness = top1 <= 5 m only, fixed recall denominator.
Metrics per sequence: AP, max F1, R@100P, R@99P, R@95P. Aggregation:
Strict-v2 six-group sequence-macro -> overall = macro of group means. NSD
rows aggregate seeds as mean +/- std; the NSD decision score is the FROZEN
val selection (confidence_selection_{variant}.json: absolute), and the
operating-point row applies each seed's frozen tau* from the same file.
Baselines use their native score arrays (mapping below), single run.

Output: results/pd2026_pr/pr_summary.json + markdown to stdout.
"""
from __future__ import annotations
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

import _pr_eval_lib as prl

REPO = Path(__file__).resolve().parents[1]
DUMPS = REPO / "results/pd2026_pr"
MANIFEST = REPO / "scripts/_pd_test_cache_manifest.json"

STRICT_EXCLUDE = {("kitti", "00"), ("kitti360", "0004"),
                  ("helipr", "Bridge04"),
                  ("nclt", "2012-02-04"), ("nclt", "2012-03-17")}
GROUP_OF = {}          # (sensor, seq) -> group label, filled in main
GROUPS = ["kitti", "helipr_town", "helipr_daejeon", "mulran", "kitti360", "mcd"]

# method -> (dump name pattern method slug, score array key)
BASELINES = {
    "scpp": ("scpp", "score_scpp"),
    "bevplace": ("bevplace", "score_cos"),
    "seqot": ("seqot", "score_cos"),
    "ringsharp": ("ringsharp", "score_trans"),
    "imlpr": ("imlpr", "score_cos"),
    "helios": ("helios", "score_cos"),
    "imlpr_pd": ("imlpr_pd", "score_cos"),
    "ring": ("ring", "score_corr"),
    "ringpp": ("ringpp", "score_corr"),
}
NSD_TRACKS = {
    "nsd_causal": ("pd_causal", "pbev128_main"),
    "nsdh_causal": ("pd_causal", "hspec12"),
    "nsd_frozen": ("pd", "pbev128_main"),
    "nsdh_frozen": ("pd", "hspec12"),
}
METRICS = ("AP", "max_F1", "R@100P", "R@99P", "R@95P")


def strict_roster():
    m = json.loads(MANIFEST.read_text())
    seqs = [(e["sensor"], e["sequence"]) for e in m["sequences"]]
    out = [s for s in seqs if s not in STRICT_EXCLUDE]
    assert len(out) == 37, len(out)
    for sensor, seq in out:
        if sensor == "helipr":
            GROUP_OF[(sensor, seq)] = ("helipr_town" if seq.startswith("Town")
                                       else "helipr_daejeon")
        else:
            GROUP_OF[(sensor, seq)] = sensor
    return out


def sweep_dump(path: Path, score_key: str):
    z = np.load(path, allow_pickle=True)
    scores = z[score_key].astype(np.float64)
    top1 = z["top1_geo_dist"].astype(np.float64)
    minp = z["min_past_gt_dist"].astype(np.float64)
    qc = prl.classify_queries(minp)
    return prl.pr_sweep(scores, top1, qc), qc, scores, top1


def seq_metrics(sweep) -> dict:
    return {m: float(sweep[m]) for m in METRICS}


def group_macro(per_seq: dict) -> dict:
    """per_seq: (sensor,seq) -> {metric: value} -> group means + overall macro."""
    by_group = defaultdict(list)
    for key, vals in per_seq.items():
        by_group[GROUP_OF[key]].append(vals)
    out = {}
    for g in GROUPS:
        rows = by_group.get(g, [])
        if rows:
            out[g] = {m: statistics.mean(r[m] for r in rows) for m in METRICS}
    if len(out) == len(GROUPS):
        out["macro"] = {m: statistics.mean(out[g][m] for g in GROUPS)
                        for m in METRICS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DUMPS / "pr_summary.json"))
    args = ap.parse_args()
    roster = strict_roster()
    summary = {"_meta": {
        "protocol": "pre-registered causal PR (docs/CAUSAL_PR_PREREGISTRATION.md "
                    "rev.2+A1); fixed-roster accounting via scripts/_pr_eval_lib.py; "
                    "Strict-v2 six-group macro; NSD scores = frozen val-selected "
                    "'absolute' statistic; baselines = native scores",
    }, "methods": {}}

    # ---------------- NSD tracks (3 seeds, frozen tau*) ----------------
    for label, (variant, op) in NSD_TRACKS.items():
        conf_p = DUMPS / f"confidence_selection_{variant}.json"
        conf = json.loads(conf_p.read_text())["ops"][op]
        assert conf["decision"] == "absolute", (label, conf["decision"])
        taus = {int(s): e["tau"] for s, e in conf["tau_star"].items()} \
            if isinstance(conf.get("tau_star"), dict) else {
                int(s): e["tau"] for s, e in conf["tau_star_per_seed"].items()}
        seed_groups, seed_ops = {}, {}
        missing = []
        for seed in (1, 2, 3):
            per_seq, per_seq_op = {}, {}
            for sensor, seq in roster:
                p = DUMPS / f"dump_{variant}_{op}_s{seed}_{sensor}_{seq}.npz"
                if not p.exists():
                    missing.append(p.name)
                    continue
                sweep, qc, scores, top1 = sweep_dump(p, "score_abs")
                per_seq[(sensor, seq)] = seq_metrics(sweep)
                o = prl.operating_point(scores, top1, qc, taus[seed])
                per_seq_op[(sensor, seq)] = {k: float(o[k])
                                             for k in ("precision", "recall", "F1")}
            if per_seq:
                seed_groups[seed] = group_macro(per_seq)
                by_g = defaultdict(list)
                for key, v in per_seq_op.items():
                    by_g[GROUP_OF[key]].append(v)
                gmeans = {g: {k: statistics.mean(r[k] for r in rows)
                              for k in ("precision", "recall", "F1")}
                          for g, rows in by_g.items()}
                if len(gmeans) == len(GROUPS):
                    seed_ops[seed] = {k: statistics.mean(gmeans[g][k] for g in GROUPS)
                                      for k in ("precision", "recall", "F1")}
        if missing:
            print(f"NOTE {label}: {len(missing)} dumps missing (e.g. {missing[0]})")
        if not seed_groups:
            continue
        entry = {"variant": variant, "op": op, "seeds": sorted(seed_groups),
                 "tau_star": taus, "per_seed": {}}
        for seed, g in seed_groups.items():
            entry["per_seed"][seed] = g
        macros = {m: [g["macro"][m] for g in seed_groups.values() if "macro" in g]
                  for m in METRICS}
        entry["macro_mean"] = {m: statistics.mean(v) for m, v in macros.items() if v}
        entry["macro_std"] = {m: (statistics.stdev(v) if len(v) > 1 else 0.0)
                              for m, v in macros.items() if v}
        if seed_ops:
            entry["operating_point"] = {
                "mean": {k: statistics.mean(o[k] for o in seed_ops.values())
                         for k in ("precision", "recall", "F1")},
                "std": {k: (statistics.stdev([o[k] for o in seed_ops.values()])
                            if len(seed_ops) > 1 else 0.0)
                        for k in ("precision", "recall", "F1")},
            }
        summary["methods"][label] = entry

    # ---------------- baselines (single run, native scores) ----------------
    for label, (slug, score_key) in BASELINES.items():
        per_seq = {}
        n_found = 0
        for sensor, seq in roster:
            p = DUMPS / f"dump_{slug}_{sensor}_{seq}.npz"
            if not p.exists():
                continue
            sweep, _, _, _ = sweep_dump(p, score_key)
            per_seq[(sensor, seq)] = seq_metrics(sweep)
            n_found += 1
        if not per_seq:
            continue
        g = group_macro(per_seq)
        summary["methods"][label] = {
            "score_key": score_key, "n_sequences": n_found, "groups": g}
        if n_found < 37:
            summary["methods"][label]["incomplete"] = True

    Path(args.out).write_text(json.dumps(summary, indent=2))

    # ---------------- markdown ----------------
    print("\n## Causal PR — Strict-v2 six-group macro (AP / maxF1 / R@100P)\n")
    print("| Method | AP | max F1 | R@100P | R@99P | R@95P |")
    print("|---|---|---|---|---|---|")
    order = ["scpp", "bevplace", "seqot", "ringsharp", "imlpr", "helios", "imlpr_pd", "ring", "ringpp",
             "nsd_frozen", "nsdh_frozen", "nsd_causal", "nsdh_causal"]
    labels = {"scpp": "SC++", "bevplace": "BEVPlace++", "seqot": "SeqOT (trailing)",
              "imlpr": "ImLPR (zero-shot)", "helios": "HeLiOS (zero-shot)",
              "imlpr_pd": "ImLPR (pd retrain)",
              "ringsharp": "RING#-L (pd)", "ring": "RING", "ringpp": "RING++",
              "nsd_frozen": "NSD Frozen-GNN", "nsdh_frozen": "NSD-H Frozen-GNN",
              "nsd_causal": "NSD causal-retrained",
              "nsdh_causal": "NSD-H causal-retrained"}
    for key in order:
        e = summary["methods"].get(key)
        if not e:
            continue
        if "macro_mean" in e:
            cells = [f"{e['macro_mean'][m]:.3f}±{e['macro_std'][m]:.3f}"
                     for m in METRICS]
        else:
            mac = e["groups"].get("macro")
            if not mac:
                cells = ["(incomplete)"] * len(METRICS)
            else:
                cells = [f"{mac[m]:.3f}" for m in METRICS]
        print(f"| {labels[key]} | " + " | ".join(cells) + " |")
    for key in ("nsd_causal", "nsdh_causal", "nsd_frozen", "nsdh_frozen"):
        e = summary["methods"].get(key)
        if e and "operating_point" in e:
            op = e["operating_point"]["mean"]
            print(f"| {labels[key]} @tau* | P={op['precision']:.3f} "
                  f"R={op['recall']:.3f} F1={op['F1']:.3f} |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
