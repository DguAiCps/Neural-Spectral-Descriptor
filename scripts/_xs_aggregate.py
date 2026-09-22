#!/usr/bin/env python3
"""Aggregate cross-session results (xs-eval-v1 JSONs) into the body and
appendix tables.

Gates: within each protocol, every result JSON must carry the identical
roster_sha256 (method-invariant roster) — hard fail otherwise.

Body table (site columns = mean over the site's protocols; site-balanced
macro = mean of the four site values): R@1@5m (primary), R@1@20m (auxiliary),
R@1%@20m. NSD rows aggregate seeds 1-3 as mean +/- std (per-seed site/macro
first, then moments over seeds); baselines are deterministic / single
checkpoint.

Appendix: per-protocol full metric dump, R@1%@{5,10,15,20}m, R@{5,10}, and
the coverage table (matchable@t / n_roster — the structural ceilings, e.g.
riverside_01_03 and the ~0.5 ceilings implied by the 20 m map spacing).

Output: results/pd2026_xsession/xs_summary.json + markdown to stdout.
"""
from __future__ import annotations
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import _xs_eval_lib as lib

BODY_METHODS = ["scpp", "bevplace", "seqot", "ring", "ringpp",
                "ringsharp", "imlpr", "helios", "imlpr_pd", "nsd", "nsdh"]
AUX_METHODS = ["nsdh_pdnn", "nsd_pdnn", "nsd_keyonly"]
SEEDED = {"nsd", "nsdh", "nsd_pdnn", "nsdh_pdnn", "nsd_keyonly"}
BODY_COLS = ["R@1@5m", "R@1@20m", "R1pct@20m"]
LABELS = {"scpp": "SC++", "bevplace": "BEVPlace++", "seqot": "SeqOT",
          "imlpr": "ImLPR (zero-shot)", "helios": "HeLiOS (zero-shot)",
          "imlpr_pd": "ImLPR (pd retrain)",
          "ring": "RING", "ringpp": "RING++", "ringsharp": "RING#",
          "nsd": "NSD", "nsdh": "NSD-H",
          "nsdh_pdnn": "NSD-H (no NCLT training)",
          "nsd_pdnn": "NSD (no NCLT training)",
          "nsd_keyonly": "NSD key-only (coarse)"}
SITE_ORDER = ["nclt", "dcc", "kaist", "riverside"]


def load_results():
    files = sorted(lib.OUT_DIR.glob("xs_*.json"))
    assert files, f"no xs_*.json under {lib.OUT_DIR}"
    per = defaultdict(dict)      # (method, protocol) -> {seed_or_0: payload}
    roster_sha = defaultdict(set)
    for f in files:
        d = json.loads(f.read_text())
        if d.get("schema") != "xs-eval-v1":     # e.g. xs_summary.json itself
            continue
        seed = d.get("seed", 0)
        per[(d["method"], d["protocol"])][seed] = d
        roster_sha[d["protocol"]].add(d["roster_sha256"])
    for protocol, shas in sorted(roster_sha.items()):
        assert len(shas) == 1, (
            f"ROSTER MISMATCH in {protocol}: {sorted(shas)} — methods were "
            "evaluated on different rosters; refusing to aggregate")
    return per


def method_metric(per, method, protocol, metric):
    """-> (mean, std_or_None, n_seeds) or None if absent."""
    entry = per.get((method, protocol))
    if not entry:
        return None
    vals = [d["metrics"][metric] for _, d in sorted(entry.items())]
    if method in SEEDED and len(vals) > 1:
        return (statistics.mean(vals), statistics.stdev(vals), len(vals))
    return (vals[0], None, len(vals))


def site_value(per, method, site, metric):
    """Per-seed site mean first, then moments over seeds."""
    protos = lib.SITES[site]
    entries = [per.get((method, p)) for p in protos]
    if any(e is None for e in entries):
        return None
    seed_sets = [set(e) for e in entries]
    common = set.intersection(*seed_sets)
    if not common:
        return None
    site_means = [statistics.mean(e[s]["metrics"][metric] for e in entries)
                  for s in sorted(common)]
    if method in SEEDED and len(site_means) > 1:
        return (statistics.mean(site_means), statistics.stdev(site_means),
                len(site_means))
    return (site_means[0], None, len(site_means))


def macro_value(per, method, metric):
    protos_by_site = [lib.SITES[s] for s in SITE_ORDER]
    entries = {p: per.get((method, p)) for site in protos_by_site for p in site}
    if any(e is None for e in entries.values()):
        return None
    common = set.intersection(*(set(e) for e in entries.values()))
    if not common:
        return None
    macros = []
    for s in sorted(common):
        site_means = [statistics.mean(entries[p][s]["metrics"][metric]
                                      for p in lib.SITES[site])
                      for site in SITE_ORDER]
        macros.append(statistics.mean(site_means))
    if method in SEEDED and len(macros) > 1:
        return (statistics.mean(macros), statistics.stdev(macros), len(macros))
    return (macros[0], None, len(macros))


def fmt(v, with_std=True):
    if v is None:
        return "--"
    mean, std, _ = v
    if std is not None and with_std:
        return f"{mean:.3f}±{std:.3f}"
    return f"{mean:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(lib.OUT_DIR / "xs_summary.json"))
    args = ap.parse_args()
    per = load_results()

    present = sorted({m for (m, _) in per})
    missing_body = [m for m in BODY_METHODS if m not in present]
    if missing_body:
        print(f"NOTE: body methods not yet evaluated: {missing_body}\n")

    summary = {"_meta": {
        "protocol_name": lib.PROTOCOL_NAME,
        "naming_note": lib._SPEC["naming_note"],
        "macro": "site-balanced macro (mean of nclt/dcc/kaist/riverside site "
                 "values; site value = mean over its 01->02 / 01->03 pairs)",
        "constants": lib.CONST,
        "seeded_methods": sorted(SEEDED),
    }, "body": {}, "sites": {}, "protocols": {}, "coverage": {}}

    # ---------- body table ----------
    print("## Body table — RING-style cross-session retrieval "
          "(site-balanced macro)\n")
    hdr = "| Method | " + " | ".join(s.upper() for s in SITE_ORDER) + " | Macro |"
    for metric in BODY_COLS:
        print(f"### {metric}")
        print(hdr)
        print("|" + "---|" * (len(SITE_ORDER) + 2))
        for method in BODY_METHODS + AUX_METHODS:
            if method not in present:
                continue
            cells = [fmt(site_value(per, method, s, metric)) for s in SITE_ORDER]
            mac = fmt(macro_value(per, method, metric))
            print(f"| {LABELS[method]} | " + " | ".join(cells) + f" | {mac} |")
            summary["body"].setdefault(metric, {})[method] = {
                "sites": {s: site_value(per, method, s, metric)
                          for s in SITE_ORDER},
                "macro": macro_value(per, method, metric),
            }
        print()

    # ---------- appendix: per-protocol all metrics ----------
    all_metrics = None
    for (method, protocol), entry in sorted(per.items()):
        d = entry[sorted(entry)[0]]
        if all_metrics is None:
            all_metrics = [k for k in d["metrics"] if k != "k_1pct"]
        vals = {}
        for metric in all_metrics:
            mv = method_metric(per, method, protocol, metric)
            vals[metric] = mv
        summary["protocols"].setdefault(protocol, {})[method] = {
            "metrics": vals, "k_1pct": d["metrics"]["k_1pct"],
            "n_db": d["db"]["n_db"], "n_roster": d["query"]["n_roster"],
        }

    # ---------- coverage / structural ceilings ----------
    print("## Coverage (structural ceilings; identical for every method)\n")
    print("| Protocol | n_db | n_roster | cov@5m | cov@10m | cov@15m | cov@20m |")
    print("|---|---|---|---|---|---|---|")
    for protocol in lib.PROTOCOLS:
        entry = next((per[(m, protocol)] for m in present
                      if (m, protocol) in per), None)
        if not entry:
            continue
        d = entry[sorted(entry)[0]]
        dg, n = d["diagnostics"], max(d["query"]["n_roster"], 1)
        cov = {t: dg[f"matchable@{t}m"] / n for t in (5, 10, 15, 20)}
        summary["coverage"][protocol] = {
            "n_db": d["db"]["n_db"], "n_roster": d["query"]["n_roster"],
            **{f"cov@{t}m": cov[t] for t in (5, 10, 15, 20)}}
        print(f"| {protocol} | {d['db']['n_db']} | {d['query']['n_roster']} | "
              + " | ".join(f"{cov[t]:.3f}" for t in (5, 10, 15, 20)) + " |")
    print()

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
