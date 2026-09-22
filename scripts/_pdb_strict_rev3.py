#!/usr/bin/env python3
"""Strict-v2 revision 3: swap the RING# column to the pd2026 retrain.

Protocol unchanged from Strict-v2 (37-seq roster, six-group-balanced macro;
excluded+disclosed: kitti/00, kitti360/0004, helipr/Bridge04, nclt pair) —
per advisor guidance the protocol name stays Strict-v2 and only the file
revision increments. The only change vs summary_strict.json (revision 2) is
the ringsharp entry: paper-era zero-shot weights (10 contaminated sequences
excluded via train_overlap) are replaced by the clean pd2026 retrain
evaluated on the full roster (results/pd2026_baselines/ringsharp_pd_*.json;
checkpoint selected by the pre-declared pd-val rule, sha256 recorded in
ringsharp_pd_ckpt_selection.json).

Validation: the group aggregation is recomputed from per-sequence JSONs for
scpp and ring and asserted equal to the committed revision-2 values before
the ringsharp swap is trusted.

Output: results/pd2026_baselines/summary_strict_v2_revision3.json
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "pd2026_baselines"
MANIFEST = REPO / "scripts" / "_pd_test_cache_manifest.json"

STRICT_EXCLUDE = {("kitti", "00"), ("kitti360", "0004"),
                  ("helipr", "Bridge04"),
                  ("nclt", "2012-02-04"), ("nclt", "2012-03-17")}

GROUP_LABEL = {
    "kitti": "kitti_pd (05/08)",
    "helipr_town": "helipr_town (01-03)",
    "helipr_daejeon": "helipr_daejeon",
    "mulran": "mulran_daejeon",
    "kitti360": "kitti360 (0000/0006)",
    "mcd": "mcd_zeroshot",
}


def group_of(sensor: str, seq: str) -> str:
    if sensor == "helipr":
        return GROUP_LABEL["helipr_town" if seq.startswith("Town")
                           else "helipr_daejeon"]
    return GROUP_LABEL[sensor]


def strict_roster():
    m = json.loads(MANIFEST.read_text())
    seqs = [(e["sensor"], e["sequence"]) for e in m["sequences"]]
    return [s for s in seqs if s not in STRICT_EXCLUDE]


def r1_of(method: str, sensor: str, seq: str) -> float:
    """Per-sequence R@1 across the differing result-JSON conventions."""
    p = OUT_DIR / f"{method}_{sensor}_{seq}.json"
    d = json.loads(p.read_text())
    if method == "bevplace":
        return float(d["recall_at_1"])
    if f"{sensor}_{seq}" in d:                      # ringsharp/seqot/ringsharp_pd
        return float(d[f"{sensor}_{seq}"]["recall@1"])
    return float(d["recall@1"])                     # scpp/ring/ringpp


def aggregate(method: str, roster) -> dict:
    groups = {}
    for sensor, seq in roster:
        groups.setdefault(group_of(sensor, seq), []).append(
            r1_of(method, sensor, seq))
    entry = {g: sum(v) / len(v) for g, v in sorted(groups.items())}
    entry["strict_macro"] = sum(entry.values()) / len(entry)
    return entry


def main():
    roster = strict_roster()
    assert len(roster) == 37, len(roster)
    prev = json.loads((OUT_DIR / "summary_strict.json").read_text())

    # ---- validation: my grouping must reproduce revision-2 exactly ----
    for method in ("scpp", "ring"):
        recomputed = aggregate(method, roster)
        for k, v in recomputed.items():
            assert abs(v - prev[method][k]) < 1e-9, (method, k, v, prev[method][k])
    print("validation OK: scpp/ring group aggregation reproduces revision 2")

    sel = json.loads((OUT_DIR / "ringsharp_pd_ckpt_selection.json").read_text())
    ringsharp_pd = aggregate("ringsharp_pd", roster)

    out = {"_meta": {
        "date": "2026-08-08",
        "protocol": "Strict-v2 (unchanged): 37-seq roster, six-group-balanced "
                    "macro; excluded+disclosed: kitti/00 (train 07 overlap), "
                    "kitti360/0004 (val 09 overlap), helipr/Bridge04 (train "
                    "site), nclt pair (train campus; temporal table)",
        "revision3": "RING# column replaced: paper-era zero-shot weights -> "
                     "clean pd2026 retrain on the full 37-seq roster (no "
                     "train_overlap exclusions needed)",
        "ringsharp_pd": {
            "weight": sel["selected_weight"],
            "weight_sha256": sel["weight_sha256"],
            "selected_epoch": sel["selected_epoch"],
            "selection_rule": sel["rule"],
        },
        "revision2_meta": prev["_meta"],
    }}
    for method in ("scpp", "bevplace", "seqot", "ring", "ringpp",
                   "nsd_pbev128_main", "nsd_hspec12"):
        out[method] = prev[method]
    out["ringsharp"] = ringsharp_pd

    p = OUT_DIR / "summary_strict_v2_revision3.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}\n")
    order = sorted((m for m in out if m != "_meta"),
                   key=lambda m: -out[m]["strict_macro"])
    for m in order:
        print(f"  {m:20s} strict_macro={out[m]['strict_macro']:.4f}")
    print("\nringsharp groups (revision3 vs revision2 zero-shot):")
    for g in ringsharp_pd:
        if g == "strict_macro":
            continue
        print(f"  {g:22s} {ringsharp_pd[g]:.4f}  (was {prev['ringsharp'].get(g, float('nan')):.4f})")


if __name__ == "__main__":
    main()
