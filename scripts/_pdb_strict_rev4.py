#!/usr/bin/env python3
"""Strict-v2 revision 4: add the reviewer-requested zero-shot baselines
HeLiOS (ICRA'25) and ImLPR (CoRL'25), evaluated with their OFFICIAL code and
weights on the identical frozen roster.

Protocol unchanged from Strict-v2 (37 seqs, six-group-balanced macro).
Both methods were trained by their authors on HeLiPR DCC/KAIST/Riverside
04-06 = our helipr_daejeon test group, so their entries carry BOTH:
  * strict_macro        — 6-group macro (helipr_daejeon cell train-CONTAMINATED,
                          disclosed; optimistic upper bound there)
  * strict_macro_clean5 — 5-group macro excluding helipr_daejeon (fully
                          zero-shot groups only)
All previously reported methods are copied unchanged from revision 3.
Output: results/pd2026_baselines/summary_strict_v2_revision4.json
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "pd2026_baselines"
import sys
sys.path.insert(0, str(REPO / "scripts"))
from _pdb_strict_rev3 import strict_roster, group_of, GROUP_LABEL  # noqa: E402

CONTAM_GROUP = GROUP_LABEL["helipr_daejeon"]
NEW = {"imlpr": "imlpr", "helios": "helios", "imlpr_pd": "imlpr_pd"}   # method -> json prefix
CLEAN_NEW = {"imlpr_pd"}  # pd-trained: no train/test overlap by construction


def r1_of(prefix: str, sensor: str, seq: str) -> float:
    d = json.loads((OUT_DIR / f"{prefix}_{sensor}_{seq}.json").read_text())
    if "recall_at_1" in d:
        return float(d["recall_at_1"])
    if f"{sensor}_{seq}" in d:
        return float(d[f"{sensor}_{seq}"]["recall@1"])
    return float(d["recall@1"])


def aggregate(prefix: str, roster, clean: bool = False) -> dict:
    groups = {}
    for sensor, seq in roster:
        groups.setdefault(group_of(sensor, seq), []).append(
            r1_of(prefix, sensor, seq))
    entry = {g: sum(v) / len(v) for g, v in sorted(groups.items())}
    entry["strict_macro"] = sum(entry[g] for g in groups) / len(groups)
    if not clean:
        cg = [g for g in groups if g != CONTAM_GROUP]
        entry["strict_macro_clean5"] = sum(entry[g] for g in cg) / len(cg)
        entry["train_contaminated_group"] = CONTAM_GROUP
    return entry


def main():
    roster = strict_roster()
    prev = json.loads((OUT_DIR / "summary_strict_v2_revision3.json").read_text())
    out = {"_meta": {
        "date": "2026-08-26",
        "protocol": prev["_meta"]["protocol"],
        "revision5": ("adds the CLEAN pd-retrained ImLPR row (from-scratch DINOv2 init, official recipe, pre-declared e100 selection); revision4 added zero-shot HeLiOS/ImLPR (official code+weights); "
                      "both trained on HeLiPR DCC/KAIST/Riverside 04-06 = our "
                      "helipr_daejeon test group -> that cell is disclosed as "
                      "train-contaminated and a clean 5-group macro is added "
                      "for them; all other methods copied from revision 3"),
        "revision3_meta": {k: v for k, v in prev["_meta"].items()
                           if k != "revision2_meta"},
    }}
    for k, v in prev.items():
        if k != "_meta":
            out[k] = v
    for method, prefix in NEW.items():
        out[method] = aggregate(prefix, roster, clean=method in CLEAN_NEW)
    p = OUT_DIR / "summary_strict_v2_revision5.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}\n")
    order = sorted((m for m in out if m != "_meta"),
                   key=lambda m: -out[m]["strict_macro"])
    for m in order:
        c5 = out[m].get("strict_macro_clean5")
        extra = f"  clean5={c5:.4f}" if c5 else ""
        print(f"  {m:20s} strict_macro={out[m]['strict_macro']:.4f}{extra}")
    for method in NEW:
        print(f"\n{method} groups:")
        for g, v in out[method].items():
            if isinstance(v, float) and not g.startswith("strict"):
                mark = (" (TRAIN-CONTAMINATED)"
                        if g == CONTAM_GROUP and method not in CLEAN_NEW else "")
                print(f"  {g:22s} {v:.4f}{mark}")


if __name__ == "__main__":
    main()
