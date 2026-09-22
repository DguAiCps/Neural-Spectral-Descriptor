#!/usr/bin/env python3
"""RING# pd2026 checkpoint selection (pre-declared rule).

Rule (frozen before any test contact): among epochs 1-5, select the
checkpoint with the highest pd-validation sequence-macro 5 m Recall@1
(macro over the 4 pd val sequences kitti/09, nclt/2012-01-08,
helipr/Roundabout03, mulran/Sejong03, equal weight); on ties, the EARLIER
epoch. Records the selected weight path + sha256 into
results/pd2026_baselines/ringsharp_pd_ckpt_selection.json. Only after this
file is written may the selected checkpoint touch the test roster.

Inputs: results/pd2026_baselines/_val_ringsharp_pd_e{N}_{sensor}_{seq}.json
(produced by scripts/_pdb_ringsharp_pd_eval.py --val).
"""
from __future__ import annotations
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdb_ringsharp_pd_eval import sha256_of        # noqa: E402

OUT_DIR = REPO / "results" / "pd2026_baselines"
WEIGHTS_DIR = (REPO / "_external/RINGSharp/tools/results/weights/"
               "ring_sharp_l_nsc_pd2026")
VAL_SEQS = [("kitti", "09"), ("nclt", "2012-01-08"),
            ("helipr", "Roundabout03"), ("mulran", "Sejong03")]


def main():
    epochs = {}
    for f in sorted(OUT_DIR.glob("_val_ringsharp_pd_e*_*.json")):
        m = re.match(r"_val_ringsharp_pd_(e\d+)_(.+)\.json", f.name)
        tag, seqkey = m.group(1), m.group(2)
        d = json.loads(f.read_text())
        (label, res), = [(k, v) for k, v in d.items() if k != "_meta"]
        epochs.setdefault(tag, {})[seqkey] = {
            "recall@1": res["recall@1"], "n_queries": res["n_queries"]}

    expected = {f"{s}_{q}" for s, q in VAL_SEQS}
    table = {}
    for tag in sorted(epochs, key=lambda t: int(t[1:])):
        have = set(epochs[tag])
        assert have == expected, (tag, "missing val sequences", expected - have)
        macro = sum(epochs[tag][k]["recall@1"] for k in sorted(expected)) / len(expected)
        table[tag] = {"macro_r1_5m": macro, "per_seq": epochs[tag]}
        print(f"{tag}: macro R@1@5m = {macro:.4f}  "
              + "  ".join(f"{k}={epochs[tag][k]['recall@1']:.4f}"
                          for k in sorted(expected)))

    # max macro; ties -> earlier epoch (sort by (-macro, epoch_number))
    best = min(table, key=lambda t: (-table[t]["macro_r1_5m"], int(t[1:])))
    ckpts = sorted(glob.glob(str(WEIGHTS_DIR / f"model_*_{best[1:]}.pth")))
    assert len(ckpts) == 1, (best, ckpts)
    sel = {
        "rule": ("highest pd-validation sequence-macro 5 m Recall@1 over "
                 "epochs 1-5; earlier epoch on ties (pre-declared before any "
                 "test contact)"),
        "val_sequences": [f"{s}/{q}" for s, q in VAL_SEQS],
        "per_epoch": table,
        "selected_epoch": best,
        "selected_weight": ckpts[0],
        "weight_sha256": sha256_of(ckpts[0]),
    }
    out = OUT_DIR / "ringsharp_pd_ckpt_selection.json"
    out.write_text(json.dumps(sel, indent=2))
    print(f"\nselected: {best} -> {ckpts[0]}\nsha256={sel['weight_sha256']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
