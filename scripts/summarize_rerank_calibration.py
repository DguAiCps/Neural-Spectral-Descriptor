#!/usr/bin/env python3
"""Select per-sensor rerank weights on TRAIN, apply to VAL, report.

Reads b3_train_calib_seed1.json (per-train-sequence coarse + grid), picks for
each sensor family the single (w_b, w_r) grid point---coarse counts as (0,0)---
that maximizes that sensor's query-weighted R@1 on the TRAINING sequences. Those
fixed per-sensor weights are then applied to the committed validation grids
(b3_rerank_seed{1,2,3}.json) and the 3-seed aggregate is reported alongside the
per-sequence (on-eval) and global-single-weight protocols for comparison.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results/_remetrize_twopass"
QC = {"kitti/00": 632, "kitti/05": 377, "kitti/08": 235, "nclt/2012-01-08": 1834,
      "nclt/2013-01-10": 181, "helipr/Town01": 1586, "mulran/DCC03": 2344,
      "mulran/KAIST03": 2909, "mulran/Riverside03": 2796}
ORDER = list(QC.keys())
SENS = {k: k.split("/")[0] for k in ORDER}


def agg(per):
    rq = sum(per[k] * QC[k] for k in ORDER) / sum(QC.values())
    rs = float(np.mean([per[k] for k in ORDER]))
    mac = {}
    for k in ORDER:
        mac.setdefault(SENS[k], []).append(per[k])
    macs = {s: np.mean(v) for s, v in mac.items()}
    sig = float(np.sqrt(np.mean([(m - np.mean(list(macs.values()))) ** 2 for m in macs.values()])))
    return rq, rs, sig, float(min(macs.values()))


def main():
    train = json.load(open(R / "b3_train_calib_seed1.json"))
    val = {s: json.load(open(R / f"b3_rerank_{s}.json")) for s in ["seed1", "seed2", "seed3"]}
    gks = list(val["seed1"][ORDER[0]]["grid"].keys())

    # --- calibrate per-sensor weight on TRAIN (query-weighted; coarse = (0,0)) ---
    tr_by_sensor = {}
    for name, d in train.items():
        tr_by_sensor.setdefault(name.split("/")[0], []).append((name, d))
    chosen = {}
    for sn, items in tr_by_sensor.items():
        def opt_rq(opt):
            num = den = 0.0
            for name, d in items:
                w = d.get("n_q", 1) or 1
                v = d["coarse"] if opt == "coarse" else d["grid"][opt]
                num += v * w
                den += w
            return num / den
        best = max(["coarse"] + gks, key=opt_rq)
        chosen[sn] = best
        print(f"[train] {sn:<8} best weight = {best:<32} (train qw-R@1 ={opt_rq(best):.4f}, "
              f"coarse={opt_rq('coarse'):.4f})")

    # --- apply fixed per-sensor weights to VAL (3-seed) ---
    def val_val(seed, seq, opt):
        return val[seed][seq]["coarse"] if opt == "coarse" else val[seed][seq]["grid"][opt]

    per_seed = {}
    for s in val:
        per_seed[s] = {k: val_val(s, k, chosen[SENS[k]]) for k in ORDER}
    mean_per = {k: float(np.mean([per_seed[s][k] for s in val])) for k in ORDER}

    print("\n=== train-calibrated per-sensor weights applied to VAL ===")
    for s in val:
        print(f"  {s}: Rq=%.4f Rs=%.4f sig=%.4f Rmin=%.4f" % agg(per_seed[s]))
    rq, rs, sig, rmin = agg(mean_per)
    print(f"  3-seed: Rq={rq:.4f} Rs={rs:.4f} sig={sig:.4f} Rmin={rmin:.4f}")
    print("  per-seq 3-seed means:")
    for k in ORDER:
        print(f"    {k:<22} {mean_per[k]:.4f}  (weight {chosen[SENS[k]].replace('phase_sketch_fusion_','')})")

    # --- reference protocols ---
    def three_seed(selector):
        m = {k: float(np.mean([selector(s, k) for s in val])) for k in ORDER}
        return agg(m)
    ps = three_seed(lambda s, k: max(val[s][k]["coarse"], max(val[s][k]["grid"].values())))
    gl_best = None
    for gk in gks:
        a = three_seed(lambda s, k: val[s][k]["grid"][gk])
        if gl_best is None or a[0] > gl_best[1][0]:
            gl_best = (gk, a)
    print("\n=== reference protocols (3-seed) ===")
    print("  per-sequence (on-eval):  Rq=%.4f Rs=%.4f sig=%.4f Rmin=%.4f" % ps)
    print(f"  global single ({gl_best[0].replace('phase_sketch_fusion_','')}): Rq=%.4f Rs=%.4f sig=%.4f Rmin=%.4f" % gl_best[1])

    json.dump({"chosen": chosen, "per_seq_val": mean_per, "aggregate": dict(zip(["Rq","Rs","sigma","Rmin"], (rq,rs,sig,rmin)))},
              open(R / "rerank_calib_summary.json", "w"), indent=1)
    print("\nsaved", R / "rerank_calib_summary.json")


if __name__ == "__main__":
    main()
