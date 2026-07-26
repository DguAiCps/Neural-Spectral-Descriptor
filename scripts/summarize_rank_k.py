#!/usr/bin/env python3
"""Aggregate Recall@{1,5,10} for NSD (deployed train-calibrated protocol) and
baselines (existing outputs/*.csv) into one review-response table.

NSD: results/_remetrize_twopass/b3_rankk_seed{1,2,3}.json
     deployed = KITTI grid (w_b,w_r)=(0.5,0), other sensors coarse; 3-seed mean.
Baselines: outputs/baseline_full_v2_handcrafted.csv + _mulran_ variants +
     baseline_iris.csv + bevplace_ft_main/mulran.csv (recall_at_{1,5,10}).
"""
from __future__ import annotations
import csv, json, statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
N = {"kitti/00": 632, "kitti/05": 377, "kitti/08": 235,
     "nclt/2012-01-08": 1834, "nclt/2013-01-10": 181, "helipr/Town01": 1586,
     "mulran/DCC03": 2344, "mulran/KAIST03": 2909, "mulran/Riverside03": 2796}
SEQ_ORDER = list(N)
CSV_NAME = {"kitti/00": "KITTI_00", "kitti/05": "KITTI_05", "kitti/08": "KITTI_08",
            "nclt/2012-01-08": "NCLT_2012-01-08", "nclt/2013-01-10": "NCLT_2013-01-10",
            "helipr/Town01": "HELIPR_Town01", "mulran/DCC03": "MULRAN_DCC03",
            "mulran/KAIST03": "MULRAN_KAIST03", "mulran/Riverside03": "MULRAN_Riverside03"}
KEY_TC = "phase_sketch_fusion_bev0.5_range0"  # train-calibrated KITTI weight
KS = ["R@1", "R@5", "R@10"]


def agg(per_seq):  # {seq: {R@k: v}} -> query-weighted + seq-balanced per k
    out = {}
    for k in KS:
        out[f"q_{k}"] = sum(per_seq[s][k] * N[s] for s in SEQ_ORDER) / sum(N.values())
        out[f"s_{k}"] = st.mean(per_seq[s][k] for s in SEQ_ORDER)
    return out


def nsd_rows():
    seeds = []
    for sd in (1, 2, 3):
        p = REPO / f"results/_remetrize_twopass/b3_rankk_seed{sd}.json"
        if not p.exists():
            return None, f"missing {p.name}"
        seeds.append(json.load(open(p)))
    per_seq = {}
    for s in SEQ_ORDER:
        vals = {}
        for k in KS:
            xs = []
            for d in seeds:
                e = d[s]
                src = e["grid"][KEY_TC] if s.startswith("kitti") else e["coarse"]
                xs.append(src[k])
            vals[k] = st.mean(xs)
        per_seq[s] = vals
    return per_seq, None


def baseline_rows():
    files = ["baseline_full_v2_handcrafted.csv", "baseline_mulran_handcrafted.csv",
             "baseline_iris.csv", "baseline_mulran_iris.csv",
             "bevplace_ft_main.csv", "bevplace_ft_mulran.csv"]
    per = {}
    for f in files:
        for row in csv.DictReader(open(REPO / "outputs" / f)):
            m = row["short_name"]
            per.setdefault(m, {})[row["dataset"]] = {
                "R@1": float(row["recall_at_1"]), "R@5": float(row["recall_at_5"]),
                "R@10": float(row["recall_at_10"])}
    out = {}
    for m, ds in per.items():
        try:
            out[m] = {s: ds[CSV_NAME[s]] for s in SEQ_ORDER}
        except KeyError as e:
            print(f"[warn] {m}: missing {e}, skipped")
    return out


def main():
    print(f"{'method':<12}" + "".join(f" {k:>7}" for k in KS)
          + "   (query-weighted | seq-balanced)")
    base = baseline_rows()
    for m in ("sc++", "m2dp", "fresco", "lidar_iris", "bevplace"):
        if m not in base:
            continue
        a = agg(base[m])
        print(f"{m:<12}" + "".join(f" {a[f'q_{k}']:>7.3f}" for k in KS)
              + "  |" + "".join(f" {a[f's_{k}']:>6.3f}" for k in KS))
    nsd, err = nsd_rows()
    if err:
        print(f"NSD: {err} (rank-K eval still running?)")
    else:
        a = agg(nsd)
        print(f"{'NSD (main)':<12}" + "".join(f" {a[f'q_{k}']:>7.3f}" for k in KS)
              + "  |" + "".join(f" {a[f's_{k}']:>6.3f}" for k in KS))
        print("\nNSD per-sequence (3-seed mean):")
        for s in SEQ_ORDER:
            print(f"  {s:<20}" + "".join(f" {nsd[s][k]:>7.3f}" for k in KS))


if __name__ == "__main__":
    main()
