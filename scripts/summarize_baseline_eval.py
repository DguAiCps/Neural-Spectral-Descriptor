"""Aggregate per-baseline R@1 into the four summary stats used in Table 4."""

import csv
import sys
from collections import defaultdict
from pathlib import Path


DS_ORDER = ["KITTI_00", "KITTI_05", "KITTI_08",
            "NCLT_2012-01-08", "NCLT_2013-01-10", "HELIPR_Town01"]
SENSORS = {
    "kitti": ["KITTI_00", "KITTI_05", "KITTI_08"],
    "nclt":  ["NCLT_2012-01-08", "NCLT_2013-01-10"],
    "helipr": ["HELIPR_Town01"],
}


def main(csv_paths):
    results = defaultdict(dict)
    method_names = {}
    for fname in csv_paths:
        if not Path(fname).exists():
            continue
        with open(fname) as f:
            for row in csv.DictReader(f):
                m = row["short_name"]
                method_names[m] = row["method"]
                results[m][row["dataset"]] = (
                    float(row["recall_at_1"]),
                    int(row["n_queries"]),
                    int(row["dim"]),
                )

    hdr = (f"{'Method':<14} {'Dim':>5} "
           f"{'K00':>6} {'K05':>6} {'K08':>6} {'N12':>6} {'N13':>6} {'HT01':>6} "
           f" {'R^q':>5} {'R^s':>5} {'sig':>5} {'Rmin':>5}")
    print(hdr)
    print("-" * len(hdr))

    method_order = ["sc++", "m2dp", "fresco", "lidar_iris",
                    "pointnetvlad", "bevplace", "nsc_raw", "nsc_gnn"]
    for m in method_order:
        if m not in results:
            continue
        r = results[m]
        ds_have = [d for d in DS_ORDER if d in r]
        if len(ds_have) < len(DS_ORDER):
            print(f"{m:<14}  partial: have {len(ds_have)}/{len(DS_ORDER)} sequences")
            continue
        dim = next(iter(r.values()))[2]
        row_R = [r[d][0] for d in DS_ORDER]
        row_n = [r[d][1] for d in DS_ORDER]
        Rq = sum(R * n for R, n in zip(row_R, row_n)) / sum(row_n)
        Rs = sum(row_R) / len(row_R)
        per_sensor = {
            s: sum(r[d][0] for d in dsl) / len(dsl)
            for s, dsl in SENSORS.items()
        }
        macro = sum(per_sensor.values()) / len(per_sensor)
        sigma = (sum((v - macro) ** 2 for v in per_sensor.values())
                 / len(per_sensor)) ** 0.5
        Rmin = min(per_sensor.values())
        print(f"{method_names[m]:<14} {dim:>5} "
              + " ".join(f"{x:>6.3f}" for x in row_R)
              + f"  {Rq:>5.3f} {Rs:>5.3f} {sigma:>5.3f} {Rmin:>5.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or [
        "outputs/baseline_full_v2_handcrafted.csv",
        "outputs/baseline_nsc.csv",
        "outputs/baseline_iris.csv",
    ])
