"""Aggregate 3-seed validation results into bootstrap CI for paper Table 2.

Reads per-seed validation logs (or a manifest) and computes per-sequence and
summary statistics with their bootstrap 95% confidence intervals.

Usage:
    python scripts/aggregate_seeds.py \
      --seeds 42 123 456 \
      --runs-dir results \
      --output results/seed_aggregate.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


SEQUENCES = [
    "KITTI_00", "KITTI_05", "KITTI_08",
    "NCLT_2012-01-08", "NCLT_2013-01-10",
    "HeLiPR_Town01",
    "MulRan_DCC03", "MulRan_KAIST03", "MulRan_Riverside03",
]
QUERY_COUNTS = {
    "KITTI_00": 632, "KITTI_05": 377, "KITTI_08": 235,
    "NCLT_2012-01-08": 1834, "NCLT_2013-01-10": 181,
    "HeLiPR_Town01": 1586,
    "MulRan_DCC03": 2344, "MulRan_KAIST03": 2909, "MulRan_Riverside03": 2796,
}
SENSOR_OF = {
    "KITTI_00": "HDL-64E", "KITTI_05": "HDL-64E", "KITTI_08": "HDL-64E",
    "NCLT_2012-01-08": "HDL-32E", "NCLT_2013-01-10": "HDL-32E",
    "HeLiPR_Town01": "VLP-16",
    "MulRan_DCC03": "OS1-64", "MulRan_KAIST03": "OS1-64", "MulRan_Riverside03": "OS1-64",
}


def parse_log(log_path: Path) -> dict:
    """Extract per-sequence R@1 from the training log's BEST-epoch validation block.

    Strategy: scan all per-sequence R@1 entries grouped by their preceding
    "Validation (per-dataset):" header position; pick the block whose AVERAGE
    R@1 matches the highest "AVERAGE | R@1: X.XXXX" line in the log (the best
    epoch). This matches the model checkpoint the paper reports.
    """
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    text = log_path.read_text()

    # Locate every AVERAGE line and pick the maximum R@1 (best epoch).
    avg_iter = list(re.finditer(
        r"AVERAGE\s+\|\s+R@1:\s+([0-9.]+)", text))
    if not avg_iter:
        raise RuntimeError(f"No AVERAGE lines in {log_path}")
    best_match = max(avg_iter, key=lambda m: float(m.group(1)))
    best_avg = float(best_match.group(1))
    best_pos = best_match.start()

    # Per-seq lines occur in the ~9 lines preceding the AVERAGE in the same block.
    block_start = max(0, best_pos - 4000)  # generous window
    block = text[block_start:best_pos]
    matches = re.findall(
        r"(\w[\w-]*)\s+\|\s+R@1:\s+([0-9.]+)\s+\(raw:",
        block,
    )
    per_seq = {}
    for name, r1 in matches:
        if name in QUERY_COUNTS:
            per_seq[name] = float(r1)  # last occurrence in window wins
    if not per_seq:
        raise RuntimeError(f"No per-sequence lines preceding best AVG in {log_path}")
    per_seq["_best_avg"] = best_avg
    return per_seq


def per_sensor_macro(per_seq: dict) -> dict:
    out = {}
    for sensor in set(SENSOR_OF.values()):
        seqs = [s for s, sn in SENSOR_OF.items() if sn == sensor]
        vals = [per_seq[s] for s in seqs if s in per_seq]
        out[sensor] = float(np.mean(vals)) if vals else float("nan")
    return out


def summary_stats(per_seq: dict) -> dict:
    # Drop synthetic keys before aggregating.
    per_seq = {k: v for k, v in per_seq.items() if k in QUERY_COUNTS}
    seq_vals = [per_seq.get(s, float("nan")) for s in SEQUENCES]
    queries = [QUERY_COUNTS[s] for s in SEQUENCES]
    rq = float(np.average(seq_vals, weights=queries))  # query-weighted
    rs = float(np.mean(seq_vals))                      # sequence-balanced
    sensors = per_sensor_macro(per_seq)
    sensor_vals = list(sensors.values())
    sigma_cross = float(np.std(sensor_vals))
    rmin = float(np.min(sensor_vals))
    return {"R_q": rq, "R_s": rs, "sigma_cross": sigma_cross, "R_min": rmin,
            "per_sensor": sensors}


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, ci: float = 0.95,
                 rng_seed: int = 0) -> tuple:
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    boots = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--runs-dir", type=Path, default=Path("results"))
    p.add_argument("--logs-dir", type=Path, default=Path("logs"),
                   help="Directory containing per-seed training logs (filename pattern: seed_<N>.log).")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    per_seed_summary = {}
    per_seq_runs = {s: [] for s in SEQUENCES}
    per_summary_runs = {k: [] for k in ["R_q", "R_s", "sigma_cross", "R_min"]}

    for seed in args.seeds:
        log_path = args.logs_dir / f"seed_{seed}.log"
        if not log_path.exists():
            # Fall back to most recent training log under runs-dir.
            cand = sorted((args.runs_dir / f"seed_{seed}").glob("training_*.log"))
            if cand:
                log_path = cand[-1]
        per_seq = parse_log(log_path)
        stats = summary_stats(per_seq)
        per_seed_summary[seed] = {"per_seq": per_seq, **stats}
        for s in SEQUENCES:
            if s in per_seq:
                per_seq_runs[s].append(per_seq[s])
        for k in per_summary_runs:
            per_summary_runs[k].append(stats[k])

    aggregate = {"n_seeds": len(args.seeds), "seeds": list(args.seeds),
                 "per_seed": per_seed_summary, "per_sequence": {}, "summary": {}}
    for seq, vals in per_seq_runs.items():
        if not vals:
            continue
        arr = np.array(vals)
        lo, hi = bootstrap_ci(arr)
        aggregate["per_sequence"][seq] = {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "ci95_lo": lo, "ci95_hi": hi, "n": int(len(vals)),
        }
    for k, vals in per_summary_runs.items():
        arr = np.array(vals)
        lo, hi = bootstrap_ci(arr)
        aggregate["summary"][k] = {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "ci95_lo": lo, "ci95_hi": hi,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"[ok] {args.output}")

    print("\n=== Summary across seeds ===")
    for k, v in aggregate["summary"].items():
        print(f"  {k:14s} {v['mean']:.4f} ± {v['std']:.4f} "
              f"(95% CI [{v['ci95_lo']:.4f}, {v['ci95_hi']:.4f}])")


if __name__ == "__main__":
    main()
