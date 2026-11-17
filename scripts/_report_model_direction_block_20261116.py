#!/usr/bin/env python3
"""보고모델 교체(2026-11-16) 단위 C 항목 7 보조: 방향별 학습 기여의 블록 재표집 민감도(부록 방향별 구간표의 블록 행).

scripts/_review_direction_bootstrap.py(시퀀스 + 원형 100질의 블록, 2,000회, rng [20261106, i])와
_review_direction_expanded_report.collect_learning(6도시 모두 남은 replicate만)의 절차를 그대로 쓴다.
같은 replicate 안에서 이전 원장(fixed, diff_s1..3)과 새 원장(uniform, 새 NSD 4..6)을 함께 계산해,
이전 쪽이 기존 direction_learning/{summary,fixed_city_block_summary}.json의 diff-fixed 구간을 재현하는지 확인한다.
출력: results/pdv2_report_model_20261116/analysis/main/direction_block_summary.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_model_common_20261116 as C  # noqa: E402
import _review_direction_bootstrap as RB  # noqa: E402
import _review_direction_metrics as base  # noqa: E402

GROUPS = ("all", "forward_only", "reverse_only")
METHODS = (("oldfixed",), ("old:1", "old:2", "old:3"), ("uniform",), tuple(C.tags("new")))
NAMES = ("old_fixed", "old_diff", "uniform", "new_diff")
DATA = {}


def load_data():
    for seq in (s for city in base.CITY.values() for s in city):
        with np.load(base.OUT / "directions" / (seq.replace("/", "_") + ".npz")) as z:
            group = z["group"]
        prepared = []
        for g in GROUPS:
            methods = []
            for tags in METHODS:
                runs = []
                for t in tags:
                    d = C.load(t, seq)
                    mask = np.ones(len(d["score"]), bool) if g == "all" else group == base.LABELS.index(g)
                    runs.append(RB.prepare(d["score"], d["top1"], d["gt"], mask))
                methods.append(runs)
            prepared.append(methods)
        DATA[seq] = dict(n=len(group), prepared=prepared)


def replicate(index):
    rng = np.random.default_rng([20261106, index])
    cities = np.full((len(base.CITY), len(GROUPS), len(METHODS), 4), np.nan)
    for ci, city in enumerate(base.CITY):
        samples = [[] for _ in GROUPS]
        seqs = base.CITY[city]
        for seq in rng.choice(seqs, len(seqs), replace=True):
            data = DATA[seq]; n = data["n"]
            starts = rng.integers(0, n, size=(n + 99) // 100)
            ix = ((starts[:, None] + np.arange(100)) % n).ravel()[:n]
            counts = np.bincount(ix, minlength=n)
            for gi, methods in enumerate(data["prepared"]):
                values = []
                for runs in methods:
                    seeds = [RB.weighted(row, counts) for row in runs]
                    values.append(None if seeds[0] is None else np.mean(seeds, axis=0))
                if values[0] is not None:
                    samples[gi].append(values)
        for gi, values in enumerate(samples):
            if values:
                cities[ci, gi] = np.mean(values, axis=0)
    return np.nanmean(cities, axis=0), cities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--replicates", type=int, default=2000)
    args = ap.parse_args()
    load_data()
    with Pool(args.workers) as pool:
        rows = list(pool.imap(replicate, range(args.replicates), chunksize=4))
    overall = np.stack([r[0] for r in rows]); cities = np.stack([r[1] for r in rows])
    np.savez_compressed(C.OUT / "main" / "direction_block_samples.npz", overall=overall, cities=cities)
    point = json.loads((C.OUT / "main" / "direction_summary.json").read_text())["methods"]
    prev_block = json.loads((base.OUT / "direction_learning/summary.json").read_text())["comparisons"]["diff-fixed"]
    prev_six = json.loads((base.OUT / "direction_learning/fixed_city_block_summary.json").read_text())["comparisons"]["diff-fixed"]
    out = dict(status="passed", replicates=args.replicates, comparisons={}, reproduction={})
    for (a, b, label) in ((1, 0, "previous_diff-fixed"), (3, 2, "diff-fixed")):
        res = {}
        for gi, g in enumerate(GROUPS):
            delta = 100 * (overall[:, gi, a] - overall[:, gi, b])
            valid = np.isfinite(cities[:, :, gi]).all(axis=(1, 2, 3))
            six = 100 * (cities[valid, :, gi, a].mean(1) - cities[valid, :, gi, b].mean(1))
            res[g] = dict(block=dict(overall={k: dict(ci95_pp=np.percentile(delta[:, i], [2.5, 97.5]).tolist()) for i, k in enumerate(C.METRICS)}),
                          six=dict(overall={k: dict(ci95_pp=np.percentile(six[:, i], [2.5, 97.5]).tolist()) for i, k in enumerate(C.METRICS)},
                                   complete_city_replicates=int(valid.sum())))
        out["comparisons"][label] = res
    # 점추정(원래 6도시 평균)
    for g in GROUPS:
        for k in C.METRICS:
            d = 100 * (point["diff"]["mean_std"][g][k]["mean"] - point["fixed"]["mean_std"][g][k]["mean"])
            out["comparisons"]["diff-fixed"][g]["block"]["overall"][k]["delta_pp"] = d
            out["comparisons"]["diff-fixed"][g]["six"]["overall"][k]["delta_pp"] = d
    err_b = max(abs(x - y) for g in GROUPS for k in C.METRICS for x, y in zip(out["comparisons"]["previous_diff-fixed"][g]["block"]["overall"][k]["ci95_pp"], prev_block[g]["overall"][k]["ci95_pp"]))
    err_s = max(abs(x - y) for g in GROUPS for k in C.METRICS for x, y in zip(out["comparisons"]["previous_diff-fixed"][g]["six"]["overall"][k]["ci95_pp"], prev_six[g]["overall"][k]["ci95_pp"]))
    out["reproduction"] = dict(previous_block_max_abs_err_pp=err_b, previous_six_max_abs_err_pp=err_s,
                               previous_six_complete=[prev_six[g]["complete_city_replicates"] for g in GROUPS])
    out["note"] = "block: 도시 안 시퀀스 + 원형 100질의 블록(가용 도시 평균). six: 6도시 모두 양성이 남은 replicate만. 점추정은 원래 6도시 평균."
    C.write(C.OUT / "main" / "direction_block_summary.json", out)
    print(json.dumps(out["reproduction"]))
    for g in GROUPS:
        for kind in ("block", "six"):
            o = out["comparisons"]["diff-fixed"][g][kind]["overall"]
            print(g, kind, {k: (round(o[k]["delta_pp"], 3), [round(x, 3) for x in o[k]["ci95_pp"]]) for k in C.METRICS},
                  out["comparisons"]["diff-fixed"][g][kind].get("complete_city_replicates", ""))
    assert args.replicates != 2000 or (err_b < 1e-9 and err_s < 1e-9)


if __name__ == "__main__":
    main()
