#!/usr/bin/env python3
"""NSD − 정제 없음(no temporal spectrum refining)의 블록 재표집 구간(부록 "Learning across seeds and cities" 둘째 문단).

절차는 scripts/_report_model_direction_block_20261116.py(= scripts/_review_direction_bootstrap.py: 도시 안 시퀀스 + 원형 100질의 블록,
2,000회, rng [20261106, i])와 같다. 같은 replicate 안에서 old 꼬리표 원장(identity, diff_s1..3; results/pdv2_diffattn_contrast_test_20261104)과
NSD 원장(정제 없음 = ablations/ledgers/test/<seq>/identity.npz, NSD 시드 4~6)을 함께 계산해, old 쪽이
results/pdv2_review_direction_yaw_20261105/direction_learning/{summary,fixed_city_block_summary}.json의 diff-identity 구간을 재현하는지 확인한다.
출력: results/pdv2_report_model_20261116/analysis/main/refining_block_summary.json
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
METHODS = (("oldidentity",), ("old:1", "old:2", "old:3"), ("newidentity",), tuple(C.tags("new")))
DATA = {}


def load(tag: str, seq: str) -> dict:
    name = seq.replace("/", "_")
    if tag == "oldidentity":
        p = C.OLD / "test" / name / "identity.npz"
    elif tag == "newidentity":
        p = C.OUT.parent / "ablations/ledgers/test" / name / "identity.npz"
    else:
        return C.load(tag, seq)
    with np.load(p) as z:
        return dict(score=z["score"].astype(np.float64), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"], q=z["query_idx"])


def load_data():
    for seq in (s for city in base.CITY.values() for s in city):
        with np.load(base.OUT / "directions" / (seq.replace("/", "_") + ".npz")) as z:
            group, qidx = z["group"], z["query_idx"]
        prepared = []
        for g in GROUPS:
            methods = []
            for tags in METHODS:
                runs = []
                for t in tags:
                    d = load(t, seq)
                    assert np.array_equal(d["q"], qidx)
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
    prev_block = json.loads((base.OUT / "direction_learning/summary.json").read_text())["comparisons"]["diff-identity"]
    prev_six = json.loads((base.OUT / "direction_learning/fixed_city_block_summary.json").read_text())["comparisons"]["diff-identity"]
    refining = json.loads((C.OUT.parent / "ablations/no_refining_summary.json").read_text())
    out = dict(status="passed", replicates=args.replicates, comparisons={}, reproduction={})
    for (a, b, label) in ((1, 0, "previous_diff-identity"), (3, 2, "diff-identity")):
        res = {}
        for gi, g in enumerate(GROUPS):
            delta = 100 * (overall[:, gi, a] - overall[:, gi, b])
            valid = np.isfinite(cities[:, :, gi]).all(axis=(1, 2, 3))
            six = 100 * (cities[valid, :, gi, a].mean(1) - cities[valid, :, gi, b].mean(1))
            res[g] = dict(block=dict(overall={k: dict(ci95_pp=np.percentile(delta[:, i], [2.5, 97.5]).tolist()) for i, k in enumerate(C.METRICS)}),
                          six=dict(overall={k: dict(ci95_pp=np.percentile(six[:, i], [2.5, 97.5]).tolist()) for i, k in enumerate(C.METRICS)},
                                   complete_city_replicates=int(valid.sum())))
        out["comparisons"][label] = res
    # 점추정은 단위 B 요약(원래 6도시 평균)의 diff-identity
    for g in GROUPS:
        src = refining["comparisons"]["diff-identity"][g]["overall"]
        for k in C.METRICS:
            for kind in ("block", "six"):
                out["comparisons"]["diff-identity"][g][kind]["overall"][k]["delta_pp"] = src[k]["delta_pp"]
    err_b = max(abs(x - y) for g in GROUPS for k in C.METRICS for x, y in zip(out["comparisons"]["previous_diff-identity"][g]["block"]["overall"][k]["ci95_pp"], prev_block[g]["overall"][k]["ci95_pp"]))
    err_s = max(abs(x - y) for g in GROUPS for k in C.METRICS for x, y in zip(out["comparisons"]["previous_diff-identity"][g]["six"]["overall"][k]["ci95_pp"], prev_six[g]["overall"][k]["ci95_pp"]))
    out["reproduction"] = dict(previous_block_max_abs_err_pp=err_b, previous_six_max_abs_err_pp=err_s)
    out["note"] = ("block: 도시 안 시퀀스 + 원형 100질의 블록(가용 도시 평균). six: 6도시 모두 양성이 남은 replicate만. 점추정은 원래 6도시 평균"
                   "(ablations/no_refining_summary.json comparisons.diff-identity). 정제 없음 원장은 새 고정 설정(융합 (2,1,2,2), λ=0.25).")
    out["sources"] = dict(new_identity="results/pdv2_report_model_20261116/ablations/ledgers/test/<seq>/identity.npz",
                          new_nsd="results/pdv2_scratch_search_20261116/ledgers/test/<seq>/<CFG>@s{4,5,6}@u512.npz",
                          script="scripts/_report_model_refining_block_20261116.py")
    C.write(C.OUT / "main" / "refining_block_summary.json", out)
    print(json.dumps(out["reproduction"]))
    for g in GROUPS:
        for kind in ("block", "six"):
            o = out["comparisons"]["diff-identity"][g][kind]["overall"]
            print(g, kind, {k: (round(o[k]["delta_pp"], 3), [round(x, 3) for x in o[k]["ci95_pp"]]) for k in C.METRICS},
                  out["comparisons"]["diff-identity"][g][kind].get("complete_city_replicates", ""))
    assert args.replicates != 2000 or (err_b < 1e-9 and err_s < 1e-9)


if __name__ == "__main__":
    main()
