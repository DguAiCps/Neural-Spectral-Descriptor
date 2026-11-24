#!/usr/bin/env python3
"""같은 오수락 수 비교(시험 전체 단일 전역 임계값).

scripts/_paper_matched_false_accepts.py의 build·choose·at_tau·independent·draws·replicate_curves를 그대로 쓰고
NSD 원장만 바꾼다. --nsd old 는 old 꼬리표 원장으로 /mnt/d/.../pdv2_matched_false_accepts_20261112/result.json의
기준별 값과 구간을 재현하는지 검사한다(결과는 analysis/matched_fa/reproduction_previous_nsd.json).
BEVPlace++는 원장이 같으므로 기존 값과 같아야 한다(검사).

python3 scripts/_report_model_matched_fa_20261116.py --nsd new|old [--workers 2]
출력: results/pdv2_report_model_20261116/analysis/matched_fa/result.json (기존 result.json의 criteria 스키마; per_seed 키는 실제 시드)
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_model_common_20261116 as C  # noqa: E402
import _paper_matched_false_accepts as mf  # noqa: E402

PREV = Path("/mnt/d/NSD_datasets/nsd_results/pdv2_matched_false_accepts_20261112/result.json")
OUT = C.OUT / "matched_fa"


def load(tag):
    rows = {}
    for k in mf.SEQS:
        d = C.load(tag, k)
        rows[k] = dict(score=d["score"], dist=d["top1"], query_idx=d["q"], mpgd=d["gt"], qc=C.classify_queries(d["gt"]))
    return rows


def job(args):
    key, fixed_index = args
    return key, mf.replicate_curves(mf.BUILT[key], mf.WEIGHTS, fixed_index, mf.CITY_COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsd", default="new", choices=("new", "old"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    seeds = {"nsd": [int(t.split(":")[1]) for t in C.tags(args.nsd)], "bevplace": [1, 2, 3]}
    tags = {"nsd": C.tags(args.nsd), "bevplace": C.tags("bev")}
    data = {(m, s): load(t) for m in mf.METHODS for s, t in zip(seeds[m], tags[m])}
    built = {k: mf.build(v, mf.SEQS) for k, v in data.items()}
    criteria, fixed_index = {}, {}
    for name, kind, value in mf.CRITERIA:
        block = dict(kind=kind, value=value, methods={})
        for m in mf.METHODS:
            per_seed = {}
            for s in seeds[m]:
                b = built[(m, s)]
                i = int(mf.choose(b["tp"], b["fp"], kind, value))
                r = mf.at_tau(data[(m, s)], float(b["thresholds"][i]))
                assert r["TP"] == b["tp"][i] and r["FP"] == b["fp"][i]
                ind = mf.independent(data[(m, s)], mf.SEQS, kind, value)
                assert ind["tau"] == r["tau"] and ind["TP"] == r["TP"] and ind["FP"] == r["FP"]
                if kind == "budget":
                    assert ind["closed_form_TP"] == r["TP"]
                r["curve_index"] = i
                per_seed[str(s)] = r
                fixed_index.setdefault((m, s), {})[name] = i
            pick = lambda f: mf.mean_sd([per_seed[str(s)][f] for s in seeds[m]])
            block["methods"][m] = dict(
                mean_sd=dict(**{f: pick(f) for f in ("recall", "TP", "FP", "FP_positive_query", "FP_negative_query", "accepted", "precision", "city_macro_recall")},
                             city_recall={c: mf.mean_sd([per_seed[str(s)]["city_recall"][c] for s in seeds[m]]) for c in mf.CITY},
                             city_FP={c: mf.mean_sd([per_seed[str(s)]["city_FP"][c] for s in seeds[m]]) for c in mf.CITY}),
                per_seed=per_seed)
        a, z = block["methods"]["nsd"]["mean_sd"], block["methods"]["bevplace"]["mean_sd"]
        block["difference_nsd_minus_bevplace"] = dict(
            pooled_recall_pp=100 * (a["recall"]["mean"] - z["recall"]["mean"]),
            city_macro_recall_pp=100 * (a["city_macro_recall"]["mean"] - z["city_macro_recall"]["mean"]),
            TP=a["TP"]["mean"] - z["TP"]["mean"], FP=a["FP"]["mean"] - z["FP"]["mean"],
            city_recall_pp={c: 100 * (a["city_recall"][c]["mean"] - z["city_recall"][c]["mean"]) for c in mf.CITY})
        criteria[name] = block
    npos = {k: data[("nsd", seeds["nsd"][0])][k]["qc"]["n_pos"] for k in mf.SEQS}
    mf.WEIGHTS, mf.CITY_COLS, zero = mf.draws(npos, mf.SEQS, mf.CITY, mf.B, mf.BOOT_SEED)
    mf.BUILT.update(built)
    jobs = [((m, s), fixed_index[(m, s)]) for m in mf.METHODS for s in seeds[m]]
    with mp.get_context("fork").Pool(args.workers) as pool:
        boots = dict(pool.map(job, jobs))
    for name, *_ in mf.CRITERIA:
        block = criteria[name]
        block["intervals_95_pp"] = {}
        for variant in ("reselect", "fixed"):
            sm = {m: {f: np.mean([boots[(m, s)][name][f"{variant}_{f}"] for s in seeds[m]], axis=0) for f in ("pooled", "city", "FP")} for m in mf.METHODS}
            block["intervals_95_pp"][variant if variant == "reselect" else "fixed_tau"] = dict(
                pooled_recall=mf.interval(sm["nsd"]["pooled"] - sm["bevplace"]["pooled"]),
                city_macro_recall=mf.interval(sm["nsd"]["city"] - sm["bevplace"]["city"]),
                replicate_FP_seed_mean_q025_q50_q975={m: [float(x) for x in np.quantile(sm[m]["FP"], [.025, .5, .975])] for m in mf.METHODS})
    prev = json.loads(PREV.read_text())
    checks = {}
    for name in criteria:
        pb, nb = prev["criteria"][name], criteria[name]
        checks[name + "/bevplace_mean_equal"] = all(abs(nb["methods"]["bevplace"]["mean_sd"][f]["mean"] - pb["methods"]["bevplace"]["mean_sd"][f]["mean"]) < 1e-12
                                                    for f in ("recall", "TP", "FP", "city_macro_recall"))
        if args.nsd == "old":
            checks[name + "/nsd_mean_equal"] = all(abs(nb["methods"]["nsd"]["mean_sd"][f]["mean"] - pb["methods"]["nsd"]["mean_sd"][f]["mean"]) < 1e-12
                                                   for f in ("recall", "TP", "FP", "city_macro_recall"))
            checks[name + "/interval_max_abs_err"] = max(abs(x - y) for v in ("reselect", "fixed_tau") for f in ("pooled_recall", "city_macro_recall")
                                                         for x, y in zip(nb["intervals_95_pp"][v][f], pb["intervals_95_pp"][v][f]))
    print(json.dumps(checks, indent=1))
    ok = all(v is True or (isinstance(v, float) and v < 1e-9) for v in checks.values())
    result = dict(status="passed" if ok else "check_failed", nsd_ledgers=args.nsd, seeds=seeds, threshold_selected_on="test", validation_fixed=False,
                  description=prev["description"], criteria=criteria, checks=checks,
                  intervals_design=dict(replicates=mf.B, seed=mf.BOOT_SEED, primary="reselect", secondary="fixed_tau",
                                        zero_positive_fixed=[mf.SEQS[i] for i in zero], same_draws_as=str(PREV)),
                  elapsed_s=time.time() - t0)
    C.write(OUT / ("result.json" if args.nsd == "new" else "reproduction_previous_nsd.json"), result)
    for name, b in criteria.items():
        a, z = b["methods"]["nsd"]["mean_sd"], b["methods"]["bevplace"]["mean_sd"]
        iv = b["intervals_95_pp"]["reselect"]["city_macro_recall"]
        print(f"{name:10s} NSD city {100*a['city_macro_recall']['mean']:.2f}±{100*a['city_macro_recall']['sd']:.2f} pooled {100*a['recall']['mean']:.2f} TP {a['TP']['mean']:.0f} FP {a['FP']['mean']:.1f}"
              f" | BEV city {100*z['city_macro_recall']['mean']:.2f}±{100*z['city_macro_recall']['sd']:.2f} pooled {100*z['recall']['mean']:.2f} TP {z['TP']['mean']:.0f}"
              f" | diff {b['difference_nsd_minus_bevplace']['city_macro_recall_pp']:+.2f} [{iv[0]:+.2f}, {iv[1]:+.2f}]"
              f" | NSD FP seqs {sorted({k for s in b['methods']['nsd']['per_seed'].values() for k, v in s['sequence'].items() if v['FP']})}")
    assert ok


if __name__ == "__main__":
    main()
