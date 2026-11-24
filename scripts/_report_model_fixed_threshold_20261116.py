#!/usr/bin/env python3
"""검증9에서 고정한 P99·P100·F1 기준값을 시험46에 적용.

기존 규약 scripts/_review_v23_fixed_threshold_20261107.py(choose·fixed·mean_sd를 그대로 가져다 씀)와
_report.py의 오수락 층(반환 거리 구간, 질의 종류×키프레임 간격×반환 거리)을 따른다.
--nsd old 로 돌리면 old 꼬리표 원장으로 results/pdv2_review_closure_20261107/threshold/{freeze,summary,false_accepts}.json
값을 재현하는지 검사만 한다. --nsd new(기본)는 NSD 시드 4·5·6으로 계산하고, BEVPlace++는 그 freeze.json의 τ를 다시 적용해
그 summary.json과 같은지 확인한 뒤 그대로 싣는다.

출력(results/pdv2_report_model_20261116/analysis/threshold/): freeze.json, summary.json, false_accepts.json
(스키마는 기존 파일과 같은 키; per_seed 키는 실제 시드 번호 "4","5","6")
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_model_common_20261116 as C  # noqa: E402
import _review_v23_fixed_threshold_20261107 as main  # noqa: E402
from _pdv2_pilot import VAL  # noqa: E402

OLD = C.ROOT / "results/pdv2_review_closure_20261107/threshold"
OUT = C.OUT / "threshold"
RETURNED_BINS = ((5, 7.5), (7.5, 10), (10, 25), (25, 100), (100, np.inf))
GAP_BINS = ((30, 60), (60, 300), (300, 1000), (1000, np.inf))


def bin_label(v):
    for lo, hi in RETURNED_BINS:
        if lo < v <= hi:
            return f"({lo:g},{hi:g}]" if np.isfinite(hi) else f">{lo:g}"
    raise ValueError(v)


def gap_label(v):
    for lo, hi in GAP_BINS:
        if lo <= v < hi:
            return f"[{lo:g},{hi:g})" if np.isfinite(hi) else f">={lo:g}"
    raise ValueError(v)


def val_rows(tag):
    rows = []
    for sensor, seq in VAL:
        d = C.load(tag, f"{sensor}/{seq}", "val")
        rows.append(dict(sequence=f"{sensor}/{seq}", score=d["score"], dist=d["top1"], mpgd=d["gt"]))
    return rows


def test_rows(tag):
    per = {}
    for s in C.SEQS:
        d = C.load(tag, s)
        per[s] = dict(score=d["score"], dist=d["top1"], qc=C.classify_queries(d["gt"]), query_idx=d["q"], j_star=d["j"], mpgd=d["gt"])
    return per


def strata(rows, tau):
    """오수락 층: positive/negative_returned_bin, cross_kind_gap_returned, query_kind, city, sequence."""
    out = {}
    for s, r in rows.items():
        pos, neg = r["qc"]["positive"], r["qc"]["negative"]
        wrong = ~(np.isfinite(r["dist"]) & (r["dist"] <= 5))
        fa = np.flatnonzero(np.isfinite(r["score"]) & (r["score"] >= tau) & wrong & (pos | neg))
        for i in fa:
            kind = "positive_wrong_place" if pos[i] else "negative"
            rb = bin_label(float(r["dist"][i]))
            gb = gap_label(int(r["query_idx"][i] - r["j_star"][i]))
            for key, val in (("query_kind", kind), ("returned_bin", rb), ("city", C.CITY_OF[s]), ("sequence", s),
                             ("positive_returned_bin" if pos[i] else "negative_returned_bin", rb),
                             ("cross_kind_gap_returned", f"{kind}|{gb}|{rb}")):
                out.setdefault(key, {}).setdefault(val, 0)
                out[key][val] += 1
    return out


def mean_strata(per_seed):
    keys = {}
    for st in per_seed.values():
        for k, v in st.items():
            keys.setdefault(k, set()).update(v)
    return {k: {v: float(np.mean([st.get(k, {}).get(v, 0) for st in per_seed.values()])) for v in sorted(vals)} for k, vals in keys.items()}


def run(nsd_kind):
    methods = {"nsd": C.tags(nsd_kind), "bevplace": C.tags("bev")}
    seeds = {"nsd": [t.split(":")[1] for t in methods["nsd"]], "bevplace": ["1", "2", "3"]}
    old_freeze = json.loads((OLD / "freeze.json").read_text())
    old_sum = json.loads((OLD / "summary.json").read_text())
    old_fa = json.loads((OLD / "false_accepts.json").read_text())
    freeze, report, fa = dict(thresholds={}), dict(methods={}), dict(per_seed={}, mean_over_seeds={})
    checks = {}
    for method, tags in methods.items():
        freeze["thresholds"][method] = {}
        report["methods"][method] = {}
        fa["per_seed"][method] = {}
        fa["mean_over_seeds"][method] = {}
        per_rows = {}
        for sd, t in zip(seeds[method], tags):
            freeze["thresholds"][method][sd] = main.choose(val_rows(t))
            per_rows[sd] = test_rows(t)
        if method == "bevplace" or nsd_kind == "old":
            checks[f"{method}_tau_equal_previous_freeze"] = all(
                freeze["thresholds"][method][sd][p]["tau"] == old_freeze["thresholds"][method][sd][p]["tau"]
                for sd in seeds[method] for p in main.POINTS)
        for point in main.POINTS:
            per_seed = {}
            for sd in seeds[method]:
                r = main.fixed(per_rows[sd], freeze["thresholds"][method][sd][point]["tau"])
                r["validation"] = {k: v for k, v in freeze["thresholds"][method][sd][point].items() if k != "val_per_sequence"}
                per_seed[sd] = r
            pick = lambda f: main.mean_sd([f(per_seed[s]) for s in seeds[method]])
            first = seeds[method][0]
            agg = dict(city_macro_recall=pick(lambda r: r["city_macro_recall"]),
                       **{k: pick(lambda r, k=k: r["pooled"][k]) for k in
                          ("precision", "recall", "F1", "TP", "FP", "FP_positive_query", "FP_negative_query",
                           "accepted", "FP_per_1000_eligible", "negative_query_false_accept_rate", "ambiguous_accepted")},
                       val_precision=pick(lambda r: r["validation"]["val_precision"]),
                       val_recall=pick(lambda r: r["validation"]["val_recall"]),
                       val_FP=pick(lambda r: r["validation"]["val_FP"]),
                       city_recall={c: pick(lambda r, c=c: r["city_recall"][c]) for c in C.CITY},
                       city_precision={c: pick(lambda r, c=c: r["city"][c]["precision"]) for c in C.CITY},
                       city_FP={c: pick(lambda r, c=c: r["city"][c]["FP"]) for c in C.CITY},
                       dataset={d: {k: pick(lambda r, d=d, k=k: r["dataset"][d][k]) for k in
                                    ("precision", "recall", "TP", "FP", "FP_positive_query", "FP_negative_query", "accepted")}
                                for d in per_seed[first]["dataset"]})
            report["methods"][method][point] = dict(mean_sd=agg, per_seed=per_seed)
            st = {sd: strata(per_rows[sd], freeze["thresholds"][method][sd][point]["tau"]) for sd in seeds[method]}
            for sd in seeds[method]:
                fa["per_seed"][method].setdefault(sd, {})[point] = st[sd]
                assert sum(st[sd].get("query_kind", {}).values()) == per_seed[sd]["pooled"]["FP"]
            fa["mean_over_seeds"][method][point] = mean_strata(st)
            if method == "bevplace" or nsd_kind == "old":
                o = old_sum["methods"][method][point]["mean_sd"]
                checks[f"{method}_{point}_summary_max_abs_err"] = max(
                    abs(agg[k]["mean"] - o[k]["mean"]) for k in ("city_macro_recall", "precision", "recall", "TP", "FP", "val_precision", "val_recall"))
                om = old_fa["mean_over_seeds"][method][point]
                checks[f"{method}_{point}_strata_equal"] = all(
                    abs(fa["mean_over_seeds"][method][point].get(k, {}).get(v, 0) - om[k][v]) < 1e-9
                    for k in ("positive_returned_bin", "negative_returned_bin", "cross_kind_gap_returned", "city", "sequence")
                    if k in om for v in om[k])
    return freeze, report, fa, checks


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsd", default="new", choices=("new", "old", "ref"))
    args = ap.parse_args()
    freeze, report, fa, checks = run(args.nsd)
    print(json.dumps(checks, indent=1))
    if args.nsd == "old":
        assert all(v is True or (isinstance(v, float) and v < 1e-9) for v in checks.values()), checks
        C.write(OUT / "reproduction_previous_nsd.json", dict(status="passed", checks=checks))
        return
    assert all(v is True or (isinstance(v, float) and v < 1e-9) for k, v in checks.items() if k.startswith("bevplace")), checks
    suffix = "" if args.nsd == "new" else "_seeds1to3_reference"
    meta = dict(nsd_ledgers=args.nsd, nsd_seeds=[int(t.split(":")[1]) for t in C.tags(args.nsd)], bevplace_seeds=[1, 2, 3],
                rules="기존 규약(_review_v23_fixed_threshold_20261107.choose): 검증9 합산 sweep, P99·P100은 정밀도 하한에서 최대 재현율의 가장 높은 τ, F1은 max_F1_tau",
                test_used_for_selection=False, checks=checks)
    C.write(OUT / f"freeze{suffix}.json", dict(meta, **freeze))
    C.write(OUT / f"summary{suffix}.json", dict(meta, **report))
    C.write(OUT / f"false_accepts{suffix}.json", dict(meta, **fa))
    for point in main.POINTS:
        for method in ("nsd", "bevplace"):
            a = report["methods"][method][point]["mean_sd"]
            f = fa["mean_over_seeds"][method][point]
            print(f"{point:5s} {method:9s} val {100*a['val_precision']['mean']:.2f}/{100*a['val_recall']['mean']:.2f} city {100*a['city_macro_recall']['mean']:.2f}±{100*a['city_macro_recall']['sd']:.2f} "
                  f"pooled rec {100*a['recall']['mean']:.2f} prec {100*a['precision']['mean']:.2f} TP {a['TP']['mean']:,.0f} FA {a['FP']['mean']:.1f} "
                  f"({a['FP_positive_query']['mean']:.1f}/{a['FP_negative_query']['mean']:.1f})")
            print("      city", {c: f"{100*a['city_recall'][c]['mean']:.1f}/{a['city_FP'][c]['mean']:.1f}" for c in C.CITY})
            print("      pos", f.get("positive_returned_bin", {}), "neg", f.get("negative_returned_bin", {}))
            print("      neg|[30,60)|(25,100]", f.get("cross_kind_gap_returned", {}).get("negative|[30,60)|(25,100]", 0), "seq", f.get("sequence", {}) if point != "F1" else "")


if __name__ == "__main__":
    main_()
