#!/usr/bin/env python3
"""_pdv2_single_threshold_all.py 결과의 독립 재계산(_pr_eval_lib·기존 선택 함수를 쓰지 않는다).

(A) BEVPlace++ + 시간 증거와 NSD의 P99·P100·F1 τ를 검증 합산 점수에서 직접 훑어 다시 고르고, 시험 TP·오수락·도시 재현율을 다시 센다.
(B) 모든 방법·실행·예산에서 시험 합산 점수의 고유 임계값을 전부 훑어 '오수락 ≤ k인 임계값 중 TP 최대'를 다시 구한다.
"""
import json
from pathlib import Path

import numpy as np

from _pdv2_pilot import CITY, TEST, VAL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pdv2_single_threshold_all_20261115"
NSD_TEST = ROOT / "results/pdv2_diffattn_contrast_test_20261104/test"
NSD_VAL = ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104/val"
BEV = ROOT / "results/pdv2_review_response_20261104/bev"
SUMMARY = ROOT / "results/pdv2_deepring_scaleinit_integration_20261110/baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json"
PREFIX = dict(scpp="dump_scpp", fresco="dump_fresco", ring="dump_ring", ringpp="dump_ringpp_i2", ringsharp="dump_ringsharp",
              deepring="dump_deepring", disco="dump_disco", helios="dump_helios", seqot="dump_seqotl2", bevplace="dump_bevplace")


def flags(score, dist, mpgd):
    """(수락 가능, 정답 수락, 오수락, 양성) 표지."""
    pos, neg = mpgd <= 5, mpgd > 25
    ok = np.isfinite(score); right = np.isfinite(dist) & (dist <= 5)
    return ok & pos & right, ok & (neg | (pos & ~right)), pos


def sweep(score, tp, fp):
    """고유 점수(내림차순)마다 그 점수 이상을 수락했을 때의 TP·FP 누적."""
    keep = tp | fp
    s = score[keep]; order = np.argsort(-s, kind="stable"); s = s[order]
    ctp, cfp = np.cumsum(tp[keep][order]), np.cumsum(fp[keep][order])
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1] if len(s) else np.array([], int)
    return s[last], ctp[last], cfp[last]


def load(path, key, ref):
    d = np.load(path); ix = np.searchsorted(d["query_idx"], ref["query_idx"])
    assert np.array_equal(d["query_idx"][ix], ref["query_idx"])
    return d[key][ix].astype(np.float64), d["top1_geo_dist"][ix].astype(np.float64)


def city_recall(tp_seq, npos_seq):
    return float(np.mean([np.mean([tp_seq[f"{a}/{b}"] / npos_seq[f"{a}/{b}"] for a, b in
                                   [n.split("/") for n in names] if npos_seq[f"{a}/{b}"]]) for names in CITY.values()]))


def main() -> None:
    a = json.loads((OUT / "validation_fixed.json").read_text())["methods"]
    b = json.loads((OUT / "test_single_threshold_budgets.json").read_text())["methods"]
    checks = 0
    for m, key in (("nsd", "score"), ("bevt", "score"), ("bevplace", "score_negl2")):
        for seed in (1, 2, 3):
            sc, tp, fp, npos = [], [], [], 0
            for sensor, seq in VAL:
                ref = np.load(NSD_VAL / f"{sensor}_{seq}" / "identity.npz")
                p = NSD_VAL / f"{sensor}_{seq}" / f"diff_s{seed}.npz" if m == "nsd" else BEV / "val" / f"s{seed}" / f"{sensor}_{seq}.npz"
                s, d = load(p, key, ref); t, f, pos = flags(s, d, ref["min_past_gt_dist"])
                sc.append(s); tp.append(t); fp.append(f); npos += int(pos.sum())
            th, ctp, cfp = sweep(np.concatenate(sc), np.concatenate(tp), np.concatenate(fp))
            prec, rec = ctp / (ctp + cfp), ctp / npos
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-300)
            taus = {}
            for name, floor in (("P99", .99), ("P100", 1.)):
                ok = np.flatnonzero(prec >= floor)
                taus[name] = float(th[ok[rec[ok] == rec[ok].max()][0]]) if len(ok) else np.inf
            taus["F1"] = float(th[int(np.argmax(f1))])
            test = {}
            for sensor, seq in TEST:
                ref = np.load(NSD_TEST / f"{sensor}_{seq}" / "identity.npz")
                p = NSD_TEST / f"{sensor}_{seq}" / f"diff_s{seed}.npz" if m == "nsd" else BEV / "test" / f"s{seed}" / f"{sensor}_{seq}.npz"
                s, d = load(p, key, ref); test[f"{sensor}/{seq}"] = (s, *flags(s, d, ref["min_past_gt_dist"]))
            for name, tau in taus.items():
                rec_ = a[m][name]["per_seed"][str(seed)]
                assert tau == rec_["tau"], (m, seed, name, tau, rec_["tau"])
                tps = {q: int((t & (s >= tau)).sum()) for q, (s, t, f, pos) in test.items()}
                fps = sum(int((f & (s >= tau)).sum()) for s, t, f, pos in test.values())
                nps = {q: int(pos.sum()) for q, (s, t, f, pos) in test.items()}
                assert sum(tps.values()) == rec_["TP"] and fps == rec_["FP"], (m, seed, name)
                assert abs(city_recall(tps, nps) - rec_["city_macro_recall"]) < 1e-12
                checks += 4
    summ = json.loads(SUMMARY.read_text())["methods"]
    runs = {"nsd": [(s, None, "score") for s in (1, 2, 3)], "bevt": [(s, None, "score") for s in (1, 2, 3)]}
    for m in PREFIX:
        runs[m] = [(seed, r["source"], summ[m]["score"]) for seed, r in summ[m]["per_seed"].items()]
    for m, rr in runs.items():
        got = {k: dict(TP=[], city=[]) for k in (0, 5, 10, 30, 100)}
        for seed, source, key in rr:
            test = {}
            for sensor, seq in TEST:
                ref = np.load(NSD_TEST / f"{sensor}_{seq}" / "identity.npz")
                p = (NSD_TEST / f"{sensor}_{seq}" / f"diff_s{seed}.npz" if m == "nsd" else
                     BEV / "test" / f"s{seed}" / f"{sensor}_{seq}.npz" if m == "bevt" else ROOT / source / f"{PREFIX[m]}_{sensor}_{seq}.npz")
                s, d = load(p, key, ref); test[f"{sensor}/{seq}"] = (s, *flags(s, d, ref["min_past_gt_dist"]))
            th, ctp, cfp = sweep(np.concatenate([v[0] for v in test.values()]), np.concatenate([v[1] for v in test.values()]),
                                 np.concatenate([v[2] for v in test.values()]))
            for k in got:
                ok = np.flatnonzero(cfp <= k)
                tau = float(th[ok[ctp[ok] == ctp[ok].max()][0]]) if len(ok) else np.inf
                tps = {q: int((t & (s >= tau)).sum()) for q, (s, t, f, pos) in test.items()}
                nps = {q: int(pos.sum()) for q, (s, t, f, pos) in test.items()}
                got[k]["TP"].append(sum(tps.values())); got[k]["city"].append(city_recall(tps, nps))
        for k in got:
            assert got[k]["TP"] == [int(v) for v in b[m][str(k)]["TP"]["values"]], (m, k, got[k]["TP"], b[m][str(k)]["TP"]["values"])
            assert np.allclose(got[k]["city"], b[m][str(k)]["city_macro_recall"]["values"], atol=1e-12, rtol=0), (m, k)
            checks += 2 * len(rr)
    (OUT / "verification.json").write_text(json.dumps(dict(status="passed", checks=checks, independent_of=["_pr_eval_lib", "_review_v23_fixed_threshold_20261107"]), indent=1) + "\n")
    print("VERIFIED", checks)


if __name__ == "__main__":
    main()
