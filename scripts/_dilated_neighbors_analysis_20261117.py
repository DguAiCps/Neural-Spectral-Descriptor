#!/usr/bin/env python3
"""dilated 이웃 변형의 원장 분석(호스트, torch 없음). scripts/_report_model_common_20261116.py의 함수를 그대로 쓴다.

시험46: 변형별 3시드 평균±SD, 짝지은 95% 구간(도시 안 시퀀스 재표집 10,000회), 방향별(순·역방향 전용).
검증9: 변형별 agg.json(S.agg의 실제 검색 지표)에서 시드 4~6 평균±SD.

python3 scripts/_dilated_neighbors_analysis_20261117.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import _report_model_common_20261116 as C  # noqa: E402

OUT = C.ROOT / "results/pdv2_dilated_20261117"
CFG = C.CFG
SEEDS = (4, 5, 6)
VARIANTS = [v for v in ("d2", "d3", "n20") if (OUT / v / "agg.json").exists()]
K = C.METRICS


def vload(variant: str, model: str, seq: str, split: str = "test") -> dict:
    p = OUT / variant / "ledgers" / split / seq.replace("/", "_") / (model + ".npz")
    with np.load(p) as z:
        return dict(score=z["score"].astype(np.float64), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"], q=z["query_idx"], j=z["j_star"])


def rows(tag: str, masks=None) -> dict:
    """tag: 'nsd:<s>', 'uniform', '<v>:<s>', '<v>:uniform'."""
    out = {}
    for s in C.SEQS:
        if tag == "uniform" or tag.startswith("nsd:"):
            d = C.load("uniform" if tag == "uniform" else "new:" + tag.split(":")[1], s)
        elif tag.startswith("bev"):
            d = C.load(tag, s)
        else:
            v, sd = tag.split(":")
            d = vload(v, "uniform" if sd == "uniform" else f"{CFG}@s{sd}@u512", s)
        ref = C.load("uniform", s)
        if not tag.startswith("bev"):
            assert np.array_equal(d["q"], ref["q"])      # 공통 질의 집합(BEV 원장은 같은 질의 순서인지 C 쪽에서 확인됨)
        out[s] = C.seq_metric(d, None if masks is None else masks[s])
    return out


def summary(tagset: list, masks=None) -> tuple:
    per = [rows(t, masks) for t in tagset]
    macros = [C.macro(r)["macro"] for r in per]
    return per, {k: C.mean_sd([100 * m[k] for m in macros]) for k in K}, macros


def main() -> None:
    groups = dict(all=None, forward_only=C.direction_masks("forward_only"), reverse_only=C.direction_masks("reverse_only"))
    res = dict(variants=VARIANTS, seeds=SEEDS, note="단위 %, 구간은 %p. 시드 평균의 시퀀스별 차이를 도시 안에서 재표집.")
    for g, masks in groups.items():
        sets = {"nsd": [f"nsd:{s}" for s in SEEDS], "uniform": ["uniform"],
                "bev": [f"bev:{s}" for s in C.BEV_SEEDS], "bevt": [f"bevt:{s}" for s in C.BEV_SEEDS]}
        for v in VARIANTS:
            sets[v] = [f"{v}:{s}" for s in SEEDS]
            sets[v + "_uniform"] = [f"{v}:uniform"]
        per, summ = {}, {}
        for name, tagset in sets.items():
            per[name], summ[name], macros = summary(tagset, masks)
            summ[name]["per_seed"] = [{k: 100 * m[k] for k in K} for m in macros]
        pairs = [("nsd", "uniform")]
        for v in VARIANTS:
            pairs += [(v, "nsd"), (v, v + "_uniform"), (v + "_uniform", "uniform"), (v + "_uniform", "nsd"), (v, "bev"), (v, "bevt")]
        if "d2" in VARIANTS:
            pairs += [(v, "d2") for v in VARIANTS if v != "d2"]
        comp = {f"{a}-{b}": C.paired(per[a], per[b]) for a, b in pairs}
        # 시드별 학습 이득(변형 시드 − 자기 균등 감산) AP
        seedgain = {}
        for v in ["nsd"] + VARIANTS:  # noqa
            u = summ["uniform" if v == "nsd" else v + "_uniform"]["per_seed"][0]
            seedgain[v] = [{k: s[k] - u[k] for k in K} for s in summ[v]["per_seed"]]
        res[g] = dict(summary=summ, paired=comp, seed_learning_gain_pp=seedgain)
        print("==", g)
        for name, s in summ.items():
            print(f"  {name:12s}", "  ".join(f"{k} {s[k]['mean']:.3f}±{(s[k]['sd'] or 0):.3f}" for k in K))
        for name, c in comp.items():
            print(f"  {name:22s}", "  ".join(f"{k} {c['delta_pp'][k]:+.3f} [{c['conditional_95_pp'][k][0]:+.3f},{c['conditional_95_pp'][k][1]:+.3f}]" for k in K))
    # 도시별 학습 이득 AP
    res["city_AP_delta_pp"] = {n: res["all"]["paired"][n]["city_delta_pp"] for n in res["all"]["paired"]}
    # 검증9 실제 검색(S.agg)
    val = {}
    ref = json.loads((C.SEARCH / "agg.json").read_text())["val"]["models"]
    val["nsd"] = {k: C.mean_sd([100 * ref[f"{CFG}@s{s}@u512"]["macro"][k] for s in SEEDS]) for k in K}
    val["uniform"] = {k: dict(mean=100 * ref["uniform"]["macro"][k], sd=None) for k in K}
    for v in VARIANTS:
        a = json.loads((OUT / v / "agg.json").read_text())["val"]["models"]
        val[v] = {k: C.mean_sd([100 * a[f"{CFG}@s{s}@u512"]["macro"][k] for s in SEEDS]) for k in K}
        val[v + "_uniform"] = {k: dict(mean=100 * a["uniform"]["macro"][k], sd=None) for k in K}
    res["val9"] = val
    print("== val9")
    for name, s in val.items():
        print(f"  {name:12s}", "  ".join(f"{k} {s[k]['mean']:.3f}±{(s[k]['sd'] or 0):.3f}" for k in K))
    # 접두 검사·주의 가중치
    pre = {}
    for v in VARIANTS:
        checks = [c for p in sorted((OUT / v / "prefix/test").glob("*.json")) for c in json.loads(p.read_text())["checks"]]
        pre[v] = dict(n=len(checks), lag=json.loads(next((OUT / v / "prefix/test").glob("*.json")).read_text())["lag"],
                      all_winner_equal=all(c["winner_equal"] for c in checks), max_score_error=max(c["score_error"] for c in checks),
                      max_output_error=max(c["output_max_error"] for c in checks),
                      one_shorter_changed_fraction=float(np.mean([c["one_shorter_score_change"] > 1e-4 for c in checks])))
    res["prefix"] = pre
    res["attention"] = {v: json.loads((OUT / v / "attention.json").read_text()) for v in VARIANTS if (OUT / v / "attention.json").exists()}
    print("== prefix", pre)
    C.write(OUT / "analysis.json", res)


if __name__ == "__main__":
    main()
