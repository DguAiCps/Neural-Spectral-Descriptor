#!/usr/bin/env python3
"""원장만으로 계산되는 그 밖의 NSD 수치.

1) 반환 오류 분해 표(appdx_aliasing)의 NSD 열(도시별 오류·먼 오류율, 도시 매크로)과 합산 'NSD 남은 오류 중 먼 장소 비율'.
   old 꼬리표 원장으로 _pdv2_aliasing_decomposition.city_rates의 NSD 열을 재현해 계산 경로를 확인한 뒤 new 꼬리표 원장(NSD 시드 4~6)으로 계산한다.
2) 시드별 학습 기여, 도시별 학습 기여, 방향×도시별 학습 기여, 데이터셋 MCD 최대 F1 등 서술 수치를 모은다.
출력: results/pdv2_report_model_20261116/analysis/extras/extras.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_model_common_20261116 as C  # noqa: E402
import _pdv2_aliasing_decomposition as A  # noqa: E402


def nsd_alias(kind):
    per = {}
    pooled = np.zeros(3)
    for t in C.tags(kind):
        per[t] = {}
        for s in C.SEQS:
            d = C.load(t, s)
            pos = d["gt"] <= 5
            c = A.classify(d["top1"][pos])
            cnt = np.bincount(c, minlength=3)
            pooled += cnt
            per[t][s] = cnt
    cities = {}
    for city, seqs in C.CITY.items():
        vals = {"nsd_error": [], "nsd_distant": []}
        for t in per:
            rows = [per[t][s] for s in seqs if per[t][s].sum()]
            vals["nsd_error"].append(np.mean([r[1:].sum() / r.sum() for r in rows]))
            vals["nsd_distant"].append(np.mean([r[2] / r.sum() for r in rows]))
        cities[city] = {k: 100 * float(np.mean(v)) for k, v in vals.items()}
    cities["macro"] = {k: float(np.mean([cities[c][k] for c in C.CITY])) for k in ("nsd_error", "nsd_distant")}
    share = None
    # 원고 'Distant places still account for X% of NSD's remaining errors': 시작점별 합산 비율의 평균
    shares = []
    for t in per:
        tot = sum(per[t][s] for s in C.SEQS)
        shares.append(tot[2] / tot[1:].sum())
    share = 100 * float(np.mean(shares))
    return cities, share


def main():
    old_cities, old_share = nsd_alias("old")
    ref = A.city_rates(A.transitions())
    err = max(abs(old_cities[c][k] - ref[c][k]) for c in old_cities for k in ("nsd_error", "nsd_distant"))
    print("재현(이전 NSD 열) 최대 오차", err, "| 이전 먼 오류 비율", round(old_share, 2))
    assert err < 1e-9
    new_cities, new_share = nsd_alias("new")
    main_dir = C.OUT / "main"
    learning = json.loads((main_dir / "learning.json").read_text())
    city = json.loads((main_dir / "city.json").read_text())
    dseq = json.loads((main_dir / "direction_sequence_summary.json").read_text())["comparisons"]["diff-fixed"]
    summary = json.loads((main_dir / "summary.json").read_text())
    ds = json.loads((main_dir / "dataset_methods_summary.json").read_text())
    out = dict(
        aliasing_nsd_columns=dict(new=new_cities, previous=old_cities, reproduction_max_abs_err=err,
                                  nsd_distant_share_of_errors_pooled_pct=dict(new=new_share, previous=old_share),
                                  note="% 단위. macro nsd_error = 100(1−R@1)."),
        learning_per_seed_pp={k: v["per_run"] for k, v in learning["seeds4to6"].items() if k in C.METRICS},
        learning_city_delta_pp=city["learning_city_delta_pp"],
        learning_city_negative=[(c, k, round(v[k], 3)) for c, v in city["learning_city_delta_pp"].items() for k in C.METRICS if v[k] < 0],
        learning_direction_city_delta_pp={g: {c: {k: v[k]["delta_pp"] for k in C.METRICS} for c, v in dseq[g]["cities"].items()}
                                          for g in ("forward_only", "reverse_only")},
        learning_direction_city_negative=[(g, c, k, round(v[k]["delta_pp"], 3)) for g in ("forward_only", "reverse_only")
                                          for c, v in dseq[g]["cities"].items() for k in C.METRICS if v[k]["delta_pp"] < 0],
        nsd_per_seed_percent={str(sd): {k: 100 * summary["per_case"][f"diff_s{sd}"]["macro"][k] for k in C.METRICS} for sd in C.NEW_SEEDS},
        nsd_minus_bev_text=dict(
            AP_pp=100 * (summary["mean"]["diff"]["AP"] - summary["mean"]["bev_native"]["AP"]),
            F1_pp=100 * (summary["mean"]["diff"]["max_F1"] - summary["mean"]["bev_native"]["max_F1"]),
            R100P_pp=100 * (summary["mean"]["diff"]["R@100P"] - summary["mean"]["bev_native"]["R@100P"]),
            R1_pp=100 * (summary["mean"]["diff"]["R@1"] - summary["mean"]["bev_native"]["R@1"])),
        dataset_nsd_percent={f: {k: 100 * d["methods"]["nsd"][k]["mean"] for k in C.METRICS} for f, d in ds["datasets"].items()},
    )
    C.write(C.OUT / "extras" / "extras.json", out)
    print("aliasing new", {c: {k: round(v, 2) for k, v in x.items()} for c, x in new_cities.items()}, "distant share", round(new_share, 2))
    print("learning per seed", {k: [round(x, 3) for x in v] for k, v in out["learning_per_seed_pp"].items()})
    print("learning city negatives", out["learning_city_negative"])
    print("direction city negatives", out["learning_direction_city_negative"])
    print("NSD per seed", {s: {k: round(v, 3) for k, v in x.items()} for s, x in out["nsd_per_seed_percent"].items()})
    print("NSD-BEV", {k: round(v, 3) for k, v in out["nsd_minus_bev_text"].items()})


if __name__ == "__main__":
    main()
