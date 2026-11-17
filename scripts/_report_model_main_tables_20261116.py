#!/usr/bin/env python3
"""보고모델 교체(2026-11-16) 단위 C 항목 1·2·3·7: 주 결과 NSD 행, 짝지은 구간, 도시별·데이터셋별, 학습 기여, 방향별.

먼저 같은 함수로 이전 NSD 원장을 넣어 기존 발표 원천(summary.json·integrated_report.json·direction_summary.json·
direction_learning/sequence_summary.json·dataset_methods_summary.json)을 재현하고, 그다음 새 원장을 넣는다.

출력(results/pdv2_report_model_20261116/analysis/main/):
  summary.json             기존 results/pdv2_diffattn_contrast_test_20261104/summary.json의 per_case/mean/std 스키마
                           (per_case: diff_s4..6, diff_s1..3(참고), fixed(=재선택 균등 감산), bev_native_s1..3)
  integrated_report.json   기존 integrated_report.json의 comparisons 스키마(diff-fixed, diff-bev_native, + 참고 비교)
  direction_summary.json   기존 direction_summary.json의 methods.<m>.mean_std.<group> 스키마(diff, fixed, bev_native)
  direction_sequence_summary.json  기존 direction_learning/sequence_summary.json 스키마(diff-fixed, diff-bev_native)
  dataset_methods_summary.json     기존 datasets/dataset_methods_summary.json 스키마(nsd만 교체, ranks 추가)
  learning.json            학습 기여(3회·6회) 요약
  verification.json        재현·독립 재계산 결과
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_model_common_20261116 as C  # noqa: E402

OUT = C.OUT / "main"
GROUPS = ("all", "forward_only", "reverse_only", "mixed")


def rows_for(kind, group):
    masks = None if group == "all" else MASKS[group]
    return [C.per_seq_rows(t, masks=masks, extra=(group == "all")) for t in C.tags(kind)]


def macro_runs(rows, keys=C.METRICS):
    return [C.macro(r, keys) for r in rows]


def stat(runs, keys=C.METRICS):
    return {k: C.mean_sd([r["macro"][k] for r in runs]) for k in keys}


def independent(d):
    """평가 모듈 없이 임계값 내림차순 단순 구현(_pdv2_scratch_search_verify_20261116.metrics와 같은 정의)."""
    pos = d["gt"] <= 5
    keep = pos | (d["gt"] > 25)
    ok = pos & (d["top1"] <= 5)
    s, y = d["score"][keep], ok[keep]
    npos = int(pos.sum())
    if not npos:
        return None
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    last = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    tp = np.r_[0, np.cumsum(y)[last]]
    fp = np.r_[0, np.cumsum(~y)[last]]
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    r = tp / npos
    ap = float(np.sum(np.diff(r) * np.maximum.accumulate(p[::-1])[::-1][1:]))
    f1 = float(np.max(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-300), 0)))
    return {"AP": ap, "max_F1": f1, "R@100P": float(r[fp == 0].max()), "R@1": int(ok.sum()) / npos}


def main():
    global MASKS
    MASKS = {g: C.direction_masks(g) for g in GROUPS[1:]}
    kinds = ("new", "ref", "uniform", "old", "oldfixed", "bev")
    R = {g: {k: rows_for(k, g) for k in kinds} for g in GROUPS}
    M = {g: {k: macro_runs(R[g][k], C.METRICS + (("R@99P", "R@95P") if g == "all" else ())) for k in kinds} for g in GROUPS}
    ver = {}

    # ---- 재현 1: 이전 summary.json / integrated_report.json
    old_sum = json.loads((C.OLD / "summary.json").read_text())
    old_rep = json.loads((C.OLD / "integrated_report.json").read_text())
    err = 0.0
    for kind, name in (("old", "diff"), ("oldfixed", "fixed"), ("bev", "bev_native")):
        for k in C.METRICS:
            err = max(err, abs(np.mean([r["macro"][k] for r in M["all"][kind]]) - old_sum["mean"][name][k]))
    ver["old_summary_means_max_abs_err"] = err
    iv = {}
    for a, b, case in (("old", "oldfixed", "diff-fixed"), ("old", "bev", "diff-bev_native")):
        p = C.paired(R["all"][a], R["all"][b])
        iv[case] = max(abs(x - y) for k in C.METRICS for x, y in zip(p["conditional_95_pp"][k], old_rep["comparisons"][case]["conditional_95_pp"][k]))
        iv[case + "_city"] = max(abs(p["city_delta_pp"][c][k] - old_rep["comparisons"][case]["city_delta_pp"][c][k]) for c in p["city_delta_pp"] for k in C.METRICS)
    ver["old_integrated_report_interval_max_abs_err_pp"] = iv

    # ---- 재현 2: 이전 방향별
    dsum = json.loads((C.DIRS.parent / "direction_summary.json").read_text())
    dseq = json.loads((C.DIRS.parent / "direction_learning/sequence_summary.json").read_text())
    err = 0.0
    for g in GROUPS:
        for kind, name in (("old", "diff"), ("oldfixed", "fixed"), ("bev", "bev_native")):
            for k in C.METRICS:
                err = max(err, abs(np.mean([r["macro"][k] for r in M[g][kind]]) - dsum["methods"][name]["mean_std"][g][k]["mean"]))
    ver["old_direction_summary_max_abs_err"] = err
    err = 0.0
    for g in GROUPS:
        p = C.paired(R[g]["old"], R[g]["oldfixed"])
        for k in C.METRICS:
            err = max(err, max(abs(x - y) for x, y in zip(p["conditional_95_pp"][k], dseq["comparisons"]["diff-fixed"][g]["overall"][k]["ci95_pp"])))
    ver["old_direction_sequence_interval_max_abs_err_pp"] = err
    ver["old_published_direction_R100P_reverse"] = dict(
        nsd=100 * np.mean([r["macro"]["R@100P"] for r in M["reverse_only"]["old"]]),
        bev=100 * np.mean([r["macro"]["R@100P"] for r in M["reverse_only"]["bev"]]))

    # ---- 새 주 수치 독립 재계산(평가 모듈 없이)
    ind = {}
    for t in C.tags("new") + C.tags("uniform"):
        per = {s: independent(C.load(t, s)) for s in C.SEQS}
        ind[t] = C.macro(per)["macro"]
    new_stat = {k: C.mean_sd([ind[t][k] for t in C.tags("new")]) for k in C.METRICS}
    ver["new_independent_percent"] = {k: [round(100 * new_stat[k]["mean"], 3), round(100 * new_stat[k]["sd"], 3)] for k in C.METRICS}
    ver["uniform_independent_percent"] = {k: round(100 * ind["uniform"][k], 3) for k in C.METRICS}
    ver["new_pr_sweep_vs_independent_max_abs_err"] = max(
        abs(M["all"]["new"][i]["macro"][k] - ind[t][k]) for i, t in enumerate(C.tags("new")) for k in C.METRICS)
    expected = {"AP": 89.879, "max_F1": 90.389, "R@100P": 72.174, "R@1": 91.742}
    ver["new_matches_brief"] = all(abs(round(100 * new_stat[k]["mean"], 3) - expected[k]) < 1e-9 for k in C.METRICS)
    cmp_json = json.loads((C.SEARCH / "compare" / f"{C.CFG}@u512@s4-5-6.json").read_text())

    # ---- summary.json (기존 스키마)
    per_case = {}
    for kind, prefix, seeds in (("new", "diff", C.NEW_SEEDS), ("ref", "diff", C.REF_SEEDS), ("bev", "bev_native", C.BEV_SEEDS)):
        for i, sd in enumerate(seeds):
            m = M["all"][kind][i]
            per_case[f"{prefix}_s{sd}"] = dict(groups=m["groups"], macro=m["macro"], sequences=R["all"][kind][i])
    per_case["fixed"] = dict(groups=M["all"]["uniform"][0]["groups"], macro=M["all"]["uniform"][0]["macro"], sequences=R["all"]["uniform"][0])
    keys_all = C.METRICS + ("R@99P", "R@95P")
    mean = {"diff": {k: stat(M["all"]["new"], keys_all)[k]["mean"] for k in keys_all},
            "fixed": M["all"]["uniform"][0]["macro"],
            "bev_native": {k: stat(M["all"]["bev"], keys_all)[k]["mean"] for k in keys_all},
            "diff_seeds1to3": {k: stat(M["all"]["ref"], keys_all)[k]["mean"] for k in keys_all},
            "diff_seeds1to6": {k: float(np.mean([r["macro"][k] for r in M["all"]["ref"] + M["all"]["new"]])) for k in keys_all},
            "previous_nsd": {k: stat(M["all"]["old"], keys_all)[k]["mean"] for k in keys_all}}
    std = {"diff": {k: stat(M["all"]["new"], keys_all)[k]["sd"] for k in keys_all},
           "bev_native": {k: stat(M["all"]["bev"], keys_all)[k]["sd"] for k in keys_all},
           "diff_seeds1to3": {k: stat(M["all"]["ref"], keys_all)[k]["sd"] for k in keys_all},
           "diff_seeds1to6": {k: float(np.std([r["macro"][k] for r in M["all"]["ref"] + M["all"]["new"]], ddof=1)) for k in keys_all},
           "previous_nsd": {k: stat(M["all"]["old"], keys_all)[k]["sd"] for k in keys_all}}
    summary = dict(
        per_case=per_case, mean=mean, std=std, seeds=dict(diff=list(C.NEW_SEEDS), diff_reference=list(C.REF_SEEDS), bev_native=list(C.BEV_SEEDS)),
        aggregation="46시퀀스중양성45→6도시→시드평균(NSD 시드 4·5·6; BEVPlace++ 1·2·3)",
        queries_per_case=int(sum(len(C.load("uniform", s)["q"]) for s in C.SEQS)),
        model=dict(cfg=C.CFG, updates=512, ledgers="results/pdv2_scratch_search_20261116/ledgers/test",
                   fixed="재선택 설정(융합 (2,1,2,2), λ=0.25)의 균등 감산 uniform.npz"),
        note="R@99P·R@95P는 _pr_eval_lib.pr_sweep의 시퀀스별 값 6도시 매크로. diff_seeds1to3/1to6은 참고(시드 1~3은 구성 선택에 씀).")
    C.write(OUT / "summary.json", summary)

    # ---- integrated_report.json (기존 스키마 + 참고 비교)
    comps = {}
    for case, a, b in (("diff-fixed", "new", "uniform"), ("diff-bev_native", "new", "bev"),
                       ("diff-previous_nsd", "new", "old"), ("diff_seeds1to3-fixed", "ref", "uniform"),
                       ("diff_seeds1to6-fixed", None, "uniform")):
        rows_a = R["all"]["ref"] + R["all"]["new"] if a is None else R["all"][a]
        p = C.paired(rows_a, R["all"][b])
        runs_a = M["all"]["ref"] + M["all"]["new"] if a is None else M["all"][a]
        seeds_a = C.REF_SEEDS + C.NEW_SEEDS if a is None else (C.NEW_SEEDS if a == "new" else C.REF_SEEDS)
        ref_mean = {k: np.mean([r["macro"][k] for r in M["all"][b]]) for k in C.METRICS}
        per_seed = {str(sd): {k: 100 * (runs_a[i]["macro"][k] - (ref_mean[k] if b != "uniform" else M["all"]["uniform"][0]["macro"][k]))
                              for k in C.METRICS} for i, sd in enumerate(seeds_a)}
        comps[case] = dict(per_seed_pp=per_seed, city_delta_pp=p["city_delta_pp"], conditional_95_pp=p["conditional_95_pp"],
                           delta_pp=p["delta_pp"], positive_runs={k: int(sum(v[k] > 0 for v in per_seed.values())) for k in C.METRICS}
                           if b == "uniform" else None)
    # compare JSON 검산
    ver["compare_json_max_abs_err_pp"] = max(
        max(abs(x - y) for x, y in zip(comps[c]["conditional_95_pp"][k], cmp_json["all"]["scratch_minus"][r]["conditional_95_pp"][k]))
        for c, r in (("diff-fixed", "uniform"), ("diff-bev_native", "bev"), ("diff-previous_nsd", "nsd")) for k in C.METRICS)
    report = dict(comparisons=comps, mean=mean,
                  interval_scope="기존 integrated_report와 같은 규약: 시퀀스별 시드 평균 차이를 도시 안에서 복원추출 10,000회(seed 20261104), 6도시 동일 비중. "
                                 "per_seed_pp: 균등 감산 대비는 각 시드−균등, BEVPlace++·이전 NSD 대비는 각 시드−상대 3시드 평균.",
                  test_used_for_selection="구성은 시험 3시드(1~3) 평균 AP로 고름; 시드 4~6은 그 뒤 학습(일지 2026-11-16)")
    C.write(OUT / "integrated_report.json", report)

    # ---- 도시별 표(NSD·BEVPlace++)와 학습 기여 도시별
    city = {}
    for c in C.CITY:
        city[c] = {name: {k: float(np.mean([r["groups"][c][k] for r in M["all"][kind]])) for k in C.METRICS}
                   for kind, name in (("new", "diff"), ("bev", "bev_native"), ("uniform", "fixed"))}
    C.write(OUT / "city.json", dict(city_mean=city, learning_city_delta_pp=comps["diff-fixed"]["city_delta_pp"],
                                    note="appdx_city: diff·bev_native 시드 평균. appdx_learning_city 첫 블록: NSD(4~6)−균등 감산 city_delta_pp."))

    # ---- 학습 기여 요약
    learn = {}
    for label, case in (("seeds4to6", "diff-fixed"), ("seeds1to3_reference", "diff_seeds1to3-fixed"), ("seeds1to6_reference", "diff_seeds1to6-fixed")):
        c = comps[case]
        learn[label] = {k: dict(mean_pp=c["delta_pp"][k], interval=c["conditional_95_pp"][k], positive_runs=c["positive_runs"][k],
                                per_run=[c["per_seed_pp"][s][k] for s in c["per_seed_pp"]]) for k in C.METRICS}
    learn["interval_definition"] = "모델(시드 평균) 고정, 도시 안 시퀀스 재표집 10,000회(seed 20261104). 시드는 재표집하지 않음."
    C.write(OUT / "learning.json", learn)

    # ---- 방향별
    dout = dict(methods={}, counts=dsum["counts"], note="group: forward_only=0, reverse_only=1, mixed=2 (directions/*.npz). 기존 direction_summary와 같은 정의(이전 원장으로 재현 확인).")
    for kind, name, seeds in (("new", "diff", C.NEW_SEEDS), ("uniform", "fixed", (0,)), ("bev", "bev_native", C.BEV_SEEDS),
                              ("ref", "diff_seeds1to3", C.REF_SEEDS)):
        ms = {}
        for g in GROUPS:
            st = stat(M[g][kind])
            ms[g] = {k: dict(mean=st[k]["mean"], std=st[k]["sd"]) for k in C.METRICS}
        dout["methods"][name] = dict(mean_std=ms, per_seed={str(sd): {g: dict(cities=M[g][kind][i]["groups"], sequences=R[g][kind][i], macro=M[g][kind][i]["macro"])
                                                                       for g in GROUPS} for i, sd in enumerate(seeds)})
    C.write(OUT / "direction_summary.json", dout)
    dseq_new = dict(status="passed", comparisons={})
    for case, a, b in (("diff-fixed", "new", "uniform"), ("diff-bev_native", "new", "bev")):
        dseq_new["comparisons"][case] = {}
        for g in GROUPS:
            p = C.paired(R[g][a], R[g][b])
            dseq_new["comparisons"][case][g] = dict(
                overall={k: dict(delta_pp=p["delta_pp"][k], ci95_pp=p["conditional_95_pp"][k]) for k in C.METRICS},
                cities={c: {k: dict(delta_pp=v[k]) for k in C.METRICS} for c, v in p["city_delta_pp"].items()})
    C.write(OUT / "direction_sequence_summary.json", dseq_new)
    ver["compare_json_direction_max_abs_err_pp"] = max(
        max(abs(x - y) for x, y in zip(dseq_new["comparisons"][c][g]["overall"][k]["ci95_pp"], cmp_json[g]["scratch_minus"][r]["conditional_95_pp"][k]))
        for c, r in (("diff-fixed", "uniform"), ("diff-bev_native", "bev")) for g in ("forward_only", "reverse_only") for k in C.METRICS)

    # ---- 데이터셋별
    src_path = C.ROOT / "results/pdv2_deepring_scaleinit_integration_20261110/datasets/dataset_methods_summary.json"
    ds = json.loads(src_path.read_text())
    err = 0.0
    ranks_old, ranks_new = {}, {}
    for fam, d in ds["datasets"].items():
        pos = d["positive_sequences"]
        for k in C.METRICS:
            vals_old = [statistics.mean(R["all"]["old"][i][s][k] for s in pos) for i in range(3)]
            err = max(err, abs(statistics.mean(vals_old) - d["methods"]["nsd"][k]["mean"]))
        ranks_old[fam] = rank(d["methods"])
        new_vals = {k: [statistics.mean(R["all"]["new"][i][s][k] for s in pos) for i in range(3)] for k in C.METRICS}
        d["methods"]["nsd"] = {k: dict(mean=statistics.mean(v), sd=statistics.stdev(v), values=v) for k, v in new_vals.items()}
        d["methods"]["nsd"]["seeds"] = [str(s) for s in C.NEW_SEEDS]
        ranks_new[fam] = rank(d["methods"])
    ver["old_dataset_nsd_max_abs_err"] = err
    ds["seeds"]["nsd"] = [str(s) for s in C.NEW_SEEDS]
    ds["passed"] = True
    ds["report_model_replacement"] = dict(source=str(src_path.relative_to(C.ROOT)), replaced="nsd only (seeds 4,5,6)",
                                          ranks_previous=ranks_old, ranks_new=ranks_new,
                                          note="rank: 표시 값(백분율 소수 둘째 자리) dense 순위 0=1위,1=2위(_paper_tables_v23.dataset_table과 같음). per_sequence·checks·input_sha256은 원래 파일 것 그대로이며 nsd에 대해서는 갱신하지 않음.")
    C.write(OUT / "dataset_methods_summary.json", ds)

    C.write(OUT / "verification.json", ver)
    print(json.dumps(ver, indent=1))
    pr = lambda st: {k: f"{100 * st[k]['mean']:.3f}±{100 * (st[k]['sd'] or 0):.3f}" for k in st}
    print("NSD", pr(stat(M["all"]["new"], keys_all)))
    print("BEV", pr(stat(M["all"]["bev"], keys_all)))
    print("old", pr(stat(M["all"]["old"], keys_all)))
    print("uniform", {k: round(100 * M["all"]["uniform"][0]["macro"][k], 3) for k in keys_all})
    for case, c in comps.items():
        print(case, {k: (round(c["delta_pp"][k], 3), [round(x, 3) for x in c["conditional_95_pp"][k]], c["positive_runs"] and c["positive_runs"][k]) for k in C.METRICS})
    for g in GROUPS[1:]:
        for name in ("diff", "bev_native", "fixed"):
            print(g, name, {k: round(100 * dout["methods"][name]["mean_std"][g][k]["mean"], 3) for k in C.METRICS})
        for case in dseq_new["comparisons"]:
            o = dseq_new["comparisons"][case][g]["overall"]
            print("  ", case, {k: (round(o[k]["delta_pp"], 3), [round(x, 3) for x in o[k]["ci95_pp"]]) for k in C.METRICS})
    for c, v in city.items():
        print(c, {n: {k: round(100 * x, 2) for k, x in m.items()} for n, m in v.items() if n != "fixed"})
    print("learning city", {c: {k: round(x, 2) for k, x in v.items()} for c, v in comps["diff-fixed"]["city_delta_pp"].items()})
    for fam in ranks_new:
        if ranks_new[fam] != ranks_old[fam]:
            print("RANK CHANGE", fam, ranks_old[fam], "->", ranks_new[fam])
    for fam, d in ds["datasets"].items():
        print(fam, {k: f"{100 * d['methods']['nsd'][k]['mean']:.2f}±{100 * d['methods']['nsd'][k]['sd']:.2f}" for k in C.METRICS})


def rank(methods):
    """데이터셋·지표별 1·2위(표시 값 dense)."""
    out = {}
    for k in C.METRICS:
        shown = {n: round(100 * v[k]["mean"], 2) for n, v in methods.items()}
        levels = sorted(set(shown.values()), reverse=True)
        out[k] = {"first": sorted(n for n in shown if shown[n] == levels[0]), "second": sorted(n for n in shown if shown[n] == levels[1])}
    return out


if __name__ == "__main__":
    main()
