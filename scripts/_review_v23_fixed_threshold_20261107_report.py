#!/usr/bin/env python3
"""고정 기준값의 시험 오수락을 사전 정의 층으로 나누고 원고용 표·문장을 만든다.

falseaccepts: freeze.json의 τ로 시험 오수락 전체 목록(csv)과 층별 집계(false_accepts.json)를 만든다.
  위치·시간·방향은 원장이 실제 쓴 주행 캐시의 poses_gt·timestamps만 쓴다. 원시 점군·ICP는 쓰지 않는다.
tables: summary.json·false_accepts.json으로 LaTeX 표와 manuscript_integration.json을 만든다.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import _review_v23_fixed_threshold_20261107 as main
from _pdv2_pilot import CITY, TEST

REPO, OUT = main.REPO, main.OUT
DIRECTION = REPO / "results/pdv2_review_direction_yaw_20261105/directions"
DIRECTION_LABELS = ("forward_only", "reverse_only", "mixed", "other_positive")
RETURNED_BINS = ((5, 7.5), (7.5, 10), (10, 25), (25, 100), (100, np.inf))
NEAREST_BINS = ((0, 5), (25, 50), (50, 100), (100, np.inf))
GAP_BINS = ((30, 60), (60, 300), (300, 1000), (1000, np.inf))
CITY_OF = {s: c for c, names in CITY.items() for s in names}
LABELS = {"nsd": "NSD", "bevplace": "BEVPlace++"}


def bin_label(value: float, bins) -> str:
    """(lo, hi] 구간 이름. 첫 구간 하한 0은 [0, hi]."""
    for lo, hi in bins:
        if (lo < value or (lo == 0 and value >= 0)) and value <= hi:
            return f"({lo:g},{hi:g}]" if np.isfinite(hi) else f">{lo:g}"
    raise ValueError(value)


def gap_label(value: int) -> str:
    for lo, hi in GAP_BINS:
        if lo <= value < hi:
            return f"[{lo:g},{hi:g})" if np.isfinite(hi) else f">={lo:g}"
    raise ValueError(value)


def headings(pos: np.ndarray) -> tuple:
    """방향 층과 같은 규칙의 5키프레임 GT 진행벡터와 유효 여부."""
    index = np.arange(len(pos))
    delta = pos - pos[np.maximum(index - 5, 0)]
    length = np.linalg.norm(delta, axis=1)
    return delta / np.maximum(length[:, None], 1e-12), (index >= 5) & (length >= 1.)


def context(sequence: str) -> dict:
    """원장이 쓴 캐시(해시 확인)의 위치·시간과 방향 층을 읽는다."""
    name = sequence.replace("/", "_")
    record = main.read(main.NSD_TEST / "test" / name / "reference_validation.json")
    (path, expected), = [(p, h) for p, h in record["input_sha256"].items() if "/cache_" in p]
    assert main.digest(REPO / path) == expected, path
    with np.load(REPO / path) as data:
        pos, stamps = data["poses_gt"][:, :3, 3].astype(np.float64), data["timestamps"].astype(np.float64)
    assert len(pos) == len(stamps)
    head, valid = headings(pos)
    with np.load(DIRECTION / f"{name}.npz") as d:
        direction = dict(query_idx=d["query_idx"], group=d["group"])
    with np.load(main.NSD_TEST / "test" / name / "candidate_pool.npz") as p:
        pool = dict(query=p["query"], offset=p["offset"], ids=p["ids"])
    return dict(cache=path, cache_sha256=expected, pos=pos, stamps=stamps, head=head, valid=valid,
                direction=direction, pool=pool)


def falseaccepts() -> None:
    """세 운영점의 오수락을 한 목록(가장 낮은 F1 τ 기준)에 모으고 층별로 센다."""
    frozen = main.read(OUT / "freeze.json")
    summary = main.read(OUT / "summary.json")
    assert summary["freeze_sha256"] == main.digest(OUT / "freeze.json")
    reference = main.read(main.NSD_TEST / "summary.json")
    contexts = {f"{a}/{b}": context(f"{a}/{b}") for a, b in TEST}
    records, strata, caches = [], {}, {}
    for method in main.METHODS:
        strata[method] = {}
        for seed in main.SEEDS:
            taus = {p: frozen["thresholds"][method][str(seed)][p]["tau"] for p in main.POINTS}
            assert taus["F1"] <= taus["P99"] <= taus["P100"]   # 높은 τ의 수락 집합은 F1 τ 수락 집합의 부분집합
            rows = main.test_rows(method, seed, reference)
            counters = {p: {} for p in main.POINTS}
            for sequence, r in rows.items():
                c = contexts[sequence]
                caches[c["cache"]] = c["cache_sha256"]
                q, j = r["query_idx"], r["j_star"]
                np.testing.assert_array_equal(c["direction"]["query_idx"], q)
                np.testing.assert_allclose(np.linalg.norm(c["pos"][q] - c["pos"][j], axis=1), r["dist"], atol=1e-6, rtol=0)
                pos_q, neg_q = r["qc"]["positive"], r["qc"]["negative"]
                finite = np.isfinite(r["score"])
                wrong = ~(np.isfinite(r["dist"]) & (r["dist"] <= 5))
                for point, tau in taus.items():
                    accepted = finite & (r["score"] >= tau)
                    bucket = counters[point]
                    # 방향별 양성 질의의 수락·정답·오수락(율 계산용 분모 포함)
                    for g, label in enumerate(DIRECTION_LABELS):
                        m = pos_q & (c["direction"]["group"] == g)
                        cell = bucket.setdefault("positive_direction", {}).setdefault(label, dict(queries=0, accepted=0, TP=0, FP=0))
                        cell["queries"] += int(m.sum()); cell["accepted"] += int((m & accepted).sum())
                        cell["TP"] += int((m & accepted & ~wrong).sum()); cell["FP"] += int((m & accepted & wrong).sum())
                    cell = bucket.setdefault("negative_queries", dict(queries=0, FP=0))
                    cell["queries"] += int(neg_q.sum()); cell["FP"] += int((neg_q & accepted).sum())
                fa = np.flatnonzero(finite & (r["score"] >= taus["F1"]) & wrong & (pos_q | neg_q))
                for i in fa:
                    qi, ji = int(q[i]), int(j[i])
                    dot = float(np.clip(c["head"][qi] @ c["head"][ji], -1, 1))
                    ok = bool(c["valid"][qi] and c["valid"][ji])
                    angle = float(np.degrees(np.arccos(dot))) if ok else None
                    heading = "invalid" if not ok else "same" if dot >= .5 else "reverse" if dot <= -.5 else "oblique"
                    positive = bool(pos_q[i])
                    k = int(np.searchsorted(c["pool"]["query"], qi))
                    assert c["pool"]["query"][k] == qi
                    ids = c["pool"]["ids"][c["pool"]["offset"][k]:c["pool"]["offset"][k + 1]]
                    pool_hit = bool((np.linalg.norm(c["pos"][ids] - c["pos"][qi], axis=1) <= 5).any())
                    row = dict(method=method, seed=seed, sequence=sequence, dataset=sequence.split("/")[0],
                               city=CITY_OF[sequence], query_idx=qi, returned_idx=ji, index_gap=qi - ji,
                               time_gap_s=round(float(c["stamps"][qi] - c["stamps"][ji]), 3),
                               query_kind="positive_wrong_place" if positive else "negative",
                               nearest_true_m=round(float(r["mpgd"][i]), 3), returned_m=round(float(r["dist"][i]), 3),
                               returned_bin=bin_label(float(r["dist"][i]), RETURNED_BINS),
                               nearest_bin=bin_label(float(r["mpgd"][i]), NEAREST_BINS), gap_bin=gap_label(qi - ji),
                               direction=(DIRECTION_LABELS[int(c["direction"]["group"][i])] if positive else "n/a"),
                               returned_heading=heading, heading_angle_deg=None if angle is None else round(angle, 1),
                               correct_place_in_nsd_candidates=pool_hit if positive else None,
                               score=float(r["score"][i]),
                               **{f"in_{p}": bool(r["score"][i] >= t) for p, t in taus.items()})
                    records.append(row)
                    for point in main.POINTS:
                        if not row[f"in_{point}"]:
                            continue
                        b = counters[point]
                        for key in ("query_kind", "returned_bin", "nearest_bin", "gap_bin", "returned_heading",
                                    "dataset", "city", "sequence"):
                            b.setdefault(key, {}).setdefault(row[key], 0)
                            b[key][row[key]] += 1
                        if positive:
                            for key in ("returned_bin", "returned_heading", "correct_place_in_nsd_candidates"):
                                d = b.setdefault("positive_" + key, {})
                                d[str(row[key])] = d.get(str(row[key]), 0) + 1
                        else:
                            d = b.setdefault("negative_returned_bin", {})
                            d[row["returned_bin"]] = d.get(row["returned_bin"], 0) + 1
            for point in main.POINTS:   # 집계 FP 수가 summary.json과 같은지 확인한다
                total = sum(counters[point]["query_kind"].values()) if "query_kind" in counters[point] else 0
                assert total == summary["methods"][method][point]["per_seed"][str(seed)]["pooled"]["FP"], (method, seed, point)
            strata[method][str(seed)] = counters
    fields = list(records[0])
    with (OUT / "false_accepts_all.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: ("" if v is None else f"{v:.6g}" if isinstance(v, float) and k == "score" else v)
                        for k, v in r.items()})
    with (OUT / "false_accepts_high_precision.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            if r["in_P99"] or r["in_P100"]:
                w.writerow({k: ("" if v is None else f"{v:.6g}" if isinstance(v, float) and k == "score" else v)
                            for k, v in r.items()})
    mean = {}
    for method in main.METHODS:
        mean[method] = {}
        for point in main.POINTS:
            def avg(path):
                vals = []
                for seed in main.SEEDS:
                    node = strata[method][str(seed)][point]
                    for p in path:
                        node = node.get(p, {}) if isinstance(node, dict) else 0
                    vals.append(node if isinstance(node, (int, float)) else 0)
                return float(np.mean(vals))
            keys = {}
            for seed in main.SEEDS:
                for key, sub in strata[method][str(seed)][point].items():
                    if key in ("positive_direction", "negative_queries"):
                        continue
                    keys.setdefault(key, set()).update(sub)
            mean[method][point] = {key: {v: avg((key, v)) for v in sorted(vals)} for key, vals in keys.items()}
            mean[method][point]["positive_direction"] = {
                g: {k: avg(("positive_direction", g, k)) for k in ("queries", "accepted", "TP", "FP")}
                for g in DIRECTION_LABELS}
            mean[method][point]["negative_queries"] = {k: avg(("negative_queries", k)) for k in ("queries", "FP")}
            # 질의 종류 × 키프레임 간격 × 반환 거리 교차표와 간격 구간별 시간 차(초) 중앙값
            chosen = [r for r in records if r["method"] == method and r[f"in_{point}"]]
            cross = {}
            for r in chosen:
                key = f"{r['query_kind']}|{r['gap_bin']}|{r['returned_bin']}"
                cross[key] = cross.get(key, 0) + 1 / len(main.SEEDS)
            mean[method][point]["cross_kind_gap_returned"] = dict(sorted(cross.items()))
            mean[method][point]["time_gap_s_median_by_gap_bin"] = {
                g: float(np.median([r["time_gap_s"] for r in chosen if r["gap_bin"] == g]))
                for g in sorted({r["gap_bin"] for r in chosen})}
    main.write(OUT / "false_accepts.json", dict(
        freeze_sha256=main.digest(OUT / "freeze.json"), summary_sha256=main.digest(OUT / "summary.json"),
        protocol_strata=main.read(OUT / "protocol.json")["false_accept_strata"],
        list_csv=dict(all=dict(path=str((OUT / "false_accepts_all.csv").relative_to(REPO)),
                               sha256=main.digest(OUT / "false_accepts_all.csv"), rows=len(records),
                               note="F1 τ의 오수락 전체(가장 낮은 τ). in_P99·in_P100 열이 높은 τ에서도 수락되는지 표시"),
                      high_precision=dict(path=str((OUT / "false_accepts_high_precision.csv").relative_to(REPO)),
                                          sha256=main.digest(OUT / "false_accepts_high_precision.csv"),
                                          rows=sum(r["in_P99"] or r["in_P100"] for r in records))),
        per_seed=strata, mean_over_seeds=mean, cache_sha256=caches,
        notes="correct_place_in_nsd_candidates는 NSD 공통 후보 합집합(원키·원보조 색인 상위100+100)에 5 m 안 과거 키프레임이 있었는지이다. "
              "BEVPlace++ 행에도 같은 NSD 후보 합집합 기준으로 적어 두지만 BEVPlace++ 자체 검색 후보는 아니다."))
    print(len(records), "false accepts listed")


def table_rows(summary: dict) -> list:
    rows = []
    for point in main.POINTS:
        for i, method in enumerate(main.METHODS):
            a = summary["methods"][method][point]["mean_sd"]
            rows.append(
                f"{point if i == 0 else ''} & {LABELS[method]} & "
                f"{100 * a['val_precision']['mean']:.2f} / {100 * a['val_recall']['mean']:.2f} & "
                f"${100 * a['city_macro_recall']['mean']:.2f}\\pm{100 * a['city_macro_recall']['sd']:.2f}$ & "
                f"{100 * a['recall']['mean']:.2f} & {100 * a['precision']['mean']:.2f} & "
                f"{a['TP']['mean']:,.0f} & {a['FP']['mean']:,.1f} & "
                f"{a['FP_positive_query']['mean']:,.1f} / {a['FP_negative_query']['mean']:,.1f} \\\\")
        rows.append(r"\midrule")
    return rows[:-1]


def tables() -> None:
    """원고 편집자용 LaTeX 표와 수치·문장 묶음을 만든다(원고 파일은 직접 고치지 않는다)."""
    summary = main.read(OUT / "summary.json")
    fa = main.read(OUT / "false_accepts.json")
    assert fa["summary_sha256"] == main.digest(OUT / "summary.json")
    head = [f"% scripts/_review_v23_fixed_threshold_20261107_report.py tables가 만든다. 숫자를 직접 고치지 않는다.",
            f"% 출처: results/pdv2_review_closure_20261107/threshold/summary.json (sha256 {main.digest(OUT / 'summary.json')})."]
    main_table = head + [
        r"\begin{table*}[t]",
        r"\caption{\textbf{Single deployment threshold.} One acceptance threshold per method and start is fixed on the "
        r"pooled validation queries (nine sequences) and applied unchanged to all 46 test sequences. P99 and P100 maximize "
        r"validation recall at $\geq$99\% and 100\% validation precision; F1 maximizes validation F1. These are validation "
        r"conditions and do not guarantee test precision. City recall follows Table~\ref{tab:results} (45 positive-bearing "
        r"sequences, six cities); pooled recall, precision, true accepts (TP) and false accepts (FA) count all 195,726 test "
        r"queries. FA are split into positive queries matched to a wrong place ($>$5\,m) and negative queries. Means over "
        r"three starts; city recall $\pm$ SD.}",
        r"\label{tab:fixed_threshold_current}",
        r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llccccccc@{}}\toprule",
        r"Point & Method & \shortstack{Val.\\prec./rec.} & \shortstack{City\\recall} & \shortstack{Pooled\\recall} & "
        r"\shortstack{Pooled\\prec.} & TP & FA & \shortstack{FA pos./\\neg.} \\",
        r"\midrule", *table_rows(summary), r"\bottomrule\end{tabular}", r"\end{table*}"]
    (OUT / "appdx_fixed_threshold_current.tex").write_text("\n".join(main_table) + "\n")

    cities = {"karlsruhe": "Karlsruhe", "seoul": "Seoul", "daejeon": "Daejeon",
              "ann_arbor": "Ann Arbor", "stockholm": "Stockholm", "hamburg": "Hamburg"}
    rows = []
    for point in main.POINTS:
        for i, method in enumerate(main.METHODS):
            a = summary["methods"][method][point]["mean_sd"]
            cells = [f"{100 * a['city_recall'][c]['mean']:.1f} / {a['city_FP'][c]['mean']:,.0f}" for c in cities]
            rows.append(f"{point if i == 0 else ''} & {LABELS[method]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    city_table = head + [
        r"\begin{table*}[t]",
        r"\caption{City recall (\%) / false accepts at the single validation-fixed thresholds of "
        r"Table~\ref{tab:fixed_threshold_current}. Means over three starts.}",
        r"\label{tab:fixed_threshold_city}",
        r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llcccccc@{}}\toprule",
        "Point & Method & " + " & ".join(cities.values()) + r" \\",
        r"\midrule", *rows[:-1], r"\bottomrule\end{tabular}", r"\end{table*}"]
    (OUT / "appdx_fixed_threshold_city.tex").write_text("\n".join(city_table) + "\n")

    order = ["(5,7.5]", "(7.5,10]", "(10,25]", "(25,100]", ">100"]
    rows = []
    for point in main.POINTS:
        for i, method in enumerate(main.METHODS):
            m = fa["mean_over_seeds"][method][point]
            pos = m.get("positive_returned_bin", {})
            neg = m.get("negative_returned_bin", {})
            cells = [f"{pos.get(b, 0):,.1f} / {neg.get(b, 0):,.1f}" for b in order]
            rows.append(f"{point if i == 0 else ''} & {LABELS[method]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    dist_table = head[:1] + [f"% 출처: results/pdv2_review_closure_20261107/threshold/false_accepts.json (sha256 {main.digest(OUT / 'false_accepts.json')})."] + [
        r"\begin{table*}[t]",
        r"\caption{False accepts by distance between the query and the returned place (m), split as positive / negative "
        r"queries, at the thresholds of Table~\ref{tab:fixed_threshold_current}. The first two bins lie just outside the 5\,m "
        r"correctness radius. Means over three starts.}",
        r"\label{tab:fixed_threshold_distance}",
        r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llccccc@{}}\toprule",
        r"Point & Method & (5, 7.5] & (7.5, 10] & (10, 25] & (25, 100] & $>$100 \\",
        r"\midrule", *rows[:-1], r"\bottomrule\end{tabular}", r"\end{table*}"]
    (OUT / "appdx_fixed_threshold_distance.tex").write_text("\n".join(dist_table) + "\n")
    integration(summary, fa)
    print("tables written")


def integration(summary: dict, fa: dict) -> None:
    """원고 편집자용 수치·짧은 영어 문장·표 경로와 출처 해시."""
    a = {m: {p: summary["methods"][m][p]["mean_sd"] for p in main.POINTS} for m in main.METHODS}
    f = {m: {p: fa["mean_over_seeds"][m][p] for p in main.POINTS} for m in main.METHODS}
    pct = lambda x: f"{100 * x:.2f}"
    n, b = a["nsd"], a["bevplace"]
    self_class = "negative|[30,60)|(25,100]"
    near = {m: f[m]["F1"]["positive_returned_bin"].get("(5,7.5]", 0) for m in main.METHODS}
    numbers = {m: {p: dict(city_recall_mean=a[m][p]["city_macro_recall"]["mean"], city_recall_sd=a[m][p]["city_macro_recall"]["sd"],
                           pooled_recall=a[m][p]["recall"]["mean"], pooled_precision=a[m][p]["precision"]["mean"],
                           pooled_precision_sd=a[m][p]["precision"]["sd"], TP=a[m][p]["TP"]["mean"], FA=a[m][p]["FP"]["mean"],
                           FA_positive=a[m][p]["FP_positive_query"]["mean"], FA_negative=a[m][p]["FP_negative_query"]["mean"],
                           val_precision=a[m][p]["val_precision"]["mean"], val_recall=a[m][p]["val_recall"]["mean"],
                           tau_per_seed=[summary["methods"][m][p]["per_seed"][str(s)]["tau"] for s in main.SEEDS],
                           city_recall={c: v["mean"] for c, v in a[m][p]["city_recall"].items()},
                           city_FA={c: v["mean"] for c, v in a[m][p]["city_FP"].items()})
                   for p in main.POINTS} for m in main.METHODS}
    sentences = [
        "We also fix one acceptance threshold per method and start on the pooled validation queries and apply it unchanged "
        "to all 46 test sequences (Table~\\ref{tab:fixed_threshold_current}).",
        f"At the validation P99 point, NSD keeps {pct(n['P99']['city_macro_recall']['mean'])}\\% city recall "
        f"({pct(n['P99']['recall']['mean'])}\\% pooled) at {pct(n['P99']['precision']['mean'])}\\% pooled test precision, "
        f"versus {pct(b['P99']['city_macro_recall']['mean'])}\\% ({pct(b['P99']['recall']['mean'])}\\%) at "
        f"{pct(b['P99']['precision']['mean'])}\\% for BEVPlace++.",
        f"NSD accepts more true matches ({n['P99']['TP']['mean']:,.0f} vs.\\ {b['P99']['TP']['mean']:,.0f} per start) and "
        f"therefore also more false accepts in absolute terms ({n['P99']['FP']['mean']:,.1f} vs.\\ {b['P99']['FP']['mean']:,.1f} "
        "of 195,726 queries); all of them fall in Daejeon sequences.",
        f"At the validation-F1 point, BEVPlace++ reaches higher city recall ({pct(b['F1']['city_macro_recall']['mean'])}\\% vs.\\ "
        f"{pct(n['F1']['city_macro_recall']['mean'])}\\%), mainly in Stockholm and Hamburg (NSD is higher in Seoul), while NSD has higher pooled precision "
        f"({pct(n['F1']['precision']['mean'])}\\% vs.\\ {pct(b['F1']['precision']['mean'])}\\%) and fewer false accepts "
        f"({n['F1']['FP']['mean']:,.0f} vs.\\ {b['F1']['FP']['mean']:,.0f}).",
        "The validation conditions do not transfer exactly: the P100 thresholds, which admit no validation false accept, give "
        f"{n['P100']['FP']['mean']:,.1f} (NSD) and {b['P100']['FP']['mean']:,.1f} (BEVPlace++) test false accepts per start.",
        f"At the F1 point, the largest false-accept group for both methods is negative queries matched to a keyframe 30--59 "
        f"keyframes (median {f['nsd']['F1']['time_gap_s_median_by_gap_bin']['[30,60)']:.1f} and "
        f"{f['bevplace']['F1']['time_gap_s_median_by_gap_bin']['[30,60)']:.1f}\\,s) earlier and 25--100\\,m away, "
        f"just outside the $j\\le q-30$ exclusion ({f['nsd']['F1']['cross_kind_gap_returned'].get(self_class, 0):,.1f} of "
        f"{n['F1']['FP']['mean']:,.1f} for NSD; {f['bevplace']['F1']['cross_kind_gap_returned'].get(self_class, 0):,.1f} of "
        f"{b['F1']['FP']['mean']:,.1f} for BEVPlace++); matches just beyond the 5\\,m radius (5--7.5\\,m) number "
        f"{near['nsd']:,.1f} and {near['bevplace']:,.1f}.",
    ]
    tables_out = {p.name: dict(path=str(p.relative_to(REPO)), sha256=main.digest(p))
                  for p in sorted(OUT.glob("appdx_fixed_threshold_*.tex"))}
    main.write(OUT / "manuscript_integration.json", dict(
        status="ready_for_editor", target="docs/paper/body_v2.3 (appendix; editor integrates, this worker does not edit the manuscript)",
        label="tab:fixed_threshold_current", replaces="v2.2 appendix tab:fixed_threshold (previous NSD model) — do not reuse its numbers",
        numbers=numbers, sentences_en=sentences, latex_tables=tables_out,
        claims_not_allowed=["P99/P100 guarantee test precision", "NSD has fewer false accepts at P99/P100 (it has more, with more TP)",
                            "perceptual cause of false accepts (only location/time/heading evidence was checked)",
                            "test-optimal single-threshold upper bound for other baselines (not computed)"],
        unchanged=["Table 2 values", "Figure 5 threshold sweep", "Figures 1-5"],
        provenance=dict(protocol_sha256=summary["protocol_sha256"], freeze_sha256=summary["freeze_sha256"],
                        summary_sha256=main.digest(OUT / "summary.json"), false_accepts_sha256=main.digest(OUT / "false_accepts.json"),
                        nsd="results/pdv2_diffattn_contrast_test_20261104 diff_s1..s3 (matching_only); validation "
                            "results/pdv2_diffattn_contrast_pipeline_20261104/val",
                        bevplace="results/pdv2_review_response_20261104/bev (= results/pdv2_bevplace_official{,_s2,_s3}/odom, bitwise)",
                        scripts={p: main.digest(REPO / p) for p in ("scripts/_review_v23_fixed_threshold_20261107.py",
                                                                    "scripts/_review_v23_fixed_threshold_20261107_report.py",
                                                                    "scripts/_review_v23_fixed_threshold_20261107_verify.py")})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("falseaccepts", "tables"))
    globals()[parser.parse_args().action]()
