#!/usr/bin/env python3
"""주 결과표 11개 방법의 진행 방향별(순방향 전용·역방향 전용) 최종 성능 (2026-11-17).

규약은 부록 tab:direction(NSD·BEVPlace++)과 같다: _report_model_common_20261116의 seq_metric(방향 양성만 남기고
공통 음성 유지, 다른 양성은 모호로 제외, 시퀀스별 임계 스윕)·macro(도시 안 평균 → 양성 있는 도시 동일 가중)·
direction_masks(results/pdv2_review_direction_yaw_20261105/directions). 학습 방법은 시드마다 매크로 → 시드 평균·표본 SD.
비교 방법 원장은 표2 요약(DEEPRING_SUMMARY, _paper_tables_v24.py)의 source 폴더이며, _pdv2_main_baselines_official.aggregate와
같이 공통 질의(30 <= q < N-15)로 잘라 쓴다. NSD는 보고모델 원장(시드 4·5·6).
호스트 CPU: python3 scripts/_direction_all_methods_20261117.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _report_model_common_20261116 as C  # noqa: E402

SUMMARY = ROOT / "results/pdv2_deepring_scaleinit_integration_20261110/baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json"
OUT = ROOT / "results/pdv2_report_model_20261116/analysis/direction_all"
ORDER = ["scpp", "fresco", "ring", "ringpp", "disco", "deepring", "ringsharp", "helios", "seqot", "bevplace", "nsd"]
NAMES = {"scpp": "SC++", "fresco": "FreSCo", "ring": "RING", "ringpp": "RING++", "disco": "DiSCO", "deepring": "DeepRING",
         "ringsharp": r"RING\#-L", "helios": "HeLiOS", "seqot": "SeqOT", "bevplace": "BEVPlace++", "nsd": r"\textbf{NSD}"}
PREFIX = {"scpp": "dump_scpp", "fresco": "dump_fresco", "ring": "dump_ring", "ringpp": "dump_ringpp_i2", "disco": "dump_disco",
          "deepring": "dump_deepring", "ringsharp": "dump_ringsharp", "helios": "dump_helios", "seqot": "dump_seqotl2",
          "bevplace": "dump_bevplace"}
GROUPS = ("all", "forward_only", "reverse_only")
MAIN_TABLE = {  # table2_results.tex (AP, max F1, R@100P, R@1)
    "scpp": (67.58, 71.48, 39.02, 77.37), "fresco": (63.15, 69.24, 19.20, 72.72), "ring": (63.15, 65.86, 23.66, 77.51),
    "ringpp": (62.52, 64.61, 19.59, 82.08), "disco": (55.11, 60.38, 23.58, 77.83), "deepring": (78.60, 80.68, 32.73, 88.50),
    "ringsharp": (73.42, 77.26, 14.89, 85.11), "helios": (35.50, 43.95, 5.80, 55.77), "seqot": (38.20, 44.00, 16.68, 54.40),
    "bevplace": (89.85, 89.00, 57.94, 93.90), "nsd": (89.88, 90.39, 72.17, 91.74)}
APPDX = {"nsd": {"forward_only": (90.96, 91.28, 81.91, 94.28), "reverse_only": (84.43, 85.30, 66.70, 87.77)},
         "bevplace": {"forward_only": (93.32, 92.46, 80.74, 97.22), "reverse_only": (80.50, 80.10, 47.30, 88.00)}}
JOURNAL_1110 = {"helios": {"forward_only": 58.32, "reverse_only": 14.14}, "seqot": {"forward_only": 64.16, "reverse_only": 15.34}}


def runs(method: str, summary: dict) -> list:
    """[(seed, loader(seq) -> dict)]."""
    if method == "nsd":
        return [(str(s), lambda seq, t=f"new:{s}": C.load(t, seq)) for s in C.NEW_SEEDS]
    m = summary["methods"][method]
    out = []
    for seed, entry in m["per_seed"].items():
        def loader(seq, folder=ROOT / entry["source"], score=m["score"], prefix=PREFIX[method]):
            with np.load(folder / f"{prefix}_{seq.replace('/', '_')}.npz") as z:
                return dict(score=z[score].astype(np.float64), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"],
                            q=z["query_idx"])
        out.append((seed, loader))
    return out


def main() -> None:
    t0 = time.time()
    summary = json.loads(SUMMARY.read_text())
    dirs = {}
    for s in C.SEQS:
        with np.load(C.DIRS / (s.replace("/", "_") + ".npz")) as z:
            dirs[s] = (z["query_idx"], z["group"])
    masks = {g: C.direction_masks(g) for g in GROUPS if g != "all"}
    npos = {g: int(sum(m.sum() for m in masks[g].values())) for g in masks}
    result, checks = {}, {}
    for method in ORDER:
        per_seed = {}
        for seed, loader in runs(method, summary):
            rows = {g: {} for g in GROUPS}
            for s in C.SEQS:
                d = loader(s)
                qref = dirs[s][0]
                ix = np.searchsorted(d["q"], qref)
                assert np.array_equal(d["q"][ix], qref), (method, seed, s)
                d = {k: v[ix] for k, v in d.items() if k != "j"}
                for g in GROUPS:
                    rows[g][s] = C.seq_metric(d, None if g == "all" else masks[g][s])
            per_seed[seed] = {g: C.macro(rows[g]) for g in GROUPS}
        ms = {g: {k: C.mean_sd([per_seed[sd][g]["macro"][k] for sd in per_seed]) for k in C.METRICS} for g in GROUPS}
        result[method] = dict(seeds=list(per_seed), mean_sd=ms,
                              per_seed={sd: {g: v[g]["macro"] for g in GROUPS} for sd, v in per_seed.items()})
        got = [round(100 * ms["all"][k]["mean"], 2) for k in C.METRICS]
        checks[f"{method}/all_vs_table2"] = dict(got=got, table=list(MAIN_TABLE[method]),
                                                 match=all(abs(a - b) < 0.0051 for a, b in zip(got, MAIN_TABLE[method])))
        if method in APPDX:
            for g in ("forward_only", "reverse_only"):
                got = [round(100 * ms[g][k]["mean"], 2) for k in C.METRICS]
                checks[f"{method}/{g}_vs_appdx"] = dict(got=got, table=list(APPDX[method][g]),
                                                        match=all(abs(a - b) < 0.0051 for a, b in zip(got, APPDX[method][g])))
        if method in JOURNAL_1110:
            for g in ("forward_only", "reverse_only"):
                got = round(100 * ms[g]["AP"]["mean"], 2)
                checks[f"{method}/{g}_AP_vs_journal_1110"] = dict(got=got, text=JOURNAL_1110[method][g],
                                                                  match=abs(got - JOURNAL_1110[method][g]) < 0.0051)
        print(method, {g: [round(100 * ms[g][k]["mean"], 2) for k in C.METRICS] for g in GROUPS}, flush=True)
    # NSD 저장본(direction_summary.json)과 1e-9 수준 비교
    ref = json.loads((C.OUT / "main/direction_summary.json").read_text())["methods"]["diff"]["mean_std"]
    checks["nsd/direction_summary_max_abs_err"] = max(abs(result["nsd"]["mean_sd"][g][k]["mean"] - ref[g][k]["mean"])
                                                      for g in GROUPS for k in C.METRICS)
    elapsed = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(scope="주 결과표 11방법의 진행 방향별 최종 성능(시험46, 사후 층화)",
               procedure="_report_model_common_20261116.seq_metric/macro/direction_masks; 시드마다 6도시 매크로 → 시드 평균, SD ddof=1",
               baseline_summary=str(SUMMARY.relative_to(ROOT)), nsd="new:4,5,6 (pdv2_scratch_search_20261116 ledgers)",
               positives=npos, checks=checks, all_checks_pass=all(v["match"] for k, v in checks.items() if isinstance(v, dict)),
               elapsed_s=elapsed, methods=result)
    C.write(OUT / "direction_all_methods.json", out)
    with open(OUT / "direction_all_methods.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "group", "n_runs"] + [f"{k}_{s}" for k in C.METRICS for s in ("mean", "sd")])
        for m in ORDER:
            for g in GROUPS:
                v = result[m]["mean_sd"][g]
                w.writerow([m, g, len(result[m]["seeds"])] + [x for k in C.METRICS for x in
                           (f"{100 * v[k]['mean']:.4f}", "" if v[k]["sd"] is None else f"{100 * v[k]['sd']:.4f}")])
    write_tex(result, npos)
    print(json.dumps({"all_checks_pass": out["all_checks_pass"], "elapsed_s": round(elapsed, 1),
                      "nsd_err": checks["nsd/direction_summary_max_abs_err"]}))


def write_tex(result: dict, npos: dict) -> None:
    def table(keys: list, name: str, wide: bool) -> None:
        cols = [(g, k) for g in ("forward_only", "reverse_only") for k in keys]
        vals = {(g, k): [round(100 * result[m]["mean_sd"][g][k]["mean"], 2) for m in ORDER] for g, k in cols}
        rank = {}
        for c, v in vals.items():
            order = sorted(set(v), reverse=True)
            rank[c] = {x: i for i, x in enumerate(order[:2])}
        head = {"AP": "AP", "R@100P": "R@100P", "R@1": r"\shortstack{Revisit\\R@1}"}
        n = len(keys)
        lines = [r"% scripts/_direction_all_methods_20261117.py 초안. 원고에 넣지 않았다.",
                 r"\begin{table}[t]",
                 r"\caption{Recognition by traversal direction (\%). Forward-only and reverse-only groups contain "
                 f"{npos['forward_only']:,} and {npos['reverse_only']:,} positive queries; each keeps the common negatives "
                 r"and excludes other positives. Aggregation follows Table~\ref{tab:results}.}\label{tab:direction_all}",
                 r"\centering\footnotesize\setlength{\tabcolsep}{3pt}", r"\sbox0{%",
                 r"\begin{tabular}{@{}l" + "c" * (2 * n) + "@{}}\\toprule",
                 f"& \\multicolumn{{{n}}}{{c}}{{Forward-only}} & \\multicolumn{{{n}}}{{c}}{{Reverse-only}} \\\\ "
                 f"\\cmidrule(lr){{2-{n + 1}}}\\cmidrule(l){{{n + 2}-{2 * n + 1}}}",
                 "Method & " + " & ".join(head[k] for _, k in cols) + r" \\", r"\midrule"]
        for i, m in enumerate(ORDER):
            if m in ("disco", "nsd"):
                lines.append(r"\midrule")
            cells = []
            for c in cols:
                x = vals[c][i]
                s = f"{x:.2f}"
                r = rank[c].get(x)
                cells.append(r"\textbf{" + s + "}" if r == 0 else r"\underline{" + s + "}" if r == 1 else s)
            lines.append(NAMES[m] + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule\end{tabular}}", r"\ifdim\wd0>\columnwidth\resizebox{\columnwidth}{!}{\usebox0}\else\usebox0\fi",
                  r"\end{table}"]
        (OUT / name).write_text("\n".join(lines) + "\n")
    table(["AP", "R@100P"], "direction_all_4col.tex", False)
    table(["AP", "R@100P", "R@1"], "direction_all_6col.tex", False)


if __name__ == "__main__":
    main()
