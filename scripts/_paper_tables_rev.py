#!/usr/bin/env python3
"""Generate the revision LaTeX table fragments from committed result JSONs.

Emits into docs/paper/body_v2.1/tables_rev/ (the working copy \\input's them):
  table1_strict.tex      — Main Table 1: 37-seq place-disjoint offline retrieval
                            (Strict-v2 rev3; RING# = clean pd retrain)
  table2_xsession.tex    — Main Table 2: RING-style cross-session retrieval
  table3_efficiency.tex  — Main Table 3: matched-condition efficiency (N=1000)
  appdx_xs_pairs.tex     — appendix: per-pair R@1@5m / R@1@20m, all methods
  appdx_xs_thresholds.tex— appendix: NSD/NSD-H macro threshold sweep + R@5/R@10
  appdx_coverage.tex     — appendix: per-pair coverage (structural ceilings)
  appdx_excluded.tex     — appendix: the four excluded+disclosed sequences
  appdx_sensitivity.tex  — appendix: full-42 vs strict-37 roster sensitivity

Numbers come exclusively from:
  results/pd2026_baselines/summary_strict_v2_revision3.json
  results/pd2026_baselines/strict_seed_stats.json
  results/pd2026_baselines/roster_sensitivity.json
  results/pd2026_xsession/xs_summary.json (+ raw xs_*.json for appendix depth)
  results/matched_bench_n1000.json
  results/pd2026/test_summary_intra.json (excluded-sequence rows)
Re-run after any upstream change; never hand-edit the fragments.
"""
from __future__ import annotations
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/paper/body_v2.1/tables_rev"
XS = REPO / "results/pd2026_xsession"

REV3 = json.loads((REPO / "results/pd2026_baselines/summary_strict_v2_revision4.json").read_text())
SEED = json.loads((REPO / "results/pd2026_baselines/strict_seed_stats.json").read_text())
SENS = json.loads((REPO / "results/pd2026_baselines/roster_sensitivity.json").read_text())
XSUM = json.loads((XS / "xs_summary.json").read_text())
BENCH = json.loads((REPO / "results/matched_bench_n1000.json").read_text())["methods"]
INTRA = json.loads((REPO / "results/pd2026/test_summary_intra.json").read_text())

F = lambda v: f"{v:.3f}"
PM = lambda m, s: f"{m:.3f}\\,{{\\scriptsize$\\pm${s:.3f}}}"


def w(name: str, s: str):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(s)
    print(f"wrote tables_rev/{name}")


def bold_best(rows, col_idx, fmt_rows):
    """Bold max, underline runner-up in one numeric column (mutates fmt_rows)."""
    vals = [(r[col_idx], i) for i, r in enumerate(rows) if r[col_idx] is not None]
    if len(vals) < 2:
        return
    vals.sort(reverse=True)
    b, u = vals[0][1], vals[1][1]
    fmt_rows[b][col_idx] = "\\textbf{" + fmt_rows[b][col_idx] + "}"
    fmt_rows[u][col_idx] = "\\underline{" + fmt_rows[u][col_idx] + "}"


# ---------------------------------------------------------------- Table 1
def table1():
    methods = [("ring", "RING$^{\\ddagger}$"), ("ringpp", "RING++$^{\\ddagger}$"),
               ("ringsharp", "RING\\#-L$^{\\mathsection}$"),
               ("scpp", "SC++"), ("seqot", "SeqOT"),
               ("helios", "HeLiOS$^{\\dagger}$"), ("imlpr", "ImLPR$^{\\dagger}$"),
               ("imlpr_pd", "ImLPR (pd retrain)$^{\\ast}$"),
               ("bevplace", "BEVPlace++$^{\\P}$"),
               ("nsd_pbev128_main", "\\textbf{NSD}"),
               ("nsd_hspec12", "\\textbf{NSD-H}")]
    groups = ["kitti_pd (05/08)", "helipr_town (01-03)", "helipr_daejeon",
              "mulran_daejeon", "kitti360 (0000/0006)", "mcd_zeroshot"]
    raw, fmt = [], []
    for key, _ in methods:
        e = REV3[key]
        vals = [e[g] for g in groups] + [e["strict_macro"]]
        raw.append(vals)
        row = [F(v) for v in vals]
        if key in ("helios", "imlpr"):
            ci = groups.index("helipr_daejeon")
            row[ci] = row[ci] + "$^{\\dagger}$"
            row[-1] = (F(e["strict_macro"]) + "$^{\\dagger}$\\,("
                       + F(e["strict_macro_clean5"]) + ")")
        if key == "nsd_pbev128_main":
            row[-1] = PM(e["strict_macro"], SEED["nsd"]["strict_macro"]["std"])
        if key == "nsd_hspec12":
            row[-1] = PM(e["strict_macro"], SEED["nsdh"]["strict_macro"]["std"])
        fmt.append(row)
    for c in range(len(groups) + 1):
        bold_best(raw, c, fmt)
    lines = [
        r"\begin{table*}[t]",
        r"\caption{\bedit{\textbf{Unified place-disjoint offline symmetric place retrieval under our common protocol} (37 test sequences, six groups; sequence-macro Recall@1 at 5\,m). All methods are re-evaluated on the identical frozen test roster with our common evaluation implementation: 5\,m positive threshold, $\pm$30-frame temporal exclusion, symmetric (offline) candidate set in which future keyframes are admissible; this is offline retrieval, not online loop closure. The summary column is the six-group-balanced macro. Test sequences are place-disjoint from the training and validation splits; four additional sequences were excluded after a post-hoc GPS data-quality audit and are disclosed in the supplementary material. RING\#-L is retrained on the place-disjoint training split (single seed; checkpoint selected on validation by a pre-declared rule). NSD and NSD-H are 3-seed means, $\pm$ one standard deviation on the macro. Best and second-best per column are bold and underlined.}}",
        r"\label{tab:results}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3.6pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"& \shortstack{KITTI\\05/08} & \shortstack{HeLiPR\\Town01--03} & \shortstack{HeLiPR\\Daejeon 04--06} & \shortstack{MulRan\\Daejeon 01--03} & \shortstack{KITTI-360\\0000/0006} & \shortstack{MCD\\KTH/TUHH} & \shortstack{\textit{Macro}\\\textit{(6 groups)}} \\",
        r"\midrule",
    ]
    for (key, label), row in zip(methods, fmt):
        pre = "\\rowcolor{gray!8}\n" if key.startswith("nsd") else ""
        lines.append(pre + label + " & " + " & ".join(row) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}", "",
        r"\bedit{$^{\ddagger}$Exhaustive correlation over all stored keyframes. $^{\mathsection}$Clean place-disjoint retrain (no train/test overlap). $^{\P}$Official KITTI-pretrained checkpoint followed by place-disjoint fine-tuning; NSD/NSD-H use no external pretraining. $^{\dagger}$HeLiOS/ImLPR: official code and weights, zero-shot; their authors trained on HeLiPR DCC/KAIST/Riverside 04--06 = this roster's HeLiPR-Daejeon group, so that cell is train-contaminated (optimistic) and the macro is shown with the clean five-group macro (excluding it) in parentheses. $^{\ast}$ImLPR (pd retrain): clean from-scratch retrain on the place-disjoint training split with the official trainer and recipe (checkpoint by the pre-declared validation rule); no train/test overlap.}",
        r"\end{table*}",
    ]
    w("table1_strict.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- Table 2
def table2():
    methods = [("scpp", "SC++"), ("bevplace", "BEVPlace++"), ("seqot", "SeqOT"),
               ("ring", "RING"), ("ringpp", "RING++"),
               ("helios", "HeLiOS (zero-shot)"), ("imlpr", "ImLPR (zero-shot)"),
               ("imlpr_pd", "ImLPR (pd retrain)"),
               ("nsd", "\\textbf{NSD}"), ("nsdh", "\\textbf{NSD-H}")]
    aux = [("nsdh_pdnn", "NSD-H (no NCLT training)")]
    sites = ["nclt", "dcc", "kaist", "riverside"]
    if ("ringsharp", None) and "ringsharp" in {m for (m, _) in methods}:
        pass

    def cell(entry, with_std=False):
        if entry is None:
            return None, "--"
        mean, std, _ = entry
        return mean, (PM(mean, std) if (with_std and std is not None) else F(mean))

    raw, fmt, labels, keys = [], [], [], []
    for key, label in methods + aux:
        body = XSUM["body"]
        r5 = body["R@1@5m"].get(key)
        if r5 is None:
            continue
        vals, cells = [], []
        for s in sites:
            v, c = cell(r5["sites"][s])
            vals.append(v); cells.append(c)
        for metric in ("R@1@5m", "R@1@20m", "R1pct@20m"):
            v, c = cell(body[metric][key]["macro"], with_std=True)
            vals.append(v); cells.append(c)
        raw.append(vals); fmt.append(cells); labels.append(label); keys.append(key)
    n_body = sum(1 for k in keys if k not in {a for a, _ in aux})
    for c in range(7):
        bold_best(raw[:n_body], c, fmt[:n_body])
    lines = [
        r"\begin{table*}[t]",
        r"\caption{\bedit{\textbf{RING-style cross-session retrieval under a common keyframe roster.} Map session subsampled at 20\,m spacing, query session at 5\,m; the query roster (queries with a map pose within 20\,m) is built once and reused at every threshold, so all columns share one denominator. Site columns report Recall@1 at 5\,m averaged over the site's two pairs (NCLT: 2012-02-04$\to$2012-03-17; MulRan DCC/KAIST/Riverside: 01$\to$02 and 01$\to$03); the macro is site-balanced. R@1\% uses $k=\max(\mathrm{round}(n_{\mathrm{db}}/100),1)$. Because the map is 20\,m-spaced, the 5\,m metric has a structural coverage ceiling ($\approx$0.49--0.60 of the roster; Riverside 01$\to$03 is an opposite-carriageway revisit with ceiling 0.013) --- hence the auxiliary 20\,m columns; per-pair results, all thresholds, and coverage are in the supplementary material. Map/query sets are resampled from our keyframe caches, so this is not the official RING roster. All baselines are re-evaluated on the identical map/query sets using our common evaluation implementation. NSD rows are 3-seed means ($\pm$ std on macros).}}",
        r"\label{tab:xsession}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{tabular}{lcccc ccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{R@1@5\,m per site} & \multicolumn{3}{c}{\textit{Site-balanced macro}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
        r"Method & NCLT & DCC & KAIST & Riverside & R@1@5\,m & R@1@20\,m & R@1\%@20\,m \\",
        r"\midrule",
    ]
    for i, label in enumerate(labels):
        if keys[i] == "nsdh_pdnn":
            lines.append(r"\midrule")
        pre = "\\rowcolor{gray!8}\n" if keys[i] in ("nsd", "nsdh") else ""
        lines.append(pre + label + " & " + " & ".join(fmt[i]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    w("table2_xsession.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- Table 3
def table3():
    rows = [("scpp", "SC++", "20 (+SC)"), ("bevplace", "BEVPlace++", "8{,}192"),
            ("seqot", "SeqOT", "256"), ("ring", "RING", "--"),
            ("ringpp", "RING++", "--"), ("ringsharp", "RING\\#-L", "160"),
            ("nsd", "\\textbf{NSD}", "416\\,/\\,128"),
            ("nsdh", "\\textbf{NSD-H}", "416\\,/\\,240")]
    lines = [
        r"\begin{table}[t]",
        r"\caption{\bedit{\textbf{Matched-condition efficiency} (KITTI~00, $N{=}1{,}000$ database, batch~1, point clouds preloaded, CUDA-synchronized; RTX~5080). Storage is the \emph{total} per-frame state each method must hold for retrieval to operate (e.g.\ SC++: ring key + scan-context matrix; RING\#-L: coarse key + rotation spectrum + translation feature), measured from the arrays actually built. Build amortizes NSD's batch graph stages per keyframe.}}",
        r"\label{tab:efficiency}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & \shortstack{Scored\\indexes} & \shortstack{Storage\\(KB/frame)} & \shortstack{Build\\(ms/frame)} & \shortstack{Query\\(ms)} \\",
        r"\midrule",
    ]
    for key, label, dim in rows:
        e = BENCH[key]
        kb = e["storage_bytes_per_frame"] / 1024.0
        pre = "\\rowcolor{gray!8}\n" if key.startswith("nsd") else ""
        lines.append(pre + f"{label} & {dim} & {kb:,.1f} & "
                     f"{e['build_ms']['mean']:,.1f} & "
                     f"{e['query_ms']['mean']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("table3_efficiency.tex", "\n".join(lines) + "\n")


# ------------------------------------------------------------- appendix
def load_xs_raw():
    per = defaultdict(dict)
    for f in sorted(XS.glob("xs_*.json")):
        d = json.loads(f.read_text())
        if d.get("schema") != "xs-eval-v1":
            continue
        per[(d["method"], d["protocol"])][d.get("seed", 0)] = d
    return per


def xs_mean(per, method, protocol, metric):
    e = per.get((method, protocol))
    if not e:
        return None
    return statistics.mean(d["metrics"][metric] for d in e.values())


def appdx_xs_pairs():
    per = load_xs_raw()
    protocols = list(XSUM["coverage"])
    methods = [("scpp", "SC++"), ("bevplace", "BEVPlace++"), ("seqot", "SeqOT"),
               ("ring", "RING"), ("ringpp", "RING++"),
               ("ringsharp", "RING\\#-L"),
               ("nsd", "NSD"), ("nsdh", "NSD-H"),
               ("nsdh_pdnn", "NSD-H (pdnn)"), ("nsd_keyonly", "NSD key-only")]
    plabel = {"nclt_0204_0317": "NCLT",
              "dcc_01_02": "DCC 1$\\to$2", "dcc_01_03": "DCC 1$\\to$3",
              "kaist_01_02": "KAIST 1$\\to$2", "kaist_01_03": "KAIST 1$\\to$3",
              "riverside_01_02": "Riv.\\ 1$\\to$2",
              "riverside_01_03": "Riv.\\ 1$\\to$3"}
    for metric, fname, title in [
            ("R@1@5m", "appdx_xs_pairs.tex", "Per-pair Recall@1 at 5\\,m"),
            ("R@1@20m", "appdx_xs_pairs20.tex", "Per-pair Recall@1 at 20\\,m")]:
        lines = [
            r"\begin{table*}[!htbp]",
            f"\\caption{{\\bedit{{\\textbf{{{title}}} (cross-session; fixed 20\\,m roster; NCLT = 2012-02-04$\\to$2012-03-17).}}}}",
            f"\\label{{tab:xs_pairs_{metric.replace('@','_').replace('m','')}}}",
            r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{l" + "c" * len(protocols) + "}",
            r"\toprule",
            "Method & " + " & ".join(plabel[p] for p in protocols) + r" \\",
            r"\midrule",
        ]
        for key, label in methods:
            vals = [xs_mean(per, key, p, metric) for p in protocols]
            if all(v is None for v in vals):
                continue
            lines.append(label + " & "
                         + " & ".join("--" if v is None else F(v) for v in vals)
                         + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        w(fname, "\n".join(lines) + "\n")


def appdx_thresholds():
    per = load_xs_raw()
    protocols = list(XSUM["coverage"])
    sites = XSUM["_meta"] and {"nclt": ["nclt_0204_0317"],
                               "dcc": ["dcc_01_02", "dcc_01_03"],
                               "kaist": ["kaist_01_02", "kaist_01_03"],
                               "riverside": ["riverside_01_02", "riverside_01_03"]}

    def macro(method, metric):
        site_means = []
        for ps in sites.values():
            vals = [xs_mean(per, method, p, metric) for p in ps]
            if any(v is None for v in vals):
                return None
            site_means.append(statistics.mean(vals))
        return statistics.mean(site_means)

    metrics = [("R@1@5m", "R@1@5"), ("R@1@10m", "R@1@10"), ("R@1@15m", "R@1@15"),
               ("R@1@20m", "R@1@20"), ("R@5@5m", "R@5@5"), ("R@10@5m", "R@10@5"),
               ("R1pct@5m", "R@1\\%@5"), ("R1pct@10m", "R@1\\%@10"),
               ("R1pct@15m", "R@1\\%@15"), ("R1pct@20m", "R@1\\%@20")]
    methods = [("scpp", "SC++"), ("bevplace", "BEVPlace++"), ("seqot", "SeqOT"),
               ("ring", "RING"), ("ringpp", "RING++"), ("ringsharp", "RING\\#-L"),
               ("nsd", "NSD"), ("nsdh", "NSD-H"), ("nsd_keyonly", "NSD key-only")]
    lines = [
        r"\begin{table*}[!htbp]",
        r"\caption{\bedit{\textbf{Cross-session threshold sweep} (site-balanced macro over the fixed 20\,m roster; one denominator per pair across all thresholds).}}",
        r"\label{tab:xs_thresholds}",
        r"\centering\footnotesize\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Method & " + " & ".join(m for _, m in metrics) + r" \\",
        r"\midrule",
    ]
    for key, label in methods:
        vals = [macro(key, m) for m, _ in metrics]
        if all(v is None for v in vals):
            continue
        lines.append(label + " & "
                     + " & ".join("--" if v is None else F(v) for v in vals)
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    w("appdx_xs_thresholds.tex", "\n".join(lines) + "\n")


def appdx_coverage():
    cov = XSUM["coverage"]
    lines = [
        r"\begin{table}[!htbp]",
        r"\caption{\bedit{\textbf{Cross-session coverage (structural ceilings).} Fraction of roster queries with any map pose within each radius; identical for every method. The 20\,m map spacing bounds the 5\,m column near 0.5 by construction; Riverside 01$\to$03 revisits the opposite carriageway.}}",
        r"\label{tab:xs_coverage}",
        r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Pair & $n_{\mathrm{db}}$ & roster & cov@5 & cov@10 & cov@15 & cov@20 \\",
        r"\midrule",
    ]
    short = {"nclt_0204_0317": "NCLT",
             "dcc_01_02": "DCC 1$\\to$2", "dcc_01_03": "DCC 1$\\to$3",
             "kaist_01_02": "KAIST 1$\\to$2", "kaist_01_03": "KAIST 1$\\to$3",
             "riverside_01_02": "Riv.\\ 1$\\to$2",
             "riverside_01_03": "Riv.\\ 1$\\to$3"}
    for p, e in cov.items():
        lines.append(short.get(p, p.replace("_", "\\_"))
                     + f" & {e['n_db']} & {e['n_roster']} & "
                     + " & ".join(F(e[f"cov@{t}m"]) for t in (5, 10, 15, 20)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("appdx_coverage.tex", "\n".join(lines) + "\n")


def appdx_excluded():
    excl = [("kitti", "00", "13.6\\% route overlap w/ train 07"),
            ("kitti360", "0004", "same route as val.\\ KITTI 09"),
            ("helipr", "Bridge04", "training bridge, reverse dir."),
            ("nclt", "2012-02-04", "training campus (sess.-disjoint)"),
            ("nclt", "2012-03-17", "training campus (sess.-disjoint)")]
    baselines = [("scpp", "SC++"), ("bevplace", "BEVPlace++"), ("seqot", "SeqOT"),
                 ("ringsharp_pd", "RING\\#-L"), ("ring", "RING"), ("ringpp", "RING++")]

    def r1(method, sensor, seq):
        p = REPO / f"results/pd2026_baselines/{method}_{sensor}_{seq}.json"
        d = json.loads(p.read_text())
        if method == "bevplace":
            return float(d["recall_at_1"])
        if f"{sensor}_{seq}" in d:
            return float(d[f"{sensor}_{seq}"]["recall@1"])
        return float(d["recall@1"])

    lines = [
        r"\begin{table*}[!htbp]",
        r"\caption{\bedit{\textbf{Excluded-and-disclosed sequences} (Recall@1 at 5\,m under the common protocol). These sequences fail the strict place-disjointness audit and are excluded from the Table~\ref{tab:results} macro; the NCLT pair is reported separately as a temporal-generalization case (sessions unseen; campus seen during training in the primary configuration).}}",
        r"\label{tab:excluded}",
        r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"Sequence & Exclusion reason & SC++ & BEVP++ & SeqOT & RING\# & RING & RING++ & NSD & NSD-H \\",
        r"\midrule",
    ]
    for sensor, seq, reason in excl:
        cells = [F(r1(m, sensor, seq)) for m, _ in baselines]
        for key, ik in [("nsd", "pd/pbev128_main"), ("nsdh", "pd/hspec12")]:
            cells.append(F(INTRA[ik]["per_sequence"][f"{sensor}/{seq}"]["r1_mean"]))
        lines.append(f"{sensor}/{seq.replace('_', '-')} & {reason} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    w("appdx_excluded.tex", "\n".join(lines) + "\n")


def appdx_sensitivity():
    lines = [
        r"\begin{table}[!htbp]",
        r"\caption{\bedit{\textbf{Roster sensitivity} of the proposed method: full original 42-sequence roster (seven-group macro) vs.\ the strict 37-sequence roster of Table~\ref{tab:results} (six-group macro). The exclusions do not change the conclusions.}}",
        r"\label{tab:roster_sensitivity}",
        r"\centering\footnotesize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"& Full roster (42) & Strict roster (37) & $\Delta$ \\",
        r"\midrule",
    ]
    for key, label in [("nsd", "NSD"), ("nsdh", "NSD-H")]:
        e = SENS[key]
        lines.append(f"{label} & {F(e['full42_macro'])} & {F(e['strict37_macro'])} "
                     f"& {e['delta']:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("appdx_sensitivity.tex", "\n".join(lines) + "\n")


def appdx_causal_pr():
    pr = json.loads((REPO / "results/pd2026_pr/pr_summary.json").read_text())["methods"]
    rows = [("bevplace", "BEVPlace++"), ("ring", "RING"), ("scpp", "SC++"),
            ("ringpp", "RING++"),
            ("nsd_causal", "\\textbf{NSD} (causal-retrained)"),
            ("nsdh_causal", "\\textbf{NSD-H} (causal-retrained)"),
            ("nsd_frozen", "NSD (Frozen-GNN adaptation)"),
            ("nsdh_frozen", "NSD-H (Frozen-GNN adaptation)"),
            ("ringsharp", "RING\\#-L (pd retrain)"),
            ("helios", "HeLiOS (zero-shot)$^{\\dagger}$"),
            ("imlpr", "ImLPR (zero-shot)$^{\\dagger}$"),
            ("imlpr_pd", "ImLPR (pd retrain)"),
            ("seqot", "SeqOT (trailing window)")]
    metrics = ("AP", "max_F1", "R@100P", "R@99P", "R@95P")
    heads = ("AP", "max F1", "R@100P", "R@99P", "R@95P")
    lines = [
        r"\begin{table*}[!htbp]",
        r"\caption{\bedit{\textbf{Pre-registered causal loop-closure decision evaluation} (Strict-v2 37 sequences, six-group-balanced macro). Protocol frozen before any test contact (pre-registration document in the code release): fully causal retrieval (past-only candidates \emph{and} past-only trajectory context, verified by a bit-level prefix-invariance test), every frame $t\ge30$ is a query including frames with no ground-truth revisit, query classes positive ($\le$5\,m) / ambiguous ((5,25]\,m, excluded) / negative ($>$25\,m), an accepted match is correct only within 5\,m, and the recall denominator is the fixed positive-query count. Baselines use their native decision scores; NSD uses the validation-selected raw-key-cosine confidence, and its causal-retrained rows retrain the refiner on causal graphs (3 seeds, mean$\pm$std). Our R@100P is by construction identical to the official RING accounting (the accounting difference affects only the $P{<}1$ segment; verified against the official implementation). The operating-point rows apply the validation-frozen threshold $\tau^{*}$ to the test set unchanged.}}",
        r"\label{tab:causal_pr}",
        r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        "Method & " + " & ".join(heads) + r" \\",
        r"\midrule",
    ]
    for key, label in rows:
        e = pr.get(key)
        if not e:
            continue
        if "macro_mean" in e:
            cells = [PM(e["macro_mean"][m], e["macro_std"][m]) for m in metrics]
        else:
            mac = e["groups"]["macro"]
            cells = [F(mac[m]) for m in metrics]
        pre = "\\rowcolor{gray!8}\n" if key.endswith("_causal") else ""
        lines.append(pre + label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule",
              r"\multicolumn{6}{l}{\bedit{\textit{Deployable operating point "
              r"(validation-frozen $\tau^{*}$ applied to test):}}} \\"]
    for key, label in (("nsd_causal", "NSD (causal-retrained)"),
                       ("nsdh_causal", "NSD-H (causal-retrained)")):
        op = pr[key]["operating_point"]["mean"]
        lines.append(f"{label} @$\\tau^{{*}}$ & \\multicolumn{{5}}{{l}}{{"
                     f"P = {op['precision']:.3f}, R = {op['recall']:.3f}, "
                     f"F1 = {op['F1']:.3f}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    w("appdx_causal_pr.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    table1()
    table2()
    table3()
    appdx_xs_pairs()
    appdx_thresholds()
    appdx_coverage()
    appdx_excluded()
    appdx_sensitivity()
    appdx_causal_pr()
