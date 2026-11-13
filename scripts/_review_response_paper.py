#!/usr/bin/env python3
"""완료한 현재 모델 심사 대조만 원고에 반영한다."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"results/pdv2_review_response_20261104"
PAPER = ROOT/"docs/paper/body_v2.2"
KEYS = ("AP", "max_F1", "R@100P", "R@1")
LABELS = dict(scpp="SC++", ring="RING", ringpp="RING++", ringsharp=r"RING\#-L", deepring="DeepRING",
              disco="DiSCO", helios="HeLiOS", imlpr="ImLPR", seqot="SeqOT", bevplace="BEVPlace++",
              nsd="NSD")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def table(caption: str, label: str, headings: str, rows: list, columns: str, wide: bool = False) -> str:
    env = "table*" if wide else "table"
    return (f"\\begin{{{env}}}[t]\n\\centering\\small\n\\setlength{{\\tabcolsep}}{{3.5pt}}\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{{columns}}}\n\\toprule\n"
            + headings + " \\\\\n\\midrule\n" + " \\\\\n".join(rows)
            + f" \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


def main() -> None:
    """최종 점추정과 짝지은 구간을 모델 범위와 함께 쓴다."""
    data = read(RUN/"analysis/summary.json")["methods"]
    boot = read(RUN/"analysis/bootstrap.json")["comparisons"]
    nsd = read(RUN/"nsd/test_summary.json")
    selected = read(RUN/"nsd/freeze.json")["selected_fusion"]
    tex = r"""\section{Current-model controlled comparisons}
\label{app:current_response}
All analyses here use the primary model and its final returned matches, with the main-table queries and city/seed aggregation. They do not substitute results from the historical models below.

\paragraph{Removing learned context and temporal scoring separately.}
The no-network control constructs $\mathrm{normalize}(\boldsymbol\omega\odot\mathbf d)$ directly. It evaluates no neural network or gate and appends no context block. The training-fitted diagonal weights and spectral statistics remain, so this is not a completely training-free method. The two indexes, candidate budget, spectral alignment, compression, correction, and final temporal scorer stay fixed. The center-only control instead preserves the original winners and replaces temporal confidence by its central-pair score. Their combined control removes both effects. All are evaluated on the same queries; none uses additional future observations.

For the no-network retuned row, we choose one common fusion setting by three-seed, three-family validation mean final AP from the same nine one-coordinate half/double variations used previously. Compression, confidence channels, and temporal coefficients remain fixed. The selected setting is fixed before test execution; all other removal rows retain the full model's fusion. This limited retuning does not claim an optimum over all no-network systems.
"""
    rows = []
    for arm, label in [("main", "NSD"), ("no_context__aux_half", "No neural context"),
                       ("no_context__"+selected, "No context, retuned"),
                       ("no_temporal_score", "Center-only confidence"), ("neither__aux_half", "Neither component")]:
        rows.append(label + " & " + " & ".join(f"{nsd[arm]['mean_std'][m]['mean']:.4f}" for m in KEYS))
    for arm, label in [("bevplace", "BEVPlace++")]:
        if arm in data: rows.append(label+" & "+" & ".join(f"{data[arm]['mean_std'][m]['mean']:.4f}" for m in KEYS))
    tex += table("Current-model final-result controls.", "tab:current_response", r"Method & AP & F1 & R@100P & R@1", rows, "lrrrr")
    tex += r"""
\paragraph{Conditional paired uncertainty.}
We draw 2,000 paired replicates, resampling sequences within each city and circular blocks of 100 consecutive queries within each drawn sequence. Every method receives the same sampled queries; each replicate averages its three fixed training seeds. Percentile intervals condition on these trained models and the exposed test set. They do not quantify model-selection uncertainty, establish equivalence, or correct for multiple comparisons. Threshold sweeps and direction-stratified results retain the main evaluation definitions.
"""
    rows = []
    for pair, label in [("nsd minus bevplace", "BEVPlace++"), ("nsd minus no_context__aux_half", "No neural context"),
                        ("nsd minus no_temporal_score", "Center-only confidence"),
                        ("nsd minus graph_free", "Graph-free learned projection"), ("nsd minus standard", "Standard temporal attention")]:
        if pair not in boot: continue
        cells = []
        for m in KEYS:
            v = boot[pair][m]
            lo, hi = [100*x for x in v["ci95"]]
            cells.append(f"${100*v['delta']:+.2f}\\,[{lo:+.2f},{hi:+.2f}]$")
        rows.append(label+" & "+" & ".join(cells))
    tex += table("NSD minus comparator: percentage-point difference and paired 95\% interval.", "tab:current_paired",
                 r"Comparator & AP & F1 & R@100P & R@1", rows, "lrrrr", True)
    tex += r"""
The interval for the neural-context AP difference includes zero, whereas temporal scoring improves AP and high-precision recall under this conditional analysis. Standard attention and graph-free learned projection remain close to DiffAttnConv. These results support the complete compact pipeline and temporal scoring more strongly than a distinctive accuracy benefit of the neural message-passing design.
"""
    tex += r"\section{Dataset and revisit-direction breakdowns}\label{app:current_datasets}"+"\n"
    tex += "Each dataset entry averages its positive-containing sequences and then training seeds; these dataset means differ from the six-city weighting of the main table. Values are percentages. KITTI 07 remains in pooled fixed-threshold precision but has no positive recall denominator.\n"
    methods = [m for m in LABELS if m in data]
    for left, right, title in [("kitti", "kitti360", "KITTI and KITTI-360"),
                               ("helipr", "mulran", "HeLiPR and MulRan"), ("nclt", "mcd", "NCLT and MCD")]:
        rows = []
        for method in methods:
            cells = []
            for sensor in (left, right):
                for m in KEYS:
                    value = sum(v["groups"]["dataset:"+sensor][m] for v in data[method]["per_seed"].values())/len(data[method]["per_seed"])
                    cells.append(f"{100*value:.2f}")
            rows.append(LABELS[method]+" & "+" & ".join(cells))
        headings = (r"& \multicolumn{4}{c}{"+title.split(" and ")[0]+r"} & \multicolumn{4}{c}{"+title.split(" and ")[1]+r"} \\"+"\n"
                    + r"Method & AP & F1 & R@100P & R@1 & AP & F1 & R@100P & R@1")
        tex += table(title+" final results (\%).", "tab:dataset_"+left, headings, rows, "lrrrrrrrr", True)
    rows = []
    for method in (m for m in ("nsd", "bevplace") if m in data):
        for group, label in (("same", "Same-direction available"), ("opposite_only", "Opposite-direction only")):
            means = {m: sum(v["directions"][group][m] for v in data[method]["per_seed"].values())/3 for m in KEYS}
            rows.append(LABELS[method]+" & "+label+" & "+" & ".join(f"{100*means[m]:.2f}" for m in KEYS))
    tex += table("Current models by available revisit direction (\%).", "tab:current_direction",
                 "Method & Positive-query subset & AP & F1 & R@100P & R@1", rows, "llrrrr", True)
    tex += ("Direction groups are determined from reference geometry, not from each method's prediction. A same-direction query has at least one same-direction true past match; the opposite-only group has only opposite-direction true matches. Each subset retains all original negatives and excludes the other positives. On opposite-only revisits, NSD AP is 80.16\% versus 80.50\% for BEVPlace++, while R@100P is 54.47\% versus 47.30\%. Exact yaw invariance of the analytic key does not imply invariance to changed visibility during reverse traversal.\n")
    if (RUN/"baseline_audit.json").exists():
        tex += r"""
\section{Baseline implementation checks}
\label{app:baseline_checks}
We rechecked 1,104 baseline ledgers for hashes, causal candidate prefixes, and finite returned scores; the common-query aggregates reproduce the main table. The stored RING and RING++ crosschecks compare three queries per sequence with the original correlation implementation and agree on every checked winner. We reran adapter checks for RING preprocessing/scoring, HeLiOS recall and retrieval, DiSCO L2 retrieval, ImLPR range-image/normal preprocessing, and SeqOT descriptor windows. These test the adapters on their stated synthetic or sampled inputs; they do not reproduce the original papers' benchmark scores or establish convergence under the eight-sequence training split. The main SeqOT comparison uses the official centered windows and native L2 score. Low results cannot be attributed solely to our former past-only window.

The training/selection records are retained per method and seed. ImLPR, for example, ran 100 epochs, with selected checkpoints at epochs 100, 95, and 75. This does not establish optimal training, but rules out a blanket explanation that every learned baseline received only a few epochs. Native-benchmark reproduction and the effects of training coverage, sensor changes, and different selection objectives remain separate limitations. Published checkpoints with overlapping training places cannot replace the place-disjoint main results without disclosing that overlap.
"""
    if (RUN/"false_accepts.json").exists():
        tex += r"""
\paragraph{Locations of fixed-threshold false accepts.}
At the previously validation-fixed P99 operating point, NSD has 24/23/23 false accepts across its three seeds and 22,649/22,592/22,565 true accepts. Of the false accepts, 22/21/21 occur on HeLiPR DCC05, with returned-place distances 5.01--6.32\,m; each seed also has one error on Riverside05 and one on Riverside06, both about 60\,m away. The DCC05 errors remain errors under the unchanged 5\,m rule. The original BEVPlace++ scores have 17/12/2 false accepts and 12,864/10,621/7,080 true accepts. Counts alone therefore do not determine precision: mean precision is 99.897\% and 99.909\%, respectively. These results use the original frozen thresholds.
"""
    (PAPER/"tables_v2/review_response.tex").write_text(tex)
    supp_path = PAPER/"cvpr2027_supplement_body.tex"
    supp = supp_path.read_text()
    inclusion = r"\input{tables_v2/review_response}"
    if inclusion not in supp:
        supp = supp.replace(r"\section{Reproducibility and evaluation limits}", inclusion+"\n\n"+r"\section{Reproducibility and evaluation limits}")
        supp_path.write_text(supp)
    body_path = PAPER/"cvpr2027_body.tex"
    body = body_path.read_text()
    body = body.replace("Paired uncertainty for this configuration and its component comparisons remains unmeasured; supplementary statistics for separately evaluated configurations are not evidence for these comparisons.",
                        "The supplement reports current-model dataset/direction breakdowns and paired block-bootstrap intervals. NSD minus BEVPlace++ is $-1.31$ AP points (95\\% interval $[-4.72,0.53]$), $-2.20$ F1 points ($[-4.97,-0.16]$), and $+7.61$ R@100P points ($[0.56,12.85]$). These conditional intervals do not establish equivalence or remove development-time selection uncertainty.")
    marker = "% BEGIN CURRENT RESPONSE\n"
    addition = r"""\cmedit{Directly removing the neural context and gate, while retaining training-fitted spectral weights and the full scorer, gives AP 0.8834 and R@100P 0.6628. Limited validation-only fusion retuning gives AP 0.8846. In contrast, removing temporal scoring while keeping the returned matches gives AP 0.8752 and R@100P 0.5876. The paired 95\% intervals for the temporal-scoring gains exclude zero; the neural-context AP interval includes zero. Thus the evidence supports temporal confidence more strongly than a distinct accuracy benefit of difference attention; details are in the supplement.}
"""
    body = re.sub(r"% BEGIN CURRENT RESPONSE\n.*?% END CURRENT RESPONSE\n", "", body, flags=re.S)
    before = r"\subsection{\cmedit{Matching responses and high-precision detection.}}"
    if before not in body:
        before = r"\subsection{\cmedit{High-precision detection.}}"
    assert before in body, "심사 대조 설명을 넣을 본문 위치가 없다"
    body = body.replace(before, marker+addition+"% END CURRENT RESPONSE\n\n"+before)
    body_path.write_text(body)


if __name__ == "__main__": main()
