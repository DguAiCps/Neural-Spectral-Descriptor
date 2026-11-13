#!/usr/bin/env python3
"""검산된 진단 결과의 영문 삽입안을 별도 폴더에 생성한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/pdv2_review_native_20261105"
OUT = RESULT / "paper_proposal"


def estimate(row: dict) -> str:
    """평균과 표본 표준편차를 같은 반올림으로 조판한다."""
    return rf"${row['v2_mean_percent']:.2f}\pm{row['v2_sample_std_pp']:.2f}$"


def main() -> None:
    """현재 원고를 수정하지 않고 독립 검토용 TeX를 만든다."""
    data = json.loads((RESULT / "submission_evidence.json").read_text())
    rows = data["rows"]
    seqot = [r for r in rows if r["method"] == "SeqOT"]
    imlpr = [r for r in rows if r["method"] == "ImLPR" and r["metric"] == "R@1"]
    helios = [r for r in rows if r["method"] == "HeLiOS"]
    assert len(seqot) == 3 and len(imlpr) == 6 and len(helios) == 12
    OUT.mkdir(exist_ok=True)
    (OUT / "submission_evidence_snapshot.json").write_bytes((RESULT / "submission_evidence.json").read_bytes())
    (OUT / "independent_verification_snapshot.json").write_bytes((RESULT / "learned_independent_verification.json").read_bytes())
    body = r"""% Proposed addition near the baseline comparison; current manuscript is untouched.
\paragraph{Baseline quality audit.}
We separately evaluate fixed author-released and shared-data checkpoints using the authors' retrieval implementations and available original-benchmark inputs (supplement). SeqOT reaches 92.17\% Recall@1 with author weights and $87.49\pm0.47$\% with our three retrained checkpoints under the released 15\,m labels; these labels differ from the paper's overlap criterion. ImLPR also attains high recall with author weights on regenerated MulRan inputs, while substantially lower recall remains with our shared-data weights. These diagnostics separate implementation and task differences from checkpoint quality, but do not establish optimal convergence. Their labels, candidate sets, and training-place overlap differ from the common evaluation, so author-weight results do not replace the place-disjoint main comparison.
"""
    (OUT / "body_insert.tex").write_text(body)
    lines = [r"% Verified results only. RING++ common-46 max-fix results are not yet included.",
             r"\subsection{Author-release and shared-data checkpoint diagnostics}",
             r"\label{app:native_audit}",
             r"We fix the author-released weights and the three existing shared-data checkpoints before these diagnostics, and run the same native retrieval path for all four models. Stored predictions and ground-truth lists are independently recounted. These experiments characterize the available released evaluation protocols; differences from the original papers are reported explicitly. No diagnostic test result is used to select a checkpoint or setting.",
             r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{6pt}",
             r"\begin{tabular}{llcrrr}", r"\toprule",
             r"Method & Evaluation condition & Metric & Paper reference & Author weights & Shared-data weights \\",
             r"\midrule"]
    for r in seqot:
        lines.append(f"SeqOT & NCLT Jan08 $\\rightarrow$ Feb05 & {r['metric']} & {r['paper_percent']:.2f} & {r['author_percent']:.2f} & {estimate(r)} " + r"\\")
    lines.append(r"\midrule")
    for r in imlpr:
        sequence = r["condition"].split(" / ")[0]
        k = r["condition"].split("k=")[1].split(" ")[0]
        lines.append(f"ImLPR & MulRan {sequence}, $k={k}$ & R@1 & {r['paper_percent']:.2f} & {r['author_percent']:.2f} & {estimate(r)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Released-protocol diagnostics (\%). Shared-data weights report the mean and sample SD over three fixed checkpoints. Paper values are reference values, not claims of identical-protocol replication. SeqOT uses the released distance-based labels, which differ from the paper's overlap labels. ImLPR inputs are regenerated from original scans; $k=15$ is the released normal-estimation setting and $k=25$ is the paper setting. Both are fixed in advance and reported. Author-weight ImLPR training and native evaluation overlap geographically in Daejeon.}",
              r"\label{tab:native_release_audit}", r"\end{table*}",
              r"\paragraph{SeqOT: released labels and task differences.}",
              r"For SeqOT~\citep{ma2022seqot}, we newly infer all 28,127 database and 28,239 query raw scans from NCLT 2012-01-08 and 2012-02-05. We preserve the released centered three-scan feature window, 20-scan GeM window, boundary handling, database/query strides of 10/5 raw scans, and FAISS search. The released ground-truth generator uses 3D distance below 15\,m, whereas the paper describes overlap above 0.3. The released evaluation has 5,648 queries, including 5,336 with positives; author-weight hit counts at ranks 1/5/20 are 4,918/5,063/5,171. Thus the close numerical agreement with the paper must not be described as replication of its overlap-based labels. NCLT is absent from our training and validation places.",
              r"On a fixed same-query diagnostic, 4,986 queries have valid poses and a 5\,m database match. With unchanged author predictions, recall is 96.25\% under the released labels and 91.01\% under 5\,m labels. These paired values have a different denominator from the full 92.17\% result. Reusing freshly inferred features on one common-protocol Jan08 sequence changes Recall@1 by only $-0.058$, $+0.058$, and $0.000$ percentage points across seeds (7,026 common queries, 1,717 positives), limiting the explanation attributable to numerical feature differences on that sequence.",
              r"\paragraph{ImLPR: preprocessing and weight effects.}",
              r"For ImLPR~\citep{jung2025imlpr}, the released processed archive was unavailable because of its download quota. We regenerate the inputs for MulRan DCC01/02/03 from the original OS1-64 scans with the released pipeline, 3\,m sampling, 10\,m labels, and the released temporal exclusions. This yields 1,417/1,260/1,598 frames and 768/639/901 positive queries. Eighteen fixed preprocessing comparisons (three scans per sequence, two normal settings) match the unmodified official computation exactly. All reported inference uses batch size one. The author-trained HeLiPR Daejeon locations overlap the native MulRan diagnostic locations; this diagnostic therefore cannot justify replacing the clean shared-data checkpoints. Under either fixed normal setting, a large author/shared-data weight gap remains. Our fixed-denominator retrieval F1 is also stored separately: the released F1 counts false negatives differently, so it is not interchangeable with main-table F1.",
              r"\paragraph{HeLiOS: partial sensor coverage and validation-place scope.}",
              r"For HeLiOS~\citep{jung2025helios}, we use released 8,192-point scans, the released overlap matrix with a strict 0.5 threshold, and the official retrieval function. We acquired six of twelve Roundabout sequence--sensor archives, covering 12 of 24 database/query pairs; all twelve trajectories and their overlap metadata are available. The six missing scan archives remain externally quota-limited. Table~\ref{tab:helios_partial_audit} reports every available pair rather than substituting a partial mean for the paper's full sensor-group means. Each row uses the intersection of evaluated queries across all four fixed models, since time filtering after the top-200 search can change eligibility slightly. Roundabout locations belong to our validation split, so these are checkpoint diagnostics, not an additional independent test. The author model uses four-sensor Daejeon training, while our shared-data training has different geographic and sensor coverage; these changes are not isolated causal effects.",
              r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{7pt}",
              r"\begin{tabular}{lrrrrrr}", r"\toprule",
              r"& \multicolumn{3}{c}{Database: Roundabout01-Aeva} & \multicolumn{3}{c}{Database: Roundabout01-Ouster} \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
              r"Query & Positives & Author & Shared-data & Positives & Author & Shared-data \\", r"\midrule"]
    index = {r["condition"]: r for r in helios}
    queries = sorted({r["condition"].split(" → ")[1] for r in helios})
    for query in queries:
        a = index["Roundabout01-Aeva → " + query]
        o = index["Roundabout01-Ouster → " + query]
        lines.append(f"{query} & {a['positive_queries']:,} & {a['author_percent']:.2f} & {estimate(a)} & {o['positive_queries']:,} & {o['author_percent']:.2f} & {estimate(o)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{All available HeLiOS pairs: Recall@1 (\%) on the same evaluated queries for author and three shared-data checkpoints. Shared-data values are mean $\pm$ sample SD. This is a partial released-input diagnostic on our validation places, not the original paper's complete benchmark mean.}",
              r"\label{tab:helios_partial_audit}", r"\end{table*}",
              r"\paragraph{A fixed-prediction label diagnostic.}",
              r"On the common evaluation, we additionally hold predictions, eligible candidates, the 61,682 queries positive at 5\,m, and city/seed aggregation fixed, and vary only the hit-distance tolerance. Changing 5\,m to 10\,m changes Recall@1 from 55.77 to 70.02\% for HeLiOS, 58.43 to 68.08\% for ImLPR, and 50.35 to 61.23\% for SeqOT. This isolates a label effect on the existing predictions; it does not select a new main-evaluation tolerance. Its changes are not additive with the native checkpoint differences, which use different query sets.",
              r"\paragraph{Training evidence and remaining limits.}",
              r"HeLiOS completes 80 epochs in each seed. Its training AP rises to 72.63--75.06\%, while validation AP peaks at epochs 7--10 and declines to 27.89--30.78\%. SeqOT completes 99 epochs in each of its two stages; its feature model also peaks early on validation, whereas its aggregation stage improves later. These logs do not support attributing every low result to an unfinished run. Conversely, ImLPR completes 100 epochs but its first seed's validation Recall@1 continues to rise near the budget limit, so optimal convergence is not established. The retained-checkpoint selection audit is separate from these original training-curve diagnostics. A longer run is not claimed here.",
              r"\paragraph{RING++ aggregation defect.}",
              r"The released RING++~\citep{xu2023ringpp} grid kernel updates a shared cell maximum using non-atomic reads and writes. Repeated fixed inputs produce different values. Holding point features and retrieval fixed, replacing only this operation by an exact maximum makes the checked repeated outputs identical. In the released NCLT Feb04-to-Mar17 setup (20\,m map spacing, 5\,m query spacing, 10\,m labels), the full-region R@1 is 62.52\% for RING, 55.60\% for the original RING++ run, and 62.24\% after exact maximum aggregation. With only the 992 positive queries retained from the original 1,054-query denominator, these become 66.43/59.07/66.13\%. The paper reports 70.98/73.21\% for RING/RING++; its exact spatial/input list is not established by the public release. Thus this controlled improvement identifies an implementation defect without reproducing the entire published advantage. The original CUDA result is one nondeterministic run, and its paired difference is not an expectation over repeated runs. The common-46-sequence reevaluation changes only the maximum operation and is reported separately once complete; no original timing or memory number is reinterpreted as a measurement of the corrected implementation.",
              r"% Integration note: replace the final pending sentence only after common-46 final verification."]
    (OUT / "supplement_insert.tex").write_text("\n\n".join(lines) + "\n")
    references = r"""@inproceedings{jung2025helios,
  title={{HeLiOS}: Heterogeneous {LiDAR} Place Recognition via Overlap-based Learning and Local Spherical Transformer},
  author={Jung, Minwoo and Jung, Sangwoo and Gil, Hyeonjae and Kim, Ayoung},
  booktitle={IEEE International Conference on Robotics and Automation},
  year={2025},
  url={https://arxiv.org/abs/2501.18943}
}
@inproceedings{jung2025imlpr,
  title={{ImLPR}: Image-based {LiDAR} Place Recognition using Vision Foundation Models},
  author={Jung, Minwoo and Fu, Lanke Frank Tarimo and Fallon, Maurice and Kim, Ayoung},
  booktitle={Proceedings of The 9th Conference on Robot Learning},
  series={Proceedings of Machine Learning Research},
  volume={305},
  pages={3318--3340},
  year={2025},
  url={https://proceedings.mlr.press/v305/jung25c.html}
}
"""
    bib = (ROOT / "docs/paper/body_v2.3/references.bib").read_text()
    import re
    additions = []
    for entry in re.findall(r"(?ms)^@.*?(?=^@|\Z)", references):
        key = entry.split("{", 1)[1].split(",", 1)[0]
        if "{" + key + "," not in bib:
            additions.append(entry)
    (OUT / "additional_references.bib").write_text(
        "% Append only missing entries; existing manuscript keys are reused.\n" + "\n".join(additions))
    (OUT / "review_references.bib").write_text(bib + "\n" + "\n".join(additions))
    review = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{booktabs,amsmath}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[breaklinks=true]{hyperref}
\title{Baseline Quality Audit: Proposed Insertions}
\author{Review copy}
\date{}
\begin{document}
\maketitle
\section{Proposed main-text addition}
\input{body_insert.tex}
\section{Proposed supplement addition}
\input{supplement_insert.tex}
\bibliographystyle{plainnat}
\bibliography{review_references}
\end{document}
"""
    (OUT / "review.tex").write_text(review)
    manifest = {"source": "results/pdv2_review_native_20261105/submission_evidence.json",
                "source_sha256": hashlib.sha256((RESULT / "submission_evidence.json").read_bytes()).hexdigest(),
                "frozen_source_copy": "submission_evidence_snapshot.json",
                "frozen_verification_copy": "independent_verification_snapshot.json",
                "rows": len(rows), "manuscript_modified": False,
                "integration_notes_ko": [
                    "body_insert.tex와 supplement_insert.tex는 부모 검토 후 삽입할 영문 제안이다.",
                    "현재 원고가 HeLiOS/ImLPR 키를 포함하면 그대로 재사용하며 additional_references.bib에는 없는 키만 넣는다. ImLPR PMLR305:3318–3340/2025와 HeLiOS arXiv2501.18943을 추가 확인했다.",
                    "RING 공통46 최종값과 별도 각도 대조, HeLiOS 재생성 경로는 이 시점 표에 포함하지 않는다.",
                    "기존 원고의 HeLiOS 'ten candidate epochs'는 80epoch 학습과 보존 후보 수를 혼동하므로 대체할 필요가 있다.",
                    "본문의 모든 저자 설정 동일·모든 방법 R@1 선택 표현은 별도 선택 담당 결과와 실제 예외에 맞춰 통합해야 한다.",
                    "공식 F1은 공통 F1과 정의가 달라 제안 표에는 Recall만 넣었다."]}
    (OUT / "proposal_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print({"output": str(OUT), "tables": 2, "rows": len(rows)})


if __name__ == "__main__":
    main()
