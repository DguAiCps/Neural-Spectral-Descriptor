#!/usr/bin/env python3
"""검산 완료 비교 방법 설명을 읽기 순서에 맞는 작은 원고 조각으로 나눈다.

원래 세 생성 파일과 정적 진단 표는 보존한다. 읽기용 조각에서 현재 평가 조건과
대표 결과를 유지하며 중복과 공개 실험의 역추적 과정을 줄인다.
"""
import hashlib
import json
import re
from pathlib import Path


def _blocks(text: str) -> dict:
    """문단 제목 앞의 공통 머리말과 문단을 손실 없이 분리한다."""
    matches = list(re.finditer(r"\\paragraph\{([^}]*)\}", text))
    parts = {"_head": text[:matches[0].start()]}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        assert match.group(1) not in parts
        parts[match.group(1)] = text[match.start():end]
    assert "".join(parts.values()) == text
    return parts


def _body(block: str) -> str:
    return block.split("\n", 1)[1]


def _cut(text: str, anchors: list) -> list:
    """각 경계가 한 번만 나타나는지 확인하고 원문을 그대로 나눈다."""
    starts = [0]
    for anchor in anchors:
        assert text.count(anchor) == 1, anchor
        starts.append(text.index(anchor))
    assert starts == sorted(starts)
    pieces = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    assert "".join(pieces) == text
    return pieces


def _compact_native_protocols(fragments: dict) -> list:
    """현재 평가 조건을 남기고 공개 실험의 역추적 과정만 줄인다."""
    changes = []

    def replace_once(name: str, old: str, new: str, reason: str) -> None:
        assert fragments[name].count(old) == 1, (name, old)
        fragments[name] = fragments[name].replace(old, new)
        changes.append({
            "fragment": name, "original": old, "replacement": new,
            "reason": reason, "words_before": len(old.split()),
            "words_after": len(new.split()),
        })

    def replace_body(name: str, title: str, body: str, reason: str) -> None:
        old = _blocks(fragments[name])[title]
        new = "\\paragraph{" + title + "}\n" + body.strip() + "\n\n"
        replace_once(name, old, new, reason)

    scpp = _blocks(fragments["scpp_protocol"])
    replace_once("scpp_protocol", scpp["Published-curve checks."], "",
                 "벡터 PR 그림의 좌표·분모 역추정은 원천 기록에 보존한다.")
    replace_body(
        "scpp_protocol", "Revisit-list definition.",
        r"Under the paper's time-window revisit definition, with past nodes within 8\,m, no single time gap reproduces both the reported 852 KITTI~00 and 102 KITTI~08 revisits.",
        "서로 양립하지 않는 시간 간격의 세밀한 끝점 대신 재현 불일치 결론을 남긴다.",
    )
    replace_once(
        "scpp_protocol", " rather than the 4,043 that this model fixes for the PC curve", "",
        "삭제한 그림 역추정 문단에 의존하는 비교절을 함께 제거한다.",
    )
    replace_once(
        "scpp_protocol",
        "Agreement of the rounded PC value thus depends on the AUC convention, and this single run neither reproduces the KITTI 08 results nor identifies the authors' run.",
        "Agreement of the rounded PC value depends on the AUC convention.",
        "실제 수치와 모집단 차이가 이미 보여 주는 반복 한정을 줄인다.",
    )
    sampler_source = _blocks(fragments["bevplace_protocol"])["Released-run sampler."]
    assert "0.15211 lies 0.0004 below the three-seed range of the initial sampler (0.15252--0.15262)" in sampler_source
    assert "The pre-registered primary criterion is therefore not met, and its auxiliary criterion favors the initial sampler." in sampler_source
    replace_body(
        "bevplace_protocol", "Released-run sampler.",
        r"Table~\ref{tab:results} uses the current code's random-positive sampler. First-iteration losses recomputed from the released epoch-5 and epoch-10 states match the logged values within $10^{-3}$ only with the repository's initial hardest-positive sampler (commit 4ad02e31), but the logged epoch-mean loss lies 0.0004 below the range of three repeated runs with that sampler. The exact code files and full training procedure of that run remain unverified.",
        "RNG·재개 에폭·세부 손실값을 줄이고 현재 표본기, 초기 순전파 증거와 에폭 평균 불일치, 미확인 상태를 남긴다.",
    )
    replace_body(
        "ring_protocol", "RING evaluation scope.",
        r"""We follow the published T-RO version of RING++, with 20\,m map spacing, 5\,m query spacing, and a 10\,m revisit threshold. Its NCLT table specifies neither the session pair nor the evaluation region. We use the 2012-02-04 map and 2012-03-17 query sessions and the entire trajectory: the paper's resolution ablation reports the same 73.21\% for this pair, and its runtime analysis describes 253 map scans over the full trajectory. The official generator gives 254 map scans and 1,054 queries. Over all queries, RING reaches 62.52\% and exact-maximum RING++ with released normalization reaches 62.24\% (55.60\% with the released kernel in Table~\ref{tab:published_number_check}). Over the 992 queries with a map scan within 10\,m, RING and exact-maximum RING++ reach 66.43\% and 66.13\%. The paper defines the denominator only as total positives, so we report both. The published 0.1\,m voxel downsampling before neighbor search is omitted by the release; adding it once, fixed in advance, leaves both exact-maximum RING++ values unchanged.

Neither method uses learned weights. Our evaluation-set generator and NCLT loader match the release,\footnote{\href{https://github.com/lus6-Jenny/RING/blob/c79c58584ad841629506b5873ed1877c9c278ce5/evaluation/generate_evaluation_sets.py}{Official RING evaluation-set generator}.} and independent regeneration reproduces the ordered frame lists, positions, and positive counts. The authors' exact map/query lists and sampling phase, voxel implementation, and order of voxelization relative to ground removal and cropping remain unavailable. The NCLT loader keeps points more than 1\,m above the Segway wheel axle, 0.04\,m above the lidar; this removes the eight downward-looking beams in all 1,308 evaluated scans. The square $\pm$70\,m crop and a grid containing both 0 and $2\pi$ depart from the rotation-invariance assumptions. Adding only a 70\,m circular mask gives all-query R@1 of 63.19\% for RING and 62.62\% for exact-maximum RING++; the unmasked run reproduces the released distances bitwise. Excluding $2\pi$ gives 63.09\% and 61.95\%, respectively. These changes, single-map-scan removal, the released default test region or an alternate query session, and eigenentropy or local-feature coordinate changes do not close the gap to the published 70.98\% and 73.21\%. The Radon-grid, eigenentropy, and local-feature coordinate changes are not applied to the common-test correction. Rasterization and other untested effects remain unquantified.""",
        "대체 세션·스캔 제거의 개별 수치를 압축하고 현재 평가 설정·양쪽 분모·대표 결과·미확인 조건을 보존한다. 원고의 기존 RING 범위 문장에 있던 추가 진단의 미채택 사실을 이 조각에 한 번 둔다.",
    )
    changes[-1]["additional_source"] = {
        "path": "cvpr2027_supplement_body.tex (cleanup 이전 RING 범위 문단)",
        "original": "Separate changes to the Radon angle endpoints, eigenentropy convention, and local-feature coordinate scaling do not resolve the remaining discrepancy and are not applied to the common-test correction.",
    }
    replace_body(
        "disco_protocol", "Preprocessing variants.",
        r"With the same checkpoint, pairs, and labels, restoring the 5\,m ego-vehicle exclusion used at checkpoint upload, instead of the current 1\,m exclusion, gives R@1/R@1\% of 81.52/96.90\% versus 82.05/97.29\%. An earlier public generator rescaled each scan by its per-axis maxima and stored heights rather than binary occupancy. We use the fixed input matching the paper's definition; the training input of either checkpoint is unknown.",
        "현재·업로드 당시 전처리의 실제 결과는 유지하고 변경 commit 날짜만 줄인다.",
    )
    replace_body(
        "disco_protocol", "Evaluation-list sensitivity.",
        r"Keeping every second database sample changes mean R@1 from 82.05\% to 90.90/91.52\% for the two sampling phases and reduces queries with a positive from 3,173 to 1,715/1,689. With queries, weights, and the 1.5\,m radius fixed, replacing the 2012-03-17 database list with the first public sampler's list raises five-pair R@1 from 80.73\% to 89.45\% on the 427 queries positive under both lists. Both lists cover only the first 14\% of the session and differ in spacing and phase. These fixed-checkpoint diagnostics show sensitivity to list construction; they neither reproduce the published 30-pair value nor identify the authors' lists or the cause of the gap.",
        "목록 민감도의 대표 결과와 바뀌는 모집단을 남기고 이득 분해·중간 목록 크기·중복 평균을 줄인다.",
    )
    return changes


# 부록 과잉 서술 정리(2026-11-20): 원고에서 손으로 고친 문구를 조각 생성에 반영한다. (조각, 원문, 바꾼 문구)이며
# 원문은 조각 안에 한 번만 있어야 한다. 수치는 바꾸지 않는다.
DISCLOSURE_CLEANUP_20261120 = (
    ('common_task', r"support a shared loop-detection comparison, but are not equivalent to each paper's original task and need not affect methods equally",
     r"define a shared loop-detection comparison that differs from each paper's original task"),
    ('sensor_inputs', r"format; their effect is not assumed neutral",
     r"format"),
    ('scpp_configuration', r"sampled scans, computed from point geometry only, on training or validation sequences; MCD, which has neither,",
     r"30 sampled scans per sequence, computed from point geometry only, without poses or labels; they use training or validation sequences, except for MCD, which appears only in the test split and"),
    ('seqot_configuration', r"Skipping empty SeqOT overlap bins makes its sampler executable on the new data, but",
     r"The sampler skips empty SeqOT overlap bins on the new data, which"),
    ('published_overview', r"runs. We distinguish a faithful execution of a released implementation or checkpoint from confirmation that the paper used the same inputs and weights. HeLiOS uses the released benchmark inputs and reproduces all eight reported values within 0.07 percentage points. The other rows are numerical checks with the input, label, or checkpoint differences below; closeness alone does not establish an identical-protocol reproduction",
     r"runs under each paper's evaluation setting as reproduced from the released code and data described below"),
    ('scpp_protocol', r"""0.80. This evaluation has 889 positive queries rather than the paper's 852, so its query list is not identical. With the public equidistant keyframe selector and the evaluator's revisit rule, MulRan KAIST03 has 4,224 nodes and 2,055 revisits, and Riverside02 has 4,870 and 2,174, equal to the paper's counts; equal counts do not establish identical lists. The official PC/CC code then gives 8\,m PR-AUCs of 0.993/0.991 on KAIST03 and 0.773/0.924 on Riverside02, against the published 0.99/0.99 and 0.72/0.88.\footnote{The released PR aggregation leaves precision undefined (0/0) at thresholds without acceptances, so its AUC is undefined. We set this precision to 1, consistent with its starting point at zero recall; excluding these thresholds instead gives 0.983 for KAIST03 CC.} 

\paragraph{Revisit-list definition.}
Under the paper's time-window revisit definition, with past nodes within 8\,m, no single time gap reproduces both the reported 852 KITTI~00 and 102 KITTI~08 revisits""",
     r"0.80; this evaluation has 889 positive queries, and the paper reports 852. With the public equidistant keyframe selector and the evaluator's revisit rule, MulRan KAIST03 has 4,224 nodes and 2,055 revisits, and Riverside02 has 4,870 and 2,174, equal to the paper's counts. The official PC/CC code then gives 8\,m PR-AUCs of 0.993/0.991 on KAIST03 and 0.773/0.924 on Riverside02, against the published 0.99/0.99 and 0.72/0.88"),
    ('scpp_protocol', r"(4,021 with the released fast evaluator, which omits the last frame), and 422 revisits. Its PR-AUCs are 0.545 for PC, or 0.540 when thresholds without acceptances are excluded, and 0.005 for CC, against the published 0.55 and 0.00; the 4, 2, and 1\,m values also differ from the published ones. Agreement of the rounded PC value depends on the AUC convention",
     r"and 422 revisits. Its 8\,m PR-AUCs are 0.545 for PC and 0.005 for CC, against the published 0.55 and 0.00"),
    ('ring_protocol', r"threshold. Its NCLT table specifies neither the session pair nor the evaluation region. We use the 2012-02-04 map and 2012-03-17 query sessions and the entire trajectory: the paper's resolution ablation reports the same 73.21\% for this pair, and its runtime analysis describes 253 map scans over the full trajectory. The official generator gives 254 map scans and 1,054 queries. Over all queries, RING reaches 62.52\% and exact-maximum RING++ with released normalization reaches 62.24\% (55.60\% with the released kernel in Table~\ref{tab:published_number_check}). Over the 992 queries with a map scan within 10\,m, RING and exact-maximum RING++ reach 66.43\% and 66.13\%. The paper defines the denominator only as total positives, so we report both. The published 0.1\,m voxel downsampling before neighbor search is omitted by the release; adding it once, fixed in advance",
     r"threshold, using the 2012-02-04 map and 2012-03-17 query sessions over the entire trajectory, for which the paper reports 73.21\% in its resolution ablation and describes 253 map scans in its runtime analysis. Neither method uses learned weights. Our evaluation-set generator and NCLT loader match the release,\footnote{\href{https://github.com/lus6-Jenny/RING/blob/c79c58584ad841629506b5873ed1877c9c278ce5/evaluation/generate_evaluation_sets.py}{Official RING evaluation-set generator}.} which gives 254 map scans and 1,054 queries, and independent regeneration reproduces the ordered frame lists, positions, and positive counts. Over all queries, RING reaches 62.52\% and exact-maximum RING++ with released normalization reaches 62.24\% (55.60\% with the released kernel in Table~\ref{tab:published_number_check}), against the published 70.98\% and 73.21\%; over the 992 queries with a map scan within 10\,m, they reach 66.43\% and 66.13\%. Adding the paper's 0.1\,m voxel downsampling, which the released implementation does not include"),
    ('ring_protocol', r"""Neither method uses learned weights. Our evaluation-set generator and NCLT loader match the release,\footnote{\href{https://github.com/lus6-Jenny/RING/blob/c79c58584ad841629506b5873ed1877c9c278ce5/evaluation/generate_evaluation_sets.py}{Official RING evaluation-set generator}.} and independent regeneration reproduces the ordered frame lists, positions, and positive counts. The authors' exact map/query lists and sampling phase, voxel implementation, and order of voxelization relative to ground removal and cropping remain unavailable. The NCLT loader keeps points more than 1\,m above the Segway wheel axle, 0.04\,m above the lidar; this removes the eight downward-looking beams in all 1,308 evaluated scans. The square $\pm$70\,m crop and a grid containing both 0 and $2\pi$ depart from the rotation-invariance assumptions. Adding only a 70\,m circular mask gives all-query R@1 of 63.19\% for RING and 62.62\% for exact-maximum RING++; the unmasked run reproduces the released distances bitwise. Excluding $2\pi$ gives 63.09\% and 61.95\%, respectively. These changes, single-map-scan removal, the released default test region or an alternate query session, and eigenentropy or local-feature coordinate changes do not close the gap to the published 70.98\% and 73.21\%. The Radon-grid, eigenentropy, and local-feature coordinate changes are not applied to the common-test correction. Rasterization and other untested effects remain unquantified.

""",
     r""),
    ('ringsharp_protocol', r"""released weights and recipe.}
No RING\#-L author checkpoint was found in the official sources we examined, so a released-weight check is not possible. We instead trained RING\#-L once with the released code and configuration on NCLT Protocol~1 (seed 123, 30 epochs) and evaluated only the final-epoch checkpoint, fixed before training, with the released evaluation code. Our generated inputs have the paper's counts of 28,331 training samples and 265 test queries, which does not establish that they are the authors' lists. The released configuration uses a learning rate of $10^{-4}$ and a weight decay of $10^{-5}$, whereas the paper text states $10^{-3}$ and $10^{-4}$. At 10\,m, Recall@1, maximum F1, and PR-AUC are 0.805, 0.845, and 0.852, against the published 0.85, 0.87, and 0.91, so the published values are not reproduced. For this run, the 100 thresholds of the official PR computation yield only three distinct acceptance sets, and maximum F1 is reached by accepting all 265 queries. Without the authors' weights or training runs, we cannot identify the cause of these differences""",
     r"""native protocol.}
We found no released RING\#-L checkpoint, so we trained RING\#-L once with the released code and configuration on NCLT Protocol~1 (seed 123, 30 epochs; with the paper's counts of 28,331 training samples and 265 test queries) and evaluated the final-epoch checkpoint with the released evaluation code. At 10\,m, Recall@1, maximum F1, and PR-AUC are 0.805, 0.845, and 0.852, against the published 0.85, 0.87, and 0.91. This run uses the released configuration (learning rate $10^{-4}$, weight decay $10^{-5}$); a second run with the paper-text values ($10^{-3}$ and $10^{-4}$) gives 0.793, 0.838, and 0.870"""),
    ('disco_protocol', r"""queries. We therefore treat the 3\,m result as a diagnostic of the text discrepancy; the conditions that produced the table in either version cannot be determined from the released artifacts. 

\paragraph{Released-checkpoint identity.}
The checkpoint is byte-identical to the file currently linked from the official README. After the paper's publication, the authors stated that they were training a new model and would update the link, which was subsequently replaced.\footnote{\href{https://github.com/MaverickPeter/DiSCO-pytorch/issues/3\#issuecomment-1065178609}{Author comment on the released model}; \href{https://github.com/MaverickPeter/DiSCO-pytorch/commit/8e79286f05f912e1c4596aacf440f56e68cc4e31}{README link update}.} Whether this file is the checkpoint behind the published table is therefore unverified, as is the identity of our regenerated lists with the authors' evaluation lists, which were not found in the available author artifacts. """,
     r"queries, and we keep the 1.5\,m result as the reference."),
    ('disco_protocol', r"definition; the training input of either checkpoint is unknown",
     r"definition"),
    ('disco_protocol', r"only the first 14\% of the session and differ in spacing and phase. These fixed-checkpoint diagnostics show sensitivity to list construction; they neither reproduce the published 30-pair value nor identify the authors' lists or the cause of the gap",
     r"the first 14\% of the session and differ in spacing and phase. These fixed-checkpoint diagnostics show that the recall of the released checkpoint depends on the evaluation-list construction"),
    ('seqot_distance_labels', r"The paper defines a correct reference by overlap above 0.3, whereas the released ground-truth generator uses a 3D distance below 15\,m. We retain the downloadable labels in the native diagnostic, as directed by the authors, rather than substitute our common 5\,m rule. Using the authors' released pose arrays, we regenerate all 5,648 query label lists. The downloaded labels exactly equal the 15\,m lists after removing pairs with equal raw frame indices: 129 lists lose one pair each. The two sequences are from different dates, so equal indices do not denote the same scan. The current released generator does not contain that exclusion. Including these pairs changes the author's R@1 from 92.1664\% to 92.2039\% (two additional correct queries out of 5,336); R@5 and R@20 are unchanged. The original downloadable-label result remains the reference run. This verifies the distance-based nature of the release, not equivalence to the paper's overlap labels. The release and the paper therefore cannot be treated as interchangeable ground truth",
     r"We use the released labels, which mark a reference as correct within a 3D distance of 15\,m, as directed by the authors. With them, the author checkpoint attains R@1/R@5/R@20 of 92.17/94.88/96.91\% on the 5,336 positive queries, close to the published 91.7/94.7/96.8\%"),
    ('bevplace_protocol', r"points; identity with the paper's final weights is unverified. ",
     r"points."),
    ('bevplace_protocol', r"""\paragraph{Released-run sampler.}
Table~\ref{tab:results} uses the current code's random-positive sampler. First-iteration losses recomputed from the released epoch-5 and epoch-10 states match the logged values within $10^{-3}$ only with the repository's initial hardest-positive sampler (commit 4ad02e31), but the logged epoch-mean loss lies 0.0004 below the range of three repeated runs with that sampler. The exact code files and full training procedure of that run remain unverified.

\paragraph{Paper and code recipes.}
The paper states no positive-selection rule, and its recipe (AdamW, 50 epochs, step decay, mining after 10 epochs) matches neither code version nor the released run, so neither sampler restores it. The KITTI gaps above concern the released checkpoint, not our training.

""",
     r""),
    ('learned_direction', r"""These validation-only checks instead use the paper-recipe first-stage SeqOT model of seed~1 with GeM fixed at $p=3$ and a single-seed HeLiOS model trained from scratch (learning rate $5\times10^{-4}$; epoch~10 selected with 32-scan encoding batches), not the models of Table~\ref{tab:results}. On the nine validation sequences, travel direction is derived from trajectory displacement, and a query counts as same- or opposite-direction only when all its admissible ground-truth frames agree; mixed queries are counted separately. For KITTI-family queries, SeqOT first-stage training lowers opposite-direction R@1 from 51.2\% to 12.8\% (1,460 to 364 of 2,854) at the selected epoch and raises same-direction R@1 from 83.0\% to 89.7\% (837 to 905 of 1,009). Its per-sequence first-stage training plan contains one opposite-direction pair among 8,194 revisit positives within 5\,m. With 32-scan batches, HeLiOS reaches 85.7\% same-direction but 13.6\% opposite-direction KITTI-family R@1 (14.1\% with the official single-scan encoder). The training data and the deficit correspond, but we did not test whether adding opposite-direction pairs removes it.

Two bounded four-epoch SeqOT pilots, a degenerate-overlap guard and KITTI/HeLiPR update balancing, reach best family-macro R@1 of 54.9\% and 56.2\%, respectively, against 56.2\% for the existing selection. Neither meets the criterion fixed before training, and the bounded pilots show no material validation gain, so Table~\ref{tab:results} keeps its models. Seven KITTI-360 queries lack admissible ground-truth candidates; removing them leaves the checked SeqOT and HeLiOS checkpoint selections unchanged""",
     r"On the nine validation sequences, with travel direction derived from trajectory displacement and a query counted as same- or opposite-direction only when all its admissible ground-truth frames agree, first-stage training of the paper-recipe SeqOT model (seed~1, GeM fixed at $p=3$) lowers opposite-direction KITTI-family R@1 from 51.2\% to 12.8\% (1,460 to 364 of 2,854 queries) and raises same-direction R@1 from 83.0\% to 89.7\% (837 to 905 of 1,009). Its first-stage training plan contains one opposite-direction pair among 8,194 revisit positives within 5\,m"),
    ('common_metrics', r"966 frozen baseline ledgers, including the 46 released-kernel RING++ ledgers, and from the 46 exact-maximum RING++ ledgers for each correlation normalization. The FreSCo row uses the 46 ledgers of Table~\ref{tab:results}. The RING++ row uses the unit-variance implementation of Table~\ref{tab:results}; the two rows below the rule are the released-normalization diagnostics of Table~\ref{tab:baseline_release_comparison}. The SeqOT and DeepRING rows use the paper-recipe and scale-initialized ledgers of Table~\ref{tab:results}, respectively",
     r"the frozen baseline ledgers. The rows above the rule use the ledgers of Table~\ref{tab:results}; the two rows below it are the RING++ diagnostic variants of Table~\ref{tab:baseline_release_comparison}"),
    ('helios_targets', r"targets; this observation does not establish that overlap labels are erroneous",
     r"targets"),
    ('scpp_height', r"\%; the table keeps the sensor-specific offsets. These macro differences are within 0.15 points, whereas R@100P of individual sequences changes by $-7.44$ to $+12.38$ points, and we computed no significance test or interval across sequences",
     r"\%, within 0.15 points"),
    # 2026-11-21: 원고에서 다시 손으로 고친 문구(용어 "official"/"public"→"released", "Scan Context++"→"SC++", 문단 교체·문장 추가).
    ('common_task', r"""\paragraph{Common-task adaptations.}
Geographically disjoint retraining, KISS-ICP keyframes, past candidates, and the shared 5\,m correctness rule define the common task. """,
     r"""\paragraph{Common-evaluation adaptations.}
"""),
    ('common_task', r" These choices define a shared loop-detection comparison that differs from each paper's original task.",
     r""),
    ('helios_configuration', r"HeLiOS empty-miner guards affect logged statistics; they avoid",
     r"Our handling of HeLiOS batches without mined triplets affects logged statistics; it avoids"),
    ('scpp_configuration', r"Scan Context++ in Table~", r"SC++ in Table~"),
    ('seqot_configuration', r"within the common decision support", r"within the common observation support"),
    ('seqot_configuration', r"Table~\ref{tab:results} uses Eq.~13; the released computation is kept for the code-recipe diagnostics.",
     r"The released computation is used only in the \emph{Released-code training recipe} row of Table~\ref{tab:baseline_release_comparison}."),
    ('scpp_height', r"reusing the nine NCLT ledgers, and changes the macro AP, max F1",
     r"reusing the outputs for the nine NCLT sequences, and changes the six-city macro AP, maximum F1"),
    ('learned_direction', r"""\paragraph{Direction-dependent validation loss.}
On the nine validation sequences, with travel direction derived from trajectory displacement and a query counted as same- or opposite-direction only when all its admissible ground-truth frames agree, first-stage training of the paper-recipe SeqOT model (seed~1, GeM fixed at $p=3$)""",
     r"""\paragraph{Low HeLiOS and SeqOT scores.}
HeLiOS and SeqOT have the lowest AP in Table~\ref{tab:results}. Their deficit concentrates on reverse-only revisits: in Table~\ref{tab:direction_all} of the main text, their forward-only AP is 58.32\% and 64.16\%, whereas their reverse-only AP is 14.14\% and 15.34\%. Both methods are trained on overlap-based positives rather than the 5\,m place criterion. Of the 4,090,486 directed HeLiOS positives, 5.14\% lie within 5\,m and 58.32\% lie beyond 25\,m. For SeqOT, the first-stage training tuples of the paper recipe contain one opposite-direction pair among 8,194 revisit positives within 5\,m. On the nine validation sequences, first-stage training of that recipe (seed~1, GeM fixed at $p=3$; a diagnostic model, not a model of Table~\ref{tab:results})"""),
    ('learned_direction', r"12.8\% (1,460", r"12.8\% at the selected epoch (1,460"),
    ('learned_direction', r"Its first-stage training plan contains one opposite-direction pair among 8,194 revisit positives within 5\,m.",
     r"The released checkpoints of both methods reproduce their published results under the original protocols (Sec.~\ref{app:published_reproduction})."),
    ('published_overview', r"compares published references with our released-weight or released-code runs",
     r"compares published values with our released-checkpoint or released-code runs"),
    ('published_overview', r"The native checkpoints are used for reproduction checks; the common benchmark retains shared-data training and its fixed settings.",
     r"Checkpoints trained under each paper's protocol are used for reproduction checks."),
    ('published_overview', r"Published values and released-model checks", r"Published values and released-code and released-checkpoint checks"),
    ('scpp_protocol', r"""\paragraph{Scan Context++ released code.}
The official Scan Context++ PC""",
     r"""\paragraph{SC++ released code.}
The released SC++ PC"""),
    ('scpp_protocol', r"criterion give PR-AUC of", r"criterion give area under the precision--recall curve (PR-AUC) of"),
    ('scpp_protocol', r"With the public equidistant", r"With the released equidistant"),
    ('scpp_protocol', r"The official PC/CC code", r"The released PC/CC code"),
    ('scpp_protocol', r"Published Scan Context++ values", r"Published SC++ values"),
    ('ring_protocol', r"Neither method uses learned weights. ", r""),
    ('ring_protocol', r"{Official RING evaluation-set generator}", r"{Released RING evaluation-set generator}"),
    ('ring_protocol', r"leaves both exact-maximum RING++ values unchanged.",
     r"leaves both exact-maximum RING++ values unchanged. The paper's NCLT table specifies neither the session pair nor the evaluation region, and the authors' map and query lists are unavailable, so this check does not reconstruct the published table exactly. A 70\,m circular mask, removal of the $2\pi$ grid endpoint, removal of single map scans, the released default test region, and an alternate query session do not close the gap, which remains unexplained."),
    ('ringsharp_protocol', r"RING\#-L native protocol.}", r"RING\#-L published protocol.}"),
    ('ringsharp_protocol', r"At 10\,m, Recall@1,", r"At 10\,m, R@1,"),
    ('ringsharp_protocol', r"with the paper-text values", r"with the values stated in the paper"),
    ('ringsharp_protocol', r"gives 0.793, 0.838, and 0.870.",
     r"gives 0.793, 0.838, and 0.870. The paper does not state which epoch its table reports. Evaluating all 30 epochs of both runs with the released evaluator, the best epoch remains 2.0, 1.0, and 2.3 points (released configuration) and 2.4, 1.2, and 1.7 points (values stated in the paper) below the published R@1, maximum F1, and PR-AUC, so epoch selection does not explain the gap, which remains unexplained."),
    ('disco_protocol', r"using the official preprocessing", r"using the released preprocessing"),
    ('disco_protocol', r"the current official generator", r"the current released generator"),
    ('disco_protocol', r"a 3\,m radius gives R@1/R@1\% of", r"a 3\,m radius gives R@1 and top-1\% recall (R@1\%) of"),
    ('disco_protocol', r"exclusion used at checkpoint upload,", r"exclusion used when the checkpoint was released,"),
    ('disco_protocol', r"An earlier public generator", r"An earlier released generator"),
    ('disco_protocol', r"the first public sampler's list", r"the first released sampler's list"),
    ('seqot_distance_labels', r"the author checkpoint attains", r"the released checkpoint attains"),
    ('bevplace_protocol', r"evaluated with the official inference", r"evaluated with the released inference"),
    # 2026-11-21: 발표 수치 대조표를 한 단 폭으로 줄이고(행·수치 유지) 겹치거나 과잉인 구절을 지운다.
    ('published_overview', r"""\begin{table*}[t]
\centering\small\setlength{\tabcolsep}{5pt}""",
     r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{2pt}"""),
    ('published_overview', r"& Published & Our run & Difference (points) \\", r"& Paper & Our run & Diff. \\"),
    ('published_overview', r"\end{table*}", r"\end{table}"),
    ('disco_protocol', r"not computed on the same queries, and we keep the 1.5\,m result as the reference.",
     r"not computed on the same queries."),
    ('bevplace_protocol', r"for the same checkpoint within $3\times10^{-6}$ percentage points.",
     r"for the same checkpoint."),
    ('scpp_height', r"A sensitivity run fixed before evaluation replaces", r"A sensitivity run replaces"),
    ('scpp_height', r"on all 46 test sequences, reusing the outputs for the nine NCLT sequences, and changes",
     r"on all 46 test sequences, and changes"),
    ('scpp_configuration', 'computed from point geometry only, without poses or labels; they use training or validation sequences, except for MCD, which appears only in the test split and uses four of its test sequences.',
     'computed from point geometry only, without poses or labels.'),
    ('learned_direction', ' at the selected epoch (1,460 to 364 of 2,854 queries)',
     ' (1,460 to 364 of 2,854 queries)'),
    # 2026-11-21: 발표 수치 대조표에 SC++·FreSCo·RING#-L·BEVPlace++ 행을 넣고(문단 순서) 표와 겹치는 수치 문장을 줄인다. 관측 범위·빈 배치 처리 문구를 구체화한다.
    ('published_overview',
     'HeLiOS & Aeva / Narrow & R@1 & 80.60 & 80.56 & -0.04 \\\\\nHeLiOS & Aeva / Narrow & R@5 & 88.50 & 88.44 & -0.06 \\\\\nHeLiOS & Aeva / Wide & R@1 & 84.90 & 84.90 & 0.00 \\\\\nHeLiOS & Aeva / Wide & R@5 & 94.00 & 93.99 & -0.01 \\\\\nHeLiOS & Ouster / Narrow & R@1 & 70.00 & 69.96 & -0.04 \\\\\nHeLiOS & Ouster / Narrow & R@5 & 85.20 & 85.14 & -0.06 \\\\\nHeLiOS & Ouster / Wide & R@1 & 91.20 & 91.23 & +0.03 \\\\\nHeLiOS & Ouster / Wide & R@5 & 94.60 & 94.60 & 0.00 \\\\\n\\midrule\nSeqOT & NCLT Jan08--Feb05 & R@1 & 91.70 & 92.17 & +0.47 \\\\\nSeqOT & NCLT Jan08--Feb05 & R@5 & 94.70 & 94.88 & +0.18 \\\\\nSeqOT & NCLT Jan08--Feb05 & R@20 & 96.80 & 96.91 & +0.11 \\\\\n\\midrule\nDiSCO & NCLT 30 pairs / 1.5 m & R@1 & 89.08 & 82.05 & -7.03 \\\\\nDiSCO & NCLT 30 pairs / 1.5 m & R@1\\% & 99.14 & 97.29 & -1.85 \\\\\n\\midrule\nRING & NCLT full / 10 m & R@1 & 70.98 & 62.52 & -8.46 \\\\\n\\midrule\nRING++ & NCLT full / 10 m & R@1 & 73.21 & 55.60 & -17.61 \\\\\n',
     'SC++ (PC) & KITTI 00 & PR-AUC & 84 & 88.6 & $+$4.6 \\\\\nSC++ (CC) & KITTI 00 & PR-AUC & 80 & 82.0 & $+$2.0 \\\\\nSC++ (PC) & KITTI 08 & PR-AUC & 55 & 54.5 & $-$0.5 \\\\\nSC++ (CC) & KITTI 08 & PR-AUC & 0 & 0.5 & $+$0.5 \\\\\nSC++ (PC) & KAIST03 & PR-AUC & 99 & 99.3 & $+$0.3 \\\\\nSC++ (CC) & KAIST03 & PR-AUC & 99 & 99.1 & $+$0.1 \\\\\nSC++ (PC) & Riverside02 & PR-AUC & 72 & 77.3 & $+$5.3 \\\\\nSC++ (CC) & Riverside02 & PR-AUC & 88 & 92.4 & $+$4.4 \\\\\n\\midrule\nFreSCo & KITTI 08 & Max F1 & 85.9 & 86.1 & $+$0.2 \\\\\nFreSCo & KAIST03 & Max F1 & 98.7 & 98.5 & $-$0.2 \\\\\n\\midrule\nRING & NCLT full / 10 m & R@1 & 70.98 & 62.52 & $-$8.46 \\\\\n\\midrule\nRING++ & NCLT full / 10 m & R@1 & 73.21 & 55.60 & $-$17.61 \\\\\n\\midrule\nRING\\#-L & NCLT Protocol 1 & R@1 & 85 & 80.5 & $-$4.5 \\\\\nRING\\#-L & NCLT Protocol 1 & Max F1 & 87 & 84.5 & $-$2.5 \\\\\nRING\\#-L & NCLT Protocol 1 & PR-AUC & 91 & 85.2 & $-$5.8 \\\\\n\\midrule\nDiSCO & NCLT 30 pairs / 1.5 m & R@1 & 89.08 & 82.05 & $-$7.03 \\\\\nDiSCO & NCLT 30 pairs / 1.5 m & R@1\\% & 99.14 & 97.29 & $-$1.85 \\\\\n\\midrule\nHeLiOS & Aeva / Narrow & R@1 & 80.60 & 80.56 & $-$0.04 \\\\\nHeLiOS & Aeva / Narrow & R@5 & 88.50 & 88.44 & $-$0.06 \\\\\nHeLiOS & Aeva / Wide & R@1 & 84.90 & 84.90 & 0.00 \\\\\nHeLiOS & Aeva / Wide & R@5 & 94.00 & 93.99 & $-$0.01 \\\\\nHeLiOS & Ouster / Narrow & R@1 & 70.00 & 69.96 & $-$0.04 \\\\\nHeLiOS & Ouster / Narrow & R@5 & 85.20 & 85.14 & $-$0.06 \\\\\nHeLiOS & Ouster / Wide & R@1 & 91.20 & 91.23 & $+$0.03 \\\\\nHeLiOS & Ouster / Wide & R@5 & 94.60 & 94.60 & 0.00 \\\\\n\\midrule\nSeqOT & NCLT Jan08--Feb05 & R@1 & 91.70 & 92.17 & $+$0.47 \\\\\nSeqOT & NCLT Jan08--Feb05 & R@5 & 94.70 & 94.88 & $+$0.18 \\\\\nSeqOT & NCLT Jan08--Feb05 & R@20 & 96.80 & 96.91 & $+$0.11 \\\\\n\\midrule\nBEVPlace++ & KITTI 00 & R@1 & 100.0 & 99.85 & $-$0.15 \\\\\nBEVPlace++ & KITTI 02 & R@1 & 99.3 & 99.35 & $+$0.05 \\\\\nBEVPlace++ & KITTI 05 & R@1 & 100.0 & 98.20 & $-$1.80 \\\\\nBEVPlace++ & KITTI 06 & R@1 & 100.0 & 100.00 & 0.00 \\\\\nBEVPlace++ & KITTI 08 & R@1 & 99.1 & 99.41 & $+$0.31 \\\\\n'),
    ('published_overview',
     '\\caption{Published values and released-code and released-checkpoint checks (\\%).}',
     '\\caption{Published values and released-code and released-checkpoint checks (\\%; differences in points).}'),
    ('helios_configuration',
     'Our handling of HeLiOS batches without mined triplets affects logged statistics; it avoids failures without adding a new learning objective.',
     'When the hard-triplet miner finds no valid anchor in a batch, the released statistics logging raises an error. We skip the logging for such a batch; mining and the loss, which is zero when no triplet is mined, are unchanged.'),
    ('seqot_configuration',
     'SeqOT keeps the released centered windows, available within the common observation support.',
     'SeqOT keeps the released centered windows, whose scans all arrive before keyframe $q+15$.'),
    ('scpp_protocol',
     "The released SC++ PC and CC implementations on all 4,541 KITTI 00 scans with the paper's 8\\,m criterion give area under the precision--recall curve (PR-AUC) of 0.886 and 0.820, compared with the published 0.84 and 0.80; this evaluation has 889 positive queries, and the paper reports 852.",
     "We run the released SC++ PC and CC implementations with the paper's 8\\,m criterion; Table~\\ref{tab:published_number_check} lists the areas under the precision--recall curve (PR-AUC). On all 4,541 KITTI 00 scans, this evaluation has 889 positive queries, and the paper reports 852."),
    ('scpp_protocol',
     ' The released PC/CC code then gives 8\\,m PR-AUCs of 0.993/0.991 on KAIST03 and 0.773/0.924 on Riverside02, against the published 0.99/0.99 and 0.72/0.88.',
     ''),
    ('scpp_protocol',
     ' Its 8\\,m PR-AUCs are 0.545 for PC and 0.005 for CC, against the published 0.55 and 0.00.',
     ''),
    ('ringsharp_protocol',
     'At 10\\,m, R@1, maximum F1, and PR-AUC are 0.805, 0.845, and 0.852, against the published 0.85, 0.87, and 0.91. This run uses',
     'Table~\\ref{tab:published_number_check} lists its R@1, maximum F1, and PR-AUC at 10\\,m. This run uses'),
    ('seqot_distance_labels',
     'With them, the released checkpoint attains R@1/R@5/R@20 of 92.17/94.88/96.91\\% on the 5,336 positive queries, close to the published 91.7/94.7/96.8\\%.',
     'With them, we evaluate the released checkpoint on the 5,336 positive queries (Table~\\ref{tab:published_number_check}).'),
    ('bevplace_protocol',
     'The released BEVPlace++ KITTI checkpoint, evaluated with the released inference and evaluation functions and 5\\,m labels, gives R@1 of 99.85/99.35/98.20/100.00/99.41\\% on KITTI 00/02/05/06/08, compared with the published 100.0/99.3/100.0/100.0/99.1\\%. These values equal',
     'We evaluate the released BEVPlace++ KITTI checkpoint with the released inference and evaluation functions and 5\\,m labels on KITTI 00/02/05/06/08 (Table~\\ref{tab:published_number_check}). Its R@1 values equal'),
    # 2026-11-22: J절 문단 제목을 방법 이름으로 통일하고, 원 논문 수치 대조(J.3)는 표를 가리키는 짧은 문단으로 줄인다.
    ('published_overview', "Table~\\ref{tab:published_number_check} compares published values with our released-checkpoint or released-code runs under each paper's evaluation setting as reproduced from the released code and data described below. Checkpoints trained under each paper's protocol are used for reproduction checks.",
     "Table~\\ref{tab:published_number_check} compares published values with our runs of the released code or checkpoints under each paper's own evaluation setting."),
    ('scpp_protocol', "\\paragraph{SC++ released code.}\nWe run the released SC++ PC and CC implementations with the paper's 8\\,m criterion; Table~\\ref{tab:published_number_check} lists the areas under the precision--recall curve (PR-AUC). On all 4,541 KITTI 00 scans, this evaluation has 889 positive queries, and the paper reports 852. With the released equidistant keyframe selector and the evaluator's revisit rule, MulRan KAIST03 has 4,224 nodes and 2,055 revisits, and Riverside02 has 4,870 and 2,174, equal to the paper's counts.\n\n\\paragraph{All-frame released-code check.}\nRunning the released PC and CC functions once on all 4,071 KITTI 08 frames, with the evaluator's 49-frame exclusion and the 8\\,m criterion, gives 4,022 queries and 422 revisits. Published SC++ values and definitions are those of its journal version (Tables III, VIII, and IX).",
     "\\paragraph{SC++.}\nWe run the released polar-context and cart-context implementations with the paper's 8\\,m criterion and report the area under the precision--recall curve; KITTI~08 uses all 4,071 frames (Table~\\ref{tab:published_number_check}). On MulRan KAIST03 and Riverside02, the evaluation sets have the paper's revisit counts (2,055 and 2,174); on KITTI~00 they have 889 positive queries, against the paper's 852. Published values are those of the journal version."),
    ('ring_protocol', "\\paragraph{RING evaluation scope.}\nWe follow the published T-RO version of RING++, with 20\\,m map spacing, 5\\,m query spacing, and a 10\\,m revisit threshold, using the 2012-02-04 map and 2012-03-17 query sessions over the entire trajectory, for which the paper reports 73.21\\% in its resolution ablation and describes 253 map scans in its runtime analysis. Our evaluation-set generator and NCLT loader match the release,\\footnote{\\href{https://github.com/lus6-Jenny/RING/blob/c79c58584ad841629506b5873ed1877c9c278ce5/evaluation/generate_evaluation_sets.py}{Released RING evaluation-set generator}.} which gives 254 map scans and 1,054 queries, and independent regeneration reproduces the ordered frame lists, positions, and positive counts. Over all queries, RING reaches 62.52\\% and exact-maximum RING++ with released normalization reaches 62.24\\% (55.60\\% with the released kernel in Table~\\ref{tab:published_number_check}), against the published 70.98\\% and 73.21\\%; over the 992 queries with a map scan within 10\\,m, they reach 66.43\\% and 66.13\\%. Adding the paper's 0.1\\,m voxel downsampling, which the released implementation does not include, leaves both exact-maximum RING++ values unchanged. The paper's NCLT table specifies neither the session pair nor the evaluation region, and the authors' map and query lists are unavailable, so this check does not reconstruct the published table exactly. A 70\\,m circular mask, removal of the $2\\pi$ grid endpoint, removal of single map scans, the released default test region, and an alternate query session do not close the gap, which remains unexplained.",
     "\\paragraph{RING and RING++.}\nWe follow the published protocol on NCLT (2012-02-04 map, 2012-03-17 queries, 20\\,m map spacing, 5\\,m query spacing, 10\\,m revisit threshold) with the released evaluation-set generator and loader, which give 254 map scans and 1,054 queries. RING reaches 62.52\\% and RING++ 55.60\\% with the released kernel (62.24\\% with the exact maximum), against the published 70.98\\% and 73.21\\% (Table~\\ref{tab:published_number_check}). The paper's NCLT table specifies neither the session pair nor the evaluation region, and the authors' map and query lists are unavailable. A 70\\,m circular mask, removal of the $2\\pi$ grid endpoint, removal of single map scans, the paper's 0.1\\,m voxel downsampling, the released default test region, and an alternate query session do not close the gap, which remains unexplained."),
    ('ringsharp_protocol', "\\paragraph{RING\\#-L published protocol.}\nWe found no released RING\\#-L checkpoint, so we trained RING\\#-L once with the released code and configuration on NCLT Protocol~1 (seed 123, 30 epochs; with the paper's counts of 28,331 training samples and 265 test queries) and evaluated the final-epoch checkpoint with the released evaluation code. Table~\\ref{tab:published_number_check} lists its R@1, maximum F1, and PR-AUC at 10\\,m. This run uses the released configuration (learning rate $10^{-4}$, weight decay $10^{-5}$); a second run with the values stated in the paper ($10^{-3}$ and $10^{-4}$) gives 0.793, 0.838, and 0.870. The paper does not state which epoch its table reports. Evaluating all 30 epochs of both runs with the released evaluator, the best epoch remains 2.0, 1.0, and 2.3 points (released configuration) and 2.4, 1.2, and 1.7 points (values stated in the paper) below the published R@1, maximum F1, and PR-AUC, so epoch selection does not explain the gap, which remains unexplained.",
     '\\paragraph{RING\\#-L.}\nNo RING\\#-L checkpoint is released, so we trained it once with the released code and configuration on NCLT Protocol~1 and evaluated the final epoch with the released evaluation code (Table~\\ref{tab:published_number_check}). Over all 30 epochs of this run, and of a second run with the learning rate and weight decay stated in the paper, the best epoch remains 1.0--2.4 points below the published R@1, maximum F1, and area under the precision--recall curve; the gap remains unexplained.'),
    ('disco_protocol', "\\paragraph{DiSCO released checkpoint.}\nWe evaluate the released checkpoint on the paper's NCLT protocol of six sessions, all 30 ordered database/query pairs, and a 1.5\\,m match radius, using the released preprocessing, list sampler, inference, and recall functions. Independent L2 recomputation reproduces the top-1 indices, top-25 order, and recall curve of every pair. The 1.5\\,m radius, stated in the published version and used by the current released generator, defines our reference reproduction. The first arXiv version reports the same table values but states a 3\\,m radius.\\footnote{\\href{https://arxiv.org/abs/2010.10949v1}{DiSCO arXiv v1}; \\href{https://arxiv.org/abs/2010.10949v2}{arXiv v2}.} With the same descriptors and lists, a 3\\,m radius gives R@1 and top-1\\% recall (R@1\\%) of 90.28/98.01\\%. The radius also changes the set of queries with a positive (3,173 at 1.5\\,m and 3,876 at 3\\,m), so the two results are not computed on the same queries.\n\n\\paragraph{Preprocessing variants.}\nWith the same checkpoint, pairs, and labels, restoring the 5\\,m ego-vehicle exclusion used when the checkpoint was released, instead of the current 1\\,m exclusion, gives R@1/R@1\\% of 81.52/96.90\\% versus 82.05/97.29\\%. An earlier released generator rescaled each scan by its per-axis maxima and stored heights rather than binary occupancy. We use the fixed input matching the paper's definition.\n\n\\paragraph{Evaluation-list sensitivity.}\nKeeping every second database sample changes mean R@1 from 82.05\\% to 90.90/91.52\\% for the two sampling phases and reduces queries with a positive from 3,173 to 1,715/1,689. With queries, weights, and the 1.5\\,m radius fixed, replacing the 2012-03-17 database list with the first released sampler's list raises five-pair R@1 from 80.73\\% to 89.45\\% on the 427 queries positive under both lists. Both lists cover the first 14\\% of the session and differ in spacing and phase. These fixed-checkpoint diagnostics show that the recall of the released checkpoint depends on the evaluation-list construction.",
     "\\paragraph{DiSCO.}\nWe evaluate the released checkpoint on the paper's NCLT protocol (six sessions, all 30 ordered database/query pairs) with the released preprocessing, sampler, inference, and recall functions and the 1.5\\,m match radius of the published version (Table~\\ref{tab:published_number_check}). The first arXiv version reports the same table values with a 3\\,m radius;\\footnote{\\href{https://arxiv.org/abs/2010.10949v1}{DiSCO arXiv v1}; \\href{https://arxiv.org/abs/2010.10949v2}{arXiv v2}.} that radius gives R@1 and top-1\\% recall of 90.28/98.01\\%. Recall also depends on how the evaluation lists are sampled: keeping every second database sample changes mean R@1 from 82.05\\% to 90.90/91.52\\%."),
    ('seqot_distance_labels', '\\paragraph{SeqOT labels.}\nWe use the released labels, which mark a reference as correct within a 3D distance of 15\\,m, as directed by the authors. With them, we evaluate the released checkpoint on the 5,336 positive queries (Table~\\ref{tab:published_number_check}).\\footnote{\\href{https://github.com/BIT-MJY/SeqOT/blob/6e10697a4e6a423b5581e6a6ed6df9cfb386658b/data_prepararion/gen_ground_truth.py}{Released SeqOT label generator}; \\href{https://github.com/BIT-MJY/SeqOT/issues/1\\#issuecomment-1342137823}{author guidance on released ground truth}.}',
     '\\paragraph{SeqOT.}\nWe evaluate the released checkpoint on NCLT~Jan08--Feb05 with the released centered sequence construction, raw L2 search, and the released labels, which mark a reference as correct within 15\\,m (Table~\\ref{tab:published_number_check}).\\footnote{\\href{https://github.com/BIT-MJY/SeqOT/blob/6e10697a4e6a423b5581e6a6ed6df9cfb386658b/data_prepararion/gen_ground_truth.py}{Released SeqOT label generator}; \\href{https://github.com/BIT-MJY/SeqOT/issues/1\\#issuecomment-1342137823}{author guidance on released ground truth}.}'),
    ('bevplace_protocol', '\\paragraph{BEVPlace++ released checkpoint.}',
     '\\paragraph{BEVPlace++.}'),
    ('scpp_configuration', '\\paragraph{SC++ (A-PC).}',
     '\\paragraph{SC++.}'),
    ('scpp_configuration', 'in the A-PC configuration:',
     'in the augmented polar-context configuration:'),
    ('helios_configuration', '\\paragraph{HeLiOS batches without mined triplets.}',
     '\\paragraph{HeLiOS.}'),
    ('seqot_configuration', '\\paragraph{SeqOT input and tuple construction.}\n',
     ''),
    ('scpp_height', '\\paragraph{SC++ height sensitivity.}',
     '\\paragraph{SC++.}'),
    ('learned_direction', '\\paragraph{Low HeLiOS and SeqOT scores.}',
     '\\paragraph{HeLiOS and SeqOT.}'),
    ('seqot_configuration', "the paper's Eq.~13 instead uses", 'Eq.~13 of the SeqOT paper instead uses'),
    ('ringsharp_protocol', 'Over all 30 epochs of this run, and of a second run with the learning rate and weight decay stated in the paper, the best epoch remains 1.0--2.4 points below the published R@1, maximum F1, and area under the precision--recall curve; the gap remains unexplained.',
     'The final epoch is 2.5--5.8 points below the published R@1, maximum F1, and area under the precision--recall curve. Over all 30 epochs of this run, and of a second run with the learning rate and weight decay stated in the paper, the best epoch remains 1.0--2.4 points below; the gap remains unexplained.'),
    ('common_task', 'The 5--25\\,m query exclusion and fixed positive denominator are applied to all methods.',
     'The 5--25\\,m query exclusion is applied to all methods. Recall is computed over all revisit queries, so a revisit query accepted with a wrong place counts as both a false accept and a missed revisit.'),
    ('seqot_configuration', 'SeqOT keeps the released centered windows, whose scans all arrive before keyframe $q+15$.',
     'SeqOT keeps the released centered windows.'),
    ('disco_protocol', '{DiSCO arXiv v1}',
     '{arXiv:2010.10949v1}'),
    ('disco_protocol', '{arXiv v2}',
     '{arXiv:2010.10949v2}'),
    ('seqot_distance_labels', '{Released SeqOT label generator}',
     '{Label generator in the released SeqOT repository (\\texttt{gen\\_ground\\_truth.py})}'),
    ('seqot_distance_labels', '{author guidance on released ground truth}',
     '{author guidance in issue~1 of that repository}'),
    ('published_overview', '\\caption{Published values and released-code and released-checkpoint checks (\\%; differences in points).}',
     '\\caption{Published values and released-code and released-checkpoint checks (\\%; differences in points). PC and CC: polar and cart context; PR-AUC: area under the precision--recall curve; R@$k$ and R@1\\%: top-$k$ and top-1\\% recall.}'),
)


def write_baseline_reader_fragments(paper: Path, output: Path = None) -> dict:
    """기존 생성물과 원천 표를 읽고 방법별 조각 및 보존 목록을 쓴다."""
    output = output or paper / "baseline_fragments"
    output.mkdir(parents=True, exist_ok=True)
    originals = {
        name: (paper / name).read_text()
        for name in ("baseline_published_audit.tex", "baseline_common_diagnostics.tex",
                     "baseline_sampler_controls.tex", "tables_v3/appdx_baseline_diagnostics.tex")
    }
    published = _blocks(originals["baseline_published_audit.tex"])
    common = _blocks(originals["baseline_common_diagnostics.tex"])
    fragments = {}
    mapping = {}

    def emit(name: str, text: str, origin: str) -> None:
        assert name not in fragments
        fragments[name] = text
        mapping[name] = origin

    overview = published.pop("_head").replace(r"\subsection{Published-number and protocol audit}",
                                               r"\subsection{Released-code, released-checkpoint, and published-protocol checks}")
    emit("published_overview", overview, "baseline_published_audit.tex: head and published table")
    for title, name in (
        ("SeqOT labels.", "seqot_distance_labels"),
        ("SeqOT overlap labels.", "seqot_overlap_labels"),
        ("RING evaluation scope.", "ring_protocol"),
        ("DiSCO released checkpoint.", "disco_protocol"),
        ("BEVPlace++ released checkpoint.", "bevplace_protocol"),
        ("Scan Context++ released code.", "scpp_protocol"),
        ("RING\\#-L released weights and recipe.", "ringsharp_protocol"),
        ("Common-task adaptations.", "common_task"),
    ):
        emit(name, published.pop(title), "baseline_published_audit.tex: " + title)
    sensor = _body(published.pop("Sensor and implementation adaptations."))
    titles = ("Sensor inputs.", "SeqOT input and tuple construction.", "HeLiOS batches without mined triplets.",
              "SC++ (A-PC).", "SC++ height sensitivity.")
    names = ("sensor_inputs", "seqot_configuration", "helios_configuration", "scpp_configuration", "scpp_height")
    pieces = _cut(sensor, ["SeqOT keeps", "HeLiOS empty-miner", "Scan Context++ in Table", "A sensitivity run"])
    for title, name, piece in zip(titles, names, pieces):
        emit(name, "\\paragraph{" + title + "}\n" + piece.strip() + "\n\n",
             "baseline_published_audit.tex: Sensor and implementation adaptations.")
    assert not published

    head = common.pop("_head")
    head = head.replace(r"\subsection{Common-benchmark and training-signal diagnostics}" + "\n", "")
    head = head.replace(r"\label{app:baseline_signal}" + "\n", "")
    overview = head + common.pop("Retrieval and acceptance ranking.") + common.pop("Dataset dependence.")
    # RING++의 세 진단 행은 기존 변형 표와 동일하다. 새 원고는 해당 표를 참조한다.
    overview = overview.replace(r"\ref{tab:ringpp_maximum}", r"\ref{tab:baseline_release_comparison}")
    emit("common_metrics", overview, "baseline_common_diagnostics.tex: head, Retrieval and acceptance ranking., Dataset dependence.")
    for title, name in (
        ("Training targets.", "helios_targets"),
        ("SeqOT positive signal.", "seqot_signal"),
        ("Sensor and augmentation checks.", "seqot_sensor_checks"),
        ("Controlled yaw check.", "seqot_yaw"),
        ("Direction-dependent validation loss.", "learned_direction"),
    ):
        emit(name, common.pop(title), "baseline_common_diagnostics.tex: " + title)
    assert not common
    sampler = originals["baseline_sampler_controls.tex"].replace(
        r"\subsection{SeqOT positive-sampling controls}", r"\paragraph{Positive-sampling controls.}")
    emit("seqot_sampler", sampler, "baseline_sampler_controls.tex: entire file; heading depth only")

    # 원래 세 출력은 보존하고, 분할본에서 중복 안내와 가까운 문단의 자기 참조만 정리한다.
    reader_edits = (
        ("common_task", "The main results must be interpreted under this specified task. ", ""),
        ("seqot_signal", r" (Sec.~\ref{app:baselines})", ""),
    )
    for name, old, new in reader_edits:
        assert fragments[name].count(old) == 1, (name, old)
        fragments[name] = fragments[name].replace(old, new)

    # 긴 단락 안의 조건 전환에 제목을 붙인다. 제목 외 본문은 바꾸지 않는다.
    breaks = {
        "seqot_overlap_labels": [
            ("The released code applies these poses", "Overlap coordinate composition."),
            ("Translating the projection origin", "Overlap projection origin."),
        ],
        "disco_protocol": [
            ("The checkpoint is byte-identical", "Released-checkpoint identity."),
            ("With the same checkpoint, pairs, and labels", "Preprocessing variants."),
            ("Keeping every second database sample", "Evaluation-list sensitivity."),
        ],
        "bevplace_protocol": [
            ("The released training run is consistent", "Released-run sampler."),
            ("The paper states no positive-selection rule", "Paper and code recipes."),
        ],
        "scpp_protocol": [
            ("Under the paper's time-window revisit definition", "Revisit-list definition."),
            ("The PR curves in the authors' public arXiv source", "Published-curve checks."),
            ("Running the released PC and CC functions once", "All-frame released-code check."),
        ],
    }
    for name, boundaries in breaks.items():
        original = fragments[name]
        modified = original
        for anchor, heading in boundaries:
            assert modified.count(anchor) == 1, (name, anchor)
            modified = modified.replace(anchor, "\n\n\\paragraph{" + heading + "}\n" + anchor)
        # 새 제목과 공백을 빼면 원문과 정확히 같다.
        restored = modified
        for _, heading in boundaries:
            restored = restored.replace("\n\n\\paragraph{" + heading + "}\n", "")
        assert restored == original
        fragments[name] = modified

    reader_compactions = _compact_native_protocols(fragments)
    for name, old, new in DISCLOSURE_CLEANUP_20261120:
        assert fragments[name].count(old) == 1, (name, old[:60])
        fragments[name] = fragments[name].replace(old, new)

    diagnostic_tables = originals["tables_v3/appdx_baseline_diagnostics.tex"]
    native_start = diagnostic_tables.index(r"\begin{table*}")
    native = diagnostic_tables[native_start:]
    assert r"\label{tab:baseline_native}" in native and r"\label{tab:ringpp_maximum}" not in native
    emit("native_table", native, "tables_v3/appdx_baseline_diagnostics.tex: native-protocol table")
    # 공통 시험 RING++ 세 행은 기존 variant 표에 모두 있다. 원천 표는 지우지 않는다.
    variant = (paper / "tables_v3/appdx_baseline_variants.tex").read_text()
    triples = [
        ("RING++ released kernel", "Released kernel + released normalization"),
        ("RING++ exact max", "Exact max + released normalization"),
        (r"\quad + unit variance", r"Table~\ref{tab:results}: exact max + unit variance"),
    ]
    for left, right in triples:
        a = next(line for line in diagnostic_tables.splitlines() if line.startswith(left + " &"))
        b = next(line for line in variant.splitlines() if " & " + right + " & " in line)
        assert a.split(" & ")[-4:] == b.split(" & ")[-4:], (left, right)
    ring = next(line for line in diagnostic_tables.splitlines() if line.startswith("RING &"))
    assert ring == r"RING & 63.15 & 65.86 & 23.66 & 77.51 \\", ring

    manifest = {
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in originals.items()},
        "variant_table_sha256": hashlib.sha256(variant.encode()).hexdigest(),
        "reader_edits": [{"fragment": name, "removed": old, "replacement": new}
                         for name, old, new in reader_edits],
        "reader_compactions": reader_compactions,
        "disclosure_cleanup_20261120": [{"fragment": name, "original": old, "replacement": new}
                                        for name, old, new in DISCLOSURE_CLEANUP_20261120],
        "fragments": {},
        "duplicate_ringpp_rows": "all three metric vectors equal appdx_baseline_variants.tex",
        "ring_reference_row": ring,
    }
    for name, text in fragments.items():
        # 분할된 파일만 열어도 원래 결과 원천을 찾을 수 있게 주석을 함께 둔다.
        origin = mapping[name].split(":", 1)[0]
        if origin in originals and not text.startswith("% 원천:"):
            provenance = originals[origin].split(r"\subsection", 1)[0]
            if origin.startswith("tables_v3/"):
                provenance = originals[origin].split(r"\begin{table}", 1)[0]
            text = provenance + text
        (output / (name + ".tex")).write_text(text)
        manifest["fragments"][name + ".tex"] = {
            "source": mapping[name], "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
