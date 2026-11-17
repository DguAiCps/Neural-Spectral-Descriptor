#!/usr/bin/env python3
"""보고모델 교체(2026-11-16) 단위 F2: 본문·부록 NSD 의존 수치 전수 조사표.

각 항목의 새 값은 결과 JSON에서 직접 읽어 반올림한다(차이는 반올림 전 값으로 계산된 delta_pp를 그대로 반올림).
처리: replace(원고에 \\rev{}로 교체), hold(문장 보류, 초안 파일로), pending(결과 미완), na(NSD 무관·값 불변).

python3 scripts/_report_model_number_audit_20261116.py
출력: results/pdv2_report_model_20261116/manuscript/number_audit.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/pdv2_report_model_20261116"
OUT = R / "manuscript/number_audit.json"
_cache = {}


def J(rel):
    if rel not in _cache:
        _cache[rel] = json.loads((R / rel).read_text())
    return _cache[rel]


def G(rel, *keys):
    o = J(rel)
    for k in keys:
        o = o[k]
    return o


def f2(x):
    return f"{x:.2f}"


def iv(a):
    return f"[{a[0]:.2f}, {a[1]:.2f}]"


rows = []


def add(doc, loc, quote, current, new, source, field, action, note=""):
    rows.append(dict(doc=doc, location=loc, quote=quote, current=current, new=new,
                     source=(f"results/pdv2_report_model_20261116/{source}" if source else None), field=field,
                     action=action, note=note))


B, S = "cvpr2027_body.tex", "cvpr2027_supplement_body.tex"
IR, DS = "analysis/main/integrated_report.json", "analysis/main/direction_summary.json"
CT, BT = "controls/summary.json", "analysis/bev_temporal/highprec.json"
TH, FA, MF = "analysis/threshold/summary.json", "analysis/threshold/false_accepts.json", "analysis/matched_fa/result.json"
NR, AL, IB = "ablations/no_refining_summary.json", "ablations/aliasing_summary.json", "ablations/index_budget_summary.json"
EX, HD, YW, YC, PR = ("analysis/extras/extras.json", "analysis/heading/heading_acceptance_new.json", "yaw/yaw_summary.json",
                      "yaw/causes/summary.json", "analysis/precision/summary.json")
nb = G(IR, "comparisons", "diff-bev_native")

# ---------------- 본문 ----------------
add(B, "Abstract, last sentence", "11.7 points higher recall at 100% precision", "11.7", f"{nb['delta_pp']['R@100P']:.1f}",
    IR, "comparisons.diff-bev_native.delta_pp.R@100P", "replace")
add(B, "Abstract, last sentence", "running 10.36× faster with 90.9% less resident memory", "10.36×; 90.9%", None, "cost/", "(재측정 중)", "pending",
    "cost/DONE 없음. 비용 수치는 그대로 둠.")
add(B, "1 Introduction, paragraph 'We evaluate NSD ...'", "with 90.9% less resident memory", "90.9%", None, "cost/", "(재측정 중)", "pending")
rev_r100 = G(DS, "methods", "diff", "mean_std", "reverse_only", "R@100P", "mean") * 100
add(B, "1 Introduction, paragraph 'We evaluate NSD ...'", "it raises R@100P from 47.30% for BEVPlace++ to 63.20%", "63.20", f2(rev_r100),
    DS, "methods.diff.mean_std.reverse_only.R@100P.mean", "replace",
    f"BEVPlace++ 47.30 불변({G(DS, 'methods', 'bev_native', 'mean_std', 'reverse_only', 'R@100P', 'mean') * 100:.3f}).")
add(B, "1 Introduction, contributions, third item", "with 90.9% less resident memory", "90.9%", None, "cost/", "(재측정 중)", "pending")
ap_nsd, ap_bev = G(IR, "mean", "diff", "AP"), G(IR, "mean", "bev_native", "AP")
add(B, "4.2 Final recognition performance, first paragraph", "NSD has the highest F1 and R@100P among the eleven methods and the second-highest AP and R@1.",
    "second-highest AP", f"AP 1위 (NSD {100 * ap_nsd:.2f} > BEVPlace++ {100 * ap_bev:.2f})", IR, "mean.diff.AP, mean.bev_native.AP", "hold",
    "방향 바뀐 주장: AP가 2위가 아니라 1위. 초안 (A).")
add(B, "4.2, first paragraph", "Relative to BEVPlace++, its AP is lower by 0.41 ... R@1 by 2.17 ... F1 is higher by 0.33 ... R@100P by 11.71",
    "−0.41 / −2.17 / +0.33 / +11.71",
    f"AP {nb['delta_pp']['AP']:+.2f} / R@1 {nb['delta_pp']['R@1']:+.2f} / F1 {nb['delta_pp']['max_F1']:+.2f} / R@100P {nb['delta_pp']['R@100P']:+.2f}",
    IR, "comparisons.diff-bev_native.delta_pp", "hold", "AP 'lower'가 'higher'로 바뀜. 문장 전체 보류. 초안 (A).")
c = nb["conditional_95_pp"]
add(B, "4.2, first paragraph", "The 95% intervals ... [7.42, 16.19] points for R@100P and [−3.57, −0.77] points for R@1, while the AP and F1 intervals include zero.",
    "[7.42, 16.19]; [−3.57, −0.77]; AP·F1 구간 0 포함",
    f"R@100P {iv(c['R@100P'])}; R@1 {iv(c['R@1'])}; AP {iv(c['AP'])}(0 포함); F1 {iv(c['max_F1'])}(0 제외)",
    IR, "comparisons.diff-bev_native.conditional_95_pp", "hold", "F1 구간이 0을 포함하지 않음. 초안 (A).")
add(B, "4.2, paragraph after Fig. 4", "at 99% required precision, recall is 76.10% for NSD and 64.60% for BEVPlace++",
    "76.10", f2(G(PR, "nsd", "R@99P", "mean")), PR, "nsd.R@99P.mean", "replace",
    f"BEVPlace++ {G(PR, 'bevplace', 'R@99P', 'mean'):.2f} 불변.")
m30 = G(MF, "criteria", "FA<=30")
add(B, "4.2, paragraph after Fig. 4", "NSD's city recall at 30 false accepts is 16.20% versus 10.15% ... [1.78, 9.04], and it is higher at all five budgets from 0 to 100",
    "16.20; [1.78, 9.04]; all five budgets",
    f"{100 * m30['methods']['nsd']['mean_sd']['city_macro_recall']['mean']:.2f}; "
    f"{iv(m30['intervals_95_pp']['reselect']['city_macro_recall'])}; 예산 0·5에서 NSD 낮음",
    MF, "criteria.FA<=*.methods.*.mean_sd.city_macro_recall, intervals_95_pp.reselect.city_macro_recall", "hold",
    "'all five budgets' 거짓. 초안 (A).")
rv = {m: G(DS, "methods", m, "mean_std", "reverse_only") for m in ("diff", "bev_native")}
fw = {m: G(DS, "methods", m, "mean_std", "forward_only") for m in ("diff", "bev_native")}
add(B, "4.2, direction paragraph", "trails in R@1 by 0.46 points; BEVPlace++ leads on all four forward-only metrics.",
    "0.46; all four forward-only",
    f"역방향 R@1 격차 {100 * (rv['bev_native']['R@1']['mean'] - rv['diff']['R@1']['mean']):.2f}; 순방향 R@100P는 NSD {100 * fw['diff']['R@100P']['mean']:.2f} > {100 * fw['bev_native']['R@100P']['mean']:.2f}",
    DS, "methods.{diff,bev_native}.mean_std.{reverse_only,forward_only}", "hold", "'all four forward-only' 거짓. 초안 (A).")
bt = G(BT, "comparisons", "diff-bevt")
add(B, "4.2, direction paragraph", "BEVPlace++ reaches 63.23% R@100P, 6.42 points below NSD (95% interval [2.10, 10.97])",
    "6.42; [2.10, 10.97]", f"{bt['delta_pp']['R@100P']:.2f}; {iv(bt['conditional_95_pp']['R@100P'])}", BT,
    "comparisons.diff-bevt.{delta_pp,conditional_95_pp}.R@100P", "replace",
    f"63.23 불변({100 * G(BT, 'mean', 'bevt', 'R@100P'):.2f}).")
add(B, "4.2, direction paragraph", "and leads in AP and F1 by 1.90 and 1.58 points", "1.90; 1.58",
    f"{-bt['delta_pp']['AP']:.2f}; {-bt['delta_pp']['max_F1']:.2f}", BT, "comparisons.diff-bevt.delta_pp.{AP,max_F1}", "replace",
    f"방향 유지(BEV+TE가 높음). F1 구간 {iv(bt['conditional_95_pp']['max_F1'])}은 0 포함(문장은 점추정만 말함).")
fr = G(YW, "final_recognition")
d = [(fr[a]["mean_std"]["AP"]["mean"] - fr["0.0"]["mean_std"]["AP"]["mean"]) * 100 for a in fr if a != "0.0"]
add(B, "4.2, paragraph 'Query yaw rotation.'", "changes AP by −0.36 to +0.35 points over six nonzero angles", "−0.36 to +0.35",
    f"{min(d):+.2f} to {max(d):+.2f}", YW, "final_recognition.<angle>.mean_std.AP.mean − final_recognition.0.0", "replace", "2,944 질의 불변.")
ibm = G(IB, "comparisons", "dual-minus-aux_matched")
add(B, "4.3 Ablation study, paragraph 'Component ablation.'", "raises candidate recall over BEV-only proposals by 0.89 points (95% interval [0.54, 1.31])",
    "0.89; [0.54, 1.31]", f"{ibm['delta_pp']['candidate_recall']:.2f}; {iv(ibm['conditional_95_pp']['candidate_recall'])}", IB,
    "comparisons.dual-minus-aux_matched.{delta_pp,conditional_95_pp}.candidate_recall", "replace",
    "후보 집합은 모델 무관(candidate_sets_model_independent)이나 재표집 구간 상한이 1.32로 다시 계산됨.")
nrd = G(NR, "directions", "reverse_only")
add(B, "4.3, paragraph 'No temporal spectrum refining'", "raises NSD's AP from 79.92% to 83.72%, which accounts for the advantage over BEVPlace++ (80.50%)",
    "79.92; 83.72", f"{100 * nrd['identity']['mean']['AP']:.2f}; {100 * nrd['diff']['mean']['AP']:.2f}", NR,
    "directions.reverse_only.{identity,diff}.mean.AP", "replace", "80.02 < 80.50이라 'accounts for the advantage' 유지.")
add(B, "4.3, paragraph 'No temporal spectrum refining'", "Without refining, NSD's F1 is below BEVPlace++'s 89.00%; refining brings it to a comparable level.",
    "89.00 (BEVPlace++)", f"89.00 불변; NSD F1 {100 * G(IR, 'mean', 'diff', 'max_F1'):.2f}, 차이 구간 {iv(c['max_F1'])}", NR,
    "bev_reference_max_F1", "na", "수치 불변. 'comparable'은 이제 '높다'에 가까움 — 초안 (A) 주의 항목.")
add(B, "4.3, paragraph on alignment re-ranking", "alignment re-ranking reduces this rate to 5.96%", "5.96",
    f2(G(AL, "city_rates_percent", "macro", "nsd_distant")), AL, "city_rates_percent.macro.nsd_distant", "replace",
    f"앞의 20.07(크기만)은 불변({G(AL, 'city_rates_percent', 'macro', 'magnitude_distant'):.2f}).")
add(B, "4.3, Fig. 5 caption", "Markers: 15 training seeds; diamonds: means; bars: 95% intervals from seed and sequence resampling.",
    "15 seeds", "시드 4~6 3회, 4지표×전체·순·역", "analysis/learning/learning_figure.json", "data", "hold", "그림 구조 변경. 캡션 안은 초안 (B).")
add(B, "4.3, paragraph 'Effect of DiffAttnConv learning.'", "Over 15 training seeds, DiffAttnConv learning gains 0.12 AP, 0.25 F1, and 0.82 R@100P ... 14 of the 15 runs ... full NSD gains 0.02 AP points [−0.44, 0.29]",
    "0.12/0.25/0.82; [0.02,0.22]...; 14/15; 0.02 [−0.44,0.29]",
    "NSD−균등 감산 AP {:+.2f} {}, F1 {:+.2f} {}, R@100P {:+.2f} {}, R@1 {:+.2f} {}".format(
        *[x for k in ("AP", "max_F1", "R@100P", "R@1") for x in (G("analysis/main/learning.json", "seeds4to6", k, "mean_pp"),
                                                                 iv(G("analysis/main/learning.json", "seeds4to6", k, "interval")))]),
    "analysis/main/learning.json", "seeds4to6.*", "hold", "비교 구조가 바뀜(DiffAttnConv만 학습 조건 폐기). 새 문단 안은 초안 (B).")
pb = G(CT, "test", "reverse_only", "differences", "nsd-pastonly")
pf = G(CT, "test", "forward_only", "differences", "pastonly-nsd")
add(B, "4.3, paragraph 'Symmetric window versus past-only.'", "NSD exceeds Past-only NSD by 10.74 AP points [8.36, 13.22] and 30.88 R@100P points [24.35, 37.17]",
    "10.74 [8.36, 13.22]; 30.88 [24.35, 37.17]",
    f"{pb['point_pp']['AP']:.2f} {iv(pb['conditional_95_pp']['AP'])}; {pb['point_pp']['R@100P']:.2f} {iv(pb['conditional_95_pp']['R@100P'])}",
    CT, "test.reverse_only.differences.nsd-pastonly.{point_pp,conditional_95_pp}.{AP,R@100P}", "replace")
add(B, "4.3, paragraph 'Symmetric window versus past-only.'", "Past-only NSD exceeds NSD by 1.51 AP points [0.15, 3.17]", "1.51 [0.15, 3.17]",
    f"{pf['point_pp']['AP']:.2f} {iv(pf['conditional_95_pp']['AP'])}", CT, "test.forward_only.differences.pastonly-nsd", "replace")
tn = G(HD, "text_numbers")
add(B, "4.3, paragraph 'Symmetric window versus past-only.'", "accept 70.2% and 66.9% ... below 15°, but 55.6% and 22.4% above 165°",
    "70.2/66.9; 55.6/22.4",
    f"{tn['nsd_below15']:.1f}/{tn['past_below15']:.1f}; {tn['nsd_above165']:.1f}/{tn['past_above165']:.1f}", HD, "text_numbers", "replace")
add(B, "4.3, paragraph 'Symmetric window versus past-only.'", "At 75–105°, mean acceptance is lower for NSD than for BEVPlace++; this bin holds 489 of the 61,501",
    "489/61,501", f"{tn['bin75_105']['n']}/{tn['bin75_105']['total']:,} (NSD {tn['bin75_105']['nsd']:.1f} < BEVPlace++ {tn['bin75_105']['bevplace']:.1f})",
    HD, "text_numbers.bin75_105", "na", "주장·수치 불변.")
add(B, "4.4 Runtime and memory", "NSD takes 5.18 ms ... 53.67 ms (10.36× faster) ... 31.46 MB versus 346.31 MB (90.9% less; 3.12 versus 34.30 KB per frame)",
    "5.18; 10.36×; 31.46; 90.9%; 3.12", None, "cost/", "(재측정 중)", "pending", "BEVPlace++ 53.67·346.31·34.30은 비교 방법 수치.")
add(B, "5 Conclusion", "NSD exceeds its past-only variant by 10.74 AP points on reverse-only revisits", "10.74", f"{pb['point_pp']['AP']:.2f}",
    CT, "test.reverse_only.differences.nsd-pastonly.point_pp.AP", "replace")
add(B, "5 Conclusion", "with 90.9% less resident memory", "90.9%", None, "cost/", "(재측정 중)", "pending")
add(B, "5 Conclusion", "Training only the DiffAttnConv layers improves final detection over uniform neighbor weighting.", "-", "-", None, None, "hold",
    "폐기된 조건(DiffAttnConv만 학습). 초안 (B).")

# ---------------- 부록 ----------------
add(S, "A Implementation and inference, paragraph 'Observation-bound verification.' (second paragraph)",
    "The test audit covers 17 conditions ... 1,564 checks ... at most 1.43×10^-6; ... 6.31×10^-14", "17; 1,564; 1.43e-6; 6.31e-14",
    "새 감사는 단위별로 흩어짐: NSD 시드4~6 시험 414검사 최대 1.43e-6(pdv2_scratch_search agg.json), 절제 276검사 최대 6.44e-6, 대조(Past-only·절대 K/V) 시험 828검사 최대 2.86e-6",
    "ablations/macro_summary.json; controls/summary.json", "prefix_checks", "hold", "조건 구성이 바뀜. 통합 수치는 다시 세야 함. 초안 (C).")
add(S, "B Training and controlled learning, paragraph 'Absolute key/value (K/V) attention'", "NSD gains 0.42 AP points over this control (95% interval [0.05, 0.81]); the F1 and R@100P intervals include zero",
    "0.42 [0.05, 0.81]", "abskv−NSD AP {:+.2f} {}, F1 {:+.2f} {}, R@100P {:+.2f} {}, R@1 {:+.2f} {}".format(
        *[x for k in ("AP", "max_F1", "R@100P", "R@1") for x in (G(CT, "test", "all", "differences", "abskv-nsd", "point_pp", k),
                                                                 iv(G(CT, "test", "all", "differences", "abskv-nsd", "conditional_95_pp", k)))]),
    CT, "test.all.differences.abskv-nsd", "hold", "방향 바뀐 주장(절대 K/V가 네 지표 모두 높음). main 담당 절이라 손대지 않음. 초안 (A)(D).")
add(S, "D Statistical and geographic results, paragraph 'Paired intervals.'", "The three pretrained NSD starts share the fine-tuning sample order ...", "-", "-", None, None, "hold",
    "폐기된 절차(사전학습·미세조정). 초안 (C).")
lp = G(EX, "learning_per_seed_pp", "AP")
add(S, "D, paragraph 'Learning across starts and cities.'", "The DiffAttnConv learning AP gains at the three starts are +0.17, +0.02, and +0.20 ... Hamburg AP −0.01 ... Stockholm R@100P −0.21",
    "+0.17/+0.02/+0.20; Hamburg AP −0.01; Stockholm R@100P −0.21",
    "NSD−균등 시드별 AP " + "/".join(f"{x:+.2f}" for x in lp) + "; 음수 도시·지표 " + str(G(EX, "learning_city_negative")),
    EX, "learning_per_seed_pp, learning_city_negative", "hold", "비교 주체가 바뀜(DiffAttnConv만 학습 → NSD). 초안 (B).")
nri = G(NR, "comparisons", "diff-identity", "all")
seeds_ap = [x * 100 - 100 * G(NR, "mean", "identity", "AP") for x in G(NR, "directions", "all", "diff", "per_seed", "AP")]
add(S, "D, paragraph 'Learning across starts and cities.' (second paragraph)", "similar across starts (+1.12 to +1.18 points) ... +9.32 in Seoul, +3.06 in Daejeon, +1.45 in Karlsruhe, versus −0.23 in Ann Arbor, −3.39 in Stockholm, and −3.37 in Hamburg",
    "+1.12~+1.18; 9.32/3.06/1.45 vs −0.23/−3.39/−3.37",
    f"시드별 {min(seeds_ap):+.2f}~{max(seeds_ap):+.2f}; " + ", ".join(f"{k} {v['AP']:+.2f}" for k, v in nri["city_delta_pp"].items()),
    NR, "comparisons.diff-identity.all.city_delta_pp, directions.all.diff.per_seed.AP − mean.identity.AP", "hold",
    "Ann Arbor 부호가 음→양(+0.02)으로 바뀌어 'versus' 묶음이 거짓. 초안 (A).")
add(S, "D, same paragraph", "Under the block resampling ..., its F1 and R@100P intervals remain positive ([1.40, 3.95] and [1.24, 6.69]), whereas its AP interval includes zero ([−0.26, 2.70])",
    "[1.40, 3.95]; [1.24, 6.69]; [−0.26, 2.70]", None, None, "(정제 효과 블록 재표집 결과 없음)", "pending", "새 결과 없음. 그대로 둠.")
add(S, "D, paragraph 'Independent fine-tuning repetitions.' (three paragraphs)", "15 runs ... +0.12 AP [0.02, 0.22] ... 89.30% ... 89.43% ...", "다수", "-", None, None, "hold",
    "폐기된 실험(15회 미세조정 반복). 초안 (C).")
pm = G(AL, "pooled_mean_over_starts")
add(S, "D, paragraph 'Returned-place errors.'", "NSD returns the correct place for 59.03% of these distant errors and for 73.50% of nearer errors, while 1.28% of correct magnitude-only returns become errors",
    "59.03; 73.50; 1.28",
    f"{100 * pm['distant_fixed_by_nsd']:.2f}; {100 * pm['near_fixed_by_nsd']:.2f}; {100 * pm['correct_broken_by_nsd']:.2f}", AL,
    "pooled_mean_over_starts.{distant_fixed_by_nsd,near_fixed_by_nsd,correct_broken_by_nsd}", "replace",
    f"20.07·27.89(크기만)은 불변({G(AL, 'city_rates_percent', 'macro', 'magnitude_distant'):.2f}, {G(AL, 'city_rates_percent', 'macro', 'magnitude_error'):.2f}).")
add(S, "D, paragraph 'Returned-place errors.'", "Distant places still account for 86.08% of NSD's remaining errors", "86.08",
    f"{100 * pm['nsd_distant_share_of_errors']:.2f}", AL, "pooled_mean_over_starts.nsd_distant_share_of_errors", "replace")
T = lambda m, cr, *k: G(TH, "methods", m, cr, "mean_sd", *k)
add(S, "E Acceptance operating points, paragraph 'Results.'", "NSD has higher city and pooled recall than BEVPlace++ (20.69% and 39.92% versus ...) and more true accepts (24,621 versus 10,188)",
    "20.69; 39.92; 24,621",
    f"{100 * T('nsd', 'P99', 'city_macro_recall', 'mean'):.2f}; {100 * T('nsd', 'P99', 'recall', 'mean'):.2f}; {T('nsd', 'P99', 'TP', 'mean'):,.0f}",
    TH, "methods.nsd.P99.mean_sd.{city_macro_recall,recall,TP}.mean", "replace", "방향 유지.")
add(S, "E, paragraph 'Results.'", "BEVPlace++ has higher pooled precision (99.91% versus 99.58%) and fewer false accepts (10.3 versus 103.7 per run)",
    "99.58; 103.7", f"{100 * T('nsd', 'P99', 'precision', 'mean'):.2f}; {T('nsd', 'P99', 'FP', 'mean'):.1f}", TH,
    "methods.nsd.P99.mean_sd.{precision,FP}.mean", "replace", "방향 유지.")
add(S, "E, paragraph 'Results.'", "the P100 thresholds admit no validation false accepts but yield 29.0 (NSD) and 5.0 (BEVPlace++) test false accepts",
    "29.0", f"{T('nsd', 'P100', 'FP', 'mean'):.1f}", TH, "methods.nsd.P100.mean_sd.FP.mean", "replace",
    f"검증 오수락 {T('nsd', 'P100', 'val_FP', 'values')} → 'no validation false accepts' 유지. 다캐전 외 도시 FP 0 유지.")
add(S, "E, paragraph 'Results.' (F1 point)", "BEVPlace++ has higher city recall (64.01% versus 51.14%), mainly in Stockholm and Hamburg (71.0% and 72.4% versus 25.5% and 34.3%), while NSD is higher in Seoul (42.5% versus 22.6%)",
    "51.14; 25.5; 34.3; 42.5",
    f"{100 * T('nsd', 'F1', 'city_macro_recall', 'mean'):.2f}; {100 * T('nsd', 'F1', 'city_recall', 'stockholm', 'mean'):.1f}; "
    f"{100 * T('nsd', 'F1', 'city_recall', 'hamburg', 'mean'):.1f}; {100 * T('nsd', 'F1', 'city_recall', 'seoul', 'mean'):.1f}",
    TH, "methods.nsd.F1.mean_sd.{city_macro_recall,city_recall.*}", "replace",
    "방향 유지. NSD가 대전·앤아버에서도 높아졌으나 문장이 서울만 언급(거짓 아님).")
add(S, "E, paragraph 'Results.' (F1 point)", "NSD makes no false accepts in Stockholm or Hamburg ..., although its per-sequence sweeps average 90.58% maximum F1 on MCD",
    "90.58", f2(G(EX, "dataset_nsd_percent", "mcd", "max_F1")), EX, "dataset_nsd_percent.mcd.max_F1", "replace",
    f"Stockholm·Hamburg FP {T('nsd', 'F1', 'city_FP', 'stockholm', 'mean')}, {T('nsd', 'F1', 'city_FP', 'hamburg', 'mean')} → 주장 유지.")
add(S, "E, paragraph 'Results.' (F1 point)", "NSD and BEVPlace++ have similar pooled recall (62.98% versus 64.17%), and NSD has higher pooled precision (95.33% versus 94.14%) with fewer false accepts (1,905 versus 2,466)",
    "62.98; 95.33; 1,905",
    f"{100 * T('nsd', 'F1', 'recall', 'mean'):.2f}; {100 * T('nsd', 'F1', 'precision', 'mean'):.2f}; {T('nsd', 'F1', 'FP', 'mean'):,.1f}",
    TH, "methods.nsd.F1.mean_sd.{recall,precision,FP}.mean", "hold", "'similar recall','higher precision','fewer false accepts' 모두 거짓. 초안 (A).")
fa = G(FA, "mean_over_seeds", "nsd", "F1", "cross_kind_gap_returned", "negative|[30,60)|(25,100]")
add(S, "E, paragraph 'False-accept distances.'", "This group contains 868.3 of 1,905.0 false accepts for NSD", "868.3 of 1,905.0",
    f"{fa:,.1f} of {T('nsd', 'F1', 'FP', 'mean'):,.1f}", FA, "mean_over_seeds.nsd.F1.cross_kind_gap_returned['negative|[30,60)|(25,100]']", "replace",
    "이 군이 NSD에서도 가장 큼(주장 유지).")
add(S, "E, paragraph 'Matched false-accept budgets.'", "pooled recall is 39.92% for NSD versus 16.52% for BEVPlace++", "39.92",
    f"{100 * T('nsd', 'P99', 'recall', 'mean'):.2f}", TH, "methods.nsd.P99.mean_sd.recall.mean", "replace")
add(S, "E, paragraph 'Matched false-accept budgets.'", "Table ... shows higher NSD city recall at all five budgets.", "all five budgets",
    "; ".join(f"{k} NSD {100 * v['methods']['nsd']['mean_sd']['city_macro_recall']['mean']:.2f} vs {100 * v['methods']['bevplace']['mean_sd']['city_macro_recall']['mean']:.2f}"
              for k, v in G(MF, "criteria").items() if v["kind"] == "budget"),
    MF, "criteria.FA<=*.methods.*.mean_sd.city_macro_recall.mean", "hold", "예산 0·5에서 NSD가 낮음. 초안 (A).")
add(S, "E, paragraph 'Matched false-accept budgets.'", "Paired 95% intervals exclude zero at 30 and 100 but include it at 0, 5, and 10 ... NSD's false accepts at budgets up to 10 occur only in HeLiPR DCC05",
    "-", "; ".join(f"{k} {iv(v['intervals_95_pp']['reselect']['city_macro_recall'])}" for k, v in G(MF, "criteria").items() if v["kind"] == "budget"),
    MF, "criteria.*.intervals_95_pp.reselect.city_macro_recall", "na", "주장 유지(30·100 제외, 0·5·10 포함, DCC05만).")
yc = G(YC, "results")
add(S, "F Traversal direction ..., paragraph 'Mechanism controls.'", "Both replacements together restore all 6,624 original returned places, with a maximum score change 2.38×10^-6",
    "2.38e-6", f"{max(yc[a]['arms']['ideal_both']['measurements']['confidence_change']['max'] for a in yc):.2e}", YC,
    "results.<angle>.arms.ideal_both.measurements.confidence_change.max (최대)", "replace",
    f"6,624 = {sum(yc[a]['arms']['ideal_both']['n'] for a in yc)}, winner_changes 0 유지.")
add(S, "F, paragraph 'Replay and discretization checks.'", "all 8,832 returned places match ..., with a maximum score error of 1.91×10^-6", "1.91e-6",
    f"{G('yaw/yaw_baseline_validation.json', 'max_score_error'):.2e}", "yaw/yaw_baseline_validation.json", "max_score_error", "replace",
    f"8,832 = query_seed_checks {G('yaw/yaw_baseline_validation.json', 'query_seed_checks')}, 불일치 0.")
add(S, "F, paragraph 'Full and DiffAttnConv learning by direction.'", "(문단 전체)", "-", "-", "analysis/main/direction_sequence_summary.json", "comparisons.diff-fixed", "hold",
    "폐기된 비교 구조. 초안 (C).")
add(S, "F, paragraph 'Intervals under different resampling procedures.'", "All-query and reverse-only AP intervals include zero under both block summaries ... all-query F1 lower endpoint is only +0.002",
    "AP 구간 0 포함; +0.002",
    "NSD−균등 블록: 전체 AP {}, F1 {}; 역방향 AP {}".format(
        iv(G("analysis/main/direction_block_summary.json", "comparisons", "diff-fixed", "all", "block", "overall", "AP", "ci95_pp")),
        iv(G("analysis/main/direction_block_summary.json", "comparisons", "diff-fixed", "all", "block", "overall", "max_F1", "ci95_pp")),
        iv(G("analysis/main/direction_block_summary.json", "comparisons", "diff-fixed", "reverse_only", "block", "overall", "AP", "ci95_pp"))),
    "analysis/main/direction_block_summary.json", "comparisons.diff-fixed.*.block.overall", "hold", "AP 구간이 0을 제외. 초안 (A).")
add(S, "G Temporal refining and observation windows, paragraph 'Contrast versus aggregation.'", "79.38 ... 69.67 ... 80.05 ... 83.49 ... 88.16 ... 1.12 [0.08, 2.07] ...", "다수",
    None, None, "(새 고정 설정의 균등 더하기 결과 없음)", "pending", "균등 감산 설정이 (2,1,2,2), λ=0.25로 바뀌어 수치 전체가 옛 설정. 새 결과 없음. 그대로 둠.")
bevh = G(BT, "comparisons", "diff-bevt")
add(S, "G, paragraph 'Temporal evidence applied to BEVPlace++.'", "NSD remains higher at 99% and 100% required precision, by 5.75 and 6.42 points, and the interval for R@95P includes zero.",
    "5.75; 6.42; R@95P 구간 0 포함",
    f"{bevh['delta_pp']['R@99P']:.2f}; {bevh['delta_pp']['R@100P']:.2f}; R@95P {iv(bevh['conditional_95_pp']['R@95P'])}(0 제외)",
    BT, "comparisons.diff-bevt", "hold", "R@95P 구간이 0을 제외. 초안 (A).")
add(S, "G, paragraph 'Temporal evidence applied to BEVPlace++.'", "Temporal evidence raises BEVPlace++ ... by 1.49 AP points and 5.29 R@100P points. With it, BEVPlace++ exceeds NSD in AP and maximum F1.",
    "1.49; 5.29", f"{G(BT, 'comparisons', 'bevt-bev', 'delta_pp', 'AP'):.2f}; {G(BT, 'comparisons', 'bevt-bev', 'delta_pp', 'R@100P'):.2f}", BT,
    "comparisons.bevt-bev.delta_pp", "na", "BEVPlace++만의 수치. 'exceeds NSD in AP and F1' 유지(점추정).")
add(S, "H Candidate retrieval under matched budgets, paragraph after Table", "improves candidate recall over BEV-only proposals by 0.89 points (conditional 95% interval [0.54, 1.31])",
    "[0.54, 1.31]", iv(ibm["conditional_95_pp"]["candidate_recall"]), IB, "comparisons.dual-minus-aux_matched.conditional_95_pp.candidate_recall", "replace",
    "179.6·98.82 불변.")
add(S, "H, same paragraph", "The corresponding final changes are AP +0.12, F1 −0.003, R@100P −0.28, and revisit R@1 +0.16 points.",
    "+0.12; −0.003; −0.28; +0.16",
    f"{ibm['delta_pp']['AP']:+.2f}; {ibm['delta_pp']['max_F1']:+.2f}; {ibm['delta_pp']['R@100P']:+.2f}; {ibm['delta_pp']['R@1']:+.2f}",
    IB, "comparisons.dual-minus-aux_matched.delta_pp", "replace", "AP +0.12는 같아 표시하지 않음.")
a200 = G(IB, "comparisons", "dual-minus-aux_200", "delta_pp")
add(S, "H, same paragraph", "At the top-200 budget, BEV-only proposals change each final metric by at most 0.25 points", "0.25",
    f"{max(abs(a200[k]) for k in ('AP', 'max_F1', 'R@100P', 'R@1')):.2f}", IB, "comparisons.dual-minus-aux_200.delta_pp (절댓값 최대)", "replace",
    "네 최종 지표 구간이 두 BEV-only 대조 모두에서 0 포함 → 주장 유지.")
add(S, "I Resource measurements (all paragraphs)", "3.22 ms; 1.96 ms; 5.18 ms; seed-1; 1.70/1.52 ms; 3.27e-7; 7.72e-7; 1.66→2.01 ms; 6.04→31.46 MB", "다수",
    None, "cost/", "(재측정 중)", "pending")
dsN = G(DS, "methods", "diff", "mean_std")
add(S, "J ..., paragraph 'HeLiOS and SeqOT performance gaps.'", "(HeLiOS 14.14%, SeqOT 15.34%, NSD 83.72%) ... (58.32% and 64.16%, respectively, versus 90.06%)",
    "83.72; 90.06", f"{100 * dsN['reverse_only']['AP']['mean']:.2f}; {100 * dsN['forward_only']['AP']['mean']:.2f}", DS,
    "methods.diff.mean_std.{reverse_only,forward_only}.AP.mean", "replace", "HeLiOS·SeqOT 수치는 비교 방법.")
add(S, "K Scope of illustrations, first paragraph", "Fig. learning plots the per-run gains and intervals of the independent fine-tuning repetitions ...; markers distinguish the three starting models.",
    "-", "-", "analysis/learning/learning_figure.json", None, "hold", "그림 구조 변경. 초안 (B).")
add(S, "K Scope of illustrations, second paragraph", "181 of 61,682; three runs; Past-only NSD equals its value in Table online_components", "181/61,682", "불변", HD, "text_numbers", "na")
add(S, "tables_v3/temporal_controls.tex (F1 소관 파일 안 문장)", "Symmetric versus one-sided ... (80.82%) ... (81.00%), versus 83.49% ...; Zero-lag configurations ...", "다수",
    "새 창 선택: 과거 −0.125 (검증 AP 80.69), 미래 −0.25 (80.97), 대칭 83.34", "ablations/window_selection.json", "val_macro_percent", "na",
    "F1 소관 파일이라 손대지 않음. 문장 변경 항목은 초안 (C)에 적음.")

add(S, "B Training and controlled learning, new paragraph 'Learned contrast strength.' + Table gate_learning", "AP 89.99 ... +0.11 [−0.20, 0.42] ... −0.83 [0.26, 1.41] ... 3.26 [1.23, 5.85] ... 8.20 [3.64, 12.74]", "(새 문단)",
    "게이트 자유 89.99/90.17/71.34/91.97; 자유−고정 AP +0.11 [−0.20, 0.42], F1 −0.22 [−0.48, 0.02], R@1 +0.23 [−0.07, 0.56], R@100P −0.83 [−1.41, −0.26]; ‡−균등 AP −3.26 [−5.85, −1.23], R@100P −8.20 [−12.74, −3.64]",
    "analysis/gate/gate_table.json", "rows_main", "replace", "F3 후속: 사용자 결정으로 새 문단·표 삽입(\\rev)")
add(S, "B Training and controlled learning, Table stage2_controls and its lead sentence", "Table~\\ref{tab:stage2_controls} lists the Stage-2 controls ...", "표·문장", "삭제",
    None, None, "removed", "F3 후속: 사용자 결정으로 삭제(결과 JSON 보존)")

# ---------------- 단위 F3(2026-11-17) 갱신: 비용 재측정·새 재계산·부록 구조 서술 정리 반영 ----------------
CO = "cost/summary.json"
cm = G(CO, "main", "skipped")
F3 = [  # (location 부분 문자열, quote 부분 문자열, action, new, source, field, note)
    ("Abstract", "10.36", "replace", f"{G(CO, 'bevplace', 'speedup', 'skipped'):.2f}×; {100 * G(CO, 'bevplace', 'resident_reduction', 'skipped'):.1f}% (불변)", CO, "bevplace.speedup.skipped", "F3"),
    ("1 Introduction, paragraph 'We evaluate", "90.9%", "na", "90.9% (불변)", CO, "bevplace.resident_reduction.skipped", "F3: 재측정 값 90.93%"),
    ("contributions", "90.9%", "na", "90.9% (불변)", CO, "bevplace.resident_reduction.skipped", "F3"),
    ("4.4 Runtime", "5.18", "replace", f"{cm['processing_ms']:.2f}; {G(CO, 'bevplace', 'speedup', 'skipped'):.2f}×; {cm['resident_mb']:.2f}; 90.9%; {cm['kb_per_frame']:.2f}", CO, "main.skipped", "F3. 'smallest among the eleven methods' 유지(main_table_top3.skipped.kb_per_frame 1위)."),
    ("5 Conclusion", "90.9%", "na", "90.9% (불변)", CO, "bevplace.resident_reduction.skipped", "F3"),
    ("I Resource measurements", "3.22", "replace",
     f"{cm['update_ms']:.2f}; 1.96; {cm['processing_ms']:.2f}; seed-4; {cm['encode_ms']:.2f}/{cm['state_update_ms']:.2f}; 5.34e-6; 1.96e-6; 1.68→1.94 ms; 6.01→31.42 MB",
     "cost/nsd_n10000_window.json, cost/summary.json mapsize.skipped", "timings_ms, verification", "F3. 'the final scores of all 288 timed queries are equal' 문장은 새 측정에 직접 근거가 없어 초안 C11에 남김."),
    ("Observation-bound verification", "", "replace", "5 conditions, 1,242 checks, max score error 6.44e-6, independent metrics 5.47e-14", "manuscript/prefix_audit.json", "manuscript", "F3: 문장 대체(\\rev)"),
    ("Paired intervals", "", "replace", "'Each NSD seed sets both the initialization and the sample order.'", None, None, "F3: 문장 대체(\\rev)"),
    ("Learning across starts and cities.'", "DiffAttnConv learning AP gains", "replace", "+0.51, +0.90, +0.63; AP Stockholm −0.21, Hamburg −0.01; R@1 Hamburg −0.47, Ann Arbor −0.07", EX, "learning_per_seed_pp, learning_city_negative", "F3: 문단 대체(\\rev), 제목 'seeds'"),
    ("Learning across starts and cities.' (second", "", "replace", "+1.29~+1.67; Seoul +10.41, Daejeon +3.84, Karlsruhe +2.17, Ann Arbor +0.02, Stockholm −2.63, Hamburg −5.07", NR, "comparisons.diff-identity.all.city_delta_pp", "F3: 문장 대체(\\rev)"),
    ("D, same paragraph", "block resampling", "replace", "AP [−0.09, 3.22]; F1 [2.15, 4.89]; R@100P [3.56, 9.41] (방향 불변)", "analysis/main/refining_block_summary.json", "comparisons.diff-identity.all.block.overall", "F3: 새 계산 scripts/_report_model_refining_block_20261116.py"),
    ("Independent fine-tuning repetitions", "", "replace", "문단 삭제", None, None, "F3: 사용자 결정에 따라 삭제"),
    ("Full and DiffAttnConv learning by direction", "", "replace", "'Learning by direction.' 문단으로 대체", "analysis/main/direction_sequence_summary.json", "comparisons.diff-fixed", "F3: \\rev"),
    ("Contrast versus aggregation", "", "replace",
     "검증 79.21→69.84, 정제 없음 79.74, 균등 감산 83.34; 시험 88.07/86.25/65.50/92.31; −균등 감산 −1.13 [0.15,2.09] 등; −정제 없음 −0.36 [−0.48,−0.24] 등; 서울·대전 수치",
     "ablations/uniform_addition_summary.json", "comparisons, city_percent", "F3: 수치 교체. 방향이 바뀐 두 문장('does not show that aggregation lowers AP', 'all three in Stockholm')은 보류(초안 C9)."),
    ("Absolute key/value", "", "removed", "원고에 싣지 않음(사용자 결정 2026-11-17)", "controls/summary.json", "test.all.differences.abskv-nsd", "F3: main이 부록 문단 삭제, 표 행 제거. 결과는 보존"),
    ("Scope of illustrations, first", "", "replace", "B7 문장 적용", "analysis/learning/learning_figure.json", "definition", "F3: \\rev"),
    ("temporal_controls.tex", "", "replace", "80.69/80.97/83.34; 폐기 변형 서술 삭제; 552 checks 2.86e-6", "ablations/window_selection.json; controls/summary.json; manuscript/prefix_audit.json", "", "F3: temporal_source.json narratives 고침(\\rev)"),
]
for loc, quote, action, new, source, field, note in F3:
    hit = [r for r in rows if loc in r["location"] and quote in r["quote"]]
    assert hit, (loc, quote)
    for r in hit:
        r.update(action=action, new=new, field=field, note=(r["note"] + " | " if r["note"] else "") + note)
        if source:
            r["source"] = f"results/pdv2_report_model_20261116/{source}"

OUT.parent.mkdir(parents=True, exist_ok=True)
summary = {a: sum(r["action"] == a for r in rows) for a in ("replace", "hold", "pending", "na", "removed")}
OUT.write_text(json.dumps(dict(
    unit="F2 (원고 수치 전수 조사, 2026-11-17), F3 갱신(같은 날)", manuscript="docs/paper/body_v2.4/{cvpr2027_body.tex,cvpr2027_supplement_body.tex}",
    model="x=log__n=ln__g=frozen__lr=1e-3__U=1536@u512, 시드 4·5·6", rounding="소수 둘째 자리(원고 표기 자릿수 유지), 차이는 반올림 전 delta_pp를 반올림",
    actions=dict(replace="원고에 \\rev{}로 교체", hold="문장 보류(초안 docs/paper/body_v2.4/_draft_sentences_20261116.md)",
                 pending="결과 미완(그대로 둠)", na="수치·주장 불변", removed="원고에서 뺀 내용"),
    counts=summary, rows=rows), ensure_ascii=False, indent=1) + "\n")
print(summary)
for r in rows:
    print(f"[{r['action']:7}] {r['doc'][:6]} | {r['location'][:60]} | {r['current']} -> {r['new']}")
