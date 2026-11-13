#!/usr/bin/env python3
"""검산한 비교 방법 공통 진단을 부록 표와 설명으로 생성한다."""
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pdv2_baseline_common_diagnosis_20261106"
# root 검산(2026-11-08): SeqOT 저자 저장 예측을 공개 원시·저자 자세로 다시 만든 overlap>0.3 정답으로 다시 채점한 진단.
# 원본 결과는 덮지 않고 같은 해시의 사본과 root 승인 근거(root_approval.json)를 읽는다. 표2 SeqOT 모델과 무관하다.
SEQOT_OVERLAP = ROOT / "results/pdv2_fresco_seqot_integration_20261108"
# root 검산(2026-11-08, metric_residual_finish): 같은 638,913쌍·저장 예측에서 라벨 투영 원점만 라이다 위치로 평행이동한 상·하한(회전 미적용).
SEQOT_LIDAR = ROOT / "results/pdv2_seqot_lidar_origin_20261108"
ORCH = ROOT / "results/pdv2_reproduction_orchestration_20261106"
NATIVE = ROOT / "results/pdv2_remaining_native_20261106"
# root 검산(2026-11-08): RING++ T-RO 최종판 대조·지도 한 장 누락 경계, DiSCO arXiv v1 반경 진단, BEVPlace++ 공개 run 표본기,
# SC++ MulRan 재생의 산출물을 바이트 동일 사본으로 고정한다(source_pins.json에 원본 경로·해시).
REPRO = ROOT / "results/pdv2_reproduction_appendix_20261108"
# SC++ IEEE T-RO 게재본(사용자 제공, 내부 대조용)과 arXiv v1의 표·정의 대조 결과. 게재본 PDF는 두지 않고 대조 JSON만 해시로 고정한다.
SCPP_FINAL = ROOT / "results/pdv2_scpp_final_tro_20261108"
# root 검산(2026-11-08, scpp_claim_fidelity_audit): 표2 SC++는 공식 MATLAB A-PC 경로(scripts/scpp_official, k=1, 정렬 키,
# search_ratio 0)에 센서별 높이를 더한다. 저자 PR 그림 면적·표VIII/IX 대조와 주장 범위(claim_decisions.json)를 바이트 동일 사본으로 고정한다.
SCPP_AUDIT = ROOT / "results/pdv2_scpp_claim_fidelity_20261108"
PAPER =ROOT / "docs/paper/body_v2.3"
# 저자 결정(2026-11-07): ImLPR는 현재 비교군에서 뺀다(results/pdv2_baseline_revision_20261107/decision.json).
# 원천 결과는 그대로 두고, 부록 표·문장·원장 수는 남은 방법만으로 만든다.
EXCLUDED = {"imlpr": "ImLPR"}
# 저자 결정(2026-11-07): 표2의 RING++는 정확max 구현이다. 분해표의 RING++ 자리에는 같은 46원장의 정확max 행을 두고,
# 공개 커널 행은 가로선 아래 진단 행으로 옮긴다. 원천 CSV의 두 행 값은 그대로 쓴다.
# 사용자 승인(2026-11-10): 표2의 RING++는 정확max + I2 정규화다. 분해표의 RING++ 자리에는 채택 I2 46원장의 값을 두고(AP·R@1에서
# SeqOT 행과 같은 식으로 계산), 공식 정규화의 정확max 행과 공개 커널 행은 가로선 아래 진단 행으로 둔다.
MAIN_ROW = {"RING++ exact max": None, "RING++": None}
DIAGNOSTIC_ROW = {"RING++ exact max": "RING++ exact max", "RING++": "RING++ released kernel"}


def provenance_values(disco_run: dict) -> dict:
    """공개 구현 재생·입력/가중치 동일성 검산 결과에서 부록 문장 수치를 읽는다."""
    load = lambda path: json.loads((ORCH / path).read_text())
    assert load("identity_audit_20261107/ring_inputs/verification.json")["all_passed"]
    identity = load("identity_audit_20261107/disco_identity/verification.json")
    assert identity["all_passed"] and identity["public_equals_expected"] and identity["server_equals_expected"]
    assert load("ring/date_pair_20261107/verification.json")["passed"]
    ring = {(r["query_date"], r["method"]): 100 * r["radii"]["10.0"]["R@1_all_queries"]
            for r in load("ring/date_pair_20261107/report.json")["rows"] if r["split"] == "all"}
    vintage = load("disco_protocol/preprocess_vintage_20261107/report.json")
    assert vintage["all_checks_passed"]
    released = vintage["metrics"]["released"]
    assert abs(released["ego1m"]["r1_percent"] - disco_run["mean_r1_percent"]) < 1e-9
    bev = json.loads((NATIVE / "bevplace/report.json").read_text())
    log = json.loads((NATIVE / "bevplace/author_log_verification.json").read_text())
    scpp = json.loads((NATIVE / "scpp/report.json").read_text())
    assert bev["complete"] and log["passed"]
    order = ("00", "02", "05", "06", "08")
    assert all(bev["rows"][k]["official_function_agrees"] for k in order)
    pc, cc = scpp["variants"]["pc"], scpp["variants"]["cc"]
    assert pc["positive_queries"] == cc["positive_queries"] and pc["finite_and_past_candidate_checks"]
    log_max = max(r["difference_pp"] for r in log["rows"])
    assert log_max < 3e-6
    return {
        "RING_0820": f"{ring[('2012-08-20', 'RING')]:.2f}",
        "RINGPP_0820": f"{ring[('2012-08-20', 'RING++ 정확max')]:.2f}",
        "DISCO_R1": f"{released['ego1m']['r1_percent']:.2f}",
        "DISCO_R1P": f"{released['ego1m']['r1_percentile_percent']:.2f}",
        "DISCO_5M_R1": f"{released['ego5m']['r1_percent']:.2f}",
        "DISCO_5M_R1P": f"{released['ego5m']['r1_percentile_percent']:.2f}",
        "BEV_RUN": "/".join(f"{bev['rows'][k]['R1_percent']:.2f}" for k in order),
        "BEV_PAPER": "/".join(f"{bev['rows'][k]['paper_R1_percent']:.1f}" for k in order),
        "BEV_LOG": r"$3\times10^{-6}$",
        "SCPP_FRAMES": f"{scpp['execution']['frames']:,}",
        "SCPP_PC": f"{pc['pr_auc']:.3f}", "SCPP_CC": f"{cc['pr_auc']:.3f}",
        "SCPP_PC_PAPER": f"{scpp['paper']['pc_pr_auc']:.2f}",
        "SCPP_CC_PAPER": f"{scpp['paper']['cc_pr_auc']:.2f}",
        "SCPP_POS": str(pc["positive_queries"]), "SCPP_POS_PAPER": str(scpp["paper"]["positive_queries"]),
    }


def reproduction_values(disco_run: dict) -> dict:
    """root 검산을 마친 원 논문 재현 진단(RING++·DiSCO·BEVPlace++·SC++)의 문장 수치를 고정 사본에서 읽는다."""
    pins = json.loads((REPRO / "source_pins.json").read_text())
    for rel, entry in pins["files"].items():
        assert hashlib.sha256((REPRO / rel).read_bytes()).hexdigest() == entry["sha256"], rel
    load = lambda rel: json.loads((REPRO / rel).read_text())
    assert load("root/ring_tro_scope_closure.json")["passed"] and load("root/ring_tro_final_pdf.json")["passed"]
    assert load("root/native_ring_disco_audit.json")["all_ok"]
    assert load("root/native_ring_disco_continue.json")["all_numeric_file_checks_passed"]
    assert load("root/bev_scpp_ringsharp_final_audit.json")["passed"] and load("root/native_bev_scpp_ringsharp_continue.json")["passed"]
    assert load("bev/audit_verification.json")["all_passed"] and load("disco/v1_radius_verification_ledgers.json")["all_passed"]
    # RING++: 최종판 0.1 m 복셀 1회 재생(전체 궤적)과 지도 한 장 누락 경계. 복셀 유무 모두 같은 기준값이다.
    vox = load("ring/voxel_replay_verification.json")
    assert vox["all_passed"]
    full = vox["voxel_replay"]["all"]
    assert full["R@1_all_10m"] == full["baseline_no_voxel_R@1_all"]
    omission = load("ring/omission_bound.json")["conditions"]
    best = {k: max(omission[c]["single_omission"][f"R_{k}_pct_min_max"][1] for c in ("no_voxel", "voxel_0p1"))
            for k in ("all", "pos")}
    for c in ("no_voxel", "voxel_0p1"):
        base = omission[c]["baseline"]
        assert (base["n_query"], base["positives"]) == (full["queries"], full["positives"])
        assert abs(base["R_all"] - 100 * full["R@1_all_10m"]) < 1e-9 and abs(base["R_pos"] - 100 * full["R@1_pos_10m"]) < 1e-9
        assert not omission[c]["reach_paper"]["any_omission_ge_73.205_common"]
        assert not omission[c]["reach_paper"]["any_omission_ge_73.205_positive"]
    # DiSCO: 같은 서술자·30쌍 목록에서 판정 반경만 1.5 m(기준 재현)와 arXiv v1의 3 m로 바꾼 값.
    radius = load("disco/v1_radius_result.json")["released"]
    near, far = radius["R1.5"], radius["R3.0"]
    assert near["valid_pairs"] == far["valid_pairs"] == 30
    assert abs(near["mean_over_pairs"]["r1_percent"] - disco_run["mean_r1_percent"]) < 1e-9
    # BEVPlace++: 공개 epoch5·10 상태에서 이어 학습한 반복0 손실(같은 난수 재생 가정 H0)과 epoch6 평균 손실 3시드.
    bev = load("bev/report.json")
    replay = {r["anchor"]: r for r in bev["exact_replay"]["rows"] if r["branch"] == "H0"}
    assert bev["exact_replay"]["tolerance"] == 1e-3
    assert all(replay[a]["match_A"] and not replay[a]["match_B"] for a in ("best", "last"))
    t1 = bev["statistics"]["T1_E5_avgloss_epoch6"]
    assert t1["primary_class"] is None and t1["secondary_class"] == "A"
    losses = lambda key: "/".join(f"{replay[a][key]:.4f}" for a in ("best", "last"))
    span = lambda v, d: f"{min(v):.{d}f}--{max(v):.{d}f}"
    # SC++: MulRan 노드·재방문 개수(원시 자료 재계산)와 8 m PR-AUC, KITTI 단일 시간 창의 정확 구간.
    scpp = load("scpp/independent_scpp.json")
    nan_check = load("scpp/nan_convention_check.json")
    raw = scpp["mulran_raw"]
    for seq in ("KAIST03", "Riverside02"):
        assert raw[seq]["nodes"] == raw[seq]["paper_nodes"] and raw[seq]["positives_8m"] == raw[seq]["paper_positives"]
        for kind in ("pc", "cc"):
            ledger = scpp["ledgers"][f"{seq}_{kind}"]["r8"]
            assert ledger["auc_matlab_literal"] is None and ledger["auc_fill1"] == nan_check[f"{seq}_{kind}"]["fill1"]
    auc = lambda seq: "/".join(f"{scpp['ledgers'][f'{seq}_{k}']['r8']['auc_fill1']:.3f}" for k in ("pc", "cc"))
    window = {k: scpp["kitti_exact_dt"][f"{k}_camera_xz"]["seconds"]["strict_gt"] for k in ("00", "08")}
    assert window["00"][1] < window["08"][0]
    interval = lambda k: f"[{window[k][0]:.3f}, {window[k][1]:.3f})"
    final = load("scpp/final_version_check.json")
    assert "not the IEEE version of record" in final["comparison"]["conclusion"]
    # SC++ 게재본: 표 III·VIII·IX와 재방문·판정·표본 정의가 arXiv v1과 같다(편집상 차이만). 문장의 논문 값은 게재본 표에서 읽는다.
    scpp_pins = json.loads((SCPP_FINAL / "pins.json").read_text())
    for rel, entry in scpp_pins["files"].items():
        assert hashlib.sha256((SCPP_FINAL / rel).read_bytes()).hexdigest() == entry["sha256"], rel
    vor = json.loads((SCPP_FINAL / "sourcecomparison.json").read_text())
    link = json.loads((SCPP_FINAL / "audit_link.json").read_text())
    assert vor["all_passed"] and not vor["conclusion_affecting_changes"]
    assert link["all_passed"] and not link["conclusions"]["changed_by_final_version"]
    t3, t8 = vor["tables"]["Table III"]["rows"], vor["tables"]["Table VIII"]["rows"]
    t9 = dict(zip(vor["tables"]["Table IX"]["row_labels"], vor["tables"]["Table IX"]["rows"]))
    kitti = json.loads((NATIVE / "scpp/report.json").read_text())["paper"]
    assert [t8[0][0], t8[0][2], t3[0][1]] == [f"{kitti['pc_pr_auc']:.2f}", f"{kitti['cc_pr_auc']:.2f}", str(kitti["positive_queries"])]
    assert [t3[0][1], t3[1][1]] == [str(scpp["kitti_exact_dt"][f"{k}_camera_xz"]["target"]) for k in ("00", "08")]
    for seq, row in (("KAIST03", t3[2]), ("Riverside02", t3[3])):
        assert row[1:3] == [str(raw[seq]["positives_8m"]), str(raw[seq]["nodes"])]
    return {**residual_values(disco_run),
        "RING_MAPS": str(full["maps"]), "RING_QUERIES": f"{full['queries']:,}", "RING_POS": str(full["positives"]),
        "RINGPP_VOX": f"{100 * full['R@1_all_10m']:.2f}/{100 * full['R@1_pos_10m']:.2f}",
        "OMIT_ALL": f"{best['all']:.2f}", "OMIT_POS": f"{best['pos']:.2f}",
        "DISCO_3M": f"{far['mean_over_pairs']['r1_percent']:.2f}/{far['mean_over_pairs']['r1_percentile_percent']:.2f}",
        "DISCO_POS_15": f"{near['positive_queries_total']:,}", "DISCO_POS_3": f"{far['positive_queries_total']:,}",
        "BEV_A": losses("pred_A"), "BEV_B": losses("pred_B"), "BEV_AUTHOR": losses("author"),
        "BEV_T1_AUTHOR": f"{t1['author']:.5f}", "BEV_T1_A": span(t1["A"], 5), "BEV_T1_B": span(t1["B"], 4),
        "BEV_T1_GAP": f"{min(t1['A']) - t1['author']:.4f}",
        "SCPP_KAIST_NODES": f"{raw['KAIST03']['nodes']:,}", "SCPP_KAIST_POS": f"{raw['KAIST03']['positives_8m']:,}",
        "SCPP_RIVER_NODES": f"{raw['Riverside02']['nodes']:,}", "SCPP_RIVER_POS": f"{raw['Riverside02']['positives_8m']:,}",
        "SCPP_KAIST_AUC": auc("KAIST03"), "SCPP_RIVER_AUC": auc("Riverside02"),
        "SCPP_KAIST_PAPER": f"{t9['KAIST 03'][0]}/{t9['KAIST 03'][4]}",
        "SCPP_RIVER_PAPER": f"{t9['Riverside 02'][0]}/{t9['Riverside 02'][4]}",
        "SCPP_KAIST_CC_SKIP": f"{nan_check['KAIST03_cc']['skip_nan']:.3f}",
        "SCPP_DT00": interval("00"), "SCPP_DT08": interval("08"),
        "SCPP_KITTI08_POS": str(scpp["kitti_exact_dt"]["08_camera_xz"]["target"]),
        **scpp_claim_values(t8, t9), **scpp_height_values(), **ringsharp_values(),
    }


def scpp_claim_values(t8: list, t9: dict) -> dict:
    """표2 SC++ 높이 설정과 저자 PR 그림 면적 대조 값을 고정 사본과 저장소 원천에서 읽는다."""
    pins = json.loads((SCPP_AUDIT / "pins.json").read_text())
    for rel, entry in pins["files"].items():
        assert hashlib.sha256((SCPP_AUDIT / rel).read_bytes()).hexdigest() == entry["sha256"], rel
    for rel, digest in pins["repo_sources_sha256"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    root = json.loads((SCPP_AUDIT / "root_verification.json").read_text())
    assert root["passed"] and root["integration_authorized"]
    assert json.loads((SCPP_AUDIT / "verification.json").read_text())["passed"]
    decisions = {d["id"]: d["decision"] for d in json.loads((SCPP_AUDIT / "claim_decisions.json").read_text())["decisions"]}
    assert decisions["C4_top1_equal_implies_same_pipeline"] == "삭제" and decisions["C7_kitti08_allframes"] == "수정"
    # 센서별 높이: 실행 스크립트 값 = sensor_height.json 'lowest' 시퀀스 중앙값의 센서별 중앙값. NCLT는 측정 없이 공식 1.9 m.
    script = (ROOT / "scripts/scpp_official/run_pdv2_sensor_height.sh").read_text()
    heights = {k: float(v) for k, v in re.findall(r"\[(\w+)\]=([0-9.]+)", script)}
    measured = {}
    for key, entry in json.loads((ROOT / "results/pdv2_ring_official/sensor_height.json").read_text()).items():
        if "/" in key:
            measured.setdefault(key.split("/")[0], []).append(entry["lowest"]["height"]["median"])
    assert sorted(measured) == sorted(k for k in heights if k != "nclt") and heights["nclt"] == 1.9
    assert all(round(statistics.median(v), 3) == heights[k] for k, v in measured.items()) and len(measured["mcd"]) == 4
    # 저자 PR 그림 면적과 게재본 표: KITTI 00 CC만 표IX와 반올림 일치, 표VIII과 불일치. 나머지 PC/CC 곡선은 두 자리 반올림 일치.
    figures = json.loads((SCPP_AUDIT / "decode_audit_m2.json").read_text())["figures"]
    cc00 = figures["kitti00_prcurve"]["curves"]["CC"]
    assert cc00["table"] == {"VIII": float(t8[0][2]), "IX": float(t9["KITTI 00"][4])}
    assert cc00["round2_equals"] == {"VIII": False, "IX": True}
    others = [c for f in figures.values() for name, c in f["curves"].items() if c.get("table") and c is not cc00]
    assert len(others) == 7 and all(all(c["round2_equals"].values()) for c in others)
    return {"SCPP_HMIN": f"{min(heights.values()):.3f}", "SCPP_HMAX": f"{max(heights.values()):.3f}",
            "SCPP_CC00_AREA": f"{cc00['drawn_area']:.4f}", "SCPP_CC00_VIII": t8[0][2], "SCPP_CC00_IX": t9["KITTI 00"][4]}


def sampler_appendix() -> None:
    """검증이 끝난 두 SeqOT 양성 선택 대조를 별도 부록으로 쓴다."""
    base = ROOT / "results/pdv2_seqot_ring_repair_20261106"
    reports = {name: json.loads((base / name / "verification.json").read_text())
               for name in ("seqot", "seqot_paper")}
    for report in reports.values():
        assert report["passed"] and report["aggregate_recomputed"]
        assert report["negative_labels_checked"] and report["queries_per_arm"] == 99236
    rows = [("Selected checkpoint", reports["seqot"]["selected_original"])]
    for name, prefix in (("seqot", "Code"), ("seqot_paper", "Paper")):
        for arm, suffix in (("control", "control"), ("repair", "non-self")):
            rows.append((prefix + ": " + suffix, reports[name]["macro"][arm]))
    lines = [name + " & " + " & ".join(f"{100 * values[k]:.2f}" for k in
             ("AP", "max_F1", "R@100P", "R@1")) + r" \\" for name, values in rows]
    content = r"""% 원천: results/pdv2_seqot_ring_repair_20261106/{seqot,seqot_paper}/verification.json
\subsection{SeqOT positive-sampling controls}
\label{app:seqot_sampler}
We test whether self-only positive tuples explain SeqOT's weak shared-data training signal under the released-code recipe. Two paired diagnostics start from its selected seed-1 transformer and keep GeM fixed. Within each pair, the 4,096 anchors, negative partners, number of positives, update order, model, loss, and learning rate ($5\times10^{-7}$) are identical. The control retains the released positive selection. The alternative samples different scans from the full directed overlap-positive pool. Both evaluate all nine common validation sequences (99,236 queries), using final returned places and L2 acceptance scores with three-family macro averaging. No test data is used.

The first pair retains the released overlap formula. Of 15,458 original anchors, 8,036 have a valid non-self positive; 883 of the 4,096 selected control updates are self-only, versus none after resampling. The second pair uses the paper's Eq.~13 denominator, the minimum valid-pixel count of the anchor and reprojected reference. This supplies non-self positives for 12,739 anchors. Excluding anchors whose existing negatives conflict with that formula leaves 11,524, from which the paired budget is drawn. Its control has 2,055 self-only updates, versus none after resampling. The two pairs use different eligible-anchor sets, so only within-pair differences isolate the positive-selection intervention; their difference is not a denominator-only ablation.

\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{SeqOT sampling controls on validation (\%).}
\label{tab:seqot_sampler}
\begin{tabular}{@{}lrrrr@{}}\toprule
Condition & AP & Max F1 & R@100P & R@1 \\
\midrule
@ROWS@
\bottomrule\end{tabular}
\end{table}

Resampling restores nonzero positive-pair distances in every update. The code-formula pair improves AP by 1.94 points and R@1 by 0.40 points over its control. Against the selected checkpoint, however, AP improves by only 0.32 points and R@1 decreases by 0.11 points. The paper-formula pair improves R@1 by 0.31 points over its control and by 0.24 points over the selected checkpoint, while AP is 0.13 points below the selected checkpoint. Thus restoring positive-pair signal helps within these controls but does not establish a broad performance recovery. These single-seed short continuations do not establish convergence, and neither is used in Table~\ref{tab:results}.
"""
    (PAPER / "baseline_sampler_controls.tex").write_text(content.replace("@ROWS@", "\n".join(lines)))


def seqot_overlap_values(report: dict) -> dict:
    """SeqOT overlap 정답 재생성 진단의 증명 상한·좌표 합성 검사 값을 root 승인 사본에서 읽는다.

    모든 질의의 저장 후보 22개와 양성 없는 질의의 50 m 안 지도를 모두 계산했고, 제외 질의에 알려진 적중이 없을 때만
    값을 상한으로 쓴다. 공개 15 m 정답 결과와 논문 수치는 표S13과 같은 원천인지 확인한다.
    """
    approval = json.loads((SEQOT_OVERLAP / "root_approval.json").read_text())
    for rel in ("seqot/recount.json", "seqot/coordfix_result.json", "seqot/geom_check.json", "seqot/overlap_result.json"):
        assert hashlib.sha256((SEQOT_OVERLAP / rel).read_bytes()).hexdigest() == approval["files"][rel]["sha256"], rel
    load = lambda rel: json.loads((SEQOT_OVERLAP / rel).read_text())
    recount, fix, geom, result = (load(f"seqot/{n}.json") for n in ("recount", "coordfix_result", "geom_check", "overlap_result"))
    assert recount["all_22_candidates_computed"] and fix["all_22_candidates_computed"] and recount["candidate_index0_occurrences"] == 0
    ks = ("1", "5", "20")
    for name in ("A", "B"):
        s = recount["score"][name]["all_kinds"]
        assert s["excluded_queries_exhaustive_50m"] and all(s[k]["excluded_with_known_topN_hit"] == 0 for k in ks)
        assert all(s[k]["recall"] == result["recall"]["overlap_" + ("A_official_formula" if name == "A" else "B_paper_eq13")][k] for k in ks)
    for name in ("A_fix", "B_fix"):
        f = fix[name]
        assert f["excluded_queries_exhaustive_50m"] and f["official_equals_independent"]
        assert all(f["bounds"][k]["excluded_with_known_hit"] == 0 and f["bounds"][k]["upper_proven"] == f["recall_official_cal_topN"][k] for k in ks)
    syn, real = geom["synthetic"], geom["real"]["summary"]
    assert syn["flip_consistent_median_err_m"]["max"] < 1e-12 and syn["det_P"] == 1.0 and syn["det_F"] == -1.0
    pairs = [real[g]["pairs"] for g in ("top_rotation_discrepancy", "random")]
    assert all(real[g]["flip_consistent_better_pairs"] == real[g]["pairs"] for g in ("top_rotation_discrepancy", "random"))
    seqot = report["seqot"]
    assert all(abs(result["paper_table2_2012_02_05"][k] - seqot["paper_reference"]["r" + k]) < 1e-12 for k in ks)
    assert all(abs(result["recall"]["released_15m"][k] - seqot["evaluation"]["released"]["recall"][k]) < 1e-12 for k in ks)
    triple = lambda r, d=2: "/".join(f"{100 * r[k]:.{d}f}" for k in ks)
    return {
        "SEQOT_CANDIDATES": "22",
        "SEQOT_A": triple(result["recall"]["overlap_A_official_formula"]),
        "SEQOT_B": triple(result["recall"]["overlap_B_paper_eq13"]),
        "SEQOT_AFIX": triple(fix["A_fix"]["recall_official_cal_topN"]),
        "SEQOT_BFIX": triple(fix["B_fix"]["recall_official_cal_topN"]),
        "SEQOT_PAPER": triple(result["paper_table2_2012_02_05"], 1),
        "SEQOT_15M": triple(result["recall"]["released_15m"]),
        "SEQOT_SYN": str(syn["cases"]), "SEQOT_SYN_ERR": f"{syn['official_median_err_m']['median']:.2f}",
        "SEQOT_REAL": str(sum(pairs)),
        **seqot_lidar_values(fix, result),
    }


def seqot_lidar_values(fix: dict, result: dict) -> dict:
    """라이다 위치 평행이동 진단의 상·하한과 입력 대조 범위를 root 검산 사본에서 읽는다.

    몸체 원점 두 정의가 기존 좌표 합성 수정 진단(coordfix)을 재현해야 같은 쌍·예측의 라벨 정의 비교로 쓴다.
    상한 h_N/K, 하한 h_N/5,648(양성이 확인되지 않은 질의를 모두 실패로 센다)이며 정확한 재현율이 아니다.
    """
    pins = json.loads((SEQOT_LIDAR / "pins.json").read_text())
    for rel, entry in pins["files"].items():
        assert hashlib.sha256((SEQOT_LIDAR / rel).read_bytes()).hexdigest() == entry["sha256"], rel
    for rel, digest in pins["repo_sources_sha256"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    load = lambda rel: json.loads((SEQOT_LIDAR / rel).read_text())
    root, full, protocol, physical = (load(n + ".json") for n in
                                      ("root_verification", "full_result", "protocol_full_lidar_origin", "origin_physical_check"))
    assert root["passed"] and root["integration_authorized"] and not root["main_table_change"] and full["gates_passed"]
    assert load("root_rerun_result.json")["all_pass"] and load("recount_bounds.json")["all_equal_full_result_and_root"]
    pairs = full["gates"]["F0_rows_equal_frozen_pairs"]["rows"]
    assert pairs == root["pairs"] == 638913 and full["inputs_sha256"]["predicted_L2_dis.npz"] == pins["repo_sources_sha256"][
        "results/pdv2_review_native_20261105/seqot/author/native/predicted_L2_dis.npz"]
    ks = ("1", "5", "20")
    defs = full["definitions"]
    for name, ref in (("A_axle", "A_fix"), ("B_axle", "B_fix")):
        assert all(defs[name]["rigorous"]["R"][k]["upper"] == fix[ref]["recall_official_cal_topN"][k] for k in ks), name
    db, query = full["gates"]["F3_B1_all_pairs"]["db_pack_equal"], full["gates"]["F3_B1_all_pairs"]["query_pack_equal"]
    assert len(set(db)) == 1 and len(set(query)) == 1
    # 원점 O = F x_body_vel 평행이동, 회전 미적용. 보고 고도 범위는 표본 스캔 전체의 최소·최대.
    origin = protocol["source_facts_before_run"]["origin"]
    offset = [float(v) for v in re.search(r"\(([-+0-9.]+), ([-+0-9.]+), ([-+0-9.]+)\)", origin).groups()]
    assert "rotation part" in origin and "NOT applied" in origin
    span = lambda key: [round(f(e[i] for e in physical["per_candidate"][key]["elev_min_max_deg"]))
                        for i, f in ((0, min), (1, max))]
    lidar, body = span("lidar_flipped_(0.002,-0.004,+0.957)"), span("axle_(0,0,0)")
    up, down = re.match(r"\+([0-9.]+)/-([0-9.]+) deg", protocol["fixed"]["FOV_image_range"]).groups()
    deg = lambda s: f"${s[0]:+d}^\\circ$ to ${s[1]:+d}^\\circ$"
    bound = lambda name, side: "/".join(f"{100 * defs[name]['rigorous']['R'][k][side]:.2f}" for k in ks)
    for name in ("A_lidar", "B_lidar"):
        assert all(abs(100 * defs[name]["rigorous"]["R"][k]["upper"] - root["upper_percent"][name][k]) < 1e-9 for k in ks)
        assert all(abs(100 * defs[name]["rigorous"]["R"][k]["lower"] - root["lower_percent"][name][k]) < 1e-9 for k in ks)
        assert all(defs[name]["rigorous"]["R"][k]["upper"] < result["paper_table2_2012_02_05"][k] for k in ks)
    return {
        "SEQOT_LQ": f"{query[0]:,}", "SEQOT_LM": f"{db[0]:,}", "SEQOT_LZ": f"{sum(v * v for v in offset) ** 0.5:.3f}",
        "SEQOT_LFILES": "one two three four five six seven eight nine".split()[len(physical["files"]) - 1],
        "SEQOT_LEL": deg(lidar), "SEQOT_BEL": deg(body), "SEQOT_FOV": f"$-{down}^\\circ$ to $+{up}^\\circ$",
        "SEQOT_PAIRS": f"{pairs:,}",
        "SEQOT_LA_UP": bound("A_lidar", "upper"), "SEQOT_LB_UP": bound("B_lidar", "upper"),
        "SEQOT_LA_LO": bound("A_lidar", "lower"), "SEQOT_LB_LO": bound("B_lidar", "lower"),
    }


def published_appendix() -> None:
    """원 논문 수치·공개 구현·공통 규약의 차이를 구분해 쓴다."""
    path = ROOT / "results/pdv2_published_reproduction_20261106/verified_published_comparison.json"
    report = json.loads(path.read_text())
    assert report["passed"]
    entries = []
    for r in report["helios"]["comparison"]:
        condition = r["database"].replace("Roundabout01-", "") + " / " + r["query_group"]
        entries.append(("HeLiOS", condition, r["metric"].replace("r", "R@"),
                        r["paper_reference_percent"], r["released_run_percent"]))
    for k in (1, 5, 20):
        r = report["seqot"]
        entries.append(("SeqOT", "NCLT Jan08--Feb05", f"R@{k}",
                        100 * r["paper_reference"][f"r{k}"],
                        100 * r["evaluation"]["released"]["recall"][str(k)]))
    disco = json.loads((ORCH / "disco_protocol/report.json").read_text())
    assert json.loads((ORCH / "disco_protocol/verification.json").read_text())["passed"]
    disco_run = disco["metrics"]["primary_released_sampler"]
    assert disco_run["complete_30_pair_evaluation"]
    for metric, name in [("r1_percent", "R@1"), ("r1_percentile_percent", r"R@1\%")]:
        entries.append(("DiSCO", "NCLT 30 pairs / 1.5 m", name,
                        disco["paper"][metric], disco_run["mean_" + metric]))
    for r in report["ring"]["rows"]:
        if r["variant"] == "released" and r["region"] == "all":
            entries.append(("RING" if r["model"] == "ring" else "RING++", "NCLT full / 10 m", "R@1",
                            100 * r["reference_R@1"], 100 * r["R@1"]))
    lines, previous = [], None
    for method, condition, metric, reference, measured in entries:
        if previous is not None and previous != method:
            lines.append(r"\midrule")
        delta = measured - reference
        delta_text = "0.00" if abs(delta) < 0.005 else f"{delta:+.2f}"
        lines.append(f"{method} & {condition} & {metric} & {reference:.2f} & {measured:.2f} & {delta_text}" + r" \\")
        previous = method
    values = provenance_values(disco_run)
    values.update(seqot_overlap_values(report))
    values.update(reproduction_values(disco_run))
    text = r"""% 원천: results/pdv2_published_reproduction_20261106/verified_published_comparison.json
% DiSCO·RING 동일성: results/pdv2_reproduction_orchestration_20261106/{disco_protocol,disco_protocol/preprocess_vintage_20261107,ring/date_pair_20261107,identity_audit_20261107/{ring_inputs,disco_identity}}/
% BEVPlace++·SC++: results/pdv2_remaining_native_20261106/{bevplace,scpp}/
% SeqOT overlap 정답 진단: results/pdv2_fresco_seqot_integration_20261108/{root_approval.json,seqot/}; 라이다 위치 평행이동 상·하한: results/pdv2_seqot_lidar_origin_20261108/pins.json
% RING++ 최종판·DiSCO v1 반경·BEVPlace++ 표본기·SC++ MulRan: results/pdv2_reproduction_appendix_20261108/source_pins.json
% SC++ 게재본(T-RO 38(3)) 표·정의 대조: results/pdv2_scpp_final_tro_20261108/pins.json; 표2 구현(A-PC·센서별 높이)·저자 PR 그림 면적: results/pdv2_scpp_claim_fidelity_20261108/pins.json
% RING 각도 격자(2π 제외) 단일 요인: results/pdv2_review_native_20261105/ring_angles_endpoint_false/{comparison,verification,evaluation_sets}.json
\subsection{Published-number and protocol audit}
\label{app:published_reproduction}
Table~\ref{tab:published_number_check} compares published references with our released-weight or released-code runs. We distinguish a faithful execution of a released implementation or checkpoint from confirmation that the paper used the same inputs and weights. HeLiOS uses the released benchmark inputs and reproduces all eight reported values within 0.07 percentage points. The other rows are numerical checks with the input, label, or checkpoint differences below; closeness alone does not establish an identical-protocol reproduction. The native checkpoints are used for reproduction checks; the common benchmark retains shared-data training and its fixed settings.

\begin{table*}[t]
\centering\small\setlength{\tabcolsep}{5pt}
\caption{Published values and released-model checks (\%).}
\label{tab:published_number_check}
\begin{tabular}{@{}lllrrr@{}}\toprule
Method & Evaluation & Metric & Published & Our run & Difference (points) \\
\midrule
@ROWS@
\bottomrule
\end{tabular}
\end{table*}

\paragraph{SeqOT labels.}
The paper defines a correct reference by overlap above 0.3, whereas the released ground-truth generator uses a 3D distance below 15\,m. We retain the downloadable labels in the native diagnostic, as directed by the authors, rather than substitute our common 5\,m rule. Using the authors' released pose arrays, we regenerate all 5,648 query label lists. The downloaded labels exactly equal the 15\,m lists after removing pairs with equal raw frame indices: 129 lists lose one pair each. The two sequences are from different dates, so equal indices do not denote the same scan. The current released generator does not contain that exclusion. Including these pairs changes the author's R@1 from 92.1664\% to 92.2039\% (two additional correct queries out of 5,336); R@5 and R@20 are unchanged. The original downloadable-label result remains the reference run. This verifies the distance-based nature of the release, not equivalence to the paper's overlap labels. The release and the paper therefore cannot be treated as interchangeable ground truth.\footnote{\href{https://github.com/BIT-MJY/SeqOT/blob/6e10697a4e6a423b5581e6a6ed6df9cfb386658b/data_prepararion/gen_ground_truth.py}{Released SeqOT label generator}; \href{https://github.com/BIT-MJY/SeqOT/issues/1\#issuecomment-1342137823}{author guidance on released ground truth}.}

\paragraph{SeqOT overlap labels.}
To test the paper's criterion, we regenerate overlap labels for the Feb05 queries against the Jan08 map from the public raw scans and the authors' poses, and rescore the authors' stored predictions with the unchanged official recall function. The authors' poses equal the official NCLT ground truth after an axis mapping that swaps $x$ and $y$ and negates $z$; they map body-frame points, with $z$ pointing down, to a $z$-up world. The paper's Eq.~13 defines overlap for training samples without a value for the range tolerance $\delta$, and its evaluation states only an overlap above 0.3. We therefore apply two readings with $\delta=1.2$\,m from the released code: the released overlap code counts reprojected-reference pixels whose range differs from the query by less than $\delta$ and divides by the valid query pixels, whereas a literal Eq.~13 counts pixels valid in both images with a difference of at most $\delta$ and divides by the smaller valid count. All @SEQOT_CANDIDATES@ stored candidates of every query are labeled, as are all map scans within 50\,m of queries without a positive. Positives outside this region can only add queries without a correct candidate, so the recalls below are upper bounds rather than exact values. With the released code, R@1/5/20 is at most @SEQOT_A@\%, or @SEQOT_B@\% with the literal Eq.~13. The released code applies these poses to $z$-flipped points, which is exact only when the relative motion is a pure yaw rotation without a height change; with $F=\mathrm{diag}(1,1,-1)$, the consistent composition is $F\,T_q^{-1}T_{db}\,F$. On @SEQOT_SYN@ synthetic pose pairs, the released composition has a median point error of @SEQOT_SYN_ERR@\,m, whereas the consistent composition is exact; it also aligns same-place scans more closely in all @SEQOT_REAL@ checked real pairs. Correcting only this composition, fixed before rescoring, gives upper bounds of @SEQOT_AFIX@\% (@SEQOT_BFIX@\% with the literal Eq.~13). These bounds remain below the published @SEQOT_PAPER@\%, whereas the released 15\,m labels give @SEQOT_15M@\%. The released artifacts do not identify the labels or weights behind the published table. This diagnostic involves no retraining and does not change Table~\ref{tab:results}, whose SeqOT models are trained without NCLT data. For the @SEQOT_LQ@ query and @SEQOT_LM@ map scans indexed by these labels, the stored model inputs equal the label-side projections, both from the vehicle body origin @SEQOT_LZ@\,m from the lidar; frames used only in the input windows are not checked. In @SEQOT_LFILES@ sampled scans, beam elevations span about @SEQOT_LEL@ from the lidar position, close to the configured @SEQOT_FOV@, but @SEQOT_BEL@ from the body origin. Translating the projection origin of both scans to the calibrated lidar position, without the calibration rotation, with the same @SEQOT_PAIRS@ pairs, stored predictions, $\delta$, and threshold, gives R@1/5/20 upper bounds of @SEQOT_LA_UP@\% (released overlap ratio) and @SEQOT_LB_UP@\% (literal Eq.~13), still below the published values; counting queries without a labeled positive as misses gives lower bounds of @SEQOT_LA_LO@\% and @SEQOT_LB_LO@\%. Because the predictions still come from body-origin inputs, only the label definition changes. This translation-only diagnostic is not a full sensor calibration; it shows that the labels are sensitive to the projection origin and does not identify those behind the published table.

\paragraph{RING evaluation scope.}
We follow the published T-RO version of RING++. It specifies 20\,m map spacing, 5\,m query spacing, and a revisit threshold of half the map spacing (10\,m). Its NCLT table states neither the session pair nor the evaluation region. We use the 2012-02-04 map and 2012-03-17 query sessions, for which the paper's resolution ablation reports the same 73.21\% for RING++, and the entire trajectory, which its runtime analysis describes as covered by 253 map scans. Neither condition is stated for that table itself. For this setting, the official generator yields @RING_MAPS@ map scans and @RING_QUERIES@ queries. Over all queries, RING reaches 62.52\%, and RING++ reaches 62.24\% with the exact maximum and the released normalization (55.60\% with the released kernel in Table~\ref{tab:published_number_check}). Over the @RING_POS@ queries with a map scan within 10\,m, RING and exact-maximum RING++ reach 66.43\% and 66.13\%. The paper defines the denominator only as total positives, so we report both. The published version adds a 0.1\,m voxel downsampling before neighbor search, which the released code omits; applying it once, fixed in advance, leaves both exact-maximum RING++ values unchanged at @RINGPP_VOX@\%. Holding all other scans and scores fixed, removing any single map scan yields at most @OMIT_ALL@\% (@OMIT_POS@\% over queries with a positive), so the one-scan difference in map count cannot account for the gap. The released default test region gives 62.55\% for RING and 61.00\% for exact-maximum RING++. With 2012-08-20 instead of 2012-03-17 queries against the same map, full-trajectory R@1 is @RING_0820@\% and @RINGPP_0820@\%, respectively. Neither alternative closes the gap to the published 70.98\% and 73.21\%.

Neither method uses learned weights. Our evaluation-set generator and NCLT loader are identical to the released generator and loader,\footnote{\href{https://github.com/lus6-Jenny/RING/blob/c79c58584ad841629506b5873ed1877c9c278ce5/evaluation/generate_evaluation_sets.py}{Official RING evaluation-set generator}.} and an independent reconstruction from this deterministic generator reproduces our map and query lists in order, frame identity, and position, together with their 10\,m positive counts. The authors' query and map lists, including their sampling phase, are not public, nor are the order of voxelization relative to ground removal and cropping and the voxel implementation. These factors remain untested. For NCLT, the released loader keeps points more than 1\,m above the wheel axle of the Segway body frame, @RING_CUT_LIDAR@\,m above the lidar origin; in all @RING_SCANS@ evaluated scans, this removes exactly the returns of the eight downward-looking beams. The square $\pm$70\,m crop and an angle grid containing both 0 and $2\pi$ depart from the rotation-invariance assumptions. Adding only a 70\,m circular mask to the released pipeline, whose unmasked run reproduces the released distances bitwise, changes all-query R@1 from 62.52\% to @RING_CIRC@\% for RING and from 62.24\% to @RINGPP_CIRC@\% for exact-maximum RING++, far less than the gaps. Replacing the angle grid with one that excludes $2\pi$ changes all-query R@1 to @RING_GRID@\% for RING and @RINGPP_GRID@\% for exact-maximum RING++, also far less than the gaps; the effects of rasterization and other causes are not quantified or excluded.

\paragraph{DiSCO released checkpoint.}
We evaluate the released checkpoint on the paper's NCLT protocol of six sessions, all 30 ordered database/query pairs, and a 1.5\,m match radius, using the official preprocessing, list sampler, inference, and recall functions. Independent L2 recomputation reproduces the top-1 indices, top-25 order, and recall curve of every pair. The 1.5\,m radius, stated in the published version and used by the current official generator, defines our reference reproduction. The first arXiv version reports the same table values but states a 3\,m radius.\footnote{\href{https://arxiv.org/abs/2010.10949v1}{DiSCO arXiv v1}; \href{https://arxiv.org/abs/2010.10949v2}{arXiv v2}.} With the same descriptors and lists, a 3\,m radius gives R@1/R@1\% of @DISCO_3M@\%. The radius also changes the set of queries with a positive (@DISCO_POS_15@ at 1.5\,m and @DISCO_POS_3@ at 3\,m), so the two results are not computed on the same queries. We therefore treat the 3\,m result as a diagnostic of the text discrepancy; the conditions that produced the table in either version cannot be determined from the released artifacts. The checkpoint is byte-identical to the file currently linked from the official README. After the paper's publication, the authors stated that they were training a new model and would update the link, which was subsequently replaced.\footnote{\href{https://github.com/MaverickPeter/DiSCO-pytorch/issues/3\#issuecomment-1065178609}{Author comment on the released model}; \href{https://github.com/MaverickPeter/DiSCO-pytorch/commit/8e79286f05f912e1c4596aacf440f56e68cc4e31}{README link update}.} Whether this file is the checkpoint behind the published table is therefore unverified, as is the identity of our regenerated lists with the authors' evaluation lists, which were not found in the available author artifacts. With the same checkpoint, pairs, and labels, restoring the 5\,m ego-vehicle exclusion used by the preprocessing code at the checkpoint upload, instead of the current 1\,m exclusion, gives R@1/R@1\% of @DISCO_5M_R1@/@DISCO_5M_R1P@\% versus @DISCO_R1@/@DISCO_R1P@\%. Before commit d96fe3e (2021-11-29), the public generator rescaled each scan by its per-axis maxima and stored heights, not binary occupancy; we use the fixed input matching the paper's definition, but the training input of either checkpoint is unknown. Keeping every second database sample changes mean R@1 from @DISCO_R1@\% to @DISCO_THIN@\% (two phases) and reduces the queries with a positive from @DISCO_POS_15@ to @DISCO_THIN_POS@; of these gains, @DISCO_THIN_SAME@ points arise on queries that keep a positive and @DISCO_THIN_POP@ from the changed query population. Replacing only the 2012-03-17 database list (@DISCO_CUR_N@ scans) with that of the first public sampler (@DISCO_FP_N@ scans), with queries, weights, and the 1.5\,m radius fixed, raises mean R@1 over its five pairs from @DISCO_CUR_R1@\% to @DISCO_FP_R1@\% (@DISCO_CUR_POS@ and @DISCO_FP_POS@ positive queries) and from @DISCO_INT_CUR@\% to @DISCO_INT_FP@\% on the @DISCO_INT_N@ queries positive under both lists. Both lists end within the first @DISCO_FRAC@\% of the session and differ in spacing and sampling phase. These fixed-checkpoint diagnostics show a material sensitivity of R@1 to list construction; they are not model improvements, do not reproduce the 30-pair value or establish the cause of its difference from the published value, and do not identify the authors' lists.

\paragraph{BEVPlace++ released checkpoint.}
The released BEVPlace++ KITTI checkpoint, evaluated with the official inference and evaluation functions and 5\,m labels, gives R@1 of @BEV_RUN@\% on KITTI 00/02/05/06/08, compared with the published @BEV_PAPER@\%. These values equal the authors' released training log for the same checkpoint within @BEV_LOG@ percentage points; identity with the paper's final weights is unverified. The released training run is consistent with the repository's initial hardest-positive sampler (commit 4ad02e31) rather than with the random-positive sampler of the current code, which our main-table training follows. Continuing from the released epoch-5 and epoch-10 weights and optimizer states under a reconstructed random-number sequence, the first-iteration losses are @BEV_A@ with the initial sampler and @BEV_B@ with the current sampler, against the logged @BEV_AUTHOR@; only the initial sampler meets the tolerance of $10^{-3}$ fixed in advance. The logged epoch-mean loss of @BEV_T1_AUTHOR@ lies @BEV_T1_GAP@ below the three-seed range of the initial sampler (@BEV_T1_A@), whereas the current sampler gives @BEV_T1_B@. The pre-registered primary criterion is therefore not met, and its auxiliary criterion favors the initial sampler. These forward-pass checks do not establish the identity of the original code files or of the full training procedure. Table~\ref{tab:results} retains the current-code training. The paper states no positive-selection rule, and its recipe (AdamW, 50 epochs, step decay, mining after 10 epochs) matches neither code version nor the released run, so neither sampler restores it. The KITTI gaps above concern the released checkpoint, not our training.

\paragraph{Scan Context++ released code.}
The official Scan Context++ PC and CC implementations on all @SCPP_FRAMES@ KITTI 00 scans with the paper's 8\,m criterion give PR-AUC of @SCPP_PC@ and @SCPP_CC@, compared with the published @SCPP_PC_PAPER@ and @SCPP_CC_PAPER@. This evaluation has @SCPP_POS@ positive queries rather than the paper's @SCPP_POS_PAPER@, so its query list is not identical. With the public equidistant keyframe selector and the evaluator's revisit rule, MulRan KAIST03 has @SCPP_KAIST_NODES@ nodes and @SCPP_KAIST_POS@ revisits, and Riverside02 has @SCPP_RIVER_NODES@ and @SCPP_RIVER_POS@, equal to the paper's counts; equal counts do not establish identical lists. The official PC/CC code then gives 8\,m PR-AUCs of @SCPP_KAIST_AUC@ on KAIST03 and @SCPP_RIVER_AUC@ on Riverside02, against the published @SCPP_KAIST_PAPER@ and @SCPP_RIVER_PAPER@.\footnote{The released PR aggregation leaves precision undefined (0/0) at thresholds without acceptances, so its AUC is undefined. We set this precision to 1, consistent with its starting point at zero recall; excluding these thresholds instead gives @SCPP_KAIST_CC_SKIP@ for KAIST03 CC.} Under the paper's time-window revisit definition, with past nodes within 8\,m and a single time gap, the reported @SCPP_POS_PAPER@ KITTI 00 and @SCPP_KITTI08_POS@ KITTI 08 revisits require gaps in @SCPP_DT00@\,s and @SCPP_DT08@\,s, respectively. No single gap produces both, so we could not regenerate the KITTI revisit lists within the definitions we tested. The PR curves in the authors' public arXiv source are vector figures. Their drawn areas agree with Tables VIII and IX to two decimals, except that the KITTI 00 CC area of @SCPP_CC00_AREA@ rounds to the @SCPP_CC00_IX@ of Table IX but not to the @SCPP_CC00_VIII@ of Table VIII; this agreement does not identify the run behind the tables. Under a float32 coordinate model consistent with every decoded vertex, the frame count alone fixes the query counts of the KITTI 00 curves and of the KITTI 08 PC curve. The MulRan query counts are unique only under an assumed keyframe-count bound (KAIST 03) or a query set shared across curves (Riverside 02), and recall denominators are determined only up to integer multiples. These constraints do not identify the authors' query lists or evaluation. Running the released PC and CC functions once on all @SCPP_K08_FRAMES@ KITTI 08 frames, with the evaluator's 49-frame exclusion and the 8\,m criterion, gives @SCPP_K08_Q@ queries (@SCPP_K08_QFAST@ with the released fast evaluator, which omits the last frame) rather than the @SCPP_K08_QFIG@ that this model fixes for the PC curve, and @SCPP_K08_POS@ revisits. Its PR-AUCs are @SCPP_K08_PC@ for PC, or @SCPP_K08_PC_SKIP@ when thresholds without acceptances are excluded, and @SCPP_K08_CC@ for CC, against the published @SCPP_K08_PAPER@; the 4, 2, and 1\,m values also differ from the published ones. Agreement of the rounded PC value thus depends on the AUC convention, and this single run neither reproduces the KITTI 08 results nor identifies the authors' run. Published Scan Context++ values and definitions are those of its journal version (Tables III, VIII, and IX).

\paragraph{RING\#-L released weights and recipe.}
No RING\#-L author checkpoint was found in the official sources we examined, so a released-weight check is not possible. We instead trained RING\#-L once with the released code and configuration on NCLT Protocol~1 (seed @RS_SEED@, @RS_EPOCHS@ epochs) and evaluated only the final-epoch checkpoint, fixed before training, with the released evaluation code. Our generated inputs have the paper's counts of @RS_TRAIN@ training samples and @RS_TEST@ test queries, which does not establish that they are the authors' lists. The released configuration uses a learning rate of @RS_LR@ and a weight decay of @RS_WD@, whereas the paper text states @RS_PAPER_LR@ and @RS_PAPER_WD@. At 10\,m, Recall@1, maximum F1, and PR-AUC are @RS_RUN@, against the published @RS_PAPER@, so the published values are not reproduced. For this run, the 100 thresholds of the official PR computation yield only @RS_SETS@ distinct acceptance sets, and maximum F1 is reached by accepting all @RS_TEST@ queries. Without the authors' weights or training runs, we cannot identify the cause of these differences.

\paragraph{Common-task adaptations.}
Geographically disjoint retraining, KISS-ICP keyframes, past candidates, and the shared 5\,m correctness rule define the common task. The 5--25\,m query exclusion and fixed positive denominator are applied to all methods. These choices support a shared loop-detection comparison, but are not equivalent to each paper's original task and need not affect methods equally. The main results must be interpreted under this specified task. Original method-specific training targets, including overlap or 10\,m positives, are retained.

\paragraph{Sensor and implementation adaptations.}
Where a release lacks a sensor configuration, projection bounds, beam indexing, and intensity scales are assigned from the sensor/file format; their effect is not assumed neutral. SeqOT keeps the released centered windows, available within the common decision support. Its released training-overlap denominator counts valid anchor pixels; the paper's Eq.~13 instead uses the minimum valid-pixel count of the two projected scans. Table~\ref{tab:results} uses Eq.~13; the released computation is kept for the code-recipe diagnostics. Skipping empty SeqOT overlap bins makes its sampler executable on the new data, but changes the number of available training tuples. HeLiOS empty-miner guards affect logged statistics; they avoid failures without adding a new learning objective. Scan Context++ in Table~\ref{tab:results} calls the released MATLAB descriptor, key, alignment, and distance functions in the A-PC configuration: the ring keys of each map scan and of its two copies with $\pm$2\,m lateral root shifts share one k-d tree, one candidate is retrieved, and the sector-key shift is used without further search. The released code adds a constant 1.9\,m to point heights for all datasets; we instead add sensor-specific height offsets of @SCPP_HMIN@--@SCPP_HMAX@\,m, keeping 1.9\,m for NCLT. The other offsets are medians of ground-plane fits to sampled scans, computed from point geometry only, on training or validation sequences; MCD, which has neither, uses four of its test sequences. A sensitivity run fixed before evaluation replaces only these offsets with the released 1.9\,m on all 46 test sequences, reusing the nine NCLT ledgers, and changes the macro AP, max F1, R@100P, and R@1 of Table~\ref{tab:results} from @SCPP_H19_CUR@\% to @SCPP_H19@\%; the table keeps the sensor-specific offsets. These macro differences are within @SCPP_H19_MAX@ points, whereas R@100P of individual sequences changes by @SCPP_H19_SEQ@ points, and we computed no significance test or interval across sequences.
"""
    text = text.replace("@ROWS@", "\n".join(lines))
    for key, value in values.items():
        text = text.replace("@" + key + "@", value)
    assert not re.search(r"@[A-Z0-9_]+@", text) and "ImLPR" not in text
    (PAPER / "baseline_published_audit.tex").write_text(text)


def residual_values(disco_run: dict) -> dict:
    """root 검산을 마친 RING NCLT 높이 컷·DiSCO 목록 간격 진단의 문장 수치를 고정 사본에서 읽는다."""
    # root 검산(2026-11-08): pipeline_residual_claim_audit. 원본은 덮지 않고 바이트 동일 사본을 해시로 고정한다.
    base = ROOT / "results/pdv2_pipeline_residual_appendix_20261108"
    pins = json.loads((base / "source_pins.json").read_text())
    for rel, entry in pins["files"].items():
        assert hashlib.sha256((base / rel).read_bytes()).hexdigest() == entry["sha256"], rel
    load = lambda rel: json.loads((base / rel).read_text())
    root = load("root/pipeline_residual_claim_audit.json")
    assert root["passed"] and root["integration_authorized"] and load("verification.json")["passed"]
    # RING: 공식 적재기는 몸체(축) 원점 z>1.0 m만 남긴다. 모든 평가 스캔에서 지운 점이 아래를 보는 레이저의 점과 같다.
    assert load("summary.json")["ring"]["cut"]["rule"] == "z_up_body > 1.0 m"
    ground = load("ring/ring_ground_by_beam.json")
    assert len(ground["rows"]) == ground["n_scans"] and ground["n_down_beams"] == 8
    assert all(r["frac_of_removed_from_down_beams"] == 1.0 and r["frac_removed_by_cut"] == r["frac_down_beam_points"]
               for r in ground["rows"])
    # DiSCO: 지도 솎기는 같은 모델의 평가 모집단을 바꾼다. R@1과 정답이 있는 질의 수를 함께 쓴다.
    density = load("disco/disco_density_response.json")
    assert abs(density["released"]["mean_r1"] - disco_run["mean_r1_percent"]) < 1e-9
    thin = [density["thinned"][p] for p in ("phase0", "phase1")]
    return {
        "RING_CUT_LIDAR": f"{ground['cut_above_lidar_origin_m']:.2f}", "RING_SCANS": f"{ground['n_scans']:,}",
        "DISCO_THIN": "/".join(f"{t['mean_r1']:.2f}" for t in thin),
        "DISCO_THIN_POS": "/".join(f"{t['positive_total']:,}" for t in thin),
        **followup_values(disco_run, thin),
    }


def followup_values(disco_run: dict, thin: list) -> dict:
    """root 검산을 마친 DiSCO 솎기 모집단 분해·03-17 지도 목록 대조와 RING 원형 지지·각도 격자 대조의 문장 수치를 고정 사본에서 읽는다."""
    # root 검산(2026-11-08): pipeline_residual_followup. 같은 공개 가중치·고정 규약의 진단이며 표2·native 기준값은 바꾸지 않는다.
    base = ROOT / "results/pdv2_pipeline_followup_appendix_20261108"
    pins = json.loads((base / "source_pins.json").read_text())
    for rel, entry in pins["files"].items():
        assert hashlib.sha256((base / rel).read_bytes()).hexdigest() == entry["sha256"], rel
    load = lambda rel: json.loads((base / rel).read_text())
    root = load("root/pipeline_residual_followup.json")
    assert root["passed"] and root["decisions"]["integration_authorized"] and not root["decisions"]["main_table_changes"]
    assert root["task2"]["saved_descriptor_recompute_byte_identical"] and root["task3"]["frozen_protocol_hashes_match"]
    # 솎기 분해: 30쌍 평균 이득 = 같은 질의 항 + 모집단 항. 기준·솎은 값은 위 문장(지도 솎기 원천)과 같아야 한다.
    phases = [load("disco/thinning_decomposition_summary.json")["phases"][p]["macro_30pairs"] for p in ("phase0", "phase1")]
    for ph, t in zip(phases, thin):
        assert abs(ph["R_base_P0"] - disco_run["mean_r1_percent"]) < 1e-9 and abs(ph["R_thin_C"] - t["mean_r1"]) < 1e-9
        assert abs(ph["same_queries_term"] + ph["population_term"] - ph["delta"]) < 1e-9
    # 03-17 지도 목록 대조: 지도 쪽 5쌍만. R@1%는 k가 달라 쓰지 않는다.
    score = load("disco/disco0317_score_result.json")
    assert score["all_checks_passed"] and all(c for c in score["checks"].values())
    mean, inter = score["mean_over_5_pairs"], score["intersection_over_5_pairs"]
    assert len(score["pairs"]) == 5 and all(p["database"] == "2012-03-17" for p in score["pairs"])
    cover = load("disco/disco0317_list_coverage.json")
    last = [cover[k]["session_time_frac_first_last"][1] for k in ("current_3_3p5", "first_public_1_1p5")]
    assert cover["current_3_3p5"]["n"] == score["pairs"][0]["current"]["database_frames"]
    assert cover["first_public_1_1p5"]["n"] == score["pairs"][0]["first_public"]["database_frames"]
    # RING 원형 70 m 단일 요인: 마스크 없는 통제는 기존 공식 거리와 비트 동일이다.
    ring = load("ring/ring_circle70_metrics.json")["methods"]
    circ = {}
    for name, ref in (("RING", "62.52"), ("RING++ exact max", "62.24")):
        m = ring[name]
        assert m["parity"]["control_equals_baseline_bitwise"] and m["parity"]["max_abs_pair_dist_diff_control_vs_baseline"] == 0.0
        r = m["radii"]["10.0"]
        assert f"{r['baseline']['R@1_all_pct']:.2f}" == f"{r['control_none']['R@1_all_pct']:.2f}" == ref
        circ[name] = f"{r['circle70']['R@1_all_pct']:.2f}"
    # RING 각도 격자 단일 요인(2026-11-05 실행, 2026-11-09 감사): 0과 2π를 모두 담은 공개 격자만 2π를 뺀 격자로 바꾼다.
    # 위 원형 마스크와 같은 평가 목록(해시)·같은 기준값인지 확인한다.
    angle = ROOT / "results/pdv2_review_native_20261105/ring_angles_endpoint_false"
    comparison = json.loads((angle / "comparison.json").read_text())
    assert json.loads((angle / "verification.json").read_text())["passed"] and comparison["checks_passed"]
    assert (hashlib.sha256((angle / "evaluation_sets.json").read_bytes()).hexdigest()
            == load("ring/ring_circle70_metrics.json")["evaluation_sets_sha256"])
    grid = {}
    for name, cond in (("RING", "all_ring"), ("RING++ exact max", "all_ringpp")):
        row = next(r for r in comparison["rows"] if r["condition"] == cond and r["radius_m"] == 10.0)
        base = ring[name]["radii"]["10.0"]["baseline"]
        assert row["original_endpoint_true"]["queries"] == 1054
        assert abs(100 * row["original_endpoint_true"]["official_R@1"] - base["R@1_all_pct"]) < 1e-9
        grid[name] = f"{100 * row['endpoint_false']['official_R@1']:.2f}"
    return {
        "DISCO_THIN_SAME": "/".join(f"{ph['same_queries_term']:.2f}" for ph in phases),
        "DISCO_THIN_POP": "/".join(f"{ph['population_term']:.2f}" for ph in phases),
        "DISCO_CUR_N": f"{cover['current_3_3p5']['n']:,}", "DISCO_FP_N": f"{cover['first_public_1_1p5']['n']:,}",
        "DISCO_CUR_R1": f"{mean['current']['r1_percent']:.2f}", "DISCO_FP_R1": f"{mean['first_public']['r1_percent']:.2f}",
        "DISCO_CUR_POS": f"{mean['current']['positive_queries_total']:,}",
        "DISCO_FP_POS": f"{mean['first_public']['positive_queries_total']:,}",
        "DISCO_INT_CUR": f"{inter['current_mean_r1_percent']:.2f}", "DISCO_INT_FP": f"{inter['first_public_mean_r1_percent']:.2f}",
        "DISCO_INT_N": f"{inter['positive_queries_total']:,}", "DISCO_FRAC": f"{100 * max(last):.0f}",
        "RING_CIRC": circ["RING"], "RINGPP_CIRC": circ["RING++ exact max"],
        "RING_GRID": grid["RING"], "RINGPP_GRID": grid["RING++ exact max"],
    }


def main() -> None:
    """동결 결과의 검산을 확인하고 부록 조각을 쓴다."""
    verification = json.loads((SOURCE / "verification.json").read_text())
    pilot_check = json.loads((SOURCE / "yaw_pilot/verification.json").read_text())
    assert verification["passed"] and pilot_check["passed"]
    with (SOURCE / "final_score_decomposition_percent.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row["method"] not in EXCLUDED.values()]
    ledgers = json.loads((SOURCE / "ledgers.json").read_text())["test_frozen"]
    assert verification["test_ledger_count"] == 46 * sum(len(v["seeds"]) for v in ledgers.values())
    ledger_count = 46 * sum(len(v["seeds"]) for k, v in ledgers.items() if k not in EXCLUDED)
    # SeqOT 행은 같은 수(3시드×46)의 채택 원장으로 바꾼다. 문장의 원장 수는 표에 쓰인 원장 수다.
    assert len(ledgers["seqot"]["seeds"]) == 3
    # 정확max 행이 표2가 채택한 RING++ 인식 원천과 같은 값인지 확인한다.
    from _paper_tables_v23 import ADOPTED_RINGPP, COMMON_BASELINES, RINGPP_SUMMARY, seqot_adoption
    assert hashlib.sha256(RINGPP_SUMMARY.read_bytes()).hexdigest() == ADOPTED_RINGPP["summary_sha256"]
    methods = json.loads(RINGPP_SUMMARY.read_text())["methods"]
    adopted = methods["ringpp"]["mean_std"]
    exact = next(row for row in rows if row["method"] == "RING++ exact max")
    assert all(abs(float(exact[k]) - 100 * adopted[k]["mean"]) < 1e-9 for k in ("AP", "R@1"))
    # 저자 결정(2026-11-08): 표2의 SeqOT는 논문 학습법 + 공식 GeM 수치 안정화 원장이다. 원천 CSV의 SeqOT 행(공개 코드
    # 학습법 원장)이 공통 원천과 같은지 확인한 뒤, 같은 식(손실 = 100−R@1, R@1−AP)으로 채택 138원장의 행을 만든다.
    seqot_adoption()
    previous = json.loads(COMMON_BASELINES.read_text())["methods"]["seqot"]["mean_std"]
    old = next(row for row in rows if row["method"] == "SeqOT")
    assert all(abs(float(old[k]) - 100 * previous[k]["mean"]) < 1e-9 for k in ("AP", "R@1"))
    ap, r1 = (100 * methods["seqot"]["mean_std"][k]["mean"] for k in ("AP", "R@1"))
    old.update({"AP": ap, "R@1": r1, "retrieval_loss": 100 - r1, "ranking_loss": r1 - ap})
    # 표2의 RING++(I2) 행은 채택 요약에서 같은 식으로 만든다. 대체 전 항목은 위 정확max 행과 같은 I1이어야 한다.
    from _paper_tables_v23 import ADOPTED_RINGPP_I2, RINGPP_I2_SUMMARY
    assert hashlib.sha256(RINGPP_I2_SUMMARY.read_bytes()).hexdigest() == ADOPTED_RINGPP_I2["summary_sha256"]
    i2_summary = json.loads(RINGPP_I2_SUMMARY.read_text())
    assert i2_summary["ringpp_i2_replacement"]["previous_ringpp_entry"] == methods["ringpp"]
    assert all(i2_summary["methods"][m] == methods[m] for m in methods if m != "ringpp")
    ap, r1 = (100 * i2_summary["methods"]["ringpp"]["mean_std"][k]["mean"] for k in ("AP", "R@1"))
    i2_row = {"method": "RING++", "AP": ap, "R@1": r1, "retrieval_loss": 100 - r1, "ranking_loss": r1 - ap}
    # 사용자 승인(2026-11-10): 표2의 DeepRING은 식 9 W0 초기배율 138원장이다. 원천 CSV의 DeepRING 행(W=1 시작 원장)이 대체 전
    # 항목과 같은지 확인한 뒤, 같은 식으로 채택 요약의 행을 만든다. 원장 수(3시드×46)는 같다.
    from _paper_tables_v23 import deepring_adoption
    deepring_summary, _ = deepring_adoption(i2_summary)
    previous = deepring_summary["deepring_scaleinit_replacement"]["previous_deepring_entry"]["mean_std"]
    old = next(row for row in rows if row["method"] == "DeepRING")
    assert all(abs(float(old[k]) - 100 * previous[k]["mean"]) < 1e-9 for k in ("AP", "R@1"))
    assert len(ledgers["deepring"]["seeds"]) == 3
    ap, r1 = (100 * deepring_summary["methods"]["deepring"]["mean_std"][k]["mean"] for k in ("AP", "R@1"))
    old.update({"AP": ap, "R@1": r1, "retrieval_loss": 100 - r1, "ranking_loss": r1 - ap})

    def line(row: dict, name: str) -> str:
        values = [float(row[k]) for k in ("AP", "R@1", "retrieval_loss", "ranking_loss")]
        assert abs(values[0] + values[2] + values[3] - 100) < 1e-9
        return name.replace("#", r"\#") + " & " + " & ".join(f"{v:.2f}" for v in values) + r" \\"
    lines, diagnostics = [], []
    for row in rows:
        if row["method"] == "RING++":
            lines.append(line(i2_row, "RING++"))
            diagnostics.append(line(exact, DIAGNOSTIC_ROW[exact["method"]]))
            diagnostics.append(line(row, DIAGNOSTIC_ROW[row["method"]]))
        elif row["method"] not in MAIN_ROW:
            lines.append(line(row, row["method"]))
    lines += [r"\midrule"] + diagnostics
    template = r"""% 원천: results/pdv2_baseline_common_diagnosis_20261106/
\subsection{Common-benchmark and training-signal diagnostics}
\label{app:baseline_signal}
\paragraph{Retrieval and acceptance ranking.}
We recompute all main metrics from @LEDGERS@ frozen baseline ledgers, including the 46 released-kernel RING++ ledgers, and from the 46 exact-maximum RING++ ledgers for each correlation normalization. The RING++ row uses the unit-variance implementation of Table~\ref{tab:results}; the two rows below the rule are the released-normalization diagnostics of Table~\ref{tab:ringpp_maximum}. The SeqOT and DeepRING rows use the paper-recipe and scale-initialized ledgers of Table~\ref{tab:results}, respectively. For fixed returned places, revisit R@1 is the upper bound on AP under perfect acceptance ranking. Thus $100-\mathrm{AP}=(100-\mathrm{R@1})+(\mathrm{R@1}-\mathrm{AP})$, with percentages on both sides. Table~\ref{tab:baseline_loss_decomposition} separates retrieval failures from the remaining ranking gap using the same city and seed averaging as the main table. This is a descriptive decomposition, not an estimate of performance recoverable by retraining. Monotone score calibration cannot change AP.

\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Final AP and its retrieval/ranking gaps (\%).}
\label{tab:baseline_loss_decomposition}
\begin{tabular}{@{}lrrrr@{}}\toprule
Method & AP & R@1 & Retrieval & Ranking \\
\midrule
@DECOMPOSITION@
\bottomrule
\end{tabular}
\end{table}

\paragraph{Dataset dependence.}
The validation MulRan queries contain only 2.34\% revisits among PR-eligible queries, compared with 23.07\% for the KITTI family and 25.93\% for HeLiPR. There, BEVPlace++ has R@1 of 99.53\% but AP of 40.09\%. Retrieval accuracy and rejection of nonrevisits must therefore be examined separately.

\paragraph{Training targets.}
We inspect all positive relations and replay the released HeLiOS batch sampler with three random initializations. Of 4,090,486 directed HeLiOS positives, 5.14\% lie within 5\,m and 58.32\% exceed 25\,m; the latter fraction remains 57.97--58.32\% among positive relations in replayed batches. These are relation pools, not necessarily the active pairs selected by the loss. BEVPlace++'s nonself positives are all within 5\,m. Overlap-based supervision and a 5\,m place criterion express different targets; this observation does not establish that overlap labels are erroneous.

\paragraph{SeqOT positive signal.}
This and the following SeqOT diagnostics use the released-code recipe and its selected models (Sec.~\ref{app:baselines}), not the models of Table~\ref{tab:results}. Replaying its recorded training plan shows that all selected positives are the anchor itself in 59.14\% of first-stage updates and 56.15\% of second-stage updates. With the selected first-stage model, actual forward/backward probes on 128 such cases give zero positive distance and zero positive-output gradient in both training and evaluation modes. Negative-repulsion loss can remain active. Among 9,142 first-stage self-only updates, 7,422 have no nonself positive in the source outgoing overlap table, 215 lose existing positives during normalized sampling, and 1,505 miss available positives when taking the first entries. Even after including incoming relations, 6,477 anchors lack a nonself positive under that overlap rule. Simply deleting self pairs therefore does not restore all missing supervision.

\paragraph{Sensor and augmentation checks.}
The median valid-pixel fraction in SeqOT's range inputs is 30.5--31.9\% for the four HeLiPR training sequences and 81.5--89.3\% for KITTI. In HeLiPR, 22.1--51.2\% of nonself pairs within 5\,m have overlap below 0.3. These observations identify a target/input mismatch without isolating its causal contribution to final performance. Sensor imbalance alone is insufficient: HeLiPR accounts for 88.47\% of eligible HeLiOS anchors and 88.24\% of BEVPlace++ inputs.

\paragraph{Controlled yaw check.}
SeqOT uses width-one convolutions, attention without positional encoding, and order-independent pooling, making horizontal circular shifts token permutations. On 32 inputs at three shifts, the largest descriptor-coordinate difference is $1.41\times10^{-6}$. A single-seed control continues the same selected model for 4,096 updates at learning rate $5\times10^{-7}$ with identical training pairs and order; the treatment adds independent horizontal shifts per input, shared across its three temporal scans. The GeM model is fixed. Table~\ref{tab:seqot_yaw_control} reports final retrieval and acceptance metrics on all nine common validation sequences (99,236 queries). The yaw treatment changes AP by $-0.00194$ percentage points relative to its equal-update control and leaves the other three metrics unchanged. Neither continuation is adopted in the main table. Coordinate rotation does not simulate reverse-traversal occlusion changes.

\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{SeqOT yaw control on common validation (\%).}
\label{tab:seqot_yaw_control}
\begin{tabular}{@{}lrrrr@{}}\toprule
Condition & AP & Max F1 & R@100P & R@1 \\
\midrule
@PILOT@
\bottomrule
\end{tabular}
\end{table}@LEARNED@
"""
    pilot = [("Selected model", pilot_check["original_validation"])]
    for name, arm in [("Equal-update control", "control"), ("Independent yaw", "yaw")]:
        data = json.loads((SOURCE / "yaw_pilot" / arm / "report.json").read_text())
        pilot.append((name, data["macro"]))
    pilot_lines = [name + " & " + " & ".join(f"{100 * m[k]:.4f}" for k in
                   ("AP", "max_F1", "R@100P", "R@1")) + r" \\" for name, m in pilot]
    content = (template.replace("@DECOMPOSITION@", "\n".join(lines)).replace("@PILOT@", "\n".join(pilot_lines))
               .replace("@LEDGERS@", f"{ledger_count:,}").replace("@LEARNED@", learned_direction_paragraph()))
    assert "ImLPR" not in content
    (PAPER / "baseline_common_diagnostics.tex").write_text(content)
    published_appendix()
    sampler_appendix()
    print(f"부록 공통 진단 생성: 분해{len(lines)}행, 원장{ledger_count:,}개, 회전 대조3행")


# root 검산(2026-11-08, scpp_height_followup): 표2 SC++에서 높이만 공식 상수 1.9 m로 바꾼 시험 전 고정 민감도(37시퀀스 재계산,
# NCLT 9원장 재사용)와 KITTI08 전 프레임 공개 PC/CC 단일 고정 진단. 바이트 동일 사본을 해시로 고정하며 표2는 바꾸지 않는다.
SCPP_HEIGHT = ROOT / "results/pdv2_scpp_height_sensitivity_20261108"


def scpp_height_values() -> dict:
    """SC++ 공식 1.9 m 높이 민감도와 KITTI08 전 프레임 진단의 문장 수치를 고정 사본에서 읽는다."""
    pins = json.loads((SCPP_HEIGHT / "pins.json").read_text())
    for rel, entry in pins["files"].items():
        assert hashlib.sha256((SCPP_HEIGHT / rel).read_bytes()).hexdigest() == entry["sha256"], rel
    load = lambda rel: json.loads((SCPP_HEIGHT / rel).read_text())
    root = load("root_verification.json")
    assert root["passed"] and root["integration_authorized"] and load("verification.json")["passed"]
    result, protocol = load("results_1p9.json"), load("protocol.json")
    assert result["protocol_sha256"] == hashlib.sha256((SCPP_HEIGHT / "protocol.json").read_bytes()).hexdigest()
    assert protocol["frozen_before_any_1p9_test_result"] and protocol["path_fixed"]["lidar_height_m"] == 1.9
    assert len(protocol["inputs"]["recompute_37"]) == 37 and len(protocol["inputs"]["reuse_nclt_9"]["sequences"]) == 9
    seqs = result["per_sequence"].values()
    assert (sum(s["n_queries"] for s in seqs), sum(s["n_positive"] for s in seqs)) == (195726, 61682)
    # 현재 값은 표2 SC++ 원천과 같아야 한다. 차이는 1.9 m − 현재(%p)이며 결과를 보고 설정을 고르지 않는다.
    from _paper_tables_v23 import COMMON_BASELINES
    main = json.loads(COMMON_BASELINES.read_text())["methods"]["scpp"]["mean_std"]
    metrics = ("AP", "max_F1", "R@100P", "R@1")
    current, official = result["current_main_sensor_height"]["macro"], result["official_1p9"]["macro"]
    assert all(abs(current[k] - main[k]["mean"]) < 1e-12 for k in metrics)
    assert all(abs(100 * official[k] - root["official_1p9_percent"][k]) < 1e-9 for k in metrics)
    delta = max(abs(result["delta_1p9_minus_current_pp"]["macro"][k]) for k in metrics)
    bound = f"{delta:.2f}"
    assert float(bound) >= delta
    r100 = [s["delta_pp"]["R@100P"] for s in seqs if s["n_positive"]]
    assert len(r100) == 45 and min(r100) < 0 < max(r100)
    # KITTI08: 실제 스크립트는 마지막 프레임까지 질의하고, 공개 fast evaluator 형식은 마지막 프레임을 뺀다(별도 재집계).
    k08 = json.loads((SCPP_HEIGHT / "native_kitti08/kitti08_allframes_result.json").read_text())
    pc, cc, paper = k08["variants"]["pc"], k08["variants"]["cc"], k08["paper"]
    run, fast = pc["r8_native_q50_to_N"], pc["r8_fast_evaluator_q50_to_N-1"]
    assert k08["inputs"]["velodyne_bins"] == k08["arithmetic"]["N_allframes"] and fast["queries"] == run["queries"] - 1
    assert cc["r8_native_q50_to_N"]["queries"] == run["queries"] and cc["r8_native_q50_to_N"]["revisit_queries"] == run["revisit_queries"]
    # 그림 질의 수는 float32 좌표 모형에서 PC 곡선만 하나로 정해진다(CC는 후보가 둘이라 쓰지 않는다).
    (figure_q, _), = pc["figure_end_Q_TP"]
    assert len(cc["figure_end_Q_TP"]) > 1 and figure_q not in (run["queries"], fast["queries"])
    # PC 8 m의 두 자리 일치는 수락 없는 임계값의 정밀도를 1로 두는 관례에서만 성립한다. 4·2·1 m는 PC/CC 모두 다르다.
    assert pc["vs_paper_round2"]["table_ix_8m"]["round2_equal"] and round(run["auc_nan_segments_skipped"], 2) != paper["table_ix_8m"]["pc"]
    assert not any(v["vs_paper_round2"][f"table_ix_{r}m"]["round2_equal"] for v in (pc, cc) for r in (4, 2, 1))
    pct = lambda m: "/".join(f"{100 * m[k]:.2f}" for k in metrics)
    return {
        "SCPP_H19_CUR": pct(current), "SCPP_H19": pct(official), "SCPP_H19_MAX": bound,
        "SCPP_H19_SEQ": f"${min(r100):.2f}$ to ${max(r100):+.2f}$",
        "SCPP_K08_FRAMES": f"{k08['arithmetic']['N_allframes']:,}", "SCPP_K08_Q": f"{run['queries']:,}",
        "SCPP_K08_QFAST": f"{fast['queries']:,}", "SCPP_K08_QFIG": f"{figure_q:,}", "SCPP_K08_POS": str(run["revisit_queries"]),
        "SCPP_K08_PC": f"{run['auc']:.3f}", "SCPP_K08_PC_SKIP": f"{run['auc_nan_segments_skipped']:.3f}",
        "SCPP_K08_CC": f"{cc['r8_native_q50_to_N']['auc']:.3f}",
        "SCPP_K08_PAPER": f"{paper['table_ix_8m']['pc']:.2f} and {paper['table_ix_8m']['cc']:.2f}",
    }


# root 검산(2026-11-08, ringsharp_final_closure_20261109): RING#-L 공개 코드·설정 그대로의 규약1 1회 학습(시드123, 30에폭,
# 평가 에폭30 학습 전 고정)과 공식 평가 1회. 바이트 동일 사본과 논문 원문 줄 해시(paper_facts.json)만 두며 표2는 바꾸지 않는다.
RINGSHARP = ROOT / "results/pdv2_ringsharp_protocol1_20261108"


def ringsharp_values() -> dict:
    """RING#-L 공개 설정 규약1 1시드 진단의 문장 수치를 고정 사본에서 읽는다."""
    pins = json.loads((RINGSHARP / "pins.json").read_text())
    for rel, entry in pins["files"].items():
        assert hashlib.sha256((RINGSHARP / rel).read_bytes()).hexdigest() == entry["sha256"], rel
    load = lambda rel: json.loads((RINGSHARP / rel).read_text())
    root, verification, recheck = load("root_verification.json"), load("verification.json"), load("root_rerun/recheck_result.json")
    assert root["passed"] and root["integration_authorized"] and root["compute_complete"] and not root["paper_reproduced"]
    assert recheck["all_pass"] and recheck["n_checks"] == root["root_checks"] == 25 and recheck["full_hash"]
    assert verification["all_pass"] and verification["n_items"] == 45
    # 지표: 공식 값 = 저장 배열 독립 재계산 = root 값. 논문 값은 원문 표 줄(pr_protocol1.tex:26)과 같아야 한다.
    final, ind, paper = load("final_metrics.json"), load("root_rerun/independent_metrics.json"), load("paper_facts.json")
    keys = ("Recall@1", "F1", "AUC")
    r10 = ind["r10_summary"]
    assert final["all_checks_pass"] and ind["all_ok"] and not any(r10["abs_diff"].values())
    assert all(final["official_metrics"][k] == r10["official_from_saved_arrays"][k] == root["metrics_r10"][k] for k in keys)
    assert paper["values"]["paper_nclt_protocol1"] == root["paper_values"] == {k: final["paper"][k] for k in keys}
    assert all(final["official_metrics"][k] < root["paper_values"][k] for k in keys)
    # 실행 규약: 공개 설정 lr·wd, 시드123, 30에폭, 학습 전 고정한 에폭30 한 번만 평가.
    protocol, recipe = load("frozen/eval_protocol.json"), load("frozen/recipe.json")
    assert protocol["final_epoch_prefixed"] == recipe["epochs"] == 30 and protocol["runs"].startswith("exactly one final evaluation")
    assert "random_seed=123" in recipe["seed"] and (recipe["lr"], recipe["weight_decay"]) == (1e-4, 1e-5)
    assert (paper["values"]["paper_lr"], paper["values"]["paper_weight_decay"]) == (1e-3, 1e-4)
    # 입력 개수: 생성한 학습 표본·시험 질의 수가 논문 표와 같다(저자 목록 동일성은 아니다). 시험 피클은 평가에 쓴 것과 같다.
    inputs = load("inputs/pickles_summary.json")
    assert (inputs["train"]["tuples"], inputs["test"]["query"]) == (paper["values"]["paper_protocol1_nclt_train"],
                                                                   paper["values"]["paper_protocol1_nclt_test"])
    assert inputs["test"]["sha256"] == ind["test_set"]["pickle_sha256"] and protocol["n_query"] == ind["test_set"]["n_query_pickle"]
    # 공식 100개 임계값에서 이번 실행의 수락 집합은 세 종류이고, 최대 F1은 모든 질의를 수락한 마지막 임계값에서 나온다.
    grid, best = ind["grid_diagnostic_r10_not_a_metric"], ind["per_radius"]["10m"]
    n_query = inputs["test"]["query"]
    assert grid["n_distinct_accept_counts_over_100_thresholds"] == 3 and grid["all_queries_accepted_by_last_threshold"]
    assert best["argmax_threshold_index"] == 99 and best["max_f1"] == final["official_metrics"]["F1"]
    counts = best["counts_at_best_f1"]
    assert counts["tp"] + counts["fp"] == n_query and counts["tn"] == counts["fn"] == 0
    power = lambda v: f"$10^{{{int(f'{v:.0e}'.split('e')[1])}}}$"
    return {
        "RS_SEED": "123", "RS_EPOCHS": str(recipe["epochs"]),
        "RS_TRAIN": f"{inputs['train']['tuples']:,}", "RS_TEST": str(n_query),
        "RS_LR": power(recipe["lr"]), "RS_WD": power(recipe["weight_decay"]),
        "RS_PAPER_LR": power(paper["values"]["paper_lr"]), "RS_PAPER_WD": power(paper["values"]["paper_weight_decay"]),
        "RS_RUN": ", ".join(f"{final['official_metrics'][k]:.3f}" for k in keys[:2]) + f", and {final['official_metrics']['AUC']:.3f}",
        "RS_PAPER": ", ".join(f"{root['paper_values'][k]:.2f}" for k in keys[:2]) + f", and {root['paper_values']['AUC']:.2f}",
        "RS_SETS": "three",
    }


# root 검산(2026-11-09, learned_fix_claim_closure_20261109): SeqOT 논문 학습법 단계1 시드1(GeM p=3 고정)과 HeLiOS 처음부터 학습
# 1시드(lr5e-4, B32 선택 e10)의 검증9 방향 진단과 두 4에폭 시험 수정. 바이트 동일 사본을 해시로 고정하며 표2는 바꾸지 않는다.
LEARNED_DIRECTION = ROOT / "results/pdv2_learned_direction_20261109"


def learned_direction_paragraph() -> str:
    """HeLiOS·SeqOT 같은 분할 학습의 방향별 검증 결손 진단 단락을 고정 사본에서 만든다."""
    pins = json.loads((LEARNED_DIRECTION / "pins.json").read_text())
    for rel, entry in pins["files"].items():
        assert hashlib.sha256((LEARNED_DIRECTION / rel).read_bytes()).hexdigest() == entry["sha256"], rel
    for rel, digest in pins["repo_sources_sha256"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    load = lambda rel: json.loads((LEARNED_DIRECTION / rel).read_text())
    closure, completion, numeric = (load(f"root/{n}.json") for n in
                                    ("learned_fix_claim_closure", "learned_training_fix_completion", "learned_training_fix_numeric_recheck"))
    assert closure["passed"] and closure["numeric_and_output_checks"] == 33 and closure["raw_ledger_hashes_matched_server"] == 5
    assert closure["raw_training_plan_hash_matched_server"] and completion["all_numeric_checks_passed"] and numeric["all_ok"]
    assert completion["main_table_unchanged"] and not completion["new_test46_authorized"]
    # 대상 모델: SeqOT 단계1 시드1 선택 에폭(GeM 공식 초기값 p=3 고정), HeLiOS lr5e-4의 B32 선택 e10(root 동결 쌍).
    assert load("p2_seqot.json")["gem"] == "official init p=3"
    stage1 = json.loads((ROOT / "results/pdv2_baseline_revision_20261107/seqot/s1/train_selection.json").read_text())["stage1"]
    selected = stage1["epochs"][str(stage1["chosen"]["epoch"])]["family_r1_l2"]
    assert stage1["chosen"]["epoch"] == 11 and selected == numeric["numbers"]["seqot_s1_e11"]["val"]
    decision = load("helios_scratch_decision_20.json")
    assert decision["top_candidate"] == "lr5e-4" and decision["best_at_20"]["lr5e-4"]["epoch"] == 10 and decision["extend"] is None
    helios_root = load("root/helios_scratch_final_audit.json")
    assert helios_root["rerun_identical"] and helios_root["frozen_pair"]["candidate"]["name"] == "lr5e-4 e10"
    # 방향: 위치 차분의 진행 방향. 허용 정답 프레임이 모두 같은 부류일 때만 순수 같은/반대, 섞이면 혼합.
    kitti = load("p8_family_class_counts.json")["kitti"]
    pct = lambda c, n: f"{100 * c / n:.1f}"
    words = {1: "one", 7: "Seven"}
    seq = {}
    for cls, key in (("pure_opp", "OPP"), ("pure_same", "SAME")):
        n, (c0, c1) = kitti[cls]["n"], (kitti[cls]["correct"][k] for k in ("init_s1", "s1_e11"))
        seq.update({f"{key}_N": f"{n:,}", f"{key}_C0": f"{c0:,}", f"{key}_C1": f"{c1:,}",
                    f"{key}_R0": pct(c0, n), f"{key}_R1": pct(c1, n)})
    plan = load("plan_heading.json")["control"]["revisit_le5m"]
    assert plan["opp_gt90"] == 1 and plan["undef"] == 0
    audit = load("helios_audit.json")
    b32, b1 = audit["direction_split_b32"]["e10"]["kitti"], audit["direction_split_b1_e10"]["e10"]["kitti"]
    assert b32["pure_opp"][1] == b1["pure_opp"][1] == kitti["pure_opp"]["n"] and b32["pure_same"][1] == kitti["pure_same"]["n"]
    # 두 시험 수정(F1 퇴화 overlap 가드, F2 계열 균형): 학습 전 고정한 기준 max(초기값, 대조군 최고)+0.02를 4에폭 안에 넘지 못했다.
    criterion = load("protocol_amendment_2.json")["success_criterion_val_only"]
    assert "+ 0.02 within 4 epochs" in criterion
    threshold = max(numeric["numbers"]["seqot_init_s1"]["val"], max(e["family_r1_l2"] for e in stage1["epochs"].values())) + 0.02
    best = {v: max(e["macro"] for e in numeric["pilots"][v]["epochs"].values()) for v in ("f1", "f2")}
    assert all(b < threshold for b in best.values()) and "neither meets" in completion["pilot_verdict"]
    # 검증 질의 규칙(앞 정답 i <= j-30)과 후보 규칙(|j-q|>30)의 경계로 어떤 모델도 맞힐 수 없는 질의. 빼도 선택이 바뀌지 않는다.
    boundary = load("boundary_sensitivity.json")
    assert boundary["drop"] == {"kitti360_0004": 5, "kitti360_0006": 2} and boundary["n_unreachable"] == 7
    assert boundary["ledger_summary"]["all_present"] and not boundary["ledger_summary"]["any_correct"]
    picks = list(boundary["seqot"].values()) + [v for k, v in boundary["helios"].items() if k.startswith(("table2", "finetune"))]
    picks += list(boundary["helios"]["scratch_pilot_b32"]["per_candidate"].values())
    assert len(picks) == 15 and not any(p["changed"] for p in picks)
    assert boundary["helios"]["scratch_pilot_b32"]["top"] == boundary["helios"]["scratch_pilot_b32"]["top_excl7"] == "lr5e-4/e10"
    assert not any(boundary["seqot_pilot_criterion"]["excl7"][v]["meets"] for v in ("f1", "f2"))
    values = {**seq, "PLAN_OPP": words[plan["opp_gt90"]], "PLAN_N": f"{plan['n']:,}",
              "H_SAME": pct(*b32["pure_same"][:2]), "H_OPP": pct(*b32["pure_opp"][:2]), "H_B1_OPP": pct(*b1["pure_opp"][:2]),
              "F1_BEST": f"{100 * best['f1']:.1f}", "F2_BEST": f"{100 * best['f2']:.1f}", "SELECTED": f"{100 * selected:.1f}",
              "BOUNDARY": words[boundary["n_unreachable"]]}
    text = r"""

\paragraph{Direction-dependent validation loss.}
These validation-only checks instead use the paper-recipe first-stage SeqOT model of seed~1 with GeM fixed at $p=3$ and a single-seed HeLiOS model trained from scratch (learning rate $5\times10^{-4}$; epoch~10 selected with 32-scan encoding batches), not the models of Table~\ref{tab:results}. On the nine validation sequences, travel direction is derived from trajectory displacement, and a query counts as same- or opposite-direction only when all its admissible ground-truth frames agree; mixed queries are counted separately. For KITTI-family queries, SeqOT first-stage training lowers opposite-direction R@1 from @OPP_R0@\% to @OPP_R1@\% (@OPP_C0@ to @OPP_C1@ of @OPP_N@) at the selected epoch and raises same-direction R@1 from @SAME_R0@\% to @SAME_R1@\% (@SAME_C0@ to @SAME_C1@ of @SAME_N@). Its per-sequence first-stage training plan contains @PLAN_OPP@ opposite-direction pair among @PLAN_N@ revisit positives within 5\,m. With 32-scan batches, HeLiOS reaches @H_SAME@\% same-direction but @H_OPP@\% opposite-direction KITTI-family R@1 (@H_B1_OPP@\% with the official single-scan encoder). The training data and the deficit correspond, but we did not test whether adding opposite-direction pairs removes it.

Two bounded four-epoch SeqOT pilots, a degenerate-overlap guard and KITTI/HeLiPR update balancing, reach best family-macro R@1 of @F1_BEST@\% and @F2_BEST@\%, respectively, against @SELECTED@\% for the existing selection. Neither meets the criterion fixed before training, and the bounded pilots show no material validation gain, so Table~\ref{tab:results} keeps its models. @BOUNDARY@ KITTI-360 queries lack admissible ground-truth candidates; removing them leaves the checked SeqOT and HeLiOS checkpoint selections unchanged."""
    for key, value in values.items():
        text = text.replace("@" + key + "@", value)
    assert not re.search(r"@[A-Z0-9_]+@", text)
    return text


if __name__ == "__main__":
    main()
