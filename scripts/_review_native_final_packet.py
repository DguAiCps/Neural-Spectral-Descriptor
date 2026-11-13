#!/usr/bin/env python3
"""다섯 비교 방법의 검산 완료 수치와 주장 범위를 한 전달 파일로 묶는다."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105"
RING = REPO / "results/pdv2_ringpp_maxfix_20261105"
MODELS = ["author", "v2_s1", "v2_s2", "v2_s3"]


def main() -> None:
    """기존 초기 제안·원장을 보존하고 최종 결과만 새 파일로 전달한다."""
    sources = {str(Path(__file__).relative_to(REPO)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def read(path: Path) -> dict:
        raw = path.read_bytes()
        sources[str(path.relative_to(REPO))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    learned = read(OUT / "learned_independent_verification.json")
    assert len(learned["helios"]) == 96 and len(learned["seqot"]) == 4
    assert all(r["passed"] for method in learned.values() if isinstance(method, list) for r in method)
    helios = read(OUT / "helios/full_release_final_verification.json")
    assert helios["complete"] and helios["passed"]
    ring = read(RING / "summary_verified.json")
    common_baselines = read(REPO / "results/pdv2_main_official_baselines/baselines_common.json")
    common_ring = common_baselines["methods"]["ring"]["macro"]
    geo = read(RING / "geometry_independent_verification.json")
    assert ring["checks_passed"] and geo["complete"] and geo["passed"]
    assert geo["sequences"] == 46 and geo["common_queries"] == 195726
    ring_native = read(OUT / "integrated_report.json")
    clarification = read(OUT / "ring_protocol_clarification.json")
    for path in [OUT / "verification.json", OUT / "maxfix/verification.json"]:
        verified = read(path)
        assert verified["passed"] and verified["raw_input_hashes_matched"] == 1413
    rows = []
    full10 = next(r for r in ring_native["comparisons"] if r["region"] == "all" and r["radius_m"] == 10)
    rows += [{"method": "RING", "condition": "NCLT full / XY10m", "metric": "R@1",
              "native_reference_percent": 100*full10["RING_R@1"], "v2_mean_percent": None, "v2_sample_sd_pp": None,
              "scope": "공식 native 경로; 논문 정확 입력 영역 목록 미확인, 학습 없음"},
             {"method": "RING++", "condition": "NCLT full / XY10m / retained kernel", "metric": "R@1",
              "native_reference_percent": 100*full10["official_RINGpp_R@1"], "v2_mean_percent": None, "v2_sample_sd_pp": None,
              "scope": "비결정적 kernel 전체1회; 여러 실행의 평균 아님"},
             {"method": "RING++", "condition": "NCLT full / XY10m / exact max", "metric": "R@1",
              "native_reference_percent": 100*full10["maxfix_RINGpp_R@1"], "v2_mean_percent": None, "v2_sample_sd_pp": None,
              "scope": "정확 max 한 연산만 바꾼 진단; 논문 표 완전 재현 아님"}]
    for group in helios["groups"]:
        for metric in ["r1", "r5"]:
            values = group["official_mean"]
            seeds = [100*values[m][metric] for m in MODELS[1:]]
            rows.append({"method": "HeLiOS", "condition": group["database"] + " / " + group["group"],
                "metric": metric.upper().replace("R", "R@"), "native_reference_percent": 100*values["author"][metric],
                "v2_mean_percent": statistics.mean(seeds), "v2_sample_sd_pp": statistics.stdev(seeds),
                "scope": "공개12센서세션·24쌍의 네 그룹, 그룹당6쌍 공식분모 평균; Roundabout은v2검증 장소"})
    seqot = [read(OUT / f"seqot/{m}/native/report.json") for m in MODELS]
    for rank in ["1", "5", "20"]:
        seeds = [100*r["official_recall"][rank] for r in seqot[1:]]
        rows.append({"method": "SeqOT", "condition": "NCLT Jan08→Feb05 / released 3D15m", "metric": "R@"+rank,
            "native_reference_percent": 100*seqot[0]["official_recall"][rank],
            "v2_mean_percent": statistics.mean(seeds), "v2_sample_sd_pp": statistics.stdev(seeds),
            "scope": "5336양성, 논문의 overlap>.3 정답과 다름; 저자학습은NCLTJan08이며v2학습은NCLT미포함"})
    for k in (15, 25):
        for sequence in ("DCC01", "DCC02", "DCC03"):
            reports = [read(OUT / f"imlpr/k{k}/{m}/native_batch1_{sequence}.json") for m in MODELS]
            seeds = [100*r["r1"] for r in reports[1:]]
            rows.append({"method": "ImLPR", "condition": f"MulRan {sequence} / k{k} / batch1", "metric": "R@1",
                "native_reference_percent": 100*reports[0]["r1"], "v2_mean_percent": statistics.mean(seeds),
                "v2_sample_sd_pp": statistics.stdev(seeds),
                "scope": "공식raw재생성4275장, 논문4328장과 입력목록 다름; 저자학습 대전과native대전 장소 겹침"})
    author_reproduction = read(OUT / "helios/author_full_published_comparison.json")
    assert author_reproduction["independent_passed"]
    for relative in ["author_training_place_scope.json", "seqot/protocol_label_correction.json",
                     "imlpr/published_overlap_control.json", "learned_sources/imlpr_archive_probe_20261106.json"]:
        read(OUT / relative)
    report = {"generated_kst": datetime.now().astimezone().isoformat(),
        "status": "다섯 방법의 고정 규약 진단과 RING++ 정확max 공통46 완료; 최적수렴 주장은 하지 않음",
        "rows": rows, "native_reference_definition": "학습형은 저자 공개 가중치, 비학습형은 condition에 명시한 구현의 실제 값이다. exact max는 원저자 구현 수치로 부르지 않는다.", "shared_data_checkpoint_policy": "기존 주표의 원래선택3시드. 새 공통검증 선택은 아래 별도 연결 자료로 구분한다.",
        "ringpp_common46": {"macro": ring["ringpp_maxfix"]["macro"], "previous": ring["previous_ringpp"],
                            "delta_previous_pp": ring["delta_previous_pp"], "queries": 195726, "positive_queries": 61682,
                            "RING_reference": common_ring,
                            "fixed_RINGpp_minus_RING_pp": {k: 100*(ring["ringpp_maxfix"]["macro"][k]-common_ring[k])
                                                           for k in ring["ringpp_maxfix"]["macro"]}},
        "helios_author_max_abs_paper_delta_pp": author_reproduction["max_abs_delta_pp"],
        "new_common_selection_diagnostics": {},
        "limits": ["학습형 저자가중치는 장소분리 주표를 대체하지 않는다.",
                   "native와공통46의 정답·후보·센서·집계·학습노출 차이를 분리해 명시한다. 절대값 차이를 단일요인의 인과효과로 주장하지 않는다.",
                   "원논문 전체입력/규약 수치 재현은 HeLiOS 공개24쌍에서 최대0.062%p 차이로 확인했다. RING/SeqOT/ImLPR 차이는 각 행의 범위를 따른다.",
                   "보존 학습곡선과 실제예산은 점검했으나 최적수렴 미입증. 장기 추가 재학습은 주말 마감 지시로 보류한다.",
                   "기존 Table3 비용을 정확max 구현의 실측값으로 간주하지 않는다."],
        "training_recipe": read(OUT / "training_recipe_conformance.json"),
        "paper_proposal": "results/pdv2_review_native_20261105/paper_proposal/completed_additions_20261106.pdf"}
    for method, relative in [("SeqOT", "seqot/selection_common_native_bridge.json"),
                             ("ImLPR", "imlpr/common_selection_native/independent_verification.json")]:
        path = OUT / relative
        if path.exists():
            value = read(path)
            assert value["passed"]
            report["new_common_selection_diagnostics"][method] = {"path": str(path.relative_to(REPO)), "complete": True}
    cross_host = OUT / "imlpr/common_selection_native/cross_host_verification.json"
    if cross_host.exists():
        assert read(cross_host)["passed"]
    score = read(RING / "score_separation_diagnostic.json")
    clarification = read(RING / "score_metric_clarification.json")
    assert score["passed"] and clarification["passed"]
    report["conditional_ROC_AUC_on_same_correct_queries"] = {
        "macro": score["city_macro_conditional_AUC"], "scope": score["scope"],
        "metric": clarification["metric"], "not_interchangeable_with": clarification["not_interchangeable_with"]}
    coordinate = OUT / "ring_feature_metric/comparison.json"
    if coordinate.exists():
        value = read(coordinate)
        verified = read(OUT / "ring_feature_metric/verification.json")
        assert value["checks_passed"] and value["RING_matrices_unchanged_bitwise"]
        assert verified["passed"] and verified["raw_input_hashes_matched"] == 1413
        report["separate_feature_coordinate_diagnostic"] = value
    else:
        report["separate_feature_coordinate_diagnostic"] = {"complete": False, "scope": "별도 진단이며 정확max 공통46에 적용하지 않음"}
    report["sources_sha256"] = sources
    (OUT / "final_native_delivery_20261106.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    with (OUT / "final_native_delivery_20261106.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = report["ringpp_common46"]
    text = "\\paragraph{Checking the native performance of the baselines.}\n"
    text += ("With the released HeLiOS data, weights, and retrieval code, all 24 sensor/session pairs "
             f"reproduce the eight reported recall values within {author_reproduction['max_abs_delta_pp']:.3f} percentage points. "
             "We also evaluate three shared-data checkpoints under that same native protocol. "
             "For RING++, we identify a non-atomic grid-maximum update in the released CUDA implementation. "
             "Changing only this operation to the exact maximum changes common-test AP by "
             f"${metrics['delta_previous_pp']['AP']:+.3f}$ points and R@1 by ${metrics['delta_previous_pp']['R@1']:+.3f}$ points. "
             "We retain the original results and report this correction separately. "
             "The supplement also separates released-label and preprocessing differences for SeqOT and ImLPR, "
             "and documents training-place exposure and retained training curves. "
             "These diagnostics do not establish optimal training convergence or authorize replacing the place-disjoint models with author checkpoints.\n")
    (OUT / "paper_proposal/body_insert_completed_20261106.tex").write_text(text)
    compact = "\\paragraph{Native diagnostic summary.}\n"
    compact += ("Each row uses its own released evaluation protocol; these numbers must not be used to rank methods across rows. "
                "Shared-data values use the three original main-table checkpoints, with mean and sample SD. "
                "The later common-validation selections are reported separately. All values are R@1 percentages.\n")
    compact += "\\begin{center}\\small\\begin{tabular}{llrr}\nMethod & Native condition & Author or fixed code & Shared data \\\\\n\\hline\n"
    for row in rows:
        if row["metric"] != "R@1":
            continue
        condition = row["condition"]
        if row["method"] == "RING":
            condition = "Full NCLT, original"
        elif row["method"] == "RING++":
            condition = "Full NCLT, exact max" if "exact max" in condition else "Full NCLT, retained kernel"
        elif row["method"] == "HeLiOS":
            condition = condition.replace("Roundabout01-", "")
        elif row["method"] == "SeqOT":
            condition = "Jan08--Feb05, released 15\\,m"
        elif row["method"] == "ImLPR":
            condition = condition.replace("MulRan ", "").replace(" / batch1", "")
        shared = "--" if row["v2_mean_percent"] is None else f"${row['v2_mean_percent']:.2f}\\pm{row['v2_sample_sd_pp']:.2f}$"
        compact += f"{row['method']} & {condition} & {row['native_reference_percent']:.2f} & {shared} \\\\\n"
    compact += ("\\end{tabular}\\end{center}\n"
                "RING/RING++ use the full released NCLT trajectory and a 10\\,m XY radius; the paper's exact spatial input list is not available. "
                "The retained RING++ kernel result is one nondeterministic full run. SeqOT uses released 3D 15\\,m labels, whereas the paper describes overlap labels. "
                "ImLPR uses 4,275 officially regenerated raw scans rather than the paper's 4,328-frame input list, with both normal settings fixed in advance. "
                "HeLiOS uses all 24 released sensor/session pairs. Author training-place exposure and shared-data validation-place exposure remain disclosed in the separate notes.\n")
    (OUT / "paper_proposal/native_compact_table_20261106.tex").write_text(compact)
    print(json.dumps({"complete": True, "rows": len(rows), "methods": sorted({r['method'] for r in rows})}))


if __name__ == "__main__":
    main()
