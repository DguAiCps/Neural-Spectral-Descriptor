#!/usr/bin/env python3
"""RING 후속 진단 결과를 모아 report.json과 report_ko.txt를 쓴다(호스트, JSON만 읽음).

    python3 scripts/_review_ring_followup_report.py

미완료 단계는 pending으로 남긴다. 수치는 각 단계의 결과 파일에서만 읽는다.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/pdv2_reproduction_orchestration_20261106/ring"
SCRIPTS = ["_review_ring_followup_sampling.py", "_review_ring_followup_sensor_frame.py",
           "_review_ring_followup_consistency.py", "_review_ring_followup_kernels.py",
           "_review_ring_followup_official.py", "_review_ring_followup_score.py",
           "_review_ring_followup_verify.py", "_review_ring_followup_radon_replica.py",
           "_review_ring_followup_run.sh", "_review_ring_followup_report.py"]
COMMANDS = [
    "호스트: python3 scripts/_review_ring_followup_sampling.py --nclt-root /mnt/d/NSD_datasets/nclt",
    "호스트: python3 scripts/_review_ring_followup_sensor_frame.py --nclt-root /mnt/d/NSD_datasets/nclt",
    "호스트: python3 scripts/_review_ring_followup_consistency.py",
    "서버 컨테이너: python3 scripts/_review_ring_followup_score.py freeze (2026-11-06T10:37:54Z)",
    "서버 호스트: setsid nohup bash scripts/_review_ring_followup_run.sh > results/pdv2_reproduction_orchestration_20261106/ring/runner.log 2>&1 &",
    "  └ score.py run → verify.py score → (flock GPU0) kernels.py → (flock GPU0) official.py run ×6 → official.py summarize → verify.py official",
    "서버 컨테이너(CPU 1스레드): python3 scripts/_review_ring_followup_radon_replica.py",
    "호스트: python3 scripts/_review_ring_followup_report.py",
]


def load(rel: str):
    """결과 JSON을 읽는다. 없으면 None."""
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def pct(x) -> str:
    """비율을 백분율 문자열로."""
    return "—" if x is None else f"{100 * x:.3f}"


def main() -> None:
    """모든 단계를 요약한다."""
    sampling, sensor, consistency = load("sampling/report.json"), load("sensor_frame.json"), load("paper_consistency.json")
    protocol, score, score_ver = load("score_c1/protocol.json"), load("score_c1/summary.json"), load("score_c1/verification.json")
    kernels, off_sum, off_ver = load("kernels/report.json"), load("official_eval/summary.json"), load("official_eval/verification.json")
    replica = load("kernels/radon_replica.json")
    c1_rows = []
    if score and "elapsed_s" in score:
        for tag, v in score["variants"].items():
            for r in ("5.0", "10.0", "15.0", "20.0"):
                a, s = v["official_abs_sum"][r], v["paper_signed_sum"][r]
                c1_rows.append({"condition": tag, "radius_m": float(r), "queries": a["queries"],
                                "positive_queries": a["positive_queries"],
                                "official_R@1_abs": a["official_R@1"], "official_R@1_signed": s["official_R@1"],
                                "positive_only_R@1_abs": a["positive_only_R@1"],
                                "positive_only_R@1_signed": s["positive_only_R@1"],
                                "official_grid_F1_abs": a["official_grid_max_F1"],
                                "official_grid_F1_signed": s["official_grid_max_F1"],
                                "gridfree_F1_abs": a["gridfree_official_PR_max_F1"],
                                "gridfree_F1_signed": s["gridfree_official_PR_max_F1"],
                                "official_grid_AUC_abs": a["official_grid_PR_AUC"],
                                "official_grid_AUC_signed": s["official_grid_PR_AUC"],
                                "positive_wrong_over_20m_abs": a["positive_wrong_over_20m"],
                                "positive_wrong_over_20m_signed": s["positive_wrong_over_20m"],
                                "paper_R@1": v["paper_R@1"]})
    report = {
        "worker": "ring", "task": "RING/RING++ 원 논문 NCLT R@1 미재현 원인 추가 분석·원인 대조",
        "scope": "공개 코드·공개 입력 실행 진단. 우리 분할 재학습·주표·main 코드와 무관. 설정 선택 없음",
        "inputs": {"raw_scans": 1413, "sets": {"test": {"map": 64, "query": 259}, "all": {"map": 254, "query": 1054}},
                   "pair": "NCLT 2012-02-04 지도(20m) → 2012-03-17 질의(5m), 20m 필터"},
        "commands": COMMANDS,
        "script_sha256": {s: hashlib.sha256((REPO / "scripts" / s).read_bytes()).hexdigest()
                          for s in SCRIPTS if (REPO / "scripts" / s).exists()},
        "stages": {
            "sampling_reconstruction": sampling and {
                "official_as_is_matches_reference": sampling["official_as_is_matches_reference"],
                "tolerance_fix_rejected_scans": {k: sampling["conditions"]["tolerance_1s_microseconds"]["all"][k]["rejected_scans"]
                                                 for k in ("map", "query")},
                "tolerance_fix_changes_selection": not all(
                    sampling["conditions"]["tolerance_1s_microseconds"][s][k]["timestamps_identical"]
                    for s in ("test", "all") for k in ("map", "query")),
                "csv_first_row_note": sampling.get("csv_first_row_note")},
            "sensor_frame": sensor and {
                "elevation_deg_range": [min(r["fitted_elevation_deg_min_max"][0] for r in sensor["rows"]),
                                        max(r["fitted_elevation_deg_min_max"][1] for r in sensor["rows"])],
                "beam_origin_z_m": sorted({round(r["fitted_beam_origin_z_m_median"], 3) for r in sensor["rows"]}),
                "fraction_kept_by_official_band": [round(r["fraction_kept_by_official_band"], 3) for r in sensor["rows"]]},
            "paper_consistency": consistency and {
                "nclt_violations_all_denominator": consistency["nclt_table_II_violations_all_denominator"],
                "nclt_positive_fraction_upper_bound": consistency["nclt_positive_fraction_upper_bound"],
                "released_sets_positive_fraction": {k: v["positive_fraction"]
                                                    for k, v in consistency["released_generator_sets_10m"].items()},
                "mulran_violations": sorted(m for m, v in consistency["table_II"]["MulRan"].items()
                                            if v["violates_all_denominator_reading"]),
                "oxford_violations": sorted(m for m, v in consistency["table_II"]["Oxford"].items()
                                            if v["violates_all_denominator_reading"])},
            "kernels": kernels["summary"] if kernels else "pending",
            "torch_radon_replica": ({"scans": len(replica["rows"]), **replica["summary"]} if replica else "pending"),
            "official_evaluate_as_is": ({"rows": off_sum["rows"], "verification_passed": off_ver and off_ver["passed"],
                                         "deterministic_identical": off_ver and off_ver.get(
                                             "deterministic_conditions_identical_to_adapter")}
                                        if off_sum else "pending"),
            "c1_signed_correlation": ({"protocol_frozen_utc": protocol and protocol["frozen_utc"],
                                       "protocol_sha256": score.get("protocol_sha256"),
                                       "rows": c1_rows,
                                       "verification_passed": score_ver and score_ver["passed"],
                                       "verification": score_ver and {k: v["direct_vs_fft_max_abs_diff"]
                                                                      for k, v in score_ver["conditions"].items()}}
                                      if c1_rows else "pending"),
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    narrative = load("narrative.json")
    if narrative:
        report["conclusions"] = narrative
    (ROOT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    lines = ["RING/RING++ 원 논문 NCLT 수치 미재현 원인 후속 진단 (worker ring)",
             f"작성 {report['created_utc']}", "",
             "[입력] NCLT 2012-02-04 지도 20m → 2012-03-17 질의 5m, 20m 필터. 원시 1,413스캔. "
             "test 64지도·259질의, all 254지도·1,054질의. 논문 R@1: RING++ 73.21%(표II·III·VI, 표III·VI가 03-17→02-04 명시), "
             "RING 70.98%(표II에만 있음, 날짜 쌍 미기재).", ""]
    if c1_rows:
        lines.append("[C1 사전고정 원인대조: 공식 Σω|corr| → 논문 식12 부호 있는 합, 10m]")
        lines.append("조건 | 전체분모 R@1 공식→부호 | 양성분모 R@1 공식→부호 | 20m밖 오검색 공식→부호")
        for r in c1_rows:
            if r["radius_m"] == 10.0:
                lines.append(f"{r['condition']} | {pct(r['official_R@1_abs'])} → {pct(r['official_R@1_signed'])} | "
                             f"{pct(r['positive_only_R@1_abs'])} → {pct(r['positive_only_R@1_signed'])} | "
                             f"{r['positive_wrong_over_20m_abs']} → {r['positive_wrong_over_20m_signed']}")
        lines.append(f"검산 통과: {score_ver and score_ver['passed']}")
        lines.append("")
    if off_sum:
        lines.append("[공식 evaluate() 그대로 재실행 vs 기존 어댑터, 10m]")
        for r in off_sum["rows"]:
            lines.append(f"{r['condition']} | 공식 R@1 {pct(r['official_R@1_10m'])} | 어댑터 {pct(r['adapter_R@1_10m'])} | "
                         f"1위 일치 {r['top1_agreement_vs_adapter']:.4f} | TE q50/75/95 "
                         f"{'/'.join(f'{x:.3f}' for x in r['TE_q_correct_10m'])} m | "
                         f"RE {'/'.join(f'{x:.3f}' for x in r['RE_q_correct_10m'])}°")
        lines.append("논문 표VI 120×120 RING++: R@1 73.21%, TE 0.56/0.77/1.31 m, RE 1.25/2.14/4.07°")
        lines.append(f"검산 통과: {off_ver and off_ver['passed']}")
        lines.append("")
    if kernels:
        k = kernels["summary"]
        lines.append(f"[커널 대조] torch_radon vs skimage 최소 상관 {k['radon_min_best_corr']:.6f}, 최대 상대 L2 "
                     f"{k['radon_max_best_rel_l2']:.4f}, 질량 보존 최대 상대오차 {k['projection_mass_rel_err_max']:.2e}, "
                     f"TING 캐시 비트일치 {k['tiring_cache_all_bitwise_equal']}, 점특징 상대오차 99.9% 최대 "
                     f"{k['feature_rel_diff_q999_max']:.2e}")
        if replica:
            q = replica["summary"]["quantized"]
            lines.append(f"[torch_radon 커널 numpy 복제 vs GPU TING 캐시] {len(replica['rows'])}스캔, 최대 상대 L2 "
                         f"{q['max_rel_l2']:.2e}, 최소 상관 {q['min_corr']:.9f}")
        lines.append("")
    if narrative:
        lines.append("[결론]")
        lines += [f"- {t}" for t in narrative.get("ko", [])]
    (ROOT / "report_ko.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
