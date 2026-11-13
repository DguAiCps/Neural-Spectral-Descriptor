#!/usr/bin/env python3
"""검산을 통과한 native 비교를 원고 검토용 JSON과 CSV로 정리한다."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/pdv2_review_native_20261105"
MODELS = ["author", "v2_s1", "v2_s2", "v2_s3"]


def read(relative: str) -> dict:
    """실제 완료 결과를 읽는다."""
    return json.loads((ROOT / relative).read_text())


def summary(values: list) -> dict:
    """3시드의 평균과 표본 표준편차를 백분율로 반환한다."""
    assert len(values) == 3
    return {"v2_mean_percent": float(100 * np.mean(values)),
            "v2_sample_std_pp": float(100 * np.std(values, ddof=1)),
            "v2_seeds_percent": [100 * float(v) for v in values]}


def main() -> None:
    """부분 결과를 논문 전체 벤치마크 평균과 구분해 저장한다."""
    verified = read("learned_independent_verification.json")
    rows = []
    if {r["model"] for r in verified["seqot"]} == set(MODELS):
        reports = [read(f"seqot/{m}/native/report.json") for m in MODELS]
        assert len({r["positive_queries"] for r in reports}) == 1
        for rank in ["1", "5", "20"]:
            rows.append({"method": "SeqOT", "condition": "NCLT Jan08 → Feb05 / 공개GT15m",
                         "metric": "R@" + rank, "positive_queries": reports[0]["positive_queries"],
                         "paper_percent": 100 * reports[0]["published"]["r" + rank],
                         "author_percent": 100 * reports[0]["official_recall"][rank],
                         **summary([r["official_recall"][rank] for r in reports[1:]]),
                         "scope": "공개 코드·공개 정답 재현; 논문의 overlap>0.3 정답과 다름"})
    valid_imlpr = {(r["model"], r["k"], r["sequence"]) for r in verified["imlpr"]
                   if r["inference_batch"] == 1}
    for k in [15, 25]:
        for sequence in ["DCC01", "DCC02", "DCC03"]:
            if not all((m, k, sequence) in valid_imlpr for m in MODELS):
                continue
            reports = [read(f"imlpr/k{k}/{m}/native_batch1_{sequence}.json") for m in MODELS]
            assert len({r["fixed_positive_denominator"] for r in reports}) == 1
            for metric, key in [("R@1", "r1"), ("공식 F1", "official_max_f1"),
                                ("고정 양성 분모 F1", "fixed_denominator_max_f1")]:
                paper_key = "r1" if key == "r1" else "f1"
                rows.append({"method": "ImLPR", "condition": f"{sequence} / k={k} / batch1",
                             "metric": metric, "positive_queries": reports[0]["fixed_positive_denominator"],
                             "paper_percent": None if key == "fixed_denominator_max_f1" else 100 * reports[0]["published"][paper_key],
                             "author_percent": 100 * reports[0][key],
                             **summary([r[key] for r in reports[1:]]),
                             "scope": "보유 원자료 공식 재생성; 저자 대전 학습과 native 대전 장소 겹침; v2 시험 장소 진단"})
    valid_helios = {(r["model"], r["database"], r["query"]) for r in verified["helios"]}
    # 모든 가중치에서 평가되는 동일 질의 교집합을 사용한다.
    pairs = sorted({(r["database"], r["query"]) for r in verified["helios"]})
    for database, query in pairs:
        if not all((m, database, query) in valid_helios for m in MODELS):
            continue
        name = database + "__" + query + ".npz"
        ledgers = [np.load(ROOT / "helios" / m / "native_pairs" / name) for m in MODELS]
        common = np.logical_and.reduce([d["evaluated"] for d in ledgers])
        assert common.any()
        values = [float(d["first_correct"][common].mean()) for d in ledgers]
        rows.append({"method": "HeLiOS", "condition": database + " → " + query,
                     "metric": "공통 질의 R@1", "positive_queries": int(common.sum()),
                     "paper_percent": None, "author_percent": 100 * values[0],
                     **summary(values[1:]),
                     "scope": "확보된 12/24쌍 개별 진단; v2 검증 장소; 논문 24쌍 그룹 평균 대체 불가"})
        for d in ledgers:
            d.close()
    for row in rows:
        row["author_minus_v2_mean_pp"] = row["author_percent"] - row["v2_mean_percent"]
    ring = json.loads((REPO / "results/pdv2_ringpp_maxfix_20261105/partial_verified.json").read_text())
    source_files = ["learned_independent_verification.json", "learned_native_scope.json",
                    "training_diagnostics.json", "helios/author_training_guidance.json",
                    "seqot/protocol_label_correction.json", "ring_protocol_clarification.json"]
    report = {"generated_kst": datetime.now().astimezone().isoformat(), "rows": rows,
              "standard_deviation": "3시드 표본 표준편차(ddof=1), 퍼센트포인트",
              "native_scope": read("learned_native_scope.json"),
              "ringpp_common46": {k: v for k, v in ring.items() if k != "per_sequence"},
              "sources_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in source_files},
              "training_conclusions": {
                  "HeLiOS": "80epoch 학습 기록상 train 상승·val 조기 최고 후 하락; 충분수렴 대신 과적합/분포차이 증거",
                  "SeqOT": "stage1 조기 선택·stage2 99epoch; 공개 native 저자/v2 격차와 공통 규약 효과를 분리",
                  "ImLPR": "100epoch·optimizer/RNG 복원 가능; 시드1 말기 검증 상승이라 충분수렴 미입증; 150epoch 연장은 주말 마감 우선 보류"},
              "claim_limits": [
                  "저자 가중치는 깨끗한 장소 분리 주표 대체용이 아니다.",
                  "native 값과 공통46 표2 값은 정답·질의·후보·집계가 달라 직접 순위 비교하지 않는다.",
                  "HeLiOS 누락 6개 배포 자료는 다운로드 할당량 제한이며 2개 Velodyne 원자료 재생성은 별도 경로다.",
                  "RING++ 공통46는 정확 max만 변경하며 추가 회전각 대조는 별도로 유지한다.",
                  "기존 표3 비용은 수정 max 구현의 측정값이 아니다."]}
    target = ROOT / "submission_evidence.json"
    temp = target.with_suffix(".tmp.json")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temp.replace(target)
    fields = ["method", "condition", "metric", "positive_queries", "paper_percent", "author_percent",
              "v2_mean_percent", "v2_sample_std_pp", "author_minus_v2_mean_pp", "scope"]
    with (ROOT / "submission_evidence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print({"rows": len(rows), "methods": sorted({r["method"] for r in rows}),
           "ringpp_completed": ring["completed_sequences"]})


if __name__ == "__main__":
    main()
