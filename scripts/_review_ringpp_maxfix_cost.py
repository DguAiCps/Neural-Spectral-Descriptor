#!/usr/bin/env python3
"""정확max RING++의 표3 비용을 기존 공통 규약으로 측정한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _pdv2_table3_benchmark as bench
import _pdv2_table3_ring as original
import _pdv2_table3_ringpp_v3 as measured
import _review_native_ring_maxfix as fix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pdv2_seqot_ring_repair_20261106/ringpp_cost"


def configure() -> dict:
    """원래 특징·검색·비용 규약을 유지하고 max 수정만 연결한다."""
    original.configure()
    config = measured.settings()
    config["schema"] = "pdv2-table3-ringpp-exactmax-v1"
    config["base_cache_protocol"]["encoder"] += "; grid maximum replaced by verified np.fmax.at"
    config["base_cache_protocol"]["sources"].update({
        str(Path(fix.__file__).relative_to(ROOT)): bench.sha(Path(fix.__file__)),
        str(Path(__file__).relative_to(ROOT)): bench.sha(Path(__file__)),
    })
    config["reason"] = "확인된 비원자 max 오류를 고친 동일46시험 구현의 실제 비용. 성능에 따른 설정 선택 없음."
    config["unchanged"] = "공통Sejong01·초기1만 후보·96도착×3·CPU1·RTX5080·실제점군 인코딩·같은점특징/라돈/상관/청크128"
    config["verification"] = "새 특징의 CPU max 정확성·모든288질의 공식상관 대조·3반복 특징 및 승자 동일성"
    config["repeat_variation"] = "수정된 격자 집계는 결정적이어야 함. 반복의 특징/승자/점수 변동 검사"
    config["external_memory_basis"] = "GPUFeatureExtractor는 그대로여서 그 경로의360바이트/호출 미해제 요청을 기존 계측대로 합산"
    config["matching_test_evidence"] = "results/pdv2_ringpp_maxfix_20261105/summary_verified.json"
    OUT.mkdir(parents=True, exist_ok=True)
    bench.write(OUT / "protocol.json", bench.read(bench.OUT / "protocol.json"))
    original.OUT = OUT
    measured.OUT = OUT
    measured.PROTOCOL = OUT / "ringpp_exactmax_protocol.json"
    original.settings = lambda variant: config["base_cache_protocol"]
    measured.settings = lambda: config
    fix.install_maxfix()
    return config


def main() -> None:
    """지도를 준비한 뒤 같은 도착을 세 번 처리하고 결과를 검산한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cache", "measure", "all"))
    parser.add_argument("--n-map", type=int, default=10000)
    args = parser.parse_args()
    configure()
    if args.stage in ("cache", "all"):
        measured.cache(args.n_map)
    if args.stage in ("measure", "all"):
        measured.measure(args.n_map)
        path = OUT / "methods" / f"ringpp_n{args.n_map}.json"
        result = bench.read(path)
        variation = result["official_native_repeat_variation"]
        assert variation["winner_changed_queries"] == 0
        assert variation["max_confidence_spread"] == 0
        assert all(v["max_abs_vs_repeat0"] == 0 for v in variation["raw_feature_changes"])
        result["implementation"] = "released point features + verified exact grid max"
        result["implementation_source_sha256"] = bench.sha(Path(fix.__file__))
        result["wrapper_source_sha256"] = bench.sha(Path(__file__))
        result["same_test_implementation"] = True
        bench.write(path, result)
        print(json.dumps({"passed": True, "timings_ms": result["timings_ms"],
                          "resident_bytes": result["resident_bytes"]["total"]}), flush=True)


if __name__ == "__main__":
    main()
