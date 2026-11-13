#!/usr/bin/env python3
"""동결된 감사의 기준 융합 재계산만 원래 float32 정밀도로 실행한다."""
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_candidate_verify as audit
from _pdv2_candidate_learning import OUT, read
from _pdv2_no_context_ablation import write_json, sha256
from _xs_eval_lib import minmax


def original_precision_minmax(values: np.ndarray) -> np.ndarray:
    """기존 평가의 float32 점수·동점 규칙을 그대로 재현한다."""
    assert np.array_equal(values, values.astype(np.float32))
    return minmax(values.astype(np.float32))


def main() -> None:
    """검사 수정의 이유를 먼저 기록하고 나머지 동결된 검사를 실행한다."""
    failure = OUT / "initial_verification_failure.json"
    if not failure.exists():
        write_json(failure, read(OUT / "status.json"))
    amendment = OUT / "verification_amendment.json"
    if not amendment.exists():
        write_json(amendment, {
            "created_unix": time.time(),
            "reason": "기존 평가의 융합은 float32다. 감사가 float64 저장 배열을 그대로 정규화하여 동점 후보 순서가 달라졌다.",
            "evidence": {"sequence": "helipr/Roundabout02", "kind": "gnn", "op": "pbev128_main",
                         "query": 3346, "candidates": [2295, 2297],
                         "original_float32_scores": [4.63260555267334, 4.63260555267334],
                         "audit_float64_scores": [4.6326054155067675, 4.632605568831771],
                         "fresh_static_matches_saved": True},
            "change": "기준 융합 감사의 minmax 입력만 원래 float32로 복원한다. 전 값이 float32로 정확히 표현 가능한지 먼저 확인한다.",
            "unchanged": "동결된 코드·설정·학습·후보·새 판정·체크포인트 선택·원장·최종 지표·검사 허용 오차. 원래 감사도 보존한다.",
            "protocol_sha256": sha256(OUT / "protocol.json"),
            "summary_sha256": sha256(OUT / "summary.json"),
            "wrapper_sha256": sha256(Path(__file__)),
        })
    write_json(OUT / "status.json", {"status": "running", "stage": "corrected_verification",
                                      "validation_jobs": 108, "test_used": False})
    audit.minmax = original_precision_minmax
    audit.main()
    result = read(OUT / "completion_validation.json")
    result["verification_amendment_sha256"] = sha256(amendment)
    result["wrapper_sha256"] = sha256(Path(__file__))
    write_json(OUT / "completion_validation.json", result)
    write_json(OUT / "status.json", {"status": "complete", "stage": "complete", "test_used": False,
                                      "validation_jobs": 108, "finished_unix": time.time(),
                                      "verification_amendment": str(amendment.relative_to(REPO))})


if __name__ == "__main__":
    main()
