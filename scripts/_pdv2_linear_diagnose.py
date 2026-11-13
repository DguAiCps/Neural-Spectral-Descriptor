#!/usr/bin/env python3
"""선택을 바꾸지 않고 최종 후보 선택과 판정 점수의 역할을 나눈다."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_linear_fusion import OUT, Scorer, fusion_features, protocol, base
from _pdv2_no_context_ablation import write_json, sha256
import _pdv2_candidate_diagnose as diagnosis


def main() -> None:
    """같은 2×2 사후 진단에 가중합만 주입한다."""
    protocol()
    diagnosis.CandidateScorer, diagnosis.pair_features = Scorer, fusion_features
    diagnosis.main()
    path = OUT / "ranking_confidence_diagnosis.json"
    result = base.read(path)
    result["wrapper_sha256"] = sha256(Path(__file__))
    result["protocol_sha256"] = sha256(OUT / "protocol.json")
    write_json(path, result)


if __name__ == "__main__":
    main()
