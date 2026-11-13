#!/usr/bin/env python3
"""같은 검증 선택 체크포인트의 공통시험/native 진단을 해시로 연결한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105"
SELECTION = REPO / "results/pdv2_review_selection_full_20261105"


def sha256(path: Path) -> str:
    """원천 결과의 전체 파일 해시를 반환한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """이미 독립 검산된 결과만 결합하며 선택이나 추론을 변경하지 않는다."""
    paths = [SELECTION / "final_delivery_seqot.json",
             SELECTION / "selection_delivery_seqot.json",
             OUT / "seqot/common_selection_native/independent_verification.json"]
    common, frozen, native = [json.loads(p.read_text()) for p in paths]
    assert common["status"] == "complete_for_retained_candidate_scope"
    assert native["complete"] and native["positive"] == 5336
    assert frozen["test_used_for_selection"] is False
    assert common["selection_sha256"] == frozen["selection_sha256"]
    matches = []
    for row in native["rows"]:
        c = row["candidate"]
        seed = str(c["seed"])
        assert c["stage1_sha256"] == frozen["models"][seed]["fixed_stage1"]["sha256"]
        for objective in c["selection_objectives"]:
            selected = common["selected_checkpoints"][seed][objective]
            assert c["stage2_sha256"] == selected["sha256"]
            assert c["stage2_tensor_sha256"] == selected["model_tensor_sha256"]
            assert c["epoch"] == selected["epoch"]
            matches.append({"seed": seed, "objective": objective, "epoch": c["epoch"],
                            "stage1_sha256": c["stage1_sha256"],
                            "stage2_sha256": c["stage2_sha256"]})
    assert len(matches) == 6
    rows = {}
    for objective, name in [("AP", "AP_selection"), ("R@1", "R1_selection")]:
        c = common["rows"][name]
        for metric, values in c.items():
            xs = list(values["per_seed"].values())
            assert abs(statistics.mean(xs) - values["mean"]) < 1e-12
            assert abs(statistics.stdev(xs) - values["sample_sd"]) < 1e-12
        ns = [100 * r["hits"]["1"] / native["positive"] for r in native["rows"]
              if objective in r["candidate"]["selection_objectives"]]
        assert len(ns) == 3
        n = {"mean": statistics.mean(ns), "sample_sd": statistics.stdev(ns)}
        assert abs(n["mean"] - 100 * native["groups"][objective]["1"]["mean"]) < 1e-12
        rows[objective] = {"common46_percent": c, "native_R@1_percent": n}
    changes = {"common46": {k: rows["AP"]["common46_percent"][k]["mean"] - rows["R@1"]["common46_percent"][k]["mean"]
                            for k in rows["AP"]["common46_percent"]},
               "native_R@1": rows["AP"]["native_R@1_percent"]["mean"] - rows["R@1"]["native_R@1_percent"]["mean"]}
    report = {"passed": True, "created_unix": time.time(), "matched_models": matches,
              "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in paths},
              "rows": rows, "AP_minus_R1_selection_pp": changes,
              "selection_sha256": common["selection_sha256"],
              "limits": ["같은 stage1을 고정한 stage2 선택 민감도다. 전체2단계 수렴을 입증하지 않는다.",
                         "공통46은195726질의·61682양성·5m·과거후보·6도시평균; native는공개15m·5336양성의교차세션이다.",
                         "서로 다른 평가의 R@1 절대값 차이를 단일요인의 효과라고 계산하지 않는다.",
                         "기존 시험셋이 알려진 사후 민감도이며 AP/R@1 선택 자체는 고정 공통검증만 사용했다."]}
    (OUT / "seqot/selection_common_native_bridge.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    text = "\\paragraph{The same selected checkpoints under two evaluation protocols.}\n"
    text += "We match all six stage-two checkpoints by file and tensor hashes across the common test and released-protocol diagnostic, while keeping stage one fixed. Choosing stage two by common-validation AP rather than common-validation R@1 changes common-test AP by "
    text += f"${changes['common46']['AP']:+.3f}$ percentage points and common-test R@1 by ${changes['common46']['R@1']:+.3f}$ points. "
    text += f"The same choice changes native R@1 by ${changes['native_R@1']:+.3f}$ points. "
    text += "These are within-protocol sensitivity results: the common benchmark uses past-only 5\\,m matches and six-city averaging, whereas the native diagnostic uses cross-session released 15\\,m labels. They do not establish convergence of stage one or attribute the absolute gap between protocols to checkpoint selection alone.\n"
    (OUT / "paper_proposal/seqot_common_native_bridge.tex").write_text(text)
    print(json.dumps({"passed": True, "models": len(matches), "changes_pp": changes}))


if __name__ == "__main__":
    main()
