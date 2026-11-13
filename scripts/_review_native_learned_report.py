#!/usr/bin/env python3
"""독립 검산이 끝난 native 증거와 공개/자체학습 대조만 모은다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    verified = read(OUT / "learned_independent_verification.json")
    result = {"scope": read(OUT / "learned_native_scope.json"), "complete": False}
    seqot = {}
    for row in verified["seqot"]:
        model = row["model"]
        seqot[model] = read(OUT / "seqot" / model / "native/report.json")
    result["seqot"] = {"completed_models": len(seqot), "required_models": 4, "models": seqot,
                       "comparison_scope": "저자공개GT15m와같은실행.논문overlap정답동일재현아님"}
    pairs = {}
    for row in verified["helios"]:
        model = row["model"]
        if model not in pairs:
            root = OUT / "helios" / model
            path = root / "native_report.json"
            if not path.exists():
                path = root / "native_partial_report.json"
            pairs[model] = read(path)["pairs"]
    comparisons = []
    for model, items in pairs.items():
        if model == "author":
            continue
        for pair in items:
            db, query = pair["database"], pair["query"]
            name = f"{db}__{query}.npz"
            author = np.load(OUT / "helios/author/native_pairs" / name)
            current = np.load(OUT / "helios" / model / "native_pairs" / name)
            common = author["evaluated"] & current["evaluated"]
            comparisons.append({"model": model, "database": db, "query": query,
                                "same_positive_queries": int(common.sum()),
                                "author_r1": float(author["first_correct"][common].mean()),
                                "v2_r1": float(current["first_correct"][common].mean()),
                                "excluded_due_to_different_candidate_filter": int((author["evaluated"] ^ current["evaluated"]).sum())})
    result["helios"] = {"models": pairs, "paired_same_queries": comparisons,
                        "complete_24_pairs_per_model": {k: len(v) == 24 for k, v in pairs.items()},
                        "published_group_means": "24쌍전부미완료시계산안함"}
    imlpr = []
    for row in verified["imlpr"]:
        if row["inference_batch"] != 1:
            continue
        path = OUT / f"imlpr/k{row['k']}" / row["model"] / f"native_batch1_{row['sequence']}.json"
        imlpr.append(read(path))
    result["imlpr"] = {"completed_conditions": len(imlpr), "required_conditions": 24, "rows": imlpr,
                       "f1_caveat": "공식FN은정답검색결과의거절만세므로공통retrievalF1과다름.fixed_denominator_max_f1별도제공"}
    result["common_radius_factor"] = {method: value["mean"] for method, value in read(OUT / "common_radius_factor_report.json")["methods"].items()}
    result["training_diagnostics"] = read(OUT / "training_diagnostics.json")
    result["remaining"] = {"ringpp_common46": read(REPO / "results/pdv2_ringpp_maxfix_20261105/partial_verified.json"),
                           "learned": "4모델×native 전체조건,HeLiOS다운로드누락조건,Velodyne원자료재생성대조를따로추적"}
    target = OUT / "learned_integrated_report.json"
    temp = target.with_suffix(".tmp.json")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temp.replace(target)
    print({"seqot_models": len(seqot), "helios_pairs": sum(len(v) for v in pairs.values()), "imlpr_batch1_conditions": len(imlpr)})
    from _review_native_submission_evidence import main as submission_evidence
    submission_evidence()


if __name__ == "__main__":
    main()
