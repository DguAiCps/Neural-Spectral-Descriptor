#!/usr/bin/env python3
"""저장된 최종 원장에서 높은 정밀도 이득의 위치와 경계 오답을 확인한다."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / "results/pdv2_temporal_derivatives_20261020"
METHOD = "arc_unique_b100"


def main() -> None:
    """모델 선택을 바꾸지 않고 각 시퀀스의 첫 오수락을 대조한다."""
    summary = json.loads((OUT / "summary.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    models = summary["per_seed"]["1"]
    sources = {str(Path(__file__).relative_to(REPO)): sha256(Path(__file__))}
    for name in ("summary.json", "protocol.json", "completion_validation.json"):
        sources[str((OUT / name).relative_to(REPO))] = sha256(OUT / name)
    old_order = REPO / "results/pdv2_temporal_focus_20261018/s1/standard_original/batch_order.json"
    new_order = OUT / "s1/standard/training_complete.json"
    original = json.loads(old_order.read_text())["epochs"]
    repeated = json.loads(new_order.read_text())["orders"]
    assert len(original) == len(repeated) == 20
    for a, b in zip(original, repeated):
        assert a["sha256"] == b["triplets"]
        assert a["permutation_sha256"] == b["permutation"]
    for path in (old_order, new_order):
        sources[str(path.relative_to(REPO))] = sha256(path)
    result = {}
    for sequence in protocol["val"]:
        rows, entries, witnesses = {}, {}, set()
        for arm, model in models.items():
            path = OUT / f"s1/{arm}/epoch{model['selected']['epoch']}" / (
                sequence.replace("/", "_") + ".npz")
            sources[str(path.relative_to(REPO))] = sha256(path)
            with np.load(path) as z:
                a = {k: z[k] for k in z.files}
            rows[arm] = a
            pos = a["min_past_gt_dist"] <= 5
            correct = pos & (a["top1_geo_dist"] <= 5)
            bad = np.flatnonzero((pos | (a["min_past_gt_dist"] > 25)) & ~correct)
            score = a[METHOD]
            threshold = float(score[bad].max())
            critical = bad[score[bad] == threshold]
            witnesses.update(int(a["query_idx"][i]) for i in critical)
            tp = int((correct & (score > threshold)).sum())
            info = json.loads(path.with_suffix(".json").read_text())
            assert abs(tp / max(int(pos.sum()), 1) -
                       info["metrics"][METHOD]["R@100P"]) < 1e-14
            entries[arm] = {
                "epoch": model["selected"]["epoch"], "n_positive": int(pos.sum()),
                "correct_above_first_error": tp, "first_error_score": threshold,
                "critical_queries": [int(a["query_idx"][i]) for i in critical],
            }
        comparison = {}
        for q in sorted(witnesses):
            comparison[str(q)] = {}
            for arm, a in rows.items():
                ix = np.flatnonzero(a["query_idx"] == q)
                assert len(ix) == 1
                i = ix[0]
                comparison[str(q)][arm] = {
                    "winner": int(a["j_star"][i]), "score": float(a[METHOD][i]),
                    "winner_gt_distance": float(a["top1_geo_dist"][i]),
                    "minimum_gt_distance": float(a["min_past_gt_dist"][i]),
                    "correct": bool(a["min_past_gt_dist"][i] <= 5 and
                                    a["top1_geo_dist"][i] <= 5),
                }
        result[sequence] = {"models": entries, "critical_query_comparison": comparison}
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest
    write_json(OUT / "high_precision_diagnosis.json", {
        "per_sequence": result, "sources": sources, "test_used": False,
        "original_standard_training_order_exact": True,
        "scope": "시드1 선택 모델의 사후 검증 진단. 선택·학습·판정은 바꾸지 않는다.",
    })
    print("R100_VERIFIED", len(result) * len(models), flush=True)


if __name__ == "__main__":
    main()
