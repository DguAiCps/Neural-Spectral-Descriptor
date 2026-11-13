#!/usr/bin/env python3
"""저장된 후보에서 공통 3D 정답에 대한 분류기·간선 품질을 설명한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pdv2_edge_labels3d as exp
from _pdv2_no_context_ablation import sha256, write_json


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """같은 3D 양성·음성 행에서 분류기 순위 지표를 계산한다."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    assert np.isfinite(scores).all()
    return {
        "rows": len(labels),
        "positive": int(labels.sum()),
        "negative": int((~labels).sum()),
        "AP": float(average_precision_score(labels, scores)),
        "AUC": float(roc_auc_score(labels, scores)),
    }


def main() -> None:
    """설정 선택 없이 후보 누락·선택 손실과 공통 정답의 품질을 집계한다."""
    import torch

    exp.protocol(True)
    torch.set_num_threads(1)
    artifacts = {}
    models = {}
    for name in exp.MODELS:
        targets = []
        predictions = {arm: [] for arm in ("original", "xy", "xyz")}
        coverage = {group: {"queries": 0, "candidate30": 0, "original": 0,
                            "xy": 0, "xyz": 0, "xyz_recovered_vs_xy": 0,
                            "xyz_regressed_vs_xy": 0}
                    for group in ("same_available", "opposite_only", "other_positive")}
        for sensor, seq in exp.VAL:
            path = exp.folder(name) / f"edges_{sensor}_{seq}.npz"
            quality = exp.read(exp.folder(name) / f"quality_{sensor}_{seq}.json")
            artifacts[str(path.relative_to(REPO))] = sha256(path)
            assert artifacts[str(path.relative_to(REPO))] == quality["edge_sha256"]
            with np.load(path) as z:
                q, d, valid, groups = z["query"], z["distance_3d"], z["valid"], z["group"]
                keep = valid[q] & ((d < 5) | (d >= 25))
                targets.append(d[keep] < 5)
                true_nodes = {"candidate30": np.unique(q[d <= 5])}
                for arm in predictions:
                    predictions[arm].append(z["prob_" + arm][keep])
                    chosen = z["selected_" + arm]
                    true_nodes[arm] = np.unique(q[chosen][d[chosen] <= 5])
                for code, group in enumerate(coverage):
                    mask = valid & (groups == code)
                    record = coverage[group]
                    record["queries"] += int(mask.sum())
                    for arm, nodes in true_nodes.items():
                        record[arm] += int(mask[nodes].sum())
                    for field, before, after in (
                        ("xyz_recovered_vs_xy", "xy", "xyz"),
                        ("xyz_regressed_vs_xy", "xyz", "xy"),
                    ):
                        changed = np.setdiff1d(true_nodes[after], true_nodes[before])
                        record[field] += int(mask[changed].sum())
        y = np.concatenate(targets)
        models[name] = {
            "classifier_3d": {arm: ranking_metrics(y, np.concatenate(parts))
                              for arm, parts in predictions.items()},
            "true_neighbor_coverage": coverage,
        }

    # 학습 분할 안의 확인용 두 시퀀스도 세 분류기를 같은 3D 정답으로 비교한다.
    original = exp.old.alignment.base.model(1, device="cpu")
    classifiers = {"original": original[2:4],
                   **{arm: exp.load_classifier(arm, "cpu") for arm in ("xy", "xyz")}}
    targets = []
    predictions = {arm: [] for arm in classifiers}
    for sequence in exp.CFG["holdout"]:
        path = exp.OUT / "features" / (sequence.replace("/", "_") + ".npz")
        artifacts[str(path.relative_to(REPO))] = sha256(path)
        assert artifacts[str(path.relative_to(REPO))] == exp.read(path.with_suffix(".json"))["sha256"]
        with np.load(path) as z:
            labels = z["labels_xyz"]
            keep = labels >= 0
            targets.append(labels[keep] == 1)
            x = z["features"][keep]
        for arm, (clf, blob) in classifiers.items():
            with torch.no_grad():
                chunks = [torch.sigmoid(clf(torch.from_numpy(
                    (part - blob["mu"]) / blob["sd"]).float())).numpy()
                    for part in np.array_split(x, max(1, (len(x) + 65535) // 65536))]
            predictions[arm].append(np.concatenate(chunks))
    y = np.concatenate(targets)
    write_json(exp.OUT / "quality_diagnostics.json", {
        "posthoc_description_only": True,
        "used_for_selection": False,
        "test_used": False,
        "scope": "검증 후보의 사후 설명 집계. 학습·추론·최종 지표를 변경하지 않는다.",
        "common_target": "모든 분류기에 같은 3D 정답: <5 m 양성, >=25 m 음성, 중간 제외. 노드의 정답 이웃 존재는 최종 평가의 <=5 m를 쓴다.",
        "models": models,
        "holdout_common_3d": {arm: ranking_metrics(y, np.concatenate(parts))
                              for arm, parts in predictions.items()},
        "input_sha256": artifacts,
        "protocol_sha256": sha256(exp.OUT / "protocol.json"),
        "script_sha256": sha256(Path(__file__)),
    })
    print("DIAGNOSED", flush=True)


if __name__ == "__main__":
    main()
