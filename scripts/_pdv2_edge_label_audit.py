#!/usr/bin/env python3
"""원래 간선 분류기 학습 자료의 XY 라벨을 3D 거리와 대조한다. 재학습 없음."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_edge_quality as exp
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """실제 분류기 생성 모델의 후보·라벨을 저장된 자료와 비교한다."""
    import torch
    from _lag_branch_fusion import lag_pass1_graph
    from _causal_edge_features import causal_edge_features
    from build_and_train_edge_classifier import range_phase_flat
    from run_kitti_operating_point import _normalize

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    exp.protocol(True)
    cfg, net, _, _, rvec, ckpt = exp.alignment.base.model(1)
    net.eval()
    saved_root = REPO / "results/_edge_cls_pdv2_lag15"
    sources = {Path(__file__), Path(ckpt), exp.alignment.base.classifier_path(1),
               REPO / "scripts/_pdv2_lag.py", REPO / "scripts/_causal_build_edge_classifier.py",
               REPO / "scripts/_causal_edge_features.py", REPO / "src/data/kitti_loader.py"}
    rows = []
    for seq in ("01", "02", "05", "06"):
        cache_path = exp.alignment.cache_path("kitti/" + seq)
        layout_path = exp.alignment.layout_path("kitti", seq)
        saved_path = saved_root / f"data_kitti_{seq}.npz"
        sources.update((cache_path, layout_path, saved_path))
        with np.load(cache_path) as z:
            cache = exp.ArrayCache({k: z[k] for k in z.files})
        with np.load(layout_path) as z:
            layout = z["nsd_layouts"].astype(np.float32)
        raw = cache["descriptors"].astype(np.float32)
        with torch.no_grad():
            first = net(lag_pass1_graph(cache, raw, cfg, "cuda", "kitti")).cpu().numpy()
        q, j, features, labels = causal_edge_features(
            first, _normalize(raw * rvec[None]), range_phase_flat(layout), cache["poses"])
        xyz = cache["poses"][:, :3, 3]
        distances = np.linalg.norm(xyz[q] - xyz[j], axis=1)
        keep = labels >= 0
        with np.load(saved_path) as z:
            saved_features, saved_labels = z["feats"], z["labels"]
        equal_shape = labels[keep].shape == saved_labels.shape
        # 문맥 출력·후보 순위와 무관한 특징으로 저장 행의 실제 쌍을 찾는다.
        # 부동소수점 반올림이나 근접 매칭 없이 정확히 같은 행만 인정한다.
        columns = [1, 8, 9, 10, 11, 12, 13]
        lookup = {}
        for i, feature in enumerate(features[:, columns]):
            lookup.setdefault(feature.tobytes(), []).append(i)
        matched = matched_positive = false_positive = far_positive = 0
        examples = []
        for feature, label in zip(saved_features[:, columns], saved_labels):
            found = lookup.get(feature.tobytes(), [])
            if len(found) != 1:
                continue
            i = found[0]
            assert label == labels[i], (seq, i, label, labels[i])
            matched += 1
            if label == 1:
                matched_positive += 1
                false_positive += int(distances[i] > 5)
                far_positive += int(distances[i] > 25)
                if distances[i] > 25 and len(examples) < 5:
                    examples.append({"query": int(q[i]), "neighbor": int(j[i]),
                                     "distance_xy": float(np.linalg.norm(xyz[q[i], :2] - xyz[j[i], :2])),
                                     "distance_3d": float(distances[i])})
        row = {
            "sequence": "kitti/" + seq, "classifier_role": "holdout" if seq == "06" else "fit",
            "xyz_span_m": np.ptp(xyz, axis=0).tolist(),
            "candidates": len(q), "kept_rows": int(keep.sum()),
            "xy_positive": int((labels == 1).sum()),
            "positive_with_3d_over_5m": int(((labels == 1) & (distances > 5)).sum()),
            "positive_with_3d_over_25m": int(((labels == 1) & (distances > 25)).sum()),
            "xy_negative": int((labels == 0).sum()),
            "saved_rows": len(saved_labels), "saved_positive": int(saved_labels.sum()),
            "saved_label_order_equal": bool(equal_shape and np.array_equal(labels[keep], saved_labels)),
            "saved_feature_max_error": float(np.max(np.abs(features[keep] - saved_features))) if equal_shape else None,
            "exact_pair_matching": {"feature_columns": columns, "unique_matched_rows": matched,
                                    "unique_matched_positive": matched_positive,
                                    "saved_positive_3d_over_5m_lower_bound": false_positive,
                                    "saved_positive_3d_over_25m_lower_bound": far_positive,
                                    "examples": examples},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    original_counts = {}
    for role in ("fit", "holdout"):
        positive = negative = 0
        for sensor, seq in exp.TRAIN:
            is_holdout = sensor + "/" + seq in {"kitti/06", "helipr/Bridge01"}
            if is_holdout != (role == "holdout"):
                continue
            path = saved_root / f"data_{sensor}_{seq}.npz"
            sources.add(path)
            with np.load(path) as z:
                positive += int((z["labels"] == 1).sum())
                negative += int((z["labels"] == 0).sum())
        original_counts[role] = {"positive": positive, "negative": negative,
                                 "negative_per_positive": negative / positive}
    write_json(exp.OUT / "label_audit.json", {
        "scope": "사후 원인 진단. 원래 시드1 분류기 생성 모델과 학습 캐시만 사용. 재학습·설정 선택·시험 없음.",
        "rows": rows, "original_classifier_counts": original_counts,
        "distance_rule": "기존 labels: XY<5m 양성, XY>=25m 음성. 진단은 동일 쌍의 3D 거리와 비교한다.",
        "score_caution": "양성 가중 BCE를 쓰므로 sigmoid 출력을 보정된 실제 정답 확률로 해석하지 않는다.",
        "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(sources)},
    })


if __name__ == "__main__":
    main()
