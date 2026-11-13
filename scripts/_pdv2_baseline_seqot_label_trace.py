#!/usr/bin/env python3
"""SeqOT의 원래 overlap부터 실제 선택까지 학습 신호가 소실되는 위치를 추적한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/pdv2_baseline_common_diagnosis_20261106"


def main() -> None:
    """고정 overlap 및 라벨을 읽으며 입력·모델·라벨 파일은 바꾸지 않는다."""
    import _seqot_official_prep as p
    import _seqot_official_train as s

    indices = s.load_train_indices()
    source_hashes, rows = {}, {}
    for d, (sensor, sequence) in enumerate(s.TRAIN):
        home = p.label_dir(sensor, sequence)
        with np.load(home / "frames.npz") as z:
            poses, nr = z["poses"], int(z["n_raw"])
        i, j, overlap = p.load_explicit(sensor, sequence, "code")
        raw_positive = (overlap > .3) & (i != j)
        raw_counts = np.bincount(i[raw_positive], minlength=len(poses))
        incoming_counts = np.bincount(j[raw_positive], minlength=len(poses))
        f1, f2, label = indices[d]
        norm_positive = (label > .3) & (f1 != f2)
        norm_anchors = np.unique(f1[norm_positive] // 5)
        plan = s.anchor_plan([indices[d]], lambda _, a: 1 <= a < nr - 1, *s.USE1)
        anchors = np.array([a // 5 for _, a, _, _ in plan])
        self_only = np.array([all(x == a for x in pos) for _, a, pos, _ in plan])
        has_raw = raw_counts[anchors] > 0
        has_normalized = np.isin(anchors, norm_anchors)
        counts = {"anchors": len(plan), "self_only_steps": int(self_only.sum()),
                  "self_only_no_nonself_raw_overlap": int((self_only & ~has_raw).sum()),
                  "self_only_raw_exists_but_not_sampled_into_index": int((self_only & has_raw & ~has_normalized).sum()),
                  "self_only_nonself_index_exists_but_first_positive_selection_misses": int((self_only & has_normalized).sum())}
        assert sum(counts[k] for k in counts if k.startswith("self_only_") and k != "self_only_steps") == counts["self_only_steps"]
        counts["no_outgoing_positive_but_existing_incoming_positive"] = int((self_only & ~has_raw & (incoming_counts[anchors] > 0)).sum())
        counts["no_nonself_positive_in_either_raw_direction"] = int((self_only & ~has_raw & (incoming_counts[anchors] == 0)).sum())
        # 5m 기하 근방의 양성 여부. 투영을 건너뛴 먼 쌍은 이 근방에 존재하지 않아야 한다.
        pairs = cKDTree(poses[:, :3, 3]).query_pairs(5, output_type="ndarray")
        key = i.astype(np.int64) * len(poses) + j
        nearby_keys = pairs[:, 0] * len(poses) + pairs[:, 1]
        loc = np.searchsorted(key, nearby_keys)
        assert np.all(loc < len(key)) and np.array_equal(key[loc], nearby_keys)
        near_overlap = overlap[loc]
        counts.update({"near5m_nonself_pairs": len(pairs),
                       "near5m_overlap_positive_fraction": float((near_overlap > .3).mean()),
                       "near5m_overlap_negative_fraction": float((near_overlap < .3).mean()),
                       "near5m_overlap_quantiles": np.quantile(near_overlap, [0, .25, .5, .75, .95, 1]).tolist(),
                       "raw_nonself_positive_pairs": int(raw_positive.sum()),
                       "normalized_nonself_positive_rows": int(norm_positive.sum())})
        valid_pixels = np.full(len(poses), -1, dtype=np.int64)
        for shard in home.glob("ovc_*of*.npz"):
            with np.load(shard) as z:
                ids, first = np.unique(z["i"], return_index=True)
                valid_pixels[ids] = z["valid_cur"][first]
        assert np.all(valid_pixels >= 0)
        counts["range_image_valid_pixel_fraction_quantiles"] = np.quantile(valid_pixels / (32 * 900), [0, .25, .5, .75, .95, 1]).tolist()
        rows[f"{sensor}/{sequence}"] = counts
        for file in [home / "frames.npz", home / "normalized_data.npy", *home.glob("ovc_*of*.npz")]:
            source_hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
        print(sensor, sequence, counts, flush=True)
    keys = ["anchors", "self_only_steps", "self_only_no_nonself_raw_overlap",
            "self_only_raw_exists_but_not_sampled_into_index",
            "self_only_nonself_index_exists_but_first_positive_selection_misses",
            "no_outgoing_positive_but_existing_incoming_positive", "no_nonself_positive_in_either_raw_direction"]
    report = {"per_sequence": rows, "totals": {k: sum(v[k] for v in rows.values()) for k in keys},
              "scope": "학습 시퀀스8개, 같은 GT5m 거리 쌍의 공식 overlap 라벨과 실제 선택의 관계. 5m 자체를 정답 overlap의 대체로 채택한 실험은 아니다.",
              "sources_sha256": source_hashes, "checks_passed": True,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "seqot_label_trace.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
