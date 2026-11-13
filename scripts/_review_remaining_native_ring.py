#!/usr/bin/env python3
"""보존한 RING 상관에서 분모·오류 유형·자세 대응을 독립 확인한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/pdv2_review_native_20261105"
OUT = REPO / "results/pdv2_remaining_native_20261106/ring"


def main() -> None:
    """검색을 바꾸지 않고 기존 수치 차이의 설명 가능한 범위를 잰다."""
    OUT.mkdir(parents=True, exist_ok=True)
    sets = json.loads((BASE / "evaluation_sets.json").read_text())
    rows, source_hashes = [], {}
    for region, groups in sets.items():
        query = np.array([x["xy"] for x in groups["query"]], dtype=np.float64)
        maps = np.array([x["xy"] for x in groups["map"]], dtype=np.float64)
        geo = np.linalg.norm(query[:, None] - maps[None], axis=2)
        positive = geo.min(axis=1) <= 10
        for variant, folder, key, reference in [
            ("ring", BASE, "ring", .7098),
            ("ringpp_released", BASE, "ringpp", .7321),
            ("ringpp_exactmax", BASE / "maxfix", "ringpp", .7321),
        ]:
            path = folder / f"pair_dists_{region}_{key}.npy"
            pairs = np.load(path)
            source_hashes[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
            rank = np.argsort(pairs, axis=1)
            selected_geo = geo[np.arange(len(query)), rank[:, 0]]
            correct = selected_geo <= 10
            error = positive & ~correct
            first_positive_rank = np.argmax(np.take_along_axis(geo <= 10, rank, axis=1), axis=1) + 1
            both = {"queries": len(query), "positive_queries": int(positive.sum()),
                    "correct": int(correct.sum()), "no_positive": int((~positive).sum()),
                    "positive_wrong": int(error.sum()),
                    "positive_wrong_10_to_20m": int((error & (selected_geo <= 20)).sum()),
                    "positive_wrong_over_20m": int((error & (selected_geo > 20)).sum())}
            assert both["correct"] + both["no_positive"] + both["positive_wrong"] == len(query)
            rows.append({"region": region, "variant": variant, **both,
                         "R1_all_percent": 100 * float(correct.mean()),
                         "R1_positive_only_percent": 100 * float(correct[positive].mean()),
                         "denominator_effect_pp": 100 * float(correct[positive].mean() - correct.mean()),
                         "paper_R1_percent": 100 * reference,
                         "remaining_gap_positive_only_pp": 100 * float(reference - correct[positive].mean()),
                         "positive_R5_percent": 100 * float((first_positive_rank[positive] <= 5).mean()),
                         "positive_R25_percent": 100 * float((first_positive_rank[positive] <= 25).mean())})
    pose_checks = {}
    for seq in ["2012-02-04", "2012-03-17"]:
        path = Path("/mnt/d/NSD_datasets/nclt") / f"groundtruth_{seq}.csv"
        gt = np.loadtxt(path, delimiter=",", usecols=(0, 1, 2), skiprows=1)
        raw = {x["timestamp"]: x for group in sets.values() for elements in group.values()
               for x in elements if x["raw_path"].startswith(seq)}
        timestamps = np.array(sorted(raw), dtype=np.int64)
        j = np.clip(np.searchsorted(gt[:, 0], timestamps), 1, len(gt) - 1)
        j -= (timestamps - gt[j - 1, 0]) < (gt[j, 0] - timestamps)
        expected = gt[j, 1:3]
        recorded = np.array([raw[int(t)]["xy"] for t in timestamps])
        error = np.linalg.norm(expected - recorded, axis=1)
        dt = np.abs(gt[j, 0] - timestamps) / 1e6
        assert np.isfinite(expected).all()
        assert error.max() < 1e-3
        pose_checks[seq] = {"scans": len(timestamps), "max_position_difference_m": float(error.max()),
                            "timestamp_difference_s_quantiles": np.quantile(dt, [0, .5, .99, 1]).tolist(),
                            "outside_correct_one_second": int((dt > 1).sum()),
                            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    printed = np.array([.7321, .6309, .7098, .6582, .2811, .1840, .5248, .3862, .1355])
    counts = np.arange(1, 5001)
    match = (np.abs(np.rint(counts[:, None] * printed) / counts[:, None] - printed) < .00005).all(axis=1)
    inference = {"published_table_II_NCLT_R1": printed.tolist(),
                 "compatible_common_integer_denominators_le_5000": counts[match].tolist(),
                 "assumption": "9방법의 Recall@1이 같은 질의 집합에서 정수 성공 수/질의 수를 소수4자리로 반올림한 값이라는 가정",
                 "interpretation": "최소 호환 분모989는 공개 생성기 전체1054·양성992·test259와 다르다. 원 논문의 실제 질의 수를 확정한 것은 아니다."}
    report = {"rows": rows, "pose_checks": pose_checks, "denominator_inference": inference,
              "timestamp_unit_source_issue": "공식 NCLT 코드의 1초 허용 오차는 1e9를 곱하지만 입력은 마이크로초다. 현재 선택 스캔에서 1초 초과가 없다면 이 오류는 현재 격차의 원인이 아니다.",
              "sources_sha256": source_hashes,
              "selection": "기존 입력·승자·점수 고정. 분모와 오류 유형만 분해하며 새 설정을 선택하지 않는다."}
    (OUT / "diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
