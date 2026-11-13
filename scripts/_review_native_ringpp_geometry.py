#!/usr/bin/env python3
"""RING++ 새 원장의 후보와 정답 거리를 원래 자세로 독립 재계산한다."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

from _review_native_pdv2_ringpp import OUT, REPO, sha256, write_json


def main() -> None:
    """64질의 단위 거리 행렬로 모든 과거 후보의 최소 거리를 다시 센다."""
    protocol = json.loads((OUT / "protocol.json").read_text())
    target = OUT / "geometry_independent_verification.json"
    prior = json.loads(target.read_text()).get("rows", {}) if target.exists() else {}
    rows = {}
    for seq in protocol["sequences"]:
        name = seq["sensor"] + "/" + seq["sequence"]
        ledger = OUT / "odom" / ("dump_ringpp_" + name.replace("/", "_") + ".npz")
        if not ledger.exists():
            continue
        digest = sha256(ledger)
        cache = REPO / seq["cache"]
        assert sha256(cache) == seq["cache_sha256"]
        if name in prior and prior[name]["ledger_sha256"] == digest and prior[name]["passed"]:
            rows[name] = prior[name]
            continue
        with np.load(cache, allow_pickle=True) as z:
            xyz = z["poses_gt"].astype(np.float64)[:, :3, 3]
        with np.load(ledger) as z:
            q, win = z["query_idx"], z["j_star"]
            np.testing.assert_array_equal(q, np.arange(30, len(xyz)))
            np.testing.assert_array_equal(z["n_candidates"], q - 29)
            assert np.all((win >= 0) & (win <= q - 30))
            winner_distance = np.sqrt(np.sum((xyz[q] - xyz[win])**2, axis=1))
            minimum = np.empty(len(q), np.float64)
            for first in range(0, len(q), 64):
                block = q[first:first+64]
                candidates = np.arange(block[-1] - 29)
                distance = cdist(xyz[block], xyz[candidates], metric="euclidean")
                distance[candidates[None, :] > block[:, None] - 30] = np.inf
                minimum[first:first+len(block)] = distance.min(axis=1)
            np.testing.assert_allclose(winner_distance, z["top1_geo_dist"], rtol=0, atol=1e-10)
            np.testing.assert_allclose(minimum, z["min_past_gt_dist"], rtol=0, atol=1e-10)
            for boundary in (5, 25):
                np.testing.assert_array_equal(minimum <= boundary, z["min_past_gt_dist"] <= boundary)
            np.testing.assert_array_equal(winner_distance <= 5, z["top1_geo_dist"] <= 5)
            rows[name] = {"passed": True, "queries": len(q), "common_queries": len(q)-15, "ledger_sha256": digest, "cache_sha256": seq["cache_sha256"], "winner_distance_max_error_m": float(np.max(np.abs(winner_distance-z["top1_geo_dist"]))), "minimum_distance_max_error_m": float(np.max(np.abs(minimum-z["min_past_gt_dist"]))), "candidate_counts_and_labels_exact": True}
    report = {"passed": all(r["passed"] for r in rows.values()), "complete": len(rows)==46, "sequences": len(rows), "queries": sum(r["queries"] for r in rows.values()), "common_queries": sum(r["common_queries"] for r in rows.values()), "rows": rows, "protocol_sha256": sha256(OUT / "protocol.json"), "independence": "공식 검색·정답 함수 미사용; scipy cdist로 모든 과거 후보 3D 거리 재계산; 원시 자세 동일해시"}
    write_json(target, report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}, ensure_ascii=False))


if __name__ == "__main__":
    main()
