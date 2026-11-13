#!/usr/bin/env python3
"""공통 원장의 같은 양성질의·예측에서 거리 허용값만 대조한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_pilot import CITY, TEST

OUT = REPO / "results/pdv2_review_native_20261105"
REFERENCE = REPO / "results/pdv2_symmetric3_20261013/s1"
PREFIX = {"ring": "dump_ring", "ringpp": "dump_ringpp", "helios": "dump_helios", "imlpr": "dump_imlpr", "seqot": "dump_seqotl2"}


def main() -> None:
    protocol = {"radius_m": [5, 10, 15, 25], "fixed_positive_queries": "공통원장 min_past_gt_dist<=5; 동일질의·동일예측·동일분모",
                "unchanged": "원본의입력·가중치·키프레임·과거후보·argmax·도시집계·시드가중치",
                "basis": "주표5m,ImLPRnative10m,SeqOT공개GT15m,주표음성경계25m",
                "scope": "정답거리만의원인진단.새주표나설정선택이아니며AP/F1은바꾸지않는다."}
    target = OUT / "common_radius_factor_protocol.json"
    if target.exists():
        assert json.loads(target.read_text()) == protocol
    else:
        target.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    source = json.loads((REPO / "results/pdv2_baseline_seeds/summary.json").read_text())
    methods = {}
    hashes = {}
    for name, prefix in PREFIX.items():
        per_seed = {}
        for seed, previous in source["methods"][name]["per_seed"].items():
            sequences = {}
            for sensor, seq in TEST:
                path = REPO / previous["source"] / f"{prefix}_{sensor}_{seq}.npz"
                with np.load(path) as z, np.load(REFERENCE / f"test_selected_symmetric_pbev128_main_{sensor}_{seq}.npz") as reference:
                    ix = np.searchsorted(z["query_idx"], reference["query_idx"])
                    np.testing.assert_array_equal(z["query_idx"][ix], reference["query_idx"])
                    np.testing.assert_array_equal(z["min_past_gt_dist"][ix], reference["min_past_gt_dist"])
                    positive = reference["min_past_gt_dist"] <= 5
                    distance = z["top1_geo_dist"][ix][positive]
                    sequences[f"{sensor}/{seq}"] = {"n_positive": len(distance),
                        "r1": {str(r): float((distance <= r).mean()) if len(distance) else None for r in protocol["radius_m"]}}
                hashes[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert sum(r["n_positive"] for r in sequences.values()) == 61682
            city = {c: {str(r): float(np.mean([sequences[s]["r1"][str(r)] for s in names if sequences[s]["n_positive"]])) for r in protocol["radius_m"]} for c, names in CITY.items()}
            macro = {str(r): float(np.mean([v[str(r)] for v in city.values()])) for r in protocol["radius_m"]}
            assert abs(macro["5"] - previous["macro"]["R@1"]) < 1e-12
            per_seed[seed] = {"source": previous["source"], "per_sequence": sequences, "city": city, "macro": macro}
        means = {str(r): float(np.mean([v["macro"][str(r)] for v in per_seed.values()])) for r in protocol["radius_m"]}
        methods[name] = {"seeds": per_seed, "mean": means}
    report = {"methods": methods, "input_sha256": hashes, "protocol_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "original_5m_reproduced": True}
    (OUT / "common_radius_factor_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v["mean"] for k, v in methods.items()}), flush=True)


if __name__ == "__main__":
    main()
