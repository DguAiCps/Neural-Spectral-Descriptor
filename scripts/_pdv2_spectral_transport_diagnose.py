#!/usr/bin/env python3
"""복소 전달 대조의 높은 정밀도 경계 오답을 같은 원장 검사로 확인한다."""
import json
from pathlib import Path

import numpy as np

import _pdv2_temporal_derivatives_diagnose as audit
import _pdv2_spectral_transport as trial
from _pdv2_no_context_ablation import sha256, write_json
from gnn.spectral_transport import lidar_yaw

REPO = Path(__file__).resolve().parents[1]
audit.OUT = REPO / "results/pdv2_spectral_transport_20261020"


def turning_groups() -> None:
    """±5 이웃의 최대 회전 차이 5도를 기준으로 최종 정답 수를 나눈다."""
    summary = trial.read(audit.OUT / "summary.json")
    spec = trial.protocol(True)
    results, sources = {}, {}
    for seed, models in summary["per_seed"].items():
        results[seed] = {}
        for sequence in spec["val"]:
            _, _, paths, _ = trial.base.exp.old.base.inputs("val", sequence)
            sources[str(paths["cache"].relative_to(REPO))] = sha256(paths["cache"])
            with np.load(paths["cache"]) as z:
                yaw = lidar_yaw(z["poses"])
            results[seed][sequence] = {}
            for arm, info in models.items():
                ep = info["selected"]["epoch"]
                path = audit.OUT / f"s{seed}/{arm}/epoch{ep}" / (sequence.replace("/", "_")+".npz")
                sources[str(path.relative_to(REPO))] = sha256(path)
                with np.load(path) as z:
                    q, distance, minimum = z["query_idx"], z["top1_geo_dist"], z["min_past_gt_dist"]
                neighbors = q[:, None] + np.arange(-5, 6)[None]
                rotation = np.rad2deg(abs(np.angle(np.exp(
                    1j*(yaw[neighbors]-yaw[q, None]))))).max(1)
                turn, positive = rotation >= 5., minimum <= 5
                correct = positive & (distance <= 5)
                results[seed][sequence][arm] = {
                    label: {"queries": int(mask.sum()), "positive": int((mask & positive).sum()),
                            "correct": int((mask & correct).sum())}
                    for label, mask in (("straight", ~turn), ("turning", turn))}
    pooled = {}
    for seed, sequences in results.items():
        pooled[seed] = {arm: {label: {
            k: sum(seq[arm][label][k] for seq in sequences.values())
            for k in ("queries", "positive", "correct")}
            for label in ("straight", "turning")} for arm in summary["per_seed"][seed]}
    for p in (Path(__file__), audit.OUT / "summary.json", audit.OUT / "protocol.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    write_json(audit.OUT / "turning_diagnosis.json", {
        "per_sequence": results, "pooled": pooled, "sources": sources, "test_used": False,
        "threshold_degrees": 5.,
        "scope": "KISS ±5이웃의최대yaw차이로분류. 최종승자정답수의통합집계이며수락성능/계열평균과다름. 선택변경없음.",
    })


if __name__ == "__main__":
    audit.main()
    path = audit.OUT / "high_precision_diagnosis.json"
    result = json.loads(path.read_text())
    result["sources"][str(Path(__file__).relative_to(REPO))] = sha256(Path(__file__))
    write_json(path, result)
    turning_groups()
