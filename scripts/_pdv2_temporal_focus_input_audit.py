#!/usr/bin/env python3
"""같은 검증 키프레임에서 GT·주행 추정의 시간 간선 속성을 비교한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal3 as old
from _pdv2_temporal_focus_diagnose import quantiles
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """정답을 입력으로 넣지 않는 읽기 전용 속성 진단이다."""
    rows, sources = {}, {}
    for sensor, seq in old.base.VAL:
        _, _, paths, _ = old.base.inputs("val", sensor + "/" + seq)
        with np.load(paths["cache"]) as z:
            poses, gt, scans, times = z["poses"], z["poses_gt"], z["scan_ids"], z["timestamps"]
        n = len(poses)
        # 양방향 간선은 같은 거리·회전각이므로 각 무방향 쌍을 한 번 센다.
        src = np.concatenate([np.arange(n-d) for d in range(1, 6)])
        dst = np.concatenate([np.arange(d, n) for d in range(1, 6)])
        distances, angles = [], []
        for p in (poses, gt):
            distances.append(np.linalg.norm(p[src, :3, 3] - p[dst, :3, 3], axis=1))
            rel = p[dst, :3, :3] @ np.transpose(p[src, :3, :3], (0, 2, 1))
            angles.append(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2)-1)/2, -1, 1)))
        row = {"nodes": n, "undirected_edges": len(src),
            "distance_absolute_error_m": quantiles(np.abs(distances[0]-distances[1])),
            "distance_attr_absolute_error": quantiles(np.abs(np.log1p(distances[0])-np.log1p(distances[1]))/5),
            "rotation_absolute_error_deg": quantiles(np.degrees(np.abs(angles[0]-angles[1]))),
            "rotation_attr_absolute_error": quantiles(np.abs(angles[0]-angles[1])/np.pi),
            "scan_id_step": quantiles(np.diff(scans)), "cache_timestamp_step": quantiles(np.diff(times))}
        rows[sensor + "/" + seq] = row
        sources[str(paths["cache"].relative_to(REPO))] = sha256(paths["cache"])
        print("POSE", sensor, seq, row["distance_absolute_error_m"], row["rotation_absolute_error_deg"], flush=True)
    write_json(REPO / "results/pdv2_temporal_focus_20261018/diagnosis/input_pose_audit.json", {
        "sequences": rows, "sources": sources, "script_sha256": sha256(Path(__file__)), "test_used": False,
        "scope": "같은 KISS키프레임의 GT/추정 자세 속성 차이. 다른 장소의 학습/검증 분포와 다르며 재학습 인과 대조 아님. 좌표계 회전에 불변인3D거리·상대회전각만 비교."})


if __name__ == "__main__":
    main()
