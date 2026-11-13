#!/usr/bin/env python3
"""공식 RING NCLT 생성기의 표본 규칙을 원본 자세·스캔 목록으로 독립 재구성해 대조한다.

호스트에서 torch 없이 실행한다(원본 자세 CSV와 스캔 파일 이름만 읽는다).
    python3 scripts/_review_ring_followup_sampling.py --nclt-root /mnt/d/NSD_datasets/nclt

공식 경로(_external/RING/datasets/NCLTDataset.py:150-243, evaluation/generate_evaluation_sets.py:101-151)를
numpy로 다시 쓴 것이며, 기존 evaluation_sets.json(공식 생성기 출력)과 시각·위치가 같은지 확인한다.
허용 오차 단위 오류(1e9)와 CSV 첫 행 생략(header=0)의 영향도 함께 센다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106/ring/sampling"
SETS = REPO / "results/pdv2_review_native_20261105/evaluation_sets.json"
SEQUENCES = ("2012-02-04", "2012-03-17")
TEST_CENTER = np.array([[50.0, -50.0]])
TEST_RADIUS = 220.0


def digest(path: Path) -> str:
    """파일의 SHA256을 반환한다."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_poses(csv: Path, skip_first: bool) -> tuple[np.ndarray, np.ndarray]:
    """시각과 float32 (x, y)를 읽는다. 공식 pandas header=0은 첫 데이터 행을 버린다."""
    raw = np.loadtxt(csv, delimiter=",", dtype=np.float64)
    if skip_first:
        raw = raw[1:]
    ts = raw[:, 0].astype(np.int64)
    order = np.argsort(ts, kind="quicksort")
    xy = raw[order, 1:3].astype(np.float32)
    return ts[order], xy, raw.shape[0]


def nearest(ts: int, stamps: np.ndarray) -> int:
    """utils/tools.py:57 find_nearest_ndx와 같은 규칙."""
    ndx = int(np.searchsorted(stamps, ts))
    if ndx == 0:
        return 0
    if ndx == len(stamps):
        return ndx - 1
    return ndx - 1 if ts - stamps[ndx - 1] < stamps[ndx] - ts else ndx


def sequence(root: Path, seq: str, split: str, sampling: float, tol_us: float, skip_first: bool) -> dict:
    """NCLTSequence.get_scan_poses의 연관·영역·간격 표본을 재현한다."""
    stamps, poses_xy, n_rows = read_poses(root / seq / f"groundtruth_{seq}.csv", skip_first)
    lidar = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(root / seq / "velodyne_sync")
                   if f.endswith(".bin"))
    keep_ts, keep_xy, dts, rejected = [], [], [], 0
    for ts in lidar:
        j = nearest(ts, stamps)
        delta = abs(int(stamps[j]) - ts)
        if delta > tol_us:
            rejected += 1
            continue
        keep_ts.append(ts)
        keep_xy.append(poses_xy[j])
        dts.append(delta)
    keep_ts = np.asarray(keep_ts, np.int64)
    keep_xy = np.asarray(keep_xy, np.float32)
    dts = np.asarray(dts, np.int64)
    if split == "test":
        mask = (np.linalg.norm(keep_xy.astype(np.float64)[:, None] - TEST_CENTER[None], axis=-1)
                < TEST_RADIUS).any(axis=1)
        keep_ts, keep_xy, dts = keep_ts[mask], keep_xy[mask], dts[mask]
    chosen, prev = [], None
    for i, pos in enumerate(keep_xy):
        if prev is None or np.linalg.norm(prev - pos) > sampling:   # float32 연산(공식과 같음)
            chosen.append(i)
            prev = pos
    return {"ts": keep_ts[chosen], "xy": keep_xy[chosen], "dt_us": dts[chosen],
            "lidar_files": len(lidar), "pose_rows": n_rows, "rejected": rejected}


def build(root: Path, split: str, tol_us: float, skip_first: bool) -> dict:
    """generate_evaluation_set(map 20m, query 5m, 20m 필터)."""
    m = sequence(root, SEQUENCES[0], split, 20.0, tol_us, skip_first)
    q = sequence(root, SEQUENCES[1], split, 5.0, tol_us, skip_first)
    d = np.linalg.norm(q["xy"].astype(np.float64)[:, None] - m["xy"].astype(np.float64)[None], axis=-1)
    keep = (d <= 20.0).any(axis=1)
    q_f = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in q.items()}
    return {"map": m, "query": q_f, "query_before_filter": int(len(q["ts"]))}


def main() -> None:
    """세 조건(공식 그대로·허용오차 1초 수정·첫 행 포함)을 기존 출력과 비교한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nclt-root", type=Path, default=Path("/mnt/d/NSD_datasets/nclt"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ref = json.loads(SETS.read_text())
    conditions = {"official_as_is": (1e9, True), "tolerance_1s_microseconds": (1e6, True),
                  "keep_first_csv_row": (1e9, False)}
    report = {"purpose": "공식 생성기 표본 규칙의 독립 재구성과 기존 evaluation_sets.json 대조",
              "source": ["_external/RING/datasets/NCLTDataset.py:150-243",
                         "_external/RING/evaluation/generate_evaluation_sets.py:101-151",
                         "_external/RING/utils/tools.py:57-123"],
              "reference_sets": str(SETS.relative_to(REPO)), "reference_sha256": digest(SETS),
              "pose_sha256": {s: digest(args.nclt_root / s / f"groundtruth_{s}.csv") for s in SEQUENCES},
              "conditions": {}, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for name, (tol, skip) in conditions.items():
        rows = {}
        for split in ("test", "all"):
            got = build(args.nclt_root, split, tol, skip)
            cmp = {}
            for kind in ("map", "query"):
                r_ts = np.array([e["timestamp"] for e in ref[split][kind]], np.int64)
                r_xy = np.array([e["xy"] for e in ref[split][kind]], np.float32)
                g = got[kind]
                same_ts = len(r_ts) == len(g["ts"]) and bool((r_ts == g["ts"]).all())
                cmp[kind] = {"count": int(len(g["ts"])), "reference_count": int(len(r_ts)),
                             "timestamps_identical": same_ts,
                             "xy_max_abs_diff_m": (float(np.abs(r_xy - g["xy"]).max()) if same_ts else None),
                             "max_pose_dt_s": (float(g["dt_us"].max() / 1e6) if len(g["dt_us"]) else None),
                             "nan_positions": int(np.isnan(g["xy"]).any(axis=1).sum()),
                             "lidar_files": g["lidar_files"], "pose_rows": g["pose_rows"],
                             "rejected_scans": g["rejected"]}
            cmp["query_before_filter"] = got["query_before_filter"]
            rows[split] = cmp
        report["conditions"][name] = rows
        print(name, json.dumps(rows), flush=True)
    a = report["conditions"]["official_as_is"]
    report["official_as_is_matches_reference"] = all(
        a[s][k]["timestamps_identical"] and a[s][k]["xy_max_abs_diff_m"] == 0.0
        for s in ("test", "all") for k in ("map", "query"))
    report["csv_first_row_note"] = (
        "2012-03-17 자세 CSV의 첫 행은 NaN이다. 공식 pandas header=0이 이 행을 버려 공식 출력은 정상이다. "
        "첫 행을 유지하면 GT 시작 전 스캔이 NaN 자세를 받아 첫 표본이 되고 이후 거리 비교가 모두 거짓이 된다.")
    report["script_sha256"] = digest(Path(__file__))
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"matches": report["official_as_is_matches_reference"]}), flush=True)


if __name__ == "__main__":
    main()
