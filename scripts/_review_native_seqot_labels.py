#!/usr/bin/env python3
"""같은 SeqOT native 예측에서 공개정답과 공통거리정답의 영향을 분리한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/pdv2_review_native_20261105/seqot"
DATA = Path("/mnt/d/NSD_datasets/nclt")
SOURCE_URL = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2012-02-05.csv"


def protocol() -> None:
    value = {"purpose": "동일native입력·가중치·후보·예측을 유지하고 정답기준만 대조",
             "distance_radii_m": [5, 10, 25], "threshold_basis": "공통평가5m/25m와native비교의10m; 결과를보고선택하지않음",
             "native_labels": "저자 공개 정답 배열 그대로; released generator는거리<15m,논문서술은overlap>0.3. protocol_label_correction.json참조",
             "paired_queries": "native양성 & 해당반경내DB있음 & GT시간대응성립; 같은질의에두정답기준적용",
             "poses": "공식NCLT100HzGT의최근접시간; 원시timestamp는microseconds; 1초이하차이만유효",
             "new_pose_source": SOURCE_URL,
             "scope": "정답기준 단일요인 진단. 공통46재평가나 시험설정선택이 아님"}
    p = ROOT / "label_factor_protocol_v2.json"
    if p.exists():
        assert json.loads(p.read_text()) == value
    else:
        p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def positions(role: str) -> tuple:
    seq = "2012-01-08" if role == "database" else "2012-02-05"
    source = DATA / seq / f"groundtruth_{seq}.csv" if role == "database" else ROOT / f"groundtruth_{seq}.csv"
    assert source.stat().st_size > 10_000_000
    pose = np.loadtxt(source, delimiter=",")
    assert pose.shape[1] == 7
    pose = pose[np.argsort(pose[:, 0])]
    pose = pose[np.isfinite(pose).all(axis=1)]
    raw = DATA / seq / "velodyne_sync" if role == "database" else DATA / "_seqot_dl" / seq / "velodyne_sync"
    files = sorted(raw.glob("*.bin"))
    assert len(files) == {"database": 28127, "query": 28239}[role]
    stride = 10 if role == "database" else 5
    ids = np.arange(0, len(files), stride)
    times = np.asarray([int(files[i].stem) for i in ids], np.int64)
    right = np.searchsorted(pose[:, 0], times).clip(0, len(pose) - 1)
    left = (right - 1).clip(0)
    picked = np.where(abs(pose[left, 0] - times) <= abs(pose[right, 0] - times), left, right)
    difference = abs(pose[picked, 0] - times)
    valid = difference <= 1_000_000
    provenance = {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "selected": len(ids), "valid_time_matches": int(valid.sum()),
                  "time_difference_us_quantiles": np.quantile(difference, [0, .5, .95, 1]).tolist()}
    return pose[picked, 1:4], valid, ids, provenance


def run() -> None:
    protocol()
    database, db_valid, db_ids, db_prov = positions("database")
    query, query_valid, query_ids, query_prov = positions("query")
    geo = np.linalg.norm(query[:, None] - database[None], axis=-1)
    geo[:, ~db_valid] = np.inf
    gt = np.load(REPO / "_external/seqot_official_eval/gt_120108_120205.npy", allow_pickle=True)
    assert len(query) == len(gt)
    positive = np.asarray([np.asarray(g).any() for g in gt])
    pair_distances = []
    for i, true in enumerate(gt):
        ids = np.asarray(true, np.int64)
        assert np.all(ids >= 0) and np.all(ids < 28127)
        # 공개 정답은 모든 원시DB프레임을 나열한다. 검색후보는10간격이므로
        # 동일후보 공간에 실제 포함된 정답쌍만 거리분포에 집계한다.
        ids = ids[ids % 10 == 0]
        values = geo[i, ids // 10]
        if query_valid[i]:
            pair_distances.extend(values[np.isfinite(values)].tolist())
    pair_distances = np.asarray(pair_distances)
    rows = []
    for p in ROOT.glob("*/native/report.json"):
        pred = np.load(p.parent / "predicted_L2_dis.npz")["arr_0"].reshape(-1, 22, 3)
        assert np.array_equal(pred[:, 0, 0], query_ids)
        selected_raw = pred[:, 0, 1].astype(np.int64)
        assert np.all(selected_raw % 10 == 0)
        selected = selected_raw // 10
        native_good = np.asarray([int(selected_raw[i]) in gt[i] for i in range(len(gt))])
        for radius in [5, 10, 25]:
            paired = positive & query_valid & db_valid[selected] & (geo.min(axis=1) <= radius)
            geometric_good = geo[np.arange(len(query)), selected] <= radius
            row = {"model": p.parent.parent.name, "radius_m": radius, "paired_queries": int(paired.sum()),
                   "native_public_hits": int(native_good[paired].sum()), "distance_hits": int(geometric_good[paired].sum()),
                   "native_public_r1": float(native_good[paired].mean()), "distance_r1": float(geometric_good[paired].mean()),
                   "native_correct_but_distance_wrong": int((paired & native_good & ~geometric_good).sum())}
            rows.append(row)
    report = {"pose_provenance": {"database": db_prov, "query": query_prov},
              "native_positive_pair_count_with_valid_pose": len(pair_distances),
              "native_positive_pair_distance_quantiles_m": np.quantile(pair_distances, [0, .5, .9, .95, .99, 1]).tolist(),
              "native_positive_pair_fraction_beyond": {str(r): float((pair_distances > r).mean()) for r in [5, 10, 25]},
              "same_queries_same_predictions": rows,
              "label_description_correction": "protocol_label_correction.json",
              "protocol_sha256": hashlib.sha256((ROOT / "label_factor_protocol_v2.json").read_bytes()).hexdigest()}
    (ROOT / "label_factor_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["protocol", "run"])
    a = p.parse_args()
    protocol() if a.stage == "protocol" else run()
