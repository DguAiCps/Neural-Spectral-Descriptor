#!/usr/bin/env python3
"""동결된 모델과 특징으로 같은 센서 지도 대조를 별도로 수행한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np

import _review_native_helios as native
from _review_native_learned_verify import independent_pair

OUT = native.OUT.parent / "helios_same_sensor"
REGEN = native.OUT.parent / "helios_regenerated_velodyne"
MODELS = ["author", "v2_s1", "v2_s2", "v2_s3"]
DATABASE = "Roundabout01-Velodyne"
QUERIES = ["Roundabout02-Velodyne", "Roundabout03-Velodyne"]


def prepare() -> None:
    """공개 겹침 행렬에서 새 지도 정답을 만들고 성능 계산 전에 고정한다."""
    OUT.mkdir(exist_ok=True)
    source = native.OUT / "data/query.pickle"
    queries = pickle.loads(source.read_bytes())
    offsets = np.r_[0, np.cumsum([len(q) for q in queries])]
    matrix = Path("/mnt/d/NSD_datasets/_review_native_20261105/helios/data_overlap/overlap_matrix_validation_Roundabout.txt")
    expected = json.loads((native.OUT / "input_manifest.json").read_text())["overlap_sha256"]
    assert native.sha(matrix) == expected
    targets = {}
    result = {}
    for name in QUERIES:
        position = native.NAMES.index(name)
        result[name] = {}
        for row in range(len(queries[position])):
            targets[int(offsets[position]) + row] = (name, position, row)
    db_position = native.NAMES.index(DATABASE)
    checked = 0
    with matrix.open() as stream:
        for index, line in enumerate(stream):
            if index not in targets:
                continue
            name, position, row = targets[index]
            values = np.fromstring(line, sep=" ", dtype=np.float64)
            assert len(values) == offsets[-1]
            old = queries[position][row]
            for db_key, db in enumerate(native.DATABASES):
                col = native.NAMES.index(db)
                ids = np.flatnonzero(values[offsets[col]:offsets[col + 1]] > 0.5)
                np.testing.assert_array_equal(ids, old[db_key])
            ids = np.flatnonzero(values[offsets[db_position]:offsets[db_position + 1]] > 0.5)
            result[name][row] = {k: v for k, v in old.items() if isinstance(k, str)}
            result[name][row][0] = ids.tolist()
            checked += 1
    assert checked == sum(map(len, result.values())) == 3139
    database = {i: {k: v for k, v in q.items() if isinstance(k, str)} for i, q in queries[db_position].items()}
    metadata = OUT / "metadata.pickle"
    data = pickle.dumps({"database": database, "queries": result}, protocol=4)
    if metadata.exists():
        assert metadata.read_bytes() == data
    else:
        metadata.write_bytes(data)
    protocol = {"frozen_before_evaluation": True, "models": native.check()["models"], "database": DATABASE,
                "queries": QUERIES, "metadata_sha256": native.sha(metadata), "public_overlap_sha256": expected,
                "released_query_sha256": native.sha(source), "original_database_GT_rows_verified": checked * 2,
                "positive_threshold": "strict overlap > 0.5 (float64)", "native_nearest_neighbors": 200,
                "regeneration_protocol_sha256": native.sha(REGEN / "regeneration_protocol.json"),
                "implementation_sha256": native.sha(Path(__file__)),
                "scope": "고정된 4모델·같은 Velodyne 질의에서 지도 센서를 추가 대조. 논문 원래 지도에 없는 확장이고 주표 대체 안 함. 지도 프레임·GT 집합이 달라 센서만의 완전한 인과 분리는 아니다. Roundabout은 v2 검증 장소다."}
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        native.write(path, protocol)
    print(json.dumps({"prepared": True, "queries": checked, "database": len(database), "protocol_sha256": native.sha(path)}))


def vectors(model: str, name: str) -> tuple:
    """이미 끝난 추론 묶음을 읽고 출처 해시를 남긴다."""
    root = REGEN / model / "desc" / name
    count = json.loads((root / "done.json").read_text())["total"]
    data = np.zeros((count, 8448), np.float32)
    seen = np.zeros(count, bool)
    hashes = {}
    for path in sorted(root.glob("shard_*.npz")):
        z = np.load(path)
        data[z["indices"]] = z["desc"]
        seen[z["indices"]] = True
        hashes[str(path.relative_to(REGEN))] = native.sha(path)
    assert seen.all() and np.isfinite(data).all()
    return data, hashes


def run() -> None:
    """공식 process_query와 독립 정수 검산을 함께 수행한다."""
    from sklearn.neighbors import KDTree
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["frozen_before_evaluation"]
    assert protocol["implementation_sha256"] == native.sha(Path(__file__))
    assert protocol["metadata_sha256"] == native.sha(OUT / "metadata.pickle")
    meta = pickle.loads((OUT / "metadata.pickle").read_bytes())
    official, _ = native.official()
    reports = []
    for model in MODELS:
        assert native.sha(native.REPO / protocol["models"][model]["path"]) == protocol["models"][model]["sha256"]
        database, hashes = vectors(model, DATABASE)
        tree = KDTree(database)
        for name in QUERIES:
            root = OUT / model
            root.mkdir(exist_ok=True)
            report_path = root / (name + ".json")
            if report_path.exists():
                row = json.loads(report_path.read_text())
                assert row["protocol_sha256"] == native.sha(OUT / "protocol.json")
                reports.append(row)
                continue
            started = time.time()
            query, query_hashes = vectors(model, name)
            distances, indices = tree.query(query, k=200)
            for probe in [0, len(query) // 2, len(query) - 1]:
                direct = np.sum((database.astype(np.float64) - query[probe].astype(np.float64)) ** 2, axis=1)
                assert direct.argmin() == indices[probe, 0]
            thresholds = np.linspace(0, 1, 1000)
            sums = np.zeros(30)
            cells = np.zeros((4, 1000))
            valid = 0
            class CachedTree:
                def query(self, x, k):
                    assert k == 200
                    return distances[current:current + 1], indices[current:current + 1]
            for current in range(len(query)):
                answer = official.process_query(current, query[current], meta["queries"][name][current], CachedTree(), 30,
                    max(int(round(len(database) / 1000)), 1), database, thresholds, 0,
                    database_meta=list(meta["database"].values()), is_intra=False, init_time=None)
                if answer is not None:
                    rec, _, tp, fp, tn, fn, count = answer
                    valid += count
                    sums += rec
                    cells += np.asarray([tp, fp, tn, fn])
            evaluated, good, hits, independent_cells, fixed = independent_pair(indices, distances, meta["queries"][name], meta["database"], 0, False, 30, False)
            assert int(evaluated.sum()) == valid == fixed
            np.testing.assert_array_equal(np.cumsum(sums), hits)
            np.testing.assert_array_equal(cells, independent_cells)
            pair = root / (name + ".npz")
            np.savez_compressed(pair, indices=indices, distances=distances, evaluated=evaluated, first_correct=good,
                                recall_hits=hits, tp_fp_tn_fn=cells, thresholds=thresholds)
            row = {"model": model, "database": DATABASE, "query": name, "positive_evaluated": valid,
                   "r1": float(hits[0] / valid), "r5": float(hits[4] / valid), "independent_equal": True,
                   "prediction_sha256": native.sha(pair), "protocol_sha256": native.sha(OUT / "protocol.json"),
                   "feature_sha256": dict(hashes, **query_hashes), "elapsed_s": time.time() - started}
            native.write(report_path, row)
            reports.append(row)
            native.write(OUT / "progress.json", {"completed": len(reports), "total": 8, "last": [model, name]})
            print(json.dumps({k: v for k, v in row.items() if k != "feature_sha256"}), flush=True)
    native.write(OUT / "report.json", {"complete": len(reports) == 8, "rows": reports})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    args = parser.parse_args()
    globals()[args.stage]()
