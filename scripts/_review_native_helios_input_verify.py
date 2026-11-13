#!/usr/bin/env python3
"""공개 텍스트 overlap을 float64로 재읽어 모든 native 정답을 검산한다."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pickle
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105/helios"
SOURCE = Path("/mnt/d/NSD_datasets/_review_native_20261105/helios")


def main() -> None:
    manifest = json.loads((OUT / "input_manifest.json").read_text())
    names = manifest["folders_order"]
    queries = pickle.loads((OUT / "data/query.pickle").read_bytes())
    db = pickle.loads((OUT / "data/database.pickle").read_bytes())
    offsets = np.r_[0, np.cumsum([len(q) for q in queries])]
    max_coordinate_error = 0.0
    for name, query in zip(names, queries):
        with (SOURCE / "data_validation" / name / "trajectory.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(query)
        for i, row in enumerate(rows):
            assert Path(query[i]["query"]).stem == row["timestamp"]
            error = max(abs(query[i]["northing"] - float(row["x"])),
                        abs(query[i]["easting"] - float(row["y"])))
            max_coordinate_error = max(max_coordinate_error, error)
            assert error < 1e-9  # pandas와 표준float 파서의 반올림만 허용한다.
    start = time.time()
    checked = 0
    with (SOURCE / "data_overlap/overlap_matrix_validation_Roundabout.txt").open() as stream:
        for row_index, line in enumerate(stream):
            values = np.fromstring(line, dtype=np.float64, sep=" ")
            assert len(values) == offsets[-1]
            sequence = int(np.searchsorted(offsets, row_index, side="right") - 1)
            row = queries[sequence][int(row_index - offsets[sequence])]
            for database_index, name in enumerate(["Roundabout01-Aeva", "Roundabout01-Ouster"]):
                block = names.index(name)
                expected = np.flatnonzero(values[offsets[block]:offsets[block + 1]] > .5)
                np.testing.assert_array_equal(expected, row[database_index])
                assert len(db[database_index]) == offsets[block + 1] - offsets[block]
                checked += 1
            if row_index % 2000 == 0:
                print(row_index, round(time.time() - start, 1), flush=True)
    assert row_index + 1 == offsets[-1]
    result = {"all_query_frames": row_index + 1, "all_query_database_label_lists": checked,
              "text_dtype": "float64 (공식 np.loadtxt 기본값)", "strict_overlap_threshold": .5,
              "all_metadata_csv_rows_equal": True, "all_label_lists_equal": True,
              "max_csv_coordinate_parser_difference_m": max_coordinate_error,
              "elapsed_s": time.time() - start}
    (OUT / "input_independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
