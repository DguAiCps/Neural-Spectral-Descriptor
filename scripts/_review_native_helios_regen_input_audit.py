#!/usr/bin/env python3
"""공개 자료가 있는 Velodyne03 전체를 재생성 입력과 대조한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results/pdv2_review_native_20261105"


def main() -> None:
    """같은 시각의 전수 입력에서 동일한 앞부분과 무작위 채움 차이를 센다."""
    released = ROOT / "helios"
    regen = ROOT / "helios_regenerated_velodyne"
    name = "Roundabout03-Velodyne"
    manifest = json.loads((released / "input_manifest.json").read_text())
    names = manifest["folders_order"]
    source = pickle.loads((released / "data/query.pickle").read_bytes())[names.index(name)]
    done = json.loads((regen / "prepared" / name / "done.json").read_text())
    generated = {row["timestamp"]: row for row in done["files"]}
    paths = {p.stem: p for p in (regen / "prepared" / name).glob("chunk_*/out/*.bin")}
    assert len(generated) == len(paths) == len(source) == 1734
    rows = []
    for i in range(len(source)):
        original_path = released / "data" / source[i]["query"]
        timestamp = original_path.stem
        old_bytes, new_bytes = original_path.read_bytes(), paths[timestamp].read_bytes()
        assert hashlib.sha256(old_bytes).hexdigest() == manifest["sequences"][name]["point_sha256"][i]
        assert hashlib.sha256(new_bytes).hexdigest() == generated[timestamp]["sha256"]
        a = np.frombuffer(old_bytes, np.float32).reshape(-1, 4)
        b = np.frombuffer(new_bytes, np.float32).reshape(-1, 4)
        assert a.shape == b.shape == (8192, 4) and np.isfinite(b).all()
        equal = (a == b).all(axis=1)
        prefix = int(np.flatnonzero(~equal)[0]) if not equal.all() else len(a)
        xyz_equal = (a[:, :3] == b[:, :3]).all(axis=1)
        xyz_prefix = int(np.flatnonzero(~xyz_equal)[0]) if not xyz_equal.all() else len(a)
        rows.append({"timestamp": timestamp, "identical_prefix_rows": prefix,
                     "identical_xyz_prefix_rows": xyz_prefix,
                     "identical_same_index_rows": int(equal.sum()), "total_rows": len(a)})
    prefix = np.asarray([row["identical_prefix_rows"] for row in rows])
    xyz_prefix = np.asarray([row["identical_xyz_prefix_rows"] for row in rows])
    report = {"sequence": name, "frames": len(rows), "both_input_hashes_verified": True,
              "identical_entire_frames": int((prefix == 8192).sum()),
              "identical_prefix_rows_quantiles_0_05_50_95_100": np.quantile(prefix, [0, .05, .5, .95, 1]).tolist(),
              "mean_identical_prefix_fraction": float(prefix.mean() / 8192),
              "identical_xyz_prefix_rows_quantiles_0_05_50_95_100": np.quantile(xyz_prefix, [0, .05, .5, .95, 1]).tolist(),
              "mean_identical_xyz_prefix_fraction": float(xyz_prefix.mean() / 8192),
              "intensity_only_prefix_mismatch_frames": int((prefix != xyz_prefix).sum()),
              "model_input": "공식 datasets/pnv_raw.py:22는 XYZ만 읽고 eval/evaluate.py:204는 입력특징을1로 채운다. intensity 차이는 사용되지 않음",
              "scope": "공개 시각·공식 toolbox 설정의 전수 비교. 무작위 부족점 채움 때문에 byte 동일 재현은 아니며 native 성능 대조는 별도다.",
              "rows": rows}
    (regen / "released_round03_input_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "rows"})


if __name__ == "__main__":
    main()
