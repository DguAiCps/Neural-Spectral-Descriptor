#!/usr/bin/env python3
"""공개 시각 목록에 맞춰 HeLiOS Velodyne 입력을 공식 toolbox로 다시 만든다."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _review_native_helios as native

OUT = native.OUT.parent / "helios_regenerated_velodyne"
RAW = Path(os.environ.get("NSD_DATA_ROOT", "/workspace/data")) / "helipr"
SOURCE = Path(os.environ.get("HELIOS_NATIVE_SOURCE", "/workspace/data/_review_native_20261105/helios"))


def stamps(sequence: str) -> list:
    """저자 배포 CSV의 시각과 순서를 그대로 읽는다."""
    with (SOURCE / "data_validation" / sequence / "trajectory.csv").open() as stream:
        return [row["timestamp"] for row in csv.DictReader(stream)]


def protocol() -> None:
    value = {"purpose": "공개 archive 차단에 대한 공식 원자료 재생성과 공개입력 대조",
             "query_sequences": [f"Roundabout{i:02d}-Velodyne" for i in (1, 2, 3)],
             "input_frames": "저자 CSV의 모든 시각을 그대로 사용; 시험성능으로 선택하지 않음",
             "toolbox": "_external/helios_env/build/toolbox/helipr_tool",
             "toolbox_sha256": native.sha(REPO / "_external/helios_env/build/toolbox/helipr_tool"),
             "source_sha256": {name: native.sha(REPO / "_external/HeLiPR-Pointcloud-Toolbox" / name) for name in ["src/PointCloudProcessor.cpp", "src/utility.cpp", "src/BsplineSE3.cpp"]},
             "processing": "공식 HeLiOS config: deskew250, crop100m, 8192점, normalize. Velodyne reader, distanceThreshold0만 연결(배포 CSV가 이미 프레임 선택).",
             "random_sampling": "공식 바이너리 srand(time) 후 부족점을 채움. 원출력 보존·재개하며 공개 가공파일과 byte동일을 주장하지 않음.",
             "reference": "Roundabout03-Velodyne의 공개 가공파일에 처음/중간/마지막 원시3개 재생성 대조",
             "scope": "원자료 재생성 진단. 공개입력 원장과 별도 저장하며 부족4센서조건을 대체하지 않음"}
    target = OUT / "regeneration_protocol.json"
    if target.exists():
        assert json.loads(target.read_text()) == value
    else:
        native.write(target, value)


def generate(sequence: str, smoke: bool) -> None:
    """공식 바이너리를128개씩 실행하고 모든 파일을 해시로 보존한다."""
    from _helios_official_prep import run_toolbox
    protocol()
    ids = stamps(sequence)
    if smoke:
        ids = [ids[0], ids[len(ids) // 2], ids[-1]]
    seq = sequence.split("-")[0]
    root = OUT / ("smoke" if smoke else "prepared") / sequence
    root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rows = []
    for first in range(0, len(ids), 128):
        chunk = root / f"chunk_{first:06d}"
        chunk.mkdir(exist_ok=True)
        done = chunk / "done.json"
        if done.exists():
            rows.extend(json.loads(done.read_text())["files"])
            continue
        bindir = chunk / "bin"
        bindir.mkdir(exist_ok=True)
        for stamp in ids[first:first + 128]:
            source = RAW / seq / "LiDAR/Velodyne" / f"{stamp}.bin"
            assert source.exists(), source
            if not (bindir / source.name).exists():
                (bindir / source.name).symlink_to(source)
        out = run_toolbox(chunk, bindir, RAW / seq / "LiDAR_GT/Velodyne_gt.txt", True)
        record = []
        missing = []
        for stamp in ids[first:first + 128]:
            p = out / f"{stamp}.bin"
            if not p.exists():
                missing.append(stamp)
                continue
            a = np.fromfile(p, np.float32).reshape(-1, 4)
            assert a.shape == (8192, 4) and np.isfinite(a).all(), (p, a.shape)
            row = {"timestamp": stamp, "raw_sha256": native.sha(bindir / p.name), "sha256": native.sha(p)}
            if smoke and sequence == "Roundabout03-Velodyne":
                from scipy.spatial import cKDTree
                ref = native.OUT / "data/validation" / sequence / "LiDAR" / p.name
                b = np.fromfile(ref, np.float32).reshape(-1, 4)
                distance = cKDTree(b[:, :3]).query(a[:, :3], workers=1)[0] * 100
                row.update(reference_sha256=native.sha(ref), byte_equal=np.array_equal(a, b),
                           nearest_released_point_m_quantiles=np.quantile(distance, [0, .5, .95, 1]).tolist(),
                           regenerated_xyz_centroid=a[:, :3].mean(axis=0).tolist(), released_xyz_centroid=b[:, :3].mean(axis=0).tolist())
            record.append(row)
        native.write(chunk / "result.json", {"files": record, "missing_toolbox_outputs": missing})
        assert not missing, f"공식 toolbox가 시각을 생략함: {missing}"
        native.write(done, {"files": record, "elapsed_s": time.time() - start})
        rows.extend(record)
        native.write(root / "status.json", {"frames": len(rows), "total": len(ids), "elapsed_s": time.time() - start, "pid": os.getpid()})
        print(json.dumps({"sequence": sequence, "frames": len(rows), "total": len(ids), "elapsed_s": time.time() - start}), flush=True)
    native.write(root / "done.json", {"files": rows, "elapsed_s": time.time() - start, "protocol_sha256": native.sha(OUT / "regeneration_protocol.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", choices=[f"Roundabout{i:02d}-Velodyne" for i in (1, 2, 3)])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    generate(args.sequence, args.smoke)
