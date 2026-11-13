#!/usr/bin/env python3
"""격자는 유지하고 지역 특징 계산의 좌표 비등방성만 분리한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

import _review_native_ring as native
import _review_native_ring_maxfix as maxfix

BASE = native.OUT
OUT = BASE / "ring_feature_metric"
Z_SCALE = np.float32(19.0 / 70.0)


def prepare() -> None:
    """새 성능을 읽기 전에 공식 좌표 정규화에서 도출한 조건을 고정한다."""
    OUT.mkdir(exist_ok=True)
    protocol = json.loads((BASE / "maxfix/protocol.json").read_text())
    protocol["purpose"] = "NCLT 지역 기하 특징의 비등방 좌표에 대한 단일요인 진단; 논문 정답 구현을 주장하지 않음"
    protocol["feature_coordinate_factor"] = {
        "control": "공식 normalize_pc: x/70,y/70,(z-1)/19 좌표에서30NN와지역특징 계산",
        "diagnostic": "격자좌표는 그대로;30NN·공분산·높이통계에만 z*=19/70 적용,즉 x/70,y/70,(z-1)/70",
        "unchanged": "원시프레임·포인트선택·격자주소·정확max·정규화entropy·endpoint=True각도·후보·정답·집계",
        "hypothesis": "논문은3D기하특징을 설명하지만 비등방 좌표 정규화는 명시하지 않는다. 지역거리의 축별 왜곡이 잔여 차이에 기여하는지 진단한다.",
        "limits": "논문 코드 버전과 정확 입력목록이 없으므로 이 조건을 저자 benchmark 정확 재현/정답 수정으로 부르지 않음. 주표에는 적용 안 함.",
        "source": "_external/RING/datasets/NCLTDataset.py::normalize_pc,utils/core.py::generate_bev_feats,논문TableI",
        "z_scale": float(Z_SCALE),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": "축별 단위 차이에서 정한 단일 조건이며 시험 성능으로 후보를 선택하지 않음",
    }
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        native.save_json(path, protocol)
    for name in ["evaluation_sets.json", "native_data", "raw_sha256.json", "raw_files.txt"]:
        p = OUT / name
        if not p.exists():
            p.symlink_to(Path("..") / name)


def coordinates(pc: np.ndarray) -> np.ndarray:
    """XY와 동일한 길이 단위로 Z를 바꾸며 입력은 변경하지 않는다."""
    result = np.asarray(pc, dtype=np.float32).copy()
    result[:, 2] *= Z_SCALE
    return result


def install() -> None:
    """정확 max 구현의 두 지역특징 입력만 같은 등방 좌표로 바꾼다."""
    import voxelfeat
    core = native.rol.import_ring_core()
    assert core.cfg.point_cloud["x_bound"][1] == 70
    assert core.cfg.point_cloud["y_bound"][1] == 70
    neighbors = core.build_neighbors_NN
    extractor = voxelfeat.GPUFeatureExtractor

    def metric_neighbors(pc, k):
        return neighbors(coordinates(pc), k)

    class MetricExtractor:
        def __init__(self, *args):
            changed = list(args)
            changed[0] = coordinates(np.asarray(args[0]).reshape(-1, 3)).flatten()
            self.original = extractor(*changed)

        def get_features(self):
            return self.original.get_features()

    core.build_neighbors_NN = metric_neighbors
    voxelfeat.GPUFeatureExtractor = MetricExtractor
    maxfix.install_maxfix()


def smoke() -> None:
    """고정3원시에서 격자 점유 불변·공식 좌표 변환·유한 출력을 확인한다."""
    import torch
    native.imports()
    from datasets.NCLTDataset import NCLTPointCloudLoader
    core = native.rol.import_ring_core()
    loader = NCLTPointCloudLoader()
    paths = (BASE / "raw_files.txt").read_text().splitlines()
    selected = [paths[i] for i in [0, len(paths)//2, len(paths)-1]]
    points, occupancy = [], []
    for name in selected:
        pc = loader.normalize_pc(loader.read_pc(str(BASE / "native_data" / name)))
        points.append(pc)
        with torch.no_grad():
            occupancy.append(core.generate_bev_occ(pc).cpu().numpy())
    install()
    rows = []
    for name, pc, before in zip(selected, points, occupancy):
        preserved = pc.copy()
        with torch.no_grad():
            after = core.generate_bev_occ(pc).cpu().numpy()
            features = core.generate_bev_feats(pc).cpu().numpy()
        assert np.array_equal(before, after)
        assert np.array_equal(pc, preserved)
        assert features.shape == (6, 120, 120) and np.isfinite(features).all()
        xyz = coordinates(pc)
        np.testing.assert_array_equal(xyz[:, :2], pc[:, :2])
        rows.append({"raw": name, "points": len(pc), "occupancy_bitwise_equal": True,
                     "grid_input_unmodified": True, "feature_shape": list(features.shape), "finite": True})
    native.save_json(OUT / "smoke.json", {"passed": True, "rows": rows})


def run() -> None:
    """공통46 완료 뒤 별도 native 입력 전체를 평가한다."""
    assert json.loads((OUT / "smoke.json").read_text())["passed"]
    install()
    native.OUT = OUT
    native.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "smoke", "run"])
    args = parser.parse_args()
    prepare()
    if args.stage == "smoke":
        smoke()
    elif args.stage == "run":
        run()
