#!/usr/bin/env python3
"""공식 RING++의 격자 최댓값 경쟁 상태만 분리한 진단이다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import _review_native_ring as native

BASE = native.OUT
OUT = BASE / "maxfix"


def prepare() -> None:
    """변경점을 결과 전에 고정하고 같은 평가 입력을 연결한다."""
    OUT.mkdir(exist_ok=True)
    protocol = json.loads((BASE / "protocol.json").read_text())
    protocol["purpose"] = "공식 RING++ 비원자 격자 max만 결정적 max로 교체한 원인 분리; 공식 native 결과 아님"
    protocol["diagnostic_variation"] = {
        "unchanged": "원시 입력·공식 점 특징·전처리·라돈·상관·정답·반경·평가식 모두 동일",
        "changed": "point2gridmap의 비원자 if/write 대신 같은 점 특징에 np.fmax.at을 적용",
        "selection": "숫자에 따른 설정 선택 없음; 코드의 max 의미를 결정적으로 계산하는 단일 진단",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    native.save_json(OUT / "protocol.json", protocol)
    for name in ("evaluation_sets.json", "native_data", "raw_sha256.json", "raw_files.txt"):
        dest = OUT / name
        if not dest.exists():
            dest.symlink_to(Path("..") / name)
    print(json.dumps(protocol["diagnostic_variation"], ensure_ascii=False), flush=True)


def install_maxfix() -> None:
    """공식 특징 계산을 유지하고 격자 max만 명시적으로 대체한다."""
    import torch
    import voxelfeat

    core = native.rol.import_ring_core()

    def generate_bev_feats(pc: np.ndarray) -> torch.Tensor:
        indices, _, _, eigens, _ = core.build_neighbors_NN(pc[:, :3], 30)
        features = voxelfeat.GPUFeatureExtractor(
            pc[:, :3].flatten(), len(pc), 13, 30,
            indices.flatten().astype(np.int32), eigens.flatten()).get_features()
        features = features.reshape(-1, 13)[:, [0, 1, 3, 10, 11, 12]]
        coords = pc[:, :3].copy()
        coords[coords == 0] = np.float32(.0001)
        coords[coords > 1] = np.float32(.9999)
        coords[coords < -1] = np.float32(-.9999)
        gap = np.float32(2. / 120.)
        x = np.floor((coords[:, 0].astype(np.float64) + 1.) / float(gap)).astype(np.int64)
        y = np.floor((coords[:, 1].astype(np.float64) + 1.) / float(gap)).astype(np.int64)
        exact = np.zeros((120 * 120, 6), np.float32)
        np.fmax.at(exact, x * 120 + y, features)
        exact = exact.reshape(120, 120, 6).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(exact)).to(core.device)

    core.generate_bev_feats = generate_bev_feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        assert 0 <= args.shard_index < args.shards
        native.OUT = OUT
        install_maxfix()
        native.run(shard_index=args.shard_index, shards=args.shards)
