#!/usr/bin/env python3
"""공식 RING++ 특징 격자의 반복성·최댓값 집계를 고정 세 입력으로 검사한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_native_20261105"
sys.path.insert(0, str(REPO / "scripts"))
import _ring_official_lib as rol


def main() -> None:
    """결과와 무관하게 목록의 첫·중간·마지막 원시 입력을 검사한다."""
    import torch
    import voxelfeat

    torch.set_num_threads(4)
    core = rol.import_ring_core()
    from datasets.NCLTDataset import NCLTPointCloudLoader
    paths = (OUT / "raw_files.txt").read_text().splitlines()
    picks = [paths[i] for i in (0, len(paths) // 2, len(paths) - 1)]
    loader = NCLTPointCloudLoader()
    report = {"selection_rule": "결과와 무관한 정렬 입력 목록의 첫·중간·마지막", "repetitions": 5,
              "scope": "공식 비원자 최댓값 CUDA 집계의 진단만 수행; native 결과·공식 소스를 변경하지 않음",
              "source": "_external/RING/utils/generate_bev_pointfeat_cython/src/kernel.cu:153",
              "voxelfeat_binary_sha256": hashlib.sha256(Path(voxelfeat.__file__).read_bytes()).hexdigest(),
              "scans": []}
    for rel in picks:
        pc = loader.normalize_pc(loader.read_pc(str(OUT / "native_data" / rel))[:, :3])
        indices, _, _, eigens, _ = core.build_neighbors_NN(pc, 30)
        features = voxelfeat.GPUFeatureExtractor(pc.flatten(), len(pc), 13, 30,
                                                indices.flatten().astype(np.int32), eigens.flatten()).get_features()
        features = features.reshape(-1, 13)[:, [0, 1, 3, 10, 11, 12]]
        combined = np.concatenate([pc, features], axis=1)
        flattened = combined.T.flatten().astype(np.float32)
        coords = pc.copy()
        coords[coords == 0] = np.float32(.0001)
        coords[coords > 1] = np.float32(.9999)
        coords[coords < -1] = np.float32(-.9999)
        gap = np.float32(2. / 120.)
        ix = np.floor((coords[:, 0].astype(np.float64) + 1.) / float(gap)).astype(np.int64)
        iy = np.floor((coords[:, 1].astype(np.float64) + 1.) / float(gap)).astype(np.int64)
        exact = np.zeros((120 * 120, 6), np.float32)
        np.fmax.at(exact, ix * 120 + iy, features)
        exact = exact.reshape(120, 120, 6).transpose(2, 0, 1)
        runs = []
        for repeat in range(5):
            trans = voxelfeat.GPUTransformer(flattened, len(pc), 1, 1, 120, 120, 1, 9)
            trans.transform()
            gpu = trans.retreive().reshape(120, 120, 9)[..., 3:].transpose(2, 0, 1)
            runs.append(gpu.copy())
        per = {"raw_path": rel, "n_points": len(pc),
               "nonfinite_point_features": int((~np.isfinite(features)).sum()),
               "exact_nonzero_cells": int(np.count_nonzero(exact)),
               "repeat_max_abs_diff": [float(np.max(np.abs(a - runs[0]))) for a in runs],
               "repeat_different_values": [int(np.count_nonzero(a != runs[0])) for a in runs],
               "max_abs_diff_from_exact_max": [float(np.max(np.abs(a - exact))) for a in runs],
               "different_values_from_exact_max": [int(np.count_nonzero(a != exact)) for a in runs]}
        print(json.dumps(per), flush=True)
        report["scans"].append(per)
    (OUT / "kernel_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
