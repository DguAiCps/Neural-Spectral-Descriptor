#!/usr/bin/env python3
"""논문 표와 공개 코드의 고유값 엔트로피 정규화 차이만 진단한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import _review_native_ring as native
import _review_native_ring_maxfix as maxfix

BASE = native.OUT
OUT = BASE / "ring_entropy_formula"


def prepare() -> None:
    """시험값을 보기 전에 논문에 적힌 한 수식만 대조 조건으로 고정한다."""
    OUT.mkdir(exist_ok=True)
    protocol = json.loads((BASE / "maxfix/protocol.json").read_text())
    protocol["purpose"] = "논문 Table I의 고유값 엔트로피 수식과 공개 구현의 정규화 차이 진단"
    protocol["entropy_factor"] = {
        "control": "공개코드: p=lambda/sum(lambda), H=-sum(p*log(p))",
        "diagnostic": "논문 Table I: H=-sum(lambda*log(lambda))",
        "implementation": "같은 GPU의 H와 같은 eigenvalue sum S로 H_literal=S*H-S*log(S) 계산. 다른 12특징은 비트 동일 유지",
        "unchanged": "원시·공식 정규화 좌표·30NN·고유값·다른5선택특징·정확max·원래 endpoint=True 각도·전체지도검색·모든반경·집계",
        "limits": "논문 표의 오기 가능성 있음. 이 조건을 수정본/정답 구현이라고 채택하지 않으며 공통46에도 적용하지 않음",
        "source": "논문 PDF 7쪽 Table I, _external/RING/utils/generate_bev_pointfeat_cython/src/kernel.cu:54-58",
        "source_pdf_sha256": hashlib.sha256((BASE / "ringpp_paper.pdf").read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": "코드-논문 수식 차이에서 사전 고정한 단일요인. 시험값으로 선택하지 않음"}
    destination = OUT / "protocol.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == protocol
    else:
        native.save_json(destination, protocol)
    for name in ["evaluation_sets.json", "native_data", "raw_sha256.json", "raw_files.txt"]:
        path = OUT / name
        if not path.exists():
            path.symlink_to(Path("..") / name)


def install() -> list:
    """공식 GPU 특징의 엔트로피 열만 수학적으로 변환한다."""
    import voxelfeat
    original = voxelfeat.GPUFeatureExtractor
    checks = []

    class Extractor:
        def __init__(self, *args):
            self.extractor = original(*args)
            self.eigens = np.asarray(args[5], dtype=np.float32).reshape(-1, 5)[:, :3]

        def get_features(self):
            values = np.asarray(self.extractor.get_features()).reshape(-1, 13)
            result = values.copy()
            total = self.eigens.sum(axis=1, dtype=np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                result[:, 3] = total * values[:, 3] - total * np.log(total)
                valid = (self.eigens > 0).all(axis=1) & np.isfinite(values[:, 3])
                direct = -(self.eigens[valid].astype(np.float64) * np.log(self.eigens[valid].astype(np.float64))).sum(axis=1)
            error = float(np.max(np.abs(result[valid, 3] - direct))) if valid.any() else 0.0
            keep = [i for i in range(13) if i != 3]
            assert np.array_equal(result[:, keep], values[:, keep], equal_nan=True)
            assert error < 1e-6, error
            checks.append({"points": len(values), "positive_eigenvalue_points": int(valid.sum()),
                           "formula_max_abs_error": error, "other_12_features_bitwise_equal": True})
            return result.reshape(-1)

    voxelfeat.GPUFeatureExtractor = Extractor
    maxfix.install_maxfix()
    return checks


def smoke() -> None:
    """고정 처음·중간·마지막 스캔에서 다른5격자채널 불변을 확인한다."""
    import torch
    native.imports()
    from datasets.NCLTDataset import NCLTPointCloudLoader
    core = native.rol.import_ring_core()
    maxfix.install_maxfix()
    loader = NCLTPointCloudLoader()
    paths = (BASE / "raw_files.txt").read_text().splitlines()
    selected = [paths[i] for i in [0, len(paths) // 2, len(paths) - 1]]
    points, baseline = [], []
    for name in selected:
        pc = loader.normalize_pc(loader.read_pc(str(BASE / "native_data" / name)))
        points.append(pc)
        with torch.no_grad():
            baseline.append(core.generate_bev_feats(pc).cpu().numpy())
    checks = install()
    rows = []
    for name, pc, before in zip(selected, points, baseline):
        with torch.no_grad():
            after = core.generate_bev_feats(pc).cpu().numpy()
        keep = [0, 1, 3, 4, 5]
        assert np.array_equal(before[keep], after[keep])
        assert np.isfinite(after).all()
        rows.append({"raw_path": name, "unchanged_5_bev_channels_bitwise_equal": True,
                     "entropy_bev_max_abs_change": float(np.max(np.abs(before[2] - after[2])))})
    native.save_json(OUT / "smoke.json", {"passed": True, "rows": rows, "feature_checks": checks})


def run() -> None:
    """고정 규약으로 전체 재추론하고 수식 동치 검사도 보존한다."""
    assert json.loads((OUT / "smoke.json").read_text())["passed"]
    checks = install()
    native.OUT = OUT
    native.run()
    native.save_json(OUT / "feature_formula_checks.json", {"rows": checks, "passed": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "smoke", "run"])
    args = parser.parse_args()
    prepare()
    if args.stage == "smoke":
        smoke()
    elif args.stage == "run":
        run()
