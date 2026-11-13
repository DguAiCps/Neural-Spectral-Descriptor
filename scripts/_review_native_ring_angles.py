#!/usr/bin/env python3
"""논문3도 각도 간격과 공개 GPU경로 끝점 중복을 단일요인 대조한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import _review_native_ring as native
import _review_native_ring_maxfix as maxfix

BASE = native.OUT
OUT = BASE / "ring_angles_endpoint_false"


def prepare() -> None:
    """각도 끝점만 바꾸는 대조를 성능 확인 전에 고정한다."""
    OUT.mkdir(exist_ok=True)
    protocol = json.loads((BASE / "maxfix/protocol.json").read_text())
    protocol["purpose"] = "결정적 max를 고정하고 Radon 각도 끝점 중복만 제거하는 native 원인 진단"
    protocol["angle_factor"] = {
        "control": "maxfix: np.linspace(0,2*pi,120), 3.025210084도, 0/2pi 중복",
        "diagnostic": "np.linspace(0,2*pi,120,endpoint=False), 논문120×120의3도 간격",
        "unchanged": "동일 원시·점 특징·정확 max·라돈 구현·FFT·전체지도상관·정답·평가분모",
        "selection": "소스와 논문의 명시적 차이에서 고정; 시험 성능으로 선택하지 않으며46주표 격자max재평가는 바꾸지 않음",
        "source": "_external/RING/utils/core.py:152,160; 논문 구현 세부의360/120=3도",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination = OUT / "protocol.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == protocol
    else:
        native.save_json(destination, protocol)
    for name in ("evaluation_sets.json", "native_data", "raw_sha256.json", "raw_files.txt"):
        dest = OUT / name
        if not dest.exists():
            dest.symlink_to(Path("..") / name)


def run() -> None:
    """동일 공식 함수 경로에서 각도 벡터만 명시적으로 바꾼다."""
    import torch
    prepare()
    native.OUT = OUT
    maxfix.install_maxfix()
    core = native.rol.import_ring_core()

    def generate_ring(bev):
        angles = torch.FloatTensor(np.linspace(0, 2 * np.pi, core.cfg.num_ring, endpoint=False).astype(np.float32))
        operator = core.ParallelBeam(core.cfg.num_sector, angles)
        ring = operator.forward(bev)
        tiring, _ = core.forward_row_fft(ring)
        return ring, tiring

    core.generate_RING = generate_ring
    native.run()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "run"])
    a = p.parse_args()
    prepare() if a.stage == "prepare" else run()
