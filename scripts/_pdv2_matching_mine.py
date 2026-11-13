#!/usr/bin/env python3
"""학습 분할에서 최종 검색·정렬로 어려운 오답을 수집한다.

ring_baseline 안에서 실행한다. 검증·시험 시퀀스를 받지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_pilot import TRAIN
from _pdv2_no_context_ablation import no_context_key, sha256, write_json

OUT = REPO / "results/pdv2_matching_mining_20261014"
RVEC = REPO / "artifacts/key_remetrize_r_pdv2.npy"
BASE = REPO / "results/pdv2_symmetric3_20261013"


def cache_path(sequence: str) -> Path:
    """허용된 학습 시퀀스의 불변 특징 캐시만 반환한다."""
    sensor, seq = sequence.split("/")
    assert (sensor, seq) in TRAIN
    paths = list((REPO / "data/preprocessed_v2").glob(f"cache_*_{sensor}_{seq}.npz"))
    assert len(paths) == 1, paths
    return paths[0]


def spec_path(sequence: str) -> Path:
    """실험 전용 학습 스펙트럼 위치를 반환한다."""
    cache_path(sequence)
    return OUT / "mining" / (sequence.replace("/", "_") + "_spec.npz")


def spectra(sequence: str) -> None:
    """기존 학습 키프레임의 자른 BEV와 높이 극좌표를 생성한다."""
    from _pd_build_test_caches import _make_loader
    from _pbev_branch_fusion import BP
    from _height_candidate_fusion import make_height_grids
    from encoding.phase_features import _phase_sketch, _pool_rows
    target = spec_path(sequence)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = cache_path(sequence)
    if target.exists():
        meta = json.loads(target.with_suffix(".json").read_text())
        assert meta["cache_sha256"] == sha256(source)
        return
    sensor, seq = sequence.split("/")
    with np.load(source) as cache:
        ids = cache["scan_ids"]
    loader = _make_loader(sensor, seq)
    re = np.empty((len(ids), 16, 12), np.float32)
    im = np.empty_like(re)
    heights = np.empty((len(ids), 20, 120), np.float16)
    started = time.time()
    for row, sid in enumerate(ids):
        points = loader[int(sid)]["points"]
        bev, _ = BP.project(points, keep_intensity=False)
        sketch = _phase_sketch(_pool_rows(np.maximum(bev, 0.), 16), 12)
        re[row], im[row] = sketch.real, sketch.imag
        _, raw_height, _, _ = make_height_grids(points)
        heights[row] = raw_height
        if row % 1000 == 0:
            print("SPECTRA", sequence, row, len(ids), round(time.time() - started, 1), flush=True)
    assert all(np.isfinite(a).all() for a in (re, im, heights))
    temp = target.with_suffix(".tmp.npz")
    np.savez_compressed(temp, scan_ids=ids, sk_re=re, sk_im=im, raw_height=heights)
    temp.replace(target)
    write_json(target.with_suffix(".json"), {"sequence": sequence, "n": len(ids),
               "cache_sha256": sha256(source), "sha256": sha256(target),
               "source_sha256": sha256(Path(__file__)), "runtime_s": time.time() - started})
    print("SPECTRA_DONE", sequence, flush=True)


def main() -> None:
    """학습 전용 준비 단계를 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("spectra",))
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    spectra(args.sequence)


if __name__ == "__main__":
    main()
