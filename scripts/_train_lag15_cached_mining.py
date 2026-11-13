#!/usr/bin/env python3
"""고정 원시 특징의 최난도 오답만 캐시하고 원래 학습 명령을 실행한다.

양성은 원래와 같이 매번 다시 뽑는다. NumPy 거리식·공간/시간 조건·동률
처리를 그대로 쓰며, 입력이나 조건이 달라지면 캐시를 다시 만든다.
"""
from __future__ import annotations

import hashlib
import runpy
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
_CACHE = {}


def cached_worker(args: tuple) -> tuple:
    """원래 단일 시퀀스 마이너와 같은 삼중항을 재사용 가능한 오답으로 만든다."""
    from gnn.triplet_miner import _original_lag15_worker
    (seq_id, indices, descriptors, poses, pos_radius, pos_gap, neg_radius,
     neg_gap, strategy, count, metric) = args
    if strategy != "hard" or metric != "l2":
        return _original_lag15_worker(args)
    digest = hashlib.sha256(descriptors.tobytes() + poses.tobytes() + indices.tobytes()).hexdigest()
    key = (seq_id, digest, pos_radius, pos_gap, neg_radius, neg_gap)
    if key not in _CACHE:
        n = len(indices)
        tree = cKDTree(poses[:, :3, 3])
        nearby = tree.query_ball_tree(tree, r=neg_radius)
        positives = tree.query_ball_tree(tree, r=pos_radius)
        all_ids = np.arange(n)
        pools, negatives = [], np.full(n, -1, dtype=np.int64)
        for anchor in range(n):
            pos = np.asarray(positives[anchor], dtype=np.int64)
            pos = pos[(pos != anchor) & (np.abs(pos - anchor) >= pos_gap)]
            pools.append(pos)
            if not len(pos):
                continue
            mask = np.abs(all_ids - anchor) >= neg_gap
            mask[np.asarray(nearby[anchor], dtype=np.int64)] = False
            mask[anchor] = False
            candidates = np.flatnonzero(mask)
            if not len(candidates):
                continue
            distance = np.sum((descriptors[candidates] - descriptors[anchor][None, :]) ** 2, axis=1)
            negatives[anchor] = candidates[np.argmin(distance)]
        _CACHE[key] = pools, negatives
    pools, negatives = _CACHE[key]
    triplets = []
    for anchor, (positive, negative) in enumerate(zip(pools, negatives)):
        if negative < 0:
            continue
        for _ in range(count):
            chosen = np.random.choice(positive)
            triplets.append((indices[anchor], indices[chosen], indices[negative]))
    return seq_id, triplets


def install() -> None:
    """이번 실행에서만 원래 병렬 마이너의 작업 함수를 대체한다."""
    import gnn.triplet_miner as module
    if not hasattr(module, "_original_lag15_worker"):
        module._original_lag15_worker = module._mine_sequence_worker
    module._mine_sequence_worker = cached_worker


def main() -> None:
    """필수 CLI 옵션을 전달한 상태에서 원래 학습 진입점을 실행한다."""
    assert "--encoder-preset" in sys.argv and "no_interdiff" in sys.argv
    assert "--use-gated-context" in sys.argv
    assert "--gate-initial-alpha" in sys.argv and "0.0625" in sys.argv
    install()
    sys.argv[0] = str(REPO / "train_multi_dataset.py")
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
