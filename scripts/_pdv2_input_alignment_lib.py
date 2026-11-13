"""자세·키프레임 선택 대조의 시간 그래프와 같은 실제 스캔 쌍을 만든다."""
from typing import Tuple

import numpy as np


def temporal_edges(sequences: np.ndarray) -> np.ndarray:
    """같은 시퀀스의 ±5개 이웃을 기존 노드·오프셋 순서로 연결한다."""
    n = len(sequences)
    offsets = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    source = np.broadcast_to(np.arange(n)[:, None], (n, 10))
    target = source + offsets[None]
    valid = (target >= 0) & (target < n)
    safe = np.clip(target, 0, n - 1)
    valid &= sequences[source] == sequences[safe]
    return np.stack((source[valid], target[valid]))


def pose_attributes(poses: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """좌표계에 불변인 거리·회전각으로 기존 시간 간선 5개 속성을 만든다."""
    source, target = edges
    distance = np.linalg.norm(poses[source, :3, 3] - poses[target, :3, 3], axis=1)
    rotation = poses[target, :3, :3] @ np.transpose(poses[source, :3, :3], (0, 2, 1))
    trace = np.clip(np.trace(rotation, axis1=1, axis2=2), -1., 3.)
    angle = np.arccos(np.clip((trace - 1.) / 2., -1., 1.))
    attributes = np.zeros((len(source), 5), np.float32)
    attributes[:, 0] = np.log1p(distance) / 5.
    attributes[:, 1] = angle / np.pi
    attributes[:, 4] = 1.
    assert np.isfinite(attributes).all()
    return attributes


def make_graph(raw: np.ndarray, poses: np.ndarray, sequences: np.ndarray):
    """추론과 같은 양방향 시간 그래프를 만든다."""
    import torch
    from torch_geometric.data import Data
    edges = temporal_edges(sequences)
    return Data(x=torch.from_numpy(raw.astype(np.float32)),
                edge_index=torch.from_numpy(edges),
                edge_attr=torch.from_numpy(pose_attributes(poses, edges)),
                edge_type=torch.zeros(edges.shape[1], dtype=torch.long))


def common_triplets(triplets: np.ndarray, gt_to_odom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """세 스캔이 공통이고 두 프레임 번호에서도 시차 30을 지킨 원 행만 남긴다."""
    mapped = gt_to_odom[triplets]
    keep = np.all(mapped >= 0, axis=1)
    for values in (triplets, mapped):
        keep &= np.abs(values[:, 0] - values[:, 1]) >= 30
        keep &= np.abs(values[:, 0] - values[:, 2]) >= 30
    rows = np.flatnonzero(keep)
    return rows, mapped[rows]
