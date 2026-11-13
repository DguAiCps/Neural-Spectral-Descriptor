#!/usr/bin/env python3
"""가장 큰 검증 손실 시퀀스에서 문맥 방향과 세기의 영향을 분리한다."""
from pathlib import Path

import numpy as np
import torch

import _pdv2_diffattn_reliability as exp


def parts(key: np.ndarray) -> tuple:
    """전체 정규화만 풀어 원 특징·세기·단위 문맥을 분리한다."""
    norm = np.linalg.norm(key[:, :288], axis=1, keepdims=True)
    raw = key[:, :288] / np.maximum(norm, 1e-12)
    ctx = key[:, 288:] / np.maximum(norm, 1e-12)
    alpha = np.linalg.norm(ctx, axis=1, keepdims=True)
    return raw, alpha, ctx / np.maximum(alpha, 1e-12)


def assemble(raw: np.ndarray, alpha: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    """분석용으로 다른 모델의 세기와 방향을 조립한다."""
    key = np.concatenate([raw, alpha * ctx], -1)
    return key / np.maximum(np.linalg.norm(key, axis=1, keepdims=True), 1e-12)


def main() -> None:
    """같은 승자에서 교체 개입과 거리 성분을 계산한다."""
    exp.activate()
    seq, name = "mulran/Sejong02", "mulran_Sejong02"
    folder = exp.OUT / "frozen/epoch0/val"
    row = exp.run.ARR(folder / (name + ".npz"))
    data = exp.run.ARR(folder / (name + "_evidence.npz"))
    cache, *_ = exp.run.inputs("val", seq)
    models = [exp.model(), exp.model(exp.OUT / "attenuate_lr4/epoch8.pth")]
    keys = [
        exp.run.source.make_keys(net, cache["descriptors"], cache["poses"])
        for net in models
    ]
    raw, a0, c0 = parts(keys[0])
    _, a1, c1 = parts(keys[1])
    variants = dict(
        frozen=keys[0],
        learned=keys[1],
        learned_direction=assemble(raw, a0, c1),
        learned_strength=assemble(raw, a1, c0),
    )
    metrics, scores = {}, {}
    for label, key in variants.items():
        value = exp.score_all(key, data, ("attenuate",))["attenuate_1"]
        scores[label] = value
        metrics[label] = exp.run.base.independent_metrics(row, value)
    eligible = (row["min_past_gt_dist"] <= 5) | (row["min_past_gt_dist"] > 25)
    correct = (row["min_past_gt_dist"] <= 5) & (row["top1_geo_dist"] <= 5)
    wrong = eligible & ~correct
    thresholds = {k: float(v[wrong].max()) for k, v in scores.items()}
    lost = (
        correct
        & (scores["frozen"] > thresholds["frozen"])
        & (scores["learned"] <= thresholds["learned"])
    )
    components = {}
    qi, ji, w = data["query"], data["candidate"], data["weight"]
    for label, (a, c) in (("frozen", (a0, c0)), ("learned", (a1, c1))):
        magnitude = 0.5 * (a[qi, 0] - a[ji, 0]) ** 2
        angle = a[qi, 0] * a[ji, 0] * (1 - (c[qi] * c[ji]).sum(-1))
        full = 0.5 * ((a * c)[qi] - (a * c)[ji]).__pow__(2).sum(-1)
        np.testing.assert_allclose(magnitude + angle, full, rtol=2e-5, atol=2e-6)
        mean_magnitude = (w * magnitude).sum(-1).mean(-1)
        mean_angle = (w * angle).sum(-1).mean(-1)
        components[label] = {
            group: dict(
                count=int(mask.sum()),
                magnitude=float(mean_magnitude[mask].mean()) if mask.any() else None,
                angle=float(mean_angle[mask].mean()) if mask.any() else None,
            )
            for group, mask in (
                ("correct", correct),
                ("wrong", wrong),
                ("lost_precision", lost),
            )
        }
    exp.run.WRITE(
        exp.OUT / "sejong02_components.json",
        dict(
            sequence=seq,
            metrics=metrics,
            components=components,
            wrong_score_max=thresholds,
            lost_queries=row["query_idx"][lost].tolist(),
            scope="결과를 본 뒤 선택한 최대 손실 시퀀스의 사후 진단. 초기 승자 고정·gamma1. 두 모델 성분 교체는 배포 모델이 아니다. 거리 성분 평균은 두 방향 평균이고 실제 최종 점수는 채널별 최대다.",
            test_used=False,
            source_sha256=exp.run.HASH(Path(__file__)),
        ),
    )
    print("COMPONENTS", metrics, components, "lost", int(lost.sum()), flush=True)


if __name__ == "__main__":
    exp.run.configure()
    main()
