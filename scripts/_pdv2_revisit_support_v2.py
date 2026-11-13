#!/usr/bin/env python3
"""반대 방향 학습 쌍의 주변 관측 범위와 단순 시간 평균의 구분력을 진단한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_real_revisit as run
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_revisit_fullbank import analyze, normalize
from _pdv2_revisit_observation import statistics

FULL = REPO / "results/pdv2_revisit_fullbank_20261025"
OUT = REPO / "results/pdv2_revisit_support_20261025/completed"


def neighbors(nodes: np.ndarray, groups: np.ndarray, radius: int) -> tuple:
    """같은 시퀀스 안의 중심 포함 시간 이웃과 유효 표지를 반환한다."""
    ids = nodes[:, None] + np.arange(-radius, radius + 1)
    valid = (ids >= 0) & (ids < len(groups))
    safe = ids.clip(0, len(groups) - 1)
    valid &= groups[safe] == groups[nodes, None]
    return safe, valid


def main() -> None:
    """고정 학습 자료에서만 평균 특징과 GT 주변 경로의 중첩을 측정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "protocol.json").exists()
    torch.set_num_threads(1)
    audit = run.base.read(FULL / "completion_validation.json")
    assert audit["passed"]
    sources = dict(audit["sources"])
    sources[str((FULL / "completion_validation.json").relative_to(REPO))] = sha256(
        FULL / "completion_validation.json"
    )
    sources[str(Path(__file__).relative_to(REPO))] = sha256(Path(__file__))
    for previous in (
        "scripts/_pdv2_revisit_support.py",
        "results/pdv2_revisit_support_20261025/protocol.json",
    ):
        sources[previous] = sha256(REPO / previous)
    write_json(
        OUT / "protocol.json",
        {
            "sources": sources,
            "test_used": False,
            "new_updates": 0,
            "scope": "동일1477질의·8590실제 반대 방향 학습 쌍. 원 특징의 시간±5/±10 단순 평균을 "
            "전체 적격 오답과 비교한다. GT는 주변 경로 중첩 진단에만 사용한다. "
            "입력 스캔 점군의 표면 겹침, 최종 온라인 성능, attention의 인과 효과를 측정하지 않는다.",
            "overlap_rule": "두 주변 경로에서 각각 가장 가까운 점까지2m 이하인 비율을 양방향 평균. "
            "주변 GT 경로 거리의 양방향 평균도 기록한다. 위치 자체를 특징에 넣지 않는다.",
        },
    )
    pool = run.base.exp.arrays(run.OUT / "auxiliary_pool.npz")
    raw = normalize(pool["initial_keys"][:, :288])
    positions, groups = pool["positions"], pool["groups"]
    q, p = pool["anchors"], pool["positives"]
    rows, columns = np.where(p >= 0)
    aq, ap = q[rows], p[rows, columns]
    cross = groups[aq] != groups[ap]
    arrays, averages = {}, {}
    for radius in (5, 10):
        mean = np.zeros_like(raw)
        for start in range(0, len(raw), 1024):
            ids, valid = neighbors(
                np.arange(start, min(start + 1024, len(raw))), groups, radius
            )
            mean[start : start + len(ids)] = (raw[ids] * valid[:, :, None]).sum(
                1
            ) / valid.sum(1)[:, None]
        summary, scores = analyze(mean, pool)
        averages[str(radius)] = summary
        arrays.update(
            {f"mean{radius}_" + name: value for name, value in scores.items()}
        )
        iq, vq = neighbors(aq, groups, radius)
        ip, vp = neighbors(ap, groups, radius)
        dist = np.linalg.norm(
            positions[iq][:, :, None, :] - positions[ip][:, None, :, :], axis=3
        )
        dist = np.where(vq[:, :, None] & vp[:, None, :], dist, np.inf)
        near_q, near_p = dist.min(2), dist.min(1)
        overlap = 0.5 * (
            ((near_q <= 2) & vq).sum(1) / vq.sum(1)
            + ((near_p <= 2) & vp).sum(1) / vp.sum(1)
        )
        chamfer = 0.5 * (
            np.where(vq, near_q, 0).sum(1) / vq.sum(1)
            + np.where(vp, near_p, 0).sum(1) / vp.sum(1)
        )
        arrays[f"overlap{radius}"] = overlap
        arrays[f"path_distance{radius}"] = chamfer
        print("SUPPORT_MEAN", radius, summary["any_positive_win"], flush=True)
    fixed = {}
    for arm in ("start", "context_matched", "context_clipped"):
        data = run.base.exp.arrays(FULL / (arm + ".npz"))
        for name in ("raw", "context", "key"):
            fixed[arm + "_" + name] = (
                data[name + "_positive"] - data[name + "_negative"][:, None]
            )[rows, columns]
    masks = {
        "all": np.ones(len(aq), bool),
        "within_session": ~cross,
        "cross_session": cross,
        "high_path_overlap": arrays["overlap10"] >= 0.8,
        "lower_path_overlap": arrays["overlap10"] < 0.8,
    }
    geometry = {}
    for label, mask in masks.items():
        geometry[label] = {
            "positive_pairs": int(mask.sum()),
            "unique_queries": len(np.unique(aq[mask])),
            "overlap5": statistics(arrays["overlap5"][mask]),
            "overlap10": statistics(arrays["overlap10"][mask]),
            "path_distance10": statistics(arrays["path_distance10"][mask]),
            "pair_margins": {
                name: statistics(value[mask]) for name, value in fixed.items()
            },
        }
    arrays.update(fixed)
    np.savez_compressed(
        OUT / "diagnostics.npz",
        anchors=q,
        positives=p,
        pair_queries=aq,
        pair_candidates=ap,
        cross_session=cross,
        **arrays,
    )
    # 시간 평균은 독립 누적합 계산과 대조한다. 시퀀스 경계를 넘지 않는다.
    independent_error = 0.0
    for radius in (5, 10):
        direct = np.zeros_like(raw)
        for group in np.unique(groups):
            ids = np.flatnonzero(groups == group)
            prefix = np.r_[np.zeros((1, raw.shape[1])), np.cumsum(raw[ids], axis=0)]
            k = np.arange(len(ids))
            lo = np.maximum(k - radius, 0)
            hi = np.minimum(k + radius + 1, len(ids))
            direct[ids] = (prefix[hi] - prefix[lo]) / (hi - lo)[:, None]
        unit = normalize(direct)
        actual = np.einsum("nd,nkd->nk", unit[q], unit[p.clip(0)])
        independent_error = max(
            independent_error,
            float(abs(actual - arrays[f"mean{radius}_positive"]).max()),
        )
    assert independent_error < 1e-10
    write_json(
        OUT / "summary.json",
        {
            "temporal_averages": averages,
            "geometry": geometry,
            "test_used": False,
            "new_updates": 0,
            "independent_mean_error": independent_error,
        },
    )
    for path in OUT.iterdir():
        sources[str(path.relative_to(REPO))] = sha256(path)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest, path
    write_json(
        OUT / "completion_validation.json",
        {
            "passed": True,
            "sources": sources,
            "independent_mean_error": independent_error,
            "positive_pairs": len(aq),
            "test_used": False,
        },
    )
    print(
        "SUPPORT_COMPLETE",
        {label: row["positive_pairs"] for label, row in geometry.items()},
        flush=True,
    )


if __name__ == "__main__":
    main()
