#!/usr/bin/env python3
"""기존 학습 표본이 같은 세션의 과거 전용 판정을 얼마나 다루는지 센다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/_pdv2_training_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / "results/pdv2_training_coverage_20261018"


def prefix_distance(positions: np.ndarray, block: int = 256) -> np.ndarray:
    """각 질의에서 30개 이상 이전의 최근접 거리만 계산한다."""
    pos = np.asarray(positions, dtype=np.float64)
    output = np.full(len(pos), np.nan)
    for start in range(30, len(pos), block):
        end = min(start+block, len(pos))
        cut = start-29
        base = cKDTree(pos[:cut]).query(pos[start:end], k=1)[0]
        tail = np.arange(cut, end-29)
        distances = np.linalg.norm(pos[start:end, None]-pos[tail][None], axis=2)
        distances[tail[None] > np.arange(start,end)[:, None]-30] = np.inf
        output[start:end] = np.minimum(base, distances.min(1))
    return output


def main() -> None:
    """학습 자세·네 표본 묶음만 읽고 원인 해석과 범위를 저장한다."""
    import torch
    source = REPO / "results/pdv2_pair_sampling_20261013/prepared.pt"
    state = torch.load(source, map_location="cpu", weights_only=False)
    pos, seq = np.asarray(state["poses"], dtype=np.float64)[:, :3, 3], np.asarray(state["sequences"])
    OUT.mkdir(parents=True, exist_ok=True)
    distances, local_index = np.full(len(pos), np.nan), np.zeros(len(pos), np.int64)
    by_seq = {}
    for s in np.unique(seq):
        ids = np.flatnonzero(seq==s)
        assert (np.diff(ids)==1).all()
        d = prefix_distance(pos[ids])
        for q in np.linspace(30,len(ids)-1,min(20,len(ids)-30),dtype=int):
            expected = np.linalg.norm(pos[ids[:q-29]]-pos[ids[q]], axis=1).min()
            np.testing.assert_allclose(d[q], expected, atol=1e-12, rtol=0)
        distances[ids], local_index[ids] = d, np.arange(len(ids))
        by_seq[str(s)] = {"frames":len(ids), "positive":int((d<=5).sum()), "negative":int((d>25).sum()),
                          "ambiguous":int(((d>5)&(d<=25)).sum()), "first30":int(np.isnan(d).sum())}
    buckets = {"positive": distances<=5, "negative": distances>25,
               "ambiguous": (distances>5)&(distances<=25), "first30": np.isnan(distances)}
    results, sources = {}, {str(source.relative_to(REPO)):sha256(source),
                           str(Path(__file__).relative_to(REPO)):sha256(Path(__file__))}
    for epoch in (1,6,11,16):
        path = REPO / f"results/pdv2_temporal_focus_20261018/samples/original_epoch{epoch}.npz"
        sources[str(path.relative_to(REPO))] = sha256(path)
        with np.load(path) as z:
            a,p,n = z["triplets"].T
        same = seq[a]==seq[p]
        past_positive = same & (local_index[p]<=local_index[a]-30)
        past_negative = (seq[a]==seq[n]) & (local_index[n]<=local_index[a]-30)
        assert (np.linalg.norm(pos[a]-pos[p],axis=1)<=5).all()
        rows = {}
        for group, mask in buckets.items():
            take = mask[a]
            total = int(mask.sum())
            unique = len(np.unique(a[take]))
            rows[group] = {"available_frames":total, "triplet_rows":int(take.sum()), "unique_anchors":unique,
                           "unique_anchor_coverage":unique/total if total else None,
                           "positive_past_same_session":int((take&past_positive).sum()),
                           "positive_future_same_session":int((take&same&~past_positive).sum()),
                           "positive_other_session":int((take&~same).sum()),
                           "explicit_negative_past_same_session":int((take&past_negative).sum())}
        assert rows["negative"]["positive_past_same_session"] == 0
        results[str(epoch)] = rows
    write_json(OUT / "summary.json", {"by_sequence":by_seq,"epochs":results,"sources":sources,
        "test_used":False,"validation_used":False,"prefix_checks":sum(min(20,r["frames"]-30) for r in by_seq.values()),
        "scope":"학습 GT 자세의 같은 세션 안 j<=q-30 분류.",
        "interpretation":"학습의 공통 좌표 세션 사이 양성은 기존 규약대로 유효하다. 이 건수는 잘못된 라벨을 뜻하지 않는다. 추론 때 과거 지도에 정답이 없는 질의와 미래/다른 세션 양성을 쓰는 학습의 목적 차이를 센다. 검증 실패가 같은 경로인지는 측정하지 않았다."})
    print("TRAIN_COVERAGE", results, flush=True)


if __name__ == "__main__":
    main()
