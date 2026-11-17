#!/usr/bin/env python3
"""옛 fig_geodesc(지리 거리 대 서술자 거리)의 현재 구조판 자료: 재방문 쿼리의 이중 색인 후보마다
(1) 색인 점수(두 색인 코사인의 융합 부분합, 정렬 없음)와 (2) 최종 선택 점수(네 융합 채널, 정제 스펙트럼의 순환 정렬 포함)를 저장한다.
경로는 scripts/_report_model_ablations_20261116.py와 같다(no_alignment의 static, NSD 승자 선택의 value). NSD 시드 4.
docker exec -w /workspace/Neural-Spectral-Codec ring_baseline python3 scripts/_geodesc_v24_export_20261117.py SEQ [SEQ...]
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _report_model_ablations_20261116 as A  # noqa: E402

OUT = ROOT / "results/pdv2_geodesc_v24_20261117"


def main(seq):
    import torch
    S = A.modules()
    P, run, prior, pipeline = S.P, S.run, S.prior, S.pipeline
    name = seq.replace("/", "_")
    cache, pos, minimum, travel, bank, _, _, _ = P.inputs("test", seq)
    graph, _, _ = prior.graph_inputs("test", seq)
    raw = pipeline.raw_key(cache["descriptors"])
    new_w, lam = P.selected()
    pool = run.ARR(P.TEST / "test" / name / "candidate_pool.npz")
    q, off = pool["query"], pool["offset"]
    with torch.no_grad():
        net = S.load(A.CFG, 4, A.UPD)
        change = run.base.variant_inputs(net(graph).cpu().numpy(), 8)
    rows = dict(query=[], cand=[], geo=[], index=[], final=[])
    for n, t in enumerate(q):
        if minimum[t] > 5:  # 재방문 쿼리만
            continue
        ids = pool["ids"][off[n]:off[n + 1]]
        stat = (new_w[0] * run.base.minmax(raw[ids] @ raw[t]) + new_w[1] * run.base.minmax(bank[0][ids] @ bank[0][t])).astype(np.float32)
        sa = run.base.align_scores(change[1], change[2], int(t), ids)
        sc = run.base.column_align_scores(change[3], int(t), ids)
        val = stat + new_w[2] * run.base.minmax(sa) + new_w[3] * run.base.minmax(sc)
        rows["query"].append(np.full(len(ids), t)); rows["cand"].append(ids)
        rows["geo"].append(np.linalg.norm(pos[ids] - pos[t], axis=1))
        rows["index"].append(stat / (new_w[0] + new_w[1])); rows["final"].append(val / new_w.sum())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", **{k: np.concatenate(v) for k, v in rows.items()},
                        weights=np.asarray(new_w), note="index = 색인 두 항 융합/가중합, final = 네 채널 융합/가중합 (쿼리 안 min-max 항의 가중 평균, 0~1)")
    led = np.load(S.LEDGERS / "test" / name / f"{A.CFG}@s4@u{A.UPD}.npz")
    # 검산: 최종 점수 최대 후보 = 원장 승자
    d = np.load(OUT / f"{name}.npz")
    wins = {}
    for t in np.unique(d["query"]):
        m = d["query"] == t
        wins[int(t)] = int(d["cand"][m][np.argmax(d["final"][m])])
    lq = dict(zip(led["query_idx"].tolist(), led["j_star"].tolist()))
    agree = np.mean([wins[t] == lq[t] for t in wins])
    print(seq, "queries", len(wins), "winner agreement", agree)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
