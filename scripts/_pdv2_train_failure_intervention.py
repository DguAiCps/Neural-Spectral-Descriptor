#!/usr/bin/env python3
"""학습 오수락의 키 점수만 낮추는 가상 개입으로 최종 판정 연결을 잰다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
from _pdv2_train_failure import OUT, arrays, read
from _pdv2_matching_mine import spec_path
from _pdv2_lag15_campaign import bank_for
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_sequence_fixed_test import frozen_scores
from _causal_pr_dump import prefix_topk
from _xs_eval_lib import minmax, align_scores, column_align_scores


def main() -> None:
    """기존 풀·고정 융합에서 승자 또는 그 장소의 키 점수만 0으로 놓는다."""
    spec = read(OUT/"protocol.json")
    names = spec["train_roster"]
    paths = [Path(__file__), OUT/"protocol.json", OUT/"retrieval_keys.npy", OUT/"training_metadata.npz"]
    paths += [OUT/("ledger_"+name.replace("/", "_")+".npz") for name in names]
    paths += [spec_path(name) for name in names]
    record = {"sources": {str(p.relative_to(REPO)): sha256(p) for p in paths},
        "scope": "학습 오수락 전수. 현재 후보 풀과 비키 점수 고정. 승자 1개의 정규화된 키 점수 또는 승자5m이내 과거 후보들의 키 점수를 0으로 만든다. 다른 키 점수의 재정규화나 새 후보 검색은 하지 않는다.",
        "limits": "학습으로 실현 가능한 임베딩이라는 보장이 없는 국소 점수 개입. 장소 개입은 학습 GT를 쓰는 진단용 가상 조건이다. 모델 개선 성능·검색 전체 가능성의 상한이 아니다.",
        "test_used": False}
    protocol = OUT/"intervention_protocol.json"
    if protocol.exists():
        assert read(protocol) == record
    else:
        write_json(protocol, record)
    metadata = arrays(OUT/"training_metadata.npz")
    keys = np.load(OUT/"retrieval_keys.npy", mmap_mode="r")
    conf = spec["selected"]["confidence"]
    weights, channel = conf["tuple"], conf["decision"]
    tau = spec["selected"]["new_tau"]["tau"]
    report = {}
    for ordinal, sequence in enumerate(names):
        name = sequence.replace("/", "_")
        ids = np.flatnonzero(metadata["sequences"] == ordinal)
        key, pos = keys[ids], metadata["positions"][ids]
        original = arrays(OUT/f"ledger_{name}.npz")
        selected = (original["min_past_gt_dist"] > 25) & (original["arc_unique_b100"] >= tau)
        q, old_j = original["query_idx"][selected], original["j_star"][selected]
        if not len(q):
            report[sequence] = {"false_accept": 0}
            continue
        with np.load(spec_path(sequence)) as z:
            bank = bank_for("pbev128_main", z)
        hkey, fft, norm, column = bank
        proposed = {k: np.empty(len(q), np.int64) for k in ("winner_key_zero", "place_key_zero")}
        raw = {k: np.empty(len(q)) for k in proposed}
        auxiliary = np.zeros(len(q), bool)
        for row, (query, winner) in enumerate(zip(q, old_j)):
            kp = prefix_topk(key, query, query-29, 100)
            hp = prefix_topk(hkey, query, query-29, 100)
            pool = np.unique(np.r_[kp, hp])
            sn, sh = minmax(key[pool]@key[query]), minmax(hkey[pool]@hkey[query])
            spectral, columns = align_scores(fft, norm, query, pool), column_align_scores(column, query, pool)
            sa, sc = minmax(spectral), minmax(columns)
            baseline = weights[0]*sn + weights[1]*sh + weights[2]*sa + weights[3]*sc
            assert pool[np.argsort(-baseline, kind="stable")[0]] == winner
            current_raw = columns if channel == "c_col" else spectral
            auxiliary[row] = winner in hp
            for condition in proposed:
                altered = sn.copy()
                affected = pool == winner if condition == "winner_key_zero" else np.linalg.norm(pos[pool]-pos[winner], axis=1) <= 5
                altered[affected] = 0
                score = weights[0]*altered + weights[1]*sh + weights[2]*sa + weights[3]*sc
                index = np.argsort(-score, kind="stable")[0]
                proposed[condition][row], raw[condition][row] = pool[index], current_raw[index]
        travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
        saved = {"query": q, "winner": old_j, "original_score": original["arc_unique_b100"][selected],
                 "winner_in_aux_pool": auxiliary}
        rows = {"false_accept": len(q), "winner_in_aux_pool": int(auxiliary.sum())}
        for condition, winner in proposed.items():
            scores = []
            for start in range(0, len(q), 256):
                sl = slice(start, start+256)
                scores.append(frozen_scores(bank, travel, q[sl], winner[sl], channel, raw[condition][sl])["arc_unique_b100"])
            confidence = np.concatenate(scores)
            changed = winner != old_j
            np.testing.assert_allclose(confidence[~changed], saved["original_score"][~changed], atol=1e-12, rtol=0)
            rows[condition] = {"winner_changed": int(changed.sum()), "now_rejected": int((confidence < tau).sum()),
                               "still_false_accept": int((confidence >= tau).sum())}
            saved[condition+"_winner"], saved[condition+"_score"] = winner, confidence
        path = OUT/f"intervention_{name}.npz"
        np.savez_compressed(path, **saved)
        rows["sha256"] = sha256(path)
        report[sequence] = rows
        print("INTERVENTION", sequence, rows, flush=True)
    totals = {}
    for group in ("all", "kitti", "helipr"):
        subset = [v for k, v in report.items() if v["false_accept"] and (group == "all" or k.startswith(group))]
        totals[group] = {"false_accept": sum(r["false_accept"] for r in subset),
                         "winner_in_aux_pool": sum(r["winner_in_aux_pool"] for r in subset)}
        for condition in ("winner_key_zero", "place_key_zero"):
            totals[group][condition] = {k: sum(r[condition][k] for r in subset)
                                       for k in ("winner_changed", "now_rejected", "still_false_accept")}
    write_json(OUT/"intervention_summary.json", {"status": "passed", "sequences": report, "groups": totals,
        "protocol_sha256": sha256(protocol), "scope": record["scope"], "limits": record["limits"], "test_used": False})
    print("INTERVENTION_TOTAL", totals, flush=True)


if __name__ == "__main__":
    main()
