#!/usr/bin/env python3
"""NSD의 비재방문 점수 상승을 키/후보 목록 교차로 진단한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_evidence as exp
from _causal_pr_dump import prefix_topk
from _pdv2_no_context_ablation import sha256, write_json
from _xs_eval_lib import minmax, align_scores, column_align_scores

OUT = REPO / "results/pdv2_nsd_negative_20261018"


def diagnose(sequence: str) -> None:
    """시드1의 변경된 비재방문 후보와 양성 손실을 전수 교차한다."""
    spec = exp.protocol()
    target = OUT / (sequence.replace("/", "_") + ".json")
    if target.exists():
        return
    sensor, seq, paths, _ = exp.old.base.inputs("val", sequence)
    pos = exp.arrays(paths["cache"])["poses_gt"][:, :3, 3]
    with np.load(paths["spec"]) as z:
        bank = exp.old.base.bank_for(exp.OP, z)
    hkey, fft, norm, column = bank
    conf = exp.read(exp.old.base.BASE / "s1/confidence_selection.json")["ops"][exp.OP]
    weights, decision, tau = conf["tuple"], conf["decision"], conf["tau_star"]
    ledgers, keys, sources = [], [], {str(p.relative_to(REPO)): sha256(p) for p in paths.values()}
    for model in ("capacity_point", "standard"):
        folder = REPO / spec["selected"][model]["1"]["folder"]
        path = folder / f"val_{exp.OP}_{sensor}_{seq}.npz"
        ledgers.append(exp.arrays(path))
        sources[str(path.relative_to(REPO))] = sha256(path)
        path = folder / f"keys_val_{sensor}_{seq}.npz"
        keys.append(exp.arrays(path)["embedding"])
        sources[str(path.relative_to(REPO))] = sha256(path)
    a, b = ledgers
    np.testing.assert_array_equal(a["query_idx"], b["query_idx"])
    negative = a["min_past_gt_dist"] > 25
    changed = a["j_star"] != b["j_star"]
    lost = (a["min_past_gt_dist"] <= 5) & (a["top1_geo_dist"] <= 5) & (b["top1_geo_dist"] > 5)
    rows = []
    for i in np.flatnonzero((negative & changed) | lost):
        q = int(a["query_idx"][i])
        aux_pool = prefix_topk(hkey, q, q-29, 100)
        pools = [np.unique(np.r_[prefix_topk(key, q, q-29, 100), aux_pool]) for key in keys]
        winners, confidence = np.zeros((2, 2), np.int64), np.zeros((2, 2))
        all_high, old_present, feasible = False, False, False
        for pi, pool in enumerate(pools):
            raw_a, raw_c = align_scores(fft, norm, q, pool), column_align_scores(column, q, pool)
            fixed = weights[1]*minmax(hkey[pool]@hkey[q]) + weights[2]*minmax(raw_a) + weights[3]*minmax(raw_c)
            raw = raw_c if decision == "c_col" else raw_a
            for ki, key in enumerate(keys):
                score = weights[0]*minmax(key[pool]@key[q]) + weights[1]*minmax(hkey[pool]@hkey[q])
                score = score + weights[2]*minmax(raw_a)
                score = score + weights[3]*minmax(raw_c)
                best = np.argsort(-score, kind="stable")[0]
                winners[ki, pi], confidence[ki, pi] = pool[best], raw[best]
            if pi == 1:
                all_high = bool(np.all(raw >= tau))
                old_present = bool(a["j_star"][i] in pool)
                if old_present:
                    old_idx = np.flatnonzero(pool == a["j_star"][i])[0]
                    feasible = bool(fixed[old_idx]+weights[0] >= fixed.max()-1e-6)
        assert winners[0, 0] == a["j_star"][i] and winners[1, 1] == b["j_star"][i]
        np.testing.assert_allclose(confidence.diagonal(), [a[decision][i], b[decision][i]], atol=2e-7)
        rows.append({"query": q, "cohort": "negative" if negative[i] else "positive_lost",
            "old_score": float(a[decision][i]), "new_score": float(b[decision][i]),
            "winner_key_by_pool": winners.tolist(), "score_key_by_pool": confidence.tolist(),
            "new_pool_all_above_tau": all_high, "old_winner_in_new_pool": old_present,
            "old_winner_arbitrary_key_bound_feasible": feasible,
            "distance_key_by_pool": np.linalg.norm(pos[winners]-pos[q], axis=2).tolist()})
    write_json(target, {"sequence": sequence, "seed": 1, "negative_queries": int(negative.sum()),
        "changed_negative_winners": int((negative & changed).sum()), "cases": rows,
        "decision": decision, "tau": tau, "sources": sources, "test_used": False})
    print("DIAGNOSED", sequence, len(rows), flush=True)


def main() -> None:
    """진단 대상과 원천을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    exp.protocol(True)
    write_json(OUT / "protocol.json", {"scope": "시드1 검증9의 변경된 비재방문 최종후보와 양성 정답손실 전수. 키/풀 교차는 원래 FP32덧셈순서. 풀효과에정규화포함. 임의키범위가능성은필요조건만이며학습가능성증명아님.",
        "script_sha256": sha256(Path(__file__)), "input_protocol_sha256": sha256(exp.OUT / "protocol.json"), "test_used": False})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for _ in pool.map(diagnose, exp.protocol()["val"]):
            pass
    write_json(OUT / "status.json", {"status": "complete", "test_used": False})


if __name__ == "__main__":
    main()
