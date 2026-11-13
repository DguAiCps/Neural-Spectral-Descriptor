#!/usr/bin/env python3
"""최종 순위 손실에서 검색 키 변화와 후보 목록 변화를 교차 진단한다."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_policy_training as exp
from _causal_pr_dump import prefix_topk
from _xs_eval_lib import align_scores, column_align_scores, minmax
from _pdv2_no_context_ablation import sha256, write_json


def candidate_pool(emb: np.ndarray, bank: tuple, query: int) -> np.ndarray:
    """동일한 100+100 과거 후보 합집합을 재생성한다."""
    return np.unique(np.r_[prefix_topk(emb, query, query - 29, 100),
                           prefix_topk(bank[0], query, query - 29, 100)])


def rank(emb: np.ndarray, bank: tuple, query: int, candidates: np.ndarray,
         weights: tuple) -> int:
    """후보 목록을 고정하고 기존 네 점수 융합의 1등을 반환한다."""
    hkey, fft, norm, column = bank
    sn = minmax(emb[candidates] @ emb[query])
    sh = minmax(hkey[candidates] @ hkey[query])
    sa = minmax(align_scores(fft, norm, query, candidates))
    sc = minmax(column_align_scores(column, query, candidates))
    w = weights
    score = w[0] * sn + w[1] * sh + w[2] * sa + w[3] * sc
    assert score.dtype == np.float32 and np.isfinite(score).all()
    return int(candidates[np.argsort(-score, kind="stable")[0]])


def comparisons() -> list:
    """같은 모델의 추론 변화와 같은 추론의 학습 변화를 분리한다."""
    pairs = []
    for arm in exp.ARMS:
        for policy in exp.ARMS[1:]:
            pairs.append((f"inference_{arm}_{policy}", (arm, "all"), (arm, policy)))
    for policy in exp.ARMS[1:]:
        pairs.append((f"training_{policy}", ("all", policy), (policy, policy)))
    return pairs


def sequence_report(sequence: str) -> None:
    """검증 한 시퀀스의 모든 양성 손실을 실제 검색으로 교차 재계산한다."""
    exp.protocol()
    s, q, paths, _ = exp.base.old.alignment.base.inputs("val", sequence)
    assert (s, q) in exp.VAL
    target = exp.OUT / f"ranking_{s}_{q}.json"
    if target.exists():
        return
    with np.load(paths["cache"]) as z:
        pos = z["poses_gt"][:, :3, 3]
    embeddings, ledgers, hashes = {}, {}, {}
    pairs = comparisons()
    for arm, policy in sorted({part for _, a, b in pairs for part in (a, b)}):
        dest = exp.folder(arm, policy)
        path = dest / f"keys_{s}_{q}.npz"
        with np.load(path) as z:
            embeddings[arm, policy] = z["embedding"]
        hashes[str(path.relative_to(REPO))] = sha256(path)
        for op in exp.OPS:
            path = dest / f"val_{op}_{s}_{q}.npz"
            ledgers[arm, policy, op], _ = exp.base.old.read_ledger(path)
            hashes[str(path.relative_to(REPO))] = sha256(path)
    selection = exp.base.read(exp.base.old.BASE / "s1/confidence_selection.json")["ops"]
    reports, checked = {}, 0
    for op in exp.OPS:
        with np.load(paths["spec"]) as z:
            bank = exp.base.old.alignment.base.bank_for(op, z)
        weights = tuple(float(x) for x in selection[op]["tuple"])
        for label, old, new in pairs:
            a, b = ledgers[old + (op,)], ledgers[new + (op,)]
            e0, e1 = embeddings[old], embeddings[new]
            np.testing.assert_array_equal(a["query_idx"], b["query_idx"])
            positive = a["min_past_gt_dist"] <= 5
            lost = positive & (a["top1_geo_dist"] <= 5) & (b["top1_geo_dist"] > 5)
            gained = positive & (a["top1_geo_dist"] > 5) & (b["top1_geo_dist"] <= 5)
            cases = []
            for i in np.flatnonzero(lost):
                query = int(a["query_idx"][i])
                pool0, pool1 = candidate_pool(e0, bank, query), candidate_pool(e1, bank, query)
                j00, j11 = rank(e0, bank, query, pool0, weights), rank(e1, bank, query, pool1, weights)
                assert j00 == a["j_star"][i] and j11 == b["j_star"][i]
                j10, j01 = rank(e1, bank, query, pool0, weights), rank(e0, bank, query, pool1, weights)
                key_wrong = bool(np.linalg.norm(pos[query] - pos[j10]) > 5)
                pool_wrong = bool(np.linalg.norm(pos[query] - pos[j01]) > 5)
                present = bool((np.linalg.norm(pos[pool1] - pos[query], axis=1) <= 5).any())
                assert present == b["union_hit"][i]
                category = ("both_individually_sufficient" if key_wrong and pool_wrong else
                            "key_change_only_sufficient" if key_wrong else
                            "pool_change_only_sufficient" if pool_wrong else "joint_change_required")
                cases.append({"query": query, "direction": int(a["group"][i]),
                    "new_pool_has_true": present, "old_winner_in_new_pool": bool(j00 in pool1),
                    "same_pool": bool(np.array_equal(pool0, pool1)),
                    "j_old": j00, "j_new": j11, "j_new_key_old_pool": j10,
                    "j_old_key_new_pool": j01, "category": category})
                checked += 2
            reports.setdefault(label, {})[op] = {"old": list(old), "new": list(new),
                "positive_queries": int(positive.sum()), "lost": int(lost.sum()), "gained": int(gained.sum()),
                "ranking_loss_with_true_candidate": int((lost & b["union_hit"]).sum()), "cases": cases}
    write_json(target, {"sequence": sequence, "comparisons": reports, "endpoint_checks": checked,
        "input_sha256": hashes, "script_sha256": sha256(Path(__file__)), "test_used": False})
    print("RANKING", sequence, checked, flush=True)


def report() -> None:
    """반사실 순위 대조를 자료·방향별로 합산하고 범위를 명시한다."""
    pooled, checks, hashes = {}, 0, {}
    for s, q in exp.VAL:
        path = exp.OUT / f"ranking_{s}_{q}.json"
        meta = exp.base.read(path)
        assert meta["script_sha256"] == sha256(Path(__file__))
        for name, digest in meta["input_sha256"].items():
            assert sha256(REPO / name) == digest
        hashes[str(path.relative_to(REPO))] = sha256(path)
        checks += meta["endpoint_checks"]
        family = "kitti" if s.startswith("kitti") else s
        for label, ops in meta["comparisons"].items():
            for op, values in ops.items():
                dest = pooled.setdefault(label, {}).setdefault(op, {}).setdefault(family, {})
                for key in ("positive_queries", "lost", "gained", "ranking_loss_with_true_candidate"):
                    dest[key] = dest.get(key, 0) + values[key]
                for name, mask in (("all_positive", lambda c: True), ("same", lambda c: c["direction"] == 0),
                                   ("opposite", lambda c: c["direction"] == 1)):
                    row = dest.setdefault(name, {"lost": 0, "union_missing": 0, "key_change_only_sufficient": 0,
                        "pool_change_only_sufficient": 0, "both_individually_sufficient": 0, "joint_change_required": 0,
                        "old_winner_present": 0, "same_pool": 0})
                    for case in values["cases"]:
                        if not mask(case):
                            continue
                        row["lost"] += 1
                        if not case["new_pool_has_true"]:
                            row["union_missing"] += 1
                        else:
                            row[case["category"]] += 1
                            row["old_winner_present"] += int(case["old_winner_in_new_pool"])
                            row["same_pool"] += int(case["same_pool"])
                    assert row["lost"] == sum(row[k] for k in ("union_missing", "key_change_only_sufficient",
                        "pool_change_only_sufficient", "both_individually_sufficient", "joint_change_required"))
    write_json(exp.OUT / "ranking_diagnosis.json", {"comparisons": pooled, "endpoint_checks": checks,
        "source_sha256": hashes, "script_sha256": sha256(Path(__file__)), "test_used": False,
        "scope": "최종1등정답→오답인 모든양성질의의 기존/새키×기존/새후보목록2x2. 정답은 사후진단만사용. 정답후보가남은손실을 분류하며 AP원인의백분율이나 학습기제의인과증명은 아니다. 후보목록을 바꿀때 네점수의목록내정규화도 함께 재계산한다."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence")
    args = parser.parse_args()
    sequence_report(args.sequence) if args.sequence else report()
