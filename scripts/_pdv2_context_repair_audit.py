#!/usr/bin/env python3
"""새 개입의 선택·메시지와 두 통과의 관측 범위를 독립 CPU로 검사한다."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_context_repair as fixed
import _pdv2_context_repair_train as learned
from _pdv2_no_context_ablation import sha256, write_json


def audit_fixed() -> None:
    """전체 저장 간선의 개입을 독립 정렬로 확인한다."""
    p = fixed.protocol(True)
    counts = []
    for s, q in fixed.VAL:
        with np.load(fixed.base.OUT / "main" / f"edges_{s}_{q}.npz") as z:
            query, neighbor = z["query"], z["neighbor"]
        with np.load(fixed.parent.OUT / "edges" / f"{s}_{q}.npz") as z:
            scores = {op: z["C_" + op] for op in fixed.OPS}
            old_ids = {op: z["selected__" + op + "__C"] for op in fixed.OPS}
        with np.load(fixed.OUT / f"selected_{s}_{q}.npz") as z:
            selected = {k: z[k] for k in z.files}
        bounds = np.r_[0, np.flatnonzero(query[1:] != query[:-1]) + 1, len(query)]
        for op in fixed.OPS:
            for arm in fixed.GRAPH_ARMS:
                cap = 1 if arm == "C_top1" else 3 if arm == "C_top3" else 10
                expected = np.concatenate([np.arange(a, b)[np.argsort(-scores[op][a:b], kind="stable")[:cap]]
                                           for a, b in zip(bounds[:-1], bounds[1:])])
                if arm == "C_cut":
                    expected = expected[scores[op][expected] >= p["thresholds"][op]["threshold"]]
                ids = selected[op + "__" + arm]
                np.testing.assert_array_equal(ids, expected)
                assert np.all(neighbor[ids] <= query[ids] - 30)
                if arm == "C_message":
                    np.testing.assert_array_equal(ids, old_ids[op])
                counts.append({"sequence": s + "/" + q, "op": op, "arm": arm, "edges": len(ids)})
        print("POLICY", s, q, flush=True)
    write_json(fixed.OUT / "policy_validation.json", {"status": "passed", "policies": counts,
               "audit_script_sha256": sha256(Path(__file__))})


def prefix(stage: str) -> None:
    """실제 첫 통과부터 다시 계산해 q+15 이후 입력의 영향을 검사한다."""
    from _lag_branch_fusion import TruncatedCache
    from _pdv2_attention_neighbors_verify_reference import decision
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    experiment = fixed if stage == "fixed" else learned
    experiment.protocol(True)
    arms = ("main",) if stage == "fixed" else learned.ARMS
    selection = fixed.base.read(fixed.base.old.BASE / "s1/confidence_selection.json")["ops"]
    records = []
    for arm in arms:
        bundle = fixed.base.load_model("main", "cpu")
        if stage == "training":
            path = learned.OUT / arm / "final.pth"
            assert sha256(path) == fixed.base.read(learned.OUT / arm / "complete.json")["checkpoint_sha256"]
            bundle[1].load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
            bundle[1].eval()
        for sequence in ("kitti/09", "helipr/Roundabout01"):
            sensor = sequence.split("/")[0]
            cache, layout, spec, _, _, _ = fixed.parent.rel.load_data("val", sequence)
            # 실제 입력 1,200노드와 316/916노드를 비교한다.
            full_cache = TruncatedCache(cache, 1200)
            full_spec = {k: v[:1200] for k, v in spec.items()}
            _, graph, c = fixed.base.capture(full_cache, sensor, layout[:1200], bundle, "cpu")
            if stage == "fixed":
                extra = fixed.parent.rel.extra_features(full_spec, c["query"], c["neighbor"])
                prob = fixed.parent.probabilities(c["features"], extra, "cpu")
                variants = [(op, a) for op in fixed.OPS for a in fixed.GRAPH_ARMS]
            else:
                clf = fixed.base.load_classifier("xyz", "cpu")
                _, probs, _ = fixed.base.apply_classifier(bundle[1], graph, c, clf)
                prob = {op: probs for op in fixed.OPS}
                variants = [(op, a) for op in fixed.OPS for a in ("all", "temporal")]
            full, full_ids = {}, {}
            for op, condition in variants:
                if stage == "fixed":
                    probabilities = prob["C_" + op]
                    ids = fixed.choose(c["query"], probabilities, condition, fixed.protocol()["thresholds"][op]["threshold"])
                else:
                    from _causal_branch_fusion import select_edges_per_node
                    probabilities = prob[op]
                    ids = select_edges_per_node(c["query"], probabilities, 10) if condition == "all" else np.empty(0, np.int64)
                full_ids[op, condition] = ids
                full[op, condition] = fixed.infer(bundle[1], fixed.base.replace_edges(graph, c, probabilities, ids), condition == "C_message")
            for query in (300, 900):
                end = query + 16
                short_cache = TruncatedCache(cache, end)
                short_spec = {k: v[:end] for k, v in spec.items()}
                _, cg, cc = fixed.base.capture(short_cache, sensor, layout[:end], bundle, "cpu")
                if stage == "fixed":
                    extra = fixed.parent.rel.extra_features(short_spec, cc["query"], cc["neighbor"])
                    pp = fixed.parent.probabilities(cc["features"], extra, "cpu")
                else:
                    _, probs, _ = fixed.base.apply_classifier(bundle[1], cg, cc, clf)
                    pp = {op: probs for op in fixed.OPS}
                for op, condition in variants:
                    if stage == "fixed":
                        probabilities = pp["C_" + op]
                        ids = fixed.choose(cc["query"], probabilities, condition, fixed.protocol()["thresholds"][op]["threshold"])
                    else:
                        probabilities = pp[op]
                        ids = select_edges_per_node(cc["query"], probabilities, 10) if condition == "all" else np.empty(0, np.int64)
                    fids = full_ids[op, condition]
                    # 두 번째 통과 2층에서 선택 간선은 q+5까지만 필요하다.
                    # q+10의 노드는 원래 입력만 필요하며 그 노드의 선택은 읽지 않는다.
                    fids = fids[c["query"][fids] <= query + 5]
                    relevant = ids[cc["query"][ids] <= query + 5]
                    np.testing.assert_array_equal(np.stack([c["query"][fids], c["neighbor"][fids]]),
                                                  np.stack([cc["query"][relevant], cc["neighbor"][relevant]]))
                    short = fixed.infer(bundle[1], fixed.base.replace_edges(cg, cc, probabilities, ids), condition == "C_message")
                    large = full[op, condition]
                    error = float(np.max(np.abs(large[:query + 1] - short[:query + 1])))
                    np.testing.assert_allclose(large[:query + 1], short[:query + 1], atol=2e-6, rtol=0)
                    bank0 = fixed.base.old.alignment.base.bank_for(op, full_spec)
                    bank1 = fixed.base.old.alignment.base.bank_for(op, short_spec)
                    d0, pool0 = decision(large, bank0, query, selection[op])
                    d1, pool1 = decision(short, bank1, query, selection[op])
                    np.testing.assert_array_equal(pool0, pool1)
                    assert d0["j_star"] == d1["j_star"] and d0["accepted"] == d1["accepted"]
                    records.append({"model": arm, "sequence": sequence, "query": query,
                        "op": op, "condition": condition, "max_key_error": error,
                        "same_pool_candidate_acceptance": True})
                print("PREFIX", stage, arm, sequence, query, flush=True)
    write_json(experiment.OUT / "prefix_validation.json", {"status": "passed_cpu_reference",
        "scope": "실제 입력 첫1200노드 대 q+16노드. 첫통과 후보·간선·둘째통과·최종판정을 CPU에서 재계산. 전체GPU스트리밍 안정성 검사는 아님.",
        "cases": records, "audit_script_sha256": sha256(Path(__file__))})


def finish(stage: str) -> None:
    """원장·정책·관측 범위 검사가 모두 끝난 상태를 기록한다."""
    experiment = fixed if stage == "fixed" else learned
    if stage == "fixed":
        audit_fixed()
    prefix(stage)
    ledger = fixed.base.read(experiment.OUT / "ledger_validation.json")
    assert ledger["status"] == "passed"
    write_json(experiment.OUT / "completion_validation.json", {
        "status": "passed_with_inherited_gpu_reproducibility_limit", "ledger": ledger,
        "prefix": fixed.base.read(experiment.OUT / "prefix_validation.json"),
        "test_used": False, "audit_script_sha256": sha256(Path(__file__)),
        "summary_sha256": sha256(experiment.OUT / "summary.json")})
    old = fixed.base.read(experiment.OUT / "status.json")
    write_json(experiment.OUT / "status.json", {**old, "status": "complete", "remaining": None,
        "verification_status": "passed_with_inherited_gpu_reproducibility_limit"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("fixed", "training"))
    finish(p.parse_args().stage)
