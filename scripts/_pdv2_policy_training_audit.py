#!/usr/bin/env python3
"""학습 예산·정책·최종 원장·지연 의존성과 교차 효과를 확인한다."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_policy_training as exp
from _pdv2_context_abstention_audit import independent_ids
from _pdv2_no_context_ablation import sha256, write_json


def training() -> dict:
    """같은 초기화·표본·업데이트와 학습 정책을 독립 재검산한다."""
    initial = exp.base.read(exp.OUT / exp.ARMS[0] / "initial.json")
    temporal = None
    checks = []
    for arm in exp.ARMS:
        assert exp.base.read(exp.OUT / arm / "initial.json") == initial
        completed = exp.base.read(exp.OUT / arm / "complete.json")
        assert completed["updates"] == 270 and len(completed["epochs"]) == 6
        assert sha256(exp.OUT / arm / "final.pth") == completed["checkpoint_sha256"]
        for row in completed["epochs"]:
            path = exp.OUT / arm / f"graph_epoch{row['epoch']}.npz"
            assert sha256(path) == row["graph_sha256"]
            assert sha256(exp.learned.OUT / "samples" / f"opposite_{row['epoch']}.npz") == row["sample_sha256"]
            with np.load(path) as a:
                expected = independent_ids(a["query"], a["probability"], np.zeros_like(a["probability"]),
                                           arm, {"classifier": exp.threshold()})
                np.testing.assert_array_equal(a["selected"], expected)
                assert np.all(a["neighbor"] <= a["query"] - 30)
                assert np.isfinite(a["probability"]).all()
                digest = str(a["temporal_sha256"])
                temporal = digest if temporal is None else temporal
                assert temporal == digest == row["temporal_sha256"]
                assert len(expected) == row["selected_edges"]
                checks.append({"arm": arm, "epoch": row["epoch"], "selected_edges": len(expected)})
    return {"initial": initial, "same_updates": 270, "same_temporal_sha256": temporal, "checks": checks}


def report() -> None:
    """최종 판정 지표와 연결 정책의 교차 효과를 모두 보고한다."""
    exp.protocol(True)
    train_checks = training()
    scores, transitions, quality = {}, {}, {}
    selection = exp.base.read(exp.base.old.BASE / "s1/confidence_selection.json")["ops"]
    nledgers = nrows = policy_checks = 0
    for arm in exp.ARMS:
        for policy in exp.POLICIES:
            label = arm + "__" + policy
            dest = exp.folder(arm, policy)
            rows = {s + "/" + q: exp.base.read(dest / f"val_{s}_{q}.json") for s, q in exp.VAL}
            scores[label] = {op: exp.base.old.aggregate_rows(rows, "val", op) for op in exp.OPS}
            transitions[label], quality[label] = {}, {}
            for op in exp.OPS:
                transitions[label][op], quality[label][op] = {}, {}
                for s, q in exp.VAL:
                    record = rows[s + "/" + q]["ops"][op]
                    path = REPO / record["ledger"]
                    assert sha256(path) == record["sha256"]
                    b, _ = exp.base.old.read_ledger(path)
                    a, _ = exp.base.old.read_ledger(exp.folder(arm, "all") / path.name)
                    for k in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(a[k], b[k])
                    sel = selection[op]
                    assert exp.base.old.sequence_metrics(b, sel["decision"], sel["tau_star"]) == record["metrics"]
                    assert (b["j_star"] <= b["query_idx"] - 30).all()
                    _, _, paths, _ = exp.base.old.alignment.base.inputs("val", s + "/" + q)
                    with np.load(paths["cache"]) as z:
                        pos = z["poses_gt"][:, :3, 3]
                    np.testing.assert_allclose(b["top1_geo_dist"], np.linalg.norm(pos[b["query_idx"]] - pos[b["j_star"]], axis=1), atol=1e-5, rtol=1e-7)
                    meta = exp.base.read(exp.OUT / f"{arm}_keys_{s}_{q}.json")
                    assert sha256(dest / f"keys_{s}_{q}.npz") == meta["policies"][policy]["sha256"]
                    ep = exp.OUT / f"{arm}_edges_{s}_{q}.npz"
                    assert sha256(ep) == meta["edges_sha256"]
                    with np.load(ep) as z:
                        edges = {k: z[k] for k in z.files}
                    ids = edges["selected__" + policy]
                    expected = independent_ids(edges["query"], edges["probability"], np.zeros_like(edges["probability"]),
                                               policy, {"classifier": exp.threshold()})
                    np.testing.assert_array_equal(ids, expected)
                    np.testing.assert_allclose(edges["distance"], np.linalg.norm(pos[edges["query"]] - pos[edges["neighbor"]], axis=1), atol=1e-5)
                    policy_checks += 1
                    groups, qgroups = {}, {}
                    connected, has_true, before_true = (np.zeros(len(pos), bool) for _ in range(3))
                    connected[edges["query"][ids]] = True
                    has_true[edges["query"][ids[edges["distance"][ids] <= 5]]] = True
                    original = edges["selected__all"]
                    before_true[edges["query"][original[edges["distance"][original] <= 5]]] = True
                    qi, md = b["query_idx"], b["min_past_gt_dist"]
                    for name, mask in (("positive", md <= 5), ("same", (md <= 5) & (a["group"] == 0)),
                                       ("opposite", (md <= 5) & (a["group"] == 1)),
                                       ("negative", md > 25), ("ambiguous", (md > 5) & (md <= 25))):
                        x, y = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
                        lost = mask & x & ~y
                        groups[name] = {"queries": int(mask.sum()), "correct": int((mask & y).sum()),
                            "recovered": int((mask & ~x & y).sum()), "regressed": int(lost.sum()),
                            "regressed_union_missing": int((lost & ~b["union_hit"]).sum()),
                            "regressed_original_true_edge": int((lost & before_true[qi]).sum()),
                            "regressed_disconnected": int((lost & ~connected[qi]).sum())}
                        qgroups[name] = {"queries": int(mask.sum()), "connected": int((mask & connected[qi]).sum()),
                                         "true_connected": int((mask & has_true[qi]).sum())}
                    assert qgroups["positive"]["queries"] + qgroups["negative"]["queries"] + qgroups["ambiguous"]["queries"] == len(qi)
                    assert qgroups["negative"]["true_connected"] == 0
                    assert qgroups["ambiguous"]["true_connected"] == 0
                    transitions[label][op][s + "/" + q] = groups
                    quality[label][op][s + "/" + q] = qgroups
                    nledgers += 1
                    nrows += len(qi)
    effects = {}
    for policy in exp.ARMS[1:]:
        effects[policy] = {}
        for op in exp.OPS:
            aa = scores["all__all"][op]["macro"]
            ap = scores["all__" + policy][op]["macro"]
            pa = scores[policy + "__all"][op]["macro"]
            pp = scores[policy + "__" + policy][op]["macro"]
            tt = scores[policy + "__temporal"][op]["macro"]
            effects[policy][op] = {m: {"matched_minus_control_same_inference": pp[m] - ap[m],
                "interaction": (pp[m] - pa[m]) - (ap[m] - aa[m]),
                "matched_minus_own_temporal": pp[m] - tt[m]} for m in exp.base.PRIMARY}
    write_json(exp.OUT / "summary.json", {"scores": scores, "policy_effects": effects,
        "transitions_vs_own_all": transitions, "edge_quality": quality, "test_used": False})
    write_json(exp.OUT / "ledger_validation.json", {"status": "passed", "training": train_checks,
        "ledgers": nledgers, "rows": nrows, "policy_checks": policy_checks,
        "summary_sha256": sha256(exp.OUT / "summary.json")})
    with (exp.OUT / "comparison.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["train_policy", "eval_policy", "op", *exp.base.PRIMARY, "AP_minus_own_temporal_pp",
                    "AP_kitti", "AP_helipr", "AP_mulran"])
        for arm in exp.ARMS:
            for policy in exp.POLICIES:
                for op, row in scores[arm + "__" + policy].items():
                    m = row["macro"]
                    w.writerow([arm, policy, op, *[m[k] for k in exp.base.PRIMARY],
                        100 * (m["AP"] - scores[arm + "__temporal"][op]["macro"]["AP"]),
                        *[row["by_group"][g]["AP"] for g in ("kitti", "helipr", "mulran")]])


def prefix() -> None:
    """세 학습 모델의 실제 두 통과와 입력 절단을 CPU에서 비교한다."""
    from _lag_branch_fusion import TruncatedCache
    from _pdv2_attention_neighbors_verify_reference import decision
    torch.set_num_threads(1)
    selection = exp.base.read(exp.base.old.BASE / "s1/confidence_selection.json")["ops"]
    cases = []
    for arm in exp.ARMS:
        bundle = exp.load_model(arm, "cpu")
        clf = exp.base.load_classifier("xyz", "cpu")
        for sequence in ("kitti/09", "helipr/Roundabout01"):
            cache, layout, spec, _, _, _ = exp.fixed.parent.rel.load_data("val", sequence)
            stored = {}
            for end in (1200, 316, 916):
                cc = TruncatedCache(cache, end)
                ss = {k: v[:end] for k, v in spec.items()}
                _, graph, c = exp.base.capture(cc, sequence.split("/")[0], layout[:end], bundle, "cpu")
                _, prob, _ = exp.base.apply_classifier(bundle[1], graph, c, clf)
                for policy in exp.POLICIES:
                    ids = exp.choose(c["query"], prob, end, policy, exp.threshold())
                    emb = exp.fixed.infer(bundle[1], exp.base.replace_edges(graph, c, prob, ids))
                    if end == 1200:
                        stored[policy] = (c, ids, emb, ss)
                        continue
                    query = end - 16
                    full, fi, fe, fs = stored[policy]
                    ii, jj = fi[full["query"][fi] <= query + 5], ids[c["query"][ids] <= query + 5]
                    np.testing.assert_array_equal(np.stack([full["query"][ii], full["neighbor"][ii]]),
                                                  np.stack([c["query"][jj], c["neighbor"][jj]]))
                    error = float(np.max(np.abs(fe[:query + 1] - emb[:query + 1])))
                    np.testing.assert_allclose(fe[:query + 1], emb[:query + 1], atol=2e-6, rtol=0)
                    for op in exp.OPS:
                        d0, p0 = decision(fe, exp.base.old.alignment.base.bank_for(op, fs), query, selection[op])
                        d1, p1 = decision(emb, exp.base.old.alignment.base.bank_for(op, ss), query, selection[op])
                        np.testing.assert_array_equal(p0, p1)
                        assert d0["j_star"] == d1["j_star"] and d0["accepted"] == d1["accepted"]
                        cases.append({"arm": arm, "sequence": sequence, "policy": policy, "op": op,
                                      "query": query, "max_key_error": error, "same_final_decision": True})
                print("PREFIX", arm, sequence, end, flush=True)
    write_json(exp.OUT / "prefix_validation.json", {"status": "passed_cpu_reference", "cases": cases,
        "scope": "실제 첫1200노드와 q+16 비교. 기존 GPU 재생성 수치 변동의 해결이나 실시간 속도 검사는 아니다."})


if __name__ == "__main__":
    report()
    prefix()
    write_json(exp.OUT / "completion_validation.json", {"status": "passed_with_inherited_gpu_reproducibility_limit",
        "ledger": exp.base.read(exp.OUT / "ledger_validation.json"),
        "prefix": exp.base.read(exp.OUT / "prefix_validation.json"),
        "summary_sha256": sha256(exp.OUT / "summary.json"), "audit_sha256": sha256(Path(__file__)), "test_used": False})
