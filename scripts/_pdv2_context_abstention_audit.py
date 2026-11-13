#!/usr/bin/env python3
"""연결 거절 정책·학습 하한·최종 원장·입력 범위를 독립 검사한다."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_context_abstention as exp
from _pdv2_no_context_ablation import sha256, write_json


def independent_ids(query: np.ndarray, prob: np.ndarray, alignment: np.ndarray,
                    policy: str, threshold: dict) -> np.ndarray:
    """질의별 사전식 정렬로 저장된 연결 정책을 별도로 재계산한다."""
    selected = []
    bounds = np.r_[0, np.flatnonzero(query[1:] != query[:-1]) + 1, len(query)]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        ids = np.arange(lo, hi)
        ids = ids[np.lexsort((ids, -prob[ids]))][:10]
        if policy == "temporal":
            continue
        if policy == "classifier_gate" and prob[ids].max() < threshold["classifier"]:
            continue
        if policy.startswith("alignment_gate") and alignment[ids].max() < threshold["alignment"]:
            continue
        selected.extend(ids[:1] if policy in ("top1", "alignment_gate_top1") else ids)
    return np.asarray(selected, np.int64)


def calibration() -> list:
    """보정에 쓴 정답·질의 분모와 각 시퀀스 95% 보존을 확인한다."""
    records = []
    for model in exp.MODELS:
        meta = exp.base.read(exp.OUT / (model + "_thresholds.json"))
        for source in meta["sources"]:
            path = REPO / source["path"]
            assert sha256(path) == source["sha256"]
        for sequence, source in zip(exp.CFG["calibration_sequences"], meta["sources"]):
            with np.load(REPO / source["path"]) as z:
                a = {k: z[k] for k in z.files}
            with np.load(exp.fixed.parent.rel.train_path(sequence)) as z:
                gt = z["poses_gt"][:, :3, 3]
            correct = np.linalg.norm(gt[a["query"]] - gt[a["neighbor"]], axis=1) <= 5
            np.testing.assert_array_equal(correct, a["correct"])
            assert (a["neighbor"] <= a["query"] - 30).all()
            for op in exp.OPS:
                align = a["extra_" + op][:, exp.CFG["alignment_channel"][op]]
                ids = independent_ids(a["query"], a["probability"], align, "all", {})
                eligible = np.zeros(len(gt), bool)
                eligible[a["query"][ids[correct[ids]]]] = True
                eligible[:30], eligible[len(gt) - 15:] = False, False
                np.testing.assert_array_equal(eligible, a["eligible"])
                for name, values in (("classifier", a["probability"]), ("alignment", align)):
                    maximum = np.full(len(gt), -np.inf)
                    np.maximum.at(maximum, a["query"][ids], values[ids])
                    threshold = meta["thresholds"][op][name]
                    assert threshold == min(x["threshold"] for x in meta["per_sequence"][op][name])
                    rate = float((maximum[eligible] >= threshold).mean())
                    assert rate >= exp.CFG["retention_per_calibration_sequence"]
                    records.append({"model": model, "sequence": sequence, "op": op,
                                    "signal": name, "queries": int(eligible.sum()), "retention": rate})
    return records


def report() -> None:
    """전 조건의 최종 지표와 연결 유지·정답 변화를 집계한다."""
    exp.protocol(True)
    calibration_checks = calibration()
    scores, quality, transitions, drift = {}, {}, {}, {}
    selection = exp.base.read(exp.base.old.BASE / "s1/confidence_selection.json")["ops"]
    nledgers = nrows = policy_checks = 0
    for model in exp.MODELS:
        thresholds_path = exp.OUT / (model + "_thresholds.json")
        thresholds = exp.base.read(thresholds_path)["thresholds"]
        for s, q in exp.VAL:
            meta = exp.base.read(exp.OUT / f"{model}_keys_{s}_{q}.json")
            assert sha256(thresholds_path) == meta["threshold_sha256"]
            path = exp.OUT / f"{model}_edges_{s}_{q}.npz"
            assert sha256(path) == meta["edge_sha256"]
            with np.load(path) as z:
                a = {k: z[k] for k in z.files}
            for op in exp.OPS:
                align = a["extra_" + op][:, exp.CFG["alignment_channel"][op]]
                for policy in exp.POLICIES:
                    ids = a["selected__" + op + "__" + policy]
                    expected = independent_ids(a["query"], a["probability"], align, policy, thresholds[op])
                    np.testing.assert_array_equal(ids, expected)
                    assert np.all(a["neighbor"][ids] <= a["query"][ids] - 30)
                    key_path = exp.folder(model, policy) / f"keys_{op}_{s}_{q}.npz"
                    assert sha256(key_path) == meta["ops"][op][policy]["sha256"]
                    policy_checks += 1
                    groups = quality.setdefault(model + "__" + policy, {}).setdefault(op, {})
                    seq_groups = {}
                    for i, name in enumerate(("same", "opposite", "other", "negative", "ambiguous")):
                        mask = a["valid"] & (a["group"] == i)
                        keep = ids[mask[a["query"][ids]]]
                        d = a["distance"][keep]
                        values = {"queries": int(mask.sum()), "connected": len(np.unique(a["query"][keep])),
                                  "true_connected": len(np.unique(a["query"][keep][d <= 5])),
                                  "edges": len(keep), "true_edges": int((d <= 5).sum()),
                                  "far_edges": int((d > 25).sum())}
                        seq_groups[name] = values
                    groups[s + "/" + q] = seq_groups
        for policy in exp.POLICIES:
            label = model + "__" + policy
            root = exp.folder(model, policy)
            rows = {s + "/" + q: exp.base.read(root / f"val_{s}_{q}.json") for s, q in exp.VAL}
            scores[label] = {op: exp.base.old.aggregate_rows(rows, "val", op) for op in exp.OPS}
            transitions[label], drift[label] = {}, {}
            for op in exp.OPS:
                transitions[label][op] = {}
                for s, q in exp.VAL:
                    meta = rows[s + "/" + q]["ops"][op]
                    path = REPO / meta["ledger"]
                    assert sha256(path) == meta["sha256"]
                    b, _ = exp.base.old.read_ledger(path)
                    a, _ = exp.base.old.read_ledger(exp.folder(model, "temporal") / path.name)
                    for key in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(a[key], b[key])
                    sel = selection[op]
                    assert exp.base.old.sequence_metrics(b, sel["decision"], sel["tau_star"]) == meta["metrics"]
                    assert (b["j_star"] <= b["query_idx"] - 30).all()
                    _, _, paths, _ = exp.base.old.alignment.base.inputs("val", s + "/" + q)
                    with np.load(paths["cache"]) as z:
                        pos = z["poses_gt"][:, :3, 3]
                    np.testing.assert_allclose(b["top1_geo_dist"], np.linalg.norm(pos[b["query_idx"]] - pos[b["j_star"]], axis=1), atol=1e-5, rtol=1e-7)
                    stats = {}
                    for name, mask in (("positive", a["min_past_gt_dist"] <= 5),
                                       ("same", a["group"] == 0), ("opposite", a["group"] == 1)):
                        x, y = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
                        stats[name] = {"queries": int(mask.sum()), "correct": int((mask & y).sum()),
                                       "recovered": int((mask & ~x & y).sum()),
                                       "regressed": int((mask & x & ~y).sum()),
                                       "regressed_union_missing": int((mask & x & ~y & ~b["union_hit"]).sum())}
                    transitions[label][op][s + "/" + q] = stats
                    nledgers += 1
                    nrows += len(b["query_idx"])
                if policy in ("all", "temporal"):
                    previous = exp.base.read(exp.learned.OUT / "summary.json")["scores"][model + "_" + policy][op]["macro"]
                    drift[label][op] = {k: scores[label][op]["macro"][k] - previous[k] for k in exp.base.PRIMARY}
    effects = {}
    for policy in exp.POLICIES:
        effects[policy] = {op: {m: scores["opposite__" + policy][op]["macro"][m] - scores["control__" + policy][op]["macro"][m]
                                for m in exp.base.PRIMARY} for op in exp.OPS}
    write_json(exp.OUT / "summary.json", {"scores": scores, "edge_quality": quality,
        "transitions_vs_own_temporal": transitions, "opposite_minus_control": effects,
        "previous_baseline_metric_change": drift, "test_used": False})
    write_json(exp.OUT / "ledger_validation.json", {"status": "passed", "ledgers": nledgers,
        "rows": nrows, "policy_checks": policy_checks, "calibration": calibration_checks,
        "summary_sha256": sha256(exp.OUT / "summary.json")})
    with (exp.OUT / "comparison.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["model", "policy", "op", *exp.base.PRIMARY, "AP_minus_own_temporal_pp", "AP_kitti", "AP_helipr", "AP_mulran"])
        for model in exp.MODELS:
            for policy in exp.POLICIES:
                for op, value in scores[model + "__" + policy].items():
                    macro = value["macro"]
                    delta = macro["AP"] - scores[model + "__temporal"][op]["macro"]["AP"]
                    w.writerow([model, policy, op, *[macro[k] for k in exp.base.PRIMARY], 100 * delta,
                                *[value["by_group"][g]["AP"] for g in ("kitti", "helipr", "mulran")]])


def prefix() -> None:
    """실제 두 통과와 연결 거절을 CPU에서 전체·절단 입력으로 비교한다."""
    from _lag_branch_fusion import TruncatedCache
    from _pdv2_attention_neighbors_verify_reference import decision
    torch.set_num_threads(1)
    selection = exp.base.read(exp.base.old.BASE / "s1/confidence_selection.json")["ops"]
    cases = []
    for model in exp.MODELS:
        bundle = exp.load_model(model, "cpu")
        thresholds = exp.base.read(exp.OUT / (model + "_thresholds.json"))["thresholds"]
        clf = exp.base.load_classifier("xyz", "cpu")
        for sequence in ("kitti/09", "helipr/Roundabout01"):
            cache, layout, spec, _, _, _ = exp.fixed.parent.rel.load_data("val", sequence)
            stored = {}
            for end in (1200, 316, 916):
                cc = TruncatedCache(cache, end)
                ss = {k: v[:end] for k, v in spec.items()}
                _, graph, c = exp.base.capture(cc, sequence.split("/")[0], layout[:end], bundle, "cpu")
                _, prob, _ = exp.base.apply_classifier(bundle[1], graph, c, clf)
                extra = exp.fixed.parent.rel.extra_features(ss, c["query"], c["neighbor"])
                for op in exp.OPS:
                    align = extra["extra_" + op][:, exp.CFG["alignment_channel"][op]]
                    bank = exp.base.old.alignment.base.bank_for(op, ss)
                    for policy in exp.POLICIES:
                        ids = exp.choose(c["query"], prob, align, end, policy, thresholds[op])
                        emb = exp.fixed.infer(bundle[1], exp.base.replace_edges(graph, c, prob, ids))
                        if end == 1200:
                            stored[op, policy] = (c, ids, emb, bank)
                            continue
                        query = end - 16
                        full, full_ids, full_emb, full_bank = stored[op, policy]
                        ii = full_ids[full["query"][full_ids] <= query + 5]
                        jj = ids[c["query"][ids] <= query + 5]
                        np.testing.assert_array_equal(np.stack([full["query"][ii], full["neighbor"][ii]]),
                                                      np.stack([c["query"][jj], c["neighbor"][jj]]))
                        error = float(np.max(np.abs(full_emb[:query + 1] - emb[:query + 1])))
                        np.testing.assert_allclose(full_emb[:query + 1], emb[:query + 1], atol=2e-6, rtol=0)
                        d0, p0 = decision(full_emb, full_bank, query, selection[op])
                        d1, p1 = decision(emb, bank, query, selection[op])
                        np.testing.assert_array_equal(p0, p1)
                        assert d0["j_star"] == d1["j_star"] and d0["accepted"] == d1["accepted"]
                        cases.append({"model": model, "sequence": sequence, "op": op, "policy": policy,
                                      "query": query, "max_key_error": error, "same_final_decision": True})
                print("PREFIX", model, sequence, end, flush=True)
    write_json(exp.OUT / "prefix_validation.json", {"status": "passed_cpu_reference", "cases": cases,
        "scope": "실제첫1200노드와q+16 비교. 전체GPU재생성 안정성이나 실시간처리 검사는 아님."})


if __name__ == "__main__":
    report()
    prefix()
    write_json(exp.OUT / "completion_validation.json", {
        "status": "passed_with_inherited_gpu_reproducibility_limit",
        "ledger": exp.base.read(exp.OUT / "ledger_validation.json"),
        "prefix": exp.base.read(exp.OUT / "prefix_validation.json"),
        "summary_sha256": sha256(exp.OUT / "summary.json"),
        "audit_sha256": sha256(Path(__file__)), "test_used": False})
