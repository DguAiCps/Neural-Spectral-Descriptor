#!/usr/bin/env python3
"""개선한 간선 판별을 같은 GNN의 최종 장소 인식 성능으로 검증한다."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_edge_reliability as rel
import _pdv2_similarity_causes as causes
from _pdv2_no_context_ablation import sha256, write_json, evaluate_keys

base = rel.base
VAL, OPS, PRIMARY = base.VAL, base.OPS, base.PRIMARY
OUT = REPO / "results/pdv2_edge_reliability_final_20261016"
CONFIG = REPO / "configs/pdv2_edge_reliability_final.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
ARMS, REFS = tuple(CFG["arms"]), tuple(CFG["reference_arms"])


def folder(arm: str = "") -> Path:
    """새 결과의 저장 폴더를 만든다."""
    path = OUT / arm
    path.mkdir(parents=True, exist_ok=True)
    return path


def reference_root(arm: str) -> Path:
    """이미 검증한 대조군 산출물을 읽기 전용으로 재사용한다."""
    assert arm in REFS
    return base.OUT / "main" / ("xyz" if arm == "A" else arm)


def protocol(data: bool = False) -> dict:
    """고정한 실행 파일과 자료의 변경을 막는다."""
    p = base.read(OUT / "protocol.json")
    for name, h in p["runtime_sha256"].items():
        assert sha256(REPO / name) == h, name
    if data:
        for name, h in p["input_sha256"].items():
            assert sha256(REPO / name) == h, name
    return p


def prepare() -> None:
    """새 최종 성능을 보기 전에 간선 선택과 속성 대조를 고정한다."""
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    parents = [rel.protocol(True), causes.protocol(True)]
    runtime = {REPO / p for parent in parents for p in parent["runtime_sha256"]}
    inputs = {REPO / p for parent in parents for p in parent["input_sha256"]}
    runtime.update(REPO / p for p in (
        "configs/pdv2_edge_reliability_final.yaml", "scripts/_pdv2_edge_reliability_final.py",
        "scripts/_pdv2_edge_reliability_final_run.py", "scripts/_pdv2_edge_reliability_final_verify.py",
        "scripts/_pdv2_attention_neighbors_verify_reference.py", "tests/gnn/test_edge_reliability_final.py"))
    inputs.update(rel.OUT / name for name in ("protocol.json", "summary.json", "completion_validation.json"))
    for arm in rel.ARMS[1:]:
        inputs.update(rel.OUT / "classifiers" / (arm + ext) for ext in (".pt", ".json"))
    for s, q in VAL:
        for kind in ("features", "predictions"):
            inputs.update(rel.OUT / kind / (f"val_{s}_{q}" + ext) for ext in (".npz", ".json"))
    write_json(folder() / "protocol.json", {"created_unix": time.time(), "config": CFG, "test_used": False,
        "scope": "주평가 GNN 시드1·검증9의 최종 AP/F1/R@100P/R@1. 간선 판별 개선의 최종 효과를 측정한다. 새 학습·선택·시험·주표 교체 없음.",
        "fixed": "같은 GNN·첫통과·후보30·원래d·시간간선·재계량·q+15·j<=q-30·100+100 검색·정렬·융합·수락 점수/기준값. 점수 하한 없이 노드당 최대10개를 연결한다.",
        "references": "temporal은 유사간선 없는 같은 GNN. A는 수정3D 분류기. original은 원고의 기존 분류기. 저장키/원장은 해시를 고정하며 A/temporal의 키를 새 경로에서 재현한다.",
        "B": "온라인 입력으로 학습한 B의 상위10·분류기 점수 속성을 사용한다. 추가 정보 없이 입력 조건만 맞춘 최종 대조다.",
        "C": "해당 운영점의 정렬2점수를 추가한 C의 상위10·분류기 점수 속성을 사용한다.",
        "C_selection": "C와 같은 상위10을 고르되 해당 간선의 A 점수 속성을 유지한다. C_selection->C는 연결을 고정한 점수 속성 변경이다.",
        "alignment_column": "각 운영점의 원시 스펙트럼 상관 또는 열정렬 점수로 상위10을 직접 고르고 해당 간선의 A 점수 속성을 유지한다. 원시 점수를 확률 속성으로 대입하지 않는다. C_selection과 같은 속성 규칙이다.",
        "operating_points": "NSD/H의 C·정렬선택은 각각 자기 운영점 점수를 쓰므로 같은 GNN 가중치라도 최종 문맥 키가 서로 달라진다. 기본 모델처럼 동일 문맥 키를 공유하는 조건은 아니다.",
        "tail": "앞 실험은 평가 질의 q<N-15만 저장했다. 이번 둘째통과는 끝15개 노드의 간선도 필요하므로 전체 후보의 점수를 계산한다. 유효질의의 점수는 저장값과 대조한 뒤 그대로 보존한다. 마지막 유효질의까지 잘린입력 검사에 포함한다.",
        "selection": "사전 정한 두 직접 점수와 모든 분류기 조건을 보고한다. 추가 threshold/융합/체크포인트 탐색 없음. 단일시드·반복 노출된 검증 진단이며 통계적 확증/새 독립시험이 아니다.",
        "runtime_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(runtime)},
        "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(inputs)}})
    print("PREPARED", len(runtime), len(inputs), flush=True)


def probabilities(features: np.ndarray, extra: dict, device: str = "cuda") -> dict:
    """모든 후보에 동결한 A/B/C 점수를 계산한다."""
    import torch
    result = {}
    for arm in rel.ARMS:
        clf, blob = rel.classifier(arm, device)
        x = rel.model_input({"features": features, **extra}, arm)
        values = []
        with torch.no_grad():
            for start in range(0, len(x), 65536):
                inputs = torch.from_numpy((x[start:start + 65536] - blob["mu"]) / blob["sd"]).to(device)
                values.append(torch.sigmoid(clf(inputs)).cpu().numpy())
        result[arm] = np.concatenate(values)
        assert np.isfinite(result[arm]).all()
    return result


def choose(c: dict, scores: dict, op: str, arm: str) -> tuple:
    """선택용 점수와 GNN에 주는 점수 속성을 분리한다. 정답은 읽지 않는다."""
    from _causal_branch_fusion import select_edges_per_node
    score, attribute = scores["A"], scores["A"]
    if arm == "temporal":
        return attribute, np.empty(0, np.int64)
    if arm == "B":
        score = attribute = scores["B"]
    elif arm in ("C", "C_selection"):
        score = scores["C_" + op]
        if arm == "C":
            attribute = score
    elif arm in ("alignment", "column"):
        score = scores["extra_" + op][:, 0 if arm == "alignment" else 1]
    else:
        assert arm == "A", arm
    ids = select_edges_per_node(c["query"], score, CFG["selected_count"])
    assert np.all(c["neighbor"][ids] <= c["query"][ids] - 30)
    return attribute, ids


def keys(sequence: str) -> None:
    """전체 노드에 유사 간선을 적용하고 운영점별 최종 키를 저장한다."""
    import torch
    from _lag_branch_fusion import lag_pass1_graph
    protocol()
    assert tuple(sequence.split("/")) in VAL
    s, q = sequence.split("/")
    done = OUT / f"keys_{s}_{q}.json"
    if done.exists():
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache, _, spec, gt, _, _ = rel.load_data("val", sequence)
    with np.load(base.OUT / "main" / f"edges_{s}_{q}.npz") as z:
        c = {k: z[k] for k in z.files}
    keep = c["valid"][c["query"]]
    extra = {}
    with np.load(rel.feature_path("val", sequence)) as z:
        for k in ("query", "neighbor", "features"):
            np.testing.assert_array_equal(c[k][keep], z[k])
        for op in OPS:
            extra["extra_" + op] = np.empty((len(keep), 2), np.float32)
            extra["extra_" + op][keep] = z["extra_" + op]
    tail = rel.extra_features(spec, c["query"][~keep], c["neighbor"][~keep])
    for name in extra:
        extra[name][~keep] = tail[name]
    scores = probabilities(c["features"], extra)
    errors = {}
    with np.load(rel.OUT / "predictions" / f"val_{s}_{q}.npz") as z:
        for arm in rel.ARMS:
            errors[arm] = float(np.max(np.abs(scores[arm][keep] - z[arm])))
            np.testing.assert_allclose(scores[arm][keep], z[arm], atol=2e-6, rtol=0)
            scores[arm][keep] = z[arm]
    np.testing.assert_allclose(scores["A"], c["prob_xyz"], atol=2e-6, rtol=0)
    scores["A"] = c["prob_xyz"]
    scores.update(extra)
    cfg, net, *_ = base.load_model("main")
    cfg = copy.deepcopy(cfg)
    cfg["keyframe"].setdefault("graph", {})["similarity_threshold"] = 2.
    raw = cache["descriptors"].astype(np.float32)
    graph = lag_pass1_graph(cache, raw, cfg, "cuda", s)
    assert not bool((graph.edge_type == 1).any())
    record = {"sequence": sequence, "nodes": len(raw), "tail_edges": int((~keep).sum()),
              "score_reproduction_error": errors, "reference_key_error": {}, "ops": {}}
    saved = {**scores, "query": c["query"], "neighbor": c["neighbor"]}
    for op in OPS:
        record["ops"][op] = {}
        for arm in ("temporal", "A", *ARMS):
            p, ids = choose(c, scores, op, arm)
            g = base.replace_edges(graph, c, p, ids)
            emb, attn = causes.infer_attention(net, g, c["group"], c["valid"], gt[:, :3, 3])
            assert emb.shape == (len(raw), 416) and np.isfinite(emb).all()
            np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1, atol=3e-7, rtol=0)
            if arm in REFS:
                with np.load(reference_root(arm) / f"keys_{s}_{q}.npz") as z:
                    error = float(np.max(np.abs(emb - z["embedding"])))
                    np.testing.assert_allclose(emb, z["embedding"], atol=1e-6, rtol=0)
                record["reference_key_error"][arm] = error
            else:
                dest = folder(arm) / f"keys_{op}_{s}_{q}.npz"
                np.savez_compressed(dest, embedding=emb, scan_ids=cache["scan_ids"])
            saved[f"selected__{op}__{arm}"] = ids
            record["ops"][op][arm] = {"edges": len(ids), "attention": attn,
                "key_sha256": sha256(dest) if arm not in REFS else None,
                "quality": {g: base.old.edge_counts(c["query"][ids], c["distance_3d"][ids], c["distance_xy"][ids], p[ids], c["valid"] & mask)
                            for g, mask in [("all", np.ones(len(raw), bool)), *[(g, c["group"] == i) for i, g in enumerate(causes.GROUPS)]]}}
    ep = folder("edges") / f"{s}_{q}.npz"
    np.savez_compressed(ep, **saved)
    record["edge_sha256"] = sha256(ep)
    write_json(done, record)
    print("KEYS", sequence, flush=True)


def score(arm: str, sequence: str) -> None:
    """운영점에 맞는 문맥 키를 최종 검색·정렬·수락까지 평가한다."""
    protocol()
    assert arm in ARMS
    s, q, paths, min_past = base.old.alignment.base.inputs("val", sequence)
    target = folder(arm) / f"val_{s}_{q}.json"
    if target.exists():
        return
    with np.load(paths["cache"]) as z:
        gt, scan_ids = z["poses_gt"][:, :3, 3], z["scan_ids"]
    selection = base.read(base.old.BASE / "s1/confidence_selection.json")["ops"]
    result = {}
    for op in OPS:
        kp = folder(arm) / f"keys_{op}_{s}_{q}.npz"
        assert sha256(kp) == base.read(OUT / f"keys_{s}_{q}.json")["ops"][op][arm]["key_sha256"]
        with np.load(kp) as z:
            embedding = z["embedding"]
            np.testing.assert_array_equal(scan_ids, z["scan_ids"])
        with np.load(paths["spec"]) as z:
            np.testing.assert_array_equal(scan_ids, z["scan_ids"])
            bank = base.old.alignment.base.bank_for(op, z)
        sel = selection[op]
        a = evaluate_keys(embedding, bank, gt, min_past, {1: tuple(float(w) for w in sel["tuple"])})[1]
        keep = a["query_idx"] < len(gt) - 15
        a = {k: v[keep] for k, v in a.items()}
        ref, _ = base.old.read_ledger(reference_root("A") / f"val_{op}_{s}_{q}.npz")
        for k in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(a[k], ref[k])
        a["group"] = ref["group"]
        dest = folder(arm) / f"val_{op}_{s}_{q}.npz"
        np.savez_compressed(dest, **a, meta=json.dumps({"protocol_sha256": sha256(OUT / "protocol.json")}))
        result[op] = {"metrics": base.old.sequence_metrics(a, sel["decision"], sel["tau_star"]),
                      "ledger": str(dest.relative_to(REPO)), "sha256": sha256(dest)}
    write_json(target, {"arm": arm, "sequence": sequence, "ops": result})
    print("SCORE", arm, sequence, flush=True)


def report() -> None:
    """최종 지표와 방향별 정답 전환·간선 품질·주의 비중을 모두 보고한다."""
    protocol(True)
    scores, quality, attention, transitions = {}, {}, {}, {}
    for arm in (*REFS, *ARMS):
        root = reference_root(arm) if arm in REFS else OUT / arm
        rows = {s + "/" + q: base.read(root / f"val_{s}_{q}.json") for s, q in VAL}
        scores[arm] = {op: base.old.aggregate_rows(rows, "val", op) for op in OPS}
        transitions[arm] = {}
        for op in OPS:
            transitions[arm][op] = {}
            for refarm in ("temporal", "A"):
                groups = {}
                for s, q in VAL:
                    a, _ = base.old.read_ledger(reference_root(refarm) / f"val_{op}_{s}_{q}.npz")
                    b, _ = base.old.read_ledger(root / f"val_{op}_{s}_{q}.npz")
                    for name, mask in [("positive", a["min_past_gt_dist"] <= 5), *[(g, a["group"] == i) for i, g in enumerate(causes.GROUPS[:3])]]:
                        x, y = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
                        counts = {"queries": int(mask.sum()), "before_correct": int((mask & x).sum()),
                                  "after_correct": int((mask & y).sum()), "recovered": int((mask & ~x & y).sum()),
                                  "regressed": int((mask & x & ~y).sum())}
                        r = groups.setdefault(name, {k: 0 for k in counts})
                        for k, v in counts.items():
                            r[k] += v
                transitions[arm][op][refarm] = groups
        if arm == "original":
            continue
        quality[arm], attention[arm] = {}, {}
        for op in OPS:
            records = [base.read(OUT / f"keys_{s}_{q}.json")["ops"][op][arm] for s, q in VAL]
            quality[arm][op] = {g: base.old.add_counts([r["quality"][g] for r in records]) for g in ("all", *causes.GROUPS)}
            attention[arm][op] = [{g: {k: sum(r["attention"][layer][g][k] for r in records)
                                      for k in records[0]["attention"][layer][g]}
                                  for g in ("all", *causes.GROUPS)} for layer in range(2)]
    write_json(OUT / "summary.json", {"scores": scores, "edge_quality": quality, "attention": attention,
        "transitions": transitions, "scope": "같은 GNN 시드1·검증9의 최종 판정. 시퀀스→3계열 평균. 새 선택·시험 없음.",
        "test_used": False, "protocol_sha256": sha256(OUT / "protocol.json")})
    with (OUT / "comparison.csv").open("w") as handle:
        w = csv.writer(handle)
        w.writerow(["arm", "operating_point", *PRIMARY, "AP_difference_vs_temporal_pp", "AP_difference_vs_A_pp"])
        for arm, ops in scores.items():
            for op, row in ops.items():
                m = row["macro"]
                w.writerow([arm, op, *[m[k] for k in PRIMARY],
                            100 * (m["AP"] - scores["temporal"][op]["macro"]["AP"]),
                            100 * (m["AP"] - scores["A"][op]["macro"]["AP"])])
    print("REPORT", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "keys", "score", "report"))
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--sequence")
    args = p.parse_args()
    if args.action in ("prepare", "report"):
        globals()[args.action]()
    elif args.action == "score":
        score(args.arm, args.sequence)
    else:
        keys(args.sequence)
