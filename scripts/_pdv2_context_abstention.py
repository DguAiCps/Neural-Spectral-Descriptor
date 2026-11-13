#!/usr/bin/env python3
"""역방향 학습 표현과 정답 독립 연결 거절의 최종 기여를 비교한다."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_context_repair_train as learned
from _causal_branch_fusion import select_edges_per_node
from _pdv2_no_context_ablation import evaluate_keys, sha256, write_json
from retrieval.edge_reliability import retention_threshold

fixed, base = learned.frozen, learned.base
OUT = REPO / "results/pdv2_context_abstention_20261017"
CONFIG = REPO / "configs/pdv2_context_abstention.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
MODELS, POLICIES = tuple(CFG["models"]), tuple(CFG["policies"])
VAL, OPS = base.VAL, base.OPS


def protocol(data: bool = False) -> dict:
    """사전에 동결한 실행 코드와 입력을 확인한다."""
    p = base.read(OUT / "protocol.json")
    for name, digest in p["runtime_sha256"].items():
        assert sha256(REPO / name) == digest, name
    if data:
        for name, digest in p["input_sha256"].items():
            assert sha256(REPO / name) == digest, name
    return p


def folder(model: str, policy: str) -> Path:
    """모델과 연결 조건별 독립 경로를 반환한다."""
    assert model in MODELS and policy in POLICIES
    return OUT / (model + "__" + policy)


def prepare() -> None:
    """학습 내부 보정과 모든 검증 조건을 결과 생성 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    lp, fp = learned.protocol(True), fixed.protocol(True)
    runtime = {REPO / x for x in lp["runtime_sha256"]}
    runtime.update((CONFIG, Path(__file__), REPO / "scripts/_pdv2_context_abstention_run.py",
                    REPO / "scripts/_pdv2_context_abstention_audit.py",
                    REPO / "tests/retrieval/test_context_abstention.py"))
    inputs = {REPO / x for x in fp["input_sha256"]}
    inputs.update((learned.OUT / "protocol.json", learned.OUT / "summary.json",
                   learned.OUT / "completion_validation.json"))
    for sequence in CFG["calibration_sequences"]:
        inputs.add(fixed.parent.rel.train_path(sequence))
    for model in MODELS:
        inputs.update((learned.OUT / model / "final.pth", learned.OUT / model / "complete.json"))
        for s, q in VAL:
            for policy in ("all", "temporal"):
                p = learned.OUT / (model + "_" + policy)
                inputs.add(p / f"keys_{s}_{q}.npz")
                inputs.add(p / f"val_{s}_{q}.json")
                inputs.update(p / f"val_{op}_{s}_{q}.npz" for op in OPS)
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "config": CFG, "test_used": False,
        "scope": "검증9·시드1 탐색. 앞선 결과에 근거해 일반/역방향 추가학습 모델을 비교 대상으로 고른 후 이번 조건을 고정한다. 시험·주표 교체 없음.",
        "models": "두 고정6에포크 모델. 추가 학습·새 추론 모듈·새 파라미터 없음. A 분류기·시간간선·속성·융합·수락 규칙 고정.",
        "policies": "all=A상위10; temporal=둘째통과유사간선없음; top1=A상위1. classifier_gate/alignment_gate는 A상위10의 최대 분류기/원시정렬 점수로 질의 전체 유사간선을 허용하거나 거절. alignment_gate_top1은 동일 상위10 기반 허용판단 뒤 A상위1만 전달.",
        "alignment": "NSD 열정렬, NSD-H 스펙트럼정렬을 사전에 고정한다. 연결 선택순위와 속성은 A를 유지한다. 현재 저장 단일스캔 스펙트럼을 사용하며 후보별 정렬을 먼저 계산하는 비용은 추가된다.",
        "calibration": "GNN 학습에 포함된 KITTI06·Bridge01의 온라인 입력에서 보정한다. A상위10에 정답<=5m가 있는 유효질의의 최대점수를 시퀀스마다 최소95% 보존하는 하한으로 정하고 두 하한의 최솟값을 공유한다. 각 모델·점수별 하나의 하한이며 검증계열별 조정 없음. GNN 독립 홀드아웃은 아니다.",
        "comparison": "모델별 동일한 새 첫통과 후보를 여섯 정책에 공유한다. all/temporal도 새로 생성·평가해 GPU 재생성 차이가 정책효과에 섞이지 않게 한다. 이전 결과와 차이는 따로 기록한다.",
        "causality": "q+15 관측, j<=q-30,100+100검색 유지. 학습GT는 하한/정답 작성만, 검증GT는 진단/평가만 사용한다. 마지막15노드도 간선 입력으로 생성한다.",
        "runtime_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(runtime)},
        "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(inputs)}})


def load_model(model: str, device: str = "cuda") -> tuple:
    """완료된 추가 학습 모델을 정확한 해시로 불러온다."""
    path = learned.OUT / model / "final.pth"
    assert sha256(path) == base.read(path.parent / "complete.json")["checkpoint_sha256"]
    bundle = base.load_model("main", device)
    bundle[1].load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    bundle[1].eval()
    return bundle


def query_signals(query: np.ndarray, probability: np.ndarray, alignment: np.ndarray,
                  nodes: int) -> tuple:
    """같은 A 상위10 이웃의 최대 점수로 질의별 연결 근거를 구한다."""
    ids = select_edges_per_node(query, probability, 10)
    a, b = np.full(nodes, -np.inf), np.full(nodes, -np.inf)
    np.maximum.at(a, query[ids], probability[ids])
    np.maximum.at(b, query[ids], alignment[ids])
    return ids, a, b


def choose(query: np.ndarray, probability: np.ndarray, alignment: np.ndarray,
           nodes: int, policy: str, thresholds: dict) -> np.ndarray:
    """GT나 다른 질의의 통계를 읽지 않고 연결을 거절한다."""
    assert policy in POLICIES
    if policy == "temporal":
        return np.empty(0, np.int64)
    ids, classifier_score, alignment_score = query_signals(query, probability, alignment, nodes)
    if policy in ("top1", "alignment_gate_top1"):
        ids = select_edges_per_node(query, probability, 1)
    if policy == "classifier_gate":
        ids = ids[classifier_score[query[ids]] >= thresholds["classifier"]]
    elif policy in ("alignment_gate", "alignment_gate_top1"):
        ids = ids[alignment_score[query[ids]] >= thresholds["alignment"]]
    return ids


def capture(role: str, model: str, sequence: str, device: str = "cuda") -> tuple:
    """정답을 분리한 입력으로 첫 통과·후보·A 점수·정렬 점수를 구한다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache, layout, spec, gt, min_past, paths = fixed.parent.rel.load_data(role, sequence)
    assert "poses_gt" not in cache.files
    bundle = load_model(model, device)
    _, graph, c = base.capture(cache, sequence.split("/")[0], layout, bundle, device)
    emb, prob, ids = base.apply_classifier(bundle[1], graph, c, base.load_classifier("xyz", device))
    extra = fixed.parent.rel.extra_features(spec, c["query"], c["neighbor"])
    return cache, gt, min_past, bundle, graph, c, emb, prob, ids, extra


def calibrate(model: str) -> None:
    """학습 내부 두 시퀀스의 정답 연결 질의 보존율로 하한을 정한다."""
    protocol(True)
    target = OUT / (model + "_thresholds.json")
    if target.exists():
        return
    sources, results = [], {op: {} for op in OPS}
    for sequence in CFG["calibration_sequences"]:
        cache, gt, _, _, _, c, _, prob, ids, extra = capture("train", model, sequence)
        nodes = len(gt)
        correct = np.linalg.norm(gt[c["query"], :3, 3] - gt[c["neighbor"], :3, 3], axis=1) <= 5
        eligible = np.zeros(nodes, bool)
        eligible[c["query"][ids[correct[ids]]]] = True
        eligible[:30], eligible[nodes - 15:] = False, False
        assert eligible.any()
        arrays = {"query": c["query"], "neighbor": c["neighbor"], "probability": prob,
                  "correct": correct, "eligible": eligible, **extra}
        path = OUT / (model + "_calibration_" + sequence.replace("/", "_") + ".npz")
        np.savez_compressed(path, **arrays)
        sources.append({"path": str(path.relative_to(REPO)), "sha256": sha256(path)})
        for op in OPS:
            _, a, b = query_signals(c["query"], prob, extra["extra_" + op][:, CFG["alignment_channel"][op]], nodes)
            for name, scores in (("classifier", a), ("alignment", b)):
                value = scores[eligible]
                threshold = retention_threshold(value, CFG["retention_per_calibration_sequence"])
                results[op].setdefault(name, []).append({"sequence": sequence, "queries": len(value),
                    "threshold": threshold, "retained": int((value >= threshold).sum())})
        print("CALIBRATE", model, sequence, int(eligible.sum()), flush=True)
    thresholds = {op: {name: min(row["threshold"] for row in rows) for name, rows in signals.items()}
                  for op, signals in results.items()}
    write_json(target, {"model": model, "thresholds": thresholds, "per_sequence": results,
        "sources": sources, "protocol_sha256": sha256(OUT / "protocol.json"), "test_used": False})


def keys(model: str, sequence: str) -> None:
    """같은 첫 통과에서 연결 정책과 두 기준 조건의 키를 함께 생성한다."""
    protocol()
    s, q = sequence.split("/")
    assert (s, q) in VAL
    metadata = OUT / f"{model}_keys_{s}_{q}.json"
    if metadata.exists():
        return
    threshold_path = OUT / (model + "_thresholds.json")
    thresholds = base.read(threshold_path)["thresholds"]
    cache, gt, min_past, bundle, graph, c, all_emb, prob, all_ids, extra = capture("val", model, sequence)
    n = len(gt)
    ref, _ = base.old.read_ledger(learned.OUT / (model + "_all") / f"val_{OPS[0]}_{s}_{q}.npz")
    group = np.full(n, -1, np.int8)
    group[ref["query_idx"]] = ref["group"]
    valid = (np.arange(n) >= 30) & (np.arange(n) < n - 15)
    saved = {"query": c["query"], "neighbor": c["neighbor"], "probability": prob,
             "features": c["features"], "l2": c["l2"], "group": group, "valid": valid,
             "distance": np.linalg.norm(gt[c["query"], :3, 3] - gt[c["neighbor"], :3, 3], axis=1), **extra}
    records, drift = {}, {}
    temporal = fixed.infer(bundle[1], base.replace_edges(graph, c, prob, np.empty(0, np.int64)))
    for op in OPS:
        align = extra["extra_" + op][:, CFG["alignment_channel"][op]]
        records[op] = {}
        for policy in POLICIES:
            ids = choose(c["query"], prob, align, n, policy, thresholds[op])
            if policy == "all":
                np.testing.assert_array_equal(ids, all_ids)
            emb = all_emb if policy == "all" else temporal if policy == "temporal" else fixed.infer(bundle[1], base.replace_edges(graph, c, prob, ids))
            dest = folder(model, policy)
            dest.mkdir(exist_ok=True)
            path = dest / f"keys_{op}_{s}_{q}.npz"
            np.savez_compressed(path, embedding=emb, scan_ids=cache["scan_ids"])
            records[op][policy] = {"sha256": sha256(path), "edges": len(ids)}
            saved["selected__" + op + "__" + policy] = ids
        for policy, emb in (("all", all_emb), ("temporal", temporal)):
            with np.load(learned.OUT / (model + "_" + policy) / f"keys_{s}_{q}.npz") as old:
                drift[policy] = float(np.max(np.abs(emb - old["embedding"])))
    path = OUT / f"{model}_edges_{s}_{q}.npz"
    np.savez_compressed(path, **saved)
    write_json(metadata, {"model": model, "sequence": sequence, "ops": records,
        "edge_sha256": sha256(path), "threshold_sha256": sha256(threshold_path),
        "previous_key_max_error": drift})
    print("KEYS", model, sequence, drift, flush=True)


def score(model: str, policy: str, sequence: str) -> None:
    """기존 최종 검색·판정 평가기로 모든 고정 조건을 평가한다."""
    protocol()
    s, q, paths, min_past = base.old.alignment.base.inputs("val", sequence)
    dest = folder(model, policy)
    if (dest / f"val_{s}_{q}.json").exists():
        return
    with np.load(paths["cache"]) as z:
        gt, scan_ids = z["poses_gt"][:, :3, 3], z["scan_ids"]
    selection = base.read(base.old.BASE / "s1/confidence_selection.json")["ops"]
    records = {}
    for op in OPS:
        with np.load(dest / f"keys_{op}_{s}_{q}.npz") as z:
            emb = z["embedding"]
            np.testing.assert_array_equal(z["scan_ids"], scan_ids)
        with np.load(paths["spec"]) as z:
            bank = base.old.alignment.base.bank_for(op, z)
        sel = selection[op]
        a = evaluate_keys(emb, bank, gt, min_past, {1: tuple(float(w) for w in sel["tuple"])})[1]
        keep = a["query_idx"] < len(gt) - 15
        a = {k: v[keep] for k, v in a.items()}
        ref, _ = base.old.read_ledger(learned.OUT / (model + "_all") / f"val_{op}_{s}_{q}.npz")
        for key in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(a[key], ref[key])
        a["group"] = ref["group"]
        path = dest / f"val_{op}_{s}_{q}.npz"
        np.savez_compressed(path, **a, meta=json.dumps({"protocol_sha256": sha256(OUT / "protocol.json")}))
        records[op] = {"metrics": base.old.sequence_metrics(a, sel["decision"], sel["tau_star"]),
                       "ledger": str(path.relative_to(REPO)), "sha256": sha256(path)}
    write_json(dest / f"val_{s}_{q}.json", {"model": model, "policy": policy, "sequence": sequence, "ops": records})
    print("SCORE", model, policy, sequence, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "calibrate", "keys", "score"))
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--policy", choices=POLICIES)
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "prepare":
        prepare()
    elif a.action == "calibrate":
        calibrate(a.model)
    elif a.action == "keys":
        keys(a.model, a.sequence)
    else:
        score(a.model, a.policy, a.sequence)
