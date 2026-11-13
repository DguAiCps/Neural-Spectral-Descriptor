#!/usr/bin/env python3
"""유사 간선 개수·정보 전달·최종 융합을 현재 고정 모델에서 비교한다."""
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
import _pdv2_edge_reliability_final as parent
from _pdv2_no_context_ablation import evaluate_keys, sha256, write_json

OUT = REPO / "results/pdv2_context_repair_20261016"
CONFIG = REPO / "configs/pdv2_context_repair.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
GRAPH_ARMS = tuple(CFG["graph_arms"])
FUSION_ARMS = tuple(CFG["fusion_arms"])
ARMS = GRAPH_ARMS + FUSION_ARMS
REFS = tuple(CFG["references"])
VAL, OPS = parent.VAL, parent.OPS
base = parent.base


def root(arm: str) -> Path:
    """기존 결과는 읽기 전용 경로로 반환한다."""
    if arm == "C":
        return parent.OUT / arm
    if arm in ("temporal", "A"):
        return parent.reference_root(arm)
    return OUT / arm


def protocol(data: bool = False) -> dict:
    """비교 규칙과 동결된 입력을 검증한다."""
    p = base.read(OUT / "protocol.json")
    for name, h in p["runtime_sha256"].items():
        assert sha256(REPO / name) == h, name
    if data:
        for name, h in p["input_sha256"].items():
            assert sha256(REPO / name) == h, name
    return p


def prepare() -> None:
    """새 결과 전에 실행 코드·비교 조건·원천 결과를 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    p = parent.protocol(True)
    runtime = {REPO / f for f in p["runtime_sha256"]}
    runtime.update((Path(__file__), CONFIG,
                    REPO / "scripts/_pdv2_context_repair_run.py",
                    REPO / "tests/gnn/test_context_repair.py"))
    inputs = {REPO / f for f in p["input_sha256"]}
    inputs.add(parent.OUT / "summary.json")
    for s, q in VAL:
        inputs.update((parent.OUT / "edges" / f"{s}_{q}.npz",
                       base.OUT / "main" / f"edges_{s}_{q}.npz"))
        for arm in REFS:
            inputs.add(root(arm) / f"val_{s}_{q}.json")
            for op in OPS:
                inputs.add(root(arm) / f"val_{op}_{s}_{q}.npz")
        for op in OPS:
            inputs.add(root("C") / f"keys_{op}_{s}_{q}.npz")
    thresholds = base.read(REPO / CFG["threshold_source"])["thresholds"]
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "config": CFG, "test_used": False,
        "scope": "검증9·현재 대칭 GNN시드1. 후보30·모델·시간간선·q+15·j<=q-30·100+100·수락규칙 고정.",
        "selection": "모든 조건을 보고하며 추가 모델·가중치·하한 선택과 시험 적용은 하지 않는다.",
        "topk": "C의 순위·점수 속성을 유지하며 최대1개 또는3개만 연결한다.",
        "cut": "학습 내부 확인2의 정답후보 질의95%를 유지한 기존 C 하한으로 각 간선을 제거한다. 질의 최대점수로 고른 기준의 간선별 적용이며 배포 보정이나 검증 선택이 아니다.",
        "message": "C와 같은10개·같은속성·같은attention식. 두 층 유사간선의 실제 메시지에 C 점수를 곱한다. 시간간선은1. 학습 파라미터 추가 없음. attention 분모는 그대로라 확률0이 시간간선 전용과 같지는 않다.",
        "fusion": "C의 키·후보합집합·보조/스펙트럼/열 점수를 유지하고 최종 융합의 키 가중치만0.5배 또는0으로 한다. 후보검색의 키는 유지한다.",
        "causality": "저장 첫통과 후보를 공유한다. q+15에서 그래프 절단·C 메시지 적용을 검사하되 기존 GPU 후보 재생성 수치 예외의 해결로 해석하지 않는다.",
        "thresholds": {op: thresholds["C_" + op] for op in OPS},
        "runtime_sha256": {str(f.relative_to(REPO)): sha256(f) for f in sorted(runtime)},
        "input_sha256": {str(f.relative_to(REPO)): sha256(f) for f in sorted(inputs)},
    })


def choose(query: np.ndarray, scores: np.ndarray, arm: str,
           threshold: float) -> np.ndarray:
    """정답·시퀀스 전체 통계 없이 질의별 연결을 선택한다."""
    from _causal_branch_fusion import select_edges_per_node
    assert arm in GRAPH_ARMS
    count = 1 if arm == "C_top1" else 3 if arm == "C_top3" else 10
    ids = select_edges_per_node(query, scores, count)
    return ids[scores[ids] >= threshold] if arm == "C_cut" else ids


def message_gates(edge_type, edge_attr):
    """시간 간선은 보존하고 유사 간선만 기존 확률로 축소한다."""
    import torch
    assert torch.isfinite(edge_attr).all()
    return torch.where(edge_type[:, None] == 1,
                       edge_attr[:, 4:5].clamp(0, 1),
                       torch.ones_like(edge_attr[:, 4:5]))


def infer(net, graph, use_message: bool = False) -> np.ndarray:
    """두 층의 실제 값 전달에만 고정 게이트를 적용한다."""
    import torch
    from _xs_eval_lib import normalize
    originals = []
    if use_message:
        gates = message_gates(graph.edge_type, graph.edge_attr)
        for layer in net.gnn.convs:
            original = layer.forward
            originals.append((layer, original))

            def forward(*args, _original=original, **kwargs):
                assert kwargs.get("edge_value_gate") is None
                assert args[1].shape[1] == len(gates)
                kwargs["edge_value_gate"] = gates
                return _original(*args, **kwargs)

            layer.forward = forward
    try:
        with torch.no_grad():
            result = normalize(net(graph).float().cpu().numpy())
    finally:
        for layer, original in originals:
            layer.forward = original
    assert result.shape[1] == 416 and np.isfinite(result).all()
    return result


def keys(sequence: str) -> None:
    """공통 저장 후보에서 네 그래프 개입의 키를 생성한다."""
    import torch
    from _lag_branch_fusion import lag_pass1_graph
    p = protocol()
    s, q = sequence.split("/")
    assert (s, q) in VAL
    target = OUT / f"keys_{s}_{q}.json"
    if target.exists():
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache, _, _, _, _, _ = parent.rel.load_data("val", sequence)
    with np.load(base.OUT / "main" / f"edges_{s}_{q}.npz") as z:
        c = {k: z[k] for k in z.files}
    with np.load(parent.OUT / "edges" / f"{s}_{q}.npz") as z:
        scores = {op: z["C_" + op] for op in OPS}
    cfg, net, *_ = base.load_model("main")
    cfg = copy.deepcopy(cfg)
    cfg["keyframe"]["graph"]["similarity_threshold"] = 2.
    graph = lag_pass1_graph(cache, cache["descriptors"].astype(np.float32), cfg, "cuda", s)
    assert not bool((graph.edge_type == 1).any())
    metadata, selected = {}, {}
    for op in OPS:
        metadata[op] = {}
        for arm in GRAPH_ARMS:
            ids = choose(c["query"], scores[op], arm, p["thresholds"][op]["threshold"])
            g = base.replace_edges(graph, c, scores[op], ids)
            emb = infer(net, g, arm == "C_message")
            folder = root(arm)
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / f"keys_{op}_{s}_{q}.npz"
            np.savez_compressed(dest, embedding=emb, scan_ids=cache["scan_ids"])
            selected[op + "__" + arm] = ids
            counts = np.bincount(c["query"][ids], minlength=len(emb))
            metadata[op][arm] = {"key_sha256": sha256(dest), "edges": len(ids),
                "empty_valid_queries": int((counts[c["valid"]] == 0).sum()),
                "mean_edges_valid": float(counts[c["valid"]].mean())}
        # C 통과를 새 실행 경로에서 재현한다. 후보는 재생성하지 않는다.
        ids = choose(c["query"], scores[op], "C_message", 0.)
        baseline = infer(net, base.replace_edges(graph, c, scores[op], ids))
        with np.load(root("C") / f"keys_{op}_{s}_{q}.npz") as z:
            error = float(np.max(np.abs(baseline - z["embedding"])))
            np.testing.assert_allclose(baseline, z["embedding"], atol=1e-6, rtol=0)
        metadata[op]["C_reproduction_max_error"] = error
    np.savez_compressed(OUT / f"selected_{s}_{q}.npz", **selected)
    write_json(target, {"sequence": sequence, "ops": metadata})
    print("KEYS", sequence, flush=True)


def score(arm: str, sequence: str) -> None:
    """원래 최종 평가기로 고정한 설정을 적용한다."""
    protocol()
    assert arm in (*GRAPH_ARMS, "fusion")
    arms = FUSION_ARMS if arm == "fusion" else (arm,)
    s, q, paths, min_past = base.old.alignment.base.inputs("val", sequence)
    if all((root(a) / f"val_{s}_{q}.json").exists() for a in arms):
        return
    with np.load(paths["cache"]) as z:
        gt, scan_ids = z["poses_gt"][:, :3, 3], z["scan_ids"]
    selection = base.read(base.old.BASE / "s1/confidence_selection.json")["ops"]
    records = {a: {} for a in arms}
    for op in OPS:
        kp = root("C" if arm == "fusion" else arm) / f"keys_{op}_{s}_{q}.npz"
        with np.load(kp) as z:
            emb = z["embedding"]
            np.testing.assert_array_equal(z["scan_ids"], scan_ids)
        with np.load(paths["spec"]) as z:
            bank = base.old.alignment.base.bank_for(op, z)
        sel = selection[op]
        weights = {}
        for a in arms:
            w = [float(x) for x in sel["tuple"]]
            if a in FUSION_ARMS:
                w[0] *= .5 if a == "C_half_key" else 0.
            weights[a] = tuple(w)
        values = evaluate_keys(emb, bank, gt, min_past, weights)
        ref, _ = base.old.read_ledger(root("A") / f"val_{op}_{s}_{q}.npz")
        for a, ledger in values.items():
            keep = ledger["query_idx"] < len(gt) - 15
            ledger = {k: v[keep] for k, v in ledger.items()}
            for k in ("query_idx", "min_past_gt_dist"):
                np.testing.assert_array_equal(ledger[k], ref[k])
            ledger["group"] = ref["group"]
            folder = root(a)
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / f"val_{op}_{s}_{q}.npz"
            np.savez_compressed(dest, **ledger, meta=json.dumps({"protocol_sha256": sha256(OUT / "protocol.json")}))
            records[a][op] = {"metrics": base.old.sequence_metrics(ledger, sel["decision"], sel["tau_star"]),
                              "ledger": str(dest.relative_to(REPO)), "sha256": sha256(dest)}
    for a in arms:
        write_json(root(a) / f"val_{s}_{q}.json", {"arm": a, "sequence": sequence, "ops": records[a]})
    print("SCORE", arm, sequence, flush=True)


def report() -> None:
    """전체 조건의 최종 지표와 방향별 손실을 재검산한다."""
    protocol(True)
    selection = base.read(base.old.BASE / "s1/confidence_selection.json")["ops"]
    scores, directions = {}, {}
    count = rows = 0
    for arm in (*REFS, *ARMS):
        records = {s + "/" + q: base.read(root(arm) / f"val_{s}_{q}.json") for s, q in VAL}
        scores[arm] = {op: base.old.aggregate_rows(records, "val", op) for op in OPS}
        directions[arm] = {}
        for op in OPS:
            stats = {}
            for s, q in VAL:
                meta = records[s + "/" + q]["ops"][op]
                path = REPO / meta["ledger"]
                assert sha256(path) == meta["sha256"]
                b, _ = base.old.read_ledger(path)
                a, _ = base.old.read_ledger(root("temporal") / f"val_{op}_{s}_{q}.npz")
                for k in ("query_idx", "min_past_gt_dist", "group"):
                    np.testing.assert_array_equal(a[k], b[k])
                assert np.all(b["j_star"] <= b["query_idx"] - 30)
                sel = selection[op]
                assert base.old.sequence_metrics(b, sel["decision"], sel["tau_star"]) == meta["metrics"]
                _, _, paths, _ = base.old.alignment.base.inputs("val", s + "/" + q)
                with np.load(paths["cache"]) as z:
                    pos = z["poses_gt"][:, :3, 3]
                np.testing.assert_allclose(b["top1_geo_dist"], np.linalg.norm(pos[b["query_idx"]] - pos[b["j_star"]], axis=1), atol=1e-5, rtol=1e-7)
                for name, mask in [("positive", a["min_past_gt_dist"] <= 5),
                                   ("same", a["group"] == 0), ("opposite", a["group"] == 1)]:
                    x, y = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
                    target = stats.setdefault(name, {k: 0 for k in ("queries", "correct", "recovered", "regressed")})
                    for k, v in {"queries": mask.sum(), "correct": (mask & y).sum(),
                                 "recovered": (mask & ~x & y).sum(), "regressed": (mask & x & ~y).sum()}.items():
                        target[k] += int(v)
                count += 1
                rows += len(b["query_idx"])
            directions[arm][op] = stats
    write_json(OUT / "summary.json", {"scores": scores, "directions_vs_temporal": directions, "test_used": False})
    with (OUT / "comparison.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["arm", "op", *base.PRIMARY, "AP_vs_temporal_pp", "AP_vs_C_pp"])
        for arm, ops in scores.items():
            for op, value in ops.items():
                m = value["macro"]
                w.writerow([arm, op, *[m[k] for k in base.PRIMARY],
                            100 * (m["AP"] - scores["temporal"][op]["macro"]["AP"]),
                            100 * (m["AP"] - scores["C"][op]["macro"]["AP"])])
    write_json(OUT / "ledger_validation.json", {"status": "passed", "ledgers": count,
               "rows": rows, "test_used": False, "summary_sha256": sha256(OUT / "summary.json")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "keys", "score", "report"))
    p.add_argument("--sequence")
    p.add_argument("--arm", choices=(*GRAPH_ARMS, "fusion"))
    a = p.parse_args()
    if a.action in ("prepare", "report"):
        globals()[a.action]()
    elif a.action == "keys":
        keys(a.sequence)
    else:
        score(a.arm, a.sequence)
