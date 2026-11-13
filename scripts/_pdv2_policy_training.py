#!/usr/bin/env python3
"""같은 역방향 모델에서 학습·추론 연결 정책 일치의 효과를 비교한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_context_abstention as previous
from _pdv2_no_context_ablation import evaluate_keys, sha256, write_json
from gnn.pair_sampling import same_place_mask, masked_infonce

learned, base, fixed = previous.learned, previous.base, previous.fixed
OUT = REPO / "results/pdv2_policy_training_20261017"
CONFIG = REPO / "configs/pdv2_policy_training.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
ARMS, POLICIES = tuple(CFG["train_policies"]), tuple(CFG["eval_policies"])
VAL, OPS = base.VAL, base.OPS


def protocol(data: bool = False) -> dict:
    """동결 코드와 입력을 검사한다."""
    p = base.read(OUT / "protocol.json")
    for name, digest in p["runtime_sha256"].items():
        assert sha256(REPO / name) == digest, name
    if data:
        for name, digest in p["input_sha256"].items():
            assert sha256(REPO / name) == digest, name
    return p


def threshold() -> float:
    """초기 역방향 모델의 학습 내부 하한을 고정해 반환한다."""
    t = base.read(REPO / CFG["threshold_source"])["thresholds"]
    assert t[OPS[0]]["classifier"] == t[OPS[1]]["classifier"]
    return float(t[OPS[0]]["classifier"])


def folder(arm: str, policy: str) -> Path:
    """학습·추론 정책별 결과 경로다."""
    assert arm in ARMS and policy in POLICIES
    return OUT / (arm + "__" + policy)


def choose(query: np.ndarray, probability: np.ndarray, nodes: int,
           policy: str, lower: float) -> np.ndarray:
    """학습·추론 공통의 정답 독립 연결 정책을 적용한다."""
    assert policy in POLICIES
    return previous.choose(query, probability, np.zeros_like(probability), nodes,
                           policy, {"classifier": lower})


def restrict_graph(graph, policy: str, lower: float) -> dict:
    """A 상위10 학습 그래프에서 시간 간선을 보존하고 정책을 적용한다."""
    temporal = graph.edge_type == 0
    sim = torch.nonzero(~temporal, as_tuple=False).flatten()
    query = graph.edge_index[1, sim].cpu().numpy()
    neighbor = graph.edge_index[0, sim].cpu().numpy()
    probability = graph.edge_attr[sim, 4].cpu().numpy()
    ids = choose(query, probability, graph.num_nodes, policy, lower)
    assert np.all(neighbor <= query - 30)
    keep = torch.cat([torch.nonzero(temporal, as_tuple=False).flatten(),
                      sim[torch.as_tensor(ids, device=sim.device)]])
    temporal_digest = hashlib.sha256(
        graph.edge_index[:, temporal].cpu().numpy().tobytes()
        + graph.edge_attr[temporal].cpu().numpy().tobytes()).hexdigest()
    graph.edge_index = graph.edge_index[:, keep]
    graph.edge_attr = graph.edge_attr[keep]
    graph.edge_type = graph.edge_type[keep]
    return {"query": query, "neighbor": neighbor, "probability": probability,
            "selected": ids, "temporal_sha256": temporal_digest}


def prepare() -> None:
    """이전 역방향 표본·하한을 재사용하고 결과 전에 조건을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    p, lp = previous.protocol(True), learned.protocol(True)
    runtime = {REPO / name for name in p["runtime_sha256"]}
    runtime.update((CONFIG, Path(__file__), REPO / "scripts/_pdv2_policy_training_run.py",
                    REPO / "scripts/_pdv2_policy_training_audit.py",
                    REPO / "tests/gnn/test_policy_training.py"))
    inputs = {REPO / name for name in p["input_sha256"]}
    inputs.update(REPO / name for name in lp["input_sha256"])
    inputs.update((REPO / CFG["initial_checkpoint"], REPO / CFG["threshold_source"],
                   previous.OUT / "protocol.json", previous.OUT / "final_summary.json"))
    for policy in POLICIES:
        for s, q in VAL:
            for op in OPS:
                inputs.add(previous.folder("opposite", policy) / f"val_{op}_{s}_{q}.npz")
    assert CFG["encoder_preset"] == "no_interdiff" and CFG["use_gated_context"]
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "config": CFG, "test_used": False,
        "scope": "검증9·시드1 탐색. 같은 역방향 고정모델에서 세 연결정책으로6에포크 추가학습하고 네 추론정책을 교차평가한다. 새 장소·모듈·파라미터·시험 사용 없음.",
        "budget": "모든조건 같은초기화·초기옵티마이저·역방향25%표본·배치순서·270업데이트. 이전6에포크 표본 재사용이므로 반복학습 및 과적합 가능성을 별도로 남긴다.",
        "policy": "둘째통과의 A상위10/all, A상위1/top1, A상위10 최대분류기점수에 의한 질의별 거절. 같은 choose 함수를 학습·추론에 사용하며 시간간선·A속성은 보존한다. NSD/H에 동일 모델·정책을 사용한다.",
        "graph": "학습은 매에포크 시작 현재모델로 첫통과·후보30·A상위10을 갱신하고 정책을 적용한다. 에포크안 그래프는 고정이므로 매업데이트 추론그래프와 완전히 같은 것은 아니다.",
        "threshold": threshold(),
        "threshold_scope": "초기역방향모델의 KITTI06·Bridge01 각각95%보존 하한의 최솟값. 학습중 및 검증에서 재보정하지 않는다. 분포이동으로 학습후95%보존은 보장하지 않는다.",
        "evaluation": "최종6에포크 고정. all/temporal/top1/classifier_gate 전체보고. q+15·j<=q-30·100+100·융합·수락 고정. 다른정렬채널로 NSD/H별 학습모델을 만드는 조건은 포함하지 않는다.",
        "analysis": "동일추론정책의 학습효과와 (정책추론-전체추론)의 학습정책별 차이의차이를 함께보고. 정답후보존재/최종순위손실을 방향·자료별 분리한다. 초기모델 결과는 별도참고이며 새동일예산대조가 주요비교다.",
        "runtime_sha256": {str(x.relative_to(REPO)): sha256(x) for x in sorted(runtime)},
        "input_sha256": {str(x.relative_to(REPO)): sha256(x) for x in sorted(inputs)}})


def load_model(arm: str, device: str = "cuda", initial: bool = False) -> tuple:
    """공통 초기 모델 또는 완료된 정책별 모델을 읽는다."""
    bundle = base.load_model("main", device)
    path = REPO / CFG["initial_checkpoint"] if initial else OUT / arm / "final.pth"
    if not initial:
        assert sha256(path) == base.read(path.parent / "complete.json")["checkpoint_sha256"]
    blob = torch.load(path, map_location="cpu", weights_only=False)
    bundle[1].load_state_dict(blob["model_state_dict"], strict=True)
    bundle[1].eval()
    return bundle


def train(arm: str) -> None:
    """같은 역방향 표본과 업데이트 수로 정책만 다르게 학습한다."""
    protocol(True)
    assert arm in ARMS
    dest = OUT / arm
    dest.mkdir(exist_ok=True)
    if (dest / "complete.json").exists():
        return
    torch.set_num_threads(2)
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    state = torch.load(learned.OUT / "prepared.pt", map_location="cpu", weights_only=False)
    graph = state["graph"].to("cuda")
    _, model, *_ = load_model(arm, initial=True)
    model.gnn.key_scale.requires_grad_(False)
    classifier, blob = base.load_classifier("xyz", "cuda")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    positions = torch.as_tensor(state["positions"], device="cuda")
    frames = torch.as_tensor(state["frames"], device="cuda")
    bsize, accum = CFG["batch_size"], CFG["accumulation_steps"]
    history, updates, tick = [], 0, time.monotonic()
    write_json(dest / "initial.json", {"checkpoint_sha256": sha256(REPO / CFG["initial_checkpoint"]),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad), "threshold": threshold()})
    for epoch in range(CFG["epochs"]):
        started = time.monotonic()
        refreshed = learned.refresh(model, graph, state["records"], classifier, blob)
        audit = restrict_graph(graph, arm, threshold())
        path = dest / f"graph_epoch{epoch + 1}.npz"
        np.savez_compressed(path, **audit)
        sample = learned.OUT / "samples" / f"opposite_{epoch + 1}.npz"
        with np.load(sample) as z:
            trips = z["triplets"]
        trips = trips[np.random.default_rng(8000 + epoch).permutation(len(trips))]
        n_batches = (len(trips) + bsize - 1) // bsize
        model.train()
        losses = []
        for start in range(0, n_batches, accum):
            stop = min(start + accum, n_batches)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb = model(graph)
            loss = torch.zeros((), device="cuda")
            for batch in range(start, stop):
                a, p, n = trips[batch * bsize:(batch + 1) * bsize].T
                mask = same_place_mask(positions[a], positions[p], frames[a], frames[p])
                loss += masked_infonce(emb[a].float(), emb[p].float(), emb[n].float(),
                                       mask, CFG["temperature"]) / (stop - start)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "updates": updates,
            "seconds": time.monotonic() - started, "refresh": refreshed,
            "selected_edges": len(audit["selected"]), "connected": len(np.unique(audit["query"][audit["selected"]])),
            "graph_sha256": sha256(path), "sample_sha256": sha256(sample),
            "temporal_sha256": audit["temporal_sha256"]})
        write_json(dest / "progress.json", {"arm": arm, "epochs": history, "status": "running"})
        print("EPOCH", arm, history[-1], flush=True)
    assert updates == 270
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    path = dest / "final.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": CFG["epochs"] - 1,
                "updates": updates, "policy": arm, "protocol_sha256": sha256(OUT / "protocol.json")}, path)
    write_json(dest / "complete.json", {"arm": arm, "epochs": history, "updates": updates,
        "checkpoint_sha256": sha256(path), "runtime_s": time.monotonic() - tick})


def keys(arm: str, sequence: str) -> None:
    """각 학습 모델의 첫 통과를 공유하여 네 추론 정책의 키를 만든다."""
    protocol()
    s, q = sequence.split("/")
    assert (s, q) in VAL
    meta_path = OUT / f"{arm}_keys_{s}_{q}.json"
    if meta_path.exists():
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache, layout, _, gt, _, _ = fixed.parent.rel.load_data("val", sequence)
    bundle = load_model(arm)
    _, graph, c = base.capture(cache, s, layout, bundle, "cuda")
    all_emb, prob, all_ids = base.apply_classifier(bundle[1], graph, c, base.load_classifier("xyz"))
    saved = {"query": c["query"], "neighbor": c["neighbor"], "probability": prob,
        "distance": np.linalg.norm(gt[c["query"], :3, 3] - gt[c["neighbor"], :3, 3], axis=1)}
    records = {}
    for policy in POLICIES:
        ids = choose(c["query"], prob, len(gt), policy, threshold())
        if policy == "all":
            np.testing.assert_array_equal(ids, all_ids)
        emb = all_emb if policy == "all" else fixed.infer(bundle[1], base.replace_edges(graph, c, prob, ids))
        dest = folder(arm, policy)
        dest.mkdir(exist_ok=True)
        path = dest / f"keys_{s}_{q}.npz"
        np.savez_compressed(path, embedding=emb, scan_ids=cache["scan_ids"])
        records[policy] = {"sha256": sha256(path), "selected_edges": len(ids)}
        saved["selected__" + policy] = ids
    path = OUT / f"{arm}_edges_{s}_{q}.npz"
    np.savez_compressed(path, **saved)
    write_json(meta_path, {"arm": arm, "sequence": sequence, "policies": records,
                           "edges_sha256": sha256(path), "threshold": threshold()})
    print("KEYS", arm, sequence, flush=True)


def score(arm: str, policy: str, sequence: str) -> None:
    """최종 검색·융합·수락 원장을 만든다."""
    protocol()
    s, q, paths, min_past = base.old.alignment.base.inputs("val", sequence)
    dest = folder(arm, policy)
    if (dest / f"val_{s}_{q}.json").exists():
        return
    with np.load(paths["cache"]) as z:
        gt, scan_ids = z["poses_gt"][:, :3, 3], z["scan_ids"]
    with np.load(dest / f"keys_{s}_{q}.npz") as z:
        emb = z["embedding"]
        np.testing.assert_array_equal(scan_ids, z["scan_ids"])
    selection = base.read(base.old.BASE / "s1/confidence_selection.json")["ops"]
    records = {}
    for op in OPS:
        with np.load(paths["spec"]) as z:
            bank = base.old.alignment.base.bank_for(op, z)
        sel = selection[op]
        a = evaluate_keys(emb, bank, gt, min_past, {1: tuple(float(w) for w in sel["tuple"])})[1]
        a = {k: v[a["query_idx"] < len(gt) - 15] for k, v in a.items()}
        ref, _ = base.old.read_ledger(previous.folder("opposite", "all") / f"val_{op}_{s}_{q}.npz")
        for k in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(a[k], ref[k])
        a["group"] = ref["group"]
        path = dest / f"val_{op}_{s}_{q}.npz"
        np.savez_compressed(path, **a, meta=json.dumps({"protocol_sha256": sha256(OUT / "protocol.json")}))
        records[op] = {"metrics": base.old.sequence_metrics(a, sel["decision"], sel["tau_star"]),
                       "ledger": str(path.relative_to(REPO)), "sha256": sha256(path)}
    write_json(dest / f"val_{s}_{q}.json", {"arm": arm, "policy": policy, "sequence": sequence, "ops": records})
    print("SCORE", arm, policy, sequence, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "train", "keys", "score"))
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--policy", choices=POLICIES)
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "prepare":
        prepare()
    elif a.action == "train":
        train(a.arm)
    elif a.action == "keys":
        keys(a.arm, a.sequence)
    else:
        score(a.arm, a.policy, a.sequence)
