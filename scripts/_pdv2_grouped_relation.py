#!/usr/bin/env python3
"""관계 종류와 attention 공유 범위의 2x2 최종 판정 대조."""
import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_nonlinear_difference as previous
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from gnn.grouped_relation import attach_grouped_relation

base = previous.base
OUT = REPO / "results/pdv2_grouped_relation_20261021"
CONFIG = REPO / "configs/pdv2_grouped_relation.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
ARMS = tuple(CFG["arms"])
ALL_ARMS = (*ARMS, CFG["reference"])
METHOD = CFG["final_score"]
read = previous.read
BASE_EVALUATE = base.evaluate_keys


def model(arm: str) -> torch.nn.Module:
    """Q/K/V 행렬을 유지하며 head마다 가중치 묶음을 1개 또는 8개로 만든다."""
    if arm == "point":
        return previous.BASE_MODEL("point")
    relation, mode = arm.split("_", 1)
    net = previous.BASE_MODEL(relation)
    attach_grouped_relation(net, CFG["groups_per_head"][mode])
    return net


def protocol(inputs: bool = False) -> dict:
    """동결된 소스와 선택적 입력 해시를 확인한다."""
    spec = read(OUT / "protocol.json")
    for field in (("runtime", "inputs") if inputs else ("runtime",)):
        for path, expected in spec[field].items():
            assert sha256(REPO / path) == expected, path
    return spec


def prepare() -> None:
    """학습 전에 네 조건·대조군·재사용·확장 규칙을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    old = previous.protocol(True)
    sources = dict(old["inputs"])
    reused = {**CFG["reused_arms"], "point": "point"}
    for name in ("protocol.json", "summary.json", "completion_validation.json"):
        p = previous.OUT / name
        sources[str(p.relative_to(REPO))] = sha256(p)
    for arm, ref in reused.items():
        folder = previous.OUT / "s1" / ref
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix in (".json", ".npz", ".pth"):
                sources[str(p.relative_to(REPO))] = sha256(p)
        complete = read(folder / "training_complete.json")
        for name, h in complete["checkpoints"].items():
            assert sha256(folder / "checkpoints" / name) == h
    initial, counts = {}, {}
    ref_initial = read(previous.OUT / "s1/diff_tanh/initial_model.json")
    for arm in ALL_ARMS:
        torch.manual_seed(1)
        net = model(arm)
        counts[arm] = sum(p.numel() for p in net.parameters())
        initial[arm] = {k: array_hash(v.numpy()) for k, v in net.state_dict().items()}
        if arm != "point":
            assert initial[arm] == ref_initial and counts[arm] == 167561
    common = set.intersection(*(set(v) for v in initial.values()))
    assert all(len({v[k] for v in initial.values()}) == 1 for k in common)
    original = torch.load(base.GRAPH, map_location="cpu", weights_only=False)
    graph = base.temporal_subgraph(original, np.arange(original.num_nodes)).cuda()
    del original
    smoke = {}
    for arm in CFG["train_arms"]:
        torch.manual_seed(1)
        net = model(arm).cuda().train()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.float16):
            result = net(graph)
            loss = result.float().square().mean()
        loss.backward()
        assert result.shape == (72306, 416) and torch.isfinite(result).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        smoke[arm] = {"shape": list(result.shape), "finite": True,
                      "parameters_with_gradient": sum(p.grad is not None for p in net.parameters()),
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        del result, loss, net
    del graph
    torch.cuda.empty_cache()
    write_json(OUT / "amp_smoke.json", smoke)
    for arm, ref in reused.items():
        source = previous.OUT / "s1" / ref
        target = base.root(1, arm)
        (target / "checkpoints").mkdir(parents=True, exist_ok=True)
        for name in read(source / "training_complete.json")["checkpoints"]:
            shutil.copy2(source / "checkpoints" / name, target / "checkpoints" / name)
        for name in ("initial_model.json", "training_complete.json", "training_progress.json"):
            shutil.copy2(source / name, target / name)
        write_json(target / "reuse_provenance.json", {
            "source": str(source.relative_to(REPO)), "existing_updates": 900,
            "retrained_in_this_experiment": False, "evaluation_replayed": True})
    runtime = dict(old["runtime"])
    for p in (CONFIG, Path(__file__), REPO / "scripts/_pdv2_grouped_relation_run.py",
              REPO / "src/gnn/grouped_relation.py", REPO / "tests/gnn/test_grouped_relation.py"):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "config": CFG, "runtime": runtime, "inputs": sources,
        "val": old["val"], "train": old["train"], "confidence": old["confidence"],
        "graph_hashes": old["graph_hashes"], "parameters": counts,
        "initial_attention_tensors": len(ref_initial), "initial_common_tensors": len(common),
        "test_used": False,
        "operator": "Q=Wq hi; K=tanh(Wk r); V=Wv r. r=hj-hi 또는 hj. 각 head32채널에 attention1개 또는 4채널씩8개. 묶음별 QK 내적/sqrt(묶음폭), 이웃 softmax, 원래 head 간선 편향 공유. 128채널2층±5시간간선,추가학습매개변수0.",
        "operator_ko": "집계 값은 선형으로 고정한다. 4head 가중치4개를 32개로 늘리되 기존 행렬과 간선 편향 행렬을 유지한다. 동일 매개변수 수지만 attention 활성화와 연산량은 증가한다. dropout은 같은 seed이나 가중치 배열 크기가 달라 기준과 같은 마스크는 아니다.",
        "training": "학습8 GT72306노드722820시간간선,같은183249삼중쌍/초기텐서/순서. 20epoch900업데이트,InfoNCE/Adam5e-4/weight_decay1e-5/AMP. no_interdiff288·gate초기.0625. 25m제외 손실은 쓰지 않아 연산효과만 분리. 기존 train 함수 직접 재사용.",
        "evaluation": "KISS검증9·99236질의·q+15·j<=q-30·100+100·시드별 기존 재계량/융합/판정 고정. 모두 후보/승자 재검색. 최종arc_unique_b100 AP/F1/R100/R1,raw보조. 이웃없는투영도같은최종시간판정이므로전체무문맥대조아님.",
        "selection": "조건별10/20중최종AP최대,1e-12동률이른epoch. 차이후보 각각을diff_linear/standard_linear/대응standard/point와비교. 모든대조대비AP+.1%p,F1및R100손실<=.1%p,계열AP손실<=.25%p면확장가능. 통과후보중AP최고(동률config순서)와네대조군을시드2,3으로확장. 실패후보사후승격없음. 시험/원고교체없음.",
        "novelty": "Point Transformer의 vector attention 등 선행연구가 있다. 묶음별 가중치 자체를 신규성으로 주장하지 않는다. 차이/표준의 개선 폭을 같은 epoch에서도 비교한다.",
    })
    print("PREPARED", counts, smoke, flush=True)


def validation(seed: int, arm: str, epoch: int, sequence: str) -> None:
    """기존 최종 평가를 재사용하면서 후보 포함 여부도 보존한다."""
    folder = base.root(seed, arm) / f"epoch{epoch}"
    target = folder / ("coverage_" + sequence.replace("/", "_") + ".npz")

    def capture(emb, bank, positions, min_past, weights_by_seed):
        records = BASE_EVALUATE(emb, bank, positions, min_past, weights_by_seed)
        row = records[seed]
        keep = row["query_idx"] < len(emb) - 15
        np.savez_compressed(target, **{k: v[keep] for k, v in row.items()})
        write_json(target.with_suffix(".json"), {"sha256": sha256(target), "test_used": False})
        return records

    base.evaluate_keys = capture
    try:
        base.validation(seed, arm, epoch, sequence)
    finally:
        base.evaluate_keys = BASE_EVALUATE
    assert sha256(target) == read(target.with_suffix(".json"))["sha256"]


def report() -> dict:
    """전체 조건과 같은 epoch 효과를 보고하고 사전 기준대로 확장한다."""
    spec = protocol()
    per_seed = {}
    for seed in (1, 2, 3):
        rows = {}
        for arm in ALL_ARMS:
            candidates = []
            for epoch in (10, 20):
                paths = [base.root(seed, arm) / f"epoch{epoch}" /
                         (s.replace("/", "_") + ".json") for s in spec["val"]]
                if not all(p.exists() for p in paths):
                    continue
                records = {s: read(p) for s, p in zip(spec["val"], paths)}
                candidates.append({"epoch": epoch, **{m: base.macro({
                    s: r["metrics"][m] for s, r in records.items()}) for m in ("raw", METHOD)}})
            if len(candidates) == 2:
                chosen = max(candidates, key=lambda x: (round(x[METHOD]["macro"]["AP"], 12), -x["epoch"]))
                rows[arm] = {"selected": chosen, "candidates": candidates}
                print("FINAL", seed, arm, chosen["epoch"], chosen[METHOD]["macro"], flush=True)
        if rows:
            per_seed[str(seed)] = rows
    assert set(per_seed["1"]) == set(ALL_ARMS)
    checks, effects, interactions = {}, {}, {}
    pilot = per_seed["1"]
    for arm in CFG["candidate_arms"]:
        selected = pilot[arm]["selected"][METHOD]
        matched = arm.replace("diff_", "standard_", 1)
        checks[arm], effects[arm], interactions[arm] = {}, {}, {}
        for ref in ("diff_linear", "standard_linear", matched, "point"):
            other = pilot[ref]["selected"][METHOD]
            delta = {m: 100 * (selected["macro"][m] - other["macro"][m]) for m in base.exp.METRICS}
            effects[arm][ref] = delta
            checks[arm][ref] = {
                "AP": delta["AP"] >= 100 * CFG["promotion"]["ap_gain_min"],
                "F1": delta["max_F1"] >= -100 * CFG["promotion"]["f1_loss_max"],
                "R100": delta["R@100P"] >= -100 * CFG["promotion"]["r100_loss_max"],
                "family_AP": all(selected["families"][f]["AP"] >=
                    other["families"][f]["AP"] - CFG["promotion"]["family_ap_loss_max"]
                    for f in other["families"]),
            }
        for epoch in (10, 20):
            interactions[arm][str(epoch)] = {m: 100 * sum(sign * next(
                c for c in pilot[a]["candidates"] if c["epoch"] == epoch)[METHOD]["macro"][m]
                for a, sign in ((arm, 1), ("diff_linear", -1), (matched, -1), ("standard_linear", 1)))
                for m in base.exp.METRICS}
    eligible = [a for a in CFG["candidate_arms"] if all(all(v.values()) for v in checks[a].values())]
    best = max(eligible, key=lambda a: pilot[a]["selected"][METHOD]["macro"]["AP"]) if eligible else None
    expansion = {"candidate": best, "arms": ([best, "diff_linear", "standard_linear",
        best.replace("diff_", "standard_", 1), "point"] if best else []),
        "seeds": CFG["expansion_seeds"] if best else [], "test_used": False}
    selection_path = OUT / "expansion_selection.json"
    if selection_path.exists():
        assert read(selection_path) == expansion
    else:
        write_json(selection_path, expansion)
    summary = {"per_seed": per_seed, "candidate_minus_other_pp": effects,
        "same_epoch_interaction_pp": interactions, "promotion_checks": checks,
        "eligible_candidates": eligible, "expansion": expansion,
        "protocol_sha256": sha256(OUT / "protocol.json"), "test_used": False}
    write_json(OUT / "summary.json", summary)
    print("EXPANSION", expansion, flush=True)
    return summary


base.OUT, base.CONFIG, base.CFG, base.ARMS = OUT, CONFIG, CFG, ALL_ARMS
base.model, base.protocol = model, protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "val", "report"))
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--arm", choices=ALL_ARMS)
    parser.add_argument("--epoch", type=int, choices=(10, 20))
    parser.add_argument("--sequence")
    args = parser.parse_args()
    torch.set_num_threads(4 if args.action == "train" else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        base.train(args.seed, args.arm)
    elif args.action == "val":
        validation(args.seed, args.arm, args.epoch, args.sequence)
    else:
        report()
