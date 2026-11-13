#!/usr/bin/env python3
"""차이/표준 관계와 선형/tanh 키의 동일 예산 최종 판정 대조."""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal_derivatives as base
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from gnn.nonlinear_relation import attach_nonlinear_relation

PREVIOUS = base.OUT
BASE_MODEL = base.model
OUT = REPO / "results/pdv2_nonlinear_difference_20261021"
CONFIG = REPO / "configs/pdv2_nonlinear_difference.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
ARMS = tuple(CFG["arms"])
METHOD = CFG["final_score"]
read = base.read


def model(arm: str):
    """공통 초기 Q/K/V를 보존하고 관계 키에만 tanh를 추가한다."""
    net = BASE_MODEL(arm.split("_")[0])
    if arm.endswith("_tanh"):
        attach_nonlinear_relation(net)
    return net


def protocol(inputs: bool = False) -> dict:
    """실험 전 고정한 소스와 입력을 확인한다."""
    spec = read(OUT / "protocol.json")
    for field in (("runtime", "inputs") if inputs else ("runtime",)):
        for p, h in spec[field].items():
            assert sha256(REPO / p) == h, p
    return spec


def prepare() -> None:
    """성능 관측 전에 모델·자료·선택 규칙과 실제 AMP 실행을 확인한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    previous = read(PREVIOUS / "protocol.json")
    for field in ("runtime", "inputs"):
        for p, h in previous[field].items():
            assert sha256(REPO / p) == h, p
    counts, initial = {}, {}
    for arm in ARMS:
        torch.manual_seed(1)
        net = model(arm)
        counts[arm] = sum(p.numel() for p in net.parameters())
        initial[arm] = {k: array_hash(v.numpy()) for k, v in net.state_dict().items()}
    assert initial["diff"] == initial["standard"] == initial["diff_tanh"] == initial["standard_tanh"]
    common = set.intersection(*(set(v) for v in initial.values()))
    assert all(len({v[k] for v in initial.values()}) == 1 for k in common)
    original = torch.load(base.GRAPH, map_location="cpu", weights_only=False)
    graph = base.temporal_subgraph(original, np.arange(original.num_nodes)).cuda()
    del original
    smoke = {}
    for arm in ("diff_tanh", "standard_tanh"):
        torch.manual_seed(1)
        net = model(arm).cuda().train()
        with torch.autocast("cuda", dtype=torch.float16):
            output = net(graph)
            loss = output.float().square().mean()
        loss.backward()
        assert output.shape == (graph.num_nodes, 416) and torch.isfinite(output).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        smoke[arm] = {"shape": list(output.shape), "loss": float(loss.detach()),
                      "parameters_with_gradient": sum(p.grad is not None for p in net.parameters()),
                      "finite": True}
        del output, loss, net
    del graph
    torch.cuda.empty_cache()
    write_json(OUT / "amp_smoke.json", smoke)
    runtime = dict(previous["runtime"])
    for p in (CONFIG, Path(__file__), REPO / "scripts/_pdv2_nonlinear_difference_run.py",
              REPO / "src/gnn/nonlinear_relation.py", REPO / "tests/gnn/test_nonlinear_relation.py"):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    inputs = {**previous["inputs"], str((PREVIOUS / "protocol.json").relative_to(REPO)):
              sha256(PREVIOUS / "protocol.json")}
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "config": CFG, "runtime": runtime, "inputs": inputs,
        "val": previous["val"], "train": previous["train"], "confidence": previous["confidence"],
        "graph_hashes": previous["graph_hashes"], "parameters": counts,
        "initial_common_tensors": len(common), "test_used": False,
        "hypothesis": "선형 차이 K의 중심항은softmax에서상쇄된다. tanh(Wk(hj-hi))는그항을분리할수없어 실제관계주의를만든다. 성능과신규성은미확인.",
        "operator": "Q=Wq hi; K=tanh(Wk 관계); V=Wv 관계. 관계는diff(hj-hi)/standard(hj). 기존간선편향·드롭아웃·정규화·잔차·게이트 유지. 새매개변수0,2층,±5시간간선만.",
        "training": "시드1·모든5조건 새로20에포크900업데이트. 동일72306노드시간그래프/초기공통텐서/183249삼중쌍/배치/InfoNCE/Adam5e-4/weight_decay1e-5/AMP. no_interdiff288 캐시·게이트초기.0625,직접GNNTrainer. 학습GT·검증KISS고정.",
        "evaluation": "검증9·99236질의·q+15·j<=q-30·100+100·기존융합/재계량/채널 고정. 후보와승자재검색. 주판정arc_unique_b100; raw보조. 후보경로·새점수상한은붙이지않아연산효과분리.",
        "selection": "각조건10/20중시퀀스→3계열 최종AP최대,1e-12동률이른epoch. diff_tanh가diff/standard/standard_tanh/point각각보다AP+.1%p,R100손실<=.1%p,계열AP손실<=.25%p일때만후속3시드후보. 이번실행은시드1까지,시험/원고교체없음.",
        "novelty": "EdgeConv·PointTransformer에비선형상대특징의선례가있다. tanh나차이사용만으로신규성주장하지않음. difference에서출발한원인분리실험이며임의새명칭으로기여확정하지않음.",
    })
    print("PREPARED", counts, smoke, flush=True)


def report() -> None:
    """조건별 최종 AP로 선택하고 2x2 효과와 채택 기준을 공개한다."""
    spec = protocol()
    models = {}
    for arm in ARMS:
        candidates = []
        for epoch in (10, 20):
            paths = [base.root(1, arm) / f"epoch{epoch}" / (s.replace("/", "_")+".json") for s in spec["val"]]
            rows = {s: read(p) for s, p in zip(spec["val"], paths)}
            candidates.append({"epoch": epoch, **{m: base.macro({s: r["metrics"][m] for s, r in rows.items()})
                              for m in ("raw", METHOD)}})
        chosen = max(candidates, key=lambda x: (round(x[METHOD]["macro"]["AP"], 12), -x["epoch"]))
        models[arm] = {"selected": chosen, "candidates": candidates}
        print("FINAL", arm, chosen["epoch"], chosen[METHOD]["macro"], flush=True)
    chosen = models[CFG["primary"]]["selected"][METHOD]
    checks, effects = {}, {}
    for arm in CFG["promotion"]["comparators"]:
        reference = models[arm]["selected"][METHOD]
        effects[arm] = {m: 100*(chosen["macro"][m]-reference["macro"][m]) for m in base.exp.METRICS}
        checks[arm] = {"AP": effects[arm]["AP"] >= .1,
                       "R100": effects[arm]["R@100P"] >= -.1,
                       "family_AP": all(chosen["families"][f]["AP"] >= reference["families"][f]["AP"]-.0025
                                        for f in reference["families"])}
    interaction = {m: 100*((models["diff_tanh"]["selected"][METHOD]["macro"][m]-models["diff"]["selected"][METHOD]["macro"][m])
                   -(models["standard_tanh"]["selected"][METHOD]["macro"][m]-models["standard"]["selected"][METHOD]["macro"][m]))
                   for m in base.exp.METRICS}
    write_json(OUT / "summary.json", {"per_seed": {"1": models}, "primary_minus_other_pp": effects,
        "interaction_pp": interaction, "promotion_checks": checks,
        "pilot_eligible": all(all(x.values()) for x in checks.values()),
        "protocol_sha256": sha256(OUT / "protocol.json"), "test_used": False})
    print("PILOT_ELIGIBLE", all(all(x.values()) for x in checks.values()), "INTERACTION", interaction, flush=True)


# 기존 실험 코드는 수정하지 않는다. 이 프로세스의 재사용 함수에만 위치·모델을 주입한다.
base.OUT, base.CONFIG, base.CFG, base.ARMS = OUT, CONFIG, CFG, ARMS
base.model, base.protocol = model, protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "val", "report"))
    parser.add_argument("--seed", type=int, choices=(1,), default=1)
    parser.add_argument("--arm", choices=ARMS)
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
        base.validation(args.seed, args.arm, args.epoch, args.sequence)
    else:
        report()
