#!/usr/bin/env python3
"""차이 키를 유지한 중심 값 보존의 동일 예산 최종 판정 대조."""
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
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_nonlinear_difference as previous
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from gnn.center_preserving_relation import attach_center_preservation

base = previous.base
OUT = REPO/"results/pdv2_center_preservation_20261021"
CONFIG = REPO/"configs/pdv2_center_preservation.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
ARMS = tuple(CFG["arms"])
METHOD = CFG["final_score"]
read = previous.read
BASE_EVALUATE = base.evaluate_keys


def model(arm: str) -> torch.nn.Module:
    """같은 차이+tanh 초기화에서 고정 중심 보존량만 바꾼다."""
    net = previous.model("diff_tanh")
    attach_center_preservation(net, CFG["beta"][arm])
    return net


def protocol(inputs: bool = False) -> dict:
    """동결된 실행 소스와 선택적 입력 해시를 확인한다."""
    spec = read(OUT/"protocol.json")
    for field in (("runtime", "inputs") if inputs else ("runtime",)):
        for path, expected in spec[field].items():
            assert sha256(REPO/path) == expected, path
    return spec


def prepare() -> None:
    """학습 전에 연산·예산·평가·재사용 기준 모델을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"protocol.json").exists():
        protocol(True)
        return
    old = previous.protocol(True)
    sources = dict(old["inputs"])
    for p in (previous.OUT/"protocol.json", previous.OUT/"summary.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    ref_arms = ("diff_tanh", *CFG["references"])
    for arm in ref_arms:
        folder = previous.OUT/"s1"/arm
        complete = read(folder/"training_complete.json")
        for filename in ("initial_model.json", "training_complete.json", "training_progress.json"):
            path = folder/filename
            sources[str(path.relative_to(REPO))] = sha256(path)
        for name, expected in complete["checkpoints"].items():
            path = folder/"checkpoints"/name
            assert sha256(path) == expected
            sources[str(path.relative_to(REPO))] = expected
        for epoch in (10, 20):
            for seq in old["val"]:
                p = folder/f"epoch{epoch}"/(seq.replace("/", "_")+".npz")
                info = read(p.with_suffix(".json"))
                for path, expected in ((p, info["sha256"]), (p.with_name("keys_"+p.name), info["keys_sha256"])):
                    assert sha256(path) == expected
                    sources[str(path.relative_to(REPO))] = expected
                sources[str(p.with_suffix('.json').relative_to(REPO))] = sha256(p.with_suffix('.json'))
    counts, initial = {}, {}
    reference_initial = read(previous.OUT/"s1/diff_tanh/initial_model.json")
    for arm in ARMS:
        torch.manual_seed(1)
        net = model(arm)
        counts[arm] = sum(p.numel() for p in net.parameters())
        initial[arm] = {k: array_hash(v.numpy()) for k, v in net.state_dict().items()}
        assert initial[arm] == reference_initial
    assert len(set(counts.values())) == 1 and counts[ARMS[0]] == 167561
    original = torch.load(base.GRAPH, map_location="cpu", weights_only=False)
    graph = base.temporal_subgraph(original, np.arange(original.num_nodes)).cuda()
    del original
    smoke = {}
    for arm in CFG["train_arms"]:
        torch.manual_seed(1)
        net = model(arm).cuda().train()
        with torch.autocast("cuda", dtype=torch.float16):
            output = net(graph)
            loss = output.float().square().mean()
        loss.backward()
        assert output.shape == (72306, 416) and torch.isfinite(output).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        smoke[arm] = {"shape": list(output.shape), "finite": True,
            "parameters_with_gradient": sum(p.grad is not None for p in net.parameters())}
        del net, output, loss
    del graph
    torch.cuda.empty_cache()
    write_json(OUT/"amp_smoke.json", smoke)
    # 원본 모델·결과는 수정하지 않는다. 0 조건의 입력 가중치만 별도 복사한다.
    source = previous.OUT/"s1/diff_tanh"
    target = base.root(1, "center_zero")
    (target/"checkpoints").mkdir(parents=True, exist_ok=True)
    for name in read(source/"training_complete.json")["checkpoints"]:
        shutil.copy2(source/"checkpoints"/name, target/"checkpoints"/name)
    for name in ("initial_model.json", "training_complete.json", "training_progress.json"):
        shutil.copy2(source/name, target/name)
    write_json(target/"reuse_provenance.json", {"source": str(source.relative_to(REPO)),
        "retrained_in_this_experiment": False, "existing_updates": 900,
        "evaluation_replayed": True, "zero_operator_exact": True})
    runtime = dict(old["runtime"])
    for p in (Path(__file__), CONFIG, REPO/"scripts/_pdv2_center_preservation_run.py",
              REPO/"src/gnn/center_preserving_relation.py", REPO/"tests/gnn/test_center_preserving_relation.py"):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT/"protocol.json", {"created_unix": time.time(), "runtime": runtime, "inputs": sources,
        "config": CFG, "val": old["val"], "train": old["train"], "confidence": old["confidence"],
        "graph_hashes": old["graph_hashes"], "parameters": counts,
        "initial_all_tensors_identical": True, "test_used": False,
        "operator": "Q=Wq hi,K=tanh(Wk(hj-hi)) 유지. V=Wv(hj-(1-beta)hi). beta=0/.5/1 두층공통고정,추가파라미터0. dropout없음/값게이트없음/이웃존재에서D+beta*C. AMP 혼합입력은 대수적분해와 반올림차이 가능. 기존BN/잔차/출력/게이트유지.",
        "training": "GT 학습8·72306노드·722820시간간선·기존183249삼중쌍·초기텐서/표본순서동일. 절반/전체각새20epoch900업데이트. InfoNCE/Adam5e-4/1e-5/AMP/3epochwarmup/cosine. no_interdiff288·gate.0625 직접GNNTrainer. beta0은동일기존900회가중치재사용.",
        "evaluation": "KISS검증9·99236질의·q+15·j<=q-30·100+100·기존융합/재계량/시간판정고정. 세조건모두10/20epoch를재검색평가. 각시퀀스coverage별도보존. beta0 원장/키/최종지표재현검사.",
        "selection": "각조건10/20중시퀀스→3계열 최종arc_unique_b100 AP최대,1e-12동률이면이른epoch. primary는center_half,center_full은절대값대조. 두조건과동일epoch모두보고. raw는보조. 시드1예비까지,추가시드/시험/원고자동승격없음.",
        "diagnosis": "최종지표선택후공통양성10036의후보누락/융합손실/정답손익·방향별/계열별을보고. 표준_tanh와의실제손실사례에서비교정답후보누락/키선호융합역전/키오답선호를재현. 기존91개고정사례회복도별도표시. 고정tau의참수락/오수락은보조,새임계값선택없음.",
        "novelty": "차이 키를남기는같은용량원인대조. beta조절이나중심보존자체를신규성으로주장하지않음."})
    print("PREPARED", counts, smoke, flush=True)


def validation(seed: int, arm: str, epoch: int, sequence: str) -> None:
    """기존 최종 평가를 실행하며 중간 후보 누락도 같은 실행에서 보존한다."""
    folder = base.root(seed, arm)/f"epoch{epoch}"
    target = folder/("coverage_"+sequence.replace("/", "_")+".npz")

    def capture_candidates(emb, bank, positions, min_past, weights_by_seed):
        result = BASE_EVALUATE(emb, bank, positions, min_past, weights_by_seed)
        ledger = result[seed]
        keep = ledger["query_idx"] < len(emb)-15
        np.savez_compressed(target, **{k: v[keep] for k, v in ledger.items()})
        write_json(target.with_suffix(".json"), {"sha256": sha256(target), "test_used": False})
        return result

    base.evaluate_keys = capture_candidates
    try:
        base.validation(seed, arm, epoch, sequence)
    finally:
        base.evaluate_keys = BASE_EVALUATE
    assert sha256(target) == read(target.with_suffix(".json"))["sha256"]


def report() -> None:
    """미리 정한 최종 지표로 고르고 모든 조건과 기준 모델을 보고한다."""
    spec = protocol()
    models = {}
    for arm in ARMS:
        candidates = []
        for epoch in (10, 20):
            rows = {s: read(base.root(1, arm)/f"epoch{epoch}"/(s.replace("/", "_")+".json")) for s in spec["val"]}
            candidates.append({"epoch": epoch, **{m: base.macro({s: r["metrics"][m] for s, r in rows.items()}) for m in ("raw", METHOD)}})
        selected = max(candidates, key=lambda x: (round(x[METHOD]["macro"]["AP"], 12), -x["epoch"]))
        models[arm] = {"selected": selected, "candidates": candidates}
        print("FINAL", arm, selected["epoch"], selected[METHOD]["macro"], flush=True)
    old = read(previous.OUT/"summary.json")["per_seed"]["1"]
    refs = {arm: old[arm] for arm in CFG["references"]}
    for actual, expected in zip(models["center_zero"]["candidates"], old["diff_tanh"]["candidates"]):
        assert actual == expected
    effects = {arm: {other: {m: 100*(models[arm]["selected"][METHOD]["macro"][m]-r["selected"][METHOD]["macro"][m])
                            for m in base.exp.METRICS} for other, r in {**models, **refs}.items()}
               for arm in CFG["train_arms"]}
    same_epoch = {arm: {str(epoch): {m: 100*(next(c for c in models[arm]["candidates"] if c["epoch"] == epoch)[METHOD]["macro"][m]
        -next(c for c in models["center_zero"]["candidates"] if c["epoch"] == epoch)[METHOD]["macro"][m])
        for m in base.exp.METRICS} for epoch in (10, 20)} for arm in CFG["train_arms"]}
    write_json(OUT/"summary.json", {"per_seed": {"1": models}, "references": refs,
        "effects_pp": effects, "same_epoch_vs_zero_pp": same_epoch,
        "protocol_sha256": sha256(OUT/"protocol.json"), "test_used": False, "zero_replay_exact": True})


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
        assert args.arm in CFG["train_arms"]
        base.train(args.seed, args.arm)
    elif args.action == "val":
        validation(args.seed, args.arm, args.epoch, args.sequence)
    else:
        report()
