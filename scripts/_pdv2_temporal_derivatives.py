#!/usr/bin/env python3
"""시간 간선의 표준·차이·거리 차분을 동일 학습과 최종 판정으로 대조한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/_pdv2_temporal_derivatives_run.py
"""
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_evidence as exp
import _pdv2_sequence_fixed_test as fixed
from _pdv2_context_accept import macro
from _pdv2_encoder_place_lib import temporal_subgraph
from _pdv2_input_alignment_lib import make_graph
from _pdv2_no_context_ablation import evaluate_keys, sha256, write_json
from _pdv2_temporal_capacity import create_capacity_model
from _pdv2_train_failure import GRAPH, PREPARED
from _train_attention_neighbors import array_hash
from _xs_eval_lib import normalize, align_scores, column_align_scores, minmax
from _causal_pr_dump import prefix_topk
from gnn.model import create_spectral_gnn
from gnn.temporal_derivatives import TemporalDerivativeGNN, attach_derivatives

OUT = REPO / "results/pdv2_temporal_derivatives_20261020"
CONFIG = REPO / "configs/pdv2_temporal_derivatives.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
ARMS = tuple(CFG["arms"])
SAMPLES = exp.old.old.SAMPLES
EPOCHS = tuple(CFG["checkpoint_epochs"])
METHOD = CFG["final_score"]
FIELDS = ("query_idx", "j_star", "top1_geo_dist", "min_past_gt_dist", "group")


def read(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(path.read_text())


def protocol(check_inputs: bool = False) -> dict:
    """학습 전 고정한 소스와 선택적으로 모든 입력을 검사한다."""
    spec = read(OUT / "protocol.json")
    for field in (("runtime", "inputs") if check_inputs else ("runtime",)):
        for p, h in spec[field].items():
            assert sha256(REPO / p) == h, p
    return spec


def model(arm: str):
    """공통 모듈 초기화 뒤 조건별 집계만 교체한다."""
    cfg = yaml.safe_load((exp.old.OUT / "standard.yaml").read_text())
    factory = create_capacity_model if arm == "point" else create_spectral_gnn
    net = factory(input_dim=288, hidden_dim=128, context_dim=128, n_layers=2,
                  n_heads=4, dropout=.1, norm_type="batch_norm",
                  use_residual_gate=True, gate_hidden_dim=64, gate_initial_alpha=.0625,
                  diffattn_value_source="diff" if arm == "diff" else "standard",
                  edge_encoder_config=cfg["gnn"]["edge_encoding"])
    if arm in ("arc", "frame"):
        net.gnn = TemporalDerivativeGNN(net.gnn)
    return net


def root(seed: int, arm: str) -> Path:
    """새 모델과 결과를 기존 실험에서 분리한다."""
    return OUT / f"s{seed}" / arm


def triplets(epoch: int) -> np.ndarray:
    """기존 원래 표본 네 묶음을 그대로 읽는다."""
    refresh = 1 + 5 * ((epoch - 1) // 5)
    return np.load(SAMPLES / f"original_epoch{refresh}.npz")["triplets"]


def prepare() -> None:
    """표본·연산·파라미터 수·선택·확장 규칙을 결과 관측 전에 동결한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    previous = exp.old.protocol(True)
    state = torch.load(PREPARED, map_location="cpu", weights_only=False)
    original = torch.load(GRAPH, map_location="cpu", weights_only=False)
    graph = temporal_subgraph(original, np.arange(original.num_nodes))
    audit = read(SAMPLES / "audit.json")
    with np.load(SAMPLES / "training_inputs.npz") as z:
        for k in ("descriptors", "poses", "sequences"):
            np.testing.assert_array_equal(state[k], z[k])
    reference = read(exp.old.OUT / "s1/standard_original/training_graph.json")
    hashes = {k: array_hash(getattr(graph, k).numpy())
              for k in ("x", "edge_index", "edge_attr", "edge_type")}
    assert hashes == reference["hashes"]
    assert bool((graph.edge_type == 0).all())
    inputs = {PREPARED, GRAPH, SAMPLES / "audit.json", SAMPLES / "training_inputs.npz",
              exp.old.OUT / "standard.yaml", exp.old.base.RVEC}
    for epoch in (1, 6, 11, 16):
        p = SAMPLES / f"original_epoch{epoch}.npz"
        assert sha256(p) == audit["sample_sha256"][p.name]
        assert triplets(epoch).shape == (183249, 3)
        inputs.add(p)
    plans = {}
    for arm in ("arc", "frame"):
        g = graph.clone()
        plan = attach_derivatives(g, state["poses"][:, :3, 3], state["sequences"], arm)
        path = OUT / f"{arm}_training_graph.pt"
        torch.save(g, path)
        inputs.add(path)
        plans[arm] = {
            "valid_fraction": plan["valid"].mean(0).tolist(),
            "radii_quantiles": [np.quantile(plan["radii"][plan["valid"][:, k], k],
                                           [0, .1, .5, .9, 1]).tolist() for k in (0, 1)],
            "same_scale_fraction": float(np.mean(plan["radii"][:, 0] == plan["radii"][:, 1])),
            "coefficient_sha256": array_hash(plan["coefficients"]),
            "all_support_inside_temporal_edges": True,
        }
    counts, initial = {}, {}
    for arm in ARMS:
        torch.manual_seed(1)
        net = model(arm)
        counts[arm] = sum(p.numel() for p in net.parameters())
        initial[arm] = {k: array_hash(v.numpy()) for k, v in net.state_dict().items()}
    common = set.intersection(*(set(v) for v in initial.values()))
    assert all(len({initial[a][k] for a in ARMS}) == 1 for k in common)
    assert initial["standard"] == initial["diff"]
    assert initial["arc"] == initial["frame"]
    val = [s + "/" + q for s, q in exp.old.base.VAL]
    confidence = {}
    for seed in (1, 2, 3):
        p = exp.old.base.BASE / f"s{seed}/confidence_selection.json"
        confidence[str(seed)] = read(p)["ops"][exp.OP]
        inputs.add(p)
    for sequence in val:
        sensor, seq, paths, _ = exp.old.base.inputs("val", sequence)
        inputs.update(paths.values())
        for seed in (1, 2, 3):
            inputs.add(fixed.VAL / f"s{seed}/standard_{sensor}_{seq}.npz")
    runtime = dict(previous["runtime_sha256"])
    for p in (CONFIG, Path(__file__), REPO / "scripts/_pdv2_temporal_derivatives_run.py",
              REPO / "src/gnn/temporal_derivatives.py",
              REPO / "tests/gnn/test_temporal_derivatives.py",
              REPO / "scripts/_pdv2_temporal_capacity.py",
              REPO / "scripts/_pdv2_encoder_place_lib.py",
              REPO / "scripts/_pdv2_input_alignment_lib.py",
              REPO / "scripts/_pdv2_context_accept.py",
              REPO / "scripts/_pdv2_sequence_fixed_test.py",
              REPO / "scripts/_pdv2_spectral_evidence.py",
              REPO / "src/retrieval/sequence_evidence.py",
              REPO / "src/retrieval/spectral_evidence.py"):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "config": CFG, "val": val,
        "train": previous["train_roster"], "confidence": confidence,
        "parameters": counts, "initial_common_tensors": len(common), "plans": plans,
        "graph_hashes": hashes, "graph_nodes": graph.num_nodes,
        "graph_edges": graph.edge_index.shape[1], "test_used": False,
        "training": "GT 원 72306노드·같은 시간간선/183249삼중쌍/순열/초기 공통 텐서. 각 20에포크900업데이트. 기존 InfoNCE·Adam5e-4/1e-5·AMP·3epoch warmup/cosine. 다섯 조건 모두 새로 학습. no_interdiff 캐시288·게이트초기.0625. 직접 GNNTrainer 사용.",
        "operator": "arc: ±5 간선 안에서 누적거리 ±1.6/3.2m 선형보간. 양측 도달거리로 축소,0.4m미만/한쪽누락은차분0. 정지동일거리평균. [h,|v1|,a1,|v2|,a2]640→68→128 두층. raw/정규화/잔차/게이트 유지. 층별최대5,두층최대10. frame: 같은 연산·가중치·용량,거리대신번호±2/4. 역순불변,실제반대경로불변보장아님.",
        "comparison": "167433 대 standard/diff167561,point167705. 모델용량은근사,함수형태/계산량/중간활성화수는같지않음. frame도변화연산이며arc와의차이는거리매개화. 단순평균/차이전파나새신규성증명으로표현안함.",
        "evaluation": "동일 KISS 검증9·99236질의·q+15·j<=q-30·100+100. 시드별재계량/융합/판정채널고정. 각모델후보와승자재계산. 주판정arc_unique_b100,단일쌍raw보조. 추가문맥가중합/새임계값선택없음.",
        "selection": "각조건10/20중NSD 최종 주판정 AP의 시퀀스→3계열평균최대,1e-12동률이른epoch. 동일epoch결과모두보존. 예비arc가standard/point모두대비AP+.1%p,R100손실<=.1%p,계열AP손실<=.25%p이면standard/point/arc시드2,3확장. frame사후주조건승격없음. 시험/원고교체없음.",
        "runtime": runtime,
        "inputs": {str(p.relative_to(REPO)): sha256(p) for p in sorted(inputs)},
    })
    print("PREPARED", counts, plans, flush=True)


def train(seed: int, arm: str) -> None:
    """같은 표본·배치·업데이트 수로 한 조건을 처음부터 학습한다."""
    from gnn.trainer import GNNTrainer
    spec = protocol()
    folder = root(seed, arm)
    marker = folder / "training_complete.json"
    if marker.exists():
        for p, h in read(marker)["checkpoints"].items():
            assert sha256(folder / "checkpoints" / p) == h
        return
    assert not (folder / "checkpoints").exists(), "미완료 실행을 덮어쓰지 않는다"
    folder.mkdir(parents=True, exist_ok=True)
    state = torch.load(PREPARED, map_location="cpu", weights_only=False)
    if arm in ("arc", "frame"):
        graph = torch.load(OUT / f"{arm}_training_graph.pt", map_location="cpu", weights_only=False)
    else:
        original = torch.load(GRAPH, map_location="cpu", weights_only=False)
        graph = temporal_subgraph(original, np.arange(original.num_nodes))
        del original
    for k, h in spec["graph_hashes"].items():
        assert array_hash(getattr(graph, k).numpy()) == h
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = model(arm)
    write_json(folder / "initial_model.json",
               {k: array_hash(v.numpy()) for k, v in net.state_dict().items()})
    trainer = GNNTrainer(net, learning_rate=5e-4, weight_decay=1e-5, temperature=.1,
                         checkpoint_dir=str(folder / "checkpoints"), use_multi_gpu=False,
                         patience=100, use_amp=True, loss_type="infonce")
    trainer.scaler = torch.amp.GradScaler("cuda", init_scale=128., growth_interval=1000000)
    assert trainer.context_aux_lambda == trainer.phase_token_aux_lambda == 0
    updates, orders, records = [], [], []
    trainer.optimizer.register_step_post_hook(lambda *args: updates.append(1))

    class FixedMiner:
        def mine_triplets(self, **kwargs):
            np.testing.assert_array_equal(kwargs["descriptors"], state["descriptors"])
            return triplets(trainer.epoch + 1).tolist()

    original_shuffle, original_epoch = np.random.shuffle, trainer.train_epoch

    def audited_shuffle(values):
        before = np.random.get_state()
        original_shuffle(values)
        if isinstance(values, np.ndarray) and values.shape == (183249, 3):
            after = np.random.get_state()
            permutation = np.arange(len(values))
            np.random.set_state(before)
            original_shuffle(permutation)
            np.random.set_state(after)
            np.testing.assert_array_equal(values, triplets(trainer.epoch + 1)[permutation])
            orders.append({"permutation": array_hash(permutation), "triplets": array_hash(values)})

    def audited_epoch(*args, **kwargs):
        epoch = trainer.epoch + 1
        epoch_seed = seed * 100000 + epoch
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        torch.cuda.manual_seed_all(epoch_seed)
        start = time.time()
        value = original_epoch(*args, **kwargs)
        assert np.isfinite(value) and len(updates) == epoch * 45 and len(orders) == epoch
        assert all(torch.isfinite(v).all() for v in net.state_dict().values())
        records.append({"epoch": epoch, "updates": len(updates), "loss": float(value),
                        "seconds": time.time() - start, "lr": trainer.optimizer.param_groups[0]["lr"],
                        **orders[-1]})
        write_json(folder / "training_progress.json", {"records": records})
        print("EPOCH", seed, arm, epoch, round(value, 6), round(time.time()-start, 2), flush=True)
        return value

    trainer.train_epoch, np.random.shuffle = audited_epoch, audited_shuffle
    start = time.time()
    try:
        trainer.train(graph, state["poses"], state["descriptors"], state["sequences"],
                      val_datasets=None, n_epochs=20, triplet_miner=FixedMiner(),
                      mine_every_n_epochs=5, n_triplets_per_anchor=3)
    finally:
        np.random.shuffle = original_shuffle
    hashes = {}
    for epoch in EPOCHS:
        p = folder / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
        checkpoint = torch.load(p, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == epoch - 1
        assert all(int(v["step"].item()) == epoch*45
                   for v in checkpoint["optimizer_state_dict"]["state"].values())
        hashes[p.name] = sha256(p)
    write_json(marker, {"epochs": 20, "updates": len(updates), "checkpoints": hashes,
                        "seconds": time.time()-start, "orders": orders, "test_used": False})


def make_keys(net, cache: dict, arm: str, end: int = None) -> tuple:
    """추론 자세로 시간 간선을 만들고 검색 키를 추출한다."""
    raw = cache["descriptors"][:end].astype(np.float32)
    pose = cache["poses"][:end]
    sequence = np.zeros(len(raw), np.int64)
    graph = make_graph(raw, pose, sequence)
    plan = attach_derivatives(graph, pose[:, :3, 3], sequence, arm) if arm in ("arc", "frame") else None
    with torch.no_grad():
        key = normalize(net(graph.cuda()).float().cpu().numpy())
    assert key.shape == (len(raw), 416) and np.isfinite(key).all()
    return key, plan


def validation(seed: int, arm: str, epoch: int, sequence: str) -> None:
    """후보를 새로 검색하고 고정한 최종 시간 판정으로 평가한다."""
    spec = protocol()
    folder = root(seed, arm) / f"epoch{epoch}"
    folder.mkdir(exist_ok=True)
    target = folder / (sequence.replace("/", "_") + ".npz")
    if target.exists() and target.with_suffix(".json").exists():
        assert sha256(target) == read(target.with_suffix(".json"))["sha256"]
        return
    checkpoint = root(seed, arm) / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
    assert sha256(checkpoint) == read(root(seed, arm) / "training_complete.json")["checkpoints"][checkpoint.name]
    net = model(arm).cuda().eval()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(exp.old.base.RVEC)).float().cuda(),
                                          requires_grad=False)
    sensor, seq, paths, min_past = exp.old.base.inputs("val", sequence)
    cache = exp.arrays(paths["cache"])
    tick = time.time()
    key, plan = make_keys(net, cache, arm)
    key_seconds = time.time()-tick
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        bank = exp.old.base.bank_for(exp.OP, z)
    conf = spec["confidence"][str(seed)]
    ledger = evaluate_keys(key, bank, cache["poses_gt"][:, :3, 3], min_past,
                           {seed: tuple(conf["tuple"])})[seed]
    keep = ledger["query_idx"] < len(key)-15
    ledger = {k: v[keep] for k, v in ledger.items()}
    ref = exp.arrays(fixed.VAL / f"s{seed}/standard_{sensor}_{seq}.npz")
    for k in ("query_idx", "min_past_gt_dist"):
        np.testing.assert_array_equal(ledger[k], ref[k])
    ledger["group"] = ref["group"]
    q, j = ledger["query_idx"], ledger["j_star"]
    assert np.all(j <= q-30)
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    chunks = [fixed.frozen_scores(bank, travel, q[i:i+256], j[i:i+256],
              conf["decision"], ledger[conf["decision"]][i:i+256])
              for i in range(0, len(q), 256)]
    scores = {k: np.concatenate([r[k] for r in chunks]) for k in ("raw", METHOD)}
    prefixes = []
    for ix in (len(q)//3, 2*len(q)//3):
        t, end = int(q[ix]), int(q[ix]+16)
        small, _ = make_keys(net, cache, arm, end)
        np.testing.assert_allclose(small[:t+1], key[:t+1], atol=4e-6, rtol=4e-5)
        ids = np.unique(np.r_[prefix_topk(small, t, t-29, 100),
                              prefix_topk(bank[0], t, t-29, 100)])
        values = (small[ids] @ small[t], bank[0][ids] @ bank[0][t],
                  align_scores(bank[1], bank[2], t, ids),
                  column_align_scores(bank[3], t, ids))
        fused = sum(float(w)*minmax(v) for w, v in zip(conf["tuple"], values))
        winner = int(ids[np.argsort(-fused, kind="stable")[0]])
        small_score = fixed.frozen_scores(exp.truncate_bank(bank, end), travel[:end],
                                         q[ix:ix+1], j[ix:ix+1], conf["decision"],
                                         ledger[conf["decision"]][ix:ix+1])
        np.testing.assert_array_equal(small_score[METHOD], scores[METHOD][ix:ix+1])
        prefixes.append({"q": t, "key_max_error": float(abs(small[:t+1]-key[:t+1]).max()),
                         "prefix_winner": winner, "full_winner": int(j[ix]),
                         "winner_equal": winner == int(j[ix]), "score_error": 0.})
    np.savez_compressed(target, **{k: ledger[k] for k in FIELDS}, **scores)
    key_path = folder / ("keys_" + target.name)
    np.savez_compressed(key_path, embedding=key, scan_ids=cache["scan_ids"])
    write_json(target.with_suffix(".json"), {
        "sha256": sha256(target), "keys_sha256": sha256(key_path),
        "metrics": {k: exp.metrics(v, ledger) for k, v in scores.items()},
        "prefixes": prefixes, "key_seconds": key_seconds, "total_seconds": time.time()-tick,
        "valid_fraction": None if plan is None else plan["valid"].mean(0).tolist(),
        "test_used": False,
    })
    print("VAL", seed, arm, epoch, sequence, round(time.time()-tick, 2), flush=True)


def report() -> dict:
    """최종 판정 AP로 체크포인트를 고르고 사전 확장 기준을 검사한다."""
    spec = protocol()
    per_seed = {}
    for seed in (1, 2, 3):
        per_seed[str(seed)] = {}
        for arm in ARMS:
            candidates = []
            for epoch in EPOCHS:
                paths = [root(seed, arm) / f"epoch{epoch}" / (s.replace("/", "_")+".json")
                         for s in spec["val"]]
                if not all(p.exists() for p in paths):
                    continue
                rows = {s: read(p) for s, p in zip(spec["val"], paths)}
                aggregate = {m: macro({s: r["metrics"][m] for s, r in rows.items()})
                             for m in ("raw", METHOD)}
                candidates.append({"epoch": epoch, **aggregate})
            if len(candidates) == len(EPOCHS):
                chosen = max(candidates, key=lambda r: (round(r[METHOD]["macro"]["AP"], 12), -r["epoch"]))
                per_seed[str(seed)][arm] = {"selected": chosen, "candidates": candidates}
        if not per_seed[str(seed)]:
            del per_seed[str(seed)]
    assert set(per_seed["1"]) == set(ARMS)
    effects, checks = {}, {}
    for seed, models in per_seed.items():
        chosen = models["arc"]["selected"][METHOD]
        effects[seed], checks[seed] = {}, {}
        for arm in models:
            baseline = models[arm]["selected"][METHOD]
            effects[seed][arm] = {m: 100*(chosen["macro"][m]-baseline["macro"][m])
                                  for m in exp.METRICS}
            if arm in CFG["promotion"]["comparators"]:
                checks[seed][arm] = {
                    "AP": chosen["macro"]["AP"] >= baseline["macro"]["AP"]+.001,
                    "R100": chosen["macro"]["R@100P"] >= baseline["macro"]["R@100P"]-.001,
                    "family_AP": all(chosen["families"][f]["AP"] >= baseline["families"][f]["AP"]-.0025
                                     for f in baseline["families"]),
                }
    eligible = all(all(v.values()) for v in checks["1"].values())
    summary = {"per_seed": per_seed, "arc_minus_other_pp": effects, "promotion_checks": checks,
               "pilot_eligible": eligible, "test_used": False,
               "protocol_sha256": sha256(OUT / "protocol.json")}
    write_json(OUT / "summary.json", summary)
    for seed, models in per_seed.items():
        for arm, row in models.items():
            print("FINAL", seed, arm, row["selected"]["epoch"], row["selected"][METHOD]["macro"], flush=True)
    print("PILOT_ELIGIBLE", eligible, checks, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "val", "report"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--epoch", type=int, choices=EPOCHS)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    torch.set_num_threads(4 if args.action == "train" else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        train(args.seed, args.arm)
    elif args.action == "val":
        validation(args.seed, args.arm, args.epoch, args.sequence)
    else:
        report()
