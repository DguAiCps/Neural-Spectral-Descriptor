#!/usr/bin/env python3
"""GT/KISS 고유 학습 쌍과 두 표본 예산을 같은 최종 검증으로 비교한다."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_input_alignment as align
from _pdv2_native_pairs_lib import nested_rows, positive_pools
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from gnn.matching_mining import paired_triplets

INPUT = align.OUT
OUT = REPO / "results/pdv2_native_pairs_20261020"
ARMS = {"gt_small": ("g", "small"), "kiss_small": ("o", "small"),
        "gt_large": ("g", "large"), "kiss_large": ("o", "large")}
EPOCHS, SEED = 20, 1
read = align.read


def protocol() -> dict:
    """고정한 전체 근거와 새 소스·표본을 검사한다."""
    spec = read(OUT / "protocol.json")
    for p, h in spec["sources"].items():
        assert sha256(REPO / p) == h, p
    return spec


def prepare() -> None:
    """검증 전에 고유 스캔의 원래 마이닝과 동일 표본 예산을 동결한다."""
    from _pdv2_pair_train import hard_negatives
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol()
        return
    parent = align.protocol()
    upstream = read(INPUT / "completion_validation.json")
    assert upstream["status"] == "passed"
    for p, h in upstream["sources"].items():
        assert sha256(REPO / p) == h, p
    data = {p: align.previous.load(INPUT / (p + "_inputs.npz")) for p in ("g", "o")}
    state = torch.load(align.old.previous.PREPARED, weights_only=False, map_location="cpu")
    pools = {p: positive_pools(d["gt"][:, :3, 3], d["sequences"]) for p, d in data.items()}
    for a, b in zip(pools["g"], state["intra"]):
        np.testing.assert_array_equal(a, b)
    negative_path = OUT / "kiss_hard_negative.npz"
    if negative_path.exists():
        assert sha256(negative_path) == read(negative_path.with_suffix(".json"))["sha256"]
        negative = np.load(negative_path)["negative"]
    else:
        negative, checks = hard_negatives(data["o"]["raw"], data["o"]["gt"][:, :3, 3], data["o"]["sequences"])
        np.savez_compressed(negative_path, negative=negative)
        write_json(negative_path.with_suffix(".json"), {"sha256": sha256(negative_path), "checks": checks,
            "inputs": {str((INPUT / "o_inputs.npz").relative_to(REPO)): sha256(INPUT / "o_inputs.npz")}})
    negatives = {"g": state["negative"], "o": negative}
    native, sources = {}, dict(upstream["sources"])
    sources[str(INPUT.relative_to(REPO) / "protocol.json")] = sha256(INPUT / "protocol.json")
    paths = {negative_path, negative_path.with_suffix(".json"), INPUT / "summary.json"}
    for population in data:
        native[population] = {}
        for epoch in align.old.REFRESH:
            trips, unchanged = paired_triplets(pools[population], negatives[population],
                np.full((len(data[population]["raw"]), 1), -1, np.int64), SEED*100000+epoch, fraction=0.)
            np.testing.assert_array_equal(trips, unchanged)
            if population == "g":
                np.testing.assert_array_equal(trips, np.load(align.old.SAMPLES / f"original_epoch{epoch}.npz")["triplets"])
            a, p, n = trips.T
            xyz, seq = data[population]["gt"][:, :3, 3], data[population]["sequences"]
            assert np.all(seq[a] == seq[p]) and np.all(seq[a] == seq[n])
            assert np.all(abs(a-p) >= 30) and np.all(abs(a-n) >= 30)
            assert np.all(np.linalg.norm(xyz[a]-xyz[p], axis=1) <= 5.)
            assert np.all(np.linalg.norm(xyz[a]-xyz[n], axis=1) > 10.)
            native[population][epoch] = trips
            target = OUT / f"{population}_native_epoch{epoch}.npz"
            np.savez_compressed(target, triplets=trips)
            paths.add(target)
    large = min(len(t) for value in native.values() for t in value.values())
    small = parent["triplets_per_epoch"]
    assert (small, large) == (143810, 182919)
    plans, sample_audit = {}, {}
    mapping = align.previous.load(INPUT / "mapping.npz")
    for arm, (pop, budget) in ARMS.items():
        root = OUT / arm
        root.mkdir(exist_ok=True)
        size = small if budget == "small" else large
        plans[arm] = {"population": pop, "budget": budget, "triplets_per_epoch": size,
            "updates_per_epoch": int(np.ceil(size/4096)),
            "graph": str((INPUT / ("gg_graph.pt" if pop == "g" else "oo_graph.pt")).relative_to(REPO))}
        sample_audit[arm] = {}
        exclusive = mapping["gt_to_odom" if pop == "g" else "odom_to_gt"] < 0
        for epoch in align.old.REFRESH:
            original = native[pop][epoch]
            rows = nested_rows(len(original), large, small, 202610200+epoch)[budget]
            trips = original[rows]
            target = root / f"triplets_epoch{epoch}.npz"
            np.savez_compressed(target, triplets=trips, original_rows=rows)
            paths.add(target)
            a = trips[:, 0]
            sample_audit[arm][str(epoch)] = {
                "native_pool_rows": len(original), "selected_rows": len(trips),
                "unique_anchors": len(np.unique(a)), "exclusive_anchor_rows": int(exclusive[a].sum()),
                "unique_exclusive_anchors": int(exclusive[np.unique(a)].sum()),
                "any_exclusive_scan_rows": int(exclusive[trips].any(1).sum()),
                "sequence_rows": {s: int(np.sum(data[pop]["sequences"][a] == i)) for i, s in enumerate(parent["train"])}}
    paths.update(REPO / p for p in (
        "scripts/_pdv2_native_pairs.py", "scripts/_pdv2_native_pairs_lib.py",
        "scripts/_pdv2_native_pairs_run.py", "tests/gnn/test_native_pairs.py",
        "scripts/_pdv2_pair_train.py", "src/gnn/matching_mining.py"))
    sources.update({str(p.relative_to(REPO)): sha256(p) for p in paths})
    spec = {"created_unix": time.time(), "seed": SEED, "epochs": EPOCHS,
        "train": parent["train"], "val": parent["val"], "plans": plans, "sample_audit": sample_audit,
        "primary": "kiss_large", "test_used": False,
        "design": "GT/KISS 각각의 전체 키프레임에서 기존 규칙으로 삼중쌍 생성. 각 모집단의 작은 예산은 큰 예산의 원 행 부분집합. 2모집단×2예산, 같은 예산의 표본 수·배치 크기·업데이트 수를 정확히 맞춤.",
        "mining": "정답 GT거리<=5m·시차>=30, 같은시퀀스, 앵커당3쌍. 원 특징L2최난도오답 GT거리>10m·시차>=30. 양성표본RNG=100000+갱신epoch. GT4묶음을원표본과배열정확일치확인. KISS고유스캔포함, 방향보강·미래금지추가·새손실없음.",
        "budgets": {"small": small, "large": large, "updates_small": 720, "updates_large": 900},
        "training": "현재표준시간모델·no_interdiff288·게이트초기.0625·±5·2층·128문맥. 같은초기값·epoch별seed·InfoNCE/Adam/AMP/LR20epoch. epoch20고정, 검증선택없음. 작은표본의추가복원반복없음.",
        "validation": "동일KISS검증9·q+15·j<=q-30·100+100후보·실제승자재검색. 기존시간판정과이전OOF문맥가중합계수고정. 새로운계수적합없음. 원래GT전체900회및공통쌍GT/KISS720회참고.",
        "promotion": "주조건kiss_large가gt_large와기존전체GT900각각 평균AP+.001, R100손실<=.001, 각계열AP손실<=.0025일때해당판정조건의추가시드확증후보. 두판정조건별독립보고, 다른학습조건사후선택안함. 이번실험은시험과원고교체없음.",
        "limits": "한시드·반복검증. GT/KISS비교는선택프레임/자세/양성과최난도오답/배치내용/전체그래프통계가함께달라지는운영자료구성효과이며고유앵커만의순수효과아님. 같은예산두모델은행순열만같고실제스캔쌍은다름. 720→900은더많은고유표본과업데이트가함께늘어나는예산효과이며고정표본반복효과아님. GT큰표본은원183249행에서330행줄이고재순서화하므로기존900모델과별도로보고.",
        "sources": sources}
    write_json(OUT / "protocol.json", spec)
    print("PREPARED", {a: (p["triplets_per_epoch"], p["updates_per_epoch"]*20) for a, p in plans.items()}, flush=True)


def train(arm: str) -> None:
    """각 예산에서 같은 초기화·배치 크기·행 순열로 끝까지 학습한다."""
    from gnn.trainer import GNNTrainer
    spec = protocol()
    plan, root = spec["plans"][arm], OUT / arm
    if (root / "training_complete.json").exists():
        assert sha256(root / "checkpoints/final_model.pth") == read(root / "training_complete.json")["checkpoint_sha256"]
        return
    assert not (root / "checkpoints").exists(), "미완료 가중치를 덮어쓰지 않는다"
    data = align.previous.load(INPUT / (plan["population"] + "_inputs.npz"))
    graph = torch.load(REPO / plan["graph"], weights_only=False, map_location="cpu")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    net = align.old.model()
    write_json(root / "initial_model.json", {k: array_hash(v.numpy()) for k, v in net.state_dict().items()})
    trainer = GNNTrainer(net, learning_rate=5e-4, weight_decay=1e-5, temperature=.1,
        checkpoint_dir=str(root / "checkpoints"), use_multi_gpu=False, patience=100, use_amp=True, loss_type="infonce")
    trainer.scaler = torch.amp.GradScaler("cuda", init_scale=128., growth_interval=1000000)
    updates, records, orders = [], [], []
    trainer.optimizer.register_step_post_hook(lambda *args: updates.append(1))

    class FixedMiner:
        """각 키프레임 모집단에서 동결한 기존 규칙 표본을 읽는다."""
        def mine_triplets(self, **kwargs):
            np.testing.assert_array_equal(kwargs["descriptors"], data["raw"])
            epoch = trainer.epoch+1
            assert epoch in align.old.REFRESH
            return np.load(root / f"triplets_epoch{epoch}.npz")["triplets"].tolist()

    original_shuffle, original_epoch = np.random.shuffle, trainer.train_epoch
    def audited_shuffle(values):
        before = np.random.get_state()
        original_shuffle(values)
        if isinstance(values, np.ndarray) and values.shape == (plan["triplets_per_epoch"], 3):
            after = np.random.get_state()
            permutation = np.arange(len(values))
            np.random.set_state(before); original_shuffle(permutation); np.random.set_state(after)
            source_epoch = 1+5*((trainer.epoch)//5)
            original = np.load(root / f"triplets_epoch{source_epoch}.npz")["triplets"]
            np.testing.assert_array_equal(values, original[permutation])
            orders.append({"permutation_sha256": array_hash(permutation), "triplets_sha256": array_hash(values)})

    def audited_epoch(*args, **kwargs):
        epoch = trainer.epoch+1
        seed = SEED*100000+epoch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        value = original_epoch(*args, **kwargs)
        assert len(updates) == epoch*plan["updates_per_epoch"] and len(orders) == epoch
        assert np.isfinite(value) and all(torch.isfinite(v).all() for v in net.state_dict().values())
        records.append({"epoch": epoch, "loss": float(value), "updates": len(updates), **orders[-1]})
        write_json(root / "training_progress.json", {"records": records})
        print("EPOCH", arm, epoch, round(float(value), 6), len(updates), flush=True)
        return value

    np.random.shuffle, trainer.train_epoch = audited_shuffle, audited_epoch
    started = time.time()
    try:
        trainer.train(graph, data["gt"], data["raw"], data["sequences"], val_datasets=None,
            n_epochs=EPOCHS, triplet_miner=FixedMiner(), mine_every_n_epochs=5, n_triplets_per_anchor=3)
    finally:
        np.random.shuffle = original_shuffle
    write_json(root / "training_complete.json", {"runtime_s": time.time()-started, "epochs": EPOCHS,
        "updates": len(updates), "order_hashes": orders,
        "initial_sha256": sha256(root / "initial_model.json"),
        "checkpoint_sha256": sha256(root / "checkpoints/final_model.pth"), "test_used": False})


def validation(arm: str) -> None:
    """해시로 고정한 앞선 평가기를 새 결과 디렉터리에 적용한다."""
    protocol()
    align.OUT = OUT
    align.validation(arm)


def report() -> None:
    """최종 지표와 표본 구성·예산의 교차 차이를 남긴다."""
    spec = protocol()
    earlier = read(INPUT / "summary.json")
    per = {"reference": earlier["per_sequence"]["reference"],
           "common_gt": earlier["per_sequence"]["gg"], "common_kiss": earlier["per_sequence"]["oo"]}
    outputs = {}
    for arm in ARMS:
        per[arm] = {m: {} for m in ("baseline", "context_score")}
        for sequence in spec["val"]:
            p = OUT / arm / "val" / (sequence.replace("/", "_") + ".npz")
            info = read(p.with_suffix(".json"))
            assert sha256(p) == info["sha256"]
            for method in per[arm]:
                per[arm][method][sequence] = info["metrics"][method]
            outputs[str(p.relative_to(REPO))] = sha256(p)
    aggregate = {arm: {m: align.accept.macro(rows) for m, rows in values.items()} for arm, values in per.items()}
    effects, checks = {}, {}
    for method in ("baseline", "context_score"):
        effects[method] = {}
        for name, a, b in (("native_gt_same720", "common_gt", "gt_small"),
                           ("native_kiss_same720", "common_kiss", "kiss_small"),
                           ("gt_budget", "gt_small", "gt_large"), ("kiss_budget", "kiss_small", "kiss_large"),
                           ("input_small", "gt_small", "kiss_small"), ("input_large", "gt_large", "kiss_large"),
                           ("gt_resample900", "reference", "gt_large"), ("primary_vs_reference", "reference", "kiss_large")):
            effects[method][name] = {m: 100*(aggregate[b][method]["macro"][m]-aggregate[a][method]["macro"][m])
                                     for m in align.accept.exp.METRICS}
        now = aggregate["kiss_large"][method]
        checks[method] = {}
        for reference in ("gt_large", "reference"):
            base = aggregate[reference][method]
            checks[method][reference] = {
                "mean_AP": now["macro"]["AP"] >= base["macro"]["AP"]+.001,
                "mean_R100": now["macro"]["R@100P"] >= base["macro"]["R@100P"]-.001,
                "family_AP": all(now["families"][f]["AP"] >= base["families"][f]["AP"]-.0025 for f in base["families"])}
    expansion = {m: all(all(v.values()) for v in values.values()) for m, values in checks.items()}
    write_json(OUT / "summary.json", {"seed": SEED, "epochs": EPOCHS, "aggregate": aggregate,
        "per_sequence": per, "effects_pp": effects, "expansion_checks": checks,
        "eligible_for_three_seed_confirmation": expansion, "outputs": outputs,
        "protocol_sha256": sha256(OUT / "protocol.json"), "test_used": False})
    for arm, scores in aggregate.items():
        print("FINAL", arm, {m: r["macro"] for m, r in scores.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "val", "report"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    if args.action == "prepare": prepare()
    elif args.action == "train": train(args.arm)
    elif args.action == "val": validation(args.arm)
    else: report()
