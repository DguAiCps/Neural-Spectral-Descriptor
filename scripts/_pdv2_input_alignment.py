#!/usr/bin/env python3
"""같은 스캔 쌍으로 자세 입력과 키프레임 선택을 교차 학습한다."""
from __future__ import annotations

import argparse
import concurrent.futures
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
import _pdv2_encoder_place as old
import _pdv2_encoder_place_eval as previous
from _pdv2_candidate_data import train_path
from _pdv2_context_accept_lib import pair_features, unit
from _pdv2_context_place_lib import score
from _pdv2_encoder_place_lib import temporal_subgraph
from _pdv2_input_alignment_lib import common_triplets, make_graph, pose_attributes
from _pdv2_matching_mine import cache_path
from _pdv2_no_context_ablation import evaluate_keys, ledger_path, local_path, sha256, write_json
from _train_attention_neighbors import array_hash
from _xs_eval_lib import normalize

OUT = REPO / "results/pdv2_input_alignment_20261019"
ARMS = {"gg": ("g", "g"), "go": ("g", "o"), "og": ("o", "g"), "oo": ("o", "o")}
EPOCHS, SEED = 20, 1
accept = previous.accept


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def val_paths(sequence: str) -> dict:
    """검증 정답 배열을 읽지 않고 메타데이터에서 경로만 찾는다."""
    sensor, seq = sequence.split("/")
    ledger = ledger_path("val", 1, "pbev128_main", sensor, seq)
    with np.load(ledger) as z:
        meta = json.loads(str(z["meta"]))
    return {**{k: local_path(meta["caches"][k]) for k in ("cache", "layouts", "spec")},
            "metadata_ledger": ledger}


def protocol() -> dict:
    """결과를 보기 전에 고정한 소스·입력을 확인한다."""
    value = read(OUT / "protocol.json")
    for p, h in value["sources"].items():
        assert sha256(REPO / p) == h, p
    return value


def prepare() -> None:
    """네 그래프와 동일 실제 스캔 삼중쌍·20에포크를 동결한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol()
        return
    source = torch.load(old.previous.PREPARED, map_location="cpu", weights_only=False)
    original = torch.load(old.previous.GRAPH, map_location="cpu", weights_only=False)
    reference_graph = temporal_subgraph(original, np.arange(original.num_nodes))
    roster = read(old.OUT / "protocol.json")["train"]
    paths = {old.previous.PREPARED, old.previous.GRAPH, old.focus.base.RVEC,
             old.OUT / "full/checkpoints/final_model.pth", old.OUT / "calibration_freeze.json",
             old.OUT / "summary.json", old.OUT / "completion_validation.json", old.SAMPLES / "audit.json",
             accept.OUT / "protocol.json"}
    collected = {population: {k: [] for k in ("raw", "gt", "odom", "sequences", "scan_ids")}
                 for population in ("g", "o")}
    mappings, audits, g_offset, o_offset = [], {}, 0, 0
    for ordinal, sequence in enumerate(roster):
        gp, op = cache_path(sequence), train_path(sequence)
        fullp = op.with_name(op.stem + "_odometry.npz")
        assert sha256(op) == read(op.with_suffix(".json"))["sha256"]
        g, o, all_poses = previous.load(gp), previous.load(op), previous.load(fullp)
        gi, oi = g["scan_ids"], o["scan_ids"]
        common, ia, ib = np.intersect1d(gi, oi, return_indices=True)
        np.testing.assert_array_equal(g["descriptors"][ia], o["descriptors"][ib])
        np.testing.assert_array_equal(g["poses"][ia], o["poses_gt"][ib])
        np.testing.assert_array_equal(g["poses"], all_poses["gt_poses"][gi])
        np.testing.assert_array_equal(o["poses"], all_poses["odom_poses"][oi])
        mapping = np.full(len(gi), -1, np.int64)
        mapping[ia] = ib + o_offset
        mappings.append(mapping)
        for population, raw, gt, odom, ids in (
            ("g", g["descriptors"], g["poses"], all_poses["odom_poses"][gi], gi),
            ("o", o["descriptors"], o["poses_gt"], o["poses"], oi)):
            values = (raw, gt, odom, np.full(len(ids), ordinal, np.int64), ids)
            for key, value in zip(collected[population], values):
                collected[population][key].append(value)
        audits[sequence] = {"gt_frames": len(gi), "odom_frames": len(oi), "common_frames": len(common),
            "common_fraction_gt": len(common) / len(gi), "common_fraction_odom": len(common) / len(oi),
            "raw_common_max_error": 0., "gt_common_max_error": 0.}
        paths.update((gp, op, fullp, op.with_suffix(".json")))
        g_offset += len(gi)
        o_offset += len(oi)
    data = {p: {k: np.concatenate(v) for k, v in values.items()} for p, values in collected.items()}
    gt_to_odom = np.concatenate(mappings)
    odom_to_gt = np.full(o_offset, -1, np.int64)
    present = np.flatnonzero(gt_to_odom >= 0)
    odom_to_gt[gt_to_odom[present]] = present
    np.testing.assert_array_equal(data["g"]["raw"], source["descriptors"])
    np.testing.assert_array_equal(data["g"]["gt"], source["poses"])
    np.testing.assert_array_equal(data["g"]["sequences"], source["sequences"])
    for population in data:
        np.savez_compressed(OUT / (population + "_inputs.npz"), **data[population])
        paths.add(OUT / (population + "_inputs.npz"))
    np.savez_compressed(OUT / "mapping.npz", gt_to_odom=gt_to_odom, odom_to_gt=odom_to_gt)
    paths.add(OUT / "mapping.npz")
    graphs = {}
    for arm, (population, pose_kind) in ARMS.items():
        d = data[population]
        graph = make_graph(d["raw"], d["gt" if pose_kind == "g" else "odom"], d["sequences"])
        if arm == "gg":
            np.testing.assert_array_equal(graph.edge_index.numpy(), reference_graph.edge_index.numpy())
            np.testing.assert_allclose(graph.edge_attr.numpy(), reference_graph.edge_attr.numpy(), atol=1e-7, rtol=0)
            graph = reference_graph
        p = OUT / (arm + "_graph.pt")
        torch.save(graph, p)
        paths.add(p)
        graphs[arm] = {"nodes": graph.num_nodes, "edges": graph.edge_index.shape[1],
                       "hashes": {k: array_hash(getattr(graph, k).numpy()) for k in ("x", "edge_index", "edge_attr", "edge_type")}}
    retained = {}
    for epoch in old.REFRESH:
        p = old.SAMPLES / f"original_epoch{epoch}.npz"
        rows = np.load(p)["triplets"]
        valid, _ = common_triplets(rows, gt_to_odom)
        retained[epoch] = (rows, valid)
        paths.add(p)
    count = min(len(v[1]) for v in retained.values())
    sample_audit = {}
    for epoch, (rows, valid) in retained.items():
        chosen = np.random.default_rng(202610190 + epoch).choice(valid, count, replace=False)
        canonical = rows[chosen]
        gt_trips, odom_trips = canonical, gt_to_odom[canonical]
        np.testing.assert_array_equal(odom_to_gt[odom_trips], gt_trips)
        a, p, n = canonical.T
        assert np.all(data["g"]["sequences"][a] == data["g"]["sequences"][p])
        assert np.all(data["g"]["sequences"][a] == data["g"]["sequences"][n])
        assert np.all(np.linalg.norm(data["g"]["gt"][a, :3, 3] - data["g"]["gt"][p, :3, 3], axis=1) <= 5.)
        assert np.all(np.linalg.norm(data["g"]["gt"][a, :3, 3] - data["g"]["gt"][n, :3, 3], axis=1) > 10.)
        target = OUT / f"triplets_epoch{epoch}.npz"
        np.savez_compressed(target, g=gt_trips, o=odom_trips, original_rows=chosen)
        paths.add(target)
        sample_audit[str(epoch)] = {"original": len(rows), "eligible_common": len(valid), "selected": count,
            "anchor_counts": {s: int(np.sum(data["g"]["sequences"][a] == i)) for i, s in enumerate(roster)}}
    val = [s + "/" + q for s, q in accept.VAL]
    for sequence in val:
        paths.update(val_paths(sequence).values())
        paths.update((previous.feature_path("full", sequence, "val"),
                      old.OUT / "full/val_keys" / (sequence.replace("/", "_") + ".npz")))
    paths.update(REPO / p for p in (
        "scripts/_pdv2_input_alignment.py", "scripts/_pdv2_input_alignment_lib.py",
        "scripts/_pdv2_input_alignment_run.py", "tests/gnn/test_input_alignment.py",
        "scripts/_pdv2_encoder_place.py", "scripts/_pdv2_encoder_place_eval.py",
        "scripts/_pdv2_context_accept_lib.py", "scripts/_pdv2_context_place_lib.py",
        "scripts/_pdv2_no_context_ablation.py", "scripts/_pr_eval_lib.py",
        "src/gnn/model.py", "src/gnn/trainer.py", "src/keyframe/graph_manager.py"))
    write_json(OUT / "protocol.json", {"created_unix": time.time(), "seed": SEED, "epochs": EPOCHS,
        "train": roster, "val": val, "arms": ARMS, "primary": "oo", "test_used": False,
        "graphs": graphs, "population_audit": audits, "samples": sample_audit,
        "triplets_per_epoch": count, "updates_per_epoch": int(np.ceil(count / 4096)),
        "training": "처음부터같은초기값·표준attention2층·시간±5·no_interdiff288·gate초기.0625. "
                    "기존InfoNCE/Adamlr5e-4/wd1e-5/온도.1/20epoch/AMPscale128. "
                    "네조건같은실제스캔삼중쌍·순서·업데이트. 공통원행을비복원선택,추가반복없음. "
                    "GT는라벨에만공통사용,go/oo그래프속성에는KISS만사용. BatchNorm·전체그래프노드구성차이는프레임선택효과에포함.",
        "diagnostic": "기존full20모델을검증KISS동일스캔에서 GT시간간선속성으로도추론. "
                      "검색을다시하고최종판정의거리경로는KISS로고정. 이GT입력은oracle진단이며배포후보아님.",
        "validation": "네재학습조건모두동일KISS키프레임·KISS간선속성·q+15·j<=q-30·100+100. "
                      "기존시간판정과기존OOF문맥가중합계수고정두점수. 문맥가중합은새모델재적합이아닌고정계수전이진단. "
                      "현재full20원장도참고. 에포크20고정·검증체크포인트선택없음.",
        "promotion": "oo가기존시간판정에서gg와현재full20각각 AP+.001,R100손실<=.001,각계열AP손실<=.0025이면추가시드대상. "
                     "고정문맥가중합은같은점수끼리별도기준보고. oracleGT는선택불가. 이번은1시드·시험평가없음.",
        "limits": "입력정합예비. 원전체삼중쌍보다공통표본이줄어기존full모델과학습량차이있음. "
                  "네조건은쌍과예산동일하나선택프레임·전체그래프BN/드롭아웃모집단다름. 같은시드가같은노드별드롭아웃을보장하지않음. "
                  "기존재계량·융합·문맥수락계수는고정. 반복사용검증이며새독립확증아님.",
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)}})
    print("PREPARED", count, "triplets", 20 * int(np.ceil(count / 4096)), "updates", flush=True)


def train(arm: str) -> None:
    """동일 실제 스캔 표본·초기화·업데이트 수로 고정 마지막 에포크까지 학습한다."""
    from gnn.trainer import GNNTrainer
    spec = protocol()
    root = OUT / arm
    root.mkdir(exist_ok=True)
    if (root / "training_complete.json").exists():
        assert sha256(root / "checkpoints/final_model.pth") == read(root / "training_complete.json")["checkpoint_sha256"]
        return
    assert not (root / "checkpoints").exists(), "미완료 체크포인트를 덮어쓰지 않는다"
    population = ARMS[arm][0]
    data = previous.load(OUT / (population + "_inputs.npz"))
    mapping = previous.load(OUT / "mapping.npz")
    canonical = np.arange(len(data["raw"])) if population == "g" else mapping["odom_to_gt"]
    graph = torch.load(OUT / (arm + "_graph.pt"), map_location="cpu", weights_only=False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    net = old.model()
    write_json(root / "initial_model.json", {k: array_hash(v.numpy()) for k, v in net.state_dict().items()})
    trainer = GNNTrainer(net, learning_rate=5e-4, weight_decay=1e-5, temperature=.1,
                        checkpoint_dir=str(root / "checkpoints"), use_multi_gpu=False,
                        patience=100, use_amp=True, loss_type="infonce")
    trainer.scaler = torch.amp.GradScaler("cuda", init_scale=128., growth_interval=1000000)
    updates, records, orders = [], [], []
    trainer.optimizer.register_step_post_hook(lambda *args: updates.append(1))

    class FixedMiner:
        """실제 스캔 기준으로 같은 표본을 반환한다."""
        def mine_triplets(self, **kwargs):
            np.testing.assert_array_equal(kwargs["descriptors"], data["raw"])
            epoch = trainer.epoch + 1
            assert epoch in old.REFRESH
            return np.load(OUT / f"triplets_epoch{epoch}.npz")[population].tolist()

    original_shuffle = np.random.shuffle
    def audited_shuffle(value):
        original_shuffle(value)
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 3:
            assert len(value) == spec["triplets_per_epoch"]
            assert np.all(canonical[value] >= 0)
            orders.append(array_hash(canonical[value]))
    np.random.shuffle = audited_shuffle
    original_epoch = trainer.train_epoch
    def audited_epoch(*args, **kwargs):
        epoch = trainer.epoch + 1
        seed = SEED * 100000 + epoch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        loss = original_epoch(*args, **kwargs)
        assert len(updates) == epoch * spec["updates_per_epoch"] and len(orders) == epoch
        assert np.isfinite(loss) and all(torch.isfinite(v).all() for v in net.state_dict().values())
        records.append({"epoch": epoch, "loss": float(loss), "updates": len(updates), "order_sha256": orders[-1]})
        write_json(root / "training_progress.json", {"records": records})
        print("EPOCH", arm, epoch, round(float(loss), 6), len(updates), flush=True)
        return loss
    trainer.train_epoch = audited_epoch
    started = time.time()
    try:
        trainer.train(graph, data["gt"], data["raw"], data["sequences"], val_datasets=None,
                      n_epochs=EPOCHS, triplet_miner=FixedMiner(), mine_every_n_epochs=5, n_triplets_per_anchor=3)
    finally:
        np.random.shuffle = original_shuffle
    write_json(root / "training_complete.json", {"epochs": EPOCHS, "updates": len(updates),
        "checkpoint_sha256": sha256(root / "checkpoints/final_model.pth"),
        "initial_sha256": sha256(root / "initial_model.json"), "order_hashes": orders,
        "runtime_s": time.time() - started, "test_used": False})


def load_net(arm: str):
    """학습 때 없는 고정 재계량을 현재 추론과 같은 위치에 적용한다."""
    checkpoint = old.OUT / "full/checkpoints/final_model.pth" if arm == "oracle_gt" else OUT / arm / "checkpoints/final_model.pth"
    net = old.model().cuda().eval()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    net.gnn.key_scale = torch.nn.Parameter(torch.from_numpy(np.load(old.focus.base.RVEC)).float().cuda(), requires_grad=False)
    return net


def generate_keys(arm: str, sequence: str, net) -> None:
    """같은 검증 스캔에서 문맥 키와 q+15 접두 대조를 저장한다."""
    root = OUT / arm / "keys"
    root.mkdir(parents=True, exist_ok=True)
    target = root / (sequence.replace("/", "_") + ".npz")
    if target.exists():
        assert sha256(target) == read(target.with_suffix(".json"))["sha256"]
        return
    cache = previous.load(val_paths(sequence)["cache"])
    raw = cache["descriptors"].astype(np.float32)
    poses = cache["poses_gt"] if arm == "oracle_gt" else cache["poses"]
    graph = make_graph(raw, poses, np.zeros(len(raw), np.int64)).cuda()
    with torch.no_grad():
        key = normalize(net(graph).float().cpu().numpy())
    prefix = []
    for q in (len(raw) // 3, 2 * len(raw) // 3):
        end = q + 16
        small = make_graph(raw[:end], poses[:end], np.zeros(end, np.int64)).cuda()
        with torch.no_grad():
            one = normalize(net(small).float().cpu().numpy())[q]
        np.testing.assert_allclose(one, key[q], atol=3e-6, rtol=3e-5)
        prefix.append({"query": q, "max_error": float(abs(one - key[q]).max())})
        del small
    reference = previous.load(old.OUT / "full/val_keys" / target.name)["embedding"]
    diagnostics = {}
    if arm == "oracle_gt":
        original_graph = make_graph(raw, cache["poses"], np.zeros(len(raw), np.int64)).cuda()
        with torch.no_grad():
            original_key = normalize(net(original_graph).float().cpu().numpy())
        np.testing.assert_allclose(original_key, reference, atol=3e-6, rtol=3e-5)
        diagnostics["reconstruction_max_error"] = float(abs(original_key - reference).max())
        delta = np.linalg.norm(unit(key[:, 288:]) - unit(reference[:, 288:]), axis=1)
        diagnostics["unit_context_l2_quantiles"] = np.quantile(delta, [.5, .9, .99, 1.]).tolist()
        del original_graph
    np.testing.assert_allclose(normalize(key[:, :288]), normalize(raw * np.load(old.focus.base.RVEC)[None]), atol=3e-7)
    assert key.shape == (len(raw), 416) and np.isfinite(key).all()
    np.savez_compressed(target, embedding=key, scan_ids=cache["scan_ids"])
    write_json(target.with_suffix(".json"), {"sha256": sha256(target), "prefix": prefix, "diagnostics": diagnostics})
    del graph
    torch.cuda.empty_cache()
    print("KEYS", arm, sequence, flush=True)


def evaluate(arm: str, sequence: str) -> None:
    """실제 후보 재검색·최종 승자·두 고정 판정 점수를 계산한다."""
    root = OUT / arm / "val"
    root.mkdir(parents=True, exist_ok=True)
    target = root / (sequence.replace("/", "_") + ".npz")
    if target.exists():
        assert sha256(target) == read(target.with_suffix(".json"))["sha256"]
        return
    paths = val_paths(sequence)
    cache = previous.load(paths["cache"])
    key = previous.load(OUT / arm / "keys" / target.name)["embedding"]
    ref = previous.load(previous.feature_path("full", sequence, "val"))
    min_past = np.full(len(key), np.nan)
    min_past[ref["query_idx"]] = ref["min_past_gt_dist"]
    selected = read(accept.OUT / "protocol.json")["seeds"]["1"]
    conf = selected["confidence"]
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        bank = old.focus.base.bank_for("pbev128_main", z)
    ledger = evaluate_keys(key, bank, cache["poses_gt"][:, :3, 3], min_past, {1: tuple(conf["tuple"])})[1]
    use = ledger["query_idx"] < len(key) - 15
    ledger = {f: v[use] for f, v in ledger.items()}
    for f in ("query_idx", "min_past_gt_dist"):
        np.testing.assert_array_equal(ledger[f], ref[f])
    # GT 입력 진단에서도 최종 거리 대응은 같은 KISS 경로로 고정한다.
    path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(cache["poses"][:, :3, 3], axis=0), axis=1))]
    parts = []
    for start in range(0, len(ledger["query_idx"]), 256):
        sl = slice(start, start + 256)
        parts.append(pair_features(key, bank, path, ledger["query_idx"][sl], ledger["j_star"][sl],
                                   conf["decision"], ledger[conf["decision"]][sl]))
    row = {f: np.concatenate([p[f] for p in parts]) for f in parts[0]}
    row.update({f: ledger[f] for f in accept.FIELDS})
    fitted = read(old.OUT / "calibration_freeze.json")["models"]["crossfit"]
    row["context_score"] = score(row, fitted, 1.)
    np.savez_compressed(target, **row)
    write_json(target.with_suffix(".json"), {"sha256": sha256(target), "rows": len(row["query_idx"]),
        "metrics": {f: accept.exp.metrics(row[f], row) for f in ("baseline", "context_score")},
        "winner_changes_from_reference": int(np.sum(row["j_star"] != ref["j_star"]))})
    print("FINAL_SEQUENCE", arm, sequence, flush=True)


def validation(arm: str) -> None:
    """네 학습 조건과 자세만 바꾼 진단을 같은 검증 규약으로 평가한다."""
    spec = protocol()
    net = load_net(arm)
    for sequence in spec["val"]:
        generate_keys(arm, sequence, net)
    del net
    torch.cuda.empty_cache()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda sequence: evaluate(arm, sequence), spec["val"]))


def report() -> None:
    """전체 최종 지표·계열별 손익·2×2 학습 효과·확장 기준을 기록한다."""
    spec = protocol()
    per = {arm: {f: {} for f in ("baseline", "context_score")} for arm in ("reference", "oracle_gt", *ARMS)}
    outputs = {}
    for sequence in spec["val"]:
        ref = previous.load(previous.feature_path("full", sequence, "val"))
        for f, column in (("baseline", "baseline"), ("context_score", "crossfit")):
            per["reference"][f][sequence] = accept.exp.metrics(ref[column], ref)
        for arm in ("oracle_gt", *ARMS):
            p = OUT / arm / "val" / (sequence.replace("/", "_") + ".npz")
            record = read(p.with_suffix(".json"))
            assert sha256(p) == record["sha256"]
            for f in per[arm]:
                per[arm][f][sequence] = record["metrics"][f]
            outputs[str(p.relative_to(REPO))] = sha256(p)
    aggregate = {arm: {f: accept.macro(rows) for f, rows in scores.items()} for arm, scores in per.items()}
    checks = {}
    for method in ("baseline", "context_score"):
        now = aggregate["oo"][method]
        checks[method] = {}
        for reference in ("gg", "reference"):
            base = aggregate[reference][method]
            checks[method][reference] = {
                "mean_AP": now["macro"]["AP"] >= base["macro"]["AP"] + .001,
                "mean_R100": now["macro"]["R@100P"] >= base["macro"]["R@100P"] - .001,
                "family_AP": all(now["families"][f]["AP"] >= base["families"][f]["AP"] - .0025 for f in base["families"])}
    effects = {}
    for method in ("baseline", "context_score"):
        effects[method] = {name: {m: 100 * (aggregate[b][method]["macro"][m] - aggregate[a][method]["macro"][m])
                                for m in accept.exp.METRICS}
                           for name, a, b in (("pose_with_gt_frames", "gg", "go"),
                                             ("pose_with_odom_frames", "og", "oo"),
                                             ("frames_with_gt_pose", "gg", "og"),
                                             ("frames_with_odom_pose", "go", "oo"),
                                             ("both", "gg", "oo"), ("oracle_pose", "reference", "oracle_gt"))}
    write_json(OUT / "summary.json", {"seed": SEED, "epochs": EPOCHS, "aggregate": aggregate,
        "per_sequence": per, "effects_pp": effects, "expansion_checks": checks,
        "eligible_for_three_seed_confirmation": all(all(v.values()) for v in checks["baseline"].values()),
        "context_transfer_pass": all(all(v.values()) for v in checks["context_score"].values()),
        "outputs": outputs, "protocol_sha256": sha256(OUT / "protocol.json"), "test_used": False})
    for arm, scores in aggregate.items():
        print("FINAL_VALIDATION", arm, {f: r["macro"] for f, r in scores.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "val", "report"))
    parser.add_argument("--arm", choices=(*ARMS, "oracle_gt"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logging.basicConfig(level=logging.WARNING)
    if args.action == "prepare": prepare()
    elif args.action == "train": train(args.arm)
    elif args.action == "val": validation(args.arm)
    else: report()
