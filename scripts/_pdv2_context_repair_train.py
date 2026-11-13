#!/usr/bin/env python3
"""방향 균형과 유사 간선 제거 학습을 현재 대칭 모델에서 교차 비교한다."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_context_repair as frozen
from _pdv2_no_context_ablation import evaluate_keys, sha256, write_json
from gnn.pair_sampling import past_heading, positive_pools, sample_triplets, same_place_mask, masked_infonce

base = frozen.base
CONFIG = REPO / "configs/pdv2_context_repair_training.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
OUT = REPO / "results/pdv2_context_repair_training_20261016"
ARMS = tuple(CFG["arms"])
TRAIN, VAL, OPS = base.TRAIN, base.VAL, base.OPS


def protocol(data: bool = False) -> dict:
    """학습·평가 코드와 입력을 동결된 해시로 검사한다."""
    p = base.read(OUT / "protocol.json")
    for name, h in p["runtime_sha256"].items():
        assert sha256(REPO / name) == h, name
    if data:
        for name, h in p["input_sha256"].items():
            assert sha256(REPO / name) == h, name
    return p


def node_dropout_mask(edge_type: np.ndarray, target: np.ndarray, n: int,
                      probability: float, seed: int) -> np.ndarray:
    """무작위 노드의 유사 간선만 모두 제거하며 시간 간선은 보존한다."""
    assert 0 <= probability <= 1
    drop = np.random.default_rng(seed).random(n) < probability
    return (edge_type == 0) | ~drop[target]


def prepare() -> None:
    """학습 8개만으로 입력·정답 쌍·오답을 준비하고 비교를 고정한다."""
    from torch_geometric.data import Batch
    from _lag_branch_fusion import lag_pass1_graph
    from _pdv2_pair_train import hard_negatives
    from build_and_train_edge_classifier import range_phase_flat
    from _xs_eval_lib import normalize
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    frozen.protocol(True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg, _, _, _, rvec, _ = base.load_model("main")
    assert CFG["encoder_preset"] == "no_interdiff" and CFG["use_gated_context"]
    assert cfg["gnn"]["input_dim"] == 288
    records, graphs, positions, desc, seqs, frames = [], [], [], [], [], []
    inputs = {REPO / CFG["initial_checkpoint"], frozen.OUT / "protocol.json"}
    offset = 0
    for seqid, (s, q) in enumerate(TRAIN):
        sequence = s + "/" + q
        cache, layout, _, gt, _, paths = frozen.parent.rel.load_data("train", sequence)
        raw = cache["descriptors"].astype(np.float32)
        first = lag_pass1_graph(cache, raw, cfg, "cpu", s)
        temporal_cfg = copy.deepcopy(cfg)
        temporal_cfg["keyframe"]["graph"]["similarity_threshold"] = 2.
        temporal = lag_pass1_graph(cache, raw, temporal_cfg, "cpu", s)
        assert (temporal.edge_type == 0).all()
        frame = 100 if sequence in CFG["shared_coordinate_groups"][0] else seqid
        records.append({"sequence": sequence, "offset": offset, "first_graph": first,
                        "poses": cache["poses"], "d_rm": normalize(raw * rvec[None]),
                        "phase": range_phase_flat(layout)})
        graphs.append(temporal)
        positions.append(gt[:, :3, 3])
        desc.append(raw)
        seqs.extend([seqid] * len(raw))
        frames.extend([frame] * len(raw))
        inputs.update(paths.values())
        offset += len(raw)
    positions, desc = np.concatenate(positions), np.concatenate(desc)
    seqs, frames = np.asarray(seqs), np.asarray(frames)
    intra, opposite = positive_pools(positions, seqs, frames)
    heading, valid = past_heading(positions, seqs)
    negative, negative_checks = hard_negatives(desc, positions, seqs)
    graph = Batch.from_data_list(graphs)
    state = {"graph": graph, "records": records, "positions": positions,
             "sequences": seqs, "frames": frames, "intra": intra, "opposite": opposite,
             "negative": negative, "heading": heading, "heading_valid": valid}
    torch.save(state, OUT / "prepared.pt")
    samples = OUT / "samples"
    samples.mkdir(exist_ok=True)
    counts = {}
    for policy, fraction in [("control", 0.), ("opposite", CFG["opposite_fraction"])]:
        counts[policy] = []
        for epoch in range(CFG["epochs"]):
            trips = sample_triplets(intra, opposite, negative, 7000 + epoch, fraction)
            a, p, n = trips.T
            assert np.all(frames[a] == frames[p])
            assert np.all(np.linalg.norm(positions[a] - positions[p], axis=1) <= 5)
            assert np.all(np.linalg.norm(positions[a] - positions[n], axis=1) > 10)
            isopp = valid[a] & valid[p] & ((heading[a] * heading[p]).sum(1) <= -.5)
            counts[policy].append({"epoch": epoch + 1, "triplets": len(trips),
                "opposite": int(isopp.sum()), "unique_opposite_pairs": len(np.unique(trips[isopp, :2], axis=0)),
                "cross_sequence": int((seqs[a] != seqs[p]).sum())})
            path = samples / f"{policy}_{epoch + 1}.npz"
            np.savez_compressed(path, triplets=trips)
            inputs.add(path)
    assert [r["triplets"] for r in counts["control"]] == [r["triplets"] for r in counts["opposite"]]
    inputs.add(OUT / "prepared.pt")
    parent = frozen.protocol()
    runtime = {REPO / x for x in parent["runtime_sha256"]}
    runtime.update((Path(__file__), CONFIG, REPO / "scripts/_pdv2_context_repair_train_run.py",
                    REPO / "tests/gnn/test_context_repair_training.py", REPO / "scripts/_pdv2_pair_train.py",
                    REPO / "src/gnn/pair_sampling.py"))
    inputs.add(base.OUT / "classifiers/xyz.pt")
    write_json(OUT / "protocol.json", {"created_unix": time.time(), "config": CFG,
        "scope": "현재 대칭시드1에서 학습8·KISS입력으로6에포크추가학습. 단일시드 검증예비실험. 새로운 장소나 모듈 없음.",
        "factorial": "control/opposite/dropout/combined는 역방향쌍25%교체 여부와 노드별 유사간선50%제거학습 여부의2x2. 같은초기화·표본수·배치순서·업데이트수·학습률. 제거는 정답을 읽지 않는 노드균등 무작위이며 시간간선은 유지한다.",
        "pairs": "실제 다른방향의 같은장소쌍만 사용한다. 공통좌표가 검증된 Bridge세션 사이 양성을 허용한다. 반복추출이며 역재생을 다른방향 관측으로 간주하지 않는다. 모든조건에서 같은장소 배치오답을 마스킹한다.",
        "graph": "매에포크 시작에 현재모델의첫통과·과거후보30·A분류기상위10·실제속성을 갱신한다. 에포크안에서는 그래프고정이므로 매업데이트 재생성과 동일하지 않다. 재계량은 기존값고정. 유사간선선택에는 GT를 쓰지 않는다.",
        "evaluation": "마지막6에포크고정. 각체크포인트를 A간선/시간간선만으로 각각최종평가. 첫통과는같고 둘째통과 유사간선만제거. 융합·수락·q+15·j<=q-30·100+100고정. 데이터셋별설정선택·시험 없음.",
        "samples": counts, "nodes": offset, "negative_checks": negative_checks,
        "opposite_anchors": sum(bool(len(x)) for x in opposite), "test_used": False,
        "runtime_sha256": {str(x.relative_to(REPO)): sha256(x) for x in sorted(runtime)},
        "input_sha256": {str(x.relative_to(REPO)): sha256(x) for x in sorted(inputs)}})
    print("PREPARED", offset, counts, flush=True)


def refresh(model, graph, records: list, classifier, blob: dict) -> dict:
    """현재 가중치와 배포 규칙으로 학습 그래프의 유사 간선을 갱신한다."""
    from _causal_edge_features import causal_edge_features
    from _causal_branch_fusion import select_edges_per_node
    model.eval()
    indices, attrs = [], []
    tick = time.monotonic()
    with torch.no_grad():
        for record in records:
            g = record["first_graph"].to("cuda")
            emb = model(g).float().cpu().numpy()
            record["first_graph"] = g.cpu()
            q, j, x, unused = causal_edge_features(emb, record["d_rm"], record["phase"], record["poses"], want_labels=False)
            assert unused is None
            chunks = []
            for start in range(0, len(x), 65536):
                t = torch.from_numpy((x[start:start + 65536] - blob["mu"]) / blob["sd"]).float().cuda()
                chunks.append(torch.sigmoid(classifier(t)).cpu().numpy())
            prob = np.concatenate(chunks)
            ids = select_edges_per_node(q, prob, 10)
            qs, js = q[ids], j[ids]
            assert (js <= qs - 30).all()
            a = np.stack([np.zeros(len(ids)), np.zeros(len(ids)), x[ids, 0],
                          np.log1p(np.linalg.norm(emb[qs] - emb[js], axis=1)) / 5., prob[ids]], axis=1).astype(np.float32)
            indices.append(np.stack([js, qs]) + record["offset"])
            attrs.append(a)
    temporal = graph.edge_type == 0
    temporal_index, temporal_attr = graph.edge_index[:, temporal], graph.edge_attr[temporal]
    index = torch.from_numpy(np.concatenate(indices, axis=1)).cuda()
    attr = torch.from_numpy(np.concatenate(attrs)).cuda()
    graph.edge_index = torch.cat([temporal_index, index], dim=1)
    graph.edge_attr = torch.cat([temporal_attr, attr])
    graph.edge_type = torch.cat([torch.zeros(len(temporal_attr), dtype=torch.long, device="cuda"),
                                 torch.ones(len(attr), dtype=torch.long, device="cuda")])
    return {"seconds": time.monotonic() - tick, "similarity_edges": len(attr),
            "mean_probability": float(attr[:, 4].mean())}


def train(arm: str) -> None:
    """같은 초기화에서 마지막 에포크까지 고정 예산으로 학습한다."""
    protocol(True)
    assert arm in ARMS
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    if (folder / "complete.json").exists():
        return
    torch.set_num_threads(2)
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    state = torch.load(OUT / "prepared.pt", map_location="cpu", weights_only=False)
    graph = state["graph"].to("cuda")
    _, model, *_ = base.load_model("main")
    model.gnn.key_scale.requires_grad_(False)
    classifier, blob = base.load_classifier("xyz", "cuda")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    positions = torch.as_tensor(state["positions"], device="cuda")
    frames = torch.as_tensor(state["frames"], device="cuda")
    bsize, accum = CFG["batch_size"], CFG["accumulation_steps"]
    policy = "opposite" if arm in ("opposite", "combined") else "control"
    dropout = arm in ("dropout", "combined")
    updates, history = 0, []
    started = time.monotonic()
    write_json(folder / "initial.json", {"checkpoint_sha256": sha256(REPO / CFG["initial_checkpoint"]),
               "parameter_count": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    for epoch in range(CFG["epochs"]):
        tick = time.monotonic()
        refreshed = refresh(model, graph, state["records"], classifier, blob)
        all_index, all_attr, all_type = graph.edge_index, graph.edge_attr, graph.edge_type
        types, targets = all_type.cpu().numpy(), all_index[1].cpu().numpy()
        with np.load(OUT / "samples" / f"{policy}_{epoch + 1}.npz") as z:
            trips = z["triplets"]
        trips = trips[np.random.default_rng(8000 + epoch).permutation(len(trips))]
        n_batches = (len(trips) + bsize - 1) // bsize
        model.train()
        losses, edge_counts = [], []
        for start in range(0, n_batches, accum):
            stop = min(start + accum, n_batches)
            if dropout:
                mask = node_dropout_mask(types, targets, len(positions), CFG["similarity_node_dropout"], 9000 + updates)
                mask = torch.from_numpy(mask).cuda()
                graph.edge_index, graph.edge_attr, graph.edge_type = all_index[:, mask], all_attr[mask], all_type[mask]
            edge_counts.append(int((graph.edge_type == 1).sum()))
            optimizer.zero_grad(set_to_none=True)
            # BF16은 FP16 스케일러의 스텝 생략 없이 같은 업데이트 수를 유지한다.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb = model(graph)
            loss = torch.zeros((), device="cuda")
            for batch in range(start, stop):
                a, p, n = trips[batch * bsize:(batch + 1) * bsize].T
                mask = same_place_mask(positions[a], positions[p], frames[a], frames[p])
                term = masked_infonce(emb[a].float(), emb[p].float(), emb[n].float(), mask, CFG["temperature"])
                loss = loss + term / (stop - start)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
        graph.edge_index, graph.edge_attr, graph.edge_type = all_index, all_attr, all_type
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "updates": updates,
                        "seconds": time.monotonic() - tick, "refresh": refreshed,
                        "mean_training_similarity_edges": float(np.mean(edge_counts))})
        write_json(folder / "progress.json", {"arm": arm, "epochs": history, "status": "running"})
        print("EPOCH", arm, history[-1], flush=True)
    path = folder / "final.pth"
    torch.save({"epoch": CFG["epochs"] - 1, "model_state_dict": model.state_dict(),
                "updates": updates, "arm": arm, "protocol_sha256": sha256(OUT / "protocol.json")}, path)
    write_json(folder / "complete.json", {"arm": arm, "epochs": history, "updates": updates,
               "checkpoint_sha256": sha256(path), "runtime_s": time.monotonic() - started})


def keys(arm: str, sequence: str) -> None:
    """추가 학습 모델을 전체 간선과 시간 간선 전용으로 각각 추론한다."""
    protocol()
    assert arm in ARMS
    s, q = sequence.split("/")
    assert (s, q) in VAL
    target = OUT / arm / f"keys_{s}_{q}.json"
    if target.exists():
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache, layout, _, _, _, _ = frozen.parent.rel.load_data("val", sequence)
    bundle = base.load_model("main")
    checkpoint = OUT / arm / "final.pth"
    assert sha256(checkpoint) == base.read(OUT / arm / "complete.json")["checkpoint_sha256"]
    bundle[1].load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    bundle[1].eval()
    _, graph, c = base.capture(cache, s, layout, bundle, "cuda")
    classifier = base.load_classifier("xyz", "cuda")
    emb, probability, ids = base.apply_classifier(bundle[1], graph, c, classifier)
    temporal = frozen.infer(bundle[1], base.replace_edges(graph, c, probability, np.empty(0, np.int64)))
    metadata = {}
    for mode, value in [("all", emb), ("temporal", temporal)]:
        folder = OUT / (arm + "_" + mode)
        folder.mkdir(exist_ok=True)
        path = folder / f"keys_{s}_{q}.npz"
        np.savez_compressed(path, embedding=value, scan_ids=cache["scan_ids"])
        metadata[mode] = sha256(path)
    write_json(target, {"sequence": sequence, "arm": arm, "key_sha256": metadata, "selected_edges": len(ids)})
    print("KEYS", arm, sequence, flush=True)


def score(arm: str, sequence: str) -> None:
    """추가 학습 조건의 최종 AP·F1·R@100P·R@1을 계산한다."""
    protocol()
    s, q, paths, min_past = base.old.alignment.base.inputs("val", sequence)
    folder = OUT / arm
    if (folder / f"val_{s}_{q}.json").exists():
        return
    with np.load(paths["cache"]) as z:
        gt, scan_ids = z["poses_gt"][:, :3, 3], z["scan_ids"]
    with np.load(folder / f"keys_{s}_{q}.npz") as z:
        emb = z["embedding"]
        np.testing.assert_array_equal(scan_ids, z["scan_ids"])
    selection = base.read(base.old.BASE / "s1/confidence_selection.json")["ops"]
    result = {}
    for op in OPS:
        with np.load(paths["spec"]) as z:
            bank = base.old.alignment.base.bank_for(op, z)
        sel = selection[op]
        a = evaluate_keys(emb, bank, gt, min_past, {1: tuple(float(w) for w in sel["tuple"])})[1]
        keep = a["query_idx"] < len(gt) - 15
        a = {k: v[keep] for k, v in a.items()}
        ref, _ = base.old.read_ledger(frozen.root("A") / f"val_{op}_{s}_{q}.npz")
        for k in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(a[k], ref[k])
        a["group"] = ref["group"]
        path = folder / f"val_{op}_{s}_{q}.npz"
        np.savez_compressed(path, **a, meta=json.dumps({"protocol_sha256": sha256(OUT / "protocol.json")}))
        result[op] = {"metrics": base.old.sequence_metrics(a, sel["decision"], sel["tau_star"]),
                      "ledger": str(path.relative_to(REPO)), "sha256": sha256(path)}
    write_json(folder / f"val_{s}_{q}.json", {"arm": arm, "sequence": sequence, "ops": result})
    print("SCORE", arm, sequence, flush=True)


def report() -> None:
    """예산 일치와 최종 원장을 검증하고 모든 조건을 보고한다."""
    protocol(True)
    completed = [base.read(OUT / a / "complete.json") for a in ARMS]
    assert len({x["updates"] for x in completed}) == 1
    scores, direction = {}, {}
    selection = base.read(base.old.BASE / "s1/confidence_selection.json")["ops"]
    nledgers = nrows = 0
    for arm in ARMS:
        for mode in ("all", "temporal"):
            label = arm + "_" + mode
            folder = OUT / label
            records = {s + "/" + q: base.read(folder / f"val_{s}_{q}.json") for s, q in VAL}
            scores[label] = {op: base.old.aggregate_rows(records, "val", op) for op in OPS}
            direction[label] = {}
            for op in OPS:
                groups = {g: {"queries": 0, "correct": 0} for g in ("same", "opposite", "positive")}
                for s, q in VAL:
                    meta = records[s + "/" + q]["ops"][op]
                    path = REPO / meta["ledger"]
                    assert sha256(path) == meta["sha256"]
                    a, _ = base.old.read_ledger(path)
                    sel = selection[op]
                    assert base.old.sequence_metrics(a, sel["decision"], sel["tau_star"]) == meta["metrics"]
                    ref, _ = base.old.read_ledger(frozen.root("A") / f"val_{op}_{s}_{q}.npz")
                    for k in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(a[k], ref[k])
                    assert (a["j_star"] <= a["query_idx"] - 30).all()
                    _, _, paths, _ = base.old.alignment.base.inputs("val", s + "/" + q)
                    with np.load(paths["cache"]) as z:
                        gt = z["poses_gt"][:, :3, 3]
                    np.testing.assert_allclose(a["top1_geo_dist"], np.linalg.norm(gt[a["query_idx"]] - gt[a["j_star"]], axis=1), atol=1e-5, rtol=1e-7)
                    for g, mask in [("same", a["group"] == 0), ("opposite", a["group"] == 1), ("positive", a["min_past_gt_dist"] <= 5)]:
                        groups[g]["queries"] += int(mask.sum())
                        groups[g]["correct"] += int((mask & (a["top1_geo_dist"] <= 5)).sum())
                    nledgers += 1
                    nrows += len(a["query_idx"])
                direction[label][op] = groups
    write_json(OUT / "summary.json", {"scores": scores, "directions": direction,
               "training": completed, "test_used": False, "selection": "고정6에포크; 최종지표검증만"})
    write_json(OUT / "ledger_validation.json", {"status": "passed", "ledgers": nledgers,
               "rows": nrows, "same_optimizer_updates": completed[0]["updates"], "test_used": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "train", "keys", "score", "report"))
    p.add_argument("--arm")
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action in ("prepare", "report"):
        globals()[a.action]()
    elif a.action == "train":
        train(a.arm)
    elif a.action == "keys":
        keys(a.arm, a.sequence)
    else:
        score(a.arm, a.sequence)
