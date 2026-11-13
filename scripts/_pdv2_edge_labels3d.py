#!/usr/bin/env python3
"""고정 GNN에서 간선 분류기의 XY/3D 라벨 재학습과 최종 판정을 비교한다."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_edge_quality as old
from retrieval.edge_labels import edge_labels_3d
from _pdv2_no_context_ablation import sha256, write_json, evaluate_keys

OUT = REPO / "results/pdv2_edge_labels3d_20261016"
CONFIG = REPO / "configs/pdv2_edge_labels3d.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
MODELS, ARMS = tuple(CFG["models"]), tuple(CFG["arms"])
TRAIN, VAL, OPS, PRIMARY = old.TRAIN, old.VAL, old.OPS, old.PRIMARY
read = old.read


def folder(model: str, arm: str = "") -> Path:
    """이 실험의 전용 저장 경로를 반환한다."""
    assert model in MODELS and (not arm or arm in ARMS)
    p = OUT / model / arm
    p.mkdir(parents=True, exist_ok=True)
    return p


def protocol(data: bool = False) -> dict:
    """실행 코드와 선택 전 동결한 자료를 확인한다."""
    value = read(OUT / "protocol.json")
    for path, digest in value["runtime_sha256"].items():
        assert sha256(REPO / path) == digest, path
    if data:
        for path, digest in value["input_sha256"].items():
            assert sha256(REPO / path) == digest, path
    return value


def prepare() -> None:
    """새 특징·라벨·검증 결과를 보기 전에 비교 범위를 고정한다."""
    import torch
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    parent = old.protocol(True)
    ckpt = REPO / "checkpoints/pdv2_lag15_s1/best_model.pth"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    models = {"main": {"checkpoint": str(ckpt.relative_to(REPO)),
                       "checkpoint_sha256": sha256(ckpt), "epoch": int(blob["epoch"]) + 1},
              **{n: {k: parent["models"][n][k] for k in ("checkpoint", "checkpoint_sha256", "epoch")}
                 for n in ("current", "aligned")}}
    runtime = {REPO / p for p in parent["runtime_sha256"]}
    runtime.update(REPO / p for p in (
        "configs/pdv2_edge_labels3d.yaml", "src/retrieval/edge_labels.py",
        "scripts/_pdv2_edge_labels3d.py", "scripts/_pdv2_edge_labels3d_run.py",
        "scripts/_pdv2_edge_labels3d_verify.py", "scripts/_pdv2_edge_labels3d_plot.py",
        "scripts/_causal_build_edge_classifier.py", "tests/retrieval/test_edge_labels.py"))
    inputs = {REPO / p for p in parent["input_sha256"]}
    inputs.update(old.OUT / p for p in ("protocol.json", "summary.json", "completion_validation.json",
                                        "label_audit.json", "label_audit_validation.json"))
    inputs.update(REPO / f"results/_edge_cls_pdv2_lag15/data_{s}_{q}.npz" for s, q in TRAIN)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "config": CFG, "models": models, "test_used": False,
        "train_roster": [s + "/" + q for s, q in TRAIN],
        "val_roster": [s + "/" + q for s, q in VAL],
        "fixed": "세 GNN은 고정한다. 주평가 시드1 모델로 학습8의 첫 통과 후보30·14차원 특징을 한 번 만들어 XY/XYZ에 공유한다. 원래6적합/2확인 분할, 표준화 절차, 양성가중BCE, 초기시드0, AdamW,8에포크,최종에포크 사용. GNN 재학습과 분류기별 표현 재적합 없음.",
        "labels": "XY는 기존 앞두축거리, XYZ는 세축거리. 양성<5m,음성>=25m,나머지제외로 경계규칙 고정. 라벨 변경으로 제외행·평균/표준편차·양성가중치·업데이트수가 달라지는 것은 원래 학습 절차의 결과이며 모두 기록한다. 정확한 경계 사례수도 기록한다.",
        "controls": "original은 저장된 분류기. xy는 공유 특징으로 기존 라벨 재학습, xyz는 같은 특징의 3D 라벨 재학습. temporal은 같은 GNN에서 유사 간선을 제거한 대조. 원래 분류기 대비 효과와 동일 재학습 XY 대비 효과를 나누어 보고한다.",
        "inference": "같은 KISS 검증9, q+15, j<=q-30, 후보30 중 점수상위10 유지. 점수하한 없음. 한 모델·시퀀스의 첫통과와 후보·특징을 공유하며 둘째통과의 선택간선·분류기점수만 바뀐다. 재계량·시간간선·100+100·정렬·융합·수락점수·기준값 고정.",
        "selection": "분류기최종8에포크와 하한없음으로 사전고정한다. 새 검증 수치로 하한·가중치·모델·에포크를 고르지 않는다. 단일시드 원인 비교이며 시험과 주표는 교체하지 않는다.",
        "runtime_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(runtime)},
        "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(inputs)},
    })
    print("PREPARED", len(runtime), len(inputs), flush=True)


def load_model(name: str, device: str = "cuda") -> tuple:
    """동결한 GNN 가중치와 기존 분류기를 읽는다."""
    import torch
    spec = protocol()["models"][name]
    bundle = old.alignment.base.model(1, device=device)
    assert sha256(REPO / spec["checkpoint"]) == spec["checkpoint_sha256"]
    state = torch.load(REPO / spec["checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"]
    missing, extra = bundle[1].load_state_dict(state, strict=False)
    assert missing == ["gnn.key_scale"] and not extra
    assert all(torch.equal(bundle[1].state_dict()[k].cpu(), v) for k, v in state.items())
    bundle[1].eval()
    return bundle


def training_features(sequence: str) -> None:
    """원래 학습 표현의 후보를 한 번 추출하고 두 라벨을 붙인다."""
    import torch
    from _lag_branch_fusion import lag_pass1_graph
    from _causal_edge_features import causal_edge_features
    from build_and_train_edge_classifier import range_phase_flat
    from run_kitti_operating_point import _normalize
    protocol()
    assert sequence in protocol()["train_roster"]
    sensor, seq = sequence.split("/")
    path = OUT / "features" / f"{sensor}_{seq}.npz"
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        assert sha256(path) == read(meta_path)["sha256"]
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(old.alignment.cache_path(sequence)) as z:
        cache = old.ArrayCache({k: z[k] for k in z.files})
    import _pd_build_edge_classifier as data
    data.PD_CACHE = REPO / "data/preprocessed_v2"
    layout = data._layouts(sensor, seq, cache).astype(np.float32)
    cfg, net, _, _, rvec, _ = load_model("main")
    raw = cache["descriptors"].astype(np.float32)
    with torch.no_grad():
        first = net(lag_pass1_graph(cache, raw, cfg, "cuda", sensor)).cpu().numpy()
    q, j, features, unused = causal_edge_features(
        first, _normalize(raw * rvec[None]), range_phase_flat(layout), cache["poses"], want_labels=False)
    assert unused is None and np.all(j <= q - 30)
    xyz = cache["poses"][:, :3, 3]
    planar = xyz.copy()
    planar[:, 2] = 0
    xy, d3 = (np.linalg.norm((xyz[q] - xyz[j])[:, :2], axis=1),
              np.linalg.norm(xyz[q] - xyz[j], axis=1))
    labels = {"xy": edge_labels_3d(planar, q, j), "xyz": edge_labels_3d(xyz, q, j)}
    legacy = np.full(len(q), -1, np.int8)
    legacy[xy < 5] = 1
    legacy[xy >= 25] = 0
    np.testing.assert_array_equal(labels["xy"], legacy)
    np.savez_compressed(path, features=features, query=q, neighbor=j, distance_xy=xy,
                        distance_3d=d3, labels_xy=labels["xy"], labels_xyz=labels["xyz"])
    original = REPO / f"results/_edge_cls_pdv2_lag15/data_{sensor}_{seq}.npz"
    with np.load(original) as z:
        original_counts = {str(k): int((z["labels"] == k).sum()) for k in (0, 1)}
    write_json(meta_path, {"sequence": sequence, "sha256": sha256(path), "candidates": len(q),
        "labels": {name: {str(k): int((v == k).sum()) for k in (-1, 0, 1)} for name, v in labels.items()},
        "original_saved_counts": original_counts,
        "transition_xy_to_xyz": {f"{a}_to_{b}": int(((labels['xy'] == a) & (labels['xyz'] == b)).sum())
                                  for a in (-1, 0, 1) for b in (-1, 0, 1)},
        "boundary_counts": {f"{name}_{d}": int((v == d).sum()) for name, v in (("xy", xy), ("xyz", d3)) for d in (5, 25)},
        "protocol_sha256": sha256(OUT / "protocol.json")})
    print("FEATURES", sequence, len(q), flush=True)


def tensor_digest(state: dict) -> str:
    """초기화·가중치 동일성을 텐서의 실제 바이트로 기록한다."""
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train(arm: str) -> None:
    """동일 특징에 대한 라벨별 기존 8에포크 분류기 학습을 실행한다."""
    import torch
    from build_and_train_edge_classifier import EdgeMLP
    from sklearn.metrics import average_precision_score, roc_auc_score
    assert arm in ("xy", "xyz")
    protocol(True)
    target = OUT / "classifiers" / (arm + ".pt")
    complete = target.with_suffix(".json")
    if complete.exists():
        assert sha256(target) == read(complete)["sha256"]
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    arrays = {role: {"x": [], "y": []} for role in ("fit", "holdout")}
    features_sha = {}
    for sensor, seq in TRAIN:
        path = OUT / "features" / f"{sensor}_{seq}.npz"
        features_sha[str(path.relative_to(REPO))] = sha256(path)
        assert features_sha[str(path.relative_to(REPO))] == read(path.with_suffix(".json"))["sha256"]
        role = "holdout" if sensor + "/" + seq in CFG["holdout"] else "fit"
        with np.load(path) as z:
            y = z["labels_" + arm]
            keep = y >= 0
            arrays[role]["x"].append(z["features"][keep])
            arrays[role]["y"].append(y[keep].astype(np.float32))
    x, y = (np.concatenate(arrays["fit"][k]) for k in ("x", "y"))
    hx, hy = (np.concatenate(arrays["holdout"][k]) for k in ("x", "y"))
    mu, sd = x.mean(0), x.std(0) + 1e-6
    pos_w = float((y == 0).sum() / (y == 1).sum())
    xt, yt = torch.from_numpy((x - mu) / sd).cuda(), torch.from_numpy(y).cuda()
    hxt = torch.from_numpy((hx - mu) / sd).cuda()
    torch.manual_seed(CFG["classifier_seed"])
    clf = EdgeMLP().cuda()
    initial = tensor_digest(clf.state_dict())
    opt = torch.optim.AdamW(clf.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device="cuda"))
    history, updates = [], 0
    for epoch in range(CFG["classifier_epochs"]):
        perm = torch.randperm(len(xt), device="cuda")
        total = 0.
        for start in range(0, len(xt), CFG["batch_size"]):
            take = perm[start:start + CFG["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(clf(xt[take]), yt[take])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(take)
            updates += 1
        with torch.no_grad():
            scores = torch.sigmoid(clf(hxt)).cpu().numpy()
        history.append({"epoch": epoch + 1, "loss": total / len(xt),
                        "holdout_AP_own_label": float(average_precision_score(hy, scores)),
                        "holdout_AUC_own_label": float(roc_auc_score(hy, scores))})
        print("TRAIN", arm, history[-1], flush=True)
    assert all(torch.isfinite(v).all() for v in clf.state_dict().values())
    torch.save({"state_dict": clf.state_dict(), "mu": mu, "sd": sd,
                "label_space": arm, "protocol_sha256": sha256(OUT / "protocol.json")}, target)
    write_json(complete, {"arm": arm, "sha256": sha256(target), "initial_state_sha256": initial,
        "features_sha256": features_sha, "epochs": len(history), "updates": updates,
        "rows": {role: {"positive": int(sum((v == 1).sum() for v in a["y"])),
                         "negative": int(sum((v == 0).sum() for v in a["y"]))} for role, a in arrays.items()},
        "positive_weight": pos_w, "mu": mu.tolist(), "sd": sd.tolist(), "history": history,
        "holdout_used_for_selection": False, "protocol_sha256": sha256(OUT / "protocol.json")})


def load_classifier(arm: str, device: str = "cuda") -> tuple:
    """새 분류기와 적합 자료에서만 구한 표준화를 읽는다."""
    import torch
    from build_and_train_edge_classifier import EdgeMLP
    path = OUT / "classifiers" / (arm + ".pt")
    assert arm in ("xy", "xyz") and sha256(path) == read(path.with_suffix(".json"))["sha256"]
    blob = torch.load(path, map_location="cpu", weights_only=False)
    clf = EdgeMLP().to(device).eval()
    clf.load_state_dict(blob["state_dict"])
    return clf, blob


def capture(cache, sensor: str, layout: np.ndarray, bundle: tuple, device: str) -> tuple:
    """원래 실행 경로의 첫 통과 특징을 세 분류기에 공통으로 제공한다."""
    import _lag_branch_fusion as lag
    original, details = lag.causal_edge_features, {}

    def tap(*args, **kwargs):
        assert kwargs["want_labels"] is False
        result = original(*args, **kwargs)
        details["features"] = result[2]
        details["l2"] = np.linalg.norm(args[0][result[0]] - args[0][result[1]], axis=1)
        return result

    lag.causal_edge_features = tap
    try:
        baseline, graphs, candidate = old.capture(cache, sensor, layout, bundle, device)
    finally:
        lag.causal_edge_features = original
    return baseline, graphs[1], {**candidate, **details}


def replace_edges(graph, candidate: dict, probability: np.ndarray, selected: np.ndarray):
    """시간 간선과 원래 노드 입력을 유지하고 선택된 유사 간선을 넣는다."""
    import torch
    result = old.filter_graph(graph, np.zeros(int((graph.edge_type == 1).sum()), bool))
    device = result.edge_index.device
    q, j = candidate["query"][selected], candidate["neighbor"][selected]
    cos, l2, p = candidate["features"][selected, 0], candidate["l2"][selected], probability[selected]
    attrs = np.stack([np.zeros_like(p), np.zeros_like(p), cos, np.log1p(l2) / 5, p], axis=1)
    result.edge_index = torch.cat([result.edge_index, torch.as_tensor(np.stack([j, q]), device=device)], dim=1)
    result.edge_type = torch.cat([result.edge_type, torch.ones(len(q), dtype=torch.long, device=device)])
    result.edge_attr = torch.cat([result.edge_attr, torch.as_tensor(attrs, dtype=torch.float32, device=device)])
    return result


def apply_classifier(net, graph, candidate: dict, classifier: tuple) -> tuple:
    """같은 후보 30개에서 점수 상위 10개를 골라 최종 키를 계산한다."""
    import torch
    from _causal_branch_fusion import select_edges_per_node
    from _xs_eval_lib import normalize
    clf, blob = classifier
    device = graph.edge_index.device
    features = (candidate["features"] - blob["mu"].astype(np.float32)) / blob["sd"].astype(np.float32)
    with torch.no_grad():
        probs = torch.sigmoid(clf(torch.from_numpy(features).float().to(device))).cpu().numpy()
        selected = select_edges_per_node(candidate["query"], probs, CFG["selected_count"])
        result = normalize(net(replace_edges(graph, candidate, probs, selected)).float().cpu().numpy())
    return result, probs, selected


def keys(name: str, sequence: str) -> None:
    """검증 시퀀스에서 네 조건의 키와 실제 간선 선택을 저장한다."""
    import torch
    from _xs_eval_lib import normalize
    protocol()
    assert sequence in protocol()["val_roster"]
    sensor, seq, paths, min_past = old.alignment.base.inputs("val", sequence)
    target = folder(name) / f"quality_{sensor}_{seq}.json"
    if target.exists():
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(paths["cache"]) as z:
        cache = old.ArrayCache({k: z[k] for k in z.files if k != "poses_gt"})
        gt = z["poses_gt"][:, :3, 3]
    with np.load(paths["layouts"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        layout = z["nsd_layouts"].astype(np.float32)
    bundle = load_model(name)
    baseline, graph, c = capture(cache, sensor, layout, bundle, "cuda")
    q, j = c["query"], c["neighbor"]
    d3, d2 = np.linalg.norm(gt[q] - gt[j], axis=1), np.linalg.norm((gt[q] - gt[j])[:, :2], axis=1)
    ref, _ = old.read_ledger(old.BASE / f"s1/val_selected_symmetric_{OPS[0]}_{sensor}_{seq}.npz")
    groups = np.full(len(gt), -1, np.int8)
    groups[ref["query_idx"]] = ref["group"]
    groups[min_past > 25] = 3
    groups[(min_past > 5) & (min_past <= 25)] = 4
    valid = (np.arange(len(gt)) >= 30) & (np.arange(len(gt)) < len(gt) - 15)
    eligible = {"all": valid, **{g: valid & (groups == i) for i, g in enumerate(
        ("same_available", "opposite_only", "other_positive", "negative", "ambiguous"))}}
    quality = {"model": name, "sequence": sequence, "nodes": len(gt), "arms": {}, "keys_sha256": {}}
    saved = {k: c[k] for k in ("query", "neighbor", "features", "l2")}
    saved.update(distance_3d=d3, distance_xy=d2, group=groups, valid=valid)
    for arm in ARMS:
        if arm == "original":
            embedding, probs, selected = baseline, c["probability"], c["selected_order"]
        elif arm == "temporal":
            probs, selected = np.zeros(len(q), np.float32), np.empty(0, np.int64)
            with torch.no_grad():
                embedding = normalize(bundle[1](replace_edges(graph, c, probs, selected)).float().cpu().numpy())
        else:
            embedding, probs, selected = apply_classifier(bundle[1], graph, c, load_classifier(arm))
        saved["prob_" + arm], saved["selected_" + arm] = probs, selected
        quality["arms"][arm] = {g: old.edge_counts(q[selected], d3[selected], d2[selected], probs[selected], mask)
                                  for g, mask in eligible.items()}
        meta = {"model": name, "arm": arm, "protocol_sha256": sha256(OUT / "protocol.json")}
        assert embedding.shape == (len(gt), 416) and np.isfinite(embedding).all()
        key_path = folder(name, arm) / f"keys_{sensor}_{seq}.npz"
        np.savez_compressed(key_path, embedding=embedding,
                            scan_ids=cache["scan_ids"], meta=json.dumps(meta))
        quality["keys_sha256"][arm] = sha256(key_path)
    edge_path = folder(name) / f"edges_{sensor}_{seq}.npz"
    np.savez_compressed(edge_path, **saved)
    quality.update(edge_sha256=sha256(edge_path), protocol_sha256=sha256(OUT / "protocol.json"))
    write_json(target, quality)
    print("KEYS", name, sequence, flush=True)


def score(name: str, arm: str, sequence: str) -> None:
    """공통 최종 검색·매칭·수락 규칙으로 평가한다."""
    protocol()
    sensor, seq, paths, min_past = old.alignment.base.inputs("val", sequence)
    target = folder(name, arm) / f"val_{sensor}_{seq}.json"
    if target.exists():
        return
    with np.load(target.parent / f"keys_{sensor}_{seq}.npz") as z:
        embedding, scan_ids = z["embedding"], z["scan_ids"]
    with np.load(paths["cache"]) as z:
        np.testing.assert_array_equal(scan_ids, z["scan_ids"])
        gt = z["poses_gt"][:, :3, 3]
    selection = read(old.BASE / "s1/confidence_selection.json")["ops"]
    result = {}
    for op in OPS:
        with np.load(paths["spec"]) as z:
            np.testing.assert_array_equal(scan_ids, z["scan_ids"])
            bank = old.alignment.base.bank_for(op, z)
        sel = selection[op]
        arrays = evaluate_keys(embedding, bank, gt, min_past, {1: tuple(float(w) for w in sel["tuple"])})[1]
        keep = arrays["query_idx"] < len(gt) - 15
        arrays = {k: v[keep] for k, v in arrays.items()}
        ref, _ = old.read_ledger(old.BASE / f"s1/val_selected_symmetric_{op}_{sensor}_{seq}.npz")
        for key in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(arrays[key], ref[key])
        arrays["group"] = ref["group"]
        dest = target.parent / f"val_{op}_{sensor}_{seq}.npz"
        np.savez_compressed(dest, **arrays, meta=json.dumps({"protocol_sha256": sha256(OUT / "protocol.json")}))
        result[op] = {"metrics": old.sequence_metrics(arrays, sel["decision"], sel["tau_star"]),
                      "ledger": str(dest.relative_to(REPO)), "sha256": sha256(dest)}
    write_json(target, {"model": name, "arm": arm, "sequence": sequence, "ops": result,
                       "protocol_sha256": sha256(OUT / "protocol.json")})


def report() -> None:
    """라벨 수정과 재학습 자체의 효과를 나누어 최종 지표를 집계한다."""
    protocol(True)
    scores, quality = {}, {}
    for name in MODELS:
        scores[name], quality[name] = {}, {}
        for arm in ARMS:
            rows = {s + "/" + q: read(folder(name, arm) / f"val_{s}_{q}.json") for s, q in VAL}
            ops = {op: old.aggregate_rows(rows, "val", op) for op in OPS}
            for op in OPS:
                dirs = {g: {"n": 0, "correct": 0, "recovered": 0, "regressed": 0}
                        for g in ("same_available", "opposite_only", "other_positive")}
                for row in rows.values():
                    a, _ = old.read_ledger(REPO / row["ops"][op]["ledger"])
                    s, q = row["sequence"].split("/")
                    b, _ = old.read_ledger(folder(name, "original") / f"val_{op}_{s}_{q}.npz")
                    before, after = b["top1_geo_dist"] <= 5., a["top1_geo_dist"] <= 5.
                    for i, g in enumerate(dirs):
                        mask = (a["min_past_gt_dist"] <= 5) & (a["group"] == i)
                        for k, condition in (("n", mask), ("correct", mask & after),
                                             ("recovered", mask & ~before & after), ("regressed", mask & before & ~after)):
                            dirs[g][k] += int(condition.sum())
                ops[op]["directions"] = dirs
                ops[op]["negative_false_accepts"] = sum(r["ops"][op]["metrics"]["negative_false_accepts"] for r in rows.values())
            scores[name][arm] = {"ops": ops}
            qr = [read(folder(name) / f"quality_{s}_{q}.json")["arms"][arm] for s, q in VAL]
            quality[name][arm] = {g: old.add_counts([r[g] for r in qr]) for g in qr[0]}
    effects = {n: {op: {m: {"xyz_minus_original": scores[n]["xyz"]["ops"][op]["macro"][m] - scores[n]["original"]["ops"][op]["macro"][m],
                           "xyz_minus_xy": scores[n]["xyz"]["ops"][op]["macro"][m] - scores[n]["xy"]["ops"][op]["macro"][m],
                           "xyz_minus_temporal": scores[n]["xyz"]["ops"][op]["macro"][m] - scores[n]["temporal"]["ops"][op]["macro"][m]}
                           for m in PRIMARY} for op in OPS} for n in MODELS}
    write_json(OUT / "summary.json", {"scores": scores, "effects": effects, "quality": quality,
        "test_used": False, "scope": "고정 GNN·분할 v2 검증·시드 1. 라벨 수정 이외의 최적화 없음.",
        "classifiers": {a: read(OUT / "classifiers" / (a + ".json")) for a in ("xy", "xyz")},
        "protocol_sha256": sha256(OUT / "protocol.json")})
    print("REPORTED", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "features", "train", "keys", "score", "report"))
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "prepare": prepare()
    elif a.action == "features": training_features(a.sequence)
    elif a.action == "train": train(a.arm)
    elif a.action == "keys": keys(a.model, a.sequence)
    elif a.action == "score": score(a.model, a.arm, a.sequence)
    else: report()
