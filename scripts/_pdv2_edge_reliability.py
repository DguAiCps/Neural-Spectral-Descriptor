#!/usr/bin/env python3
"""고정 GNN의 간선 신뢰도에서 입력 정보와 학습 조건의 영향을 분리한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_pdv2_edge_reliability_run.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_edge_labels3d as base
from _pdv2_candidate_data import load_data, train_path
from _pdv2_no_context_ablation import sha256, write_json
from retrieval.edge_reliability import (
    binary_metrics, candidate_matrices, query_metrics, summarize_queries,
    retention_threshold, confidence_bins,
)

OUT = REPO / "results/pdv2_edge_reliability_20261016"
CONFIG = REPO / "configs/pdv2_edge_reliability.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
TRAIN, VAL, OPS = base.TRAIN, base.VAL, base.OPS
ARMS = tuple(CFG["arms"])
SIGNALS = {"key_cos": (0, 1), "raw_cos": (1, 1), "context_cos": (2, 1),
           "query_margin": (5, 1), "cylinder_alignment": (8, 1),
           "cylinder_peak_gap": (11, 1), "cylinder_sharpness": (12, -1)}


def protocol(data: bool = False) -> dict:
    """동결된 코드·모델·캐시를 검사한다."""
    p = base.read(OUT / "protocol.json")
    for name, h in p["runtime_sha256"].items():
        assert sha256(REPO / name) == h, name
    if data:
        for name, h in p["input_sha256"].items():
            assert sha256(REPO / name) == h, name
    return p


def prepare() -> None:
    """신규 특징과 판별 결과를 보기 전에 비교·분모·예산을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    parent = base.protocol(True)
    runtime = {REPO / p for p in parent["runtime_sha256"]}
    runtime.update(REPO / p for p in (
        "configs/pdv2_edge_reliability.yaml", "scripts/_pdv2_edge_reliability.py",
        "scripts/_pdv2_edge_reliability_run.py", "scripts/_pdv2_edge_reliability_audit.py",
        "scripts/_pdv2_candidate_data.py", "src/retrieval/edge_reliability.py",
        "tests/retrieval/test_edge_reliability.py"))
    inputs = {REPO / p for p in parent["input_sha256"]}
    inputs.update(base.OUT / p for p in ("protocol.json", "classifiers/xyz.pt", "classifiers/xyz.json", "quality_diagnostics.json"))
    for s, q in TRAIN:
        p = train_path(s + "/" + q)
        meta = base.read(p.with_suffix(".json"))
        assert sha256(p) == meta["sha256"]
        assert meta["script_sha256"] == sha256(REPO / "scripts/_pdv2_candidate_data.py")
        inputs.update((p, p.with_suffix(".json")))
    for s, q in VAL:
        inputs.update(base.OUT / "main" / f"{kind}_{s}_{q}.{ext}" for kind, ext in (("edges", "npz"), ("quality", "json")))
    write_json(OUT / "protocol.json", {"created_unix": time.time(), "config": CFG,
        "test_used": False, "scope": "요청한 1·2단계: 간선 판별 정보·분류기 학습 조건 진단. GNN 재학습·최종 시스템 평가·시험·주표 교체 없음.",
        "A": "기존 3D 라벨 분류기. 주평가 시드1 GNN의 GT 학습 키프레임/자세 특징으로 적합한 가중치 그대로.",
        "B": "같은 GNN·14특징·3D 라벨. 기존 학습8의 KISS-ICP 키프레임/자세/원통 특징으로 온라인 첫통과와 과거 후보30을 생성하여 분류기만 재학습. GT는 라벨·진단에만 사용한다.",
        "C": "B와 동일 행·라벨·첫14특징에 해당 운영점의 원시 스펙트럼 정렬점수와 열정렬점수2개만 추가. NSD/H 별도 분류기다. 새로운 저장 블록은 요구하지 않으나 간선후보에서 정렬을 먼저 계산하는 비용이 든다.",
        "budget": "적합6/확인2(KITTI06·Bridge01). 8에포크 최종모델, AdamW .001/weightdecay .0001, batch65536, 양성가중BCE. 표준화는 적합6에서만 구한다. B/C는16입력64/64/1의 같은 초기값·순서·업데이트수. B의 추가2입력과 해당 초기가중치는0이며 C의 추가 초기가중치도0. 공통14입력 초기값은 A 학습의 초기값과 같다.",
        "remaining_differences": "A->B는 자세·키프레임·원통 캐시·유효 q<N-15·후보분포를 함께 맞추는 입력조건 묶음이다. 행수·표준화·양성가중치·업데이트수 변화도 기록하며 순수한 자세 단독 효과로 해석하지 않는다. GNN 자체의 GT 학습과 검증 센서/장소 차이는 남는다.",
        "queries": "train/val 공통30<=q<N-15, 후보j<=q-30. 검증 후보·14특징은 저장 main 후보를 그대로 사용한다. 새 특징은 단일스캔의 저장스펙트럼에서만 계산한다.",
        "labels": "학습/쌍PR은3D<5m 양성,>=25m 음성,중간제외. 실제 정답이웃/질의 보존은<=5m. 경계수 기록. 후보에 GT 정답 강제삽입 없음.",
        "metrics": "질의 내부 AUC는 양/음 후보가 함께 있는 질의만(동점0.5). 정답후보가 있을때 Hit1/Hit10, 전체 재방문질의 Hit10, 선택10 정확도, 방향별 분모 병기. 전체쌍AP/AUC와 양성비율은 보조. query_margin은 질의상수라 내부AUC0.5이며 정답후보존재 판단용이다.",
        "abstention": "질의의 후보30 최대점수로 정답간선 존재를 판별. 전체지도 양성<=5/음성>25 질의만 사용하고 모호질의 제외. 지도에는 정답이 있지만 후보30에서 빠진 질의는 별도 보고. 임계값은 확인2에서 정답후보존재 질의 최소95% 보존으로 고정하고 검증에서 보존율·비재방문오연결·누락질의오연결을 측정한다. 배포용 임계값 선택이나 확률 보정은 아니다.",
        "selection": "최종8에포크 고정, 새 검증으로 특징 부호·모델·임계값·에포크 선택 없음. 단일 GNN시드1/분류기시드0 진단. 동점은 기존 후보순서 유지. 최고 수치만 골라 보고하지 않는다.",
        "runtime_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(runtime)},
        "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(inputs)}})
    print("PREPARED", len(runtime), len(inputs), flush=True)


def feature_path(role: str, sequence: str) -> Path:
    """공유 특징의 저장 경로를 반환한다."""
    return OUT / "features" / (role + "_" + sequence.replace("/", "_") + ".npz")


def nearest_past(positions: np.ndarray) -> np.ndarray:
    """정답 작성용 최근접 과거 거리를 정확히 계산한다."""
    from scipy.spatial import cKDTree
    result = np.full(len(positions), np.nan)
    for start in range(30, len(positions), 256):
        end = min(start + 256, len(positions))
        prefix = start - 29
        distance, _ = cKDTree(positions[:prefix]).query(positions[start:end])
        for row, q in enumerate(range(start, end)):
            extra = positions[prefix:q - 29]
            result[q] = min(distance[row], np.linalg.norm(extra - positions[q], axis=1).min()) if len(extra) else distance[row]
    return result


def extra_features(spec: dict, q: np.ndarray, j: np.ndarray) -> dict:
    """기존 최종 정렬 함수를 같은 후보쌍에 적용한다."""
    from _xs_eval_lib import align_scores, column_align_scores
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1, len(q)]
    result = {}
    for op in OPS:
        _, fft, norm, column = base.old.alignment.base.bank_for(op, spec)
        x = np.empty((len(q), 2), np.float32)
        for start, end in zip(starts[:-1], starts[1:]):
            query, candidates = int(q[start]), j[start:end]
            x[start:end, 0] = align_scores(fft, norm, query, candidates)
            x[start:end, 1] = column_align_scores(column, query, candidates)
        assert np.isfinite(x).all()
        result["extra_" + op] = x
    return result


def features(role: str, sequence: str) -> None:
    """정답을 추론 입력에서 분리하고 공통 후보·특징을 한 번 저장한다."""
    import torch
    from _lag_branch_fusion import lag_pass1_graph
    from _causal_edge_features import causal_edge_features
    from build_and_train_edge_classifier import range_phase_flat
    from _xs_eval_lib import normalize
    from _pdv2_pair_evaluate import direction_groups
    protocol()
    assert tuple(sequence.split("/")) in (TRAIN if role == "train" else VAL)
    path = feature_path(role, sequence)
    if path.with_suffix(".json").exists():
        assert sha256(path) == base.read(path.with_suffix(".json"))["sha256"]
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache, layout, spec, gt, min_past, paths = load_data(role, sequence)
    assert "poses_gt" not in cache.files
    n = len(gt)
    valid = (np.arange(n) >= 30) & (np.arange(n) < n - 15)
    if role == "train":
        cfg, net, _, _, r, _ = base.load_model("main")
        raw = cache["descriptors"].astype(np.float32)
        with torch.no_grad():
            emb = net(lag_pass1_graph(cache, raw, cfg, "cuda", sequence.split("/")[0])).cpu().numpy()
        q, j, x, unused = causal_edge_features(emb, normalize(raw * r[None]), range_phase_flat(layout), cache["poses"], want_labels=False)
        assert unused is None
        min_past = nearest_past(gt[:, :3, 3])
        group = direction_groups(gt[:, :3, 3], min_past)
        probability = None
    else:
        s, seq = sequence.split("/")
        src = base.OUT / "main" / f"edges_{s}_{seq}.npz"
        assert sha256(src) == base.read(base.OUT / "main" / f"quality_{s}_{seq}.json")["edge_sha256"]
        with np.load(src) as z:
            q, j, x, group = z["query"], z["neighbor"], z["features"], z["group"]
            probability = z["prob_xyz"]
            np.testing.assert_array_equal(valid, z["valid"])
    keep = valid[q]
    q, j, x = q[keep], j[keep], x[keep]
    d = np.linalg.norm(gt[q, :3, 3] - gt[j, :3, 3], axis=1)
    y = np.full(len(q), -1, np.int8)
    y[d < 5], y[d >= 25] = 1, 0
    group[min_past > 25] = 3
    group[(min_past > 5) & (min_past <= 25)] = 4
    assert np.all(j <= q - 30) and x.shape == (len(q), 14) and np.isfinite(x).all()
    extra = extra_features(spec, q, j)
    saved = {"query": q, "neighbor": j, "features": x, "distance": d, "labels": y,
             "valid": valid, "min_past": min_past, "group": group, "scan_ids": cache["scan_ids"], **extra}
    if probability is not None:
        saved["probability_A"] = probability[keep]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **saved)
    write_json(path.with_suffix(".json"), {"role": role, "sequence": sequence, "nodes": n, "edges": len(q),
        "labels": {str(i): int((y == i).sum()) for i in (-1, 0, 1)}, "sha256": sha256(path),
        "boundaries": {str(k): int((d == k).sum()) for k in (5, 25)},
        "inputs_sha256": {str(v.relative_to(REPO)): sha256(v) for v in paths.values()},
        "protocol_sha256": sha256(OUT / "protocol.json")})
    print("FEATURES", role, sequence, len(q), flush=True)


def model_input(z, arm: str) -> np.ndarray:
    """B의 추가 두 열은0이고 C는 해당 운영점의 두 정렬 점수다."""
    if arm == "A":
        return z["features"]
    extra = np.zeros((len(z["features"]), 2), np.float32) if arm == "B" else z["extra_" + arm[2:]]
    return np.concatenate([z["features"], extra], axis=1)


def initial_classifier():
    """A와 같은14차원 초기값을 공유하고 새 입력 가중치만0으로 확장한다."""
    import torch
    from build_and_train_edge_classifier import EdgeMLP
    torch.manual_seed(CFG["classifier_seed"])
    original = EdgeMLP(14)
    original_hash = base.tensor_digest(original.state_dict())
    assert original_hash == base.read(base.OUT / "classifiers/xyz.json")["initial_state_sha256"]
    net = EdgeMLP(16)
    state = original.state_dict()
    state["net.0.weight"] = torch.cat([state["net.0.weight"], torch.zeros(64, 2)], dim=1)
    net.load_state_dict(state)
    return net, original_hash


def train(arm: str) -> None:
    """GNN을 고정하고 세 분류기에 동일한 행·초기값·최적화 순서를 적용한다."""
    import torch
    protocol(True)
    assert arm in ARMS and arm != "A"
    path = OUT / "classifiers" / (arm + ".pt")
    if path.with_suffix(".json").exists():
        assert sha256(path) == base.read(path.with_suffix(".json"))["sha256"]
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    arrays = {role: {"x": [], "y": []} for role in ("fit", "holdout")}
    sources = {}
    for s, q in TRAIN:
        p = feature_path("train", s + "/" + q)
        sources[str(p.relative_to(REPO))] = sha256(p)
        assert sources[str(p.relative_to(REPO))] == base.read(p.with_suffix(".json"))["sha256"]
        role = "holdout" if s + "/" + q in CFG["holdout"] else "fit"
        with np.load(p) as z:
            keep = z["labels"] >= 0
            arrays[role]["x"].append(model_input(z, arm)[keep])
            arrays[role]["y"].append(z["labels"][keep].astype(np.float32))
    arrays = {r: {k: np.concatenate(v) for k, v in a.items()} for r, a in arrays.items()}
    x, y = arrays["fit"]["x"], arrays["fit"]["y"]
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt, yt = torch.from_numpy((x - mu) / sd).cuda(), torch.from_numpy(y).cuda()
    hx = torch.from_numpy((arrays["holdout"]["x"] - mu) / sd).cuda()
    net, initial14 = initial_classifier()
    initial16 = base.tensor_digest(net.state_dict())
    net = net.cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    weight = float((y == 0).sum() / (y == 1).sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device="cuda"))
    rng = torch.Generator(device="cuda").manual_seed(CFG["classifier_seed"])
    history, order_hashes, updates = [], [], 0
    for epoch in range(CFG["classifier_epochs"]):
        perm = torch.randperm(len(x), generator=rng, device="cuda")
        import hashlib
        order_hashes.append(hashlib.sha256(perm.cpu().numpy().tobytes()).hexdigest())
        total = 0.
        for start in range(0, len(x), CFG["batch_size"]):
            ids = perm[start:start + CFG["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(net(xt[ids]), yt[ids])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(ids)
            updates += 1
        with torch.no_grad():
            hp = torch.sigmoid(net(hx)).cpu().numpy()
        history.append({"epoch": epoch + 1, "train_loss": total / len(x),
                        "holdout_edges": binary_metrics(arrays["holdout"]["y"], hp)})
        print("TRAIN", arm, epoch + 1, history[-1], flush=True)
    assert all(torch.isfinite(v).all() for v in net.state_dict().values())
    if arm == "B":
        assert torch.count_nonzero(net.net[0].weight[:, 14:]) == 0
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd, "arm": arm,
                "protocol_sha256": sha256(OUT / "protocol.json")}, path)
    write_json(path.with_suffix(".json"), {"arm": arm, "sha256": sha256(path), "initial14_sha256": initial14,
        "initial16_sha256": initial16, "order_sha256": order_hashes, "features_sha256": sources,
        "rows": {r: {"positive": int((a["y"] == 1).sum()), "negative": int((a["y"] == 0).sum())} for r, a in arrays.items()},
        "epochs": len(history), "updates": updates, "positive_weight": weight, "mu": mu.tolist(), "sd": sd.tolist(),
        "history": history, "selected_epoch": CFG["classifier_epochs"], "validation_used_for_selection": False})


def classifier(arm: str, device: str = "cuda") -> tuple:
    """해시를 확인하고 고정된 분류기를 읽는다."""
    import torch
    from build_and_train_edge_classifier import EdgeMLP
    if arm == "A":
        return base.load_classifier("xyz", device)
    path = OUT / "classifiers" / (arm + ".pt")
    assert sha256(path) == base.read(path.with_suffix(".json"))["sha256"]
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = EdgeMLP(16).to(device).eval()
    net.load_state_dict(blob["state_dict"])
    return net, blob


def predict(role: str, sequence: str) -> None:
    """같은 후보에 네 분류기의 점수를 기록한다."""
    import torch
    protocol()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = OUT / "predictions" / (role + "_" + sequence.replace("/", "_") + ".npz")
    if out.with_suffix(".json").exists():
        assert sha256(out) == base.read(out.with_suffix(".json"))["sha256"]
        return
    predictions, error = {}, None
    with np.load(feature_path(role, sequence)) as z:
        for arm in ARMS:
            net, blob = classifier(arm)
            x = model_input(z, arm)
            values = []
            with torch.no_grad():
                for start in range(0, len(x), 65536):
                    part = torch.from_numpy((x[start:start + 65536] - blob["mu"]) / blob["sd"]).cuda()
                    values.append(torch.sigmoid(net(part)).cpu().numpy())
            p = np.concatenate(values)
            assert p.dtype == np.float32 and np.isfinite(p).all()
            if role == "val" and arm == "A":
                error = float(np.max(np.abs(p - z["probability_A"])))
                np.testing.assert_allclose(p, z["probability_A"], atol=2e-6, rtol=0)
                p = z["probability_A"]
            predictions[arm] = p
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **predictions)
    write_json(out.with_suffix(".json"), {"role": role, "sequence": sequence, "sha256": sha256(out),
        "feature_sha256": sha256(feature_path(role, sequence)), "A_reproduction_error": error})
    print("PREDICT", role, sequence, flush=True)


def analyze_sequence(role: str, sequence: str, baseline: bool = False) -> None:
    """각 특징과 분류기의 질의 내부 판별·연결 거절 통계를 계산한다."""
    protocol()
    path = OUT / ("stage1" if baseline else "diagnostics") / (role + "_" + sequence.replace("/", "_"))
    if path.with_suffix(".json").exists():
        return
    with np.load(feature_path(role, sequence)) as z:
        c = {k: z[k] for k in z.files}
    pp = OUT / "predictions" / (role + "_" + sequence.replace("/", "_") + ".npz")
    if baseline:
        assert role == "val"
        scores = {"A": c["probability_A"]}
    else:
        with np.load(pp) as z:
            scores = {k: z[k] for k in z.files}
    scores.update({name: c["features"][:, col] * sign for name, (col, sign) in SIGNALS.items()})
    for op in OPS:
        scores[op + "_alignment"] = c["extra_" + op][:, 0]
        scores[op + "_column"] = c["extra_" + op][:, 1]
    n = len(c["valid"])
    d, _ = candidate_matrices(c["query"], c["distance"], n)
    masks = {"all": c["valid"], "positive_query": c["valid"] & (c["min_past"] <= 5),
             **{g: c["valid"] & (c["group"] == i) for i, g in enumerate(("same_available", "opposite_only", "other_positive", "negative", "ambiguous"))}}
    fields, result = {k: c[k] for k in ("valid", "min_past", "group")}, {}
    edge_keep = c["labels"] >= 0
    for name, score in scores.items():
        sm, _ = candidate_matrices(c["query"], score, n)
        qm = query_metrics(sm, d, c["valid"])
        result[name] = {"edges": binary_metrics(c["labels"][edge_keep], score[edge_keep]),
                        "groups": {g: summarize_queries(qm, mask) for g, mask in masks.items()}}
        for k, v in qm.items():
            fields[name + "__" + k] = v
        eligible = masks["positive_query"] | masks["negative"]
        result[name]["query_existence"] = binary_metrics(qm["has_correct"][eligible], qm["max_score"][eligible])
        if name in ARMS:
            result[name]["bins_all"] = confidence_bins(score, c["distance"])
            from _causal_branch_fusion import select_edges_per_node
            ids = select_edges_per_node(c["query"], score, 10)
            result[name]["bins_selected10"] = confidence_bins(score[ids], c["distance"][ids])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **fields)
    write_json(path.with_suffix(".json"), {"sequence": sequence, "role": role, "scores": result,
        "node_metrics_sha256": sha256(path.with_suffix(".npz")), "feature_sha256": sha256(feature_path(role, sequence)),
        "prediction_sha256": sha256(pp) if not baseline else None})
    print("DIAGNOSE", role, sequence, flush=True)


def mean_nested(rows: list) -> dict:
    """시퀀스 간 결측 분모는 제외하고 지표별 평균을 계산한다."""
    return {key: (float(np.mean([r[key] for r in rows if r[key] is not None])) if any(r[key] is not None for r in rows) else None) for key in rows[0]}


def report() -> None:
    """확인용 임계값을 고정하고 검증 판별력을 전 조건에 대해 보고한다."""
    from _pdv2_pilot import FAMILY
    protocol(True)
    records = {role: {s + "/" + q: base.read(OUT / "diagnostics" / f"{role}_{s}_{q}.json") for s, q in roster}
               for role, roster in (("train", TRAIN), ("val", VAL))}
    names = list(next(iter(records["val"].values()))["scores"])
    families = {FAMILY.get(s, s) for s, _ in VAL}
    thresholds, pooled, threshold_rows = {}, {}, {}
    for name in names:
        positives = []
        for sequence in CFG["holdout"]:
            with np.load(OUT / "diagnostics" / ("train_" + sequence.replace("/", "_") + ".npz")) as z:
                mask = z["valid"] & z[name + "__has_correct"]
                positives.append(z[name + "__max_score"][mask])
        threshold = retention_threshold(np.concatenate(positives), CFG["holdout_query_retention"])
        thresholds[name] = {"threshold": threshold, "holdout_positive_queries": sum(len(v) for v in positives),
                            "holdout_retention": float((np.concatenate(positives) >= threshold).mean())}
        pool = {k: [] for k in ("labels", "scores", "has_correct", "max_score")}
        threshold_rows[name] = {}
        for s, q in VAL:
            seq = s + "/" + q
            with np.load(OUT / "diagnostics" / f"val_{s}_{q}.npz") as z:
                valid, minimum = z["valid"], z["min_past"]
                pos, neg = minimum <= 5, minimum > 25
                has, maximum = z[name + "__has_correct"], z[name + "__max_score"]
                masks = {"candidate_has_correct": valid & has, "no_revisit": valid & neg,
                         "positive_but_missing_candidate": valid & pos & ~has,
                         "no_correct_candidate": valid & (pos | neg) & ~has,
                         "same_available": valid & (z["group"] == 0) & has,
                         "opposite_only": valid & (z["group"] == 1) & has}
                threshold_rows[name][seq] = {g: {"n": int(m.sum()), "connected": int((m & (maximum >= threshold)).sum()),
                    "rate": float((maximum[m] >= threshold).mean()) if m.any() else None} for g, m in masks.items()}
                use = valid & (pos | neg)
                pool["has_correct"].append(has[use])
                pool["max_score"].append(maximum[use])
            with np.load(feature_path("val", seq)) as z:
                keep = z["labels"] >= 0
                pool["labels"].append(z["labels"][keep])
                if name in SIGNALS:
                    col, sign = SIGNALS[name]
                    score = z["features"][:, col] * sign
                elif name not in ARMS:
                    op, field = name.rsplit("_", 1)
                    score = z["extra_" + op][:, 0 if field == "alignment" else 1]
                else:
                    with np.load(OUT / "predictions" / f"val_{s}_{q}.npz") as pp:
                        score = pp[name]
                pool["scores"].append(score[keep])
        pooled[name] = {"edges": binary_metrics(np.concatenate(pool["labels"]), np.concatenate(pool["scores"])),
                        "query_existence": binary_metrics(np.concatenate(pool["has_correct"]), np.concatenate(pool["max_score"]))}
    macro, family_macro, within_pooled = {}, {}, {}
    for name in names:
        family_macro[name], macro[name] = {}, {}
        for family in sorted(families):
            rows = [r["scores"][name] for seq, r in records["val"].items() if FAMILY.get(seq.split("/")[0], seq.split("/")[0]) == family]
            family_macro[name][family] = {"edges": mean_nested([r["edges"] for r in rows]),
                "query_existence": mean_nested([r["query_existence"] for r in rows]),
                "groups": {g: mean_nested([r["groups"][g] for r in rows]) for g in rows[0]["groups"]},
                "threshold_rates": {g: mean_nested([threshold_rows[name][seq][g] for seq in threshold_rows[name]
                                     if FAMILY.get(seq.split("/")[0], seq.split("/")[0]) == family]) for g in next(iter(threshold_rows[name].values()))}}
        rows = list(family_macro[name].values())
        macro[name] = {"edges": mean_nested([r["edges"] for r in rows]), "query_existence": mean_nested([r["query_existence"] for r in rows]),
                      "groups": {g: mean_nested([r["groups"][g] for r in rows]) for g in rows[0]["groups"]},
                      "threshold_rates": {g: mean_nested([r["threshold_rates"][g] for r in rows]) for g in rows[0]["threshold_rates"]}}
        within_pooled[name] = {}
        for g in ("positive_query", "same_available", "opposite_only", "negative"):
            values = [r["scores"][name]["groups"][g] for r in records["val"].values()]
            counts = {k: int(sum(r[k] for r in values)) for k in ("queries", "candidate_has_correct", "mixed_queries", "selected_edges", "selected_correct")}
            for k, denom in (("within_query_AUC", "mixed_queries"), ("hit1_given_available", "candidate_has_correct"), ("hit10_given_available", "candidate_has_correct"), ("hit10_all_queries", "queries")):
                counts[k] = sum(r[k] * r[denom] for r in values if r[k] is not None) / counts[denom] if counts[denom] else None
            within_pooled[name][g] = counts
    write_json(OUT / "summary.json", {"scope": "간선 판별 진단이며 최종 장소인식 AP가 아니다. GNN시드1·분류기시드0·검증9.",
        "test_used": False, "pooled": pooled, "within_query_pooled": within_pooled, "macro": macro,
        "family_macro": family_macro, "thresholds": thresholds, "threshold_per_sequence": threshold_rows,
        "per_sequence": records, "training": {arm: base.read(OUT / "classifiers" / (arm + ".json")) for arm in ARMS if arm != "A"},
        "protocol_sha256": sha256(OUT / "protocol.json")})
    print("REPORTED", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "features", "train", "predict", "analyze_sequence", "baseline", "report"))
    p.add_argument("--role", choices=("train", "val"))
    p.add_argument("--sequence")
    p.add_argument("--arm", choices=ARMS)
    args = p.parse_args()
    if args.action in ("prepare", "report"):
        globals()[args.action]()
    elif args.action == "train":
        train(args.arm)
    elif args.action == "baseline":
        analyze_sequence("val", args.sequence, baseline=True)
    else:
        globals()[args.action](args.role, args.sequence)
