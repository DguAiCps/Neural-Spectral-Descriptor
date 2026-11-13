#!/usr/bin/env python3
"""유사 간선의 품질·확률 하한과 최종 판정 성능을 진단한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    python3 scripts/_pdv2_edge_quality_run.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_graph_alignment as alignment
from _pdv2_pilot import TRAIN, VAL, OPS
from _pdv2_no_context_ablation import sha256, write_json, evaluate_keys
from _pdv2_lag15_report import read as read_ledger, sequence_metrics
from _pdv2_checkpoint_select import aggregate_rows, PRIMARY
from _pdv2_pair_evaluate import ArrayCache

CONFIG = REPO / "configs/pdv2_edge_quality.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
OUT = REPO / "results/pdv2_edge_quality_20261016"
MODELS = tuple(CFG["models"])
POLICIES = tuple(CFG["policies"])
BASE = alignment.base.BASE


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def folder(model: str, policy: str = "") -> Path:
    """실험 전용 출력 경로를 만든다."""
    assert model in MODELS and (not policy or policy in POLICIES)
    p = OUT / model / policy
    p.mkdir(parents=True, exist_ok=True)
    return p


def protocol(data: bool = False) -> dict:
    """코드·모델·자료 동결을 검사한다."""
    value = read(OUT / "protocol.json")
    for name, digest in value["runtime_sha256"].items():
        assert sha256(REPO / name) == digest, name
    if data:
        for name, digest in value["input_sha256"].items():
            assert sha256(REPO / name) == digest, name
    return value


def prepare() -> None:
    """추가 점수나 간선 품질을 보기 전에 조건·선택 범위를 고정한다."""
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    previous = alignment.protocol(True)
    selected = read(alignment.OUT / "summary.json")["arms"]
    models = {name: selected[arm] for name, arm in CFG["models"].items()}
    runtime = {REPO / p for p in previous["runtime_sha256"]}
    runtime.update(REPO / p for p in (
        "configs/pdv2_edge_quality.yaml", "scripts/_pdv2_edge_quality.py",
        "scripts/_pdv2_edge_quality_run.py", "scripts/_pdv2_edge_quality_verify.py",
        "scripts/_pdv2_edge_quality_plot.py", "tests/gnn/test_edge_quality.py",
        "scripts/_pdv2_candidate_learning.py", "scripts/_pdv2_pair_evaluate.py"))
    data = {REPO / p for p in previous["input_sha256"]}
    data.update((alignment.OUT / "summary.json", alignment.OUT / "completion_validation.json"))
    for value in models.values():
        data.add(REPO / value["checkpoint"])
        root = alignment.arm_root(next(a for a, r in selected.items() if r == value))
        data.update(p for p in (root / f"epoch{value['epoch']}").rglob("*") if p.is_file())
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "config": CFG, "models": models,
        "train_roster": [s + "/" + q for s, q in TRAIN],
        "val_roster": [s + "/" + q for s, q in VAL], "test_used": False,
        "fixed": "기존 학습/온라인 정합 학습의 검증선택 체크포인트를 고정한다. 재학습·분류기 재적합·첫 통과 변경 없음. 같은 KISS 키프레임, q+15, 과거 후보j<=q-30, 재계량·100+100·정렬·융합·판정점수·기준값 유지.",
        "intervention": "원래 둘째 통과에서 고른 최대10개 중 확률 하한0/.25/.5/.75/.9를 넘는 간선만 남긴다. 시간 간선과 남은 간선 속성은 그대로다. temporal은 같은 모델에서 유사 간선을 전부 제거한다. 별도 temporal 학습 모델의 결과도 참고한다.",
        "random_control": "random050은 노드마다 p050과 동일 개수만 원래10개에서 고정 해시 순서로 남긴다. 간선 품질과 개수 감소 효과를 분리하는 단일 무작위 대조이며 난수시드1601 고정.",
        "oracle": "oracle_true는 원래10개 중 GT 3D거리<=5m인 간선만 남기는 특권정보 진단이다. 실제 배포/선택/방법 성능 주장에 쓰지 않는다. 새 정답 간선을 추가하지 않으며 성능 상한이라고 가정하지 않는다.",
        "labels": "정답은 간선 선택 후 별도 진단에만 사용한다(oracle 예외). 최종과 같은3D<=5m/5~25m/>25m와 분류기 학습의 평면<5m/>=25m를 분리한다. 확률은 보정된 참 확률이라고 가정하지 않는다. 검증 진단은 공통질의30<=q<N-15에서 집계하며 비재방문/방향별로 나눈다. 학습 자료 진단은 기존 기준자세 키프레임으로 별도 집계한다.",
        "selection": "기존 모델별로 p000/p025/p050/p075/p090만 검증 시퀀스→3계열→NSD/H 평균 최종AP로 비교한다.1e-12동률이면 작은하한. random/temporal/oracle는 선택대상이 아니다. 같은 검증으로 고른 결과는 탐색 진단으로 표시한다.",
        "scope": "시드1 고정표현의 간선 제거 진단. 재학습된 조건의 비교나 시험 성능/통계적 유의성 주장이 아니다. 주표를 교체하지 않는다.",
        "runtime_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(runtime)},
        "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(data)},
    })
    print("PREPARED", len(runtime), len(data), flush=True)


def masks(query: np.ndarray, neighbor: np.ndarray, probability: np.ndarray,
          true_edge: np.ndarray) -> dict:
    """원래 선택 간선에 대한 제거 마스크를 만든다. 시간 간선은 받지 않는다."""
    assert query.shape == neighbor.shape == probability.shape == true_edge.shape
    assert np.all(np.diff(query) >= 0)
    result = {name: probability >= threshold for name, threshold in CFG["thresholds"].items()}
    random = np.zeros(len(query), dtype=bool)
    key = (query.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
           ^ neighbor.astype(np.uint64) ^ np.uint64(CFG["random_seed"]))
    key = (key ^ (key >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    key = (key ^ (key >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    key ^= key >> np.uint64(31)
    bounds = np.r_[np.flatnonzero(np.r_[True, np.diff(query) != 0]), len(query)] if len(query) else [0]
    for a, b in zip(bounds[:-1], bounds[1:]):
        count = int(result["p050"][a:b].sum())
        random[a + np.argsort(key[a:b], kind="stable")[:count]] = True
    result.update(random050=random, temporal=np.zeros(len(query), bool), oracle_true=true_edge.copy())
    return result


def load_model(name: str, device: str = "cuda") -> tuple:
    """고정한 선택 모델과 원래 분류기·재계량을 읽는다."""
    import torch
    spec = protocol()["models"][name]
    bundle = alignment.base.model(1, device=device)
    state = torch.load(REPO / spec["checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"]
    assert sha256(REPO / spec["checkpoint"]) == spec["checkpoint_sha256"]
    missing, extra = bundle[1].load_state_dict(state, strict=False)
    assert missing == ["gnn.key_scale"] and not extra
    assert all(torch.equal(bundle[1].state_dict()[k].cpu(), v) for k, v in state.items())
    bundle[1].eval()
    return bundle


def capture(cache, sensor: str, layout: np.ndarray, bundle: tuple, device: str) -> tuple:
    """운영 함수를 그대로 실행하면서 선택 전 후보와 두 그래프를 보존한다."""
    import _lag_branch_fusion as lag
    cfg, net, clf, blob, rvec, _ = bundle
    candidates, probabilities, chosen, graphs = [], [], [], []
    old_features, old_select = lag.causal_edge_features, lag.select_edges_per_node

    def features(*args, **kwargs):
        assert kwargs.get("want_labels") is False
        value = old_features(*args, **kwargs)
        candidates.append((value[0], value[1]))
        return value

    def select(src, probs, keep_k=10):
        probabilities.append(probs.copy())
        value = old_select(src, probs, keep_k)
        chosen.append(value)
        return value

    lag.causal_edge_features, lag.select_edges_per_node = features, select
    hook = net.register_forward_pre_hook(lambda module, args: graphs.append(args[0]))
    try:
        embedding = lag.nsd_embedding_lag(cache, sensor, cache["descriptors"], layout,
                                          cfg, net, clf, blob, rvec, device)
    finally:
        hook.remove()
        lag.causal_edge_features, lag.select_edges_per_node = old_features, old_select
    assert len(graphs) == 2 and len(candidates) == len(probabilities) == len(chosen) == 1
    src, dst = candidates[0]
    keep = np.zeros(len(src), bool)
    keep[chosen[0]] = True
    return embedding, graphs, {"query": src, "neighbor": dst, "probability": probabilities[0],
                               "selected": keep, "selected_order": chosen[0]}


def edge_counts(q: np.ndarray, d3: np.ndarray, d2: np.ndarray, p: np.ndarray,
                eligible_nodes: np.ndarray) -> dict:
    """간선과 노드의 분모를 명시한 품질 집계를 만든다."""
    valid = eligible_nodes[q]
    q, d3, d2, p = q[valid], d3[valid], d2[valid], p[valid]
    return {"edges": len(q), "positive_3d": int((d3 <= 5).sum()),
            "ambiguous_3d": int(((d3 > 5) & (d3 <= 25)).sum()), "far_3d": int((d3 > 25).sum()),
            "positive_xy": int((d2 < 5).sum()), "negative_xy": int((d2 >= 25).sum()),
            "probability_sum": float(p.astype(np.float64).sum()),
            "nodes": int(eligible_nodes.sum()), "nodes_with_edge": len(np.unique(q)),
            "nodes_with_true_edge": len(np.unique(q[d3 <= 5]))}


def diagnose(candidate: dict, first_graph, gt: np.ndarray, groups: np.ndarray,
             valid: np.ndarray) -> dict:
    """선택이 끝난 뒤에만 GT로 후보/선택 간선의 품질을 계산한다."""
    q, j, p = (candidate[k] for k in ("query", "neighbor", "probability"))
    delta = gt[q, :3, 3] - gt[j, :3, 3]
    d3, d2 = np.linalg.norm(delta, axis=1), np.linalg.norm(delta[:, :2], axis=1)
    order = candidate["selected_order"]
    policy = masks(q[order], j[order], p[order], d3[order] <= 5)
    groups_named = {"all": valid, **{name: valid & (groups == code) for code, name in enumerate(
        ("same_available", "opposite_only", "other_positive", "negative", "ambiguous"))}}
    result = {"groups": {}, "probability_bins": {}, "policy": {}}
    for group, eligible in groups_named.items():
        result["groups"][group] = {
            "candidates30": edge_counts(q, d3, d2, p, eligible),
            "selected10": edge_counts(q[order], d3[order], d2[order], p[order], eligible)}
    for label, select in (("candidates30", np.ones(len(q), bool)), ("selected10", candidate["selected"])):
        result["probability_bins"][label] = []
        for lo, hi in zip(CFG["probability_bins"][:-1], CFG["probability_bins"][1:]):
            take = select & (p >= lo) & (p < hi)
            result["probability_bins"][label].append({"lo": lo, "hi": min(hi, 1.),
                **edge_counts(q[take], d3[take], d2[take], p[take], valid)})
    for name, take in policy.items():
        ids = order[take]
        result["policy"][name] = {group: edge_counts(q[ids], d3[ids], d2[ids], p[ids], eligible)
                                  for group, eligible in groups_named.items()}
    index = first_graph.edge_index[:, first_graph.edge_type == 1].cpu().numpy()
    ds = gt[index[1], :3, 3] - gt[index[0], :3, 3]
    result["first_pass"] = edge_counts(index[1], np.linalg.norm(ds, axis=1),
                                        np.linalg.norm(ds[:, :2], axis=1), np.ones(index.shape[1]), valid)
    candidate.update(distance_3d=d3, distance_xy=d2)
    return result


def filter_graph(graph, keep: np.ndarray):
    """시간 간선과 보존 간선의 속성을 그대로 유지한다."""
    import torch
    similar = graph.edge_type == 1
    assert len(keep) == int(similar.sum())
    mask = ~similar
    mask[similar] = torch.from_numpy(keep).to(mask.device)
    result = copy.copy(graph)
    for key in ("edge_index", "edge_attr", "edge_type"):
        value = getattr(graph, key)
        setattr(result, key, value[:, mask] if key == "edge_index" else value[mask])
    assert torch.equal(result.edge_attr[result.edge_type == 0], graph.edge_attr[~similar])
    return result


def extract(name: str, role: str, sequence: str) -> None:
    """공통 첫 통과를 재사용해 여덟 제거 조건의 키와 품질을 저장한다."""
    import torch
    from _xs_eval_lib import normalize
    from _pdv2_pair_evaluate import direction_groups
    protocol()
    assert role in ("train", "val") and sequence in protocol()[role + "_roster"]
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sensor, seq = sequence.split("/")
    target = folder(name) / f"quality_{role}_{sensor}_{seq}.json"
    if target.exists():
        return
    if role == "val":
        _, _, paths, min_past = alignment.base.inputs("val", sequence)
        with np.load(paths["cache"]) as z:
            cache = ArrayCache({k: z[k] for k in z.files if k != "poses_gt"})
            gt = z["poses_gt"]
        with np.load(paths["layouts"]) as z:
            np.testing.assert_array_equal(cache["scan_ids"], z["scan_ids"])
            layout = z["nsd_layouts"]
        ledger, _ = read_ledger(BASE / f"s1/val_selected_symmetric_{OPS[0]}_{sensor}_{seq}.npz")
        groups = np.full(len(gt), -1, np.int8)
        groups[ledger["query_idx"]] = ledger["group"]
    else:
        from _pdv2_candidate_learning import nearest_past
        import _pd_build_edge_classifier as data
        data.PD_CACHE = REPO / "data/preprocessed_v2"
        path = alignment.cache_path(sequence)
        with np.load(path) as z:
            cache = ArrayCache({k: z[k] for k in z.files})
        gt = cache["poses"].copy()
        layout = data._layouts(sensor, seq, cache)
        min_past = nearest_past(gt[:, :3, 3])
        groups = direction_groups(gt[:, :3, 3], min_past)
    groups[min_past > 25] = 3
    groups[(min_past > 5) & (min_past <= 25)] = 4
    valid = (np.arange(len(gt)) >= 30) & (np.arange(len(gt)) < len(gt) - 15)
    bundle = load_model(name)
    t = time.time()
    baseline, graphs, candidates = capture(cache, sensor, layout, bundle, "cuda")
    quality = diagnose(candidates, graphs[0], gt, groups, valid)
    edge_file = folder(name) / f"edges_{role}_{sensor}_{seq}.npz"
    np.savez_compressed(edge_file, **candidates, node_group=groups, valid_node=valid)
    if role == "val":
        order = candidates["selected_order"]
        q, j, p = (candidates[k][order] for k in ("query", "neighbor", "probability"))
        policy = masks(q, j, p, candidates["distance_3d"][order] <= 5.)
        index = graphs[1].edge_index[:, graphs[1].edge_type == 1].cpu().numpy()
        np.testing.assert_array_equal(index, np.stack([j, q]))
        np.testing.assert_array_equal(graphs[1].edge_attr[graphs[1].edge_type == 1, 4].cpu().numpy(), p)
        for policy_name in POLICIES:
            if policy_name == "p000":
                result = baseline
            else:
                graph = filter_graph(graphs[1], policy[policy_name])
                with torch.no_grad():
                    result = normalize(bundle[1](graph).float().cpu().numpy())
            assert result.shape == (len(gt), 416) and np.isfinite(result).all()
            meta = {"model": name, "policy": policy_name, "protocol_sha256": sha256(OUT / "protocol.json"),
                    "checkpoint_sha256": protocol()["models"][name]["checkpoint_sha256"],
                    "selected_edges": int(policy[policy_name].sum()), "oracle": policy_name == "oracle_true"}
            np.savez_compressed(folder(name, policy_name) / f"keys_{sensor}_{seq}.npz",
                                embedding=result, scan_ids=cache["scan_ids"], meta=json.dumps(meta))
        old_arm = CFG["models"][name]
        epoch = protocol()["models"][name]["epoch"]
        with np.load(alignment.folder(1, old_arm, epoch) / f"keys_val_{sensor}_{seq}.npz") as z:
            quality["previous_key_max_error"] = float(np.abs(baseline - z["embedding"]).max())
    quality.update(model=name, role=role, sequence=sequence, nodes=len(gt),
                   runtime_s=time.time() - t, edge_file=str(edge_file.relative_to(REPO)),
                   edge_sha256=sha256(edge_file), protocol_sha256=sha256(OUT / "protocol.json"))
    write_json(target, quality)
    print("EXTRACTED", name, role, sequence, round(time.time() - t, 2), flush=True)


def score(name: str, policy: str, sequence: str) -> None:
    """간선 품질과 별도로 최종 검색·매칭·수락 지표를 계산한다."""
    protocol()
    sensor, seq, paths, min_past = alignment.base.inputs("val", sequence)
    target = folder(name, policy) / f"val_{sensor}_{seq}.json"
    if target.exists():
        return
    with np.load(target.parent / f"keys_{sensor}_{seq}.npz") as z:
        embedding, scan_ids = z["embedding"], z["scan_ids"]
    with np.load(paths["cache"]) as z:
        np.testing.assert_array_equal(scan_ids, z["scan_ids"])
        gt = z["poses_gt"][:, :3, 3]
    selection = read(BASE / "s1/confidence_selection.json")["ops"]
    result = {}
    for op in OPS:
        with np.load(paths["spec"]) as z:
            np.testing.assert_array_equal(scan_ids, z["scan_ids"])
            bank = alignment.base.bank_for(op, z)
        sel = selection[op]
        arrays = evaluate_keys(embedding, bank, gt, min_past,
                               {1: tuple(float(w) for w in sel["tuple"])})[1]
        keep = arrays["query_idx"] < len(gt) - 15
        arrays = {k: v[keep] for k, v in arrays.items()}
        ref, _ = read_ledger(BASE / f"s1/val_selected_symmetric_{op}_{sensor}_{seq}.npz")
        for key in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(arrays[key], ref[key])
        arrays["group"] = ref["group"]
        dest = target.parent / f"val_{op}_{sensor}_{seq}.npz"
        np.savez_compressed(dest, **arrays, meta=json.dumps({"protocol_sha256": sha256(OUT / "protocol.json")}))
        result[op] = {"metrics": sequence_metrics(arrays, sel["decision"], sel["tau_star"]),
                      "ledger": str(dest.relative_to(REPO)), "sha256": sha256(dest)}
    write_json(target, {"model": name, "policy": policy, "sequence": sequence, "ops": result,
               "protocol_sha256": sha256(OUT / "protocol.json")})


def add_counts(rows: list) -> dict:
    """분모를 합친 뒤 품질 비율을 계산한다."""
    result = {k: sum(r[k] for r in rows) for k in rows[0]}
    n = result["edges"]
    for k in ("positive_3d", "far_3d", "positive_xy", "negative_xy"):
        result[k + "_fraction"] = result[k] / n if n else None
    result["mean_probability"] = result["probability_sum"] / n if n else None
    return result


def report() -> None:
    """고정표현 비교·간선 품질·특권 진단을 구분해 집계한다."""
    protocol(True)
    quality, scores, chosen = {}, {}, {}
    for name in MODELS:
        scores[name] = {}
        quality[name] = {}
        for role, roster in (("train", TRAIN), ("val", VAL)):
            qrows = [read(folder(name) / f"quality_{role}_{s}_{q}.json") for s, q in roster]
            quality[name][role] = {"groups": {}, "probability_bins": {}, "policy": {}}
            for group in qrows[0]["groups"]:
                quality[name][role]["groups"][group] = {kind: add_counts([r["groups"][group][kind] for r in qrows])
                                                        for kind in ("candidates30", "selected10")}
            for kind in ("candidates30", "selected10"):
                quality[name][role]["probability_bins"][kind] = []
                for i, row in enumerate(qrows[0]["probability_bins"][kind]):
                    summed = add_counts([{k: v for k, v in r["probability_bins"][kind][i].items() if k not in ("lo", "hi")} for r in qrows])
                    quality[name][role]["probability_bins"][kind].append({"lo": row["lo"], "hi": row["hi"], **summed})
            for policy in POLICIES:
                quality[name][role]["policy"][policy] = {g: add_counts([r["policy"][policy][g] for r in qrows])
                                                        for g in qrows[0]["policy"][policy]}
            quality[name][role]["first_pass"] = add_counts([r["first_pass"] for r in qrows])
        for policy in POLICIES:
            rows = {s + "/" + q: read(folder(name, policy) / f"val_{s}_{q}.json") for s, q in VAL}
            ops = {op: aggregate_rows(rows, "val", op) for op in OPS}
            for op in OPS:
                directions = {g: {"n": 0, "correct": 0, "recovered": 0, "regressed": 0}
                              for g in ("same_available", "opposite_only", "other_positive")}
                for row in rows.values():
                    arrays, _ = read_ledger(REPO / row["ops"][op]["ledger"])
                    s, q = row["sequence"].split("/")
                    before, _ = read_ledger(folder(name, "p000") / f"val_{op}_{s}_{q}.npz")
                    old, new = before["top1_geo_dist"] <= 5., arrays["top1_geo_dist"] <= 5.
                    for i, group in enumerate(directions):
                        mask = (arrays["min_past_gt_dist"] <= 5.) & (arrays["group"] == i)
                        d = directions[group]
                        d["n"] += int(mask.sum())
                        d["correct"] += int((mask & new).sum())
                        d["recovered"] += int((mask & ~old & new).sum())
                        d["regressed"] += int((mask & old & ~new).sum())
                ops[op]["directions"] = directions
                ops[op]["negative_false_accepts"] = sum(r["ops"][op]["metrics"]["negative_false_accepts"] for r in rows.values())
            scores[name][policy] = {"ops": ops, "joint_AP": float(np.mean([r["macro"]["AP"] for r in ops.values()]))}
        best = max(scores[name][p]["joint_AP"] for p in CFG["selection_policies"])
        chosen[name] = next(p for p in CFG["selection_policies"] if best - scores[name][p]["joint_AP"] <= 1e-12)
    write_json(OUT / "quality_summary.json", quality)
    write_json(OUT / "summary.json", {"scores": scores, "chosen_threshold": chosen,
               "trained_temporal_reference": read(alignment.OUT / "summary.json")["arms"]["diff_temporal"],
               "test_used": False, "protocol_sha256": sha256(OUT / "protocol.json"),
               "scope": "검증셋 고정모델 간선 제거 진단. oracle_true는 GT를 쓰는 특권 진단이며 선택 제외."})
    print("SELECTED", chosen, flush=True)
    print("SCORES", {n: {p: r["joint_AP"] for p, r in rows.items()} for n, rows in scores.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "extract", "score", "report"))
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--policy", choices=POLICIES)
    p.add_argument("--role", choices=("train", "val"), default="val")
    p.add_argument("--sequence")
    a = p.parse_args()
    if a.action == "prepare":
        prepare()
    elif a.action == "extract":
        extract(a.model, a.role, a.sequence)
    elif a.action == "score":
        score(a.model, a.policy, a.sequence)
    else:
        report()
