#!/usr/bin/env python3
"""고정 모델의 유사 간선 손실을 연결·속성·검색/순위 단계로 분해한다."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_edge_labels3d as base
from _pdv2_no_context_ablation import sha256, write_json, evaluate_keys

OUT = REPO / "results/pdv2_similarity_causes_20261016"
CONFIG = REPO / "configs/pdv2_similarity_causes.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
ARMS = tuple(CFG["interventions"])
REFS = tuple(CFG["reference_arms"])
VAL, OPS, PRIMARY = base.VAL, base.OPS, base.PRIMARY
GROUPS = ("same_available", "opposite_only", "other_positive", "negative", "ambiguous")


def folder(arm: str = "") -> Path:
    """진단 전용 저장 폴더를 만든다."""
    p = OUT / arm
    p.mkdir(parents=True, exist_ok=True)
    return p


def protocol(data: bool = False) -> dict:
    """동결한 실행 파일과 자료를 확인한다."""
    value = base.read(OUT / "protocol.json")
    for k, h in value["runtime_sha256"].items():
        assert sha256(REPO / k) == h, k
    if data:
        for k, h in value["input_sha256"].items():
            assert sha256(REPO / k) == h, k
    return value


def prepare() -> None:
    """검증 전 진단 조건을 동결한다. 재학습·설정 선택·시험 사용은 없다."""
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    parent = base.protocol(True)
    runtime = {REPO / p for p in parent["runtime_sha256"]}
    runtime.update((CONFIG, Path(__file__), REPO / "scripts/_pdv2_similarity_causes_run.py",
                    REPO / "tests/gnn/test_similarity_causes.py"))
    inputs = {REPO / p for p in parent["input_sha256"]}
    inputs.update(base.OUT / p for p in ("protocol.json", "summary.json", "completion_validation.json"))
    for s, q in VAL:
        inputs.add(base.OUT / "main" / f"edges_{s}_{q}.npz")
        for arm in (*REFS, "original"):
            root = base.OUT / "main" / arm
            inputs.update(root / f for f in (f"keys_{s}_{q}.npz", f"val_{s}_{q}.json"))
            inputs.update(root / f"val_{op}_{s}_{q}.npz" for op in OPS)
    inputs.update(base.OUT / "classifiers" / ("xyz" + ext) for ext in (".pt", ".json"))
    write_json(folder() / "protocol.json", {
        "created_unix": time.time(), "config": CFG, "test_used": False,
        "fixed": "주평가 시드1 GNN과 3D 라벨 분류기 고정. 저장된 첫통과 후보30·점수·상위10·KISS 검증9 사용. 시간간선, 원래 d 입력, 재계량, q+15, j<=q-30, 100+100, 정렬·융합·수락 규칙 고정.",
        "oracle": "gt_true_only는 선택10 중 GT3D<=5m, gt_nontrue_only는 나머지(모호구간 포함), gt_random_matched는 노드별 gt_true_only와 같은 개수의 임의 부분집합이다. 세 조건 모두 특권정보 진단이며 배포/선택/상한 주장이 아니다. 무작위는 단일 추출이다.",
        "non_oracle": "cosine_top10은 같은30 중 첫통과코사인 상위10으로 ID만 바꾸고 해당 분류기 점수는 유지한다. probability_one은 같은10의 다섯째 속성만1로 한다. raw_key_attributes는 같은10·점수를 유지하며 코사인/L2 속성만 학습그래프처럼 원래288d에서 계산한다. 학습분포 전체를 일치시키는 실험은 아니다.",
        "cross": "temporal/xyz 검색키로 후보색인을 만들고 temporal/xyz 키로 융합의 키 점수를 계산하는 2x2 교차. 양 끝은 저장 최종원장 재현 검사를 한다. 후보변경에는 minmax 정규화 집합 변화도 포함한다.",
        "attention": "실제 두 층의 정규화된 attention을 기록한다. 이 값만으로 인과 효과를 주장하지 않는다.",
        "ledger": "temporal->original/xyz AP차이를 정답유형별 Shapley 회계로 분해한다. 회계는 원인 추정이 아니다.",
        "selection": "단일시드 검증 원인진단. 모든 조건을 보고하며 최고 설정 선택·추가 학습·시험 평가·주표 교체 없음.",
        "runtime_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(runtime)},
        "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted(inputs)}})
    print("PREPARED", len(runtime), len(inputs), flush=True)


def intervention(c: dict, raw: np.ndarray, arm: str) -> tuple:
    """공통 선택 간선에서 한 요소만 개입한 후보·점수·인덱스를 반환한다."""
    from _causal_branch_fusion import select_edges_per_node
    p, ids = c["prob_xyz"], c["selected_xyz"]
    cand = c
    if arm == "temporal":
        ids = np.empty(0, np.int64)
    elif arm == "gt_true_only":
        ids = ids[c["distance_3d"][ids] <= 5]
    elif arm == "gt_nontrue_only":
        ids = ids[c["distance_3d"][ids] > 5]
    elif arm == "gt_random_matched":
        q = c["query"][ids]
        n = len(raw)
        counts = np.bincount(q[c["distance_3d"][ids] <= 5], minlength=n)
        order = np.lexsort((np.random.default_rng(CFG["random_seed"]).random(len(ids)), q))
        sq = q[order]
        rank = np.arange(len(ids)) - np.repeat(np.r_[0, np.cumsum(np.bincount(sq, minlength=n))[:-1]], np.bincount(sq, minlength=n))
        ids = ids[order[rank < counts[sq]]]
    elif arm == "cosine_top10":
        ids = select_edges_per_node(c["query"], c["features"][:, 0], 10)
    elif arm == "probability_one":
        p = np.ones_like(p)
    elif arm == "raw_key_attributes":
        # 원래 288차원의 크기를 유지한 L2이며 코사인 계산만 정규화한다.
        q, j = c["query"], c["neighbor"]
        norm = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
        features = c["features"].copy()
        l2 = c["l2"].copy()
        for start in range(0, len(ids), 8192):
            take = ids[start:start + 8192]
            features[take, 0] = np.einsum("ij,ij->i", norm[q[take]], norm[j[take]])
            l2[take] = np.linalg.norm(raw[q[take]] - raw[j[take]], axis=1)
        cand = {**c, "features": features, "l2": l2}
    else:
        assert arm == "xyz", arm
    assert ids.dtype == np.int64 and np.all(c["neighbor"][ids] <= c["query"][ids] - 30)
    return cand, p, ids


def infer_attention(net, graph, groups: np.ndarray, valid: np.ndarray, gt: np.ndarray) -> tuple:
    """실제 forward를 유지하며 각 층의 간선 종류별 attention 질량을 잰다."""
    import torch
    from _xs_eval_lib import normalize
    captured, originals = [], []
    for layer in net.gnn.convs:
        original = layer.forward
        originals.append(original)

        def tapped(*args, _original=original, **kwargs):
            kwargs["return_attention_weights"] = True
            out, (index, weights) = _original(*args, **kwargs)
            captured.append((index.detach().cpu().numpy(), weights.detach().cpu().numpy()))
            return out

        layer.forward = tapped
    try:
        with torch.no_grad():
            emb = normalize(net(graph).float().cpu().numpy())
    finally:
        for layer, original in zip(net.gnn.convs, originals):
            layer.forward = original
    edge_type = graph.edge_type.cpu().numpy()
    stats = []
    for index, weights in captured:
        src, dst = index
        w = weights.mean(1).astype(np.float64)
        n = len(groups)
        total = np.bincount(dst, weights=w, minlength=n)
        np.testing.assert_allclose(total, np.ones(n), atol=5e-7, rtol=0)
        d = np.linalg.norm(gt[src] - gt[dst], axis=1)
        masks = {"temporal": edge_type == 0, "similarity": edge_type == 1,
                 "similarity_true": (edge_type == 1) & (d <= 5),
                 "similarity_far": (edge_type == 1) & (d > 25),
                 "similarity_ambiguous": (edge_type == 1) & (d > 5) & (d <= 25)}
        node_mass = {k: np.bincount(dst[m], weights=w[m], minlength=n) for k, m in masks.items()}
        stats.append({name: {"nodes": int(mask.sum()),
                            **{k + "_mass_sum": float(v[mask].sum()) for k, v in node_mass.items()}}
                      for name, mask in [("all", valid), *[(g, valid & (groups == i)) for i, g in enumerate(GROUPS)]]})
    return emb, stats


def keys(sequence: str) -> None:
    """같은 시간 그래프에서 연결·속성 개입의 최종 검색 키를 계산한다."""
    import torch
    from _lag_branch_fusion import lag_pass1_graph
    protocol()
    assert sequence in [s + "/" + q for s, q in VAL]
    s, q, paths, _ = base.old.alignment.base.inputs("val", sequence)
    done = OUT / f"keys_{s}_{q}.json"
    if done.exists():
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(paths["cache"]) as z:
        cache = base.old.ArrayCache({k: z[k] for k in z.files if k != "poses_gt"})
        gt = z["poses_gt"][:, :3, 3]
    with np.load(base.OUT / "main" / f"edges_{s}_{q}.npz") as z:
        c = {k: z[k] for k in z.files}
    cfg, net, *_ = base.load_model("main")
    cfg = copy.deepcopy(cfg)
    cfg["keyframe"].setdefault("graph", {})["similarity_threshold"] = 2.0
    raw = cache["descriptors"].astype(np.float32)
    graph = lag_pass1_graph(cache, raw, cfg, "cuda", s)
    assert not bool((graph.edge_type == 1).any())
    record = {"sequence": sequence, "arms": {}, "reference_max_error": {}}
    for arm in (*REFS, *ARMS):
        cand, prob, ids = intervention(c, raw, arm)
        changed = base.replace_edges(graph, cand, prob, ids)
        emb, attn = infer_attention(net, changed, c["group"], c["valid"], gt)
        assert emb.shape == (len(raw), 416) and np.isfinite(emb).all()
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1, atol=2e-7, rtol=0)
        if arm in REFS:
            with np.load(base.OUT / "main" / arm / f"keys_{s}_{q}.npz") as z:
                error = float(np.max(np.abs(emb - z["embedding"])))
                np.testing.assert_allclose(emb, z["embedding"], atol=1e-6, rtol=0)
            record["reference_max_error"][arm] = error
        else:
            dest = folder(arm) / f"keys_{s}_{q}.npz"
            np.savez_compressed(dest, embedding=emb, scan_ids=cache["scan_ids"])
        record["arms"][arm] = {"edges": len(ids), "attention": attn,
            "quality": {g: base.old.edge_counts(c["query"][ids], c["distance_3d"][ids], c["distance_xy"][ids], prob[ids], c["valid"] & (c["group"] == i))
                        for i, g in enumerate(GROUPS)}}
        if arm not in REFS:
            record["arms"][arm]["keys_sha256"] = sha256(dest)
        print("KEY", sequence, arm, len(ids), flush=True)
    write_json(done, record)


def save_scores(arm: str, sequence: str, arrays_by_op: dict) -> dict:
    """최종 판정 원장과 공통 지표를 저장한다."""
    s, q = sequence.split("/")
    selection = base.read(base.old.BASE / "s1/confidence_selection.json")["ops"]
    result = {}
    for op, arrays in arrays_by_op.items():
        sel = selection[op]
        dest = folder(arm) / f"val_{op}_{s}_{q}.npz"
        np.savez_compressed(dest, **arrays, meta=json.dumps({"protocol_sha256": sha256(OUT / "protocol.json")}))
        result[op] = {"metrics": base.old.sequence_metrics(arrays, sel["decision"], sel["tau_star"]),
                      "ledger": str(dest.relative_to(REPO)), "sha256": sha256(dest)}
    row = {"arm": arm, "sequence": sequence, "ops": result}
    write_json(folder(arm) / f"val_{s}_{q}.json", row)
    return row


def score(arm: str, sequence: str) -> None:
    """추가 간선 개입을 중간 검색률이 아닌 최종 판정으로 평가한다."""
    protocol()
    s, q, paths, min_past = base.old.alignment.base.inputs("val", sequence)
    if (folder(arm) / f"val_{s}_{q}.json").exists():
        return
    with np.load(folder(arm) / f"keys_{s}_{q}.npz") as z:
        embedding, scan_ids = z["embedding"], z["scan_ids"]
    with np.load(paths["cache"]) as z:
        gt = z["poses_gt"][:, :3, 3]
        np.testing.assert_array_equal(scan_ids, z["scan_ids"])
    selection = base.read(base.old.BASE / "s1/confidence_selection.json")["ops"]
    result = {}
    for op in OPS:
        with np.load(paths["spec"]) as z:
            np.testing.assert_array_equal(scan_ids, z["scan_ids"])
            bank = base.old.alignment.base.bank_for(op, z)
        a = evaluate_keys(embedding, bank, gt, min_past, {1: tuple(float(w) for w in selection[op]["tuple"])})[1]
        keep = a["query_idx"] < len(gt) - 15
        a = {k: v[keep] for k, v in a.items()}
        ref, _ = base.old.read_ledger(base.OUT / "main/xyz" / f"val_{op}_{s}_{q}.npz")
        np.testing.assert_array_equal(a["query_idx"], ref["query_idx"])
        a["group"] = ref["group"]
        result[op] = a
    save_scores(arm, sequence, result)


def crossed(sequence: str) -> None:
    """후보검색 키와 최종 융합 키를 2x2로 교차하여 양 끝을 검증한다."""
    from _causal_pr_dump import prefix_topk
    from _xs_eval_lib import align_scores, column_align_scores, minmax
    protocol()
    s, q, paths, _ = base.old.alignment.base.inputs("val", sequence)
    done = OUT / f"cross_{s}_{q}.json"
    if done.exists():
        return
    selection = base.read(base.old.BASE / "s1/confidence_selection.json")["ops"]
    embeddings = {}
    for arm in REFS:
        with np.load(base.OUT / "main" / arm / f"keys_{s}_{q}.npz") as z:
            embeddings[arm] = z["embedding"]
            scan_ids = z["scan_ids"]
    with np.load(paths["cache"]) as z:
        gt = z["poses_gt"][:, :3, 3]
        np.testing.assert_array_equal(scan_ids, z["scan_ids"])
    results = {f"pool_{a}_score_{b}": {} for a in REFS for b in REFS}
    checks = {}
    for op in OPS:
        refs = {arm: base.old.read_ledger(base.OUT / "main" / arm / f"val_{op}_{s}_{q}.npz")[0] for arm in REFS}
        ref = refs["xyz"]
        weights = [float(w) for w in selection[op]["tuple"]]
        with np.load(paths["spec"]) as z:
            hkey, hfft, hnorm, column = base.old.alignment.base.bank_for(op, z)
        arrays = {arm: {k: v.copy() for k, v in ref.items()} for arm in results}
        for qi, query in enumerate(ref["query_idx"]):
            query = int(query)
            aux = prefix_topk(hkey, query, query - 29, 100)
            ranked = {a: prefix_topk(emb, query, query - 29, 100) for a, emb in embeddings.items()}
            pools = {a: np.unique(np.r_[ids, aux]) for a, ids in ranked.items()}
            union = np.union1d(*pools.values())
            raw_a = align_scores(hfft, hnorm, query, union)
            raw_c = column_align_scores(column, query, union)
            for pool, candidates in pools.items():
                indexes = np.searchsorted(union, candidates)
                a, c = raw_a[indexes], raw_c[indexes]
                d = np.linalg.norm(gt[candidates] - gt[query], axis=1)
                kd = np.linalg.norm(gt[ranked[pool]] - gt[query], axis=1)
                common = {"n_candidates": len(candidates), "key_top1_geo": kd[0], "key_hit": np.any(kd <= 5),
                          "aux_hit": np.any(np.linalg.norm(gt[aux] - gt[query], axis=1) <= 5), "union_hit": np.any(d <= 5)}
                for scoring, emb in embeddings.items():
                    fusion = (weights[0] * minmax(emb[candidates] @ emb[query]) +
                              weights[1] * minmax(hkey[candidates] @ hkey[query]) +
                              weights[2] * minmax(a) + weights[3] * minmax(c))
                    assert fusion.dtype == np.float32
                    order = np.argsort(-fusion, kind="stable")
                    best = int(order[0])
                    j = int(candidates[best])
                    values = {**common, "j_star": j, "top1_geo_dist": d[best], "c_spec": a[best], "c_col": c[best],
                              "score_abs": emb[query] @ emb[j], "score_margin": fusion[best] - fusion[order[1]] if len(order) > 1 else 0.}
                    for field, value in values.items():
                        arrays[f"pool_{pool}_score_{scoring}"][field][qi] = value
        checks[op] = {}
        for arm in REFS:
            a, b = arrays[f"pool_{arm}_score_{arm}"], refs[arm]
            np.testing.assert_array_equal(a["j_star"], b["j_star"])
            errors = {}
            for field in a:
                if a[field].dtype.kind in "biu":
                    np.testing.assert_array_equal(a[field], b[field])
                else:
                    np.testing.assert_allclose(a[field], b[field], atol=2e-6, rtol=0)
                    errors[field] = float(np.max(np.abs(a[field] - b[field])))
            checks[op][arm] = errors
        for arm, a in arrays.items():
            results[arm][op] = a
    for arm, arrays in results.items():
        save_scores(arm, sequence, arrays)
    write_json(done, {"sequence": sequence, "endpoint_checks": checks})
    print("CROSS", sequence, flush=True)


def ledger() -> None:
    """저장 최종 원장의 정답 탈락과 AP 차이를 질의 유형별로 집계한다."""
    from _pr_eval_lib import classify_queries, pr_sweep
    from _pdv2_pilot import FAMILY
    from _pdv2_context_attribution import transitions
    protocol()
    categories = ("recovered", "regressed", "both_correct", "both_wrong", "negative")
    out = {}
    selection = base.read(base.old.BASE / "s1/confidence_selection.json")["ops"]
    for after in ("original", "xyz"):
        out[after] = {}
        for op in OPS:
            rows, counts = {}, {}
            for s, q in VAL:
                a, b = [base.old.read_ledger(base.OUT / "main" / arm / f"val_{op}_{s}_{q}.npz")[0] for arm in ("temporal", after)]
                for key in ("query_idx", "min_past_gt_dist", "group"):
                    np.testing.assert_array_equal(a[key], b[key])
                qc = classify_queries(a["min_past_gt_dist"])
                good0, good1 = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
                masks = [qc["positive"] & ~good0 & good1, qc["positive"] & good0 & ~good1,
                         qc["positive"] & good0 & good1, qc["positive"] & ~good0 & ~good1, qc["negative"]]
                for name, mask in [("positive", qc["positive"]), *[(g, qc["positive"] & (a["group"] == i)) for i, g in enumerate(GROUPS[:3])]]:
                    count = transitions(a, b, mask)
                    target = counts.setdefault(name, {k: 0 for k in count})
                    for k, v in count.items():
                        target[k] += v
                if not qc["n_pos"]:
                    continue
                decision = selection[op]["decision"]
                values = {}
                for subset in range(32):
                    mask = np.zeros(len(good0), bool)
                    for i, m in enumerate(masks):
                        if subset & (1 << i):
                            mask |= m
                    values[subset] = float(pr_sweep(np.where(mask, b[decision], a[decision]),
                                                  np.where(mask, b["top1_geo_dist"], a["top1_geo_dist"]), qc)["AP"])
                shares = {}
                for i, name in enumerate(categories):
                    shares[name] = sum(math.factorial(bin(k).count("1")) * math.factorial(4 - bin(k).count("1")) / 120 *
                                       (values[k | (1 << i)] - values[k]) for k in range(32) if not k & (1 << i))
                assert abs(sum(shares.values()) - (values[31] - values[0])) < 1e-12
                rows[s + "/" + q] = {"family": FAMILY.get(s, s), "before": values[0], "after": values[31], **shares}
            families = {r["family"] for r in rows.values()}
            macro = {k: float(np.mean([np.mean([r[k] for r in rows.values() if r["family"] == f]) for f in families])) for k in ("before", "after", *categories)}
            out[after][op] = {"macro": macro, "transitions": counts, "per_sequence": rows}
    write_json(OUT / "ledger_diagnosis.json", out)


def report() -> None:
    """모든 조건을 보고하고 저장 원장에서 최종 지표를 다시 계산한다."""
    protocol(True)
    selection = base.read(base.old.BASE / "s1/confidence_selection.json")["ops"]
    arms = (*REFS, *ARMS, *[f"pool_{a}_score_{b}" for a in REFS for b in REFS])
    scores, verified, nrows = {}, 0, 0
    for arm in arms:
        root = base.OUT / "main" / arm if arm in REFS else OUT / arm
        rows = {s + "/" + q: base.read(root / f"val_{s}_{q}.json") for s, q in VAL}
        for sequence, row in rows.items():
            s, q = sequence.split("/")
            for op in OPS:
                r = row["ops"][op]
                path = REPO / r["ledger"]
                assert sha256(path) == r["sha256"]
                a, _ = base.old.read_ledger(path)
                ref, _ = base.old.read_ledger(base.OUT / "main/xyz" / f"val_{op}_{s}_{q}.npz")
                for k in ("query_idx", "min_past_gt_dist", "group"):
                    np.testing.assert_array_equal(a[k], ref[k])
                assert np.all(a["j_star"] <= a["query_idx"] - 30)
                fresh = base.old.sequence_metrics(a, selection[op]["decision"], selection[op]["tau_star"])
                assert fresh == r["metrics"], (arm, sequence, op)
                verified += 1
                nrows += len(a["query_idx"])
        scores[arm] = {op: base.old.aggregate_rows(rows, "val", op) for op in OPS}
    attention, quality = {}, {}
    for arm in (*REFS, *ARMS):
        records = [base.read(OUT / f"keys_{s}_{q}.json")["arms"][arm] for s, q in VAL]
        attention[arm] = [{g: {k: sum(r["attention"][layer][g][k] for r in records) for k in records[0]["attention"][layer][g]} for g in ("all", *GROUPS)} for layer in range(2)]
        quality[arm] = {g: base.old.add_counts([r["quality"][g] for r in records]) for g in GROUPS}
    cross_effects = {}
    for op in OPS:
        v = {(a, b): scores[f"pool_{a}_score_{b}"][op]["macro"] for a in REFS for b in REFS}
        cross_effects[op] = {m: {"pool": .5 * ((v["xyz", "temporal"][m] - v["temporal", "temporal"][m]) + (v["xyz", "xyz"][m] - v["temporal", "xyz"][m])),
                                "score": .5 * ((v["temporal", "xyz"][m] - v["temporal", "temporal"][m]) + (v["xyz", "xyz"][m] - v["xyz", "temporal"][m]))} for m in PRIMARY}
    write_json(OUT / "summary.json", {"scores": scores, "attention": attention, "edge_quality": quality,
        "cross_effects": cross_effects, "test_used": False, "scope": "주평가 시드1·검증9·3계열 평균. GT 조건은 특권정보 진단이며 실제방법 성능이 아니다.", "protocol_sha256": sha256(OUT / "protocol.json")})
    write_json(OUT / "completion_validation.json", {"status": "passed", "ledgers": verified, "rows": nrows,
        "reference_key_max_error": max(max(base.read(OUT / f"keys_{s}_{q}.json")["reference_max_error"].values()) for s, q in VAL),
        "cross_endpoints": 36, "test_used": False, "protocol_sha256": sha256(OUT / "protocol.json")})
    print("REPORT", verified, nrows, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "keys", "score", "crossed", "ledger", "report"))
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--sequence")
    args = p.parse_args()
    if args.action in ("prepare", "ledger", "report"):
        globals()[args.action]()
    elif args.action == "score":
        score(args.arm, args.sequence)
    else:
        globals()[args.action](args.sequence)
