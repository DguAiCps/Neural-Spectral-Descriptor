#!/usr/bin/env python3
"""보존 원장과 학습 쌍으로 비교 방법의 공통 손실 원인을 분리한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  python3 scripts/_pdv2_baseline_common_diagnosis.py --part pairs|ledgers
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import copy
import random
from types import SimpleNamespace
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _pdv2_pilot import CITY, DATASET, TEST, TRAIN, VAL
from _pr_eval_lib import classify_queries, pr_sweep

OUT = ROOT / "results/pdv2_baseline_common_diagnosis_20261106"
SOURCES = {}


def source(relative: str) -> Path:
    """로컬 원천 또는 서버에서 복사한 불변 원천을 해시와 함께 연다."""
    path = ROOT / relative
    if not path.exists():
        path = OUT / "inputs" / relative
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def save(name: str, data: dict) -> None:
    """분석 결과를 원자적으로 저장한다."""
    path = OUT / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def quantiles(values: np.ndarray) -> list:
    """비어 있지 않은 유한 표본의 사분위와 꼬리를 반환한다."""
    values = np.asarray(values)
    return np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist() if values.size else []


def pair_stats(a: np.ndarray, b: np.ndarray, seq: np.ndarray,
               pose: np.ndarray, raw: np.ndarray, cache: np.ndarray) -> dict:
    """방향 있는 양성 쌍의 거리·방향·시간 분포를 계산한다."""
    same = seq[a] == seq[b]
    distance = np.linalg.norm(pose[a, :3, 3] - pose[b, :3, 3], axis=1)
    # 같은 세계 좌표계를 공유하는 쌍만 입력한다. 3차원 전방 벡터의 각도다.
    dot = np.sum(pose[a, :3, 0] * pose[b, :3, 0], axis=1)
    angle = np.rad2deg(np.arccos(np.clip(dot, -1, 1)))
    gap = np.abs(cache[a] - cache[b])
    flags = {"self_pair": a == b, "same_sequence": same,
             "cross_sequence": ~same, "within_5m": distance <= 5,
             "over_10m": distance > 10, "over_25m": distance > 25,
             "heading_over_90deg": angle > 90,
             "same_sequence_raw_gap_le30": same & (np.abs(raw[a] - raw[b]) <= 30),
             "same_sequence_keyframe_gap_ge30": same & (gap >= 30),
             "past_keyframe_gap_ge30": same & (cache[b] <= cache[a] - 30),
             "same_sequence_gap_ge30_within5m": same & (gap >= 30) & (distance <= 5),
             "cross_or_keyframe_gap_ge30": ~same | (gap >= 30)}
    return {"pairs": len(a), "unique_directed_pairs": int(len(np.unique(np.stack([a, b], 1), axis=0))),
            "counts": {k: int(v.sum()) for k, v in flags.items()},
            "fractions": {k: float(v.mean()) if len(v) else None for k, v in flags.items()},
            "distance_3d_m_quantiles": quantiles(distance),
            "heading_deg_quantiles": quantiles(angle)}


def tuple_pairs(method: str) -> dict:
    """HeLiOS·ImLPR의 실제 저장 양성 쌍 전체를 읽는다."""
    rel = ("_external/helios_official_pdv2/data/training.pickle" if method == "helios"
           else "_external/imlpr_official_pdv2/data/training_pdv2_global.pickle")
    data = pickle.loads(source(rel).read_bytes())
    assert sorted(data) == list(range(len(data)))
    names = [f"{s}_{q}" for s, q in TRAIN]
    seq, pose, raw, cache = [], [], [], []
    indices = {}
    for name in names:
        path = (f"_external/helios_official_pdv2/data/train/{name}/index.npz" if method == "helios"
                else f"_external/imlpr_official_pdv2/data/training/{name}/index.npz")
        with np.load(source(path)) as z:
            keep = np.flatnonzero(z["keep"]) if method == "helios" else np.arange(len(z["scan_ids"]))
            indices[name] = (z["lidar_poses"][keep] if method == "helios" else z["poses_lidar"],
                             z["scan_ids"][keep], keep if method == "helios" else z["cache_idx"])
    left, right, eligible = [], [], []
    for i, row in data.items():
        parts = row["rel_scan_filepath"].split("/")
        name, idx = parts[1], int(Path(parts[-1]).stem)
        p, r, c = indices[name]
        if method == "helios":
            local = np.searchsorted(r, idx)
            assert local < len(r) and r[local] == idx
            idx = int(local)
        assert idx < len(p)
        seq.append(names.index(name)); pose.append(p[idx]); raw.append(r[idx]); cache.append(c[idx])
        positives = row["positives"]
        if len(positives):
            eligible.append(i)
            left.extend([i] * len(positives)); right.extend(positives)
    a, b = np.asarray(left), np.asarray(right)
    seq, pose, raw, cache = map(np.asarray, [seq, pose, raw, cache])
    assert np.all(a != b)
    rows = {}
    for k, name in enumerate(names):
        mask = seq[a] == k
        rows[name] = {"frames": int((seq == k).sum()),
                      "eligible_anchors": int((seq[eligible] == k).sum()),
                      **pair_stats(a[mask], b[mask], seq, pose, raw, cache)}
        print(method, name, rows[name]["frames"], rows[name]["pairs"], flush=True)
    near_counts = np.bincount(a[np.linalg.norm(pose[a, :3, 3] - pose[b, :3, 3], axis=1) <= 5], minlength=len(data))
    sampler_path = f'_external/{"HeLiOS" if method == "helios" else "ImLPR"}/datasets/samplers.py'
    tree = ast.parse(source(sampler_path).read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ["ListDict", "BatchSampler"]]
    ns = {"copy": copy, "random": random, "Sampler": object, "TrainingDataset": object, "VERBOSE": False}
    exec(compile(ast.Module(body=classes, type_ignores=[]), sampler_path, "exec"), ns)
    probes = {}
    for seed in [1, 2, 3]:
        random.seed(seed)
        dataset = SimpleNamespace(queries=data, get_positives=lambda i: data[i]["positives"])
        batches = list(ns["BatchSampler"](dataset, batch_size=2048))
        ids = np.concatenate(batches)
        # 인접 두 항목은 sampler가 명시적으로 만든 양성 쌍이다.
        pa, pb = ids[::2], ids[1::2]
        la, lb = [], []
        for batch in batches:
            batch = np.asarray(batch)
            for i in batch:
                partners = batch[np.isin(batch, data[int(i)]["positives"])]
                la.extend([int(i)] * len(partners)); lb.extend(partners)
        probes[str(seed)] = {"batches": len(batches), "presented_frames": len(ids),
            "helipr_presented_fraction": float((seq[ids] >= 4).mean()),
            "sampled_pairs": pair_stats(pa, pb, seq, pose, raw, cache),
            "in_batch_positive_relations": pair_stats(np.asarray(la), np.asarray(lb), seq, pose, raw, cache)}
    return {"scope": "저장된 전체 양성 관계. 배치 안의 실제 손실 쌍 빈도와 구별한다.",
            "frames": len(data), "eligible_anchors": len(eligible),
            "helipr_anchor_fraction": float(np.mean(seq[eligible] >= 4)),
            "within5m_positive_coverage": {"at_least_one_fraction": float((near_counts > 0).mean()),
                                            "at_least_four_fraction": float((near_counts >= 4).mean()),
                                            "count_quantiles": quantiles(near_counts)},
            "sampler_probe": probes,
            "sampler_scope": "공식 sampler를 세 RNG 초기값에서 한 epoch씩 재생. 과거 학습의 모든 배치를 복원한 것은 아니다. 배치 안의 관계는 miner가 고른 활성 손실 쌍과 구별한다.",
            "per_sequence": rows, **pair_stats(a, b, seq, pose, raw, cache)}


def seqot_pairs() -> dict:
    """실제 공개 학습기의 기준 프레임·양성 선택을 그대로 재구성한다."""
    code = source("scripts/_seqot_official_train.py").read_text()
    tree = ast.parse(code)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "anchor_plan"]
    ns = {"np": np, "OVERLAP_THRESH": .3, "NEG_RULE": "lt"}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "anchor_plan", "exec"), ns)
    indices, nr, poses, raw, caches = [], [], [], [], []
    names = [f"{s}_{q}" for s, q in TRAIN]
    for name in names:
        arr = np.load(source(f"_external/seqot_official_pdv2/labels/{name}/normalized_data.npy"))
        indices.append((arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64), arr[:, 2].astype(float)))
        with np.load(source(f"_external/seqot_official_pdv2/labels/{name}/frames.npz")) as z:
            nr.append(int(z["n_raw"])); poses.append(z["poses"]); raw.append(z["raw_ids"])
        sensor, sequence = name.split("_", 1)
        path = list((ROOT / "data/preprocessed_v2").glob(f"cache_????????_{sensor}_{sequence}.npz"))
        assert len(path) == 1
        with np.load(source(str(path[0].relative_to(ROOT)))) as z:
            # 5번째 원시 프레임의 시각을 가장 최근 키프레임 위치로 옮긴 근사 간격이다.
            caches.append(np.searchsorted(z["scan_ids"], raw[-1], side="right") - 1)
    offsets = np.r_[0, np.cumsum([len(x) for x in raw])]
    seq = np.concatenate([np.full(len(p), i) for i, p in enumerate(poses)])
    pose, rawall, cache = np.concatenate(poses), np.concatenate(raw), np.concatenate(caches)
    result = {}
    for stage, count in [(1, 3), (2, 6)]:
        ok = (lambda d, a: 1 <= a < nr[d] - 1) if stage == 1 else (lambda d, a: 10 <= a <= nr[d] - 9)
        plan = ns["anchor_plan"](indices, ok, count, count)
        meta = ("_external/seqot_official_pdv2/runs/stage1/meta.json" if stage == 1
                else "_external/seqot_official_pdv2/centered/runs/stage2/meta.json")
        expected = json.loads(source(meta).read_text())
        actual = {name: sum(d == k for d, _, _, _ in plan) for k, name in enumerate(names)}
        assert actual == expected["anchors_per_sequence"], (stage, actual, expected)
        aa, bb, na, nb = [], [], [], []
        for d, anchor, pos, neg in plan:
            assert anchor % 5 == 0 and all(x % 5 == 0 for x in pos + neg)
            aa.extend([offsets[d] + anchor // 5] * len(pos)); bb.extend(offsets[d] + np.asarray(pos) // 5)
            na.extend([offsets[d] + anchor // 5] * len(neg)); nb.extend(offsets[d] + np.asarray(neg) // 5)
        a, b, na, nb = map(np.asarray, [aa, bb, na, nb])
        rows = {name: {"frames": len(raw[k]), "eligible_anchors": actual[name],
                       **pair_stats(a[seq[a] == k], b[seq[a] == k], seq, pose, rawall, cache)}
                for k, name in enumerate(names)}
        result[f"stage{stage}"] = {"anchors": len(plan), "reproduced_logged_counts": True,
            "helipr_anchor_fraction": float(np.mean([d >= 4 for d, _, _, _ in plan])),
            "per_sequence": rows, **pair_stats(a, b, seq, pose, rawall, cache),
            "negative_pair_geometry": pair_stats(na, nb, seq, pose, rawall, cache)}
        result[f"stage{stage}"]["all_selected_positives_are_self"] = sum(all(x == anchor for x in pos) for _, anchor, pos, _ in plan)
        result[f"stage{stage}"]["all_selected_positives_are_self_fraction"] = float(np.mean([all(x == anchor for x in pos) for _, anchor, pos, _ in plan]))
        # 학습을 하지 않는 구성 대조: 같은 원본 overlap 라벨의 양성 풀에서 자기쌍·중복만 제거한다.
        rng = np.random.default_rng(20261106)
        pools = []
        for f1, f2, ov in indices:
            pools.append({int(anchor): np.unique(f2[(f1 == anchor) & (ov > .3) & (f2 != anchor)])
                          for anchor in np.unique(f1)})
        ca, cb, poola, poolb = [], [], [], []
        missing = 0
        for d, anchor, pos, _ in plan:
            pool = pools[d][anchor]
            if not len(pool):
                missing += 1
                continue
            chosen = rng.choice(pool, size=min(count, len(pool)), replace=False)
            ca.extend([offsets[d] + anchor // 5] * len(chosen)); cb.extend(offsets[d] + chosen // 5)
            poola.extend([offsets[d] + anchor // 5] * len(pool)); poolb.extend(offsets[d] + pool // 5)
        result[f"stage{stage}"]["nonself_unique_pool"] = pair_stats(np.asarray(poola), np.asarray(poolb), seq, pose, rawall, cache)
        result[f"stage{stage}"]["nonself_unique_uniform_sampling_diagnostic"] = {
            "unusable_original_anchors": missing,
            "scope": "기존 유효 기준 프레임·라벨 유지; 양성 자기쌍·중복 제거 후 무작위 추출. 재학습 성능이 아니다.",
            **pair_stats(np.asarray(ca), np.asarray(cb), seq, pose, rawall, cache)}
        print("seqot", stage, len(plan), result[f"stage{stage}"]["fractions"], flush=True)
    return result


def bev_pairs() -> dict:
    """잘 동작하는 BEVPlace++의 원래 양성 정의를 대조군으로 재구성한다."""
    from scipy.spatial import cKDTree

    record = json.loads(source("results/pdv2_bevplace_official/train_selection.json").read_text())
    expected = record["_meta"]["train_sequences"]
    pose, raw, cache, seq = [], [], [], []
    aa, bb, counts = [], [], {}
    offset = 0
    for k, (sensor, sequence) in enumerate(TRAIN):
        path = list((ROOT / "data/preprocessed_v2").glob(f"cache_????????_{sensor}_{sequence}.npz"))
        assert len(path) == 1
        with np.load(source(str(path[0].relative_to(ROOT)))) as z:
            p, r = z["poses"], z["scan_ids"]
        name = f"{sensor}_{sequence}"
        assert len(p) == expected[name]
        indices = cKDTree(p[:, :3, 3]).query_ball_point(p[:, :3, 3], 5)
        eligible = 0
        for i, js in enumerate(indices):
            js = np.asarray(js)
            js = js[(js != i) & (np.linalg.norm(p[js, :3, 3] - p[i, :3, 3], axis=1) < 5)]
            eligible += bool(len(js))
            aa.extend([offset + i] * len(js)); bb.extend(offset + js)
        counts[name] = {"frames": len(p), "eligible_anchors": eligible}
        seq.extend([k] * len(p)); pose.extend(p); raw.extend(r); cache.extend(np.arange(len(p)))
        offset += len(p)
    seq, pose, raw, cache, a, b = map(np.asarray, [seq, pose, raw, cache, aa, bb])
    return {"scope": "원래 <5m, 자기쌍 제외 양성 풀의 기하 재구성. 영상의 독립 회전 증강 전 관계다.",
            "frames": len(pose), "eligible_anchors": sum(v["eligible_anchors"] for v in counts.values()),
            "helipr_frame_fraction": float((seq >= 4).mean()),
            "augmentation": "질의·양성 BEV 영상을 각각 독립적인 [0,360)도 정수 회전. 실제 반대 시점 관측과 같은 것은 아니다.",
            "per_sequence": {name: {**counts[name], **pair_stats(a[seq[a] == k], b[seq[a] == k], seq, pose, raw, cache)}
                             for k, name in enumerate(counts)},
            **pair_stats(a, b, seq, pose, raw, cache)}


def auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """동점을 0.5로 세는 정확 순위 AUC를 계산한다."""
    if not len(pos) or not len(neg):
        return None
    ordered = np.sort(neg)
    return float(np.mean((np.searchsorted(ordered, pos, "left") +
                          np.searchsorted(ordered, pos, "right")) / (2 * len(neg))))


def measure(z: dict, score: str) -> dict:
    """최종 AP 손실을 검색 실패와 점수 순서 손실로 정확 분해한다."""
    qc = classify_queries(z["min_past_gt_dist"])
    s, d = z[score], z["top1_geo_dist"]
    pos, neg = qc["positive"], qc["negative"]
    correct = pos & (d <= 5)
    eligible = pos | neg
    assert np.all(np.isfinite(s))
    assert np.all(z["j_star"] <= z["query_idx"] - 30)
    result = {"n_queries": len(s), "n_positive": int(pos.sum()), "n_negative": int(neg.sum())}
    if not pos.any():
        return result
    sweep = pr_sweep(s, d, qc)
    recall = float(correct.sum() / pos.sum())
    oracle = pr_sweep(correct.astype(float), d, qc)
    assert abs(oracle["AP"] - recall) < 1e-12
    assert sweep["AP"] <= recall + 1e-12
    result.update({k: sweep[k] for k in ["AP", "max_F1", "R@100P"]})
    result.update({"R@1": recall, "retrieval_loss": 1 - recall,
                   "ranking_loss": recall - sweep["AP"],
                   "AP_over_R1": sweep["AP"] / recall if recall else 0,
                   "correct_vs_wrong_AUC": auc(s[correct], s[eligible & ~correct]),
                   "correct_vs_nonrevisit_AUC": auc(s[correct], s[neg]),
                   "correct_vs_mislocalized_AUC": auc(s[correct], s[pos & ~correct])})
    for radius in [10, 15, 25]:
        result[f"same_queries_R1_{radius}m"] = float((d[pos] <= radius).mean())
    assert abs(result["retrieval_loss"] + result["ranking_loss"] + result["AP"] - 1) < 1e-12
    return result


def aggregate(rows: dict, groups: dict) -> dict:
    """양성 없는 시퀀스를 뺀 뒤 그룹에 같은 가중치를 준다."""
    keys = [k for k in next(v for v in rows.values() if "AP" in v) if not k.startswith("n_")]
    by_group = {}
    for name, members in groups.items():
        valid = [rows[s] for s in members if s in rows and "AP" in rows[s]]
        if valid:
            by_group[name] = {k: float(np.mean([v[k] for v in valid if v[k] is not None]))
                              if any(v[k] is not None for v in valid) else None for k in keys}
    mean = {k: float(np.mean([v[k] for v in by_group.values() if v[k] is not None])) for k in keys}
    return {"groups": by_group, "macro": mean}


def ledgers() -> dict:
    """기존 10개 비교 방법의 모든 시드와 원래 선택 검증 원장을 분석한다."""
    from _pdv2_main_baselines_official import METHODS
    baseline = json.loads(source("results/pdv2_baseline_seeds/summary.json").read_text())
    results = {}
    for method, _, prefix, score, _, _ in METHODS:
        seeds = {}
        for seed, info in baseline["methods"][method]["per_seed"].items():
            rows = {}
            for sensor, sequence in TEST:
                path = f'{info["source"]}/{prefix}_{sensor}_{sequence}.npz'
                with np.load(source(path)) as z, np.load(source(
                    f"results/pdv2_symmetric3_20261013/s1/test_selected_symmetric_pbev128_main_{sensor}_{sequence}.npz")) as ref:
                    ix = np.searchsorted(z["query_idx"], ref["query_idx"])
                    np.testing.assert_array_equal(z["query_idx"][ix], ref["query_idx"])
                    np.testing.assert_array_equal(z["min_past_gt_dist"][ix], ref["min_past_gt_dist"])
                    arrays = {k: z[k][ix] for k in ["query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", score]}
                rows[f"{sensor}/{sequence}"] = measure(arrays, score)
            summary = aggregate(rows, CITY)
            assert sum(v["n_queries"] for v in rows.values()) == 195726
            for k in ["AP", "max_F1", "R@100P", "R@1"]:
                assert abs(summary["macro"][k] - info["macro"][k]) < 1e-12
            seeds[seed] = {"sequences": rows, **summary, "datasets": aggregate(rows, DATASET)["groups"]}
        keys = next(iter(seeds.values()))["macro"]
        results[method] = {"seeds": seeds, "mean": {k: float(np.mean([v["macro"][k] for v in seeds.values()])) for k in keys}}
        print(method, results[method]["mean"], flush=True)
    validation = {}
    base = "results/pdv2_review_selection_full_20261105/retained"
    groups = {"kitti": [f"{s}/{q}" for s, q in VAL if s.startswith("kitti")],
              "helipr": [f"{s}/{q}" for s, q in VAL if s == "helipr"],
              "mulran": [f"{s}/{q}" for s, q in VAL if s == "mulran"]}
    for method, score in [("imlpr", "score_cos"), ("seqot", "score_disco"), ("disco", "score_disco"), ("bevplace", "score_negl2")]:
        selection = json.loads(source(f"{base}/selection_{method}.json").read_text())
        validation[method] = {}
        for seed, choice in selection["original_choices"].items():
            # Original-policy IDs are immutable; no alternative checkpoint is selected here.
            cid = choice["candidate"]["id"]
            rows = {}
            for sensor, sequence in VAL:
                relative = f"{base}/val/{method}/s{seed}/{cid}/dump_{sensor}_{sequence}.npz"
                if method == "bevplace":
                    record = json.loads(source(f"{base}/val/{method}/s{seed}/{cid}/{sensor}_{sequence}.json").read_text())
                    relative = record["audit"]["reused_ledger"]
                    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == record["audit"]["sha256"]
                with np.load(source(relative)) as z:
                    rows[f"{sensor}/{sequence}"] = measure(z, score)
            assert sum(v["n_queries"] for v in rows.values()) == 99236
            validation[method][seed] = {"checkpoint_id": cid, "sequences": rows, **aggregate(rows, groups)}
    return {"test_frozen": results, "validation_original_models": validation,
            "identity": "1−AP = (1−R@1) + (R@1−AP). 후자는 반환 후보를 고정한 채 정답/오답을 완벽 정렬하는 oracle과의 차이이며 달성 가능한 개선량을 보장하지 않는다.",
            "all_main_metrics_reproduced": True, "sources_sha256": SOURCES}


def main() -> None:
    """선택이나 재학습 없이 공통 진단을 완료한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["pairs", "ledgers", "control"], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"scope": "저장 학습 쌍과 고정 모델의 원장 공통 진단",
                "selection": "설정·체크포인트·주표·그림 변경 없음; 시험은 동결 원장 기술 통계만 사용",
                "pairs": "전체 양성 풀 및 SeqOT 실제 선택; 5/10/25m·자기쌍·30키프레임·전방각90도 기준 사전 고정",
                "decomposition": "1−AP = 검색실패(1−R1) + 수락점수순서손실(R1−AP)",
                "validation": "완료된 공통 검증 원장 중 원래 선택된 가중치만 사용",
                "remaining": "진단으로 유효 학습·입력 원인을 좁힌 뒤 학습/검증에서만 수정 검증"}
    pp = OUT / "protocol.json"
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol
    else:
        save("protocol.json", protocol)
    start = time.time()
    if args.part == "pairs":
        result = {"helios": tuple_pairs("helios"), "imlpr": tuple_pairs("imlpr"), "seqot": seqot_pairs()}
        result["sources_sha256"] = SOURCES
        result["scope_notes"] = ["HeLiOS/ImLPR는 전체 관계 풀, SeqOT는 매 에포크 고정 선택 빈도다.",
            "거리·방향은 GT로 설명하는 진단이며 모델 입력 변경이 아니다.",
            "SeqOT의 키프레임 간격은 원시 프레임에 최근 키프레임 번호를 매긴 근사다."]
    elif args.part == "control":
        result = {"bevplace": bev_pairs()}
        rows = {}
        for sensor, sequence in TEST:
            with np.load(source(f"results/pdv2_ringpp_maxfix_20261105/odom/dump_ringpp_{sensor}_{sequence}.npz")) as z, np.load(source(
                f"results/pdv2_symmetric3_20261013/s1/test_selected_symmetric_pbev128_main_{sensor}_{sequence}.npz")) as ref:
                ix = np.searchsorted(z["query_idx"], ref["query_idx"])
                np.testing.assert_array_equal(z["query_idx"][ix], ref["query_idx"])
                arrays = {k: z[k][ix] for k in ["query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "score_corr"]}
                rows[f"{sensor}/{sequence}"] = measure(arrays, "score_corr")
        result["ringpp_exactmax"] = {"sequences": rows, **aggregate(rows, CITY)}
        expected = json.loads(source("results/pdv2_ringpp_maxfix_20261105/summary_verified.json").read_text())["ringpp_maxfix"]["macro"]
        for k in ["AP", "max_F1", "R@100P", "R@1"]:
            assert abs(expected[k] - result["ringpp_exactmax"]["macro"][k]) < 1e-12
        result["sources_sha256"] = SOURCES
    else:
        result = ledgers()
    result["elapsed_s"] = time.time() - start
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(args.part + ".json", result)


if __name__ == "__main__":
    main()
