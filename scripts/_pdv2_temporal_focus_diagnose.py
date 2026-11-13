#!/usr/bin/env python3
"""시간 전용 3시드의 최종 AP·순위·물리적 문맥을 저장 자료로 분석한다."""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal3 as old
from _pdv2_no_context_ablation import sha256, write_json
from _pr_eval_lib import classify_queries, pr_sweep

OUT = REPO / "results/pdv2_temporal_focus_20261018/diagnosis"
CATEGORIES = ("recovered", "regressed", "both_correct", "both_wrong", "negative")


def load(path: Path) -> dict:
    """NPZ의 수치 배열을 읽는다."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files if k != "meta"}


def quantiles(x: np.ndarray) -> dict:
    """유한한 관측의 개수와 분위수를 반환한다."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return {"n": len(x), **dict(zip(("p10", "median", "p90"),
            np.quantile(x, [.1, .5, .9]).tolist() if len(x) else [None] * 3))}


def cohort_masks(a: dict, b: dict) -> list:
    """평가에서 모호한 질의를 제외한 다섯 상호 배타적 군이다."""
    qc = classify_queries(a["min_past_gt_dist"])
    g0, g1 = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
    return [qc["positive"] & ~g0 & g1, qc["positive"] & g0 & ~g1,
            qc["positive"] & g0 & g1, qc["positive"] & ~g0 & ~g1, qc["negative"]]


def ap_parts(a: dict, b: dict, decision: str) -> dict:
    """교체 순서를 모두 평균한 정확한 AP 회계 분해다(인과 추정 아님)."""
    qc = classify_queries(a["min_past_gt_dist"])
    masks = cohort_masks(a, b)
    assert sum(int(m.sum()) for m in masks) == int((qc["positive"] | qc["negative"]).sum())
    if not qc["n_pos"]:
        return {"n_pos": 0}
    values = []
    for bits in range(32):
        use = np.zeros(len(a["query_idx"]), bool)
        for i, m in enumerate(masks):
            if bits & (1 << i):
                use |= m
        values.append(float(pr_sweep(np.where(use, b[decision], a[decision]),
                                    np.where(use, b["top1_geo_dist"], a["top1_geo_dist"]), qc)["AP"]))
    parts = {}
    for i, label in enumerate(CATEGORIES):
        parts[label] = sum(math.factorial(bin(bits).count("1")) * math.factorial(4 - bin(bits).count("1"))
                           / math.factorial(5) * (values[bits | (1 << i)] - values[bits])
                           for bits in range(32) if not bits & (1 << i))
    assert abs(sum(parts.values()) - (values[-1] - values[0])) < 1e-12
    return {"n_pos": qc["n_pos"], "before": values[0], "after": values[-1], **parts}


def ledger_pair(seed: int, sensor: str, seq: str, op: str) -> tuple:
    """선택된 두 모델의 동일 질의 원장을 읽는다."""
    selection = old.read(old.OUT / "summary.json")["per_seed"][str(seed)]
    paths = [old.OUT / f"s{seed}/{arm}/epoch{selection[arm]['epoch']}/val_{op}_{sensor}_{seq}.npz"
             for arm in ("point", "temporal")]
    a, b = map(load, paths)
    for k in ("query_idx", "min_past_gt_dist", "group"):
        np.testing.assert_array_equal(a[k], b[k])
    return a, b, paths


def ledger() -> None:
    """세 시드·두 운영점의 AP 차이를 재현하고 군별로 분해한다."""
    result, sources = {}, {}
    summary = old.read(old.OUT / "summary.json")
    for seed in old.SEEDS:
        conf = old.read(old.base.BASE / f"s{seed}/confidence_selection.json")["ops"]
        for op in old.base.OPS:
            rows = {}
            for sensor, seq in old.base.VAL:
                a, b, paths = ledger_pair(seed, sensor, seq, op)
                sources.update({str(p.relative_to(REPO)): sha256(p) for p in paths})
                same = a["j_star"] == b["j_star"]
                np.testing.assert_allclose(a[conf[op]["decision"]][same], b[conf[op]["decision"]][same], atol=2e-6)
                row = ap_parts(a, b, conf[op]["decision"])
                row.update(family="kitti" if sensor == "kitti360" else sensor,
                           counts={k: int(m.sum()) for k, m in zip(CATEGORIES, cohort_masks(a, b))})
                rows[sensor + "/" + seq] = row
            fam = {f: {k: float(np.mean([r[k] for r in rows.values() if r["n_pos"] and r["family"] == f]))
                       for k in ("before", "after", *CATEGORIES)} for f in ("kitti", "helipr", "mulran")}
            macro = {k: float(np.mean([r[k] for r in fam.values()])) for k in ("before", "after", *CATEGORIES)}
            for arm, key in (("point", "before"), ("temporal", "after")):
                assert abs(macro[key] - summary["per_seed"][str(seed)][arm]["ops"][op]["macro"]["AP"]) < 1e-12
            result[f"s{seed}_{op}"] = {"macro": macro, "families": fam, "sequences": rows}
            print("AP", seed, op, macro, flush=True)
    write_json(OUT / "ap_decomposition.json", {"results": result, "sources": sources,
               "scope": "검증 AP의 정확한 회계 분해. 인과 추정 아님.", "test_used": False})


def window_features(cache: dict, radius: int) -> dict:
    """키프레임 창의 경로 길이·시간·자세 변화·특징 중복을 구한다."""
    poses, raw = cache["poses"], cache["descriptors"].astype(np.float64)
    n = len(raw)
    q = np.arange(n)
    lo, hi = np.maximum(0, q - radius), np.minimum(n - 1, q + radius)
    pos = poses[:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
    rotation = poses[hi, :3, :3] @ np.transpose(poses[lo, :3, :3], (0, 2, 1))
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    cosine, count = np.zeros(n), np.zeros(n)
    for off in range(-radius, radius + 1):
        if not off:
            continue
        j = q + off
        valid = (j >= 0) & (j < n)
        cosine[valid] += (raw[valid] * raw[j[valid]]).sum(1)
        count[valid] += 1
    out = {"path_m": travel[hi] - travel[lo], "rotation_deg": angle,
           "neighbor_cosine": cosine / np.maximum(count, 1)}
    if "timestamps" in cache:
        t = cache["timestamps"]
        assert np.isfinite(t).all() and (np.diff(t) >= 0).all()
        out["cache_timestamp_span"] = t[hi] - t[lo]
    return out


def windows() -> None:
    """학습과 검증의 창 및 정답 쌍 주변 경로를 비교한다."""
    result, sources = {}, {}
    train_paths = [REPO / p for p in old.read(old.SAMPLES / "audit.json")["sha256"] if p.startswith("data/")]
    for path in train_paths:
        cache = load(path)
        sources[str(path.relative_to(REPO))] = sha256(path)
        result[path.stem] = {"role": "train", "windows": {str(r): {k: quantiles(v) for k, v in window_features(cache, r).items()} for r in (5, 10)}}
    for sensor, seq in old.base.VAL:
        _, _, paths, _ = old.base.inputs("val", sensor + "/" + seq)
        cache = load(paths["cache"])
        sources[str(paths["cache"].relative_to(REPO))] = sha256(paths["cache"])
        pos = cache["poses_gt"][:, :3, 3]
        tree = cKDTree(pos)
        a, _, _ = ledger_pair(1, sensor, seq, old.base.OPS[0])
        queries = a["query_idx"]
        features = {r: window_features(cache, r) for r in (5, 10)}
        near = tree.query_ball_point(pos[queries], 5.)
        overlap = {r: np.full(len(queries), np.nan) for r in (5, 10)}
        for row, q in enumerate(queries):
            candidates = np.asarray([j for j in near[row] if j <= q - 30], dtype=np.int64)
            if not len(candidates):
                continue
            j = candidates[np.argmin(np.linalg.norm(pos[candidates] - pos[q], axis=1))]
            for r in (5, 10):
                u = pos[max(q-r, 0):q+r+1] - pos[q]
                v = pos[max(j-r, 0):min(j+r+1, len(pos))] - pos[j]
                d = np.linalg.norm(u[:, None] - v[None], axis=2)
                overlap[r][row] = .5 * (d.min(0).mean() + d.min(1).mean())
        row = {"role": "val", "windows": {str(r): {k: quantiles(v[queries]) for k, v in features[r].items()} for r in features}, "cohorts": {}}
        for seed, op in itertools.product(old.SEEDS, old.base.OPS):
            a, b, ledgers = ledger_pair(seed, sensor, seq, op)
            sources.update({str(p.relative_to(REPO)): sha256(p) for p in ledgers})
            groups = {k: m for k, m in zip(CATEGORIES, cohort_masks(a, b))}
            groups.update({f"direction_{g}": (a["group"] == g) & (a["min_past_gt_dist"] <= 5) for g in (0, 1, 2)})
            row["cohorts"][f"s{seed}_{op}"] = {k: {str(r): {**{f: quantiles(v[queries][m]) for f, v in features[r].items()},
                 "centered_path_mismatch_m": quantiles(overlap[r][m])} for r in features} for k, m in groups.items()}
        result[sensor + "/" + seq] = row
        print("WINDOW", sensor, seq, row["windows"]["5"], flush=True)
    write_json(OUT / "windows.json", {"sequences": result, "sources": sources, "test_used": False,
       "scope": "창은 입력 자세 기반. 정답 쌍은 가장 가까운 과거 GT 정답 하나를 고정하고 중심 평행이동 후 양방향 평균 최근접 경로 거리를 측정. 3D GT 분석용이며 배포 입력 아님. cache_timestamp_span은 캐시 단위이고 원시 시간 정확성은 별도 감사 대상."})


def ranking() -> None:
    """정답을 잃거나 얻은 질의에서 키와 후보 집합을 교차한다."""
    from _causal_pr_dump import prefix_topk
    from _xs_eval_lib import minmax, align_scores, column_align_scores
    selection = old.read(old.OUT / "summary.json")["per_seed"]
    for sensor, seq in old.base.VAL:
        target = OUT / f"ranking_{sensor}_{seq}.json"
        if target.exists():
            continue
        _, _, paths, _ = old.base.inputs("val", sensor + "/" + seq)
        pos = load(paths["cache"])["poses_gt"][:, :3, 3]
        result, sources = {}, {str(p.relative_to(REPO)): sha256(p) for p in paths.values()}
        for seed in old.SEEDS:
            conf = old.read(old.base.BASE / f"s{seed}/confidence_selection.json")["ops"]
            keys = []
            for arm in ("point", "temporal"):
                p = old.OUT / f"s{seed}/{arm}/epoch{selection[str(seed)][arm]['epoch']}/keys_val_{sensor}_{seq}.npz"
                keys.append(load(p)["embedding"])
                sources[str(p.relative_to(REPO))] = sha256(p)
            for op in old.base.OPS:
                a, b, ledgers = ledger_pair(seed, sensor, seq, op)
                sources.update({str(p.relative_to(REPO)): sha256(p) for p in ledgers})
                with np.load(paths["spec"]) as z:
                    bank = old.base.bank_for(op, z)
                hkey, fft, norm, column = bank
                weights = conf[op]["tuple"]
                cases = []
                for label, mask in zip(CATEGORIES[:2], cohort_masks(a, b)[:2]):
                    for i in np.flatnonzero(mask):
                        q = int(a["query_idx"][i])
                        pools = [np.unique(np.r_[prefix_topk(e, q, q-29, 100), prefix_topk(hkey, q, q-29, 100)]) for e in keys]
                        winners = np.zeros((2, 2), np.int64)
                        for pi, pool in enumerate(pools):
                            aux, sa, sc = minmax(hkey[pool] @ hkey[q]), minmax(align_scores(fft, norm, q, pool)), minmax(column_align_scores(column, q, pool))
                            for ki, e in enumerate(keys):
                                # 원 평가와 같은 덧셈 순서로 FP32 순위를 재현한다.
                                score = weights[0] * minmax(e[pool] @ e[q]) + weights[1] * aux
                                score = score + weights[2] * sa
                                score = score + weights[3] * sc
                                winners[ki, pi] = pool[np.argsort(-score, kind="stable")[0]]
                        assert winners[0, 0] == a["j_star"][i] and winners[1, 1] == b["j_star"][i]
                        cases.append({"query": q, "type": label, "direction": int(a["group"][i]),
                                      "new_pool_has_true": bool(b["union_hit"][i]),
                                      "correct_matrix_key_by_pool": (np.linalg.norm(pos[winners] - pos[q], axis=2) <= 5).tolist()})
                result[f"s{seed}_{op}"] = cases
        write_json(target, {"cases": result, "sources": sources, "test_used": False})
        print("RANK", sensor, seq, sum(map(len, result.values())), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("ledger", "windows", "ranking"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    old.protocol()
    globals()[args.action]()
