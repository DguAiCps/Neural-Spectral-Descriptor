#!/usr/bin/env python3
"""고정된 시드 1 기록에서 문맥 게이트·순위 역전·접근 방향을 진단한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
STAGES = ("weighted", "pass1", "context")
GROUPS = ("same_available", "opposite_only", "other")


def stats(values: np.ndarray) -> dict:
    """유한 표본 수와 분위수를 반환한다."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {"n": len(values), "quantiles": dict(zip(
        ("min", "p01", "p25", "median", "p75", "p99", "max"),
        np.quantile(values, (0, .01, .25, .5, .75, .99, 1)).tolist()
        if len(values) else [None] * 7))}


def heading(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """좌표축에 무관한 과거 5개 키프레임의 3차원 이동 방향을 구한다."""
    idx = np.arange(len(pos))
    delta = pos - pos[np.maximum(idx - 5, 0)]
    length = np.linalg.norm(delta, axis=1)
    return delta / np.maximum(length[:, None], 1e-12), (idx >= 5) & (length >= 1.0)


def angle_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """단위 이동 벡터 사이의 0~180도 각도를 구한다."""
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


def train_direction() -> dict:
    """학습 캐시의 가능한 양성 쌍을 센다. 실제 추출 이력은 아니다."""
    from _pdv2_pilot import TRAIN

    rows = []
    for sensor, seq in TRAIN:
        paths = list((REPO / "data/preprocessed_v2").glob(
            f"cache_*_{sensor}_{seq}.npz"))
        assert len(paths) == 1, paths
        path = paths[0]
        with np.load(path) as archive:
            pos = archive["poses"][:, :3, 3]
        h, valid = heading(pos)
        tree = cKDTree(pos)
        counts = {"all": 0, "valid_heading": 0, "same": 0, "opposite": 0}
        anchor_fractions = []
        for i in range(len(pos)):
            candidates = np.asarray(tree.query_ball_point(pos[i], 5.0), dtype=np.int64)
            candidates = candidates[np.abs(candidates - i) >= 30]
            counts["all"] += len(candidates)
            angle = angle_degrees(h[i], h[candidates])
            ok = valid[i] & valid[candidates]
            opposite = ok & (angle >= 120)
            counts["valid_heading"] += int(ok.sum())
            counts["same"] += int((ok & (angle <= 60)).sum())
            counts["opposite"] += int(opposite.sum())
            if len(candidates):
                anchor_fractions.append(float(opposite.mean()))
        rows.append({"sequence": sensor + "/" + seq, "n_frames": len(pos),
                     "n_anchors": len(anchor_fractions), "pairs": counts,
                     "anchor_mean_opposite_fraction": float(np.mean(anchor_fractions))
                     if anchor_fractions else None,
                     "cache": str(path.relative_to(REPO)),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    counts = {k: sum(row["pairs"][k] for row in rows) for k in rows[0]["pairs"]}
    return {"scope": "5m 이하, |i-j|>=30 양방향의 가능한 양성 쌍. 실제 학습 샘플 빈도와 구분.",
            "counts": counts, "opposite_fraction_of_all": counts["opposite"] / counts["all"],
            "opposite_fraction_valid_heading": counts["opposite"] / counts["valid_heading"],
            "per_sequence": rows}


def collect(root: Path) -> dict:
    """저장된 검색 순위를 바꾸지 않고 방향별 정답률과 점수 항을 분해한다."""
    original = json.loads((root / "summary.json").read_text())
    rows, sources, pooled = [], {}, {}
    for entry in original["per_sequence"]:
        sequence = entry["sequence"]
        stem = "s1_" + sequence.replace("/", "_")
        with np.load(root / (stem + ".npz")) as archive:
            z = {key: archive[key] for key in archive.files if key != "meta"}
            meta = json.loads(str(archive["meta"]))
        key_path = root / ("keys_" + stem + ".npz")
        with np.load(key_path) as archive:
            f = archive["context"].astype(np.float64)
        cache_path = REPO / meta["caches"]["cache"]
        with np.load(cache_path) as archive:
            pos = archive["poses_gt"][:, :3, 3]
        for path in (root / (stem + ".npz"), key_path, cache_path):
            sources[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
        block_norm = np.stack((np.linalg.norm(f[:, :288], axis=1),
                               np.linalg.norm(f[:, 288:], axis=1)), axis=1)
        assert np.isfinite(f).all() and np.all(block_norm > 0)
        alpha = block_norm[:, 1] / block_norm[:, 0]
        key = f[:, :288] / block_norm[:, :1]
        ctx = f[:, 288:] / block_norm[:, 1:]
        h, valid = heading(pos)
        positive = z["min_past_gt_dist"] <= 5.0
        q = z["query_idx"][positive]
        correct = {s: z[s + "_top1_geo"][positive] <= 5 for s in STAGES}
        group = np.full(len(q), 2, np.int8)
        nearest = np.empty(len(q), np.int64)
        nearest_angle = np.full(len(q), np.nan)
        tree = cKDTree(pos)
        for k, i in enumerate(q):
            candidates = np.asarray(tree.query_ball_point(pos[i], 5.0), dtype=np.int64)
            candidates = candidates[candidates <= i - 30]
            assert len(candidates)
            a = candidates[np.argmin(np.linalg.norm(pos[candidates] - pos[i], axis=1))]
            assert np.isclose(np.linalg.norm(pos[i] - pos[a]),
                              z["min_past_gt_dist"][positive][k], atol=1e-6)
            nearest[k] = a
            angle = angle_degrees(h[i], h[candidates])
            if valid[i]:
                if np.any(valid[candidates] & (angle <= 60)):
                    group[k] = 0
                elif np.all(valid[candidates] & (angle >= 120)):
                    group[k] = 1
                if valid[a]:
                    nearest_angle[k] = angle_degrees(h[i], h[a])
        regresses = correct["weighted"] & ~correct["context"]
        recovers = ~correct["weighted"] & correct["context"]
        a = z["weighted_top1_idx"][positive]
        b = z["context_top1_idx"][positive]
        # 양수이면 키 항은 정답 a를, 문맥 항은 오답 b를 지지한다.
        key_adv = np.einsum("ij,ij->i", f[q, :288], f[a, :288] - f[b, :288])
        ctx_adv = np.einsum("ij,ij->i", f[q, 288:], f[b, 288:] - f[a, 288:])
        key_cos_adv = np.einsum("ij,ij->i", key[q], key[a] - key[b])
        combined_adv = np.einsum("ij,ij->i", f[q], f[b] - f[a])
        assert np.allclose(ctx_adv - key_adv, combined_adv, atol=1e-12)
        assert np.all(key_cos_adv[regresses] >= -2e-6)
        assert np.all(combined_adv[regresses] >= -2e-6)
        paired_key = np.einsum("ij,ij->i", key[q], key[nearest])
        paired_ctx = np.einsum("ij,ij->i", ctx[q], ctx[nearest])
        arrays = {"query_idx": q, "group": group, "nearest_positive": nearest,
                  "nearest_angle": nearest_angle, "paired_key_cos": paired_key,
                  "paired_context_cos": paired_ctx, "regresses": regresses,
                  "recovers": recovers, "key_contribution_true_advantage": key_adv,
                  "context_contribution_wrong_advantage": ctx_adv,
                  **{s + "_correct": correct[s] for s in STAGES}}
        np.savez_compressed(root / ("context_diagnosis_" + stem + ".npz"), **arrays)
        for name, arr in {**arrays, "alpha": alpha}.items():
            pooled.setdefault(name, []).append(arr)
        row = {"sequence": sequence, "city": entry["city"], "alpha": stats(alpha),
               "n_positive": len(q), "groups": {}}
        for gi, name in enumerate(GROUPS):
            mask = group == gi
            row["groups"][name] = {"n": int(mask.sum()), **{
                s: float(correct[s][mask].mean()) if mask.any() else None for s in STAGES}}
        rows.append(row)
        print(sequence, len(q), "regresses", int(regresses.sum()), flush=True)
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    groups = {}
    for gi, name in enumerate(GROUPS):
        mask = pooled["group"] == gi
        cities = {}
        for city in sorted({row["city"] for row in rows}):
            subset = [r["groups"][name] for r in rows
                      if r["city"] == city and r["groups"][name]["n"]]
            if subset:
                cities[city] = {s: float(np.mean([r[s] for r in subset])) for s in STAGES}
        groups[name] = {"n": int(mask.sum()), "pooled_r1": {
            s: float(pooled[s + "_correct"][mask].mean()) if mask.any() else None
            for s in STAGES}, "n_cities": len(cities), "by_city": cities,
            "city_macro": {s: float(np.mean([c[s] for c in cities.values()]))
                           for s in STAGES}}
    true_pairs = {}
    for name, mask in (("same", pooled["nearest_angle"] <= 60),
                       ("opposite", pooled["nearest_angle"] >= 120)):
        true_pairs[name] = {s: stats(pooled["paired_" + s + "_cos"][mask])
                            for s in ("key", "context")}
    mask = pooled["regresses"]
    return {"scope": "현재 고정 시드 1, 시험 46시퀀스의 사후 진단. 설정 선택 없음.",
            "heading_rule": "GT 3차원 과거 5키프레임 이동 벡터 사이 각도. 변위 >=1m, 5개 미만은 제외. 좌표축 회전에 불변.",
            "groups_rule": "same: 60도 이하 정답 존재; opposite: 모든 정답의 방향 유효하고 120도 이상; 나머지 other",
            "limits": "방향 그룹 간 분포 차이는 통제하지 않았다. 인과 효과나 전체 파이프라인 제거 실험이 아니다.",
            "n_positive": len(pooled["query_idx"]), "alpha": stats(pooled["alpha"]),
            "frac_alpha_gt_099": float((pooled["alpha"] > .99).mean()),
            "regresses": int(mask.sum()), "recovers": int(pooled["recovers"].sum()),
            "regression_key_contribution_advantage": stats(pooled["key_contribution_true_advantage"][mask]),
            "regression_context_contribution_advantage": stats(pooled["context_contribution_wrong_advantage"][mask]),
            "regression_context_supports_wrong_fraction": float((
                pooled["context_contribution_wrong_advantage"][mask] > 0).mean()),
            "groups": groups, "nearest_true_pair_cosine": true_pairs,
            "train_positive_direction": train_direction(),
            "per_sequence": rows, "input_sha256": sources,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main() -> None:
    """저장한 키와 검색 기록에서 재현 가능한 진단 결과를 쓴다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO / "results/pdv2_collision_20261013")
    args = parser.parse_args()
    result = collect(args.root.resolve())
    path = args.root / "context_diagnosis.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(path, flush=True)


if __name__ == "__main__":
    main()
