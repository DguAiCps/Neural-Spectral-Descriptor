#!/usr/bin/env python3
"""현재 최종 원장을 진행 방향 정답 집합별로 다시 집계한다."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from _pdv2_pilot import CITY, TEST
from _pr_eval_lib import classify_queries, pr_sweep

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
OUT = ROOT / "results/pdv2_review_direction_yaw_20261105"
KEYS = ("AP", "max_F1", "R@100P", "R@1")
LABELS = ("forward_only", "reverse_only", "mixed", "other_positive")


def read(path: Path) -> dict:
    """JSON 원천을 읽는다."""
    return json.loads(path.read_text())


def write(path: Path, data: dict) -> None:
    """검산 가능한 JSON 결과를 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    """원천 SHA256을 구한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_path(sequence: str) -> Path:
    """현재 원장이 실제 사용한 주행 추정 캐시를 찾는다."""
    source = read(MAIN / "test" / sequence.replace("/", "_") / "reference_validation.json")
    return ROOT / next(p for p in source["input_sha256"] if "/cache_" in p)


def classify_direction(pos: np.ndarray, q: np.ndarray, minimum: np.ndarray) -> dict:
    """모든 과거 정답의 3차원 진행각으로 서로 겹치지 않는 집합을 만든다."""
    index = np.arange(len(pos))
    delta = pos - pos[np.maximum(index - 5, 0)]
    length = np.linalg.norm(delta, axis=1)
    valid = (index >= 5) & (length >= 1.)
    heading = delta / np.maximum(length[:, None], 1e-12)
    tree = cKDTree(pos)
    group = np.full(len(q), -1, np.int8)
    legacy = np.full(len(q), -1, np.int8)
    counts = np.zeros((len(q), 4), np.int32)
    for row in np.flatnonzero(minimum <= 5):
        t = q[row]
        ids = np.asarray(tree.query_ball_point(pos[t], 5), np.int64)
        ids = ids[ids <= t - 30]
        assert len(ids)
        dot = np.clip(heading[ids] @ heading[t], -1., 1.)
        good = valid[t] & valid[ids]
        same, reverse = good & (dot >= .5), good & (dot <= -.5)
        counts[row] = (len(ids), same.sum(), reverse.sum(), (~good).sum())
        group[row] = 0 if same.all() else 1 if reverse.all() else 2 if same.any() and reverse.any() else 3
        legacy[row] = 0 if same.any() else 1 if reverse.all() else 2
    return dict(group=group, legacy=legacy, counts=counts)


def metric(score: np.ndarray, top1: np.ndarray, minimum: np.ndarray, mask=None) -> dict:
    """선택 방향 양성과 공통 음성을 유지한 최종 PR을 계산한다."""
    qc = classify_queries(minimum)
    if mask is not None:
        qc["positive"] &= mask
        qc["ambiguous"] = ~(qc["positive"] | qc["negative"])
        qc["n_pos"] = int(qc["positive"].sum())
        qc["n_ambiguous"] = int(qc["ambiguous"].sum())
    if not qc["n_pos"]:
        return None
    values = pr_sweep(score, top1, qc)
    return {**{k: float(values[k]) for k in KEYS[:3]},
            "R@1": float(np.mean(top1[qc["positive"]] <= 5)),
            "n_pos": qc["n_pos"], "n_neg": qc["n_neg"]}


def aggregate(rows: dict, groups: dict) -> dict:
    """양성 없는 시퀀스·묶음만 제외하고 고정 순서로 평균한다."""
    result = {}
    for label, seqs in groups.items():
        valid = [s for s in seqs if rows[s] is not None]
        result[label] = None if not valid else dict(
            **{k: float(np.mean([rows[s][k] for s in valid])) for k in KEYS},
            sequences=valid, n_pos=sum(rows[s]["n_pos"] for s in valid))
    return result


def main() -> None:
    """17개 현 모델 조건과 BEV 기본의 방향별 지표 및 원래 총계를 검산한다."""
    spec = read(MAIN / "protocol.json")
    protocol = dict(created_unix=time.time(), source=str(MAIN.relative_to(ROOT)),
        truth="j<=q-30의 3D GT 거리<=5m인 모든 후보가 정답. 진행벡터는 GT 위치의 과거5키프레임 차이이며 이동>=1m, 인덱스>=5만 유효.",
        groups=dict(forward_only="정답 모두 유효하며 진행각<=60도", reverse_only="정답 모두 유효하며 진행각>=120도",
                    mixed="<=60도 정답과 >=120도 정답이 둘 다 있음. 다른 각도·무효 정답 동반 가능", other_positive="나머지 양성. 중간 각도·진행벡터 무효 포함"),
        legacy="same_available는 <=60도 정답 하나 이상. reverse_only는 과거 opposite_only와 동일. 새 엄격 그룹과 별도 보고.",
        accounting="방향별 AP/F1/R100은 해당 방향 양성과 전체 공통 음성(dmin>25m)을 유지한다. 다른 방향 양성과 원래 5<dmin<=25m는 제외. 정답 판정은 반환 승자 거리<=5m이며 승자 방향으로 재라벨하지 않는다. R1 분모는 선택 방향 양성.",
        aggregation="시퀀스별 PR -> 양성이 있는 시퀀스만 도시 내 평균 -> 해당 방향 양성이 있는 도시 평균 -> 3시작점 평균. 데이터셋별은 별도 센서 내 시퀀스 평균. 전체와 방향별 도시 수를 보존.",
        selection="저장 결과의 사후 층화이며 새 모델·임계값 선택 없음. AP/F1/R100은 점수 sweep 통계.",
        sources={str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), MAIN / "protocol.json", MAIN / "summary.json")})
    path = OUT / "direction_protocol.json"
    if not path.exists(): write(path, protocol)
    datasets = {s: [a + "/" + b for a, b in TEST if a == s] for s in sorted({s for s, _ in TEST})}
    direction, sources, count_rows = {}, {}, {}
    for sequence in spec["test"]:
        folder = MAIN / "test" / sequence.replace("/", "_")
        with np.load(folder / "diff_s1.npz") as z:
            row = dict(z)
        cp = cache_path(sequence)
        with np.load(cp) as data: pos = data["poses_gt"][:, :3, 3]
        d = classify_direction(pos, row["query_idx"], row["min_past_gt_dist"])
        np.testing.assert_array_equal(d["legacy"], row["group"])
        direction[sequence] = d
        sources[str(cp.relative_to(ROOT))] = digest(cp)
        count_rows[sequence] = dict(queries=len(row["query_idx"]),
            positive=int((row["min_past_gt_dist"] <= 5).sum()),
            negative=int((row["min_past_gt_dist"] > 25).sum()),
            **{label: int((d["group"] == i).sum()) for i, label in enumerate(LABELS)},
            same_available=int((d["legacy"] == 0).sum()))
        dp = OUT / "directions" / (sequence.replace("/", "_") + ".npz")
        dp.parent.mkdir(exist_ok=True)
        np.savez_compressed(dp, query_idx=row["query_idx"], **d)
    methods = {}
    for name in ("identity", "fixed", "diff", "convs_only", "abs_kv", "point_gate", "other_only", "bev_native"):
        seeds = (0,) if name in ("identity", "fixed") else (1, 2, 3)
        per_seed = {}
        for seed in seeds:
            rows = {}
            for sequence in spec["test"]:
                sname = sequence.replace("/", "_")
                p = (ROOT / "results/pdv2_review_response_20261104/bev/test" / f"s{seed}" / (sname + ".npz")
                     if name == "bev_native" else MAIN / "test" / sname / ((name if not seed else f"{name}_s{seed}") + ".npz"))
                sources[str(p.relative_to(ROOT))] = digest(p)
                assert sources[str(p.relative_to(ROOT))] == read(p.with_suffix(".json"))["sha256"]
                with np.load(p) as z: a = dict(z)
                d = direction[sequence]
                with np.load(MAIN / "test" / sname / "diff_s1.npz") as reference:
                    for k in ("query_idx", "min_past_gt_dist"): np.testing.assert_array_equal(a[k], reference[k])
                score = a["score_negl2"] if name == "bev_native" else a["score"]
                masks = dict(all=None, **{g: d["group"] == i for i, g in enumerate(LABELS)}, same_available=d["legacy"] == 0)
                rows[sequence] = {g: metric(score, a["top1_geo_dist"], a["min_past_gt_dist"], mask) for g, mask in masks.items()}
            groups = {}
            for group in masks:
                values = {s: r[group] for s, r in rows.items()}
                cities = aggregate(values, CITY)
                active = [x for x in cities.values() if x is not None]
                groups[group] = dict(macro={k: float(np.mean([x[k] for x in active])) for k in KEYS},
                                     active_cities=[k for k, v in cities.items() if v is not None],
                                     cities=cities, datasets=aggregate(values, datasets), sequences=values)
            per_seed[str(seed)] = groups
        stats = {g: {k: dict(mean=float(np.mean([x[g]["macro"][k] for x in per_seed.values()])),
                            std=float(np.std([x[g]["macro"][k] for x in per_seed.values()], ddof=1)) if len(seeds)>1 else None)
                     for k in KEYS} for g in masks}
        methods[name] = dict(mean_std=stats, per_seed=per_seed)
    original = read(MAIN / "summary.json")["mean"]
    for method in methods:
        for k in KEYS:
            np.testing.assert_allclose(methods[method]["mean_std"]["all"][k]["mean"], original[method][k], atol=1e-12, rtol=0)
    totals = {k: sum(r[k] for r in count_rows.values()) for k in next(iter(count_rows.values()))}
    assert totals["queries"] == 195726 and totals["positive"] == 61682
    write(OUT / "direction_summary.json", dict(methods=methods, counts=totals, per_sequence_counts=count_rows,
          original_totals_reproduced=True, legacy_direction_labels_reproduced=True,
          protocol_sha256=digest(path), source_sha256=sources))
    print(json.dumps(dict(status="passed", counts=totals, main={m: methods[m]["mean_std"] for m in ("diff", "bev_native")}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
