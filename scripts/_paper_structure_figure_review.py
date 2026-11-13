#!/usr/bin/env python3
"""고정 시험 원장에서 방향 차이별 검색 단계 효과를 재집계한다.

호스트에서 NumPy로 저장 결과만 집계한다. 모델 추론이나 설정 선택은 하지 않는다.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/paper_structure_figure_review_20261031"
MAIN = REPO / "results/pdv2_spectral_whitening_20261026/fixed_test"
CONTROL = REPO / "results/pdv2_table4_current_20261030"
KEYS = REPO / "results/pdv2_spectral_keys_20261025"
BINS = np.arange(0., 181., 30.)
STAGES = ("key_top1", "zero_shift", "full", "key_pool", "aux_pool", "union_pool")


def read(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    """검토용 JSON을 기록한다."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    """원천 해시를 계산한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(path: Path) -> dict:
    """작은 원장의 배열만 메모리로 읽는다."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def analyze() -> None:
    """같은 양성 질의의 단계별 성공 여부를 기하 구간으로 나눈다."""
    OUT.mkdir(exist_ok=True)
    spec = read(MAIN / "freeze.json")
    protocol = {
        "date_kst": "2026-10-31", "scope": "사후 구조 해석용, 모델·설정 선택 없음",
        "roster": spec["test"], "seeds": [1, 2, 3],
        "positive": "기존 공통 원장의 min_past_gt_dist <= 5 m",
        "heading_reference": "j <= q-30의 GT 최근접 정답; 거리 동률은 작은 키프레임 번호",
        "bins_deg": BINS.tolist(), "bin_edges": "좌측 포함, 마지막 구간만 180도 포함",
        "aggregation": "각 구간의 같은 질의에 세 시드 성공을 평균한다. 도시 평균 표2와 다르다.",
        "stages": list(STAGES),
        "controls": "0-shift와 전체 모델은 같은 두 색인 후보 합집합과 학습 가중치를 사용한다.",
        "limitations": "R@1은 수락 임계값을 적용하지 않는다. 정렬은 기존 연산이며 신규 모듈 우위 증거가 아니다.",
    }
    write(OUT / "protocol.json", protocol)
    records, hashes, paths = [], {}, {}
    totals = {name: {str(seed): {} for seed in (1, 2, 3)} for name in STAGES}
    for sid, sequence in enumerate(spec["test"]):
        name = sequence.replace("/", "_") + ".npz"
        info = read(CONTROL / "s1/no_alignment/test" / Path(name).with_suffix(".json"))
        cache_name = next(p for p in info["inputs"] if "/cache_" in p)
        cache_path = REPO / cache_name
        assert digest(cache_path) == info["inputs"][cache_name]
        hashes[cache_name] = info["inputs"][cache_name]
        paths[sequence] = dict(cache=cache_name,
                               spectrum=next(p for p in info["inputs"] if "/testspec_" in p))
        with np.load(cache_path) as cache:
            pose = cache["poses_gt"]
        xyz = pose[:, :3, 3]
        yaw = np.arctan2(pose[:, 1, 0], pose[:, 0, 0])
        reference = archive(MAIN / "s1" / name)
        positive = reference["min_past_gt_dist"] <= 5
        q = reference["query_idx"][positive]
        if not len(q):
            assert sequence == "kitti/07"
            continue
        neighbors = cKDTree(xyz).query_ball_point(xyz[q], 5.000001)
        nearest, distances = [], []
        for query, candidates in zip(q, neighbors):
            eligible = np.array(sorted(j for j in candidates if j <= query - 30))
            assert len(eligible)
            d = np.linalg.norm(xyz[eligible] - xyz[query], axis=1)
            index = int(np.argmin(d))
            nearest.append(int(eligible[index]))
            distances.append(float(d[index]))
        nearest = np.asarray(nearest)
        np.testing.assert_allclose(distances, reference["min_past_gt_dist"][positive], atol=1e-5, rtol=0)
        degrees = np.degrees(np.abs(np.angle(np.exp(1j * (yaw[q] - yaw[nearest])))))
        item = dict(sequence=np.full(len(q), sid), query=q, nearest=nearest,
                    distance=np.array(distances), heading_deg=degrees)
        scores = {name: [] for name in STAGES}
        for seed in (1, 2, 3):
            main_path = MAIN / f"s{seed}" / name
            zero_path = CONTROL / f"s{seed}/no_alignment/test" / name
            coverage_path = KEYS / f"s{seed}/octave288/test" / ("coverage_" + name)
            paths_to_load = (main_path, zero_path, coverage_path)
            for path in paths_to_load:
                hashes[str(path.relative_to(REPO))] = digest(path)
            assert hashes[str(main_path.relative_to(REPO))] == read(main_path.with_suffix(".json"))["sha256"]
            assert hashes[str(zero_path.relative_to(REPO))] == read(zero_path.with_suffix(".json"))["sha256"]
            key_info = read(coverage_path.with_name(name).with_suffix(".json"))
            assert hashes[str(coverage_path.relative_to(REPO))] == key_info["coverage_sha256"]
            main, zero, coverage = [archive(path) for path in paths_to_load]
            for data in (main, zero, coverage):
                np.testing.assert_array_equal(data["query_idx"], reference["query_idx"])
                np.testing.assert_array_equal(data["min_past_gt_dist"], reference["min_past_gt_dist"])
            np.testing.assert_array_equal(zero["union_hit"], coverage["union_hit"])
            np.testing.assert_array_equal(zero["n_candidates"], coverage["n_candidates"])
            np.testing.assert_array_equal(main["j_star"], coverage["j_star"])
            for stage, values in (
                ("key_top1", coverage["key_top1_geo"] <= 5),
                ("zero_shift", zero["top1_geo_dist"] <= 5),
                ("full", main["top1_geo_dist"] <= 5),
                ("key_pool", coverage["key_hit"]),
                ("aux_pool", coverage["aux_hit"]),
                ("union_pool", coverage["union_hit"]),
            ):
                scores[stage].append(values[positive])
                totals[stage][str(seed)][sequence] = float(values[positive].mean())
        item.update({stage: np.stack(values) for stage, values in scores.items()})
        records.append(item)
        print(sequence, len(q), flush=True)
    arrays = {key: np.concatenate([r[key] for r in records], axis=1 if key in STAGES else 0)
              for key in records[0]}
    assert len(arrays["query"]) == 61682
    index = np.minimum(np.searchsorted(BINS, arrays["heading_deg"], side="right") - 1, len(BINS)-2)
    rows = []
    for i in range(len(BINS)-1):
        keep = index == i
        rows.append(dict(low_deg=float(BINS[i]), high_deg=float(BINS[i+1]),
                         queries=int(keep.sum()), sequences=len(np.unique(arrays["sequence"][keep])),
                         stages={stage: dict(mean=float(arrays[stage][:, keep].mean()),
                                             per_seed=arrays[stage][:, keep].mean(1).tolist())
                                 for stage in STAGES}))
    main_summary = read(MAIN / "table2_summary.json")
    for seed in (1, 2, 3):
        city_means = [np.mean([totals["full"][str(seed)][seq] for seq in sequences
                               if seq in totals["full"][str(seed)]])
                      for sequences in spec["cities"].values()]
        np.testing.assert_allclose(np.mean(city_means), main_summary["per_seed"][str(seed)]["macro"]["R@1"], rtol=0, atol=1e-12)
    np.savez_compressed(OUT / "per_query.npz", **arrays)
    report = dict(positive_queries=len(arrays["query"]), seeds=3, bins=rows,
                  pooled={stage: float(arrays[stage].mean()) for stage in STAGES},
                  gains=dict(alignment_rescued=int((arrays["full"] & ~arrays["zero_shift"]).sum()),
                             alignment_lost=int((~arrays["full"] & arrays["zero_shift"]).sum())),
                  raw_table2_r1_reproduced=True, matched_candidate_coverage_and_counts=True,
                  paths=paths, sources=hashes)
    write(OUT / "summary.json", report)
    print(json.dumps({k: report[k] for k in ("positive_queries", "pooled", "gains", "bins")}))


if __name__ == "__main__":
    analyze()
