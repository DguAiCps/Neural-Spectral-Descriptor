#!/usr/bin/env python3
"""표 2의 고정 최종 원장으로 비교 그림의 근거를 재집계한다."""
from pathlib import Path
import hashlib
import json

import numpy as np

from _pdv2_main_baselines_official import METHODS
from _pdv2_baseline_seeds import SEED_FOLDERS, TRAINING_FREE
from _pdv2_pilot import CITY, TEST
from _pr_eval_lib import classify_queries, pr_sweep


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/paper_baseline_figure_review_20261031"
MAIN = REPO / "results/pdv2_spectral_whitening_20261026/fixed_test"
STRUCTURE = REPO / "results/paper_structure_figure_review_20261031"
GRID = np.unique(np.r_[np.linspace(0, 1, 1001), np.linspace(.99, 1, 1001), .9, .95, .99, 1.])
METRICS = ("AP", "max_F1", "R@100P", "R@1")
LABELS = dict(scpp="SC++", ring="RING", ringpp="RING++", ringsharp="RING#-L",
              deepring="DeepRING", disco="DiSCO", helios="HeLiOS", seqot="SeqOT",
              imlpr="ImLPR", bevplace="BEVPlace++", nsd="NSD")


def read(path: Path) -> dict:
    """JSON 근거를 읽는다."""
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    """한글과 배열을 보존해 JSON을 저장한다."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    """원천의 해시를 계산한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recall_at_precision(sweep: dict, requirements: np.ndarray) -> np.ndarray:
    """각 정밀도 하한을 만족하는 실제 동점 묶음 중 최대 재현율을 구한다."""
    order = np.argsort(sweep["precision"], kind="stable")
    precision = sweep["precision"][order]
    best_recall = np.maximum.accumulate(sweep["recall"][order][::-1])[::-1]
    return best_recall[np.searchsorted(precision, requirements, side="left")]


def analyze(exclude: tuple = (), base_path: Path = None, overrides: dict = None) -> None:
    """동일 질의·최종 점수·표 2 집계 규칙을 지키는 그림 자료를 만든다.

    exclude에 준 방법은 원장을 읽지 않고 그림 자료에서 뺀다. 기본값은 전체 방법이다.
    base_path는 표 2 기대값 요약이고, overrides는 방법별 원장 폴더와 그 해시({"folder", "input_sha256"})다.
    """
    OUT.mkdir(exist_ok=True)
    base_path = base_path or REPO / "results/pdv2_baseline_seeds/summary.json"
    overrides = overrides or {}
    assert all(m in TRAINING_FREE for m in overrides)
    spec = read(MAIN / "freeze.json")
    base = read(base_path)
    nsd = read(MAIN / "table2_summary.json")
    geometry = dict(np.load(STRUCTURE / "per_query.npz"))
    roster = read(STRUCTURE / "protocol.json")["roster"]
    assert roster == spec["test"]
    bins = np.minimum((geometry["heading_deg"] // 30).astype(int), 5)
    protocol = dict(
        date_kst="2026-10-31", scope="고정 시험 결과의 사후 시각화; 설정·가중치·후보 변경 없음",
        methods={k: v for k, v in LABELS.items() if k not in exclude}, sequences=spec["test"], precision_grid=GRID.tolist(),
        queries="표 2 공통 질의 30 <= q < N-15, 195726개/실행",
        candidates="j <= q-30", positives="min_past_gt_dist <= 5 m, 61682개/실행",
        negatives="min_past_gt_dist > 25 m; 5–25 m 질의는 PR 제외",
        false_positive="양성·음성 질의 모두 반환 후보가 5 m 밖인데 수락하면 오탐",
        precision_curve="각 시퀀스에서 정밀도 하한을 만족하는 동점 포함 임계값의 최대 재현율",
        aggregation="시퀀스 → 동일 가중치 6도시 → 시드 평균; 표 2와 동일",
        seeds="학습 방법 3회(DeepRING 1,3,4; 나머지 1,2,3); 비학습 방법 1회",
        interpretation="시험 임계값 전체 곡선 통계다. 배포용으로 선택한 단일 임계값 성능은 아니다.",
        heading_diagnostic="GT 최근접 정답과 30도 구간, 양성 질의 평균; 표 2 도시 평균과 구별",
        table2_summary=str(base_path.relative_to(REPO)), method_sources={k: v["folder"] for k, v in overrides.items()},
    )
    write(OUT / "protocol.json", protocol)
    hashes = {str(Path(__file__).relative_to(REPO)): digest(Path(__file__))}
    for path in (MAIN / "table2_summary.json", base_path,
                 STRUCTURE / "protocol.json", STRUCTURE / "per_query.npz",
                 REPO / "scripts/_pr_eval_lib.py"):
        hashes[str(path.relative_to(REPO))] = digest(path)
    references = {}
    for sensor, seq in TEST:
        with np.load(MAIN / "s1" / f"{sensor}_{seq}.npz") as data:
            references[f"{sensor}/{seq}"] = {k: data[k] for k in data.files}
    methods = [(m[0], overrides[m[0]]["folder"], *m[2:]) if m[0] in overrides else m
               for m in METHODS if m[0] not in exclude] + [("nsd", "", "", "score", "", "")]
    output, errors = {}, []
    for method, folder, prefix, score, _, _ in methods:
        runs = ([(seed, str((MAIN / f"s{seed}").relative_to(REPO))) for seed in (1, 2, 3)]
                if method == "nsd" else
                [(None, folder)] if method in TRAINING_FREE else SEED_FOLDERS[method])
        seed_reports = {}
        for seed, source in runs:
            seed_key = "-" if seed is None else str(seed)
            per_sequence = {}
            correct_by_heading = np.zeros(6, dtype=np.int64)
            heading_counts = np.zeros(6, dtype=np.int64)
            count = positive_count = 0
            for sensor, seq in TEST:
                sequence = f"{sensor}/{seq}"
                name = f"{sensor}_{seq}.npz"
                path = REPO / source / (name if method == "nsd" else prefix + "_" + name)
                relative = str(path.relative_to(REPO))
                hashes[relative] = digest(path)
                expected = (read(path.with_suffix(".json"))["sha256"] if method == "nsd"
                            else overrides[method]["input_sha256"][relative] if method in overrides
                            else base["input_sha256"][relative])
                assert hashes[relative] == expected, relative
                with np.load(path) as data:
                    ref = references[sequence]
                    ix = np.searchsorted(data["query_idx"], ref["query_idx"])
                    np.testing.assert_array_equal(data["query_idx"][ix], ref["query_idx"])
                    np.testing.assert_array_equal(data["min_past_gt_dist"][ix], ref["min_past_gt_dist"])
                    assert np.all(data["j_star"][ix] <= ref["query_idx"] - 30)
                    qc = classify_queries(ref["min_past_gt_dist"])
                    count += len(ix)
                    positive_count += qc["n_pos"]
                    if not qc["n_pos"]:
                        assert sequence == "kitti/07"
                        continue
                    distances = data["top1_geo_dist"][ix]
                    sw = pr_sweep(data[score][ix], distances, qc)
                curve = recall_at_precision(sw, GRID)
                assert np.all(np.diff(curve) <= 0)
                for requirement, key in ((.95, "R@95P"), (.99, "R@99P"), (1., "R@100P")):
                    assert curve[np.flatnonzero(GRID == requirement)[0]] == sw[key]
                metrics = {key: sw[key] for key in METRICS[:-1]}
                metrics["R@1"] = float((distances[qc["positive"]] <= 5).mean())
                per_sequence[sequence] = dict(metrics=metrics, recall=curve.tolist())
                keep = geometry["sequence"] == roster.index(sequence)
                np.testing.assert_array_equal(geometry["query"][keep], ref["query_idx"][qc["positive"]])
                correctness = distances[qc["positive"]] <= 5
                for b in range(6):
                    bkeep = bins[keep] == b
                    heading_counts[b] += bkeep.sum()
                    correct_by_heading[b] += correctness[bkeep].sum()
            assert (count, positive_count) == (195726, 61682)
            cities = {city: dict(
                recall=np.mean([per_sequence[s]["recall"] for s in names if s in per_sequence], axis=0).tolist(),
                metrics={k: float(np.mean([per_sequence[s]["metrics"][k] for s in names if s in per_sequence]))
                         for k in METRICS}) for city, names in CITY.items()}
            macro = {k: float(np.mean([c["metrics"][k] for c in cities.values()])) for k in METRICS}
            expected = (nsd["per_seed"][seed_key]["macro"] if method == "nsd" else
                        base["methods"][method]["per_seed"][seed_key]["macro"])
            for metric in METRICS:
                error = abs(macro[metric] - expected[metric])
                errors.append(error)
                assert error < 1e-12, (method, seed, metric, error)
            seed_reports[seed_key] = dict(
                recall=np.mean([c["recall"] for c in cities.values()], axis=0).tolist(),
                metrics=macro, cities=cities,
                heading_r1=(correct_by_heading / heading_counts).tolist(),
                heading_counts=heading_counts.tolist(), per_sequence=per_sequence)
        curves = np.array([r["recall"] for r in seed_reports.values()])
        heading = np.array([r["heading_r1"] for r in seed_reports.values()])
        output[method] = dict(label=LABELS[method], per_seed=seed_reports,
                              mean=curves.mean(axis=0).tolist(),
                              std=curves.std(axis=0, ddof=1).tolist() if len(curves) > 1 else None,
                              heading_r1_mean=heading.mean(axis=0).tolist())
        print(method, {p: round(float(curves.mean(axis=0)[np.flatnonzero(GRID == p)[0]])*100, 3)
                       for p in (.9, .95, .99, 1.)}, flush=True)
    write(OUT / "curves.json", dict(precision_grid=GRID.tolist(), methods=output))
    write(OUT / "validation.json", dict(checks_passed=True, table2_max_absolute_error=max(errors),
          table2_metric_checks=len(errors), source_count=len(hashes), input_sha256=hashes,
          full_query_count=195726, positive_queries=61682, positive_sequences=45,
          model_inference=False, model_selection=False))


if __name__ == "__main__":
    analyze()
