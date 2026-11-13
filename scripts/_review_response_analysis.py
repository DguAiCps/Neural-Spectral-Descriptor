#!/usr/bin/env python3
"""현 원고의 최종 원장으로 데이터셋·방향·짝지은 불확실성을 집계한다."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import _paper_fixed_threshold as fixed
import _pdv2_paired_bootstrap as old_boot
from _pr_eval_lib import classify_queries, pr_sweep

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/pdv2_review_response_20261104/analysis"
KEYS = ("AP", "max_F1", "R@100P", "R@1")
DATA, CLASSES, SORTED = {}, {}, {}


def prepare_sorted(score: np.ndarray, top1: np.ndarray, qc: dict) -> tuple:
    """고정 점수 순서를 보존하여 재표집마다 불필요한 정렬을 피한다."""
    finite = (qc["positive"] | qc["negative"]) & np.isfinite(score)
    order = np.flatnonzero(finite)[np.argsort(-score[finite], kind="stable")]
    last = np.flatnonzero(np.r_[score[order][1:] != score[order][:-1], True]) if len(order) else np.array([], int)
    correct = qc["positive"] & (top1 <= 5)
    return order, last, correct[order], ~correct[order], qc["positive"], correct


def weighted_metric(prepared: tuple, counts: np.ndarray) -> np.ndarray:
    """추출 횟수로 계산한 PR이 복제한 원장의 PR과 같도록 한다."""
    order, last, correct, incorrect, pos, pos_correct = prepared
    npos = counts[pos].sum()
    if not npos: return None
    tp = np.r_[0, np.cumsum(counts[order]*correct)[last]]
    fp = np.r_[0, np.cumsum(counts[order]*incorrect)[last]]
    recall = tp/npos
    precision = np.where(tp+fp > 0, tp/np.maximum(tp+fp, 1), 1.)
    ap = np.sum(np.diff(recall)*np.maximum.accumulate(precision[::-1])[::-1][1:])
    f1 = 2*precision*recall/np.maximum(precision+recall, 1e-300)
    return np.array([ap, f1.max(), recall[precision >= 1].max(), counts[pos_correct].sum()/npos])


def weighted_check() -> dict:
    """동점·누락 점수·중복 표본을 원래 PR 라이브러리와 검산한다."""
    rng = np.random.default_rng(104)
    for n in (1, 40, 200):
        qc = classify_queries(rng.choice([2., 12., 40.], n))
        score = rng.integers(-3, 4, n).astype(float)
        score[::17] = np.nan
        top1 = np.where(qc["negative"], 40., rng.choice([2., 8.], n))
        prepared = prepare_sorted(score, top1, qc)
        for _ in range(10):
            ix = rng.integers(0, n, size=n)
            expected = old_boot.seq_metrics(score, top1, qc, ix)
            actual = weighted_metric(prepared, np.bincount(ix, minlength=n))
            if expected is None: assert actual is None
            else: np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    return dict(status="passed", duplicated_query_reference_cases=30)


def write(path: Path, value: dict) -> None:
    fixed.write(path, value)


def metric(score: np.ndarray, top1: np.ndarray, qc: dict) -> dict:
    if not qc["n_pos"]:
        return None
    v = pr_sweep(score, top1, qc)
    return {**{k: float(v[k]) for k in KEYS[:3]}, "R@1": float(np.mean(top1[qc["positive"]] <= 5))}


def load_controls(runs: dict) -> None:
    """완료된 대조만 전체 시퀀스를 확인한 뒤 추가한다."""
    families = {"graph_free": ROOT/"results/pdv2_spectral_power_20261101/adoption/table4",
                "standard": ROOT/"results/pdv2_spectral_power_20261101/adoption/table4"}
    nsd = ROOT/"results/pdv2_review_response_20261104/nsd"
    if (nsd/"test_summary.json").exists():
        selected = fixed.read(nsd/"freeze.json")["selected_fusion"]
        for arm in dict.fromkeys(["no_context__aux_half", "no_context__"+selected,
                                 "no_temporal_score", "neither__aux_half"]):
            families[arm] = nsd/"test"
    for arm, root in families.items():
        runs[arm] = {}
        for seed in (1, 2, 3):
            per = {}
            for sensor, seq in fixed.TEST:
                p = root/f"s{seed}"/f"{sensor}_{seq}_{arm}.npz"
                with np.load(p) as d:
                    per[f"{sensor}/{seq}"] = (d["score"], d["top1_geo_dist"], classify_queries(d["min_past_gt_dist"]))
                    ref = fixed.POWER/"test"/f"s{seed}"/f"{sensor}_{seq}_{fixed.ARM}.npz"
                    with np.load(ref) as r:
                        for key in ("query_idx", "min_past_gt_dist"):
                            np.testing.assert_array_equal(d[key], r[key])
            runs[arm][str(seed)] = per
    bev = ROOT/"results/pdv2_review_response_20261104/bev/test"
    if all((bev/f"s{s}"/f"{sensor}_{seq}.json").exists() for s in (1, 2, 3) for sensor, seq in fixed.TEST):
        runs["bev_temporal"] = {}
        for seed in (1, 2, 3):
            per = {}
            for sensor, seq in fixed.TEST:
                path = bev/f"s{seed}"/f"{sensor}_{seq}.npz"
                record = fixed.read(path.with_suffix(".json"))
                assert fixed.digest(path) == record["sha256"]
                with np.load(path) as d:
                    per[f"{sensor}/{seq}"] = (d["score"], d["top1_geo_dist"], classify_queries(d["min_past_gt_dist"]))
                    raw = runs["bevplace"][str(seed)][f"{sensor}/{seq}"]
                    np.testing.assert_array_equal(d["score_negl2"], raw[0])
                    np.testing.assert_array_equal(d["top1_geo_dist"], raw[1])
            runs["bev_temporal"][str(seed)] = per


def summary(runs: dict) -> dict:
    """원고와 동일 도시 평균 및 별도 데이터셋 평균을 낸다."""
    groups = dict(fixed.CITY)
    for sensor in sorted({s for s, _ in fixed.TEST}):
        groups["dataset:"+sensor] = [s+"/"+q for s, q in fixed.TEST if s == sensor]
    directions = {}
    for sensor, seq in fixed.TEST:
        with np.load(fixed.POWER/"test/s1"/f"{sensor}_{seq}_{fixed.ARM}.npz") as d:
            directions[sensor+"/"+seq] = d["group"]
    result = {}
    for method, seeds in runs.items():
        per_seed = {}
        for seed, per in seeds.items():
            seq_metrics = {s: metric(*r) for s, r in per.items()}
            group_metrics = {g: {m: float(np.mean([seq_metrics[s][m] for s in seqs if seq_metrics[s] is not None]))
                                 for m in KEYS} for g, seqs in groups.items()}
            main = {m: float(np.mean([group_metrics[c][m] for c in fixed.CITY])) for m in KEYS}
            direction_metrics = {}
            for label, group in (("same", 0), ("opposite_only", 1)):
                values = {}
                for seq, (score, top1, qc) in per.items():
                    pos = qc["positive"] & (directions[seq] == group)
                    selected = dict(qc, positive=pos, n_pos=int(pos.sum()),
                                    ambiguous=qc["ambiguous"] | (qc["positive"] & ~pos))
                    selected["n_ambiguous"] = int(selected["ambiguous"].sum())
                    values[seq] = metric(score, top1, selected)
                cities = [{m: float(np.mean([values[s][m] for s in seqs if values[s] is not None])) for m in KEYS}
                          for seqs in fixed.CITY.values() if any(values[s] is not None for s in seqs)]
                direction_metrics[label] = {m: float(np.mean([v[m] for v in cities])) for m in KEYS}
            per_seed[seed] = dict(macro=main, groups=group_metrics, directions=direction_metrics, sequences=seq_metrics)
        stats = {m: dict(mean=float(np.mean([v["macro"][m] for v in per_seed.values()])),
                         std=float(np.std([v["macro"][m] for v in per_seed.values()], ddof=1)) if len(per_seed)>1 else None)
                 for m in KEYS}
        result[method] = dict(mean_std=stats, per_seed=per_seed)
    main = fixed.read(fixed.POWER/"table2_summary.json")["mean_std"][fixed.ARM]
    for m in KEYS:
        np.testing.assert_allclose(result["nsd"]["mean_std"][m]["mean"], main[m]["mean"], atol=1e-12, rtol=0)
    source = fixed.read(ROOT/"results/pdv2_baseline_seeds/summary.json")["methods"]
    for method in source:
        for m in KEYS:
            np.testing.assert_allclose(result[method]["mean_std"][m]["mean"], source[method]["mean_std"][m]["mean"], atol=1e-12, rtol=0)
    write(OUT/"summary.json", dict(methods=result, all_main_points_reproduced=True,
          scope="도시 평균은 표2 그대로. 데이터셋은 각 센서·벤치마크 안 양성 시퀀스 평균이므로 도시 평균과 별개.",
          directions="현재 주모델 질의의 GT 방향 그룹: 같은 방향 정답 있음 / 반대 방향만 있음. 해당 양성과 공통 음성 유지.",
          input_manifest_sha256=fixed.digest(ROOT/"results/pdv2_baseline_seeds/summary.json")))
    return result


def draw(rep: int) -> dict:
    """모든 방법에 같은 도시 내 시퀀스와 원형 100질의 블록을 뽑는다."""
    rng = np.random.default_rng([20261104, rep])
    output = {m: [] for m in DATA}
    for city, seqs in fixed.CITY.items():
        values = {m: [] for m in DATA}
        for seq in rng.choice(seqs, len(seqs), replace=True):
            n = len(CLASSES[seq]["positive"])
            starts = rng.integers(0, n, size=(n+99)//100)
            ix = ((starts[:, None]+np.arange(100)) % n).ravel()[:n]
            counts = np.bincount(ix, minlength=n)
            for method, seeds in DATA.items():
                metrics = [weighted_metric(row[seq], counts) for row in SORTED[method]]
                if metrics[0] is not None:
                    values[method].append(np.mean(metrics, axis=0))
        for method in DATA:
            if values[method]: output[method].append(np.mean(values[method], axis=0))
    return {m: np.mean(v, axis=0) for m, v in output.items()}


def bootstrap(runs: dict, points: dict, workers: int) -> None:
    """고정 실행에 조건부인 짝지은 95% 구간을 보존한다."""
    global DATA, CLASSES, SORTED
    methods = [m for m in ("nsd", "bevplace", "graph_free", "standard", "no_context__aux_half",
                           "no_temporal_score", "neither__aux_half", "bev_temporal") if m in runs]
    CLASSES = {s: r[2] for s, r in runs["nsd"]["1"].items()}
    DATA = {m: [{s: r[:2] for s, r in per.items()} for per in runs[m].values()] for m in methods}
    SORTED = {m: [{s: prepare_sorted(*r, CLASSES[s]) for s, r in per.items()} for per in seeds]
              for m, seeds in DATA.items()}
    write(OUT/"bootstrap_protocol.json", dict(methods=methods, replicates=2000, block_queries=100, seed=20261104,
         weighted_replay_check=weighted_check(),
         resampling="도시 내 시퀀스 복원 추출, 각 추출마다 원형 연속 100질의 블록 복원 추출, 방법 공통",
         interpretation="평가한 학습 실행들에 조건부인 사후 백분위 구간. 시드 재표집·개발 선택 불확실성·동등성 검정은 아님."))
    with Pool(workers) as pool:
        samples = []
        for i, value in enumerate(pool.imap(draw, range(2000), chunksize=5), 1):
            samples.append(value)
            if i % 100 == 0:
                write(OUT/"bootstrap_status.json", dict(done=i, total=2000))
                print("BOOTSTRAP", i, flush=True)
    arrays = {m: np.stack([r[m] for r in samples]) for m in methods}
    np.savez_compressed(OUT/"bootstrap_samples.npz", **arrays)
    comparisons = {}
    pairs = [("nsd", m) for m in methods if m != "nsd"]
    if "bev_temporal" in methods: pairs.append(("bev_temporal", "bevplace"))
    for a, b in pairs:
        delta = arrays[a]-arrays[b]
        comparisons[a+" minus "+b] = {m: dict(delta=points[a]["mean_std"][m]["mean"]-points[b]["mean_std"][m]["mean"],
                     ci95=np.percentile(delta[:, k], [2.5, 97.5]).tolist()) for k, m in enumerate(KEYS)}
    write(OUT/"bootstrap.json", dict(comparisons=comparisons, protocol_sha256=fixed.digest(OUT/"bootstrap_protocol.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    runs = fixed.test_runs()
    load_controls(runs)
    results = summary(runs)
    print("ANALYZED", list(results), flush=True)
    if args.bootstrap: bootstrap(runs, results, args.workers)
