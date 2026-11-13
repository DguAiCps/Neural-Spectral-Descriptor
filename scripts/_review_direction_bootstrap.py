#!/usr/bin/env python3
"""실제 재방문 방향별 학습 기여의 짝지은 조건부 구간을 계산한다."""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import _review_direction_metrics as base

OUT = base.OUT / "direction_learning"
GROUPS = ("all", "forward_only", "reverse_only")
METHODS = ("identity", "fixed", "diff", "convs_only")
CITIES = tuple(base.CITY)
DATA = {}


def prepare(score: np.ndarray, geo: np.ndarray, minimum: np.ndarray, mask: np.ndarray) -> tuple:
    """동점 점수 전체를 같이 수락하는 고정 PR 정렬을 만든다."""
    pos, neg = (minimum <= 5) & mask, minimum > 25
    finite = (pos | neg) & np.isfinite(score)
    order = np.flatnonzero(finite)[np.argsort(-score[finite], kind="stable")]
    last = np.flatnonzero(np.r_[score[order][1:] != score[order][:-1], True]) if len(order) else np.array([], int)
    correct = pos & (geo <= 5)
    return order, last, correct[order], np.flatnonzero(pos), np.flatnonzero(correct)


def weighted(row: tuple, counts: np.ndarray):
    """블록 재표집으로 복제한 원장과 같은 AP/F1/R100/R1을 계산한다."""
    order, last, correct, pos, matches = row
    npos = counts[pos].sum()
    if not npos:
        return None
    weights = counts[order]
    tp = np.r_[0, np.cumsum(weights*correct)[last]]
    fp = np.r_[0, np.cumsum(weights*(~correct))[last]]
    recall = tp/npos
    precision = np.where(tp+fp > 0, tp/np.maximum(tp+fp, 1), 1.)
    ap = np.sum(np.diff(recall)*np.maximum.accumulate(precision[::-1])[::-1][1:])
    f1 = 2*precision*recall/np.maximum(precision+recall, 1e-300)
    return np.array([ap, f1.max(), recall[precision >= 1].max(), counts[matches].sum()/npos])


def load_data() -> None:
    """원래 총계와 그룹 지표를 확인한 고정 실행을 메모리에 준비한다."""
    summary = base.read(base.OUT / "direction_summary.json")
    for seq in (s for city in base.CITY.values() for s in city):
        name = seq.replace("/", "_")
        with np.load(base.OUT / "directions" / (name+".npz")) as archive:
            directions = dict(archive)
        prepared = []
        for group in GROUPS:
            methods = []
            for method in METHODS:
                seeds = (0,) if method in ("identity", "fixed") else (1, 2, 3)
                runs = []
                for seed in seeds:
                    tag = method if seed == 0 else f"{method}_s{seed}"
                    path = base.MAIN / "test" / name / (tag+".npz")
                    assert base.digest(path) == summary["source_sha256"][str(path.relative_to(base.ROOT))]
                    with np.load(path) as archive: data = dict(archive)
                    mask = np.ones(len(data["score"]), bool) if group == "all" else directions["group"] == base.LABELS.index(group)
                    row = prepare(data["score"], data["top1_geo_dist"], data["min_past_gt_dist"], mask)
                    point = weighted(row, np.ones(len(mask), np.int32))
                    expected = summary["methods"][method]["per_seed"][str(seed)][group]["sequences"][seq]
                    if expected is None: assert point is None
                    else: np.testing.assert_allclose(point, [expected[k] for k in base.KEYS], atol=1e-12, rtol=0)
                    runs.append(row)
                methods.append(runs)
            prepared.append(methods)
        DATA[seq] = dict(n=len(directions["group"]), prepared=prepared)


def replicate(index: int) -> tuple:
    """모든 방법·방향에 같은 도시 내 시퀀스와100질의 블록을 뽑는다."""
    rng = np.random.default_rng([20261106, index])
    cities = np.full((len(CITIES), len(GROUPS), len(METHODS), 4), np.nan)
    for ci, city in enumerate(CITIES):
        samples = [[] for _ in GROUPS]
        seqs = base.CITY[city]
        for seq in rng.choice(seqs, len(seqs), replace=True):
            data = DATA[seq]
            n = data["n"]
            starts = rng.integers(0, n, size=(n+99)//100)
            ix = ((starts[:, None]+np.arange(100)) % n).ravel()[:n]
            counts = np.bincount(ix, minlength=n)
            for gi, methods in enumerate(data["prepared"]):
                values = []
                for runs in methods:
                    seeds = [weighted(row, counts) for row in runs]
                    values.append(None if seeds[0] is None else np.mean(seeds, axis=0))
                if values[0] is not None:
                    samples[gi].append(values)
        for gi, values in enumerate(samples):
            if values: cities[ci, gi] = np.mean(values, axis=0)
    return np.nanmean(cities, axis=0), cities


def main() -> None:
    """2000번 조건부 재표집의 도시별·전체 차이 구간을 기록한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    protocol = dict(created_unix=time.time(), replicates=2000, block_queries=100, rng_seed=20261106,
        groups=GROUPS, methods=METHODS, comparisons=["convs_only-fixed", "diff-fixed", "diff-identity"],
        resampling="고정 도시 안 시퀀스 복원추출, 선택된 시퀀스마다 원형 연속100질의 블록 복원추출. 같은 replicate에서 모든 방법·시작점·방향에 같은 표본을 적용. 시작점3개는 평균하고 다시 추출하지 않음.",
        zero_positive="선택 방향 양성이 없는 시퀀스는 원래 점추정처럼 제외. 재표집에서 한 도시의 해당 양성이 모두 사라지면 그 도시를 해당 replicate 방향평균에서 제외하고 도시별 유효 replicate 수와 도시 누락 횟수를 명시.",
        interpretation="고정된 데이터·체크포인트·사전학습시작점3개에 조건부인 사후 percentile95%구간. 학습선택 불확실성, 독립확증, 다중비교 보정·동등성 검정이 아니다. 기존 원장·정답 정의·최종 점수를 바꾸지 않는다.",
        sources={str(p.relative_to(base.ROOT)):base.digest(p) for p in (__import__('pathlib').Path(__file__), base.OUT/'direction_summary.json', base.OUT/'direction_protocol.json')})
    if (OUT / "protocol.json").exists():
        old = base.read(OUT / "protocol.json")
        assert old["sources"] == protocol["sources"]
    else: base.write(OUT / "protocol.json", protocol)
    load_data()
    with Pool(args.workers) as pool:
        rows = []
        for i, value in enumerate(pool.imap(replicate, range(2000), chunksize=4), 1):
            rows.append(value)
            if i % 100 == 0:
                base.write(OUT / "status.json", dict(done=i, total=2000))
                print("BOOTSTRAP", i, flush=True)
    overall = np.stack([r[0] for r in rows])
    cities = np.stack([r[1] for r in rows])
    np.savez_compressed(OUT / "samples.npz", overall=overall, cities=cities)
    source = base.read(base.OUT / "direction_summary.json")
    results = {}
    for first, second in (("convs_only", "fixed"), ("diff", "fixed"), ("diff", "identity")):
        mi, mj = METHODS.index(first), METHODS.index(second)
        result = {}
        for gi, group in enumerate(GROUPS):
            part = {}
            for ki, metric in enumerate(base.KEYS):
                point = 100*(source["methods"][first]["mean_std"][group][metric]["mean"]-source["methods"][second]["mean_std"][group][metric]["mean"])
                delta = 100*(overall[:, gi, mi, ki]-overall[:, gi, mj, ki])
                part[metric] = dict(delta_pp=point, ci95_pp=np.percentile(delta, [2.5, 97.5]).tolist())
            city_report = {}
            for ci, city in enumerate(CITIES):
                city_report[city] = {}
                for ki, metric in enumerate(base.KEYS):
                    def city_point(method):
                        runs=source["methods"][method]["per_seed"]
                        return np.mean([v[group]["cities"][city][metric] for v in runs.values()])
                    delta = 100*(cities[:, ci, gi, mi, ki]-cities[:, ci, gi, mj, ki])
                    finite=np.isfinite(delta)
                    city_report[city][metric]=dict(delta_pp=100*(city_point(first)-city_point(second)),
                        ci95_pp=np.percentile(delta[finite],[2.5,97.5]).tolist(),valid_replicates=int(finite.sum()),omitted_replicates=int((~finite).sum()))
            result[group]=dict(overall=part,cities=city_report)
        results[first+"-"+second]=result
    base.write(OUT / "summary.json",dict(status="passed",comparisons=results,protocol_sha256=base.digest(OUT/'protocol.json'),
        samples_sha256=base.digest(OUT/'samples.npz'),point_replay_passed=True))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
