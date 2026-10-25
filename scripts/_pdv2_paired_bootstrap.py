#!/usr/bin/env python3
"""주표 비교의 짝지은 부트스트랩(할 일 C8): NSD·NSD-H − 비교 방법 차이의 신뢰구간과 p값.

설계(실행 전 고정, 2026-10-16). 이미 본 시험 기록에 대한 사후 분석이다.
- 기록: 주표와 같은 공통 질의(30 <= q < N-15, j <= q-30). NSD·NSD-H는 시드 1·2·3의 대칭 문맥
  기록과 시드별로 검증에서 고른 판정 점수, 비교 방법은 _pdv2_main_baselines_official.METHODS의
  점수와 _pdv2_baseline_seeds.SEED_FOLDERS의 시드 기록이다(학습 없는 SC++·RING·RING++는 실행 하나;
  2026-10-25 사용자 결정으로 시드 1에서 시드 평균으로 바꿈). 판정 점수와 질의 분류(양성 5 m 이내·음성 25 m 초과·나머지 제외)는 그대로 둔다.
- 재표집(두 단계, 모든 방법에 같은 표본): ① 도시마다 그 도시의 시퀀스를 복원 추출,
  ② 시퀀스마다 연속 질의 BLOCK=100개 블록을 복원 추출해 원래 길이만큼 이어 붙인다.
- 지표: 표본마다 시퀀스별 AP·최대 F1·R@100P·R@1을 다시 계산 → 도시 평균(양성이 있는 시퀀스만)
  → 6개 도시 평균. 시드가 여럿인 방법은 같은 표본에서 시드마다 계산해 평균한다(주표의 시드 평균).
- 반복 B=2000, 난수 시드 20261016. 차이의 95% 백분위 구간, 한쪽 p = P(차이 <= 0),
  지표·운영점마다 비교 방법들에 대해 Holm 보정.
- 구간은 평가한 학습 실행(시드 3개씩)에 조건을 둔다. 시드 간 변동은 재표집하지 않는다.
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pdv2_baseline_seeds import SEED_FOLDERS, TRAINING_FREE
from _pdv2_main_baselines_official import METHODS, sha256, write_json
from _pdv2_pilot import CITY, TEST
from _pr_eval_lib import classify_queries, pr_sweep

OUT = REPO / "results/pdv2_paired_bootstrap"
NSD_DIR = REPO / "results/pdv2_symmetric3_20261013"
OPS = (("pbev128_main", "NSD"), ("hspec12", "NSD-H"))
KEYS = ("AP", "max_F1", "R@100P", "R@1")
BLOCK, SEED = 100, 20261016
B = int(os.environ.get("BOOT_B", "2000"))      # 설계값 2000; 시험 실행만 줄인다
SEQS = [f"{s}/{q}" for s, q in TEST]
DATA = {}          # 작업자 프로세스가 fork로 물려받는다: method -> seq -> (score, top1); "_class" -> seq -> masks


def seq_metrics(score, top1, cls, ix):
    pos, neg = cls["positive"][ix], cls["negative"][ix]
    if not pos.any():
        return None
    qc = {"positive": pos, "negative": neg, "ambiguous": ~pos & ~neg, "n_pos": int(pos.sum()),
          "n_neg": int(neg.sum()), "n_ambiguous": int((~pos & ~neg).sum()), "n_total": len(ix)}
    sw = pr_sweep(score[ix], top1[ix], qc)
    return (float(sw["AP"]), float(sw["max_F1"]), float(sw["R@100P"]), float((top1[ix][pos] <= 5).mean()))


def macro(per_seq, seq_draw):
    """seq_draw: 도시 -> 뽑힌 시퀀스 이름 목록(중복 허용). per_seq: 이름 -> 지표 또는 None."""
    cities = []
    for names in seq_draw.values():
        rows = [per_seq[n] for n in names if per_seq[n] is not None]
        if rows:
            cities.append(np.mean(rows, axis=0))
    return np.mean(cities, axis=0)


def one_replicate(r):
    rng = np.random.default_rng([SEED, r])
    seq_draw = {c: list(rng.choice(names, size=len(names), replace=True)) for c, names in CITY_SEQS.items()}
    idx = {}
    for name in sorted(set(n for v in seq_draw.values() for n in v)):   # 같은 시퀀스는 같은 블록 표본; 정렬해 실행마다 같은 난수 순서
        n = len(DATA["_class"][name]["positive"])
        starts = np.arange(0, n, BLOCK)
        pick = rng.choice(starts, size=len(starts), replace=True)
        idx[name] = np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in pick])[:n]
    out = {}
    for method, seqs in DATA.items():
        if method == "_class":
            continue
        if isinstance(seqs, list):                               # 시드 셋: 시드마다 계산해 평균
            vals = [macro({nm: seq_metrics(*sd[nm], DATA["_class"][nm], idx[nm]) if nm in idx else None
                           for nm in SEQS}, seq_draw) for sd in seqs]
            out[method] = np.mean(vals, axis=0)
        else:
            out[method] = macro({nm: seq_metrics(*seqs[nm], DATA["_class"][nm], idx[nm]) if nm in idx else None
                                 for nm in SEQS}, seq_draw)
    return out


def holm(pvals: dict) -> dict:
    order = sorted(pvals, key=pvals.get)
    m, running, adj = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    global CITY_SEQS
    CITY_SEQS = {c: list(names) for c, names in CITY.items()}   # CITY: 도시 -> "sensor/seq" 목록
    sources = {}

    def track(path: Path) -> Path:
        sources[str(path.relative_to(REPO))] = sha256(path)
        return path

    track(Path(__file__))
    sel = json.loads(track(NSD_DIR / "summary.json").read_text())["stages"]["test_selected"]["ops"]
    DATA["_class"] = {}
    for op, label in OPS:
        DATA[label] = [{}, {}, {}]
    missing = []
    runs = {m: ([folder] if m in TRAINING_FREE else [f for _, f in SEED_FOLDERS[m]])
            for m, folder, *_ in METHODS}
    for method, folder, prefix, score_name, _, _ in METHODS:
        if not all((REPO / f / f"{prefix}_{s}_{q}.npz").exists() for f in runs[method] for s, q in TEST):
            missing.append(method)
        else:
            DATA[method] = [{} for _ in runs[method]] if len(runs[method]) > 1 else {}
    for s, q in TEST:
        nm = f"{s}/{q}"
        ref = np.load(track(NSD_DIR / f"s1/test_selected_symmetric_pbev128_main_{s}_{q}.npz"))
        DATA["_class"][nm] = classify_queries(ref["min_past_gt_dist"])
        for op, label in OPS:
            for k, sd in enumerate(("1", "2", "3")):
                n = np.load(track(NSD_DIR / f"s{sd}/test_selected_symmetric_{op}_{s}_{q}.npz"))
                np.testing.assert_array_equal(n["query_idx"], ref["query_idx"])
                dec = sel[op]["per_seed"][sd]["selection"]["decision"]
                DATA[label][k][nm] = (n[dec], n["top1_geo_dist"])
        for method, folder, prefix, score_name, _, _ in METHODS:
            if method in missing:
                continue
            for k, f in enumerate(runs[method]):
                b = np.load(track(REPO / f / f"{prefix}_{s}_{q}.npz"))
                ix = np.searchsorted(b["query_idx"], ref["query_idx"])
                np.testing.assert_array_equal(b["query_idx"][ix], ref["query_idx"])
                np.testing.assert_array_equal(b["min_past_gt_dist"][ix], ref["min_past_gt_dist"])
                row = DATA[method][k] if isinstance(DATA[method], list) else DATA[method]
                row[nm] = (b[score_name][ix], b["top1_geo_dist"][ix])
    # 점추정: 재표집 없이 원래 시퀀스·질의 전부(주표 값과 같아야 한다)
    full_idx = {nm: np.arange(len(DATA["_class"][nm]["positive"])) for nm in SEQS}
    full_draw = {c: list(v) for c, v in CITY_SEQS.items()}
    point = {}
    for method, seqs in DATA.items():
        if method == "_class":
            continue
        if isinstance(seqs, list):
            point[method] = np.mean([macro({nm: seq_metrics(*sd[nm], DATA["_class"][nm], full_idx[nm])
                                            for nm in SEQS}, full_draw) for sd in seqs], axis=0)
        else:
            point[method] = macro({nm: seq_metrics(*seqs[nm], DATA["_class"][nm], full_idx[nm])
                                   for nm in SEQS}, full_draw)
    with Pool(int(os.environ.get("NPROC", "16"))) as pool:
        reps = pool.map(one_replicate, range(B), chunksize=10)
    baselines = [m for m in DATA if m not in ("_class", "NSD", "NSD-H")]
    result = {"point": {m: dict(zip(KEYS, map(float, v))) for m, v in point.items()}, "comparisons": {}}
    for _, label in OPS:
        for ki, k in enumerate(KEYS):
            pv = {}
            rows = {}
            for m in baselines:
                d = np.array([r[label][ki] - r[m][ki] for r in reps])
                rows[m] = {"diff": float(point[label][ki] - point[m][ki]),
                           "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                           "p_one_sided": float((np.sum(d <= 0) + 1) / (len(d) + 1))}
                pv[m] = rows[m]["p_one_sided"]
            for m, a in holm(pv).items():
                rows[m]["p_holm"] = float(a)
            result["comparisons"].setdefault(label, {})[k] = rows
    write_json(OUT / "summary.json", {
        "design": {"block_queries": BLOCK, "replicates": B, "seed": SEED,
                   "levels": "도시 안 시퀀스 복원 추출 → 시퀀스 안 연속 질의 블록 복원 추출(모든 방법 공통)",
                   "nsd": "시드 1·2·3을 같은 표본에서 계산해 평균",
                   "baselines": "학습 방법은 시드 3개(DeepRING 1·3·4)를 같은 표본에서 계산해 평균, 학습 없는 방법은 실행 하나",
                   "seed_runs": runs,
                   "p": "한쪽 p = (#{차이 <= 0} + 1) / (B + 1), 지표·운영점마다 Holm 보정",
                   "conditional_on": "평가한 학습 실행(시드 3개씩); 시드 간 변동은 재표집하지 않음",
                   "post_hoc": "이미 본 시험 기록에 대한 사후 분석"},
        "missing_methods": missing, **result, "input_sha256": sources, "checks_passed": True})
    for _, label in OPS:
        for m, row in result["comparisons"][label]["AP"].items():
            print(f"{label} - {m:9s} AP diff {row['diff']:+.4f} CI [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] "
                  f"p={row['p_one_sided']:.4f} holm={row['p_holm']:.4f}")


if __name__ == "__main__":
    main()
