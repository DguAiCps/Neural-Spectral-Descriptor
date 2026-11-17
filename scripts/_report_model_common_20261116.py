#!/usr/bin/env python3
"""보고모델 교체(2026-11-16) 단위 C 공통: 원장 경로, 도시 구성, 시퀀스 지표, 6도시 매크로, 짝지은 구간.

새 보고모델 원장(results/pdv2_scratch_search_20261116/ledgers)과 이전 보고모델 원장
(results/pdv2_diffattn_contrast_test_20261104)을 같은 함수로 읽어, 이전 원장으로 기존 발표 수치를 재현한 뒤
새 원장을 넣는다. torch 없이 호스트에서 돈다.

모델 꼬리표:
  new:<seed>     새 NSD (CFG@s<seed>@u512)          시험·검증
  uniform        재선택 설정의 균등 감산                시험·검증
  old:<seed>     이전 NSD diff_s<seed>                시험·검증(results/pdv2_diffattn_contrast_pipeline_20261104/val)
  oldfixed       이전 균등 감산(fixed)                시험
  bev:<seed>     BEVPlace++ (score_negl2)             시험·검증
  bevt:<seed>    BEVPlace++ + 시간 증거 (score)        시험
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from _pr_eval_lib import classify_queries, pr_sweep  # noqa: E402

CFG = "x=log__n=ln__g=frozen__lr=1e-3__U=1536"
SEARCH = ROOT / "results/pdv2_scratch_search_20261116"
OLD = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
BEV = ROOT / "results/pdv2_review_response_20261104/bev"
DIRS = ROOT / "results/pdv2_review_direction_yaw_20261105/directions"
OUT = ROOT / "results/pdv2_report_model_20261116/analysis"
NEW_SEEDS = (4, 5, 6)
REF_SEEDS = (1, 2, 3)          # 구성 선택에 쓴 시드(참고)
BEV_SEEDS = (1, 2, 3)
OLD_SEEDS = (1, 2, 3)
METRICS = ("AP", "max_F1", "R@100P", "R@1")
CITY = json.loads((ROOT / "results/pdv2_scratch_pilot_20261116/defaultbn_3seed/city.json").read_text())
CITY_OF = {s: c for c, v in CITY.items() for s in v}
SEQS = [s for v in CITY.values() for s in v]
BOOT = 10000
BOOT_SEED = 20261104


def path(tag: str, name: str, split: str = "test") -> tuple:
    """(npz 경로, 점수 필드)."""
    kind, _, seed = tag.partition(":")
    if kind == "new":
        return SEARCH / "ledgers" / split / name / f"{CFG}@s{seed}@u512.npz", "score"
    if kind == "uniform":
        return SEARCH / "ledgers" / split / name / "uniform.npz", "score"
    if kind == "old":
        base = OLD / "test" if split == "test" else ROOT / "results/pdv2_diffattn_contrast_pipeline_20261104/val"
        return base / name / f"diff_s{seed}.npz", "score"
    if kind == "oldfixed":
        assert split == "test"
        return OLD / "test" / name / "fixed.npz", "score"
    if kind == "bev":
        return BEV / split / f"s{seed}" / (name + ".npz"), "score_negl2"
    if kind == "bevt":   # BEVPlace++ + 같은 거리 대응 시간 증거(원장의 score 필드; NSD 모델과 무관)
        return BEV / split / f"s{seed}" / (name + ".npz"), "score"
    raise KeyError(tag)


def load(tag: str, seq: str, split: str = "test") -> dict:
    """원장 한 개: score(높을수록 수락), top1, gt(min_past_gt_dist), q, j."""
    p, f = path(tag, seq.replace("/", "_"), split)
    with np.load(p) as z:
        out = dict(score=z[f].astype(np.float64), top1=z["top1_geo_dist"], gt=z["min_past_gt_dist"], q=z["query_idx"])
        out["j"] = z["j_star"] if "j_star" in z.files else (z["top1_idx"] if "top1_idx" in z.files else None)
    return out


def tags(kind: str) -> list:
    return {"new": [f"new:{s}" for s in NEW_SEEDS], "ref": [f"new:{s}" for s in REF_SEEDS],
            "six": [f"new:{s}" for s in REF_SEEDS + NEW_SEEDS], "uniform": ["uniform"],
            "old": [f"old:{s}" for s in OLD_SEEDS], "oldfixed": ["oldfixed"],
            "bev": [f"bev:{s}" for s in BEV_SEEDS], "bevt": [f"bevt:{s}" for s in BEV_SEEDS]}[kind]


def seq_metric(d: dict, mask=None, extra=False) -> dict | None:
    """시퀀스 하나의 AP·최대F1·R@100P(+R@99P·R@95P)·R@1. mask는 양성 부분집합(방향별)."""
    qc = classify_queries(d["gt"])
    if mask is not None:
        qc["positive"] &= mask
        qc["ambiguous"] = ~(qc["positive"] | qc["negative"])
        qc["n_pos"] = int(qc["positive"].sum())
        qc["n_ambiguous"] = int(qc["ambiguous"].sum())
    if not qc["n_pos"]:
        return None
    v = pr_sweep(d["score"], d["top1"], qc)
    keys = ("AP", "max_F1", "R@100P") + (("R@99P", "R@95P") if extra else ())
    out = {k: float(v[k]) for k in keys}
    out["R@1"] = float(np.mean(d["top1"][qc["positive"]] <= 5))
    out["n_pos"] = qc["n_pos"]
    return out


def macro(per_seq: dict, keys=METRICS) -> dict:
    """시퀀스별 값(None=양성 없음) → 도시 평균 → 6도시 동일 가중(양성 있는 도시만)."""
    cities = {}
    for c, v in CITY.items():
        vals = [per_seq[s] for s in v if per_seq.get(s) is not None]
        if vals:
            cities[c] = {k: float(np.mean([x[k] for x in vals])) for k in keys}
    return dict(groups=cities, macro={k: float(np.mean([g[k] for g in cities.values()])) for k in keys})


def mean_sd(values: list) -> dict:
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a) > 1 else None)


def paired(rows_a: list, rows_b: list, keys=METRICS, seed=BOOT_SEED, n=BOOT) -> dict:
    """짝지은 차이와 도시 안 시퀀스 재표집 95% 구간(%p). rows_*: 시드별 {seq: metric|None}.

    기존 규약(_pdv2_contrast_pipeline_report.py): 시퀀스마다 각 쪽 시드 평균의 차이 → 도시 안 복원추출 →
    6도시 평균. 양성 판단은 rows_a[0] 기준.
    """
    rng = np.random.default_rng(seed)
    draws, city_delta = [], {}
    for c, v in CITY.items():
        avail = [s for s in v if rows_a[0].get(s) is not None]
        if not avail:
            continue
        mat = np.array([[100 * (np.mean([r[s][k] for r in rows_a]) - np.mean([r[s][k] for r in rows_b])) for k in keys]
                        for s in avail])
        city_delta[c] = dict(zip(keys, mat.mean(0).tolist()))
        idx = rng.integers(0, len(avail), size=(n, len(avail)))
        draws.append(mat[idx].mean(1))
    boot = np.stack(draws).mean(0)
    point = {k: float(np.mean([city_delta[c][k] for c in city_delta])) for k in keys}
    return dict(delta_pp=point, city_delta_pp=city_delta,
                conditional_95_pp={k: np.quantile(boot[:, i], [0.025, 0.975]).tolist() for i, k in enumerate(keys)})


def per_seq_rows(tag: str, split="test", masks=None, extra=False) -> dict:
    """{seq: metric}. masks: {seq: bool array} (방향별)."""
    out = {}
    for s in SEQS:
        try:
            d = load(tag, s, split)
        except FileNotFoundError:
            if split == "val":
                continue
            raise
        out[s] = seq_metric(d, None if masks is None else masks[s], extra)
    return out


def direction_masks(group: str) -> dict:
    """방향 층: forward_only=0, reverse_only=1 (results/pdv2_review_direction_yaw_20261105/directions)."""
    code = {"forward_only": 0, "reverse_only": 1, "mixed": 2}[group]
    out = {}
    for s in SEQS:
        with np.load(DIRS / (s.replace("/", "_") + ".npz")) as z:
            out[s] = z["group"] == code
    return out


def write(path_: Path, obj) -> None:
    path_.parent.mkdir(parents=True, exist_ok=True)
    path_.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
