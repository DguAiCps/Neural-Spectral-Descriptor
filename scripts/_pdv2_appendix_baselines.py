#!/usr/bin/env python3
"""주표 비교의 부록 분석: 방향별 지표, 정답 반경 10 m, NSD 판정 통계 이식.

주표(results/pdv2_main_official_baselines)와 같은 공통 질의(30 <= q < N-15, j <= q-30)와
같은 시험 기록을 다시 집계한다. 새 학습·추론·선택은 없다.
- 방향별: NSD 기록의 질의 분류(group: 0 같은 방향 정답 있음, 1 반대 방향 정답만, 2 기타)로
  양성을 나눈다. 다른 방향의 양성은 그 분석에서 애매한 질의처럼 뺀다. 음성은 그대로 둔다.
- 10 m: 양성 정의(5 m 이내)는 그대로 두고, 돌려준 1위 후보를 맞았다고 보는 반경만 10 m로 넓힌다.
- 판정 통계 이식: 각 비교 방법이 고른 1위 후보에 NSD의 학습 없는 판정 통계 c_spec
  (조감도 스펙트럼, 시드와 무관)을 고정해서 붙인다. 고르기·조정은 없다. R@1은 1위 후보가
  같으므로 바뀌지 않는다. 다시 계산한 c_spec이 NSD 기록 값과 같은지 먼저 확인한다.
NSD·NSD-H와 학습하는 비교 방법은 시드 3개 평균과 표본 표준편차(시드 목록은 _pdv2_baseline_seeds.SEED_FOLDERS,
DeepRING 1·3·4), 학습 없는 SC++·RING·RING++는 실행 하나의 값이다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _pr_eval_lib as P
from _pdv2_baseline_seeds import SEED_FOLDERS, TRAINING_FREE
from _pdv2_main_baselines_official import METHODS, sha256, write_json
from _pdv2_pilot import CITY, TEST
from _pr_eval_lib import classify_queries, pr_sweep
from _xs_eval_lib import variant_inputs

OUT = REPO / "results/pdv2_appendix_baselines"
NSD_DIR = REPO / "results/pdv2_symmetric3_20261013"
SPEC = REPO / "data/preprocessed_v2_odom_bevclip/testspec_{}_{}.npz"
OPS = (("pbev128_main", "NSD"), ("hspec12", "NSD-H"))
KEYS = ("AP", "max_F1", "R@100P", "R@1")
VARIANTS = ("all", "same", "opposite", "r10", "cspec")


def pair_spec(hfft: np.ndarray, hnorm: np.ndarray, q: np.ndarray, j: np.ndarray) -> np.ndarray:
    """NSD의 c_spec: 순환 이동 상관의 최댓값을 두 크기로 나눈다(학습 없음)."""
    out = np.full(len(q), np.nan)
    idx = np.nonzero(j >= 0)[0]
    n = (hfft.shape[-1] - 1) * 2
    for s in range(0, len(idx), 2000):
        k = idx[s:s + 2000]
        c = np.fft.irfft(hfft[j[k]] * np.conj(hfft[q[k]]), n=n, axis=-1).sum(axis=1).max(axis=-1)
        out[k] = c / (hnorm[j[k]] * hnorm[q[k]])
    return out


def metrics(score, top1, mpd, group=None, subset=None, radius=5.0):
    """한 시퀀스의 AP·최대 F1·R@100P·R@1. 양성이 없으면 None."""
    qc = classify_queries(mpd)
    if subset is not None:
        keep = qc["positive"] & (group == subset)
        qc = dict(qc)
        qc["ambiguous"] = qc["ambiguous"] | (qc["positive"] & ~keep)
        qc["positive"], qc["n_pos"] = keep, int(keep.sum())
    if not qc["n_pos"]:
        return None
    old = P.MATCH_RADIUS_M
    P.MATCH_RADIUS_M = radius
    try:
        sw = pr_sweep(score, top1, qc)
    finally:
        P.MATCH_RADIUS_M = old
    return {"AP": float(sw["AP"]), "max_F1": float(sw["max_F1"]), "R@100P": float(sw["R@100P"]),
            "R@1": float((top1[qc["positive"]] <= radius).mean())}


def macro(per: dict) -> dict:
    """시퀀스 → 도시 6개 평균 → 매크로(양성이 있는 시퀀스만)."""
    cities = []
    for names in CITY.values():
        rows = [per[s] for s in names if per.get(s) is not None]
        if rows:
            cities.append({k: np.mean([r[k] for r in rows]) for k in KEYS})
    return {k: float(np.mean([c[k] for c in cities])) for k in KEYS}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}

    def track(path: Path) -> Path:
        sources[str(path.relative_to(REPO))] = sha256(path)
        return path

    track(Path(__file__))
    sel = json.loads(track(NSD_DIR / "summary.json").read_text())["stages"]["test_selected"]["ops"]
    per = {}                     # name -> variant -> seq -> metrics
    max_err = 0.0
    for sensor, seq in TEST:
        key = f"{sensor}/{seq}"
        sp = np.load(track(Path(str(SPEC).format(sensor, seq))))
        sk = sp["sk_re"].astype(np.float32) + 1j * sp["sk_im"].astype(np.float32)
        _, bfft, bnorm, _ = variant_inputs(sk, 8)
        ref = np.load(track(NSD_DIR / f"s1/test_selected_symmetric_pbev128_main_{sensor}_{seq}.npz"))
        group = ref["group"]
        for op, label in OPS:
            for sd in ("1", "2", "3"):
                n = np.load(track(NSD_DIR / f"s{sd}/test_selected_symmetric_{op}_{sensor}_{seq}.npz"))
                np.testing.assert_array_equal(n["query_idx"], ref["query_idx"])
                dec = sel[op]["per_seed"][sd]["selection"]["decision"]
                if op == "pbev128_main" and sd == "1":        # c_spec 재계산이 기록과 같은지 확인
                    cs = pair_spec(bfft, bnorm, n["query_idx"], n["j_star"])
                    f = np.isfinite(n["c_spec"])
                    max_err = max(max_err, float(np.nanmax(np.abs(cs[f] - n["c_spec"][f]))))
                row = per.setdefault(f"{label}|s{sd}", {})
                s, t, m = n[dec], n["top1_geo_dist"], n["min_past_gt_dist"]
                row.setdefault("all", {})[key] = metrics(s, t, m)
                row.setdefault("same", {})[key] = metrics(s, t, m, group, 0)
                row.setdefault("opposite", {})[key] = metrics(s, t, m, group, 1)
                row.setdefault("r10", {})[key] = metrics(s, t, m, radius=10.0)
        for method, folder, prefix, score_name, _, _ in METHODS:
            runs = [(None, folder)] if method in TRAINING_FREE else SEED_FOLDERS[method]
            for seed, f in runs:
                path = REPO / f / f"{prefix}_{sensor}_{seq}.npz"
                if not path.exists():
                    continue
                b = np.load(track(path))
                ix = np.searchsorted(b["query_idx"], ref["query_idx"])
                np.testing.assert_array_equal(b["query_idx"][ix], ref["query_idx"])
                s, t, m = b[score_name][ix], b["top1_geo_dist"][ix], b["min_past_gt_dist"][ix]
                row = per.setdefault(method if seed is None else f"{method}|s{seed}", {})
                row.setdefault("all", {})[key] = metrics(s, t, m)
                row.setdefault("same", {})[key] = metrics(s, t, m, group, 0)
                row.setdefault("opposite", {})[key] = metrics(s, t, m, group, 1)
                row.setdefault("r10", {})[key] = metrics(s, t, m, radius=10.0)
                cs = pair_spec(bfft, bnorm, b["query_idx"][ix], b["j_star"][ix])
                row.setdefault("cspec", {})[key] = metrics(cs, t, m)
        print(key, flush=True)
    assert max_err < 1e-5, f"c_spec 재계산이 NSD 기록과 다르다: {max_err}"
    result = {}
    for name, variants in per.items():
        if any(len(v) != len(TEST) for v in variants.values()):
            continue                                    # 46개가 다 없는 방법은 뺀다
        result[name] = {v: macro(seqs) for v, seqs in variants.items()}
    groups = {label: [f"{label}|s{sd}" for sd in ("1", "2", "3")] for _, label in OPS}
    groups.update({m: [f"{m}|s{sd}" for sd, _ in runs] for m, runs in SEED_FOLDERS.items()})
    for name, keys in groups.items():
        assert all(k in result for k in keys), (name, [k for k in keys if k not in result])
        seeds = [result.pop(k) for k in keys]
        result[name] = {v: {k: {"mean": float(np.mean([s[v][k] for s in seeds])),
                                "std": float(np.std([s[v][k] for s in seeds], ddof=1)),
                                "values": [float(s[v][k]) for s in seeds]}
                            for k in KEYS} for v in seeds[0]}
    write_json(OUT / "summary.json", {
        "scope": "주표 공통 질의의 부록 분석(방향별, 정답 반경 10 m, NSD c_spec 이식); 새 선택·추론 없음",
        "direction_groups": {"same": 0, "opposite": 1},
        "r10": "양성 5 m 정의 유지, 1위 후보 정답 반경 10 m",
        "cspec": "각 방법의 1위 후보에 NSD 조감도 c_spec(학습 없음)을 고정 적용",
        "cspec_recompute_max_abs_err": max_err,
        "methods": result, "input_sha256": sources, "checks_passed": True})
    for name, v in result.items():
        print(name, {var: round((x["AP"]["mean"] if isinstance(x["AP"], dict) else x["AP"]), 4)
                     for var, x in v.items()})


if __name__ == "__main__":
    main()
