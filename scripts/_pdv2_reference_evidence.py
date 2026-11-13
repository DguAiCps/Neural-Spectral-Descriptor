#!/usr/bin/env python3
"""NSD의 배경 유사도 보정과 진폭을 보존한 시간 점수 혼합을 비교한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_evidence as exp
from _pdv2_no_context_ablation import sha256, write_json
from retrieval.spectral_evidence import correlation_profiles

OUT = REPO / "results/pdv2_reference_evidence_20261018"
METHODS = ("raw", "reference_centered", "reference_z", "temporal_blend25", "temporal_blend50")


def reference_indices(query: np.ndarray, count: int = 64) -> tuple:
    """j≤q−30의 접두 지도에서 중복 없이 등간격 기준 프레임을 뽑는다."""
    q = np.asarray(query, dtype=np.int64)
    assert q.ndim == 1 and (q >= 30).all()
    ends = q-30
    sizes = np.minimum(ends+1, count)
    k = np.arange(count)[None]
    indices = np.minimum(k*ends[:, None]//np.maximum(sizes[:, None]-1, 1), ends[:, None])
    return indices, k < sizes[:, None]


def calibrate(raw: np.ndarray, temporal: np.ndarray, refs: np.ndarray,
              valid: np.ndarray, scores: np.ndarray, winner: np.ndarray) -> dict:
    """선택 후보 주변을 제외한 배경 평균·편차를 추정한다."""
    mask = valid & (np.abs(refs-winner[:, None]) > 30)
    count = mask.sum(1)
    mean = (scores*mask).sum(1)/np.maximum(count, 1)
    var = (((scores-mean[:, None])**2)*mask).sum(1)/np.maximum(count, 1)
    usable = (count >= 8) & (var > 1e-16)
    return {"raw": raw, "reference_centered": np.where(usable, raw-mean, raw),
            "reference_z": np.where(usable, (raw-mean)/np.maximum(np.sqrt(var), 1e-8), raw),
            "temporal_blend25": .75*raw+.25*temporal,
            "temporal_blend50": .5*raw+.5*temporal,
            "background_count": count, "background_mean": mean,
            "background_std": np.sqrt(var), "fallback": ~usable}


def prepare() -> None:
    """첫 신호처리 결과를 본 후의 작은 후속 대조임을 기록한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    spec = exp.protocol(True)
    if (OUT / "protocol.json").exists():
        assert sha256(Path(__file__)) == exp.read(OUT / "protocol.json")["script_sha256"]
        return
    assert exp.read(exp.OUT / "status.json")["status"] == "complete"
    write_json(OUT / "protocol.json", {"created_unix": time.time(), "script_sha256": sha256(Path(__file__)),
        "input_protocol_sha256": sha256(exp.OUT / "protocol.json"),
        "input_summary_sha256": sha256(exp.OUT / "summary.json"), "methods": list(METHODS),
        "reference_count": 64, "guard_keyframes": 30, "minimum_background": 8,
        "seeds": [1, 2, 3], "models": spec["config"]["models"], "test_used": False,
        "scope": "단순PSR/시간평균의결과를본뒤고정한후속검증. 추가학습없음. 같은선택후보·원장·시드별정렬채널. 등간격과거참조64개에서승자±30프레임제외.8개미만또는분산0은기존점수로복귀. 진폭보존혼합은시간평균25/50%만고정비교. CFAR오경보율보장이나확률·독립확증아님."})


def worker(sequence: str) -> None:
    """참조의 회전 피크는 시퀀스마다 한 번 계산해 모든 대조에 공유한다."""
    target = OUT / f"{sequence.replace('/', '_')}.json"
    if target.exists():
        return
    spec = exp.protocol()
    sensor, seq, paths, _ = exp.old.base.inputs("val", sequence)
    with np.load(paths["spec"]) as z:
        bank = exp.old.base.bank_for(exp.OP, z)
    reference = exp.arrays(exp.OUT / f"s1/standard_{sensor}_{seq}.npz")
    q = reference["query_idx"]
    refs, valid = reference_indices(q)
    background = {k: np.empty(refs.shape) for k in ("c_col", "c_spec")}
    tick = time.perf_counter()
    for start in range(0, len(q), 128):
        sl = slice(start, start+128)
        qq = np.broadcast_to(q[sl, None], refs[sl].shape).reshape(-1)
        jj = refs[sl].reshape(-1)
        for channel in background:
            background[channel][sl] = correlation_profiles(bank, qq, jj, channel).max(1).reshape(refs[sl].shape)
    runtime = time.perf_counter()-tick
    result = {}
    for seed in (1, 2, 3):
        channel = exp.read(exp.old.base.BASE / f"s{seed}/confidence_selection.json")["ops"][exp.OP]["decision"]
        result[str(seed)] = {}
        for model in spec["config"]["models"]:
            source = exp.OUT / f"s{seed}/{model}_{sensor}_{seq}.npz"
            a = exp.arrays(source)
            np.testing.assert_array_equal(a["query_idx"], q)
            scores = calibrate(a["raw"], a["temporal_independent"], refs, valid, background[channel], a["j_star"])
            assert all(np.isfinite(v).all() for v in scores.values())
            dest = OUT / f"s{seed}/{model}_{sensor}_{seq}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **{k: a[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")}, **scores)
            result[str(seed)][model] = {"metrics": {m: exp.metrics(scores[m], a) for m in METHODS},
                "ledger": str(dest.relative_to(REPO)), "sha256": sha256(dest),
                "source_sha256": sha256(source), "fallback_fraction": float(scores["fallback"].mean())}
    write_json(target, {"sequence": sequence, "results": result, "shared_background_runtime_s": runtime,
        "queries": len(q), "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in paths.values()}})
    print("REFERENCE", sequence, runtime, flush=True)


def report() -> None:
    """모든 시드의 동일한 최종 평가를 집계한다."""
    spec = exp.protocol(True)
    rows = {seq: exp.read(OUT / f"{seq.replace('/', '_')}.json")["results"] for seq in spec["val"]}
    values, families = {}, {}
    count = 0
    for seed in ("1", "2", "3"):
        values[seed], families[seed] = {}, {}
        for model in spec["config"]["models"]:
            values[seed][model], families[seed][model] = {}, {}
            for seq, row in rows.items():
                item = row[seed][model]
                path = REPO / item["ledger"]
                assert sha256(path) == item["sha256"]
                ledger = exp.arrays(path)
                count += len(ledger["query_idx"])
                for method in METHODS:
                    assert exp.metrics(ledger[method], ledger) == item["metrics"][method]
            for method in METHODS:
                fam = {}
                for name in ("kitti", "helipr", "mulran"):
                    selected = [r[seed][model]["metrics"][method] for seq, r in rows.items() if seq.startswith(name) and r[seed][model]["metrics"][method]["n_pos"]]
                    fam[name] = {m: float(np.mean([v[m] for v in selected])) for m in exp.METRICS}
                families[seed][model][method] = fam
                values[seed][model][method] = {m: float(np.mean([v[m] for v in fam.values()])) for m in exp.METRICS}
    means = {model: {method: {m: {"mean": float(np.mean([values[s][model][method][m] for s in values])),
                                "std": float(np.std([values[s][model][method][m] for s in values], ddof=1))} for m in exp.METRICS} for method in METHODS} for model in spec["config"]["models"]}
    old = exp.read(exp.OUT / "summary.json")
    for seed in values:
        for model in values[seed]:
            assert values[seed][model]["raw"] == old["per_seed"][seed][model]["raw"]
    write_json(OUT / "summary.json", {"per_seed": values, "families": families, "mean_std": means, "test_used": False})
    write_json(OUT / "completion_validation.json", {"status": "passed", "rows": count, "ledgers": 54, "raw_reproduced": True, "test_used": False})
    print("REFERENCE_SUMMARY", means, flush=True)


def main() -> None:
    """작은 후속 대조를 3시드에서 완료한다."""
    prepare()
    tick = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for _ in pool.map(worker, exp.protocol()["val"]):
            pass
    report()
    write_json(OUT / "status.json", {"status": "complete", "runtime_s": time.time()-tick, "test_used": False})


if __name__ == "__main__":
    main()
