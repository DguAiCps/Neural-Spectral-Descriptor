#!/usr/bin/env python3
"""현재 모델의 기존 융합 가중치와 스펙트럼 판정 채널을 검증한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import _pdv2_minimal_rerank as base

REPO = base.REPO
OUT = REPO / "results/pdv2_current_calibration_20261101"
LOG = REPO / "logs/current_calibration_20261101"
FUSIONS = ("original", "key_half", "key_double", "aux_half", "aux_double",
           "spec_half", "spec_double", "col_half", "col_double")
CHANNELS = ("legacy", "spectral", "column", "equal")
ARMS = tuple(f + "__" + c for f in FUSIONS for c in CHANNELS)
BASELINE = "original__legacy"
read, arrays = base.read, base.arrays
sha256, write_json = base.sha256, base.write_json


def fusion_weights(conf: dict) -> np.ndarray:
    """기존 네 가중치 중 하나만 절반 또는 두 배로 바꾼다."""
    weights = np.tile(np.asarray(conf["tuple"], np.float32), (len(FUSIONS), 1))
    for column in range(4):
        weights[1 + 2 * column, column] *= .5
        weights[2 + 2 * column, column] *= 2.
    return weights


def candidates(key: np.ndarray, bank: tuple, weights: np.ndarray,
               q: int) -> np.ndarray:
    """같은 두 과거 색인 합집합에서 각 융합식의 승자 하나를 고른다."""
    ids = np.unique(np.r_[base.prefix_topk(key, q, q - 29, 100),
                          base.prefix_topk(bank[0], q, q - 29, 100)])
    sn = base.minmax(key[ids] @ key[q])
    sh = base.minmax(bank[0][ids] @ bank[0][q])
    sa = base.minmax(base.align_scores(bank[1], bank[2], q, ids))
    sc = base.minmax(base.column_align_scores(bank[3], q, ids))
    winners = []
    for w in weights:
        score = w[0] * sn + w[1] * sh + w[2] * sa + w[3] * sc
        winners.append(ids[np.argmax(score)])
    return np.asarray(winners, np.int64)


def confidence(spec: np.ndarray, col: np.ndarray, channel: str,
               legacy: str) -> np.ndarray:
    """주파수 상관의 해석적 크기를 맞춘 후 기존 두 판정만 조합한다."""
    if channel == "legacy":
        return spec if legacy == "c_spec" else col
    if channel == "spectral":
        return spec
    if channel == "column":
        return col
    assert channel == "equal"
    return .5 * (30. * spec + col)


def synthetic_check() -> dict:
    """주파수 척도·회전 불변성과 영 입력의 유한성을 확인한다."""
    rng = np.random.default_rng(77)
    x = (rng.normal(size=(3, 16, 8)) + 1j * rng.normal(size=(3, 16, 8))).astype(np.complex64)
    x[1] = x[0] * np.exp(2j * np.pi * np.arange(1, 9) * 7 / 60)
    x[2] = 0
    bank = base.variant_inputs(x, 8)
    p = base.correlation_profiles(bank, np.array([0, 0, 2]), np.array([0, 1, 2]), "c_spec")
    c = base.correlation_profiles(bank, np.array([0, 0, 2]), np.array([0, 1, 2]), "c_col")
    np.testing.assert_allclose(30 * p[:2].max(1), 1., atol=5e-7, rtol=0)
    np.testing.assert_allclose(c[:2].max(1), 1., atol=5e-7, rtol=0)
    assert np.isfinite(p).all() and np.isfinite(c).all()
    np.testing.assert_array_equal(confidence(p.max(1), c.max(1), "legacy", "c_col"), c.max(1))
    return {"status": "passed", "spectral_self_match_times_30": float(30 * p[0].max()),
            "rotation_peak_shift": int(p[1].argmax()), "zero_input_finite": True}


def prepare() -> dict:
    """결과를 보기 전에 36조건과 채택 기준을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = read(target)
        for p, h in spec["sources"].items():
            assert sha256(REPO / p) == h, p
        return spec
    old = read(base.white.OUT / "protocol.json")
    sources = {Path(__file__), Path(base.__file__), Path(base.white.__file__),
               REPO / "scripts/_xs_eval_lib.py", REPO / "scripts/_causal_pr_dump.py",
               REPO / "scripts/_pdv2_sequence_fixed_test.py",
               REPO / "src/retrieval/sequence_evidence.py",
               REPO / "src/retrieval/spectral_evidence.py",
               REPO / "scripts/_pdv2_context_tail_finish.py",
               REPO / "scripts/_pdv2_context_accept.py",
               base.white.OUT / "protocol.json", base.white.OUT / "statistics.npz"}
    for seed in old["seeds"]:
        for sequence in old["val"]:
            name = sequence.replace("/", "_") + ".npz"
            sources.add(base.source_folder(seed, "val", old) / ("keys_" + name))
            sources.add(base.reference_path(seed, "val", name))
    spec = {k: old[k] for k in ("seeds", "val", "models", "confidence")}
    spec.update({"created_unix": time.time(), "arms": ARMS, "baseline": BASELINE,
        "scope": "현재288D+DiffAttnConv128D·재계량·통계·100+100후보·q+15·j<=q-30 고정. 인근 프레임 재선택은 사용하지 않는다. 재학습/신경망/저장특징 추가 없음.",
        "fusion": "기존 네 가중치의 한 축만 x0.5 또는 x2. 원래 가중치 포함9조건. 현재 minmax와 동률 작은 인덱스 유지.",
        "acceptance": "현재75:25 원본/상관보정 arc_unique_b100을 두 기존 채널에 계산. legacy, c_spec, c_col, (30*c_spec+c_col)/2의4조건. 마지막 조건은 최종 두 점수의 평균이며 경로별 혼합이 아니다. 30은60점 irfft의2/N에서 해석적으로 고정한다.",
        "selection": {"mean_AP_min_pp": .5, "mean_F1_min_pp": .5,
                      "minimum_seed_AP_pp": -1e-9, "mean_R1_min_pp": -1e-9,
                      "mean_R100_min_pp": -1., "minimum_family_AP_pp": -.5,
                      "rule": "모든 기준을 통과한 조건 중 3시드·계열 평균 AP 최대. 동률은 목록 순서. 최대 한 조건만 시험에 넘긴다. 불합격이면 시험 없이 진단한다."},
        "target": "표2 AP/F1 격차를 실제로 줄이는 연구. +.5%p 검증 승격은 comparable 달성 보장이 아니다.",
        "data_policy": "GT는 지표와 사후 손익 집계에만 사용. 선택은 검증만 사용. 시험은 과거 개발에 노출된 자료이며 새로운 독립 확증으로 부르지 않는다.",
        "synthetic_check": synthetic_check(),
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in sorted(sources)}})
    write_json(target, spec)
    return spec


def worker(seed: int, sequence: str, limit: int = 0) -> None:
    """검증 전체 질의에 최종 승자와 판정을 계산하고 원장을 보존한다."""
    spec = read(OUT / "protocol.json")
    assert sha256(Path(__file__)) == spec["sources"][str(Path(__file__).relative_to(REPO))]
    name = sequence.replace("/", "_") + ("_smoke" if limit else "")
    folder = OUT / "val" / f"s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (name + ".json")
    if target.exists():
        assert read(target)["status"] == "passed"
        return
    ref_path = base.reference_path(seed, "val", sequence.replace("/", "_") + ".npz")
    key_path = base.source_folder(seed, "val", spec) / ("keys_" + sequence.replace("/", "_") + ".npz")
    ref, key_data = arrays(ref_path), arrays(key_path)
    key = key_data["embedding"]
    _, _, paths, _ = base.white.prior.BASE.exp.old.base.inputs("val", sequence)
    cache = arrays(paths["cache"])
    np.testing.assert_array_equal(cache["scan_ids"], key_data["scan_ids"])
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(cache["scan_ids"], z["scan_ids"])
    x = base.white.coefficients(paths["spec"])
    bank = base.variant_inputs(x, 8)
    corrected = base.variant_inputs(base.white.transform(x, arrays(base.white.OUT / "statistics.npz")["full"]), 8)
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    positions = cache["poses_gt"][:, :3, 3]
    conf = spec["confidence"][str(seed)]
    if limit:
        ix = np.linspace(0, len(ref["query_idx"]) - 1, limit).astype(int)
        ref = {k: v[ix] for k, v in ref.items()}
    q, weights = ref["query_idx"], fusion_weights(conf)
    tick = time.perf_counter()
    ids = np.empty((len(q), len(FUSIONS)), np.int64)
    for i, t in enumerate(q):
        ids[i] = candidates(key, bank, weights, int(t))
    construction = time.perf_counter() - tick
    np.testing.assert_array_equal(ids[:, 0], ref["j_star"])
    assert (ids >= 0).all() and (ids <= q[:, None] - 30).all()
    tick = time.perf_counter()
    pair, inv = np.unique((q[:, None] * len(key) + ids).ravel(), return_inverse=True)
    scores = {}
    for channel in ("c_spec", "c_col"):
        values = base.scores_for(bank, corrected, travel, pair // len(key), pair % len(key), channel)
        scores[channel] = values[inv].reshape(ids.shape)
    elapsed = time.perf_counter() - tick
    error = float(abs(scores[conf["decision"]][:, 0] - ref["score"]).max())
    np.testing.assert_allclose(scores[conf["decision"]][:, 0], ref["score"], atol=1e-7, rtol=0)
    geo = np.linalg.norm(positions[q, None] - positions[ids], axis=-1)
    rows, changes = {}, {}
    pos = ref["min_past_gt_dist"] <= 5
    old_ok = pos & (ref["top1_geo_dist"] <= 5)
    for fi, fusion in enumerate(FUSIONS):
        row = {k: ref[k] for k in base.FIELDS}
        row.update(j_star=ids[:, fi], top1_geo_dist=geo[:, fi])
        new_ok = pos & (geo[:, fi] <= 5)
        changes[fusion] = {"winner_changed": int((ids[:, fi] != ref["j_star"]).sum()),
                           "gained_correct": int((new_ok & ~old_ok).sum()),
                           "lost_correct": int((old_ok & ~new_ok).sum())}
        for channel in CHANNELS:
            value = confidence(scores["c_spec"][:, fi], scores["c_col"][:, fi], channel, conf["decision"])
            metrics = base.independent_metrics(row, value)
            other = base.white.prior.BASE.exp.metrics(value, row)
            for m in base.METRICS:
                if metrics["n_pos"] and metrics[m] is not None:
                    np.testing.assert_allclose(metrics[m], other[m], atol=1e-12, rtol=0)
            rows[fusion + "__" + channel] = metrics
    expected = base.independent_metrics(ref, ref["score"])
    for m in base.METRICS:
        if expected[m] is not None:
            np.testing.assert_allclose(rows[BASELINE][m], expected[m], atol=1e-10, rtol=0)
    prefixes = []
    for i in (len(q) // 3, 2 * len(q) // 3):
        end, t = int(q[i]) + 16, int(q[i])
        b = base.white.prior.BASE.exp.truncate_bank(bank, end)
        c = base.white.prior.BASE.exp.truncate_bank(corrected, end)
        np.testing.assert_array_equal(candidates(key[:end], b, weights, t), ids[i])
        for ch in ("c_spec", "c_col"):
            tiny = base.scores_for(b, c, travel[:end], np.full(len(FUSIONS), t), ids[i], ch)
            np.testing.assert_array_equal(tiny, scores[ch][i])
        prefixes.append({"query": t, "candidates_and_two_scores_equal": True})
    ledger = folder / (name + ".npz")
    np.savez_compressed(ledger, **{k: ref[k] for k in base.FIELDS}, candidates=ids,
                        candidate_geo=geo, **scores)
    write_json(target, {"status": "passed", "seed": seed, "sequence": sequence,
        "rows": len(q), "arms": rows, "changes": changes,
        "prefix_checks": prefixes, "baseline_score_max_error": error,
        "timing_s": {"construction": construction, "two_final_scores": elapsed},
        "ledger": str(ledger.relative_to(REPO)), "sha256": sha256(ledger),
        "inputs": {str(p.relative_to(REPO)): sha256(p) for p in (ref_path, key_path, *paths.values())}})
    print("DONE", seed, sequence, len(q), round(construction + elapsed, 1), flush=True)


def finish() -> None:
    """원장과 독립 지표를 검산한 후 사전 기준으로 한 조건을 선택한다."""
    spec = prepare()
    records = {}
    for seed in spec["seeds"]:
        for seq in spec["val"]:
            rec = read(OUT / "val" / f"s{seed}" / (seq.replace("/", "_") + ".json"))
            assert rec["status"] == "passed"
            assert sha256(REPO / rec["ledger"]) == rec["sha256"]
            data = arrays(REPO / rec["ledger"])
            for fi, fusion in enumerate(FUSIONS):
                row = {k: data[k] for k in base.FIELDS}
                row.update(j_star=data["candidates"][:, fi], top1_geo_dist=data["candidate_geo"][:, fi])
                for channel in CHANNELS:
                    score = confidence(data["c_spec"][:, fi], data["c_col"][:, fi], channel,
                                       spec["confidence"][str(seed)]["decision"])
                    assert base.independent_metrics(row, score) == rec["arms"][fusion + "__" + channel]
            records[(seed, seq)] = rec
    per_seed, stats, decisions = {}, {}, {}
    for arm in ARMS:
        per_seed[arm] = {}
        for seed in spec["seeds"]:
            rows = {seq: records[(seed, seq)]["arms"][arm] for seq in spec["val"]}
            per_seed[arm][str(seed)] = {**base.macro(rows), "sequences": rows}
        stats[arm] = {m: {"mean": float(np.mean([v["macro"][m] for v in per_seed[arm].values()])),
                         "std": float(np.std([v["macro"][m] for v in per_seed[arm].values()], ddof=1))}
                      for m in base.METRICS}
    rule = spec["selection"]
    for arm in ARMS[1:]:
        delta = {m: 100 * (stats[arm][m]["mean"] - stats[BASELINE][m]["mean"]) for m in base.METRICS}
        seeds = {str(s): 100 * (per_seed[arm][str(s)]["macro"]["AP"] - per_seed[BASELINE][str(s)]["macro"]["AP"]) for s in spec["seeds"]}
        families = {f: 100 * float(np.mean([per_seed[arm][str(s)]["families"][f]["AP"] - per_seed[BASELINE][str(s)]["families"][f]["AP"] for s in spec["seeds"]])) for f in ("kitti", "helipr", "mulran")}
        checks = {"AP": delta["AP"] >= rule["mean_AP_min_pp"],
                  "F1": delta["max_F1"] >= rule["mean_F1_min_pp"],
                  "seed_AP": min(seeds.values()) >= rule["minimum_seed_AP_pp"],
                  "R1": delta["R@1"] >= rule["mean_R1_min_pp"],
                  "R100": delta["R@100P"] >= rule["mean_R100_min_pp"],
                  "family_AP": min(families.values()) >= rule["minimum_family_AP_pp"]}
        decisions[arm] = {"delta_pp": delta, "seed_AP_delta_pp": seeds,
                          "family_AP_delta_pp": families, "checks": checks, "eligible": all(checks.values())}
    eligible = [a for a in ARMS[1:] if decisions[a]["eligible"]]
    selected = max(eligible, key=lambda a: stats[a]["AP"]["mean"]) if eligible else None
    write_json(OUT / "summary.json", {"stats": stats, "per_seed": per_seed, "decisions": decisions,
                                      "selected": selected, "test_used": False})
    write_json(OUT / "completion_validation.json", {"status": "passed", "ledger_count": len(records),
        "final_metric_sets": len(records) * len(ARMS), "rows": sum(r["rows"] for r in records.values()),
        "baseline_max_error": max(r["baseline_score_max_error"] for r in records.values()),
        "prefix_checks": sum(len(r["prefix_checks"]) for r in records.values()),
        "test_used": False, "summary_sha256": sha256(OUT / "summary.json")})
    best = sorted(ARMS, key=lambda a: stats[a]["AP"]["mean"], reverse=True)[:6]
    print("SUMMARY", {a: {"metrics": stats[a], "decision": decisions.get(a)} for a in best},
          "SELECTED", selected, flush=True)


def run() -> None:
    """검증 시퀀스를 세 CPU 작업으로 실행한다."""
    spec = prepare()
    LOG.mkdir(parents=True, exist_ok=True)
    def invoke(item: tuple) -> None:
        seed, seq = item
        with (LOG / f"s{seed}_{seq.replace('/', '_')}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__)), "worker", "--seed", str(seed),
                            "--sequence", seq], stdout=log, stderr=subprocess.STDOUT, check=True)
    jobs = [(s, q) for s in spec["seeds"] for q in spec["val"]]
    tick = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(invoke, item): item for item in jobs}
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            write_json(OUT / "status.json", {"status": "running", "done": n, "total": len(jobs),
                                              "elapsed_s": time.time() - tick})
            print("PROGRESS", n, len(jobs), futures[future], flush=True)
    finish()
    write_json(OUT / "status.json", {"status": "complete", "elapsed_s": time.time() - tick})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "run", "finish"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sequence", default="kitti360/0004")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.seed, args.sequence, args.limit)
    else:
        globals()[args.action]()
