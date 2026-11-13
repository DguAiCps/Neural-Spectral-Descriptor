#!/usr/bin/env python3
"""복소 위상을 유지한 크기 압축을 최종 판정에만 대조한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import _pdv2_current_calibration as cal

base = cal.base
REPO = cal.REPO
OUT = REPO / "results/pdv2_spectral_power_20261101"
LOG = REPO / "logs/spectral_power_20261101"
POWERS = {"p075": .75, "p050": .5}
read, arrays, sha256, write_json = cal.read, cal.arrays, cal.sha256, cal.write_json


def transform(x: np.ndarray, power: float) -> np.ndarray:
    """위상을 유지하고 각 복소 성분의 크기를 power승으로 압축한다."""
    magnitude = abs(x)
    factor = np.zeros_like(magnitude)
    np.power(magnitude, power - 1., out=factor, where=magnitude > 0)
    return (x * factor).astype(np.complex64)


def prepare() -> dict:
    """두 압축률과 기존36조건 교차를 새 측정 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = read(target)
        for p, h in spec["sources"].items():
            assert sha256(REPO / p) == h, p
        return spec
    prior = cal.prepare()
    rng = np.random.default_rng(5)
    x = (rng.normal(size=(2, 16, 8)) + 1j * rng.normal(size=(2, 16, 8))).astype(np.complex64)
    x[0, 0, 0] = 0
    phase = np.exp(1j * np.arange(1, 9) * .7)
    checks = {}
    np.testing.assert_array_equal(transform(x, 1.), x)
    for name, p in POWERS.items():
        y = transform(x, p)
        assert np.isfinite(y).all() and y[0, 0, 0] == 0 and y.shape == x.shape
        error = float(abs(transform((x * phase).astype(np.complex64), p) - y * phase).max())
        assert error < 5e-7
        np.testing.assert_allclose(abs(y), abs(x) ** p, atol=5e-7, rtol=0)
        checks[name] = {"rotation_equivariance_error": error, "zero_finite": True}
    paths = {Path(__file__), Path(cal.__file__), cal.OUT / "protocol.json"}
    spec = {k: prior[k] for k in ("seeds", "val", "models", "confidence", "selection")}
    spec.update({"created_unix": time.time(), "powers": POWERS,
        "scope": "현재 고정 모델·두색인100+100·9개기존융합승자·시간대응·원본/보정75:25·q+15·j<=q-30 유지. 인근 프레임 재선택 없음.",
        "method": "T_p(z)=z*abs(z)^(p-1), z=0이면0. p=.75,.5만 대조. 원본x와기존W*x에각각적용후기존판정. 학습통계 재적합이나검증분포정규화 없음. 스캔별 변환이며 미래를참조하지않음.",
        "cross": "기존검증캠페인의9융합x4판정채널과교차. p=1의36조건과새72조건을같은승격기준으로통합선택. 시험은전체에서한조건만. 추가계수탐색없음.",
        "motivation": "강한일부성분의지배를줄이는가설. 효과는측정전미확인. 조건설계시첫캠페인의일부검증출력은관찰했음. 독립확증이나순수사전등록으로표현하지않는다.",
        "synthetic": checks,
        "sources": {**prior["sources"], **{str(p.relative_to(REPO)): sha256(p) for p in paths}}})
    write_json(target, spec)
    return spec


def worker(seed: int, sequence: str) -> None:
    """고정 후보에서 두 새 변환의 최종 두 채널을 계산한다."""
    spec = read(OUT / "protocol.json")
    assert sha256(Path(__file__)) == spec["sources"][str(Path(__file__).relative_to(REPO))]
    name = sequence.replace("/", "_")
    folder = OUT / "val" / f"s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (name + ".json")
    if target.exists():
        assert read(target)["status"] == "passed"
        return
    source = cal.OUT / "val" / f"s{seed}" / (name + ".npz")
    prior_info = read(source.with_suffix(".json"))
    assert sha256(source) == prior_info["sha256"]
    saved = arrays(source)
    q, ids = saved["query_idx"], saved["candidates"]
    assert (ids <= q[:, None] - 30).all() and (ids >= 0).all()
    _, _, paths, _ = base.white.prior.BASE.exp.old.base.inputs("val", sequence)
    cache = arrays(paths["cache"])
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    x = base.white.coefficients(paths["spec"])
    wx = base.white.transform(x, arrays(base.white.OUT / "statistics.npz")["full"])
    pair, inv = np.unique((q[:, None] * len(x) + ids).ravel(), return_inverse=True)
    values, metrics, prefix = {}, {}, []
    tick = time.perf_counter()
    for name_power, power in POWERS.items():
        b = base.variant_inputs(transform(x, power), 8)
        w = base.variant_inputs(transform(wx, power), 8)
        for ch in ("c_spec", "c_col"):
            v = base.scores_for(b, w, travel, pair // len(x), pair % len(x), ch)
            values[name_power + "_" + ch] = v[inv].reshape(ids.shape)
        for i in (len(q) // 3, 2 * len(q) // 3):
            end = int(q[i]) + 16
            # 변환도 접두 입력부터 다시 만들어 관측 범위를 확인한다.
            short = base.variant_inputs(transform(x[:end], power), 8)
            sw = base.variant_inputs(transform(wx[:end], power), 8)
            for ch in ("c_spec", "c_col"):
                v = base.scores_for(short, sw, travel[:end], np.full(len(cal.FUSIONS), q[i]), ids[i], ch)
                np.testing.assert_array_equal(v, values[name_power + "_" + ch][i])
            prefix.append({"query": int(q[i]), "power": power, "exact": True})
        for fi, fusion in enumerate(cal.FUSIONS):
            row = {k: saved[k] for k in base.FIELDS}
            row.update(j_star=ids[:, fi], top1_geo_dist=saved["candidate_geo"][:, fi])
            for channel in cal.CHANNELS:
                score = cal.confidence(values[name_power + "_c_spec"][:, fi],
                                       values[name_power + "_c_col"][:, fi], channel,
                                       spec["confidence"][str(seed)]["decision"])
                a = base.independent_metrics(row, score)
                other = base.white.prior.BASE.exp.metrics(score, row)
                for m in base.METRICS:
                    if a["n_pos"]:
                        np.testing.assert_allclose(a[m], other[m], atol=1e-12, rtol=0)
                metrics[name_power + "__" + fusion + "__" + channel] = a
    ledger = folder / (name + ".npz")
    np.savez_compressed(ledger, query_idx=q, **values)
    write_json(target, {"status": "passed", "arms": metrics, "rows": len(q),
        "prefix_checks": prefix, "elapsed_s": time.perf_counter() - tick,
        "source": str(source.relative_to(REPO)), "source_sha256": sha256(source),
        "ledger": str(ledger.relative_to(REPO)), "sha256": sha256(ledger),
        "inputs": {str(p.relative_to(REPO)): sha256(p) for p in paths.values()}})
    print("DONE", seed, sequence, round(time.perf_counter() - tick, 1), flush=True)


def finish() -> None:
    """108조건의 최종 검증 결과를 비교하고 한 조건만 선택한다."""
    spec = prepare()
    old = read(cal.OUT / "summary.json")
    stats = {"p100__" + k: v for k, v in old["stats"].items()}
    per_seed = {"p100__" + k: v for k, v in old["per_seed"].items()}
    decisions = {"p100__" + k: v for k, v in old["decisions"].items()}
    records = {(s, q): read(OUT / "val" / f"s{s}" / (q.replace("/", "_") + ".json"))
               for s in spec["seeds"] for q in spec["val"]}
    for (seed, seq), r in records.items():
        assert r["status"] == "passed" and sha256(REPO / r["ledger"]) == r["sha256"]
        assert sha256(REPO / r["source"]) == r["source_sha256"]
        src, vals = arrays(REPO / r["source"]), arrays(REPO / r["ledger"])
        np.testing.assert_array_equal(src["query_idx"], vals["query_idx"])
        for fi, fusion in enumerate(cal.FUSIONS):
            row = {k: src[k] for k in base.FIELDS}
            row.update(j_star=src["candidates"][:, fi], top1_geo_dist=src["candidate_geo"][:, fi])
            for power in POWERS:
                for channel in cal.CHANNELS:
                    score = cal.confidence(vals[power + "_c_spec"][:, fi], vals[power + "_c_col"][:, fi],
                                           channel, spec["confidence"][str(seed)]["decision"])
                    assert base.independent_metrics(row, score) == r["arms"][power + "__" + fusion + "__" + channel]
    baseline = "p100__" + cal.BASELINE
    rule = spec["selection"]
    for arm in next(iter(records.values()))["arms"]:
        per_seed[arm] = {}
        for seed in spec["seeds"]:
            rows = {seq: records[(seed, seq)]["arms"][arm] for seq in spec["val"]}
            per_seed[arm][str(seed)] = {**base.macro(rows), "sequences": rows}
        stats[arm] = {m: {"mean": float(np.mean([v["macro"][m] for v in per_seed[arm].values()])),
                         "std": float(np.std([v["macro"][m] for v in per_seed[arm].values()], ddof=1))}
                      for m in base.METRICS}
        delta = {m: 100 * (stats[arm][m]["mean"] - stats[baseline][m]["mean"]) for m in base.METRICS}
        seed_delta = {str(s): 100 * (per_seed[arm][str(s)]["macro"]["AP"] - per_seed[baseline][str(s)]["macro"]["AP"]) for s in spec["seeds"]}
        family = {f: 100 * float(np.mean([per_seed[arm][str(s)]["families"][f]["AP"] - per_seed[baseline][str(s)]["families"][f]["AP"] for s in spec["seeds"]])) for f in ("kitti", "helipr", "mulran")}
        checks = {"AP": delta["AP"] >= rule["mean_AP_min_pp"], "F1": delta["max_F1"] >= rule["mean_F1_min_pp"],
                  "seed_AP": min(seed_delta.values()) >= rule["minimum_seed_AP_pp"], "R1": delta["R@1"] >= rule["mean_R1_min_pp"],
                  "R100": delta["R@100P"] >= rule["mean_R100_min_pp"], "family_AP": min(family.values()) >= rule["minimum_family_AP_pp"]}
        decisions[arm] = {"delta_pp": delta, "seed_AP_delta_pp": seed_delta, "family_AP_delta_pp": family,
                          "checks": checks, "eligible": all(checks.values())}
    eligible = [a for a, d in decisions.items() if d["eligible"]]
    selected = max(eligible, key=lambda a: stats[a]["AP"]["mean"]) if eligible else None
    write_json(OUT / "summary.json", {"stats": stats, "per_seed": per_seed, "decisions": decisions,
        "selected": selected, "test_used": False, "baseline": baseline})
    write_json(OUT / "completion_validation.json", {"status": "passed", "ledgers": len(records),
        "new_metric_sets": len(records) * len(POWERS) * len(cal.ARMS), "test_used": False,
        "prefix_checks": sum(len(r["prefix_checks"]) for r in records.values()),
        "summary_sha256": sha256(OUT / "summary.json")})
    print("SELECTED", selected, "BEST", sorted(stats, key=lambda a: stats[a]["AP"]["mean"], reverse=True)[:6], flush=True)


def run() -> None:
    """고정한 검증27작업을 세 개씩 실행한다."""
    spec = prepare()
    assert read(cal.OUT / "completion_validation.json")["status"] == "passed"
    LOG.mkdir(parents=True, exist_ok=True)
    def invoke(item: tuple) -> None:
        seed, seq = item
        with (LOG / f"s{seed}_{seq.replace('/', '_')}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__)), "worker", "--seed", str(seed),
                            "--sequence", seq], stdout=log, stderr=subprocess.STDOUT, check=True)
    tick = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(invoke, (s, q)) for s in spec["seeds"] for q in spec["val"]]
        for n, f in enumerate(concurrent.futures.as_completed(jobs), 1):
            f.result()
            write_json(OUT / "status.json", {"status": "running", "done": n, "total": len(jobs),
                                              "elapsed_s": time.time() - tick})
            print("PROGRESS", n, len(jobs), flush=True)
    finish()
    write_json(OUT / "status.json", {"status": "complete", "elapsed_s": time.time() - tick})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "run", "finish"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sequence", default="kitti360/0004")
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.seed, args.sequence)
    else:
        globals()[args.action]()
