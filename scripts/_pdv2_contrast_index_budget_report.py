#!/usr/bin/env python3
"""색인 후보 예산 실험의 원장·정답 포함률·집계를 독립 검산한다."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

import _pdv2_contrast_index_budget as study

ROOT, OUT = study.ROOT, study.OUT
METRICS = ("AP", "max_F1", "R@100P", "R@1")


def aggregate(values: dict, cities: dict, metric_names: tuple) -> dict:
    """양성 시퀀스→도시 균등 평균을 직접 계산한다."""
    groups = {}
    for city, sequences in cities.items():
        usable = [values[s] for s in sequences if values[s]["n_pos"] > 0]
        groups[city] = {m: float(np.mean([v[m] for v in usable])) for m in metric_names}
    return dict(groups=groups, macro={m: float(np.mean([g[m] for g in groups.values()]))
                                     for m in metric_names}, sequences=values)


def finish() -> dict:
    """완료 원장을 전수 검산하고 표2와 같은 집계 및 짝지은 구간을 쓴다."""
    started = time.time()
    spec = study.read(OUT / "protocol.json")
    for path, expected in spec["sources"].items():
        assert study.digest(ROOT / path) == expected, path
    cities = study.run.current.CITY
    rows = {f"{arm}_s{seed}": {} for arm in study.ARMS for seed in spec["seeds"]}
    counts = {a: 0 for a in study.ARMS}
    checks = dict(ledgers=0, queries=0, positive_queries=0,
                  maximum_metric_error=0.0, maximum_reference_score_error=0.0,
                  exact_matched_counts=True, past_only_candidates=True,
                  original_winners_exact=True)
    sources = {str(OUT.relative_to(ROOT) / "protocol.json"): study.digest(OUT / "protocol.json")}
    for sequence in spec["test"]:
        name = sequence.replace("/", "_")
        folder = OUT / "test" / name
        meta = study.read(folder / "summary.json")
        assert meta["status"] == "passed"
        sources[str((folder / "summary.json").relative_to(ROOT))] = study.digest(folder / "summary.json")
        pools = study.run.ARR(folder / "candidate_pools.npz")
        assert study.digest(folder / "candidate_pools.npz") == meta["candidate_pools_sha256"]
        cache_path = ROOT / next(iter(meta["input_sha256"]))
        for path, sha in meta["input_sha256"].items():
            assert study.digest(ROOT / path) == sha
        with np.load(cache_path) as cache:
            positions = cache["poses_gt"][:, :3, 3]
        q = pools["query_idx"]
        np.testing.assert_array_equal(q, np.arange(30, len(positions)-15))
        checks["queries"] += len(q)
        ref = study.run.ARR(study.MAIN / "test" / name / "diff_s1.npz")
        checks["positive_queries"] += int((ref["min_past_gt_dist"] <= 5).sum())
        generated_hits, sizes = {}, {}
        for arm in study.ARMS:
            offsets, ids = pools[arm + "_offset"], pools[arm + "_ids"]
            assert len(offsets) == len(q)+1 and offsets[0] == 0 and offsets[-1] == len(ids)
            size = np.diff(offsets)
            assert (size > 0).all() and (size <= 200).all()
            assert (ids >= 0).all() and (ids <= np.repeat(q-30, size)).all()
            repeated = np.repeat(q, size)
            pair_hit = np.linalg.norm(positions[ids]-positions[repeated], axis=1) <= 5
            hits = np.logical_or.reduceat(pair_hit, offsets[:-1])
            generated_hits[arm], sizes[arm] = hits, size
            counts[arm] += int(size.sum())
            for start, end in zip(offsets[:-1], offsets[1:]):
                assert (np.diff(ids[start:end]) > 0).all()
        for arm in ("key_matched", "aux_matched"):
            np.testing.assert_array_equal(sizes[arm], sizes["dual_100_100"])
        for arm in ("key_200", "aux_200"):
            np.testing.assert_array_equal(sizes[arm], np.minimum(q-29, 200))
        for seed in spec["seeds"]:
            old = study.run.ARR(study.MAIN / "test" / name / f"diff_s{seed}.npz")
            for arm in study.ARMS:
                tag = f"{arm}_s{seed}"
                rec = meta["records"][tag]
                path = ROOT / rec["path"]
                assert study.digest(path) == rec["sha256"]
                row = study.run.ARR(path)
                for field in ("query_idx", "min_past_gt_dist", "group"):
                    np.testing.assert_array_equal(row[field], old[field])
                np.testing.assert_array_equal(row["candidate_hit"], generated_hits[arm])
                np.testing.assert_array_equal(row["candidate_count"], sizes[arm])
                np.testing.assert_allclose(row["top1_geo_dist"],
                    np.linalg.norm(positions[q]-positions[row["j_star"]], axis=1), atol=0, rtol=0)
                offsets, ids = pools[arm + "_offset"], pools[arm + "_ids"]
                for i, winner in enumerate(row["j_star"]):
                    assert winner in ids[offsets[i]:offsets[i+1]]
                if arm == "dual_100_100":
                    np.testing.assert_array_equal(row["j_star"], old["j_star"])
                    err = float(np.max(abs(row["score"]-old["score"])))
                    assert err <= spec["score_atol"]
                    checks["maximum_reference_score_error"] = max(checks["maximum_reference_score_error"], err)
                actual = study.run.base.independent_metrics(row, row["score"])
                other = study.run.white.prior.BASE.exp.metrics(row["score"], row)
                for metric in METRICS:
                    if actual[metric] is not None:
                        err = max(abs(actual[metric]-other[metric]), abs(actual[metric]-rec["metrics"][metric]))
                        assert err < 1e-12
                        checks["maximum_metric_error"] = max(checks["maximum_metric_error"], err)
                positive = row["min_past_gt_dist"] <= 5
                recall = float(np.mean(row["candidate_hit"][positive])) if positive.any() else None
                assert recall == rec["candidate_recall"]
                if recall is not None:
                    assert actual["R@1"] <= recall + 1e-12
                rows[tag][sequence] = dict(actual, candidate_recall=recall)
                checks["ledgers"] += 1
    assert checks["queries"] == 195726
    assert checks["ledgers"] == 46*3*len(study.ARMS)
    metric_names = (*METRICS, "candidate_recall")
    per = {tag: aggregate(values, cities, metric_names) for tag, values in rows.items()}
    means, stds = {}, {}
    for arm in study.ARMS:
        means[arm] = {m: float(np.mean([per[f"{arm}_s{s}"]["macro"][m] for s in spec["seeds"]])) for m in metric_names}
        stds[arm] = {m: float(np.std([per[f"{arm}_s{s}"]["macro"][m] for s in spec["seeds"]], ddof=1)) for m in metric_names}
    rng = np.random.default_rng(20261106)
    comparisons = {}
    for arm in study.ARMS:
        if arm == "dual_100_100":
            continue
        samples = np.zeros((10000, len(metric_names)))
        for city, sequences in cities.items():
            usable = [s for s in sequences if rows["dual_100_100_s1"][s]["n_pos"] > 0]
            difference = np.array([[np.mean([rows[f"dual_100_100_s{i}"][s][m]-rows[f"{arm}_s{i}"][s][m]
                                             for i in spec["seeds"]]) for m in metric_names] for s in usable])
            selected = rng.integers(0, len(usable), size=(10000, len(usable)))
            samples += difference[selected].mean(axis=1) / len(cities)
        comparisons["dual-minus-" + arm] = dict(
            delta_pp={m: 100*(means["dual_100_100"][m]-means[arm][m]) for m in metric_names},
            conditional_95_pp={m: (100*np.quantile(samples[:, k], [.025, .975])).tolist()
                               for k, m in enumerate(metric_names)})
    main = study.read(study.MAIN / "summary.json")
    result = dict(status="passed", protocol_sha256=study.digest(OUT / "protocol.json"),
        per_case=per, mean=means, std=stds, comparisons=comparisons,
        mean_candidates={a: counts[a]/checks["queries"] for a in study.ARMS},
        original_main_metric_difference={m: means["dual_100_100"][m]-main["mean"]["diff"][m] for m in METRICS},
        aggregation=spec["aggregation"], checks=checks, sources=sources,
        bootstrap=dict(resamples=10000, seed=20261106, unit="within-city sequence",
                       fixed_start_mean=True, multiple_comparison_correction=False))
    study.write(OUT / "summary.json", result)
    study.write(OUT / "integrated_validation.json", dict(status="passed", **checks,
        summary_sha256=study.digest(OUT / "summary.json"),
        protocol_sha256=study.digest(OUT / "protocol.json"),
        verifier_sha256=study.digest(Path(__file__)), seconds=time.time()-started))
    print({"mean": means, "counts": result["mean_candidates"], "checks": checks}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    finish()
