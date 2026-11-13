#!/usr/bin/env python3
"""검증에서 고정한 25% 상관 보정만 표2 시험 규약으로 평가한다."""
import argparse
import concurrent.futures
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import _pdv2_spectral_whitening as source
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_no_context_ablation import sha256, write_json

run = source.prior
OUT = source.OUT / "fixed_test"
LOG = source.LOG / "fixed_test"
FRACTION = 0.25
METRICS = run.METRICS


def freeze() -> dict:
    """검증 선택값과 실행·통계·시험 입력을 새 평가 전에 고정한다."""
    OUT.mkdir(exist_ok=True)
    target = OUT / "freeze.json"
    if target.exists():
        spec = run.read(target)
        for path, digest in spec["sources"].items():
            assert sha256(run.REPO / path) == digest, path
        return spec
    validation = run.read(source.OUT / "blend/summary.json")
    assert validation["selected"] == "quarter"
    assert all(validation["decisions"]["quarter"]["checks"].values())
    old = source.protocol()
    original_freeze = run.read(run.PRIOR / "test_freeze.json")
    original_table = run.read(run.PRIOR / "table2_summary.json")
    assert original_table["checks_passed"]
    for seed in run.SEEDS:
        assert (
            old["models"][str(seed)]
            == original_freeze["models"][str(seed)]["octave288"]["epoch"]
        )
    paths = {
        Path(__file__),
        source.OUT / "protocol.json",
        source.OUT / "statistics.npz",
        source.OUT / "blend/summary.json",
        source.OUT / "blend/protocol.json",
        source.OUT / "blend/completion_validation.json",
        run.PRIOR / "table2_summary.json",
        run.PRIOR / "completion_validation.json",
        run.PRIOR / "test_freeze.json",
    }
    paths.update(run.REPO / p for p in old["sources"])
    for sequence in original_freeze["test"]:
        paths.update(run.BASE.exp.old.base.inputs("test", sequence)[2].values())
        for seed in run.SEEDS:
            p = (
                run.root(seed, "control")
                / "test"
                / (sequence.replace("/", "_") + ".npz")
            )
            paths.update((p, p.with_suffix(".json"), p.with_name("coverage_" + p.name)))
    spec = {
        "created_unix": time.time(),
        "fraction": FRACTION,
        "test": original_freeze["test"],
        "cities": original_freeze["cities"],
        "seeds": run.SEEDS,
        "confidence": old["confidence"],
        "models": {
            str(s): original_freeze["models"][str(s)]["octave288"] for s in run.SEEDS
        },
        "scope": "검증에서선택한기존75%+전체상관보정25%만평가. 학습통계/모델/후보/승자/q+15/j<=q-30동일. 기존195726공통질의·46원장·45양성시퀀스→6도시→3시드표본평균/표준편차. 원고자동교체없음.",
        "test_used_for_selection": False,
        "test_exposure": "이전개발에서이미평가한시험분할이며새로운미노출확증으로표현하지않는다. 이시험결과로혼합률이나보정통계를수정하지않는다.",
        "sources": {str(p.relative_to(run.REPO)): sha256(p) for p in sorted(paths)},
    }
    write_json(target, spec)
    print("FROZEN_TEST", len(paths), flush=True)
    return spec


def evaluate(seed: int, sequence: str) -> None:
    """기존 최종 원장과 같은 승자에 고정된 새 판정만 적용한다."""
    spec = run.read(OUT / "freeze.json")
    assert spec["fraction"] == FRACTION
    assert (
        sha256(Path(__file__))
        == spec["sources"][str(Path(__file__).relative_to(run.REPO))]
    )
    name = sequence.replace("/", "_") + ".npz"
    folder = OUT / f"s{seed}"
    folder.mkdir(exist_ok=True)
    target = folder / name
    assert not target.exists(), target
    original = run.root(seed, "control") / "test" / name
    row = run.BASE.exp.arrays(original)
    info = run.read(original.with_suffix(".json"))
    assert sha256(original) == info["sha256"]
    assert info["checkpoint_sha256"] == spec["models"][str(seed)]["sha256"]
    coverage = run.BASE.exp.arrays(original.with_name("coverage_" + name))
    _, _, paths, _ = run.BASE.exp.old.base.inputs("test", sequence)
    cache = run.BASE.exp.arrays(paths["cache"])
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(cache["scan_ids"], z["scan_ids"])
    x = source.coefficients(paths["spec"])
    q, j = row["query_idx"], row["j_star"]
    assert (j <= q - 30).all() and (q + 15 < len(x)).all()
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    channel = spec["confidence"][str(seed)]["decision"]
    matrix = run.BASE.exp.arrays(source.OUT / "statistics.npz")["full"]
    values, prefix = {}, {}
    for arm in ("identity", "full"):
        modified = x if arm == "identity" else source.transform(x, matrix)
        bank = source.variant_inputs(modified, 8)
        raw = (
            coverage[channel]
            if arm == "identity"
            else run.BASE.fixed.correlation_profiles(bank, q, j, channel).max(1)
        )
        chunks = []
        for start in range(0, len(q), 512):
            sl = slice(start, start + 512)
            chunks.append(
                run.BASE.fixed.frozen_scores(
                    bank, travel, q[sl], j[sl], channel, raw[sl]
                )[source.METHOD]
            )
        values[arm] = np.concatenate(chunks)
        if arm == "identity":
            np.testing.assert_array_equal(values[arm], row[source.METHOD])
        prefix[arm] = []
        for index in (len(q) // 3, 2 * len(q) // 3):
            end = int(q[index]) + 16
            small = run.BASE.exp.truncate_bank(bank, end)
            check = run.BASE.fixed.frozen_scores(
                small,
                travel[:end],
                q[index : index + 1],
                j[index : index + 1],
                channel,
                raw[index : index + 1],
            )[source.METHOD]
            np.testing.assert_array_equal(check, values[arm][index : index + 1])
            prefix[arm].append({"query": int(q[index]), "error": 0.0})
    score = (1 - FRACTION) * values["identity"] + FRACTION * values["full"]
    assert np.isfinite(score).all()
    metric = independent_metrics(row, score)
    other = run.BASE.exp.metrics(score, row)
    for key in METRICS:
        if metric[key] is not None:
            np.testing.assert_allclose(metric[key], other[key], atol=1e-12, rtol=0)
    np.savez_compressed(
        target,
        **{k: row[k] for k in run.BASE.FIELDS},
        baseline=values["identity"],
        score=score,
    )
    write_json(
        target.with_suffix(".json"),
        {
            "sha256": sha256(target),
            "metrics": metric,
            "prefixes": prefix,
            "original_sha256": sha256(original),
            "freeze_sha256": sha256(OUT / "freeze.json"),
            "winner_unchanged": True,
        },
    )


def macro(rows: dict, cities: dict) -> dict:
    """양성 시퀀스와 도시를 균등 평균한다."""
    grouped = {}
    for city, sequences in cities.items():
        selected = [rows[s] for s in sequences if rows[s]["n_pos"]]
        grouped[city] = {m: float(np.mean([r[m] for r in selected])) for m in METRICS}
    return {
        "cities": grouped,
        "macro": {m: float(np.mean([r[m] for r in grouped.values()])) for m in METRICS},
    }


def finish() -> None:
    """표2 최종 지표를 재계산하고 보존된 공식3시드와 비교한다."""
    spec = freeze()
    previous = run.read(run.PRIOR / "table2_summary.json")
    per_seed, per_sequence, counts = {}, {}, {}
    sources = {}
    for seed in run.SEEDS:
        rows, total = {}, 0
        for sequence in spec["test"]:
            p = OUT / f"s{seed}" / (sequence.replace("/", "_") + ".npz")
            info, row = run.read(p.with_suffix(".json")), run.BASE.exp.arrays(p)
            assert sha256(p) == info["sha256"]
            sources[str(p.relative_to(run.REPO))] = info["sha256"]
            original = run.BASE.exp.arrays(run.root(seed, "control") / "test" / p.name)
            for k in run.BASE.FIELDS:
                np.testing.assert_array_equal(row[k], original[k])
            np.testing.assert_array_equal(row["baseline"], original[source.METHOD])
            baseline = independent_metrics(row, row["baseline"])
            assert (
                baseline == previous["per_sequence"]["octave288"][str(seed)][sequence]
            )
            rows[sequence] = independent_metrics(row, row["score"])
            assert rows[sequence] == info["metrics"]
            assert all(r["error"] == 0 for v in info["prefixes"].values() for r in v)
            total += len(row["query_idx"])
        assert total == 195726
        counts[str(seed)] = total
        per_sequence[str(seed)], per_seed[str(seed)] = rows, macro(rows, spec["cities"])
    stats = {
        m: {
            "mean": float(np.mean([r["macro"][m] for r in per_seed.values()])),
            "std": float(np.std([r["macro"][m] for r in per_seed.values()], ddof=1)),
        }
        for m in METRICS
    }
    comparison = {
        "octave288": previous["mean_std"]["octave288"],
        "whitening25": stats,
        "bevplace": previous["mean_std"]["bevplace"],
    }
    delta = {
        ref: {m: 100 * (stats[m]["mean"] - comparison[ref][m]["mean"]) for m in METRICS}
        for ref in ("octave288", "bevplace")
    }
    write_json(
        OUT / "table2_summary.json",
        {
            "mean_std": comparison,
            "per_seed": per_seed,
            "per_sequence": per_sequence,
            "delta_pp": delta,
            "query_counts": counts,
            "freeze_sha256": sha256(OUT / "freeze.json"),
            "test_used_for_selection": False,
            "paper_changed": False,
            "checks_passed": True,
            "sources": sources,
        },
    )
    with (OUT / "table2.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method"] + [m + "_" + k for m in METRICS for k in ("mean_pct", "std_pp")]
        )
        for name, values in comparison.items():
            writer.writerow(
                [name] + [100 * values[m][k] for m in METRICS for k in ("mean", "std")]
            )
    write_json(
        OUT / "completion_validation.json",
        {
            "status": "passed",
            "ledgers": 138,
            "query_rows": sum(counts.values()),
            "independent_metric_checks": 552,
            "prefix_checks": 552,
            "winner_and_baseline_exact": True,
            "summary_sha256": sha256(OUT / "table2_summary.json"),
            "csv_sha256": sha256(OUT / "table2.csv"),
        },
    )
    print("TABLE2_COMPLETE", comparison, delta, flush=True)


def run_all() -> None:
    """고정한 138시퀀스 평가와 최종 집계를 실행한다."""
    LOG.mkdir(parents=True, exist_ok=True)
    spec = freeze()

    def worker(job):
        seed, sequence = job
        p = LOG / f"s{seed}_{sequence.replace('/','_')}.log"
        with p.open("x") as handle:
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "eval",
                    "--seed",
                    str(seed),
                    "--sequence",
                    sequence,
                ],
                cwd=run.REPO,
                env=dict(
                    os.environ,
                    PYTHONDONTWRITEBYTECODE="1",
                    OPENBLAS_NUM_THREADS="1",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                ),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(p.read_text()[-4000:])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, (s, q)) for s in run.SEEDS for q in spec["test"]]
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if number % 10 == 0 or number == len(futures):
                print("TEST_DONE", number, len(futures), flush=True)
    finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "eval", "finish"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.action == "run":
        run_all()
    elif args.action == "eval":
        evaluate(args.seed, args.sequence)
    else:
        finish()
