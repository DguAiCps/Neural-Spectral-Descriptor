#!/usr/bin/env python3
"""고정한 상관 보정의 불확실성·오류 분해·추가 판정 비용을 확인한다."""
import platform
import time
from pathlib import Path

import numpy as np

import _pdv2_spectral_whitening as source
from _pdv2_final_error_budget import split_error
from _pdv2_no_context_ablation import sha256, write_json

run = source.prior
OUT = run.REPO / "results/pdv2_whitening_support_20261026"
SEQUENCES = ("kitti360/0004", "helipr/Roundabout01", "mulran/Sejong01")


def statistics(values: list) -> dict:
    """개별 측정값의 평균·중앙값·95백분위수를 반환한다."""
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "p50": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
    }


def bootstrap() -> dict:
    """도시와 세 시드를 고정하고 각 도시 안의 경로를 짝지어 재추출한다."""
    current = run.read(source.OUT / "fixed_test/table2_summary.json")
    old = run.read(run.PRIOR / "table2_summary.json")
    cities = run.read(source.OUT / "fixed_test/freeze.json")["cities"]
    rng = np.random.default_rng(20261026)
    differences = {}
    for city, sequences in cities.items():
        selected = [s for s in sequences if current["per_sequence"]["1"][s]["n_pos"]]
        values = np.array(
            [
                [
                    np.mean(
                        [
                            current["per_sequence"][str(seed)][seq][metric]
                            - old["per_sequence"]["octave288"][str(seed)][seq][metric]
                            for seed in run.SEEDS
                        ]
                    )
                    * 100
                    for metric in run.METRICS
                ]
                for seq in selected
            ]
        )
        differences[city] = {"sequences": selected, "delta_pp": values.tolist()}
    samples = np.zeros((20000, len(run.METRICS)))
    for row in differences.values():
        values = np.asarray(row["delta_pp"])
        indices = rng.integers(0, len(values), size=(len(samples), len(values)))
        samples += values[indices].mean(1) / len(differences)
    result = {}
    for i, metric in enumerate(run.METRICS):
        mean = float(
            np.mean(
                [np.asarray(r["delta_pp"])[:, i].mean() for r in differences.values()]
            )
        )
        np.testing.assert_allclose(
            mean, current["delta_pp"]["octave288"][metric], atol=1e-12, rtol=0
        )
        result[metric] = {
            "delta_pp": mean,
            "percentile95_pp": np.quantile(samples[:, i], [0.025, 0.975]).tolist(),
            "bootstrap_nonpositive_fraction": float(np.mean(samples[:, i] <= 0)),
        }
    return {
        "metrics": result,
        "city_differences": differences,
        "resamples": len(samples),
        "scope": "고정6도시·고정3시드 평균. 도시내45양성경로를 짝지어 복원추출. 경로 독립 가정·고정시드 조건부 불확실성이며 새 도시/시드/반복개발 선택불확실성을 포함하지 않는다. 시험결과로 설정선택 안함.",
    }


def error_budget() -> dict:
    """기존과25%보정의 검증 AP 손실 원인을 같은 정답 분모로 나눈다."""
    spec = source.protocol()
    results, means = {}, {}
    for arm in ("original", "whitening25"):
        results[arm] = {}
        for seed in run.SEEDS:
            rows = {}
            for sequence in spec["val"]:
                p = source.OUT / f"s{seed}" / (sequence.replace("/", "_") + ".npz")
                row = run.BASE.exp.arrays(p)
                score = (
                    row["identity"]
                    if arm == "original"
                    else 0.75 * row["identity"] + 0.25 * row["full"]
                )
                rows[sequence] = split_error(row, score)
            families = {}
            for family in ("kitti", "helipr", "mulran"):
                chosen = [
                    r for s, r in rows.items() if s.startswith(family) and r["positive"]
                ]
                families[family] = {
                    k: float(np.mean([r["components"][k] for r in chosen]))
                    for k in chosen[0]["components"]
                }
            macro = {
                k: float(np.mean([r[k] for r in families.values()]))
                for k in next(iter(families.values()))
            }
            results[arm][str(seed)] = {"families": families, "macro": macro}
        means[arm] = {
            k: float(np.mean([r["macro"][k] for r in results[arm].values()]))
            for k in results[arm]["1"]["macro"]
        }
    return {
        "per_seed": results,
        "mean": means,
        "scope": "검증GT를 쓰는 오류 분해 진단 상한. 온라인 실현 성능이나 인과 기여율이 아니다.",
    }


def timing() -> dict:
    """현재 참조 코드의 판정 단계만 한 질의씩 추가 비용을 잰다."""
    spec = source.protocol()
    matrix = run.BASE.exp.arrays(source.OUT / "statistics.npz")["full"]
    cases, pool = {}, {"original_ms": [], "whitening25_ms": [], "extra_ms": []}
    for sequence in SEQUENCES:
        paths = run.BASE.exp.old.base.inputs("val", sequence)[2]
        x = source.coefficients(paths["spec"])
        cache = run.BASE.exp.arrays(paths["cache"])
        xyz = cache["poses"][:, :3, 3]
        travel = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        original = source.variant_inputs(x, 8)
        white = source.variant_inputs(source.transform(x, matrix), 8)
        memory = [a.nbytes / len(x) for a in (white[0], white[1], white[2], *white[3])]
        for seed in (1, 2):
            row = run.BASE.exp.arrays(
                source.OUT / f"s{seed}" / (sequence.replace("/", "_") + ".npz")
            )
            channel = spec["confidence"][str(seed)]["decision"]
            ids = np.linspace(0, len(row["query_idx"]) - 1, 96, dtype=int)
            data = {
                "original_ms": [],
                "whitening25_ms": [],
                "extra_ms": [],
                "transform_and_bank_ms": [],
            }

            def score(bank, i):
                q, j = row["query_idx"][i : i + 1], row["j_star"][i : i + 1]
                raw = run.BASE.fixed.correlation_profiles(bank, q, j, channel).max(1)
                return run.BASE.fixed.frozen_scores(bank, travel, q, j, channel, raw)[
                    source.METHOD
                ]

            for i in ids[:16]:
                score(original, i)
                score(white, i)
            for repeat in range(3):
                for i in ids:
                    elapsed, values = {}, {}
                    # 순서에 따른 캐시 편향을 줄이도록 측정 순서를 교대한다.
                    order = (
                        ("original_ms", "whitening25_ms")
                        if repeat % 2 == 0
                        else ("whitening25_ms", "original_ms")
                    )
                    for name in order:
                        tick = time.perf_counter_ns()
                        v = score(original, i)
                        if name == "whitening25_ms":
                            v = 0.75 * v + 0.25 * score(white, i)
                        elapsed[name] = (time.perf_counter_ns() - tick) / 1e6
                        values[name] = v
                    np.testing.assert_array_equal(
                        values["original_ms"], row["identity"][i : i + 1]
                    )
                    np.testing.assert_array_equal(
                        values["whitening25_ms"],
                        (0.75 * row["identity"] + 0.25 * row["full"])[i : i + 1],
                    )
                    for name in elapsed:
                        data[name].append(elapsed[name])
                    data["extra_ms"].append(
                        elapsed["whitening25_ms"] - elapsed["original_ms"]
                    )
                    q = int(row["query_idx"][i])
                    tick = time.perf_counter_ns()
                    source.variant_inputs(source.transform(x[q : q + 1], matrix), 8)
                    data["transform_and_bank_ms"].append(
                        (time.perf_counter_ns() - tick) / 1e6
                    )
            case = sequence + "/" + channel
            cases[case] = {
                "timing": {k: statistics(v) for k, v in data.items()},
                "new_bank_bytes_per_frame": sum(memory),
                "new_bank_breakdown_bytes": dict(
                    zip(
                        (
                            "auxiliary_key",
                            "spectral_fft",
                            "spectral_norm",
                            "column_fft",
                            "column_mask_fft",
                        ),
                        memory,
                    )
                ),
                "active_channel_arrays_bytes_per_frame": (
                    sum(memory[3:]) if channel == "c_col" else sum(memory[1:3])
                ),
                "frames": len(x),
            }
            for name in pool:
                pool[name].extend(data[name])
            np.savez_compressed(OUT / (case.replace("/", "_") + "_timing.npz"), **data)
            print("TIMING", case, cases[case]["timing"], flush=True)
    return {
        "cases": cases,
        "pooled": {k: statistics(v) for k, v in pool.items()},
        "matrix_bytes": matrix.nbytes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu": "AMD Ryzen 7 9800X3D",
            "container": "ring_baseline",
            "threads": 1,
        },
        "scope": "검증3경로·2채널·각96질의×3회,16워밍업. CPU한질의씩. 메모리는 실제 ndarray payload이며 RSS/임시작업공간 아님. 원시점군 인코딩/GNN/후보검색/파일IO 제외. 연구참조 frozen_scores가 만들지만 사용하지 않는 대조연산도 포함한다. BEVPlace++와 전체속도 비교나 실시간배포 검증이 아니다.",
    }


def main() -> None:
    """고정 결과의 근거와 비용을 보존하고 검산한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "summary.json").exists()
    paths = {
        Path(__file__),
        source.OUT / "fixed_test/table2_summary.json",
        source.OUT / "fixed_test/completion_validation.json",
        run.PRIOR / "table2_summary.json",
        source.OUT / "statistics.npz",
        source.OUT / "blend/summary.json",
    }
    spec = source.protocol()
    paths.update(run.REPO / p for p in spec["sources"])
    paths.update(
        source.OUT / f"s{s}" / (seq.replace("/", "_") + ".npz")
        for s in run.SEEDS
        for seq in spec["val"]
    )
    write_json(
        OUT / "protocol.json",
        {
            "sources": {str(p.relative_to(run.REPO)): sha256(p) for p in sorted(paths)},
            "test_for_selection": False,
            "timing_roster": SEQUENCES,
            "bootstrap_seed": 20261026,
            "resamples": 20000,
        },
    )
    result = {
        "bootstrap": bootstrap(),
        "validation_error_budget": error_budget(),
        "timing": timing(),
    }
    write_json(OUT / "summary.json", result)
    write_json(
        OUT / "completion_validation.json",
        {
            "status": "passed",
            "timed_scores_match_frozen_ledgers": True,
            "bootstrap_mean_matches_table2": True,
            "summary_sha256": sha256(OUT / "summary.json"),
            "protocol_sha256": sha256(OUT / "protocol.json"),
        },
    )
    print(
        "SUPPORT_COMPLETE",
        result["bootstrap"]["metrics"],
        result["validation_error_budget"]["mean"],
        result["timing"]["pooled"],
        flush=True,
    )


if __name__ == "__main__":
    main()
