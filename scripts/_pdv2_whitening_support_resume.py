#!/usr/bin/env python3
"""검색에서 이미 구한 원시 점수를 재사용하는 실제 판정 경로의 비용을 잰다."""
import time
from pathlib import Path

import numpy as np

import _pdv2_whitening_support as prior
from _pdv2_no_context_ablation import sha256, write_json

source, run = prior.source, prior.run
OUT = prior.OUT / "completed"


def timing() -> dict:
    """원본 중심 점수는 검색 원장을 사용하고 보정 중심 점수만 추가 계산한다."""
    spec = source.protocol()
    matrix = run.BASE.exp.arrays(source.OUT / "statistics.npz")["full"]
    cases, pooled = {}, {
        k: []
        for k in ("original_ms", "whitening25_ms", "extra_ms", "transform_and_bank_ms")
    }
    for sequence in prior.SEQUENCES:
        paths = run.BASE.exp.old.base.inputs("val", sequence)[2]
        x = source.coefficients(paths["spec"])
        cache = run.BASE.exp.arrays(paths["cache"])
        xyz = cache["poses"][:, :3, 3]
        travel = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        original = source.variant_inputs(x, 8)
        white = source.variant_inputs(source.transform(x, matrix), 8)
        memory = [a.nbytes / len(x) for a in (white[0], white[1], white[2], *white[3])]
        for seed in (1, 2):
            name = sequence.replace("/", "_") + ".npz"
            row = run.BASE.exp.arrays(source.OUT / f"s{seed}" / name)
            channel = spec["confidence"][str(seed)]["decision"]
            coverage = run.BASE.exp.arrays(
                run.root(seed, "control")
                / f"epoch{spec['models'][str(seed)]}"
                / ("coverage_" + name)
            )
            ids = np.linspace(0, len(row["query_idx"]) - 1, 96, dtype=int)
            data = {k: [] for k in pooled}

            def score(bank, i, raw):
                q, j = row["query_idx"][i : i + 1], row["j_star"][i : i + 1]
                if raw is None:
                    raw = run.BASE.fixed.correlation_profiles(bank, q, j, channel).max(
                        1
                    )
                return run.BASE.fixed.frozen_scores(bank, travel, q, j, channel, raw)[
                    source.METHOD
                ]

            for i in ids[:16]:
                score(original, i, coverage[channel][i : i + 1])
                score(white, i, None)
            for repeat in range(3):
                for i in ids:
                    elapsed, values = {}, {}
                    order = (
                        ("original_ms", "whitening25_ms")
                        if repeat % 2 == 0
                        else ("whitening25_ms", "original_ms")
                    )
                    for condition in order:
                        tick = time.perf_counter_ns()
                        v = score(original, i, coverage[channel][i : i + 1])
                        if condition == "whitening25_ms":
                            v = 0.75 * v + 0.25 * score(white, i, None)
                        elapsed[condition] = (time.perf_counter_ns() - tick) / 1e6
                        values[condition] = v
                    np.testing.assert_array_equal(
                        values["original_ms"], row["identity"][i : i + 1]
                    )
                    np.testing.assert_array_equal(
                        values["whitening25_ms"],
                        (0.75 * row["identity"] + 0.25 * row["full"])[i : i + 1],
                    )
                    for condition, value in elapsed.items():
                        data[condition].append(value)
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
                "timing": {k: prior.statistics(v) for k, v in data.items()},
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
            for condition in pooled:
                pooled[condition].extend(data[condition])
            np.savez_compressed(OUT / (case.replace("/", "_") + "_timing.npz"), **data)
            print("TIMING", case, cases[case]["timing"], flush=True)
    return {
        "cases": cases,
        "pooled": {k: prior.statistics(v) for k, v in pooled.items()},
        "matrix_bytes": matrix.nbytes,
        "environment": {
            "cpu": "AMD Ryzen 7 9800X3D",
            "container": "ring_baseline",
            "threads": 1,
            "numpy": np.__version__,
        },
        "scope": "검증3경로·2채널·각96질의×3회,16워밍업. CPU한질의씩. 원시점군 인코딩/GNN/후보검색/파일IO 제외. 원시 점수는 이미 수행된 검색에서 재사용한다. 보정의 중심 점수 계산은 추가 판정 시간에 포함한다. 연구참조 frozen_scores의 불필요 대조 계산도 포함한다. ndarray payload만 측정하며 RSS/임시 작업공간 아님. 전체 실시간 배포/BEV 속도 비교 아님.",
    }


def main() -> None:
    """첫 시도 차이를 명시하고 엄격한 점수 일치를 유지한 채 재측정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "summary.json").exists()
    old_spec = run.read(prior.OUT / "protocol.json")
    for p, h in old_spec["sources"].items():
        assert sha256(run.REPO / p) == h
    spec = dict(old_spec)
    spec["sources"] = dict(old_spec["sources"])
    for p in (Path(__file__), prior.OUT / "protocol.json"):
        spec["sources"][str(p.relative_to(run.REPO))] = sha256(p)
    spec["amendment"] = (
        "첫 비용 측정은 원본 raw를 다른 상관 함수로 재계산하여 최종0.2221007470289866 대0.22210075149933498(4.47035e-9)로 정확 일치 검사 실패. 실제 평가처럼 검색 coverage의 raw를 재사용하도록 비용 측정만 수정. 점수 허용오차 완화 없음. 모델/평가 원장/선택/원 코드는 변경하지 않음. 보정 raw 계산은 추가 시간에 포함. 실패프로토콜 보존."
    )
    write_json(OUT / "protocol.json", spec)
    result = {
        "bootstrap": prior.bootstrap(),
        "validation_error_budget": prior.error_budget(),
        "timing": timing(),
    }
    write_json(OUT / "summary.json", result)
    write_json(
        OUT / "completion_validation.json",
        {
            "status": "passed",
            "timed_scores_match_frozen_ledgers_exact": True,
            "timed_conditions": 3456,
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
