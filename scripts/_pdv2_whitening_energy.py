#!/usr/bin/env python3
"""상관 보정에서 주파수 에너지 변화와 반경 방향 변화를 분리한다."""
import argparse
import concurrent.futures
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
OUT = run.REPO / "results/pdv2_whitening_energy_20261026"
LOG = run.REPO / "logs/whitening_energy_20261026"
MODES = ("population", "scan")
ARMS = ("baseline", "population25", "scan25", "population100", "scan100")


def population_matrix(moment: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """학습 분포의 주파수별 기대 에너지를 보존하도록 보정 행렬을 맞춘다."""
    before = np.trace(moment, axis1=1, axis2=2).real
    after = np.trace(
        matrix @ moment @ matrix.conj().transpose(0, 2, 1), axis1=1, axis2=2
    ).real
    return matrix * np.sqrt(before / np.maximum(after, 1e-30))[:, None, None]


def transform(x: np.ndarray, matrix: np.ndarray, mode: str) -> np.ndarray:
    """반경 보정 후 필요하면 각 스캔의 주파수별 원래 에너지를 복원한다."""
    assert mode in MODES
    y = source.transform(x, matrix)
    if mode == "scan":
        before = np.sqrt(np.sum(abs(x.astype(np.complex128)) ** 2, axis=1))
        after = np.sqrt(np.sum(abs(y.astype(np.complex128)) ** 2, axis=1))
        y = (y * (before / np.maximum(after, 1e-30))[:, None]).astype(np.complex64)
    return y


def protocol(check: bool = True) -> dict:
    """사전 고정 명세와 해시를 읽는다."""
    spec = run.read(OUT / "protocol.json")
    if check:
        for path, digest in spec["sources"].items():
            assert sha256(run.REPO / path) == digest, path
    return spec


def prepare() -> None:
    """두 구조 대조와 승격 규칙을 검증 점수 계산 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol()
        return
    old = source.protocol()
    stats = run.BASE.exp.arrays(source.OUT / "statistics.npz")
    matrix = population_matrix(stats["moment"], stats["full"])
    np.savez_compressed(OUT / "statistics.npz", population=matrix, scan=stats["full"])
    energy = np.trace(stats["moment"], axis1=1, axis2=2).real
    corrected = np.trace(
        stats["full"] @ stats["moment"] @ stats["full"].conj().transpose(0, 2, 1),
        axis1=1,
        axis2=2,
    ).real
    paths = {
        Path(__file__),
        run.REPO / "tests/retrieval/test_whitening_energy.py",
        OUT / "statistics.npz",
        source.OUT / "statistics.npz",
        source.OUT / "protocol.json",
        source.OUT / "blend/summary.json",
    }
    paths.update(run.REPO / p for p in old["sources"])
    paths.update(
        source.OUT / f"s{s}" / (q.replace("/", "_") + ".npz")
        for s in run.SEEDS
        for q in old["val"]
    )
    write_json(
        OUT / "protocol.json",
        {
            "created_unix": time.time(),
            "val": old["val"],
            "seeds": run.SEEDS,
            "models": old["models"],
            "confidence": old["confidence"],
            "arms": ARMS,
            "hypothesis": "전체 보정은 고주파5~8의 학습 에너지 비율을37.8%에서50.7%로 높인다. HeLiPR 검증 손실이 주파수 가중치 변화 때문인지 반경 상관 제거와 분리한다.",
            "methods": {
                "population": "W_k에sqrt(tr(C_k)/tr(W_k C_k W_k^H))를 곱해 학습 평균 주파수 에너지를 보존한다. 검증 통계로 적합하지 않는다.",
                "scan": "기존W_k 적용 뒤 각 스캔의 각 주파수 반경L2를 원본과 같게 맞춘다. 현재 스캔만 필요하며 미래 통계나GT를 사용하지 않는다.",
            },
            "fit": "기존 train8·72306frames의 고정10%수축 통계 재사용. 추가학습/모듈/주파수/라벨 없음.",
            "selection": "현재 whitening25 기준. 새25%조건 중 평균AP>=+.1%p, 모든시드AP>0, 평균R100>=-.1%p, 각계열AP>=-.25%p를 모두 만족하면 최고평균AP 선택. 동률population 우선. 100%조건은 기전 대조만이며 승격 불가. 실패하면 baseline 유지. 시험 미사용.",
            "train_energy_share": (energy / energy.sum()).tolist(),
            "whitened_train_energy_share": (corrected / corrected.sum()).tolist(),
            "sources": {str(p.relative_to(run.REPO)): sha256(p) for p in sorted(paths)},
            "test_used": False,
            "independent_confirmation": False,
        },
    )


def evaluate(seed: int, sequence: str) -> None:
    """고정 승자의 두 새 보정 점수와 접두 동일성을 평가한다."""
    spec = protocol(False)
    assert (
        sha256(Path(__file__))
        == spec["sources"][str(Path(__file__).relative_to(run.REPO))]
    )
    name = sequence.replace("/", "_") + ".npz"
    folder = OUT / f"s{seed}"
    folder.mkdir(exist_ok=True)
    target = folder / name
    assert not target.exists()
    row = run.BASE.exp.arrays(source.OUT / f"s{seed}" / name)
    paths = run.BASE.exp.old.base.inputs("val", sequence)[2]
    cache = run.BASE.exp.arrays(paths["cache"])
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
    x = source.coefficients(paths["spec"])
    q, j = row["query_idx"], row["j_star"]
    assert (j <= q - 30).all() and (q + 15 < len(x)).all()
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    channel = spec["confidence"][str(seed)]["decision"]
    matrices = run.BASE.exp.arrays(OUT / "statistics.npz")
    scores = {"baseline": 0.75 * row["identity"] + 0.25 * row["full"]}
    reference = run.BASE.exp.arrays(source.OUT / "blend/quarter" / f"s{seed}" / name)
    np.testing.assert_array_equal(scores["baseline"], reference["score"])
    checks, diagnostic = {}, {}
    for mode in MODES:
        y = transform(x, matrices[mode], mode)
        before = (abs(x.astype(np.complex128)) ** 2).sum(1)
        after = (abs(y.astype(np.complex128)) ** 2).sum(1)
        if mode == "scan":
            np.testing.assert_allclose(after, before, rtol=4e-7, atol=1e-12)
        diagnostic[mode] = {
            "original_high_frequency_share": float(
                np.mean(before[:, 4:].sum(1) / np.maximum(before.sum(1), 1e-30))
            ),
            "modified_high_frequency_share": float(
                np.mean(after[:, 4:].sum(1) / np.maximum(after.sum(1), 1e-30))
            ),
        }
        bank = source.variant_inputs(y, 8)
        raw = run.BASE.fixed.correlation_profiles(bank, q, j, channel).max(1)
        pieces = []
        for start in range(0, len(q), 512):
            sl = slice(start, start + 512)
            pieces.append(
                run.BASE.fixed.frozen_scores(
                    bank, travel, q[sl], j[sl], channel, raw[sl]
                )[source.METHOD]
            )
        value = np.concatenate(pieces)
        assert np.isfinite(value).all()
        prefixes = []
        for i in (len(q) // 3, 2 * len(q) // 3):
            end = int(q[i]) + 16
            # 스캔별 정규화도 접두 입력부터 직접 다시 계산한다.
            small = source.variant_inputs(transform(x[:end], matrices[mode], mode), 8)
            raw_small = run.BASE.fixed.correlation_profiles(
                small, q[i : i + 1], j[i : i + 1], channel
            ).max(1)
            v = run.BASE.fixed.frozen_scores(
                small, travel[:end], q[i : i + 1], j[i : i + 1], channel, raw_small
            )[source.METHOD]
            np.testing.assert_array_equal(v, value[i : i + 1])
            prefixes.append({"query": int(q[i]), "error": 0.0})
        scores[mode + "25"] = 0.75 * row["identity"] + 0.25 * value
        scores[mode + "100"] = value
        checks[mode] = prefixes
    metrics = {}
    for arm, value in scores.items():
        metrics[arm] = independent_metrics(row, value)
        other = run.BASE.exp.metrics(value, row)
        for key in run.METRICS:
            if metrics[arm][key] is not None:
                np.testing.assert_allclose(
                    metrics[arm][key], other[key], atol=1e-12, rtol=0
                )
    np.savez_compressed(target, **{k: row[k] for k in run.BASE.FIELDS}, **scores)
    write_json(
        target.with_suffix(".json"),
        {
            "sha256": sha256(target),
            "metrics": metrics,
            "prefixes": checks,
            "energy": diagnostic,
            "winner_unchanged": True,
        },
    )


def finish() -> None:
    """세 시드·계열별 최종 성능으로 사전 승격 규칙을 적용한다."""
    spec = protocol()
    per_seed, stats, ledgers = {}, {}, []
    for arm in ARMS:
        per_seed[arm] = {}
        for seed in run.SEEDS:
            rows = {}
            for seq in spec["val"]:
                p = OUT / f"s{seed}" / (seq.replace("/", "_") + ".npz")
                info, row = run.read(p.with_suffix(".json")), run.BASE.exp.arrays(p)
                assert sha256(p) == info["sha256"]
                rows[seq] = independent_metrics(row, row[arm])
                assert rows[seq] == info["metrics"][arm]
                old = run.BASE.exp.arrays(source.OUT / f"s{seed}" / p.name)
                for key in run.BASE.FIELDS:
                    np.testing.assert_array_equal(row[key], old[key])
                if arm == "baseline":
                    ledgers.append(
                        {
                            "path": str(p.relative_to(run.REPO)),
                            "sha256": info["sha256"],
                            "rows": len(row["query_idx"]),
                        }
                    )
            per_seed[arm][str(seed)] = {**run.BASE.macro(rows), "sequences": rows}
        stats[arm] = {
            m: {
                "mean": float(np.mean([r["macro"][m] for r in per_seed[arm].values()])),
                "std": float(
                    np.std([r["macro"][m] for r in per_seed[arm].values()], ddof=1)
                ),
            }
            for m in run.METRICS
        }
    previous = run.read(source.OUT / "blend/summary.json")
    assert stats["baseline"] == previous["stats"]["quarter"]
    decisions = {}
    for arm in ARMS[1:]:
        delta = {
            m: 100 * (stats[arm][m]["mean"] - stats["baseline"][m]["mean"])
            for m in run.METRICS
        }
        seeds = {
            str(s): 100
            * (
                per_seed[arm][str(s)]["macro"]["AP"]
                - per_seed["baseline"][str(s)]["macro"]["AP"]
            )
            for s in run.SEEDS
        }
        family = {
            f: 100
            * float(
                np.mean(
                    [
                        per_seed[arm][str(s)]["families"][f]["AP"]
                        - per_seed["baseline"][str(s)]["families"][f]["AP"]
                        for s in run.SEEDS
                    ]
                )
            )
            for f in ("kitti", "helipr", "mulran")
        }
        checks = {
            "mean_AP": delta["AP"] >= 0.1,
            "all_seeds_AP": min(seeds.values()) > 0,
            "R100": delta["R@100P"] >= -0.1,
            "family_AP": min(family.values()) >= -0.25,
            "eligible_arm": arm.endswith("25"),
        }
        decisions[arm] = {
            "delta_pp": delta,
            "seed_AP_delta_pp": seeds,
            "family_AP_delta_pp": family,
            "checks": checks,
            "promoted": all(checks.values()),
        }
    eligible = [a for a, d in decisions.items() if d["promoted"]]
    chosen = (
        max(eligible, key=lambda a: stats[a]["AP"]["mean"]) if eligible else "baseline"
    )
    write_json(
        OUT / "summary.json",
        {
            "stats": stats,
            "per_seed": per_seed,
            "decisions": decisions,
            "selected": chosen,
            "test_used": False,
        },
    )
    write_json(
        OUT / "completion_validation.json",
        {
            "status": "passed",
            "ledgers": ledgers,
            "query_rows": sum(r["rows"] for r in ledgers),
            "independent_metric_checks": 135,
            "prefix_checks": 108,
            "winner_and_baseline_exact": True,
            "summary_sha256": sha256(OUT / "summary.json"),
            "protocol_sha256": sha256(OUT / "protocol.json"),
        },
    )
    print("RESULT", stats, decisions, "SELECTED", chosen, flush=True)


def run_all() -> None:
    """독립 시퀀스 점수 계산을 두 프로세스씩 실행한다."""
    prepare()
    LOG.mkdir(parents=True, exist_ok=True)
    spec = protocol()

    def worker(job):
        seed, seq = job
        p = LOG / f"s{seed}_{seq.replace('/', '_')}.log"
        with p.open("x") as handle:
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "val",
                    "--seed",
                    str(seed),
                    "--sequence",
                    seq,
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
        futures = [pool.submit(worker, (s, q)) for s in run.SEEDS for q in spec["val"]]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print("DONE", count, len(futures), flush=True)
    finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "val", "finish"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.action == "run":
        run_all()
    elif args.action == "val":
        evaluate(args.seed, args.sequence)
    else:
        finish()
