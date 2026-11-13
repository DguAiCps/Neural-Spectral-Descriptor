#!/usr/bin/env python3
"""고정 승자의 최종 스펙트럼 판정에 학습 통계 백색화를 대조한다."""
import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import _pdv2_key_metric_alignment as prior
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_matching_mine import spec_path
from _pdv2_no_context_ablation import sha256, write_json
from _xs_eval_lib import variant_inputs

REPO = prior.REPO
OUT = REPO / "results/pdv2_spectral_whitening_20261026"
LOG = REPO / "logs/spectral_whitening_20261026"
ARMS = ("identity", "diagonal", "full")
METHOD = prior.METHOD


def coefficients(path: Path) -> np.ndarray:
    """NSD가 이미 보존하는 양의 주파수 1~8만 읽는다."""
    with np.load(path) as data:
        return data["sk_re"][:, :, :8].astype(np.float32) + 1j * data["sk_im"][
            :, :, :8
        ].astype(np.float32)


def whitening_matrix(moment: np.ndarray, full: bool) -> np.ndarray:
    """10% 등방 수축을 고정한 주파수별 Hermitian 역제곱근을 만든다."""
    result = []
    for original in moment:
        cov = original if full else np.diag(np.real(np.diag(original)))
        mean = float(np.trace(cov).real / len(cov))
        regularized = 0.9 * cov + 0.1 * max(mean, 1e-12) * np.eye(len(cov))
        value, vectors = np.linalg.eigh(regularized)
        assert np.min(value) > 0
        matrix = (vectors * (1 / np.sqrt(value))[None]) @ vectors.conj().T
        np.testing.assert_allclose(
            matrix @ regularized @ matrix.conj().T, np.eye(len(cov)), atol=1e-10, rtol=0
        )
        result.append(matrix)
    return np.asarray(result)


def transform(spectrum: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """각 주파수 안에서 반경 행을 섞어 yaw 위상 변환을 보존한다."""
    return np.einsum("kij,njk->nik", matrix, spectrum, optimize=True).astype(
        np.complex64
    )


def protocol() -> dict:
    """고정한 입력·실행·학습 통계 해시를 검사한다."""
    spec = prior.read(OUT / "protocol.json")
    for path, digest in spec["sources"].items():
        assert sha256(REPO / path) == digest, path
    return spec


def prepare() -> None:
    """검증 결과 계산 전에 학습 통계와 두 대조를 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol()
        return
    old = prior.protocol(True)
    selected = prior.read(prior.PRIOR / "validation_summary.json")["per_seed"]
    paths = {
        Path(__file__),
        REPO / "tests/retrieval/test_spectral_whitening.py",
        prior.OUT / "protocol.json",
        prior.PRIOR / "validation_summary.json",
        REPO / "scripts/_xs_eval_lib.py",
        REPO / "scripts/_pdv2_sequence_fixed_test.py",
        REPO / "src/retrieval/spectral_evidence.py",
        REPO / "src/retrieval/sequence_evidence.py",
    }
    moments, counts = [], {}
    for sequence in old["train"]:
        path = spec_path(sequence)
        paths.add(path)
        x = coefficients(path).astype(np.complex128)
        x /= np.maximum(np.sqrt(np.sum(abs(x) ** 2, axis=(1, 2))), 1e-12)[:, None, None]
        moments.append(np.einsum("nik,njk->kij", x, x.conj(), optimize=True) / len(x))
        counts[sequence] = len(x)
    # 학습 센서 2계열의 각 4시퀀스에 같은 가중치를 준다.
    moment = np.mean(moments, axis=0)
    np.testing.assert_allclose(
        moment, moment.conj().transpose(0, 2, 1), atol=1e-14, rtol=0
    )
    matrices = {
        "identity": np.repeat(np.eye(16)[None], 8, axis=0).astype(np.complex128),
        "diagonal": whitening_matrix(moment, False),
        "full": whitening_matrix(moment, True),
    }
    np.savez_compressed(OUT / "statistics.npz", moment=moment, **matrices)
    paths.add(OUT / "statistics.npz")
    models = {}
    for seed in prior.SEEDS:
        epoch = selected[str(seed)]["octave288"]["selected"]["epoch"]
        models[str(seed)] = epoch
        for sequence in old["val"]:
            p = (
                prior.root(seed, "control")
                / f"epoch{epoch}"
                / (sequence.replace("/", "_") + ".npz")
            )
            paths.update((p, p.with_suffix(".json"), p.with_name("coverage_" + p.name)))
            paths.update(prior.BASE.exp.old.base.inputs("val", sequence)[2].values())
    write_json(
        OUT / "protocol.json",
        {
            "created_unix": time.time(),
            "train": old["train"],
            "val": old["val"],
            "seeds": prior.SEEDS,
            "models": models,
            "arms": ARMS,
            "confidence": old["confidence"],
            "training_frames": counts,
            "hypothesis": "비재방문 오수락이 최종 AP 손실의 주된 부분이므로 보존 스펙트럼의 공통 성분이 과대평가되는지 검사한다.",
            "fit": "학습GT8·72306프레임만. 각프레임복소16x8전체L2정규화→주파수별중심화없는2차모멘트→시퀀스균등평균(센서2계열균등). 대각/전체행상관에고정10%등방수축후역제곱근. GT라벨불사용·새신경망없음·보존계수256실수변경없음. 검증에서수축률을고르지않음.",
            "evaluation": "기존octave288선택모델3시드의후보/승자/지연고정. 최종스펙트럼중심과이웃판정만변환된계수로재계산. 시드별c_col/c_spec과arc_unique_b100동일. 같은승자이므로R1불변. identity는원최종점수를재현해야함. 검증9·99236질의·시험미사용.",
            "selection": "두조건모두보고. 평균AP+.1%p이상·모든시드AP개선·평균R100손실.1%p이하·계열AP손실.25%p이하만후속후보. 이번실행에서주표/시험평가없음.",
            "sources": {str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)},
            "test_used": False,
        },
    )
    print("FROZEN_WHITENING", counts, flush=True)


def evaluate(seed: int, sequence: str) -> None:
    """승자와 분모를 고정한 채 세 점수의 최종 원장을 만든다."""
    spec = protocol()
    name = sequence.replace("/", "_") + ".npz"
    folder = OUT / f"s{seed}"
    folder.mkdir(exist_ok=True)
    target = folder / name
    assert not target.exists(), target
    original = prior.root(seed, "control") / f"epoch{spec['models'][str(seed)]}" / name
    row = prior.BASE.exp.arrays(original)
    coverage = prior.BASE.exp.arrays(original.with_name("coverage_" + name))
    _, _, paths, _ = prior.BASE.exp.old.base.inputs("val", sequence)
    cache = prior.BASE.exp.arrays(paths["cache"])
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
    x = coefficients(paths["spec"])
    q, j = row["query_idx"], row["j_star"]
    assert (j <= q - 30).all() and (q + 15 < len(x)).all()
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    channel = spec["confidence"][str(seed)]["decision"]
    matrices = prior.BASE.exp.arrays(OUT / "statistics.npz")
    scores, checks = {}, {}
    for arm in ARMS:
        modified = transform(x, matrices[arm])
        if arm == "identity":
            np.testing.assert_array_equal(modified, x)
        bank = variant_inputs(modified, 8)
        raw = (
            coverage[channel]
            if arm == "identity"
            else prior.BASE.fixed.correlation_profiles(bank, q, j, channel).max(1)
        )
        pieces = []
        for start in range(0, len(q), 512):
            sl = slice(start, start + 512)
            pieces.append(
                prior.BASE.fixed.frozen_scores(
                    bank, travel, q[sl], j[sl], channel, raw[sl]
                )[METHOD]
            )
        value = np.concatenate(pieces)
        assert np.isfinite(value).all()
        if arm == "identity":
            np.testing.assert_array_equal(value, row[METHOD])
        prefix = []
        for index in (len(q) // 3, 2 * len(q) // 3):
            end = int(q[index]) + 16
            small = prior.BASE.exp.truncate_bank(bank, end)
            check = prior.BASE.fixed.frozen_scores(
                small,
                travel[:end],
                q[index : index + 1],
                j[index : index + 1],
                channel,
                raw[index : index + 1],
            )[METHOD]
            np.testing.assert_array_equal(check, value[index : index + 1])
            prefix.append({"query": int(q[index]), "error": 0.0})
        metric = independent_metrics(row, value)
        other = prior.BASE.exp.metrics(value, row)
        for field in prior.METRICS:
            if metric[field] is not None:
                np.testing.assert_allclose(
                    metric[field], other[field], atol=1e-12, rtol=0
                )
        scores[arm], checks[arm] = value, {"metrics": metric, "prefixes": prefix}
    np.savez_compressed(target, **{k: row[k] for k in prior.BASE.FIELDS}, **scores)
    write_json(
        target.with_suffix(".json"),
        {
            "sha256": sha256(target),
            "checks": checks,
            "original_sha256": sha256(original),
            "unchanged_winner": True,
            "test_used": False,
        },
    )
    print("EVALUATED", seed, sequence, flush=True)


def finish() -> None:
    """전수 검산 후 세 조건의 평균·시드·계열별 손익을 보고한다."""
    spec = protocol()
    per_seed, stats = {}, {}
    for arm in ARMS:
        per_seed[arm] = {}
        for seed in prior.SEEDS:
            rows = {}
            for sequence in spec["val"]:
                path = OUT / f"s{seed}" / (sequence.replace("/", "_") + ".npz")
                info = prior.read(path.with_suffix(".json"))
                assert sha256(path) == info["sha256"]
                row = prior.BASE.exp.arrays(path)
                rows[sequence] = independent_metrics(row, row[arm])
                assert rows[sequence] == info["checks"][arm]["metrics"]
            per_seed[arm][str(seed)] = {**prior.BASE.macro(rows), "sequences": rows}
        stats[arm] = {
            m: {
                "mean": float(np.mean([r["macro"][m] for r in per_seed[arm].values()])),
                "std": float(
                    np.std([r["macro"][m] for r in per_seed[arm].values()], ddof=1)
                ),
            }
            for m in prior.METRICS
        }
    decisions = {}
    for arm in ARMS[1:]:
        delta = {
            m: 100 * (stats[arm][m]["mean"] - stats["identity"][m]["mean"])
            for m in prior.METRICS
        }
        family = {
            f: 100
            * float(
                np.mean(
                    [
                        per_seed[arm][str(s)]["families"][f]["AP"]
                        - per_seed["identity"][str(s)]["families"][f]["AP"]
                        for s in prior.SEEDS
                    ]
                )
            )
            for f in ("kitti", "helipr", "mulran")
        }
        checks = {
            "mean_AP": delta["AP"] >= 0.1,
            "all_seeds_AP": all(
                per_seed[arm][str(s)]["macro"]["AP"]
                > per_seed["identity"][str(s)]["macro"]["AP"]
                for s in prior.SEEDS
            ),
            "R100": delta["R@100P"] >= -0.1,
            "family_AP": min(family.values()) >= -0.25,
        }
        decisions[arm] = {
            "delta_pp": delta,
            "family_AP_delta_pp": family,
            "checks": checks,
            "promoted": all(checks.values()),
        }
    write_json(
        OUT / "summary.json",
        {
            "stats": stats,
            "per_seed": per_seed,
            "decisions": decisions,
            "test_used": False,
        },
    )
    write_json(
        OUT / "completion_validation.json",
        {
            "status": "passed",
            "ledgers": 27,
            "score_conditions": 81,
            "query_rows": 297708,
            "prefix_checks": 162,
            "identity_exact": True,
            "independent_metrics": 81,
            "summary_sha256": sha256(OUT / "summary.json"),
            "protocol_sha256": sha256(OUT / "protocol.json"),
            "test_used": False,
        },
    )
    print("FINAL_WHITENING", stats, decisions, flush=True)


def run_all() -> None:
    """학습 통계 적합 뒤 독립 시퀀스 판정을 두 개씩 계산한다."""
    LOG.mkdir(parents=True, exist_ok=True)
    prepare()
    spec = protocol()

    def worker(job):
        seed, sequence = job
        path = LOG / f"s{seed}_{sequence.replace('/', '_')}.log"
        with path.open("x") as handle:
            process = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "val",
                    "--seed",
                    str(seed),
                    "--sequence",
                    sequence,
                ],
                cwd=REPO,
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
        if process.returncode:
            raise RuntimeError(path.read_text()[-4000:])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [(s, q) for s in prior.SEEDS for q in spec["val"]]
        futures = [pool.submit(worker, j) for j in jobs]
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print("DONE", number, len(jobs), flush=True)
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
