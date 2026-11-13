#!/usr/bin/env python3
"""학습 재방문 차이의 공분산으로 불안정한 스펙트럼 방향을 낮춘다."""
import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import _pdv2_whitening_energy as evaluator
from _pdv2_matching_mine import cache_path, spec_path
from _pdv2_no_context_ablation import sha256, write_json

source = evaluator.source
run = source.prior
OUT = run.REPO / "results/pdv2_whitening_revisit_20261026"
LOG = run.REPO / "logs/whitening_revisit_20261026"


def align_pairs(a: np.ndarray, b: np.ndarray) -> tuple:
    """60방향 상관 최대각으로 학습 정답 쌍을 회전 정렬한다."""
    assert a.shape == b.shape and a.shape[1:] == (16, 8)
    product = np.zeros((len(a), 31), dtype=np.complex128)
    product[:, 1:9] = (b * a.conj()).sum(1)
    profile = np.fft.irfft(product, n=60, axis=1)
    shift = profile.argmax(1)
    phase = np.exp(2j * np.pi * shift[:, None] * np.arange(1, 9)[None] / 60)
    aligned = b * phase[:, None]
    delta = a - aligned
    cost = np.sum(abs(delta) ** 2, axis=(1, 2))
    before = np.sum(abs(a - b) ** 2, axis=(1, 2))
    assert np.all(cost <= before + 1e-12)
    return delta, shift, before, cost


def transformed(x: np.ndarray, matrix: np.ndarray, mode: str) -> np.ndarray:
    """검증에서 적합 없이 고정 재방문 보정 행렬만 적용한다."""
    assert mode in ("noise", "shrink")
    return source.transform(x, matrix)


def configure() -> None:
    """완료된 에너지 대조의 공통 평가기를 새 출력에 연결한다."""
    evaluator.OUT = OUT
    evaluator.MODES = ("noise", "shrink")
    evaluator.ARMS = ("baseline", "noise25", "shrink25", "noise100", "shrink100")
    evaluator.transform = transformed


def prepare() -> None:
    """학습 정답 쌍만으로 행렬을 만들고 평가 전에 명세를 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        evaluator.protocol()
        return
    old = source.protocol()
    paths = {
        Path(__file__),
        Path(evaluator.__file__),
        run.REPO / "tests/retrieval/test_whitening_revisit.py",
        source.OUT / "statistics.npz",
        source.OUT / "blend/summary.json",
    }
    paths.update(run.REPO / p for p in old["sources"])
    sequence_stats, moments = {}, {}
    for sequence in old["train"]:
        path = cache_path(sequence)
        paths.update((path, spec_path(sequence)))
        cache = run.BASE.exp.arrays(path)
        with np.load(spec_path(sequence)) as z:
            np.testing.assert_array_equal(cache["scan_ids"], z["scan_ids"])
        x = source.coefficients(spec_path(sequence)).astype(np.complex128)
        x /= np.maximum(np.sqrt(np.sum(abs(x) ** 2, axis=(1, 2))), 1e-30)[:, None, None]
        xyz = cache["poses"][:, :3, 3]
        neighbors = cKDTree(xyz).query_ball_point(xyz, r=5.0)
        qs, js, ws = [], [], []
        anchors = 0
        for q, candidates in enumerate(neighbors):
            valid = np.asarray(
                sorted(j for j in candidates if j <= q - 30), dtype=np.int64
            )
            if not len(valid):
                continue
            anchors += 1
            qs.extend([q] * len(valid))
            js.extend(valid.tolist())
            ws.extend([1.0 / len(valid)] * len(valid))
        if not anchors:
            sequence_stats[sequence] = {
                "anchors": 0,
                "pairs": 0,
                "excluded": "적격 재방문 없음",
            }
            continue
        q, j, weight = np.asarray(qs), np.asarray(js), np.asarray(ws) / anchors
        assert (j <= q - 30).all()
        assert np.max(np.linalg.norm(xyz[q] - xyz[j], axis=1)) <= 5.0 + 1e-10
        np.testing.assert_allclose(weight.sum(), 1.0, atol=1e-12)
        moment = np.zeros((8, 16, 16), np.complex128)
        before_mean = after_mean = 0.0
        shift_count = np.zeros(60, np.int64)
        for start in range(0, len(q), 2048):
            sl = slice(start, start + 2048)
            delta, shift, before, after = align_pairs(x[q[sl]], x[j[sl]])
            moment += 0.5 * np.einsum(
                "nik,njk,n->kij", delta, delta.conj(), weight[sl], optimize=True
            )
            before_mean += float(weight[sl] @ before)
            after_mean += float(weight[sl] @ after)
            shift_count += np.bincount(shift, minlength=60)
        moments[sequence] = moment
        sequence_stats[sequence] = {
            "anchors": anchors,
            "pairs": len(q),
            "mean_squared_difference_before": before_mean,
            "mean_squared_difference_after": after_mean,
            "shift_histogram": shift_count.tolist(),
        }
        np.savez_compressed(
            OUT / (sequence.replace("/", "_") + "_pairs.npz"),
            query=q,
            candidate=j,
            weight=weight,
        )
        print("FIT", sequence, anchors, len(q), flush=True)
    family_moments = [
        np.mean([v for seq, v in moments.items() if seq.startswith(f + "/")], axis=0)
        for f in ("kitti", "helipr")
    ]
    noise = np.mean(family_moments, axis=0)
    old_stats = run.BASE.exp.arrays(source.OUT / "statistics.npz")
    shrink = 0.5 * old_stats["moment"] + 0.5 * noise
    matrices = {
        "noise": source.whitening_matrix(noise, True),
        "shrink": source.whitening_matrix(shrink, True),
    }
    np.savez_compressed(
        OUT / "statistics.npz", noise_moment=noise, shrink_moment=shrink, **matrices
    )
    write_json(
        OUT / "training_fit.json",
        {
            "sequences": sequence_stats,
            "families": ["kitti", "helipr"],
            "anchors": sum(v["anchors"] for v in sequence_stats.values()),
            "pairs": sum(v["pairs"] for v in sequence_stats.values()),
            "test_used": False,
        },
    )
    paths.update((OUT / "statistics.npz", OUT / "training_fit.json"))
    paths.update(OUT.glob("*_pairs.npz"))
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
            "hypothesis": "전체 분포 상관만으로는 장소 구별 성분과 동일 장소에서 흔들리는 성분을 구분하지 못한다. 재방문 차이 공분산으로 불안정한 방향을 낮춘다.",
            "fit": "학습GT8내 j<=q-30·거리<=5m 전체 정답쌍. 프레임16x8전체L2정규화. 기존60방향 원시 복소 상관으로 각 쌍 회전정렬. .5*E[(a-b)(a-b)^H], 각앵커의후보균등→시퀀스균등→KITTI/HeLiPR각1/2. KITTI01은 정답쌍0으로 적합 제외. 검증/시험 통계·GT 적합 없음.",
            "methods": {
                "noise": "재방문 차이 공분산의10%등방수축 역제곱근",
                "shrink": "기존 전체분포 공분산50%+재방문 차이 공분산50%를 섞은뒤10%등방수축 역제곱근",
            },
            "selection": "기존white25기준. 새25%조건만 평균AP>=+.1%p·모든시드AP>0·평균R100>=-.1%p·각계열AP>=-.25%p 모두 통과하면 최고AP승격. 100%는 기전대조만. 실패하면기존유지. 검증반복탐색이며 독립확증 아님.",
            "evaluation": "기존 모델·후보·승자·q+15·원장·집계 고정. 동일 공통 평가기 재사용, 새 보정행렬만 변경. 추가 신경망 없음.",
            "sources": {str(p.relative_to(run.REPO)): sha256(p) for p in sorted(paths)},
            "test_used": False,
        },
    )


def run_all() -> None:
    """두 조건의 세 시드 최종 검증을 실행한다."""
    prepare()
    spec = evaluator.protocol()
    LOG.mkdir(parents=True, exist_ok=True)

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
    evaluator.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "val", "finish"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    configure()
    if args.action == "run":
        run_all()
    elif args.action == "val":
        evaluator.evaluate(args.seed, args.sequence)
    else:
        evaluator.finish()
