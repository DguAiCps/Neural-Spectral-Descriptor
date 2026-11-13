#!/usr/bin/env python3
"""학습에서 정한 시간 평균 점수를 고정 후보와 최종 시간 판정에서 진단한다."""
import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_fusion_factor as factor
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_fusion_factor_lib import fuse_with_range
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_temporal_test_control_finish import scalar_score

OUT = REPO / "results/pdv2_temporal_mean_score_20261025"
LOG = REPO / "logs/temporal_mean_score_20261025"
SUPPORT = REPO / "results/pdv2_revisit_support_20261025/completed"
VARIANTS = ("context", "mean5")
base, METHOD = factor.base, factor.METHOD


def mean_key(key: np.ndarray) -> np.ndarray:
    """시퀀스 안에서 앞뒤5와 중심의 원 특징을 평균하고 정규화한다."""
    raw = key[:, :288].astype(np.float64)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    prefix = np.r_[np.zeros((1, 288)), np.cumsum(raw, axis=0)]
    idx = np.arange(len(raw))
    lo, hi = np.maximum(idx - 5, 0), np.minimum(idx + 6, len(raw))
    avg = (prefix[hi] - prefix[lo]) / (hi - lo)[:, None]
    return avg / np.linalg.norm(avg, axis=1, keepdims=True)


def prepare() -> dict:
    """완료된 학습 진단으로 반경을 고정하고 입력·코드 해시를 동결한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "protocol.json").exists()
    sources = {}
    for parent in (SUPPORT, factor.OUT):
        name = "completion_validation.json"
        audit = base.read(parent / name)
        assert audit["passed"]
        sources.update(audit["sources"])
        sources[str((parent / name).relative_to(REPO))] = sha256(parent / name)
    sources[str(Path(__file__).relative_to(REPO))] = sha256(Path(__file__))
    spec = factor.protocol(True)
    result = {
        "sources": sources,
        "train": spec["train"],
        "val": spec["val"],
        "confidence": spec["confidence"],
        "radius": 5,
        "test_used": False,
        "scope": "학습 전체오답 진단에서 정답우위 질의255개인±5 평균을 고정한다. "
        "학습GT8·검증KISS9, 시작 interaction 모델의 후보100+100을 유지하고 "
        "키 점수만 원 특징 시간 평균의 코사인으로 교체한다. 정규화·융합계수·"
        "최종시간판정은 동일. 중간 특징 비교이며 재학습이나 새모델 채택이 아니다.",
    }
    write_json(OUT / "protocol.json", result)
    return result


def replay(role: str, sequence: str) -> None:
    """두 점수 조건의 최종 원장을 만들고 기존 결과·직접 수식·접두 입력과 대조한다."""
    spec = base.read(OUT / "protocol.json")
    assert sequence in spec[role]
    cache = base.exp.arrays(factor.paths_for(role, sequence)["cache"])
    data = base.exp.arrays(factor.key_path(role, "interaction", sequence))
    np.testing.assert_array_equal(cache["scan_ids"], data["scan_ids"])
    key = data["embedding"]
    mean = mean_key(key)
    with np.load(factor.paths_for(role, sequence)["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        bank = base.exp.old.base.bank_for(base.exp.OP, z)
    parent = factor.OUT / role / sequence.replace("/", "_")
    expected = base.exp.arrays(parent / "new.npz")
    q = expected["query_idx"]
    xyz = cache["poses"][:, :3, 3]
    positions = cache["poses" if role == "train" else "poses_gt"][:, :3, 3].astype(
        np.float64
    )
    travel = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    conf = spec["confidence"]["1"]
    weights = tuple(conf["tuple"])
    ledgers = {
        name: {
            "query_idx": q,
            "min_past_gt_dist": expected["min_past_gt_dist"],
            "j_star": np.empty(len(q), np.int64),
            "top1_geo_dist": np.empty(len(q)),
            "union_hit": expected["union_hit"],
            "raw": np.empty(len(q)),
        }
        for name in VARIANTS
    }
    checks = []
    for i, query in enumerate(q):
        t = int(query)
        pool = np.unique(
            np.r_[
                base.prefix_topk(key, t, t - 29, 100),
                base.prefix_topk(bank[0], t, t - 29, 100),
            ]
        )
        assert pool.max() <= t - 30
        common = (
            bank[0][pool] @ bank[0][t],
            base.align_scores(bank[1], bank[2], t, pool),
            base.column_align_scores(bank[3], t, pool),
        )
        distance = np.linalg.norm(positions[pool] - positions[t], axis=1)
        assert bool((distance <= 5).any()) == bool(expected["union_hit"][i])
        for name, scorekey in (("context", key), ("mean5", mean)):
            channels = np.column_stack((scorekey[pool] @ scorekey[t], *common))
            score, _ = fuse_with_range(channels, weights, channels)
            winner = int(np.argsort(-score, kind="stable")[0])
            row = ledgers[name]
            row["j_star"][i] = pool[winner]
            row["top1_geo_dist"][i] = distance[winner]
            row["raw"][i] = channels[winner, 3 if conf["decision"] == "c_col" else 2]
            if i in (0, len(q) // 3, 2 * len(q) // 3, len(q) - 1):
                direct = sum(
                    float(w)
                    * (channels[:, c] - channels[:, c].min())
                    / max(float(np.ptp(channels[:, c])), 1e-8)
                    for c, w in enumerate(weights)
                )
                assert abs(direct - score).max() < 2e-6
                assert int(pool[np.argmax(direct)]) == int(row["j_star"][i])
        if i in (len(q) // 3, 2 * len(q) // 3):
            small = mean_key(key[: t + 16])
            error = float(abs(small[: t + 1] - mean[: t + 1]).max())
            assert error < 1e-12
            checks.append({"q": t, "mean_prefix_error": error})
    folder = OUT / role / sequence.replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)
    for name, row in ledgers.items():
        row[METHOD] = np.concatenate(
            [
                base.fixed.frozen_scores(
                    bank,
                    travel,
                    q[s : s + 256],
                    row["j_star"][s : s + 256],
                    conf["decision"],
                    row["raw"][s : s + 256],
                )[METHOD]
                for s in range(0, len(q), 256)
            ]
        )
        if name == "context":
            np.testing.assert_array_equal(row["j_star"], expected["j_star"])
            np.testing.assert_allclose(row[METHOD], expected[METHOD], atol=1e-7, rtol=0)
        for record in checks:
            t = record["q"]
            i = t - 30
            j = int(row["j_star"][i])
            direct = scalar_score(
                bank, travel, t, j, conf["decision"], float(row["raw"][i])
            )
            small = base.fixed.frozen_scores(
                base.exp.truncate_bank(bank, t + 16),
                travel[: t + 16],
                q[i : i + 1],
                row["j_star"][i : i + 1],
                conf["decision"],
                row["raw"][i : i + 1],
            )[METHOD][0]
            assert abs(direct - row[METHOD][i]) < 1e-12
            assert abs(small - row[METHOD][i]) < 1e-12
            record[name + "_final_error"] = float(abs(direct - row[METHOD][i]))
        np.savez_compressed(folder / (name + ".npz"), **row)
    write_json(
        folder / "audit.json",
        {
            "checks": checks,
            "passed": True,
            "metrics": {
                name: independent_metrics(row, row[METHOD])
                for name, row in ledgers.items()
            },
            "sources": {
                str((folder / (name + ".npz")).relative_to(REPO)): sha256(
                    folder / (name + ".npz")
                )
                for name in VARIANTS
            },
        },
    )


def finish() -> None:
    """최종 원장의 독립 지표·시퀀스와 계열 평균·정답 손익·해시를 검산한다."""
    spec = base.read(OUT / "protocol.json")
    sources = dict(spec["sources"])
    results, changes = {}, {}
    ledgers = rows = checks = 0
    error = 0.0
    for role in ("train", "val"):
        metrics = {name: {} for name in VARIANTS}
        changes[role] = {}
        for seq in spec[role]:
            folder = OUT / role / seq.replace("/", "_")
            audit = base.read(folder / "audit.json")
            sources.update(audit["sources"])
            sources[str((folder / "audit.json").relative_to(REPO))] = sha256(
                folder / "audit.json"
            )
            correct = {}
            for name in VARIANTS:
                row = base.exp.arrays(folder / (name + ".npz"))
                actual = independent_metrics(row, row[METHOD])
                canonical = base.exp.metrics(row[METHOD], row)
                for metric in base.exp.METRICS:
                    if actual[metric] is None:
                        assert canonical[metric] is None
                    else:
                        error = max(error, abs(actual[metric] - canonical[metric]))
                        assert (
                            abs(actual[metric] - audit["metrics"][name][metric]) < 5e-13
                        )
                metrics[name][seq] = actual
                correct[name] = row["top1_geo_dist"] <= 5
                assert np.all(row["j_star"] <= row["query_idx"] - 30)
                ledgers += 1
                rows += len(row["query_idx"])
            changes[role][seq] = {
                "gained": int((correct["mean5"] & ~correct["context"]).sum()),
                "lost": int((~correct["mean5"] & correct["context"]).sum()),
            }
            checks += len(audit["checks"])
        results[role] = {name: base.macro(data) for name, data in metrics.items()}
        print(
            "MEAN_FINAL",
            role,
            {name: data["macro"] for name, data in results[role].items()},
            flush=True,
        )
    assert error < 5e-13
    write_json(
        OUT / "summary.json",
        {
            "results": results,
            "changes": changes,
            "test_used": False,
            "training_updates": 0,
            "scope": spec["scope"],
        },
    )
    for path in (OUT / "summary.json", OUT / "protocol.json"):
        sources[str(path.relative_to(REPO))] = sha256(path)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest, path
    write_json(
        OUT / "completion_validation.json",
        {
            "passed": True,
            "sources": sources,
            "ledgers": ledgers,
            "rows": rows,
            "prefix_checks": checks,
            "maximum_metric_error": error,
            "test_used": False,
            "training_updates": 0,
        },
    )


def main() -> None:
    """독립 시퀀스를 두 개씩 평가하고 최종 결과를 검산한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "replay", "finish"))
    parser.add_argument("--role", choices=("train", "val"))
    parser.add_argument("--sequence")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "replay":
        replay(args.role, args.sequence)
    elif args.action == "finish":
        finish()
    else:
        spec = prepare()
        LOG.mkdir(parents=True, exist_ok=True)

        def worker(job: tuple) -> None:
            role, seq = job
            path = LOG / (role + "_" + seq.replace("/", "_") + ".log")
            with path.open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        __file__,
                        "replay",
                        "--role",
                        role,
                        "--sequence",
                        seq,
                    ],
                    cwd=REPO,
                    env=os.environ.copy(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                raise RuntimeError(path.read_text()[-6000:])

        jobs = [(role, seq) for role in ("train", "val") for seq in spec[role]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                future.result()
                write_json(
                    OUT / "status.json",
                    {"done": count, "total": len(jobs), "test_used": False},
                )
                print("MEAN_REPLAY_DONE", count, len(jobs), flush=True)
        finish()
        write_json(OUT / "status.json", {"status": "complete", "test_used": False})


if __name__ == "__main__":
    main()
