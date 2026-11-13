#!/usr/bin/env python3
"""방향을 맞춘 복소 차이 전달을 최종 시간 판정에서 검증한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import _pdv2_diffattn_gain_readout as base
from gnn.spectral_difference import transport_difference
from gnn.spectral_transport import attach_transport, lidar_yaw

run = base.run
OUT = ROOT / "results/pdv2_diffattn_complex_difference_20261104"
PARENT = base.OUT
CASES = [("identity", 0.0, True)] + [
    (f"{name}_{eta:g}", eta, transport)
    for name, transport in (("transport", True), ("plain", False))
    for eta in (0.125, 0.25, 0.5)
]


def prepare() -> dict:
    """학습 없는 물리적 차이 전달의 조건을 측정 전에 고정한다."""
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        spec = run.READ(path)
        for p, digest in spec["sources"].items():
            assert run.HASH(ROOT / p) == digest, p
    else:
        parent = run.READ(PARENT / "protocol.json")
        paths = [
            Path(__file__),
            ROOT / "src/gnn/spectral_difference.py",
            ROOT / "src/gnn/spectral_transport.py",
            ROOT / "tests/gnn/test_spectral_difference.py",
            PARENT / "protocol.json",
            run.white.OUT / "statistics.npz",
        ]
        paths += list((PARENT / "pools").glob("*/*/pool.npz"))
        spec = dict(
            created_unix=time.time(),
            train=parent["train"],
            val=parent["val"],
            cases=CASES,
            sources={str(p.relative_to(ROOT)): run.HASH(p) for p in paths},
            representation="단위 L2의16x8 복소 계수를 yaw 공통 좌표로 옮겨 ±5 이웃 차이를 두 층 전달한다. 층별 eta0/.125/.25/.5, 학습 없음. 마지막에 센서 좌표로 되돌린다.",
            protocol="원키288+보조100+100의 공통 고정 승자. 처리 뒤 진폭 제곱근·기존 상관 보정75:25·60각도·양방향 시간 판정·q+15. 상수 정규화 크기는 점수에서 상쇄된다.",
            scope="학습 전 고정 평균 효과의 진단이다. DiffAttnConv 기여나 비교 가능성 확증이 아니다. 시험은 쓰지 않고 같은 연결의 학습 대조를 별도로 고정한다.",
            pose="학습 KITTI만 공식Tr로 camera 자세를LiDAR축으로 변환한다. 검증KISS는LiDAR축이다. yaw만 보정하며 병진·roll/pitch·가림은 보정하지 않는다.",
        )
        run.WRITE(path, spec)
    run.OUT = OUT
    return spec


def graph_inputs(role: str, sequence: str, end: int = None) -> tuple:
    """평가 정답을 자세 입력에 섞지 않고 실제 센서 축의 그래프를 만든다."""
    cache, _, _, travel, _, _, _, paths = run.inputs(role, sequence)
    x, poses = cache["descriptors"][:end], cache["poses"][:end]
    z = run.white.coefficients(paths[1])[:end]
    extrinsic = None
    calibration = None
    if role == "train" and sequence.startswith("kitti/"):
        path = (
            Path("/workspace/data/kitti/dataset/sequences")
            / sequence.split("/")[1]
            / "calib.txt"
        )
        lines = [
            line for line in path.read_text().splitlines() if line.startswith("Tr:")
        ]
        assert len(lines) == 1
        extrinsic = np.eye(4)
        extrinsic[:3] = np.fromstring(lines[0].split(":", 1)[1], sep=" ").reshape(3, 4)
        np.testing.assert_allclose(
            extrinsic[:3, :3].T @ extrinsic[:3, :3], np.eye(3), atol=1e-6
        )
        calibration = dict(path=str(path), sha256=run.HASH(path))
    yaw = lidar_yaw(poses, extrinsic)
    graph = run.make_graph(x, poses, np.zeros(len(x), np.int64)).cuda()
    attach_transport(graph, z, yaw, np.zeros(len(x), np.int64))
    audit = dict(
        inputs={str(p.relative_to(ROOT)): run.HASH(p) for p in paths},
        calibration=calibration,
        role=role,
        nodes=len(z),
    )
    return graph, travel[:end], audit


def filtered(graph, eta: float, transport: bool = True) -> torch.Tensor:
    """학습 없는 공통 세기의 두 층 전달을 계산한다."""
    strength = torch.full((2, len(graph.x), 1), eta, device=graph.x.device)
    return transport_difference(
        graph.spectrum,
        graph.transport_indices,
        graph.transport_mask,
        graph.transport_phase,
        strength,
        transport=transport,
    )


def coefficients(z: torch.Tensor) -> torch.Tensor:
    """전달된 복소 계수에 기존의 두 스펙트럼 변환을 적용한다."""
    matrix = torch.from_numpy(run.ARR(run.white.OUT / "statistics.npz")["full"]).to(
        z.device, z.dtype
    )
    white = torch.einsum("kij,njk->nik", matrix, z)

    def power(value):
        amplitude = value.abs()
        return torch.where(amplitude > 0, value / amplitude.clamp_min(1e-12).sqrt(), 0)

    return torch.stack((power(z), power(white)), dim=1)


def score(z: torch.Tensor, plan: dict) -> torch.Tensor:
    """미분 가능한 기존 최종 판정과 학습8의 고정 척도를 사용한다."""
    dummy = torch.zeros(len(z), 416, device=z.device)
    dummy[:, 0] = 1
    return base.score(dummy, coefficients(z), plan, strength=0.0)


def evaluate(sequence: str) -> None:
    """동일 승자에서 모든 고정 평균 조건의 최종 지표를 측정한다."""
    prepare()
    name = sequence.replace("/", "_")
    folder = OUT / "val"
    folder.mkdir(exist_ok=True)
    target = folder / (name + ".json")
    if target.exists():
        return
    graph, travel, audit = graph_inputs("val", sequence)
    data = run.ARR(PARENT / "pools/val" / name / "pool.npz")
    row = {
        k: data[k]
        for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")
    }
    plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
    values, metrics, checks = {}, {}, []
    with torch.no_grad():
        for tag, eta, transport in CASES:
            z = filtered(graph, eta, transport)
            parts = []
            for start in range(0, len(row["query_idx"]), 128):
                pp = {
                    k: torch.from_numpy(v[start : start + 128]).cuda()
                    for k, v in plan.items()
                }
                parts.append(score(z, pp).cpu().numpy())
            value = np.concatenate(parts).astype(np.float64)
            for at in (len(value) // 3, 2 * len(value) // 3):
                t = int(row["query_idx"][at])
                small, short_travel, _ = graph_inputs("val", sequence, t + 16)
                short = filtered(small, eta, transport)
                error = float((short[: t + 6] - z[: t + 6]).abs().max())
                assert error < 4e-6
                pp = {
                    k: torch.from_numpy(v).cuda()
                    for k, v in base.prior.plan(
                        short_travel,
                        row["query_idx"][at : at + 1],
                        row["j_star"][at : at + 1],
                    ).items()
                }
                for k, v in pp.items():
                    np.testing.assert_array_equal(v.cpu(), plan[k][at : at + 1])
                check = score(short, pp).cpu().numpy()
                np.testing.assert_allclose(check, value[at : at + 1], atol=8e-6, rtol=0)
                checks.append(
                    dict(
                        condition=tag,
                        query=t,
                        coefficient_error=error,
                        score_error=float(abs(check - value[at : at + 1]).max()),
                    )
                )
                del small, short
            values[tag] = value
            metrics[tag] = run.base.independent_metrics(row, value)
    old = run.ARR(PARENT / "val_checked" / (name + ".npz"))
    identity_error = float(abs(values["identity"] - old["uniform"]).max())
    assert identity_error < 8e-6, identity_error
    path = target.with_suffix(".npz")
    np.savez_compressed(path, **row, **values)
    run.WRITE(
        target,
        dict(
            metrics=metrics,
            prefix_checks=checks,
            identity_error=identity_error,
            ledger_sha256=run.HASH(path),
            audit=audit,
            test_used=False,
        ),
    )
    print("VAL", sequence, flush=True)


def finish() -> None:
    """3계열 평균과 계열별 결과를 보고하며 학습 기여로 해석하지 않는다."""
    spec = prepare()
    rows = {
        s: run.READ(OUT / "val" / (s.replace("/", "_") + ".json")) for s in spec["val"]
    }
    results = {
        k: run.base.macro({s: v["metrics"][k] for s, v in rows.items()})
        for k, _, _ in CASES
    }
    run.WRITE(
        OUT / "summary.json", dict(results=results, learned=False, test_used=False)
    )
    print("FINISHED", {k: v["macro"] for k, v in results.items()}, flush=True)


def campaign() -> None:
    """검증 시퀀스별 진단을 진행한다."""
    spec = prepare()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)

    def call(s):
        with (logs / (s.replace("/", "_") + ".log")).open("w") as stream:
            subprocess.run(
                [sys.executable, __file__, "val", "--sequence", s],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call, s) for s in spec["val"]]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            run.WRITE(
                OUT / "status.json",
                dict(phase="validation", done=i, total=len(futures)),
            )
            print("VALIDATED", i, flush=True)
    finish()
    run.WRITE(OUT / "status.json", dict(phase="complete", test_used=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "val", "finish", "run"))
    parser.add_argument("--sequence", default="kitti/09")
    args = parser.parse_args()
    run.configure()
    if args.action == "val":
        evaluate(args.sequence)
    elif args.action == "run":
        campaign()
    else:
        globals()[args.action]()
