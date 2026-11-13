#!/usr/bin/env python3
"""기존 DiffAttnConv 출력으로 스펙트럼 성분을 조절하는 대조 실험."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import _pdv2_diffattn_reliability as previous
from retrieval.context_spectral_gain import gain_score
from retrieval.gated_context import gated_context_score
from retrieval.sequence_evidence import neighbor_plans, repeated_candidate_weights

run = previous.run
PARENT = previous.PARENT
OUT = ROOT / "results/pdv2_diffattn_spectral_gain_20261104"
STRENGTHS = (0.0, 0.25, 0.5, 1.0)
ARMS = {"conv_lr4": 1e-4, "conv_lr3": 1e-3}
EPOCHS = (2, 4, 8)


def prepare() -> dict:
    """비교 전에 입력·변경 범위·판정 기준을 고정한다."""
    OUT.mkdir(exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = run.READ(target)
        for p, h in spec["sources"].items():
            assert run.HASH(ROOT / p) == h, p
    else:
        parent = run.READ(PARENT / "protocol.json")
        initial = PARENT / "recenter_frozen/epoch0.pth"
        paths = [
            Path(__file__),
            ROOT / "src/retrieval/context_spectral_gain.py",
            ROOT / "tests/retrieval/test_context_spectral_gain.py",
            initial,
            ROOT / "src/gnn/geometry_interaction.py",
            ROOT / "src/gnn/model.py",
            ROOT / "src/retrieval/gated_context.py",
            Path(run.__file__),
            ROOT / "src/retrieval/sequence_evidence.py",
            PARENT / "calibration_recenter.json",
        ]
        for s in parent["train"]:
            name = s.replace("/", "_")
            paths += [
                PARENT / "pools" / name / "graph.pt",
                previous.OUT / "pools" / name / "pool.npz",
            ]
        spec = dict(
            created_unix=time.time(),
            seed=1,
            test_used=False,
            initial=str(initial.relative_to(ROOT)),
            train=parent["train"],
            val=parent["val"],
            confidence=parent["confidence"],
            strengths=STRENGTHS,
            arms=ARMS,
            epochs=EPOCHS,
            training_strength=0.5,
            steps_per_sequence=8,
            batch_queries=64,
            method="기존128차원게이트문맥u에서g=exp(s*tanh(sqrt(128)*u))를16행8주파수에대응. 원본/보정제곱근진폭계수에g를곱하고기존60각도·시간대응·75/25·두채널판정. 양의이득이라위상유지. 새학습층없음. 키와기존가산문맥0.5도유지.",
            train_scope="gnn.convs.*14텐서만학습. 투영/게이트가중치/BN통계/재계량고정. 게이트출력은간접변화. 시드1포화완화시작. 고정초기학습승자·정답/오답균등BCE·같은배치512갱신. 기존학습보정척도고정. 검증후보전체재검색.",
            selection="학습후s>0가초기최적강도및동일강도및자체s0보다AP+.1%p. 동일강도F1손실<=.3%p,R100<=1%p,R1<=.3%p. BEV동일시간검증AP/F1열세2%p이내,R100/R1열세3%p이내. 통과중AP최대·작은s·작은lr·이른epoch. 성공후고정3시드/그래프없는대조/표준attention대조. 새시험없음.",
            limitations="128출력좌표와주파수의대응은새귀납편향이고초기값의물리적의미없음. 반복검증탐색. 단일시드이득은operator고유우위아님. 원고/그림유지.",
            sources={str(p.relative_to(ROOT)): run.HASH(p) for p in paths},
        )
        run.WRITE(target, spec)
    run.OUT = OUT
    return spec


def model(checkpoint: Path = None) -> torch.nn.Module:
    """현재 구조에서 두 차이 attention 층만 학습 가능하게 한다."""
    spec = prepare()
    net = previous.prior.gate.MODEL(
        ROOT / spec["initial"] if checkpoint is None else checkpoint
    )
    for name, p in net.named_parameters():
        p.requires_grad_(name.startswith("gnn.convs."))
    return net


def plan(travel: np.ndarray, q: np.ndarray, j: np.ndarray) -> dict:
    """기존 중복 보정 시간 대응과 중심 가중치를 구성한다."""
    qn, jn, valid = neighbor_plans(travel, q, j)["arc_unique"]
    w = repeated_candidate_weights(jn, valid)
    w = np.concatenate((np.ones(w.shape[:2] + (1,)), w), -1)
    w /= w.sum(-1, keepdims=True)
    w *= 11 / 15
    w[..., 0] += 4 / 15
    shape = qn.shape[:2] + (1,)
    qi = np.concatenate((np.broadcast_to(q[None, :, None], shape), qn), -1)
    ji = np.concatenate((np.broadcast_to(j[None, :, None], shape), jn), -1)
    return {
        "query": np.clip(qi, 0, len(travel) - 1).transpose(1, 0, 2).copy(),
        "candidate": np.clip(ji, 0, len(travel) - 1).transpose(1, 0, 2).copy(),
        "weight": w.transpose(1, 0, 2).astype(np.float32),
    }


def coefficients(path: Path) -> torch.Tensor:
    """학습 고정 상관 변환과 진폭 제곱근까지 적용한 계수를 읽는다."""
    z = run.white.coefficients(path)
    wz = run.white.transform(z, run.ARR(run.white.OUT / "statistics.npz")["full"])
    return torch.from_numpy(
        np.stack([run.power.transform(x, 0.5) for x in (z, wz)], 1)
    ).cuda()


def total_score(key: torch.Tensor, z: torch.Tensor, p: dict, strength: float) -> tuple:
    """기존 학습 보정 척도와 가산 문맥을 유지한 최종 점수를 반환한다."""
    fit = run.READ(PARENT / "calibration_recenter.json")
    spec = gain_score(key, z, p, strength)
    affine = fit["spectral"]
    slope = affine["coefficients"][0] / affine["scale"][0]
    context = gated_context_score(key, p)
    total = slope * (spec - affine["mean"][0]) + affine["intercept"]
    total = total + 0.5 * (context - fit["context_mean"]) / fit["context_scale"]
    return total, spec


def scores(key: np.ndarray, z: torch.Tensor, p: dict) -> dict:
    """모든 강도에서 같은 최종 판정식을 실행한다."""
    values = {f"s{s:g}": [] for s in STRENGTHS}
    embedding = torch.from_numpy(key).cuda()
    with torch.no_grad():
        for start in range(0, len(p["query"]), 128):
            pp = {
                k: torch.from_numpy(v[start : start + 128]).cuda() for k, v in p.items()
            }
            for strength in STRENGTHS:
                value, _ = total_score(embedding, z, pp, strength)
                values[f"s{strength:g}"].append(value.cpu().numpy())
    return {k: np.concatenate(v).astype(np.float64) for k, v in values.items()}


def preflight() -> None:
    """실제 입력에서 항등 점수·캐시 그래프·두 층의 미분을 검증한다."""
    spec = prepare()
    seq = "kitti/06"
    cache, pos, minimum, travel, bank, nb, nc, paths = run.inputs("train", seq)
    net = model()
    key = run.source.make_keys(net, cache["descriptors"], cache["poses"])
    row = run.ARR(previous.OUT / "pools/kitti_06/pool.npz")
    q, j = row["query_idx"][::83], row["j_star"][::83]
    p = {k: torch.from_numpy(v).cuda() for k, v in plan(travel, q, j).items()}
    z = coefficients(paths[1])
    with torch.no_grad():
        _, actual = total_score(torch.from_numpy(key).cuda(), z, p, 0)
    expected = run.current.adoption.scores(nb, nc, travel, q, j)["score"]
    error = float(np.max(abs(actual.cpu().numpy() - expected)))
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=5e-7, rtol=0)
    graph = torch.load(
        PARENT / "pools/kitti_06/graph.pt", map_location="cuda", weights_only=False
    )
    newkey = F.normalize(net(graph), dim=-1)
    np.testing.assert_allclose(newkey.detach().cpu().numpy(), key, atol=2e-6, rtol=0)
    loss = total_score(newkey, z, p, 0.5)[0].mean()
    loss.backward()
    norms = {
        n: float(v.grad.norm()) for n, v in net.named_parameters() if v.requires_grad
    }
    assert all(np.isfinite(v) for v in norms.values()) and sum(norms.values()) > 0
    run.WRITE(
        OUT / "preflight.json",
        dict(
            status="passed",
            identity_max_abs=error,
            gradients=norms,
            protocol_sha256=run.HASH(OUT / "protocol.json"),
        ),
    )
    print("PREFLIGHT", error, sum(norms.values()), flush=True)


def train(arm: str) -> None:
    """최종 점수의 BCE를 스펙트럼 연산을 거쳐 두 층에 역전파한다."""
    spec = prepare()
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    if (folder / "training_complete.json").exists():
        return
    assert not list(folder.glob("epoch*.pth"))
    torch.manual_seed(1)
    net = model()
    initial = {n: p.detach().cpu().clone() for n, p in net.state_dict().items()}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=ARMS[arm], weight_decay=1e-5)
    records, hashes, updates = [], {}, 0
    tick = time.time()
    for epoch in range(1, max(EPOCHS) + 1):
        for ordinal, seq in enumerate(spec["train"]):
            name = seq.replace("/", "_")
            data = run.ARR(previous.OUT / "pools" / name / "pool.npz")
            graph = torch.load(
                PARENT / "pools" / name / "graph.pt",
                map_location="cuda",
                weights_only=False,
            )
            z = coefficients(run.spec_path(seq))
            pp = {
                k: torch.from_numpy(data["plan_" + k]).cuda()
                for k in ("query", "candidate", "weight")
            }
            pp["weight"] = pp["weight"].float()
            sets = [
                np.flatnonzero(data["eligible"] & (data["correct"] == c))
                for c in (0, 1)
            ]
            sets = [v for v in sets if len(v)]
            rng = np.random.default_rng(41000 + epoch * 1000 + ordinal)
            losses, grads, orders = [], [], []
            for step in range(spec["steps_per_sequence"]):
                count = spec["batch_queries"] // len(sets)
                chosen = rng.permutation(
                    np.concatenate(
                        [rng.choice(v, count, replace=len(v) < count) for v in sets]
                    )
                )
                orders.append(run.array_hash(chosen))
                ix = torch.from_numpy(chosen).cuda()
                opt.zero_grad(set_to_none=True)
                key = F.normalize(net(graph), dim=-1)
                score, _ = total_score(key, z, {k: v[ix] for k, v in pp.items()}, 0.5)
                target = torch.from_numpy(data["correct"][chosen]).cuda().float()
                loss = F.binary_cross_entropy_with_logits(score, target)
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
                assert torch.isfinite(grad)
                opt.step()
                losses.append(float(loss))
                grads.append(float(grad))
                updates += 1
            record = dict(
                epoch=epoch,
                sequence=seq,
                updates=updates,
                loss=float(np.mean(losses)),
                grad=float(np.mean(grads)),
                batch_hashes=orders,
            )
            records.append(record)
            run.WRITE(folder / "progress.json", record)
            del graph, key, pp, z, score, loss
            torch.cuda.empty_cache()
        if epoch in EPOCHS:
            cp = folder / f"epoch{epoch}.pth"
            torch.save(
                dict(
                    model_state_dict=net.state_dict(),
                    epoch=epoch,
                    updates=updates,
                    protocol_sha256=run.HASH(OUT / "protocol.json"),
                ),
                cp,
            )
            hashes[cp.name] = run.HASH(cp)
        print("TRAIN", arm, epoch, updates, round(time.time() - tick, 1), flush=True)
    changed = [
        n
        for n, p in net.state_dict().items()
        if not torch.equal(p.detach().cpu(), initial[n])
    ]
    assert changed and all(n.startswith("gnn.convs.") for n in changed)
    run.WRITE(
        folder / "training_complete.json",
        dict(
            status="passed",
            updates=updates,
            records=records,
            checkpoints=hashes,
            changed_parameters=changed,
            seconds=time.time() - tick,
        ),
    )


def validate(arm: str, epoch: int, sequence: str) -> None:
    """키 검색부터 최종 판정까지 실행하고 q+15 접두 입력을 확인한다."""
    spec = prepare()
    folder = OUT / arm / f"epoch{epoch}/val"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (sequence.replace("/", "_") + ".json")
    if target.exists():
        assert run.READ(target)["status"] == "passed"
        return
    cp = ROOT / spec["initial"] if arm == "frozen" else OUT / arm / f"epoch{epoch}.pth"
    tick = time.time()
    net = model(cp)
    cache, pos, minimum, travel, bank, nb, nc, paths = run.inputs("val", sequence)
    key = run.source.make_keys(net, cache["descriptors"], cache["poses"])
    row, _ = run.retrieve(key, bank, pos, minimum)
    reference = run.ARR(
        previous.OUT / "frozen/epoch0/val" / (sequence.replace("/", "_") + ".npz")
    )
    row["group"] = reference["group"]
    np.testing.assert_array_equal(row["query_idx"], reference["query_idx"])
    z = coefficients(paths[1])
    p = plan(travel, row["query_idx"], row["j_star"])
    values = scores(key, z, p)
    metrics = {k: run.base.independent_metrics(row, v) for k, v in values.items()}
    for k, v in values.items():
        other = run.white.prior.BASE.exp.metrics(v, row)
        for m in run.base.METRICS:
            if metrics[k][m] is not None:
                np.testing.assert_allclose(metrics[k][m], other[m], atol=1e-12, rtol=0)
    prefix = []
    for at in (len(row["query_idx"]) // 3, 2 * len(row["query_idx"]) // 3):
        t = int(row["query_idx"][at])
        end = t + 16
        short = run.source.make_keys(net, cache["descriptors"], cache["poses"], end)
        np.testing.assert_allclose(short[: t + 6], key[: t + 6], atol=4e-6, rtol=4e-5)
        pp = plan(
            travel[:end], row["query_idx"][at : at + 1], row["j_star"][at : at + 1]
        )
        out = scores(short, z[:end], pp)
        for k, v in out.items():
            np.testing.assert_allclose(v, values[k][at : at + 1], atol=8e-6, rtol=0)
        prefix.append(t)
    ledger = target.with_suffix(".npz")
    np.savez_compressed(ledger, **row, **values)
    run.WRITE(
        target,
        dict(
            status="passed",
            arm=arm,
            epoch=epoch,
            sequence=sequence,
            metrics=metrics,
            prefix_checks=prefix,
            rows=len(key),
            checkpoint_sha256=run.HASH(cp),
            ledger_sha256=run.HASH(ledger),
            inputs={str(p.relative_to(ROOT)): run.HASH(p) for p in paths},
            seconds=time.time() - tick,
        ),
    )
    print("VAL", arm, epoch, sequence, round(time.time() - tick, 1), flush=True)


def finish() -> None:
    """새 역할의 학습 기여와 강도0 대조 및 BEV 비교를 함께 집계한다."""
    spec = prepare()
    records = {}
    for arm, epochs in [("frozen", (0,))] + [(a, EPOCHS) for a in ARMS]:
        for epoch in epochs:
            seqs = {
                s: run.READ(
                    OUT / arm / f"epoch{epoch}/val" / (s.replace("/", "_") + ".json")
                )
                for s in spec["val"]
            }
            for s, rec in seqs.items():
                path = OUT / arm / f"epoch{epoch}/val" / (s.replace("/", "_") + ".npz")
                assert run.HASH(path) == rec["ledger_sha256"]
            for strength in STRENGTHS:
                rows = {s: r["metrics"][f"s{strength:g}"] for s, r in seqs.items()}
                records[f"{arm}_{epoch}_{strength:g}"] = dict(
                    arm=arm,
                    epoch=epoch,
                    strength=strength,
                    sequences=rows,
                    **run.base.macro(rows),
                )
    best_frozen = max(
        [r for r in records.values() if r["arm"] == "frozen"],
        key=lambda r: r["macro"]["AP"],
    )
    bev = run.READ(previous.prior.gate.previous.OUT / "bev_validation.json")[
        "temporal"
    ]["macro"]
    eligible = []
    for r in records.values():
        if r["arm"] == "frozen":
            continue
        strength, m = r["strength"], r["macro"]
        base = records[f"frozen_0_{strength:g}"]["macro"]
        zero = records[f"{r['arm']}_{r['epoch']}_0"]["macro"]
        checks = dict(
            uses_gain=strength > 0,
            same_gain=m["AP"] >= base["AP"] + 0.001,
            best_frozen=m["AP"] >= best_frozen["macro"]["AP"] + 0.001,
            beats_zero=m["AP"] >= zero["AP"] + 0.001,
        )
        checks.update(
            {
                f"preserve_{k}": m[k] >= base[k] - tol
                for k, tol in {"max_F1": 0.003, "R@100P": 0.01, "R@1": 0.003}.items()
            }
        )
        checks.update(
            {
                f"bev_{k}": m[k] >= bev[k] - tol
                for k, tol in {
                    "AP": 0.02,
                    "max_F1": 0.02,
                    "R@100P": 0.03,
                    "R@1": 0.03,
                }.items()
            }
        )
        r["checks"] = checks
        if all(checks.values()):
            eligible.append(r)
    selected = (
        min(
            eligible,
            key=lambda r: (
                -r["macro"]["AP"],
                r["strength"],
                ARMS[r["arm"]],
                r["epoch"],
            ),
        )
        if eligible
        else None
    )
    run.WRITE(
        OUT / "summary.json",
        dict(
            records=records,
            best_frozen=best_frozen,
            selected=selected,
            bev_temporal=bev,
            test_used=False,
        ),
    )
    print(
        "FINISH",
        (
            None
            if selected is None
            else (
                selected["arm"],
                selected["epoch"],
                selected["strength"],
                selected["macro"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "preflight", "train", "val", "finish"))
    p.add_argument("--arm", default="conv_lr4")
    p.add_argument("--epoch", type=int, default=4)
    p.add_argument("--sequence", default="kitti360/0004")
    args = p.parse_args()
    run.configure()
    if args.action == "train":
        train(args.arm)
    elif args.action == "val":
        validate(args.arm, args.epoch, args.sequence)
    else:
        globals()[args.action]()
