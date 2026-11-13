#!/usr/bin/env python3
"""입력에 무관한 128개 공통 이득만으로 새 효과가 설명되는지 대조한다."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _pdv2_diffattn_spectral_only as phase
from retrieval.context_spectral_gain import spectral_gain

e = phase.base
OUT = e.ROOT / "results/pdv2_diffattn_spectral_global_20261104"


def prepare() -> dict:
    """장소별 문맥 없이 고정 성분 비중만 학습하는 대조를 고정한다."""
    spec = phase.prepare()
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        record = e.run.READ(path)
        assert record["source_sha256"] == e.run.HASH(Path(__file__))
    else:
        e.run.WRITE(
            path,
            dict(
                created_unix=time.time(),
                source_sha256=e.run.HASH(Path(__file__)),
                parent_sha256=e.run.HASH(phase.OUT / "protocol.json"),
                scope="원래동결모델의키검색·승자유지. 학습8시퀀스균등평균게이트문맥128개로초기화한공통벡터만학습. 스캔입력과시간간선에무관한스펙트럼이득. 두lr와512갱신·동일양음배치·에포크2/4/8·강도0/.25/.5/1. 가산문맥없음. 동적모델고유기여를주장하기전고정가중치설명을확인. 용량정합대조가아님.",
                test_used=False,
            ),
        )
    return spec


def mask_key(value: torch.Tensor, n: int) -> torch.Tensor:
    """공통 이득 벡터를 기존 미분 연산이 읽는 형태로 포장한다."""
    raw = torch.zeros(n, 288, device=value.device)
    raw[:, 0] = 1
    return torch.cat((raw, value[None].expand(n, -1)), -1)


def train() -> None:
    """신경망 없이 공통 128개 비중만 같은 최종 판정 손실로 학습한다."""
    spec = prepare()
    init_path = OUT / "initial.npy"
    if not init_path.exists():
        model = e.model()
        means = []
        for s in spec["train"]:
            graph = torch.load(
                e.PARENT / "pools" / s.replace("/", "_") / "graph.pt",
                map_location="cuda",
                weights_only=False,
            )
            with torch.no_grad():
                key = model(graph)
                means.append(key[:, 288:].mean(0).cpu().numpy())
        np.save(init_path, np.mean(means, 0))
        del graph, model, key
    for arm, lr in e.ARMS.items():
        folder = OUT / arm
        folder.mkdir(exist_ok=True)
        if (folder / "training_complete.json").exists():
            continue
        value = torch.nn.Parameter(torch.from_numpy(np.load(init_path)).cuda())
        opt = torch.optim.AdamW([value], lr=lr, weight_decay=1e-5)
        records, updates = [], 0
        for epoch in range(1, 9):
            for ordinal, s in enumerate(spec["train"]):
                data = e.run.ARR(
                    e.previous.OUT / "pools" / s.replace("/", "_") / "pool.npz"
                )
                z = e.coefficients(e.run.spec_path(s))
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
                rng = np.random.default_rng(41000 + 1000 * epoch + ordinal)
                losses, orders = [], []
                for _ in range(8):
                    count = 64 // len(sets)
                    chosen = rng.permutation(
                        np.concatenate(
                            [rng.choice(v, count, replace=len(v) < count) for v in sets]
                        )
                    )
                    orders.append(e.run.array_hash(chosen))
                    ix = torch.from_numpy(chosen).cuda()
                    opt.zero_grad(set_to_none=True)
                    score, _ = phase.total_score(
                        mask_key(value, len(z)),
                        z,
                        {k: v[ix] for k, v in pp.items()},
                        0.5,
                    )
                    loss = F.binary_cross_entropy_with_logits(
                        score, torch.from_numpy(data["correct"][chosen]).cuda().float()
                    )
                    loss.backward()
                    assert torch.isfinite(value.grad).all()
                    torch.nn.utils.clip_grad_norm_([value], 1)
                    opt.step()
                    losses.append(float(loss.detach()))
                    updates += 1
                records.append(
                    dict(
                        epoch=epoch,
                        sequence=s,
                        loss=float(np.mean(losses)),
                        batch_hashes=orders,
                    )
                )
                del z, pp, score, loss
            if epoch in e.EPOCHS:
                np.save(folder / f"epoch{epoch}.npy", value.detach().cpu().numpy())
        assert updates == 512
        e.run.WRITE(
            folder / "training_complete.json",
            dict(
                status="passed",
                updates=updates,
                records=records,
                checkpoints={
                    f"epoch{ep}.npy": e.run.HASH(folder / f"epoch{ep}.npy")
                    for ep in e.EPOCHS
                },
            ),
        )
        print("TRAIN_GLOBAL", arm, updates, flush=True)


def validate(sequence: str) -> None:
    """동결 검색 승자를 쓰는 공통 이득 방법의 최종 성능을 측정한다."""
    prepare()
    folder = OUT / "val"
    folder.mkdir(exist_ok=True)
    target = folder / (sequence.replace("/", "_") + ".json")
    if target.exists():
        return
    _, _, _, travel, _, _, _, paths = e.run.inputs("val", sequence)
    source = phase.OUT / "frozen/epoch0/val" / (sequence.replace("/", "_") + ".npz")
    row = {k: v for k, v in e.run.ARR(source).items() if not k.startswith("s")}
    plan = e.plan(travel, row["query_idx"], row["j_star"])
    z = e.coefficients(paths[1])
    models = [("initial", OUT / "initial.npy")] + [
        (f"{a}_{ep}", OUT / a / f"epoch{ep}.npy") for a in e.ARMS for ep in e.EPOCHS
    ]
    values, metrics = {}, {}
    for tag, path in models:
        key = mask_key(torch.from_numpy(np.load(path)).cuda(), len(z)).cpu().numpy()
        vals = e.scores(key, z, plan)
        for s, v in vals.items():
            values[tag + "_" + s] = v
            metrics[tag + "_" + s] = e.run.base.independent_metrics(row, v)
    ledger = target.with_suffix(".npz")
    np.savez_compressed(ledger, **row, **values)
    e.run.WRITE(
        target,
        dict(
            status="passed",
            sequence=sequence,
            metrics=metrics,
            source_sha256=e.run.HASH(source),
            ledger_sha256=e.run.HASH(ledger),
            test_used=False,
        ),
    )
    print("VAL_GLOBAL", sequence, flush=True)


def finish() -> None:
    """공통 비중의 최적 최종 성능과 같은 강도 대조를 저장한다."""
    spec = prepare()
    rec = {
        s: e.run.READ(OUT / "val" / (s.replace("/", "_") + ".json"))
        for s in spec["val"]
    }
    keys = next(iter(rec.values()))["metrics"]
    results = {
        k: e.run.base.macro({s: r["metrics"][k] for s, r in rec.items()}) for k in keys
    }
    best = max(results, key=lambda k: results[k]["macro"]["AP"])
    e.run.WRITE(OUT / "summary.json", dict(results=results, best=best, test_used=False))
    print("GLOBAL_BEST", best, results[best]["macro"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "val", "finish"))
    parser.add_argument("--sequence", default="kitti360/0004")
    args = parser.parse_args()
    e.run.configure()
    if args.action == "val":
        validate(args.sequence)
    else:
        globals()[args.action]()
