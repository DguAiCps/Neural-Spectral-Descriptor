#!/usr/bin/env python3
"""정규화 전 출력층을 함께 학습하며 차이 attention의 기여를 분리한다."""
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
import _pdv2_diffattn_spectral_gain as prior
import _pdv2_table4_current as table4
from gnn.spectral_gain_readout import GlobalGainReadout, SpectralGainReadout

run = prior.run
OUT = ROOT / "results/pdv2_diffattn_gain_readout_20261104"
GLOBAL = ROOT / "results/pdv2_diffattn_spectral_global_20261104/conv_lr3/epoch8.npy"
KINDS = ("diff", "head_only", "standard", "graph_free", "global")
ARMS = {
    f"{k}_{tag}": (k, lr) for k in KINDS for tag, lr in (("lr4", 1e-4), ("lr3", 1e-3))
}
EPOCHS = (4, 8)


def prepare() -> dict:
    """동일 초기 판정·학습 예산·구조 대조를 먼저 고정한다."""
    OUT.mkdir(exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = run.READ(target)
        for p, h in spec["sources"].items():
            assert run.HASH(ROOT / p) == h, p
    else:
        parent = run.READ(prior.OUT / "protocol.json")
        checkpoints = {
            "diff": str(
                (
                    run.source.root(1, "octave288")
                    / "checkpoints/checkpoint_epoch_20.pth"
                ).relative_to(ROOT)
            )
        }
        checkpoints.update(
            {
                k: str(
                    (
                        table4.root(1, k) / "checkpoints/checkpoint_epoch_20.pth"
                    ).relative_to(ROOT)
                )
                for k in ("standard", "graph_free")
            }
        )
        paths = [
            Path(__file__),
            ROOT / "src/gnn/spectral_gain_readout.py",
            GLOBAL,
            ROOT / "src/retrieval/context_spectral_gain.py",
            Path(prior.__file__),
            Path(table4.__file__),
            Path(run.source.__file__),
            Path(run.__file__),
            ROOT / "src/gnn/model.py",
            ROOT / "src/gnn/geometry_interaction.py",
            prior.PARENT / "calibration_recenter.json",
        ] + [ROOT / p for p in checkpoints.values()]
        spec = dict(
            created_unix=time.time(),
            seed=1,
            train=parent["train"],
            val=parent["val"],
            confidence=parent["confidence"],
            checkpoints=checkpoints,
            arms=ARMS,
            epochs=EPOCHS,
            train_strength=0.5,
            steps_per_sequence=8,
            batch_queries=64,
            architecture="기존두층DiffAttnConv유지. 정규화전기존128출력층을스펙트럼이득입력u로사용. 출력weight0·bias=선행공통이득학습벡터. 모든구조의초기스펙트럼판정완전동일. 새층없음. 게이트/문맥L2를이득출력에서사용하지않음. 원288키+보조키100+100검색은학습중불변. 원복소계수256실수+원키288+이득128=672실수저장가능;배포효율실측아님.",
            training="학습8실제원키검색승자재생성·고정. 기존학습척도BCE. 두층+출력층만학습;head_only는Diff사전학습층고정·출력층만. BN통계/다른투영/재계량고정. 가산문맥없음. 각512갱신·동일64양음균등표본·AdamW감쇠1e-5·clip1.",
            controls="Diff/표준/그래프없는대조는각자의동일학습8·동일900회·20epoch사전학습가중치로시작. 원래20epoch선택을모두고정하며검증선택체크포인트차이를없앰. 그래프없는MLP는거의같은167705대167561용량. head_only/global추가대조. 각구조두lr·4/8epoch동일선택기회.",
            score="s=.5고정·g=exp(s*tanh(sqrt128*u)). 원본/보정제곱근계수·위상유지·60각도·75/25·두채널·기존시간판정. q+15·j<=q-30. 이득없는0과공통이득초기를동일검색에서보고.",
            selection="diff최적AP가공통시작점및각대조최적AP보다+.1%p. 대조들대비F1손실<=.3%p,R100<=1%p. 같은시간BEV검증AP/F1열세2%p,R100/R1열세3%p이내. 통과하면고정설정3시드확인. 동일시드단일검증으로operator우위주장금지. 시험사용없음.",
            motivation="정규화문맥128좌표를그대로읽는조건은실패했고공통이득만학습하면AP/F1/R100이함께개선. 출력층고정제약을풀고역할을스펙트럼조절로집중. 구조별사전학습가중치가달라추가학습효과와처음부터의우위는구분. 반복검증개발.",
            sources={str(p.relative_to(ROOT)): run.HASH(p) for p in paths},
            test_used=False,
        )
        run.WRITE(target, spec)
    run.OUT = OUT
    return spec


def model(kind: str, checkpoint: Path = None) -> torch.nn.Module:
    """모든 구조의 출력 초기값을 같은 공통 이득으로 맞춘다."""
    spec = prepare()
    initial = torch.from_numpy(np.load(GLOBAL)).cuda()
    if kind == "global":
        net = GlobalGainReadout(initial)
    else:
        architecture = "diff" if kind == "head_only" else kind
        base = (
            run.source.factory(288)
            if architecture == "diff"
            else table4.model(architecture, 1)
        )
        state = torch.load(
            ROOT / spec["checkpoints"][architecture],
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"]
        base.load_state_dict(state, strict=True)
        base.gnn.key_scale = torch.nn.Parameter(
            torch.from_numpy(
                np.load(run.source.OUT / "remetric_octave288.npy")
            ).float(),
            requires_grad=False,
        )
        net = SpectralGainReadout(
            base.cuda().eval(), initial, head_only=kind == "head_only"
        )
    if checkpoint is not None:
        net.load_state_dict(
            torch.load(checkpoint, map_location="cuda", weights_only=False)[
                "model_state_dict"
            ],
            strict=True,
        )
    return net.cuda().eval()


def pool(role: str, sequence: str) -> None:
    """모든 구조가 공유할 원 특징 검색의 실제 승자와 시간 대응을 만든다."""
    prepare()
    folder = OUT / "pools" / role / sequence.replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "audit.json").exists():
        assert (
            run.HASH(folder / "pool.npz") == run.READ(folder / "audit.json")["sha256"]
        )
        return
    cache, pos, minimum, travel, bank, nb, nc, paths = run.inputs(role, sequence)
    # 원 특징 변환을 참조 구현에서 꺼내 별도 구현의 불일치를 피한다.
    frozen = prior.previous.prior.gate.MODEL(
        prior.PARENT / "recenter_frozen/epoch0.pth"
    )
    key = run.source.make_keys(frozen, cache["descriptors"], cache["poses"])
    raw = key[:, :288].copy()
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    row, _ = run.retrieve(raw, bank, pos, minimum)
    p = prior.plan(travel, row["query_idx"], row["j_star"])
    eligible = (row["min_past_gt_dist"] <= 5) | (row["min_past_gt_dist"] > 25)
    correct = (row["min_past_gt_dist"] <= 5) & (row["top1_geo_dist"] <= 5)
    np.savez_compressed(
        folder / "pool.npz",
        **row,
        **{f"plan_{k}": v for k, v in p.items()},
        eligible=eligible,
        correct=correct,
    )
    run.WRITE(
        folder / "audit.json",
        dict(
            sequence=sequence,
            role=role,
            rows=len(correct),
            correct=int(correct.sum()),
            eligible=int(eligible.sum()),
            sha256=run.HASH(folder / "pool.npz"),
            inputs={str(p.relative_to(ROOT)): run.HASH(p) for p in paths},
        ),
    )
    print("POOL", role, sequence, flush=True)


def score(
    key: torch.Tensor, z: torch.Tensor, p: dict, strength: float = 0.5
) -> torch.Tensor:
    """스펙트럼 판정을 기존 학습8 척도로 바꿔 반환한다."""
    v = prior.gain_score(key, z, p, strength)
    fit = run.READ(prior.PARENT / "calibration_recenter.json")["spectral"]
    return (v - fit["mean"][0]) * fit["coefficients"][0] / fit["scale"][0] + fit[
        "intercept"
    ]


def train(arm: str) -> None:
    """같은 실제 판정 표본으로 기존 층과 출력층의 필요한 부분을 학습한다."""
    spec = prepare()
    kind, lr = ARMS[arm]
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    if (folder / "training_complete.json").exists():
        return
    assert not list(folder.glob("epoch*.pth"))
    torch.manual_seed(1)
    net = model(kind)
    initial = {n: v.detach().cpu().clone() for n, v in net.state_dict().items()}
    params = [v for v in net.parameters() if v.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    records = []
    hashes = {}
    updates = 0
    tick = time.time()
    for epoch in range(1, 9):
        for ordinal, s in enumerate(spec["train"]):
            name = s.replace("/", "_")
            data = run.ARR(OUT / "pools/train" / name / "pool.npz")
            graph = torch.load(
                prior.PARENT / "pools" / name / "graph.pt",
                map_location="cuda",
                weights_only=False,
            )
            z = prior.coefficients(run.spec_path(s))
            pp = {
                k: torch.from_numpy(data["plan_" + k]).cuda()
                for k in ("query", "candidate", "weight")
            }
            sets = [
                np.flatnonzero(data["eligible"] & (data["correct"] == c))
                for c in (0, 1)
            ]
            sets = [v for v in sets if len(v)]
            rng = np.random.default_rng(45000 + 1000 * epoch + ordinal)
            losses = []
            orders = []
            grads = []
            for step in range(8):
                count = 64 // len(sets)
                chosen = rng.permutation(
                    np.concatenate(
                        [rng.choice(v, count, replace=len(v) < count) for v in sets]
                    )
                )
                orders.append(run.array_hash(chosen))
                ix = torch.from_numpy(chosen).cuda()
                opt.zero_grad(set_to_none=True)
                key = net(graph)
                values = score(key, z, {k: v[ix] for k, v in pp.items()})
                loss = F.binary_cross_entropy_with_logits(
                    values, torch.from_numpy(data["correct"][chosen]).cuda().float()
                )
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(params, 1)
                assert torch.isfinite(grad)
                opt.step()
                updates += 1
                losses.append(float(loss.detach()))
                grads.append(float(grad))
            records.append(
                dict(
                    epoch=epoch,
                    sequence=s,
                    loss=float(np.mean(losses)),
                    grad=float(np.mean(grads)),
                    batch_hashes=orders,
                )
            )
            run.WRITE(
                folder / "progress.json", dict(epoch=epoch, sequence=s, updates=updates)
            )
            del graph, z, pp, key, values, loss
        if epoch in EPOCHS:
            cp = folder / f"epoch{epoch}.pth"
            torch.save(
                dict(
                    model_state_dict=net.state_dict(),
                    epoch=epoch,
                    updates=updates,
                    kind=kind,
                    protocol_sha256=run.HASH(OUT / "protocol.json"),
                ),
                cp,
            )
            hashes[cp.name] = run.HASH(cp)
        print("TRAIN", arm, epoch, updates, round(time.time() - tick, 1), flush=True)
    changed = [
        n for n, v in net.state_dict().items() if not torch.equal(v.cpu(), initial[n])
    ]
    allowed = {n for n, v in net.named_parameters() if v.requires_grad}
    assert set(changed) <= allowed and updates == 512
    run.WRITE(
        folder / "training_complete.json",
        dict(
            status="passed",
            updates=updates,
            records=records,
            checkpoints=hashes,
            changed_parameters=changed,
            trainable_parameters=sum(p.numel() for p in params),
            seconds=time.time() - tick,
        ),
    )


def evaluate(sequence: str) -> None:
    """공유 검색 승자에서 각 구조의 실제 최종 판정을 함께 평가한다."""
    spec = prepare()
    name = sequence.replace("/", "_")
    folder = OUT / "val"
    folder.mkdir(exist_ok=True)
    target = folder / (name + ".json")
    if target.exists():
        return
    cache, _, _, travel, _, _, _, paths = run.inputs("val", sequence)
    data = run.ARR(OUT / "pools/val" / name / "pool.npz")
    row = {
        k: data[k]
        for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")
    }
    z = prior.coefficients(paths[1])
    p = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
    graph = run.make_graph(
        cache["descriptors"], cache["poses"], np.zeros(len(cache["poses"]), np.int64)
    ).cuda()
    conditions = [("initial", "global", None, 0.5), ("uniform", "global", None, 0)] + [
        (f"{a}_{ep}", k, OUT / a / f"epoch{ep}.pth", 0.5)
        for a, (k, _) in ARMS.items()
        for ep in EPOCHS
    ]
    values = {}
    metrics = {}
    prefix = []
    for tag, kind, cp, strength in conditions:
        net = model(kind, cp)
        with torch.no_grad():
            key = net(graph)
            parts = []
            for start in range(0, len(row["query_idx"]), 128):
                pp = {
                    k: torch.from_numpy(v[start : start + 128]).cuda()
                    for k, v in p.items()
                }
                parts.append(score(key, z, pp, strength).cpu().numpy())
            value = np.concatenate(parts).astype(np.float64)
            at = len(row["query_idx"]) // 2
            t = int(row["query_idx"][at])
            end = t + 16
            small = run.make_graph(
                cache["descriptors"][:end],
                cache["poses"][:end],
                np.zeros(end, np.int64),
            ).cuda()
            short = net(small)
            np.testing.assert_allclose(
                short[: t + 6].cpu(), key[: t + 6].cpu(), atol=4e-6, rtol=4e-5
            )
            pp = {
                k: torch.from_numpy(v).cuda()
                for k, v in prior.plan(
                    travel[:end],
                    row["query_idx"][at : at + 1],
                    row["j_star"][at : at + 1],
                ).items()
            }
            check = score(short, z[:end], pp, strength).cpu().numpy()
            np.testing.assert_allclose(check, value[at : at + 1], atol=8e-6, rtol=0)
        values[tag] = value
        metrics[tag] = run.base.independent_metrics(row, value)
        prefix.append(dict(condition=tag, query=t))
        del net, key, short, small
    ledger = target.with_suffix(".npz")
    np.savez_compressed(ledger, **row, **values)
    run.WRITE(
        target,
        dict(
            status="passed",
            sequence=sequence,
            metrics=metrics,
            prefix_checks=prefix,
            ledger_sha256=run.HASH(ledger),
            test_used=False,
        ),
    )
    print("VAL", sequence, flush=True)


def finish() -> None:
    """각 대조의 동일 예산 최적값을 넘어서는 DiffAttnConv 후보를 고른다."""
    spec = prepare()
    rec = {
        s: run.READ(OUT / "val" / (s.replace("/", "_") + ".json")) for s in spec["val"]
    }
    tags = next(iter(rec.values()))["metrics"]
    results = {
        k: run.base.macro({s: r["metrics"][k] for s, r in rec.items()}) for k in tags
    }
    best = {
        kind: max(
            [f"{a}_{ep}" for a, (k, _) in ARMS.items() if k == kind for ep in EPOCHS],
            key=lambda tag: results[tag]["macro"]["AP"],
        )
        for kind in KINDS
    }
    candidate = best["diff"]
    m = results[candidate]["macro"]
    controls = ["initial"] + [best[k] for k in KINDS if k != "diff"]
    checks = {f"AP_{c}": m["AP"] >= results[c]["macro"]["AP"] + 0.001 for c in controls}
    checks.update(
        {
            f"{metric}_{c}": m[metric] >= results[c]["macro"][metric] - tol
            for c in controls
            for metric, tol in (("max_F1", 0.003), ("R@100P", 0.01))
        }
    )
    bev = run.READ(prior.previous.prior.gate.previous.OUT / "bev_validation.json")[
        "temporal"
    ]["macro"]
    checks.update(
        {
            f"bev_{k}": m[k] >= bev[k] - tol
            for k, tol in (
                ("AP", 0.02),
                ("max_F1", 0.02),
                ("R@100P", 0.03),
                ("R@1", 0.03),
            )
        }
    )
    selected = candidate if all(checks.values()) else None
    run.WRITE(
        OUT / "summary.json",
        dict(
            results=results,
            best=best,
            checks=checks,
            selected=selected,
            bev_temporal=bev,
            test_used=False,
        ),
    )
    print(
        "FINISHED",
        selected,
        {k: results[v]["macro"] for k, v in best.items()},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "pool", "train", "val", "finish"))
    parser.add_argument("--role", default="train")
    parser.add_argument("--sequence", default="kitti/06")
    parser.add_argument("--arm", default="diff_lr4")
    args = parser.parse_args()
    run.configure()
    if args.action == "pool":
        pool(args.role, args.sequence)
    elif args.action == "train":
        train(args.arm)
    elif args.action == "val":
        evaluate(args.sequence)
    else:
        globals()[args.action]()
