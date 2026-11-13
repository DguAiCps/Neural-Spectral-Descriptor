#!/usr/bin/env python3
"""복소 차이 전달의 주의·세기 학습을 같은 최종 판정에서 대조한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

import _pdv2_diffattn_complex_difference as prior
from gnn.learned_spectral_difference import LearnedSpectralDifference

run, ROOT = prior.run, prior.ROOT
OUT = ROOT / "results/pdv2_diffattn_complex_learn_20261104"
KINDS = ("diff", "abs_k", "abs_kv", "point_gate", "gate_only")
ARMS = {
    f"{kind}_{tag}": (kind, lr)
    for kind in KINDS
    for tag, lr in (("lr5", 1e-5), ("lr4", 1e-4))
}
EPOCHS = (4, 8)


def configure() -> None:
    """GPU 반복 차이를 줄이도록 결정적 연산을 필수로 사용한다."""
    run.configure()
    torch.use_deterministic_algorithms(True)


def prepare() -> dict:
    """모델·학습·평가 조건과 비교 기준을 학습 전에 고정한다."""
    OUT.mkdir(exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = run.READ(target)
        for p, h in spec["sources"].items():
            assert run.HASH(ROOT / p) == h, p
    else:
        parent = prior.prepare()
        paths = [
            Path(__file__),
            Path(prior.__file__),
            ROOT / "src/gnn/learned_spectral_difference.py",
            ROOT / "src/gnn/spectral_difference.py",
            prior.OUT / "summary.json",
            prior.OUT / "protocol.json",
        ]
        paths += [
            ROOT / p
            for p in run.READ(prior.PARENT / "protocol.json")["checkpoints"].values()
        ]
        spec = dict(
            created_unix=time.time(),
            train=parent["train"],
            val=parent["val"],
            sources={str(p.relative_to(ROOT)): run.HASH(p) for p in paths},
            arms=ARMS,
            epochs=EPOCHS,
            architecture="기존128숨김·두층·네머리·기하결합·게이트 재사용. 출력층 제거. 두 스펙트럼 차이 전달에 각 층 attention 연결. 머리별2주파수. 첫 세기는점별h0,둘째h2이며.5*sigmoid로0~.5. 기존q+15범위유지.",
            initialization="동일 pretrained Diff20epoch 가중치에서 K/V 차이 여부만 교체. Q와기하MLP최종층0,게이트최종층0으로모두균등attention·세기.25. point_gate는자체기존20epoch점별투영이나동일최종초기점수. 사전학습이Diff에유리할수있음. gate_only는Diff기반게이트만갱신.",
            training="학습8의공통원키실제승자,정답/오답균등64,시퀀스당8step*8epoch=512,AdamW감쇠1e-5,clip1,lr1e-5/1e-4,기존학습8BCE척도,모든사용파라미터학습(BN누적통계고정). 배치순서는출력대조와동일. 결정적CUDA필수.",
            validation="검증9최종AP/최대F1/R100/R1. 원키검색승자·60각도·75:25·양방향시간판정고정. 같은복소차이뒤진폭제곱근/상관보정. 원본/동일균등시작점/절대K/절대KV/점별게이트/게이트만대조. 시험사용없음.",
            selection="Diff최적AP가모든대조의최적및전달없음·균등전달각조건보다+.1%p,F1손실<=.3%p,R100<=1%p면매칭후보. 후보검색R1차이가BEV보다3.065%p로고정되어있으므로검색복구·3시드·설정고정후시험전까지목표완료주장없음. 반복검증탐색.",
            verification="두지점q+15실제접두최종점수atol8e-6및계수4e-6. 두독립지표구현. 시작출력일치와실제기울기, 동일조건반복학습검사.",
        )
        run.WRITE(target, spec)
    run.OUT = OUT
    return spec


def model(kind: str, checkpoint: Path = None) -> torch.nn.Module:
    """같은 최종 초기 점수에서 기존 부분을 재사용한다."""
    torch.manual_seed(1)
    if kind == "point_gate":
        ref = prior.base.table4.model("graph_free", 1)
        name = "graph_free"
    else:
        ref = run.source.factory(288)
        name = "diff"
    paths = run.READ(prior.PARENT / "protocol.json")["checkpoints"]
    ref.load_state_dict(
        torch.load(ROOT / paths[name], map_location="cpu", weights_only=False)[
            "model_state_dict"
        ],
        strict=True,
    )
    net = LearnedSpectralDifference(ref, kind).cuda().eval()
    if checkpoint is not None:
        net.load_state_dict(
            torch.load(checkpoint, map_location="cuda", weights_only=False)[
                "model_state_dict"
            ],
            strict=True,
        )
    return net


def prepare_graphs() -> None:
    """학습8만의 복소 입력·자세 축과 그래프를 저장한다."""
    spec = prepare()
    folder = OUT / "train_inputs"
    folder.mkdir(exist_ok=True)
    for s in spec["train"]:
        path = folder / (s.replace("/", "_") + ".pt")
        if path.exists():
            assert run.HASH(path) == run.READ(path.with_suffix(".json"))["graph_sha256"]
            continue
        graph, _, audit = prior.graph_inputs("train", s)
        original = torch.load(
            prior.base.prior.PARENT / "pools" / s.replace("/", "_") / "graph.pt",
            map_location="cpu",
            weights_only=False,
        )
        for k in ("x", "edge_index", "edge_attr", "edge_type"):
            torch.testing.assert_close(
                getattr(graph, k).cpu(), getattr(original, k), rtol=0, atol=0
            )
        torch.save(graph.cpu(), path)
        run.WRITE(
            path.with_suffix(".json"), dict(audit=audit, graph_sha256=run.HASH(path))
        )


def preflight() -> None:
    """실제 학습 입력에서 공통 출발점·미분·결정성을 검사한다."""
    spec = prepare()
    prepare_graphs()
    s = spec["train"][0]
    graph = torch.load(
        OUT / "train_inputs" / (s.replace("/", "_") + ".pt"),
        map_location="cuda",
        weights_only=False,
    )
    data = run.ARR(prior.PARENT / "pools/train" / s.replace("/", "_") / "pool.npz")
    pp = {
        k: torch.from_numpy(data["plan_" + k][:32]).cuda()
        for k in ("query", "candidate", "weight")
    }
    expected = prior.filtered(graph, 0.25)
    checks = {}
    for kind in KINDS:
        net = model(kind)
        z = net(graph)
        error = float((z - expected).abs().max())
        assert error < 4e-6, (kind, error)
        repeat = net(graph)
        assert torch.equal(z, repeat), kind
        score = prior.score(z, pp)
        loss = F.binary_cross_entropy_with_logits(
            score, torch.from_numpy(data["correct"][:32]).cuda().float()
        )
        loss.backward()
        gradients = {
            n: float(p.grad.norm())
            for n, p in net.named_parameters()
            if p.grad is not None
        }
        assert all(np.isfinite(v) for v in gradients.values())
        assert sum(gradients.values()) > 0
        if kind in ("diff", "abs_k", "abs_kv"):
            assert sum(v for n, v in gradients.items() if "W_q" in n) > 0
        checks[kind] = dict(
            initial_error=error,
            gradients=gradients,
            trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
        )
        del net, z, repeat, score, loss
    run.WRITE(
        OUT / "preflight.json", dict(status="passed", checks=checks, deterministic=True)
    )
    print("PREFLIGHT PASSED", flush=True)


def train(arm: str, repeat: bool = False) -> None:
    """같은 실제 판정 표본으로 512회 학습한다."""
    spec = prepare()
    assert run.READ(OUT / "preflight.json")["status"] == "passed"
    kind, lr = ARMS[arm]
    folder = OUT / (arm + ("_repeat" if repeat else ""))
    folder.mkdir(exist_ok=True)
    if (folder / "training_complete.json").exists():
        return
    assert not list(folder.glob("epoch*.pth"))
    net = model(kind)
    initial = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    params = [v for v in net.parameters() if v.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    records, hashes = [], {}
    updates = 0
    tick = time.time()
    for epoch in range(1, 9):
        for ordinal, s in enumerate(spec["train"]):
            name = s.replace("/", "_")
            graph = torch.load(
                OUT / "train_inputs" / (name + ".pt"),
                map_location="cuda",
                weights_only=False,
            )
            data = run.ARR(prior.PARENT / "pools/train" / name / "pool.npz")
            plan = {
                k: torch.from_numpy(data["plan_" + k]).cuda()
                for k in ("query", "candidate", "weight")
            }
            sets = [
                np.flatnonzero(data["eligible"] & (data["correct"] == c))
                for c in (0, 1)
            ]
            sets = [v for v in sets if len(v)]
            rng = np.random.default_rng(45000 + 1000 * epoch + ordinal)
            orders, losses, grads = [], [], []
            for _ in range(8):
                n = 64 // len(sets)
                chosen = rng.permutation(
                    np.concatenate([rng.choice(v, n, replace=len(v) < n) for v in sets])
                )
                orders.append(run.array_hash(chosen))
                ix = torch.from_numpy(chosen).cuda()
                opt.zero_grad(set_to_none=True)
                z = net(graph)
                value = prior.score(z, {k: v[ix] for k, v in plan.items()})
                loss = F.binary_cross_entropy_with_logits(
                    value, torch.from_numpy(data["correct"][chosen]).cuda().float()
                )
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
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
                    strength_mean=float(net.last_strengths.detach().mean()),
                )
            )
            run.WRITE(
                folder / "progress.json", dict(epoch=epoch, sequence=s, updates=updates)
            )
            del graph, z, value, loss, plan
        if epoch in EPOCHS:
            path = folder / f"epoch{epoch}.pth"
            torch.save(
                dict(
                    model_state_dict=net.state_dict(),
                    epoch=epoch,
                    updates=updates,
                    kind=kind,
                    protocol_sha256=run.HASH(OUT / "protocol.json"),
                ),
                path,
            )
            hashes[path.name] = run.HASH(path)
        print("TRAIN", arm, epoch, updates, round(time.time() - tick, 1), flush=True)
    changed = [
        n for n, v in net.state_dict().items() if not torch.equal(v.cpu(), initial[n])
    ]
    assert set(changed) <= {n for n, p in net.named_parameters() if p.requires_grad}
    run.WRITE(
        folder / "training_complete.json",
        dict(
            status="passed",
            updates=updates,
            records=records,
            checkpoints=hashes,
            changed_parameters=changed,
            seconds=time.time() - tick,
            trainable_parameters=sum(p.numel() for p in params),
        ),
    )


def evaluate(sequence: str) -> None:
    """학습 모델을 최종 판정하고 실제 접두부와 비교한다."""
    prepare()
    name = sequence.replace("/", "_")
    folder = OUT / "val"
    folder.mkdir(exist_ok=True)
    target = folder / (name + ".json")
    if target.exists():
        return
    graph, travel, audit = prior.graph_inputs("val", sequence)
    data = run.ARR(prior.PARENT / "pools/val" / name / "pool.npz")
    row = {
        k: data[k]
        for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")
    }
    plan = {k: data["plan_" + k] for k in ("query", "candidate", "weight")}
    # 앞 고정 평균 원장을 그대로 보존하고 학습 조건만 추가한다.
    old = run.ARR(prior.OUT / "val" / (name + ".npz"))
    values = {k: old[k] for k, _, _ in prior.CASES}
    metrics = {k: run.base.independent_metrics(row, v) for k, v in values.items()}
    checks, strengths = [], {}
    with torch.no_grad():
        for arm, (kind, _) in ARMS.items():
            for epoch in EPOCHS:
                tag = f"{arm}_{epoch}"
                net = model(kind, OUT / arm / f"epoch{epoch}.pth")
                z = net(graph)
                strengths[tag] = np.quantile(
                    net.last_strengths.cpu(), [0.0, 0.1, 0.5, 0.9, 1.0]
                ).tolist()
                parts = []
                for start in range(0, len(row["query_idx"]), 128):
                    pp = {
                        k: torch.from_numpy(v[start : start + 128]).cuda()
                        for k, v in plan.items()
                    }
                    parts.append(prior.score(z, pp).cpu().numpy())
                value = np.concatenate(parts).astype(np.float64)
                for at in (len(value) // 3, 2 * len(value) // 3):
                    t = int(row["query_idx"][at])
                    small, short_travel, _ = prior.graph_inputs("val", sequence, t + 16)
                    short = net(small)
                    error = float((short[: t + 6] - z[: t + 6]).abs().max())
                    assert error < 4e-6, (tag, error)
                    pp = {
                        k: torch.from_numpy(v).cuda()
                        for k, v in prior.base.prior.plan(
                            short_travel,
                            row["query_idx"][at : at + 1],
                            row["j_star"][at : at + 1],
                        ).items()
                    }
                    for k, v in pp.items():
                        np.testing.assert_array_equal(v.cpu(), plan[k][at : at + 1])
                    check = prior.score(short, pp).cpu().numpy()
                    np.testing.assert_allclose(
                        check, value[at : at + 1], atol=8e-6, rtol=0
                    )
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
                other = run.white.prior.BASE.exp.metrics(value, row)
                for m in run.base.METRICS:
                    if metrics[tag][m] is not None:
                        np.testing.assert_allclose(
                            other[m], metrics[tag][m], rtol=0, atol=1e-12
                        )
                del net, z
    path = target.with_suffix(".npz")
    np.savez_compressed(path, **row, **values)
    run.WRITE(
        target,
        dict(
            status="passed",
            metrics=metrics,
            strengths=strengths,
            prefix_checks=checks,
            audit=audit,
            ledger_sha256=run.HASH(path),
            test_used=False,
        ),
    )
    print("VAL", sequence, flush=True)


def finish() -> None:
    """학습 효과를 모든 대조와 비교한다."""
    spec = prepare()
    rows = {
        s: run.READ(OUT / "val" / (s.replace("/", "_") + ".json")) for s in spec["val"]
    }
    results = {
        k: run.base.macro({s: v["metrics"][k] for s, v in rows.items()})
        for k in next(iter(rows.values()))["metrics"]
    }
    best = {
        kind: max(
            [f"{a}_{ep}" for a, (k, _) in ARMS.items() if k == kind for ep in EPOCHS],
            key=lambda t: results[t]["macro"]["AP"],
        )
        for kind in KINDS
    }
    controls = [k for k, _, _ in prior.CASES] + [
        v for k, v in best.items() if k != "diff"
    ]
    m = results[best["diff"]]["macro"]
    checks = {
        f"{c}_{metric}": m[metric] >= results[c]["macro"][metric] + delta
        for c in controls
        for metric, delta in (("AP", 0.001), ("max_F1", -0.003), ("R@100P", -0.01))
    }
    selected = best["diff"] if all(checks.values()) else None
    run.WRITE(
        OUT / "summary.json",
        dict(
            results=results,
            best=best,
            checks=checks,
            selected_matching_candidate=selected,
            comparable_not_confirmed=True,
            test_used=False,
        ),
    )
    print(
        "FINISHED",
        selected,
        {k: results[v]["macro"] for k, v in best.items()},
        flush=True,
    )


def campaign() -> None:
    """모든 학습·대조·검증을 완료한다."""
    spec = prepare()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)

    def call(args, tag):
        with (logs / (tag + ".log")).open("w") as stream:
            subprocess.run(
                [sys.executable, __file__, *args],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    if not (OUT / "preflight.json").exists():
        preflight()
    for i, arm in enumerate(ARMS, 1):
        call(["train", "--arm", arm], "train_" + arm)
        run.WRITE(OUT / "status.json", dict(phase="training", done=i, total=len(ARMS)))
        print("TRAINED", arm, flush=True)
    call(["train", "--arm", "diff_lr4", "--repeat"], "repeat_diff_lr4")
    for ep in EPOCHS:
        original = torch.load(
            OUT / "diff_lr4" / f"epoch{ep}.pth", map_location="cpu", weights_only=False
        )["model_state_dict"]
        repeat = torch.load(
            OUT / "diff_lr4_repeat" / f"epoch{ep}.pth",
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"]
        assert all(torch.equal(v, repeat[k]) for k, v in original.items()), ep
    run.WRITE(
        OUT / "repetition.json",
        dict(status="passed", exact_weights=True, epochs=EPOCHS),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(call, ["val", "--sequence", s], "val_" + s.replace("/", "_"))
            for s in spec["val"]
        ]
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
    parser.add_argument(
        "action", choices=("prepare", "preflight", "train", "val", "finish", "run")
    )
    parser.add_argument("--arm", default="diff_lr4")
    parser.add_argument("--sequence", default="kitti/09")
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    configure()
    if args.action == "train":
        train(args.arm, args.repeat)
    elif args.action == "val":
        evaluate(args.sequence)
    elif args.action == "run":
        campaign()
    else:
        globals()[args.action]()
