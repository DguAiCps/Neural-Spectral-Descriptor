#!/usr/bin/env python3
"""DiffAttnConv 갱신 범위와 기존 검색 마진 보호를 같은 예산에서 대조한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import _pdv2_nonrevisit_learning as run
from retrieval.teacher_margin import teacher_margin_loss, triplet_margin

PRIOR = run.OUT
BALANCED = ROOT / "results/pdv2_nonrevisit_balanced_20261104"
OUT = ROOT / "results/pdv2_diffattn_margin_20261104"
ARMS = ("full_control", "conv_only", "conv_guard", "full_guard")


def prepare() -> dict:
    """새 검증 전에 비교 범위·손실·선택 규칙을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = run.READ(target)
        sources = dict(spec["sources"])
        amendments = OUT / "runtime_amendments.json"
        if amendments.exists():
            for change in run.READ(amendments)["changes"]:
                assert sources[change["path"]] == change["before"]
                sources[change["path"]] = change["after"]
        for name, digest in sources.items():
            assert run.HASH(ROOT / name) == digest, name
        return spec
    spec = run.READ(PRIOR / "protocol.json")
    paths = [ROOT / p for p in spec["sources"]]
    paths += [Path(__file__), ROOT / "src/retrieval/teacher_margin.py",
              ROOT / "tests/retrieval/test_teacher_margin.py",
              PRIOR / "protocol.json", PRIOR / "runtime_amendments.json",
              PRIOR / "calibration.json", BALANCED / "protocol.json",
              BALANCED / "positive_only/epoch4.pth"]
    gain = run.READ(BALANCED / "protocol.json")["acceptance_gain"]
    spec.update(created_unix=time.time(), arms=ARMS, epochs=[4], lambdas=[0., .5],
        acceptance_gain=gain, guard_gain=10., guard_temperature=.1,
        objective="재방문 질의 수락 BCE와 InfoNCE. full_control은 앞 보정 양성 전용 학습 재현. conv_only는 gnn.convs.* 2층의 Q/K/V·기하 bias MLP만 갱신. conv_guard는 여기에 초기 416D 키의 양수 삼중쌍 마진 감소 벌점 추가. full_guard는 전체 갱신에 같은 벌점. 손실은 InfoNCE+12.685962594044891*BCE+10*guard. guard는 relu((초기마진-현재마진)/.1)^2, 초기마진>0만 적용하고 같은 전체 삼중쌍 배치 분모를 사용.",
        controls="같은 초기 모델, 학습8, 배치순서, 학습률, 256회, 마지막 epoch4만 비교. 모든 조건의 문맥 혼합 .5 고정. 계수0은 검색 경로 진단용이며 선택하지 않음. guard 계수10은 새 검증 전에 고정한 탐색 설정. 실제 융합 순위 또는 미관측 질의 보호를 보장하지 않음.",
        selection="일차 비교는 같은 .5 판정의 재학습 없는 모델 대비 DiffAttnConv 개선. AP+.1%p 이상이며 F1/R100/R1이 모두 낮아지지 않는 조건을 후속 후보로 삼고 그중 AP 최대. 기존 positive_only 보정과 새 full_control보다 나아지는지도 별도 보고. 비재방문 추가의 이득·표준 attention 우월은 이번 진행의 전제 아님. 반복 검증 단일 초기 시드의 탐색이며 독립 확증 아님. 시험·원고·그림 변경 없음.",
        sources={str(p.relative_to(ROOT)): run.HASH(p) for p in paths})
    run.WRITE(target, spec)
    (OUT / "pools").symlink_to(PRIOR / "pools", target_is_directory=True)
    fit = run.READ(PRIOR / "calibration.json")
    fit.update(protocol_sha256=run.HASH(target), reused_from=str(PRIOR.relative_to(ROOT)))
    run.WRITE(OUT / "calibration.json", fit)
    for seq in spec["train"]:
        folder = OUT / "pools" / seq.replace("/", "_")
        audit = run.READ(folder / "audit.json")
        for name, digest in audit["outputs"].items():
            assert run.HASH(folder / name) == digest
    return spec


def activate() -> dict:
    """독립 프로세스에서 검증 엔진에 이번 고정 설정을 공급한다."""
    spec = prepare()
    run.OUT = OUT
    run.prepare = prepare
    run.EPOCHS = (4,)
    run.LAMBDAS = (0., .5)
    return spec


def train(arm: str) -> None:
    """고정 검색 재계량과 정규화 통계 아래 같은 표본으로 추가 학습한다."""
    spec = activate()
    fit = run.calibration()
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    if (folder / "training_complete.json").exists():
        return
    assert not list(folder.glob("epoch*.pth")), "기존 부분 학습을 덮어쓰지 않는다"
    torch.manual_seed(1)
    net = run.model()
    conv_only = arm.startswith("conv_")
    if conv_only:
        for name, parameter in net.named_parameters():
            parameter.requires_grad_(name.startswith("gnn.convs."))
    initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    trainable = {k for k, v in net.named_parameters() if v.requires_grad}
    if conv_only:
        assert len(trainable) == 14
    teacher = run.model()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    targets = {}
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                 lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    contrastive = run.InfoNCELoss(temperature=.1)
    records, updates = [], 0
    started = time.time()
    for epoch in range(1, 5):
        for ordinal, seq in enumerate(spec["train"]):
            pooldir = OUT / "pools" / seq.replace("/", "_")
            graph = torch.load(pooldir / "graph.pt", map_location="cuda", weights_only=False)
            pool = run.ARR(pooldir / "pool.npz")
            mask = pool["eligible"]
            tensors = {k: torch.from_numpy(pool[k][mask]).cuda() for k in ("positive", "correct")}
            plan = {k: torch.from_numpy(pool["plan_"+k]).cuda() for k in ("query", "candidate", "weight")}
            logit = torch.from_numpy(run.predict_score(pool["spectral"][mask, None], fit["spectral"])).float().cuda()
            triplets = torch.from_numpy(pool["triplets"]).cuda()
            if seq not in targets:
                with torch.no_grad():
                    targets[seq] = triplet_margin(F.normalize(teacher(graph), dim=1), triplets)
            rng = np.random.default_rng(1000*epoch+ordinal)
            terms, hashes = [], []
            for step in range(spec["steps_per_sequence"]):
                chosen = rng.choice(len(logit), spec["batch_queries"], replace=len(logit)<spec["batch_queries"])
                tr = rng.choice(len(triplets), spec["retrieval_batch"], replace=len(triplets)<spec["retrieval_batch"]) if len(triplets) else np.empty(0, np.int64)
                hashes.append(run.array_hash(np.r_[chosen, tr]))
                rows, ti = torch.from_numpy(chosen).cuda(), torch.from_numpy(tr).cuda()
                optimizer.zero_grad(set_to_none=True)
                embedding = F.normalize(net(graph), dim=1)
                ids = triplets[ti]
                ranking = contrastive(embedding[ids[:, 0]], embedding[ids[:, 1]], embedding[ids[:, 2]]) if len(tr) else embedding.sum()*0.
                ctx = run.context_score(embedding, {k: v[rows] for k, v in plan.items()})
                logits = logit[rows] + (ctx-fit["context_mean"])/fit["context_scale"]
                bce = run.acceptance_loss(logits, tensors["correct"][rows], tensors["positive"][rows], False)
                guard = teacher_margin_loss(triplet_margin(embedding, ids), targets[seq][ti], spec["guard_temperature"])
                loss = ranking + spec["acceptance_gain"] * bce
                if arm.endswith("guard"):
                    loss = loss + spec["guard_gain"] * guard
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(net.parameters(), spec["clip"])
                assert torch.isfinite(grad)
                optimizer.step()
                updates += 1
                terms.append([float(v.detach()) for v in (loss, ranking, bce, guard, grad)])
            assert embedding.shape[1] == 416
            record = dict(epoch=epoch, sequence=seq, updates=updates,
                          loss_rank_bce_guard_grad=np.mean(terms, axis=0).tolist(), batch_hashes=hashes)
            records.append(record)
            run.WRITE(folder / "progress.json", record)
            print("TRAIN", arm, epoch, seq, updates, record["loss_rank_bce_guard_grad"], flush=True)
            del graph, embedding, loss, ranking, ctx, logits, bce, guard, plan, tensors, logit, triplets
            torch.cuda.empty_cache()
    changed = []
    for name, value in net.state_dict().items():
        same = torch.equal(value.detach().cpu(), initial[name])
        if name not in trainable:
            assert same, name
        elif not same:
            changed.append(name)
    assert changed and (not conv_only or all(k.startswith("gnn.convs.") for k in changed))
    checkpoint = folder / "epoch4.pth"
    torch.save(dict(model_state_dict=net.state_dict(), epoch=4, updates=updates,
                    protocol_sha256=run.HASH(OUT / "protocol.json")), checkpoint)
    reproduction = None
    if arm == "full_control":
        prior = torch.load(BALANCED / "positive_only/epoch4.pth", map_location="cpu", weights_only=False)["model_state_dict"]
        reproduction = max(float((value.detach().cpu()-prior[k]).abs().max()) for k, value in net.state_dict().items())
        # 동일 학습 절차라도 가중치의 비트 단위 재현을 가정하지 않는다.
        # 실제 차이를 저장하고 이번 대조도 검색부터 별도로 평가한다.
        assert np.isfinite(reproduction)
    run.WRITE(folder / "training_complete.json", dict(arm=arm, updates=updates, records=records,
        changed_parameters=changed, trainable_parameters=sorted(trainable),
        parameter_count=sum(p.numel() for p in net.parameters() if p.requires_grad),
        frozen_state_exact=True, prior_control_max_parameter_error=reproduction,
        checkpoints={checkpoint.name: run.HASH(checkpoint)}, seconds=time.time()-started,
        test_used=False))


def finish() -> None:
    """최종 원장을 독립 재계산하고 고정 혼합의 학습 전후를 비교한다."""
    spec = activate()
    results, diagnostics, ledgers = {}, {}, 0
    roots = {"frozen": PRIOR / "initial/epoch0", **{a: OUT / a / "epoch4" for a in ARMS}}
    for arm, root in roots.items():
        all_metrics = {label: {} for label in ("lambda_0", "lambda_0.5")}
        for seq in spec["val"]:
            path = root / "val" / (seq.replace("/", "_")+".json")
            record, row = run.READ(path), run.ARR(path.with_suffix(".npz"))
            assert record["status"] == "passed" and len(record["prefix_checks"]) == 2
            assert run.HASH(path.with_suffix(".npz")) == record["ledger_sha256"]
            if arm != "frozen":
                assert record["checkpoint_sha256"] == run.HASH(OUT / arm / "epoch4.pth")
            for label in all_metrics:
                metric = run.base.independent_metrics(row, row[label])
                assert metric == record["metrics"][label]
                all_metrics[label][seq] = metric
            ledgers += 1
        results[arm] = dict(**run.base.macro(all_metrics["lambda_0.5"]), sequences=all_metrics["lambda_0.5"])
        diagnostics[arm] = dict(**run.base.macro(all_metrics["lambda_0"]), sequences=all_metrics["lambda_0"])
    train = {a: run.READ(OUT / a / "training_complete.json") for a in ARMS}
    assert {r["updates"] for r in train.values()} == {256}
    orders = [[r["batch_hashes"] for r in t["records"]] for t in train.values()]
    assert all(v == orders[0] for v in orders)
    previous_order = [r["batch_hashes"] for r in run.READ(BALANCED / "positive_only/training_complete.json")["records"]]
    assert orders[0] == previous_order
    changes, checks = {}, {}
    for arm in ARMS:
        changes[arm] = {ref: {m: 100*(results[arm]["macro"][m]-results[ref]["macro"][m]) for m in run.base.METRICS}
                        for ref in ("frozen", "full_control")}
        d = changes[arm]["frozen"]
        checks[arm] = d["AP"] >= .1 and all(d[m] >= 0 for m in ("max_F1", "R@100P", "R@1"))
    eligible = [a for a in ARMS if checks[a]]
    selected = max(eligible, key=lambda a: results[a]["macro"]["AP"]) if eligible else None
    summary = dict(results=results, key_path_only=diagnostics, delta_pp=changes,
                   advancement_checks=checks, selected=selected, fixed_coefficient=.5,
                   equal_updates_and_samples=True, test_used=False,
                   protocol_sha256=run.HASH(OUT / "protocol.json"))
    run.WRITE(OUT / "summary.json", summary)
    run.WRITE(OUT / "completion_validation.json", dict(status="passed", ledgers=ledgers,
        metric_rows=ledgers*2, prefix_checks=ledgers*2, checkpoints=4,
        frozen_parameter_audit=True, test_used=False))
    run.WRITE(OUT / "status.json", dict(status="complete", selected=selected, test_used=False))
    print("RESULT", {k: v["macro"] for k, v in results.items()}, "SELECTED", selected, flush=True)


def campaign() -> None:
    """학습 후 2개 검증 작업만 병렬 실행하며 진행 상태를 기록한다."""
    spec = prepare()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    def call(args: list, name: str) -> None:
        with (logs / (name+".log")).open("w") as log:
            subprocess.run([sys.executable, __file__, *args], stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        for arm in ARMS:
            run.WRITE(OUT / "status.json", dict(status="running", phase="training", arm=arm))
            call(["train", "--arm", arm], "train_"+arm)
        jobs = [(a, s) for a in ARMS for s in spec["val"]]
        def evaluate(job: tuple) -> None:
            arm, seq = job
            call(["val", "--arm", arm, "--sequence", seq], "val_"+arm+"_"+seq.replace("/", "_"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for i, future in enumerate(concurrent.futures.as_completed([pool.submit(evaluate, j) for j in jobs]), 1):
                future.result()
                run.WRITE(OUT / "status.json", dict(status="running", phase="validation", done=i, total=len(jobs)))
                print("VAL_PROGRESS", i, len(jobs), flush=True)
        finish()
    except Exception as exc:
        run.WRITE(OUT / "status.json", dict(status="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "val", "finish", "run"))
    parser.add_argument("--arm", choices=ARMS, default="conv_guard")
    parser.add_argument("--sequence", default="kitti/09")
    args = parser.parse_args()
    run.configure()
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        train(args.arm)
    elif args.action == "val":
        activate()
        run.validate(args.arm, 4, args.sequence)
    elif args.action == "finish":
        finish()
    else:
        campaign()
