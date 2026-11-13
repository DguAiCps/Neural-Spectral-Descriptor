#!/usr/bin/env python3
"""게이트의 최종 판정 경로와 포화를 분리하는 학습·검증 대조."""
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
import _pdv2_diffattn_comparable as previous
from retrieval.gated_context import gated_context_score

run = previous.run
OUT = ROOT / "results/pdv2_diffattn_gate_learning_20261104"
MODEL, LOSS = run.model, run.acceptance_loss
ARMS = ("native_all", "recenter_all", "recenter_convs", "recenter_gate")
FROZEN = ("native_frozen", "recenter_frozen")
EPOCHS = (4, 8)


def prepare() -> dict:
    """새 학습 전에 대조·학습 예산·검증 선택을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = run.READ(target)
        for name, digest in spec["sources"].items():
            assert run.HASH(ROOT / name) == digest, name
        return spec
    spec = run.READ(previous.OUT / "protocol.json")
    paths = [ROOT / name for name in spec["sources"]]
    paths += [Path(__file__), ROOT / "src/retrieval/gated_context.py",
              ROOT / "tests/retrieval/test_gated_context.py",
              previous.OUT / "three_seed/summary.json", previous.OUT / "gate_gradient_audit.json"]
    spec.update(created_unix=time.time(), arms=ARMS, frozen=FROZEN, epochs=EPOCHS,
        objective="게이트 보존 최종 문맥 점수. 기존 전체 문맥 학습 / 학습8로 게이트 포화를 완화한 전체 문맥 / 같은 시작에서 DiffAttnConv 두 층만 / 게이트만 학습. 각512갱신,4·8epoch 확인. 기존 InfoNCE+BCE비중12.685962594044891, 학습 문맥계수1, BN·재계량 고정.",
        gate_recenter="초기 모델 gate 마지막 선형층의 logit을 학습8 시퀀스 균등 평균·표준편차로 중심화·척도화한다(표준편차 하한1). 기존 마지막 가중치·절편에 흡수하므로 모듈·매개변수 증가 없음. 검증/시험별 재적합 없음.",
        acceptance="normalize([raw,alpha*context])에서 문맥 블록/원 특징 노름으로 alpha*unit_context를 복원하고 기존 경로 가중 내적·방향max를 사용. 초기 학습8의 같은 고정 승자에서 조건별 문맥 평균/척도 적합. 스펙트럼 affine 점수 적합은 공통. 모든 학습 및 동결 대조에 동일5개 추론 비중 제공.",
        controls="모든 조건은 같은 초기 시드1·같은 원래 학습 승자·삼중쌍·배치 순서. 포화 완화 전후 각각 동결 대조. 게이트만 학습과 conv만 학습으로 기여 범위를 구분. 평가 검색은 각 실제 키로 모두 다시 실행.",
        selection="검증9에서 AP 최대. 동률 작은 문맥비중→이른epoch→이름. 시험 확장에는 학습 전 두 조건 및 기존 정규화 문맥의 검증 최적 AP보다+.1%p, 같은 시작/비중보다+.1%p, BEV 동일시간 판정 대비 AP/F1 -2%p 이내·R100/R1 -3%p 이내. conv 고유 기여는 conv만 학습의 양의 변화 또는 전체학습이 gate만학습을 넘는지 별도 확인. 이는 작업용 기준이며 통계적 동등성 아님. 통과 후보만 모델/비중을 고정한 뒤 시험; 원고 자동 수정 없음.",
        limitations="반복 사용 검증과 과거 시험 개발 노출이 있는 탐색. 보유 학습 승자는 고정. 전체학습은 투영층·기하인코더도 포함하므로 이를 conv만의 효과로 부르지 않음.",
        sources={str(p.relative_to(ROOT)):run.HASH(p) for p in paths})
    run.WRITE(target, spec)
    (OUT / "pools").symlink_to(previous.PRIOR / "pools", target_is_directory=True)
    return spec


def mode(arm: str) -> str:
    """조건이 사용하는 초기 게이트를 반환한다."""
    return "native" if arm.startswith("native") else "recenter"


def load_model(checkpoint: Path = None, arm: str = "native_all") -> torch.nn.Module:
    """기존 모델을 그대로 불러오고 초기 게이트 또는 학습 범위만 설정한다."""
    net = MODEL(checkpoint)
    if checkpoint is None and mode(arm) == "recenter":
        stats = run.READ(OUT / "gate_statistics.json")
        with torch.no_grad():
            net.gnn.gate[-1].weight.div_(stats["scale"])
            net.gnn.gate[-1].bias.sub_(stats["mean"]).div_(stats["scale"])
    if arm in ("recenter_gate", "recenter_convs"):
        prefix = "gnn.gate." if arm == "recenter_gate" else "gnn.convs."
        for name, parameter in net.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))
    return net


def activate(arm: str) -> dict:
    """독립 프로세스에서 기존 학습·평가 엔진에 실험 설정을 공급한다."""
    spec = prepare()
    run.OUT, run.EPOCHS = OUT, EPOCHS
    run.prepare = prepare
    run.model = lambda checkpoint=None: load_model(checkpoint, arm)
    run.context_score = gated_context_score
    run.calibration = lambda: run.READ(OUT / ("calibration_" + mode(arm) + ".json"))
    run.acceptance_loss = lambda *a, **kw: spec["acceptance_gain"] * LOSS(*a, **kw)
    return spec


def setup() -> None:
    """학습 자료만으로 초기 게이트와 고정 점수 척도를 만든다."""
    spec = activate("native_all")
    if (OUT / "setup_complete.json").exists():
        return
    net = MODEL()
    values, sources = [], {}
    for seq in spec["train"]:
        folder = OUT / "pools" / seq.replace("/", "_")
        graph = torch.load(folder / "graph.pt", map_location="cuda", weights_only=False)
        captured = []
        hook = net.gnn.gate.register_forward_hook(lambda m, x, y: captured.append(y.detach().cpu().numpy().ravel()))
        with torch.no_grad():
            net(graph)
        hook.remove()
        value = np.concatenate(captured).astype(np.float64)
        values.append(value)
        for name in ("graph.pt", "pool.npz"):
            p = folder / name
            sources[str(p.relative_to(ROOT))] = run.HASH(p)
        del graph
    mean = float(np.mean([v.mean() for v in values]))
    scale = max(float(np.sqrt(np.mean([np.mean((v-mean)**2) for v in values]))), 1.)
    run.WRITE(OUT / "gate_statistics.json", dict(mean=mean, scale=scale,
        per_sequence={s:dict(mean=float(v.mean()), std=float(v.std()), minimum=float(v.min()), maximum=float(v.max())) for s,v in zip(spec["train"],values)},
        sources=sources, validation_used=False, test_used=False))
    fit = run.READ(previous.PRIOR / "calibration.json")
    for frozen in FROZEN:
        net = load_model(arm=frozen)
        folder = OUT / frozen
        folder.mkdir(exist_ok=True)
        torch.save(dict(model_state_dict=net.state_dict(), epoch=0, updates=0,
            protocol_sha256=run.HASH(OUT / "protocol.json")), folder / "epoch0.pth")
        cs, ws, alphas, gradients = [], [], {}, {}
        for seq in spec["train"]:
            pooldir = OUT / "pools" / seq.replace("/", "_")
            pool = run.ARR(pooldir / "pool.npz")
            graph = torch.load(pooldir / "graph.pt", map_location="cuda", weights_only=False)
            key = F.normalize(net(graph), dim=1)
            alpha = net.gnn._last_alpha.cpu().numpy().ravel()
            alphas[seq] = dict(mean=float(alpha.mean()), minimum=float(alpha.min()), maximum=float(alpha.max()),
                               saturated_fraction=float(np.mean((alpha < .01) | (alpha > .99))))
            plan = {k:torch.from_numpy(pool["plan_"+k]).cuda() for k in ("query", "candidate", "weight")}
            with torch.no_grad():
                score = torch.cat([gated_context_score(key, {k:v[i:i+256] for k,v in plan.items()})
                                   for i in range(0,len(plan["query"]),256)]).cpu().numpy().astype(np.float64)
            cs.append(score); ws.append(np.full(len(score), 1./len(score)))
            sample = gated_context_score(key, {k:v[:128] for k,v in plan.items()})
            params = [p for n,p in net.named_parameters() if n.startswith("gnn.gate.")]
            grad = torch.autograd.grad(sample.mean(), params)
            gradients[seq] = float(torch.sqrt(sum(g.square().sum() for g in grad)))
            del graph, key, sample, plan
        c, w = np.concatenate(cs), np.concatenate(ws)
        w /= w.sum()
        m = float(w@c)
        s = max(float(np.sqrt(w@((c-m)**2))), spec["context_scale_floor"])
        run.WRITE(OUT / ("calibration_"+mode(frozen)+".json"), dict(spectral=fit["spectral"],
            context_mean=m, context_scale=s, sources=sources, test_used=False,
            alpha=alphas, gate_gradient=gradients, protocol_sha256=run.HASH(OUT / "protocol.json")))
    run.WRITE(OUT / "setup_complete.json", dict(status="passed", train_sequences=len(spec["train"]),
        gate_recenter=dict(mean=mean,scale=scale), validation_used=False,test_used=False))
    print("SETUP",mean,scale,flush=True)


def finish() -> None:
    """모든 원장을 재검산하고 검증에서만 선택한다."""
    spec = prepare()
    results, checks = {}, 0
    for arm in (*FROZEN, *ARMS):
        for epoch in ((0,) if arm in FROZEN else EPOCHS):
            per = {lam:{} for lam in run.LAMBDAS}
            for seq in spec["val"]:
                path = OUT / arm / f"epoch{epoch}/val" / (seq.replace("/", "_")+".json")
                meta, row = run.READ(path), run.ARR(path.with_suffix(".npz"))
                assert meta["status"] == "passed" and len(meta["prefix_checks"]) == 2
                assert run.HASH(path.with_suffix(".npz")) == meta["ledger_sha256"]
                assert run.HASH(OUT/arm/f"epoch{epoch}.pth") == meta["checkpoint_sha256"]
                for lam in run.LAMBDAS:
                    label = f"lambda_{lam:g}"
                    metric = run.base.independent_metrics(row,row[label])
                    assert metric == meta["metrics"][label]
                    per[lam][seq] = metric
                    checks += 1
            for lam in run.LAMBDAS:
                results[f"{arm}_{epoch}_{lam:g}"] = dict(arm=arm,epoch=epoch,coefficient=lam,
                    **run.base.macro(per[lam]),sequences=per[lam])
    order = lambda v:(-round(v["macro"]["AP"],12),v["coefficient"],v["epoch"],v["arm"])
    best = {arm:sorted([v for v in results.values() if v["arm"]==arm],key=order)[0] for arm in (*FROZEN,*ARMS)}
    frozen = sorted([best[a] for a in FROZEN],key=order)[0]
    prior = run.READ(previous.OUT/"validation_summary.json")["frozen_best"]
    bev = run.READ(previous.OUT/"bev_validation.json")
    eligible, tests = [], {}
    for name,row in results.items():
        if row["arm"] in FROZEN:
            continue
        same = results[f"{mode(row['arm'])}_frozen_0_{row['coefficient']:g}"]
        criteria = {m:row["macro"][m] >= bev["temporal"]["macro"][m]-tol
                    for m,tol in {"AP":.02,"max_F1":.02,"R@100P":.03,"R@1":.03}.items()}
        criteria.update(over_frozen=row["macro"]["AP"] >= max(frozen["macro"]["AP"],prior["macro"]["AP"])+.001,
                        over_same=row["macro"]["AP"]>=same["macro"]["AP"]+.001)
        tests[name] = criteria
        if all(criteria.values()):
            eligible.append(row)
    selected = sorted(eligible,key=order)[0] if eligible else None
    training = {a:run.READ(OUT/a/"training_complete.json") for a in ARMS}
    orders = [[r["batch_hashes"] for r in v["records"]] for v in training.values()]
    assert all(v==orders[0] for v in orders)
    for arm,prefix in (("recenter_gate","gnn.gate."),("recenter_convs","gnn.convs.")):
        assert all(p.startswith(prefix) for p in training[arm]["changed_parameters"])
    run.WRITE(OUT/"validation_summary.json",dict(all_results=results,best=best,selected=selected,
        frozen_best=frozen, prior_normalized_frozen=prior, bev=bev,advancement_checks=tests,test_used=False))
    run.WRITE(OUT/"validation_checks.json",dict(status="passed",new_ledgers=90,metric_rows=checks,
        prefix_checks=180,equal_training_samples=True,parameter_scope_verified=True,test_used=False))
    if selected is not None:
        checkpoint=OUT/selected["arm"]/f"epoch{selected['epoch']}.pth"
        run.WRITE(OUT/"test_freeze.json",dict(selected=selected,checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=run.HASH(checkpoint),protocol_sha256=run.HASH(OUT/"protocol.json"),
            validation_sha256=run.HASH(OUT/"validation_summary.json"),test_used_for_selection=False))
    run.WRITE(OUT/"status.json",dict(status="validation_complete",selected=selected,test_used=False))
    print("RESULT",{a:v["macro"] for a,v in best.items()},"SELECTED",None if selected is None else selected["arm"],flush=True)


def campaign() -> None:
    """학습과 전체 검증을 실행하고 시험 전에 멈춰 선택을 확인한다."""
    spec=prepare()
    logs=OUT/"logs"
    logs.mkdir(exist_ok=True)
    def call(args: list, name: str) -> None:
        with (logs/(name+".log")).open("w") as out:
            subprocess.run([sys.executable,__file__,*args],stdout=out,stderr=subprocess.STDOUT,check=True)
    try:
        call(["setup"],"setup")
        for arm in ARMS:
            run.WRITE(OUT/"status.json",dict(status="running",phase="training",arm=arm))
            call(["train","--arm",arm],"train_"+arm)
        jobs=[(a,e,s) for a in (*FROZEN,*ARMS) for e in ((0,) if a in FROZEN else EPOCHS) for s in spec["val"]]
        def evaluate(job: tuple) -> None:
            a,e,s=job
            call(["val","--arm",a,"--epoch",str(e),"--sequence",s],f"val_{a}_{e}_{s.replace('/','_')}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for i,future in enumerate(concurrent.futures.as_completed([pool.submit(evaluate,j) for j in jobs]),1):
                future.result()
                run.WRITE(OUT/"status.json",dict(status="running",phase="validation",done=i,total=len(jobs)))
                print("VALIDATION",i,len(jobs),flush=True)
        finish()
    except Exception as exc:
        run.WRITE(OUT/"status.json",dict(status="failed",error=str(exc)))
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","setup","train","val","finish","run"))
    parser.add_argument("--arm",choices=(*FROZEN,*ARMS),default="native_all")
    parser.add_argument("--epoch",type=int,default=8)
    parser.add_argument("--sequence",default="kitti/09")
    args=parser.parse_args()
    run.configure()
    if args.action=="prepare": prepare()
    elif args.action=="setup": setup()
    elif args.action=="train": activate(args.arm); run.train(args.arm)
    elif args.action=="val": activate(args.arm); run.validate(args.arm,args.epoch,args.sequence)
    elif args.action=="finish": finish()
    else: campaign()
