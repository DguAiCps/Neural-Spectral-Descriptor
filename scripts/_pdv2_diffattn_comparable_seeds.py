#!/usr/bin/env python3
"""선택을 바꾸지 않고 같은 추가 학습 규칙을 초기 시드2·3에 적용한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import _pdv2_diffattn_comparable as experiment
import _pdv2_diffattn_comparable_test as evaluation

run = experiment.run
PARENT = experiment.OUT
OUT = PARENT / "three_seed"
CONTEXT = run.context_score
LOSS = run.acceptance_loss


def prepare() -> dict:
    """시드1 시험 결과를 본 확장임을 밝히고 공통 설정을 다시 선택하지 않는다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = run.READ(target)
        for path, digest in spec["sources"].items():
            assert run.HASH(ROOT / path) == digest, path
        return spec
    selected = run.READ(PARENT / "test_freeze.json")
    main = run.READ(run.power.OUT / "protocol.json")
    checkpoints = {str(s):run.source.root(s, "octave288") / "checkpoints" / f"checkpoint_epoch_{main['models'][str(s)]}.pth"
                   for s in (1,2,3)}
    paths = [Path(__file__), Path(experiment.__file__), Path(evaluation.__file__), Path(run.__file__),
             PARENT / "protocol.json", PARENT / "test_freeze.json", PARENT / "fixed_test/summary.json",
             run.power.OUT / "protocol.json", *checkpoints.values()]
    spec = dict(created_unix=time.time(), seeds=[1,2,3], selected=selected,
        initial={s:str(p.relative_to(ROOT)) for s,p in checkpoints.items()},
        confidence=main["confidence"],
        rule="시드1 시험의 AP/F1 열세를 확인한 뒤 원고와 같은3초기시드 평균을 확인하는 확장. 학습계수1·추론계수1·추가8에포크512회·학습률1e-5·BCE비중12.685962594044891을 공통 고정. 시드별 초기 현재모델과 고정 검색 융합을 사용하며 학습 승자·삼중쌍·점수 척도는 같은 절차로 학습8에서 다시 만든다. 추가 학습의 표본 RNG는 공통. 원래 사전학습의3시드를 뜻하며 초기 시드별 최적 epoch 재선택은 하지 않음.",
        controls="학습 전 모델도 같은 추론계수1과 시드1 검증에서 선택한 공통계수.5를 모두 보고. 이후 검증은 선택에 사용하지 않음. 비교 방법의 같은3시드 원장과 짝지은 도시6평균. 원고·그림 변경 없음.",
        test_exposure="초기시드1 시험 결과를 본 뒤 정한 반복 확인 범위이며 독립 미노출 확증이 아님. 추가 시드 결과로 학습 설정을 다시 선택하지 않음.",
        sources={str(p.relative_to(ROOT)):run.HASH(p) for p in paths})
    run.WRITE(target, spec)
    return spec


def seed_prepare(seed: int) -> dict:
    """각 초기 시드에 필요한 학습 자료와 고정 설정을 분리한다."""
    common = prepare()
    root = OUT / f"s{seed}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "protocol.json"
    if path.exists():
        return run.READ(path)
    spec = run.READ(PARENT / "protocol.json")
    spec.update(seed=seed, initial=common["initial"][str(seed)],
                confidence=common["confidence"][str(seed)], epochs=[8], arms=["unit_weight"],
                fixed_replication=True, parent_sha256=run.HASH(OUT / "protocol.json"))
    run.WRITE(path, spec)
    return spec


def activate(seed: int, training: bool = False) -> dict:
    """독립 프로세스 안에서 기존 학습·최종 평가 엔진에 시드 경로를 공급한다."""
    spec = seed_prepare(seed)
    root = OUT / f"s{seed}"
    run.OUT = root
    run.prepare = lambda: seed_prepare(seed)
    run.EPOCHS = (8,)
    if training:
        fit = run.calibration()
        mean = fit["context_mean"]
        run.context_score = lambda *args, **kw: mean + (CONTEXT(*args, **kw)-mean)
        run.acceptance_loss = lambda *args, **kw: spec["acceptance_gain"]*LOSS(*args, **kw)
    experiment.OUT = root
    evaluation.OUT = root / "evaluation"
    evaluation.prepare = lambda: evaluation_prepare(seed)
    return spec


def evaluation_prepare(seed: int) -> dict:
    """공통 고정 규칙과 실제 시드 체크포인트를 연결해 평가 전에 저장한다."""
    common = prepare()
    root = OUT / f"s{seed}"
    folder = root / "evaluation"
    folder.mkdir(exist_ok=True)
    path = folder / "protocol.json"
    if path.exists():
        protocol = run.READ(path)
        for p,h in protocol["sources"].items():
            assert run.HASH(ROOT / p) == h, p
        return protocol
    checkpoint = root / "unit_weight/epoch8.pth"
    selection = dict(common["selected"])
    selection.update(seed=seed, selected_checkpoint=str(checkpoint.relative_to(ROOT)),
                     checkpoint_sha256=run.HASH(checkpoint),
                     selected={"arm":"unit_weight", "epoch":8, "coefficient":1.})
    paths = [OUT / "protocol.json", root / "protocol.json", root / "calibration.json",
             checkpoint, Path(__file__), Path(evaluation.__file__)]
    protocol = dict(selection=selection, created_unix=time.time(),
        sources={str(p.relative_to(ROOT)):run.HASH(p) for p in paths})
    run.WRITE(path, protocol)
    return protocol


def finish() -> None:
    """NSD와 BEVPlace++의 같은3시드를 최종 원장에서 집계한다."""
    spec = prepare()
    labels = ("learned", "frozen_same", "frozen_retuned", "manuscript", "bev_native", "bev_temporal")
    per_seed = {label:{} for label in labels}
    for seed in (1,2,3):
        rows = {label:{} for label in labels}
        query_count = 0
        folder = PARENT / "fixed_test" if seed == 1 else OUT / f"s{seed}/evaluation"
        for seq in spec["selected"]["test"]:
            name = seq.replace("/", "_")
            record = run.READ(folder / "test" / (name+".json"))
            assert record["status"] == "passed"
            query_count += record["queries"]
            for arm in ("learned", "frozen"):
                info = record["records"][arm]
                p = ROOT / info["path"]
                assert run.HASH(p) == info["sha256"] and len(info["prefix_checks"]) == 2
                row = run.ARR(p)
                label = "learned" if arm == "learned" else "frozen_same"
                rows[label][seq] = run.base.independent_metrics(row, row["score"])
                assert rows[label][seq] == info["metrics"]["score"]
                if arm == "frozen":
                    rows["frozen_retuned"][seq] = run.base.independent_metrics(row, row["score_retuned"])
                    assert rows["frozen_retuned"][seq] == info["metrics"]["score_retuned"]
            main = run.ARR(run.power.OUT / f"test/s{seed}" / (name+"_p050__aux_half__equal.npz"))
            rows["manuscript"][seq] = run.base.independent_metrics(main, main["score"])
            bp = experiment.BEV / f"test/s{seed}" / (name+".npz")
            assert run.HASH(bp) == run.READ(bp.with_suffix(".json"))["sha256"]
            bev = run.ARR(bp)
            for other in (main, bev):
                for field in ("query_idx", "min_past_gt_dist"):
                    np.testing.assert_array_equal(row[field], other[field])
            rows["bev_native"][seq] = run.base.independent_metrics(bev, bev["score_negl2"])
            rows["bev_temporal"][seq] = run.base.independent_metrics(bev, bev["score"])
        assert query_count == 195726
        for label, metrics in rows.items():
            per_seed[label][str(seed)] = dict(**run.current.adoption.previous.macro(metrics, run.current.CITY), sequences=metrics)
    stats = {label:{m:{"mean":float(np.mean([v["macro"][m] for v in seeds.values()])),
                       "std":float(np.std([v["macro"][m] for v in seeds.values()], ddof=1))}
                    for m in run.base.METRICS} for label,seeds in per_seed.items()}
    delta = {label:{m:100*(stats["learned"][m]["mean"]-stats[label][m]["mean"])
                    for m in run.base.METRICS} for label in labels if label != "learned"}
    run.WRITE(OUT / "summary.json", dict(per_seed=per_seed, mean_std=stats,
        learned_delta_pp=delta, fixed_rule=True, test_used_for_parameter_selection=False,
        scope=spec["test_exposure"], protocol_sha256=run.HASH(OUT / "protocol.json")))
    for seed in (2,3):
        own = run.READ(OUT / f"s{seed}/unit_weight/training_complete.json")
        assert own["updates"] == 512 and own["fixed_buffers_verified"]
    run.WRITE(OUT / "completion_validation.json", dict(status="passed", seeds=3,
        test_sequences=46, queries_per_seed=195726, total_ledgers=276,
        total_score_conditions=414, prefix_checks=552, fixed_updates_per_model=512,
        no_parameter_reselection=True))
    run.WRITE(OUT / "status.json", dict(status="complete", fixed_rule=True))
    print("THREE_SEED_RESULT", stats, flush=True)


def campaign() -> None:
    """자료 구성→학습→고정 평가를 수행하며 추가 선택은 하지 않는다."""
    common = prepare()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    def call(args: list, name: str) -> None:
        with (logs / (name+".log")).open("w") as log:
            subprocess.run([sys.executable, __file__, *args], stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        for seed in (2,3):
            spec = seed_prepare(seed)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                jobs = [pool.submit(call, ["pool", "--seed",str(seed),"--sequence",seq], f"pool_s{seed}_{seq.replace('/', '_')}") for seq in spec["train"]]
                for i, future in enumerate(concurrent.futures.as_completed(jobs),1):
                    future.result()
                    run.WRITE(OUT / "status.json",dict(status="running",phase="pools",seed=seed,done=i,total=8))
                    print("POOL_PROGRESS",seed,i,8,flush=True)
            call(["train","--seed",str(seed)],f"train_s{seed}")
            evaluation_prepare(seed)
        jobs = [(s,seq) for s in (2,3) for seq in common["selected"]["test"]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(call,["evaluate","--seed",str(s),"--sequence",seq],f"test_s{s}_{seq.replace('/', '_')}") for s,seq in jobs]
            for i, future in enumerate(concurrent.futures.as_completed(futures),1):
                future.result()
                run.WRITE(OUT / "status.json",dict(status="running",phase="test",done=i,total=len(jobs)))
                print("TEST_PROGRESS",i,len(jobs),flush=True)
        finish()
    except Exception as exc:
        run.WRITE(OUT / "status.json",dict(status="failed",error=str(exc)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","pool","train","evaluate","finish","run"))
    parser.add_argument("--seed",type=int,choices=(2,3),default=2)
    parser.add_argument("--sequence",default="kitti/01")
    args = parser.parse_args()
    run.configure()
    if args.action == "prepare":
        prepare()
    elif args.action == "pool":
        activate(args.seed)
        run.make_pool(args.sequence)
    elif args.action == "train":
        activate(args.seed,training=True)
        run.train("unit_weight")
    elif args.action == "evaluate":
        activate(args.seed)
        evaluation.worker("test",args.sequence)
    elif args.action == "finish":
        finish()
    else:
        campaign()
