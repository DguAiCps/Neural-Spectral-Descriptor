#!/usr/bin/env python3
"""검증에서 고정한 추가 학습 모델과 학습 전 모델을 시험46에 적용한다."""
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
import _pdv2_diffattn_comparable as experiment

run = experiment.run
OUT = experiment.OUT / "fixed_test"


def prepare() -> dict:
    """검증 선택과 시험 실행 코드를 해시로 묶고 이후 설정 변경을 차단한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    spec = experiment.activate()
    frozen = run.READ(experiment.OUT / "test_freeze.json")
    assert frozen["protocol_sha256"] == run.HASH(experiment.OUT / "protocol.json")
    assert frozen["validation_sha256"] == run.HASH(experiment.OUT / "validation_summary.json")
    assert frozen["checkpoint_sha256"] == run.HASH(ROOT / frozen["selected_checkpoint"])
    assert run.READ(experiment.OUT / "validation_checks.json")["status"] == "passed"
    target = OUT / "protocol.json"
    if target.exists():
        protocol = run.READ(target)
        for path, digest in protocol["sources"].items():
            assert run.HASH(ROOT / path) == digest, path
        return protocol
    paths = [Path(__file__), Path(experiment.__file__), Path(run.__file__),
             experiment.OUT / "test_freeze.json", experiment.OUT / "calibration.json",
             ROOT / frozen["selected_checkpoint"], ROOT / spec["initial"]]
    protocol = dict(created_unix=time.time(), selection=frozen,
        source_protocol_sha256=run.HASH(experiment.OUT / "protocol.json"),
        working_tolerances_pp={"AP": 2., "max_F1": 2., "R@100P": 3., "R@1": 3.},
        conditions="추가 학습과 학습 전의 전체 검색 재실행. 학습 전은 검증 최적 비중 및 선택 모델과 같은 비중을 함께 보고. 비교 방법은 보유 BEVPlace++ 시드1 기본·동일 시간 판정. 양쪽 같은 질의와 라벨 확인. 시험은 설정 선택에 사용하지 않음.",
        sources={str(p.relative_to(ROOT)):run.HASH(p) for p in paths})
    run.WRITE(target, protocol)
    return protocol


def inputs(role: str, sequence: str) -> tuple:
    """고정 평가 입력과 라벨을 읽으며 GT는 지표 계산에만 사용한다."""
    if role == "val":
        return run.inputs(role, sequence)
    assert role == "test"
    _, _, paths, minimum = run.white.prior.BASE.exp.old.base.inputs(role, sequence)
    cp, sp = paths["cache"], paths["spec"]
    cache = run.ARR(cp)
    x = run.white.coefficients(sp)
    wx = run.white.transform(x, run.ARR(run.white.OUT / "statistics.npz")["full"])
    with np.load(sp) as data:
        np.testing.assert_array_equal(data["scan_ids"], cache["scan_ids"])
    bank = run.base.variant_inputs(x, 8)
    nb, nc = [run.base.variant_inputs(run.power.transform(v, .5), 8) for v in (x, wx)]
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    return cache, cache["poses_gt"][:, :3, 3], minimum, travel, bank, nb, nc, (cp, sp)


def worker(role: str, sequence: str) -> None:
    """같은 입력에서 두 모델을 재실행하고 미래 관측 범위를 검산한다."""
    protocol = prepare()
    frozen = protocol["selection"]
    assert sequence in (frozen["test"] if role == "test" else run.READ(experiment.OUT / "protocol.json")["val"])
    name = sequence.replace("/", "_")
    folder = OUT / role
    target = folder / (name+".json")
    if target.exists():
        assert run.READ(target)["status"] == "passed"
        return
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    fit = run.calibration()
    cache, pos, minimum, travel, bank, nb, nc, paths = inputs(role, sequence)
    refpath = (ROOT / "results/paper_fixed_threshold_20261101/nsd_val/s1" / (name+".npz")
               if role == "val" else run.power.OUT / "test/s1" / (name+"_p050__aux_half__equal.npz"))
    reference = run.ARR(refpath)
    records = {}
    coefficient = frozen["selected"]["coefficient"]
    for arm in ("frozen", "learned"):
        checkpoint = ROOT / frozen["selected_checkpoint"] if arm == "learned" else None
        net = run.model(checkpoint)
        key = run.source.make_keys(net, cache["descriptors"], cache["poses"])
        row, _ = run.retrieve(key, bank, pos, minimum)
        for field in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(row[field], reference[field])
        row["group"] = reference["group"]
        q, j = row["query_idx"], row["j_star"]
        spectral = run.current.adoption.scores(nb, nc, travel, q, j)["score"]
        context = run.neural_scores(key, travel, q, j)
        baseline = run.predict_score(spectral[:, None], fit["spectral"])
        residual = (context-fit["context_mean"])/fit["context_scale"]
        scores = {"score":baseline+coefficient*residual}
        if arm == "frozen":
            scores["score_retuned"] = baseline+frozen["frozen_best_coefficient"]*residual
        metrics = {}
        for label, score in scores.items():
            assert np.isfinite(score).all()
            metrics[label] = run.base.independent_metrics(row, score)
            independent = run.white.prior.BASE.exp.metrics(score, row)
            for metric in run.base.METRICS:
                if metrics[label][metric] is not None:
                    np.testing.assert_allclose(metrics[label][metric], independent[metric], atol=1e-12, rtol=0)
        prefix_checks = []
        for at in (len(q)//3, 2*len(q)//3):
            t, end = int(q[at]), int(q[at])+16
            short = run.source.make_keys(net, cache["descriptors"], cache["poses"], end)
            np.testing.assert_allclose(short[:t+6], key[:t+6], atol=4e-6, rtol=4e-5)
            sbank = run.current.adoption.previous.source.base.exp.truncate_bank(bank, end)
            weights = run.cal.fusion_weights(run.READ(experiment.OUT / "protocol.json")["confidence"])[[run.cal.FUSIONS.index("aux_half")]]
            assert run.cal.candidates(short, sbank, weights, t)[0] == j[at]
            partial = run.neural_scores(short, travel[:end], q[at:at+1], j[at:at+1])
            np.testing.assert_allclose(partial, context[at:at+1], atol=2e-6, rtol=0)
            b, c = [run.current.adoption.previous.source.base.exp.truncate_bank(v, end) for v in (nb, nc)]
            np.testing.assert_array_equal(run.current.adoption.scores(b, c, travel[:end], q[at:at+1], j[at:at+1])["score"], spectral[at:at+1])
            prefix_checks.append(t)
        path = folder / (name+"_"+arm+".npz")
        np.savez_compressed(path, **row, spectral=spectral, context=context, **scores)
        records[arm] = dict(path=str(path.relative_to(ROOT)), sha256=run.HASH(path),
            metrics=metrics, prefix_checks=prefix_checks,
            checkpoint_sha256=run.HASH(checkpoint if checkpoint is not None else ROOT / run.READ(experiment.OUT / "protocol.json")["initial"]))
        del net, key
        torch.cuda.empty_cache()
    run.WRITE(target, dict(status="passed", role=role, sequence=sequence,
        queries=len(reference["query_idx"]), records=records, seconds=time.time()-started,
        protocol_sha256=run.HASH(OUT / "protocol.json"),
        input_sha256={str(p.relative_to(ROOT)):run.HASH(p) for p in (*paths, refpath)}))
    print("EVALUATED", role, sequence, round(time.time()-started, 1), flush=True)


def preflight() -> None:
    """시험 실행 전에 독립 평가 경로가 검증 원장을 재현하는지 확인한다."""
    spec = prepare()["selection"]
    seq = "kitti360/0004"
    worker("val", seq)
    own = run.READ(OUT / "val/kitti360_0004.json")
    for arm in ("frozen", "learned"):
        path = (experiment.PRIOR / "initial/epoch0/val/kitti360_0004.npz" if arm == "frozen" else
                experiment.OUT / spec["selected"]["arm"] / f"epoch{spec['selected']['epoch']}/val/kitti360_0004.npz")
        previous, actual = run.ARR(path), run.ARR(ROOT / own["records"][arm]["path"])
        np.testing.assert_array_equal(actual["j_star"], previous["j_star"])
        np.testing.assert_allclose(actual["score"], previous[f"lambda_{spec['selected']['coefficient']:g}"], atol=2e-6, rtol=0)
    run.WRITE(OUT / "preflight.json", dict(status="passed", validation_replay=seq,
        selected_checkpoint_sha256=spec["checkpoint_sha256"], test_used=False))


def finish() -> None:
    """시퀀스→도시6 집계로 BEVPlace++와 학습 전후의 최종 차이를 보고한다."""
    protocol = prepare()
    spec = protocol["selection"]
    labels = ("learned", "frozen_same", "frozen_retuned", "manuscript", "bev_native", "bev_temporal")
    rows = {name:{} for name in labels}
    ledger_count, queries = 0, 0
    for seq in spec["test"]:
        name = seq.replace("/", "_")
        info = run.READ(OUT / "test" / (name+".json"))
        assert info["status"] == "passed"
        queries += info["queries"]
        reference = None
        for arm in ("learned", "frozen"):
            record = info["records"][arm]
            path = ROOT / record["path"]
            assert run.HASH(path) == record["sha256"] and len(record["prefix_checks"]) == 2
            row = run.ARR(path)
            reference = row
            for score_label, target in (("score", "learned" if arm == "learned" else "frozen_same"),):
                metric = run.base.independent_metrics(row, row[score_label])
                assert metric == record["metrics"][score_label]
                rows[target][seq] = metric
            if arm == "frozen":
                metric = run.base.independent_metrics(row, row["score_retuned"])
                assert metric == record["metrics"]["score_retuned"]
                rows["frozen_retuned"][seq] = metric
            ledger_count += 1
        main = run.ARR(run.power.OUT / "test/s1" / (name+"_p050__aux_half__equal.npz"))
        rows["manuscript"][seq] = run.base.independent_metrics(main, main["score"])
        bevpath = experiment.BEV / "test/s1" / (name+".npz")
        bev = run.ARR(bevpath)
        assert run.HASH(bevpath) == run.READ(bevpath.with_suffix(".json"))["sha256"]
        for other in (main, bev):
            for field in ("query_idx", "min_past_gt_dist"):
                np.testing.assert_array_equal(reference[field], other[field])
        rows["bev_native"][seq] = run.base.independent_metrics(bev, bev["score_negl2"])
        rows["bev_temporal"][seq] = run.base.independent_metrics(bev, bev["score"])
    results = {label:dict(**run.current.adoption.previous.macro(per, run.current.CITY), sequences=per)
               for label,per in rows.items()}
    differences = {ref:{metric:100*(results["learned"]["macro"][metric]-results[ref]["macro"][metric])
                       for metric in run.base.METRICS} for ref in labels if ref != "learned"}
    reference_checks = {ref:{m:differences[ref][m]>=-tol
                            for m,tol in protocol["working_tolerances_pp"].items()}
                        for ref in ("bev_native", "bev_temporal")}
    intervals = {}
    for ref in ("frozen_same", "frozen_retuned", "bev_native", "bev_temporal"):
        rng = np.random.default_rng(20261104)
        samples = []
        for city, sequences in run.current.CITY.items():
            available = [s for s in sequences if rows["learned"][s]["n_pos"]]
            matrix = np.array([[100*(rows["learned"][s][m]-rows[ref][s][m])
                                for m in run.base.METRICS] for s in available])
            draw = rng.integers(0, len(available), size=(10000, len(available)))
            samples.append(matrix[draw].mean(1))
        boot = np.stack(samples).mean(0)
        intervals[ref] = {m:np.quantile(boot[:, i], [.025,.975]).tolist()
                          for i,m in enumerate(run.base.METRICS)}
    run.WRITE(OUT / "summary.json", dict(results=results, learned_delta_pp=differences,
        reference_working_checks=reference_checks,
        conditional_city_stratified_sequence_bootstrap_95_pp=intervals,
        interval_note="고정된 단일 초기시드의 도시 내 짝지은 시퀀스 재표집. 장소 상관·학습 시드 분산·과거 시험 노출을 해결하는 독립 확증 구간이 아님. 허용폭 통과를 통계적 동등성으로 해석하지 않음.",
        queries=queries, seed=1, selected=spec["selected"], test_used_for_selection=False,
        scope="시험46→양성45→도시6 평균. 단일 초기시드·과거 개발노출 시험. BEV 기본과 같은시간 판정 모두 보고. 원고 결과 초과를 요구하지 않으며 시험 결과로 재선택하지 않음.",
        freeze_sha256=run.HASH(experiment.OUT / "test_freeze.json")))
    run.WRITE(OUT / "completion_validation.json", dict(status="passed", new_ledgers=ledger_count,
        metric_rows=46*3, prefix_checks=ledger_count*2, queries=queries,
        paired_query_labels=True, preflight_passed=True, test_used_for_selection=False))
    run.WRITE(OUT / "status.json", dict(status="complete", test_used_for_selection=False))
    print("TEST_RESULT", {k:v["macro"] for k,v in results.items()}, flush=True)


def campaign() -> None:
    """고정·검증 재현 후에만 시험46을 실행한다."""
    spec = prepare()["selection"]
    preflight()
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    def evaluate(sequence: str) -> None:
        with (logs / (sequence.replace("/", "_")+".log")).open("w") as log:
            subprocess.run([sys.executable, __file__, "worker", "--sequence", sequence],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for i,future in enumerate(concurrent.futures.as_completed([pool.submit(evaluate,s) for s in spec["test"]]),1):
                future.result()
                run.WRITE(OUT / "status.json", dict(status="running", done=i, total=len(spec["test"])))
                print("TEST_PROGRESS", i, len(spec["test"]), flush=True)
        finish()
    except Exception as exc:
        run.WRITE(OUT / "status.json", dict(status="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "preflight", "finish", "run"))
    parser.add_argument("--sequence", default="kitti/00")
    parser.add_argument("--role", choices=("val", "test"), default="test")
    args = parser.parse_args()
    run.configure()
    if args.action == "prepare":
        prepare()
    elif args.action == "worker":
        worker(args.role, args.sequence)
    elif args.action == "preflight":
        preflight()
    elif args.action == "finish":
        finish()
    else:
        campaign()
