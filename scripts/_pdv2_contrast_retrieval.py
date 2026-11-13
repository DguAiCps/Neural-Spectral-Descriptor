#!/usr/bin/env python3
"""검증에서 고정한 복소 차이 출력을 실제 후보 선택에도 연결한다."""
import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import _pdv2_diffattn_contrast_seeds as source

e = source.e
run = e.run
ROOT = e.ROOT
OUT = ROOT / "results/pdv2_diffattn_contrast_retrieval_20261104"
SOURCE = source.SOURCE
MODELS = ("fixed", "diff", "abs_kv", "point_gate")
POLICIES = ("matching_only", "aux_and_matching")


def prepare() -> dict:
    """검색 정책의 두 선택과 승격 기준을 검증 실행 전에 고정한다."""
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        spec = run.READ(path)
        for p, h in spec["sources"].items():
            assert run.HASH(ROOT / p) == h, p
    else:
        initial = run.READ(SOURCE / "protocol.json")
        selected = run.READ(SOURCE / "summary.json")["best"]
        paths = [
            Path(__file__),
            SOURCE / "summary.json",
            source.CAMPAIGN / "summary.json",
            source.CAMPAIGN / "freeze.json",
            source.CAMPAIGN / "cache_validation.json",
        ]
        for k in MODELS[1:]:
            arm, epoch = selected[k].rsplit("_", 1)
            paths.append(SOURCE / arm / f"epoch{epoch}.pth")
        spec = dict(
            created_unix=time.time(),
            train=initial["train"],
            val=initial["val"],
            seed=1,
            confidence=run.READ(e.prior.PARENT / "protocol.json")["confidence"],
            models=MODELS,
            policies=POLICIES,
            selected=selected,
            sources={str(p.relative_to(ROOT)): run.HASH(p) for p in paths},
            protocol="원키288 검색/같은100+100예산/융합계수/과거후보/q+15 유지. matching_only는두검색색인과코사인점수유지,두정렬점수만전달후복소계수로교체. aux_and_matching은보조128색인·보조코사인도함께교체. 최종시간판정은전달후계수의기존제곱근·75:25 그대로.",
            selection="첫시드고정모델/체크포인트/학습을유지. Diff의원검색대비AP/F1손실<=.3%p,R100<=1%p,R1은기존BEV동일시간검증R1-3%p이상인정책중최고AP를고른다. 두정책모두미달이면원검색유지하되R1한계가남았다고보고. 각대조도동일선택기회와동일정책결과모두기록. 시험미사용.",
            interpretation="원장최종승자를다시선택하는추론연결실험이며학습승자는고정이었던제약을남긴다. 원키+복소계수값의교체이며새신경망추가없음. 메모리/시간개선은아직실측주장안함.",
        )
        run.WRITE(path, spec)
    run.OUT = OUT
    return spec


def choose_bank(original: tuple, transformed: tuple, policy: str) -> tuple:
    """공통 후보 예산에서 정렬만 또는 보조 색인도 교체한다."""
    if policy == "matching_only":
        return (original[0], *transformed[1:])
    assert policy == "aux_and_matching"
    return transformed


def load_model(kind: str):
    """첫 시드에서 선택한 저장 모델을 고정해서 불러온다."""
    if kind == "fixed":
        return None
    tag = run.READ(SOURCE / "summary.json")["best"][kind]
    arm, epoch = tag.rsplit("_", 1)
    return source.model(kind, SOURCE / arm / f"epoch{epoch}.pth")


def transform(net, graph):
    """동결 감산 또는 선택된 학습 전달을 수행한다."""
    return e.prior.filtered(graph, -0.5) if net is None else net(graph)


def evaluate(sequence: str) -> None:
    """실제 과거 후보를 다시 검색하고 최종 판정·접두부를 검증한다."""
    spec = prepare()
    name = sequence.replace("/", "_")
    folder = OUT / "val" / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summary.json").exists():
        return
    cache, pos, minimum, travel, bank, _, _, _ = run.inputs("val", sequence)
    graph, _, audit = e.prior.graph_inputs("val", sequence)
    cp = (
        ROOT / "results/pdv2_diffattn_gate_learning_20261104/recenter_frozen/epoch0.pth"
    )
    frozen = run.model(cp)
    key = run.source.make_keys(frozen, cache["descriptors"], cache["poses"])
    raw = key[:, :288].copy()
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    baseline, _ = run.retrieve(raw, bank, pos, minimum)
    old = run.ARR(e.prior.PARENT / "pools/val" / name / "pool.npz")
    for k, v in baseline.items():
        np.testing.assert_array_equal(v, old[k])
    original = run.READ(SOURCE / "val" / (name + ".json"))
    metrics = {}
    for kind in MODELS:
        original_tag = "transport_-0.5" if kind == "fixed" else spec["selected"][kind]
        metrics[kind + "__original"] = original["metrics"][original_tag]
    weights = run.cal.fusion_weights(spec["confidence"])[
        [run.cal.FUSIONS.index("aux_half")]
    ]
    checks = []
    with torch.no_grad():
        for kind in MODELS:
            net = load_model(kind)
            z = transform(net, graph)
            changed = run.base.variant_inputs(z.cpu().numpy(), 8)
            for policy in POLICIES:
                tag = kind + "__" + policy
                path = folder / (tag + ".npz")
                selected_bank = choose_bank(bank, changed, policy)
                row, _ = run.retrieve(raw, selected_bank, pos, minimum)
                plan = e.prior.base.prior.plan(travel, row["query_idx"], row["j_star"])
                values = []
                for start in range(0, len(row["query_idx"]), 128):
                    pp = {
                        k: torch.from_numpy(v[start : start + 128]).cuda()
                        for k, v in plan.items()
                    }
                    values.append(e.prior.score(z, pp).cpu().numpy())
                value = np.concatenate(values).astype(np.float64)
                metric = run.base.independent_metrics(row, value)
                other = run.white.prior.BASE.exp.metrics(value, row)
                for m in run.base.METRICS:
                    if metric[m] is not None:
                        np.testing.assert_allclose(
                            metric[m], other[m], atol=1e-12, rtol=0
                        )
                for at in (len(value) // 3, 2 * len(value) // 3):
                    q = int(row["query_idx"][at])
                    end = q + 16
                    small, _, _ = e.prior.graph_inputs("val", sequence, end)
                    short = transform(net, small)
                    skey = run.source.make_keys(
                        frozen, cache["descriptors"][:end], cache["poses"][:end]
                    )[:, :288]
                    skey /= np.maximum(
                        np.linalg.norm(skey, axis=1, keepdims=True), 1e-12
                    )
                    sbank = choose_bank(
                        tuple(b[:end] for b in bank),
                        run.base.variant_inputs(short.cpu().numpy(), 8),
                        policy,
                    )
                    winner = run.cal.candidates(skey, sbank, weights, q)
                    assert int(winner[0]) == int(row["j_star"][at]), (
                        tag,
                        q,
                        winner,
                        row["j_star"][at],
                    )
                    pp = {
                        k: torch.from_numpy(v).cuda()
                        for k, v in e.prior.base.prior.plan(
                            travel[:end], np.array([q]), winner
                        ).items()
                    }
                    out = e.prior.score(short, pp).cpu().numpy()
                    np.testing.assert_allclose(
                        out, value[at : at + 1], atol=8e-6, rtol=0
                    )
                    checks.append(
                        dict(
                            condition=tag,
                            query=q,
                            winner=int(winner[0]),
                            score_error=float(abs(out - value[at : at + 1]).max()),
                        )
                    )
                    del small, short
                np.savez_compressed(path, **row, score=value)
                metrics[tag] = metric
                run.WRITE(
                    path.with_suffix(".json"),
                    dict(
                        metrics=metric,
                        ledger_sha256=run.HASH(path),
                        changed_winners=int(
                            (row["j_star"] != baseline["j_star"]).sum()
                        ),
                        test_used=False,
                    ),
                )
                print("RETRIEVED", sequence, tag, metric, flush=True)
            del net, z, changed
    run.WRITE(
        folder / "summary.json",
        dict(
            metrics=metrics,
            prefix_checks=checks,
            audit=audit,
            original_retrieval_reproduced=True,
            test_used=False,
        ),
    )


def finish() -> None:
    """같은 두 정책의 최종 지표와 원 검색 대비 변화를 비교한다."""
    spec = prepare()
    rows = {
        s: run.READ(OUT / "val" / s.replace("/", "_") / "summary.json")
        for s in spec["val"]
    }
    results = {
        k: run.base.macro({s: r["metrics"][k] for s, r in rows.items()})
        for k in next(iter(rows.values()))["metrics"]
    }
    bev = run.READ(
        ROOT / "results/pdv2_diffattn_comparable_20261104/bev_validation.json"
    )["temporal"]["macro"]
    tests = {}
    baseline = results["diff__original"]["macro"]
    for policy in POLICIES:
        m = results["diff__" + policy]["macro"]
        tests[policy] = {
            k: m[k] >= baseline[k] - tol
            for k, tol in (("AP", 0.003), ("max_F1", 0.003), ("R@100P", 0.01))
        }
        tests[policy]["R1_bev"] = m["R@1"] >= bev["R@1"] - 0.03
    eligible = [p for p, c in tests.items() if all(c.values())]
    selected = (
        max(eligible, key=lambda p: results["diff__" + p]["macro"]["AP"])
        if eligible
        else "original"
    )
    run.WRITE(
        OUT / "summary.json",
        dict(
            results=results,
            checks=tests,
            selected_policy=selected,
            bev_temporal_seed1=bev,
            test_used=False,
        ),
    )
    print("FINISHED", selected, {k: v["macro"] for k, v in results.items()}, flush=True)


def campaign() -> None:
    """시퀀스별 실제 검색을 병렬 실행한다."""
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
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "val", "finish", "run"))
    p.add_argument("--sequence", default="kitti/09")
    a = p.parse_args()
    e.configure()
    if a.action == "run":
        campaign()
    elif a.action == "val":
        evaluate(a.sequence)
    else:
        globals()[a.action]()
