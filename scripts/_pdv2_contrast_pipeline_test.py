#!/usr/bin/env python3
"""검증에서 고정한 복소 차이 전달과 대조를 시험46에 적용한다."""
import argparse
import concurrent.futures
import functools
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import _pdv2_contrast_pipeline as pipeline
import _pdv2_contrast_pipeline_verify as verifier
import _pdv2_diffattn_comparable_test as test_inputs

run, ROOT = pipeline.run, pipeline.ROOT
PARENT = pipeline.OUT
OUT = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
BEV = ROOT / "results/pdv2_review_response_20261104/bev"
VAL_INPUTS = run.inputs


def prepare() -> dict:
    """설정·학습 상태·실행 코드를 시험 전에 동결한다."""
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        spec = run.READ(path)
        for name, digest in spec["sources"].items():
            assert run.HASH(ROOT / name) == digest, name
        return spec
    old = run.READ(PARENT / "protocol.json")
    summary = run.READ(PARENT / "summary.json")
    assert summary["advance_to_test"]
    assert run.READ(PARENT / "integrated_validation.json")["status"] == "passed"
    paths = [
        Path(__file__),
        Path(pipeline.__file__),
        Path(verifier.__file__),
        Path(test_inputs.__file__),
        PARENT / "protocol.json",
        PARENT / "summary.json",
        PARENT / "integrated_validation.json",
    ]
    sources = dict(old["sources"])
    sources.update({str(p.relative_to(ROOT)): run.HASH(p) for p in paths})
    spec = dict(old)
    spec.update(
        created_unix=time.time(),
        sources=sources,
        test=["/".join(s) for s in run.current.TEST],
        test_criteria="전체Diff의고정감산및absKV대비평균AP+.1%p,전체및convs_only의고정감산대비AP양수최소2/3. BEV기본과동일시간모두비교하며작업상comparable허용폭AP/F1-2%p,R100/R1-3%p를별도로판정. 통계적동등성주장아님. 점별대조/다른층학습도함께보고하고불리한결과를숨기지않음.",
        working_tolerances_pp={"AP": 2.0, "max_F1": 2.0, "R@100P": 3.0, "R@1": 3.0},
        test_policy="검증9에서고정한17조건을모두시험46에실행. 다른사전학습시드3개,추가학습표본순서는공통. 시험선택없음. 과거시험개발노출은있어독립확증이아님. 원고/그림변경없음.",
    )
    run.WRITE(path, spec)
    return spec


@functools.lru_cache(maxsize=1)
def inputs(role: str, sequence: str) -> tuple:
    """검증 입력을 재사용하고 동결 뒤에만 시험 자료를 연다."""
    if role == "val":
        return VAL_INPUTS(role, sequence)
    assert sequence in prepare()["test"]
    return test_inputs.inputs(role, sequence)


def activate() -> dict:
    """동일 평가 코드에 시험 자료 연결만 공급한다."""
    spec = prepare()
    pipeline.OUT = OUT
    run.OUT = OUT
    pipeline.prepare = prepare
    run.inputs = inputs
    return spec


def attach_reference(role: str, sequence: str) -> None:
    """공통 질의·라벨을 확인하고 방향 그룹과 출처를 연결한다."""
    spec = prepare()
    name = sequence.replace("/", "_")
    folder = OUT / role / name
    target = folder / "reference_validation.json"
    if target.exists():
        assert run.READ(target)["status"] == "passed"
        return
    refpath = (
        PARENT / "val" / name / "identity.npz"
        if role == "val"
        else run.power.OUT / "test/s1" / (name + "_p050__aux_half__equal.npz")
    )
    reference = run.ARR(refpath)
    if role == "test":
        _, _, _, _, _, _, _, inputpaths = inputs(role, sequence)
    else:
        inputpaths = ()
    hashes = {str(p.relative_to(ROOT)): run.HASH(p) for p in (*inputpaths, refpath)}
    for tag in spec["cases"]:
        path = folder / (tag + ".npz")
        row = run.ARR(path)
        for field in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(row[field], reference[field])
        record = run.READ(path.with_suffix(".json"))
        if role == "val":
            old = run.ARR(PARENT / "val" / name / (tag + ".npz"))
            for field in ("query_idx", "j_star", "top1_geo_dist"):
                np.testing.assert_array_equal(row[field], old[field])
            np.testing.assert_allclose(row["score"], old["score"], atol=8e-6, rtol=0)
        elif not np.array_equal(row["group"], reference["group"]):
            row["group"] = reference["group"]
            np.savez_compressed(path, **row)
            record["sha256"] = run.HASH(path)
            record["group_source_sha256"] = run.HASH(refpath)
            run.WRITE(path.with_suffix(".json"), record)
    run.WRITE(
        target,
        dict(
            status="passed",
            queries=len(reference["query_idx"]),
            paired_labels=True,
            input_sha256=hashes,
        ),
    )


def worker(role: str, sequence: str) -> None:
    """동일 검색·최종 판정을 실행한 뒤 공통 원장을 대조한다."""
    activate()
    pipeline.worker(role, sequence)
    attach_reference(role, sequence)


def finish() -> None:
    """최종 결과를 시퀀스→도시6→시드3 순서로 집계한다."""
    spec = prepare()
    per = {tag: {} for tag in spec["cases"]}
    for seed in (1, 2, 3):
        for label in ("bev_native", "bev_temporal", "manuscript"):
            per[f"{label}_s{seed}"] = {}
    count = 0
    for sequence in spec["test"]:
        name = sequence.replace("/", "_")
        folder = OUT / "test" / name
        meta = run.READ(folder / "summary.json")
        refcheck = run.READ(folder / "reference_validation.json")
        assert refcheck["status"] == "passed"
        count += refcheck["queries"]
        reference = run.ARR(folder / "identity.npz")
        for tag in spec["cases"]:
            per[tag][sequence] = meta["metrics"][tag]
        for seed in (1, 2, 3):
            bpath = BEV / f"test/s{seed}" / (name + ".npz")
            bev = run.ARR(bpath)
            assert run.HASH(bpath) == run.READ(bpath.with_suffix(".json"))["sha256"]
            mpath = (
                run.power.OUT / f"test/s{seed}" / (name + "_p050__aux_half__equal.npz")
            )
            manuscript = run.ARR(mpath)
            for row in (bev, manuscript):
                for field in ("query_idx", "min_past_gt_dist"):
                    np.testing.assert_array_equal(row[field], reference[field])
            for label, row, field in (
                ("bev_native", bev, "score_negl2"),
                ("bev_temporal", bev, "score"),
                ("manuscript", manuscript, "score"),
            ):
                per[f"{label}_s{seed}"][sequence] = run.base.independent_metrics(
                    row, row[field]
                )
    assert count == 195726
    results = {
        tag: dict(
            **run.current.adoption.previous.macro(rows, run.current.CITY),
            sequences=rows,
        )
        for tag, rows in per.items()
    }
    means, stds = {}, {}
    for kind in (
        "identity",
        "fixed",
        "diff",
        "abs_kv",
        "point_gate",
        "convs_only",
        "other_only",
        "bev_native",
        "bev_temporal",
        "manuscript",
    ):
        tags = (
            [kind]
            if kind in ("identity", "fixed")
            else [f"{kind}_s{s}" for s in (1, 2, 3)]
        )
        means[kind] = {
            m: float(np.mean([results[t]["macro"][m] for t in tags]))
            for m in run.base.METRICS
        }
        stds[kind] = {
            m: (
                float(np.std([results[t]["macro"][m] for t in tags], ddof=1))
                if len(tags) > 1
                else 0.0
            )
            for m in run.base.METRICS
        }
    delta = {
        k: {m: 100 * (means["diff"][m] - v[m]) for m in run.base.METRICS}
        for k, v in means.items()
        if k != "diff"
    }
    checks = {
        "diff_vs_" + c: means["diff"]["AP"] >= means[c]["AP"] + 0.001
        for c in ("fixed", "abs_kv")
    }
    for kind in ("diff", "convs_only"):
        checks[kind + "_positive_two_seeds"] = (
            sum(
                results[f"{kind}_s{s}"]["macro"]["AP"] > means["fixed"]["AP"]
                for s in (1, 2, 3)
            )
            >= 2
        )
    comparable = {
        c: {m: delta[c][m] >= -tol for m, tol in spec["working_tolerances_pp"].items()}
        for c in ("bev_native", "bev_temporal")
    }
    report = dict(
        per_case=results,
        mean=means,
        std=stds,
        diff_delta_pp=delta,
        learning_checks=checks,
        working_comparability=comparable,
        aggregation="46시퀀스중양성45→6도시→3사전학습시드평균",
        queries_per_case=count,
        test_used_for_selection=False,
        historical_test_exposure=True,
    )
    run.WRITE(OUT / "summary.json", report)
    print("TEST", means, "CHECKS", checks, "COMPARABLE", comparable, flush=True)
    verifier.verify(OUT, "test")
    run.WRITE(OUT / "status.json", dict(status="complete"))


def campaign() -> None:
    """검증 재현 뒤 세 작업씩 시험하며 설정을 바꾸지 않는다."""
    spec = prepare()
    worker("val", "kitti360/0004")
    run.WRITE(
        OUT / "preflight.json",
        dict(status="passed", cases=len(spec["cases"]), test_used=False),
    )
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)

    def call(sequence):
        with (logs / (sequence.replace("/", "_") + ".log")).open("a") as stream:
            subprocess.run(
                [sys.executable, __file__, "worker", "--sequence", sequence],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(call, s) for s in spec["test"]]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            run.WRITE(
                OUT / "status.json", dict(status="running", done=i, total=len(futures))
            )
            print("TEST_SEQUENCE", i, len(futures), flush=True)
    finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "worker", "finish", "run"))
    p.add_argument("--sequence", default="kitti/00")
    a = p.parse_args()
    pipeline.e.configure()
    if a.action == "worker":
        worker("test", a.sequence)
    elif a.action == "run":
        campaign()
    else:
        globals()[a.action]()
