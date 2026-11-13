#!/usr/bin/env python3
"""검증에서 고정한 크기 변환·융합·판정 한 조건만 시험한다."""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import _pdv2_spectral_power as study
import _pdv2_minimal_rerank_test as report

base, cal = study.base, study.cal
OUT, REPO, white = study.OUT, study.REPO, base.white
METRICS = base.METRICS
read, arrays, sha256, write_json = study.read, study.arrays, study.sha256, study.write_json
independent_metrics = base.independent_metrics


def freeze() -> dict:
    """검증 기준을 통과한 한 조건과 실행·입력 해시를 고정한다."""
    target = OUT / "test_freeze.json"
    if target.exists():
        spec = read(target)
        for p, h in spec["sources"].items():
            assert sha256(REPO / p) == h, p
        return spec
    protocol = study.prepare()
    result = read(OUT / "summary.json")
    selected = result["selected"]
    assert selected and result["decisions"][selected]["eligible"], "검증 승격 기준 미달"
    assert read(OUT / "completion_validation.json")["status"] == "passed"
    previous = read(white.OUT / "fixed_test/freeze.json")
    paths = {Path(__file__), Path(report.__file__), OUT / "summary.json", OUT / "protocol.json",
             OUT / "completion_validation.json", cal.OUT / "summary.json",
             white.OUT / "fixed_test/freeze.json", white.OUT / "fixed_test/table2_summary.json",
             REPO / "results/pdv2_baseline_seeds/summary.json"}
    for seed in protocol["seeds"]:
        for sequence in previous["test"]:
            name = sequence.replace("/", "_") + ".npz"
            paths.add(base.reference_path(seed, "test", name))
            paths.add(base.source_folder(seed, "test", protocol) / ("keys_" + name))
    spec = {"created_unix": time.time(), "selected": selected,
            "test": previous["test"], "cities": previous["cities"], "seeds": protocol["seeds"],
            "scope": "검증108조건에서고정한한조건과기존원고NSD만시험. 시험성적으로설정재선택없음. 195726질의/시드,45양성시퀀스→6도시→3시드. 원고자동교체없음. 기존개발노출이있는시험이며미노출확증이아님.",
            "sources": {**protocol["sources"], **{str(p.relative_to(REPO)): sha256(p) for p in paths}}}
    write_json(target, spec)
    return spec


def worker(seed: int, sequence: str, role: str = "test") -> None:
    """선택된 설정만 재생성하고 현재 최종 판정과 함께 검산한다."""
    frozen = read(OUT / "test_freeze.json")
    assert sha256(Path(__file__)) == frozen["sources"][str(Path(__file__).relative_to(REPO))]
    spec = read(OUT / "protocol.json")
    selected = frozen["selected"]
    name_power, fusion, channel = selected.split("__")
    power = study.POWERS.get(name_power, 1.)
    fi = cal.FUSIONS.index(fusion)
    name = sequence.replace("/", "_")
    folder = OUT / ("test" if role == "test" else "validation_replay") / f"s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (name + ".json")
    if target.exists():
        assert read(target)["status"] == "passed"
        return
    ref_path = base.reference_path(seed, role, name + ".npz")
    keys_path = base.source_folder(seed, role, spec) / ("keys_" + name + ".npz")
    ref, keys = arrays(ref_path), arrays(keys_path)
    _, _, paths, _ = white.prior.BASE.exp.old.base.inputs(role, sequence)
    cache = arrays(paths["cache"])
    np.testing.assert_array_equal(cache["scan_ids"], keys["scan_ids"])
    x = white.coefficients(paths["spec"])
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
    wx = white.transform(x, arrays(white.OUT / "statistics.npz")["full"])
    original, corrected = base.variant_inputs(x, 8), base.variant_inputs(wx, 8)
    new_bank = base.variant_inputs(study.transform(x, power), 8)
    new_corrected = base.variant_inputs(study.transform(wx, power), 8)
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    pos = cache["poses_gt"][:, :3, 3]
    conf, q = spec["confidence"][str(seed)], ref["query_idx"]
    weights = cal.fusion_weights(conf)[[0, fi]]
    tick = time.perf_counter()
    ids = np.asarray([cal.candidates(keys["embedding"], original, weights, int(t)) for t in q])
    np.testing.assert_array_equal(ids[:, 0], ref["j_star"])
    assert (ids >= 0).all() and (ids <= q[:, None] - 30).all()
    baseline = base.scores_for(original, corrected, travel, q, ids[:, 0], conf["decision"])
    np.testing.assert_array_equal(baseline, ref["score"])
    endpoint = {c: base.scores_for(new_bank, new_corrected, travel, q, ids[:, 1], c)
                for c in ("c_spec", "c_col")}
    value = cal.confidence(endpoint["c_spec"], endpoint["c_col"], channel, conf["decision"])
    rows, checks = {}, []
    for arm, winner, score in (("baseline", ids[:, 0], baseline), (selected, ids[:, 1], value)):
        row = {k: ref[k] for k in base.FIELDS}
        row.update(j_star=winner, top1_geo_dist=np.linalg.norm(pos[q] - pos[winner], axis=1), score=score)
        metrics = independent_metrics(row, score)
        other = white.prior.BASE.exp.metrics(score, row)
        for m in METRICS:
            if metrics["n_pos"]:
                np.testing.assert_allclose(metrics[m], other[m], atol=1e-12, rtol=0)
        if arm == "baseline":
            assert independent_metrics(ref, ref["score"]) == metrics
        ledger = folder / f"{name}_{arm}.npz"
        np.savez_compressed(ledger, **row)
        rows[arm] = {"path": str(ledger.relative_to(REPO)), "sha256": sha256(ledger), "metrics": metrics}
    if role == "val":
        old = arrays(cal.OUT / "val" / f"s{seed}" / (name + ".npz"))
        np.testing.assert_array_equal(ids[:, 1], old["candidates"][:, fi])
        stored = old if name_power == "p100" else arrays(OUT / "val" / f"s{seed}" / (name + ".npz"))
        prefix = "" if name_power == "p100" else name_power + "_"
        expected = cal.confidence(stored[prefix + "c_spec"][:, fi], stored[prefix + "c_col"][:, fi], channel, conf["decision"])
        np.testing.assert_array_equal(value, expected)
    for i in (len(q) // 3, 2 * len(q) // 3):
        end, t = int(q[i]) + 16, int(q[i])
        small = white.prior.BASE.exp.truncate_bank(original, end)
        np.testing.assert_array_equal(cal.candidates(keys["embedding"][:end], small, weights, t), ids[i])
        b = base.variant_inputs(study.transform(x[:end], power), 8)
        c = base.variant_inputs(study.transform(wx[:end], power), 8)
        ss = {ch: base.scores_for(b, c, travel[:end], q[i:i+1], ids[i:i+1, 1], ch)
              for ch in ("c_spec", "c_col")}
        v = cal.confidence(ss["c_spec"], ss["c_col"], channel, conf["decision"])
        np.testing.assert_array_equal(v, value[i:i+1])
        checks.append({"query": t, "winner_and_final_score_equal": True})
    positive = ref["min_past_gt_dist"] <= 5
    a = positive & (ref["top1_geo_dist"] <= 5)
    b = positive & (np.linalg.norm(pos[q] - pos[ids[:, 1]], axis=1) <= 5)
    write_json(target, {"status": "passed", "arms": rows, "rows": len(q), "prefix_checks": checks,
        "baseline_score_max_error": float(abs(baseline - ref["score"]).max()),
        "winner_changes": int((ids[:, 1] != ref["j_star"]).sum()),
        "gained_correct": int((b & ~a).sum()), "lost_correct": int((a & ~b).sum()),
        "elapsed_s": time.perf_counter() - tick,
        "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in (ref_path, keys_path, *paths.values())}})
    print("DONE", seed, sequence, role, round(time.perf_counter() - tick, 1), flush=True)


def finish() -> None:
    """검산된 공통 표2 집계기를 현재 원장 형식에 적용한다."""
    report.run = sys.modules[__name__]
    report.OUT, report.REPO, report.freeze = OUT, REPO, freeze
    report.finish()


def run() -> None:
    """검증 재현 후 동결한 하나의 설정만 시험46x3으로 실행한다."""
    frozen = freeze()
    # 시험 지표를 보기 전에 같은 코드가 검증 선택 결과를 재현하는지 확인한다.
    for seed in frozen["seeds"]:
        worker(seed, "kitti360/0004", "val")
    logs = study.LOG / "test"
    logs.mkdir(parents=True, exist_ok=True)
    def invoke(item: tuple) -> None:
        seed, seq = item
        with (logs / f"s{seed}_{seq.replace('/', '_')}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__)), "worker", "--seed", str(seed),
                            "--sequence", seq], stdout=log, stderr=subprocess.STDOUT, check=True)
    tick = time.time()
    jobs = [(s, q) for s in frozen["seeds"] for q in frozen["test"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(invoke, item): item for item in jobs}
        for n, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            write_json(OUT / "test_status.json", {"status": "running", "done": n, "total": len(jobs),
                                                   "elapsed_s": time.time() - tick})
            print("TEST", n, len(jobs), futures[f], flush=True)
    finish()
    write_json(OUT / "test_status.json", {"status": "complete", "elapsed_s": time.time() - tick})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "worker", "run", "finish"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sequence", default="kitti360/0004")
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.seed, args.sequence)
    else:
        globals()[args.action]()
