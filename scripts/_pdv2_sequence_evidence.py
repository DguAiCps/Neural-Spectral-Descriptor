#!/usr/bin/env python3
"""NSD의 관측·표현을 고정하고 시퀀스 판정의 대응·중복·비중을 비교한다.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/_pdv2_sequence_evidence.py
"""
from __future__ import annotations

import concurrent.futures
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_evidence as exp
from _pdv2_no_context_ablation import sha256, write_json
from retrieval.sequence_evidence import sequence_scores

OUT = REPO / "results/pdv2_sequence_evidence_20261018"
CONFIG = REPO / "configs/pdv2_sequence_evidence.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
BASELINE = "arc_sparse_b25"
FIELDS = ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")


def prepare() -> dict:
    """원천·수치 기준·방법군을 새 평가 전에 동결한다."""
    spec = exp.protocol(True)
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = {str(p.relative_to(REPO)): sha256(p) for p in (
        Path(__file__), CONFIG, REPO / "src/retrieval/sequence_evidence.py",
        REPO / "tests/retrieval/test_sequence_evidence.py")}
    previous = OUT / "protocol.json"
    if previous.exists():
        assert exp.read(previous)["runtime_sha256"] == runtime
        return spec
    write_json(previous, {"created_unix": time.time(), "config": CFG, "runtime_sha256": runtime,
        "input_protocol_sha256": sha256(exp.OUT / "protocol.json"), "test_used": False,
        "scope": "NSD만 검증9·3시드·두모델. 기존최종승자·시드별판정채널고정. q±5·j±10·q+15판정. 중심/2이웃의.25/.5/1혼합에맞춰밀집10이웃총비중을환산. 전체21조건공개. 거리개선관찰뒤후속탐색.",
        "methods": "2이웃거리,10이웃프레임,10이웃거리,양끝거리로정한구간별일정속도,8개고정속도가설,거리대응의중복후보가중치. 방향별25%도보고. 중복가중치는같은후보그룹의합1이며유효이웃수감소도포함.",
        "fairness": "표현·관측범위동일. 희소최대5쌍,밀집두가설최대21쌍,속도8가설은더많은쌍. 비용을같다고하지않고실제고유쌍수를보고. 원래SeqSLAM의영상·전역정규화를재현한공식결과아님.",
        "selection": "세시드각각 기존거리25%보다AP>=.1%p,R100손실<=.1%p인조건만후보. 두모델의합격조건중평균AP최고를선택. 여기서는시험을실행하지않음.",
        "target": "BEVPlace++와같은최종평가에서NSD우위. 기존시험표격차를검증76점대와직접비교하지않음."})
    return spec


def worker(sequence: str) -> None:
    """한 시퀀스의 모든 대조 점수와 실제 가용 입력 검사를 저장한다."""
    target = OUT / (sequence.replace("/", "_")+".json")
    if target.exists():
        return
    spec = exp.protocol()
    sensor, seq, paths, _ = exp.old.base.inputs("val", sequence)
    cache = exp.arrays(paths["cache"])
    xyz = cache["poses"][:, :3, 3]
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(cache["scan_ids"], z["scan_ids"])
        bank = exp.old.base.bank_for(exp.OP, z)
    results = {}
    for seed in ("1", "2", "3"):
        ch = exp.read(exp.old.base.BASE / f"s{seed}/confidence_selection.json")["ops"][exp.OP]["decision"]
        results[seed] = {}
        for model in CFG["models"]:
            source = exp.OUT / f"s{seed}/{model}_{sensor}_{seq}.npz"
            a = exp.arrays(source)
            q, j = a["query_idx"], a["j_star"]
            assert (j <= q-30).all() and (q < len(travel)-15).all()
            chunks, costs = [], []
            tick = time.perf_counter()
            for start in range(0, len(q), CFG["batch_size"]):
                sl = slice(start, start+CFG["batch_size"])
                score, cost = sequence_scores(bank, travel, q[sl], j[sl], ch, a["raw"][sl])
                chunks.append(score)
                costs.append(cost)
            runtime = time.perf_counter()-tick
            scores = {k: np.concatenate([r[k] for r in chunks]) for k in chunks[0]}
            old = exp.arrays(REPO / f"results/pdv2_arc_evidence_20261018/s{seed}/{model}_{sensor}_{seq}.npz")
            np.testing.assert_array_equal(scores[BASELINE], old["arc_match10"])
            prefix = []
            for i in (len(q)//3, 2*len(q)//3):
                end = int(q[i]+16)
                small, _ = sequence_scores(exp.truncate_bank(bank, end), travel[:end],
                                            q[i:i+1], j[i:i+1], ch, a["raw"][i:i+1])
                full, _ = sequence_scores(bank, travel, q[i:i+1], j[i:i+1], ch, a["raw"][i:i+1])
                for key in scores:
                    np.testing.assert_array_equal(small[key], full[key])
                    np.testing.assert_array_equal(small[key], scores[key][i:i+1])
                prefix.append({"q": int(q[i]), "max_error": 0.})
            path = OUT / f"s{seed}/{model}_{sensor}_{seq}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{f: a[f] for f in FIELDS}, **scores)
            cost_summary = {}
            for name in costs[0]:
                pair = np.concatenate([c[name]["pair_count"] for c in costs])
                count = np.concatenate([c[name]["effective_neighbors"].ravel() for c in costs])
                cost_summary[name] = {"pairs_mean": float(pair.mean()), "pairs_max": int(pair.max()),
                                      "effective_neighbors_mean": float(count.mean())}
            results[seed][model] = {"metrics": {k: exp.metrics(v, a) for k, v in scores.items()},
                "ledger": str(path.relative_to(REPO)), "sha256": sha256(path), "source_sha256": sha256(source),
                "prefix_checks": prefix, "all_conditions_runtime_s": runtime, "pair_counts": cost_summary}
            print("SEQUENCE", sequence, seed, model, round(runtime, 2), flush=True)
    write_json(target, {"results": results, "sequence": sequence, "test_used": False})


def report(spec: dict) -> None:
    """최종 원장을 재계산해 모든 조건의 집계와 같은 시드 차이를 보고한다."""
    records = {seq: exp.read(OUT / (seq.replace("/", "_")+".json"))["results"] for seq in spec["val"]}
    first = next(iter(records.values()))["1"][CFG["models"][0]]["metrics"]
    methods = list(first)
    values, families, ledgers, rows, prefixes = {}, {}, 0, 0, 0
    for seed in ("1", "2", "3"):
        values[seed], families[seed] = {}, {}
        for model in CFG["models"]:
            values[seed][model], families[seed][model] = {}, {}
            for seq, record in records.items():
                item = record[seed][model]
                path = REPO / item["ledger"]
                assert sha256(path) == item["sha256"]
                a = exp.arrays(path)
                source = exp.arrays(exp.OUT / f"s{seed}/{model}_{seq.replace('/', '_')}.npz")
                for field in FIELDS:
                    np.testing.assert_array_equal(a[field], source[field])
                for method in methods:
                    assert np.isfinite(a[method]).all()
                    assert exp.metrics(a[method], a) == item["metrics"][method]
                ledgers += 1
                rows += len(a["query_idx"])
                prefixes += len(item["prefix_checks"])
            for method in methods:
                group = {}
                for name in ("kitti", "helipr", "mulran"):
                    selected = [r[seed][model]["metrics"][method] for seq, r in records.items()
                                if seq.startswith(name) and r[seed][model]["metrics"][method]["n_pos"]]
                    group[name] = {m: float(np.mean([r[m] for r in selected])) for m in exp.METRICS}
                families[seed][model][method] = group
                values[seed][model][method] = {m: float(np.mean([r[m] for r in group.values()])) for m in exp.METRICS}
    means, delta, eligible = {}, {}, []
    for model in CFG["models"]:
        means[model], delta[model] = {}, {}
        for method in methods:
            means[model][method] = {m: {"mean": float(np.mean([values[s][model][method][m] for s in values])),
                "std": float(np.std([values[s][model][method][m] for s in values], ddof=1))} for m in exp.METRICS}
            delta[model][method] = {m: [100*(values[s][model][method][m]-values[s][model][BASELINE][m]) for s in values] for m in exp.METRICS}
            if (min(delta[model][method]["AP"]) >= 100*CFG["minimum_paired_ap_gain"] and
                    min(delta[model][method]["R@100P"]) >= -100*CFG["maximum_paired_r100_loss"]):
                eligible.append({"model": model, "method": method, "AP": means[model][method]["AP"]["mean"]})
    selected = max(eligible, key=lambda r: r["AP"]) if eligible else None
    write_json(OUT / "summary.json", {"per_seed": values, "families": families, "mean_std": means,
        "paired_delta_pp": delta, "eligible": eligible, "selected": selected, "test_used": False})
    write_json(OUT / "completion_validation.json", {"status": "passed", "ledgers": ledgers,
        "rows": rows, "prefix_checks": prefixes, "prefix_max_error": 0., "baseline_exact": True, "test_used": False})
    print("SELECTED", selected, flush=True)
    for model in means:
        print("RANK", model, sorted([(m, means[model][m]["AP"]["mean"], means[model][m]["R@100P"]["mean"]) for m in methods], key=lambda x:-x[1]), flush=True)


def main() -> None:
    """두 시퀀스씩 실행하고 성공·실패 조건을 모두 보존한다."""
    spec = prepare()
    tick = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(worker, sequence): sequence for sequence in spec["val"]}
        for number, future in enumerate(concurrent.futures.as_completed(pending), 1):
            future.result()
            write_json(OUT / "status.json", {"status": "running", "done": number, "total": 9, "test_used": False})
    report(spec)
    write_json(OUT / "status.json", {"status": "complete", "runtime_s": time.time()-tick, "test_used": False})


if __name__ == "__main__":
    main()
