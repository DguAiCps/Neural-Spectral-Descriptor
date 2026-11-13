#!/usr/bin/env python3
"""NSD 최종 후보를 고정한 신호처리 판정 대조.

docker exec -w /workspace/Neural-Spectral-Codec ring_baseline \
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 scripts/_pdv2_spectral_evidence.py run
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal_focus as old
from _pdv2_no_context_ablation import sha256, write_json
from _pr_eval_lib import classify_queries, pr_sweep
from retrieval.spectral_evidence import temporal_scores

CONFIG = REPO / "configs/pdv2_spectral_evidence.yaml"
CFG = yaml.safe_load(CONFIG.read_text())
OUT = REPO / "results/pdv2_spectral_evidence_20261018"
CAP = REPO / "results/pdv2_temporal_capacity_20261018"
OP = CFG["op"]
METRICS = ("AP", "max_F1", "R@100P", "R@1")


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def arrays(path: Path) -> dict:
    """수치 배열만 읽는다."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files if k != "meta"}


def protocol(data: bool = False) -> dict:
    """새 실험의 동결 코드와 입력을 확인한다."""
    value = read(OUT / "protocol.json")
    for field in ("runtime_sha256", "input_sha256") if data else ("runtime_sha256",):
        for name, expected in value[field].items():
            assert sha256(REPO / name) == expected, name
    return value


def prepare() -> None:
    """점수 후보·3시드·NSD 선택 기준을 추가 평가 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    original = old.protocol(True)
    runtime = dict(original["runtime_sha256"])
    for path in (Path(__file__), CONFIG, REPO / "src/retrieval/spectral_evidence.py",
                 REPO / "tests/retrieval/test_spectral_evidence.py"):
        runtime[str(path.relative_to(REPO))] = sha256(path)
    inputs, selected = {}, {}
    for name, root, arm in (("standard", old.OUT, "standard_original"),
                            ("capacity_point", CAP, "point_original")):
        assert read(root / "completion_validation.json")["status"] == "passed"
        summary = read(root / "summary.json")
        inputs[str((root / "summary.json").relative_to(REPO))] = sha256(root / "summary.json")
        selected[name] = {}
        for seed in CFG["seeds"]:
            rows = summary["per_seed"][str(seed)][arm]["candidates"]
            best = max(row["ops"][OP]["macro"]["AP"] for row in rows)
            chosen = min((r for r in rows if best-r["ops"][OP]["macro"]["AP"] <= 1e-12), key=lambda r: r["epoch"])
            folder = root / f"s{seed}/{arm}/epoch{chosen['epoch']}"
            selected[name][str(seed)] = {"folder": str(folder.relative_to(REPO)),
                "epoch": chosen["epoch"], "raw_metrics": chosen["ops"][OP]["macro"]}
            confidence = old.base.BASE / f"s{seed}/confidence_selection.json"
            inputs[str(confidence.relative_to(REPO))] = sha256(confidence)
            for sensor, sequence in old.base.VAL:
                path = folder / f"val_{OP}_{sensor}_{sequence}.npz"
                inputs[str(path.relative_to(REPO))] = sha256(path)
                path = folder / f"keys_val_{sensor}_{sequence}.npz"
                inputs[str(path.relative_to(REPO))] = sha256(path)
    for sensor, sequence in old.base.VAL:
        _, _, paths, _ = old.base.inputs("val", sensor + "/" + sequence)
        for path in paths.values():
            inputs[str(path.relative_to(REPO))] = sha256(path)
    write_json(OUT / "protocol.json", {"created_unix": time.time(), "config": CFG,
        "selected": selected, "val": [s+"/"+q for s,q in old.base.VAL], "test_used": False,
        "scope": "NSD만의 검증9·3시드·두기존모델·6판정점수. 기존10/20후보 중NSD AP로선택하며 실제선택은 기존공동AP선택과같음. 최종후보·검색·융합은불변. 각시드 기존c_col/c_spec과 그 회전곡선을공유. 새점수는 수락임계값을훑어 AP/F1/R100 평가. 재학습·추가분류기·조건별새튜닝없음.",
        "temporal": "중심과±5키프레임, 정/역방향두가설, 후보중심j<=q-30. 관측은q+15까지. 단순평균/공통회전/주행yaw보정공통회전/PSR를분리. 투영+시간점수도시간문맥을쓰므로무문맥대조라고하지않음.",
        "novelty": "PSR·시간평균·주행보정의도입만으로신규성주장안함. coherent는광학LiDAR위상이아닌BEV영상의각도푸리에위상. CFAR확률보장·독립증거가산·독립시험확증이아님.",
        "runtime_sha256": runtime, "input_sha256": inputs})
    print("PREPARED", len(runtime), len(inputs), flush=True)


def metrics(score: np.ndarray, ledger: dict) -> dict:
    """기존 최종 PR 평가와 같은 정의를 사용한다."""
    qc = classify_queries(ledger["min_past_gt_dist"])
    result = pr_sweep(score, ledger["top1_geo_dist"], qc)
    return {**{m: float(result[m]) for m in METRICS[:3]}, "n_pos": qc["n_pos"],
            "R@1": float((ledger["top1_geo_dist"][qc["positive"]] <= 5).mean()) if qc["n_pos"] else None}


def truncate_bank(bank: tuple, n: int) -> tuple:
    """실제 가용 입력 접두 구간을 만든다."""
    return bank[0][:n], bank[1][:n], bank[2][:n], tuple(x[:n] for x in bank[3])


def worker(seed: int, sequence: str) -> None:
    """같은 입력에서 두 모델의 고정 최종 후보를 평가한다."""
    spec = protocol()
    target = OUT / f"s{seed}/{sequence.replace('/', '_')}.json"
    if target.exists():
        return
    sensor, seq, paths, _ = old.base.inputs("val", sequence)
    cache = arrays(paths["cache"])
    with np.load(paths["spec"]) as z:
        np.testing.assert_array_equal(z["scan_ids"], cache["scan_ids"])
        bank = old.base.bank_for(OP, z)
    poses = cache["poses"]
    yaw = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
    conf = read(old.base.BASE / f"s{seed}/confidence_selection.json")["ops"][OP]
    result = {}
    for model in CFG["models"]:
        folder = REPO / spec["selected"][model][str(seed)]["folder"]
        source = folder / f"val_{OP}_{sensor}_{seq}.npz"
        ledger = arrays(source)
        q, j = ledger["query_idx"], ledger["j_star"]
        assert np.all(j <= q-30) and np.all(q < len(yaw)-15)
        chunks, tick = [], time.perf_counter()
        for start in range(0, len(q), CFG["batch_size"]):
            sl = slice(start, start+CFG["batch_size"])
            chunks.append(temporal_scores(bank, yaw, q[sl], j[sl], conf["decision"],
                CFG["temporal_offset"], CFG["psr_guard_bins"]))
        elapsed = time.perf_counter()-tick
        scores = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
        error = float(np.max(np.abs(scores["raw"]-ledger[conf["decision"]])))
        np.testing.assert_allclose(scores["raw"], ledger[conf["decision"]], atol=2e-7, rtol=2e-6)
        # 기존 동점 순서와 완전히 같은 기준 지표를 보존한다.
        scores["raw"] = ledger[conf["decision"]]
        assert all(np.isfinite(x).all() for x in scores.values())
        prefix_checks = []
        for i in (len(q)//3, 2*len(q)//3):
            end = int(q[i]+16)
            small = temporal_scores(truncate_bank(bank, end), yaw[:end], q[i:i+1], j[i:i+1], conf["decision"], CFG["temporal_offset"], CFG["psr_guard_bins"])
            full = temporal_scores(bank, yaw, q[i:i+1], j[i:i+1], conf["decision"], CFG["temporal_offset"], CFG["psr_guard_bins"])
            assert all(np.array_equal(small[k], full[k]) for k in full)
            prefix_checks.append({"q": int(q[i]), "end": end, "max_error": 0.})
        values = {k: ledger[k] for k in ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")}
        dest = OUT / f"s{seed}/{model}_{sensor}_{seq}.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, **values, **scores, meta=json.dumps({"source": str(source.relative_to(REPO)), "protocol_sha256": sha256(OUT / "protocol.json")}))
        result[model] = {"metrics": {method: metrics(scores[method], ledger) for method in CFG["methods"]},
            "ledger": str(dest.relative_to(REPO)), "sha256": sha256(dest), "queries": len(q),
            "raw_curve_max_error": error, "prefix_checks": prefix_checks,
            "shared_six_scores_runtime_s": elapsed, "source_channel": conf["decision"],
            "full_temporal_support_fraction": float(((scores["support_forward"]==3) & (scores["support_reverse"]==3)).mean())}
    write_json(target, {"seed": seed, "sequence": sequence, "results": result})
    print("SCORED", seed, sequence, flush=True)


def report() -> None:
    """최종 지표를 재계산하고 자료별·같은 시드 효과를 보고한다."""
    spec = protocol(True)
    values, details, mean_std = {}, {}, {}
    total_rows, prefix_checks, max_error = 0, 0, 0.
    for seed in CFG["seeds"]:
        records = {sequence: read(OUT / f"s{seed}/{sequence.replace('/', '_')}.json")["results"] for sequence in spec["val"]}
        values[str(seed)], details[str(seed)] = {}, {}
        for model in CFG["models"]:
            values[str(seed)][model], details[str(seed)][model] = {}, {}
            for sequence, record in records.items():
                row = record[model]
                path = REPO / row["ledger"]
                assert sha256(path) == row["sha256"]
                ledger = arrays(path)
                total_rows += len(ledger["query_idx"])
                prefix_checks += len(row["prefix_checks"])
                max_error = max(max_error, row["raw_curve_max_error"])
                for method in CFG["methods"]:
                    assert metrics(ledger[method], ledger) == row["metrics"][method]
            for method in CFG["methods"]:
                families = {}
                for family in ("kitti", "helipr", "mulran"):
                    selected = [r[model]["metrics"][method] for seq, r in records.items()
                                if seq.startswith(family) and r[model]["metrics"][method]["n_pos"]]
                    assert selected
                    families[family] = {m: float(np.mean([r[m] for r in selected])) for m in METRICS}
                macro = {m: float(np.mean([r[m] for r in families.values()])) for m in METRICS}
                values[str(seed)][model][method] = macro
                details[str(seed)][model][method] = families
                if method == "raw":
                    expected = spec["selected"][model][str(seed)]["raw_metrics"]
                    for m in METRICS:
                        assert abs(macro[m]-expected[m]) < 1e-12, (seed, model, m)
    for model in CFG["models"]:
        mean_std[model] = {method: {m: {"mean": float(np.mean([values[str(s)][model][method][m] for s in CFG["seeds"]])),
            "std": float(np.std([values[str(s)][model][method][m] for s in CFG["seeds"]], ddof=1))} for m in METRICS} for method in CFG["methods"]}
    delta = {model: {method: {m: [(values[str(s)][model][method][m]-values[str(s)][model]["raw"][m])*100 for s in CFG["seeds"]] for m in METRICS} for method in CFG["methods"]} for model in CFG["models"]}
    write_json(OUT / "summary.json", {"per_seed": values, "families": details, "mean_std": mean_std,
        "paired_delta_pp": delta, "test_used": False, "protocol_sha256": sha256(OUT / "protocol.json")})
    write_json(OUT / "completion_validation.json", {"status": "passed", "ledgers": 54,
        "rows": total_rows, "prefix_checks": prefix_checks, "raw_curve_max_error": max_error,
        "same_final_winners": True, "same_queries": True, "test_used": False})
    print("SUMMARY", mean_std, flush=True)


def run() -> None:
    """전체 검증만 실행하고 원고·시험 결과와 분리한다."""
    prepare()
    tick = time.time()
    logs = REPO / "logs/spectral_evidence_20261018"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, sequence) for seed in CFG["seeds"] for sequence in protocol()["val"]]

    def invoke(job: tuple) -> None:
        seed, sequence = job
        with (logs / f"s{seed}_{sequence.replace('/', '_')}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__)), "worker", "--seed", str(seed), "--sequence", sequence], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(invoke, job): job for job in jobs}
        for count, f in enumerate(concurrent.futures.as_completed(pending), 1):
            f.result()
            write_json(OUT / "status.json", {"status": "running", "done": count, "total": len(jobs), "test_used": False})
            print("DONE", count, len(jobs), pending[f], flush=True)
    report()
    write_json(OUT / "status.json", {"status": "complete", "runtime_s": time.time()-tick, "test_used": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "report", "run"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.seed, args.sequence)
    else:
        globals()[args.action]()
