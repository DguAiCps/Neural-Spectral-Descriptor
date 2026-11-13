#!/usr/bin/env python3
"""고정 후보 경로에서 중심 거절 완화와 최종 점수 상승 제한을 비교한다."""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_candidate_paths as old
from _pdv2_no_context_ablation import sha256, write_json
from _pr_eval_lib import classify_queries, operating_point
from retrieval.path_confidence import confidence_variants

OUT = REPO / "results/pdv2_path_confidence_20261021"
CONFIG = REPO / "configs/pdv2_path_confidence.yaml"
CFG = yaml.safe_load(CONFIG.read_text())


def read(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return old.exp.read(path)


def protocol(inputs: bool = False) -> dict:
    """고정 코드와 필요할 때 모든 입력을 확인한다."""
    spec = read(OUT / "protocol.json")
    for field in (("runtime", "inputs") if inputs else ("runtime",)):
        for p, h in spec[field].items():
            assert sha256(REPO / p) == h, p
    return spec


def prepare() -> None:
    """새 점수를 보기 전에 대조·수식·선택 기준을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        protocol(True)
        return
    previous = old.protocol(True)
    audit = read(old.OUT / "completion_validation.json")
    assert audit["status"] == "passed" and audit["prefix_winner_changes"] == 0
    for p, h in audit["sources"].items():
        assert sha256(REPO / p) == h, p
    runtime = dict(previous["runtime"])
    for p in (Path(__file__), CONFIG, REPO / "src/retrieval/path_confidence.py",
              REPO / "tests/retrieval/test_path_confidence.py"):
        runtime[str(p.relative_to(REPO))] = sha256(p)
    inputs = {p: h for p, h in audit["sources"].items() if p not in runtime}
    for p in (old.OUT / "summary.json", old.OUT / "completion_validation.json",
              old.OUT / "high_precision_diagnosis.json"):
        inputs[str(p.relative_to(REPO))] = sha256(p)
    write_json(OUT / "protocol.json", {
        "created_unix": time.time(), "config": CFG, "val": previous["val"],
        "runtime": runtime, "inputs": inputs, "test_used": False,
        "scope": "검증9·3시드·표준/용량투영의 기존 후보 경로와 체크포인트 고정. 재학습·새 신경망·검색 후보 변경 없음. 이전 결과를 본 뒤 설계한 후속 탐색.",
        "baseline": "baseline=경로전 선택,hard_null=기존경로 중심거절,forced=기존경로 중심강제. 세기준의 저장원장을 비트단위로 재사용하고 별도 점수를 검산한다.",
        "change": "새4조건은 forced 승자 유지. baseline_score=원래승자점수를그대로이전; soft_null=S_forced*exp(min(null_margin,0)); capped=min(S_forced,S_original); soft_capped=capped*exp(min(null_margin,0))(주조건). 척도1은고정이며검증튜닝없음. 새로운nullthreshold없음.",
        "meaning": "soft는후보를영구거절하지않고경로부족만반영한다. capped는각질의의기존최종점수상한으로점수상승을막는다. 후보가바뀌므로오수락증가방지·확률보정·AP단조개선보장은없다.",
        "availability": "기존/새승자모두q+15내기존arc_unique_b100으로계산가능. 원래승자판정추가계산비용이있지만이미필요한스펙트럼과관측만사용. GT는사후평가지표만.",
        "selection": "세시드모두공개. 주조건standard/soft_capped가각시드의forced,hard_null각각보다AP+.1%p/R100손실<=.1%p/계열AP손실<=.25%p일때만후속주후보. 다른조건사후승격·시험·원고교체없음.",
        "diagnostic_basis": "기존hard_null은forced정답을시드별66/61/51개버린다. Sejong01 q12106/j11875의높아진판정점수가R100을낮춘다. 두현상의완화만대조하며새로운장소구별정보추가로주장하지않음.",
    })
    print("PREPARED", len(runtime), len(inputs), flush=True)


def bundle_path(seed: int, model: str, sequence: str) -> Path:
    """한 모델·시퀀스의 모든 대조 결과 위치다."""
    return OUT / f"s{seed}/{model}_{sequence.replace('/', '_')}.npz"


def row(bundle: dict, arm: str) -> dict:
    """행렬 원장에서 기존 최종 지표에 필요한 한 조건을 꺼낸다."""
    k = CFG["arms"].index(arm)
    return {"query_idx": bundle["query_idx"], "min_past_gt_dist": bundle["min_past_gt_dist"],
            "group": bundle["group"], "j_star": bundle["winners"][:, k],
            "top1_geo_dist": bundle["distances"][:, k], "score": bundle["scores"][:, k]}


def worker(seed: int, model: str, sequence: str, tau: float) -> dict:
    """같은 질의·승자를 유지하고 네 고정 점수를 새로 평가한다."""
    p = bundle_path(seed, model, sequence)
    p.parent.mkdir(exist_ok=True)
    source = {arm: old.exp.arrays(old.OUT / f"s{seed}/{model}_{arm}_{sequence.replace('/', '_')}.npz")
              for arm in ("baseline", "joint_null", "joint_forced")}
    base, hard, forced = (source[k] for k in ("baseline", "joint_null", "joint_forced"))
    for a in source.values():
        for k in ("query_idx", "min_past_gt_dist", "group"):
            np.testing.assert_array_equal(a[k], base[k])
    variants = confidence_variants(base["score"], forced["score"], forced["path_margin"])
    scores = {"baseline": base["score"], "hard_null": hard["score"], "forced": forced["score"], **variants}
    data = {k: base[k] for k in ("query_idx", "min_past_gt_dist", "group")}
    data["scores"] = np.column_stack([scores[a] for a in CFG["arms"]])
    data["winners"] = np.column_stack([(base if a == "baseline" else hard if a == "hard_null" else forced)["j_star"] for a in CFG["arms"]])
    data["distances"] = np.column_stack([(base if a == "baseline" else hard if a == "hard_null" else forced)["top1_geo_dist"] for a in CFG["arms"]])
    data["path_margin"] = forced["path_margin"]
    np.savez_compressed(p, **data)
    metrics, diagnostics = {}, {}
    qc = classify_queries(data["min_past_gt_dist"])
    for arm in CFG["arms"]:
        a = row(data, arm)
        metrics[arm] = old.exp.metrics(a["score"], a)
        valid = np.isfinite(a["score"])
        correct = qc["positive"] & (a["top1_geo_dist"] <= 5)
        diagnostics[arm] = {
            "returned": int(valid.sum()), "correct": int(correct.sum()),
            "fixed_tau": operating_point(a["score"], a["top1_geo_dist"], qc, tau),
            "directions": {str(g): int((correct & (a["group"] == g)).sum()) for g in (0, 1, 2)}}
        wrong = valid & (qc["positive"] | qc["negative"]) & ~correct
        if wrong.any():
            ix = np.flatnonzero(wrong)[np.argmax(a["score"][wrong])]
            diagnostics[arm]["first_wrong"] = {"q": int(a["query_idx"][ix]), "j": int(a["j_star"][ix]),
                "score": float(a["score"][ix]), "distance": float(a["top1_geo_dist"][ix])}
    negative_margin = data["path_margin"] < 0
    confidence_increase = forced["score"] > base["score"]
    rejection = {}
    for name, mask in (("correct", qc["positive"] & (forced["top1_geo_dist"] <= 5)),
                       ("wrong_positive", qc["positive"] & (forced["top1_geo_dist"] > 5)),
                       ("negative", qc["negative"])):
        rejection[name] = {"n": int(mask.sum()), "hard_rejected": int((mask & negative_margin).sum()),
                           "score_increased": int((mask & confidence_increase).sum())}
    record = {"seed": seed, "model": model, "sequence": sequence, "metrics": metrics,
              "diagnostics": diagnostics, "mechanism": rejection, "sha256": sha256(p),
              "test_used": False}
    write_json(p.with_suffix(".json"), record)
    return record


def report(records: dict) -> dict:
    """3시드 평균·표준편차와 사전 주조건의 조건부 채택 여부를 기록한다."""
    per_seed = {}
    for seed in CFG["seeds"]:
        per_seed[str(seed)] = {}
        for model in CFG["models"]:
            per_seed[str(seed)][model] = {
                arm: old.macro({s: r["metrics"][arm] for s, r in records[str(seed)][model].items()})
                for arm in CFG["arms"]}
    mean_std = {model: {arm: {m: {
        "mean": float(np.mean([per_seed[str(s)][model][arm]["macro"][m] for s in CFG["seeds"]])),
        "std": float(np.std([per_seed[str(s)][model][arm]["macro"][m] for s in CFG["seeds"]], ddof=1))}
        for m in old.exp.METRICS} for arm in CFG["arms"]} for model in CFG["models"]}
    checks = {}
    for seed in CFG["seeds"]:
        chosen = per_seed[str(seed)][CFG["primary_model"]][CFG["primary_arm"]]
        checks[str(seed)] = {}
        for arm in CFG["promotion"]["comparators"]:
            reference = per_seed[str(seed)][CFG["primary_model"]][arm]
            checks[str(seed)][arm] = {
                "AP": chosen["macro"]["AP"] >= reference["macro"]["AP"]+.001,
                "R100": chosen["macro"]["R@100P"] >= reference["macro"]["R@100P"]-.001,
                "family_AP": all(chosen["families"][f]["AP"] >= reference["families"][f]["AP"]-.0025 for f in reference["families"])}
    accepted = all(all(v.values()) for s in checks.values() for v in s.values())
    result = {"per_seed": per_seed, "mean_std": mean_std, "promotion_checks": checks,
              "primary_eligible": accepted, "test_used": False, "protocol_sha256": sha256(OUT / "protocol.json")}
    write_json(OUT / "summary.json", result)
    with (OUT / "final_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "arm", *old.exp.METRICS, "AP_std"])
        writer.writeheader()
        for model in CFG["models"]:
            for arm in CFG["arms"]:
                value = {"model": model, "arm": arm,
                         **{m: 100*mean_std[model][arm][m]["mean"] for m in old.exp.METRICS},
                         "AP_std": 100*mean_std[model][arm]["AP"]["std"]}
                writer.writerow(value)
                print("FINAL", model, arm, {m: round(value[m], 4) for m in old.exp.METRICS}, flush=True)
    print("PRIMARY_ELIGIBLE", accepted, flush=True)
    return result


def main() -> None:
    """고정된 모든 조건을 재학습 없이 평가한다."""
    started = time.perf_counter()
    prepare()
    spec = protocol()
    records = {}
    for seed in CFG["seeds"]:
        records[str(seed)] = {}
        tau = old.fixed.choose_tau(seed)["tau"]
        for model in CFG["models"]:
            records[str(seed)][model] = {}
            for sequence in spec["val"]:
                records[str(seed)][model][sequence] = worker(seed, model, sequence, tau)
            write_json(OUT / "status.json", {"status": "running", "seed": seed, "model": model})
            print("DONE", seed, model, flush=True)
    report(records)
    write_json(OUT / "status.json", {"status": "evaluation_complete", "elapsed_s": time.perf_counter()-started,
                                     "seeds": CFG["seeds"], "test_used": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "prepare"))
    args = parser.parse_args()
    if args.action == "run":
        main()
    else:
        prepare()
