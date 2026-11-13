#!/usr/bin/env python3
"""소수 양성의 한 질의 단위를 고려한 학습 내부 비중 선택을 대조한다."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_context_place as experiment
from _pdv2_context_place_lib import FACTORS, PLACES, score
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / "results/pdv2_context_place_smallstep_20261018"


def prepare() -> dict:
    """학습 OOF의 정수 정답 손실 기준을 새 검증 점수 계산 전에 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/"protocol.json"
    if path.exists():
        data = experiment.old.exp.read(path)
        for p, expected in data["sources"].items():
            assert sha256(REPO/p) == expected
        return data
    sources = (Path(__file__), experiment.OUT/"protocol.json", experiment.OUT/"freeze.json",
               experiment.OUT/"oof_summary.json", REPO/"scripts/_pdv2_context_place_lib.py")
    data = {"created_unix": time.time(), "reason": "앞선학습OOF5%조건의엄격한0.1%p검사탈락이KITTI05정답한개에해당. 작은표본의정수해상도를고려하는별도탐색이며원주선택0%는바꾸지않음.",
        "selection": "문맥이웃만. 사전에정한0/5/10/25/50/100%중각시드OOF평균AP+.001이상/R100비감소,각양성시퀀스AP손실<=.001/R100정답수손실<=1,무양성장소오답수락률증가<=.001인양수중최소비율. 없으면0. 검증손익으로변경안함.",
        "validation": "학습내선택후freeze저장. 선택한한비율과기준만검증. encoder·후보·승자·기존전체학습적합계수동결. 시험없음.",
        "promotion": "기존검증기준유지:각시드AP+.001이상/R100손실<=.001,평균각계열AP손실<=.0025. 통과해도시험우위주장아님.",
        "caveat": "학습OOF를본뒤설계한후속진단. 문맥모델은모든학습장소를보았고,모델차원의장소분리검증이아님.",
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in sources}, "test_used": False}
    write_json(path, data)
    return data


def select() -> dict:
    """검증 자료 없이 최소 양수 비중을 고른다."""
    target = OUT/"freeze.json"
    if target.exists():
        return experiment.old.exp.read(target)
    data = experiment.old.exp.read(experiment.OUT/"oof_summary.json")
    grid = data["grids"]["context_arc"]
    evaluated = []
    for row in grid:
        factor = row["factor"]
        checks = []
        for index, seed in enumerate((1,2,3)):
            base = grid[0]["per_seed"][index]
            now = row["per_seed"][index]
            checks.append({"seed": seed, "scope": "macro", "pass": now["AP"] >= base["AP"]+.001 and now["R@100P"] >= base["R@100P"]})
            for place in PLACES:
                p = data["statistics"][str(seed)][place]["context_arc"]
                for sequence, current in p[str(factor)]["per_sequence"].items():
                    reference = p["0.0"]["per_sequence"][sequence]
                    n = current["n_pos"]
                    if n:
                        tp_loss = int(round((reference["R@100P"]-current["R@100P"])*n))
                        ok = current["AP"] >= reference["AP"]-.001 and tp_loss <= 1
                        checks.append({"seed": seed, "scope": sequence, "n_pos": n, "R100_tp_loss": tp_loss, "AP_delta": current["AP"]-reference["AP"], "pass": ok})
                    else:
                        delta = current["incorrect_accept"]/current["incorrect"]-reference["incorrect_accept"]/reference["incorrect"]
                        checks.append({"seed": seed, "scope": sequence, "negative_rate_delta": delta, "pass": delta <= .001})
        evaluated.append({"factor": factor, "pass": factor > 0 and all(c["pass"] for c in checks), "checks": checks})
    eligible = [r["factor"] for r in evaluated if r["pass"]]
    frozen = {"created_unix": time.time(), "factor": min(eligible) if eligible else 0.,
              "eligible_factors": eligible, "evaluated": evaluated, "test_used": False,
              "protocol_sha256": sha256(OUT/"protocol.json"),
              "oof_summary_sha256": sha256(experiment.OUT/"oof_summary.json"),
              "validation_evaluated_before_freeze": False}
    write_json(target, frozen)
    print("SMALLSTEP_FROZEN", frozen["factor"], eligible, flush=True)
    return frozen


def main() -> None:
    """학습 내부 선택 한 개만 검증하고 별도 결과를 보존한다."""
    prepare()
    selected = select()
    spec = experiment.old.exp.read(experiment.OUT/"protocol.json")
    previous = experiment.old.exp.read(experiment.OUT/"freeze.json")
    results, outputs = {}, {}
    frozen_hash = sha256(OUT/"freeze.json")
    for seed in (1,2,3):
        data = experiment.data_for("val", seed, spec)
        metrics = {"baseline": {}, "smallstep": {}}
        model = previous["full_models"][str(seed)]["context_arc"]
        for sequence, row in data.items():
            arrays = {f: row[f] for f in experiment.old.FIELDS}
            arrays.update(baseline=row["baseline"], smallstep=score(row, model, selected["factor"]))
            for method in metrics:
                metrics[method][sequence] = experiment.old.exp.metrics(arrays[method], row)
            path = OUT/f"val/s{seed}/{sequence.replace('/', '_')}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            outputs[str(path.relative_to(REPO))] = sha256(path)
        results[str(seed)] = {m: {**experiment.old.macro(rows), "per_sequence": rows} for m,rows in metrics.items()}
    mean_std = {}
    for method in ("baseline", "smallstep"):
        rows = [results[str(s)][method] for s in (1,2,3)]
        mean_std[method] = {"macro": {m: {"mean": float(np.mean([r["macro"][m] for r in rows])),
                    "std": float(np.std([r["macro"][m] for r in rows], ddof=1))} for m in experiment.old.exp.METRICS},
                    "families": {f: {m: float(np.mean([r["families"][f][m] for r in rows])) for m in experiment.old.exp.METRICS} for f in rows[0]["families"]}}
    passes = []
    for seed in (1,2,3):
        a,b = results[str(seed)]["smallstep"]["macro"],results[str(seed)]["baseline"]["macro"]
        passes.append(a["AP"] >= b["AP"]+.001 and a["R@100P"] >= b["R@100P"]-.001)
    a,b = mean_std["smallstep"]["families"],mean_std["baseline"]["families"]
    family_pass = all(a[f]["AP"] >= b[f]["AP"]-.0025 for f in a)
    assert sha256(OUT/"freeze.json") == frozen_hash
    result = {"factor": selected["factor"], "per_seed": results, "mean_std": mean_std,
        "seeds_pass": passes, "families_pass": family_pass, "promote": all(passes) and family_pass,
        "outputs": outputs, "freeze_sha256": frozen_hash, "test_used": False}
    write_json(OUT/"summary.json", result)
    for method, row in mean_std.items():
        print("FINAL_VAL", method, {m: round(v["mean"]*100,4) for m,v in row["macro"].items()}, flush=True)


if __name__ == "__main__":
    main()
