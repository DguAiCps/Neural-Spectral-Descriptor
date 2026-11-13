#!/usr/bin/env python3
"""장소 제외 인코더 출력에서 높은 오답 점수 손실을 선택하고 검증한다."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_encoder_place_eval as previous
from _pdv2_context_accept_lib import labels
from _pdv2_context_place_lib import fit_fold, false_rate_threshold, score
from _pdv2_context_tail_lib import ARMS, fit_tail
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / "results/pdv2_context_tail_20261019"
PRIOR = previous.OUT
accept = previous.accept


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def training_paths() -> dict:
    """각 장소를 보지 않은 인코더의 실제 최종 원장 위치를 반환한다."""
    spec = read(PRIOR / "protocol.json")
    return {s: previous.feature_path(spec["pairs"][p]["leave"], s)
            for p, names in spec["places"].items() for s in names}


def prepare() -> dict:
    """학습 결과를 보기 전에 조건·선택·검증 기준을 고정한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        result = read(target)
        for path, digest in result["sources"].items():
            assert sha256(REPO / path) == digest, path
        return result
    paths = set(training_paths().values())
    paths.update(previous.feature_path("full", s + "/" + q, "val") for s, q in accept.VAL)
    paths.update(PRIOR / f for f in ("protocol.json", "calibration_freeze.json", "summary.json",
                                   "completion_validation.json", "completion_status.json"))
    paths.update(REPO / f for f in (
        "scripts/_pdv2_context_tail.py", "scripts/_pdv2_context_tail_lib.py",
        "tests/retrieval/test_context_tail.py", "scripts/_pdv2_context_accept_lib.py",
        "scripts/_pdv2_context_place_lib.py", "scripts/_pdv2_context_accept.py",
        "scripts/_pdv2_encoder_place_eval.py", "scripts/_pdv2_spectral_evidence.py",
        "scripts/_pdv2_no_context_ablation.py", "scripts/_pr_eval_lib.py"))
    spec = {
        "created_unix": time.time(), "seed": 1, "test_used": False, "encoder_updates": 0,
        "arms": ARMS, "places": read(PRIOR / "protocol.json")["places"],
        "fit": "인코더 장소 제외 원장8. 기존두점수·양수계수·BCE·L2=.001 유지. "
               "적합장소만으로 BCE모델 적합→각시퀀스 오답 상위1% 또는최고1개 고정→"
               "해당평균점수가 각정답보다낮아지도록 softplus(오답평균-정답) 추가. "
               "BCE와순위손실 각각계열→시퀀스균등. 양성없는시퀀스는 BCE에만포함. λ=1,4.",
        "selection": "점수적합4장소제외. 각조건의제외예측을합친 최종AP로선택. "
                     "평균AP>=기존시간판정+.001,평균R100>=기존,각양성장소AP>=기존-.0025,"
                     "각양성장소R100>=기존-.001,비양성장소의학습1%오답임계값FPR<=기존+.001. "
                     "통과중평균AP최대,동률R100최대→λ작은순→이름순. 전부실패시 같은순서1개만"
                     "진단용선택하여검증하되확장불가. 검증결과로다른조건재선택하지않음.",
        "validation": "조건선택후OOF원장8전체재적합·해시동결→이미고정한full20epoch모델 "
                      "검증9동일99236질의·최종승자. 기존시간판정·기존OOF-BCE·선택손실1개 비교.",
        "expansion": "학습통과필수. 검증평균AP>=기존시간+.001,평균R100>=기존시간-.001,"
                     "각계열AP>=기존시간-.0025를기존기준으로보고. 추가로BCE대비평균AP손실<=.001,"
                     "평균R100비감소,각계열R100>=기존시간-.01이어야추가시드대상. 시험평가없음.",
        "limits": "기존검증결과를보고설계한1시드후속. 점수적합장소제외는완전한중첩인코더검증이아님:"
                  "나머지OOF특징을만든인코더는점수제외장소를학습했을수있음. 기존재계량·융합고정. "
                  "GT학습과KISS검증입력차이유지. 모든조건은결국두원점수계수비만바꿈.",
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)}}
    write_json(target, spec)
    print("PROTOCOL_FROZEN", len(paths), flush=True)
    return spec


def predicted(row: dict, model: dict | None) -> np.ndarray:
    """기존 점수 또는 고정된 두 점수 적합을 적용한다."""
    return row["baseline"] if model is None else score(row, model, 1.)


def threshold(data: dict, excluded: tuple, model: dict | None) -> float:
    """점수 적합 자료의 오답만으로 1% 수락 임계값을 정한다."""
    values = []
    for name, row in data.items():
        if name in excluded:
            continue
        keep, y = labels(row)
        values.append(predicted(row, model)[keep & (y == 0)])
    return false_rate_threshold(np.concatenate(values), .01)


def training(spec: dict) -> None:
    """검증을 읽기 전에 점수 적합의 장소 제외 선택을 끝낸다."""
    target = OUT / "selection_freeze.json"
    if target.exists():
        assert read(target)["protocol_sha256"] == sha256(OUT / "protocol.json")
        return
    data = {s: previous.load(p) for s, p in training_paths().items()}
    results = {name: {} for name in ("baseline", "bce", *ARMS)}
    fits, thresholds, outputs = {}, {}, {}
    for place, excluded in spec["places"].items():
        excluded = tuple(excluded)
        models = {"baseline": None, "bce": fit_fold(data, excluded, "context_arc")}
        for name, parameters in ARMS.items():
            models[name] = fit_tail(data, excluded, **parameters)
        fits[place] = models
        thresholds[place] = {name: threshold(data, excluded, model) for name, model in models.items()}
        for sequence in excluded:
            row = data[sequence]
            scores = {name: predicted(row, model) for name, model in models.items()}
            keep, y = labels(row)
            for name, value in scores.items():
                results[name][sequence] = {**accept.exp.metrics(value, row),
                    "incorrect_accept_rate": float(np.mean(value[keep & (y == 0)] >= thresholds[place][name]))}
            target_scores = OUT / "train_cv" / (sequence.replace("/", "_") + ".npz")
            target_scores.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target_scores, **{f: row[f] for f in accept.FIELDS}, **scores)
            outputs[str(target_scores.relative_to(REPO))] = sha256(target_scores)
        print("TRAIN_PLACE", place, flush=True)
    aggregate = {name: accept.macro(rows) for name, rows in results.items()}
    checks = {}
    base = aggregate["baseline"]["macro"]
    for name in ARMS:
        now = aggregate[name]["macro"]
        guards = {"mean_AP": now["AP"] >= base["AP"] + .001,
                  "mean_R100": now["R@100P"] >= base["R@100P"]}
        for place, names in spec["places"].items():
            positive = [s for s in names if results["baseline"][s]["n_pos"]]
            if positive:
                for metric, limit in (("AP", .0025), ("R@100P", .001)):
                    delta = np.mean([results[name][s][metric] - results["baseline"][s][metric]
                                     for s in positive])
                    guards[place + "/" + metric] = bool(delta >= -limit)
            else:
                guards[place + "/FPR"] = all(results[name][s]["incorrect_accept_rate"] <=
                    results["baseline"][s]["incorrect_accept_rate"] + .001 for s in names)
        checks[name] = {"checks": guards, "pass": all(guards.values())}
    eligible = [name for name in ARMS if checks[name]["pass"]]
    candidates = eligible or list(ARMS)
    selected = sorted(candidates, key=lambda name: (-aggregate[name]["macro"]["AP"],
        -aggregate[name]["macro"]["R@100P"], ARMS[name]["strength"], name))[0]
    model = fit_tail(data, (), **ARMS[selected])
    reference = fit_fold(data, (), "context_arc")
    old = read(PRIOR / "calibration_freeze.json")["models"]["crossfit"]
    for key in ("mean", "scale", "coefficients", "intercept"):
        np.testing.assert_allclose(reference[key], old[key], rtol=0, atol=1e-12)
    record = {"created_unix": time.time(), "selected": selected,
              "training_pass": bool(eligible), "eligible": eligible, "checks": checks,
              "aggregate": aggregate, "per_sequence": results, "fold_models": fits,
              "fold_thresholds": thresholds, "model": model, "reference_model": reference,
              "protocol_sha256": sha256(OUT / "protocol.json"), "outputs": outputs,
              "validation_read_during_selection": False, "test_used": False}
    write_json(target, record)
    print("SELECTION_FROZEN", selected, "training_pass", bool(eligible),
          "ratio", model["raw_context_ratio"], flush=True)


def validation() -> None:
    """선택한 조건 하나만 같은 최종 승자에 적용한다."""
    frozen = read(OUT / "selection_freeze.json")
    per = {name: {} for name in ("baseline", "bce", "tail")}
    output = {}
    for sensor, seq in accept.VAL:
        sequence = sensor + "/" + seq
        row = previous.load(previous.feature_path("full", sequence, "val"))
        values = {"baseline": row["baseline"], "bce": score(row, frozen["reference_model"], 1.),
                  "tail": score(row, frozen["model"], 1.)}
        np.testing.assert_allclose(values["bce"], row["crossfit"], rtol=0, atol=1e-12)
        target = OUT / "val" / (sequence.replace("/", "_") + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **{f: row[f] for f in accept.FIELDS}, **values)
        for name, value in values.items():
            per[name][sequence] = accept.exp.metrics(value, row)
        output[str(target.relative_to(REPO))] = sha256(target)
    aggregate = {name: accept.macro(rows) for name, rows in per.items()}
    now, base, ref = [aggregate[name] for name in ("tail", "baseline", "bce")]
    prior_guard = {
        "mean_AP": now["macro"]["AP"] >= base["macro"]["AP"] + .001,
        "mean_R100": now["macro"]["R@100P"] >= base["macro"]["R@100P"] - .001,
        "family_AP": all(now["families"][f]["AP"] >= base["families"][f]["AP"] - .0025
                         for f in base["families"])}
    repair_guard = {
        "retain_BCE_mean_AP": now["macro"]["AP"] >= ref["macro"]["AP"] - .001,
        "improve_BCE_mean_R100": now["macro"]["R@100P"] >= ref["macro"]["R@100P"],
        "family_R100": all(now["families"][f]["R@100P"] >= base["families"][f]["R@100P"] - .01
                           for f in base["families"])}
    write_json(OUT / "summary.json", {
        "seed": 1, "selected": frozen["selected"], "training_pass": frozen["training_pass"],
        "aggregate": aggregate, "per_sequence": per, "prior_guard": prior_guard,
        "repair_guard": repair_guard,
        "eligible_for_three_seed_confirmation": frozen["training_pass"] and
            all(prior_guard.values()) and all(repair_guard.values()),
        "outputs": output, "selection_sha256": sha256(OUT / "selection_freeze.json"),
        "test_used": False, "encoder_updates": 0})
    for name, result in aggregate.items():
        print("FINAL_VALIDATION", name, result["macro"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "run"))
    args = parser.parse_args()
    protocol = prepare()
    if args.action != "prepare":
        training(protocol)
    if args.action == "run":
        validation()
