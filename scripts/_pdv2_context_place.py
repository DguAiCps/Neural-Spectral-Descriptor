#!/usr/bin/env python3
"""학습 내부의 장소 제외로 문맥 비중을 선택하고 고정 검증에 적용한다."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_context_accept_fp32 as previous
previous.configure()
old = previous.run
from _pdv2_context_accept_lib import labels
from _pdv2_context_place_lib import (
    FACTORS, FEATURES, PLACES, choose, false_rate_threshold, fit_fold,
    robust_pass, score,
)
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / "results/pdv2_context_place_20261018"
GEO = REPO / "results/split_audit/geo_overlap.json"
CONDITIONS = ("baseline", "context_full", "context_ap", "context_robust", "d_robust")


def key(feature: str, factor: float) -> str:
    """점수 원장의 고정 열 이름을 만든다."""
    return f"{feature}_f{int(round(factor*100)):03d}"


def audit_places() -> dict:
    """기존 위치 감사와 학습 좌표로 장소 묶음을 확인한다."""
    geographic = old.exp.read(GEO)
    kitti, bridge = {}, {}
    assignment = {s: place for place, seqs in PLACES.items() for s in seqs}
    for a in assignment:
        for b in assignment:
            if a >= b or not a.startswith("kitti") or not b.startswith("kitti"):
                continue
            ab, ba = geographic["pairs_same_frame"][a+"|"+b], geographic["pairs_same_frame"][b+"|"+a]
            kitti[a+"|"+b] = {"forward": ab, "reverse": ba, "same_place_group": assignment[a] == assignment[b]}
            if assignment[a] != assignment[b]:
                assert max(ab["frac_len_50m"], ba["frac_len_50m"], ab["frac_scan_50m"], ba["frac_scan_50m"]) <= .01
    meta = old.exp.arrays(old.PRIOR/"training_metadata.npz")
    roster = old.exp.read(old.OUT/"protocol.json")["train"]
    for a in PLACES["helipr_bridge"]:
        ids = np.flatnonzero(meta["sequences"] == roster.index(a))
        assert (meta["coordinate_groups"][ids] == 100).all()
        for b in PLACES["helipr_bridge"]:
            if a == b:
                continue
            other = np.flatnonzero(meta["sequences"] == roster.index(b))
            distance = cKDTree(meta["positions"][other]).query(meta["positions"][ids])[0]
            bridge[a+"|"+b] = {"minimum_m": float(distance.min()), "fraction_keyframes_50m": float((distance <= 50).mean())}
    return {"groups": PLACES, "kitti_geographic": kitti, "bridge_training_coordinates": bridge,
            "cross_region": "KITTI독일·HeLiPR Bridge서울. 공통미터좌표끼리만직접비교. 기존geo/bridge위치감사재사용.",
            "scope": "가중치적합의장소분리. 동결문맥모델·재계량·승자생성은전체학습자료를본상태이므로완전한미관측장소모델검증이아님."}


def prepare() -> dict:
    """새 결과 계산 전에 장소·비중 후보·선택·검증 범위를 동결한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/"protocol.json"
    if target.exists():
        protocol = old.exp.read(target)
        for p, h in protocol["sources"].items():
            assert sha256(REPO/p) == h, p
        return protocol
    check = old.exp.read(old.OUT/"completion_validation.json")
    assert check["status"] == "passed"
    for p, h in check["sources"].items():
        assert sha256(REPO/p) == h, p
    prior = old.exp.read(old.OUT/"protocol.json")
    sources = {Path(__file__), REPO/"scripts/_pdv2_context_place_lib.py",
               REPO/"tests/retrieval/test_context_place.py", GEO,
               REPO/"configs/training_place_disjoint_v2.yaml",
               REPO/"results/split_audit/bridge/bridge_overlap.json",
               old.PRIOR/"training_metadata.npz", old.OUT/"fitted.json",
               old.OUT/"protocol.json", old.OUT/"completion_validation.json",
               old.OUT/"report.json", REPO/"scripts/_pdv2_context_accept.py",
               REPO/"scripts/_pdv2_context_accept_fp32.py",
               REPO/"scripts/_pdv2_context_accept_lib.py", REPO/"scripts/_pr_eval_lib.py"}
    for role in ("train", "val"):
        for seed in (1,2,3):
            for sequence in prior[role]:
                sources.add(old.feature_file(role, seed, sequence))
    audit = audit_places()
    write_json(OUT/"place_audit.json", audit)
    sources.add(OUT/"place_audit.json")
    protocol = {"created_unix": time.time(), "seeds": [1,2,3], "places": PLACES,
        "factors": FACTORS, "features": FEATURES, "train": prior["train"], "val": prior["val"],
        "fit": "한장소전체제외. 남은계열균등→시퀀스균등→유효질의균등. 기존양의BCE+L2.001,통계도제외장소사용금지. 새모듈·인코더업데이트없음.",
        "factor": "학습한추가특징계수만0/5/10/25/50/100%로줄임. 기준스펙트럼계수/표준화/편향은유지. 0은기준과단조동일,100은기존가중합.",
        "mean_ap_selection": "제외장소별원장결합,양성시퀀스→2계열→3시드최종AP최고. 동률작은비중. 진단용이며주선택아님.",
        "robust_selection": "주선택. 각양성장소·각시드AP/R100손실<=.001,양성없는장소오답수락률증가<=.001,전체평균AP이득>=.001. 통과중AP최고,동률작은비중,없으면0.",
        "negative_place_threshold": "각fold적합자료의오답전체에서수락률<=1%가되는임계값을조건별로고정. 제외KITTI01오답수락률진단. 최종배포임계값아님.",
        "validation": "학습OOF선택이끝나면freeze저장. 기존전체학습적합계수에선택비율적용. baseline/기존100%/평균AP선택/보수적문맥선택/보수적원래특징대조만검증. 검증에서비율선택안함.",
        "promotion": "주선택context_robust만판단. 검증각시드AP+.001이상/R100손실<=.001,3시드평균각계열AP손실<=.0025. 다른조건으로사후교체안함. 시험실행없음.",
        "limitation": audit["scope"]+" 검증9는이전개발에사용된자료. 이OOF선택값은중첩교차검증의불편추정량이아님. Bridge한장소만학습에있어자료다양성제한.",
        "test_used": False, "encoder_updates": 0,
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in sorted(sources)}}
    write_json(target, protocol)
    return protocol


def data_for(role: str, seed: int, spec: dict) -> dict:
    """검증된 기존 고정 승자 특징 원장을 읽는다."""
    return {sequence: old.exp.arrays(old.feature_file(role, seed, sequence)) for sequence in spec[role]}


def train_selection(spec: dict) -> dict:
    """제외한 장소의 최종 지표만으로 공통 축소 비율을 고정한다."""
    target = OUT/"freeze.json"
    if target.exists():
        return old.exp.read(target)
    all_fits, statistics = {}, {}
    for seed in spec["seeds"]:
        data = data_for("train", seed, spec)
        all_fits[str(seed)], statistics[str(seed)] = {}, {}
        for place, heldout in PLACES.items():
            all_fits[str(seed)][place], statistics[str(seed)][place] = {}, {}
            outputs = {s: {f: data[s][f] for f in old.FIELDS} for s in heldout}
            for feature in FEATURES:
                fitted = fit_fold(data, heldout, feature)
                all_fits[str(seed)][place][feature] = fitted
                statistics[str(seed)][place][feature] = {}
                for factor in FACTORS:
                    name = key(feature, factor)
                    negatives = []
                    for s in fitted["training"]:
                        keep, y = labels(data[s])
                        negatives.append(score(data[s], fitted, factor)[keep & (y == 0)])
                    negative_score = np.concatenate(negatives)
                    tau = false_rate_threshold(negative_score)
                    assert (negative_score >= tau).mean() <= .01
                    records = {}
                    for s in heldout:
                        prediction = score(data[s], fitted, factor)
                        outputs[s][name] = prediction
                        keep, y = labels(data[s])
                        wrong = keep & (y == 0)
                        records[s] = {**old.exp.metrics(prediction, data[s]),
                            "incorrect": int(wrong.sum()), "incorrect_accept": int((wrong & (prediction >= tau)).sum())}
                    positives = [r for r in records.values() if r["n_pos"]]
                    wrong_n = sum(r["incorrect"] for r in records.values())
                    aggregate = {m: float(np.mean([r[m] for r in positives])) if positives else None for m in old.exp.METRICS}
                    aggregate.update(n_pos=sum(r["n_pos"] for r in records.values()),
                        incorrect_accept_rate=sum(r["incorrect_accept"] for r in records.values())/wrong_n)
                    statistics[str(seed)][place][feature][str(factor)] = {"per_sequence": records,
                        "place": aggregate, "fit_negative_threshold": tau, "fit_negative_count": len(negative_score)}
            for sequence, arrays in outputs.items():
                path = OUT/f"oof/s{seed}/{sequence.replace('/', '_')}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
            print("HELDOUT", seed, place, flush=True)
    choices, grids = {}, {}
    for feature in FEATURES:
        grid = []
        for factor in FACTORS:
            macros, base_macros, deltas = [], [], []
            for seed in spec["seeds"]:
                current, reference = {}, {}
                for place in PLACES:
                    rows = statistics[str(seed)][place][feature]
                    now, base = rows[str(factor)], rows["0.0"]
                    current.update(now["per_sequence"])
                    reference.update(base["per_sequence"])
                    a, b = now["place"], base["place"]
                    deltas.append({"seed": seed, "place": place, "n_pos": a["n_pos"],
                        "AP": a["AP"]-b["AP"] if a["n_pos"] else None,
                        "R@100P": a["R@100P"]-b["R@100P"] if a["n_pos"] else None,
                        "incorrect_accept_rate": a["incorrect_accept_rate"]-b["incorrect_accept_rate"]})
                macros.append(old.macro(current)["macro"])
                base_macros.append(old.macro(reference)["macro"])
            average = {m: float(np.mean([v[m] for v in macros])) for m in old.exp.METRICS}
            gain = average["AP"]-float(np.mean([v["AP"] for v in base_macros]))
            grid.append({"factor": factor, **average, "AP_gain": gain,
                         "deltas": deltas, "pass": robust_pass(deltas, gain), "per_seed": macros})
        choices[feature], grids[feature] = choose(grid), grid
    write_json(OUT/"oof_fits.json", all_fits)
    write_json(OUT/"oof_summary.json", {"statistics": statistics, "grids": grids, "choices": choices})
    full = old.exp.read(old.OUT/"fitted.json")
    frozen = {"created_unix": time.time(), "choices": choices, "primary": "context_robust",
        "full_models": {str(s): {"context_arc": full["models"][str(s)]["add_context_arc"],
                                  "d_arc": full["models"][str(s)]["add_d_arc"]} for s in spec["seeds"]},
        "conditions": {"baseline": None, "context_full": ["context_arc", 1.],
            "context_ap": ["context_arc", choices["context_arc"]["mean_ap_factor"]],
            "context_robust": ["context_arc", choices["context_arc"]["robust_factor"]],
            "d_robust": ["d_arc", choices["d_arc"]["robust_factor"]]},
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in (OUT/"protocol.json", OUT/"oof_fits.json", OUT/"oof_summary.json", old.OUT/"fitted.json")},
        "validation_evaluated_before_freeze": False, "test_used": False}
    write_json(target, frozen)
    print("FROZEN_CHOICES", choices, flush=True)
    return frozen


def validation(spec: dict, frozen: dict) -> None:
    """선택한 비율만 검증하고 주선택의 채택 기준을 확인한다."""
    freeze_hash = sha256(OUT/"freeze.json")
    results, outputs = {}, {}
    for seed in spec["seeds"]:
        data = data_for("val", seed, spec)
        metrics = {m: {} for m in CONDITIONS}
        for sequence, row in data.items():
            arrays = {f: row[f] for f in old.FIELDS}
            for name, condition in frozen["conditions"].items():
                prediction = row["baseline"] if condition is None else score(row, frozen["full_models"][str(seed)][condition[0]], condition[1])
                arrays[name] = prediction
                metrics[name][sequence] = old.exp.metrics(prediction, row)
            path = OUT/f"val/s{seed}/{sequence.replace('/', '_')}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            outputs[str(path.relative_to(REPO))] = sha256(path)
        results[str(seed)] = {m: {**old.macro(v), "per_sequence": v} for m, v in metrics.items()}
    mean_std = {}
    for method in CONDITIONS:
        rows = [results[str(s)][method] for s in spec["seeds"]]
        mean_std[method] = {"macro": {m: {"mean": float(np.mean([v["macro"][m] for v in rows])),
                            "std": float(np.std([v["macro"][m] for v in rows], ddof=1))} for m in old.exp.METRICS},
                            "families": {f: {m: float(np.mean([v["families"][f][m] for v in rows])) for m in old.exp.METRICS} for f in rows[0]["families"]}}
    seed_pass = []
    for seed in spec["seeds"]:
        a, b = results[str(seed)]["context_robust"]["macro"], results[str(seed)]["baseline"]["macro"]
        seed_pass.append(a["AP"] >= b["AP"]+.001 and a["R@100P"] >= b["R@100P"]-.001)
    a, b = mean_std["context_robust"]["families"], mean_std["baseline"]["families"]
    family_pass = all(a[f]["AP"] >= b[f]["AP"]-.0025 for f in a)
    assert sha256(OUT/"freeze.json") == freeze_hash
    summary = {"per_seed": results, "mean_std": mean_std, "choices": frozen["choices"],
        "primary": frozen["primary"], "promote": all(seed_pass) and family_pass,
        "seeds_pass": seed_pass, "families_pass": family_pass, "test_used": False,
        "freeze_sha256": freeze_hash, "outputs": outputs}
    write_json(OUT/"summary.json", summary)
    for name, row in mean_std.items():
        print("FINAL_VAL", name, {m: round(v["mean"]*100,4) for m,v in row["macro"].items()}, flush=True)


def main() -> None:
    """장소 확인·학습 내부 선택·동결 검증 순서로 실행한다."""
    started = time.time()
    spec = prepare()
    frozen = train_selection(spec)
    validation(spec, frozen)
    write_json(OUT/"status.json", {"status": "complete", "runtime_s": time.time()-started, "test_used": False})


if __name__ == "__main__":
    main()
