#!/usr/bin/env python3
"""과거 후보의 공통 유사도를 보정하고 최종 수락 성능을 대조한다."""
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
from _pdv2_context_relative_lib import STYLES, relative_features
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / "results/pdv2_context_relative_20261019"
PRIOR = previous.OUT
accept = previous.accept
FEATURES = ("context_arc", "d_arc") + tuple(n + "_" + s for n in ("context", "d") for s in STYLES)


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def sources(role: str, sequence: str) -> dict:
    """같은 시퀀스·같은 인코더의 원장·키·자세 파일만 반환한다."""
    if role == "train":
        spec = read(PRIOR / "protocol.json")
        place = next(p for p, names in spec["places"].items() if sequence in names)
        root = PRIOR / spec["pairs"][place]["leave"]
        return {"row": root / "train" / (sequence.replace("/", "_") + ".npz"),
                "key": root / "retrieval_keys.npy", "poses": root / "training_metadata.npz"}
    assert role == "val"
    _, _, paths, _ = previous.focus.base.inputs("val", sequence)
    return {"row": previous.feature_path("full", sequence, "val"),
            "key": PRIOR / "full/val_keys" / (sequence.replace("/", "_") + ".npz"),
            "poses": paths["cache"]}


def inputs(role: str, sequence: str) -> tuple:
    """학습은 GT 경로, 검증은 KISS-ICP 경로를 읽는다."""
    paths = sources(role, sequence)
    row = previous.load(paths["row"])
    if role == "train":
        meta = previous.load(paths["poses"])
        roster = read(PRIOR / "protocol.json")["train"]
        ids = np.flatnonzero(meta["sequences"] == roster.index(sequence))
        assert np.all(np.diff(ids) == 1)
        key = np.load(paths["key"], mmap_mode="r")[ids]
        xyz = meta["positions"][ids].astype(np.float64)
    else:
        key = previous.load(paths["key"])["embedding"]
        with np.load(paths["poses"]) as z:
            xyz = z["poses"][:, :3, 3]
    path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    return key, path, row


def feature_path(role: str, sequence: str) -> Path:
    """새 실험 폴더에만 특징을 저장한다."""
    return OUT / "features" / role / (sequence.replace("/", "_") + ".npz")


def prepare() -> dict:
    """새 특징을 보기 전에 조건과 선택·확장 기준을 동결한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        spec = read(target)
        for p, h in spec["sources"].items():
            assert sha256(REPO / p) == h, p
        return spec
    old = read(PRIOR / "protocol.json")
    train = old["train"]
    val = [s + "/" + q for s, q in accept.VAL]
    paths = set()
    for role, names in (("train", train), ("val", val)):
        for sequence in names:
            paths.update(sources(role, sequence).values())
    paths.update(PRIOR / name for name in ("protocol.json", "calibration_freeze.json", "summary.json",
                                         "completion_validation.json", "completion_status.json"))
    paths.update(REPO / name for name in (
        "scripts/_pdv2_context_relative.py", "scripts/_pdv2_context_relative_lib.py",
        "tests/retrieval/test_context_relative.py", "scripts/_pdv2_encoder_place_eval.py",
        "scripts/_pdv2_context_accept.py", "scripts/_pdv2_context_accept_lib.py",
        "scripts/_pdv2_context_place_lib.py", "scripts/_pdv2_no_context_ablation.py",
        "scripts/_pdv2_spectral_evidence.py", "scripts/_pr_eval_lib.py",
        "src/retrieval/sequence_evidence.py"))
    spec = {
        "created_unix": time.time(), "seed": 1, "train": train, "val": val,
        "places": old["places"], "styles": STYLES, "features": FEATURES,
        "test_used": False, "encoder_updates": 0,
        "scope": "인코더 장소 제외 원장8·full20epoch 검증9. 모델·검색후보·최종승자·스펙트럼방향고정.",
        "normalization": "단위문맥128과단위재계량원특징288 각각. q마다같은시퀀스의 "
                         "0..q-30 모든과거후보평균mu를구함. GT라벨·최근접오답필터없음,정답도포함가능. "
                         "기존거리대응각쌍(x,y)에 excess=x.y-x.mu,centered_dot=(x-mu).(y-mu),"
                         "centered_cos=cos(x-mu,y-mu). 각잔차노름하한1e-6. "
                         "기존방향·중심비중·반복가중치로평균. mean은현재시퀀스에서만매질의갱신.",
        "online": "q+15에서판정. mu에포함되는최대키q-30의문맥은q-20까지관측. "
                  "평가이웃도모두q+15이내. 전체시퀀스평균·미래후보사용없음. "
                  "누적합은O(D)상태로구현가능하지만전체배포시간은이번범위아님.",
        "fit": "기존BCE+L2=.001,baseline양수·추가계수비음수,학습계열→시퀀스→질의균등. "
               "점수적합4장소제외. 보정문맥3종과같은원특징3종·미보정문맥/원특징을같이평가. "
               "보정문맥중에서만선택하고같은보정원특징을대조로사용. 다른손실·비중격자없음.",
        "selection": "학습내부평균AP>=baseline+.001,평균R100>=baseline, "
                     "각양성장소AP손실<=.0025,R100손실<=.001,양성없는장소학습1%오답임계값FPR증가<=.001. "
                     "통과중평균AP최대→R100최대→정해진styles순. 없으면같은기준1개진단용검증·확장불가. "
                     "선택뒤전체OOF재적합을동결하고검증적용. 검증에서다른보정재선택안함.",
        "validation": "baseline,미보정문맥,선택보정문맥,미보정원특징,같은보정원특징 5조건. "
                      "최종AP/F1/R100/R1·계열별손익,같은99236질의.",
        "expansion": "학습통과,검증AP>=baseline+.001,R100>=baseline-.001,각계열AP>=baseline-.0025. "
                     "추가로미보정문맥대비평균AP손실<=.001,평균R100비감소,각계열R100>=baseline-.01. "
                     "이번실험에서는시험·3시드·원고교체없음. 보정문맥의원특징대비차이는별도보고.",
        "limits": "1시드반복검증예비. 점수적합장소제외는중첩인코더검증아님. "
                  "OOF인코더별학습량·분포차이와GT학습/KISS검증차이남음. "
                  "새정규화효과가시간간선의인과적기여나신규성증거는아님.",
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)}}
    write_json(target, spec)
    print("PROTOCOL_FROZEN", len(paths), flush=True)
    return spec


def generate(role: str, sequence: str) -> None:
    """한 시퀀스의 새 점수와 q+15 접두 대조를 저장한다."""
    target = feature_path(role, sequence)
    if target.exists():
        assert sha256(target) == read(target.with_suffix(".json"))["sha256"]
        return
    started = time.perf_counter()
    key, path, row = inputs(role, sequence)
    values = relative_features(key, path, row)
    errors = {}
    for name in ("d", "context"):
        errors[name] = float(abs(values[name + "_arc_check"] - row[name + "_arc"]).max())
        np.testing.assert_allclose(values[name + "_arc_check"], row[name + "_arc"], atol=8e-7, rtol=0)
    checks = []
    for i in np.unique(np.linspace(0, len(row["query_idx"]) - 1, 5, dtype=int)):
        one = {k: v[i:i + 1] for k, v in row.items()}
        q = int(row["query_idx"][i])
        prefix = relative_features(key[:q + 16], path[:q + 16], one)
        error = max(float(abs(prefix[f][0] - values[f][i])) for f in values)
        assert error < 1e-10
        checks.append({"query": q, "max_error": error, "max_background_index": q - 30})
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **row, **values)
    write_json(target.with_suffix(".json"), {"sha256": sha256(target), "rows": len(row["query_idx"]),
        "arc_reconstruction_max_error": errors, "prefix_checks": checks,
        "feature_generation_seconds": time.perf_counter() - started,
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in sources(role, sequence).values()}})
    print("FEATURES", role, sequence, "rows", len(row["query_idx"]), flush=True)


def threshold(data: dict, excluded: tuple, model: dict) -> float:
    """적합 자료 오답의 1% 수락 임계값을 구한다."""
    arrays = []
    for name, row in data.items():
        if name not in excluded:
            keep, y = labels(row)
            values = row["baseline"] if model is None else score(row, model, 1.)
            arrays.append(values[keep & (y == 0)])
    return false_rate_threshold(np.concatenate(arrays), .01)


def training(spec: dict) -> None:
    """보정 문맥 하나를 학습 내부에서 선택하고 같은 원특징 대조를 동결한다."""
    target = OUT / "selection_freeze.json"
    if target.exists():
        assert read(target)["protocol_sha256"] == sha256(OUT / "protocol.json")
        return
    for sequence in spec["train"]:
        generate("train", sequence)
    data = {s: previous.load(feature_path("train", s)) for s in spec["train"]}
    results = {name: {} for name in ("baseline", *FEATURES)}
    fits, thresholds, outputs = {}, {}, {}
    for place, excluded in spec["places"].items():
        excluded = tuple(excluded)
        models = {"baseline": None, **{f: fit_fold(data, excluded, f) for f in FEATURES}}
        fits[place] = models
        thresholds[place] = {f: threshold(data, excluded, model) for f, model in models.items()}
        for sequence in excluded:
            row = data[sequence]
            keep, y = labels(row)
            values = {f: row["baseline"] if model is None else score(row, model, 1.)
                      for f, model in models.items()}
            for f, value in values.items():
                results[f][sequence] = {**accept.exp.metrics(value, row),
                    "incorrect_accept_rate": float(np.mean(value[keep & (y == 0)] >= thresholds[place][f]))}
            p = OUT / "train_cv" / (sequence.replace("/", "_") + ".npz")
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **{f: row[f] for f in accept.FIELDS}, **values)
            outputs[str(p.relative_to(REPO))] = sha256(p)
        print("TRAIN_PLACE", place, flush=True)
    aggregate = {f: accept.macro(rows) for f, rows in results.items()}
    checks = {}
    for f in ("context_" + s for s in STYLES):
        now, base = aggregate[f]["macro"], aggregate["baseline"]["macro"]
        guards = {"mean_AP": now["AP"] >= base["AP"] + .001,
                  "mean_R100": now["R@100P"] >= base["R@100P"]}
        for place, names in spec["places"].items():
            positive = [s for s in names if results["baseline"][s]["n_pos"]]
            if positive:
                for metric, limit in (("AP", .0025), ("R@100P", .001)):
                    delta = np.mean([results[f][s][metric] - results["baseline"][s][metric] for s in positive])
                    guards[place + "/" + metric] = bool(delta >= -limit)
            else:
                guards[place + "/FPR"] = all(results[f][s]["incorrect_accept_rate"] <=
                    results["baseline"][s]["incorrect_accept_rate"] + .001 for s in names)
        checks[f] = {"checks": guards, "pass": all(guards.values())}
    eligible = [f for f in checks if checks[f]["pass"]]
    order = ["context_" + s for s in STYLES]
    selected = sorted(eligible or order, key=lambda f: (-aggregate[f]["macro"]["AP"],
        -aggregate[f]["macro"]["R@100P"], order.index(f)))[0]
    matched = "d_" + selected[len("context_"):]
    models = {"context_uncorrected": fit_fold(data, (), "context_arc"),
              "context_relative": fit_fold(data, (), selected),
              "d_uncorrected": fit_fold(data, (), "d_arc"),
              "d_relative": fit_fold(data, (), matched)}
    old = read(PRIOR / "calibration_freeze.json")["models"]["crossfit"]
    for key in ("mean", "scale", "coefficients", "intercept"):
        np.testing.assert_allclose(models["context_uncorrected"][key], old[key], atol=1e-12, rtol=0)
    write_json(target, {"created_unix": time.time(), "selected": selected, "matched_raw": matched,
        "training_pass": bool(eligible), "eligible": eligible, "checks": checks,
        "aggregate": aggregate, "per_sequence": results, "fold_models": fits, "fold_thresholds": thresholds,
        "models": models, "outputs": outputs, "test_used": False,
        "validation_read_during_selection": False, "protocol_sha256": sha256(OUT / "protocol.json"),
        "training_features": {str(feature_path("train", s).relative_to(REPO)): sha256(feature_path("train", s))
                              for s in spec["train"]}})
    print("SELECTION_FROZEN", selected, "training_pass", bool(eligible), flush=True)


def validation(spec: dict) -> None:
    """동결한 보정 문맥과 동일한 원특징 대조만 최종 검증한다."""
    frozen = read(OUT / "selection_freeze.json")
    results = {name: {} for name in ("baseline", *frozen["models"])}
    outputs = {}
    for sequence in spec["val"]:
        generate("val", sequence)
        row = previous.load(feature_path("val", sequence))
        values = {"baseline": row["baseline"], **{name: score(row, model, 1.) for name, model in frozen["models"].items()}}
        np.testing.assert_allclose(values["context_uncorrected"], row["crossfit"], atol=1e-12, rtol=0)
        for name, value in values.items():
            results[name][sequence] = accept.exp.metrics(value, row)
        path = OUT / "val" / (sequence.replace("/", "_") + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f: row[f] for f in accept.FIELDS}, **values)
        outputs[str(path.relative_to(REPO))] = sha256(path)
    aggregate = {f: accept.macro(rows) for f, rows in results.items()}
    now, base, old = [aggregate[f] for f in ("context_relative", "baseline", "context_uncorrected")]
    prior_guard = {"mean_AP": now["macro"]["AP"] >= base["macro"]["AP"] + .001,
        "mean_R100": now["macro"]["R@100P"] >= base["macro"]["R@100P"] - .001,
        "family_AP": all(now["families"][f]["AP"] >= base["families"][f]["AP"] - .0025 for f in base["families"])}
    repair_guard = {"retain_context_AP": now["macro"]["AP"] >= old["macro"]["AP"] - .001,
        "improve_context_R100": now["macro"]["R@100P"] >= old["macro"]["R@100P"],
        "family_R100": all(now["families"][f]["R@100P"] >= base["families"][f]["R@100P"] - .01 for f in base["families"])}
    write_json(OUT / "summary.json", {"seed": 1, "selected": frozen["selected"], "training_pass": frozen["training_pass"],
        "aggregate": aggregate, "per_sequence": results, "prior_guard": prior_guard, "repair_guard": repair_guard,
        "eligible_for_three_seed_confirmation": frozen["training_pass"] and all(prior_guard.values()) and all(repair_guard.values()),
        "outputs": outputs, "selection_sha256": sha256(OUT / "selection_freeze.json"), "test_used": False,
        "encoder_updates": 0, "validation_features": {str(feature_path("val", s).relative_to(REPO)): sha256(feature_path("val", s))
                                                      for s in spec["val"]}})
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
        validation(protocol)
