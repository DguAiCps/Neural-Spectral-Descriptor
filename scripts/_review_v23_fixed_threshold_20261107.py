#!/usr/bin/env python3
"""현재 v2.3 NSD와 BEVPlace++의 검증 고정 단일 판정 기준값을 시험46에 적용한다.

select: 검증 9시퀀스 원장만 읽어 방법·시드마다 τ를 정하고 freeze.json에 고정한다. 읽은 파일을 모두 기록한다.
evaluate: freeze.json의 원천 해시를 확인한 뒤 고정 τ를 시험 46시퀀스에 그대로 적용한다.
규약은 results/pdv2_review_closure_20261107/threshold/protocol.json이다. 모델·후보·승자·점수는 바꾸지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from _pdv2_pilot import CITY, TEST, VAL
from _pr_eval_lib import classify_queries, operating_point, pr_sweep

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_review_closure_20261107/threshold"
NSD_VAL = REPO / "results/pdv2_diffattn_contrast_pipeline_20261104/val"
NSD_TEST = REPO / "results/pdv2_diffattn_contrast_test_20261104"
BEV = REPO / "results/pdv2_review_response_20261104/bev"
SEEDS = (1, 2, 3)
METHODS = ("nsd", "bevplace")
POINTS = {"P99": .99, "P100": 1.0, "F1": None}
SCORE = {"nsd": "score", "bevplace": "score_negl2"}
SUMMARY_CASE = {"nsd": "diff_s{}", "bevplace": "bev_native_s{}"}
TRACE: list = []   # select가 연 파일 목록(시험 원장 차단 확인용)


def read(path: Path) -> dict:
    """JSON을 읽고 읽기 추적에 남긴다."""
    TRACE.append(str(Path(path).relative_to(REPO)))
    return json.loads(Path(path).read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path: Path) -> dict:
    """NPZ 전체를 읽고 읽기 추적에 남긴다."""
    TRACE.append(str(Path(path).relative_to(REPO)))
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def ledger(method: str, role: str, seed: int, sensor: str, seq: str) -> tuple:
    """원장 경로와 원장 기록(json)의 sha256을 확인해 돌려준다."""
    name = f"{sensor}_{seq}"
    if method == "nsd":
        path = (NSD_VAL if role == "val" else NSD_TEST / "test") / name / f"diff_s{seed}.npz"
    else:
        path = BEV / role / f"s{seed}" / f"{name}.npz"
    record = read(path.with_suffix(".json"))
    assert digest(path) == record["sha256"], path
    return path, record


def val_rows(method: str, seed: int, protocol: dict) -> list:
    """검증 9시퀀스의 (시퀀스, 점수, 반환 거리, 과거 최근접 GT 거리). 두 방법의 질의·라벨을 대조한다."""
    rows = []
    for sensor, seq in VAL:
        path, record = ledger(method, "val", seed, sensor, seq)
        if method == "nsd":
            assert record["checkpoint_sha256"] == protocol["checkpoint_sha256"][f"diff_s{seed}"]
        else:
            assert record["status"] == "passed" and record["role"] == "val" and record["seed"] == seed
            assert record["checkpoint"]["sha256"] == protocol["bev_checkpoint_sha256"][str(seed)]
        data = arrays(path)
        other = arrays(NSD_VAL / f"{sensor}_{seq}" / "identity.npz")
        for key in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(data[key], other[key])
        assert data[SCORE[method]].dtype == np.float64
        rows.append(dict(sequence=f"{sensor}/{seq}", score=data[SCORE[method]], dist=data["top1_geo_dist"],
                         mpgd=data["min_past_gt_dist"], path=str(path.relative_to(REPO)), sha256=record["sha256"]))
    return rows


def choose(rows: list) -> dict:
    """검증 합산 sweep에서 세 운영점의 τ를 고른다(동점은 가장 높은 τ)."""
    scores = np.concatenate([r["score"] for r in rows])
    dists = np.concatenate([r["dist"] for r in rows])
    qc = classify_queries(np.concatenate([r["mpgd"] for r in rows]))
    sweep = pr_sweep(scores, dists, qc)
    chosen = {}
    for name, floor in POINTS.items():
        if floor is None:
            tau = sweep["max_F1_tau"]   # 라이브러리 규칙: 최대 F1의 첫(가장 높은) τ
        else:
            ok = np.flatnonzero(sweep["precision"] >= floor)
            best = sweep["recall"][ok].max()
            tau = float(sweep["thresholds"][ok[sweep["recall"][ok] == best][0]])
        point = operating_point(scores, dists, qc, tau)
        per_sequence = {}
        for r in rows:
            p = operating_point(r["score"], r["dist"], classify_queries(r["mpgd"]), tau)
            per_sequence[r["sequence"]] = {k: p[k] for k in ("TP", "FP", "FN", "TN")}
        chosen[name] = dict(tau=tau, tau_hex=float(tau).hex(), val_precision=point["precision"],
                            val_recall=point["recall"], val_F1=point["F1"], val_TP=point["TP"], val_FP=point["FP"],
                            val_accepted=point["TP"] + point["FP"], val_per_sequence=per_sequence)
    chosen["val_counts"] = sweep["counts"]
    chosen["val_sweep"] = dict(AP=sweep["AP"], max_F1=sweep["max_F1"], R100P=sweep["R@100P"], R99P=sweep["R@99P"],
                               n_thresholds=int(len(sweep["thresholds"])))
    return chosen


def select() -> None:
    """검증 원장만으로 τ를 고정한다. 이미 고정했으면 멈춘다."""
    target = OUT / "freeze.json"
    if target.exists():
        print("FROZEN_ALREADY"); return
    protocol_path = OUT / "protocol.json"
    spec = read(protocol_path)
    nsd_protocol = read(REPO / "results/pdv2_diffattn_contrast_pipeline_20261104/protocol.json")
    bev_protocol = read(BEV / "protocol.json")
    checks = dict(checkpoint_sha256={c: nsd_protocol["cases"][c]["checkpoint_sha256"] for c in spec["methods"]["nsd"]["cases"]},
                  bev_checkpoint_sha256={s: bev_protocol["checkpoints"][s]["sha256"] for s in ("1", "2", "3")})
    assert nsd_protocol["selected_policy"] == "matching_only"
    for case, path in spec["methods"]["nsd"]["checkpoints"].items():
        assert digest(REPO / path) == checks["checkpoint_sha256"][case], path
    for s in ("1", "2", "3"):
        assert digest(REPO / bev_protocol["checkpoints"][s]["path"]) == checks["bev_checkpoint_sha256"][s]
    thresholds, ledgers = {}, {}
    for method in METHODS:
        thresholds[method] = {}
        for seed in SEEDS:
            rows = val_rows(method, seed, checks)
            ledgers.update({r["path"]: r["sha256"] for r in rows})
            thresholds[method][str(seed)] = choose(rows)
            print(method, seed, {k: (v["tau"], v["val_precision"], v["val_recall"], v["val_FP"])
                                 for k, v in thresholds[method][str(seed)].items() if k in POINTS}, flush=True)
    opened = sorted(set(TRACE))
    assert not [p for p in opened if "/test/" in p or "_test_" in p], "시험 원장을 열었다"
    code = {str(Path(__file__).relative_to(REPO)): digest(Path(__file__)),
            "scripts/_pr_eval_lib.py": digest(REPO / "scripts/_pr_eval_lib.py"),
            "scripts/_pdv2_pilot.py": digest(REPO / "scripts/_pdv2_pilot.py")}
    write(target, dict(created_unix=time.time(), protocol_sha256=digest(protocol_path), rules=spec["threshold_selection"],
                       thresholds=thresholds, validation=[f"{s}/{q}" for s, q in VAL],
                       validation_ledgers_sha256=ledgers, checkpoints=checks, code_sha256=code,
                       read_trace=opened, test_files_opened=0, test_used_for_selection=False))
    print(target, digest(target))


def test_rows(method: str, seed: int, summary: dict) -> dict:
    """주표와 같은 시험 원장을 읽고 질의·라벨·주표 지표 일치를 확인한다."""
    per, metrics = {}, {}
    for sensor, seq in TEST:
        key = f"{sensor}/{seq}"
        path, record = ledger(method, "test", seed, sensor, seq)
        if method == "bevplace":
            assert record["status"] == "passed" and record["role"] == "test" and record["seed"] == seed
        data = arrays(path)
        reference = arrays(NSD_TEST / "test" / f"{sensor}_{seq}" / "identity.npz")
        for k in ("query_idx", "min_past_gt_dist"):
            np.testing.assert_array_equal(data[k], reference[k])
        qc = classify_queries(data["min_past_gt_dist"])
        row = dict(score=data[SCORE[method]].astype(np.float64), dist=data["top1_geo_dist"], qc=qc,
                   query_idx=data["query_idx"], j_star=data["j_star"], mpgd=data["min_past_gt_dist"],
                   path=str(path.relative_to(REPO)), sha256=record["sha256"])
        per[key] = row
        if qc["n_pos"]:
            sweep = pr_sweep(row["score"], row["dist"], qc)
            metrics[key] = dict(AP=sweep["AP"], max_F1=sweep["max_F1"], R100P=sweep["R@100P"],
                                R1=float(np.mean(row["dist"][qc["positive"]] <= 5)))
    expected = summary["per_case"][SUMMARY_CASE[method].format(seed)]
    for key, m in metrics.items():   # 주표 원천과 같은 원장·같은 지표인지 시퀀스마다 확인한다
        e = expected["sequences"][key]
        for a, b in (("AP", "AP"), ("max_F1", "max_F1"), ("R100P", "R@100P"), ("R1", "R@1")):
            assert abs(m[a] - e[b]) < 1e-12, (method, seed, key, a, m[a], e[b])
    macro, _ = city_mean({k: m["AP"] for k, m in metrics.items()})
    assert abs(macro - expected["macro"]["AP"]) < 1e-12
    return per


def city_mean(per_sequence: dict) -> tuple:
    """양성 있는 시퀀스 → 도시 평균 → 6도시 평균."""
    cities = {c: float(np.mean([per_sequence[s] for s in names if s in per_sequence])) for c, names in CITY.items()}
    return float(np.mean(list(cities.values()))), cities


def dataset_of(sequence: str) -> str:
    return sequence.split("/")[0]


def counts(rows: dict, tau: float) -> dict:
    """고정 τ에서 시퀀스별 판정 칸과 오수락 종류를 센다."""
    out = {}
    for s, r in rows.items():
        p = operating_point(r["score"], r["dist"], r["qc"], tau)
        accepted = np.isfinite(r["score"]) & (r["score"] >= tau)
        wrong = ~(np.isfinite(r["dist"]) & (r["dist"] <= 5))
        out[s] = dict(TP=p["TP"], FP=p["FP"], FN=p["FN"], TN=p["TN"], n_pos=r["qc"]["n_pos"], n_neg=r["qc"]["n_neg"],
                      FP_positive_query=int((accepted & wrong & r["qc"]["positive"]).sum()),
                      FP_negative_query=int((accepted & r["qc"]["negative"]).sum()),
                      ambiguous_accepted=int((accepted & r["qc"]["ambiguous"]).sum()),
                      ambiguous_accepted_wrong=int((accepted & wrong & r["qc"]["ambiguous"]).sum()))
        assert out[s]["FP"] == out[s]["FP_positive_query"] + out[s]["FP_negative_query"]
    return out


def pooled(cells: dict, names) -> dict:
    """주어진 시퀀스들의 판정 칸을 합산해 정밀도·재현율·F1과 오수락 비율을 만든다."""
    tot = {k: sum(cells[s][k] for s in names) for k in next(iter(cells.values()))}
    acc = tot["TP"] + tot["FP"]
    precision = tot["TP"] / acc if acc else 1.0
    recall = tot["TP"] / tot["n_pos"] if tot["n_pos"] else None
    f1 = (2 * precision * recall / (precision + recall) if recall is not None and precision + recall else
          (0.0 if recall is not None else None))
    eligible = tot["n_pos"] + tot["n_neg"]
    return dict(**tot, accepted=acc, precision=precision, recall=recall, F1=f1,
                FP_per_1000_eligible=1000 * tot["FP"] / eligible if eligible else None,
                negative_query_false_accept_rate=tot["FP_negative_query"] / tot["n_neg"] if tot["n_neg"] else None)


def fixed(rows: dict, tau: float) -> dict:
    """고정 τ의 시험 판정: 도시 평균 재현율, 46시퀀스·도시·데이터셋·시퀀스별 합산."""
    cells = counts(rows, tau)
    recall, cities = city_mean({s: c["TP"] / c["n_pos"] for s, c in cells.items() if c["n_pos"]})
    datasets = sorted({dataset_of(s) for s in rows})
    return dict(tau=tau, city_macro_recall=recall, city_recall=cities,
                pooled=pooled(cells, list(rows)),
                city=({c: pooled(cells, names) for c, names in CITY.items()}),
                dataset={d: pooled(cells, [s for s in rows if dataset_of(s) == d]) for d in datasets},
                sequence={s: pooled(cells, [s]) for s in rows})


def mean_sd(values: list) -> dict:
    v = [x for x in values if x is not None]
    return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, values=values)


def evaluate() -> None:
    """고정 τ를 시험에 적용한다. freeze 원천 해시를 먼저 확인한다."""
    frozen = read(OUT / "freeze.json")
    assert frozen["test_used_for_selection"] is False and frozen["test_files_opened"] == 0
    assert frozen["protocol_sha256"] == digest(OUT / "protocol.json")
    for path, value in {**frozen["validation_ledgers_sha256"], **frozen["code_sha256"]}.items():
        if path != str(Path(__file__).relative_to(REPO)):
            assert digest(REPO / path) == value, path
    script_unchanged = digest(Path(__file__)) == frozen["code_sha256"][str(Path(__file__).relative_to(REPO))]
    summary = read(NSD_TEST / "summary.json")
    assert summary["queries_per_case"] == 195726
    report = dict(protocol_sha256=frozen["protocol_sha256"], freeze_sha256=digest(OUT / "freeze.json"),
                  select_script_unchanged_since_freeze=script_unchanged, methods={}, test_ledgers_sha256={})
    for method in METHODS:
        report["methods"][method] = {}
        per_seed_rows = {}
        for seed in SEEDS:
            rows = test_rows(method, seed, summary)
            assert sum(len(r["query_idx"]) for r in rows.values()) == 195726
            report["test_ledgers_sha256"].update({r["path"]: r["sha256"] for r in rows.values()})
            per_seed_rows[seed] = rows
        for point in POINTS:
            per_seed = {}
            for seed in SEEDS:
                result = fixed(per_seed_rows[seed], frozen["thresholds"][method][str(seed)][point]["tau"])
                result["validation"] = {k: v for k, v in frozen["thresholds"][method][str(seed)][point].items()
                                        if k != "val_per_sequence"}
                per_seed[str(seed)] = result
            pick = lambda f: mean_sd([f(per_seed[str(s)]) for s in SEEDS])
            agg = dict(city_macro_recall=pick(lambda r: r["city_macro_recall"]),
                       **{k: pick(lambda r, k=k: r["pooled"][k]) for k in
                          ("precision", "recall", "F1", "TP", "FP", "FP_positive_query", "FP_negative_query",
                           "accepted", "FP_per_1000_eligible", "negative_query_false_accept_rate", "ambiguous_accepted")},
                       val_precision=pick(lambda r: r["validation"]["val_precision"]),
                       val_recall=pick(lambda r: r["validation"]["val_recall"]),
                       val_FP=pick(lambda r: r["validation"]["val_FP"]),
                       city_recall={c: pick(lambda r, c=c: r["city_recall"][c]) for c in CITY},
                       city_precision={c: pick(lambda r, c=c: r["city"][c]["precision"]) for c in CITY},
                       city_FP={c: pick(lambda r, c=c: r["city"][c]["FP"]) for c in CITY},
                       dataset={d: {k: pick(lambda r, d=d, k=k: r["dataset"][d][k]) for k in
                                    ("precision", "recall", "TP", "FP", "FP_positive_query", "FP_negative_query", "accepted")}
                                for d in per_seed["1"]["dataset"]})
            report["methods"][method][point] = dict(mean_sd=agg, per_seed=per_seed)
    write(OUT / "summary.json", report)
    for point in POINTS:
        print("==", point)
        for method in METHODS:
            a = report["methods"][method][point]["mean_sd"]
            print(f"  {method:9s} recall {100*a['city_macro_recall']['mean']:6.2f}±{100*a['city_macro_recall']['sd']:.2f} "
                  f"prec {100*a['precision']['mean']:6.2f}±{100*a['precision']['sd']:.2f} FP {a['FP']['mean']:.1f} "
                  f"(pos {a['FP_positive_query']['mean']:.1f}, neg {a['FP_negative_query']['mean']:.1f}) TP {a['TP']['mean']:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "evaluate"))
    globals()[parser.parse_args().action]()
