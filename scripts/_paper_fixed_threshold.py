#!/usr/bin/env python3
"""검증에서 고정한 단일 판정 기준값으로 시험 최종 판정을 비교한다.

nsd_val: 원고 주모델(`p050__aux_half__equal`)의 검증 9시퀀스 최종 점수를 원래 식으로 재구성한다(도커, torch 필요).
select: 검증 원장만으로 NSD·BEVPlace++의 시드마다 τ를 정해 freeze.json에 고정한다. 시험 원장은 읽지 않는다.
evaluate: 고정 τ를 시험 46시퀀스에 그대로 적용한다. 나머지 9개 방법은 시험에서 고른 단일 τ의 상한과 비교한다.
table: summary.json으로 원고 부록 표를 만든다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from _pdv2_main_baselines_official import METHODS
from _pdv2_baseline_seeds import SEED_FOLDERS, TRAINING_FREE
from _pdv2_pilot import CITY, TEST, VAL
from _pr_eval_lib import classify_queries, operating_point, pr_sweep

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/paper_fixed_threshold_20261101"
MAIN = REPO / "results/pdv2_spectral_whitening_20261026/fixed_test"   # 표 2 공통 질의 기준
POWER = REPO / "results/pdv2_spectral_power_20261101"
ARM = "p050__aux_half__equal"   # 2026-11-01 원고 주모델(adoption_decision.json)
NSD_VAL = OUT / "nsd_val"
TABLE = REPO / "docs/paper/body_v2.2/tables_v2/appdx_fixed_threshold.tex"   # 부록 표
BEV_VAL = {1: "results/pdv2_bevplace_val_odom_20261018/dump_bevplace_{}.npz",
           2: "results/pdv2_bevplace_val_odom_seeds_20261101/dump_bevplace_s2_{}.npz",
           3: "results/pdv2_bevplace_val_odom_seeds_20261101/dump_bevplace_s3_{}.npz"}
SEEDS = (1, 2, 3)
FIXED = ("nsd", "bevplace")
POINTS = {"P99": .99, "P100": 1.0, "F1": None}
LABELS = dict(scpp="SC++", ring="RING", ringpp="RING++", ringsharp="RING#-L", deepring="DeepRING",
              disco="DiSCO", helios="HeLiOS", seqot="SeqOT", imlpr="ImLPR", bevplace="BEVPlace++", nsd="NSD")
RULES = {
    "purpose": "그림 5의 시퀀스별 최적 τ(모든 방법에 같은 규칙)를 배포형 단일 τ로 바꿔 NSD와 BEVPlace++를 비교한다.",
    "methods": f"NSD는 2026-11-01 원고 주모델 {ARM}(3시드), BEVPlace++는 공식 3시드(표 2와 같은 체크포인트).",
    "threshold": "방법·시드마다 τ 하나를 검증 9시퀀스(KISS-ICP 키프레임, 99,236질의)의 합산 질의로 고르고 시험 46시퀀스 "
                 "전체에 그대로 쓴다. 시험 원장은 τ 선택에 쓰지 않는다.",
    "operating_points": "P99: 검증 합산 정밀도 ≥ 99%인 τ 중 검증 재현율 최대. P100: 검증 오수락 0인 τ 중 검증 재현율 최대. "
                        "F1: 검증 합산 F1 최대. 같은 재현율이면 가장 높은 τ(라이브러리 F1 동률 규칙과 같음).",
    "accounting": "_pr_eval_lib 그대로: 양성 ≤5 m, 음성 >25 m, 5–25 m 질의는 제외. 점수 ≥ τ면 수락. 수락한 1등이 5 m "
                  "밖이면 질의 종류와 상관없이 오수락. 재현율 분모는 양성 질의 수.",
    "test_metrics": "재현율은 양성 있는 45시퀀스 → 동일 가중치 6도시 → 시드 평균(표 2와 같음). 정밀도와 오수락 수는 "
                    "KITTI 07을 포함한 46시퀀스 합산. 도시별 재현율·합산 정밀도도 보고한다.",
    "upper_bound": "나머지 9개 방법은 검증 원장이 없으므로 시험에서 직접 고른 단일 τ(합산 정밀도 하한 아래 도시 평균 재현율 "
                   "최대)를 어떤 단일 τ로도 넘을 수 없는 상한으로 쓴다. 하한은 같은 운영점에서 NSD가 얻은 시험 합산 정밀도다.",
    "scope": "모델·후보·승자·점수 변경 없음. 표 2와 그림 5의 값은 바꾸지 않는다.",
}


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nsd_val() -> None:
    """주모델의 검증 최종 점수·승자·거리를 `_pdv2_spectral_power.finish`와 같은 식으로 재구성해 검산한다."""
    import _pdv2_current_calibration as cal
    spec = cal.read(POWER / "protocol.json")
    power, fusion, channel = ARM.split("__")
    fi = cal.FUSIONS.index(fusion)
    checks = []
    for seed in SEEDS:
        for sensor, seq in VAL:
            name = f"{sensor}_{seq}"
            record = cal.read(POWER / "val" / f"s{seed}" / f"{name}.json")
            assert record["status"] == "passed"
            assert digest(REPO / record["ledger"]) == record["sha256"]
            assert digest(REPO / record["source"]) == record["source_sha256"]
            source, values = cal.arrays(REPO / record["source"]), cal.arrays(REPO / record["ledger"])
            np.testing.assert_array_equal(source["query_idx"], values["query_idx"])
            score = cal.confidence(values[power + "_c_spec"][:, fi], values[power + "_c_col"][:, fi], channel,
                                   spec["confidence"][str(seed)]["decision"])
            row = {k: source[k] for k in cal.base.FIELDS}
            row.update(j_star=source["candidates"][:, fi], top1_geo_dist=source["candidate_geo"][:, fi])
            assert cal.base.independent_metrics(row, score) == record["arms"][ARM]
            replay = POWER / "validation_replay" / f"s{seed}" / f"{name}_{ARM}.npz"
            if replay.exists():
                with np.load(replay) as other:
                    np.testing.assert_array_equal(other["score"], score)
                    for key in ("query_idx", "j_star", "top1_geo_dist", "min_past_gt_dist"):
                        np.testing.assert_array_equal(other[key], row[key])
            path = NSD_VAL / f"s{seed}" / f"{name}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, score=score, **row)
            checks.append(dict(seed=seed, sequence=f"{sensor}/{seq}", metrics_equal_record=True,
                               replay_equal=replay.exists(), sha256=digest(path)))
    write(NSD_VAL / "validation.json", dict(arm=ARM, status="passed", ledgers=checks, test_used=False))


def val_rows(method: str, seed: int) -> list:
    """검증 9시퀀스의 (시퀀스, 점수, 1등 거리, 과거 최근접 거리, 원장 경로). 질의는 NSD 시드 1과 대조한다."""
    rows = []
    for sensor, seq in VAL:
        if method == "nsd":
            path = NSD_VAL / f"s{seed}" / f"{sensor}_{seq}.npz"
            key = "score"
        else:
            path = REPO / BEV_VAL[seed].format(f"{sensor}_{seq}")
            key = "score_negl2"
        with np.load(path) as data, np.load(NSD_VAL / "s1" / f"{sensor}_{seq}.npz") as reference:
            for name in ("query_idx", "min_past_gt_dist"):
                np.testing.assert_array_equal(data[name], reference[name])
            rows.append((f"{sensor}/{seq}", data[key].astype(np.float64), data["top1_geo_dist"],
                         data["min_past_gt_dist"], path))
    return rows


def choose(rows: list) -> dict:
    """검증 합산 질의로 세 운영점의 τ를 고른다."""
    scores = np.concatenate([r[1] for r in rows])
    dists = np.concatenate([r[2] for r in rows])
    qc = classify_queries(np.concatenate([r[3] for r in rows]))
    sweep = pr_sweep(scores, dists, qc)
    chosen = {}
    for name, floor in POINTS.items():
        if floor is None:
            tau = sweep["max_F1_tau"]
        else:
            ok = np.flatnonzero(sweep["precision"] >= floor)
            best = sweep["recall"][ok].max()
            tau = float(sweep["thresholds"][ok[sweep["recall"][ok] == best][0]])
        point = operating_point(scores, dists, qc, tau)
        chosen[name] = dict(tau=tau, val_precision=point["precision"], val_recall=point["recall"],
                            val_FP=point["FP"], val_TP=point["TP"])
    chosen["val_counts"] = sweep["counts"]
    return chosen


def select() -> None:
    """검증만으로 τ를 고정한다. 이미 고정했으면 아무것도 하지 않는다."""
    if (OUT / "freeze.json").exists():
        print("FROZEN_ALREADY"); return
    thresholds, sources = {}, {str(Path(__file__).relative_to(REPO)): digest(Path(__file__)),
                               "scripts/_pr_eval_lib.py": digest(REPO / "scripts/_pr_eval_lib.py")}
    for method in FIXED:
        thresholds[method] = {}
        for seed in SEEDS:
            rows = val_rows(method, seed)
            for row in rows:
                sources[str(row[4].relative_to(REPO))] = digest(row[4])
            thresholds[method][str(seed)] = choose(rows)
            print(method, seed, {k: v for k, v in thresholds[method][str(seed)].items() if k != "val_counts"}, flush=True)
    write(OUT / "freeze.json", dict(rules=RULES, thresholds=thresholds, validation=[f"{s}/{q}" for s, q in VAL],
                                    sources=sources, test_used_for_selection=False))


def test_runs() -> dict:
    """표 2와 같은 시험 원장·공통 질의를 방법·시드별로 읽고 원천 해시를 확인한다."""
    base = read(REPO / "results/pdv2_baseline_seeds/summary.json")
    references = {}
    for sensor, seq in TEST:
        with np.load(MAIN / "s1" / f"{sensor}_{seq}.npz") as data:
            references[f"{sensor}/{seq}"] = dict(query_idx=data["query_idx"], mpgd=data["min_past_gt_dist"])
    runs = {}
    for method, folder, prefix, score, _, _ in list(METHODS) + [("nsd", "", "", "score", "", "")]:
        seeds = ([(s, POWER / "test" / f"s{s}") for s in SEEDS] if method == "nsd" else
                 [(None, REPO / folder)] if method in TRAINING_FREE else
                 [(s, REPO / f) for s, f in SEED_FOLDERS[method]])
        runs[method] = {}
        for seed, source in seeds:
            per = {}
            for sensor, seq in TEST:
                name = f"{sensor}_{seq}.npz"
                path = source / (f"{sensor}_{seq}_{ARM}.npz" if method == "nsd" else prefix + "_" + name)
                expected = (read(source / f"{sensor}_{seq}.json")["arms"][ARM]["sha256"] if method == "nsd"
                            else base["input_sha256"][str(path.relative_to(REPO))])
                assert digest(path) == expected, path
                ref = references[f"{sensor}/{seq}"]
                with np.load(path) as data:
                    ix = np.searchsorted(data["query_idx"], ref["query_idx"])
                    np.testing.assert_array_equal(data["query_idx"][ix], ref["query_idx"])
                    np.testing.assert_array_equal(data["min_past_gt_dist"][ix], ref["mpgd"])
                    per[f"{sensor}/{seq}"] = (data[score][ix].astype(np.float64), data["top1_geo_dist"][ix],
                                              classify_queries(ref["mpgd"]))
            runs[method]["-" if seed is None else str(seed)] = per
    return runs


def city_mean(per_sequence: dict) -> tuple:
    """양성 있는 시퀀스 → 도시 평균 → 6도시 평균."""
    cities = {c: float(np.mean([per_sequence[s] for s in names if s in per_sequence])) for c, names in CITY.items()}
    return float(np.mean(list(cities.values()))), cities


def fixed(per: dict, tau: float) -> dict:
    """고정 τ의 시험 판정: 도시 평균 재현율, 46시퀀스 합산 정밀도·오수락."""
    points = {s: operating_point(*row, tau) for s, row in per.items()}
    recall, cities = city_mean({s: p["recall"] for s, p in points.items() if per[s][2]["n_pos"]})
    tp, fp = sum(p["TP"] for p in points.values()), sum(p["FP"] for p in points.values())
    city_precision = {}
    for city, names in CITY.items():
        t, f = sum(points[s]["TP"] for s in names), sum(points[s]["FP"] for s in names)
        city_precision[city] = t / (t + f) if t + f else 1.0
    return dict(tau=tau, recall=recall, precision=tp / (tp + fp) if tp + fp else 1.0, TP=tp, FP=fp,
                accepted=tp + fp, city_recall=cities, city_precision=city_precision,
                negative_query_FP=sum(int((np.isfinite(r[0]) & (r[0] >= tau) & r[2]["negative"]).sum())
                                      for r in per.values()))


def single_tau_curve(per: dict) -> tuple:
    """모든 시퀀스에 같은 τ를 쓸 때 τ별 합산 정밀도와 도시 평균 재현율."""
    prepared = []
    for s, (score, dist, qc) in per.items():
        pos, neg = qc["positive"], qc["negative"]
        keep = (pos | neg) & np.isfinite(score)
        correct = np.isfinite(dist) & (dist <= 5)
        order = np.argsort(-score[keep], kind="stable")
        prepared.append((s, score[keep][order], np.r_[0, np.cumsum((pos & correct)[keep][order])],
                         np.r_[0, np.cumsum((neg | (pos & ~correct))[keep][order])], qc["n_pos"]))
    taus = np.unique(np.concatenate([p[1] for p in prepared]))[::-1]
    tp_total, fp_total = np.zeros(len(taus)), np.zeros(len(taus))
    recalls = {}
    for s, ordered, tp, fp, n_pos in prepared:
        k = np.searchsorted(-ordered, -taus, side="right")
        tp_total += tp[k]; fp_total += fp[k]
        if n_pos:
            recalls[s] = tp[k] / n_pos
    cities = [np.mean([recalls[s] for s in names if s in recalls], axis=0) for names in CITY.values()]
    precision = np.where(tp_total + fp_total > 0, tp_total / np.maximum(tp_total + fp_total, 1), 1.0)
    return taus, precision, np.mean(cities, axis=0)


def upper_bound(curve: tuple, floor: float) -> dict:
    """합산 정밀도 하한을 지키는 단일 τ 중 도시 평균 재현율 최대(시험에서 직접 고른 상한)."""
    taus, precision, recall = curve
    ok = np.flatnonzero(precision >= floor - 1e-12)
    if not len(ok):
        return dict(recall=0.0, tau=None, precision=None)
    i = ok[np.argmax(recall[ok])]
    return dict(recall=float(recall[i]), tau=float(taus[i]), precision=float(precision[i]))


def evaluate() -> None:
    """고정 τ를 시험에 적용하고 나머지 방법의 단일 τ 상한과 비교한다."""
    frozen = read(OUT / "freeze.json")
    for path, value in frozen["sources"].items():
        if path != str(Path(__file__).relative_to(REPO)):
            assert digest(REPO / path) == value, path
    runs = test_runs()
    nsd_table = read(POWER / "table2_summary.json")["per_seed"][ARM]
    base = read(REPO / "results/pdv2_baseline_seeds/summary.json")
    for method in FIXED:   # 표 2의 R@100P를 다시 계산해 같은 원장을 읽었는지 확인한다
        for seed, per in runs[method].items():
            value, _ = city_mean({s: pr_sweep(*row)["R@100P"] for s, row in per.items() if row[2]["n_pos"]})
            expected = (nsd_table[seed] if method == "nsd" else base["methods"][method]["per_seed"][seed])["macro"]["R@100P"]
            assert abs(value - expected) < 1e-12, (method, seed, value, expected)
    curves = {m: {seed: single_tau_curve(per) for seed, per in seeds.items()} for m, seeds in runs.items()}
    report = dict(rules=frozen["rules"], fixed={}, upper_bounds={})
    for method in FIXED:
        report["fixed"][method] = {}
        for point in POINTS:
            per_seed = {}
            for seed, per in runs[method].items():
                result = fixed(per, frozen["thresholds"][method][seed][point]["tau"])
                taus, precision, recall = curves[method][seed]
                above = np.flatnonzero(taus >= result["tau"])   # 점수 ≥ τ인 수락 집합은 이 시험 점수의 것과 같다
                if len(above):
                    i = above[-1]
                    assert abs(recall[i] - result["recall"]) < 1e-12 and abs(precision[i] - result["precision"]) < 1e-12
                else:
                    assert result["accepted"] == 0
                result["test_upper_bound_same_precision"] = upper_bound(curves[method][seed], result["precision"])
                result["validation"] = frozen["thresholds"][method][seed][point]
                per_seed[seed] = result
            summary = {k: dict(mean=float(np.mean([r[k] for r in per_seed.values()])),
                               sd=float(np.std([r[k] for r in per_seed.values()], ddof=1)))
                       for k in ("recall", "precision", "FP", "negative_query_FP", "accepted")}
            summary["city_recall"] = {c: float(np.mean([r["city_recall"][c] for r in per_seed.values()])) for c in CITY}
            summary["city_precision"] = {c: float(np.mean([r["city_precision"][c] for r in per_seed.values()])) for c in CITY}
            report["fixed"][method][point] = dict(summary=summary, per_seed=per_seed)
    for point in POINTS:
        floor = report["fixed"]["nsd"][point]["summary"]["precision"]["mean"]
        nsd_recall = report["fixed"]["nsd"][point]["summary"]["recall"]["mean"]
        rows = {}
        for method in runs:
            bounds = {seed: upper_bound(curves[method][seed], floor) for seed in runs[method]}
            mean = float(np.mean([b["recall"] for b in bounds.values()]))
            rows[method] = dict(mean_recall=mean, per_seed=bounds, below_nsd_fixed=mean < nsd_recall)
        report["upper_bounds"][point] = dict(precision_floor=floor, nsd_fixed_recall=nsd_recall, methods=rows)
    write(OUT / "summary.json", report)
    for point in POINTS:
        print(f"== {point}")
        for method in FIXED:
            s = report["fixed"][method][point]["summary"]
            print(f"  {LABELS[method]:11s} recall {s['recall']['mean']*100:6.2f}±{s['recall']['sd']*100:.2f}  "
                  f"precision {s['precision']['mean']*100:6.2f}±{s['precision']['sd']*100:.2f}  FP {s['FP']['mean']:.0f}")
        u = report["upper_bounds"][point]
        print(f"  upper bounds at precision >= {u['precision_floor']*100:.2f}% (NSD fixed recall {u['nsd_fixed_recall']*100:.2f})")
        for method, row in u["methods"].items():
            print(f"    {LABELS[method]:11s} {row['mean_recall']*100:6.2f} {'below' if row['below_nsd_fixed'] else 'NOT below'}")


def table() -> None:
    """부록 표 `tab:fixed_threshold`를 summary.json에서 만든다."""
    names = {"P99": "P99", "P100": "P100", "F1": "F1"}
    report = read(OUT / "summary.json")
    rows = []
    for point in POINTS:
        for i, method in enumerate(FIXED):
            s = report["fixed"][method][point]["summary"]
            rows.append(f"{names[point] if i == 0 else ''} & {LABELS[method]} & "
                        f"${s['recall']['mean'] * 100:.2f}\\pm{s['recall']['sd'] * 100:.2f}$ & "
                        f"{s['precision']['mean'] * 100:.2f} & {s['FP']['mean']:,.0f} \\\\")
        rows.append(r"\midrule")
    lines = [
        "% scripts/_paper_fixed_threshold.py table이 만든다. 숫자를 직접 고치지 않는다.",
        f"% 출처: results/paper_fixed_threshold_20261101/summary.json (sha256 {digest(OUT / 'summary.json')}).",
        r"\begin{table}[!htbp]",
        r"\caption{\cmedit{\textbf{Single deployment threshold.} One acceptance threshold per method and seed, "
        r"fixed on validation and applied to all 46 test sequences. Recall is city-averaged; precision and "
        r"false accepts pool all test queries. P99 and P100 maximize validation recall at $\geq$99\% and 100\% "
        r"validation precision; F1 maximizes validation F1. Means over three seeds; recall $\pm$ seed SD.}}",
        r"\label{tab:fixed_threshold}",
        r"\centering\footnotesize\color{blue}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Threshold & Method & Recall (\%) & Prec.\ (\%) & False accepts \\",
        r"\midrule",
        *rows[:-1],
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    TABLE.write_text("\n".join(lines) + "\n")
    print(TABLE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("nsd_val", "select", "evaluate", "table"))
    globals()[parser.parse_args().action]()
