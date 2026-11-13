#!/usr/bin/env python3
"""대칭 문맥 3시드 결과를 감사하고 짝지은 비교표와 그림을 만든다."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _pdv2_lag15_campaign import (
    ARMS, CITY, FAMILY, LAG, OPS, OUT, REPO, SEEDS, STATS, TEST, VAL,
    effective_hashes, frozen_guard, prefix_verified_error, protocol, seed_dir,
)
from _pdv2_no_context_ablation import sha256, write_json
from _pr_eval_lib import AGG_METRICS, classify_queries, operating_point, pr_sweep

GROUPS = ("same_available", "opposite_only", "other")
METRICS = (*AGG_METRICS, "R@1", "P_at_tau", "R_at_tau", "key_R@1", "key_R@100", "union_R@200")


def read(path: Path) -> tuple:
    """수치 기록과 메타데이터를 분리해 읽는다."""
    with np.load(path) as archive:
        return ({k: archive[k] for k in archive.files if k != "meta"},
                json.loads(str(archive["meta"])))


def stats(values: list) -> dict:
    """시드 평균과 표본 표준편차를 구한다."""
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            "values": [float(v) for v in values]}


def sequence_metrics(a: dict, decision: str, tau: float) -> dict:
    """공통 질의에서 PR과 검색 지표, 비재방문 오수락을 센다."""
    qc = classify_queries(a["min_past_gt_dist"])
    positive = qc["positive"]
    confidence = a[STATS[decision]]
    row = {k: None for k in METRICS}
    if positive.any():
        sweep = pr_sweep(confidence, a["top1_geo_dist"], qc)
        op = operating_point(confidence, a["top1_geo_dist"], qc, tau)
        row.update({k: float(sweep[k]) for k in AGG_METRICS})
        row.update({"R@1": float((a["top1_geo_dist"][positive] <= 5).mean()),
                    "P_at_tau": float(op["precision"]), "R_at_tau": float(op["recall"]),
                    "key_R@1": float((a["key_top1_geo"][positive] <= 5).mean()),
                    "key_R@100": float(a["key_hit"][positive].mean()),
                    "union_R@200": float(a["union_hit"][positive].mean())})
    row.update(n_queries=qc["n_total"], n_positive=qc["n_pos"], n_negative=qc["n_neg"],
               n_ambiguous=qc["n_ambiguous"],
               negative_false_accepts=int((qc["negative"] & (confidence >= tau)).sum()))
    return row


def summarize(stage: str) -> dict:
    """시퀀스→검증 계열/시험 도시→세 시드의 순서로 평균을 낸다."""
    role, mode = stage.split("_")
    roster = VAL if role == "val" else TEST
    arms = ARMS if mode == "fixed" else ("no_context", "symmetric")
    grouping = CITY if role == "test" else {}
    if role == "val":
        for sensor, seq in roster:
            grouping.setdefault(FAMILY.get(sensor, sensor), []).append(sensor + "/" + seq)
    result = {"groups": grouping, "ops": {}}
    for op in OPS:
        seed_results = {}
        for seed in SEEDS:
            selection = (protocol()["fixed_selection"] if mode == "fixed" else
                         json.loads((seed_dir(seed) / "confidence_selection.json").read_text())["ops"])[op]
            outputs = {arm: {"per_sequence": {}, "direction": {g: {"n": 0, "key": 0, "full": 0,
                        "key_hit": 0, "union_hit": 0} for g in GROUPS}} for arm in arms}
            paired = {other: {g: {"n": 0, "key_recovered": 0, "key_regressed": 0,
                                 "full_recovered": 0, "full_regressed": 0}
                              for g in ("all", *GROUPS)} for other in arms if other != "symmetric"}
            for sensor, seq in roster:
                records = {}
                for arm in arms:
                    a, meta = read(seed_dir(seed) / f"{stage}_{arm}_{op}_{sensor}_{seq}.npz")
                    records[arm] = a
                    row = sequence_metrics(a, selection["decision"], selection["tau_star"])
                    outputs[arm]["per_sequence"][sensor + "/" + seq] = row
                    for i, group in enumerate(GROUPS):
                        mask = (a["min_past_gt_dist"] <= 5) & (a["group"] == i)
                        d = outputs[arm]["direction"][group]
                        d["n"] += int(mask.sum())
                        for name, ok in (("key", a["key_top1_geo"] <= 5), ("full", a["top1_geo_dist"] <= 5),
                                         ("key_hit", a["key_hit"]), ("union_hit", a["union_hit"])):
                            d[name] += int(ok[mask].sum())
                after = records["symmetric"]
                for other, groups in paired.items():
                    before = records[other]
                    for i, group in enumerate(("all", *GROUPS)):
                        mask = after["min_past_gt_dist"] <= 5
                        if i:
                            mask &= after["group"] == i - 1
                        d = groups[group]
                        d["n"] += int(mask.sum())
                        for key, field in (("key", "key_top1_geo"), ("full", "top1_geo_dist")):
                            old, new = before[field] <= 5, after[field] <= 5
                            d[key + "_recovered"] += int((mask & ~old & new).sum())
                            d[key + "_regressed"] += int((mask & old & ~new).sum())
            for output in outputs.values():
                per_seq = output["per_sequence"]
                group_values = {}
                for group, names in grouping.items():
                    valid = [per_seq[name] for name in names if per_seq[name]["n_positive"] > 0]
                    assert valid, group
                    group_values[group] = {k: float(np.mean([r[k] for r in valid])) for k in METRICS}
                output["macro"] = {k: float(np.mean([r[k] for r in group_values.values()])) for k in METRICS}
                output["group_macro"] = group_values
                output["excluded_no_positive"] = [k for k, r in per_seq.items() if not r["n_positive"]]
                for d in output["direction"].values():
                    for name, numerator in (("key_R@1", "key"), ("full_R@1", "full"),
                                            ("key_R@100", "key_hit"), ("union_R@200", "union_hit")):
                        d[name] = d[numerator] / d["n"] if d["n"] else None
            seed_results[str(seed)] = {"arms": outputs, "selection": selection, "symmetric_paired_vs": paired}
        means = {arm: {k: stats([v["arms"][arm]["macro"][k] for v in seed_results.values()])
                       for k in METRICS} for arm in arms}
        deltas = {other: {k: stats([v["arms"]["symmetric"]["macro"][k] - v["arms"][other]["macro"][k]
                                   for v in seed_results.values()]) for k in METRICS}
                  for other in arms if other != "symmetric"}
        result["ops"][op] = {"per_seed": seed_results, "mean_std": means, "symmetric_minus": deltas}
    return result


def audit(include_test: bool) -> dict:
    """모든 기록의 해시·동일 질의·과거 후보·유한성·재사용 원천을 확인한다."""
    spec = protocol()
    if include_test:
        frozen_guard(full=True)
    ph = sha256(OUT / "protocol.json")
    checked = {}
    def check(name: str, expected: str) -> None:
        if name not in checked:
            checked[name] = sha256(REPO / name)
        assert checked[name] == expected, name
    for name, expected in effective_hashes(spec).items():
        check(name, expected)
    amendment_path = OUT / "prefix_verification_amendment.json"
    diagnosis = None
    if amendment_path.exists():
        amendment = json.loads(amendment_path.read_text())
        dp = OUT / "seed3_prefix_numerical_diagnosis.json"
        check(str(dp.relative_to(REPO)), amendment["diagnosis_sha256"])
        diagnosis = json.loads(dp.read_text())
        assert diagnosis["cpu"]["max_prefix_error"] < 2e-6
    count, rows, prefix, query_counts = 0, 0, [], {}
    stages = ("val_fixed", "val_selected", "test_selected") if include_test else ("val_fixed", "val_selected")
    for seed in SEEDS:
        for stage in stages:
            role, mode = stage.split("_")
            roster = VAL if role == "val" else TEST
            selection = (spec["fixed_selection"] if mode == "fixed" else
                         json.loads((seed_dir(seed) / "confidence_selection.json").read_text())["ops"])
            for sensor, seq in roster:
                kp = seed_dir(seed) / f"keys_{role}_{sensor}_{seq}.npz"
                keys, km = read(kp)
                assert km["protocol_sha256"] == ph
                for name, expected in {**km["inputs_sha256"], **km["artifacts_sha256"]}.items():
                    check(name, expected)
                if stage == "val_fixed":
                    prefix.extend({"seed": seed, "sequence": f"{sensor}/{seq}", **r} for r in km["prefix_checks"])
                n = len(keys["scan_ids"])
                q = np.arange(30, n - LAG)
                for arm in ("past", "symmetric") if role == "val" else ("symmetric",):
                    emb = keys[arm]
                    assert emb.dtype == np.float32 and emb.shape == (n, 416)
                    assert np.isfinite(emb).all()
                    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1, atol=2e-7)
                reference = None
                for op in OPS:
                    for arm in ARMS if mode == "fixed" else ("no_context", "symmetric"):
                        path = seed_dir(seed) / f"{stage}_{arm}_{op}_{sensor}_{seq}.npz"
                        a, meta = read(path)
                        assert meta["protocol_sha256"] == ph
                        assert meta["tuple"] == selection[op]["tuple"]
                        check(str(kp.relative_to(REPO)), meta["keys_sha256"])
                        if role == "test":
                            assert meta["freeze_sha256"] == sha256(OUT / "freeze.json")
                        np.testing.assert_array_equal(q, a["query_idx"])
                        for name, arr in a.items():
                            assert arr.shape == (len(q),) and np.isfinite(arr).all(), (path.name, name)
                        assert ((a["j_star"] >= 0) & (a["j_star"] <= q - 30)).all()
                        assert ((a["n_candidates"] >= 1) & (a["n_candidates"] <= 200)).all()
                        assert np.isin(a["group"], [-1, 0, 1, 2]).all()
                        if reference is None:
                            reference = a
                            query_counts[f"s{seed}/{stage}/{sensor}/{seq}"] = {
                                "queries": len(q), "positive": int((a["min_past_gt_dist"] <= 5).sum())}
                        else:
                            for name in ("query_idx", "min_past_gt_dist", "group"):
                                np.testing.assert_array_equal(reference[name], a[name])
                        reused = meta.get("reused_identical_settings")
                        if reused:
                            check(reused["path"], reused["sha256"])
                            old, om = read(REPO / reused["path"])
                            for name in a:
                                np.testing.assert_array_equal(a[name], old[name])
                        count += 1
                        rows += len(q)
    assert len(prefix) == 27 and max(prefix_verified_error(r) for r in prefix) < 2e-6
    expected_count = len(SEEDS) * len(OPS) * (len(VAL) * 5 + (len(TEST) * 2 if include_test else 0))
    assert count == expected_count, (count, expected_count)
    return {"passed": True, "ledgers": count, "rows": rows, "checked_sha256": checked,
            "prefix_checks": prefix, "query_counts": query_counts,
            "preserved_gpu_tolerance_failure_and_cpu_reference": diagnosis}


def figures(summary: dict) -> None:
    """개별 시드와 방향별 결과를 논문 검토용 PDF·PNG로 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fontpath = OUT / "UnDotum.ttf"
    if fontpath.exists():
        prop = FontProperties(fname=str(fontpath))
        matplotlib.font_manager.fontManager.addfont(str(fontpath))
        plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    stages = list(summary["stages"])
    fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 3.8),
                             sharey=True, constrained_layout=True)
    fig.suptitle("점: 개별 시드 · 검은 마름모: 평균 ± 표본 표준편차", fontsize=11)
    for ax, stage in zip(np.atleast_1d(axes), stages):
        data = summary["stages"][stage]
        for i, op in enumerate(OPS):
            d = data["ops"][op]["symmetric_minus"]["no_context"]["AP"]
            x = i + np.array([-.08, 0, .08])
            ax.scatter(x, np.array(d["values"]) * 100, s=35, color=("#176caa", "#dd762b")[i])
            ax.errorbar(i + .17, d["mean"] * 100, yerr=d["std"] * 100, fmt="D", color="black", capsize=4)
        ax.axhline(0, color="gray", linewidth=.8)
        ax.set_xticks([0, 1], ["NSD", "NSD-H"])
        ax.set_title({"val_fixed": "검증: 검색 설정 공통 고정", "val_selected": "검증: 시드별 설정 선택",
                      "test_selected": "시험: 검증 선택 설정 고정"}[stage])
        ax.set_ylabel("문맥 제거 대비 AP 차이 (%p)")
        ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "three_seed_context_gain.pdf")
    fig.savefig(OUT / "three_seed_context_gain.png", dpi=180)
    plt.close(fig)
    stage = "test_selected" if "test_selected" in summary["stages"] else "val_fixed"
    data = summary["stages"][stage]["ops"][OPS[0]]["per_seed"]
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    for i, arm in enumerate(("no_context", "symmetric")):
        values = np.array([[v["arms"][arm]["direction"][g]["key_R@1"] for g in GROUPS[:2]]
                           for v in data.values()]) * 100
        ax.bar(np.arange(2) + (i - .5) * .32, values.mean(0), .32,
               yerr=values.std(0, ddof=1), capsize=4, label=("문맥 제거", "대칭 문맥")[i])
    ax.set_xticks([0, 1], ["같은 방향 재방문 가능", "반대 방향 재방문만 가능"])
    ax.set_ylabel("검색 키 R@1 (%)")
    ax.set_title("시험 방향별 검색 키 성능" if stage.startswith("test") else "검증 방향별 검색 키 성능")
    ax.legend()
    fig.savefig(OUT / "direction_key_three_seed.pdf")
    fig.savefig(OUT / "direction_key_three_seed.png", dpi=180)
    plt.close(fig)


def timing_and_gates(include_test: bool) -> dict:
    """키프레임 대기 시간과 실제 문맥 블록의 상대 길이를 기록한다."""
    result = {}
    for role in (("val", "test") if include_test else ("val",)):
        roster = VAL if role == "val" else TEST
        seconds, travel, per_seq = [], [], {}
        gates = {str(s): {a: [] for a in (("past", "symmetric") if role == "val" else ("symmetric",))}
                 for s in SEEDS}
        def spread(a):
            return {"min": float(a.min()), "median": float(np.median(a)),
                    "p95": float(np.quantile(a, .95)), "max": float(a.max())}
        for sensor, seq in roster:
            keys, meta = read(seed_dir(1) / f"keys_{role}_{sensor}_{seq}.npz")
            cache_name = next(n for n in meta["inputs_sha256"] if Path(n).name.startswith("cache_"))
            with np.load(REPO / cache_name) as cache:
                q = np.arange(30, len(cache["scan_ids"]) - LAG)
                delay = cache["timestamps"][q + LAG] - cache["timestamps"][q]
                pos = cache["poses"][:, :3, 3]
                distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
                moved = distance[q + LAG] - distance[q]
                assert np.isfinite(delay).all() and (delay >= 0).all()
            seconds.append(delay)
            travel.append(moved)
            per_seq[sensor + "/" + seq] = {"seconds": spread(delay), "odom_travel_m": spread(moved)}
            for seed in SEEDS:
                current = keys if seed == 1 else read(seed_dir(seed) / f"keys_{role}_{sensor}_{seq}.npz")[0]
                for arm in gates[str(seed)]:
                    emb = current[arm][q]
                    alpha = np.linalg.norm(emb[:, 288:], axis=1) / np.linalg.norm(emb[:, :288], axis=1)
                    gates[str(seed)][arm].append(alpha)
        result[role] = {"delay": {"seconds": spread(np.concatenate(seconds)),
                                  "odom_travel_m": spread(np.concatenate(travel)), "per_sequence": per_seq},
                        "gate_alpha": {seed: {arm: spread(np.concatenate(parts)) for arm, parts in arms.items()}
                                       for seed, arms in gates.items()}}
    return result


def main() -> None:
    """검증 중간 집계 또는 완료된 전체 집계를 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-only", action="store_true")
    args = parser.parse_args()
    stages = ["val_fixed", "val_selected"] + ([] if args.val_only else ["test_selected"])
    result = {"protocol_sha256": sha256(OUT / "protocol.json"),
              "report_script_sha256": sha256(Path(__file__)),
              "note": "검증은 시퀀스→3계열, 시험은 시퀀스→6도시 평균. 양성 없는 시퀀스는 AP·재현율 평균에서 공통 제외. ±는 3시드 표본 표준편차이며 유의성 검정이 아니다.",
              "stages": {stage: summarize(stage) for stage in stages},
              "timing_and_gates": timing_and_gates(not args.val_only)}
    verified = audit(not args.val_only)
    write_json(OUT / ("audit_val.json" if args.val_only else "audit.json"), verified)
    write_json(OUT / ("summary_val.json" if args.val_only else "summary.json"), result)
    with (OUT / "metrics_three_seed.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stage", "op", "arm", "metric", "mean", "std", "s1", "s2", "s3"))
        for stage, content in result["stages"].items():
            for op, value in content["ops"].items():
                for arm, metrics in value["mean_std"].items():
                    for metric, row in metrics.items():
                        writer.writerow((stage, op, arm, metric, row["mean"], row["std"], *row["values"]))
    figures(result)
    print(json.dumps({stage: {op: {arm: vals["AP"] for arm, vals in row["mean_std"].items()}
                              for op, row in content["ops"].items()}
                      for stage, content in result["stages"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
