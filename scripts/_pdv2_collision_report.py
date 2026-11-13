#!/usr/bin/env python3
"""고정 충돌·퍼짐 측정의 집계, 그림, 재현 가능한 사례 목록을 만든다."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _collision_diagnostics_lib import CATEGORIES, transition_counts  # noqa: E402
from _pdv2_pilot import CITY, TEST  # noqa: E402
from _pr_eval_lib import classify_queries, operating_point, pr_sweep  # noqa: E402

STAGES = ("raw", "weighted", "pass1", "context", "pbev128_main", "hspec12")
LABELS = ("Raw d", "Weighted d", "Pass 1", "Final key", "NSD", "NSD-H")
OPS = STAGES[-2:]
CITY_OF = {seq: city for city, sequences in CITY.items() for seq in sequences}


def ratio(numerator: int, denominator: int) -> float | None:
    """빈 분모는 0으로 바꾸지 않는다."""
    return float(numerator / denominator) if denominator else None


def write_json(path: Path, value: dict) -> None:
    """NaN 없는 JSON을 쓴다."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def summarize(rows: list[dict]) -> dict:
    """질의 통합과 시퀀스→도시 평균을 각각 계산한다."""
    keys = [key for key in rows[0]["rates"]]
    cities = {}
    for city in CITY:
        subset = [row for row in rows if row["city"] == city]
        cities[city] = {key: float(np.mean([r["rates"][key] for r in subset
                                          if r["rates"][key] is not None]))
                        for key in keys if any(r["rates"][key] is not None for r in subset)}
    return {"by_city": cities,
            "city_macro": {key: float(np.mean([c[key] for c in cities.values() if key in c]))
                           for key in keys if any(key in c for c in cities.values())},
            "n_cities_with_denominator": {key: sum(key in c for c in cities.values()) for key in keys},
            "n_sequences_with_denominator": {key: sum(r["rates"][key] is not None for r in rows)
                                               for key in keys}}


def collect(out: Path) -> tuple[dict, list[dict]]:
    """46시퀀스의 분모·시간 제약·순위 일관성을 검사하고 집계한다."""
    protocol_hash = hashlib.sha256((out / "protocol.json").read_bytes()).hexdigest()
    rows, arrays, all_cases = [], [], []
    for sensor, seq in TEST:
        path = out / f"s1_{sensor}_{seq}.npz"
        with np.load(path) as archive:
            z = {key: archive[key] for key in archive.files if key != "meta"}
            meta = json.loads(str(archive["meta"]))
        assert meta["protocol_sha256"] == protocol_hash
        name = f"{sensor}/{seq}"
        positive = z["min_past_gt_dist"] <= 5
        cat = z["raw_category_eps_0.1"]
        n = int(positive.sum())
        assert n == meta["n_positive"] and len(cat) == meta["n_queries"]
        assert np.array_equal(cat == 4, ~positive)
        assert np.array_equal(cat == 0, positive & (z["raw_top1_geo"] <= 5))
        row = {"sequence": name, "city": CITY_OF[name], "n_queries": len(cat),
               "n_positive": n, "counts": {}, "rates": {},
               "existing_ledger_differences": meta["existing_ledger_differences"]}
        for stage in STAGES:
            correct = positive & (z[f"{stage}_top1_geo"] <= 5)
            row["counts"][f"r1_{stage}"] = int(correct.sum())
            row["rates"][f"r1_{stage}"] = ratio(int(correct.sum()), n)
            idx_key = f"{stage}_top1_idx" if stage not in OPS else f"{stage}_j_star"
            assert np.all(z[idx_key][positive] <= z["query_idx"][positive] - 30)
            if stage not in OPS:
                assert np.array_equal(z[f"{stage}_best_pos_rank"][positive] == 1, correct[positive])
            for category, label in enumerate(CATEGORIES[:4]):
                selected = cat == category
                num = int((selected & correct).sum())
                row["rates"][f"recovery_{label}_{stage}"] = ratio(num, int(selected.sum()))
        for category, label in enumerate(CATEGORIES[:4]):
            row["counts"][f"category_{label}"] = int((cat == category).sum())
            row["rates"][f"category_{label}"] = ratio(int((cat == category).sum()), n)
            if category:
                row["rates"][f"share_of_misses_{label}"] = ratio(
                    int((cat == category).sum()), int((positive & (cat != 0)).sum()))
        for eps in (.05, .1, .2):
            category_array = z[f"raw_category_eps_{eps:g}"]
            for category, label in enumerate(CATEGORIES[1:4], 1):
                row["rates"][f"epsilon_{eps:g}_{label}"] = ratio(int((category_array == category).sum()), n)
        for op in OPS:
            for field in ("key_hit", "aux_hit", "union_hit"):
                row["rates"][f"{op}_{field}"] = ratio(int((positive & z[f"{op}_{field}"]).sum()), n)
            assert np.all((z[f"{op}_top1_geo"] > 5) | z[f"{op}_union_hit"])
            query_class = classify_queries(z["min_past_gt_dist"])
            sweep = pr_sweep(z[f"{op}_c_spec"], z[f"{op}_top1_geo"], query_class)
            fixed = operating_point(z[f"{op}_c_spec"], z[f"{op}_top1_geo"], query_class,
                                    meta["confidence"][op]["tau_star"])
            for metric in ("AP", "max_F1", "R@100P"):
                row["rates"][f"{op}_{metric}"] = float(sweep[metric])
            row["rates"][f"{op}_P_at_tau"] = float(fixed["precision"])
            row["rates"][f"{op}_R_at_tau"] = float(fixed["recall"])
        rows.append(row)
        arrays.append(z)
        for i in np.flatnonzero(positive & (cat != 0)):
            record = {"sequence": name, "query_idx": int(z["query_idx"][i]),
                      "query_scan_id": int(z["query_scan_id"][i]), "category": CATEGORIES[int(cat[i])],
                      "raw_best_pos_dist": float(z["raw_best_pos_dist"][i])}
            for stage in STAGES:
                fields = ("top1_idx", "top1_geo", "best_pos_idx", "best_pos_rank") if stage not in OPS else (
                    "j_star", "top1_geo", "c_spec", "accepted", "best_pos_candidate_rank", "union_hit")
                for field in fields:
                    record[f"{stage}_{field}"] = z[f"{stage}_{field}"][i].item()
            all_cases.append(record)
    joined = {key: np.concatenate([z[key] for z in arrays]) for key in arrays[0]}
    positive = joined["min_past_gt_dist"] <= 5
    cat = joined["raw_category_eps_0.1"]
    n = int(positive.sum())
    result = {"protocol_sha256": protocol_hash, "seed": 1, "n_sequences": len(rows),
              "n_queries": len(cat), "n_positive": n, "aggregation": summarize(rows),
              "per_sequence": rows, "query_pooled": {}, "transitions": {}, "sensitivity": {},
              "existing_ledger_differences": {}, "final_matching": {}}
    for stage in STAGES:
        result["query_pooled"][f"r1_{stage}"] = float((joined[f"{stage}_top1_geo"][positive] <= 5).mean())
    for category, label in enumerate(CATEGORIES[:4]):
        chosen = cat == category
        result["query_pooled"][label] = {
            "n": int(chosen.sum()), "share_of_revisits": ratio(int(chosen.sum()), n),
            "share_of_raw_misses": (ratio(int(chosen.sum()), int((positive & (cat != 0)).sum()))
                                    if category else None),
            "correct_by_stage": {s: int((chosen & (joined[f"{s}_top1_geo"] <= 5)).sum()) for s in STAGES},
            "correct_rate_by_stage": {s: ratio(int((chosen & (joined[f"{s}_top1_geo"] <= 5)).sum()),
                                               int(chosen.sum())) for s in STAGES}}
    for before, after in zip(STAGES[:-2], STAGES[1:-1]):
        result["transitions"][f"{before}_to_{after}"] = transition_counts(
            joined[f"{before}_top1_geo"] <= 5, joined[f"{after}_top1_geo"] <= 5, positive)
    for after in OPS:
        for before in ("raw", "context"):
            result["transitions"][f"{before}_to_{after}"] = transition_counts(
                joined[f"{before}_top1_geo"] <= 5, joined[f"{after}_top1_geo"] <= 5, positive)
    for eps in (.05, .1, .2):
        c = joined[f"raw_category_eps_{eps:g}"]
        result["sensitivity"][str(eps)] = {label: int((c == i).sum()) for i, label in enumerate(CATEGORIES)}
    for op in OPS:
        correct = joined[f"{op}_top1_geo"] <= 5
        noalign = joined[f"{op}_noalign_top1_geo"] <= 5
        accepted = joined[f"{op}_accepted"]
        negative = joined["min_past_gt_dist"] > 25
        result["existing_ledger_differences"][op] = {
            key: sum(row["existing_ledger_differences"][op][key] for row in rows)
            for key in ("top1_changed", "correctness_changed")}
        result["final_matching"][op] = {
            "same_candidate_alignment_transition": transition_counts(noalign, correct, positive),
            "correct_accepted": int((positive & correct & accepted).sum()),
            "correct_rejected": int((positive & correct & ~accepted).sum()),
            "wrong_accepted_revisit": int((positive & ~correct & accepted).sum()),
            "wrong_rejected_revisit": int((positive & ~correct & ~accepted).sum()),
            "negative_queries": int(negative.sum()),
            "false_accepted_negative": int((negative & accepted).sum()),
            "missed_at_union": int((positive & ~joined[f"{op}_union_hit"]).sum()),
            "missed_at_ranking": int((positive & joined[f"{op}_union_hit"] & ~correct).sum())}
    result["limits"] = ["고정된 시드 1의 시험 사후 진단이다.",
                        "각 키의 검색 비교이며 순차 입력 변환이나 재학습 제거 실험이 아니다.",
                        "실패 분류만으로 위상 소실·옥타브 압축·센서 변화의 원인별 기여가 분리되지는 않는다.",
                        "R@1은 수락 기준 적용 전이다. 수락·거절 건수는 별도로 집계한다.",
                        "종류별 회복률은 원래 키에서 정한 고정 질의를 분모로 쓴다.",
                        "정렬 제외 진단은 같은 후보와 기존 앞의 두 가중치를 쓴 순위 비교다. AP 기여가 아니다."]
    return result, all_cases


def plot_summary(out: Path, result: dict) -> None:
    """원고용 PDF와 확인용 PNG를 만든다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.5), layout="constrained")
    macro = result["aggregation"]["city_macro"]
    colors = ["#5e7189"] * 4 + ["#137d78", "#459d96"]
    values = [100 * macro[f"r1_{s}"] for s in STAGES]
    axes[0].bar(np.arange(6), values, color=colors)
    axes[0].set(xticks=np.arange(6), xticklabels=LABELS, ylim=(0, 106),
                ylabel="Recall@1 (%)", title="(a) Retrieval comparisons | 6-city mean")
    axes[0].tick_params(axis="x", labelrotation=45)
    for x, y in enumerate(values):
        axes[0].text(x, y + 1.4, f"{y:.1f}", ha="center", fontsize=8)
    cats = ("spread", "collision", "near_field")
    names = ("Spread", "Far collision", "Near-field")
    cat_colors = ("#d9a443", "#ba5c5c", "#837da8")
    for i, (c, label, color) in enumerate(zip(cats, names, cat_colors)):
        v = 100 * macro[f"category_{c}"]
        axes[1].barh(i, v, color=color)
        axes[1].text(v + .3, i, f"{v:.1f}%", va="center")
    axes[1].set(yticks=np.arange(3), yticklabels=names, xlabel="Share of all revisit queries (%)",
                title="(b) Raw-key errors | 6-city mean", xlim=(0, max(100 * macro[f"category_{c}"] for c in cats) * 1.4))
    axes[1].invert_yaxis()
    data = np.array([[100 * result["query_pooled"][c]["correct_rate_by_stage"][s]
                      for s in STAGES[1:]] for c in cats])
    axes[2].imshow(data, vmin=0, vmax=100, cmap="GnBu", aspect="auto")
    axes[2].set(xticks=np.arange(5), xticklabels=LABELS[1:], yticks=np.arange(3),
                yticklabels=[f"{name}\nn={result['query_pooled'][c]['n']:,}" for c, name in zip(cats, names)],
                title="(c) Correct after retrieval (%) | pooled")
    axes[2].tick_params(axis="x", labelrotation=45)
    for i in range(3):
        for j in range(5):
            axes[2].text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                         color="white" if data[i, j] > 60 else "black", fontsize=8)
    for ax in axes[:2]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Seed 1 | 46 sequences | raw-key epsilon = 0.1 | true match <= 5 m | far match >= 25 m | index gap >= 30",
                  fontsize=8)
    fig.savefig(out / "collision_recovery.pdf", bbox_inches="tight")
    fig.savefig(out / "collision_recovery.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def select_cases(cases: list[dict]) -> dict:
    """각 종류에서 사례가 가장 많은 시퀀스의 중앙 순위 질의를 고른다."""
    selected = {}
    for category in ("collision", "spread"):
        for recovered in (True, False):
            pool = [c for c in cases if c["category"] == category
                    and (c["pbev128_main_top1_geo"] <= 5) == recovered]
            if not pool:
                continue
            counts = {s: sum(c["sequence"] == s for c in pool) for s in sorted({c["sequence"] for c in pool})}
            sequence = sorted(counts, key=lambda s: (-counts[s], s))[0]
            subset = sorted([c for c in pool if c["sequence"] == sequence],
                            key=lambda c: (c["raw_best_pos_rank"], c["query_idx"]))
            key = f"{category}_{'recovered' if recovered else 'unresolved'}"
            selected[key] = {"n_pool": len(pool), "n_sequence_pool": len(subset),
                             "selected": subset[len(subset) // 2]}
    return {"selection_rule": "종류×NSD top1 회복 여부마다 건수가 가장 많은 시퀀스를 고른다. "
                              "그 시퀀스에서 (raw 정답 순위, 질의 번호)로 정렬한 중앙 질의를 택한다. "
                              "동률 시퀀스는 이름순이다. 그림 품질로 고르지 않는다.", "cases": selected}


def main() -> None:
    """완료된 고정 측정만 보고서로 만든다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "results/pdv2_collision_20261013")
    args = parser.parse_args()
    result, cases = collect(args.out)
    existing = json.loads((REPO / "results/pdv2_c3_bevclip_s1/odom/pr_summary_align.json").read_text())
    macro = result["aggregation"]["city_macro"]
    result["main_metric_replay"] = {
        op: {metric: {"current": macro[f"{op}_{metric}"],
                      "existing": existing[op]["city_macro6"][metric],
                      "difference": macro[f"{op}_{metric}"] - existing[op]["city_macro6"][metric]}
             for metric in ("AP", "max_F1", "R@100P", "P_at_tau", "R_at_tau")}
        for op in OPS}
    result["report_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(args.out / "summary.json", result)
    write_json(args.out / "selected_cases.json", select_cases(cases))
    with (args.out / "all_error_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    with (args.out / "per_sequence.csv").open("w", newline="") as handle:
        fields = ["sequence", "city", "n_queries", "n_positive"] + list(result["per_sequence"][0]["rates"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**{k: r[k] for k in fields[:4]}, **r["rates"]} for r in result["per_sequence"])
    plot_summary(args.out, result)
    print(json.dumps({k: result[k] for k in ("n_sequences", "n_queries", "n_positive",
                                           "query_pooled", "existing_ledger_differences")}, indent=2))


if __name__ == "__main__":
    main()
