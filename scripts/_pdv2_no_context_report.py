#!/usr/bin/env python3
"""고정 문맥 제거 결과의 검증셋 집계·CSV·논문용 그림을 만든다."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from _pdv2_no_context_ablation import (
    OPS, OUT, REPO, SEEDS, VAL, ledger_path, metrics, sha256, write_json,
)


def validation_summary() -> dict:
    """설정 선택 없이 검증 9시퀀스의 계열 평균을 확인한다."""
    protocol = json.loads((OUT / "protocol.json").read_text())
    result = {}
    for op in OPS:
        result[op] = {}
        for seed in SEEDS:
            per = {}
            tau = protocol["selection_by_seed"][str(seed)][op]["tau_star"]
            for sensor, seq in VAL:
                path = OUT / f"val_{op}_s{seed}_{sensor}_{seq}.npz"
                with np.load(path) as archive:
                    new = {k: archive[k] for k in archive.files if k != "meta"}
                source = ledger_path("val", seed, op, sensor, seq)
                with np.load(source) as old, np.load(source.with_name("align_" + source.name)) as align:
                    baseline = {k: old[k] for k in ("min_past_gt_dist", "top1_geo_dist")}
                    np.testing.assert_array_equal(old["j_star"], align["j_star"])
                    baseline["c_spec"] = align["c_spec"]
                    np.testing.assert_array_equal(old["query_idx"], new["query_idx"])
                    np.testing.assert_array_equal(old["min_past_gt_dist"], new["min_past_gt_dist"])
                before, after = metrics(baseline, tau), metrics(new, tau)
                per[sensor + "/" + seq] = {"with_context": before, "without_context": after,
                                           "delta": {m: after[m] - before[m] for m in before}}
            groups = {f: [s + "/" + q for s, q in VAL if ("kitti" if s == "kitti360" else s) == f]
                      for f in ("kitti", "helipr", "mulran")}
            family = {f: {c: {m: float(np.mean([per[name][c][m] for name in names]))
                              for m in per[names[0]][c]}
                           for c in per[names[0]]}
                      for f, names in groups.items()}
            macro = {c: {m: float(np.mean([v[c][m] for v in family.values()]))
                         for m in next(iter(family.values()))[c]}
                     for c in next(iter(family.values()))}
            result[op][str(seed)] = {"family_macro3": macro, "family_groups": family,
                                    "per_sequence": per}
    write_json(OUT / "validation_summary.json", result)
    return result


def audit() -> dict:
    """완료 표시·질의 기록·입력 해시를 전체 확인한다."""
    protocol_path = OUT / "protocol.json"
    spec = json.loads(protocol_path.read_text())
    protocol_hash = sha256(protocol_path)
    for name, expected in spec["sha256"].items():
        assert sha256(REPO / name) == expected, name
    checked_inputs, block_errors = {}, []
    n_records = 0
    total_queries = {"val": 0, "test": 0}
    for role, roster in spec["roster"].items():
        for sequence in roster:
            tag = sequence.replace("/", "_")
            done = json.loads((OUT / f"done_{role}_{tag}.json").read_text())
            assert done["protocol_sha256"] == protocol_hash
            for seed in SEEDS:
                for op in OPS:
                    with np.load(OUT / f"{role}_{op}_s{seed}_{tag}.npz") as z:
                        meta = json.loads(str(z["meta"]))
                        assert meta["protocol_sha256"] == protocol_hash
                        q = z["query_idx"]
                        np.testing.assert_array_equal(q, np.arange(30, meta["n_frames"]))
                        assert np.all(z["j_star"] <= q - 30)
                        assert np.all((z["n_candidates"] > 0) & (z["n_candidates"] <= 200))
                        assert np.all(z["union_hit"] == (z["key_hit"] | z["aux_hit"]))
                        for name in z.files:
                            if name != "meta":
                                assert z[name].shape == q.shape and np.isfinite(z[name]).all(), name
                        if seed == 1 and op == OPS[0]:
                            total_queries[role] += len(q)
                        if meta["key_block_max_abs_difference"] is not None:
                            block_errors.append(meta["key_block_max_abs_difference"])
                    sources = {**meta["inputs_sha256"], meta["source"]: meta["source_sha256"]}
                    for name, expected in sources.items():
                        if name not in checked_inputs:
                            checked_inputs[name] = sha256(REPO / name)
                        assert checked_inputs[name] == expected, name
                    n_records += 1
    result = {"n_records": n_records, "n_distinct_inputs": len(checked_inputs),
              "n_queries_per_setting": total_queries,
              "key_block_max_abs_difference": max(block_errors),
              "protocol_sha256": protocol_hash,
              "checks": "분모, 과거 제약, 후보 상한, 유한성, 모양, 후보 합집합, 원천 해시 일치"}
    write_json(OUT / "audit.json", result)
    return result


def tables(result: dict) -> None:
    """도시·시퀀스별 짝지은 값을 CSV로 쓴다."""
    for kind in ("city_groups", "per_sequence"):
        rows = []
        for op in OPS:
            for seed, run in result["ops"][op]["by_seed_setting"].items():
                for group, values in run[kind].items():
                    for metric in values["with_context"]:
                        rows.append({"op": op, "seed_setting": seed, "group": group,
                                     "metric": metric, **{c: values[c][metric] for c in (
                                         "with_context", "without_context", "delta")}})
        with (OUT / f"{kind}.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def figure(result: dict) -> None:
    """전체 점수와 도시별 제거 효과를 원고용 PDF와 PNG로 그린다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.8), layout="constrained")
    colors = ("#197f7b", "#d99447")
    names = ("NSD AP", "NSD R@1", "NSD-H AP", "NSD-H R@1")
    for ci, condition in enumerate(("with_context", "without_context")):
        stats = [result["ops"][op]["across_three_settings"][condition][m]
                 for op in OPS for m in ("AP", "R@1_revisit")]
        x = np.arange(4) + (ci - .5) * .36
        y = [100 * s["mean"] for s in stats]
        axes[0].bar(x, y, .36, yerr=[100 * s["std"] for s in stats],
                    color=colors[ci], label=("With context", "Without context")[ci], capsize=2)
        for xx, yy in zip(x, y):
            axes[0].text(xx, yy + 1, f"{yy:.1f}", ha="center", fontsize=7)
    axes[0].set(xticks=np.arange(4), xticklabels=names, ylabel="Score (%)",
                ylim=(0, 105), title="(a) Full pipeline | six-city mean")
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[0].legend(fontsize=8, loc="lower left")
    city_order = ("karlsruhe", "seoul", "daejeon", "ann_arbor", "stockholm", "hamburg")
    for ax, metric, title in zip(axes[1:], ("AP", "R@1_revisit"),
                                  ("(b) AP change after removal", "(c) R@1 change after removal")):
        for oi, op in enumerate(OPS):
            rows = result["ops"][op]["by_seed_setting"]
            delta = np.array([[100 * rows[str(s)]["city_groups"][city]["delta"][metric]
                               for s in SEEDS] for city in city_order])
            x = np.arange(6) + (oi - .5) * .36
            ax.bar(x, delta.mean(axis=1), .36, yerr=delta.std(axis=1, ddof=1),
                   color=colors[oi], label=("NSD", "NSD-H")[oi], capsize=2)
        ax.axhline(0, color="#555555", lw=.7)
        ax.set(xticks=np.arange(6), xticklabels=[x.replace("_", " ").title() for x in city_order],
               ylabel="Without minus with (percentage points)", title=title)
        ax.tick_params(axis="x", labelrotation=45)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("46 test sequences | original settings paired across three seeds | no retuning | error bars: sample SD across settings",
                  fontsize=8)
    fig.savefig(OUT / "context_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "context_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """완료된 고정 실험 결과만 읽어 보고서를 만든다."""
    result = json.loads((OUT / "summary.json").read_text())
    audit()
    validation_summary()
    tables(result)
    figure(result)
    write_json(OUT / "report_provenance.json", {
        "script_sha256": sha256(Path(__file__)), "summary_sha256": sha256(OUT / "summary.json"),
        "protocol_sha256": sha256(OUT / "protocol.json"),
        "note": "제거 대조군의 세 설정 차이는 기존 융합·임계값 차이이며 세 번의 재학습 변동이 아니다."})
    print("REPORT COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    main()
