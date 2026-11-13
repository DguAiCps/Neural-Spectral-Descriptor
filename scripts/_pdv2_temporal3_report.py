#!/usr/bin/env python3
"""완료된 3시드 대조의 방향별 정답 변화와 그림을 만든다(호스트 실행)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_temporal3_20261017"


def digest(path: Path) -> str:
    """파일 해시를 계산한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """학습·선택 변경 없이 저장 원장을 짝지어 분석한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    summary = json.loads((OUT / "summary.json").read_text())
    audit = json.loads((OUT / "completion_validation.json").read_text())
    assert audit["status"] == "passed"
    protocol = json.loads((OUT / "protocol.json").read_text())
    manifest = {str(p.relative_to(REPO)): digest(p) for p in
                (Path(__file__), OUT / "summary.json", OUT / "completion_validation.json")}
    seed_checks = {}
    graphs = []
    for arm in ("temporal", "point"):
        initial, orders = [], []
        for seed in (1, 2, 3):
            folder = OUT / f"s{seed}/{arm}"
            for name in ("initial_model.json", "batch_order.json", "training_graph.json"):
                p = folder / name
                manifest[str(p.relative_to(REPO))] = digest(p)
            initial.append(digest(folder / "initial_model.json"))
            orders.append(digest(folder / "batch_order.json"))
            graphs.append(json.loads((folder / "training_graph.json").read_text()))
        assert len(set(initial)) == len(set(orders)) == 3
        seed_checks[arm] = {"distinct_initializations": 3, "distinct_batch_orders": 3}
    assert all(g == graphs[0] for g in graphs)
    transitions, datasets, fixed = {}, {}, {}
    ops = list(summary["mean_std"])
    for seed, selected in summary["per_seed"].items():
        transitions[seed], datasets[seed], fixed[seed] = {}, {}, {}
        for op in ops:
            transitions[seed][op] = {}
            datasets[seed][op] = {
                family: {m: selected["temporal"]["ops"][op]["by_group"][family][m]
                         - selected["point"]["ops"][op]["by_group"][family][m]
                         for m in ("AP", "max_F1", "R@100P", "R@1")}
                for family in selected["temporal"]["ops"][op]["by_group"]}
            fixed[seed][op] = {
                epoch: {m: cells[seed]["temporal"][op]["macro"][m]
                        - cells[seed]["point"][op]["macro"][m]
                        for m in ("AP", "max_F1", "R@100P", "R@1")}
                for epoch, cells in summary["fixed_epoch"].items()}
            for seq in protocol["val_roster"]:
                loaded = {}
                for arm in ("temporal", "point"):
                    p = OUT / f"s{seed}/{arm}/epoch{selected[arm]['epoch']}/val_{op}_{seq.replace('/', '_')}.npz"
                    manifest[str(p.relative_to(REPO))] = digest(p)
                    with np.load(p) as z:
                        loaded[arm] = {k: z[k] for k in ("query_idx", "min_past_gt_dist", "top1_geo_dist", "group", "union_hit")}
                a, b = loaded["point"], loaded["temporal"]
                np.testing.assert_array_equal(a["query_idx"], b["query_idx"])
                pos = a["min_past_gt_dist"] <= 5.
                old, new = a["top1_geo_dist"] <= 5., b["top1_geo_dist"] <= 5.
                groups = {}
                for index, label in enumerate(("same_available", "opposite_only", "other")):
                    mask = pos & (a["group"] == index)
                    groups[label] = {"n": int(mask.sum()), "point_correct": int((mask & old).sum()),
                                     "temporal_correct": int((mask & new).sum()),
                                     "gained": int((mask & ~old & new).sum()),
                                     "lost": int((mask & old & ~new).sum()),
                                     "lost_missing_positive_in_pool": int((mask & old & ~new & ~b["union_hit"].astype(bool)).sum())}
                    assert groups[label]["temporal_correct"] - groups[label]["point_correct"] == groups[label]["gained"] - groups[label]["lost"]
                transitions[seed][op][seq] = groups
    for name, value in (("selected_transitions.json", transitions),
                        ("dataset_deltas.json", datasets), ("fixed_epoch_deltas.json", fixed)):
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    font = FontProperties(fname="/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    for ax, metric, title in zip(axes, ("AP", "R@100P"), ("최종 AP", "정밀도 100%에서 재현율")):
        for i, op in enumerate(ops):
            for j, (arm, label, color) in enumerate((("point", "이웃 없는 투영", "#777777"), ("temporal", "시간 문맥", "#1967A3"))):
                r = summary["mean_std"][op][arm][metric]
                x = i + (j - .5) * .24
                vals = np.asarray(r["values"]) * 100
                ax.errorbar(x, r["mean"] * 100, yerr=r["std"] * 100,
                            fmt="o", color=color, capsize=4, label=label if i == 0 else None)
                ax.scatter(x + np.array([-.035, 0, .035]), vals, color=color, s=15, alpha=.6)
            a = np.array(summary["mean_std"][op]["point"][metric]["values"]) * 100
            b = np.array(summary["mean_std"][op]["temporal"][metric]["values"]) * 100
            for av, bv in zip(a, b):
                ax.plot([i - .12, i + .12], [av, bv], color="#AAAAAA", alpha=.5, linewidth=.8)
        ax.set_xticks([0, 1], ["NSD", "NSD-H"])
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.grid(axis="y", alpha=.2)
        ax.margins(x=.25, y=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("검증 9시퀀스 · 3시드 평균과 표본 표준편차 · 동일 최종 판정 규칙")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"comparison.{suffix}", dpi=180)
    plt.close(fig)
    (OUT / "report_validation.json").write_text(json.dumps({
        "status": "passed", "paired_transition_accounting": True,
        "seed_checks": seed_checks, "same_training_graph_across_all_runs": True,
        "inputs_sha256": manifest, "test_used": False,
        "artifacts_sha256": {p.name: digest(p) for p in (OUT / "selected_transitions.json",
            OUT / "dataset_deltas.json", OUT / "fixed_epoch_deltas.json", OUT / "comparison.png", OUT / "comparison.pdf")}},
        ensure_ascii=False, indent=2) + "\n")
    print("REPORT_PASSED", len(manifest))


if __name__ == "__main__":
    main()
