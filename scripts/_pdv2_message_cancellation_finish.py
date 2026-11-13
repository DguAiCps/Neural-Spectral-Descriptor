#!/usr/bin/env python3
"""메시지 상쇄 진단 검산과 고정 최종 수락 오류의 보조 분석."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_message_cancellation as run
from _pdv2_no_context_ablation import sha256, write_json


def matched(values: dict, lost: np.ndarray, kept: np.ndarray, layer: int) -> dict:
    """행별 구간 사전으로 같은 조건의 두 집단 평균 차이를 구한다."""
    buckets = defaultdict(lambda: [[], []])
    for index in np.flatnonzero(lost | kept):
        key = (int(values["group"][index]), bool(values["turning"][index]),
               int(values[f"l{layer}_mass_bin"][index]))
        buckets[key][0 if lost[index] else 1].append(index)
    result = []
    for key, (a, b) in sorted(buckets.items()):
        if min(len(a), len(b)) < 3:
            continue
        result.append({"stratum": list(key), "lost": len(a), "kept": len(b), "weight": min(len(a), len(b)),
            **{name: float(values[f"l{layer}_{name}"][a].astype(float).mean() - values[f"l{layer}_{name}"][b].astype(float).mean())
               for name in ("cancellation", "between", "within")},
            "candidate_between": float(values[f"j_l{layer}_between"][a].astype(float).mean()
                                        - values[f"j_l{layer}_between"][b].astype(float).mean())})
    return {"raw_lost": int(lost.sum()), "raw_kept": int(kept.sum()), "strata": result}


def summarize(cells: list) -> dict:
    """구간 표본 수로 가중한 평균 차이와 계열별 범위를 기록한다."""
    fields = ("between", "within", "cancellation", "candidate_between")
    def weighted(selected):
        return {f: float(np.average([x[f] for x in selected], weights=[x["weight"] for x in selected]))
                if selected else None for f in fields}
    return {"strata": cells, "matched_lost": sum(x["lost"] for x in cells),
            "weighted_difference": weighted(cells),
            "families": {f: {"matched_lost": sum(x["lost"] for x in cells if x["sequence"].startswith(f)),
                             **weighted([x for x in cells if x["sequence"].startswith(f)])}
                         for f in ("kitti", "helipr", "mulran")}}


def main() -> None:
    """모든 저장 행의 합산식과 고정 원장을 검산하고 결과를 보고한다."""
    out = run.OUT
    report = run.previous.read(out / "summary.json")
    spec = run.previous.read(out / "protocol.json")
    assert sha256(out / "protocol.json") == report["protocol_sha256"]
    sources = dict(report["sources"])
    for p, h in sources.items():
        assert sha256(REPO / p) == h, p
    metric_rows = 0
    identity_error = 0.
    records, training = {}, []
    for arm in run.ARMS:
        train = run.run.exp.arrays(out / f"train_{arm}.npz")
        for layer in (1, 2):
            means = {field: [] for field in ("cancellation", "within", "between")}
            for seq in np.unique(train["sequence"]):
                mask = train["sequence"] == seq
                expected = report["train"][arm][str(layer)][str(seq)]
                for field in means:
                    x = train[f"l{layer}_{field}"][mask].astype(float)
                    np.testing.assert_allclose(x.mean(), expected[field]["mean"], atol=1e-12)
                    means[field].append(float(x.mean()))
            training.append({"arm": arm, "layer": layer, **{k: 100*np.mean(v) for k, v in means.items()}})
        records[arm] = {}
        for p in [out / f"train_{arm}.npz", *[out / (arm + "_" + seq.replace("/", "_") + ".npz")
                  for seq in run.previous.read(run.previous.OUT / "protocol.json")["val"]]]:
            a = run.run.exp.arrays(p)
            for layer in (1, 2):
                for prefix in ("", "j_") if "query_idx" in a else ("",):
                    base = f"{prefix}l{layer}_"
                    mass, surviving = a[base+"mass"].astype(float), a[base+"surviving"].astype(float)
                    c, w, b = (a[base+k].astype(float) for k in ("cancellation", "within", "between"))
                    expected = np.clip(np.where(mass > 1e-12, 1-surviving/np.maximum(mass, 1e-12), 0), 0, 1)
                    error = max(float(abs(c-expected).max()), float(abs(c-w-b).max()))
                    assert error < 1e-6
                    identity_error = max(identity_error, error)
                    for key in run.FIELDS:
                        assert np.isfinite(a[base+key]).all()
                    metric_rows += len(mass)
            if "query_idx" not in a:
                continue
            sequence = p.stem[len(arm)+1:]
            records[arm][sequence] = a
    old_summary = run.previous.read(run.previous.OUT / "summary.json")
    source_metrics = {}
    for arm in run.ARMS:
        epoch = old_summary["per_seed"]["1"][arm]["selected"]["epoch"]
        source_metrics[arm] = old_summary["per_seed"]["1"][arm]["selected"][run.run.METHOD]["macro"]
        for sequence, a in records[arm].items():
            original = run.run.exp.arrays(run.run.root(1, arm) / f"epoch{epoch}" / (sequence + ".npz"))
            for key, value in original.items():
                np.testing.assert_array_equal(a[key], value)
    associations, acceptance_counts = {}, {}
    tau = spec["fixed_tau"]
    for arm, reference in (("diff", "standard"), ("diff_tanh", "standard_tanh")):
        associations[arm], acceptance_counts[arm] = {}, {}
        for layer in (1, 2):
            events = defaultdict(list)
            for sequence, a in records[arm].items():
                b = records[reference][sequence]
                pos, neg = a["min_past_gt_dist"] <= 5, a["min_past_gt_dist"] > 25
                ca, cb = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
                aa, ab = a[run.run.METHOD] >= tau, b[run.run.METHOD] >= tau
                masks = {
                    "retrieval_loss": (pos & cb & ~ca, pos & cb & ca),
                    "true_accept_loss": (pos & cb & ab & ~(ca & aa), pos & cb & ab & ca & aa),
                    "new_negative_false_accept": (neg & ~ab & aa, neg & ~ab & ~aa)}
                for name, (lost, kept) in masks.items():
                    r = matched(a, lost, kept, layer)
                    for row in r["strata"]:
                        row["sequence"] = sequence.replace("_", "/", 1)
                        events[name].append(row)
                    if layer == 2:
                        acceptance_counts[arm].setdefault(name, {})[sequence] = {k: r[k] for k in ("raw_lost", "raw_kept")}
            associations[arm][str(layer)] = {name: summarize(cells) for name, cells in events.items()}
            reproduced = associations[arm][str(layer)]["retrieval_loss"]
            original = report["associations"][arm][str(layer)]
            assert reproduced["matched_lost"] == original["matched_lost"]
            np.testing.assert_allclose(reproduced["weighted_difference"]["between"], original["weighted_between_difference"], atol=1e-7)
    primary = associations["diff_tanh"]["2"]["retrieval_loss"]
    support = sum(v["between"]["quantiles"][2] >= .1 for v in report["train"]["diff_tanh"]["2"].values())
    checks = {"train_cross_cancellation": support >= 2, "matched_loss_count": primary["matched_lost"] >= 20,
        "matched_between_difference": primary["weighted_difference"]["between"] >= .02,
        "two_families_positive": sum(v["between"] is not None and v["between"] > 0 for v in primary["families"].values()) >= 2}
    assert checks == report["followup_checks"] and all(checks.values()) == report["followup_eligible"]
    write_json(out / "acceptance_report.json", {"fixed_tau": tau, "associations": associations,
        "counts": acceptance_counts, "source_final_metrics": source_metrics,
        "scope": "검색 손실의 주판정을 그대로 재검산. 참수락 누락과 새 비재방문 오수락의 층화는 주결과를 본 뒤 추가한 보조 진단이며 진입 기준을 바꾸지 않음.",
        "test_used": False})
    with (out / "training_cancellation.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training[0]))
        writer.writeheader()
        writer.writerows(training)
    plot(training, associations)
    for p in (Path(__file__), out / "protocol.json", out / "summary.json", out / "acceptance_report.json",
              out / "training_cancellation.csv", out / "message_cancellation.png", out / "message_cancellation.pdf"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    write_json(out / "completion_validation.json", {"status": "passed", "sources": sources,
        "bundles": 40, "statistic_rows": metric_rows, "identity_max_error": identity_error,
        "fixed_ledger_arrays_exact": True, "strata_and_followup_gate_recomputed": True,
        "output_reconstruction_max_error": report["max_output_reconstruction_error"],
        "stored_key_max_error": report["max_saved_key_error"], "prefix_checks": report["prefix_checks"],
        "prefix_cancellation_max_error": report["max_prefix_cancellation_error"], "test_used": False})
    print("VERIFIED", metric_rows, "FOLLOWUP", report["followup_eligible"], flush=True)
    for arm, layers in associations.items():
        for event, value in layers["2"].items():
            print("FINAL_ERROR_ASSOCIATION", arm, event, value["matched_lost"], value["weighted_difference"], flush=True)


def plot(training: list, associations: dict) -> None:
    """상쇄가 존재한다는 사실과 오류 증가 관계를 분리해 그린다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams["axes.unicode_minus"] = False
    font = FontProperties(fname=str(REPO / "results/pdv2_sequence_report_20261018/UnDotum.ttf"))
    chosen = [x for x in training if x["layer"] == 2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    within = [x["within"] for x in chosen]
    between = [x["between"] for x in chosen]
    axes[0].bar(range(4), within, color="#92a8bc", label="방향 내부")
    bars = axes[0].bar(range(4), between, bottom=within, color="#bd7745", label="과거/미래 사이")
    axes[0].bar_label(bars, labels=[f'{x["cancellation"]:.2f}' for x in chosen], padding=3)
    axes[0].set_xticks(range(4), ["차이", "차이+tanh", "표준", "표준+tanh"], fontproperties=font)
    axes[0].set_title("학습 8시퀀스 균등 평균 L1 상쇄 (%) · 마지막 층", fontproperties=font)
    axes[0].set_ylim(0, 35)
    axes[0].legend(prop=font)
    primary = associations["diff_tanh"]["2"]["retrieval_loss"]["families"]
    values = [100*primary[f]["between"] for f in ("kitti", "helipr", "mulran")]
    labels = [f'{label}\n(n={primary[family]["matched_lost"]})'
              for label, family in (("KITTI", "kitti"), ("HeLiPR", "helipr"), ("MulRan", "mulran"))]
    bars = axes[1].bar(labels, values,
                       color=["#bd7745" if x > 0 else "#4e8e84" for x in values])
    axes[1].bar_label(bars, fmt="%+.3f", padding=3)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_ylim(min(values)-1., max(values)+1.)
    axes[1].set_title("차이+tanh: 잃은 정답 - 유지 정답\n층화한 과거/미래 상쇄 차이 (%p)", fontproperties=font)
    fig.suptitle("검증 오답과의 관계 진단 · 새 성능 개선 결과 아님", fontproperties=font)
    fig.tight_layout()
    fig.savefig(run.OUT / "message_cancellation.png", dpi=180)
    fig.savefig(run.OUT / "message_cancellation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
