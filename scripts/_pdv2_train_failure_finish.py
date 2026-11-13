#!/usr/bin/env python3
"""학습 오수락의 추가 파라미터·후보 개입 기록을 재검산하고 그림을 만든다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
from _pdv2_train_failure import OUT, arrays, read
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """원천·전체 파생 원장의 건수와 유한성·과거 범위를 검증한다."""
    sources = {}
    for filename in ("protocol.json", "parameter_protocol.json", "parameter_fine_protocol.json", "intervention_protocol.json"):
        for path, expected in read(OUT/filename)["sources"].items():
            assert sha256(REPO/path) == expected, path
            sources[path] = expected
    assert read(OUT/"completion_validation.json")["status"] == "passed"
    s, p, c = [read(OUT/name) for name in ("summary.json", "parameter_fine_summary.json", "intervention_summary.json")]
    a, response = arrays(OUT/"exposure_arrays.npz"), arrays(OUT/"parameter_fine_response.npz")
    assert p["response_sha256"] == sha256(OUT/"parameter_fine_response.npz")
    q = response["query"]
    for metric in ("response_eps0001", "response_eps00005", "error_estimate"):
        assert np.isfinite(response[metric]).all()
    assert np.all(response["winner"] <= q-30)
    assert not (response["stable_decrease"] & response["stable_increase"]).any()
    for group in ("all", "kitti", "helipr"):
        group_mask = np.ones(len(q), bool) if group == "all" else (a["sequence"][q] < 4 if group == "kitti" else a["sequence"][q] >= 4)
        for name, cid in (("false_accept", 1), ("negative_reject", 0), ("positive_correct_accept", 2)):
            mask = group_mask & (a["cohort"][q] == cid)
            row = p["groups"][group][name]
            assert row["n"] == mask.sum()
            assert row["stable_decrease"] == (mask & response["stable_decrease"]).sum()
            assert row["stable_increase"] == (mask & response["stable_increase"]).sum()
            assert row["n"] == row["stable_decrease"]+row["stable_increase"]+row["uncertain"]
    conditions = ("winner_key_zero", "place_key_zero")
    tau = read(OUT/"protocol.json")["selected"]["new_tau"]["tau"]
    for name, record in c["sequences"].items():
        if not record["false_accept"]:
            continue
        path = OUT/("intervention_"+name.replace("/", "_")+".npz")
        assert sha256(path) == record["sha256"]
        x = arrays(path)
        assert len(x["query"]) == record["false_accept"]
        assert np.all(x["original_score"] >= tau)
        assert x["winner_in_aux_pool"].sum() == record["winner_in_aux_pool"]
        for condition in conditions:
            scores = x[condition+"_score"]
            winner = x[condition+"_winner"]
            assert np.isfinite(scores).all() and np.all(winner <= x["query"]-30)
            assert record[condition]["winner_changed"] == (winner != x["winner"]).sum()
            assert record[condition]["now_rejected"] == (scores < tau).sum()
            assert record[condition]["still_false_accept"] == (scores >= tau).sum()
    for group in ("all", "kitti", "helipr"):
        subset = [v for k, v in c["sequences"].items() if v["false_accept"] and (group == "all" or k.startswith(group))]
        row = c["groups"][group]
        assert row["false_accept"] == sum(r["false_accept"] for r in subset)
        assert row["winner_in_aux_pool"] == sum(r["winner_in_aux_pool"] for r in subset)
        for condition in conditions:
            for field in ("winner_changed", "now_rejected", "still_false_accept"):
                assert row[condition][field] == sum(r[condition][field] for r in subset)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    n = s["cohorts"]["all"]["false_accept"]["n"]
    exposure = s["cohorts"]["all"]["false_accept"]
    categories = ["Anchor", "Exact explicit", "Exact any source", "Nearby place"]
    values = [100*exposure[k]/n for k in ("anchor_seen", "explicit_exact_seen", "exact_any_forward_seen", "nearby_forward_seen")]
    axes[0].bar(range(4), values, color="#228833")
    axes[0].set_xticks(range(4), categories, rotation=35, ha="right")
    axes[0].set_title("Exposure in training batches")
    parameter = p["groups"]["all"]["false_accept"]
    values2 = [100*parameter[k]/n for k in ("stable_decrease", "stable_increase", "uncertain")]
    axes[1].bar(range(3), values2, color=["#228833", "#CC6677", "#BBBBBB"])
    axes[1].set_xticks(range(3), ["Decrease", "Increase", "Uncertain"], rotation=35, ha="right")
    axes[1].set_title("Wrong-pair key similarity\nLocal parameter gradient")
    values3 = [100*c["groups"]["all"][condition]["now_rejected"]/n for condition in conditions]
    axes[2].bar(range(2), values3, color="#4477AA")
    axes[2].set_xticks(range(2), ["Winner key = 0", "Nearby keys = 0"], rotation=35, ha="right")
    axes[2].set_title("False accepts now rejected\nFixed-pool score intervention")
    for ax, values in zip(axes, (values, values2, values3)):
        ax.set_ylim(0, 105)
        ax.set_ylabel("False-accept queries (%)")
        ax.spines[["top", "right"]].set_visible(False)
        for i, value in enumerate(values):
            ax.text(i, value+2, f"{value:.1f}", ha="center", fontsize=9)
    fig.suptitle("NSD training diagnosis: 5,482 false accepts, seed 1 / epoch 10")
    for ext in ("png", "pdf"):
        fig.savefig(OUT/f"training_bottleneck.{ext}", dpi=180)
    plt.close(fig)
    write_json(OUT/"report_validation.json", {"status": "passed", "source_files": len(sources),
        "false_accept_queries": n, "parameter_queries": len(q), "intervention_conditions": list(conditions),
        "initial_large_step_probe": "failed finite-difference check; not used for conclusions; original script/protocol/log preserved",
        "small_step_relative_errors": [r["relative_error"] for r in p["linear_checks"]],
        "test_used": False, "script_sha256": sha256(Path(__file__)),
        "reports_sha256": {name: sha256(OUT/name) for name in ("summary.json", "parameter_fine_summary.json", "intervention_summary.json")}})
    print("VERIFIED_REPORT", n, len(q), len(sources), flush=True)


if __name__ == "__main__":
    main()
