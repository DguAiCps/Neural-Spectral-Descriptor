#!/usr/bin/env python3
"""학습 오수락 원장과 배치 노출의 독립 표본 재계산 및 비교 그림이다."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_no_context_ablation import sha256, write_json

OUT = REPO / "results/pdv2_train_failure_20261018"


def arrays(path: Path) -> dict:
    """NPZ를 읽는다."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def read(path: Path) -> dict:
    """JSON을 읽는다."""
    return json.loads(path.read_text())


def main() -> None:
    """기록된 전체 결과와 256개 이하의 배치 노출을 별도 알고리즘으로 검사한다."""
    assert read(OUT / "status.json")["status"] == "complete"
    spec, summary = read(OUT/"protocol.json"), read(OUT/"summary.json")
    for rel, expected in spec["sources"].items():
        assert sha256(REPO/rel) == expected, rel
    a = arrays(OUT/"exposure_arrays.npz")
    metadata = arrays(OUT/"training_metadata.npz")
    positions, seqids = metadata["positions"], metadata["sequences"]
    tau = spec["selected"]["new_tau"]["tau"]
    selections, rows = [], []
    rng = np.random.default_rng(20261018)
    from _pr_eval_lib import classify_queries, pr_sweep
    for ordinal, sequence in enumerate(spec["train_roster"]):
        source = OUT/("ledger_"+sequence.replace("/", "_")+".npz")
        ledger = arrays(source)
        record = summary["replay"][sequence]
        assert sha256(source) == record["ledger_sha256"]
        ids = np.flatnonzero(seqids == ordinal)
        np.testing.assert_array_equal(ledger["query_idx"], np.arange(30, len(ids)-15))
        np.testing.assert_array_equal(ledger["global_query"], ids[ledger["query_idx"]])
        q, j = ledger["global_query"], ledger["global_winner"]
        assert np.all(j <= q-30) and np.all(seqids[q] == seqids[j])
        np.testing.assert_array_equal(ledger["top1_geo_dist"], np.linalg.norm(positions[q]-positions[j], axis=1))
        qc = classify_queries(ledger["min_past_gt_dist"])
        curve = pr_sweep(ledger["arc_unique_b100"], ledger["top1_geo_dist"], qc)
        for key in ("AP", "max_F1", "R@100P"):
            assert curve[key] == record["metrics"][key]
        negative = q[qc["negative"]]
        picked = rng.choice(negative, min(16, len(negative)), replace=False)
        false = q[qc["negative"] & (ledger["arc_unique_b100"] >= tau)]
        picked = np.unique(np.r_[picked, rng.choice(false, min(16, len(false)), replace=False)])
        selections.extend(picked)
        for query in picked:
            valid = ids[ids <= query-30]
            expected = np.linalg.norm(positions[valid]-positions[query], axis=1).min()
            np.testing.assert_allclose(a["min_past_gt_dist"][query], expected, atol=1e-12, rtol=0)
        counts = record["counts"]
        assert len(false) == counts["negative_false_accept"]
        rows.append({"sequence": sequence, **counts,
                     "negative_false_accept_rate": len(false)/len(negative) if len(negative) else None,
                     "AP": curve["AP"]})
    selections = np.asarray(selections, dtype=np.int64)
    fields = ("anchors", "hard_exact", "batch_exact", "hard_nearby", "batch_nearby", "reverse_exact")
    expected = {name: np.zeros(len(selections), np.int64) for name in fields}
    orders = read(REPO/"results/pdv2_temporal_focus_20261018/s1/standard_original/batch_order.json")["epochs"]
    for epoch in range(1, 11):
        refresh = 1 if epoch <= 5 else 6
        batch = arrays(REPO/f"results/pdv2_temporal_focus_20261018/samples/original_epoch{refresh}.npz")["triplets"]
        np.random.RandomState(100000+epoch).shuffle(batch)
        for index, query in enumerate(selections):
            winner = a["winner"][query]
            anchored = np.flatnonzero(batch[:, 0] == query)
            expected["anchors"][index] += len(anchored)
            for row in anchored:
                start = (row//1024)*1024
                end = min(start+1024, len(batch))
                positives = batch[start:end, 1]
                columns = np.arange(start, end) != row
                nearby = ((seqids[positives] == seqids[query]) & (positives <= query-30)
                          & (np.linalg.norm(positions[positives]-positions[winner], axis=1) <= 5))
                negative = batch[row, 2]
                expected["hard_exact"][index] += int(negative == winner)
                expected["batch_exact"][index] += int(((positives == winner) & columns).sum())
                expected["hard_nearby"][index] += int(seqids[negative] == seqids[query] and negative <= query-30
                    and np.linalg.norm(positions[negative]-positions[winner]) <= 5)
                expected["batch_nearby"][index] += int((nearby & columns).sum())
            for row in np.flatnonzero(batch[:, 1] == query):
                start = (row//1024)*1024
                end = min(start+1024, len(batch))
                expected["reverse_exact"][index] += int(((batch[start:end, 0] == winner)
                                                        & (np.arange(start, end) != row)).sum())
            expected["reverse_exact"][index] += int(((batch[:, 0] == winner) & (batch[:, 2] == query)).sum())
    for key in fields:
        np.testing.assert_array_equal(expected[key], a["count_"+key][selections])
    assert a["count_anchors"].sum() == 183249*10
    assert sum(r["queries"] for r in rows) == 72306-8*45
    with (OUT/"replay_by_sequence.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Plot measured counts and loss pressure; this is not a new model comparison.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), constrained_layout=True)
    axes[0].barh([r["sequence"] for r in rows], [100*r["negative_false_accept_rate"] for r in rows], color="#4477AA")
    axes[0].set_xlabel("False acceptance among non-revisits (%)")
    axes[0].set_title("Training replay, fixed validation threshold")
    false = summary["cohorts"]["all"]["false_accept"]
    values = [100*false[k]/false["n"] for k in ("anchor_seen", "explicit_exact_seen", "exact_any_forward_seen", "nearby_forward_seen")]
    axes[1].barh(["Anchor included", "Exact winner: explicit negative", "Exact winner: either negative source", "Within 5 m of winner"], values, color="#228833")
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Covered false-accept queries (%)")
    axes[1].set_title("Actual batches through epoch 10")
    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT/f"training_exposure.{suffix}", dpi=180)
    plt.close(fig)
    write_json(OUT/"completion_validation.json", {"status": "passed", "ledgers": len(rows),
        "queries": sum(r["queries"] for r in rows), "independent_exposure_queries": len(selections),
        "checked_epochs": 10, "exposure_fields": list(fields), "source_hashes": len(spec["sources"]),
        "test_used": False, "script_sha256": sha256(Path(__file__)),
        "summary_sha256": sha256(OUT/"summary.json")})
    print("VERIFIED", len(selections), len(rows), flush=True)


if __name__ == "__main__":
    main()
