#!/usr/bin/env python3
"""공통 입력·정답·지표·학습 예산·지연 범위를 검증하고 진단 그림을 만든다."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_edge_reliability as exp
from _pdv2_no_context_ablation import sha256, write_json


def prefix_check() -> list:
    """CPU 기준으로 q+15까지의 입력만으로 후보 특징·점수가 재현되는지 검사한다."""
    import torch
    from _pdv2_pair_evaluate import ArrayCache
    from _lag_branch_fusion import lag_pass1_graph
    from _causal_edge_features import causal_edge_features
    from build_and_train_edge_classifier import range_phase_flat
    from _xs_eval_lib import normalize
    torch.set_num_threads(1)
    cfg, net, _, _, r, _ = exp.base.load_model("main", "cpu")
    classifiers = {a: exp.classifier(a, "cpu") for a in exp.ARMS}
    results = []
    for role, seq in (("train", "kitti/06"), ("val", "kitti/09")):
        cache, layout, spec, _, _, _ = exp.load_data(role, seq)
        raw = cache["descriptors"].astype(np.float32)
        with torch.no_grad():
            full = net(lag_pass1_graph(cache, raw, cfg, "cpu", "kitti")).numpy()
        for query in (300, 800):
            n = query + 16
            cut = ArrayCache({k: cache[k][:n] for k in cache.files})
            with torch.no_grad():
                prefix = net(lag_pass1_graph(cut, raw[:n], cfg, "cpu", "kitti")).numpy()
            embedding_error = float(np.max(np.abs(full[:query + 1] - prefix[:query + 1])))
            np.testing.assert_allclose(full[:query + 1], prefix[:query + 1], atol=2e-6, rtol=0)
            pairs = []
            for emb in (full[:n], prefix):
                q, j, x, unused = causal_edge_features(emb, normalize(raw[:n] * r[None]),
                    range_phase_flat(layout[:n]), cut["poses"], want_labels=False)
                assert unused is None
                mask = q == query
                pairs.append((q[mask], j[mask], x[mask]))
            qa, ja, xa = pairs[0]
            qb, jb, xb = pairs[1]
            np.testing.assert_array_equal(qa, qb)
            np.testing.assert_array_equal(ja, jb)
            np.testing.assert_allclose(xa, xb, atol=2e-5, rtol=0)
            ea = exp.extra_features(spec, qa, ja)
            eb = exp.extra_features({k: v[:n] for k, v in spec.items()}, qb, jb)
            for k in ea:
                np.testing.assert_array_equal(ea[k], eb[k])
            errors = {}
            for arm, (clf, blob) in classifiers.items():
                probs = []
                for x, extra in ((xa, ea), (xb, eb)):
                    inputs = exp.model_input({"features": x, **extra}, arm)
                    with torch.no_grad():
                        probs.append(torch.sigmoid(clf(torch.from_numpy((inputs - blob["mu"]) / blob["sd"]))).numpy())
                errors[arm] = float(np.max(np.abs(probs[0] - probs[1])))
                np.testing.assert_allclose(probs[0], probs[1], atol=2e-5, rtol=0)
            results.append({"role": role, "sequence": seq, "query": query,
                            "input_last_keyframe": n - 1, "embedding_error": embedding_error,
                            "feature_error": float(np.max(np.abs(xa - xb))), "probability_error": errors})
    return results


def figures(summary: dict) -> None:
    """최종 장소인식 지표와 구분한 간선 진단 그림과 표를 쓴다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dest = exp.OUT / "figures"
    dest.mkdir(exist_ok=True)
    names = list(exp.ARMS)
    labels = ["A: existing", "B: online inputs", "C: + NSD scores", "C: + NSD-H scores"]
    colors = ["#7c8a99", "#4285b4", "#329b75", "#c58537"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    panels = [("Within-query AUC (higher is better)",
               [summary["macro"][a]["groups"]["positive_query"]["within_query_AUC"] for a in names]),
              ("Hit@1 when candidate30 has a match",
               [summary["macro"][a]["groups"]["positive_query"]["hit1_given_available"] for a in names]),
              ("False connection on no-revisit queries\n(lower is better; train-holdout threshold)",
               [summary["macro"][a]["threshold_rates"]["no_revisit"]["rate"] for a in names])]
    for ax, (title, values) in zip(axes, panels):
        bars = ax.bar(np.arange(4), values, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_xticks(np.arange(4), labels, rotation=28, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Frozen GNN / same candidate30 / validation 9 sequences / 3-family macro\nEdge diagnostics only; these are not final place-recognition AP", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(dest / ("edge_reliability." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fields = ["score", "pooled_edge_AP", "pooled_edge_AUC", "macro_within_query_AUC",
              "macro_hit1_given_available", "macro_hit10_given_available", "macro_no_revisit_connection",
              "macro_correct_candidate_retention", "holdout_threshold"]
    with (exp.OUT / "diagnostic_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, macro in summary["macro"].items():
            writer.writerow(dict(zip(fields, [name, summary["pooled"][name]["edges"]["AP"],
                summary["pooled"][name]["edges"]["AUC"], macro["groups"]["positive_query"]["within_query_AUC"],
                macro["groups"]["positive_query"]["hit1_given_available"], macro["groups"]["positive_query"]["hit10_given_available"],
                macro["threshold_rates"]["no_revisit"]["rate"], macro["threshold_rates"]["candidate_has_correct"]["rate"],
                summary["thresholds"][name]["threshold"]])))


def main() -> None:
    """해시·경계·독립 AUC 계산·원시 정렬 재계산·동일 예산을 검증한다."""
    from sklearn.metrics import roc_auc_score
    p = exp.protocol(True)
    summary = exp.base.read(exp.OUT / "summary.json")
    records = []
    for role, roster in (("train", exp.TRAIN), ("val", exp.VAL)):
        for s, q in roster:
            sequence = s + "/" + q
            fp = exp.feature_path(role, sequence)
            meta = exp.base.read(fp.with_suffix(".json"))
            assert sha256(fp) == meta["sha256"]
            with np.load(fp) as z:
                c = {k: z[k] for k in z.files}
            cache, layout, spec, gt, minimum, _ = exp.load_data(role, sequence)
            n = len(gt)
            assert "poses_gt" not in cache.files
            np.testing.assert_array_equal(c["scan_ids"], cache["scan_ids"])
            np.testing.assert_array_equal(c["valid"], (np.arange(n) >= 30) & (np.arange(n) < n - 15))
            query, neighbor = c["query"], c["neighbor"]
            assert np.all(c["valid"][query]) and np.all(neighbor <= query - 30)
            distance = np.linalg.norm(gt[query, :3, 3] - gt[neighbor, :3, 3], axis=1)
            np.testing.assert_array_equal(distance, c["distance"])
            labels = np.where(distance < 5, 1, np.where(distance >= 25, 0, -1))
            np.testing.assert_array_equal(labels, c["labels"])
            for qi in np.linspace(30, n - 16, 20, dtype=int):
                direct = np.linalg.norm(gt[:qi - 29, :3, 3] - gt[qi, :3, 3], axis=1).min()
                np.testing.assert_allclose(direct, c["min_past"][qi], atol=1e-8, rtol=1e-8)
            sample = np.linspace(0, len(query) - 1, 64, dtype=int)
            extra = exp.extra_features(spec, query[sample], neighbor[sample])
            for k in extra:
                np.testing.assert_array_equal(extra[k], c[k][sample])
            if role == "val":
                with np.load(exp.base.OUT / "main" / f"edges_{s}_{q}.npz") as old:
                    keep = old["valid"][old["query"]]
                    for key in ("query", "neighbor", "features"):
                        np.testing.assert_array_equal(c[key], old[key][keep])
                    np.testing.assert_array_equal(c["probability_A"], old["prob_xyz"][keep])
            pred = exp.OUT / "predictions" / f"{role}_{s}_{q}.npz"
            pm = exp.base.read(pred.with_suffix(".json"))
            assert sha256(pred) == pm["sha256"] and pm["feature_sha256"] == sha256(fp)
            dp = exp.OUT / "diagnostics" / f"{role}_{s}_{q}.npz"
            dm = exp.base.read(dp.with_suffix(".json"))
            assert sha256(dp) == dm["node_metrics_sha256"] and dm["prediction_sha256"] == sha256(pred)
            checked = 0
            with np.load(dp) as d, np.load(pred) as predictions:
                mixed = np.flatnonzero(np.isfinite(d["A__within_query_AUC"]))
                for qi in mixed[np.linspace(0, len(mixed) - 1, min(40, len(mixed)), dtype=int)]:
                    lo, hi = np.searchsorted(query, [qi, qi + 1])
                    use = labels[lo:hi] >= 0
                    for arm in exp.ARMS:
                        actual = roc_auc_score(labels[lo:hi][use], predictions[arm][lo:hi][use])
                        np.testing.assert_allclose(actual, d[arm + "__within_query_AUC"][qi], atol=1e-12, rtol=0)
                        checked += 1
            records.append({"role": role, "sequence": sequence, "nodes": n, "edges": len(query),
                            "independent_AUC_checks": checked, "raw_alignment_pair_checks": len(sample)})
            print("AUDIT", role, sequence, flush=True)
    reference = exp.base.read(exp.base.OUT / "quality_diagnostics.json")["models"]["main"]["classifier_3d"]["xyz"]
    for k in ("AP", "AUC", "positive"):
        np.testing.assert_allclose(summary["pooled"]["A"]["edges"][k], reference[k], atol=1e-12, rtol=0)
    assert summary["pooled"]["A"]["edges"]["n"] == reference["rows"]
    train = [summary["training"][a] for a in exp.ARMS[1:]]
    for row in train:
        assert row["selected_epoch"] == 8 and not row["validation_used_for_selection"]
        for key in ("initial14_sha256", "initial16_sha256", "order_sha256", "features_sha256", "rows", "updates", "positive_weight"):
            assert row[key] == train[0][key]
        assert row["mu"][:14] == train[0]["mu"][:14] and row["sd"][:14] == train[0]["sd"][:14]
    for name, values in summary["thresholds"].items():
        scores = []
        for seq in exp.CFG["holdout"]:
            with np.load(exp.OUT / "diagnostics" / ("train_" + seq.replace("/", "_") + ".npz")) as z:
                scores.extend(z[name + "__max_score"][z["valid"] & z[name + "__has_correct"]])
        assert values["threshold"] == exp.retention_threshold(np.array(scores), .95)
        assert values["holdout_retention"] >= .95
    prefixes = prefix_check()
    figures(summary)
    write_json(exp.OUT / "completion_validation.json", {"status": "passed", "test_used": False,
        "runtime_files": len(p["runtime_sha256"]), "input_files": len(p["input_sha256"]),
        "sequences": records, "prefix_checks_CPU_reference": prefixes,
        "A_pooled_metrics_reproduced": True, "B_C_matched_initialization_rows_budget": True,
        "threshold_fit_scope": exp.CFG["holdout"], "summary_sha256": sha256(exp.OUT / "summary.json")})
    print("VERIFIED", len(records), len(prefixes), flush=True)


if __name__ == "__main__":
    main()
