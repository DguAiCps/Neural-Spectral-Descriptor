#!/usr/bin/env python3
"""간선 개입·최종 원장·15키프레임 입력 범위를 독립 검사하고 그림을 만든다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_edge_reliability_final as exp
from _pdv2_no_context_ablation import sha256, write_json


def verify() -> None:
    """선택·속성·개수·정답·최종 지표와 실제 잘린 입력 추론을 검증한다."""
    import torch
    from _lag_branch_fusion import TruncatedCache
    from _xs_eval_lib import normalize
    from _pdv2_attention_neighbors_verify_reference import decision
    p = exp.protocol(True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    summary = exp.base.read(exp.OUT / "summary.json")
    selection = exp.base.read(exp.base.old.BASE / "s1/confidence_selection.json")["ops"]
    ledgers = rows = 0
    policies, prefixes = [], []
    for arm in (*exp.REFS, *exp.ARMS):
        root = exp.reference_root(arm) if arm in exp.REFS else exp.OUT / arm
        records = {s + "/" + q: exp.base.read(root / f"val_{s}_{q}.json") for s, q in exp.VAL}
        for op in exp.OPS:
            assert exp.base.old.aggregate_rows(records, "val", op) == summary["scores"][arm][op]
            for s, q in exp.VAL:
                meta = records[s + "/" + q]["ops"][op]
                path = REPO / meta["ledger"]
                assert sha256(path) == meta["sha256"]
                a, _ = exp.base.old.read_ledger(path)
                ref, _ = exp.base.old.read_ledger(exp.reference_root("A") / f"val_{op}_{s}_{q}.npz")
                for k in ("query_idx", "min_past_gt_dist", "group"):
                    np.testing.assert_array_equal(a[k], ref[k])
                assert np.all(a["j_star"] <= a["query_idx"] - 30)
                sel = selection[op]
                assert exp.base.old.sequence_metrics(a, sel["decision"], sel["tau_star"]) == meta["metrics"]
                _, _, paths, _ = exp.base.old.alignment.base.inputs("val", s + "/" + q)
                with np.load(paths["cache"]) as z:
                    gt = z["poses_gt"][:, :3, 3]
                d = np.linalg.norm(gt[a["query_idx"]] - gt[a["j_star"]], axis=1)
                np.testing.assert_allclose(a["top1_geo_dist"], d, atol=1e-5, rtol=1e-7)
                ledgers += 1
                rows += len(a["query_idx"])
    bundle = exp.base.load_model("main")
    net = bundle[1]
    for s, q in exp.VAL:
        sequence = s + "/" + q
        record = exp.base.read(exp.OUT / f"keys_{s}_{q}.json")
        ep = exp.OUT / "edges" / f"{s}_{q}.npz"
        assert sha256(ep) == record["edge_sha256"]
        with np.load(ep) as z:
            scores = {k: z[k] for k in z.files}
        with np.load(exp.base.OUT / "main" / f"edges_{s}_{q}.npz") as z:
            c = {k: z[k] for k in z.files}
        n = record["nodes"]
        for k in ("query", "neighbor"):
            np.testing.assert_array_equal(scores[k], c[k])
        assert c["query"].max() == n - 1 and record["tail_edges"] == 450
        np.testing.assert_array_equal(scores["A"], c["prob_xyz"])
        keep = c["valid"][c["query"]]
        with np.load(exp.rel.OUT / "predictions" / f"val_{s}_{q}.npz") as z:
            for arm in exp.rel.ARMS:
                np.testing.assert_array_equal(scores[arm][keep], z[arm])
        controls = {}
        for op in exp.OPS:
            for arm in ("temporal", "A", *exp.ARMS):
                prob, ids = exp.choose(c, scores, op, arm)
                np.testing.assert_array_equal(ids, scores[f"selected__{op}__{arm}"])
                if arm == "temporal":
                    assert len(ids) == 0
                else:
                    expected = np.minimum(np.bincount(c["query"], minlength=n), 10)
                    np.testing.assert_array_equal(np.bincount(c["query"][ids], minlength=n), expected)
                if arm in ("C_selection", "alignment", "column"):
                    np.testing.assert_array_equal(prob, scores["A"])
                controls[op, arm] = ids
                if arm in exp.ARMS:
                    kp = exp.OUT / arm / f"keys_{op}_{s}_{q}.npz"
                    assert sha256(kp) == record["ops"][op][arm]["key_sha256"]
                    with np.load(kp) as z, np.load(exp.reference_root("A") / f"keys_{s}_{q}.npz") as ref:
                        np.testing.assert_array_equal(z["scan_ids"], ref["scan_ids"])
                        np.testing.assert_allclose(normalize(z["embedding"][:, :288]), normalize(ref["embedding"][:, :288]), atol=3e-7, rtol=0)
            np.testing.assert_array_equal(controls[op, "C"], controls[op, "C_selection"])
            np.testing.assert_array_equal(controls[op, "A"], c["selected_xyz"])
        policies.append({"sequence": sequence, "nodes": n, "tail_edges": record["tail_edges"],
                         "reference_key_error": record["reference_key_error"], "same_edge_counts": True})
        print("POLICIES", sequence, flush=True)
        if (s, q) not in (("kitti", "09"), ("mulran", "Sejong01")):
            continue
        cache, layout, spec, _, _, _ = exp.rel.load_data("val", sequence)
        banks = {op: exp.base.old.alignment.base.bank_for(op, spec) for op in exp.OPS}
        for query in (300, 900, n - 16):
            cut = TruncatedCache(cache, query + 16)
            _, graph, cc = exp.base.capture(cut, s, layout[:query + 16], bundle, "cuda")
            short_spec = {k: v[:query + 16] for k, v in spec.items()}
            extra = exp.rel.extra_features(short_spec, cc["query"], cc["neighbor"])
            pp = {**exp.probabilities(cc["features"], extra), **extra}
            for op in exp.OPS:
                short_bank = exp.base.old.alignment.base.bank_for(op, short_spec)
                for arm in exp.ARMS:
                    prob, ids = exp.choose(cc, pp, op, arm)
                    with torch.no_grad():
                        short = normalize(net(exp.base.replace_edges(graph, cc, prob, ids)).float().cpu().numpy())
                    with np.load(exp.OUT / arm / f"keys_{op}_{s}_{q}.npz") as z:
                        full = z["embedding"]
                    error = float(np.max(np.abs(full[:query + 1] - short[:query + 1])))
                    np.testing.assert_allclose(full[:query + 1], short[:query + 1], atol=2e-6, rtol=0)
                    d0, pool0 = decision(full, banks[op], query, selection[op])
                    d1, pool1 = decision(short, short_bank, query, selection[op])
                    np.testing.assert_array_equal(pool0, pool1)
                    assert d0 == d1, (sequence, query, op, arm, d0, d1)
                    prefixes.append({"sequence": sequence, "query": query, "op": op, "arm": arm,
                                     "max_key_error": error, "same_pool_final_candidate_acceptance": True})
            print("PREFIX", sequence, query, flush=True)
    write_json(exp.OUT / "completion_validation.json", {"status": "passed", "ledgers": ledgers, "rows": rows,
        "runtime_files": len(p["runtime_sha256"]), "input_files": len(p["input_sha256"]),
        "policies": policies, "prefixes": prefixes, "test_used": False,
        "summary_sha256": sha256(exp.OUT / "summary.json")})
    print("VERIFIED", ledgers, rows, len(prefixes), flush=True)


def plot() -> None:
    """최종 AP 변화와 방향별 최종 정답 수 변화를 별도 그림에 표시한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = exp.base.read(exp.OUT / "summary.json")
    names = {"temporal": "Temporal only", "A": "A: corrected 3D classifier", "B": "B: online input classifier",
             "C": "C: selection + score attribute", "C_selection": "C: selection only (A attribute)",
             "alignment": "Spectral alignment (A attribute)", "column": "Column alignment (A attribute)"}
    arms = list(names)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, op, title in zip(axes, exp.OPS, ("NSD", "NSD-H")):
        ref = s["scores"]["temporal"][op]["macro"]["AP"]
        delta = [100 * (s["scores"][a][op]["macro"]["AP"] - ref) for a in arms]
        bars = ax.barh(np.arange(len(arms)), delta, color=["#7d8991" if a in exp.REFS else "#4389b8" for a in arms])
        ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=8)
        ax.set_yticks(np.arange(len(arms)), [names[a] for a in arms])
        ax.axvline(0, color="black", linewidth=.7)
        ax.set_title(title)
        ax.set_xlabel("Final AP difference vs temporal only (percentage points)")
        ax.margins(x=.25)
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Frozen GNN / Same candidate30 and top10 / Validation 9 sequences / 3-family macro")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(exp.OUT / ("final_ap." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, op, title in zip(axes, exp.OPS, ("NSD", "NSD-H")):
        for i, (group, label) in enumerate((("same_available", "Same direction available"), ("opposite_only", "Opposite direction only"))):
            values = [s["transitions"][a][op]["temporal"][group]["recovered"] - s["transitions"][a][op]["temporal"][group]["regressed"] for a in exp.ARMS]
            ax.bar(np.arange(len(exp.ARMS)) + (i - .5) * .36, values, .36, label=label)
        ax.set_xticks(np.arange(len(exp.ARMS)), list(exp.ARMS), rotation=25)
        ax.set_title(title)
        ax.set_ylabel("Change in final correct queries vs temporal only")
        ax.axhline(0, color="black", linewidth=.7)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(exp.OUT / ("direction_correct." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    verify()
    plot()
