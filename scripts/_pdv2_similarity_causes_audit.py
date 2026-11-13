#!/usr/bin/env python3
"""유사 간선 진단의 실제 개입·지연 의존성을 검사하고 결과 그림을 만든다."""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_similarity_causes as exp
from _pdv2_no_context_ablation import sha256, write_json


def audit() -> None:
    """실제 자료의 개수 통제와 q+15 절단 뒤 최종 후보 일치를 확인한다."""
    import torch
    from _lag_branch_fusion import lag_pass1_graph, TruncatedCache
    from _pdv2_attention_neighbors_verify_reference import decision
    from _xs_eval_lib import normalize
    exp.protocol(True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    bundle = exp.base.load_model("main")
    cfg, net, *_ = bundle
    cfg_t = copy.deepcopy(cfg)
    cfg_t["keyframe"].setdefault("graph", {})["similarity_threshold"] = 2.0
    records, prefixes, bins = [], [], {}
    selection = exp.base.read(exp.base.old.BASE / "s1/confidence_selection.json")["ops"]
    for s, q in exp.VAL:
        _, _, paths, _ = exp.base.old.alignment.base.inputs("val", s + "/" + q)
        with np.load(paths["cache"]) as z:
            cache = exp.base.old.ArrayCache({k: z[k] for k in z.files if k != "poses_gt"})
            raw, gt = z["descriptors"].astype(np.float32), z["poses_gt"][:, :3, 3]
        with np.load(exp.base.OUT / "main" / f"edges_{s}_{q}.npz") as z:
            c = {k: z[k] for k in z.files}
        n = len(raw)
        np.testing.assert_array_equal(c["distance_3d"], np.linalg.norm(gt[c["query"]] - gt[c["neighbor"]], axis=1))
        controls = {arm: exp.intervention(c, raw, arm) for arm in (*exp.REFS, *exp.ARMS)}
        ids_true, ids_nontrue, ids_random = [controls[a][2] for a in ("gt_true_only", "gt_nontrue_only", "gt_random_matched")]
        np.testing.assert_array_equal(np.sort(np.r_[ids_true, ids_nontrue]), np.sort(c["selected_xyz"]))
        np.testing.assert_array_equal(np.bincount(c["query"][ids_true], minlength=n), np.bincount(c["query"][ids_random], minlength=n))
        for arm in exp.ARMS:
            cand, prob, ids = controls[arm]
            record = exp.base.read(exp.OUT / f"keys_{s}_{q}.json")["arms"][arm]
            assert len(ids) == record["edges"]
            path = exp.OUT / arm / f"keys_{s}_{q}.npz"
            assert sha256(path) == record["keys_sha256"]
            with np.load(path) as z, np.load(exp.base.OUT / "main/xyz" / f"keys_{s}_{q}.npz") as ref:
                np.testing.assert_array_equal(z["scan_ids"], ref["scan_ids"])
                np.testing.assert_allclose(normalize(z["embedding"][:, :288]), normalize(ref["embedding"][:, :288]), atol=3e-7, rtol=0)
        graph = lag_pass1_graph(cache, raw, cfg_t, "cuda", s)
        cand, p, ids = controls["xyz"]
        changed = exp.base.replace_edges(graph, cand, p, ids)
        exp.infer_attention(net, changed, c["group"], c["valid"], gt)
        qids = c["query"][ids]
        # p가 낮은 유사 간선이 실제 attention에서 무시되는지 확인한다.
        for layer, conv in enumerate(net.gnn.convs):
            weights = conv._attn_weights.cpu().numpy().mean(1)[-len(ids):]
            for name, mask in [("all", c["valid"][qids]), *[(g, c["valid"][qids] & (c["group"][qids] == i)) for i, g in enumerate(exp.GROUPS)]]:
                for lo, hi in ((0., .1), (.1, .5), (.5, .9), (.9, 1.000001)):
                    m = mask & (p[ids] >= lo) & (p[ids] < hi)
                    key = f"layer{layer + 1}/{name}/{lo}_{hi}"
                    r = bins.setdefault(key, {"edges": 0, "attention_sum": 0., "probability_sum": 0., "far_edges": 0})
                    r["edges"] += int(m.sum())
                    r["attention_sum"] += float(weights[m].astype(np.float64).sum())
                    r["probability_sum"] += float(p[ids][m].astype(np.float64).sum())
                    r["far_edges"] += int((m & (c["distance_3d"][ids] > 25)).sum())
        records.append({"sequence": s + "/" + q, "policies_checked": list(controls)})
        if (s, q) not in (("kitti", "09"), ("mulran", "Sejong01")):
            continue
        with np.load(paths["layouts"]) as z:
            layout = z["nsd_layouts"].astype(np.float32)
        with np.load(paths["spec"]) as z:
            banks = {op: exp.base.old.alignment.base.bank_for(op, z) for op in exp.OPS}
        for query in (300, 900):
            cut = TruncatedCache(cache, query + 16)
            _, cg, cc = exp.base.capture(cut, s, layout[:query + 16], bundle, "cuda")
            _, probs, selected = exp.base.apply_classifier(net, cg, cc, exp.base.load_classifier("xyz"))
            cc.update(prob_xyz=probs, selected_xyz=selected)
            for arm in ("cosine_top10", "probability_one", "raw_key_attributes"):
                cand, prob, ids = exp.intervention(cc, raw[:query + 16], arm)
                with torch.no_grad():
                    short = normalize(net(exp.base.replace_edges(cg, cand, prob, ids)).float().cpu().numpy())
                with np.load(exp.OUT / arm / f"keys_{s}_{q}.npz") as z:
                    full = z["embedding"]
                error = float(np.max(np.abs(short[:query + 1] - full[:query + 1])))
                np.testing.assert_allclose(short[:query + 1], full[:query + 1], atol=2e-6, rtol=0)
                for op in exp.OPS:
                    d0, pool0 = decision(full, banks[op], query, selection[op])
                    d1, pool1 = decision(short, banks[op], query, selection[op])
                    np.testing.assert_array_equal(pool0, pool1)
                    assert d0 == d1
                prefixes.append({"sequence": s + "/" + q, "query": query, "arm": arm, "max_error": error,
                                 "same_pool_final_candidate_and_acceptance": True})
        print("AUDIT", s, q, flush=True)
    write_json(exp.OUT / "supplemental_validation.json", {"status": "passed", "policies": records,
        "prefixes": prefixes, "probability_attention_bins": bins, "test_used": False,
        "script_sha256": sha256(Path(__file__)), "protocol_sha256": sha256(exp.OUT / "protocol.json")})


def plot() -> None:
    """최종 성능과 attention을 별도 패널로 그리고 전체 지표를 CSV로 쓴다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = exp.base.read(exp.OUT / "summary.json")
    labels = {"temporal": "Temporal only", "xyz": "Similarity top-10 (3D classifier)",
              "gt_true_only": "GT: correct edges only", "gt_nontrue_only": "GT: non-correct edges only",
              "gt_random_matched": "GT: count-matched random", "cosine_top10": "Cosine top-10 (same scores)",
              "probability_one": "Classifier attribute = 1", "raw_key_attributes": "Raw-key cosine / L2 attributes"}
    arms = list(labels)
    with (exp.OUT / "comparison.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "operating_point", *exp.PRIMARY])
        for arm, ops in result["scores"].items():
            for op, row in ops.items():
                writer.writerow([arm, op, *[row["macro"][m] for m in exp.PRIMARY]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, op, name in zip(axes, exp.OPS, ("NSD", "NSD-H")):
        before = result["scores"]["temporal"][op]["macro"]["AP"]
        delta = [(result["scores"][a][op]["macro"]["AP"] - before) * 100 for a in arms]
        colors = ["#555555" if a == "temporal" else "#e59c45" if a.startswith("gt_") else "#377eb8" for a in arms]
        ax.barh(np.arange(len(arms)), delta, color=colors)
        ax.set_yticks(np.arange(len(arms)), [labels[a] for a in arms])
        ax.axvline(0, color="black", linewidth=.7)
        ax.set_title(name)
        ax.set_xlabel("Final AP difference vs temporal only (percentage points)")
        ax.grid(axis="x", alpha=.25)
    axes[0].invert_yaxis()
    fig.suptitle("Frozen main seed 1 | Validation only | Orange: privileged GT diagnostics")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(exp.OUT / ("final_ap_causes." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    names = ("same_available", "opposite_only", "negative")
    components = ("temporal", "similarity_true", "similarity_ambiguous", "similarity_far")
    colors = ("#377eb8", "#4daf4a", "#ffd166", "#e15759")
    for layer, ax in enumerate(axes):
        bottom = np.zeros(3)
        for component, color in zip(components, colors):
            values = np.array([result["attention"]["xyz"][layer][g][component + "_mass_sum"] / result["attention"]["xyz"][layer][g]["nodes"] * 100 for g in names])
            ax.bar(np.arange(3), values, bottom=bottom, label=component.replace("_", " "), color=color)
            bottom += values
        ax.set_xticks(np.arange(3), ("Same available", "Opposite only", "No revisit"))
        ax.set_title(f"Attention layer {layer + 1}")
    axes[0].set_ylabel("Mean incoming attention mass (%)")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2)
    fig.suptitle("Attention allocation is descriptive, not a causal effect estimate")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(exp.OUT / ("attention_allocation." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "plot"))
    globals()[parser.parse_args().action]()
