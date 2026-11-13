#!/usr/bin/env python3
"""입력 정합 대조의 동일 학습량·실제 검색·최종 지표를 독립 검산한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_input_alignment as run
from _causal_pr_dump import prefix_topk, minmax, align_scores, column_align_scores
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_encoder_place_verify import context_reference
from _pdv2_input_alignment_lib import common_triplets
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_sequence_fixed_test import frozen_scores, METHOD
from _train_attention_neighbors import array_hash


def check_training(spec: dict) -> dict:
    """네 조건의 실제 스캔 쌍·초기값·배치 순서·업데이트를 대조한다."""
    data = {p: run.previous.load(run.OUT / (p + "_inputs.npz")) for p in ("g", "o")}
    mapping = run.previous.load(run.OUT / "mapping.npz")
    graphs = {a: torch.load(run.OUT / (a + "_graph.pt"), weights_only=False, map_location="cpu") for a in run.ARMS}
    for a, (pop, pose) in run.ARMS.items():
        graph = graphs[a]
        assert graph.edge_type.eq(0).all()
        for field, h in spec["graphs"][a]["hashes"].items():
            assert array_hash(getattr(graph, field).numpy()) == h
        source, target = graph.edge_index.numpy()
        assert np.all(data[pop]["sequences"][source] == data[pop]["sequences"][target])
        assert np.max(abs(source - target)) == 5
        poses = data[pop]["gt" if pose == "g" else "odom"]
        for edge in np.linspace(0, len(source) - 1, 13).astype(int):
            i, j = source[edge], target[edge]
            distance = np.linalg.norm(poses[i, :3, 3] - poses[j, :3, 3])
            rot = poses[j, :3, :3] @ poses[i, :3, :3].T
            angle = np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))
            np.testing.assert_allclose(graph.edge_attr[edge].numpy(), [np.log1p(distance)/5, angle/np.pi, 0, 0, 1], atol=1e-7, rtol=0)
    for a, b in (("gg", "go"), ("og", "oo")):
        for f in ("x", "edge_index", "edge_type"):
            assert torch.equal(getattr(graphs[a], f), getattr(graphs[b], f))
        assert not torch.equal(graphs[a].edge_attr, graphs[b].edge_attr)
    for epoch in run.old.REFRESH:
        original = np.load(run.old.SAMPLES / f"original_epoch{epoch}.npz")["triplets"]
        valid, _ = common_triplets(original, mapping["gt_to_odom"])
        selected = np.random.default_rng(202610190 + epoch).choice(valid, spec["triplets_per_epoch"], replace=False)
        rows = run.previous.load(run.OUT / f"triplets_epoch{epoch}.npz")
        assert len(np.unique(selected)) == len(selected)
        np.testing.assert_array_equal(rows["original_rows"], selected)
        np.testing.assert_array_equal(rows["g"], original[selected])
        np.testing.assert_array_equal(mapping["odom_to_gt"][rows["o"]], rows["g"])
        for pop in ("g", "o"):
            a, p, n = rows[pop].T
            assert np.all(abs(a - p) >= 30) and np.all(abs(a - n) >= 30)
            xyz = data[pop]["gt"][:, :3, 3]
            assert np.all(np.linalg.norm(xyz[a] - xyz[p], axis=1) <= 5)
            assert np.all(np.linalg.norm(xyz[a] - xyz[n], axis=1) > 10)
        for f in ("scan_ids", "sequences", "raw", "gt"):
            np.testing.assert_array_equal(data["g"][f][rows["g"]], data["o"][f][rows["o"]])
    initial, order = set(), set()
    for arm in run.ARMS:
        root = run.OUT / arm
        completed = run.read(root / "training_complete.json")
        initial.add(sha256(root / "initial_model.json"))
        order.add(tuple(completed["order_hashes"]))
        checkpoint = root / "checkpoints/final_model.pth"
        assert sha256(checkpoint) == completed["checkpoint_sha256"]
        blob = torch.load(checkpoint, weights_only=False, map_location="cpu")
        assert blob["epoch"] == 19 and blob["val_metrics"] == []
        assert all(torch.isfinite(v).all() for v in blob["model_state_dict"].values())
        updates = spec["updates_per_epoch"] * 20
        assert {int(v["step"]) for v in blob["optimizer_state_dict"]["state"].values()} == {updates}
        assert completed["updates"] == updates and len(completed["order_hashes"]) == 20
    assert len(initial) == len(order) == 1
    # 공통 중심 스캔에서 실제로 같은 물리 스캔을 이웃으로 읽는 정도다.
    physical_neighbors = {}
    native_coverage = {}
    from scipy.spatial import cKDTree
    offsets = np.r_[-5:0, 1:6]
    for ordinal, sequence in enumerate(spec["train"]):
        g_ids = np.flatnonzero(data["g"]["sequences"] == ordinal)
        o_ids = np.flatnonzero(data["o"]["sequences"] == ordinal)
        common, gi, oi = np.intersect1d(data["g"]["scan_ids"][g_ids], data["o"]["scan_ids"][o_ids], return_indices=True)
        interior = (gi >= 5) & (gi < len(g_ids)-5) & (oi >= 5) & (oi < len(o_ids)-5)
        gn = data["g"]["scan_ids"][g_ids[gi[interior, None] + offsets]]
        on = data["o"]["scan_ids"][o_ids[oi[interior, None] + offsets]]
        overlap = (gn[:, :, None] == on[:, None, :]).any(2).sum(1)
        physical_neighbors[sequence] = {"interior_common_centers": int(interior.sum()),
            "mean_shared_neighbors_out_of_10": float(overlap.mean()),
            "fraction_all_ten_equal": float(np.mean(overlap == 10)),
            "mean_neighbor_jaccard": float(np.mean(overlap/(20-overlap)))}
        native_coverage[sequence] = {}
        for population, ids in (("g", g_ids), ("o", o_ids)):
            xyz = data[population]["gt"][ids, :3, 3]
            neighbors = cKDTree(xyz).query_ball_point(xyz, 5.)
            eligible = np.array([any(abs(i-j) >= 30 for j in js) for i, js in enumerate(neighbors)])
            shared = np.isin(data[population]["scan_ids"][ids], common)
            native_coverage[sequence][population] = {
                "frames_with_positive_at_5m_gap30": int(eligible.sum()),
                "of_these_only_in_this_population": int(np.sum(eligible & ~shared)),
                "scope": "GT 거리·번호차의 후보 존재 진단. 실제 마이너의 추가 제외·선택은 적용하지 않음"}
    write_json(run.OUT / "physical_neighbors.json", physical_neighbors)
    write_json(run.OUT / "native_pair_coverage.json", native_coverage)
    return {"models": 4, "triplets_per_epoch": spec["triplets_per_epoch"], "updates_per_model": updates,
            "same_initialization_and_physical_batch_order": True, "extra_sampling_replacement": False}


def check_validation(spec: dict, summary: dict) -> dict:
    """멀리 떨어진 질의의 실제 검색과 독립 PR·거리 판정을 검산한다."""
    selected = run.read(run.accept.OUT / "protocol.json")["seeds"]["1"]["confidence"]
    weights = selected["tuple"]
    fit = run.read(run.old.OUT / "calibration_freeze.json")["models"]["crossfit"]
    coefficients = np.asarray(fit["coefficients"]) / fit["scale"]
    bias = fit["intercept"] - np.dot(fit["mean"], coefficients)
    total, retrieval, prefix, metrics = 0, 0, 0, 0
    metric_error = 0.
    changes, oracle = {}, {}
    sources = {}
    for sequence in spec["val"]:
        paths = run.val_paths(sequence)
        cache = run.previous.load(paths["cache"])
        bank = run.old.focus.base.bank_for("pbev128_main", run.previous.load(paths["spec"]))
        hkey, hfft, hnorm, column = bank
        xyz = cache["poses_gt"][:, :3, 3]
        path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(cache["poses"][:, :3, 3], axis=0), axis=1))]
        reference = run.previous.load(run.previous.feature_path("full", sequence, "val"))
        for arm in ("oracle_gt", *run.ARMS):
            stem = sequence.replace("/", "_") + ".npz"
            p = run.OUT / arm / "val" / stem
            kp = run.OUT / arm / "keys" / stem
            row, key = run.previous.load(p), run.previous.load(kp)["embedding"]
            info, ki = run.read(p.with_suffix(".json")), run.read(kp.with_suffix(".json"))
            assert sha256(p) == info["sha256"] and sha256(kp) == ki["sha256"]
            prefix += len(ki["prefix"])
            if arm == "oracle_gt":
                oracle[sequence] = ki["diagnostics"]
            for field in ("query_idx", "min_past_gt_dist"):
                np.testing.assert_array_equal(row[field], reference[field])
            q, j = row["query_idx"], row["j_star"]
            assert np.all(j <= q - 30) and q[-1] == len(key) - 16
            np.testing.assert_allclose(np.linalg.norm(xyz[q] - xyz[j], axis=1), row["top1_geo_dist"], atol=1e-10)
            expected = np.column_stack([row[f] for f in fit["fields"]]) @ coefficients + bias
            np.testing.assert_allclose(expected, row["context_score"], atol=3e-12, rtol=0)
            for field in ("baseline", "context_score"):
                one = independent_metrics(row, row[field])
                for metric, value in one.items():
                    if value is None:
                        assert info["metrics"][field][metric] is None
                    else:
                        metric_error = max(metric_error, abs(value-info["metrics"][field][metric]))
                        # PR 곡선 적분의 float64 합산 순서 차이를 허용한다.
                        np.testing.assert_allclose(value, info["metrics"][field][metric], atol=5e-13, rtol=0)
                assert info["metrics"][field] == summary["per_sequence"][arm][field][sequence]
                metrics += 1
            for index in (0, len(q)//3, 2*len(q)//3, len(q)-1):
                t = q[index]
                candidates = np.unique(np.r_[prefix_topk(key, t, t-29, 100), prefix_topk(hkey, t, t-29, 100)])
                channels = (key[candidates] @ key[t], hkey[candidates] @ hkey[t],
                            align_scores(hfft, hnorm, t, candidates), column_align_scores(column, t, candidates))
                fused = sum(w * minmax(v) for w, v in zip(weights, channels))
                winner = candidates[np.argsort(-fused, kind="stable")[0]]
                assert winner == j[index], (arm, sequence, int(t))
                raw = channels[2 if selected["decision"] == "c_spec" else 3][np.flatnonzero(candidates == winner)[0]]
                one = frozen_scores(bank, path, q[index:index+1], j[index:index+1], selected["decision"], np.array([raw]))
                np.testing.assert_allclose(one[METHOD][0], row["baseline"][index], atol=2e-7, rtol=0)
                np.testing.assert_allclose(context_reference(key, path, row, index), row["context_arc"][index], atol=3e-7, rtol=0)
                minimum = np.linalg.norm(xyz[:t-29] - xyz[t], axis=1).min()
                np.testing.assert_allclose(minimum, row["min_past_gt_dist"][index], atol=1e-10)
                retrieval += 1
            old_correct = reference["top1_geo_dist"] <= 5
            new_correct = row["top1_geo_dist"] <= 5
            changes.setdefault(arm, {})[sequence] = {
                "winner_changed": int(np.sum(j != reference["j_star"])),
                "winner_correct_gained": int(np.sum(new_correct & ~old_correct)),
                "winner_correct_lost": int(np.sum(old_correct & ~new_correct))}
            total += len(q)
            for source in (p, p.with_suffix(".json"), kp, kp.with_suffix(".json")):
                sources[str(source.relative_to(REPO))] = sha256(source)
        print("CHECKED", sequence, flush=True)
    for arm, per in summary["per_sequence"].items():
        for method, values in per.items():
            assert run.accept.macro(values) == summary["aggregate"][arm][method]
    write_json(run.OUT / "diagnostics.json", {"oracle_same_scan": oracle, "winner_changes": changes})
    return {"validation_ledgers": 45, "validation_rows": total, "independent_metric_comparisons": metrics,
            "max_independent_metric_error": metric_error,
            "scalar_retrieval_and_score_checks": retrieval, "gpu_prefix_checks": prefix, "outputs": sources}


def plot(summary: dict) -> None:
    """최종 성능과 네 조건의 같은 예산 차이를 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/tmp/nsd_context_accept_font.ttf")
    plt.rcParams["font.family"] = font_manager.FontProperties(fname="/tmp/nsd_context_accept_font.ttf").get_name()
    plt.rcParams["axes.unicode_minus"] = False
    arms = ("reference", "oracle_gt", "gg", "go", "og", "oo")
    names = ("기준 모델\n900회", "같은 모델\nGT 자세 진단", "GT 선택\nGT 자세", "GT 선택\nKISS 자세", "KISS 선택\nGT 자세", "KISS 선택\nKISS 자세")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, metric in zip(axes, ("AP", "R@100P", "R@1")):
        for method, color, offset, label in (("baseline", "#52789d", -.18, "기존 시간 판정"),
                                              ("context_score", "#ce8541", .18, "고정 문맥 가중합")):
            values = [100*summary["aggregate"][a][method]["macro"][metric] for a in arms]
            bars = ax.bar(np.arange(len(arms)) + offset, values, .35, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        ax.set_xticks(np.arange(len(arms)), names, fontsize=8)
        ax.set_title(metric + " (%)")
        ax.set_ylim((65, 100) if metric == "AP" else ((25, 60) if metric == "R@100P" else (90, 100)))
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("검증 9 · 시드 1 · 뒤의 네 조건은 같은 실제 스캔 쌍 / 각 720회 업데이트", fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(run.OUT / ("input_alignment." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """학습·최종 평가 검산과 그림을 끝낸다."""
    spec = run.protocol()
    upstream = run.read(run.old.OUT / "completion_validation.json")["sources"]
    for p, h in upstream.items():
        assert sha256(REPO / p) == h, p
    summary = run.read(run.OUT / "summary.json")
    assert summary["protocol_sha256"] == sha256(run.OUT / "protocol.json")
    training = check_training(spec)
    validation = check_validation(spec, summary)
    sources = {**upstream, **spec["sources"]}
    sources.update(validation.pop("outputs"))
    for arm in run.ARMS:
        for name in ("training_complete.json", "initial_model.json", "training_progress.json", "checkpoints/final_model.pth"):
            p = run.OUT / arm / name
            sources[str(p.relative_to(REPO))] = sha256(p)
    for p in (Path(__file__), run.OUT / "summary.json", run.OUT / "diagnostics.json", run.OUT / "physical_neighbors.json", run.OUT / "native_pair_coverage.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    plot(summary)
    write_json(run.OUT / "completion_validation.json", {"status": "passed", **training, **validation,
        "sources": sources, "test_used": False, "scope": "같은 실제 스캔 쌍·초기값·순서·업데이트, 네 그래프 요인 분리, 최종 승자·점수·독립 PR 검산"})
    print("VERIFIED", training, validation, len(sources), flush=True)


if __name__ == "__main__":
    main()
