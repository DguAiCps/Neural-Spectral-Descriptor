#!/usr/bin/env python3
"""고유 스캔 표본·같은 예산·최종 검색과 PR 지표를 별도로 검산한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_native_pairs as run
from _causal_pr_dump import prefix_topk, minmax, align_scores, column_align_scores
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_encoder_place_verify import context_reference
from _pdv2_native_pairs_lib import nested_rows, positive_pools
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_sequence_fixed_test import frozen_scores, METHOD
from _train_attention_neighbors import array_hash
from gnn.matching_mining import paired_triplets


def check_training(spec: dict) -> dict:
    """새 스캔 실제 포함·원래 마이너 재현·같은 예산과 순열을 확인한다."""
    data = {p: run.align.previous.load(run.INPUT / (p+"_inputs.npz")) for p in ("g", "o")}
    mapping = run.align.previous.load(run.INPUT / "mapping.npz")
    state = torch.load(run.align.old.previous.PREPARED, weights_only=False, map_location="cpu")
    negative = {"g": state["negative"], "o": np.load(run.OUT / "kiss_hard_negative.npz")["negative"]}
    source_checks = 0
    for pop, d in data.items():
        pools = positive_pools(d["gt"][:, :3, 3], d["sequences"])
        for epoch in run.align.old.REFRESH:
            generated, _ = paired_triplets(pools, negative[pop], np.full((len(pools), 1), -1),
                100000+epoch, fraction=0.)
            saved = np.load(run.OUT / f"{pop}_native_epoch{epoch}.npz")["triplets"]
            np.testing.assert_array_equal(generated, saved)
            if pop == "g":
                np.testing.assert_array_equal(saved, np.load(run.align.old.SAMPLES / f"original_epoch{epoch}.npz")["triplets"])
            anchors, counts = np.unique(saved[:, 0], return_counts=True)
            np.testing.assert_array_equal(counts, np.full(len(anchors), 3))
            assert len(anchors) == (61083 if pop == "g" else 60973)
            for field in (1, 2):
                assert np.all(d["sequences"][saved[:, 0]] == d["sequences"][saved[:, field]])
                assert np.all(abs(saved[:, 0]-saved[:, field]) >= 30)
            xyz = d["gt"][:, :3, 3]
            assert np.all(np.linalg.norm(xyz[saved[:, 0]]-xyz[saved[:, 1]], axis=1) <= 5.)
            assert np.all(np.linalg.norm(xyz[saved[:, 0]]-xyz[saved[:, 2]], axis=1) > 10.)
        # 실제 KISS 고유 앵커와 공통 앵커의 오답을 전체 원 특징 거리로 검산한다.
        exclusive = mapping["gt_to_odom" if pop == "g" else "odom_to_gt"] < 0
        for seq in np.unique(d["sequences"]):
            ids = np.flatnonzero(d["sequences"] == seq)
            for own in (False, True):
                choices = ids[(exclusive[ids] == own) & np.array([bool(len(pools[i])) for i in ids])]
                if not len(choices):
                    continue
                for a in choices[np.unique(np.linspace(0, len(choices)-1, min(16, len(choices))).astype(int))]:
                    candidates = ids[(abs(ids-a) >= 30) & (np.linalg.norm(xyz[ids]-xyz[a], axis=1) > 10.)]
                    distance = np.sum((d["raw"][candidates]-d["raw"][a])**2, axis=1)
                    assert candidates[np.argmin(distance)] == negative[pop][a]
                    source_checks += 1
    initial, permutations = set(), {"small": [], "large": []}
    cohort = {}
    for arm, plan in spec["plans"].items():
        root = run.OUT / arm
        pop, budget = plan["population"], plan["budget"]
        exclusive = mapping["gt_to_odom" if pop == "g" else "odom_to_gt"] < 0
        cohort[arm] = {}
        for epoch in run.align.old.REFRESH:
            native = np.load(run.OUT / f"{pop}_native_epoch{epoch}.npz")["triplets"]
            chosen = nested_rows(len(native), spec["budgets"]["large"], spec["budgets"]["small"], 202610200+epoch)
            saved = run.align.previous.load(root / f"triplets_epoch{epoch}.npz")
            assert len(np.unique(saved["original_rows"])) == plan["triplets_per_epoch"]
            np.testing.assert_array_equal(saved["original_rows"], chosen[budget])
            np.testing.assert_array_equal(saved["triplets"], native[chosen[budget]])
            assert np.isin(chosen["small"], chosen["large"]).all()
            anchors = np.unique(saved["triplets"][:, 0])
            count = int(exclusive[anchors].sum())
            assert count == spec["sample_audit"][arm][str(epoch)]["unique_exclusive_anchors"] and count > 0
            if arm == "kiss_large":
                assert count == 6489
            cohort[arm][str(epoch)] = count
        completed = run.read(root / "training_complete.json")
        checkpoint = root / "checkpoints/final_model.pth"
        assert sha256(checkpoint) == completed["checkpoint_sha256"]
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert blob["epoch"] == 19 and blob["val_metrics"] == []
        assert all(torch.isfinite(v).all() for v in blob["model_state_dict"].values())
        expected = plan["updates_per_epoch"]*20
        assert completed["updates"] == expected
        assert {int(v["step"]) for v in blob["optimizer_state_dict"]["state"].values()} == {expected}
        initial.add(sha256(root / "initial_model.json"))
        permutations[budget].append(tuple(x["permutation_sha256"] for x in completed["order_hashes"]))
        for epoch, record in enumerate(completed["order_hashes"], 1):
            trips = np.load(root / f"triplets_epoch{1+5*((epoch-1)//5)}.npz")["triplets"]
            permutation = np.arange(len(trips))
            np.random.RandomState(100000+epoch).shuffle(permutation)
            assert array_hash(permutation) == record["permutation_sha256"]
            assert array_hash(trips[permutation]) == record["triplets_sha256"]
    assert len(initial) == 1
    assert initial == {sha256(run.INPUT / "gg/initial_model.json")}
    assert all(len(set(values)) == 1 for values in permutations.values())
    from scipy.spatial import cKDTree
    native_o = np.load(run.OUT / "o_native_epoch1.npz")["triplets"]
    active_o = np.unique(native_o[:, 0])
    unique_o = active_o[mapping["odom_to_gt"][active_o] < 0]
    geometry, distances = {}, []
    for ordinal, sequence in enumerate(spec["train"]):
        selected = unique_o[data["o"]["sequences"][unique_o] == ordinal]
        reference = np.flatnonzero(data["g"]["sequences"] == ordinal)
        if not len(selected):
            geometry[sequence] = {"exclusive_anchors": 0}
            continue
        distance, _ = cKDTree(data["g"]["gt"][reference, :3, 3]).query(data["o"]["gt"][selected, :3, 3])
        distances.extend(distance.tolist())
        geometry[sequence] = {"exclusive_anchors": len(selected),
            "nearest_gt_selected_distance_q50_q90_q99_max_m": np.quantile(distance, [.5, .9, .99, 1.]).tolist()}
    geometry["all"] = {"exclusive_anchors": len(distances),
        "nearest_gt_selected_distance_q50_q90_q99_max_m": np.quantile(distances, [.5, .9, .99, 1.]).tolist(),
        "scope": "같은 학습 시퀀스의 기존 GT 선택 스캔까지 GT 3차원 거리. 새 독립 장소의 추가가 아님."}
    write_json(run.OUT / "exclusive_geometry.json", geometry)
    return {"models": 4, "negative_reference_checks": source_checks,
            "same_initialization": True, "same_permutation_within_budget": True,
            "exclusive_anchors_by_refresh": cohort}


def check_validation(spec: dict, summary: dict) -> dict:
    """정답·과거 후보·실제 승자·판정 점수·독립 PR를 확인한다."""
    align = run.align
    conf = run.read(align.accept.OUT / "protocol.json")["seeds"]["1"]["confidence"]
    fit = run.read(align.old.OUT / "calibration_freeze.json")["models"]["crossfit"]
    coef = np.asarray(fit["coefficients"])/fit["scale"]
    intercept = fit["intercept"]-np.dot(fit["mean"], coef)
    outputs, comparison = {}, {}
    rows, metric_checks, scalar_checks, prefix, error = 0, 0, 0, 0, 0.
    for sequence in spec["val"]:
        cache = align.previous.load(align.val_paths(sequence)["cache"])
        bank = align.old.focus.base.bank_for("pbev128_main", align.previous.load(align.val_paths(sequence)["spec"]))
        xyz = cache["poses_gt"][:, :3, 3]
        path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(cache["poses"][:, :3, 3], axis=0), axis=1))]
        reference = align.previous.load(align.previous.feature_path("full", sequence, "val"))
        ledgers = {}
        for arm in run.ARMS:
            name = sequence.replace("/", "_")+".npz"
            p, kp = run.OUT / arm / "val" / name, run.OUT / arm / "keys" / name
            row, key = align.previous.load(p), align.previous.load(kp)["embedding"]
            ledgers[arm] = row
            info, ki = run.read(p.with_suffix(".json")), run.read(kp.with_suffix(".json"))
            assert sha256(p) == info["sha256"] and sha256(kp) == ki["sha256"]
            prefix += len(ki["prefix"])
            for f in ("query_idx", "min_past_gt_dist"):
                np.testing.assert_array_equal(row[f], reference[f])
            q, j = row["query_idx"], row["j_star"]
            assert np.all(j <= q-30) and q[-1] == len(key)-16
            np.testing.assert_allclose(np.linalg.norm(xyz[q]-xyz[j], axis=1), row["top1_geo_dist"], atol=1e-10)
            expected = np.column_stack([row[f] for f in fit["fields"]])@coef+intercept
            np.testing.assert_allclose(expected, row["context_score"], atol=3e-12, rtol=0)
            for method in ("baseline", "context_score"):
                independent = independent_metrics(row, row[method])
                for metric, value in independent.items():
                    if value is None:
                        assert info["metrics"][method][metric] is None
                    else:
                        error = max(error, abs(value-info["metrics"][method][metric]))
                        np.testing.assert_allclose(value, info["metrics"][method][metric], atol=5e-13, rtol=0)
                assert info["metrics"][method] == summary["per_sequence"][arm][method][sequence]
                metric_checks += 1
            hkey, hfft, hnorm, column = bank
            for index in (0, len(q)//3, 2*len(q)//3, len(q)-1):
                t = q[index]
                cand = np.unique(np.r_[prefix_topk(key, t, t-29, 100), prefix_topk(hkey, t, t-29, 100)])
                channels = (key[cand]@key[t], hkey[cand]@hkey[t],
                            align_scores(hfft, hnorm, t, cand), column_align_scores(column, t, cand))
                fused = sum(w*minmax(v) for w, v in zip(conf["tuple"], channels))
                best = np.argsort(-fused, kind="stable")[0]
                assert cand[best] == j[index]
                raw = np.asarray([channels[2 if conf["decision"] == "c_spec" else 3][best]])
                score = frozen_scores(bank, path, q[index:index+1], j[index:index+1], conf["decision"], raw)[METHOD][0]
                np.testing.assert_allclose(score, row["baseline"][index], atol=2e-7, rtol=0)
                np.testing.assert_allclose(context_reference(key, path, row, index), row["context_arc"][index], atol=3e-7, rtol=0)
                np.testing.assert_allclose(np.linalg.norm(xyz[:t-29]-xyz[t], axis=1).min(), row["min_past_gt_dist"][index], atol=1e-10)
                scalar_checks += 1
            rows += len(q)
            for source in (p, p.with_suffix(".json"), kp, kp.with_suffix(".json")):
                outputs[str(source.relative_to(REPO))] = sha256(source)
        comparison[sequence] = {}
        for budget in ("small", "large"):
            g, o = ledgers["gt_"+budget], ledgers["kiss_"+budget]
            correct_g, correct_o = g["top1_geo_dist"] <= 5, o["top1_geo_dist"] <= 5
            comparison[sequence][budget] = {
                "winner_changes": int(np.sum(g["j_star"] != o["j_star"])),
                "correct_gained": int(np.sum(correct_o & ~correct_g)),
                "correct_lost": int(np.sum(correct_g & ~correct_o))}
        print("CHECKED", sequence, flush=True)
    for arm, methods in summary["per_sequence"].items():
        for method, values in methods.items():
            assert align.accept.macro(values) == summary["aggregate"][arm][method]
    write_json(run.OUT / "winner_changes.json", comparison)
    return {"validation_ledgers": 36, "validation_rows": rows, "independent_metric_comparisons": metric_checks,
        "max_metric_error": error, "retrieval_and_scalar_score_checks": scalar_checks,
        "gpu_prefix_checks": prefix, "outputs": outputs}


def plot(summary: dict) -> None:
    """학습 표본·예산별 최종 성능을 점 그래프로 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/tmp/nsd_context_accept_font.ttf")
    plt.rcParams["font.family"] = font_manager.FontProperties(fname="/tmp/nsd_context_accept_font.ttf").get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, (method, title) in enumerate((("baseline", "기존 시간 판정"), ("context_score", "고정 문맥 가중합"))):
        for row, metric in enumerate(("AP", "R@100P")):
            ax = axes[row, col]
            for pop, color, arms, offset in (
                ("GT", "#426b98", ("common_gt", "gt_small", "gt_large"), 10),
                ("KISS", "#b16430", ("common_kiss", "kiss_small", "kiss_large"), -16)):
                values = [100*summary["aggregate"][a][method]["macro"][metric] for a in arms]
                ax.plot(range(3), values, "o-", label=pop+" 기준", color=color, linewidth=1.5)
                for i, value in enumerate(values):
                    ax.annotate(f"{value:.3f}", (i, value), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=8, color=color)
            ref = 100*summary["aggregate"]["reference"][method]["macro"][metric]
            ax.axhline(ref, color="#777777", linestyle="--", label=f"기존 전체 쌍 900회 ({ref:.3f})")
            ax.set_xticks(range(3), ["공통 쌍\n720회", "각 선택의 쌍\n720회", "각 선택의 쌍\n900회"])
            ax.set_title(title+" · "+metric+" (%)")
            ax.margins(x=.15, y=.35)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("새 스캔 학습 표본과 예산 대조 · 검증 9 · 시드 1")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(run.OUT / ("native_pairs."+suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """모든 근거와 최종 평가를 확인하고 분석 그림을 만든다."""
    spec = run.protocol()
    summary = run.read(run.OUT / "summary.json")
    assert sha256(run.OUT / "protocol.json") == summary["protocol_sha256"]
    training = check_training(spec)
    validation = check_validation(spec, summary)
    sources = {**spec["sources"], **validation.pop("outputs")}
    for arm in run.ARMS:
        for name in ("training_complete.json", "training_progress.json", "initial_model.json", "checkpoints/final_model.pth"):
            p = run.OUT / arm / name
            sources[str(p.relative_to(REPO))] = sha256(p)
    for p in (Path(__file__), run.OUT / "summary.json", run.OUT / "winner_changes.json", run.OUT / "exclusive_geometry.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    plot(summary)
    write_json(run.OUT / "completion_validation.json", {"status": "passed", **training, **validation,
        "sources": sources, "test_used": False})
    print("VERIFIED", training, validation, len(sources), flush=True)


if __name__ == "__main__":
    main()
