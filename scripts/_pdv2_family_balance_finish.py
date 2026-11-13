#!/usr/bin/env python3
"""같은 배치 학습·계열 기울기·네 시점의 최종 평가를 검산한다."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_family_balance as run
from _causal_pr_dump import prefix_topk, minmax, align_scores, column_align_scores
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_encoder_place_verify import context_reference
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_sequence_fixed_test import frozen_scores, METHOD
from _train_attention_neighbors import array_hash


def verify_training(spec: dict) -> dict:
    """같은 초기값·배치·LR·실제 업데이트와 연속 체크포인트를 확인한다."""
    initial, schedules, source_files = set(), [], []
    for arm in run.ARMS:
        root = run.OUT / arm
        done = run.read(root / "training_complete.json")
        records = run.read(root / "training_progress.json")["records"]
        assert done["updates"] == 900 and len(records) == 20
        assert sha256(root / "checkpoints/final_model.pth") == done["checkpoint_sha256"]
        initial.add(sha256(root / "initial_model.json"))
        schedules.append([r["learning_rate"] for r in records])
        for epoch, record in enumerate(records, 1):
            triplets = run.trips(epoch)
            indices = np.arange(len(triplets))
            np.random.RandomState(100000+epoch).shuffle(indices)
            assert record["updates"] == epoch*45
            assert array_hash(indices) == record["permutation_sha256"]
            assert array_hash(triplets[indices]) == record["triplets_sha256"]
            assert record["family_counts_verified"]
        for updates, epoch, name in ((720, 15, "update_720.pth"), (900, 19, "final_model.pth")):
            path = run.OUT / f"{arm}_{updates}" / "checkpoints/final_model.pth"
            assert sha256(path) == done["checkpoints"][str(updates)]
            assert sha256(path) == sha256(root / "checkpoints" / name)
            blob = torch.load(path, weights_only=False, map_location="cpu")
            assert blob["epoch"] == epoch and blob["val_metrics"] == []
            assert len(blob["train_losses"]) == epoch+1 and blob["global_step"] == 179*(epoch+1)
            assert {int(x["step"]) for x in blob["optimizer_state_dict"]["state"].values()} == {updates}
            assert all(torch.isfinite(v).all() for v in blob["model_state_dict"].values())
            source_files.append(path)
        source_files.extend(root / n for n in ("training_complete.json", "training_progress.json", "initial_model.json"))
    assert len(initial) == 1 and initial == {sha256(run.SAMPLES / "initial_model.json")}
    assert schedules[0] == schedules[1]
    assert schedules[0][15] > schedules[0][19]
    original = torch.load(run.SAMPLES / "checkpoints/final_model.pth", map_location="cpu", weights_only=False)["model_state_dict"]
    repeated = torch.load(run.OUT / "natural/checkpoints/final_model.pth", map_location="cpu", weights_only=False)["model_state_dict"]
    differences = []
    for name, value in original.items():
        delta = value.double()-repeated[name].double()
        differences.append({"name": name, "max_abs": float(delta.abs().max()),
                            "relative_l2": float(delta.norm()/value.double().norm().clamp(min=1e-30)),
                            "reference_max_abs": float(value.double().abs().max())})
    write_json(run.OUT / "control_replay_detail.json", {
        "tensors": differences, "scope": "같은 초기값·표본·배치 순서·LR·업데이트 재실행의 차이. 비트 단위 재현으로 주장하지 않는다."})
    return {"models": 2, "checkpoints": 4, "same_initialization_batch_and_schedule": True,
            "natural_reference_replay": run.read(run.OUT / "natural/training_complete.json")["reference_replay"],
            "sources": {str(p.relative_to(REPO)): sha256(p) for p in source_files}}


def verify_diagnostics() -> dict:
    """저장된 기울기 벡터에서 크기·방향과 국소 손익을 별도로 계산한다."""
    records, pooled, sources, maximum_error = {}, {}, {}, 0.
    batches = run.align.previous.load(run.OUT / "probe_batches.npz")
    data = run.training_data()
    for stage in ("initial", "reference", *run.EVAL_ARMS):
        p = run.OUT / "diagnostics" / (stage+".npz")
        arrays, meta = run.align.previous.load(p), run.read(p.with_suffix(".json"))
        assert sha256(p) == meta["sha256"] and meta["state_restored_exactly"]
        assert sha256(run.OUT / "protocol.json") == meta["protocol_sha256"]
        if meta["checkpoint"]:
            assert sha256(REPO / meta["checkpoint"]) == meta["checkpoint_sha256"]
        assert all(x["relative_l2_error"] < .002 for x in meta["additivity_checks"])
        a, b = arrays["gradient_kitti"].astype(float), arrays["gradient_bridge"].astype(float)
        assert a.shape == b.shape == (24, sum(x["size"] for x in meta["parameters"]))
        assert np.isfinite(a).all() and np.isfinite(b).all()
        norm_a, norm_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        fraction = arrays["kitti_fraction"]
        natural = fraction[:, None]*a+(1-fraction[:, None])*b
        balanced = (a+b)/2
        dot = lambda x, y: np.einsum("ij,ij->i", x, y)
        expected = {
            "mean_gradient_norm_kitti": norm_a, "mean_gradient_norm_bridge": norm_b,
            "gradient_cosine": dot(a, b)/(norm_a*norm_b),
            "natural_bridge_to_kitti_norm_ratio": (1-fraction)*norm_b/(fraction*norm_a),
            "natural_kitti_dot": dot(a, natural), "natural_bridge_dot": dot(b, natural),
            "balanced_kitti_dot": dot(a, balanced), "balanced_bridge_dot": dot(b, balanced),
        }
        for field, values in expected.items():
            saved = np.array([x[field] for x in meta["batches"]])
            maximum_error = max(maximum_error, float(abs(values-saved).max()))
            np.testing.assert_allclose(values, saved, rtol=1e-10, atol=1e-12)
        for item in meta["batches"]:
            ids = batches[str(item["batch_id"])]
            assert len(ids) == item["rows"]
            assert int((data["sequences"][ids[:, 0]] < 4).sum()) == item["n_kitti"]
            np.testing.assert_allclose(item["kitti_fraction"], item["n_kitti"]/item["rows"], atol=0, rtol=0)
            p0 = item["kitti_fraction"]
            l0, l1 = item["loss_mean_kitti"], item["loss_mean_bridge"]
            np.testing.assert_allclose(item["natural_loss_bridge_fraction"], (1-p0)*l1/(p0*l0+(1-p0)*l1))
        summary = meta["summary"]
        np.testing.assert_allclose(summary["median_gradient_cosine"], np.median(expected["gradient_cosine"]))
        np.testing.assert_allclose(summary["median_natural_bridge_to_kitti_gradient_norm_ratio"],
                                   np.median(expected["natural_bridge_to_kitti_norm_ratio"]))
        assert summary["opposed_batches"] == int((expected["gradient_cosine"] < 0).sum())
        for method in ("natural", "balanced"):
            for family in ("kitti", "bridge"):
                assert summary[f"{method}_direction_harms_{family}_batches"] == int((expected[f"{method}_{family}_dot"] < 0).sum())
        records[stage] = summary
        mean_a, mean_b = a.mean(0), b.mean(0)
        part_a, part_b = (fraction[:, None]*a).mean(0), ((1-fraction[:, None])*b).mean(0)
        total, equal = part_a+part_b, (mean_a+mean_b)/2
        pooled[stage] = {
            "scope": "진단 24미니배치의 기울기 평균. 전체 학습 배치나 Adam 업데이트의 재현은 아님.",
            "mean_loss_kitti": float(np.mean([x["loss_mean_kitti"] for x in meta["batches"]])),
            "mean_loss_bridge": float(np.mean([x["loss_mean_bridge"] for x in meta["batches"]])),
            "natural_contribution_cosine": float(np.dot(part_a, part_b)/np.linalg.norm(part_a)/np.linalg.norm(part_b)),
            "natural_bridge_to_kitti_norm_ratio": float(np.linalg.norm(part_b)/np.linalg.norm(part_a)),
            "natural_kitti_dot": float(np.dot(mean_a, total)),
            "natural_bridge_dot": float(np.dot(mean_b, total)),
            "balanced_kitti_dot": float(np.dot(mean_a, equal)),
            "balanced_bridge_dot": float(np.dot(mean_b, equal)),
        }
        sources[str(p.relative_to(REPO))] = sha256(p)
        sources[str(p.with_suffix(".json").relative_to(REPO))] = sha256(p.with_suffix(".json"))
    write_json(run.OUT / "diagnostic_summary.json", records)
    write_json(run.OUT / "pooled_diagnostics.json", pooled)
    return {"gradient_states": 6, "gradient_batches": 144, "gradient_max_absolute_error": maximum_error,
            "sources": sources}


def verify_validation(spec: dict, summary: dict) -> dict:
    """독립 PR·과거 후보·스칼라 검색과 점수를 재계산한다."""
    align = run.align
    conf = run.read(align.accept.OUT / "protocol.json")["seeds"]["1"]["confidence"]
    fit = run.read(align.old.OUT / "calibration_freeze.json")["models"]["crossfit"]
    coefficient = np.array(fit["coefficients"])/fit["scale"]
    intercept = fit["intercept"]-np.dot(fit["mean"], coefficient)
    sources, changes = {}, {}
    rows = metrics = scalar = prefix = 0
    max_error = 0.
    for sequence in spec["val"]:
        cache = align.previous.load(align.val_paths(sequence)["cache"])
        bank = align.old.focus.base.bank_for("pbev128_main", align.previous.load(align.val_paths(sequence)["spec"]))
        xyz = cache["poses_gt"][:, :3, 3]
        path = np.r_[0., np.cumsum(np.linalg.norm(np.diff(cache["poses"][:, :3, 3], axis=0), axis=1))]
        reference = align.previous.load(align.previous.feature_path("full", sequence, "val"))
        ledgers = {}
        for arm in run.EVAL_ARMS:
            name = sequence.replace("/", "_")+".npz"
            p, kp = run.OUT/arm/"val"/name, run.OUT/arm/"keys"/name
            ledger, key = align.previous.load(p), align.previous.load(kp)["embedding"]
            info, ki = run.read(p.with_suffix(".json")), run.read(kp.with_suffix(".json"))
            assert sha256(p) == info["sha256"] and sha256(kp) == ki["sha256"]
            for field in ("query_idx", "min_past_gt_dist"):
                np.testing.assert_array_equal(ledger[field], reference[field])
            q, j = ledger["query_idx"], ledger["j_star"]
            assert np.all(j <= q-30) and q[-1] == len(key)-16
            np.testing.assert_allclose(ledger["top1_geo_dist"], np.linalg.norm(xyz[q]-xyz[j], axis=1), atol=1e-10)
            expected = np.column_stack([ledger[f] for f in fit["fields"]])@coefficient+intercept
            np.testing.assert_allclose(expected, ledger["context_score"], atol=3e-12, rtol=0)
            for method in ("baseline", "context_score"):
                independent = independent_metrics(ledger, ledger[method])
                for metric, value in independent.items():
                    saved = info["metrics"][method][metric]
                    if value is None:
                        assert saved is None
                    else:
                        max_error = max(max_error, abs(value-saved))
                        np.testing.assert_allclose(value, saved, atol=5e-13, rtol=0)
                assert info["metrics"][method] == summary["per_sequence"][arm][method][sequence]
                metrics += 1
            hkey, hfft, hnorm, column = bank
            for index in (0, len(q)//3, 2*len(q)//3, len(q)-1):
                t = q[index]
                candidates = np.unique(np.r_[prefix_topk(key, t, t-29, 100), prefix_topk(hkey, t, t-29, 100)])
                channels = (key[candidates]@key[t], hkey[candidates]@hkey[t],
                            align_scores(hfft, hnorm, t, candidates), column_align_scores(column, t, candidates))
                fused = sum(w*minmax(v) for w, v in zip(conf["tuple"], channels))
                selected = np.argsort(-fused, kind="stable")[0]
                assert candidates[selected] == j[index]
                raw = np.asarray([channels[2 if conf["decision"] == "c_spec" else 3][selected]])
                score = frozen_scores(bank, path, q[index:index+1], j[index:index+1], conf["decision"], raw)[METHOD][0]
                np.testing.assert_allclose(score, ledger["baseline"][index], atol=2e-7, rtol=0)
                np.testing.assert_allclose(context_reference(key, path, ledger, index), ledger["context_arc"][index], atol=3e-7, rtol=0)
                np.testing.assert_allclose(np.linalg.norm(xyz[:t-29]-xyz[t], axis=1).min(), ledger["min_past_gt_dist"][index], atol=1e-10)
                scalar += 1
            ledgers[arm] = ledger
            rows += len(q); prefix += len(ki["prefix"])
            for source in (p, kp, p.with_suffix(".json"), kp.with_suffix(".json")):
                sources[str(source.relative_to(REPO))] = sha256(source)
        changes[sequence] = {}
        for updates in (720, 900):
            natural, balanced = ledgers[f"natural_{updates}"], ledgers[f"balanced_{updates}"]
            c0, c1 = natural["top1_geo_dist"] <= 5, balanced["top1_geo_dist"] <= 5
            changes[sequence][str(updates)] = {
                "winner_changes": int(np.sum(natural["j_star"] != balanced["j_star"])),
                "correct_gained": int(np.sum(c1 & ~c0)), "correct_lost": int(np.sum(c0 & ~c1))}
        print("CHECKED", sequence, flush=True)
    for arm, methods in summary["per_sequence"].items():
        for method, per_sequence in methods.items():
            assert align.accept.macro(per_sequence) == summary["aggregate"][arm][method]
    write_json(run.OUT / "winner_changes.json", changes)
    assert (rows, metrics, scalar, prefix) == (396944, 72, 144, 72)
    return {"validation_rows": rows, "metric_comparisons": metrics, "scalar_retrieval_score_checks": scalar,
            "gpu_prefix_checks": prefix, "max_metric_error": max_error, "sources": sources}


def plot(summary: dict) -> None:
    """같은 학습 과정의 720·900회 최종 성능을 비교한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = "/tmp/nsd_context_accept_font.ttf"
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, (method, title) in enumerate((("baseline", "기존 시간 판정"), ("context_score", "고정 문맥 가중합"))):
        for row, metric in enumerate(("AP", "R@100P")):
            ax = axes[row, col]
            values = [[100*summary["aggregate"][f"{a}_{n}"][method]["macro"][metric] for n in (720, 900)]
                      for a in run.ARMS]
            for i, (name, color) in enumerate((("기존 학습", "#426b98"), ("계열 비중 보정", "#b16430"))):
                ax.plot([720, 900], values[i], "o-", label=name, color=color)
                for j, value in enumerate(values[i]):
                    above = value >= values[1-i][j]
                    ax.annotate(f"{value:.3f}", ([720, 900][j], value), xytext=(0, 10 if above else -16),
                                textcoords="offset points", ha="center", color=color, fontsize=9)
            ax.set_xticks([720, 900], ["720회 · 16바퀴", "900회 · 20바퀴"])
            ax.set_title(title+" · "+metric+" (%)", pad=30)
            ax.margins(x=.2, y=.3); ax.grid(alpha=.2)
            ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.005), ncol=2, frameon=False, fontsize=8)
    fig.suptitle("같은 KISS 학습 쌍 · 최종 검증 9 · 시드 1")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(run.OUT / ("family_balance."+suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """동결 소스와 독립 검산 결과를 완료 근거로 보존한다."""
    spec, summary = run.protocol(), run.read(run.OUT / "summary.json")
    assert sha256(run.OUT / "protocol.json") == summary["protocol_sha256"]
    sources, results = dict(spec["sources"]), {}
    for name, result in (("training", verify_training(spec)), ("diagnostics", verify_diagnostics()),
                         ("validation", verify_validation(spec, summary))):
        sources.update(result.pop("sources"))
        results[name] = result
    previous = run.read(run.parent.OUT / "summary.json")["aggregate"]["kiss_large"]
    replay = {
        method: {
            "macro_difference_pp": {metric: 100*(summary["aggregate"]["natural_900"][method]["macro"][metric]-value)
                                    for metric, value in previous[method]["macro"].items()},
            "family_difference_pp": {
                family: {metric: 100*(summary["aggregate"]["natural_900"][method]["families"][family][metric]-value)
                         for metric, value in values.items()}
                for family, values in previous[method]["families"].items()},
        } for method in ("baseline", "context_score")
    }
    write_json(run.OUT / "replay_final_difference.json", replay)
    plot(summary)
    for p in (Path(__file__), run.OUT/"summary.json", run.OUT/"diagnostic_summary.json",
              run.OUT/"winner_changes.json", run.OUT/"family_balance.png", run.OUT/"family_balance.pdf",
              run.OUT/"replay_final_difference.json", run.OUT/"control_replay_detail.json",
              run.OUT/"pooled_diagnostics.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    write_json(run.OUT / "completion_validation.json", {"status": "passed", **results, "sources": sources, "test_used": False})
    print("VERIFIED", results, len(sources), flush=True)


if __name__ == "__main__":
    main()
