#!/usr/bin/env python3
"""복소 전달의 입력·최종 원장·정렬 기여와 방향별 손익을 검산한다."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_transport as trial
trial.configure()
run = trial.base
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_temporal_test_control_finish import scalar_score
from _pdv2_no_context_ablation import sha256, write_json
from _train_attention_neighbors import array_hash
from _pr_eval_lib import classify_queries, operating_point


def aggregate(rows: dict) -> dict:
    """생산 집계기와 별도로 시퀀스→계열 평균을 계산한다."""
    families = {}
    for family in ("kitti", "helipr", "mulran"):
        selected = [v for s, v in rows.items() if s.startswith(family) and v["n_pos"]]
        families[family] = {m: sum(v[m] for v in selected)/len(selected)
                            for m in run.exp.METRICS}
    return {"families": families,
            "macro": {m: sum(v[m] for v in families.values())/3 for m in run.exp.METRICS}}


def check_training_plans() -> dict:
    """학습 그래프의 회전 연산을 직접 상대 위상 식으로 검산한다."""
    from gnn.spectral_transport import spectral_mean
    g = torch.load(run.OUT / "training_graph.pt", map_location="cpu", weights_only=False)
    z, phase = g.spectrum.numpy(), g.transport_phase.numpy()
    index, mask = g.transport_indices.numpy(), g.transport_mask.numpy()
    with np.load(run.SAMPLES / "training_inputs.npz") as raw:
        sequences = raw["sequences"]
    source, target = g.edge_index.numpy()
    assert np.all(sequences[source] == sequences[target])
    assert np.all(abs(source-target) <= 5)
    assert int(mask.sum()) == len(source)
    np.testing.assert_allclose(abs(phase), 1., atol=1e-7)
    # 72,306개 모든 노드의 실제 입력을 사용한다.
    measured = spectral_mean(g.spectrum, g.transport_indices, g.transport_mask,
                             g.transport_phase, "transported").numpy()
    expected = np.zeros_like(z, dtype=np.complex128)
    for slot in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10):
        ids = index[:, slot]
        factor = phase.astype(np.complex128) * phase[ids].astype(np.complex128).conj()
        expected += z[ids] * factor[:, None] * mask[:, slot, None, None]
    expected /= mask.sum(1).clip(1)[:, None, None]
    error = abs(measured-expected)
    np.testing.assert_allclose(measured, expected, atol=3e-5, rtol=4e-5)
    rng = np.random.default_rng(39)
    change = np.exp(1j*rng.uniform(-np.pi, np.pi, len(z))[:, None]*np.arange(1, 9))
    moved = spectral_mean(torch.from_numpy((z*change[:, None]).astype(np.complex64)),
        g.transport_indices, g.transport_mask,
        torch.from_numpy((phase*change).astype(np.complex64)), "transported").numpy()
    np.testing.assert_allclose(moved, measured*change[:, None], atol=4e-5, rtol=5e-5)
    return {"nodes": len(z), "edges": len(source), "support_within_five": True,
            "explicit_relative_rotation_max_error": float(error.max()),
            "local_gauge_max_error": float(abs(moved-measured*change[:, None]).max())}


def main() -> None:
    """원장 전체의 지표와 표본 검색을 재계산하고 최종 표·그림을 만든다."""
    spec = run.protocol(True)
    summary = run.read(run.OUT / "summary.json")
    assert summary["protocol_sha256"] == sha256(run.OUT / "protocol.json")
    assert run.read(run.OUT / "status.json")["status"] in ("evaluation_complete", "complete")
    sources = {**spec["runtime"], **spec["inputs"]}
    plan_checks = check_training_plans()
    for p in (Path(__file__), REPO / "scripts/_pdv2_context_tail_finish.py",
              REPO / "scripts/_pdv2_temporal_test_control_finish.py",
              run.OUT / "protocol.json", run.OUT / "summary.json",
              run.OUT / "training_geometry.json", run.OUT / "amp_smoke.json"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    ledgers = rows_count = metric_sets = scalar_checks = prefixes = 0
    max_metric_error = max_scalar_error = max_prefix_error = 0.
    all_rows, diagnostics, common_counts, reference_replay = {}, {}, {}, {}
    for seed, models in summary["per_seed"].items():
        s = int(seed)
        initial = {}
        reference_orders = None
        for arm in models:
            folder = run.root(s, arm)
            initial[arm] = run.read(folder / "initial_model.json")
            complete = run.read(folder / "training_complete.json")
            progress = run.read(folder / "training_progress.json")["records"]
            assert complete["updates"] == 900 and len(progress) == 20
            assert [v["updates"] for v in progress] == list(range(45, 901, 45))
            assert all(np.isfinite(v["loss"]) for v in progress)
            if reference_orders is None:
                reference_orders = complete["orders"]
            assert complete["orders"] == reference_orders
            for name, expected in complete["checkpoints"].items():
                p = folder / "checkpoints" / name
                assert sha256(p) == expected
                sources[str(p.relative_to(REPO))] = expected
                checkpoint = torch.load(p, map_location="cpu", weights_only=False)
                assert all(torch.isfinite(v).all() for v in checkpoint["model_state_dict"].values())
                updates = (checkpoint["epoch"]+1)*45
                assert all(int(v["step"].item()) == updates
                           for v in checkpoint["optimizer_state_dict"]["state"].values())
            for name in ("initial_model.json", "training_complete.json", "training_progress.json"):
                p = folder / name
                sources[str(p.relative_to(REPO))] = sha256(p)
        common = set.intersection(*(set(v) for v in initial.values()))
        assert all(len({v[k] for v in initial.values()}) == 1 for k in common)
        common_counts[seed] = len(common)
        assert initial["spectral_local"] == initial["untransported"] == initial["transported"]
        old_initial = run.read(run.exp.old.OUT / f"s{s}/standard_original/initial_model.json")
        assert initial["standard"] == old_initial["tensor_sha256"]
        reference_replay[seed] = {"initial_exact": True, "checkpoints": {}}
        for epoch in run.EPOCHS:
            old_path = run.exp.old.checkpoint_dir(s, "standard_original") / f"checkpoint_epoch_{epoch}.pth"
            new_path = run.root(s, "standard") / "checkpoints" / old_path.name
            old_state = torch.load(old_path, map_location="cpu", weights_only=False)["model_state_dict"]
            new_state = torch.load(new_path, map_location="cpu", weights_only=False)["model_state_dict"]
            assert set(old_state) == set(new_state)
            reference_replay[seed]["checkpoints"][str(epoch)] = {
                "all_exact": all(torch.equal(old_state[k], new_state[k]) for k in old_state),
                "max_parameter_or_buffer_difference": max(float((old_state[k].double()-new_state[k].double()).abs().max())
                                                          for k in old_state)}
            sources[str(old_path.relative_to(REPO))] = sha256(old_path)

        all_rows[seed], diagnostics[seed] = {}, {}
        tau = run.fixed.choose_tau(s)["tau"]
        for arm, saved in models.items():
            collected = {}
            for epoch in run.EPOCHS:
                metrics = {m: {} for m in ("raw", run.METHOD)}
                records = {}
                for sequence in spec["val"]:
                    sensor, seq, paths, _ = run.exp.old.base.inputs("val", sequence)
                    cache = run.exp.arrays(paths["cache"])
                    path = run.root(s, arm) / f"epoch{epoch}" / (sequence.replace("/", "_")+".npz")
                    info = run.read(path.with_suffix(".json"))
                    key_path = path.with_name("keys_"+path.name)
                    assert sha256(path) == info["sha256"]
                    assert sha256(key_path) == info["keys_sha256"]
                    for p in (path, key_path, path.with_suffix(".json")):
                        sources[str(p.relative_to(REPO))] = sha256(p)
                    a = run.exp.arrays(path)
                    keys = run.exp.arrays(key_path)
                    q, j, key = a["query_idx"], a["j_star"], keys["embedding"]
                    np.testing.assert_array_equal(q, np.arange(30, len(key)-15))
                    np.testing.assert_array_equal(keys["scan_ids"], cache["scan_ids"])
                    reference = run.exp.arrays(run.fixed.VAL / f"s{s}/standard_{sensor}_{seq}.npz")
                    for field in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(a[field], reference[field])
                    xyz = cache["poses_gt"][:, :3, 3]
                    np.testing.assert_allclose(np.linalg.norm(xyz[q]-xyz[j], axis=1),
                                               a["top1_geo_dist"], atol=1e-10, rtol=0)
                    assert (j <= q-30).all() and (j >= 0).all()
                    assert key.shape == (len(xyz), 416) and np.isfinite(key).all()
                    np.testing.assert_allclose(np.linalg.norm(key, axis=1), 1., atol=2e-6)
                    np.testing.assert_allclose(run.normalize(key[:, :288]),
                        run.normalize(cache["descriptors"]*np.load(run.exp.old.base.RVEC)),
                        atol=3e-7, rtol=3e-5)
                    for method in metrics:
                        assert np.isfinite(a[method]).all()
                        measured = independent_metrics(a, a[method])
                        for metric, value in measured.items():
                            expected = info["metrics"][method][metric]
                            if value is None:
                                assert expected is None
                            else:
                                max_metric_error = max(max_metric_error, abs(value-expected))
                                np.testing.assert_allclose(value, expected, atol=5e-13, rtol=0)
                        metrics[method][sequence] = measured
                        metric_sets += 1
                    with np.load(paths["spec"]) as z:
                        bank = run.exp.old.base.bank_for(run.exp.OP, z)
                    positions = cache["poses"][:, :3, 3]
                    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
                    conf = spec["confidence"][seed]
                    for ix in (0, len(q)//3, 2*len(q)//3, len(q)-1):
                        t = int(q[ix])
                        ids = np.unique(np.r_[run.prefix_topk(key, t, t-29, 100),
                                              run.prefix_topk(bank[0], t, t-29, 100)])
                        channels = (key[ids] @ key[t], bank[0][ids] @ bank[0][t],
                                    run.align_scores(bank[1], bank[2], t, ids),
                                    run.column_align_scores(bank[3], t, ids))
                        fused = sum(float(w)*run.minmax(v) for w, v in zip(conf["tuple"], channels))
                        winner = int(ids[np.argsort(-fused, kind="stable")[0]])
                        assert winner == int(j[ix])
                        channel_index = 2 if conf["decision"] == "c_spec" else 3
                        raw = float(channels[channel_index][np.flatnonzero(ids == winner)[0]])
                        np.testing.assert_allclose(raw, a["raw"][ix], atol=1e-12, rtol=0)
                        score = scalar_score(bank, travel, t, winner, conf["decision"], raw)
                        max_scalar_error = max(max_scalar_error, abs(score-a[run.METHOD][ix]))
                        np.testing.assert_allclose(score, a[run.METHOD][ix], atol=1e-12, rtol=0)
                        scalar_checks += 1
                    for check in info["prefixes"]:
                        assert check["winner_equal"] and check["score_error"] == 0
                        max_prefix_error = max(max_prefix_error, check["key_max_error"])
                        prefixes += 1
                    if epoch == saved["selected"]["epoch"]:
                        records[sequence] = a
                        pos = a["min_past_gt_dist"] <= 5
                        correct = pos & (a["top1_geo_dist"] <= 5)
                        alpha = np.linalg.norm(key[:, 288:], axis=1)/np.maximum(np.linalg.norm(key[:, :288], axis=1), 1e-12)
                        diagnostics[seed].setdefault(arm, {})[sequence] = {
                            "directions": {str(g): {"n": int((pos & (a["group"] == g)).sum()),
                                "correct": int((correct & (a["group"] == g)).sum())} for g in (0, 1, 2)},
                            "fixed_tau": operating_point(a[run.METHOD], a["top1_geo_dist"],
                                                         classify_queries(a["min_past_gt_dist"]), tau),
                            "gate_quantiles": np.quantile(alpha, [0, .1, .5, .9, 1]).tolist(),
                            "valid_fraction": info["valid_fraction"],
                            "negative_false_accepts": int(((a["min_past_gt_dist"] > 25) & (a[run.METHOD] >= tau)).sum()),
                        }
                    ledgers += 1
                    rows_count += len(q)
                measured_aggregate = {m: aggregate(v) for m, v in metrics.items()}
                old = next(r for r in saved["candidates"] if r["epoch"] == epoch)
                for m in metrics:
                    for key in ("macro",):
                        for metric in run.exp.METRICS:
                            np.testing.assert_allclose(measured_aggregate[m][key][metric],
                                                       old[m][key][metric], atol=5e-13)
                    for family in ("kitti", "helipr", "mulran"):
                        for metric in run.exp.METRICS:
                            np.testing.assert_allclose(measured_aggregate[m]["families"][family][metric],
                                                       old[m]["families"][family][metric], atol=5e-13)
                collected[epoch] = measured_aggregate
                if records:
                    all_rows[seed][arm] = records
            chosen = max(collected, key=lambda e: (round(collected[e][run.METHOD]["macro"]["AP"], 12), -e))
            assert chosen == saved["selected"]["epoch"]

    transitions = {}
    for seed, models in all_rows.items():
        transitions[seed] = {}
        for reference in models:
            if reference == "transported":
                continue
            counts = {"gained": 0, "lost": 0, "winner_changed": 0}
            for seq, b in models["transported"].items():
                a = models[reference][seq]
                pos = a["min_past_gt_dist"] <= 5
                ca, cb = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
                counts["gained"] += int((pos & ~ca & cb).sum())
                counts["lost"] += int((pos & ca & ~cb).sum())
                counts["winner_changed"] += int((a["j_star"] != b["j_star"]).sum())
            transitions[seed][reference] = counts
    write_json(run.OUT / "diagnostics.json", {
        "per_sequence": diagnostics, "transitions_to_transported": transitions,
        "original_standard_replay": reference_replay,
        "fixed_tau_source": "기존 standard 모델의 검증 F1 최대 임계값을 모든 조건에 공통 적용",
        "test_used": False,
    })
    make_artifacts(summary)
    for p in (run.OUT / "diagnostics.json", run.OUT / "final_metrics.csv",
              run.OUT / "spectral_transport.png", run.OUT / "spectral_transport.pdf"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    for p, h in sources.items():
        assert sha256(REPO / p) == h
    write_json(run.OUT / "completion_validation.json", {
        "status": "passed", "ledgers": ledgers, "rows": rows_count,
        "metric_sets": metric_sets, "metric_max_error": max_metric_error,
        "scalar_retrieval_and_scores": scalar_checks, "scalar_max_error": max_scalar_error,
        "prefix_checks": prefixes, "prefix_key_max_error": max_prefix_error,
        "prefix_winner_changes": 0, "common_initial_tensors": common_counts,
        "transport_graph_checks": plan_checks,
        "same_triplets_orders_and_900_updates": True, "sources": sources, "test_used": False,
    })
    print("VERIFIED", ledgers, rows_count, metric_sets, scalar_checks, prefixes, flush=True)


def make_artifacts(summary: dict) -> None:
    """최종 판정 지표와 계열별 효과를 CSV·그림으로 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    rows = []
    for seed, models in summary["per_seed"].items():
        for arm, value in models.items():
            selected = value["selected"]
            rows.append({"seed": int(seed), "arm": arm, "epoch": selected["epoch"],
                         **{m: 100*selected[run.METHOD]["macro"][m] for m in run.exp.METRICS}})
    with (run.OUT / "final_metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fp = FontProperties(fname=str(REPO / "results/pdv2_sequence_report_20261018/UnDotum.ttf"))
    labels = {"standard": "표준 attention", "point": "간선 없는 투영",
              "spectral_local": "단일 복소 특징", "untransported": "무보정 복소 전달",
              "transported": "회전 보정 복소 전달"}
    arms = list(summary["per_seed"]["1"])
    selected = {a: summary["per_seed"]["1"][a]["selected"][run.METHOD] for a in arms}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    colors = ["#6a7f93", "#98aabd", "#b2bbc4", "#deac64", "#bd663b"]
    for ax, metric in zip(axes[:2], ("AP", "R@100P")):
        values = [100*selected[a]["macro"][metric] for a in arms]
        bars = ax.bar(np.arange(len(arms)), values, color=colors)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
        ax.set_ylim(min(values)-max(1., np.ptp(values)*.4), max(values)+max(1., np.ptp(values)*.3))
        ax.set_xticks(np.arange(len(arms)), [labels[a] for a in arms], rotation=30,
                      ha="right", fontproperties=fp)
        ax.set_title("최종 "+metric+" (%)", fontproperties=fp)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    families = ("kitti", "helipr", "mulran")
    delta = [100*(selected["transported"]["families"][f]["AP"]-selected["standard"]["families"][f]["AP"])
             for f in families]
    bars = axes[2].bar(families, delta, color=["#42836b" if v >= 0 else "#bd663b" for v in delta])
    axes[2].bar_label(bars, fmt="%+.3f", padding=3)
    axes[2].axhline(0, color="black", lw=.8)
    axes[2].set_title("복소 전달 - 표준 AP (%p)", fontproperties=fp)
    axes[2].margins(y=.25)
    fig.suptitle("검증 9시퀀스 · 시드 1 · 같은 최종 시간 판정", fontproperties=fp, fontsize=14)
    fig.tight_layout()
    fig.savefig(run.OUT / "spectral_transport.png", dpi=180)
    fig.savefig(run.OUT / "spectral_transport.pdf")
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
