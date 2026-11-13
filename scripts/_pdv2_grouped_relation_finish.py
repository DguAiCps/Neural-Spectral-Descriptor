#!/usr/bin/env python3
"""묶음 attention 대조의 최종 원장·기준 재현·학습 순서·방향별 손익을 검산한다."""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_grouped_relation as run
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_temporal_test_control_finish import scalar_score
from _pdv2_no_context_ablation import sha256, write_json


def aggregate(rows: dict) -> dict:
    """독립적으로 양성 있는 시퀀스→세 계열 평균을 계산한다."""
    families = {}
    for family in ("kitti", "helipr", "mulran"):
        chosen = [v for s, v in rows.items() if s.startswith(family) and v["n_pos"]]
        families[family] = {m: sum(v[m] for v in chosen) / len(chosen)
                            for m in run.base.exp.METRICS}
    return {"families": families, "macro": {
        m: sum(v[m] for v in families.values()) / 3 for m in run.base.exp.METRICS}}


def counts(row: dict, coverage: dict, tau: float) -> dict:
    """최종 정답·후보 누락·공통 임계값의 수락 수를 계산한다."""
    positive = row["min_past_gt_dist"] <= 5
    correct = positive & (row["top1_geo_dist"] <= 5)
    accepted = row[run.METHOD] >= tau
    hit = coverage["union_hit"]
    assert not np.any(correct & ~hit)
    result = {
        "queries": len(positive), "positive": int(positive.sum()),
        "correct": int(correct.sum()), "all_positives_absent": int((positive & ~hit).sum()),
        "wrong_with_positive_in_pool": int((positive & hit & ~correct).sum()),
        "true_accept": int((correct & accepted).sum()),
        "wrong_positive_accept": int((positive & ~correct & accepted).sum()),
        "negative_false_accept": int(((row["min_past_gt_dist"] > 25) & accepted).sum()),
    }
    for group in (0, 1, 2):
        selected = positive & (row["group"] == group)
        result[f"group{group}_positive"] = int(selected.sum())
        result[f"group{group}_correct"] = int((selected & correct).sum())
    return result


def compare(a: dict, b: dict) -> dict:
    """같은 질의의 정답 회복·손실을 계열과 방향별로 나눈다."""
    result = {}
    for seq in a:
        x, y = a[seq], b[seq]
        pos = x["min_past_gt_dist"] <= 5
        old, new = x["top1_geo_dist"] <= 5, y["top1_geo_dist"] <= 5
        result[seq] = {"gained": int((pos & ~old & new).sum()),
                       "lost": int((pos & old & ~new).sum()),
                       "winner_changed": int((x["j_star"] != y["j_star"]).sum())}
        for group in (0, 1, 2):
            mask = pos & (x["group"] == group)
            result[seq][f"group{group}_gained"] = int((mask & ~old & new).sum())
            result[seq][f"group{group}_lost"] = int((mask & old & ~new).sum())
    return {"sequences": result, "total": {
        k: sum(r[k] for r in result.values()) for k in next(iter(result.values()))}}


def artifacts(summary: dict) -> None:
    """최종 판정 표와 내보낼 수 있는 비교 그림을 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = list(run.base.exp.METRICS)
    with (run.OUT / "final_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "selected_epoch", "aggregation", *metrics])
        for seed, models in summary["per_seed"].items():
            for arm, record in models.items():
                row = record["selected"]
                scores = row[run.METHOD]
                for family, values in {"macro": scores["macro"], **scores["families"]}.items():
                    writer.writerow([seed, arm, row["epoch"], family,
                                     *[100 * values[m] for m in metrics]])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), layout="constrained")
    selected = summary["per_seed"]["1"]
    modes = ("linear", "grouped")
    labels = ("32 channels / weight", "4 channels / weight")
    x = np.arange(2)
    for ax, metric in zip(axes, ("AP", "max_F1", "R@100P")):
        for relation, offset, color in (("diff", -.18, "#2166ac"),
                                         ("standard", .18, "#d6604d")):
            values = [100 * selected[f"{relation}_{mode}"]["selected"][run.METHOD]["macro"][metric]
                      for mode in modes]
            ax.plot(x + offset, values, "o", label=relation, color=color)
            for px, value in zip(x + offset, values):
                ax.annotate(f"{value:.3f}", (px, value), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=7)
        point = 100 * selected["point"]["selected"][run.METHOD]["macro"][metric]
        ax.axhline(point, color="gray", linestyle="--", linewidth=1, label="point control")
        ax.set_xticks(x, labels, rotation=15)
        ax.set_ylabel(metric + " (%)")
        all_values = [100 * r["selected"][run.METHOD]["macro"][metric] for r in selected.values()]
        lo, hi = min(all_values), max(all_values)
        margin = max(.08, (hi - lo) * .25)
        ax.set_ylim(lo - margin, hi + 2 * margin)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Final validation decisions | seed 1 | fixed retrieval and scoring")
    fig.savefig(run.OUT / "grouped_relation.png", dpi=180)
    fig.savefig(run.OUT / "grouped_relation.pdf")
    plt.close(fig)


def main() -> None:
    """독립 지표와 표본 검색 검산이 끝난 경우에만 완료를 기록한다."""
    spec, summary = run.protocol(True), run.read(run.OUT / "summary.json")
    status = run.read(run.OUT / "status.json")
    assert status["status"] in ("evaluation_complete", "complete")
    assert summary["protocol_sha256"] == sha256(run.OUT / "protocol.json")
    sources = {**spec["runtime"], **spec["inputs"]}
    for p in (Path(__file__), run.OUT / "summary.json", run.OUT / "protocol.json",
              run.OUT / "amp_smoke.json", run.OUT / "expansion_selection.json",
              REPO / "scripts/_pdv2_context_tail_finish.py",
              REPO / "scripts/_pdv2_temporal_test_control_finish.py"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    smoke = run.read(run.OUT / "amp_smoke.json")
    assert all(v["finite"] and v["shape"] == [72306, 416]
               and v["parameters_with_gradient"] == 33 for v in smoke.values())
    ledgers = rows_count = metric_sets = scalar_checks = prefixes = replays = 0
    max_metric_error = max_scalar_error = max_prefix_error = max_reference_key_error = 0.
    details, totals, transitions, common = {}, {}, {}, {}
    old_summary = run.read(run.previous.OUT / "summary.json")
    reused = {**run.CFG["reused_arms"], "point": "point"}
    for seed_text, models in summary["per_seed"].items():
        seed = int(seed_text)
        initial, orders, records = {}, None, {}
        details[seed_text], totals[seed_text], transitions[seed_text] = {}, {}, {}
        tau = run.base.fixed.choose_tau(seed)["tau"]
        for arm, saved in models.items():
            folder = run.base.root(seed, arm)
            initial[arm] = run.read(folder / "initial_model.json")
            complete = run.read(folder / "training_complete.json")
            progress = run.read(folder / "training_progress.json")["records"]
            assert complete["updates"] == 900 and len(progress) == 20
            assert [r["updates"] for r in progress] == list(range(45, 901, 45))
            assert all(np.isfinite(r["loss"]) for r in progress)
            if orders is None:
                orders = complete["orders"]
            assert orders == complete["orders"]
            for name in ("initial_model.json", "training_complete.json", "training_progress.json"):
                p = folder / name
                sources[str(p.relative_to(REPO))] = sha256(p)
            for name, expected in complete["checkpoints"].items():
                p = folder / "checkpoints" / name
                assert sha256(p) == expected
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
                assert all(torch.isfinite(v).all() for v in ckpt["model_state_dict"].values())
                assert all(int(v["step"].item()) == (ckpt["epoch"] + 1) * 45
                           for v in ckpt["optimizer_state_dict"]["state"].values())
                sources[str(p.relative_to(REPO))] = expected
            records[arm], details[seed_text][arm] = {}, {}
            collected = []
            for epoch in (10, 20):
                measured = {m: {} for m in ("raw", run.METHOD)}
                for seq in spec["val"]:
                    _, _, paths, _ = run.base.exp.old.base.inputs("val", seq)
                    cache = run.base.exp.arrays(paths["cache"])
                    p = folder / f"epoch{epoch}" / (seq.replace("/", "_") + ".npz")
                    info = run.read(p.with_suffix(".json"))
                    kp, cp = p.with_name("keys_" + p.name), p.with_name("coverage_" + p.name)
                    for path, expected in ((p, info["sha256"]), (kp, info["keys_sha256"]),
                                            (cp, run.read(cp.with_suffix(".json"))["sha256"])):
                        assert sha256(path) == expected
                        sources[str(path.relative_to(REPO))] = expected
                    for path in (p.with_suffix(".json"), cp.with_suffix(".json")):
                        sources[str(path.relative_to(REPO))] = sha256(path)
                    row, key_data, coverage = (run.base.exp.arrays(path) for path in (p, kp, cp))
                    q, j, key = row["query_idx"], row["j_star"], key_data["embedding"]
                    np.testing.assert_array_equal(q, np.arange(30, len(key) - 15))
                    np.testing.assert_array_equal(key_data["scan_ids"], cache["scan_ids"])
                    ref_path = run.base.root(seed, "diff_linear") / f"epoch{epoch}" / p.name
                    ref = run.base.exp.arrays(ref_path)
                    for field in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(row[field], ref[field])
                    for field in ("query_idx", "j_star", "top1_geo_dist", "min_past_gt_dist"):
                        np.testing.assert_array_equal(row[field], coverage[field])
                    np.testing.assert_array_equal(coverage["union_hit"], coverage["key_hit"] | coverage["aux_hit"])
                    assert np.all((0 <= j) & (j <= q - 30))
                    xyz = cache["poses_gt"][:, :3, 3]
                    np.testing.assert_allclose(np.linalg.norm(xyz[q] - xyz[j], axis=1),
                                               row["top1_geo_dist"], atol=1e-10, rtol=0)
                    assert key.shape == (len(xyz), 416) and np.isfinite(key).all()
                    np.testing.assert_allclose(np.linalg.norm(key, axis=1), 1., atol=2e-6)
                    for method in measured:
                        metrics = independent_metrics(row, row[method])
                        for name, value in metrics.items():
                            expected = info["metrics"][method][name]
                            if value is None:
                                assert expected is None
                            else:
                                max_metric_error = max(max_metric_error, abs(value - expected))
                                np.testing.assert_allclose(value, expected, atol=5e-13, rtol=0)
                        measured[method][seq] = metrics
                        metric_sets += 1
                    with np.load(paths["spec"]) as z:
                        bank = run.base.exp.old.base.bank_for(run.base.exp.OP, z)
                    pos = cache["poses"][:, :3, 3]
                    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
                    conf = spec["confidence"][seed_text]
                    for ix in (0, len(q) // 3, 2 * len(q) // 3, len(q) - 1):
                        t = int(q[ix])
                        ids = np.unique(np.r_[run.base.prefix_topk(key, t, t - 29, 100),
                            run.base.prefix_topk(bank[0], t, t - 29, 100)])
                        channels = (key[ids] @ key[t], bank[0][ids] @ bank[0][t],
                            run.base.align_scores(bank[1], bank[2], t, ids),
                            run.base.column_align_scores(bank[3], t, ids))
                        fused = sum(float(w) * run.base.minmax(v) for w, v in zip(conf["tuple"], channels))
                        winner = int(ids[np.argsort(-fused, kind="stable")[0]])
                        assert winner == int(j[ix])
                        channel = 2 if conf["decision"] == "c_spec" else 3
                        raw = float(channels[channel][np.flatnonzero(ids == winner)[0]])
                        np.testing.assert_allclose(raw, row["raw"][ix], atol=1e-12, rtol=0)
                        value = scalar_score(bank, travel, t, winner, conf["decision"], raw)
                        max_scalar_error = max(max_scalar_error, abs(value - row[run.METHOD][ix]))
                        np.testing.assert_allclose(value, row[run.METHOD][ix], atol=1e-12, rtol=0)
                        scalar_checks += 1
                    for check in info["prefixes"]:
                        assert check["winner_equal"] and check["score_error"] == 0
                        max_prefix_error = max(max_prefix_error, check["key_max_error"])
                        prefixes += 1
                    if seed == 1 and arm in reused:
                        old_folder = run.previous.OUT / "s1" / reused[arm] / f"epoch{epoch}"
                        old_row = run.base.exp.arrays(old_folder / p.name)
                        old_key = run.base.exp.arrays(old_folder / kp.name)["embedding"]
                        for field in row:
                            np.testing.assert_array_equal(row[field], old_row[field])
                        max_reference_key_error = max(max_reference_key_error, float(np.max(abs(key - old_key))))
                        np.testing.assert_allclose(key, old_key, atol=4e-6, rtol=4e-5)
                        replays += 1
                    if epoch == saved["selected"]["epoch"]:
                        records[arm][seq] = row
                        details[seed_text][arm][seq] = counts(row, coverage, tau)
                    ledgers += 1
                    rows_count += len(q)
                current = {"epoch": epoch, **{m: aggregate(rows) for m, rows in measured.items()}}
                expected = next(c for c in saved["candidates"] if c["epoch"] == epoch)
                for method in measured:
                    for group, values in {"macro": current[method]["macro"], **current[method]["families"]}.items():
                        ref = expected[method]["macro"] if group == "macro" else expected[method]["families"][group]
                        for metric, value in values.items():
                            np.testing.assert_allclose(value, ref[metric], atol=5e-13, rtol=0)
                collected.append(current)
            chosen = max(collected, key=lambda c: (round(c[run.METHOD]["macro"]["AP"], 12), -c["epoch"]))
            assert chosen["epoch"] == saved["selected"]["epoch"]
            if seed == 1 and arm in reused:
                assert saved == old_summary["per_seed"]["1"][reused[arm]]
            total = details[seed_text][arm]
            totals[seed_text][arm] = {k: sum(v[k] for v in total.values())
                                    for k in next(iter(total.values()))}
            assert totals[seed_text][arm]["queries"] == 99236
            assert totals[seed_text][arm]["positive"] == 10036
        keys = set.intersection(*(set(v) for v in initial.values()))
        assert all(len({v[k] for v in initial.values()}) == 1 for k in keys)
        attention_initial = [v for a, v in initial.items() if a != "point"]
        assert all(v == attention_initial[0] for v in attention_initial)
        common[seed_text] = {"all_models": len(keys), "attention_models": len(attention_initial[0])}
        for arm in run.CFG["candidate_arms"]:
            if arm not in records:
                continue
            transitions[seed_text][arm] = {}
            for ref in ("diff_linear", "standard_linear", arm.replace("diff_", "standard_", 1), "point"):
                transitions[seed_text][arm][ref] = compare(records[ref], records[arm])
    write_json(run.OUT / "diagnostics.json", {"totals": totals, "per_sequence": details,
        "transitions": transitions, "common_initial_tensors": common,
        "same_epoch_interaction_pp": summary["same_epoch_interaction_pp"], "test_used": False,
        "fixed_tau_source": "기존 standard의 시드별 검증 F1 최대 임계값. 새 선택에는 사용하지 않는다."})
    artifacts(summary)
    for name in ("diagnostics.json", "final_metrics.csv", "grouped_relation.png", "grouped_relation.pdf"):
        p = run.OUT / name
        sources[str(p.relative_to(REPO))] = sha256(p)
    for p, h in sources.items():
        assert sha256(REPO / p) == h, p
    validation = {"status": "passed", "ledgers": ledgers, "rows": rows_count,
        "metric_sets": metric_sets, "scalar_checks": scalar_checks, "prefix_checks": prefixes,
        "exact_reference_ledger_replays": replays, "max_reference_key_error": max_reference_key_error,
        "reference_key_tolerance": "기존 접두 재현과 같은 atol4e-6, rtol4e-5. 원장과 최종 지표는 완전 일치 요구.",
        "max_metric_error": max_metric_error,
        "max_scalar_error": max_scalar_error, "max_prefix_key_error": max_prefix_error,
        "source_count": len(sources), "sources": sources, "test_used": False}
    write_json(run.OUT / "completion_validation.json", validation)
    write_json(run.OUT / "status.json", {**status, "status": "complete", "audit_passed": True,
        "finalized_unix": time.time(), "eligible_candidates": summary["eligible_candidates"],
        "source_count": len(sources), "test_used": False})
    print("PASSED", {k: v for k, v in validation.items() if k != "sources"}, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
