#!/usr/bin/env python3
"""시간 간선 대조의 최종 시험 지표·검색·판정을 독립 검산한다."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_temporal_test_control as run
from _causal_pr_dump import prefix_topk, minmax, align_scores, column_align_scores
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_no_context_ablation import sha256, write_json
from _pr_eval_lib import classify_queries, operating_point


def scalar_score(bank: tuple, travel: np.ndarray, q: int, j: int,
                 channel: str, raw: float) -> float:
    """거리 창을 직접 훑어 중복 후보 가중치의 최종 점수를 검산한다."""
    hkey, hfft, hnorm, column = bank
    def peak(a: int, b: int) -> float:
        if channel == "c_spec":
            return float(align_scores(hfft, hnorm, a, np.asarray([b]))[0])
        return float(column_align_scores(column, a, np.asarray([b]))[0])
    center = peak(q, j)
    means = []
    candidates = np.arange(max(0, j - 10), min(len(travel), j + 11))
    for direction in (1, -1):
        pairs = []
        for offset in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
            qn = q + offset
            if not 0 <= qn < len(travel):
                continue
            target = travel[j] + direction * (travel[qn] - travel[q])
            distance = np.abs(travel[candidates] - target)
            closest = candidates[distance == distance.min()]
            # 양쪽 동률은 아래 거리, 정지 구간의 아래 거리는 마지막 번호다.
            below = closest[travel[closest] < target]
            jn = int(below[-1] if len(below) else closest[0])
            if jn != j:
                pairs.append((qn, jn))
        total, weight = center, 1.
        for qn, jn in pairs:
            w = 1. / sum(b == jn for _, b in pairs)
            total += w * peak(qn, jn)
            weight += w
        means.append(total / weight)
    beta = (2. / 3.) * (11. / 10.)
    return (1 - beta) * raw + beta * max(means)


def verify() -> dict:
    """보존 원장과 새 원장을 같은 규칙으로 확인한다."""
    spec = run.freeze(True)
    summary = run.exp.read(run.OUT / "summary.json")
    assert summary["freeze_sha256"] == sha256(run.OUT / "freeze.json")
    sources = {**spec["runtime_sha256"], **spec["sources_sha256"]}
    checked_metrics = rows = scalar_checks = prefixes = prefix_winner_changes = 0
    maximum_metric_error = maximum_scalar_error = maximum_key_error = 0.
    fixed_counts = {model: {} for model in ("standard", "capacity_point")}
    for sensor, seq in run.fixed.TEST:
        sequence = sensor + "/" + seq
        _, _, paths, _ = run.exp.old.base.inputs("test", sequence)
        cache = run.exp.arrays(paths["cache"])
        xyz = cache["poses_gt"][:, :3, 3]
        pose = cache["poses"][:, :3, 3]
        travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pose, axis=0), axis=1))]
        with np.load(paths["spec"]) as z:
            bank = run.exp.old.base.bank_for(run.exp.OP, z)
        for model in ("standard", "capacity_point"):
            folder = run.fixed.OUT / "test" if model == "standard" else run.OUT / "test" / model
            info_path = folder / f"{sensor}_{seq}.json"
            record = run.exp.read(info_path)
            sources[str(info_path.relative_to(REPO))] = sha256(info_path)
            if model == "capacity_point":
                assert record["freeze_sha256"] == summary["freeze_sha256"]
            for seed in ("1", "2", "3"):
                meta = record["results"][seed]
                path = REPO / meta["ledger"]
                assert sha256(path) == meta["sha256"]
                sources[str(path.relative_to(REPO))] = meta["sha256"]
                a = run.exp.arrays(path)
                q, j = a["query_idx"], a["j_star"]
                np.testing.assert_array_equal(q, np.arange(30, len(xyz) - 15))
                assert (j <= q - 30).all() and (j >= 0).all()
                np.testing.assert_allclose(np.linalg.norm(xyz[q] - xyz[j], axis=1),
                                           a["top1_geo_dist"], atol=1e-10, rtol=0)
                rows += len(q)
                for method in run.SCORES:
                    assert np.isfinite(a[method]).all()
                    measured = independent_metrics(a, a[method])
                    expected = summary["per_sequence"][model][seed][sequence][method]
                    assert expected == meta["metrics"][method]
                    for metric, value in measured.items():
                        if value is None:
                            assert expected[metric] is None
                        else:
                            maximum_metric_error = max(maximum_metric_error, abs(value - expected[metric]))
                            np.testing.assert_allclose(value, expected[metric], atol=5e-13, rtol=0)
                    checked_metrics += 1
                tau = spec["models"][model][seed]["new_tau"]["tau"]
                point = operating_point(a[run.METHOD], a["top1_geo_dist"],
                                        classify_queries(a["min_past_gt_dist"]), tau)
                assert point == meta["fixed_tau"]
                fixed_counts[model].setdefault(seed, {})[sequence] = point
                if model != "capacity_point":
                    continue
                assert meta["graph"]["edges"] == 0 and meta["graph"]["similarity"] == 0
                assert meta["checkpoint_sha256"] == spec["models"][model][seed]["checkpoint_sha256"]
                kp = REPO / meta["keys"]
                assert sha256(kp) == meta["keys_sha256"]
                sources[str(kp.relative_to(REPO))] = meta["keys_sha256"]
                keydata = run.exp.arrays(kp)
                key = keydata["embedding"]
                np.testing.assert_array_equal(keydata["scan_ids"], cache["scan_ids"])
                assert key.shape == (len(xyz), 416) and np.isfinite(key).all()
                np.testing.assert_allclose(np.linalg.norm(key, axis=1), 1., atol=2e-6, rtol=0)
                conf = spec["models"][model][seed]["confidence"]
                for index in (0, len(q) // 3, 2 * len(q) // 3, len(q) - 1):
                    t = int(q[index])
                    hkey, hfft, hnorm, column = bank
                    candidates = np.unique(np.r_[prefix_topk(key, t, t - 29, 100),
                                                 prefix_topk(hkey, t, t - 29, 100)])
                    channels = (key[candidates] @ key[t], hkey[candidates] @ hkey[t],
                                align_scores(hfft, hnorm, t, candidates),
                                column_align_scores(column, t, candidates))
                    fused = sum(float(w) * minmax(v) for w, v in zip(conf["tuple"], channels))
                    best = int(np.argsort(-fused, kind="stable")[0])
                    assert int(candidates[best]) == int(j[index])
                    raw = float(channels[2 if conf["decision"] == "c_spec" else 3][best])
                    np.testing.assert_allclose(raw, a["raw"][index], atol=2e-7, rtol=0)
                    score = scalar_score(bank, travel, t, int(j[index]), conf["decision"], raw)
                    maximum_scalar_error = max(maximum_scalar_error, abs(score - a[run.METHOD][index]))
                    np.testing.assert_allclose(score, a[run.METHOD][index], atol=3e-7, rtol=0)
                    np.testing.assert_allclose(np.linalg.norm(xyz[:t-29] - xyz[t], axis=1).min(),
                                               a["min_past_gt_dist"][index], atol=1e-10, rtol=0)
                    scalar_checks += 1
                prefixes += len(meta["prefix_checks"])
                for probe in meta["prefix_checks"]:
                    assert probe["score_max_error"] == 0.
                    maximum_key_error = max(maximum_key_error, probe["key_max_error"])
                    assert probe["winner_equal"] == (probe["prefix_winner"] == probe["full_winner"])
                    prefix_winner_changes += int(not probe["winner_equal"])
        print("VERIFIED", sequence, flush=True)
    assert rows == 195726 * 6 and checked_metrics == 46 * 3 * 2 * 2
    # 도시 집계와 같은 시드 차이를 별도 행 목록에서 계산한다.
    for model in ("standard", "capacity_point"):
        for method in run.SCORES:
            for metric in run.exp.METRICS:
                values = []
                for seed in ("1", "2", "3"):
                    cities = [
                        np.mean([summary["per_sequence"][model][seed][seq][method][metric]
                                 for seq in members
                                 if summary["per_sequence"][model][seed][seq][method]["n_pos"]])
                        for members in run.fixed.CITY.values()]
                    values.append(float(np.mean(cities)))
                np.testing.assert_allclose([np.mean(values), np.std(values, ddof=1)],
                    [summary["mean_std"][model][method][metric]["mean"],
                     summary["mean_std"][model][method][metric]["std"]], atol=1e-14, rtol=0)
    write_json(run.OUT / "fixed_threshold_counts.json", fixed_counts)
    return {"status": "passed", "ledgers": 276, "new_ledgers": 138, "rows": rows,
            "metric_sets": checked_metrics, "metric_max_error": maximum_metric_error,
            "scalar_retrieval_and_score_checks": scalar_checks, "scalar_score_max_error": maximum_scalar_error,
            "gpu_prefix_checks": prefixes, "gpu_key_max_error": maximum_key_error,
            "gpu_prefix_winner_changes": prefix_winner_changes,
            "same_query_and_city_protocol": True, "sources": sources, "test_used": True}


def artifacts(summary: dict) -> None:
    """최종 대조 표와 도시별 손익 그림을 만든다."""
    with (run.OUT / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric", "temporal_mean_percent", "point_mean_percent",
                         "delta_mean_pp", "delta_seed1_pp", "delta_seed2_pp", "delta_seed3_pp"])
        for method in run.SCORES:
            for metric in run.exp.METRICS:
                delta = summary["temporal_minus_point_pp"][method][metric]
                writer.writerow([method, metric,
                    summary["mean_std"]["standard"][method][metric]["mean"] * 100,
                    summary["mean_std"]["capacity_point"][method][metric]["mean"] * 100,
                    float(np.mean(delta)), *delta])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(REPO / "results/pdv2_sequence_report_20261018/UnDotum.ttf"))
    plt.rcParams.update({"font.family": "UnDotum", "axes.unicode_minus": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, metric in zip(axes[:2], ("AP", "R@100P")):
        vals = [summary["mean_std"][m][run.METHOD][metric] for m in ("capacity_point", "standard")]
        for i, (row, color) in enumerate(zip(vals, ("#888888", "#316ba8"))):
            ax.errorbar(i, row["mean"] * 100, yerr=row["std"] * 100,
                        fmt="o", color=color, capsize=5)
            ax.annotate(f'{row["mean"]*100:.3f}', (i, (row["mean"]+row["std"])*100),
                        xytext=(0, 10), textcoords="offset points", ha="center")
        ax.set_xticks([0, 1], ["시간 간선 없음", "시간 간선 사용"])
        ax.set_xlim(-.5, 1.5)
        ax.margins(y=.4)
        ax.set_title(f"최종 {metric} (%)")
        ax.grid(axis="y", alpha=.2)
    cities = summary["city_temporal_minus_point_pp"]
    names = {"karlsruhe": "카를스루에", "seoul": "서울", "daejeon": "대전",
             "ann_arbor": "앤아버", "stockholm": "스톡홀름", "hamburg": "함부르크"}
    labels = [names[c] for c in cities]
    vals = [cities[c]["AP"] for c in cities]
    bars = axes[2].barh(labels, vals, color=["#316ba8" if v >= 0 else "#be703c" for v in vals])
    axes[2].bar_label(bars, fmt="%+.3f", padding=4)
    axes[2].axvline(0, color="black", linewidth=.7)
    axes[2].margins(x=.35)
    axes[2].set_title("도시별 AP 차이 (%p)\n시간 간선 사용 − 없음")
    axes[2].grid(axis="x", alpha=.2)
    fig.suptitle("시험 46시퀀스 · 같은 최종 판정 · 3시드 평균±표본 표준편차")
    fig.tight_layout(rect=(0, 0, 1, .92))
    for ext in ("png", "pdf"):
        fig.savefig(run.OUT / f"temporal_test_control.{ext}", dpi=180)
    plt.close(fig)


def main() -> None:
    """검산을 끝내고 결과와 도표의 해시를 보존한다."""
    result = verify()
    summary = run.exp.read(run.OUT / "summary.json")
    artifacts(summary)
    files = [Path(__file__), run.OUT / "summary.json", run.OUT / "comparison.csv",
             run.OUT / "fixed_threshold_counts.json", run.OUT / "validation_replay.json",
             run.OUT / "temporal_test_control.png", run.OUT / "temporal_test_control.pdf",
             REPO / "scripts/_pdv2_context_tail_finish.py"]
    for p in files:
        result["sources"][str(p.relative_to(REPO))] = sha256(p)
    write_json(run.OUT / "completion_validation.json", result)
    print("COMPLETE", {k: v for k, v in result.items() if k != "sources"},
          "HASHES", len(result["sources"]), flush=True)


if __name__ == "__main__":
    main()
