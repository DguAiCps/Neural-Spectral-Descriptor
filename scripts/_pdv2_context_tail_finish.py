#!/usr/bin/env python3
"""높은 오답 점수 손실의 수렴·최종 지표를 별도로 검산하고 정리한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_context_tail as run
from _pdv2_context_accept_lib import labels
from _pdv2_context_place_lib import score
from _pdv2_no_context_ablation import sha256, write_json


def independent_metrics(row: dict, value: np.ndarray) -> dict:
    """동점 묶음과 전체 양성 분모를 별도 계산한다."""
    positive = row["min_past_gt_dist"] <= 5
    keep = positive | (row["min_past_gt_dist"] > 25)
    correct = positive & (row["top1_geo_dist"] <= 5)
    ids = np.flatnonzero(keep & np.isfinite(value))
    ids = ids[np.argsort(-value[ids], kind="stable")]
    end = np.flatnonzero(np.r_[value[ids][1:] != value[ids][:-1], True])
    tp = np.r_[0, np.cumsum(correct[ids])[end]]
    fp = np.r_[0, np.cumsum(~correct[ids])[end]]
    precision = np.ones(len(tp), dtype=float)
    np.divide(tp, tp + fp, out=precision, where=tp + fp > 0)
    recall = tp / max(int(positive.sum()), 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return {"AP": float(np.diff(recall) @ envelope[1:]),
            "max_F1": float(np.max(2 * precision * recall / np.maximum(precision + recall, 1e-300))),
            "R@100P": float(np.max(recall[fp == 0])), "n_pos": int(positive.sum()),
            "R@1": float(correct.sum() / positive.sum()) if positive.any() else None}


def check_fit(data: dict, fitted: dict) -> dict:
    """학습 통계·오답 선정·손실의 기울기를 독립적으로 계산한다."""
    names = fitted["training"]
    assert not set(names) & set(fitted["heldout"])
    families = {s.split("/")[0] for s in names}
    mean, second = np.zeros(2), np.zeros(2)
    bce_loss, rank_loss = 0., 0.
    bce_grad, rank_grad = np.zeros(3), np.zeros(3)
    mining = {p["sequence"]: p for p in fitted["mining"]}
    usable = [s for s in names if mining[s]["positive_count"] and mining[s]["negative_count"]]
    rank_families = {s.split("/")[0] for s in usable}
    theta = np.r_[fitted["coefficients"], fitted["intercept"]]
    for sequence in names:
        row = data[sequence]
        family = sequence.split("/")[0]
        mass = 1. / (len(families) * sum(s.split("/")[0] == family for s in names))
        keep, correct = labels(row)
        x = np.column_stack((row["baseline"][keep], row["context_arc"][keep]))
        y = correct[keep]
        mean += mass * x.mean(0)
        second += mass * (x * x).mean(0)
        z = (x - fitted["mean"]) / fitted["scale"]
        value = z @ theta[:2] + theta[2]
        bce_loss += mass * np.mean(np.logaddexp(0, value) - y * value)
        residual = expit(value) - y
        bce_grad += mass * np.r_[z.T @ residual / len(y), residual.mean()]
        negative = np.flatnonzero(y == 0)
        ref_score = score(row, fitted["mining_model"], 1.)[keep]
        count = max(1, int(np.ceil(fitted["fraction"] * len(negative))))
        hard = negative[np.argsort(-ref_score[negative], kind="stable")[:count]]
        np.testing.assert_array_equal(row["query_idx"][keep][hard], mining[sequence]["hard_query_idx"])
        if sequence in usable:
            w = 1. / (len(rank_families) * sum(s.split("/")[0] == family for s in usable))
            delta = z[hard].mean(0) - z[y == 1]
            gap = delta @ theta[:2]
            rank_loss += w * np.logaddexp(0, gap).mean()
            rank_grad[:2] += w * delta.T @ expit(gap) / len(gap)
    np.testing.assert_allclose(mean, fitted["mean"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.sqrt(second - mean ** 2), fitted["scale"], atol=1e-12, rtol=0)
    loss = bce_loss + fitted["strength"] * rank_loss + .0005 * (theta[:2] @ theta[:2])
    gradient = bce_grad + fitted["strength"] * rank_grad + np.r_[.001 * theta[:2], 0.]
    np.testing.assert_allclose(loss, fitted["loss"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(gradient, fitted["gradient"], atol=1e-12, rtol=0)
    projected = gradient.copy()
    for i, lower in ((0, 1e-6), (1, 0.)):
        assert theta[i] >= lower
        if theta[i] <= lower + 1e-10:
            projected[i] = min(0., gradient[i])
    assert abs(projected).max() < 2e-5
    return {"bce_loss": bce_loss, "rank_loss": rank_loss,
            "bce_gradient": bce_grad.tolist(), "rank_gradient": rank_grad.tolist(),
            "full_gradient": gradient.tolist(), "projected_max": float(abs(projected).max())}


def protection_bound(row: dict) -> dict:
    """기존 R100 정답을 하나도 잃지 않는 양의 문맥 계수비 경계를 진단한다."""
    keep, y = labels(row)
    b, c = row["baseline"], row["context_arc"]
    negative = np.flatnonzero(keep & (y == 0))
    protected = np.flatnonzero(keep & (y == 1) & (b > b[negative].max()))
    bound, witness = np.inf, None
    for start in range(0, len(protected), 128):
        positive = protected[start:start + 128]
        gap_b = b[positive, None] - b[negative][None]
        gap_c = c[negative][None] - c[positive, None]
        ratios = np.full(gap_b.shape, np.inf)
        np.divide(gap_b, gap_c, out=ratios, where=gap_c > 0)
        if ratios.size and ratios.min() < bound:
            i, j = np.unravel_index(np.argmin(ratios), ratios.shape)
            p, n = positive[i], negative[j]
            bound = float(ratios[i, j])
            witness = {"correct_query": int(row["query_idx"][p]),
                       "incorrect_query": int(row["query_idx"][n]),
                       "correct_winner": int(row["j_star"][p]),
                       "incorrect_winner": int(row["j_star"][n]),
                       "baseline_gap": float(b[p] - b[n]), "context_gap": float(c[n] - c[p])}
    return {"protected_correct": len(protected), "ratio_upper_strict": bound if np.isfinite(bound) else None,
            "witness": witness,
            "scope": "학습 OOF 사후진단. 기존 정답집합 보존의 경계이지 R100 지표 비감소의 필요조건은 아님. 선택에 사용하지 않음."}


def main() -> None:
    """동결 입력·적합·최종 원장을 검산하고 그림을 저장한다."""
    protocol = run.prepare()
    frozen = run.read(run.OUT / "selection_freeze.json")
    summary = run.read(run.OUT / "summary.json")
    assert summary["selection_sha256"] == sha256(run.OUT / "selection_freeze.json")
    upstream = run.read(run.PRIOR / "completion_validation.json")
    for path, digest in upstream["sources"].items():
        assert sha256(REPO / path) == digest, path
    data = {s: run.previous.load(p) for s, p in run.training_paths().items()}
    fitted_checks = {}
    for place, models in frozen["fold_models"].items():
        for name in run.ARMS:
            fitted_checks[place + "/" + name] = check_fit(data, models[name])
    fitted_checks["full_selected"] = check_fit(data, frozen["model"])
    ledgers, rows, comparisons = 0, 0, 0
    for role, record in (("train_cv", frozen), ("val", summary)):
        for path, digest in record["outputs"].items():
            assert sha256(REPO / path) == digest, path
            saved = run.previous.load(REPO / path)
            name = Path(path).stem.replace("_", "/", 1)
            source = data[name] if role == "train_cv" else run.previous.load(
                run.previous.feature_path("full", name, "val"))
            for field in run.accept.FIELDS:
                np.testing.assert_array_equal(saved[field], source[field])
            for method in record["per_sequence"]:
                actual = independent_metrics(saved, saved[method])
                expected = record["per_sequence"][method][name]
                for metric, value in actual.items():
                    if value is None:
                        assert expected[metric] is None
                    else:
                        np.testing.assert_allclose(value, expected[metric], atol=1e-12, rtol=0)
                comparisons += 1
            if role == "val":
                for method, model in (("tail", frozen["model"]), ("bce", frozen["reference_model"])):
                    raw = np.asarray(model["coefficients"]) / model["scale"]
                    expanded = raw[0] * source["baseline"] + raw[1] * source["context_arc"]
                    expanded += model["intercept"] - np.dot(raw, model["mean"])
                    np.testing.assert_allclose(expanded, saved[method], atol=1e-12, rtol=0)
                if frozen["model"]["coefficients"][1] == 0:
                    np.testing.assert_array_equal(np.argsort(saved["tail"], kind="stable"),
                                                  np.argsort(saved["baseline"], kind="stable"))
            ledgers += 1
            rows += len(saved["query_idx"])
    bounds = {s: protection_bound(r) for s, r in data.items()}
    raw = np.asarray(frozen["reference_model"]["coefficients"]) / frozen["reference_model"]["scale"]
    report = {"selected": frozen["selected"], "training_pass": frozen["training_pass"],
              "reference_ratio": float(raw[1] / raw[0]), "selected_ratio": frozen["model"]["raw_context_ratio"],
              "optimization": fitted_checks, "training_protection_bounds": bounds,
              "family_delta_vs_baseline_pp": {
                  f: {m: 100 * (summary["aggregate"]["tail"]["families"][f][m] - base[m])
                      for m in run.accept.exp.METRICS}
                  for f, base in summary["aggregate"]["baseline"]["families"].items()},
              "test_used": False, "scope": "손실만 변경한 1시드 판정 실험. 인코더·후보·최종승자 고정."}
    write_json(run.OUT / "report.json", report)
    write_json(run.OUT / "completion_validation.json", {
        "status": "passed", "fitted_models": len(fitted_checks), "ledgers": ledgers,
        "rows": rows, "metric_comparisons": comparisons,
        "upstream_hashes": len(upstream["sources"]), "protocol_hashes": len(protocol["sources"]),
        "test_used": False, "sources": {str(p.relative_to(REPO)): sha256(p) for p in
            (Path(__file__), run.OUT / "protocol.json", run.OUT / "selection_freeze.json",
             run.OUT / "summary.json", run.OUT / "report.json")}})
    plot(summary)
    print("VERIFIED", ledgers, rows, comparisons, "models", len(fitted_checks), flush=True)
    print("GRADIENT", fitted_checks["full_selected"], flush=True)


def plot(summary: dict) -> None:
    """최종 검증 지표와 계열별 R100을 공유 가능한 그림으로 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="/tmp/nsd_context_accept_font.ttf")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False})
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font.get_file())
    methods = ("baseline", "bce", "tail")
    names = ("기존 시간 판정", "장소 제외 BCE", "높은 오답 억제")
    colors = ("#7c8798", "#277da1", "#d98324")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metrics, labels_, source in (
        (axes[0], ("AP", "max_F1", "R@100P"), ("AP", "최대 F1", "R@100P"), "macro"),
        (axes[1], ("kitti", "helipr", "mulran"), ("KITTI 계열", "HeLiPR", "MulRan"), "families")):
        x = np.arange(3)
        for index, (method, name, color) in enumerate(zip(methods, names, colors)):
            block = summary["aggregate"][method][source]
            values = [100 * (block[m] if source == "macro" else block[m]["R@100P"]) for m in metrics]
            bars = ax.bar(x + (index - 1) * .25, values, width=.25, color=color, label=name)
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
        ax.set_xticks(x, labels_)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylabel("성능 (%)")
    axes[0].set_title("최종 검증 평균 · 동일 모델/승자 · 시드 1")
    axes[1].set_title("계열별 R@100P")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -.15), ncol=3, frameon=False)
    fig.suptitle("높은 오답 억제 학습은 문맥 계수를 0으로 낮춰 기존 판정으로 돌아간다", fontsize=12)
    fig.tight_layout(rect=(0, .04, 1, .92))
    for suffix in ("png", "pdf"):
        fig.savefig(run.OUT / ("context_tail." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
