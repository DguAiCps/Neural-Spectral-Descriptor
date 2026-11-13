#!/usr/bin/env python3
"""과거 평균 보정의 관측 범위·수식·적합·최종 성능을 검산한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_context_relative as run
from _pdv2_context_accept_lib import auc, labels
from _pdv2_context_place_lib import score
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_no_context_ablation import sha256, write_json
from retrieval.sequence_evidence import neighbor_plans


def scalar_features(key: np.ndarray, path: np.ndarray, row: dict, index: int) -> dict:
    """직접 과거 평균과 쌍별 반복문으로 보정 수식을 확인한다."""
    q, j, direction = (int(row[f][index]) for f in ("query_idx", "j_star", "direction"))
    qn, jn, valid = neighbor_plans(path, np.array([q]), np.array([j]))["arc_unique"]
    valid = valid[direction, 0]
    qs, js = qn[direction, 0][valid], jn[direction, 0][valid]
    weights = np.array([1. / np.sum(js == value) for value in js])
    beta = (2. / 3.) * (11. / 10.)
    mass = np.r_[1 - beta + beta / (1 + weights.sum()), beta * weights / (1 + weights.sum())]
    result = {}
    for name, sl in (("d", slice(0, 288)), ("context", slice(288, 416))):
        vectors = key[:, sl].astype(np.float64)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        past = vectors[:q - 29]
        mean = past.mean(axis=0)
        sums = {k: 0. for k in ("arc_check", "excess", "centered_dot", "centered_cos", "background")}
        minimum = np.inf
        for a, b, weight in zip(np.r_[q, qs], np.r_[j, js], mass):
            left, right = vectors[a], vectors[b]
            background = float(np.mean(past @ left))
            u, v = left - mean, right - mean
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            dot = float(left @ right)
            sums["arc_check"] += weight * dot
            sums["background"] += weight * background
            sums["excess"] += weight * (dot - background)
            sums["centered_dot"] += weight * (u @ v)
            sums["centered_cos"] += weight * (u @ v) / (max(nu, 1e-6) * max(nv, 1e-6))
            minimum = min(minimum, nu, nv)
        result.update({name + "_" + k: float(v) for k, v in sums.items()})
        result[name + "_mean_norm"] = float(np.linalg.norm(mean))
        result[name + "_residual_min"] = float(minimum)
    return result


def verify_fit(data: dict, model: dict) -> dict:
    """제외 장소 없이 적합 통계와 BCE 경계 기울기를 독립 계산한다."""
    names = model["training"]
    assert set(names).isdisjoint(model["heldout"])
    families = {s.split("/")[0] for s in names}
    mean, second = np.zeros(2), np.zeros(2)
    gradient = np.zeros(3)
    loss = 0.
    contributions = {}
    coefficients = np.asarray(model["coefficients"])
    for sequence in names:
        row = data[sequence]
        keep, y = labels(row)
        y = y[keep]
        x = np.column_stack([row[f][keep] for f in model["fields"]])
        family = sequence.split("/")[0]
        mass = 1. / (len(families) * sum(s.split("/")[0] == family for s in names))
        mean += mass * x.mean(0)
        second += mass * np.square(x).mean(0)
        z = (x - model["mean"]) / model["scale"]
        value = z @ coefficients + model["intercept"]
        residual = expit(value) - y
        part = mass * np.r_[z.T @ residual / len(y), residual.mean()]
        contributions[sequence] = part.tolist()
        gradient += part
        loss += mass * np.mean(np.logaddexp(0, value) - y * value)
    np.testing.assert_allclose(mean, model["mean"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.maximum(np.sqrt(np.maximum(0., second - mean ** 2)), 1e-6),
                               model["scale"], atol=1e-12, rtol=0)
    loss += .0005 * (coefficients @ coefficients)
    gradient[:2] += .001 * coefficients
    np.testing.assert_allclose(loss, model["loss"], atol=1e-12, rtol=0)
    projected = gradient.copy()
    for i, lower in ((0, 1e-6), (1, 0.)):
        assert coefficients[i] >= lower
        if coefficients[i] <= lower + 1e-10:
            projected[i] = min(0., projected[i])
    assert np.abs(projected).max() < 2e-5
    raw = coefficients / model["scale"]
    return {"gradient": gradient.tolist(), "projected_max": float(abs(projected).max()),
            "sequence_gradient": contributions, "raw_coefficient_ratio": float(raw[1] / raw[0])}


def diagnose(row: dict, fields: list) -> dict:
    """전체 및 기존 점수 상위 10%의 정답·오답 구분력을 사후 진단한다."""
    keep, y = labels(row)
    top = keep & (row["baseline"] >= np.quantile(row["baseline"][keep], .9))
    result = {}
    for cohort, mask in (("all_eligible", keep), ("baseline_top10pct", top)):
        result[cohort] = {"rows": int(mask.sum()), "correct": int(y[mask].sum()), "features": {}}
        for field in fields:
            stats = {"auc": auc(row[field][mask], y[mask])}
            for name, target in (("correct", 1), ("incorrect", 0)):
                value = row[field][mask & (y == target)]
                stats[name] = {"mean": float(value.mean()),
                               "q10_q50_q90": np.quantile(value, [.1, .5, .9]).tolist()} if len(value) else None
            result[cohort]["features"][field] = stats
    return result


def main() -> None:
    """검증된 결과와 원인 진단·그림을 새 결과 폴더에 저장한다."""
    spec = run.prepare()
    frozen = run.read(run.OUT / "selection_freeze.json")
    summary = run.read(run.OUT / "summary.json")
    assert frozen["protocol_sha256"] == sha256(run.OUT / "protocol.json")
    assert summary["selection_sha256"] == sha256(run.OUT / "selection_freeze.json")
    upstream = run.read(run.PRIOR / "completion_validation.json")
    for p, h in upstream["sources"].items():
        assert sha256(REPO / p) == h, p
    data = {s: run.previous.load(run.feature_path("train", s)) for s in spec["train"]}
    fitted_checks = {}
    for place, models in frozen["fold_models"].items():
        for name, model in models.items():
            if model is not None:
                fitted_checks[place + "/" + name] = verify_fit(data, model)
    for name, model in frozen["models"].items():
        fitted_checks["full/" + name] = verify_fit(data, model)
    diagnostics, cases, feature_hashes = {}, {}, {}
    ledgers, rows, scalar_checks, prefix_checks, metric_checks = 0, 0, 0, 0, 0
    for role in ("train", "val"):
        diagnostics[role] = {}
        for sequence in spec[role]:
            p = run.feature_path(role, sequence)
            metadata = run.read(p.with_suffix(".json"))
            assert sha256(p) == metadata["sha256"]
            for source, digest in metadata["sources"].items():
                assert sha256(REPO / source) == digest
            row = run.previous.load(p)
            key, path, original = run.inputs(role, sequence)
            for field, value in original.items():
                np.testing.assert_array_equal(row[field], value)
            checks = np.unique(np.linspace(0, len(row["query_idx"]) - 1, 7, dtype=int))
            for i in checks:
                scalar = scalar_features(key, path, row, int(i))
                for field, value in scalar.items():
                    np.testing.assert_allclose(value, row[field][i], atol=3e-10, rtol=0)
                scalar_checks += 1
            prefix_checks += len(metadata["prefix_checks"])
            fields = ["baseline", "context_arc", frozen["selected"], "d_arc", frozen["matched_raw"]]
            diagnostics[role][sequence] = diagnose(row, fields)
            if role == "train" and sequence in ("helipr/Bridge02", "helipr/Bridge03"):
                query = 6835 if sequence.endswith("02") else 1099
                index = int(np.flatnonzero(row["query_idx"] == query)[0])
                case_fields = ["query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "context_arc",
                               "context_excess", "context_centered_dot", "context_centered_cos",
                               "context_background", "context_mean_norm", "context_residual_min"]
                cases[sequence] = {f: float(row[f][index]) for f in case_fields}
            record = frozen if role == "train" else summary
            score_path = run.OUT / ("train_cv" if role == "train" else "val") / p.name
            assert sha256(score_path) == record["outputs"][str(score_path.relative_to(REPO))]
            predictions = run.previous.load(score_path)
            for field in run.accept.FIELDS:
                np.testing.assert_array_equal(predictions[field], row[field])
            for method in record["per_sequence"]:
                actual = independent_metrics(row, predictions[method])
                expected = record["per_sequence"][method][sequence]
                for metric, value in actual.items():
                    if value is None:
                        assert expected[metric] is None
                    else:
                        np.testing.assert_allclose(value, expected[metric], atol=1e-12, rtol=0)
                metric_checks += 1
            if role == "val":
                for name, model in frozen["models"].items():
                    coefficients = np.asarray(model["coefficients"]) / model["scale"]
                    expected = sum(coefficients[i] * row[f] for i, f in enumerate(model["fields"]))
                    expected += model["intercept"] - np.dot(coefficients, model["mean"])
                    np.testing.assert_allclose(expected, predictions[name], atol=1e-12, rtol=0)
                if frozen["models"]["context_relative"]["coefficients"][1] == 0:
                    np.testing.assert_array_equal(np.argsort(predictions["context_relative"], kind="stable"),
                                                  np.argsort(predictions["baseline"], kind="stable"))
            ledgers += 1
            rows += len(row["query_idx"])
            feature_hashes[str(p.relative_to(REPO))] = sha256(p)
            print("VERIFIED_SEQUENCE", role, sequence, flush=True)
    # 시퀀스→계열→전체 평균도 결과 표와 별도로 대조한다.
    for record in (frozen, summary):
        for method, per_sequence in record["per_sequence"].items():
            groups = {}
            for sequence, item in per_sequence.items():
                if item["n_pos"]:
                    family = "kitti" if sequence.startswith("kitti") else sequence.split("/")[0]
                    groups.setdefault(family, []).append(item)
            for metric in run.accept.exp.METRICS:
                values = []
                for family, items in groups.items():
                    value = sum(item[metric] for item in items) / len(items)
                    np.testing.assert_allclose(value, record["aggregate"][method]["families"][family][metric], atol=1e-12)
                    values.append(value)
                np.testing.assert_allclose(sum(values) / len(values), record["aggregate"][method]["macro"][metric], atol=1e-12)
    report = {"selected": frozen["selected"], "training_pass": frozen["training_pass"],
        "fits": fitted_checks, "diagnostics": diagnostics, "cases": cases, "test_used": False,
        "scope": "진단은 선택 후 계산. 새 조건 적합·검증 선택에 사용하지 않음.",
        "sources": feature_hashes}
    write_json(run.OUT / "report.json", report)
    plot(summary)
    paths = [Path(__file__), REPO / "scripts/_pdv2_context_tail_finish.py"]
    paths += [run.OUT / p for p in ("protocol.json", "selection_freeze.json", "summary.json", "report.json",
                                    "context_relative.png", "context_relative.pdf")]
    write_json(run.OUT / "completion_validation.json", {"status": "passed", "fitted_models": len(fitted_checks),
        "feature_and_score_ledgers": ledgers, "rows": rows, "scalar_queries": scalar_checks,
        "prefix_queries": prefix_checks, "metric_comparisons": metric_checks,
        "protocol_hashes": len(spec["sources"]), "upstream_hashes": len(upstream["sources"]),
        "test_used": False, "sources": {str(p.relative_to(REPO)): sha256(p) for p in paths}})
    print("COMPLETE_VALIDATION", ledgers, rows, scalar_checks, prefix_checks, metric_checks, flush=True)


def plot(summary: dict) -> None:
    """최종 지표와 계열별 높은 정밀도를 그림으로 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = font_manager.FontProperties(fname="/tmp/nsd_context_accept_font.ttf")
    font_manager.fontManager.addfont(font.get_file())
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False})
    methods = ("baseline", "context_uncorrected", "context_relative", "d_relative")
    names = ("기존 시간 판정", "미보정 문맥", "과거 평균 보정 문맥", "같은 보정 원 특징")
    colors = ("#7c8798", "#277da1", "#d98324", "#729f48")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, keys, labels_, category in (
        (axes[0], ("AP", "max_F1", "R@100P"), ("AP", "최대 F1", "R@100P"), "macro"),
        (axes[1], ("kitti", "helipr", "mulran"), ("KITTI 계열", "HeLiPR", "MulRan"), "families")):
        x = np.arange(3)
        for i, (method, name, color) in enumerate(zip(methods, names, colors)):
            block = summary["aggregate"][method][category]
            values = [100 * (block[k] if category == "macro" else block[k]["R@100P"]) for k in keys]
            bars = ax.bar(x + (i - 1.5) * .2, values, width=.2, color=color, label=name)
            ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=3)
        ax.set_xticks(x, labels_)
        ax.set_ylim(0, 100)
        ax.set_ylabel("성능 (%)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_title("최종 검증 평균 · 동일 모델/승자 · 시드 1")
    axes[1].set_title("계열별 R@100P")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -.15), ncol=4, frameon=False)
    fig.suptitle("과거 후보 평균을 뺀 내적: 문맥 비중 0, 원 특징 대조도 최종 성능 하락", fontsize=12)
    fig.tight_layout(rect=(0, .03, 1, .92))
    for suffix in ("png", "pdf"):
        fig.savefig(run.OUT / ("context_relative." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
