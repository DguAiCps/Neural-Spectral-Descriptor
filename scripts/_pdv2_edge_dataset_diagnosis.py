#!/usr/bin/env python3
"""저장된 검증 결과를 주행 형태·간선·최종 오류로 나누어 분석한다."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pr_eval_lib import classify_queries, pr_sweep

FINAL = REPO / "results/pdv2_edge_reliability_final_20261016"
EDGE = REPO / "results/pdv2_edge_labels3d_20261016/main"
REL = REPO / "results/pdv2_edge_reliability_20261016"
OUT = REPO / "results/pdv2_edge_dataset_diagnosis_20261016"
OPS = ("pbev128_main", "hspec12")
GROUPS = ("same_available", "opposite_only", "other_positive", "negative", "ambiguous")
SOURCES = {}


def read(path: Path) -> dict:
    """입력 해시와 함께 JSON 또는 NPZ를 읽는다."""
    SOURCES[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def stats(x: np.ndarray) -> dict:
    """유한한 값의 개수와 사분위·꼬리값을 반환한다."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return {"n": int(len(x)), "mean": float(np.mean(x)) if len(x) else None,
            **dict(zip(("p05", "p25", "median", "p75", "p95"),
                       np.quantile(x, [.05, .25, .5, .75, .95]).tolist() if len(x) else [None] * 5))}


def trajectory(cache: dict, queries: np.ndarray) -> tuple:
    """기준 자세에서 거리·시간·회전·문맥의 물리적 길이를 잰다."""
    pos = cache["poses_gt"][:, :3, 3]
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(step)]
    t = cache["timestamps"]
    q = queries[(queries >= 5) & (queries + 5 < len(pos))]
    before, after = pos[q] - pos[q - 5], pos[q + 5] - pos[q]
    lengths = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    angle = np.degrees(np.arccos(np.clip(np.sum(before * after, axis=1) / np.maximum(lengths, 1e-12), -1, 1)))
    angle[lengths < 1.] = np.nan
    turns = np.full(len(pos), np.nan)
    turns[q] = angle
    span = np.full(len(pos), np.nan)
    span[q] = arc[q + 5] - arc[q - 5]
    good_dt = np.diff(t) > 0
    return pos, turns, span, {
        "nodes": len(pos), "path_km": float(arc[-1] / 1000),
        "extent_xyz_m": np.ptp(pos, axis=0).tolist(), "step_m": stats(step),
        "step_seconds": stats(np.diff(t)[good_dt]),
        "speed_mps": stats(step[good_dt] / np.diff(t)[good_dt]),
        "context_pm5_path_m": stats(span[q]), "context_pm5_turn_degrees": stats(angle),
        "context_turn_over30_fraction": float(np.mean(angle[np.isfinite(angle)] > 30)),
    }


def decomposition(a: dict, b: dict, decision: str) -> dict:
    """AP 차이를 질의군별 교체 회계로 분해하며 인과효과로 해석하지 않는다."""
    qc = classify_queries(a["min_past_gt_dist"])
    if not qc["n_pos"]:
        return {}
    g0, g1 = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
    names = ("recovered", "regressed", "both_correct", "both_wrong", "negative")
    masks = [qc["positive"] & ~g0 & g1, qc["positive"] & g0 & ~g1,
             qc["positive"] & g0 & g1, qc["positive"] & ~g0 & ~g1, qc["negative"]]
    values = {}
    for subset in range(32):
        mask = np.zeros(len(g0), bool)
        for i, m in enumerate(masks):
            if subset & (1 << i):
                mask |= m
        values[subset] = float(pr_sweep(np.where(mask, b[decision], a[decision]),
                                       np.where(mask, b["top1_geo_dist"], a["top1_geo_dist"]), qc)["AP"])
    shares = {}
    for i, name in enumerate(names):
        shares[name] = sum(math.factorial(bin(k).count("1")) * math.factorial(4 - bin(k).count("1")) / 120 *
                           (values[k | (1 << i)] - values[k]) for k in range(32) if not k & (1 << i))
    assert abs(sum(shares.values()) - (values[31] - values[0])) < 1e-12
    return {"before": values[0], "after": values[31], "shares": shares}


def changes(a: dict, b: dict, decision: str, turns: np.ndarray, span: np.ndarray) -> dict:
    """최종 오류를 방향·거리·후보 보존 여부로 구분한다."""
    for key in ("query_idx", "min_past_gt_dist", "group"):
        np.testing.assert_array_equal(a[key], b[key])
    positive = a["min_past_gt_dist"] <= 5
    groups = a["group"].copy()
    groups[a["min_past_gt_dist"] > 25] = 3
    groups[(a["min_past_gt_dist"] > 5) & (a["min_past_gt_dist"] <= 25)] = 4
    assert np.isin(groups, np.arange(5)).all()
    good0, good1 = a["top1_geo_dist"] <= 5, b["top1_geo_dist"] <= 5
    regressed, recovered = positive & good0 & ~good1, positive & ~good0 & good1
    rows = {}
    for name, mask in [("positive", positive), *[(g, groups == i) for i, g in enumerate(GROUPS)]]:
        lost, gained = regressed & mask, recovered & mask
        rows[name] = {
            "n": int(mask.sum()), "regressed": int(lost.sum()), "recovered": int(gained.sum()),
            "regressed_pool_missing": int((lost & ~b["union_hit"]).sum()),
            "regressed_pool_present": int((lost & b["union_hit"]).sum()),
            "regressed_distance_5to10": int((lost & (b["top1_geo_dist"] <= 10)).sum()),
            "regressed_distance_10to25": int((lost & (b["top1_geo_dist"] > 10) & (b["top1_geo_dist"] <= 25)).sum()),
            "regressed_distance_over25": int((lost & (b["top1_geo_dist"] > 25)).sum()),
            "candidate_changed": int((mask & (a["j_star"] != b["j_star"])).sum()),
            "decision_score_change": stats((b[decision] - a[decision])[mask]),
            "turn_degrees": stats(turns[a["query_idx"][mask]]),
            "regressed_turn_degrees": stats(turns[a["query_idx"][lost]]),
            "context_span_m": stats(span[a["query_idx"][mask]]),
            "regressed_context_span_m": stats(span[a["query_idx"][lost]]),
        }
    assert sum(rows[g]["n"] for g in GROUPS) == len(positive)
    rows["ap_accounting"] = decomposition(a, b, decision)
    rows["regressed_cases"] = [{"query": int(b["query_idx"][i]), "group": GROUPS[int(b["group"][i])],
        "candidate_before": int(a["j_star"][i]), "candidate_after": int(b["j_star"][i]),
        "distance_before": float(a["top1_geo_dist"][i]), "distance_after": float(b["top1_geo_dist"][i]),
        "pool_present": bool(b["union_hit"][i]), "turn_degrees": float(turns[b["query_idx"][i]])}
        for i in np.flatnonzero(regressed)]
    return rows


def analyze() -> None:
    """검증 9시퀀스와 학습 8시퀀스의 저장 자료만 집계한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read(FINAL / "summary.json")
    selection = read(REPO / "results/pdv2_symmetric3_20261013/s1/confidence_selection.json")["ops"]
    result = {"scope": "검증9·주모델seed1의 사후 원인진단. 학습8의 저장 입력은 분포 비교용. 시험·재학습·설정선택 없음.",
              "definitions": {"positive": "전체 과거지도에 3D 5m 이내 정답 있음", "negative": "전체 과거지도의 최근접 거리가 25m 초과",
                  "opposite_only": "정답이 모두 이동방향 차이120도 이상", "positive_height_fraction": "센서 높이보다 높은 점이 기록된 20x120 셀 비율. 점군 점유율이 아님",
                  "ap_accounting": "5개 질의군의 원장 교체 Shapley 합. 인과 추정이 아닌 AP 차이의 회계",
                  "curvature": "과거5와 미래5의 3차원 이동방향 차이. 움직임 길이 곱1m2 미만 제외"},
              "val": {}, "train": {}, "test_used": False}
    plot_rows = []
    sequences = summary["scores"]["C"][OPS[0]]["per_sequence"]
    for seq in sequences:
        slug = seq.replace("/", "_")
        meta = read(REL / "features" / f"val_{slug}.json")
        paths = {"cache" if Path(p).name.startswith("cache_") else "layouts" if "layouts_" in p else "spec": REPO / p
                 for p in meta["inputs_sha256"]}
        cache = read(paths["cache"])
        c = read(EDGE / f"edges_{slug}.npz")
        q = np.flatnonzero(c["valid"])
        pos, turns, span, route = trajectory(cache, q)
        assert len(q) == summary["scores"]["C"][OPS[0]]["per_sequence"][seq]["n_queries"]
        groups = {name: int((c["group"][q] == i).sum()) for i, name in enumerate(GROUPS)}
        assert sum(groups.values()) == len(q)
        has_true = np.bincount(c["query"][c["distance_3d"] <= 5], minlength=len(pos)) > 0
        positive = np.isin(c["group"][q], [0, 1, 2])
        layout = read(paths["layouts"])["nsd_layouts"]
        spec = read(paths["spec"])
        occupancy = np.mean(spec["raw_height"] > 0, axis=(1, 2))
        features = read(REL / "features" / f"val_{slug}.npz")
        record = read(FINAL / f"keys_{slug}.json")
        row = {"trajectory": route, "queries": len(q), "groups": groups,
               "positive_fraction": float(positive.mean()),
               "candidate30_positive_coverage": float(has_true[q][positive].mean()) if positive.any() else None,
               "positive_height_fraction": stats(occupancy[q]),
               "range_layout_nonzero_fraction": stats(np.mean(layout[q] > 0, axis=(1, 2))),
               "ops": {}}
        ledgers = {}
        for op in OPS:
            ledgers[op] = {}
            for arm in ("temporal", "A", "C"):
                root = FINAL / "C" if arm == "C" else EDGE / ("xyz" if arm == "A" else arm)
                ledgers[op][arm] = read(root / f"val_{op}_{slug}.npz")
            before, after = ledgers[op]["A"], ledgers[op]["C"]
            decision = selection[op]["decision"]
            comparisons = {name: changes(ledgers[op][aa], ledgers[op][bb], decision, turns, span)
                           for name, aa, bb in [("A_to_C", "A", "C"), ("temporal_to_A", "temporal", "A")]}
            for name, aa, bb in [("A_to_C", "A", "C"), ("temporal_to_A", "temporal", "A")]:
                accounting = comparisons[name]["ap_accounting"]
                if accounting:
                    for field, arm in [("before", aa), ("after", bb)]:
                        assert abs(accounting[field] - summary["scores"][arm][op]["per_sequence"][seq]["AP"]) < 1e-12
            quality = {}
            for arm in ("A", "C"):
                data = record["ops"][op][arm]
                quality[arm] = {}
                for g in GROUPS:
                    v, att = data["quality"][g], data["attention"][1][g]
                    quality[arm][g] = {**v,
                        "far_edge_fraction": v["far_3d"] / v["edges"] if v["edges"] else None,
                        "true_edge_fraction": v["positive_3d"] / v["edges"] if v["edges"] else None,
                        "mean_probability": v["probability_sum"] / v["edges"] if v["edges"] else None,
                        "layer2_far_mass": att["similarity_far_mass_sum"] / att["nodes"] if att["nodes"] else None,
                        "layer2_true_mass": att["similarity_true_mass_sum"] / att["nodes"] if att["nodes"] else None}
            signals = {}
            for name, mask in [("true", features["distance"] <= 5), ("far", features["distance"] > 25),
                               ("same_available_true", (features["distance"] <= 5) & (features["group"][features["query"]] == 0)),
                               ("opposite_true", (features["distance"] <= 5) & (features["group"][features["query"]] == 1))]:
                signals[name] = {"raw_cos": stats(features["features"][mask, 1]),
                                 "pass1_key_cos": stats(features["features"][mask, 0]),
                                 "pass1_context_cos": stats(features["features"][mask, 2]),
                                 "alignment": stats(features["extra_" + op][mask, 0]),
                                 "column": stats(features["extra_" + op][mask, 1])}
            row["ops"][op] = {"comparisons": comparisons, "edge_quality": quality, "candidate30_signals": signals}
        result["val"][seq] = row
        plot_rows.append((seq, pos, q, c["group"][q], ledgers, row))
        print("VAL", seq, groups, flush=True)
    for p in sorted((REL / "features").glob("train_*.json")):
        meta = read(p)
        seq = meta["sequence"]
        cache_path = next(iter(meta["inputs_sha256"]))
        cache = read(REPO / cache_path)
        features = read(p.with_suffix(".npz"))
        q = np.flatnonzero(features["valid"])
        _, _, _, route = trajectory(cache, q)
        groups = {name: int((features["group"][q] == i).sum()) for i, name in enumerate(GROUPS)}
        result["train"][seq] = {"queries": len(q), "trajectory": route, "groups": groups,
            "positive_height_fraction": stats(np.mean(cache["raw_height"][q] > 0, axis=(1, 2)))}
        print("TRAIN", seq, groups, flush=True)
    result["source_sha256"] = SOURCES
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    plot(plot_rows)
    print("DONE", OUT, flush=True)


def plot(rows: list) -> None:
    """실제 궤적 위에 정답 유형과 최종 정답 손실을 표시한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for ax, (seq, pos, q, groups, ledgers, row) in zip(axes.flat, rows):
        # KITTI 기준 자세는 카메라 좌표여서 수평축이 x,z이다.
        axes_xy = [0, 2] if seq.startswith("kitti/") else [0, 1]
        xy = (pos[:, axes_xy] - pos[0, axes_xy]) / 1000
        ax.plot(xy[:, 0], xy[:, 1], color="0.8", lw=.8, zorder=1)
        for i, color, label in [(0, "#167ca3", "Same available"), (1, "#e19a24", "Opposite only"), (2, "#9a73bc", "Other positive")]:
            mask = groups == i
            ax.scatter(xy[q[mask], 0], xy[q[mask], 1], s=4, c=color, label=label, zorder=2)
        for op, marker, color, label in [(OPS[0], "x", "#bc1642", "NSD loss"), (OPS[1], "+", "#171717", "NSD-H loss")]:
            a, b = ledgers[op]["A"], ledgers[op]["C"]
            lost = (a["min_past_gt_dist"] <= 5) & (a["top1_geo_dist"] <= 5) & (b["top1_geo_dist"] > 5)
            ax.scatter(xy[q[lost], 0], xy[q[lost], 1], s=42, marker=marker, c=color, label=label, zorder=3)
        npos = sum(row["groups"][g] for g in GROUPS[:3])
        reverse = row["groups"]["opposite_only"] / max(npos, 1)
        ax.set_title(f"{seq}\nRevisit {100 * row['positive_fraction']:.1f}% | Opposite among revisits {100 * reverse:.1f}%")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)" if seq.startswith("kitti/") else "y (km)")
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle("Validation trajectories and final top-1 regressions: A to C (seed 1)")
    fig.tight_layout(rect=[0, .04, 1, .96])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("trajectory_errors." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    analyze()
