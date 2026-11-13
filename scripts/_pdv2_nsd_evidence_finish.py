#!/usr/bin/env python3
"""NSD 판정·학습 대조의 원장을 재검사하고 표와 그림을 만든다."""
from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_spectral_evidence as exp
import _pdv2_reference_evidence as ref
import _pdv2_nsd_channel as channel
import _pdv2_arc_evidence as arc
from _pdv2_no_context_ablation import sha256, write_json
from retrieval.spectral_evidence import correlation_profiles, temporal_scores

OUT = exp.OUT / "reports"
MODELS = ("standard", "capacity_point")
FIELDS = ("query_idx", "j_star", "min_past_gt_dist", "top1_geo_dist", "group")
NEG = REPO / "results/pdv2_nsd_negative_20261018"
LOSS = REPO / "results/pdv2_nsd_stratified_20261018"


def aggregate(rows: dict) -> dict:
    """양성이 있는 시퀀스→세 계열을 차례로 평균한다."""
    families = {}
    for family in ("kitti", "helipr", "mulran"):
        selected = [r for seq, r in rows.items() if seq.startswith(family) and r["n_pos"]]
        families[family] = {m: float(np.mean([r[m] for r in selected])) for m in exp.METRICS}
    return {"macro": {m: float(np.mean([r[m] for r in families.values()])) for m in exp.METRICS},
            "families": families}


def check_metrics(actual: dict, expected: dict) -> None:
    """집계 수치를 원래 결과와 엄격히 비교한다."""
    for metric in exp.METRICS:
        np.testing.assert_allclose(actual[metric], expected[metric], atol=1e-14, rtol=0)


def audit_ledgers(spec: dict) -> tuple:
    """최종 후보·질의·점수 지표와 후속 조건의 기준 재현을 확인한다."""
    roots = {"spectral": exp.OUT, "reference": ref.OUT, "channel": channel.OUT, "arc": arc.OUT}
    methods = {"spectral": exp.CFG["methods"], "reference": ref.METHODS,
               "channel": ("raw", "blend25"), "arc": arc.METHODS}
    data, counts, hashes = {}, {}, {}
    for name, root in roots.items():
        summary = exp.read(root / "summary.json")
        counts[name] = {"ledgers": 0, "rows": 0}
        data[name] = {}
        for seed in ("1", "2", "3"):
            data[name][seed] = {}
            for model in MODELS:
                data[name][seed][model] = {}
                for ch in ("c_spec", "c_col") if name == "channel" else ("selected",):
                    by_method = {m: {} for m in methods[name]}
                    for sequence in spec["val"]:
                        sensor, seq = sequence.split("/")
                        suffix = f"{model}_{ch}_" if name == "channel" else f"{model}_"
                        path = root / f"s{seed}/{suffix}{sensor}_{seq}.npz"
                        a = exp.arrays(path)
                        base = exp.arrays(exp.OUT / f"s{seed}/{model}_{sensor}_{seq}.npz")
                        for field in FIELDS:
                            np.testing.assert_array_equal(a[field], base[field])
                        assert (a["j_star"] <= a["query_idx"]-30).all()
                        for method in methods[name]:
                            assert np.isfinite(a[method]).all()
                            by_method[method][sequence] = exp.metrics(a[method], a)
                        if name == "reference":
                            np.testing.assert_array_equal(a["raw"], base["raw"])
                        if name == "arc":
                            previous = exp.arrays(ref.OUT / f"s{seed}/{model}_{sensor}_{seq}.npz")
                            np.testing.assert_array_equal(a["frame5"], previous["temporal_blend25"])
                        if name == "channel":
                            source = REPO / spec["selected"][model][seed]["folder"] / f"val_{exp.OP}_{sensor}_{seq}.npz"
                            np.testing.assert_array_equal(a["raw"], exp.arrays(source)[ch])
                        counts[name]["ledgers"] += 1
                        counts[name]["rows"] += len(a["query_idx"])
                        hashes[str(path.relative_to(REPO))] = sha256(path)
                    result = {method: aggregate(rows) for method, rows in by_method.items()}
                    data[name][seed][model][ch] = result
                    for method, value in result.items():
                        expected = summary["per_seed"][seed][model]
                        expected = expected[ch][method] if name == "channel" else expected[method]
                        check_metrics(value["macro"], expected)
        hashes[str((root / "summary.json").relative_to(REPO))] = sha256(root / "summary.json")
    return data, counts, hashes


def prefix_audit(spec: dict) -> dict:
    """실제 q+16 접두 입력에서 배경 참조와 거리 대응을 재계산한다."""
    checks, geometry, saved_error = [], {}, 0.
    for sequence in spec["val"]:
        sensor, seq, paths, _ = exp.old.base.inputs("val", sequence)
        poses = exp.arrays(paths["cache"])["poses"]
        yaw = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
        travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
        with np.load(paths["spec"]) as z:
            bank = exp.old.base.bank_for(exp.OP, z)
        geometry[sequence] = {}
        for seed in ("1", "2", "3"):
            ch = exp.read(exp.old.base.BASE / f"s{seed}/confidence_selection.json")["ops"][exp.OP]["decision"]
            for model in MODELS:
                a = exp.arrays(exp.OUT / f"s{seed}/{model}_{sensor}_{seq}.npz")
                for i in (len(a["query_idx"])//3, 2*len(a["query_idx"])//3):
                    q, j = a["query_idx"][i:i+1], a["j_star"][i:i+1]
                    end = int(q[0]+16)
                    small = exp.truncate_bank(bank, end)
                    refs, valid = ref.reference_indices(q)
                    assert refs.max() <= q[0]-30
                    # 실행기의 공유 배경 배열은 float64다. 같은 자료형으로 재계산한다.
                    full_bg = correlation_profiles(bank, np.repeat(q, refs.size), refs.ravel(), ch).max(1).reshape(refs.shape).astype(np.float64)
                    small_bg = correlation_profiles(small, np.repeat(q, refs.size), refs.ravel(), ch).max(1).reshape(refs.shape).astype(np.float64)
                    np.testing.assert_array_equal(full_bg, small_bg)
                    temporal = temporal_scores(small, yaw[:end], q, j, ch)["temporal_independent"]
                    np.testing.assert_array_equal(temporal, temporal_scores(bank, yaw, q, j, ch)["temporal_independent"])
                    calculated = ref.calibrate(a["raw"][i:i+1], temporal, refs, valid, small_bg, j)
                    saved = exp.arrays(ref.OUT / f"s{seed}/{model}_{sensor}_{seq}.npz")
                    for method in ref.METHODS:
                        saved_error = max(saved_error, float(np.max(np.abs(calculated[method]-saved[method][i:i+1]))))
                        np.testing.assert_allclose(calculated[method], saved[method][i:i+1], atol=1e-12, rtol=0)
                    saved = exp.arrays(arc.OUT / f"s{seed}/{model}_{sensor}_{seq}.npz")
                    for method in arc.METHODS:
                        full_score, _ = arc.scores(bank, travel, q, j, ch, method)
                        small_score, _ = arc.scores(small, travel[:end], q, j, ch, method)
                        np.testing.assert_array_equal(full_score, small_score)
                        reproduced = .75*a["raw"][i:i+1]+.25*small_score
                        saved_error = max(saved_error, float(np.max(np.abs(reproduced-saved[method][i:i+1]))))
                        np.testing.assert_allclose(reproduced, saved[method][i:i+1], atol=1e-12, rtol=0)
                    checks.append({"sequence": sequence, "seed": seed, "model": model,
                                   "q": int(q[0]), "reference_max_error": 0., "arc_max_error": 0.})
                if seed == "1" and model == "standard":
                    q, j = a["query_idx"], a["j_star"]
                    for method in arc.METHODS:
                        errors, offsets, dropped = [], [], 0
                        for direction in (1, -1):
                            for step in (-5, 5):
                                qn = q+step
                                delta = travel[qn]-travel[q]
                                if method == "arc_match10":
                                    jn = arc.arc_neighbor(travel, j, direction*delta)
                                    good = jn != j
                                else:
                                    jn = j+direction*step*(2 if method == "candidate_frame10" else 1)
                                    good = (jn >= 0) & (jn < len(travel))
                                assert (jn[good] <= q[good]-20).all()
                                errors.extend(np.abs(np.abs(travel[jn[good]]-travel[j[good]])-np.abs(delta[good])))
                                offsets.extend(np.abs(jn[good]-j[good]))
                                dropped += int((~good).sum())
                        geometry[sequence][method] = {"distance_error_m_p10_p50_p90": np.quantile(errors, [.1,.5,.9]).tolist(),
                            "frame_offset_p10_p50_p90": np.quantile(offsets, [.1,.5,.9]).tolist(),
                            "dropped_fraction": dropped/(4*len(q)),
                            "scope": "시드1시간모델의모든최종후보,정/역두가설을함께집계.정답만의경로오차가아님."}
    return {"checks": checks, "geometry": geometry, "saved_batch_vs_single_max_error": saved_error}


def negative_audit(spec: dict) -> dict:
    """고정 임계값에서 새 오수락의 키/후보 목록 개입을 분리한다."""
    rows, hashes = {}, {}
    for sequence in spec["val"]:
        path = NEG / (sequence.replace("/", "_")+".json")
        d = exp.read(path)
        for name, value in d["sources"].items():
            assert sha256(REPO / name) == value, name
            hashes[name] = value
        counts = {k: 0 for k in ("changed_negative", "new_false_accept", "key_only", "pool_only", "either", "joint",
                                 "old_winner_retained", "arbitrary_bound_feasible", "all_pool_above_tau", "positive_lost")}
        for row in d["cases"]:
            if row["cohort"] == "positive_lost":
                counts["positive_lost"] += 1
                continue
            counts["changed_negative"] += 1
            s = np.asarray(row["score_key_by_pool"])
            np.testing.assert_allclose(s.diagonal(), [row["old_score"], row["new_score"]], atol=2e-7, rtol=0)
            if row["old_score"] < d["tau"] <= row["new_score"]:
                counts["new_false_accept"] += 1
                key, pool = s[1, 0] >= d["tau"], s[0, 1] >= d["tau"]
                counts["either" if key and pool else "key_only" if key else "pool_only" if pool else "joint"] += 1
                counts["old_winner_retained"] += row["old_winner_in_new_pool"]
                counts["arbitrary_bound_feasible"] += row["old_winner_arbitrary_key_bound_feasible"]
                counts["all_pool_above_tau"] += row["new_pool_all_above_tau"]
        assert counts["changed_negative"] == d["changed_negative_winners"]
        assert sum(counts[k] for k in ("key_only", "pool_only", "either", "joint")) == counts["new_false_accept"]
        rows[sequence] = counts
    return {"by_sequence": rows, "total": {k: sum(r[k] for r in rows.values()) for k in counts},
            "source_hashes": hashes, "scope": "시드1,새로기존임계값을넘은오수락의조건부교차.인과AP비율이나학습가능성증명아님."}


def figures(data: dict, negative: dict) -> None:
    """최종 판정 지표와 교차 진단을 독립 그림으로 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf")
    font_manager.fontManager.addfont(str(font if font.exists() else OUT / "UnDotum.ttf"))
    plt.rcParams.update({"font.family": "UnDotum", "axes.unicode_minus": False, "pdf.fonttype": 42})
    conditions = [("spectral", "raw", "현재 판정"), ("spectral", "temporal_independent", "시간 평균만"),
                  ("spectral", "temporal_coherent", "회전 보정 평균"), ("reference", "temporal_blend25", "프레임 대응 혼합"),
                  ("arc", "arc_match10", "거리 대응 혼합")]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for axis, metric in zip(axes, ("AP", "R@100P")):
        for model, offset, color, label in (("standard", -.12, "#276fbf", "표준 시간 모델"),
                                             ("capacity_point", .12, "#e18634", "용량 맞춘 투영")):
            scores = np.array([[data[stage][str(s)][model]["selected"][method]["macro"][metric]*100
                                for s in (1, 2, 3)] for stage, method, _ in conditions])
            x = np.arange(len(conditions))+offset
            axis.errorbar(x, scores.mean(1), yerr=scores.std(1, ddof=1), fmt="o-", capsize=3, color=color, label=label)
            for i in range(3):
                axis.scatter(x+(i-1)*.025, scores[:, i], s=10, color=color, alpha=.5)
        axis.set_xticks(np.arange(len(conditions)), [r[2] for r in conditions], rotation=20, ha="right")
        axis.set_ylabel(metric+" (%)")
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("NSD 최종 판정 · 검증 9시퀀스 · 3시드 평균±표본 표준편차", fontsize=12)
    fig.text(.5, .015, "최종 후보는 동일. 판정 단계의 시간 정보 효과이며, 반복 사용한 검증셋의 탐색 결과다.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.96))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("final_confidence."+ext), dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7.8,4))
    labels = ["키 개입만 기준 초과", "목록 개입만 기준 초과", "각 개입 모두 기준 초과", "두 개입 함께 필요"]
    values = [negative["total"][k] for k in ("key_only", "pool_only", "either", "joint")]
    bars = axis.barh(labels[::-1], values[::-1], color=["#888888", "#7c64af", "#e18634", "#276fbf"])
    axis.bar_label(bars, padding=4)
    axis.set_xlim(0, max(values)*1.12)
    axis.set_xlabel("새 비재방문 오수락 수 (시드 1, 고정된 기존 임계값)")
    axis.set_title("용량 맞춘 투영 → 표준 시간 모델: 키·후보 목록 교차 진단")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("negative_crossing."+ext), dpi=180)
    plt.close(fig)


def main() -> None:
    """실험 원천을 확인한 뒤 재계산 결과만 보고한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    spec = exp.protocol(True)
    for root, script in ((ref.OUT, ref.__file__), (channel.OUT, channel.__file__), (arc.OUT, arc.__file__)):
        p = exp.read(root / "protocol.json")
        assert p["script_sha256"] == sha256(Path(script))
        assert p["input_protocol_sha256"] == sha256(exp.OUT / "protocol.json")
        assert exp.read(root / "status.json")["status"] == "complete"
    correction = exp.read(NEG / "comment_correction.json")
    snapshot, current = NEG / "runtime_snapshot.py", REPO / "scripts/_pdv2_nsd_negative.py"
    assert sha256(snapshot) == correction["before_sha256"] == exp.read(NEG / "protocol.json")["script_sha256"]
    assert sha256(current) == correction["after_sha256"]
    def executable(path: Path) -> str:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
                if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    node.body.pop(0)
        return ast.dump(tree)
    assert executable(snapshot) == executable(current)
    for name in ("completion_validation", "control_validation"):
        assert exp.read(LOSS / (name+".json"))["status"] == "passed"
    loss_protocol = exp.read(LOSS / "protocol.json")
    for field in ("runtime_sha256", "input_sha256"):
        for name, expected in loss_protocol[field].items():
            assert sha256(REPO / name) == expected, name
    data, counts, hashes = audit_ledgers(spec)
    print("LEDGERS_PASSED", counts, flush=True)
    prefixes = prefix_audit(spec)
    negative = negative_audit(spec)
    summary = {"experiments": data, "negative": negative, "prefix": prefixes,
               "loss_summary": exp.read(LOSS / "summary.json"), "loss_expansion": exp.read(LOSS / "expansion.json"),
               "test_used": False, "order": "신호처리→진폭보존혼합→채널교차·거리대응. 후속조건은앞결과를본뒤고정한탐색이며독립확증아님."}
    write_json(OUT / "combined_summary.json", summary)
    with (OUT / "final_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "model", "channel", "method", "metric", "mean_percent", "sample_sd_percent", "seed1_percent", "seed2_percent", "seed3_percent"])
        for stage, seeds in data.items():
            for model, channels in seeds["1"].items():
                for ch, methods in channels.items():
                    for method in methods:
                        for metric in exp.METRICS:
                            v = [seeds[s][model][ch][method]["macro"][metric]*100 for s in seeds]
                            writer.writerow([stage, model, ch, method, metric, np.mean(v), np.std(v, ddof=1), *v])
    figures(data, negative)
    write_json(OUT / "completion_validation.json", {"status": "passed", "counts": counts,
        "prefix_query_checks": len(prefixes["checks"]), "reference_and_arc_prefix_max_error": 0.,
        "saved_batch_vs_single_max_error": prefixes["saved_batch_vs_single_max_error"],
        "arc_frame5_exactly_reproduces_reference_blend25": True, "same_final_queries_and_winners": True,
        "negative_docstring_only_change_verified": True, "source_hashes": hashes,
        "report_script_sha256": sha256(Path(__file__)), "test_used": False})
    print("COMPLETE", negative["total"], flush=True)


if __name__ == "__main__":
    main()
