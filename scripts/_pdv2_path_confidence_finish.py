#!/usr/bin/env python3
"""경로 판정 수정의 최종 원장·점수·관측 범위와 손익을 검산한다."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_path_confidence as run
from _pdv2_candidate_paths_finish import independent_metrics, scalar_path
from _pdv2_temporal_test_control_finish import scalar_score
from _pdv2_no_context_ablation import sha256, write_json


def average(rows: dict) -> dict:
    """양성이 있는 시퀀스→계열을 별도로 평균한다."""
    groups = {"kitti": ("kitti/", "kitti360/"), "helipr": ("helipr/",),
              "mulran": ("mulran/",)}
    families = {f: {m: float(np.mean([r[m] for s, r in rows.items()
                 if s.startswith(prefix) and r["n_pos"] > 0]))
                 for m in run.old.exp.METRICS} for f, prefix in groups.items()}
    return {"families": families, "macro": {m: float(np.mean([r[m] for r in families.values()]))
                                             for m in run.old.exp.METRICS}}


def main() -> None:
    """저장된 모든 조건과 접두 표본의 최종 점수를 독립 계산한다."""
    spec = run.protocol(True)
    summary = run.read(run.OUT / "summary.json")
    prior = run.old.protocol()
    sources = {**spec["runtime"], **spec["inputs"]}
    metrics_error = score_error = prefix_score_error = 0.
    ledgers = rows = boundaries = prefix_count = scalar_count = 0
    diagnosis = {}
    for seed in run.CFG["seeds"]:
        diagnosis[str(seed)] = {}
        conf = run.read(run.old.exp.old.base.BASE / f"s{seed}/confidence_selection.json")["ops"][run.old.exp.OP]
        for model in run.CFG["models"]:
            aggregate = {a: {} for a in run.CFG["arms"]}
            detail = {}
            for sequence in spec["val"]:
                p = run.bundle_path(seed, model, sequence)
                bundle = run.old.exp.arrays(p)
                record = run.read(p.with_suffix(".json"))
                assert sha256(p) == record["sha256"]
                for path in (p, p.with_suffix(".json")):
                    sources[str(path.relative_to(REPO))] = sha256(path)
                name = sequence.replace("/", "_")
                references = {a: run.old.exp.arrays(run.old.OUT / f"s{seed}/{model}_{b}_{name}.npz")
                              for a, b in (("baseline", "baseline"), ("forced", "joint_forced"),
                                           ("hard_null", "joint_null"))}
                baseline, forced = references["baseline"], references["forced"]
                for arm, expected in references.items():
                    for key, value in run.row(bundle, arm).items():
                        np.testing.assert_array_equal(value, expected[key])
                for arm in run.CFG["arms"][3:]:
                    actual = run.row(bundle, arm)
                    for key in ("j_star", "top1_geo_dist"):
                        np.testing.assert_array_equal(actual[key], forced[key])
                # 보정 함수를 부르지 않고 math.exp로 독립 계산한다.
                original, selected = baseline["score"], forced["score"]
                factor = np.array([math.exp(min(float(x), 0.)) for x in bundle["path_margin"]])
                cap = np.array([min(float(x), float(y)) for x, y in zip(original, selected)])
                expected_scores = {"baseline_score": original, "soft_null": selected*factor,
                                   "capped": cap, "soft_capped": cap*factor}
                for arm, expected in expected_scores.items():
                    error = float(abs(run.row(bundle, arm)["score"]-expected).max())
                    score_error = max(score_error, error)
                    assert error < 5e-15
                sensor, seq, paths, _ = run.old.exp.old.base.inputs("val", sequence)
                cache = run.old.exp.arrays(paths["cache"])
                gt, xyz = cache["poses_gt"][:, :3, 3], cache["poses"][:, :3, 3]
                travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
                q = bundle["query_idx"]
                np.testing.assert_array_equal(q, np.arange(30, len(gt)-15))
                positive = bundle["min_past_gt_dist"] <= 5
                eligible = positive | (bundle["min_past_gt_dist"] > 25)
                arms = {}
                for arm in run.CFG["arms"]:
                    a = run.row(bundle, arm)
                    valid = a["j_star"] >= 0
                    assert np.array_equal(np.isfinite(a["score"]), valid)
                    assert np.array_equal(np.isfinite(a["top1_geo_dist"]), valid)
                    assert (a["j_star"][valid] <= q[valid]-30).all()
                    np.testing.assert_allclose(a["top1_geo_dist"][valid],
                        np.linalg.norm(gt[q[valid]]-gt[a["j_star"][valid]], axis=1), atol=1e-10, rtol=0)
                    measured = independent_metrics(a)
                    for m, value in measured.items():
                        expected = record["metrics"][arm][m]
                        if value is None:
                            assert expected is None
                        else:
                            metrics_error = max(metrics_error, abs(value-expected))
                            assert abs(value-expected) < 5e-13
                    correct = positive & (a["top1_geo_dist"] <= 5)
                    wrong = eligible & valid & ~correct
                    # 최초 오답과 동점인 정답은 precision=1 구간에 넣지 않는다.
                    high = float(a["score"][wrong].max()) if wrong.any() else -np.inf
                    strict = int((correct & (a["score"] > high)).sum())
                    r100 = strict / max(int(positive.sum()), 1)
                    assert abs(r100-measured["R@100P"]) < 1e-14
                    arms[arm] = {**record["diagnostics"][arm], "before_first_wrong": strict,
                                 "R100_boundary_score": high}
                    aggregate[arm][sequence] = measured
                    ledgers += 1
                    rows += len(q)
                    boundaries += 1
                # 고정 키의 실제 접두 후보·경로·스펙트럼으로 새 점수를 확인한다.
                # 키 생성 자체의 108개 접두 재실행은 이전 감사와 그 해시를 재사용한다.
                selected_model = prior["selected"][model][str(seed)]
                key = run.old.exp.arrays(REPO / selected_model["folder"] /
                      f"keys_val_{sensor}_{seq}.npz")["embedding"]
                cb = run.old.exp.arrays(run.old.OUT / f"s{seed}/candidates_{model}_{name}.npz")
                with np.load(paths["spec"]) as z:
                    bank = run.old.exp.old.base.bank_for(run.old.exp.OP, z)
                probes = {len(q)//3, 2*len(q)//3}
                discarded = positive & (forced["top1_geo_dist"] <= 5) & (bundle["path_margin"] < 0)
                if discarded.any():
                    probes.add(int(np.flatnonzero(discarded)[0]))
                wrong = eligible & ~(forced["top1_geo_dist"] <= 5)
                if wrong.any():
                    probes.add(int(np.flatnonzero(wrong)[np.argmax(forced["score"][wrong])]))
                prefix_records = []
                for ix in sorted(probes):
                    t, end = int(q[ix]), int(q[ix]+16)
                    small = run.old.exp.truncate_bank(bank, end)
                    local = {k: v[:end].copy() for k, v in cb.items()}
                    raw_by_id = {}
                    for qt in range(t-5, t+6):
                        jj, uu, rr = run.old.candidates(key[:end], small, conf["tuple"], conf["decision"], qt)
                        count = min(20, len(jj))
                        local["indices"][qt, :count] = jj[:count]
                        local["unary"][qt, :count] = uu[:count]
                        if qt == t:
                            assert int(jj[0]) == int(baseline["j_star"][ix])
                            raw_by_id = dict(zip(jj.tolist(), rr.tolist()))
                    new, joint, margin = scalar_path(local, travel[:end], t)
                    assert new == int(forced["j_star"][ix])
                    assert joint == int(references["hard_null"]["j_star"][ix])
                    assert abs(margin-bundle["path_margin"][ix]) < 2e-7
                    bs = scalar_score(small, travel[:end], t, int(baseline["j_star"][ix]),
                                      conf["decision"], raw_by_id[int(baseline["j_star"][ix])])
                    fs = scalar_score(small, travel[:end], t, new, conf["decision"], raw_by_id[new])
                    predicted = {"baseline_score": bs, "soft_null": fs*math.exp(min(margin, 0.)),
                                 "capped": min(bs, fs), "soft_capped": min(bs, fs)*math.exp(min(margin, 0.))}
                    for arm, value in predicted.items():
                        error = abs(value-run.row(bundle, arm)["score"][ix])
                        prefix_score_error = max(prefix_score_error, error)
                        assert error < 3e-7
                    scalar_count += 2
                    prefix_count += 1
                    prefix_records.append({"q": t, "forced": new, "hard_null": joint,
                                           "margin": margin, "old_score": bs, "new_score": fs})
                detail[sequence] = {"arms": arms, "mechanism": record["mechanism"],
                                    "prefix_probes": prefix_records}
            for arm in run.CFG["arms"]:
                actual = average(aggregate[arm])
                expected = summary["per_seed"][str(seed)][model][arm]
                for field in ("macro", "families"):
                    a = actual[field] if field == "macro" else {f+"/"+m: v for f, d in actual[field].items() for m, v in d.items()}
                    b = expected[field] if field == "macro" else {f+"/"+m: v for f, d in expected[field].items() for m, v in d.items()}
                    for k in a:
                        assert abs(a[k]-b[k]) < 5e-13
            diagnosis[str(seed)][model] = detail
            print("AUDITED", seed, model, flush=True)
    write_json(run.OUT / "diagnostics.json", {"per_seed": diagnosis, "test_used": False,
        "scope": "저장검증 전원장 PR·수식·정답거리, 고정키의 접두 후보와 스칼라경로·최종판정 재계산. 키생성 접두108개는 해시검증한 이전실험 감사 재사용."})
    plot(summary)
    for p in (Path(__file__), run.OUT / "protocol.json", run.OUT / "summary.json",
              run.OUT / "diagnostics.json", run.OUT / "final_metrics.csv",
              run.OUT / "path_confidence.png", run.OUT / "path_confidence.pdf",
              REPO / "results/pdv2_sequence_report_20261018/UnDotum.ttf"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    audit = {"status": "passed", "bundles": 54, "ledgers": ledgers, "rows": rows,
             "metric_sets": ledgers, "metric_max_error": metrics_error,
             "score_formula_max_error": score_error, "first_wrong_boundaries": boundaries,
             "fixed_key_prefixes": prefix_count, "scalar_final_scores": scalar_count,
             "prefix_score_max_error": prefix_score_error, "prefix_winner_changes": 0,
             "reused_key_regenerated_prefixes": 108, "sources": sources, "test_used": False}
    write_json(run.OUT / "completion_validation.json", audit)
    print("VERIFIED", {k: v for k, v in audit.items() if k != "sources"}, flush=True)


def plot(summary: dict) -> None:
    """조건별 최종 AP·R100을 같은 축에서 그린다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=str(REPO / "results/pdv2_sequence_report_20261018/UnDotum.ttf"))
    labels = ["기존 선택", "경로\n중심 거절", "경로\n중심 강제", "기존 점수\n이전", "완만한\n감점", "점수\n상한", "감점+상한\n주조건"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, metric in zip(axes, ("AP", "R@100P")):
        for model, color, offset in (("standard", "#277b8b", -.19), ("capacity_point", "#d99535", .19)):
            values = [100*summary["mean_std"][model][a][metric]["mean"] for a in run.CFG["arms"]]
            bars = ax.bar(np.arange(7)+offset, values, .35, color=color,
                          label="표준 시간 모델" if model == "standard" else "이웃 없는 학습 투영")
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=3)
        ax.set_xticks(np.arange(7), labels, fontproperties=font, fontsize=9)
        ax.set_ylabel(metric+" (%)")
        ax.set_ylim(0, 100 if metric == "AP" else 62)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(prop=font, loc="upper left", ncol=2)
    fig.suptitle("후보 경로의 최종 판정 수정 · 검증 9 · 3시드 평균", fontproperties=font, fontsize=14)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(run.OUT / ("path_confidence."+extension), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
