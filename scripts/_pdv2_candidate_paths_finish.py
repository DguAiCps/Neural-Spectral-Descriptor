#!/usr/bin/env python3
"""후보 경로 실험의 최종 지표·경로·관측 접두와 손익을 독립 검산한다."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_candidate_paths as run
from _pdv2_no_context_ablation import sha256, write_json
from _pdv2_temporal_test_control_finish import scalar_score
from _pdv2_temporal_capacity import create_capacity_model
from _pdv2_pair_evaluate import ArrayCache
from gnn.model import create_spectral_gnn
from _pr_eval_lib import classify_queries, operating_point


def independent_metrics(row: dict) -> dict:
    """NaN 거절과 동점을 처리하면서 생산 PR 함수 없이 지표를 구한다."""
    positive = row["min_past_gt_dist"] <= 5
    valid = (positive | (row["min_past_gt_dist"] > 25)) & np.isfinite(row["score"])
    correct = positive & (row["top1_geo_dist"] <= 5)
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(-row["score"][indices], kind="stable")]
    if len(indices):
        stop = np.flatnonzero(np.r_[np.diff(row["score"][indices]) != 0, True])
        tp = np.r_[0, np.cumsum(correct[indices])[stop]]
        fp = np.r_[0, np.cumsum(~correct[indices])[stop]]
    else:
        tp = fp = np.array([0])
    precision = np.divide(tp, np.maximum(tp+fp, 1)).astype(float)
    precision[tp+fp == 0] = 1.
    recall = tp/max(positive.sum(), 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return {"AP": float(np.diff(recall) @ envelope[1:]),
            "max_F1": float(np.max(2*precision*recall/np.maximum(precision+recall, 1e-300))),
            "R@100P": float(np.max(recall[fp == 0])),
            "R@1": float(correct.sum()/positive.sum()) if positive.any() else None,
            "n_pos": int(positive.sum())}


def scalar_path(cb: dict, path: np.ndarray, q: int) -> tuple:
    """21개 상태의 각 전이를 직접 계산하는 독립 최대합 대조다."""
    k = cb["indices"].shape[1]
    ids = cb["indices"][q-5:q+6]
    unary = np.c_[cb["unary"][q-5:q+6], np.full(11, .5)]
    all_scores = []
    for direction in (1, -1):
        costs = []
        for t in range(10):
            step = path[q-4+t]-path[q-5+t]
            cost = np.zeros((k+1, k+1))
            for a in range(k+1):
                for b in range(k+1):
                    if a == k or b == k:
                        cost[a, b] = 0. if a == b else .15
                    elif ids[t, a] < 0 or ids[t+1, b] < 0:
                        cost[a, b] = np.inf
                    else:
                        delta = direction*(path[ids[t+1, b]]-path[ids[t, a]])
                        cost[a, b] = np.inf if delta < -.8 or delta > 3*step+3.2 else .25*min(abs(delta-step)/max(step, .8), 3.)
            costs.append(cost)
        left = unary[0].copy()
        for t in range(1, 6):
            left = unary[t] + np.max(left[:, None]-costs[t-1], axis=0)
        right = unary[-1].copy()
        for t in range(9, 4, -1):
            right = unary[t] + np.max(right[None, :]-costs[t], axis=1)
        score = left+right-np.where(np.isfinite(unary[5]), unary[5], 0.)
        all_scores.append(score)
    best = np.maximum(*all_scores)
    forced = int(np.argmax(best[:k]))
    joint = int(np.argmax(best))
    return int(ids[5, forced]), (-1 if joint == k else int(ids[5, joint])), float(best[forced]-best[-1])


def load_model(model: str, seed: int, epoch: int) -> tuple:
    """과거 실행기의 전역 설정 변경 없이 고정 가중치를 읽는다."""
    cfg = yaml.safe_load((run.exp.old.OUT / ("standard.yaml" if model == "standard" else "diff.yaml")).read_text())
    factory = create_spectral_gnn if model == "standard" else create_capacity_model
    net = factory(input_dim=288, hidden_dim=128, context_dim=128, n_layers=2, n_heads=4,
                  dropout=.1, norm_type="batch_norm", use_residual_gate=True,
                  gate_hidden_dim=64, gate_initial_alpha=.0625,
                  diffattn_value_source=cfg["gnn"]["diffattn_value_source"],
                  edge_encoder_config=cfg["gnn"]["edge_encoding"],
                  key_remetrize_config={"enabled": True, "learnable": False,
                                        "init_path": str(run.exp.old.base.RVEC)}).cuda().eval()
    ck = (run.exp.old.checkpoint_dir(seed, "standard_original") if model == "standard"
          else REPO / f"checkpoints/pdv2_temporal_capacity_s{seed}_point_original")
    state = torch.load(ck / f"checkpoint_epoch_{epoch}.pth", map_location="cpu",
                       weights_only=False)["model_state_dict"]
    missing, extra = net.load_state_dict(state, strict=False)
    assert missing == ["gnn.key_scale"] and not extra
    return cfg, net


def main() -> None:
    """모든 원장을 검산하고 설명용 표·그림·방향별 진단을 저장한다."""
    spec = run.protocol(True)
    summary = run.exp.read(run.OUT / "summary.json")
    assert summary["protocol_sha256"] == sha256(run.OUT / "protocol.json")
    sources = {**spec["runtime"], **spec["inputs"]}
    ledgers = rows_count = metric_sets = scalar_count = path_count = prefix_count = 0
    max_metric_error = max_scalar_error = max_key_error = 0.
    prefix_winner_changes = 0
    diagnostics, records_csv, prefix_records = {}, [], []
    for seed in summary["per_seed"]:
        s = int(seed)
        tau = run.fixed.choose_tau(s)["tau"]
        diagnostics[seed] = {}
        for model in run.CFG["models"]:
            aggregate = {arm: {} for arm in run.CFG["arms"]}
            details = {}
            selected = spec["selected"][model][seed]
            cfg, net = load_model(model, s, selected["epoch"])
            for sequence in spec["val"]:
                name = sequence.replace("/", "_")
                sensor, seq, paths, _ = run.exp.old.base.inputs("val", sequence)
                cache = run.exp.arrays(paths["cache"])
                xyz = cache["poses"][:, :3, 3]
                gt = cache["poses_gt"][:, :3, 3]
                travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
                with np.load(paths["spec"]) as z:
                    bank = run.exp.old.base.bank_for(run.exp.OP, z)
                conf = run.exp.read(run.exp.old.base.BASE / f"s{s}/confidence_selection.json")["ops"][run.exp.OP]
                cp = run.OUT / f"s{s}/candidates_{model}_{name}.npz"
                cb = run.exp.arrays(cp)
                meta_path = run.OUT / f"s{s}/{name}.json"
                meta = run.exp.read(meta_path)["models"][model]
                assert sha256(cp) == meta["candidate_sha256"]
                for p in (cp, meta_path):
                    sources[str(p.relative_to(REPO))] = sha256(p)
                keydata = run.exp.arrays(REPO / selected["folder"] / f"keys_val_{sensor}_{seq}.npz")
                key = keydata["embedding"]
                baseline = run.exp.arrays(run.OUT / f"s{s}/{model}_baseline_{name}.npz")
                q = baseline["query_idx"]
                positive = baseline["min_past_gt_dist"] <= 5
                correct_base = positive & (baseline["top1_geo_dist"] <= 5)
                detail = {"positive": int(positive.sum()),
                          "union_reachable": int(cb["union_hit"][q[positive]].sum()),
                          "shortlist_reachable": int(cb["shortlist_hit"][q[positive]].sum()),
                          "timing_s": meta["timing_s"], "arms": {}}
                arms = {}
                for arm in run.CFG["arms"]:
                    p = run.OUT / f"s{s}/{model}_{arm}_{name}.npz"
                    a = run.exp.arrays(p)
                    arms[arm] = a
                    assert sha256(p) == meta["arms"][arm]["sha256"]
                    sources[str(p.relative_to(REPO))] = sha256(p)
                    for field in ("query_idx", "min_past_gt_dist", "group"):
                        np.testing.assert_array_equal(a[field], baseline[field])
                    valid = a["j_star"] >= 0
                    assert np.array_equal(np.isnan(a["score"]), ~valid)
                    assert np.array_equal(np.isnan(a["top1_geo_dist"]), ~valid)
                    assert np.array_equal(np.isnan(a["raw"]), ~valid)
                    assert np.all(a["j_star"][valid] <= q[valid]-30)
                    np.testing.assert_allclose(a["top1_geo_dist"][valid],
                        np.linalg.norm(gt[q[valid]]-gt[a["j_star"][valid]], axis=1), atol=1e-10, rtol=0)
                    assert np.all((a["j_star"][valid, None] == cb["indices"][q[valid]]).any(1))
                    m = independent_metrics(a)
                    for metric, value in m.items():
                        expected = meta["arms"][arm]["metrics"][metric]
                        if value is None:
                            assert expected is None
                        else:
                            error = abs(value-expected)
                            max_metric_error = max(max_metric_error, error)
                            assert error < 5e-13, (sequence, model, arm, metric)
                    aggregate[arm][sequence] = m
                    correct = positive & (a["top1_geo_dist"] <= 5)
                    gains, losses = correct & ~correct_base, correct_base & ~correct
                    detail["arms"][arm] = {
                        "correct": int(correct.sum()), "gains": int(gains.sum()), "losses": int(losses.sum()),
                        "null": int((~valid).sum()), "positive_null": int((positive & ~valid).sum()),
                        "winner_changes": int((a["j_star"] != baseline["j_star"]).sum()),
                        "directions": {str(g): {"n": int((positive & (a["group"] == g)).sum()),
                            "correct": int((correct & (a["group"] == g)).sum()),
                            "gains": int((gains & (a["group"] == g)).sum()),
                            "losses": int((losses & (a["group"] == g)).sum())} for g in (0, 1, 2)},
                        "fixed_tau": operating_point(a["score"], a["top1_geo_dist"],
                                                     classify_queries(a["min_past_gt_dist"]), tau)}
                    eligible = positive | (a["min_past_gt_dist"] > 25)
                    incorrect = eligible & valid & ~(a["top1_geo_dist"] <= 5)
                    if incorrect.any():
                        first = np.flatnonzero(incorrect)[np.argmax(a["score"][incorrect])]
                        detail["arms"][arm]["first_wrong"] = {
                            "q": int(q[first]), "j": int(a["j_star"][first]),
                            "distance": float(a["top1_geo_dist"][first]), "score": float(a["score"][first]),
                            "positive": bool(positive[first])}
                    for ix in np.linspace(0, len(q)-1, 5, dtype=int):
                        if not valid[ix]:
                            continue
                        score = scalar_score(bank, travel, int(q[ix]), int(a["j_star"][ix]),
                                             conf["decision"], float(a["raw"][ix]))
                        error = abs(score-a["score"][ix])
                        max_scalar_error = max(max_scalar_error, error)
                        assert error < 2e-7
                        scalar_count += 1
                    metric_sets += 1
                    ledgers += 1
                    rows_count += len(q)
                for ix in np.linspace(0, len(q)-1, 5, dtype=int):
                    forced, joint, margin = scalar_path(cb, travel, int(q[ix]))
                    assert forced == arms["joint_forced"]["j_star"][ix]
                    assert joint == arms["joint_null"]["j_star"][ix]
                    np.testing.assert_allclose(margin, arms["joint_null"]["path_margin"][ix], atol=2e-12, rtol=0)
                    path_count += 1
                for ix in (len(q)//3, 2*len(q)//3):
                    t, end = int(q[ix]), int(q[ix]+16)
                    tiny = ArrayCache({k: v[:end] for k, v in cache.items()})
                    small_key, graph = run.exp.old.old.embedding(
                        tiny, sensor, tiny["descriptors"].astype(np.float32), cfg, net,
                        "temporal" if model == "standard" else "point")
                    error = float(abs(small_key[:t+6]-key[:t+6]).max())
                    max_key_error = max(max_key_error, error)
                    np.testing.assert_allclose(small_key[:t+6], key[:t+6], atol=4e-6, rtol=4e-5)
                    small_cb = {k: v[:end].copy() for k, v in cb.items()}
                    for qt in range(t-5, t+6):
                        jj, uu, rr = run.candidates(small_key, run.exp.truncate_bank(bank, end),
                                                   conf["tuple"], conf["decision"], qt)
                        m = min(run.CFG["candidate_top_k"], len(jj))
                        small_cb["indices"][qt, :m] = jj[:m]
                        small_cb["unary"][qt, :m] = uu[:m]
                    p = run.path_selection(small_cb, travel[:end], np.array([t]))
                    changed = int(p["joint"][0] != arms["joint_null"]["j_star"][ix])
                    prefix_winner_changes += changed
                    prefix_records.append({"seed": s, "model": model, "sequence": sequence,
                                           "q": t, "key_max_error": error, "winner_changed": changed})
                    prefix_count += 1
                details[sequence] = detail
            del net
            for arm in run.CFG["arms"]:
                actual = run.macro(aggregate[arm])
                expected = summary["per_seed"][seed][model][arm]
                for metric in run.exp.METRICS:
                    np.testing.assert_allclose(actual["macro"][metric], expected["macro"][metric], atol=5e-13)
                records_csv.append({"seed": seed, "model": model, "arm": arm,
                                    **{m: 100*actual["macro"][m] for m in run.exp.METRICS}})
            diagnostics[seed][model] = {"sequences": details, "fixed_tau": tau}
    with (run.OUT / "final_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "model", "arm", *run.exp.METRICS])
        writer.writeheader()
        writer.writerows(records_csv)
    write_json(run.OUT / "diagnostics.json", {"per_seed": diagnostics, "test_used": False,
        "scope": "방향별 수는 질의 통합값. 상한은GT사후진단이며모델입력아님. 같은기존standard검증F1임계값을고정.",
        "group_definition": "0=유효한같은방향정답존재,1=가능한모든과거정답이유효한반대방향,2=나머지"})
    plot(records_csv)
    for p in (Path(__file__), REPO / "scripts/_pdv2_temporal_test_control_finish.py",
              run.OUT / "protocol.json", run.OUT / "summary.json", run.OUT / "diagnostics.json",
              run.OUT / "final_metrics.csv", run.OUT / "candidate_paths.png", run.OUT / "candidate_paths.pdf"):
        sources[str(p.relative_to(REPO))] = sha256(p)
    write_json(run.OUT / "completion_validation.json", {
        "status": "passed", "ledgers": ledgers, "rows": rows_count, "metric_sets": metric_sets,
        "metric_max_error": max_metric_error, "scalar_final_scores": scalar_count,
        "scalar_score_max_error": max_scalar_error, "scalar_paths": path_count,
        "regenerated_prefixes": prefix_count, "prefix_key_max_error": max_key_error,
        "prefix_winner_changes": prefix_winner_changes, "prefix_records": prefix_records,
        "sources": sources, "test_used": False})
    print("VERIFIED", ledgers, rows_count, scalar_count, path_count, prefix_count,
          "PREFIX_WINNER_CHANGES", prefix_winner_changes, flush=True)


def plot(rows: list) -> None:
    """검증 시드별 최종 AP와 높은 정밀도 재현율의 비교 그림을 만든다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=str(REPO / "results/pdv2_sequence_report_20261018/UnDotum.ttf"))
    labels = {"baseline": "기존 선택", "unary_null": "단일 선택·거절",
              "joint_forced": "경로·중심 강제", "joint_null": "경로·거절 허용"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, metric in zip(axes, ("AP", "R@100P")):
        for model, color, offset in (("standard", "#24788c", -.18), ("capacity_point", "#e2a143", .18)):
            values = [np.mean([r[metric] for r in rows if r["model"] == model and r["arm"] == arm])
                      for arm in run.CFG["arms"]]
            bars = ax.bar(np.arange(4)+offset, values, width=.34, color=color,
                          label="표준 시간 모델" if model == "standard" else "이웃 없는 학습 투영")
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set_xticks(np.arange(4), [labels[a] for a in run.CFG["arms"]], fontproperties=font, fontsize=9)
        ax.set_ylabel(metric+" (%)")
        ax.set_ylim(0, max([r[metric] for r in rows])*1.18)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(prop=font, fontsize=8)
    fig.suptitle(f"후보 대응 시간 제약: 최종 검증 지표, {len({r['seed'] for r in rows})}시드", fontproperties=font)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(run.OUT / ("candidate_paths."+extension), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
