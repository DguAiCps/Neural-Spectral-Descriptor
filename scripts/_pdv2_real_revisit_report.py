#!/usr/bin/env python3
"""실제 재방문 학습의 최종 성능·방향별 손익·학습 쌍 구분력을 보고한다."""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_real_revisit as run
from _pdv2_pair_evaluate import direction_groups
from _pdv2_temporal_focus_diagnose import ap_parts, CATEGORIES
from _pdv2_no_context_ablation import sha256, write_json


def unit(x: np.ndarray) -> np.ndarray:
    """배정밀도에서 행별 코사인 입력을 만든다."""
    x = x.astype(np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def auxiliary_scores(key: np.ndarray, pool: dict, context: bool) -> dict:
    """고정된 모든 실제 정답 각각을 모든 오답과 독립 비교한다."""
    value = unit(key[:, 288:] if context else key)
    q, pp, nn = pool["anchors"], pool["positives"], pool["negatives"]
    pv, nv = pp >= 0, nn >= 0
    ps = np.einsum("nd,nkd->nk", value[q], value[pp.clip(0)])
    ns = np.einsum("nd,nkd->nk", value[q], value[nn.clip(0)])
    negative = np.where(nv, ns, -np.inf)
    loss = np.logaddexp(0.0, logsumexp(negative / 0.1, axis=1)[:, None] - ps / 0.1)
    margins = ps - negative.max(1)[:, None]
    per_query = {
        "loss": np.where(pv, loss, 0).sum(1) / pv.sum(1),
        "margin": np.where(pv, margins, 0).sum(1) / pv.sum(1),
        "all_positive_win": np.all((margins > 0) | ~pv, axis=1),
        "any_positive_win": np.any((margins > 0) & pv, axis=1),
    }
    return {k: float(v.mean()) for k, v in per_query.items()}


def main() -> None:
    """최종 원장 검산에 방향·보조 학습 대조와 도표를 더한다."""
    spec = run.protocol(True)
    summary = run.base.read(run.OUT / "summary.json")
    audit = run.base.read(run.OUT / "completion_validation.json")
    auxiliary_audit = run.base.read(run.OUT / "auxiliary_validation.json")
    assert audit["passed"] and auxiliary_audit["passed"]
    sources = {**audit["sources"], **auxiliary_audit["sources"]}
    pool = run.base.exp.arrays(run.OUT / "auxiliary_pool.npz")
    arms = ["start"] + run.CFG["arms"]
    table = []
    auxiliary = {}
    for arm in arms:
        if arm == "start":
            key = pool["initial_keys"]
        else:
            epoch = summary["selected_epoch"][arm]
            key = np.concatenate(
                [
                    run.base.exp.arrays(
                        run.OUT
                        / arm
                        / f"epoch{epoch}"
                        / "train"
                        / ("keys_" + s.replace("/", "_") + ".npz")
                    )["embedding"]
                    for s in spec["train"]
                ]
            )
        auxiliary[arm] = {
            mode: auxiliary_scores(key, pool, mode == "context")
            for mode in ("key", "context")
        }
        for role in ("train", "val"):
            table.append(
                {
                    "arm": arm,
                    "role": role,
                    "epoch": 0 if arm == "start" else summary["selected_epoch"][arm],
                    **{
                        k: 100 * v
                        for k, v in summary["results"][role][arm]["macro"].items()
                    },
                }
            )
    with (run.OUT / "final_metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    # 학습 전 교정 손실은 질의 수 가중을 적용했으므로 전체 질의 독립 계산과 일치한다.
    calibration = run.base.read(run.OUT / "calibration.json")
    for mode in ("key", "context"):
        expected = np.mean([v[mode + "_loss"] for v in calibration["batches"]])
        assert abs(expected - auxiliary["start"][mode]["loss"]) < 5e-5, (
            mode,
            expected,
            auxiliary["start"][mode],
        )
    by_direction = {}
    gate_stats = {}
    for role in ("train", "val"):
        gate_values = {arm: [] for arm in arms}
        totals = {
            arm: {
                str(g): dict(positive=0, old_correct=0, new_correct=0, gained=0, lost=0)
                for g in range(3)
            }
            for arm in run.CFG["arms"]
        }
        for sequence in spec[role]:
            old = run.base.exp.arrays(
                run.factor.OUT / role / sequence.replace("/", "_") / "new.npz"
            )
            q = old["query_idx"]
            cache = run.base.exp.arrays(run.factor.paths_for(role, sequence)["cache"])
            pose = cache["poses" if role == "train" else "poses_gt"]
            past = np.full(len(pose), np.inf)
            past[q] = old["min_past_gt_dist"]
            group = direction_groups(pose[:, :3, 3].astype(np.float64), past)[q]
            first = old["top1_geo_dist"] <= 5
            initial_key = run.base.exp.arrays(
                run.factor.key_path(role, "interaction", sequence)
            )["embedding"]
            gate_values["start"].append(
                np.linalg.norm(initial_key[q, 288:], axis=1)
                / np.linalg.norm(initial_key[q, :288], axis=1)
            )
            for arm in run.CFG["arms"]:
                epoch = summary["selected_epoch"][arm]
                new = run.base.exp.arrays(
                    run.OUT
                    / arm
                    / f"epoch{epoch}"
                    / role
                    / (sequence.replace("/", "_") + ".npz")
                )
                np.testing.assert_array_equal(new["query_idx"], q)
                key_path = (
                    run.OUT
                    / arm
                    / f"epoch{epoch}"
                    / role
                    / ("keys_" + sequence.replace("/", "_") + ".npz")
                )
                current_key = run.base.exp.arrays(key_path)["embedding"]
                np.testing.assert_allclose(
                    unit(current_key[:, :288]),
                    unit(initial_key[:, :288]),
                    atol=4e-6,
                    rtol=4e-5,
                )
                alpha = np.linalg.norm(current_key[q, 288:], axis=1) / np.linalg.norm(
                    current_key[q, :288], axis=1
                )
                assert np.all(alpha >= 0) and np.all(alpha <= 1.00001)
                gate_values[arm].append(alpha)
                second = new["top1_geo_dist"] <= 5
                for g in range(3):
                    mask = group == g
                    row = {
                        "positive": int(mask.sum()),
                        "old_correct": int((mask & first).sum()),
                        "new_correct": int((mask & second).sum()),
                        "gained": int((mask & ~first & second).sum()),
                        "lost": int((mask & first & ~second).sum()),
                    }
                    for k, v in row.items():
                        totals[arm][str(g)][k] += v
        for arm in run.CFG["arms"]:
            epoch = summary["selected_epoch"][arm]
            reference = summary["candidate_transitions"][arm][role][str(epoch)]
            for k in ("gained", "lost"):
                assert sum(v[k] for v in totals[arm].values()) == reference[k]
        by_direction[role] = totals
        gate_stats[role] = {
            arm: {
                "mean": float(np.concatenate(v).mean()),
                "p05": float(np.quantile(np.concatenate(v), 0.05)),
                "p95": float(np.quantile(np.concatenate(v), 0.95)),
            }
            for arm, v in gate_values.items()
        }
    prior = run.base.read(
        REPO / "results/pdv2_temporal_augmentation_20261024/summary.json"
    )
    control = {
        m: 100
        * (
            summary["results"]["val"]["clean_rank"]["macro"][m]
            - prior["results"]["val"]["clean_rank"]["macro"][m]
        )
        for m in run.base.exp.METRICS
    }
    decomposition = {}
    for arm in ("pair_key", "pair_context"):
        rows = {}
        for sequence in spec["val"]:
            ledgers = []
            for name in ("clean_rank", arm):
                epoch = summary["selected_epoch"][name]
                path = (
                    run.OUT
                    / name
                    / f"epoch{epoch}"
                    / "val"
                    / (sequence.replace("/", "_") + ".npz")
                )
                ledgers.append(run.base.exp.arrays(path))
            row = ap_parts(*ledgers, run.METHOD)
            row["family"] = (
                "kitti" if sequence.startswith("kitti") else sequence.split("/")[0]
            )
            rows[sequence] = row
        families = {
            f: {
                k: float(
                    np.mean(
                        [v[k] for v in rows.values() if v["n_pos"] and v["family"] == f]
                    )
                )
                for k in ("before", "after", *CATEGORIES)
            }
            for f in ("kitti", "helipr", "mulran")
        }
        macro = {
            k: float(np.mean([v[k] for v in families.values()]))
            for k in ("before", "after", *CATEGORIES)
        }
        assert (
            abs(
                macro["before"] - summary["results"]["val"]["clean_rank"]["macro"]["AP"]
            )
            < 1e-12
        )
        assert (
            abs(macro["after"] - summary["results"]["val"][arm]["macro"]["AP"]) < 1e-12
        )
        decomposition[arm] = {"macro": macro, "families": families, "sequences": rows}
    report = {
        "final_metrics_percent": table,
        "auxiliary_pair_metrics": auxiliary,
        "direction_transitions": by_direction,
        "context_gate_on_queries": gate_stats,
        "validation_AP_accounting_vs_clean": decomposition,
        "coefficients": auxiliary_audit["coefficient_checks"],
        "pool": run.base.read(run.OUT / "pool_audit.json"),
        "same_epoch_deltas": summary["same_epoch_minus_clean_pp"],
        "effects": summary["minus_reference_pp"],
        "control_vs_prior_selected_pp": control,
        "eligible_candidates": summary["eligible_candidates"],
        "test_used": False,
        "interpretation": "실제 학습 쌍의 구분력과 정상 입력 최종 성능을 구분한다. 문맥 계수는 사전 하한으로 "
        "잘려 전체키 조건과 초기 기울기 크기가 같지 않다. 시드1 검증 예비이며 시험 성능이나 구조 신규성의 증명이 아니다.",
    }
    write_json(run.OUT / "integrated_report.json", report)
    names = ["Start", "Rank only", "+ Key pairs", "+ Context pairs"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for role, label, color in [
        ("train", "Train", "#597da2"),
        ("val", "Validation", "#b76447"),
    ]:
        ys = [100 * summary["results"][role][a]["macro"]["AP"] for a in arms]
        axes[0, 0].plot(range(4), ys, "o-", label=label, color=color)
    axes[0, 0].set(
        xticks=range(4),
        xticklabels=names,
        ylabel="Final AP (%)",
        title="A. Full pipeline performance",
    )
    axes[0, 0].legend()
    for i, a in enumerate(arms[1:]):
        v = summary["minus_reference_pp"][a]["clean_rank"]
        axes[0, 1].plot(
            [0, 1, 2],
            [v[k] for k in ("AP", "max_F1", "R@100P")],
            "o-",
            label=names[i + 1],
        )
    axes[0, 1].axhline(0, color="gray", lw=0.8)
    axes[0, 1].set(
        xticks=[0, 1, 2],
        xticklabels=["AP", "Max F1", "R@100P"],
        ylabel="Change vs rank only (pp)",
        title="B. Validation changes",
    )
    axes[0, 1].legend(fontsize=8)
    for i, a in enumerate(arms[1:]):
        row = by_direction["val"][a]
        axes[1, 0].bar(
            np.arange(3) + (i - 1) * 0.25,
            [row[str(g)]["gained"] - row[str(g)]["lost"] for g in range(3)],
            width=0.24,
            label=names[i + 1],
        )
    axes[1, 0].axhline(0, color="gray", lw=0.8)
    axes[1, 0].set(
        xticks=range(3),
        xticklabels=["Same available", "Opposite only", "Other"],
        ylabel="Gained - lost queries",
        title="C. Validation choice changes vs start",
    )
    axes[1, 0].legend(fontsize=8)
    for mode, label in [("key", "Full key"), ("context", "Context")]:
        axes[1, 1].plot(
            range(4), [auxiliary[a][mode]["margin"] for a in arms], "o-", label=label
        )
    axes[1, 1].axhline(0, color="gray", lw=0.8)
    axes[1, 1].set(
        xticks=range(4),
        xticklabels=names,
        ylabel="Positive - strongest negative",
        title="D. Fixed real training pairs",
    )
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(run.OUT / f"real_revisit.{ext}", dpi=180)
    plt.close(fig)
    for name in (
        "scripts/_pdv2_real_revisit_report.py",
        "scripts/_pdv2_pair_evaluate.py",
        "scripts/_pdv2_context_diagnosis.py",
        "scripts/_pdv2_temporal_focus_diagnose.py",
        "results/pdv2_temporal_augmentation_20261024/summary.json",
    ):
        sources[name] = sha256(REPO / name)
    for name in (
        "integrated_report.json",
        "auxiliary_validation.json",
        "final_metrics.csv",
        "real_revisit.png",
        "real_revisit.pdf",
    ):
        path = run.OUT / name
        sources[str(path.relative_to(REPO))] = sha256(path)
    for path, digest in sources.items():
        assert sha256(REPO / path) == digest, path
    write_json(
        run.OUT / "integrated_validation.json",
        {
            **{k: v for k, v in audit.items() if k != "sources"},
            "sources": sources,
            "auxiliary_validation_passed": True,
            "test_used": False,
        },
    )
    print(
        "REAL_REVISIT_REPORT_COMPLETE",
        summary["eligible_candidates"],
        len(sources),
        flush=True,
    )


if __name__ == "__main__":
    main()
