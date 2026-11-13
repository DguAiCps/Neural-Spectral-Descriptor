#!/usr/bin/env python3
"""최종 AP 손실을 검색 실패와 두 종류 오수락의 진단 상한으로 분해한다."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]
import _pdv2_key_metric_alignment as run
from _pdv2_context_tail_finish import independent_metrics
from _pdv2_no_context_ablation import sha256, write_json

OUT = run.OUT / "error_budget"


def split_error(row: dict, score: np.ndarray) -> dict:
    """고정 승자의 오수락 점수를 진단용으로만 낮춰 AP 손실을 분해한다."""
    positive = row["min_past_gt_dist"] <= 5
    nonrevisit = row["min_past_gt_dist"] > 25
    wrong_place = positive & (row["top1_geo_dist"] > 5)
    scores = {}
    # 모든 유한 점수보다 낮게 둔다. 질의를 삭제하거나 분모를 바꾸지 않는다.
    floor = float(np.min(score) - max(float(np.ptp(score)), 1.0))
    for name, mask in (
        ("actual", np.zeros(len(score), bool)),
        ("reject_nonrevisit", nonrevisit),
        ("reject_wrong_place", wrong_place),
        ("reject_both", nonrevisit | wrong_place),
    ):
        changed = score.copy()
        changed[mask] = floor
        scores[name] = independent_metrics(row, changed)
    ap = {k: v["AP"] for k, v in scores.items()}
    if not positive.any():
        return {"metrics": scores, "positive": 0}
    ceiling = scores["actual"]["R@1"]
    np.testing.assert_allclose(ap["reject_both"], ceiling, atol=1e-12, rtol=0)
    # 두 오류 제거 순서의 평균: 상호작용은 두 원인에 반씩 배분한다.
    nonrevisit_cost = 0.5 * (
        ap["reject_nonrevisit"]
        - ap["actual"]
        + ap["reject_both"]
        - ap["reject_wrong_place"]
    )
    wrong_place_cost = 0.5 * (
        ap["reject_wrong_place"]
        - ap["actual"]
        + ap["reject_both"]
        - ap["reject_nonrevisit"]
    )
    components = {
        "retrieval_miss": 1 - ceiling,
        "nonrevisit_false_accept": nonrevisit_cost,
        "wrong_place_false_accept": wrong_place_cost,
    }
    assert all(v >= -1e-12 for v in components.values())
    np.testing.assert_allclose(
        sum(components.values()), 1 - ap["actual"], atol=1e-12, rtol=0
    )
    return {
        "metrics": scores,
        "components": components,
        "positive": int(positive.sum()),
        "nonrevisit": int(nonrevisit.sum()),
        "wrong_place": int(wrong_place.sum()),
        "queries": len(score),
    }


def main() -> None:
    """선택된 3시드의 기존·정합 모델을 같은 최종 원장에서 분석한다."""
    spec = run.protocol(True)
    summary = run.read(run.OUT / "summary.json")
    OUT.mkdir(exist_ok=True)
    sources = {
        str(Path(__file__).relative_to(REPO)): sha256(Path(__file__)),
        str((run.OUT / "summary.json").relative_to(REPO)): sha256(
            run.OUT / "summary.json"
        ),
    }
    for seed in run.SEEDS:
        for arm in run.ARMS:
            epoch = summary["selected"][str(seed)][arm]["epoch"]
            for sequence in spec["val"]:
                path = (
                    run.root(seed, arm)
                    / f"epoch{epoch}"
                    / (sequence.replace("/", "_") + ".npz")
                )
                sources[str(path.relative_to(REPO))] = sha256(path)
    plan = {
        "sources": sources,
        "test_used": False,
        "scope": "선택된검증3시드고정승자. GT오류구분은분석용이며온라인사용불가. 비재방문오수락/재방문오위치오수락의점수만전체유한최솟값보다낮게둔다. 완전거절AP=최종R1검산. 두거절순서의평균으로AP손실을배분한다. 검색실패상한과오수락분해는최종성능개선실험이아니다.",
    }
    target = OUT / "protocol.json"
    if target.exists():
        assert run.read(target) == plan
    else:
        write_json(target, plan)
    results, averaged = {}, {}
    for arm in run.ARMS:
        results[arm] = {}
        for seed in run.SEEDS:
            epoch = summary["selected"][str(seed)][arm]["epoch"]
            rows = {}
            for sequence in spec["val"]:
                p = (
                    run.root(seed, arm)
                    / f"epoch{epoch}"
                    / (sequence.replace("/", "_") + ".npz")
                )
                row = run.BASE.exp.arrays(p)
                rows[sequence] = split_error(row, row[run.METHOD])
            families = {}
            for family in ("kitti", "helipr", "mulran"):
                selected = [
                    r for s, r in rows.items() if s.startswith(family) and r["positive"]
                ]
                families[family] = {
                    k: float(np.mean([r["components"][k] for r in selected]))
                    for k in selected[0]["components"]
                }
            macro = {
                k: float(np.mean([r[k] for r in families.values()]))
                for k in next(iter(families.values()))
            }
            np.testing.assert_allclose(
                sum(macro.values()),
                1 - summary["selected"][str(seed)][arm]["macro"]["AP"],
                atol=1e-12,
                rtol=0,
            )
            results[arm][str(seed)] = {
                "sequences": rows,
                "families": families,
                "macro": macro,
            }
        averaged[arm] = {
            k: float(np.mean([r["macro"][k] for r in results[arm].values()]))
            for k in results[arm]["1"]["macro"]
        }
    write_json(
        OUT / "summary.json",
        {
            "results": results,
            "mean": averaged,
            "test_used": False,
            "protocol_sha256": sha256(target),
        },
    )
    # 최종 원장 분해와 합계가 맞는지 확인한 뒤 내보낼 그림을 만든다.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    labels = ("Unaligned training", "Aligned training")
    left = np.zeros(2)
    for key, label, color in (
        ("retrieval_miss", "Missed correct place", "#6e8ca0"),
        ("nonrevisit_false_accept", "Non-revisit false acceptance", "#de9450"),
        ("wrong_place_false_accept", "Wrong-place false acceptance", "#aa6878"),
    ):
        values = 100 * np.asarray([averaged[a][key] for a in run.ARMS])
        ax.barh(labels, values, left=left, label=label, color=color)
        for n, v in enumerate(values):
            if v > 0.8:
                ax.text(
                    left[n] + v / 2, n, f"{v:.2f}", ha="center", va="center", fontsize=9
                )
        left += values
    ax.set_xlabel("Decomposition of 100 - final validation AP (percentage points)")
    ax.set_title(
        "Diagnostic upper bounds; ground-truth labels are used only in analysis"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"final_error_budget.{suffix}", dpi=180)
    plt.close(fig)
    write_json(
        OUT / "validation.json",
        {
            "status": "passed",
            "ledgers": 54,
            "sum_to_AP_deficit": True,
            "perfect_rejection_equals_R1": True,
            "summary_sha256": sha256(OUT / "summary.json"),
            "test_used": False,
        },
    )
    print("ERROR_BUDGET", averaged, flush=True)


if __name__ == "__main__":
    main()
