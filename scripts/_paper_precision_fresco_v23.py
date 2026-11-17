#!/usr/bin/env python3
"""v2.3 그림 4(요구 정밀도별 재현율)에 FreSCo 곡선을 더한다.

- 기존 10곡선(NSD + 비교 방법 9종)은 검산된 curves.json을 그대로 읽는다. 다시 집계하지 않는다.
- FreSCo 곡선은 표 2가 쓰는 46원장에서 같은 규칙(시퀀스별 임계값 → 도시 → 6도시 평균)으로 집계한다.
- 검산: FreSCo의 6도시 매크로 AP·최대 F1·R@100P·R@1이 표 2 채택 요약과 일치해야 한다.
- 배치·색·글꼴과 다른 곡선은 그대로다. FreSCo는 범례에서 표 2 순서(SC++ 다음)에 둔다.
새 학습·추론 없이 기존 원장만 읽는다. 이전 곡선 폴더는 덮지 않고 새 폴더에 쓴다.
"""
import shutil
from pathlib import Path

import numpy as np

import _paper_baseline_figure_preview as preview
import _paper_baseline_figure_review as review
from _paper_contrast_figures_v23 import DEST, PRECISION_LAYOUT, PRECISION_OUT
from _paper_tables_v23 import DEEPRING_SUMMARY
from _pdv2_pilot import CITY, TEST
from _pr_eval_lib import classify_queries, pr_sweep

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper_precision_fresco_20261113"
PREFIX, SCORE = "dump_fresco", "score_fresco"
# 같은 선 모양(-.)인 RING++·HeLiOS와의 색차(OKLab×100): 정상 21.5, 색각 이상 8.1 이상.
COLOR, STYLE = "#C2478F", "-."


def fresco_curve(summary: dict) -> dict:
    """FreSCo 46원장에서 요구 정밀도별 재현율을 집계하고 표 2 값과 대조한다."""
    entry = summary["methods"]["fresco"]
    (run,) = entry["per_seed"].values()
    per, count, positives = {}, 0, 0
    for sensor, seq in TEST:
        name = f"{sensor}_{seq}.npz"
        with np.load(PRECISION_OUT / "main_view/s1" / name) as ref, np.load(ROOT / run["source"] / f"{PREFIX}_{name}") as data:
            q = ref["query_idx"]
            ix = np.searchsorted(data["query_idx"], q)
            np.testing.assert_array_equal(data["query_idx"][ix], q)
            np.testing.assert_array_equal(data["min_past_gt_dist"][ix], ref["min_past_gt_dist"])
            assert np.all(data["j_star"][ix] <= q - 30)
            qc = classify_queries(ref["min_past_gt_dist"])
            count += len(ix)
            positives += qc["n_pos"]
            if not qc["n_pos"]:
                assert (sensor, seq) == ("kitti", "07")
                continue
            dist = data["top1_geo_dist"][ix]
            sw = pr_sweep(data[SCORE][ix], dist, qc)
        curve = review.recall_at_precision(sw, review.GRID)
        metrics = {k: sw[k] for k in ("AP", "max_F1", "R@100P")}
        metrics["R@1"] = float((dist[qc["positive"]] <= 5).mean())
        per[f"{sensor}/{seq}"] = dict(metrics=metrics, recall=curve)
    assert (count, positives) == (195726, 61682)
    cities = [[per[s] for s in names if s in per] for names in CITY.values()]
    macro = {k: float(np.mean([np.mean([r["metrics"][k] for r in c]) for c in cities])) for k in review.METRICS}
    for k in review.METRICS:
        assert abs(macro[k] - run["macro"][k]) < 1e-12, (k, macro[k], run["macro"][k])
    mean = np.mean([np.mean([r["recall"] for r in c], axis=0) for c in cities], axis=0)
    assert abs(mean[-1] - entry["mean_std"]["R@100P"]["mean"]) < 1e-12
    return dict(label="FreSCo", mean=mean.tolist(), std=None, metrics=macro, source=run["source"])


def main() -> None:
    """곡선 자료에 FreSCo를 더해 그림을 다시 그리고 원고 폴더에 둔다."""
    OUT.mkdir(parents=True, exist_ok=True)
    data = review.read(PRECISION_OUT / "precision/curves.json")
    assert list(data["methods"]) == ["scpp", "ring", "ringpp", "ringsharp", "deepring", "disco", "helios", "seqot", "bevplace", "nsd"]
    np.testing.assert_array_equal(data["precision_grid"], review.GRID)
    fresco = fresco_curve(review.read(DEEPRING_SUMMARY))
    methods = {}
    for m, row in data["methods"].items():   # 표 2 순서: SC++ 다음에 FreSCo
        methods[m] = dict(label=row["label"], mean=row["mean"], std=row["std"])
        if m == "scpp":
            methods["fresco"] = fresco
    review.write(OUT / "curves.json", dict(precision_grid=data["precision_grid"], methods=methods,
                                           base_curves=str((PRECISION_OUT / "precision/curves.json").relative_to(ROOT)),
                                           base_curves_sha256=review.digest(PRECISION_OUT / "precision/curves.json"),
                                           summary_sha256=review.digest(DEEPRING_SUMMARY)))
    labels = {}
    for k, v in preview.LABELS.items():
        labels[k] = v
        if k == "scpp":
            labels["fresco"] = "FreSCo"
    preview.LABELS.clear(); preview.LABELS.update(labels)
    preview.STYLES["fresco"] = STYLE
    preview.OUT = OUT
    layout = dict(PRECISION_LAYOUT, colors={**PRECISION_LAYOUT["colors"], "fresco": COLOR})
    preview.render(layout=layout)
    for suffix in ("pdf", "png", "svg"):
        shutil.copy2(OUT / f"baseline_precision_candidate.{suffix}", DEST / f"fig_precision_recall.{suffix}")
    grid = list(review.GRID)
    print("FreSCo", {p: round(100 * fresco["mean"][grid.index(p)], 2) for p in (.9, .95, .99, 1.)}, fresco["metrics"])


if __name__ == "__main__":
    main()
