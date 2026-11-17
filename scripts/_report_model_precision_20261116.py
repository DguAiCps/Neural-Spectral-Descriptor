#!/usr/bin/env python3
"""보고모델 교체(2026-11-16) 단위 C 항목 4: 요구 정밀도별 도시 매크로 재현율(그림 fig_precision_recall 후보).

_paper_precision_fresco_v23.py의 곡선 자료(results/paper_precision_fresco_20261113/curves.json)에서 NSD 곡선만
지정한 원장으로 다시 집계해 바꾼다. 다른 10곡선·배치·색·글꼴은 그대로다(같은 PRECISION_LAYOUT, preview.render).
먼저 이전 NSD 원장(diff_s1..3)으로 기존 NSD 곡선(평균·SD)을 재현한 뒤 새 원장을 넣는다.

python3 scripts/_report_model_precision_20261116.py [--nsd new|old] [--fig-dir DIR]
출력: results/pdv2_report_model_20261116/analysis/precision/{curves.json,summary.json}, analysis/figures/fig_precision_recall.{pdf,png,svg}
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_model_common_20261116 as C  # noqa: E402
import _paper_baseline_figure_preview as preview  # noqa: E402
import _paper_baseline_figure_review as review  # noqa: E402
from _paper_contrast_figures_v23 import PRECISION_LAYOUT  # noqa: E402
from _paper_precision_fresco_v23 import COLOR, STYLE  # noqa: E402

BASE = C.ROOT / "results/paper_precision_fresco_20261113/curves.json"
OUT = C.OUT / "precision"
FIG = C.OUT / "figures"
POINTS = (.9, .95, .99, 1.)


def curve(tags: list) -> dict:
    runs = []
    for t in tags:
        per = {}
        for s in C.SEQS:
            d = C.load(t, s)
            qc = C.classify_queries(d["gt"])
            if not qc["n_pos"]:
                continue
            per[s] = review.recall_at_precision(C.pr_sweep(d["score"], d["top1"], qc), review.GRID)
        cities = [np.mean([per[s] for s in v if s in per], axis=0) for v in C.CITY.values()]
        runs.append(np.mean(cities, axis=0))
    runs = np.array(runs)
    return dict(mean=runs.mean(0), std=runs.std(0, ddof=1), runs=runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsd", default="new", choices=("new", "old"))
    ap.add_argument("--fig-dir", default=None, help="그림 출력 폴더(기본 analysis/figures). 예: docs/paper/body_v2.4/figs/pair_analysis")
    args = ap.parse_args()
    fig_dir = Path(args.fig_dir) if args.fig_dir else FIG
    base = review.read(BASE)
    grid = list(review.GRID)
    np.testing.assert_array_equal(base["precision_grid"], review.GRID)
    old = curve(C.tags("old"))
    err_mean = float(np.max(np.abs(old["mean"] - np.asarray(base["methods"]["nsd"]["mean"]))))
    err_sd = float(np.max(np.abs(old["std"] - np.asarray(base["methods"]["nsd"]["std"]))))
    bev = curve(C.tags("bev"))
    err_bev = float(np.max(np.abs(bev["mean"] - np.asarray(base["methods"]["bevplace"]["mean"]))))
    print("재현: 이전 NSD 곡선 최대 오차 평균", err_mean, "SD", err_sd, "| BEVPlace++", err_bev)
    assert err_mean < 1e-12 and err_sd < 1e-12 and err_bev < 1e-12
    new = curve(C.tags(args.nsd))
    ref = curve(C.tags("ref"))
    methods = {m: dict(row) for m, row in base["methods"].items()}
    methods["nsd"] = dict(label=methods["nsd"]["label"], mean=new["mean"].tolist(), std=new["std"].tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    review.write(OUT / "curves.json", dict(precision_grid=base["precision_grid"], methods=methods, base_curves=str(BASE.relative_to(C.ROOT)),
                                           base_curves_sha256=review.digest(BASE), nsd_ledgers=args.nsd,
                                           nsd_seeds=list(C.NEW_SEEDS) if args.nsd == "new" else list(C.OLD_SEEDS)))
    pick = lambda c: {f"R@{int(round(100 * p))}P" if p < 1 else "R@100P":
                      dict(mean=100 * float(c["mean"][grid.index(p)]), sd=100 * float(c["std"][grid.index(p)]),
                           runs=[100 * float(x) for x in c["runs"][:, grid.index(p)]]) for p in POINTS}
    summary = dict(nsd=pick(new), nsd_seeds1to3_reference=pick(ref), previous_nsd=pick(old), bevplace=pick(bev),
                   reproduction=dict(previous_nsd_curve_max_abs_err=err_mean, previous_nsd_sd_max_abs_err=err_sd, bevplace_curve_max_abs_err=err_bev),
                   crossing=crossing(new["mean"], bev["mean"]), crossing_previous=crossing(old["mean"], bev["mean"]),
                   definition="시퀀스별 임계값 sweep에서 정밀도 하한을 만족하는 최대 재현율(review.recall_at_precision) → 도시 → 6도시 → 시드 평균, SD ddof=1.")
    review.write(OUT / "summary.json", summary)
    # 그림: 기존 그림과 같은 render, 출력만 analysis/figures
    labels = {}
    for k, v in preview.LABELS.items():
        labels[k] = v
        if k == "scpp":
            labels["fresco"] = "FreSCo"
    preview.LABELS.clear(); preview.LABELS.update(labels)
    preview.STYLES["fresco"] = STYLE
    preview.OUT = OUT
    preview.render(layout=dict(PRECISION_LAYOUT, colors={**PRECISION_LAYOUT["colors"], "fresco": COLOR}))
    for suffix in ("pdf", "png", "svg"):
        shutil.copy2(OUT / f"baseline_precision_candidate.{suffix}", fig_dir / f"fig_precision_recall.{suffix}")
    for k in ("nsd", "previous_nsd", "bevplace"):
        print(k, {m: f"{v['mean']:.3f}±{v['sd']:.3f}" for m, v in summary[k].items()})
    print("crossing", summary["crossing"], "previous", summary["crossing_previous"])


def crossing(a, b):
    """NSD가 BEVPlace++ 이상이 되기 시작하는 요구 정밀도(그 위로 계속 앞서는 최소 격자점)."""
    g = np.asarray(review.GRID)
    ahead = a >= b
    idx = len(g)
    while idx > 0 and ahead[idx - 1]:
        idx -= 1
    return dict(from_precision=100 * float(g[idx]) if idx < len(g) else None)


if __name__ == "__main__":
    main()
