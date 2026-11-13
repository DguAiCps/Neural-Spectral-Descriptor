#!/usr/bin/env python3
"""결정적으로 고른 회복·실패 사례의 실제 BEV 입력과 순위를 그린다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "src")]


def main() -> None:
    """ring_baseline에서 원시 스캔을 읽어 정렬하지 않은 입력을 표시한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _bevclip_spec import _sketch
    from _pd_build_test_caches import _make_loader
    from _pbev_branch_fusion import BP

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "results/pdv2_collision_20261013")
    args = parser.parse_args()
    selected = json.loads((args.out / "selected_cases.json").read_text())
    names = [name for name in ("collision_recovered", "spread_recovered", "collision_unresolved")
             if name in selected["cases"]]
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(len(names), 5, figsize=(10.5, 2.3 * len(names)), squeeze=False,
                             layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.45]})
    radius, theta = np.meshgrid(np.arange(1, 81), np.linspace(-np.pi, np.pi, 61), indexing="ij")
    xx, yy = radius * np.cos(theta), radius * np.sin(theta)
    metadata, grids = [], {}
    for row, name in enumerate(names):
        case = selected["cases"][name]["selected"]
        sensor, seq = case["sequence"].split("/")
        info = json.loads((args.out / f"s1_{sensor}_{seq}.json").read_text())
        cache = np.load(REPO / info["caches"]["cache"])
        spectrum = np.load(REPO / info["caches"]["spec"])
        loader = _make_loader(sensor, seq)
        query = case["query_idx"]
        indices = [query, case["raw_top1_idx"], case["raw_best_pos_idx"], case["pbev128_main_j_star"]]
        titles = ["Query", "Raw-key top 1", "Raw-key best true", "NSD top 1"]
        checks = []
        for col, (index, title) in enumerate(zip(indices, titles)):
            scan_id = int(cache["scan_ids"][index])
            points = loader[scan_id]["points"]
            bev = np.maximum(BP.project(points, keep_intensity=False)[0], 0)
            expected = spectrum["sk_re"][index] + 1j * spectrum["sk_im"][index]
            error = float(np.max(np.abs(_sketch(bev) - expected)))
            assert error <= 1e-5, (name, index, error)
            grids[f"{name}_{col}"] = bev
            ax = axes[row, col]
            artist = ax.pcolormesh(xx, yy, bev, cmap="magma", vmin=0, vmax=5, rasterized=True)
            ax.plot(0, 0, "+", color="cyan", ms=5)
            ax.set(xlim=(-80, 80), ylim=(-80, 80), aspect="equal", xticks=[], yticks=[])
            distance = float(np.linalg.norm(cache["poses_gt"][index, :3, 3] - cache["poses_gt"][query, :3, 3]))
            ax.set_title(title + (f"\n{distance:.1f} m" if col else ""), fontsize=9)
            ax.set_xlabel(f"scan {scan_id}", fontsize=8)
            if col == 0:
                label = name.replace("_recovered", ": top-1 recovered").replace("_unresolved", ": top-1 wrong")
                ax.set_ylabel(label + f"\n{sensor}/{seq}", fontsize=8)
            checks.append({"keyframe_idx": index, "scan_id": scan_id,
                           "gt_distance_m": distance, "spectrum_max_abs_diff": error})
        ax = axes[row, 4]
        ranks = [case[f"{s}_best_pos_rank"] for s in ("raw", "weighted", "pass1", "context")]
        final_rank = case["pbev128_main_best_pos_candidate_rank"]
        ranks.append(final_rank if final_rank > 0 else np.nan)
        ax.plot(np.arange(5), ranks, "o-", color="#137d78", lw=1.4, ms=4)
        ax.set_yscale("log")
        status = "accepted" if case["pbev128_main_accepted"] else "rejected"
        ax.set(xticks=np.arange(5), xticklabels=["Raw", "Weight", "Pass 1", "Key", "NSD"],
               ylabel="Best true rank (log)",
               title=f"raw distance = {case['raw_best_pos_dist']:.3f}\nNSD decision: {status}")
        ax.set_ylim(.65, max([r for r in ranks if np.isfinite(r)] + [10]) * 2)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelrotation=40, labelsize=8)
        for i, rank in enumerate(ranks):
            if np.isfinite(rank):
                ax.annotate(str(rank), (i, rank), xytext=(0, -10), textcoords="offset points",
                            ha="center", fontsize=8)
        if final_rank < 0:
            ax.text(.98, .45, "No true\ncandidate", transform=ax.transAxes, ha="right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        metadata.append({"case": name, "source": case, "scan_checks": checks})
    fig.colorbar(artist, ax=axes[:, :4], shrink=.55, pad=.01, label="Clipped height (m; display capped at 5)")
    fig.savefig(args.out / "collision_cases.pdf", bbox_inches="tight")
    fig.savefig(args.out / "collision_cases.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(args.out / "case_bev_inputs.npz", **grids)
    result = {"selection_rule": selected["selection_rule"], "cases": metadata,
              "caption_notes": ["센서 좌표계의 실제 79×60 BEV 입력이다. 시각화를 위해 회전 정렬하지 않는다.",
                                "표시 색만 높이 5 m에서 자르며 저장 입력과 측정에는 추가 절단이 없다.",
                                "각 키의 순위는 전체 과거 후보, NSD 순위는 100+100 후보 합집합 안에서 잰다.",
                                "단계마다 최고 정답의 프레임이 달라질 수 있다. Raw-key best true 그림은 원래 키 기준이다.",
                                "원시 점 구름에서 재생성한 스펙트럼을 현재 캐시와 대조했다."]}
    (args.out / "case_figure_provenance.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Rendered {len(names)} cases; all {4 * len(names)} raw-scan/cache checks passed.")


if __name__ == "__main__":
    main()
