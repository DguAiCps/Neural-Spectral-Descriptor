#!/usr/bin/env python3
"""보고모델 교체(2026-11-16) 단위 C 항목 9: 진행 방향 차이별 수락 비율(fig_heading_acceptance 후보).

scripts/_paper_heading_figure_v23.py(작업 트리 판)의 계산·그림을 그대로 옮기고, NSD와 Past-only NSD 원장 위치만 인자로 받는다.
비교 방법 10종은 같은 채택 요약 원장을 그대로 읽는다. 각도·질의 집합은 이전 NSD 시험 폴더의 캐시 기준(같은 질의).

python3 scripts/_report_model_heading_20261116.py --nsd old --past old      # 기존 발표 수치 재현 검사
python3 scripts/_report_model_heading_20261116.py --nsd new [--past DIR --past-pattern 'NAME@s{seed}.npz' --past-seeds 4,5,6]
  --past 를 주지 않으면 Past-only 곡선 없이 NSD·비교 방법만 계산·그린다(그림은 만들되 이름에 _nsd_only).
출력: results/pdv2_report_model_20261116/analysis/heading/heading_acceptance[_<tag>].json, analysis/figures/fig_heading_acceptance[...].{pdf,png}
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_model_common_20261116 as C  # noqa: E402
import _paper_heading_figure_v23 as H  # noqa: E402

OLD_PAST = Path("/mnt/d/NSD_datasets/nsd_results/pdv2_zero_lag_pretrain_20261112/p10/test_eval/test")


def compute(nsd_kind, past_dir, past_pattern, past_seeds):
    summ = json.loads(H.SUMMARY.read_text())["methods"]
    runs = {"nsd": [(lambda s, t=t: C.path(t, s)[0], "score") for t in C.tags(nsd_kind)]}
    if past_dir is not None:
        runs["past"] = [(lambda s, i=i: Path(past_dir) / s / past_pattern.format(seed=i), "score") for i in past_seeds]
    for m, pre in H.PREFIX.items():
        runs[m] = [(lambda s, src=r["source"], pre=pre: C.ROOT / src / f"{pre}_{s}.npz", summ[m]["score"]) for r in summ[m]["per_seed"].values()]
    seqs = sorted(os.listdir(H.MAIN)); angle, ref = {}, {}
    EDGES = H.EDGES
    for s in seqs:
        n = np.load(H.MAIN / s / "diff_s1.npz"); q, g = n["query_idx"], n["min_past_gt_dist"]
        src = json.loads((H.MAIN / s / "reference_validation.json").read_text())
        pos = np.load(C.ROOT / next(p for p in src["input_sha256"] if "/cache_" in p))["poses_gt"][:, :3, 3]
        idx = np.arange(len(pos)); d = pos - pos[np.maximum(idx - 5, 0)]; ln = np.linalg.norm(d, axis=1)
        head = d / np.maximum(ln[:, None], 1e-12); ok = (idx >= 5) & (ln >= 1.); a = np.full(len(q), np.nan)
        for r in np.flatnonzero(g <= 5):
            t = q[r]; j = int(np.argmin(np.linalg.norm(pos[:t - 29] - pos[t], axis=1)))
            if ok[t] and ok[j]: a[r] = np.degrees(np.arccos(np.clip(head[t] @ head[j], -1, 1)))
        angle[s], ref[s] = a, q
    out = {"edges": EDGES.tolist(), "centers": H.CENTERS.tolist(), "methods": {}}
    for m, rr in runs.items():
        share, r100 = [], []
        for path, key in rr:
            hb, per = np.zeros((len(EDGES) - 1, 2)), {}
            for s in seqs:
                d = np.load(path(s), allow_pickle=True); k = np.isin(d["query_idx"], ref[s]); assert k.sum() == len(ref[s]), (m, s)
                sc, t, g = d[key][k], d["top1_geo_dist"][k], d["min_past_gt_dist"][k]; sc = np.where(np.isfinite(sc), sc, -np.inf)
                pos, neg = g <= 5, g > 25; true = pos & (t <= 5); wrong = (pos | neg) & ~true
                acc = true & (sc > (sc[wrong].max() if wrong.any() else -np.inf)); b = np.digitize(angle[s], EDGES[1:-1]); v = pos & np.isfinite(angle[s])
                for i in range(len(EDGES) - 1): hb[i] += ((acc & v & (b == i)).sum(), (v & (b == i)).sum())
                if pos.any(): per[s] = acc.sum() / pos.sum()
            share.append(100 * hb[:, 0] / hb[:, 1]); cs = sorted({H.city(s) for s in per})
            r100.append(100 * np.mean([np.mean([per[s] for s in per if H.city(s) == c]) for c in cs]))
        out["methods"][m] = dict(share=np.mean(share, 0).tolist(), lo=np.min(share, 0).tolist(), hi=np.max(share, 0).tolist(),
                                 r100p=float(np.mean(r100)), n=hb[:, 1].astype(int).tolist(), runs=len(rr), per_run_share=[x.tolist() for x in share])
        if m in summ:
            assert abs(out["methods"][m]["r100p"] - 100 * summ[m]["mean_std"]["R@100P"]["mean"]) < 1e-6, m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsd", default="new", choices=("new", "old"))
    ap.add_argument("--past", default=None, help="'old' 또는 Past-only 시험 원장 폴더(<dir>/<seq>/<file>)")
    ap.add_argument("--past-pattern", default="zl_p10_diff_s{seed}.npz")
    ap.add_argument("--past-seeds", default="1,2,3")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fig-dir", default=None, help="그림 출력 폴더(기본 analysis/figures). 예: docs/paper/body_v2.4/figs/pair_analysis")
    args = ap.parse_args()
    past = OLD_PAST if args.past == "old" else args.past
    seeds = [int(x) for x in args.past_seeds.split(",")]
    d = compute(args.nsd, past, args.past_pattern, seeds)
    tag = args.tag or (f"{args.nsd}" + ("" if past else "_nsd_only"))
    folder = C.OUT / "heading"
    folder.mkdir(parents=True, exist_ok=True)
    d["config"] = dict(nsd=args.nsd, past=str(past) if past else None, past_pattern=args.past_pattern, past_seeds=seeds)
    M = d["methods"]
    i0, i180, i90 = 0, len(H.CENTERS) - 1, list(H.CENTERS).index(90)
    d["text_numbers"] = dict(
        nsd_below15=M["nsd"]["share"][i0], nsd_above165=M["nsd"]["share"][i180],
        past_below15=M["past"]["share"][i0] if "past" in M else None, past_above165=M["past"]["share"][i180] if "past" in M else None,
        bin75_105=dict(nsd=M["nsd"]["share"][i90], bevplace=M["bevplace"]["share"][i90], n=M["nsd"]["n"][i90], total=int(sum(M["nsd"]["n"]))),
        nsd_lower_than_bevplace_bins=[int(c) for c, a, b in zip(H.CENTERS, M["nsd"]["share"], M["bevplace"]["share"]) if a < b])
    (folder / f"heading_acceptance_{tag}.json").write_text(json.dumps(d, indent=1) + "\n")
    if args.nsd == "old" and args.past == "old":
        prev = json.loads((H.OUT / "heading_acceptance.json").read_text())
        err = max(abs(a - b) for m in prev["methods"] for a, b in zip(d["methods"][m]["share"], prev["methods"][m]["share"]))
        print("재현: 기존 heading_acceptance.json 대비 최대 오차", err)
        assert err < 1e-9
        return
    dest = Path(args.fig_dir) if args.fig_dir else C.OUT / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    draw(d, dest / ("fig_heading_acceptance" + ("" if "past" in M else "_nsd_only")))
    for m, v in M.items():
        print(m, f"R@100P {v['r100p']:.2f}", np.round(v["share"], 1).tolist())
    print(json.dumps(d["text_numbers"], indent=1))


def draw(d, stem):
    """H.draw와 같은 그림. 있는 곡선(bevplace, past, nsd)만 그리고 stem.{pdf,png}로 쓴다."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Carlito", "font.size": 7.9, "axes.edgecolor": H.MUTED, "axes.linewidth": .6,
                         "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "axes.labelcolor": H.INK, "text.color": H.INK, "xtick.color": H.INK, "ytick.color": H.INK,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.25, 2.3)); ax = fig.add_axes([.125, .355, .84, .62]); bar = fig.add_axes([.125, .175, .84, .135])
    x, EDGES = H.CENTERS, H.EDGES; M = d["methods"]; others = [m for m in M if m not in H.STYLE]
    for m in others: ax.plot(x, M[m]["share"], color=H.GREY, lw=.8, zorder=1, clip_on=False)
    ax.plot([], [], color=H.GREY, lw=.8, label=f"Other baselines ({len(others)})")
    for m in [k for k in ("bevplace", "past", "nsd") if k in M]:
        lab, col, mk = H.STYLE[m]; ax.fill_between(x, M[m]["lo"], M[m]["hi"], color=col, alpha=.15, lw=0, zorder=2)
        ax.plot(x, M[m]["share"], color=col, lw=1.6, marker=mk, ms=4.8 if m == "bevplace" else 3.8, mec="white", mew=.5, label=lab, zorder=3, clip_on=False)
    ax.set_xticks(H.CENTERS); ax.set_xticklabels([]); ax.set_xlim(0, 180); ax.set_ylim(0, 80); ax.grid(color=H.GRID, lw=.45); ax.set_axisbelow(True); ax.tick_params(length=2, pad=2)
    ax.set_ylabel("Revisits accepted (%)", labelpad=1)
    h, l = ax.get_legend_handles_labels(); ax.legend(h[::-1], l[::-1], frameon=False, fontsize=7.0, loc="upper center", handlelength=1.6, labelspacing=.25, borderaxespad=.1)
    bar.bar((EDGES[:-1] + EDGES[1:]) / 2, M["nsd"]["n"], width=np.diff(EDGES) * .88, color=H.GREY); bar.set_yscale("log"); bar.set_xlim(0, 180); bar.set_ylim(30, 80000)
    bar.set_xticks(H.CENTERS); bar.set_yticks([100, 10000]); bar.minorticks_off(); bar.tick_params(length=2, pad=2); bar.tick_params(axis="y", labelsize=6.4); bar.grid(axis="y", color=H.GRID, lw=.45); bar.set_axisbelow(True)
    bar.set_xlabel("Heading difference to the revisited place (deg)", labelpad=1); bar.set_ylabel("Queries", fontsize=6.8, labelpad=1)
    for ext in ("pdf", "png"): fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
