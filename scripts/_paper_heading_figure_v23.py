#!/usr/bin/env python3
"""v2.3 그림: 재방문 장소와의 진행 방향 차이별 수락 비율(NSD, Past-only NSD, 표 2 비교 방법 10종).

- 수락: 시퀀스마다 오수락이 0이 되는 가장 낮은 임계값(표 2 R@100P와 같은 방식)에서 받아들인 정답 반환.
- 진행 방향 차이: 질의와, 정답 자세에서 가장 가까운 과거 keyframe(j <= q-30)의 진행 방향 사이 각.
  구간은 0, 30, ..., 180도를 중앙에 둔 30도 폭이고, 방향 차이가 0~180도 값이라 양 끝 구간은 0~15도와 165~180도다.
- 집계: 시험 46 시퀀스의 재방문 질의를 구간별로 합산, 시드(시작점) 3개의 평균과 최소~최대.
- 검산: 같은 수락 집합으로 낸 6도시 매크로 R@100P가 채택 요약(표 2)과 일치해야 한다.
새 학습·추론 없이 기존 원장만 읽는다.
"""
import glob, hashlib, json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results/pdv2_diffattn_contrast_test_20261104/test"
PAST = Path("/mnt/d/NSD_datasets/nsd_results/pdv2_zero_lag_pretrain_20261112/p10/test_eval/test")
SUMMARY = ROOT / "results/pdv2_deepring_scaleinit_integration_20261110/baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json"
OUT = ROOT / "results/paper_heading_figure_20261113"
DEST = ROOT / "docs/paper/body_v2.3/figs/pair_analysis"
PREFIX = dict(scpp="dump_scpp", ring="dump_ring", ringpp="dump_ringpp_i2", ringsharp="dump_ringsharp", deepring="dump_deepring",
              disco="dump_disco", helios="dump_helios", seqot="dump_seqotl2", bevplace="dump_bevplace", fresco="dump_fresco")
CENTERS = np.arange(0, 181, 30)                                   # 구간 중앙(그림의 눈금): 0, 30, ..., 180도
EDGES = np.r_[0, (CENTERS[:-1] + CENTERS[1:]) / 2, 180]          # 0-15, 15-45, ..., 135-165, 165-180도 (양 끝 구간은 반폭)
STYLE = dict(nsd=("NSD", "#196B24", "o"), past=("Past-only NSD", "#D9773A", "^"), bevplace=("BEVPlace++", "#753A85", "s"))
INK, MUTED, GRID, GREY = "#0E2841", "#6D7780", "#ECEFF1", "#B9C0C7"   # 그림 4와 같은 글자·테두리·격자 색


def city(s: str) -> str:
    """시퀀스 폴더 이름을 평가 도시로 바꾼다(scripts/_pdv2_pilot.py CITY와 같은 묶음)."""
    if s.startswith("kitti"): return "karlsruhe"
    if s.startswith("helipr_Town"): return "seoul"
    if s.startswith(("helipr", "mulran")): return "daejeon"
    if s.startswith("nclt"): return "ann_arbor"
    return "stockholm" if s.startswith("mcd_kth") else "hamburg"


def compute() -> dict:
    """원장에서 구간별 수락 비율과 검산용 R@100P를 계산한다."""
    summ = json.loads(SUMMARY.read_text())["methods"]
    runs = {"nsd": [(lambda s, i=i: MAIN / s / f"diff_s{i}.npz", "score") for i in (1, 2, 3)],
            "past": [(lambda s, i=i: PAST / s / f"zl_p10_diff_s{i}.npz", "score") for i in (1, 2, 3)]}
    for m, pre in PREFIX.items():
        runs[m] = [(lambda s, src=r["source"], pre=pre: ROOT / src / f"{pre}_{s}.npz", summ[m]["score"]) for r in summ[m]["per_seed"].values()]
    seqs = sorted(os.listdir(MAIN)); angle, ref = {}, {}
    for s in seqs:
        n = np.load(MAIN / s / "diff_s1.npz"); q, g = n["query_idx"], n["min_past_gt_dist"]
        src = json.loads((MAIN / s / "reference_validation.json").read_text())
        pos = np.load(ROOT / next(p for p in src["input_sha256"] if "/cache_" in p))["poses_gt"][:, :3, 3]
        idx = np.arange(len(pos)); d = pos - pos[np.maximum(idx - 5, 0)]; ln = np.linalg.norm(d, axis=1)
        head = d / np.maximum(ln[:, None], 1e-12); ok = (idx >= 5) & (ln >= 1.); a = np.full(len(q), np.nan)
        for r in np.flatnonzero(g <= 5):
            t = q[r]; j = int(np.argmin(np.linalg.norm(pos[:t - 29] - pos[t], axis=1)))
            if ok[t] and ok[j]: a[r] = np.degrees(np.arccos(np.clip(head[t] @ head[j], -1, 1)))
        angle[s], ref[s] = a, q
    out = {"edges": EDGES.tolist(), "centers": CENTERS.tolist(), "methods": {}}
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
            share.append(100 * hb[:, 0] / hb[:, 1]); cs = sorted({city(s) for s in per})
            r100.append(100 * np.mean([np.mean([per[s] for s in per if city(s) == c]) for c in cs]))
        out["methods"][m] = dict(share=np.mean(share, 0).tolist(), lo=np.min(share, 0).tolist(), hi=np.max(share, 0).tolist(),
                                 r100p=float(np.mean(r100)), n=hb[:, 1].astype(int).tolist(), runs=len(rr))
        if m in summ:   # 표 2 채택 요약과 대조
            assert abs(out["methods"][m]["r100p"] - 100 * summ[m]["mean_std"]["R@100P"]["mean"]) < 1e-6, m
    assert abs(out["methods"]["nsd"]["r100p"] - 69.6485) < 5e-4 and abs(out["methods"]["past"]["r100p"] - 49.00) < 5e-3
    out["sources"] = {"summary": hashlib.sha256(SUMMARY.read_bytes()).hexdigest(), "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return out


def draw(d: dict) -> None:
    """한 단 그림을 그린다. 위: 구간별 수락 비율(점은 구간 중앙 = 눈금), 아래: 구간별 재방문 질의 수."""
    plt.rcParams.update({"font.family": "Carlito", "font.size": 7.9, "axes.edgecolor": MUTED, "axes.linewidth": .6,
                         "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.25, 2.3)); ax = fig.add_axes([.125, .355, .84, .62]); bar = fig.add_axes([.125, .175, .84, .135])
    x = CENTERS; M = d["methods"]; others = [m for m in M if m not in STYLE]
    for m in others: ax.plot(x, M[m]["share"], color=GREY, lw=.8, zorder=1, clip_on=False)
    ax.plot([], [], color=GREY, lw=.8, label=f"Other baselines ({len(others)})")
    for m in ("bevplace", "past", "nsd"):
        lab, col, mk = STYLE[m]; ax.fill_between(x, M[m]["lo"], M[m]["hi"], color=col, alpha=.15, lw=0, zorder=2)
        ax.plot(x, M[m]["share"], color=col, lw=1.6, marker=mk, ms=4.8 if m == "bevplace" else 3.8, mec="white", mew=.5, label=lab, zorder=3, clip_on=False)   # BEVPlace++ 표식은 겹쳐도 보이게 조금 크게
    ax.set_xticks(CENTERS); ax.set_xticklabels([]); ax.set_xlim(0, 180); ax.set_ylim(0, 80); ax.grid(color=GRID, lw=.45); ax.set_axisbelow(True); ax.tick_params(length=2, pad=2)
    ax.set_ylabel("Revisits accepted (%)", labelpad=1)
    h, l = ax.get_legend_handles_labels(); ax.legend(h[::-1], l[::-1], frameon=False, fontsize=7.0, loc="upper center", handlelength=1.6, labelspacing=.25, borderaxespad=.1)
    bar.bar((EDGES[:-1] + EDGES[1:]) / 2, M["nsd"]["n"], width=np.diff(EDGES) * .88, color=GREY); bar.set_yscale("log"); bar.set_xlim(0, 180); bar.set_ylim(30, 80000)
    bar.set_xticks(CENTERS); bar.set_yticks([100, 10000]); bar.minorticks_off(); bar.tick_params(length=2, pad=2); bar.tick_params(axis="y", labelsize=6.4); bar.grid(axis="y", color=GRID, lw=.45); bar.set_axisbelow(True)
    bar.set_xlabel("Heading difference to the revisited place (deg)", labelpad=1); bar.set_ylabel("Queries", fontsize=6.8, labelpad=1)
    for ext in ("pdf", "png"): fig.savefig(DEST / f"fig_heading_acceptance.{ext}", dpi=300)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); d = compute()
    (OUT / "heading_acceptance.json").write_text(json.dumps(d, indent=1) + "\n"); draw(d)
    for m, v in d["methods"].items(): print(m, f"R@100P {v['r100p']:.2f}", np.round(v["share"], 1).tolist())


if __name__ == "__main__":
    main()
