#!/usr/bin/env python3
"""v2.3 그림 시안: DiffAttnConv 학습의 이득(15회 독립 추가 학습, 균일 감산 대비).

(a) 지표별 이득: 시작점 3 x 표본 시드 5의 실행별 값, 평균, 표본 시드와 시험 시퀀스를 함께 재표집한 95% 구간(사후 분석 posthoc_crossed.json의 B_crossed, 본문 문장과 같은 구간).
(b) 같은 비율의 질의를 수락할 때 줄어드는 오수락: 시험 46 시퀀스의 평가 질의를 수락 점수로 정렬해 상위 k를 수락하고
    틀린 반환(오반환 또는 비재방문)의 수를 균일 감산과 비교. 실행별 곡선, 15회 평균, 도시 안 시퀀스 재표집 95% 띠.
검산: 원장에서 다시 낸 실행별 6도시 매크로 AP·R@100P 이득이 분석 파일의 per_run과 일치해야 한다.
새 학습·추론 없이 기존 원장만 읽는다. 양식은 그림 4(scripts/_paper_baseline_figure_preview.py)와 맞춘다.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _pr_eval_lib import classify_queries, pr_sweep  # noqa: E402

MAIN = ROOT / "results/pdv2_diffattn_contrast_test_20261104/test"
IND = Path("/mnt/d/NSD_datasets/nsd_results/pdv2_diffattn_independent_20261108")
OUT = ROOT / "results/paper_learning_figure_20261113"
NAVY, EDGE, GRID, ZERO = "#0E2841", "#6D7780", "#ECEFF1", "#3B4650"
START = (("#546578", "o"), ("#546578", "s"), ("#546578", "^"))   # 시작점은 모양으로만 구분(그림 4·6의 방법 색과 겹치지 않게)
FONT = dict(tick=7.4, label=7.9, title=8.3, note=7.2)
FRAC = np.linspace(.01, 1, 100)


def city(s: str) -> str:
    if s.startswith("kitti"): return "karlsruhe"
    if s.startswith("helipr_Town"): return "seoul"
    if s.startswith(("helipr", "mulran")): return "daejeon"
    if s.startswith("nclt"): return "ann_arbor"
    return "stockholm" if s.startswith("mcd_kth") else "hamburg"


def load(path: Path) -> tuple:
    d = np.load(path); g = d["min_past_gt_dist"]; ev = (g <= 5) | (g > 25)
    return d, d["score"][ev], ~((g <= 5) & (d["top1_geo_dist"] <= 5))[ev]


def reduction(fixed: dict, arm: dict, names: list) -> np.ndarray:
    """상위 k 수락에서 균일 감산보다 줄어든 틀린 반환 수(질의 100개당)."""
    out = []
    for src in (fixed, arm):
        sc = np.concatenate([src[s][0] for s in names]); bad = np.concatenate([src[s][1] for s in names]); n = len(sc)
        cum = np.cumsum(bad[np.argsort(-sc, kind="stable")]); out.append(100 * cum[np.maximum((FRAC * n).astype(int) - 1, 0)] / n)
    return out[0] - out[1]


def compute() -> dict:
    ana = json.loads((IND / "analysis.json").read_text()); seqs = sorted(os.listdir(MAIN)); runs = [(s, r) for s in (1, 2, 3) for r in (1, 2, 3, 4, 5)]
    fixed, fm = {}, {}
    for s in seqs:
        d, sc, bad = load(MAIN / s / "fixed.npz"); fixed[s] = (sc, bad)
        if (d["min_past_gt_dist"] <= 5).any():
            r = pr_sweep(d["score"], d["top1_geo_dist"], classify_queries(d["min_past_gt_dist"])); fm[s] = (r["AP"], r["R@100P"])
    cities = sorted({city(s) for s in fm}); macro = lambda m: np.mean([np.mean([m[s] for s in m if city(s) == c], 0) for c in cities], 0)
    arms, curves = {}, []
    for k, (st, rn) in enumerate(runs):
        arm, am = {}, {}
        for s in seqs:
            d, sc, bad = load(IND / "test_eval/test" / s / f"convs_only_s{st}_r{rn}.npz"); arm[s] = (sc, bad)
            if s in fm:
                r = pr_sweep(d["score"], d["top1_geo_dist"], classify_queries(d["min_past_gt_dist"])); am[s] = (r["AP"], r["R@100P"])
        gain = 100 * (macro(am) - macro(fm))     # 검산: 분석 파일의 실행별 이득
        for j, key in enumerate(("AP", "R@100P")): assert abs(gain[j] - ana["convs_only"]["metrics"][key]["per_run"][st - 1][rn - 1]) < 2e-5, (st, rn, key, gain[j])
        arms[k] = arm; curves.append(reduction(fixed, arm, seqs))
    rng = np.random.default_rng(20261113); groups = {}
    for s in seqs: groups.setdefault(city(s), []).append(s)
    boot = []
    for _ in range(1000):
        names = [s for v in groups.values() for s in rng.choice(v, len(v))]; boot.append(np.mean([reduction(fixed, arms[k], names) for k in arms], 0))
    lo, hi = np.quantile(boot, [.025, .975], 0); n = sum(len(v[0]) for v in fixed.values())
    crossed = json.loads((IND / "posthoc_crossed.json").read_text())["results"]["convs_only"]   # 본문이 쓰는 구간: 표본 시드와 시험 시퀀스를 함께 재표집
    return dict(metrics={k: dict(mean_pp=v["mean_pp"], interval=crossed[k]["B_crossed"], positive_runs=v["positive_runs"], per_run=v["per_run"]) for k, v in ana["convs_only"]["metrics"].items()},
                fraction=FRAC.tolist(), curves=np.array(curves).tolist(), mean=np.mean(curves, 0).tolist(), lo=lo.tolist(), hi=hi.tolist(), n_queries=int(n),
                fixed_false_at=dict(zip(("25", "50", "75", "100"), (100 * np.cumsum(np.concatenate([fixed[s][1] for s in seqs])[np.argsort(-np.concatenate([fixed[s][0] for s in seqs]), kind="stable")])[(np.array([.25, .5, .75, 1]) * n).astype(int) - 1] / n).tolist())),
                sources=dict(analysis=hashlib.sha256((IND / "analysis.json").read_bytes()).hexdigest(), script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def draw(d: dict) -> None:
    plt.rcParams.update({"font.family": "Carlito", "font.size": FONT["label"], "pdf.fonttype": 42, "text.color": NAVY, "axes.labelcolor": NAVY,
                         "axes.edgecolor": EDGE, "axes.linewidth": .6, "xtick.color": NAVY, "ytick.color": NAVY})
    only_a = "--panel-a" in sys.argv
    fig = plt.figure(figsize=(3.25, 1.62 if only_a else 3.55))
    a = fig.add_axes([.19, .255, .775, .725] if only_a else [.19, .585, .775, .345]); b = fig.add_axes([.19, .105, .775, .33]) if not only_a else None
    rows = (("R@1", "R@1"), ("R@100P", "R@100P"), ("max_F1", "Max F1"), ("AP", "AP"))
    a.axvline(0, color=EDGE, lw=.7, ls=(0, (3, 2)), zorder=2)
    for y, (key, lab) in enumerate(rows):
        m = d["metrics"][key]; lo, hi = m["interval"]
        a.plot([lo, hi], [y, y], color=NAVY, lw=.9, zorder=3, solid_capstyle="butt")
        for x in (lo, hi): a.plot([x, x], [y - .11, y + .11], color=NAVY, lw=.9, zorder=3)
        for si, (col, mk) in enumerate(START):
            a.scatter(m["per_run"][si], np.full(5, y + (1 - si) * .2), s=13, marker=mk, color=col, edgecolor="white", linewidth=.35, zorder=4, label=f"Start {si + 1}" if y == 0 else None, alpha=.9)
        a.scatter([m["mean_pp"]], [y], s=30, marker="D", color=NAVY, edgecolor="white", linewidth=.4, zorder=5)
        a.text(2.02, y, f"{m['mean_pp']:+.2f}".replace("-", "\u2212"), ha="right", va="center", fontsize=FONT["note"]); a.text(2.52, y, f"{m['positive_runs']}/15", ha="right", va="center", fontsize=FONT["note"])
    a.text(2.02, 3.66, "mean", ha="right", va="center", fontsize=FONT["note"] - .4, color=EDGE); a.text(2.52, 3.66, "runs > 0", ha="right", va="center", fontsize=FONT["note"] - .4, color=EDGE)
    a.set(yticks=range(4), yticklabels=[l for _, l in rows], xlim=(-.55, 2.56), xticks=[-.5, 0, .5, 1, 1.5], ylim=(-.55, 3.95), xlabel="Change from uniform subtraction (points)")
    a.legend(frameon=False, fontsize=FONT["note"] - .4, loc="upper left", bbox_to_anchor=(-.01, 1.02), ncol=3, handletextpad=.0, columnspacing=.5, borderaxespad=.2)
    if only_a:
        a.tick_params(labelsize=FONT["tick"], length=2, pad=2); a.grid(color=GRID, lw=.45); a.set_axisbelow(True); a.grid(axis="y", visible=False)
        dest = ROOT / "docs/paper/body_v2.3/figs/pair_analysis"
        for ext in ("pdf", "png"): fig.savefig(dest / f"fig_learning_gain.{ext}", dpi=300)
        return
    x = 100 * np.array(d["fraction"]); b.axhline(0, color=ZERO, lw=.8, zorder=2)
    b.fill_between(x, d["lo"], d["hi"], color="#156082", alpha=.18, edgecolor="none", zorder=1)
    for c, (st, _) in zip(d["curves"], [(s, r) for s in range(3) for r in range(5)]): b.plot(x, c, color=START[st][0], lw=.45, alpha=.75, zorder=3)
    b.plot(x, d["mean"], color=NAVY, lw=1.7, zorder=5)
    b.set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100], xlabel="Queries accepted (%)", ylabel="Fewer false accepts\nper 100 queries")
    for ax, t in ((a, "(a) Gain from DiffAttnConv learning, 15 runs"), (b, "(b) False accepts removed at equal acceptance")):
        ax.set_title(t, fontsize=FONT["title"], weight="bold", pad=4); ax.tick_params(labelsize=FONT["tick"], length=2, pad=2); ax.grid(color=GRID, lw=.45); ax.set_axisbelow(True)
    a.grid(axis="y", visible=False)
    for ext in ("pdf", "png"): fig.savefig(OUT / f"fig_learning_gain.{ext}", dpi=300)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); path = OUT / "learning_gain.json"
    d = json.loads(path.read_text()) if "--draw-only" in sys.argv and path.exists() else compute()
    path.write_text(json.dumps(d) + "\n"); draw(d)
    m = np.array(d["mean"]); print("mean reduction at 25/50/75/100%:", m[[24, 49, 74, 99]].round(3), "| band at 50%:", round(d["lo"][49], 3), round(d["hi"][49], 3), "| uniform false per 100 at 25/50/75/100%:", {k: round(v, 2) for k, v in d["fixed_false_at"].items()})


if __name__ == "__main__":
    main()
