#!/usr/bin/env python3
"""문맥 수락 대조의 계열별 구분력과 최종 검증 결과를 시각화한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]
import _pdv2_context_accept_fp32 as effective
run = effective.run
effective.configure()
from _pdv2_context_accept_lib import auc, labels
from _pdv2_no_context_ablation import sha256, write_json


def main() -> None:
    """선택을 바꾸지 않고 합산 진단과 표본 표준편차를 별도로 보고한다."""
    spec = run.exp.read(run.OUT / "protocol.json")
    summary = run.exp.read(run.OUT / "summary.json")
    assert run.exp.read(run.OUT / "completion_validation.json")["status"] == "passed"
    report = {"std_definition": "시드3표본표준편차ddof=1. 원summary의std는ddof=0이며평균·선택은같음.",
              "auc_definition": "각시드·계열내질의합산ROC-AUC. 최종AP가아니며모호질의제외. baseline_accepted는기존검증고정임계값영역.",
              "mean_std": {}, "discrimination": {}, "selected": summary["selected"], "test_used": False}
    for role in ("train", "val"):
        report["mean_std"][role] = {}
        for method in run.METHODS:
            report["mean_std"][role][method] = {m: {
                "mean": float(np.mean([summary["per_seed"][str(s)][role][method]["macro"][m] for s in (1,2,3)])),
                "std": float(np.std([summary["per_seed"][str(s)][role][method]["macro"][m] for s in (1,2,3)], ddof=1))} for m in run.exp.METRICS}
        report["discrimination"][role] = {}
        for seed in (1, 2, 3):
            report["discrimination"][role][str(seed)] = {}
            for group in sorted({run.family(s) for s in spec[role]}):
                arrays, predictions = [], []
                for sequence in spec[role]:
                    if run.family(sequence) != group:
                        continue
                    arrays.append(run.exp.arrays(run.feature_file(role, seed, sequence)))
                    predictions.append(run.exp.arrays(run.OUT/f"scores/{role}/s{seed}/{sequence.replace('/', '_')}.npz"))
                merged = {f: np.concatenate([a[f] for a in arrays]) for f in arrays[0]}
                scores = {f: np.concatenate([a[f] for a in predictions]) for f in predictions[0]}
                eligible, y = labels(merged)
                tau = spec["seeds"][str(seed)]["new_tau"]["tau"]
                result = {}
                for cohort, mask in (("eligible", eligible), ("baseline_accepted", eligible & (merged["baseline"] >= tau))):
                    result[cohort] = {"correct": int(y[mask].sum()), "incorrect": int((1-y[mask]).sum()),
                        "feature_auc": {f: auc(merged[f][mask], y[mask]) for f in run.FEATURES},
                        "fitted_auc": {f: auc(scores[f][mask], y[mask]) for f in run.METHODS}}
                report["discrimination"][role][str(seed)][group] = result
    write_json(run.OUT / "report.json", report)
    plot(summary, report)
    paths = [Path(__file__), run.OUT/"summary.json", run.OUT/"report.json", run.OUT/"completion_validation.json",
             run.OUT/"final_validation.png", run.OUT/"final_validation.pdf"]
    write_json(run.OUT / "report_validation.json", {"status": "passed", "test_used": False,
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in paths}})
    for method in run.METHODS:
        value = report["mean_std"]["val"][method]
        print("VAL", method, {k: [round(v["mean"]*100, 4), round(v["std"]*100, 4)] for k, v in value.items()}, flush=True)
    for role in ("train", "val"):
        print("AUC", role, report["discrimination"][role]["1"], flush=True)


def plot(summary: dict, report: dict) -> None:
    """최종 검증 지표·자료별 손익·기존 수락 영역의 구분력을 그린다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/tmp/nsd_context_accept_font.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    methods = ("baseline", "add_d_arc", "add_context_center", "add_context_arc", "add_d_context_arc")
    names = ("기존 시간 판정", "+ 원래 특징·이웃", "+ 문맥·중심", "+ 문맥·이웃", "+ 원래 특징·문맥 이웃")
    colors = ("#64748b", "#d97706", "#0891b2", "#2563eb", "#7c3aed")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    y = np.arange(len(methods))
    for ax, metric, title in ((axes[0,0], "AP", "최종 AP (%)"), (axes[0,1], "R@100P", "최종 R@100P (%)")):
        values = [report["mean_std"]["val"][m][metric] for m in methods]
        means = np.array([v["mean"]*100 for v in values])
        ax.barh(y, means, xerr=[v["std"]*100 for v in values], color=colors, alpha=.85, capsize=3)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel(title+" · 3시드 평균±표본 표준편차")
        ax.grid(axis="x", alpha=.2)
        for i, value in enumerate(means):
            ax.text(value+.15, i+.16, f"{value:.2f}", fontsize=9)
        ax.set_xlim(max(0, means.min()-8), means.max()+8)
    ax = axes[1,0]
    families = ("kitti", "helipr", "mulran")
    width = .17
    for index, method in enumerate(methods[1:]):
        delta = [100*(summary["mean_std"]["val"][method]["families"][f]["AP"]-summary["mean_std"]["val"]["baseline"]["families"][f]["AP"]) for f in families]
        ax.bar(np.arange(3)+(index-1.5)*width, delta, width, label=names[index+1], color=colors[index+1])
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(np.arange(3), ("KITTI 계열", "HeLiPR", "MulRan"))
    ax.set_ylabel("기존 판정 대비 최종 AP 차이 (%p)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=.2)
    ax = axes[1,1]
    for index, feature in enumerate(("baseline", "d_arc", "context_arc")):
        values = [np.mean([report["discrimination"]["val"][str(s)][f]["baseline_accepted"]["feature_auc"][feature] for s in (1,2,3)]) for f in families]
        ax.bar(np.arange(3)+(index-1)*.23, values, .23, color=(colors[0], colors[1], colors[3])[index], label=("스펙트럼 판정", "원래 특징·이웃", "문맥·이웃")[index])
    ax.axhline(.5, color="black", linestyle="--", linewidth=.8)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(3), ("KITTI 계열", "HeLiPR", "MulRan"))
    ax.set_ylabel("기존 수락 영역에서 정답/오답 ROC-AUC")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=.2)
    fig.suptitle("기존 문맥을 최종 수락에 연결한 고정 승자 대조\n학습 8에서만 가중치 적합 → 검증 9 적용 · 모델·검색·최종 승자 고정 · 시험 미사용", fontsize=15)
    for extension in ("png", "pdf"):
        fig.savefig(run.OUT/f"final_validation.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
