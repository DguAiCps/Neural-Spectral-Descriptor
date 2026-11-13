#!/usr/bin/env python3
"""시간 문맥 분석과 완료된 최종 판정 결과를 표·그림으로 정리한다."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_temporal_focus_20261018"
DEST = OUT / "reports"
OPS = ("pbev128_main", "hspec12")
METRICS = ("AP", "max_F1", "R@100P", "R@1")


def read(path: Path) -> dict:
    """JSON 기록을 읽는다."""
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    """내용 해시를 구한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, data: dict) -> None:
    """유한 JSON만 저장한다."""
    (DEST / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def plotting() -> object:
    """공유 가능한 독립 그림의 글꼴과 백엔드를 설정한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 10})
    return plt


def diagnosis() -> None:
    """AP 분해·순위 교차의 재집계와 시각화를 한다."""
    p = OUT / "diagnosis"
    ap = read(p / "ap_decomposition.json")["results"]
    cats = ("recovered", "regressed", "both_correct", "both_wrong", "negative")
    means = {op: {family: {k: float(np.mean([ap[f"s{s}_{op}"]["macro" if family == "macro" else "families"][k]
                       if family == "macro" else ap[f"s{s}_{op}"]["families"][family][k] for s in (1, 2, 3)]))
                       for k in ("before", "after", *cats)} for family in ("macro", "kitti", "helipr", "mulran")} for op in OPS}
    for fams in means.values():
        for row in fams.values():
            assert abs(sum(row[k] for k in cats) - (row["after"]-row["before"])) < 1e-12
    ranking = {op: collections.defaultdict(collections.Counter) for op in OPS}
    for path in sorted(p.glob("ranking_*.json")):
        for key, rows in read(path)["cases"].items():
            op = key.split("_", 1)[1]
            for r in rows:
                c = ranking[op][r["type"]]
                c["total"] += 1
                if not r["new_pool_has_true"]:
                    assert r["type"] == "regressed"
                    c["true_missing"] += 1
                    continue
                m = r["correct_matrix_key_by_pool"]
                keyeffect = bool(m[1][0]) != bool(m[0][0])
                pooleffect = bool(m[0][1]) != bool(m[0][0])
                c["both" if keyeffect and pooleffect else "key_only" if keyeffect else "pool_only" if pooleffect else "joint"] += 1
    for v in ranking.values():
        for c in v.values():
            assert c["total"] == sum(v for k, v in c.items() if k != "total")
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    names = ("정답 복구", "정답 손실", "둘 다 정답", "둘 다 오답", "비재방문")
    for ax, op, title in zip(axes, OPS, ("NSD", "NSD-H")):
        vals = np.array([means[op]["macro"][k] for k in cats]) * 100
        ax.barh(names[::-1], vals[::-1], color=["#246CA0" if x >= 0 else "#C25A4D" for x in vals[::-1]])
        ax.axvline(0, color="#444444", linewidth=.8)
        ax.set_xlim(-.9, 1.)
        for i, v in enumerate(vals[::-1]):
            ax.text(v + (.02 if v >= 0 else -.02), i, f"{v:+.3f}", ha="left" if v >= 0 else "right", va="center", fontsize=9)
        ax.set_title(f"{title}: 전체 {(means[op]['macro']['after']-means[op]['macro']['before'])*100:+.3f}%p")
        ax.set_xlabel("최종 AP 차이에 배분된 몫 (%p)")
    fig.suptitle("시간 문맥과 이웃 없는 투영의 차이 · 검증 3시드 평균\n질의군 교체 순서 전체 평균에 따른 회계 분해 (인과 효과 아님)", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(DEST / f"ap_decomposition.{suffix}", dpi=180)
    plt.close(fig)
    write("diagnosis_summary.json", {"ap": means, "ranking": ranking,
        "learning_signal": read(p / "learning_signal.json")["results"], "test_used": False,
        "scope": "같은 검증 질의가 세 시드에서 반복 집계됨. 원인 확정이나 독립 표본 수 아님."})
    sources = {str(x.relative_to(REPO)): sha(x) for x in p.glob("*.json")}
    write("diagnosis_report_validation.json", {"status": "passed", "sources": sources,
        "script_sha256": sha(Path(__file__)), "ap_additive_identity": True, "ranking_counts_reconciled": True})
    print("DIAGNOSIS", means, ranking)


def results() -> None:
    """선택된 최종 지표를 재계산하고 예비·확증을 구분해 내보낸다."""
    summary = read(OUT / "summary.json")
    assert read(OUT / "completion_validation.json")["status"] == "passed"
    rows, effects = [], {}
    for seed, arms in summary["per_seed"].items():
        effects[seed] = {}
        for arm, r in arms.items():
            for op in OPS:
                vals = r["ops"][op]["macro"]
                rows.append({"seed": seed, "arm": arm, "epoch": r["epoch"], "op": op, **{m: vals[m]*100 for m in METRICS}})
            if arm.startswith("point"):
                continue
            reference = arms["point_" + arm.split("_")[1]]
            effects[seed][arm] = {op: {m: (r["ops"][op]["macro"][m]-reference["ops"][op]["macro"][m])*100 for m in METRICS} for op in OPS}
            effects[seed][arm]["families"] = {op: {f: {m: (r["ops"][op]["by_group"][f][m]-reference["ops"][op]["by_group"][f][m])*100 for m in METRICS}
                    for f in r["ops"][op]["by_group"]} for op in OPS}
    with (DEST / "final_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    winner = summary["expansion"]["winner"]
    stats = {}
    if winner:
        for op in OPS:
            stats[op] = {m: {"mean_pp": float(np.mean([effects[s][winner][op][m] for s in ("1", "2", "3")])),
                            "std_pp": float(np.std([effects[s][winner][op][m] for s in ("1", "2", "3")], ddof=1)),
                            "values_pp": [effects[s][winner][op][m] for s in ("1", "2", "3")]} for m in METRICS}
    write("final_report.json", {"effects_pp": effects, "winner_three_seed_effects": stats, "expansion": summary["expansion"], "test_used": False})
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    labels = {"diff_original": "차이·기존", "standard_original": "표준·기존", "diff_opposite": "차이·반대25%",
              "standard_opposite": "표준·반대25%", "point_original": "투영·기존", "point_opposite": "투영·반대25%"}
    arms = list(summary["per_seed"]["1"])
    for ax, op, name in zip(axes, OPS, ("NSD", "NSD-H")):
        vals = [next(r["AP"] for r in rows if r["seed"] == "1" and r["arm"] == a and r["op"] == op) for a in arms]
        ax.plot(vals, np.arange(len(arms)), "o", color="#246CA0")
        ax.set_yticks(range(len(arms)), [labels[a] for a in arms])
        ax.invert_yaxis()
        ax.set_title(name)
        ax.set_xlabel("최종 AP (%)")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("시간 문맥 집계·학습 쌍 비교 · 검증 시드 1 · 조건별 최종 AP로 체크포인트 선택", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(DEST / f"pilot.{suffix}", dpi=180)
    plt.close(fig)
    write("final_report_validation.json", {"status": "passed", "rows": len(rows), "script_sha256": sha(Path(__file__)),
        "sources": {str(p.relative_to(REPO)): sha(p) for p in (OUT / "summary.json", OUT / "completion_validation.json", OUT / "protocol.json")}})
    print("RESULTS", rows, stats)


def final_plot() -> None:
    """세 시드와 용량 대조를 포함한 최종 AP·R@100P 그림을 만든다."""
    path = DEST / "post_analysis.json"
    report = read(path)
    plt = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    methods = ("point", "capacity_point", "diff", "standard")
    labels = ("작은 투영", "용량 맞춘 투영", "차이 시간 문맥", "표준 시간 문맥")
    colors = ("#8A8A8A", "#535353", "#C28D45", "#246CA0")
    for row, (op, name) in enumerate(zip(OPS, ("NSD", "NSD-H"))):
        for col, (metric, title) in enumerate((("AP", "최종 AP"), ("R@100P", "정밀도 100%에서 재현율"))):
            ax = axes[row, col]
            for y, (method, color) in enumerate(zip(methods, colors)):
                stats = report["mean_std"][method][op][metric]
                values = np.array([report["per_seed"][str(s)][method][op][metric] for s in (1, 2, 3)]) * 100
                ax.errorbar(stats["mean"]*100, y, xerr=stats["std"]*100,
                            fmt="o", color=color, capsize=4)
                ax.scatter(values, y + np.array([-.12, 0, .12]), color=color, s=12, alpha=.6)
            ax.set_yticks(range(len(methods)), labels)
            ax.invert_yaxis()
            ax.set_title(name + " · " + title)
            ax.set_xlabel("%")
            ax.grid(axis="x", alpha=.2)
            ax.margins(x=.15)
    fig.suptitle("동일 기존 학습 쌍 · 검증 9시퀀스 · 3시드 평균과 표본 표준편차\n시간 모델 167,561개 / 작은 투영 95,617개 / 용량 대조 167,705개 파라미터", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(DEST / f"three_seed_capacity.{suffix}", dpi=180)
    plt.close(fig)
    write("final_plot_validation.json", {"status": "passed", "source_sha256": sha(path), "script_sha256": sha(Path(__file__))})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("diagnosis", "results", "final_plot"))
    args = p.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    globals()[args.action]()
