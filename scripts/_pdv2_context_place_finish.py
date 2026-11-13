#!/usr/bin/env python3
"""장소 제외 선택과 높은 정밀도 손실을 검산하고 그림으로 정리한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/"src"), str(REPO/"scripts")]
import _pdv2_context_place as run
import _pdv2_context_place_smallstep as small
from _pdv2_matching_mine import cache_path
from _pdv2_no_context_ablation import sha256, write_json


def verify_geometry() -> dict:
    """원 자세에서 거리·국소 주행 방향·직선 정도를 다시 계산한다."""
    record = run.old.exp.read(run.OUT/"high_precision_geometry.json")
    for sequence, row in record["results"].items():
        positions = run.old.exp.arrays(cache_path(sequence))["poses"][:, :3, 3].astype(float)
        pair = row["0.0"]["first_incorrect"][0]
        q, j = pair["query"], pair["winner"]
        travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
        query_direction = positions[q+5, :2]-positions[q-5, :2]
        candidate_direction = positions[j+5, :2]-positions[j-5, :2]
        cosine = query_direction @ candidate_direction
        cosine /= np.linalg.norm(query_direction)*np.linalg.norm(candidate_direction)
        expected = {
            "keyframe_gap": q-j,
            "travel_distance_between_centers_m": float(travel[q]-travel[j]),
            "heading_difference_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
            "center_displacement_m": float(np.linalg.norm(positions[q]-positions[j])),
            "query_local_straightness": float(np.linalg.norm(positions[q+5]-positions[q-5])/(travel[q+5]-travel[q-5])),
            "candidate_local_straightness": float(np.linalg.norm(positions[j+5]-positions[j-5])/(travel[j+5]-travel[j-5])),
        }
        for name, value in expected.items():
            np.testing.assert_allclose(value, row["geometry"][name], atol=1e-12, rtol=0)
    return record


def plot(oof: dict, final: dict) -> None:
    """평균 이득과 시퀀스별 높은 정밀도 손실을 나란히 그린다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/tmp/nsd_context_accept_font.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric, title in ((axes[0, 0], "AP", "학습 내부 장소 제외: 최종 AP"),
                                (axes[0, 1], "R@100P", "학습 내부 장소 제외: 최종 R@100P")):
        for feature, label, color in (("context_arc", "문맥 이웃", "#2563eb"),
                                      ("d_arc", "원래 특징 이웃", "#d97706")):
            grid = oof["grids"][feature]
            axis.plot([r["factor"]*100 for r in grid], [r[metric]*100 for r in grid],
                      marker="o", color=color, label=label)
        axis.set_xlabel("학습한 추가 점수 비중의 적용률 (%)")
        axis.set_ylabel(metric+" (%) · 3시드 평균")
        axis.set_title(title)
        axis.grid(alpha=.2)
        axis.legend(fontsize=9)
    axis = axes[1, 0]
    methods = ("baseline", "context_ap", "context_robust")
    labels = ("기존 판정", "AP 선택: 문맥 100%", "보수적 선택: 문맥 0%")
    values = [final["mean_std"][m]["macro"]["AP"]["mean"]*100 for m in methods]
    errors = [final["mean_std"][m]["macro"]["AP"]["std"]*100 for m in methods]
    axis.barh(np.arange(3), values, xerr=errors, capsize=3,
              color=("#64748b", "#d97706", "#2563eb"))
    axis.set_yticks(np.arange(3), labels)
    axis.invert_yaxis()
    axis.set_xlim(68, 84)
    axis.set_title("검증 9: 학습 내부에서 고른 조건만 적용")
    axis.set_xlabel("최종 AP (%) · 3시드 평균±표본 표준편차")
    for i, value in enumerate(values):
        axis.text(value+errors[i]+.3, i+.07, f"{value:.2f}", fontsize=10)
    axis.grid(axis="x", alpha=.2)
    axis = axes[1, 1]
    rows = oof["statistics"]["1"]["helipr_bridge"]["context_arc"]
    sequences = [f"helipr/Bridge0{i}" for i in (1, 2, 3, 4)]
    delta = [100*(rows["0.05"]["per_sequence"][s]["R@100P"]-rows["0.0"]["per_sequence"][s]["R@100P"]) for s in sequences]
    colors = ["#2563eb" if d >= 0 else "#dc2626" for d in delta]
    axis.bar(np.arange(4), delta, color=colors)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xticks(np.arange(4), [s.split("/")[1] for s in sequences])
    axis.set_ylabel("R@100P 차이 (%p)")
    axis.set_title("문맥 5% · 시드 1: 장소 평균이 가리는 손실")
    for i, difference in enumerate(delta):
        axis.text(i, difference+(.12 if difference >= 0 else -.3),
                  f"{difference:+.2f}", ha="center")
    axis.set_ylim(min(delta)-.8, max(delta)+.6)
    axis.grid(axis="y", alpha=.2)
    fig.suptitle("문맥 비중의 장소 간 전이와 높은 정밀도 경계\n가중치 적합만 장소 분리 · 기존 문맥 모델은 학습 장소를 본 상태 · 시험 미사용", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(run.OUT/f"place_transfer.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    """완료 검사를 확인하고 해석에 필요한 정확한 수치를 저장한다."""
    for root in (run.OUT, small.OUT):
        assert run.old.exp.read(root/"completion_validation.json")["status"] == "passed"
    oof = run.old.exp.read(run.OUT/"oof_summary.json")
    final = run.old.exp.read(run.OUT/"summary.json")
    followup = run.old.exp.read(small.OUT/"summary.json")
    geometry = verify_geometry()
    plot(oof, final)
    report = {
        "choices": oof["choices"], "smallstep_factor": followup["factor"],
        "validation": final["mean_std"], "high_precision_examples": geometry["results"],
        "conclusion": "비중적합의장소분리만으로안정적문맥기여를확보하지못함. AP선택100%는검증악화,보수적선택과시퀀스정수손실기준은0%. 5%는학습OOF평균개선이나일부시퀀스R100정답23개씩감소. 동일직선주행중멀리떨어진오답에높은스펙트럼·문맥점수가함께발생.",
        "causal_limit": "두경계쌍의형태진단을전체오답원인비율로확대하지않음. 문맥모델·재계량·승자생성은전체학습자료노출,가중치적합만장소분리. 시퀀스단위1TP검사는장소단위0.1%p검사의단순완화가아니라집계단위도다른후속진단.",
        "next": "문맥인코더에서도장소를완전히제외한대조와,같은방향직선의다른위치쌍을과신하지않도록하는학습/판정연결을분리확인. 평가제외폭을늘려오답을숨기지않음.",
        "test_used": False,
    }
    write_json(run.OUT/"report.json", report)
    paths = [Path(__file__), run.OUT/"completion_validation.json", small.OUT/"completion_validation.json",
             run.OUT/"oof_summary.json", run.OUT/"summary.json", small.OUT/"summary.json",
             run.OUT/"report.json", run.OUT/"high_precision_geometry.json",
             run.OUT/"place_transfer.png", run.OUT/"place_transfer.pdf"]
    write_json(run.OUT/"report_validation.json", {
        "status": "passed", "test_used": False,
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in paths}})
    print("REPORT_PASSED", flush=True)


if __name__ == "__main__":
    main()
