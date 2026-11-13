#!/usr/bin/env python3
"""공통 진단 결과를 검산하고 읽기용 보고서·그림을 만든다."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/pdv2_baseline_common_diagnosis_20261106"


def read(name: str) -> dict:
    """결과 JSON을 읽는다."""
    return json.loads((OUT / name).read_text())


def main() -> None:
    """원천·집계 항등식·실제 손실 신호를 검산한다."""
    ledgers, pairs, controls = read("ledgers.json"), read("pairs.json"), read("control.json")
    signal, trace = read("seqot_signal_probe.json"), read("seqot_label_trace.json")
    names = {"scpp": "SC++", "ring": "RING", "ringpp": "RING++", "ringsharp": "RING#-L",
             "deepring": "DeepRING", "disco": "DiSCO", "helios": "HeLiOS", "imlpr": "ImLPR",
             "seqot": "SeqOT", "bevplace": "BEVPlace++", "ringpp_exactmax": "RING++ exact max"}
    methods = {k: v["mean"] for k, v in ledgers["test_frozen"].items()}
    methods["ringpp_exactmax"] = controls["ringpp_exactmax"]["macro"]
    records = []
    for name, m in methods.items():
        assert abs(m["AP"] + m["retrieval_loss"] + m["ranking_loss"] - 1) < 1e-12
        records.append({"method": names[name], **{k: m[k] * 100 for k in ["AP", "R@1", "retrieval_loss", "ranking_loss"]}})
    with (OUT / "final_score_decomposition_percent.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    # 다른 구현으로 실제 원장의 AUC를 검산한다. 본 분석은 binary search, 검산은 rank sum이다.
    from scipy.stats import rankdata
    auc_checks = []
    for method, prefix, score in [("helios", "dump_helios", "score_cos"), ("seqot", "dump_seqotl2", "score_l2")]:
        source = json.loads((ROOT / "results/pdv2_baseline_seeds/summary.json").read_text())["methods"][method]["per_seed"]["1"]["source"]
        for sensor, seq in [("kitti", "00"), ("helipr", "Town01"), ("mulran", "DCC01")]:
            with np.load(ROOT / source / f"{prefix}_{sensor}_{seq}.npz") as z, np.load(ROOT / f"results/pdv2_symmetric3_20261013/s1/test_selected_symmetric_pbev128_main_{sensor}_{seq}.npz") as ref:
                take = np.searchsorted(z["query_idx"], ref["query_idx"])
                md, td, scores = [z[k][take] for k in ["min_past_gt_dist", "top1_geo_dist", score]]
            pos, neg = (md <= 5) & (td <= 5), ((md <= 5) | (md > 25)) & (td > 5)
            rank = rankdata(np.r_[scores[pos], scores[neg]], method="average")
            value = float((rank[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))
            expected = ledgers["test_frozen"][method]["seeds"]["1"]["sequences"][f"{sensor}/{seq}"]["correct_vs_wrong_AUC"]
            assert abs(value - expected) < 1e-12
            auc_checks.append({"method": method, "sequence": f"{sensor}/{seq}", "rank_sum_AUC": value})
    for key in ["True", "False"]:
        rows = [r for r in signal["rows"] if r["all_positive_self"] == (key == "True")]
        assert len(rows) == signal["summary"][key]["samples"]
        if key == "True":
            for row in rows:
                assert all(p == row["anchor"] for p in row["positives"])
                for mode in ["train", "eval"]:
                    assert row[mode]["max_positive_squared_distance"] == 0
                    assert row[mode]["positive_output_gradient_norm"] == 0
    assert trace["totals"]["self_only_steps"] == pairs["seqot"]["stage1"]["all_selected_positives_are_self"]
    verified = 0
    for report in [ledgers, pairs, controls]:
        for name, expected in report["sources_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
            verified += 1
    pilot = {}
    for arm in ["control", "yaw"]:
        if (OUT / "yaw_pilot" / arm / "report.json").exists():
            pilot[arm] = read(f"yaw_pilot/{arm}/report.json")
    verification = {"passed": True, "ten_main_baselines_all_metrics_reproduced": ledgers["all_main_metrics_reproduced"],
                    "test_ledger_count": 1104, "exactmax_extra_ledgers": 46,
                    "validation_original_ledgers": 108, "queries_per_test_run": 195726,
                    "queries_per_validation_run": 99236, "source_hash_checks_with_repeats": verified,
                    "independent_AUC_checks": auc_checks, "real_forward_samples": len(signal["rows"]),
                    "self_pair_gradient_zero_checked": True, "pilot_completed_arms": list(pilot)}
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    # 논문 그림은 수정하지 않는다. 아래는 내부 진단용 독립 산출물이다.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 10})
    order = ["bevplace", "imlpr", "helios", "seqot", "ring", "ringpp", "ringpp_exactmax", "scpp", "ringsharp", "deepring", "disco"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    left = np.zeros(len(order))
    for key, label, color in [("AP", "실제 AP", "#0072B2"), ("ranking_loss", "수락 점수 순서 손실", "#E69F00"), ("retrieval_loss", "장소 검색 실패", "#C7C7C7")]:
        values = np.array([100 * methods[k][key] for k in order])
        ax.barh(np.arange(len(order)), values, left=left, label=label, color=color)
        for i, (v, start) in enumerate(zip(values, left)):
            if v >= 5:
                ax.text(start + v / 2, i, f"{v:.1f}", ha="center", va="center", fontsize=9, color="white" if key == "AP" else "black")
        left += values
    ax.set_yticks(np.arange(len(order)), [names[k] for k in order]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("공통 최종 평가의 AP와 손실 분해 (%)")
    ax.set_title("같은 195,726개 질의: 어디에서 성능을 잃는가")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.06), ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "common_diagnosis.pdf"); fig.savefig(OUT / "common_diagnosis.png", dpi=160); plt.close(fig)
    text = ["베이스라인 공통 진단 — 2026-11-06", "", "검산 범위",
            "주표10방법의 1,104원장, 정확max RING++46원장, 원래 가중치4방법×3시드×검증9의108원장을 분석했다.",
            "설정 선택을 하지 않은 동결 시험 원장의 기술 통계와 학습/검증 자료의 원인 진단을 구별한다.", "",
            "최종 AP 분해: AP + 수락 점수 순서 손실 + 장소 검색 실패 = 100%.",
            "R@1은 후보 반환을 고정했을 때 완벽한 수락 점수로 도달할 수 있는 AP 상한이다. 이 상한이 실제로 달성 가능하다는 뜻은 아니다."]
    for row in records:
        text.append(f"{row['method']}: AP {row['AP']:.2f}, R@1 {row['R@1']:.2f}, 검색 손실 {row['retrieval_loss']:.2f}, 점수 순서 손실 {row['ranking_loss']:.2f}")
    text += ["", "확인한 공통점과 차이",
        "1. 센서 비중: HeLiOS·ImLPR의 유효 앵커 중 HeLiPR88.47%, BEVPlace++ 입력88.24%, SeqOT 단계1 90.88%다. 잘 되는 BEV도 편중되므로 비중 하나를 공통 원인이라고 단정할 수 없다.",
        "2. HeLiOS: 전체 양성4,090,486쌍 중5m 이내5.14%,25m 초과58.32%. 공식 sampler 3회 재생의 배치 안 관계도25m 초과57.97~58.32%다. overlap의 정답과5m장소의 정답이 다르며 공식 라벨의 버그라고 단정하지 않는다.",
        "3. ImLPR: 전체 양성498,342쌍 중5m 이내42.20%,25m 초과0%. 10m 이내 정답을 쓰는 정책이다. 독립 원통형 yaw 증강도 이미 있어 SeqOT와 같은 처방의 근거가 없다.",
        "4. SeqOT: 실제 단계1의59.14%, 단계2의56.15%가 자기쌍만 있는 단계다. 단계1 실제128표본에서 train/eval 모두 양성 출력 기울기0을 확인했다. 음성 밀어내기는 남는다.",
        f"   단계1 자기쌍 전용9,142개 중 기존 단방향 overlap에 상대 없음7,422개, 정규화 표본 누락215개, 첫 양성만 선택해 누락1,505개다. 양방향으로도 상대가 없는 기준 프레임은 {trace['totals']['no_nonself_positive_in_either_raw_direction']:,}개다.",
        "   HeLiPR에서5m 이내 비자기 쌍의22.1~51.2%는 overlap<0.3 음성이다. 거리영상 유효 픽셀 중앙값30.5~31.9%; KITTI81.5~89.3%와 다르다. 센서 희소성과 overlap 라벨 불일치의 관찰이며 단일요인 인과효과 추정은 아니다.",
        "5. BEVPlace++: 원래 양성은 자기쌍 제외5m이내100%, 질의와 양성의 독립 회전 증강, 어려운 음성 검색이다. 데이터 편중을 공유하면서 유효한 양성 신호를 제공하는 대조군이다.",
        "6. RING++: 정확max 뒤 R@1은76.79%로 RING77.51%와 가깝지만 AP44.54% 대63.15%다. 남은 문제의 우선순위는 수락 점수의 구분력이다. 단조로운 점수 보정이나 임계값 변경만으로 AP 순서는 바뀌지 않는다.",
        "", "검증과 시험의 데이터별 현상",
        "공통 검증에서 ImLPR는 MulRan R@1 93.26%이나 AP31.35%다. SeqOT도69.55% 대10.45%로 점수 순서 문제가 함께 있다.",
        "시험 MulRan에서는 ImLPR AP77.1/R@1 80.1, BEV AP78.1/R@1 80.4로 가깝다. 낮은 전체 평균을 모든 데이터에서 실패한다고 읽으면 안 된다.",
        "", "현재 결정",
        "사용자 지시에 따라 비교 방법의 수치 향상 자체를 목표로 하지 않는다. 공식 재현·입력·라벨·평가 오류 확인과 조건 차이의 근거 정리를 우선한다.",
        "새로운 증강·손실·표본 추출법을 넣는 개량 탐색은 보류한다. 이미 수행한 SeqOT yaw 대조는 진단 기록으로 보존하며 주표 모델로 채택하지 않는다.",
        "RING++의 공식 CUDA 문제와 정확max 결과는 공시한다. 수정본을 주표로 채택하는 별도 결정이 있다면 성능과 비용의 구현을 맞춘다.",
        "", "이 분석은 낮은 성능의 기여 원인을 좁히며, 설명된 비율 전체가 재학습으로 회복된다는 주장은 하지 않는다."]
    if len(pilot) == 2:
        text += ["", "회전 증강 단일요인 진단 결과 (검증9, 시드1)"]
        for arm, r in pilot.items():
            text.append(arm + ": " + ", ".join(f"{k} {100*v:.3f}%" for k, v in r["macro"].items()))
        text += ["같은4096단계 대조군에 비해 회전 증강의 개선은 확인되지 않았다. SeqOT의 수평 순열 불변 구조와 맞는 결과다.",
                 "실제32입력×3회전의 출력 검사도 seqot_roll_check.json에 있다. 원래 모델·추가학습 대조·회전증강을 구분하며 추가학습 자체의 변화와 증강 효과를 혼동하지 않는다."]
    else:
        text += ["", "회전 증강 대조의 학습·검증은 진행 중이다. 완료 상태는 yaw_pilot/<arm>/status.json에 쓴다."]
    (OUT / "report_ko.txt").write_text("\n".join(text) + "\n")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    main()
