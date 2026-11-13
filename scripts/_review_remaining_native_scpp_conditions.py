#!/usr/bin/env python3
"""SC++ KITTI00 진단을 공개 평가기·논문 조건과 메타데이터 수준에서 대조한다.

python3 scripts/_review_remaining_native_scpp_conditions.py
MATLAB 서술자를 다시 만들지 않는다. 기존 원장(질의, 승자, SCD 거리, 승자 거리, 최소 거리)과
KITTI00 자세만 읽는다. 논문 수치에 맞춰 조건을 고르지 않고, 공개 코드·논문 문장이 정한
조건만 적용하며 나머지는 정의별 개수로 공시한다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCPP = REPO / "results/pdv2_remaining_native_20261106/scpp"
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106/disco_protocol/scpp_conditions.json"
KITTI = Path("/mnt/d/NSD_datasets/kitti/dataset")
OFFICIAL = REPO / "_external/scancontext_tro"
THRESHOLDS = np.concatenate([np.linspace(0, 0.01, 50), np.linspace(0.01, 1, 50)])
PAPER_TABLE_IX = {"pc": {8: 0.84, 4: 0.88, 2: 0.91, 1: 0.94}, "cc": {8: 0.81, 4: 0.85, 2: 0.87, 1: 0.88}}
PAPER_TABLE_VIII_RETRIEVAL_KEY = {"pc": 0.84, "cc": 0.80}


def sha256(path: Path) -> str:
    """파일 SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_auc(distance: np.ndarray, winner_geo: np.ndarray, nearest_geo: np.ndarray, radius: float) -> dict:
    """fast_evaluator/main.m의 계수와 prcurve_drawer.m의 사다리꼴 AUC를 재현한다.

    거절(거리>임계값)이면 재방문은 FN, 아니면 TN. 수락이면 승자가 반경 안일 때 TP, 아니면 FP.
    첫 정밀도는 1로 둔다(prcurve_drawer.m의 Precisions(1)=1). 수락이 없는 다른 임계값의
    정밀도는 MATLAB에서 0/0=NaN이라 공식 합이 NaN이 되므로, 기존 보고의 관례(1로 둠)와
    NaN 구간을 건너뛴 값을 함께 기록한다.
    """
    revisit = nearest_geo < radius
    correct = winner_geo < radius
    precision, recall, counts = [], [], []
    for threshold in THRESHOLDS:
        accept = ~(distance > threshold)
        tp = int((accept & correct).sum())
        fp = int((accept & ~correct).sum())
        fn = int((~accept & revisit).sum())
        tn = int((~accept & ~revisit).sum())
        counts.append([tp, fp, fn, tn])
        precision.append(tp / (tp + fp) if tp + fp else np.nan)
        recall.append(tp / (tp + fn) if tp + fn else np.nan)
    precision[0] = 1.0
    p, r = np.array(precision), np.array(recall)
    empty = np.isnan(p)
    filled = np.where(empty, 1.0, p)
    segments = 0.5 * (p[:-1] + p[1:]) * (r[1:] - r[:-1])
    auc = float(np.sum(0.5 * (filled[:-1] + filled[1:]) * (r[1:] - r[:-1])))
    f1 = 2 * filled * r / (filled + r)
    return {"auc": auc, "auc_convention": "수락 없는 임계값 정밀도=1(기존 보고와 같음)",
            "auc_nan_segments_skipped": float(np.nansum(segments)),
            "official_literal_auc_is_nan": bool(np.isnan(segments).any()),
            "thresholds_without_acceptance": int(empty.sum()),
            "max_f1": float(np.nanmax(f1)), "queries": len(distance),
            "revisit_queries": int(revisit.sum()), "correct_top1_revisit": int((revisit & correct).sum()),
            "accepted_all_counts_tp_fp_fn_tn": counts[-1]}


def revisit_counts() -> dict:
    """KITTI00 자세만으로 재방문 정의별 개수를 센다(성능 계산 없음)."""
    poses = np.loadtxt(KITTI / "poses/00.txt")
    tr = [line for line in open(KITTI / "sequences/00/calib.txt") if line.startswith("Tr:")][0].split()[1:]
    tr = np.array(tr, float).reshape(3, 4)
    rot, trans = poses.reshape(-1, 3, 4)[:, :, :3], poses.reshape(-1, 3, 4)[:, :, 3]
    lidar = np.einsum("nij,j->ni", rot, tr[:, 3]) + trans
    n, idx = len(poses), np.arange(len(poses))
    rows = []
    for frame, xy in [("camera_xz(공식 SingleTestInfo·fast_evaluator)", poses[:, [3, 11]]),
                      ("lidar_xz(Tr 적용)", lidar[:, [0, 2]])]:
        distance = np.sqrt(((xy[:, None, :] - xy[None, :, :])**2).sum(-1))
        for radius in [4, 5, 6, 7, 8, 9, 10]:
            for gap in [49, 50, 100, 300]:
                positive = ((distance < radius) & (idx[None, :] <= idx[:, None] - gap)).any(1)
                rows.append({"pose": frame, "radius_m": radius, "candidate_gap_frames": gap,
                             "all_queries": int(positive.sum()),
                             "fast_evaluator_query_range": int(positive[49:n - 1].sum()) if gap == 49 else None})
        if frame.startswith("camera"):
            gaps = {}
            for gap in range(1, 1001):
                gaps[gap] = int(((distance < 8) & (idx[None, :] <= idx[:, None] - gap)).any(1).sum())
            gap_852 = [g for g, c in gaps.items() if c == 852]
            plateau = sorted(set(gaps[g] for g in range(100, 601)))
    hits = [row for row in rows if 852 in (row["all_queries"], row["fast_evaluator_query_range"])]
    return {"frames": n, "poses_sha256": sha256(KITTI / "poses/00.txt"),
            "fast_evaluator_00csv_equals_poses": bool(np.array_equal(
                np.loadtxt(OFFICIAL / "fast_evaluator/00.csv", delimiter=","), poses)),
            "definitions": rows, "definitions_giving_852": hits,
            "camera_8m_frame_gaps_giving_852": gap_852, "camera_8m_counts_for_gap_100_to_600": plateau}


def main() -> None:
    """원장 재계산과 조건 대조를 기록한다."""
    report = json.loads((SCPP / "report.json").read_text())
    variants = {}
    for name in ["pc", "cc"]:
        ledger = np.loadtxt(SCPP / f"{name}_ledger.csv", delimiter=",")
        query, distance, winner_geo, nearest_geo = ledger[:, 0], ledger[:, 2], ledger[:, 3], ledger[:, 4]
        assert np.array_equal(query, np.arange(49, 4541)) and np.all(ledger[:, 1] <= query - 49)
        stored = report["variants"][name]
        existing = official_auc(distance, winner_geo, nearest_geo, 8)
        # fast_evaluator/main.m:61 `for query_idx = 1:num_queries - 1`: 마지막 프레임은 질의가 아니다.
        public = slice(0, len(query) - 1)
        radii = {r: official_auc(distance[public], winner_geo[public], nearest_geo[public], r) for r in [8, 4, 2, 1]}
        variants[name] = {
            "ledger_sha256": sha256(SCPP / f"{name}_ledger.csv"),
            "existing_all_4492_queries_recomputed": existing,
            "existing_report_auc": stored["pr_auc"],
            "existing_report_auc_abs_difference": abs(existing["auc"] - stored["pr_auc"]),
            "public_fast_evaluator_query_range": {"queries": "1-based 50..4540, 후보 1..q-49",
                                                  "by_radius_m": {str(r): v for r, v in radii.items()}},
            "paper_table_ix_auc": {str(r): v for r, v in PAPER_TABLE_IX[name].items()},
            "ours_minus_paper_table_ix": {str(r): radii[r]["auc"] - PAPER_TABLE_IX[name][r] for r in [8, 4, 2, 1]},
            "paper_table_viii_retrieval_key_auc": PAPER_TABLE_VIII_RETRIEVAL_KEY[name]}
    last_query_positive = bool(np.loadtxt(SCPP / "pc_ledger.csv", delimiter=",")[-1, 4] < 8)
    result = {
        "scope": "기존 SC++ KITTI00 공식 MATLAB 원장의 조건 대조. 새 서술자 계산·새 학습·시험 기반 선택 없음",
        "variants": variants, "last_frame_query_positive_8m": last_query_positive,
        "revisit_definitions": revisit_counts(),
        "conditions": [
            {"item": "질의 범위", "public_code": "fast_evaluator/main.m:61 for query_idx = 1:num_queries-1, main.m:71 length<num_candidates(50)면 건너뜀",
             "existing_diagnostic": "질의 1-based 50..4541(4,492개); 마지막 프레임 포함",
             "action": "원장에서 마지막 질의만 빼 정확 대응(4,491개). 서술자·승자 변화 없음", "exactly_matchable": True},
            {"item": "후보·정답 시간 제외", "public_code": "main.m:75 exp(1:end-(num_candidates-1)) → 후보 j≤q-49, 정답도 같은 집합(main.m:78)",
             "existing_diagnostic": "gap=49, 후보·정답 모두 1..q-49", "action": "일치", "exactly_matchable": True},
            {"item": "후보 수 k", "paper": "§VI-D-1 k-d tree 단일 후보(k=1); 표VII 기준 K=1*",
             "public_code": "fast_evaluator는 50후보 뒤 sc_dist 최소", "existing_diagnostic": "k=1",
             "action": "논문 조건 유지", "exactly_matchable": True},
            {"item": "정렬 탐색", "paper": "§V-C N(n̂*)={n̂*}로 충분하다고 명시", "existing_diagnostic": "search_ratio=0",
             "action": "일치", "exactly_matchable": True},
            {"item": "1–1.5m 등간격 표본", "paper": "§VI-B 문장. 그러나 표III KITTI00 노드 4,541 = 원시 프레임 수; KITTI08 2,377/4,071, KAIST03 6.25km/4,224",
             "public_code": "tro2022/batch_processor/loadData_augIros18verForKITTI.m:11 AllFrames 경로, makeAugmentedExperienceAllFramesTreeMerge.m:97-104 1m 이동 조건 주석 처리",
             "existing_diagnostic": "KITTI00 전체 4,541프레임", "action": "KITTI00에는 등간격 표본을 적용하지 않는 것이 표III·공식 KITTI 경로와 맞다", "exactly_matchable": True},
            {"item": "재방문 852개", "paper": "표III KITTI00 852/4541, 비동일방향 22/852",
             "existing_diagnostic": "8m·후보 j≤q-49에서 889(공개 질의 범위 888)",
             "action": "8m 카메라 자세에서 원칙적인 프레임 제외로는 852가 나오지 않는다(100~600프레임 864 평탄, 852는 834프레임만). 맞추기 위한 조건 선택은 하지 않고 미해결로 둔다",
             "exactly_matchable": False},
            {"item": "PR 산식", "public_code": "main.m:100-125 계수, prcurve_drawer.m:85-91 Precisions(1)=1 사다리꼴",
             "existing_diagnostic": "같은 임계값 100개·계수 규칙. 수락 없는 임계값 정밀도를 1로 둠",
             "action": "원장에서 독립 재계산해 기존 AUC와 대조. 최소 SCD 거리(PC 0.0131, CC 0.0208)가 0.01보다 커 첫 51개 임계값은 수락이 없고, 공식 drawer를 문자 그대로 쓰면 0/0으로 AUC가 NaN이다. 두 관례(정밀도=1, NaN 구간 생략)를 함께 공시하며 논문값에 가까운 관례를 고르지 않는다",
             "exactly_matchable": False}],
        "rerun_needed": False,
        "rerun_reason": "질의 범위 차이는 원장에서 정확히 재현되고, 다른 확인 가능한 조건은 이미 일치한다. 852 정의는 공개 자료로 확정할 수 없어 MATLAB 재실행으로 해결되지 않는다.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: {"existing": v["existing_all_4492_queries_recomputed"]["auc"],
                          "public_range": {r: x["auc"] for r, x in v["public_fast_evaluator_query_range"]["by_radius_m"].items()},
                          "diff_existing": v["existing_report_auc_abs_difference"]} for k, v in variants.items()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
