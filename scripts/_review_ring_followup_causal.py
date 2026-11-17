"""RING/RING++ 원 논문 NCLT 수치 미재현의 원인 판정 근거를 기존 완료 결과에서 모은다(새 실행 없음).

호스트에서 실행한다.
    python3 scripts/_review_ring_followup_causal.py

입력: 논문 원문 추출본, 공식 코드, 기존 완료 결과(diagnosis, score_c1, official_eval, paper_consistency).
출력: results/pdv2_reproduction_orchestration_20261106/ring/causal_audit.json
논문 인용 줄은 실제 줄 내용에 지정 문자열이 있는지 검사하고, 지표는 원장 수치에서 다시 계산한다.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/pdv2_reproduction_orchestration_20261106/ring"
PAPER = REPO / "results/pdv2_review_native_20261105/ringpp_paper.txt"
DIAG = REPO / "results/pdv2_remaining_native_20261106/ring/diagnosis.json"
C1 = OUT / "score_c1/summary.json"
OFFICIAL = OUT / "official_eval/summary.json"
CONSISTENCY = OUT / "paper_consistency.json"
SCOPE = REPO / "results/pdv2_published_reproduction_20261106/ring_published_scope.json"
NCLT_LOCAL = Path("/mnt/d/NSD_datasets/nclt")
DISCO = REPO / "results/pdv2_reproduction_orchestration_20261106/disco_protocol/report.json"
# 2026-11-06 서버 호스트 목록 조회 결과(읽기만). 이 스크립트는 서버에 접속하지 않는다.
DISCO_0820_SERVER = {
    "path": "/home/qkrrlans00/Neural-Spectral-Codec/results/pdv2_remaining_native_20261106/disco/native_data/2012-08-20",
    "listed": ["complete.json", "groundtruth.csv", "gzip.index", "members.json", "selection.json",
               "selection_no_upper_bound.json", "velodyne_sync/ (파일 1,903개)"],
    "size": "878M",
}

# (파일, 줄, 줄 안에 있어야 할 문자열, 뜻)
LINE_EVIDENCE = [
    (PAPER, 763, "2012-03-17", "그림10: 03-17 질의→02-04 지도(지도 간격 10·20·50m)"),
    (PAPER, 775, "NCLT, MulRan and Oxford", "그림11 캡션: 20m 지도 간격, 날짜 쌍 없음"),
    (PAPER, 778, "2012-03-17", "그림11 문단 시작: 03-17→02-04 외에"),
    (PAPER, 780, "2012-08-20", "그림11: NCLT 08-20→02-04 쌍"),
    (PAPER, 781, "Sejong02", "그림11: MulRan Sejong02→Sejong01"),
    (PAPER, 782, "2019-01-15-13-06-37", "그림11: Oxford 01-15→01-11"),
    (PAPER, 794, "Tab. II", "그림11~13 정성 비교 다음에 표II 언급(날짜 쌍 미기재)"),
    (PAPER, 830, "query data is 5m", "표III: 질의 5m"),
    (PAPER, 831, "revisited distance threshold", "표III: 지도 20m·재방문 10m"),
    (PAPER, 837, "2012-03-17", "표III: NCLT 03-17 질의"),
    (PAPER, 847, "TABLE II", "표II 캡션: Multi-session Scenarios, 날짜 쌍 없음"),
    (PAPER, 850, "0.7321", "표II NCLT RING++ 0.7321/0.7890/0.8374"),
    (PAPER, 853, "0.7098", "표II NCLT RING 0.7098/0.7658/0.8246 (이 표에만 있음)"),
    (PAPER, 862, "0.7451", "표II MulRan RING++ 0.7451"),
    (PAPER, 871, "0.8274", "표II Oxford RING++ 0.8274"),
    (PAPER, 887, "0.6941", "표III MulRan* RING++ 0.6941 (EgoNN 시험셋, 표II와 다름)"),
    (PAPER, 892, "0.7321", "표III NCLT RING++ 0.7321/0.7890/0.8374 (표II와 같음)"),
    (PAPER, 897, "0.8274", "표III Oxford RING++ 0.8274 (표II와 같음)"),
    (PAPER, 908, "EgoNN [40]", "표III 각주: MulRan은 EgoNN 시험셋"),
    (PAPER, 963, "73.21", "표VI 120×120 R@1 73.21%"),
    (PAPER, 1019, "2012-03-17", "표VI: 03-17→02-04 쌍"),
    (PAPER, 573, "Recall@1 =", "식30 R@1 = TP_top1/(TP_top1+FN_top1)"),
    (REPO / "_external/RING/evaluation/generate_evaluation_sets.py", 157, "2012-02-04", "공개 생성기 기본 지도"),
    (REPO / "_external/RING/evaluation/generate_evaluation_sets.py", 158, "2012-03-17", "공개 생성기 기본 질의"),
    (REPO / "_external/RING/evaluation/generate_evaluation_sets.py", 161, "20.0", "질의 필터 20m"),
    (REPO / "_external/RING/evaluation/evaluate.py", 239, "/num_queries", "공개 R@k 분모 = 전체 질의"),
    (REPO / "_external/RING/evaluation/evaluate.py", 223, "np.linspace(0.55, 1, 10)", "PR 임계값 격자 끝 1.0"),
    (REPO / "_external/RING/evaluation/plot_PR_curve.py", 32, "min(dist) < thre", "임계값 아래면 수락"),
    (REPO / "_external/RING/evaluation/plot_PR_curve.py", 52, "num_hits + num_misses", "recall 분모에서 오수락 제외"),
    (REPO / "_external/RING/evaluation/plot_PR_curve.py", 63, "DeepRING", "공개 PR 그림 범례가 DeepRING"),
    (REPO / "_external/RING/utils/generate_bev_pointfeat_cython/src/kernel.cu", 156, "max_h[", "셀별 max 비원자 비교"),
    (REPO / "_external/RING/utils/generate_bev_pointfeat_cython/src/kernel.cu", 159, "max_h[", "셀별 max 비원자 쓰기"),
    (REPO / "_external/RING/datasets/NCLTDataset.py", 199, "1000000000", "시간 허용오차 1e9 단위 오류"),
]


def sha256(path: Path) -> str:
    """파일 sha256."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_lines() -> list:
    """인용 줄 내용을 검사한다."""
    rows = []
    for path, line, needle, meaning in LINE_EVIDENCE:
        text = path.read_text(errors="replace").split("\n")[line - 1]
        rows.append({"path": str(path.relative_to(REPO)), "line": line, "needle": needle,
                     "found": needle in text, "meaning": meaning})
    return rows


def main() -> None:
    """근거를 모아 causal_audit.json을 쓴다."""
    diag = {(r["region"], r["variant"]): r for r in json.loads(DIAG.read_text())["rows"]}
    c1 = json.loads(C1.read_text())["variants"]
    official = {r["condition"]: r for r in json.loads(OFFICIAL.read_text())["rows"]}
    consistency = json.loads(CONSISTENCY.read_text())
    lines = check_lines()
    assert all(r["found"] for r in lines), [r for r in lines if not r["found"]]

    paper = {"ring": 70.98, "ringpp": 73.21}
    ring = diag[("all", "ring")]
    rel = diag[("all", "ringpp_released")]
    fix = diag[("all", "ringpp_exactmax")]
    # 비원자 max의 다른 실행(공식 evaluate() 그대로)
    rel_rerun = official["all_feat_official"]["official_R@1_10m"] * 100

    def f1(cond: str) -> float:
        return c1[cond]["official_abs_sum"]["10.0"]["official_grid_max_F1"]

    gap_all_released = paper["ringpp"] - rel["R1_all_percent"]
    maxfix_gain = fix["R1_all_percent"] - rel["R1_all_percent"]
    decomposition = {
        "assumption": "논문 73.21%가 NCLT 02-04 지도 20m·03-17 질의 5m·10m 판정, 전체 1,054질의 공개 생성기 집합과 같은 과제라는 가정. 날짜·분모는 표 대응 항목 참고",
        "ringpp_released_R1_all": rel["R1_all_percent"],
        "ringpp_released_rerun_R1_all": rel_rerun,
        "ringpp_exactmax_R1_all": fix["R1_all_percent"],
        "ringpp_exactmax_R1_positive": fix["R1_positive_only_percent"],
        "paper_ringpp": paper["ringpp"],
        "gap_released_all_pp": gap_all_released,
        "maxfix_gain_pp": maxfix_gain,
        "maxfix_gain_range_over_two_released_runs_pp": [fix["R1_all_percent"] - max(rel["R1_all_percent"], rel_rerun),
                                                         fix["R1_all_percent"] - min(rel["R1_all_percent"], rel_rerun)],
        "maxfix_share_of_released_gap": maxfix_gain / gap_all_released,
        "remaining_after_maxfix_all_pp": paper["ringpp"] - fix["R1_all_percent"],
        "remaining_after_maxfix_positive_pp": paper["ringpp"] - fix["R1_positive_only_percent"],
        "ring_unaffected_by_max": {"ring_R1_all": ring["R1_all_percent"],
                                   "note": "RING은 점유 BEV라 max 커널을 쓰지 않는다. released·exactmax RING R@1이 같다(ring_published_scope.json)"},
        "ring_remaining_all_pp": paper["ring"] - ring["R1_all_percent"],
        "ring_remaining_positive_pp": paper["ring"] - ring["R1_positive_only_percent"],
        "remaining_basis": "RING++ 잔차는 03-17→02-04가 명시된 표III·VI 73.21%와의 비교다. RING 잔차는 날짜 쌍이 적히지 않은 "
                           "표II 70.98%와 03-17 결과의 차이일 뿐이며, 표II NCLT 행이 08-20 쌍일 가능성을 배제하지 못해 "
                           "같은 조건 잔차로 확정하지 않는다(table_mapping 참고)",
        "ringpp_minus_ring_paper_pp": paper["ringpp"] - paper["ring"],
        "ringpp_minus_ring_ours_exactmax_all_pp": fix["R1_all_percent"] - ring["R1_all_percent"],
        "official_grid_max_F1_10m": {"all_ring": f1("all_ring"), "all_ringpp_exactmax": f1("all_ringpp_exactmax"),
                                     "all_ringpp_released": f1("all_ringpp_released"),
                                     "paper_ring": 0.7658, "paper_ringpp": 0.7890},
    }
    # 공개 격자 끝(전부 수락)에서 F1=2P/(1+P). 논문 F1이 같은 격자·같은 질의 집합의 최대 F1이라면(A2·A3)
    # 논문 질의 집합의 전체분모 1위 정답률 P는 F1/(2−F1) 이하다.
    nclt = consistency["table_II"]["NCLT"]
    decomposition["f1_view"] = {
        "assumption": "A2·A3(논문 F1 = 공개 격자 최대 F1, R@1과 같은 질의 집합)일 때만 성립. 미확정",
        "ring": {"paper_F1_implied_max_P_percent": 100 * nclt["RING"]["max_all_denominator_R@1_given_F1"],
                 "ours_P_percent": ring["R1_all_percent"],
                 "ours_F1_minus_paper_pp": 100 * (f1("all_ring") - 0.7658)},
        "ringpp_exactmax": {"paper_F1_implied_max_P_percent": 100 * nclt["RING++"]["max_all_denominator_R@1_given_F1"],
                            "ours_P_percent": fix["R1_all_percent"],
                            "ours_F1_minus_paper_pp": 100 * (f1("all_ringpp_exactmax") - 0.7890)},
        "accept_all_F1_ours": {"ring": 2 * ring["R1_all_percent"] / 100 / (1 + ring["R1_all_percent"] / 100),
                               "ringpp_exactmax": 2 * fix["R1_all_percent"] / 100 / (1 + fix["R1_all_percent"] / 100)},
        "reading": "우리 최대 F1은 전부 수락 점의 2P/(1+P)와 0.002 안이라 F1은 사실상 1위 정답률 P의 함수다. "
                   "이 가정에서 논문 RING F1은 논문 질의 집합의 P≤62.05%를 뜻해 우리 62.52%보다 높지 않고, "
                   "RING++은 P≤65.16%로 우리 62.24%와 최대 2.92%p 차이다. 즉 R@1 격차(8.46·10.97%p)는 "
                   "같은 표의 F1과 맞지 않으며 R@1 정의·분모·질의 집합 차이 가설(H2)을 지지하는 정황이다.",
    }

    local_sessions = sorted(p.name for p in NCLT_LOCAL.iterdir() if p.name.startswith("201")) \
        if NCLT_LOCAL.exists() else []
    disco_ver = json.loads(DISCO.read_text())["verification"]
    disco_0820 = {"sel": disco_ver["selection"]["2012-08-20"], "inp": disco_ver["inputs"]["dates"]["2012-08-20"]}
    assert disco_0820["inp"]["raw_files_rehashed"] == disco_0820["sel"]["no_upper_bound"]["stored"] == 1903
    table_mapping = {
        "question": "RING++ 73.21%는 세 표에서 같지만 RING 70.98%의 비교 날짜 쌍이 확정되는가",
        "finding": "확정되지 않는다. RING 70.98%는 표II에만 있고 표II는 날짜 쌍을 적지 않는다. 03-17→02-04는 표II NCLT의 모든 행이 같은 쌍에서 나왔다는 가정 아래 가장 일관된 해석일 뿐이다.",
        "for_03_17": [
            "표II NCLT RING++ 세 값(0.7321/0.7890/0.8374)이 03-17→02-04로 명시된 표III NCLT와 세 자리 모두 같다(ringpp_paper.txt:850, :892, :837)",
            "같은 73.21%가 03-17→02-04로 명시된 표VI 120×120에도 있다(:963, :1019)",
            "공개 생성기 기본값이 02-04 지도·03-17 질의다(generate_evaluation_sets.py:157-158)",
            "표III에서 프로토콜이 다른 MulRan*은 값이 다르고(0.6941 vs 0.7451) 같은 쌍인 Oxford는 같다(0.8274) → 저자는 같은 실행 값을 표 사이에 재사용한 정황",
        ],
        "for_08_20_or_ambiguity": [
            "표II는 그림11 문단(NCLT 08-20→02-04, MulRan Sejong02→01, Oxford 01-15→01-11) 바로 뒤에서 '정성 비교에 더해 정량 결과'로 소개된다(:778-796)",
            "표II의 MulRan·Oxford는 그림11의 쌍과 맞는 해석이 자연스럽다(표III Oxford 쌍 = 그림11 Oxford 쌍, :782·:838-840)",
            "표II 캡션·각주에 날짜 쌍이 없다(:847)",
        ],
        "conclusion": "RING++ 73.21% = 03-17→02-04 20m/5m/10m은 표III·VI 본문 근거가 있다(단, 질의 목록·분모는 미기재). RING 70.98%가 같은 쌍이라는 것은 '표II 한 블록이 한 쌍에서 나왔다'는 가정에 기댄 추론이다. 그림11의 08-20 쌍이 표II NCLT 행의 출처일 가능성을 본문만으로 배제할 수 없다.",
        "correction_to_prior_reports": [
            "작업 지시·ring_published_scope.json·diagnosis.json·report_ko.txt는 RING 70.98%를 03-17→02-04 기준값(reference_R@1)으로 놓고 격차를 계산했다. RING 쪽 기준 날짜는 '확정'이 아니라 '가정'으로 정정한다. RING++ 쪽은 표III·VI 본문 근거가 있어 유지한다.",
            "worker_status의 '표VI 120x120 R@1 73.21%=표II RING++, 03-17->02-04 쌍 명시'는 RING++에 한해 맞다. 표II 전체의 쌍이 명시됐다는 뜻으로 읽으면 안 된다.",
            "RING 70.98% 대비 격차 8.46%p(전체)·4.55%p(양성), C1·분모 반영 뒤 3.44%p(양성)와 'RING++>RING 우위 미재현'은 같은 쌍 가정 아래의 값이며 같은 조건 잔차로 확정하지 않는다.",
            "이전 재검토의 '08-20 자료 미보유·다운로드 승인 필요'는 틀렸다. 로컬 D:에만 없고, 서버에는 DiSCO 담당이 확보·검산한 08-20 선택 원시 스캔이 있다(nclt_2012_08_20_server). 서버 실험은 이미 사용자 승인 범위다.",
        ],
        "nclt_2012_08_20_available_local": "2012-08-20" in local_sessions,
        "nclt_local_sessions": local_sessions,
        "nclt_2012_08_20_server": {
            "listing": DISCO_0820_SERVER,
            "disco_report": str(DISCO.relative_to(REPO)),
            "disco_report_sha256": sha256(DISCO),
            "session_synced_scans": disco_0820["sel"]["all_scans"],
            "disco_selection_no_upper_bound": disco_0820["sel"]["no_upper_bound"]["stored"],
            "disco_selection_released": disco_0820["sel"]["released"]["stored"],
            "raw_files_rehashed": disco_0820["inp"]["raw_files_rehashed"],
            "raw_sha256_mismatches": disco_0820["inp"]["raw_sha256_mismatches"],
            "disco_selection_rule": "DiSCO 공개 생성기 규칙: 직전 선택 프레임에서 3m 이상(released는 3.5m 미만 상한 추가) "
                                    "(scripts/_review_remaining_native_disco_data.py:57)",
        },
        "nclt_2012_08_20_note": "세션 자료가 전혀 없는 상태가 아니다. 서버에 DiSCO 담당이 확보한 08-20 선택 원시 스캔 1,903장(세션 동기 스캔 "
                                "25,107장 중, 재해시 불일치 0)과 groundtruth.csv가 있다. 다만 이 목록은 DiSCO의 3m 간격 규칙으로 고른 것이라 "
                                "RING 공개 생성기의 5m 질의(20m 필터)에 필요한 원시 스캔이 모두 들어 있는지는 확인하지 않았다. "
                                "로컬 D:에는 08-20이 없다. 이번 정정에서는 다운로드·평가를 실행하지 않았다.",
    }

    confirmed = [
        {"id": "K1", "what": "공개 RING++ CUDA 셀별 max 비원자 갱신(경쟁 상태)",
         "evidence": ["_external/RING/utils/generate_bev_pointfeat_cython/src/kernel.cu:153-160",
                      "results/pdv2_review_native_20261105/kernel_audit.json",
                      "results/pdv2_review_native_20261105/maxfix/repeatability.json"],
         "effect": f"RING++ 전체분모 R@1 {rel['R1_all_percent']:.3f}%(다른 실행 {rel_rerun:.3f}%) → 결정적 max {fix['R1_all_percent']:.3f}%, +{maxfix_gain:.2f}%p. 비원자 기준 논문과 차이 {gap_all_released:.2f}%p 중 약 {100 * maxfix_gain / gap_all_released:.0f}%. RING에는 영향 없음",
         "status": "확정(공개 코드 결함, 재현 격차의 일부만 설명)"},
        {"id": "K2", "what": "R@1 분모: 공개 evaluate()는 전체 질의, 논문 식30은 양성(TP+FN) 분모로 읽힘",
         "evidence": ["_external/RING/evaluation/evaluate.py:239", "results/pdv2_review_native_20261105/ringpp_paper.txt:568-581",
                      "results/pdv2_remaining_native_20261106/ring/diagnosis.json"],
         "effect": f"정답 없는 62/1,054질의 제외 시 RING {ring['R1_all_percent']:.2f}→{ring['R1_positive_only_percent']:.2f}%, 결정적 max RING++ {fix['R1_all_percent']:.2f}→{fix['R1_positive_only_percent']:.2f}%",
         "status": "확정(정의 차이 존재). 논문이 어느 분모를 썼는지는 미확정"},
        {"id": "K3", "what": "점수 합산: 공식 circorr2 Σω|corr| vs 논문 식12 부호 있는 합(사전 고정 C1)",
         "evidence": ["results/pdv2_reproduction_orchestration_20261106/ring/score_c1/{protocol,summary,verification}.json"],
         "effect": "RING +1.04%p, 결정적 max RING++ +0.57%p(전체분모). 1위 변경 30·46질의",
         "status": "확정(작은 기여). 논문 실험이 어느 합을 썼는지는 미확정"},
    ]
    excluded = [
        {"id": "X1", "what": "공개 생성기 표본 추출·자세 대응 오류", "evidence": "sampling/report.json: 독립 재구성과 지도·질의 4목록 비트 일치"},
        {"id": "X2", "what": "타임스탬프 1e9 허용오차 단위 오류", "evidence": "NCLTDataset.py:199; diagnosis.json pose_checks: 선택 1,413스캔 dt≤0.009184s, 위치차 <3.3e-5m"},
        {"id": "X3", "what": "우리 어댑터의 검색·점수 결함", "evidence": "official_eval/: 공식 evaluate() 그대로(ICP·그림만 대체) 결정적 4조건 pair_dists 비트 일치"},
        {"id": "X4", "what": "torch_radon·voxelfeat·TING 서버 빌드 이상", "evidence": "kernels/{report,radon_replica}.json: 독립 복제 상대L2≤7.9e-5, 점특징 상대오차≤1.4e-6"},
        {"id": "X5", "what": "센서 좌표 부호(z) 오류", "evidence": "sensor_frame.json: HDL-32E 거꾸로 장착, z>1m 대역은 지면 제거로 동작"},
        {"id": "X6", "what": "BEV 해상도·이동 추정 경로가 논문 120×120과 다름", "evidence": "official_eval: 결정적 max TE 0.556/0.771/1.299m vs 논문 표VI 0.56/0.77/1.31m (질의 집합이 달라 정황 수준)"},
        {"id": "X7", "what": "정답 반경(10m) 오해", "evidence": "ringpp_paper.txt:830-831 표III 10m 명시, 기존 반경 대조(common_radius_factor_report.json)"},
    ]
    hypotheses = [
        {"id": "H1", "priority": 1, "what": "표II NCLT 행(특히 RING 70.98%)의 날짜 쌍이 03-17이 아닐 가능성(그림11의 08-20)",
         "why": "표 대응 항목 참고. RING 70.98%는 날짜 쌍이 명시된 표에 없다",
         "next_check": "서버의 08-20 groundtruth.csv·스캔 목록에 공개 생성기 기본 규칙(02-04 지도 20m, 08-20 질의 5m, 20m 필터)을 적용해 "
                       "필요한 질의 스캔이 DiSCO 확보분 1,903장에 모두 있는지 먼저 확인한다. 빠진 스캔만 추가 확보한 뒤 RING·결정적 max RING++를 "
                       "10m 판정으로 한 번 평가해 03-17 결과와 나란히 보고한다. 결과에 따라 쌍을 고르지 않는다. 서버 실험은 사용자 승인 범위이며 "
                       "이번 정정에서는 실행하지 않았다"},
        {"id": "H2", "priority": 2, "what": "논문 질의 집합·분모가 공개 생성기와 다름(영역·필터·목록)",
         "why": "같은 질의 집합·공개 격자 최대 F1·전체 분모 세 가정 아래 표II NCLT RING 계열 4행이 R@1≤F1/(2−F1)을 어김. 양성 분모로 읽으면 양성 비율≤0.869가 필요하나 공개 집합은 0.931~0.941",
         "next_check": "저자 평가 pickle(질의·지도 목록) 또는 질의 수 확인. 표본 탐색으로 맞추지 않는다"},
        {"id": "H3", "priority": 3, "what": "논문 F1이 공개 compute_PR_pairs의 격자 최대 F1이 아님(특정 임계값 F1, 다른 격자, 다른 실행)",
         "why": "논문은 'F1 score'의 산출 방식(최대 여부·격자)을 적지 않는다. 공개 evaluate()는 F1 배열만 저장하고 표 값 산출 코드는 없다",
         "next_check": "저자 확인 외에는 판정 불가. H2와 함께 일관성 검사 가정으로만 사용"},
        {"id": "H4", "priority": 4, "what": "논문 실험 코드와 공개 코드(2023-09-28 단일 커밋, PR 그림 범례 'DeepRING')의 차이",
         "why": "공개 저장소 이력이 논문 이후 단일 커밋이라 실험 당시 코드를 확인할 수 없다. 공개 실행 일치는 공개본 기준의 일치일 뿐이다",
         "next_check": "저자 실험 당시 코드·설정 확보 시 대조"},
        {"id": "H5", "priority": 5, "what": "RING++>RING 우위(논문 +2.23%p)가 재현되지 않는 이유",
         "why": f"결정적 max에서도 RING++−RING = {fix['R1_all_percent'] - ring['R1_all_percent']:+.2f}%p. 특징 6채널 정의·정렬은 공식식과 일치",
         "next_check": "H1(같은 날짜 쌍인지) 해소 후 다시 판단. 다른 쌍이면 비교 자체가 성립하지 않는다"},
    ]
    audit = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "기존 완료 결과 재사용 + 논문·공식 코드 줄 확인. 새 GPU·대형 실험·다운로드 없음. 주표·main 코드 불변",
        "script": "scripts/_review_ring_followup_causal.py",
        "table_mapping": table_mapping,
        "max_bug_decomposition": decomposition,
        "confirmed_causes": confirmed,
        "excluded_causes": excluded,
        "open_hypotheses": hypotheses,
        "paper_consistency_assumptions": {
            "statement": "아래 세 가정이 모두 맞으면 R@1 ≤ F1/(2−F1)이어야 하는데 표II NCLT RING 계열 4행이 어긴다. 따라서 세 가정 중 적어도 하나가 논문 수치에는 성립하지 않는다. 표 자체의 오류라고 단정하지 않는다",
            "assumptions": ["A1 R@1 분모 = 전체 질의(공개 evaluate.py:239)",
                            "A2 F1 = 공개 compute_PR_pairs 격자(끝 1.0에서 전부 수락)의 최대값(evaluate.py:221-224, plot_PR_curve.py:29)",
                            "A3 R@1과 F1이 같은 질의 집합·같은 실행에서 계산됨"],
            "nclt_ring_rows_violating": consistency.get("nclt_ring_rows_violations_all_denominator"),
            "positive_fraction_upper_bound_ring_rows": consistency.get("nclt_positive_fraction_upper_bound_ring_rows_only"),
        },
        "line_evidence": lines,
        "held_process_note": "보류 프로세스 컨테이너 PID 340070·1083011은 서버 호스트 PID 1976866·1015953과 같은 프로세스다"
                             "(/proc/<pid>/status NSpid 1976866→340070, 1015953→1083011, 둘 다 T). PID 차이는 호스트·컨테이너 "
                             "PID namespace 차이이며 장애가 아니다. 건드리지 않았다",
        "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                          [PAPER, DIAG, C1, OFFICIAL, CONSISTENCY, SCOPE, DISCO, Path(__file__)]},
    }
    (OUT / "causal_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"lines_ok": all(r["found"] for r in lines), **{k: decomposition[k] for k in (
        "maxfix_gain_pp", "maxfix_share_of_released_gap", "remaining_after_maxfix_all_pp",
        "remaining_after_maxfix_positive_pp", "ring_remaining_all_pp", "ring_remaining_positive_pp",
        "ringpp_minus_ring_ours_exactmax_all_pp", "official_grid_max_F1_10m")}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
