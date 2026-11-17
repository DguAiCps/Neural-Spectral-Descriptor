#!/usr/bin/env python3
"""공식 FreSCo 시험 46 평가의 사전 고정 규약 protocol.json (로컬 호스트, 2026-11-07).

  python3 scripts/fresco_official/protocol.py
시험 원장을 만들기 전에 한 번 쓴다. 이미 있으면 바꾸지 않고 해시만 다시 확인한다.
설정 값은 모두 공식 코드 기본값이며, 바뀐 것은 adaptations에만 있다. 시험 결과로 고르지 않는다.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CITY, FRESCO_COMMIT, PREP_COMMIT, REPO, RESULTS, SERVER, TEST, VAL,  # noqa
                    sha256, write_json)

OUT = RESULTS / "protocol.json"
NAS = "/mnt/aicps-nas/qkrrlans00/nsd_fresco_official_20261107"
CODE = ["common.py", "codec.py", "stream.py", "evaluate.py", "matlab_gen.py", "official_check.py",
        "verify_prep.py", "aggregate.py", "server_master.py", "supervisor.py", "protocol.py",
        "cpp/fresco_prep_driver.cpp", "cpp/build.sh", "cpp/shim/vtk_iostream_compat.h",
        "matlab_gui_shim/waitbar.m", "matlab_gui_shim/visualizeMatchPairs.m", "inventory.py"]


def remote_json(path: str) -> dict:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", SERVER, f"cat {path}"],
                       capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout)


def main():
    if OUT.exists():
        p = json.loads(OUT.read_text())
        print("protocol.json exists; frozen at", p["frozen_utc"])
        return
    build = remote_json(f"{NAS}/build/build.json")
    mgen = remote_json(f"{NAS}/build/matlab/matlab_gen.json")
    write_json(RESULTS / "build.json", build)
    write_json(RESULTS / "matlab_gen.json", mgen)
    code = {c: sha256(REPO / "scripts/fresco_official" / c) for c in CODE
            if (REPO / "scripts/fresco_official" / c).exists()}
    proto = {
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_before_test_ledgers": True,
        "scope": "공식 FreSCo(MATLAB+C++ 전처리)를 현재 원고 표2 조건(분할 v2 시험46, 공통 195,726질의)에 "
                 "학습 없이 1회 결정적 실행. 원 논문 조건 재현이 아니다.",
        "official": {
            "fresco": {"repo": "https://github.com/soytony/FreSCo", "commit": FRESCO_COMMIT,
                       "license": "CC BY-NC-SA 4.0", "paper": "https://arxiv.org/html/2206.12628v2",
                       "matlab_sha256": mgen["official_sha256"]},
            "preprocessing": {"repo": "https://github.com/soytony/Point-Cloud-Preprocessing-Tools",
                              "commit": PREP_COMMIT, "fmt_submodule": build["fmt_commit"],
                              "sources_sha256": build["official_sources_sha256"]},
            "fidelity_audit": "results/pdv2_fresco_official_20261107/official_fidelity/audit.json "
                              "(verdict reference_ready)"},
        "split_and_protocol": {
            "test": [f"{s}/{q}" for s, q in TEST], "n_test": len(TEST),
            "validation_used_for_equivalence_only": ["kitti/09"],
            "keyframes": "data/preprocessed_v2_odom/cache_*_test_*.npz scan_ids (KISS-ICP 키프레임, 다른 "
                         "비교 방법과 같은 목록)",
            "labels": "poses_gt는 top1_geo_dist·min_past_gt_dist 라벨에만 쓴다. 입력·검색에 GT 없음",
            "ledger_queries": "t >= 30 (0부터), 후보 j <= t-30",
            "common_queries": "30 <= q < N-15, 합 195,726, 양성 61,682, 양성 없는 kitti/07 원장 포함",
            "metrics": "scripts/_pr_eval_lib.py classify_queries(양성 <= 5 m, 음성 > 25 m)·pr_sweep: AP, "
                       "max_F1, R@100P, R@99P, R@95P; 재방문 R@1 = 양성 질의 중 반환 승자 <= 5 m",
            "aggregation": "양성 45시퀀스 -> 도시 6묶음 평균 -> 매크로(scripts/_pdv2_pilot.py CITY), "
                           "학습 없는 1회 결정적 실행",
            "cities": {k: v for k, v in CITY.items()}},
        "native_settings_unchanged": {
            "bev": "공식 C++ 지면 제거(markGroundPoints) + 단층 BEV 224x224, 1 m, MAX_RANGE 112, "
                   "height=int((z+2)*4) clamp, 칸 max (BatchMultiBevGen.cpp:331-373)",
            "sensor_params": "HDL_64E (64,2083,50), OS1_64 (64,1024,31), HDL_32E (32,1056,20) (Utility.cpp)",
            "descriptor": "imresize 101 bilinear, fft2/fftshift |.|, crop 31:71, Gaussian sigma 1 7x7 "
                          "conv replicate, ImToPolar(0,1,20,120), log; key 40D = [row mean; row std]/mean",
            "retrieval": "createns kdtree + knnsearch K=20 (공식 후보 예산, NSD 후보 수와 무관)",
            "matching": "computeFrescoDist 0:59 순환 L1(log-log), 승자 = L1 최소(엄격 <)",
            "acceptance": "공식 L1 <= t1 AND cos_row <= t2 (t1/t2 = 1e5)"},
        "adaptations": {
            "A1_protocol": "키프레임·정답·과거 후보를 공통 규약으로: 공통 KISS 키프레임(공식 2 m GT 선택 대신), "
                           "5 m 판정(공식 10 m), j <= q-30 (공식 LOOP_FRAME_INTERVAL 100 -> 30). 영향 중립을 주장하지 않는다",
            "A2_score": "원장 주 점수 score_fresco = -max(L1, 1e5*cos_row): 공식 2-threshold 수락 경로의 "
                        "스칼라 동치(C3). score_negl1 = -L1은 사전 등록 보조 진단",
            "A3_io": "공식 체인의 pcd·CSV 디스크 왕복과 다층 BEV·non_ground pcd를 생략하고 메모리 점군과 PNG "
                     "재읽기 uint8 BEV를 쓴다(C5 비트 동일). 원시 스캔 1개 작업 파일은 /dev/shm(RAM)에 덮어쓴다",
            "A4_knn": "createns+knnsearch를 공식 그대로 질의마다 실행(변경 없음)",
            "G1_kitti_ground": "(a) 공식 체인 그대로: KITTI 선택기의 intensity=-1 때문에 KITTI·KITTI-360 실제 점이 "
                               "지면으로 표시되지 않는 공식 동작을 유지. 수정판(b)은 실행하지 않음",
            "G2_helipr_vlp16": "공식 VLP-16 선택기·센서 값 없음. N_SCAN 16, Horizon 1800, GROUND_UPPER_SCAN 7"
                               "(LeGO-LOAM VLP-16 groundScanInd; 공식 HDL_32E 20·HDL_64E 50이 LeGO-LOAM 값과 같음), "
                               "row = 15 - ring(공식 top-down), col·label·intensity는 MulRan 선택기 규칙, "
                               "파일 크기만큼 모든 점을 읽음(MulRan eof 여분 점 특성은 재현하지 않음). audit G2가 권고한 Oxford 앙각 "
                               "규칙을 VLP-16(+15~-15도, 2도)에 옮기면 공칭 앙각에서 같은 row(= 15 - ring)를 주며, 원시 ring "
                               "필드는 반올림 오배정이 없는 실제 빔 번호라 이를 쓴다. Horizon·GROUND_UPPER_SCAN 값은 공식 근거가 없다",
            "G3_nclt_hdl32": "NCLT 선택기 없음. 공식 Oxford HDL-32E 선택기(앙각->row, 1056열)를 무수정 호출하고, "
                             "입력은 원시 z-up 센서 좌표(공유 로더의 z 반전을 되돌림; 로더 좌표 앙각 0.5~99.5% = "
                             "-4.5~33.3도로 상하 반전 확인)를 Oxford 파일 배치(-x, y, -z)로 씀",
            "G4_mcd_order": "MCD 추출 bin은 행우선 64x1024라 점 순서를 열우선으로 전치해 공식 MulRan 선택기를 "
                            "무수정 호출(k%64 = 빔, 빔0 맨 위)",
            "intensity": "KITTI는 공식이 -1로 덮어써 무관, MulRan·MCD·NCLT·HeLiPR는 ==-1 검사에만 쓰여 0으로 "
                         "대체(원시에 -1 없음 확인, C5 비트 동일)",
            "matlab_env": "MATLAB TMPDIR은 기본(/tmp) 유지: NFS TMPDIR에서 MathWorks 서비스 오류 5201. "
                          "MATLAB 산출은 NAS에만 쓴다",
            "G5_double_log": "공개 코드대로 저장한 log 기술자에 다시 log를 적용한 L1(log-log)을 주 설정으로 둔다"
                             "(createDescsAndRetrieve.m:82,97,336). 논문 식(8)의 한 번 log는 실행하지 않는다(audit G5)",
            "matlab_gui_shim": "공식 함수 무수정 대조 실행에서만 waitbar·visualizeMatchPairs를 matlab_gui_shim/(보이지 않는 "
                               "figure, 그리지 않음)으로 대체: 원래 waitbar가 -nodisplay -batch 유사도 단계에서 멈춤. 계산 줄 무관",
            "nonfinite_rule": "공식 점수가 비유한이거나 승자가 없으면 score = -1e12, 승자 없으면 j_star = "
                              "knnsearch 첫 후보, 개수는 원장 meta·summary에 기록"},
        "verification_before_test": {
            "C5_preprocessing": "pipeline/verification_prep.json: 공식 선택기+batch_multi_bev_gen 실행 파일 CSV == "
                                "드라이버 BEV 비트 동일(KITTI 학습05·Sejong01 검증·MCD 전치)",
            "C1_C3_matlab": "pipeline/verification_official_c2_kitti_09_1463.json: 공식 createDescsAndRetrieve.m "
                            "무수정 실행 대비 keys·fresco 비트 동일, 50임계 TP/FP/TN/FN 동일, 스칼라 점수 동치"},
        "build": {"build_json": "pipeline/build.json", "matlab_gen_json": "pipeline/matlab_gen.json",
                  "adapter_sha256": mgen["adapter_sha256"], "binaries_sha256": build["binaries"]},
        "code_sha256": code,
        "resources": "서버 cps2 호스트, nice 10/ionice idle, OMP/MKL/BLAS 1, MATLAB -singleCompThread, "
                     "receiver 1 + MATLAB 1 프로세스(<= 2코어), 큰 산출은 NAS 전용 루트만",
        "timing_label": "diagnostic hardware label(cps2 공유 CPU). RTX5080 원고 비용 열과 섞지 않는다",
    }
    write_json(OUT, proto)
    print("protocol.json frozen", proto["frozen_utc"])


if __name__ == "__main__":
    main()
