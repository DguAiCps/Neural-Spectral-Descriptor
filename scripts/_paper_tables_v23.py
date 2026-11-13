#!/usr/bin/env python3
"""현재 복소 차이 모델의 검증된 결과로 v2.3 표를 생성한다."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("AP", "max_F1", "R@100P", "R@1")
CURRENT = ROOT / "results/pdv2_diffattn_contrast_test_20261104"
# 저자 결정(2026-11-08): 표2의 SeqOT는 논문 학습법(20+20에폭)을 완주하고 공식 GeM을 같은 함수의 log-domain 계산으로
# 바꾼 3시드 재학습이다(paper recipe + numerical stabilization of official GeM). 공통 원천은 덮지 않고, SeqOT 항목만
# 바꾼 요약을 아래 해시로 고정해 채택한다.
COMMON_BASELINES = ROOT / "results/pdv2_baseline_seeds/summary.json"
SEQOT = ROOT / "results/pdv2_seqot_stable_gem_20261108"
BASELINES = SEQOT / "manuscript_integration_20261108/baseline_seeds_summary_seqot_stable.json"
ADOPTED_SEQOT = {
    "decided": "2026-11-08",
    "label": "paper recipe + numerical stabilization of official GeM",
    "records_sha256": "22767634596c2799273636e0a0231acdc05b735ef1952d668c89f3c7ac50c80f",
    "base_sha256": "8ee86c722504924426211407dd514a1f44f41e4e2680db024a76e72331600e7a",
    "common_sha256": "53cb0b724111661f113c49911e07b655b35d7f0645634aaada9fbc1da3f9dada",
}
# 저자 결정(2026-11-08): 표2·표S17의 SeqOT 비용은 위 채택 기록의 시드1 모델을 같은 로컬 RTX 5080·공통 규약으로 다시 잰 값이다.
# 기존 공통 비용 원천은 덮지 않고, 복사한 측정 파일을 아래 해시로 고정해 SeqOT 비용만 바꾼다. 세 반복을 모두 쓴다.
SEQOT_COST = SEQOT / "cost_20261108"
ADOPTED_SEQOT_COST = {
    "decided": "2026-11-08",
    "label": "paper recipe + numerical stabilization of official GeM",
    "cost_sha256": {"1000": "07444f920229491619d19ad4b111ede2dd5cc8efbfb1ca8fb65c84e662056c53",
                    "5000": "9f2f04aacdb67f8ddb021f1b84e6e86f5e83544b3290ff8f2a24df240a92e6a3",
                    "10000": "990162fde789a30b4c39feddfe65507c1000de0add929094206fdaf790528db5"},
    "verification_sha256": {"1000": "ef9c3bbeeda885559e88c4066a679cab26583d8320023f4605ffa045d6e15dbe",
                            "5000": "2a311343454227616454cc2a9ea79f9cf61541a4b0341be9495abd5d2c58d9ea",
                            "10000": "2408780a48a7557833f42c35181c5b8eb49eeac3489b002da79d468868913904"},
    "complete_sha256": {"1000": "02a0b530ca301b36189ba745f56e8662b33dd95a3785429ee4b047dc4c280483",
                        "5000": "51674e517e98671927b85e07d09484563eff877023f3925ac025fad77830ad16",
                        "10000": "fc612a27841dbfb25863c37ed4ae0f26b580626eca89407561b74e43344fd932"},
    "summary_sha256": "9af660f0a5c0ba445ce557a084cf4c88e21cd3668edf7b2878d5422c4311958e",
    "method_protocol_sha256": "242a2341bcf493da2800b848b6949e814c4a9d2feea95b232e87346d4fb9d19c",
    "protocol_sha256": "16ed8bf28cc91815cc728e92c6fe32d24dcc1a4639dc6477d6fb7e6fcb88e35c",
    "stage1_sha256": "e05465db4498fee6a80da3bfd63e4770f2b4a1ab52fa97cba6bffb0cbe49b407",
    "stage2_sha256": "d197f9ae67d57afeeea04b5e6f5d23ef6cf670e7b7a999b56454b03f3a979340",
    "source_sha256": "9c2fd5cb0e9f0f5734fdc05eb62442530df95c6d2c2b821ad2d4b9cdd4944184",
}
COST_PENDING = ()
EFFICIENCY = ROOT / "results/pdv2_table3_20261029/summary.json"
CURRENT_EFFICIENCY = ROOT / "results/pdv2_contrast_table3_opt_20261106/nsd_n10000_window.json"
INDEX_BUDGET = ROOT / "results/pdv2_contrast_index_budget_20261106"
DIRECTION = ROOT / "results/pdv2_review_direction_yaw_20261105"
MACRO = ROOT / "results/pdv2_macro_ablation_20261106"
MAPSIZE = ROOT / "results/pdv2_contrast_table3_mapsize_20261106"
# 저자 결정(2026-11-06): 결과를 바꾸지 않는 구현 정리만 적용한 window 측정을 표3에 쓴다.
# 측정 파일은 publish_ready=False로 남기고, 채택은 이 해시로 고정한다.
ADOPTED_COST = {"variant": "window", "decided": "2026-11-06", "sha256": "abcdbfaffac8fe2ff8a0bc9233cacae831240799d64adfa5bf9a6d8d081480ea"}
# 저자 결정(2026-11-07): 표2·표S17의 RING++는 공개 CUDA 특징의 비원자 격자 max만 정확 max(np.fmax.at)로 바꾼 구현이다.
# 인식(46시험 원장)과 비용(로컬 RTX 5080 실측)이 같은 구현이다. 공통 원천은 덮지 않고, RING++ 항목만 바꾼 사본과
# publish_ready=False로 남긴 측정 파일을 아래 해시로 고정해 채택한다. 다른 방법의 채택 조건은 그대로다.
RINGPP = ROOT / "results/pdv2_baseline_revision_20261107/ringpp"
# 2026-11-08: SeqOT 채택 요약을 기반으로 RING++ 사본을 같은 생성기로 다시 만들었다(RING++ 값은 그대로).
RINGPP_SUMMARY = SEQOT / "manuscript_integration_20261108/baseline_seeds_summary_seqot_stable_ringpp_exactmax.json"
ADOPTED_RINGPP = {
    "decided": "2026-11-07",
    "summary_sha256": "aa3fc8b157a632bcd8ede743eedbf6bd7c37c51780f306712cea9754cdbaafdf",
    "verification_sha256": "4d6dc9cd98499c87eac4414fe1ad007d8a5ea92b8080fd1f7e16582da6b753e5",
    "reverify_sha256": "6995673a78683c56a1bbc782cfaafab03b86696fe7ecc3f6098c7055b05ff225",
    "mapsize_sha256": "11bfd818dbe841691267ca843de0d8cc0daa615bca90791e9162f0ec15c3e278",
    "protocol_sha256": "16083d9c6ae27baf6eef77197c132021105d6ac555cfccce3f4491b3c99888f8",
    "cost_sha256": {"1000": "3a7290d077843135f5f3f274ef2d52d9671c76aa189c134e921910392bf0f057",
                    "5000": "2c7d770d53585e96edf0317ad53ba91af22e74953fa54ae87011dec4e344e3b1",
                    "10000": "cda79ac8b76abf0880bb3d4f889d55a0d9cc00a2f00bf190ab97f43e5321ed6a"},
}
# root 결정(2026-11-08): 표2·표S4·표S17에 FreSCo를 넣는다. 공식 MATLAB 2bf5cee·C++ d94040e의 미학습 1회 실행이다. NCLT 9개는 좌표 수정,
# KITTI 00·07·08·KITTI-360 0000은 공식 KittiPointCloudSelect.cpp에 고정 두 줄(1974440 이전 빈 칸 의미, scripts/fresco_official/cpp/kitti_groundfix.py)을 적용한 지면 수정 재평가 원장이다(소스 동작으로 채택, root kgf 검산).
# 원본 결과는 덮지 않는다. 인식은 지면 수정 패키지(FRESCO_KGF), 비용은 KITTI TU를 링크하지 않는 Sejong01 실측이라 기존 패키지(FRESCO) 사본을 해시로 고정한다.
FRESCO = ROOT / "results/pdv2_fresco_seqot_integration_20261108"
FRESCO_KGF = ROOT / "results/pdv2_fresco_kgf_integration_20261108"
FRESCO_SUMMARY = FRESCO_KGF / "baseline_seeds_summary_seqot_stable_ringpp_exactmax_fresco_kgf.json"
ADOPTED_FRESCO = {
    "decided": "2026-11-08",
    "label": "FreSCo official, untrained single run; KITTI-family ground-removal regression fix; NCLT sensor-frame adaptation",
    "summary_sha256": "a4557bfd3b517ee78eefb516b172929cf0c8b722e72abba777ac2fa785521473",
    "root_approval_sha256": "3901fe81acad16e1e1e98b247932fa5191fe758c25035d12e98182468cdb0217",
    "root_verification_sha256": "70dda3a7b742d8d9077f0c927abbd99a5d29c1d5b3ac302b183de20232ba7955",
    "corrected_summary_sha256": "105efdbc07bfcb6bc4d132755c3b0bb23cdef4a1de1cdc2cad8462f4c357c713",
    "previous_summary_sha256": "c74bfbfa18541e5cc1b3775f5ace10ee2569a08fd65140081fee12242bec0104",
    "cost_root_approval_sha256": "353c055bb31571d4f980c3a059580c281a618a143cddb82b3aba3665eaa233e8",
    "cost_sha256": {"1000": "09fe42220ad67d0ddcf391d11f19eed4a7b48a828de4d69dffac9ecfcb4af7d7",
                    "5000": "5a10705812d5feb36567224921c6c6c7297fe0e83f4ad47a527cf283c616df77",
                    "10000": "d5c2470cbf663238dc2a4f6f350f123afccc80943c3da0f6bbdfb2030cb6fab9"},
    "method_protocol_sha256": "18d435dc7789088f432986d0a400ed1efae9ece1461a23e28787b07c5091d6d7",
    "source_sha256": "8ac0aa7efbdf463906cff7f19b26982d52f6ffa49310468d1da8dd084488cf62",
}
# 사용자 승인(2026-11-10): 표2·표S4·표S17의 RING++는 정확max 특징에 상관 정규화 I2(평균0·분산1, float64 모분산, 정확 상수 채널 0, eps 없음)를
# 쓴 고정 변형이다. 공식 InstanceNorm(eps 1e-5)의 정확max 값(위 ADOPTED_RINGPP)은 보존해 부록 진단으로만 쓴다. 인식(root 검산 46원장)과
# 비용(같은 RTX 5080·규약)은 같은 I2 구현이다. 이전 원천은 덮지 않고, FreSCo 지면 수정 채택 요약에서 RING++만 바꾼 사본과 측정 파일을 해시로 고정한다.
RINGPP_I2 = ROOT / "results/pdv2_ringpp_i2_integration_20261110"
RINGPP_I2_SUMMARY = RINGPP_I2 / "baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf.json"
ADOPTED_RINGPP_I2 = {
    "decided": "2026-11-10",
    "label": "exact voxel maximum + I2 correlation normalization (zero mean, unit variance, no epsilon)",
    "summary_sha256": "1cccdb755ae14ea14aab35c508883a677252473912909cd11f766b5fba8e7cca",
    "adoption_basis_sha256": "67381ce14483fc2f9bfd8f16ec30781affd9722d3cf83caaa790db00db76b706",
    "test46_root_verification_sha256": "b1e7789665cde3da3b16d294a90b17e128575eaf2fc94c397a765be13f790867",
    "test46_protocol_sha256": "5e72d07d1eef45b12e5f188b99b89ebdfded976adea23f89f78b2442d81c85fe",
    "normalization_sha256": "051a6179a018dba565f6abfd7195c8609c5f336b4c279243a2ec872e21d4130c",
    "maxfix_protocol_sha256": "bdcc048237953a9eec3afedba17d025bc79e76db85a9e9561395097ed37eb7a8",
    "cost_protocol_sha256": "76012bb3b4c864b086713d0882b31d0d1798a0f6db6270c9fd582758b075731b",
    "cost_verification_sha256": "1b3615b16525ff60d3b6e3f34d4f2dbec94af760bdc1bc7d0c036d2693cb9254",
    "equivalence_sha256": "3a8cb9d7ca647ccfcff31156246c39057075f507d08279c278292a3d07dfca8a",
    "cost_sha256": {"1000": "a1b9a61675fc8538bae27728dc6053f30642f0a363f190d3e5016ea55ebe01fe",
                    "5000": "faa85c6bd8f1ba44405d41f9b2b560e4a818fa86906302a81b0f071a23daa485",
                    "10000": "cbd076a132d30f8c98dbcd71ccce3f395474653b030b00a04ff867a8bfb53e8a"},
    "source_sha256": "3f93c67f1441ec64df0cd69cddcb570585d12f9f3b5985ad22bc676a0d481785",
}
# 사용자 승인(2026-11-10, "교체합시다"): 표2·표S4의 DeepRING은 식 9 스칼라 헤드 W·C+b를 W0 = 1/(학습8 고정 20에피소드 2,880질의의
# 중심화 점수 RMS), b0 = 0으로 시작한 재구현이다(RING circorr2 점수, 시드 1·3·4, 검증9 선택 에폭 16·3·6). 모듈·매개변수 수·서술자·추론
# 연산이 같아 비용은 기존 같은 구현 실측(EFFICIENCY)을 그대로 쓴다. root 최종 검산(28/28·독립 재집계)을 거친 시험46 138원장 사본과
# I2 채택 요약에서 DeepRING만 바꾼 사본을 해시로 고정한다. 이전 W=1 시작 항목과 원논문 식 8 대조 결과는 보존한다.
DEEPRING = ROOT / "results/pdv2_deepring_scaleinit_integration_20261110"
DEEPRING_SUMMARY = DEEPRING / "baseline_seeds_summary_seqot_stable_ringpp_i2_fresco_kgf_deepring_scaleinit.json"
ADOPTED_DEEPRING = {
    "decided": "2026-11-10",
    "label": "paper-based reimplementation; Eq. 9 head initialized at W0 = 1/RMS(centered training scores), b0 = 0",
    "summary_sha256": "1527727946d6fccd4cf3a635ca074c6291143634fc3302a84d69ea388177891e",
    "adoption_basis_sha256": "a26c198c98b77e83bda2c69494ba67b5ec48cc71eaf752615256f4187861404b",
    "root_verification_sha256": "3517198cd10ec5b0c93b4b2ec587a7fbb478f2024267c12335787b4a37ce19b4",
    "test46_protocol_sha256": "598910a4f2759116d4806405225599549dcb039685a663c3b0cc98f159afc3d2",
    "server_sha256sums_sha256": "765c681148107f378a93aaa95df9fe2be90f33e5b5bb98c9a4940482a7e75416",
    "calibration_sha256": "0f9f6265c936cad727f0ee3dc179ce17a65594f9d58bf3b532868429f08e9712",
    "epochs": {"1": 16, "3": 3, "4": 6},
}
# 사용자 결정(2026-11-10 밤): 부록 비교 방법 절에 표2의 RING++·SeqOT·FreSCo·DeepRING을 공개 코드 변형·이전 구성과 같은 공통 시험 규약으로
# 나란히 싣는다(tab:baseline_release_comparison). 원천 목록은 /mnt/d/NSD_datasets/nsd_results/baseline_defense_20261110/facts/facts.json의
# item2_released_vs_table2이며, 생성기는 그 목록의 원래 결과 JSON을 읽는다. 공개 커널 RING++·공개 코드 학습법 SeqOT는 COMMON_BASELINES,
# 이전 W=1 시작 DeepRING은 DEEPRING_SUMMARY의 대체 전 항목이다.
RINGPP_MAXFIX = ROOT / "results/pdv2_ringpp_maxfix_20261105/summary_verified.json"
FRESCO_PREFIX = FRESCO / "baseline_seeds_summary_seqot_stable_ringpp_exactmax_fresco.json"
# 사용자 요청(2026-11-07): 부록 데이터셋 표는 모든 방법을 싣는다. 46시험 최종 원장을 데이터셋별로 다시 집계한 결과이며(원장·채택 요약 해시는 그 안),
# DeepRING 초기배율 채택 요약으로 다시 돌렸다(DeepRING 외 10방법 값이 직전 RING++ I2 집계와 같음은 그 파일 checks).
DATASET_METHODS = DEEPRING / "datasets/dataset_methods_summary.json"
DATASET_METHODS_SHA256 = "d5129726996e98a15a5a042aba9afa5a542b0c7d5fd304cdadfa71f36a5aeaf8"
# 사용자 요청(2026-11-07): 부록에 검증에서 고정한 단일 판정 기준값의 시험 결과를 싣는다. 원천은 기준값 작업 결과이며
# 규약·고정값·요약·오수락 층화·검산 파일을 아래 해시로 고정한다. 표2 수치와 그림5(기준값 sweep)는 바꾸지 않는다.
THRESHOLD = ROOT / "results/pdv2_review_closure_20261107/threshold"
THRESHOLD_SHA256 = {
    "protocol.json": "78415ff5b6db7529cd7abaf849867f3dee47e478f1f94dd6319a009855508201",
    "freeze.json": "e7561cb4d093f1efd8ef1a659de601008b9b1a5518b634e8e6937e1726dd0218",
    "summary.json": "ea7244b786507035f01cf600c583e320ae366e52d850a7c632c056b1d15ccfb3",
    "false_accepts.json": "10d32b01498f784a24bd25f1735fd085e75229ed6d9eaa297f37f761b59ed313",
    "verification.json": "224ef6eb5c9c5fd8a717c2d667375567f2930dbb2f887f777a9bdfc608680833",
}
# 사용자 요청(2026-11-07): 부록에 정확한 분할 목록과 분할 쌍 50 m 지리 감사를 싣는다. 원천은 지리 감사 작업 결과이며
# 목록·감사 요약·검산 파일을 아래 해시로 고정한다. 공통 좌표가 없는 쌍은 하한·도시 수준으로만 싣는다.
GEOGRAPHY = ROOT / "results/pdv2_review_closure_20261107/geography"
GEOGRAPHY_SHA256 = {
    "roster.json": "82c1d185d4279eebd98829cfa092ba72b7209bdb2805ca400b29edd2601bb672",
    "geo_audit.json": "d4d5b2a1df72718e56f4c47c4400b64ca132cb1e45c16805fe6b21a9ddfe8b79",
    "verification.json": "1ad34bdbc666185b1e909c79e473bbf94af2c3b1e45ae4bd0407281093364b3a",
}
# 지리 감사의 독립 방법 검토(2026-11-07). Bridge 하한은 방향 가정 없이 공개 교량 좌표의 150 m 안 통과만 가정한 값을 쓴다.
GEO_REVIEW = ROOT / "results/pdv2_review_closure_20261107/geo_audit"
GEO_REVIEW_SHA256 = {
    "verification.json": "1fda0c0541ed9b7769af94f5ad264850e278a625c60a03c7473cb35756a252b8",
    "bridge_rotation_free_check.json": "8f48c1d35b8a4a4e4a99b7d7d5add51415575896a011b31f3b056599c184ce75",
}
# 저자 결정(2026-11-07): ImLPR는 현재 비교군에서 뺀다(results/pdv2_baseline_revision_20261107/decision.json).
# 결과 파일은 보존하고, 표의 순위·비용 열은 아래 10개 방법(2026-11-08 FreSCo 추가)과 NSD만으로 계산한다.
NAMES = {
    "disco": r"DiSCO~\citep{xu2021disco}",
    "deepring": r"DeepRING~\citep{lu2022deepring}",
    "ring": r"RING~\citep{lu2022ring}", "ringpp": r"RING++~\citep{xu2023ringpp}",
    "ringsharp": r"RING\#-L~\citep{lu2024ringsharp}", "scpp": r"SC++~\citep{kim2022scancontextpp}",
    "fresco": r"FreSCo~\citep{fan2022fresco}",
    "helios": r"HeLiOS~\citep{jung2025helios}",
    "seqot": r"SeqOT~\citep{ma2022seqot}", "bevplace": r"BEVPlace++~\citep{luo2025bevplacepp}",
}
VARIANTS = {
    "identity": "No temporal spectrum refining",
    "fixed": "Uniform subtraction",
    "point_gate": r"Pointwise gate$^\dagger$",
    "abs_kv": "Absolute K/V attention",
    "other_only": "Projection, norm, and gate learning",
    "convs_only": "DiffAttnConv learning",
    "diff": r"\textbf{NSD}",
}


def read(path: Path) -> dict:
    """JSON 결과를 읽는다."""
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    """원천 파일의 SHA256을 계산한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(caption: str, label: str, header: str, rows: list,
          columns: str = "lcccc", wide: bool = False,
          compact: bool = False) -> str:
    """표를 지정 폭 안에 배치하고 자연 크기보다 확대하지 않는다."""
    env = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    padding = "2pt" if compact else "4pt"
    return ("% 현재 결과 JSON에서 자동 생성한다. 숫자를 직접 고치지 않는다.\n"
            + rf"\begin{{{env}}}[t]" + "\n"
            + rf"\caption{{{caption}}}\label{{{label}}}" + "\n"
            + rf"\centering\footnotesize\setlength{{\tabcolsep}}{{{padding}}}" + "\n"
            + r"\sbox0{%" + "\n"
            + rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}\toprule" + "\n"
            + header + r" \\" + "\n" + r"\midrule" + "\n"
            + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}}" + "\n"
            + rf"\ifdim\wd0>{width}\resizebox{{{width}}}{{!}}{{\usebox0}}\else\usebox0\fi" + "\n"
            + rf"\end{{{env}}}" + "\n")


def value_cell(value: float, sd=None, rank=None) -> str:
    """비율을 백분율 소수 둘째 자리로 바꿔 평균·표준편차·1·2위 표시를 만든다.

    2026-11-06 가독성 보완: 본문 글꼴 숫자를 쓰고, 소수 넷째 자리 비율과 같은 정밀도를 유지한다.
    """
    text = f"{100 * value:.2f}"
    if rank == 0:
        text = r"\textbf{" + text + "}"
    elif rank == 1:
        text = r"\underline{" + text + "}"
    if sd is not None:
        text += r"\textsubscript{$\pm$" + f"{100 * sd:.2f}" + "}"
    return text


def signed(value: float, digits: int = 2) -> str:
    """부호가 있는 %p 값을 수학 부호와 본문 글꼴 숫자로 쓴다.

    소수 둘째 자리로 쓰되, 0이 아닌 값이 0.00으로 보이면 부호 해석이 사라지므로 셋째 자리까지 쓴다.
    """
    if value != 0 and abs(value) < 0.005 and digits == 2:
        digits = 3
    return ("$+$" if value >= 0 else "$-$") + f"{abs(value):.{digits}f}"


def interval_cell(value: dict) -> str:
    """%p 차이와 조건부 95% 구간을 본문 글꼴 숫자로 쓴다."""
    lo, hi = value["ci95_pp"]
    return f"{signed(value['delta_pp'])} [{signed(lo)}, {signed(hi)}]"


def seqot_adoption() -> list:
    """채택한 SeqOT 요약이 독립 검산 요약·138원장 해시와 같고, 다른 방법은 공통 원천과 같은지 확인한다."""
    records = SEQOT / "server_mirror/records/summary.json"
    assert digest(records) == ADOPTED_SEQOT["records_sha256"] and digest(BASELINES) == ADOPTED_SEQOT["base_sha256"]
    assert digest(COMMON_BASELINES) == ADOPTED_SEQOT["common_sha256"]
    base, common, rec = read(BASELINES), read(COMMON_BASELINES), read(records)
    replacement = base["seqot_replacement"]
    assert replacement["label"] == rec["label"] == ADOPTED_SEQOT["label"]
    assert replacement["base_summary_sha256"] == ADOPTED_SEQOT["common_sha256"]
    assert replacement["records_summary_sha256"] == ADOPTED_SEQOT["records_sha256"]
    assert base["methods"].keys() == common["methods"].keys()
    assert all(base["methods"][m] == common["methods"][m] for m in common["methods"] if m != "seqot")
    entry = base["methods"]["seqot"]
    for k in METRICS:
        assert abs(entry["mean_std"][k]["mean"] - rec["mean_std"][k]["mean"]) < 1e-12
        assert abs(entry["mean_std"][k]["std"] - rec["mean_std"][k]["std"]) < 1e-12
    ledgers = [p for p in replacement["seqot_inputs_sha256"] if "/dump_seqotl2_" in p]
    assert len(ledgers) == 138 and all(p.startswith(str(SEQOT.relative_to(ROOT)) + "/") for p in ledgers)
    for path, expected in replacement["seqot_inputs_sha256"].items():
        assert digest(ROOT / path) == expected, path
    return [records, COMMON_BASELINES]


def ringpp_adoption() -> tuple:
    """채택한 정확max RING++의 인식 사본과 지도 크기별 비용을 검산해 돌려준다."""
    paths = {"summary": RINGPP_SUMMARY,
             "verification": RINGPP / "verification.json",
             "reverify": RINGPP / "test_reverify/verification.json",
             "mapsize": RINGPP / "costs_mapsize.json",
             "protocol": RINGPP / "cost/ringpp_exactmax_protocol.json"}
    for key, path in paths.items():
        assert digest(path) == ADOPTED_RINGPP[key + "_sha256"], path
    assert read(paths["verification"])["passed"] and read(paths["mapsize"])["passed"] and read(paths["reverify"])["passed"]
    summary, base = read(paths["summary"]), read(BASELINES)
    replacement = summary["ringpp_replacement"]
    assert replacement["base_summary_sha256"] == digest(BASELINES)
    # RING++ 외 방법은 공통 원천과 값이 같아야 한다.
    assert summary["methods"].keys() == base["methods"].keys()
    assert all(summary["methods"][m] == base["methods"][m] for m in base["methods"] if m != "ringpp")
    entry = summary["methods"]["ringpp"]
    assert entry["per_seed"]["-"]["source"] == "results/pdv2_ringpp_maxfix_20261105/odom"
    ledgers = [p for p in replacement["ringpp_inputs_sha256"] if p.startswith(entry["per_seed"]["-"]["source"] + "/dump_ringpp_")]
    assert len(ledgers) == 46
    for path, expected in replacement["ringpp_inputs_sha256"].items():
        assert digest(ROOT / path) == expected, path
    reverify = read(paths["reverify"])["macro"]
    assert all(abs(entry["mean_std"][k]["mean"] - reverify[k]) < 1e-12 for k in METRICS)
    protocol = read(paths["protocol"])
    for path, expected in protocol["sources"].items():
        assert digest(ROOT / path) == expected, path
    common = next(m for m in read(EFFICIENCY)["measurements"] if m["method"] == "ringpp" and m["n_map"] == 10000)
    costs = {}
    for n in (1000, 5000, 10000):
        path = RINGPP / f"cost/methods/ringpp_n{n}.json"
        assert digest(path) == ADOPTED_RINGPP["cost_sha256"][str(n)], path
        m = read(path)
        assert m["status"] == "passed" and m["method"] == "ringpp" and m["n_map"] == n and m["same_test_implementation"]
        assert m["method_protocol_sha256"] == ADOPTED_RINGPP["protocol_sha256"]
        # 공통 측정 규약(Sejong01, 96도착×3반복, 워밍업 12회)은 다른 방법과 같다.
        assert m["protocol_sha256"] == common["protocol_sha256"]
        assert (m["samples"], m["repeats"], m["warmup"]) == (96, 3, 12)
        assert m["environment"]["gpu"] == "NVIDIA GeForce RTX 5080" and m["environment"]["threads"] == m["environment"]["torch_threads"] == 1
        assert m["official_native_repeat_variation"]["winner_changed_queries"] == 0
        costs[n] = m
    sources = list(paths.values()) + [RINGPP / f"cost/methods/ringpp_n{n}.json" for n in (1000, 5000, 10000)]
    return summary, costs, sources


def seqot_cost_adoption() -> tuple:
    """채택한 SeqOT 시드1 모델의 지도 크기별 비용을 해시·규약·참조 검사로 검산해 돌려준다."""
    pin = ADOPTED_SEQOT_COST
    protocol_path, summary_path = SEQOT_COST / "seqot_protocol.json", SEQOT_COST / "summary.json"
    assert digest(protocol_path) == pin["method_protocol_sha256"] and digest(summary_path) == pin["summary_sha256"]
    assert digest(SEQOT_COST / "protocol.json") == pin["protocol_sha256"]
    protocol = read(protocol_path)
    assert protocol["label"] == ADOPTED_SEQOT["label"] == pin["label"] and protocol["common_protocol_sha256"] == pin["protocol_sha256"]
    assert protocol["stage1"]["sha256"] == pin["stage1_sha256"] and protocol["stage2"]["sha256"] == pin["stage2_sha256"]
    assert protocol["batch"] == 1 and protocol["threads"] == 1 and protocol["tf32"] is False
    for path, expected in protocol["sources"].items():
        assert digest(ROOT / path) == expected, path
    # 비용 모델은 표2 정확도 채택 기록의 시드1 모델이다.
    seed1 = read(SEQOT / "server_mirror/records/summary.json")["training"]["1"]
    assert seed1["stage1"]["sha256"] == pin["stage1_sha256"] and seed1["checkpoint_sha256"] == pin["stage2_sha256"]
    common = next(m for m in read(EFFICIENCY)["measurements"] if m["method"] == "seqot" and m["n_map"] == 10000)
    costs, sources = {}, [protocol_path, summary_path, SEQOT_COST / "protocol.json"]
    for n in (1000, 5000, 10000):
        paths = {"cost": SEQOT_COST / f"methods/seqot_n{n}.json",
                 "verification": SEQOT_COST / f"verification/seqot_n{n}.json",
                 "complete": SEQOT_COST / f"seqot_batch1/n{n}/complete.json"}
        for key, path in paths.items():
            assert digest(path) == pin[key + "_sha256"][str(n)], path
        m, check, complete = read(paths["cost"]), read(paths["verification"]), read(paths["complete"])
        assert m["status"] == check["status"] == complete["status"] == "passed" and m["verification"] == check
        assert m["method"] == "seqot" and m["n_map"] == n and m["seed"] == 1
        assert m["source_sha256"] == pin["source_sha256"] and m["method_protocol_sha256"] == pin["method_protocol_sha256"]
        # 공통 측정 규약(Sejong01, 96도착×3반복, 워밍업 12회)은 다른 방법과 같다.
        assert m["protocol_sha256"] == common["protocol_sha256"] == pin["protocol_sha256"]
        assert (m["samples"], m["repeats"], m["warmup"]) == (96, 3, 12) and len(m["raw_samples"]) == 288
        assert m["environment"]["gpu"] == "NVIDIA GeForce RTX 5080" and m["environment"]["threads"] == 1
        # 증분 서술자는 같은 GPU·가중치의 배치1 참조와 같고, 공식 접두 L2 검사 12개가 통과한다.
        assert check["reference_cache_manifest_sha256"] == pin["complete_sha256"][str(n)]
        assert len(check["official_prefix_checks"]) == 12 and all(c["score_error"] == 0.0 for c in check["official_prefix_checks"])
        assert check["streaming_reference_checks"] and all(c["max_error"] == 0.0 for c in check["streaming_reference_checks"])
        assert m["resident_bytes"]["total"] == m["resident_bytes"]["cpu"] + m["resident_bytes"]["gpu"]
        # 표의 평균은 세 반복 288표본 전체에서 다시 계산한 값과 같다(느린 반복도 제외하지 않는다).
        for key, field in (("update", "update"), ("query", "query_ms"), ("processing", "processing")):
            assert abs(statistics.mean(s[field] for s in m["raw_samples"]) - m["timings_ms"][key]["mean"]) < 1e-9
        assert sorted({s["repeat"] for s in m["raw_samples"]}) == [0, 1, 2]
        costs[n] = m
        sources += list(paths.values())
    return costs, sources


def fresco_adoption(ringpp_summary: dict) -> tuple:
    """root가 검산한 FreSCo 인식(46원장)·지도 크기별 비용을 해시·규약·원표본으로 검산해 돌려준다.

    요약은 정확max RING++ 채택 사본에 FreSCo 항목만 더한 것이어야 하고, 매크로는 지면 수정 요약·root 검산 값과 같아야 한다.
    """
    pin, root = ADOPTED_FRESCO, read(FRESCO_KGF / "root/kgf_root_verification.json")
    approval_path, cost_approval = FRESCO_KGF / "root_approval.json", FRESCO / "root_approval.json"
    assert digest(FRESCO_SUMMARY) == pin["summary_sha256"] and digest(approval_path) == pin["root_approval_sha256"]
    assert digest(FRESCO_KGF / "root/kgf_root_verification.json") == pin["root_verification_sha256"] and root["passed"]
    assert digest(FRESCO_KGF / "fresco/summary.json") == pin["corrected_summary_sha256"] and digest(cost_approval) == pin["cost_root_approval_sha256"]
    for base, rel, item in ([(FRESCO_KGF, r, i) for r, i in read(approval_path)["files"].items()]
                            + [(FRESCO, r, i) for r, i in read(cost_approval)["files"].items() if r.startswith("fresco/cost/")]):
        assert digest(base / rel) == item["sha256"], rel
    summary = read(FRESCO_SUMMARY)
    addition = summary["fresco_addition"]
    assert addition["base_summary_sha256"] == digest(RINGPP_SUMMARY) == ADOPTED_RINGPP["summary_sha256"]
    assert (addition["root_approval_sha256"], addition["previous_fresco_summary_sha256"]) == (pin["root_approval_sha256"], pin["previous_summary_sha256"])
    # FreSCo 외 방법은 정확max RING++ 채택 사본과 값이 같아야 한다.
    assert set(summary["methods"]) == set(ringpp_summary["methods"]) | {"fresco"}
    assert all(summary["methods"][m] == ringpp_summary["methods"][m] for m in ringpp_summary["methods"])
    entry = summary["methods"]["fresco"]
    assert entry["training_free"] and entry["seeds"] == ["-"] and entry["score"] == "score_fresco"
    ledgers = [p for p in addition["fresco_inputs_sha256"] if "/dump_fresco_" in p]
    assert len(ledgers) == 46
    for path, expected in addition["fresco_inputs_sha256"].items():
        assert digest(ROOT / path) == expected, path
    corrected = read(FRESCO_KGF / "fresco/summary.json")["scores"]["score_fresco"]["macro"]
    for k in METRICS:
        assert abs(entry["mean_std"][k]["mean"] - corrected[k]) < 1e-12 and abs(entry["mean_std"][k]["mean"] - root["scores"]["after_unrounded"][k]) < 1e-12, k
    common = next(m for m in read(EFFICIENCY)["measurements"] if m["method"] == "scpp" and m["n_map"] == 10000)
    costs, sources = {}, [FRESCO_SUMMARY, approval_path, FRESCO_KGF / "root/kgf_root_verification.json", FRESCO_KGF / "fresco/summary.json", cost_approval]
    for n in (1000, 5000, 10000):
        path = FRESCO / f"fresco/cost/fresco_n{n}.json"
        assert digest(path) == pin["cost_sha256"][str(n)], path
        m = read(path)
        assert m["status"] == m["verification"]["status"] == "passed" and m["method"] == "fresco" and m["n_map"] == n
        assert m["method_protocol_sha256"] == pin["method_protocol_sha256"] and m["source_sha256"] == pin["source_sha256"]
        # 공통 측정 규약(Sejong01, 96도착×3반복, 워밍업 12회)은 다른 방법과 같다. CPU 1스레드, GPU는 쓰지 않는다.
        assert m["protocol_sha256"] == common["protocol_sha256"]
        assert (m["samples"], m["repeats"], m["warmup"]) == (96, 3, 12) and len(m["raw_samples"]) == 288
        assert m["environment"]["threads"] == 1 and m["resident_bytes"]["gpu"] == 0
        assert m["verification"]["measured_queries_checked"] == 288 and m["verification"]["winners_and_confidence_identical_across_repeats"]
        # 상주 상태는 키 행렬·키 셀(공식 중복 보유)·기술자·마지막 kd-tree의 합이며 MATLAB 런타임·질의 기록은 뺀다.
        resident = m["resident_bytes"]
        assert resident["total"] == resident["cpu"] == sum(resident["breakdown"].values())
        # 표의 평균은 세 반복 288표본 전체에서 다시 계산한 값과 같다.
        for key, field in (("update", "update"), ("query", "query_ms"), ("processing", "processing")):
            assert abs(statistics.mean(s[field] for s in m["raw_samples"]) - m["timings_ms"][key]["mean"]) < 1e-9
        assert sorted({s["repeat"] for s in m["raw_samples"]}) == [0, 1, 2]
        costs[n] = m
        sources.append(path)
    return summary, costs, sources


def ringpp_i2_adoption(fresco_summary: dict) -> tuple:
    """I2 RING++의 인식 사본(46원장)과 지도 크기별 비용을 검산하고, 두 원천이 같은 구현인지 확인해 돌려준다."""
    pin = ADOPTED_RINGPP_I2
    paths = {"summary": RINGPP_I2_SUMMARY, "adoption_basis": RINGPP_I2 / "adoption_basis.json",
             "test46_root_verification": RINGPP_I2 / "ringpp/test46/root_verification.json",
             "test46_protocol": RINGPP_I2 / "ringpp/test46/protocol_freeze.json",
             "normalization": RINGPP_I2 / "ringpp/test46/i2_norm.py",
             "maxfix_protocol": ROOT / "results/pdv2_ringpp_maxfix_20261105/protocol.json",
             "cost_protocol": RINGPP_I2 / "cost/ringpp_i2_protocol.json",
             "cost_verification": RINGPP_I2 / "cost/verification_cost.json",
             "equivalence": RINGPP_I2 / "cost/equivalence/verification.json"}
    for key, path in paths.items():
        assert digest(path) == pin[key + "_sha256"], path
    root, verification, basis = (read(paths[k]) for k in ("test46_root_verification", "cost_verification", "adoption_basis"))
    assert root["passed"] and root["protocol_sha256"] == pin["test46_protocol_sha256"] and verification["passed"]
    for rel, item in basis["files"].items():
        assert digest(RINGPP_I2 / rel) == item["sha256"], rel
    summary = read(RINGPP_I2_SUMMARY)
    replacement = summary["ringpp_i2_replacement"]
    assert replacement["base_summary_sha256"] == digest(FRESCO_SUMMARY) == ADOPTED_FRESCO["summary_sha256"]
    assert replacement["adoption_basis_sha256"] == pin["adoption_basis_sha256"]
    # RING++ 외 방법은 FreSCo 지면 수정 채택 요약과 값이 같고, 대체 전 RING++는 정확max I1(보존)이다.
    assert summary["methods"].keys() == fresco_summary["methods"].keys()
    assert all(summary["methods"][m] == fresco_summary["methods"][m] for m in fresco_summary["methods"] if m != "ringpp")
    assert replacement["previous_ringpp_entry"] == fresco_summary["methods"]["ringpp"]
    entry = summary["methods"]["ringpp"]
    assert entry["per_seed"]["-"]["source"] == replacement["source"] == str((RINGPP_I2 / "ringpp/ledgers/test").relative_to(ROOT))
    ledgers = [p for p in replacement["ringpp_inputs_sha256"] if p.startswith(replacement["source"] + "/dump_ringpp_i2_")]
    assert len(ledgers) == 46
    for path, expected in replacement["ringpp_inputs_sha256"].items():
        assert digest(ROOT / path) == expected, path
    assert all(abs(entry["mean_std"][k]["mean"] - root["macro"]["I2"][k]) < 1e-12 for k in METRICS)
    # 인식과 비용이 같은 구현: 같은 정규화 소스, 같은 정확max 인코더 규약, 원시 스캔 재인코딩이 채택 원장과 같은 승자.
    protocol, test_protocol = read(paths["cost_protocol"]), read(paths["test46_protocol"])
    assert replacement["normalization_sha256"] == protocol["normalization"]["sha256"] == pin["normalization_sha256"]
    assert protocol["normalization"]["test46_protocol_sha256"] == replacement["test46_protocol_sha256"] == pin["test46_protocol_sha256"]
    assert test_protocol["test_inputs"]["maxfix_protocol_sha256"] == pin["maxfix_protocol_sha256"]
    assert protocol["exactmax_protocol_sha256"] == ADOPTED_RINGPP["protocol_sha256"]
    assert protocol["base_cache_protocol"] == read(RINGPP / "cost/ringpp_exactmax_protocol.json")["base_cache_protocol"]
    assert protocol["initial_cache"]["official_i1_fft_reuse"] is False
    equivalence = read(paths["equivalence"])
    dcc01 = replacement["source"] + "/dump_ringpp_i2_mulran_DCC01.npz"
    assert equivalence["passed"] and equivalence["winner_equal"] == equivalence["queries"] == 98
    assert equivalence["ledger_sha256"] == replacement["ringpp_inputs_sha256"][dcc01]
    common = next(m for m in read(EFFICIENCY)["measurements"] if m["method"] == "ringpp" and m["n_map"] == 10000)
    costs = {}
    for n in (1000, 5000, 10000):
        path = RINGPP_I2 / f"cost/methods/ringpp_i2_n{n}.json"
        assert digest(path) == pin["cost_sha256"][str(n)], path
        m = read(path)
        assert m["status"] == "passed" and m["method"] == "ringpp" and m["n_map"] == n and m["same_test_implementation"]
        assert m["normalization_source_sha256"] == pin["normalization_sha256"] and m["normalization"] == "I2"
        assert m["method_protocol_sha256"] == pin["cost_protocol_sha256"] and m["source_sha256"] == pin["source_sha256"]
        # 공통 측정 규약(Sejong01, 96도착×3반복, 워밍업 12회)과 도착 점군은 다른 방법·정확max I1 측정과 같다.
        assert m["protocol_sha256"] == common["protocol_sha256"]
        assert m["raw_input_sha256"] == read(RINGPP / f"cost/methods/ringpp_n{n}.json")["raw_input_sha256"]
        assert (m["samples"], m["repeats"], m["warmup"]) == (96, 3, 12) and len(m["raw_samples"]) == 288
        assert m["environment"]["gpu"] == "NVIDIA GeForce RTX 5080" and m["environment"]["threads"] == m["environment"]["torch_threads"] == 1
        assert m["official_native_repeat_variation"]["winner_changed_queries"] == 0
        official = m["verification"]["all_queries_same_execution"]
        assert official["status"] == "passed" and official["queries"] == official["winners_exact"] == 288
        assert verification["per_map_size"][str(n)] and all(verification["per_map_size"][str(n)].values())
        for key, field in (("update", "update"), ("query", "query_ms"), ("processing", "processing")):
            assert abs(statistics.mean(s[field] for s in m["raw_samples"]) - m["timings_ms"][key]["mean"]) < 1e-9
        costs[n] = m
    sources = list(paths.values()) + [RINGPP_I2 / f"cost/methods/ringpp_i2_n{n}.json" for n in (1000, 5000, 10000)]
    return summary, costs, sources


def deepring_adoption(ringpp_i2_summary: dict) -> tuple:
    """초기배율 DeepRING의 인식 사본(138원장)을 root 최종 검산·고정 선택·학습셋 보정 기록으로 검산해 돌려준다."""
    pin = ADOPTED_DEEPRING
    paths = {"summary": DEEPRING_SUMMARY, "adoption_basis": DEEPRING / "adoption_basis.json",
             "root_verification": DEEPRING / "deepring/audit/root_verification.json",
             "test46_protocol": DEEPRING / "deepring/test46/protocol.json",
             "server_sha256sums": DEEPRING / "deepring/test46/server_SHA256SUMS.txt",
             "calibration": DEEPRING / "deepring/calibration/calibration.json"}
    for key, path in paths.items():
        assert digest(path) == pin[key + "_sha256"], path
    root, basis = read(paths["root_verification"]), read(paths["adoption_basis"])
    assert root["all_ok"] and all(root["checks"].values())
    for rel, item in basis["files"].items():
        assert digest(DEEPRING / rel) == item["sha256"], rel
    audit = read(DEEPRING / "deepring/audit/verification.json")
    assert audit["all_ok"] and audit["n_checks"] == 28 and all(audit["checks"].values())
    summary = read(DEEPRING_SUMMARY)
    replacement = summary["deepring_scaleinit_replacement"]
    assert replacement["base_summary_sha256"] == digest(RINGPP_I2_SUMMARY) == ADOPTED_RINGPP_I2["summary_sha256"]
    assert replacement["adoption_basis_sha256"] == pin["adoption_basis_sha256"]
    assert replacement["root_verification_sha256"] == pin["root_verification_sha256"]
    assert replacement["test46_protocol_sha256"] == pin["test46_protocol_sha256"] and replacement["epochs"] == pin["epochs"]
    # DeepRING 외 방법은 I2 채택 요약과 같고, 대체 전 DeepRING은 W=1 시작 표2 항목(보존)이다.
    assert summary["methods"].keys() == ringpp_i2_summary["methods"].keys()
    assert all(summary["methods"][m] == ringpp_i2_summary["methods"][m] for m in ringpp_i2_summary["methods"] if m != "deepring")
    assert replacement["previous_deepring_entry"] == ringpp_i2_summary["methods"]["deepring"]
    entry = summary["methods"]["deepring"]
    assert entry["seeds"] == list(pin["epochs"]) and not entry["training_free"] and entry["score"] == "score_corr"
    # 시드·에폭은 시험 전에 검증9로 고정한 선택과 같다. W0 보정은 학습 8시퀀스의 고정 20에피소드(2,880질의)만 쓴다.
    selection = read(DEEPRING / "deepring/test46/fixed_selection.json")["seeds"]
    assert {s: selection[s]["epoch"] for s in pin["epochs"]} == pin["epochs"]
    calibration = read(paths["calibration"])
    assert not calibration["fails"] and len(calibration["fixed_episodes"]) == 20
    assert all(calibration["seeds"][s]["n_queries"] == 2880 and calibration["seeds"][s]["b0"] == 0.0 for s in pin["epochs"])
    assert all(e[0].split("/")[0] in ("kitti", "helipr") and e[0] in read(CURRENT / "protocol.json")["train"]
               for e in calibration["fixed_episodes"])
    audited = read(DEEPRING / "deepring/audit/audit_aggregate.json")["per_run"]["S_new"]
    ledgers = [p for p in replacement["deepring_inputs_sha256"] if p.endswith(".npz") and "/deepring/ledgers/" in p]
    assert len(ledgers) == 138
    for path in ledgers:
        assert digest(ROOT / path) == replacement["deepring_inputs_sha256"][path] == summary["input_sha256"][path], path
    for seed, run in entry["per_seed"].items():
        assert run["source"] == str((DEEPRING / f"deepring/ledgers/s{seed}/odom").relative_to(ROOT))
        assert sum(p.startswith(run["source"] + "/dump_deepring_") for p in ledgers) == 46
        # root 최종 검산의 독립 집계(공통 질의, 6도시 매크로)와 같다.
        assert all(abs(run["macro"][k] - audited[seed]["macro"][k]) < 1e-12 for k in METRICS)
    for k in METRICS:
        values = [entry["per_seed"][s]["macro"][k] for s in entry["seeds"]]
        assert abs(statistics.mean(values) - entry["mean_std"][k]["mean"]) < 1e-15
        assert abs(statistics.stdev(values) - entry["mean_std"][k]["std"]) < 1e-15
    return summary, list(paths.values())


def dataset_table(output: Path, summary: dict, baseline: dict) -> list:
    """11방법의 데이터셋별 최종 성능 부록 표를 쓴다.

    데이터셋 안 양성 있는 시퀀스를 같은 가중으로 평균한 뒤 시드 평균·표본 SD를 낸다. 1·2위는 데이터셋·지표마다
    표시 값(백분율 소수 둘째 자리)으로 정하고, 표시 값이 같으면 같은 표시를 준다(dense 순위).
    """
    assert digest(DATASET_METHODS) == DATASET_METHODS_SHA256
    src = read(DATASET_METHODS)
    assert src["passed"] and all(v["passed"] for v in src["checks"]["main_table_reproduced"].values())
    assert src["checks"]["finite"] and src["checks"]["support_identical_across_methods"]
    assert src["input_sha256"][str(RINGPP_SUMMARY.relative_to(ROOT))] == ADOPTED_RINGPP["summary_sha256"]
    assert src["input_sha256"][str(FRESCO_SUMMARY.relative_to(ROOT))] == ADOPTED_FRESCO["summary_sha256"]
    assert src["input_sha256"][str(RINGPP_I2_SUMMARY.relative_to(ROOT))] == ADOPTED_RINGPP_I2["summary_sha256"]
    assert src["input_sha256"][str(DEEPRING_SUMMARY.relative_to(ROOT))] == ADOPTED_DEEPRING["summary_sha256"]
    assert src["checks"]["deepring_scaleinit_ledgers_match_adopted_hashes"] == 138
    assert src["checks"]["previous_ten_methods_identical"] and src["checks"]["dataset_support_sums_equal_totals"]
    for path, expected in src["input_sha256"].items():
        assert digest(ROOT / path) == expected, path
    order = (*NAMES, "nsd")
    # 행 순서는 표2와 같다.
    assert set(src["order"]) == set(order) and len(src["order"]) == 11 and "imlpr" not in src["order"]
    for name in NAMES:
        assert src["seeds"][name] == baseline["methods"][name]["seeds"]
    families = ("kitti", "kitti360", "helipr", "mulran", "nclt", "mcd")
    assert tuple(src["datasets"]) == families
    for family in families:
        d = src["datasets"][family]
        # NSD 값은 현재 요약의 시퀀스별 값에서 다시 계산해 대조한다.
        for k in METRICS:
            vals = [statistics.mean(summary["per_case"][f"diff_s{i}"]["sequences"][s][k] for s in d["positive_sequences"])
                    for i in (1, 2, 3)]
            assert abs(statistics.mean(vals) - d["methods"]["nsd"][k]["mean"]) < 1e-12
            assert abs(statistics.stdev(vals) - d["methods"]["nsd"][k]["sd"]) < 1e-12

    def label(name: str) -> str:
        return r"\textbf{NSD}" if name == "nsd" else NAMES[name].split("~")[0]

    def head(a: str, b: str) -> str:
        cells = []
        for family in (a, b):
            d = src["datasets"][family]
            cells.append(rf"\multicolumn{{4}}{{c}}{{{d['label']} ({len(d['positive_sequences'])}/{len(d['sequences'])})}}")
        return (" & " + " & ".join(cells) + r" \\ \cmidrule(lr){2-5}\cmidrule(l){6-9}"
                + r"Method & AP & Max F1 & R@100P & R@1 & AP & Max F1 & R@100P & R@1")

    rows = []
    pairs = (("kitti", "kitti360"), ("helipr", "mulran"), ("nclt", "mcd"))
    for p, (a, b) in enumerate(pairs):
        if p:
            rows.append(r"\midrule" + head(a, b) + r" \\" + "\n" + r"\midrule")
        cells = {name: [] for name in order}
        for family in (a, b):
            methods = src["datasets"][family]["methods"]
            for k in METRICS:
                shown = {name: round(100 * methods[name][k]["mean"], 2) for name in order}
                levels = sorted(set(shown.values()), reverse=True)
                for name in order:
                    cells[name].append(value_cell(methods[name][k]["mean"], methods[name][k]["sd"], levels.index(shown[name])))
        rows += [label(name) + " & " + " & ".join(cells[name]) + r" \\" for name in order]
    (output / "appdx_datasets.tex").write_text(table(
        r"Final performance by dataset (\%). Parentheses: sequences with positives/total sequences. Learned methods report mean $\pm$ SD "
        r"across three runs. Bold/underlined: best/second best within each dataset.",
        "tab:dataset_current", head(*pairs[0]), rows, "lcccccccc", wide=True, compact=True))
    return [DATASET_METHODS]


def baseline_variant_table(output: Path, baseline: dict) -> list:
    """표2의 RING++·SeqOT·FreSCo·DeepRING과 공개 코드 변형·이전 구성의 공통 시험 결과를 부록 표로 쓴다.

    모든 행은 표2와 같은 공통 시험 집계(195,726질의, 6도시 매크로, 학습 방법은 시드 평균·표본 SD)다.
    묶음마다 마지막 행이 표2 행이며, 표2와 같은 값·같은 표기여야 한다.
    """
    records = SEQOT / "server_mirror/records/summary.json"
    assert digest(COMMON_BASELINES) == ADOPTED_SEQOT["common_sha256"] and digest(records) == ADOPTED_SEQOT["records_sha256"]
    assert digest(FRESCO_PREFIX) == ADOPTED_FRESCO["previous_summary_sha256"]
    common, maxfix, prefix = read(COMMON_BASELINES), read(RINGPP_MAXFIX), read(FRESCO_PREFIX)
    maxfix_macro = maxfix["ringpp_maxfix"]["macro"]
    assert maxfix["checks_passed"] and (maxfix["ringpp_maxfix"]["n_queries"], maxfix["ringpp_maxfix"]["n_positive"]) == (195726, 61682)
    # 이전 구성 행은 표2 채택 요약이 기록한 대체 전 항목과 같아야 한다(정확max 공식 정규화 = I2 대체 전, FreSCo 지면 수정 전).
    previous = baseline["ringpp_i2_replacement"]["previous_ringpp_entry"]["mean_std"]
    assert all(abs(maxfix_macro[k] - previous[k]["mean"]) < 1e-12 for k in METRICS)
    assert all(prefix["methods"]["fresco"]["mean_std"][k]["mean"] == baseline["fresco_addition"]["previous_macro"][k] for k in METRICS)
    assert common["methods"]["ringpp"]["seeds"] == ["-"] and len(common["methods"]["seqot"]["seeds"]) == 3

    def entry(item: dict) -> tuple:
        return {k: item[k]["mean"] for k in METRICS}, {k: item[k]["std"] for k in METRICS}
    groups = (
        ("ringpp", "RING++", [
            ("Released kernel + released normalization", entry(common["methods"]["ringpp"]["mean_std"])),
            ("Exact max + released normalization", ({k: maxfix_macro[k] for k in METRICS}, dict.fromkeys(METRICS))),
            (r"Table~\ref{tab:results}: exact max + unit variance", entry(baseline["methods"]["ringpp"]["mean_std"]))]),
        ("seqot", "SeqOT", [
            ("Released-code training recipe", entry(common["methods"]["seqot"]["mean_std"])),
            (r"Table~\ref{tab:results}: paper recipe + log-domain GeM", entry(read(records)["mean_std"]))]),
        ("fresco", "FreSCo", [
            ("Before KITTI preprocessing regression fix", entry(prefix["methods"]["fresco"]["mean_std"])),
            (r"Table~\ref{tab:results}: KITTI regression fixed", entry(baseline["methods"]["fresco"]["mean_std"]))]),
        ("deepring", "DeepRING", [
            (r"Earlier $W_0=1$ initialization", entry(baseline["deepring_scaleinit_replacement"]["previous_deepring_entry"]["mean_std"])),
            (r"Table~\ref{tab:results}: $W_0=1/s$", entry(baseline["methods"]["deepring"]["mean_std"]))]),
    )
    rows = []
    for g, (key, method, configs) in enumerate(groups):
        if g:
            rows.append(r"\midrule")
        for i, (config, (mean, sd)) in enumerate(configs):
            rows.append((method if not i else "") + f" & {config} & "
                        + " & ".join(value_cell(mean[k], sd[k]) for k in METRICS) + r" \\")
        # 표2 행은 table2_results.tex와 같은 값·같은 표기(순위 표시 제외)다.
        mean, sd = configs[-1][1]
        table2 = baseline["methods"][key]["mean_std"]
        assert all(abs(mean[k] - table2[k]["mean"]) < 1e-12 for k in METRICS)
        assert ([value_cell(mean[k], sd[k]) for k in METRICS]
                == [value_cell(table2[k]["mean"], table2[k]["std"]) for k in METRICS]), key
    (output / "appdx_baseline_variants.tex").write_text(table(
        r"Implementation and training variants on the common test (\%). Learned methods report mean $\pm$ sample SD across three runs; "
        r"training-free methods report one run. The released RING++ kernel is nondeterministic.",
        "tab:baseline_release_comparison",
        r"Method & Configuration & AP & Max F1 & R@100P & \shortstack{Revisit\\R@1}", rows, "llrrrr", wide=True))
    return [COMMON_BASELINES, RINGPP_MAXFIX, records, FRESCO_PREFIX, DEEPRING_SUMMARY]


def threshold_tables(output: Path, summary: dict) -> list:
    """검증 고정 단일 기준값의 시험 결과 부록 표 3개를 쓴다.

    방법·시작점마다 검증 9시퀀스 합산에서 고른 τ(P99·P100·F1)를 시험 46시퀀스에 그대로 적용한 값이다.
    도시 재현율은 표2 집계다. 합산 정밀도·재현율은 시작점마다 46시퀀스의 적격 질의(5–25 m 모호 질의 제외)를 합쳐
    계산한 값의 세 시작점 평균이고, 개수도 세 시작점 평균이다.
    """
    paths = {name: THRESHOLD / name for name in THRESHOLD_SHA256}
    for name, path in paths.items():
        assert digest(path) == THRESHOLD_SHA256[name], path
    res, fa, freeze = read(paths["summary.json"]), read(paths["false_accepts.json"]), read(paths["freeze.json"])
    assert read(paths["verification.json"])["status"] == "passed" and freeze["test_files_opened"] == 0
    assert res["freeze_sha256"] == fa["freeze_sha256"] == THRESHOLD_SHA256["freeze.json"]
    assert fa["summary_sha256"] == THRESHOLD_SHA256["summary.json"]
    # NSD 시험 원장은 표2 원천과 같은 파일이고, 시퀀스별 양성 수가 표2 요약과 같아야 한다.
    for path, expected in res["test_ledgers_sha256"].items():
        if path.startswith(str(CURRENT.relative_to(ROOT))):
            assert digest(ROOT / path) == expected, path
    for i in (1, 2, 3):
        seqs = res["methods"]["nsd"]["P99"]["per_seed"][str(i)]["sequence"]
        assert all(seqs[s]["n_pos"] == summary["per_case"][f"diff_s{i}"]["sequences"][s]["n_pos"] for s in seqs)
    points = ("P99", "P100", "F1")
    methods = (("nsd", r"\textbf{NSD}"), ("bevplace", "BEVPlace++"))
    cities = {"karlsruhe": "Karlsruhe", "seoul": "Seoul", "daejeon": "Daejeon",
              "ann_arbor": "Ann Arbor", "stockholm": "Stockholm", "hamburg": "Hamburg"}
    bins = ("(5,7.5]", "(7.5,10]", "(10,25]", "(25,100]", ">100")

    def block(cells) -> list:
        rows = []
        for p, point in enumerate(points):
            for i, (method, label) in enumerate(methods):
                rows.append((r"\midrule " if p and not i else "") + (point if not i else "") + f" & {label} & "
                            + " & ".join(cells(res["methods"][method][point]["mean_sd"], fa["mean_over_seeds"][method][point])) + r" \\")
        return rows

    (output / "appdx_fixed_threshold.tex").write_text(table(
        r"\textbf{Validation-fixed thresholds.} One acceptance threshold per method and run is selected on validation "
        r"and applied unchanged to all test sequences. Means over three runs; city recall $\pm$ SD.",
        "tab:fixed_threshold_current",
        r"Point & Method & \shortstack{Val.\\prec./rec.} & \shortstack{City\\recall} & \shortstack{Pooled\\recall} & "
        r"\shortstack{Pooled\\prec.} & TP & FA & \shortstack{FA\\pos./neg.}",
        block(lambda a, _: [f"{100 * a['val_precision']['mean']:.2f} / {100 * a['val_recall']['mean']:.2f}",
                            f"{100 * a['city_macro_recall']['mean']:.2f}" + r"\textsubscript{$\pm$" + f"{100 * a['city_macro_recall']['sd']:.2f}" + "}",
                            f"{100 * a['recall']['mean']:.2f}", f"{100 * a['precision']['mean']:.2f}",
                            f"{a['TP']['mean']:,.0f}", f"{a['FP']['mean']:,.1f}",
                            f"{a['FP_positive_query']['mean']:,.1f} / {a['FP_negative_query']['mean']:,.1f}"]),
        "llccccrrc", wide=True))
    (output / "appdx_fixed_threshold_city.tex").write_text(table(
        r"City recall (\%) / false accepts at the thresholds of Table~\ref{tab:fixed_threshold_current}.",
        "tab:fixed_threshold_city", "Point & Method & " + " & ".join(cities.values()),
        block(lambda a, _: [f"{100 * a['city_recall'][c]['mean']:.1f} / {a['city_FP'][c]['mean']:,.1f}" for c in cities]),
        "llcccccc", wide=True))
    (output / "appdx_fixed_threshold_distance.tex").write_text(table(
        r"False accepts by distance to the returned place (m), positive / negative queries, at the thresholds of "
        r"Table~\ref{tab:fixed_threshold_current}.",
        "tab:fixed_threshold_distance", r"Point & Method & (5, 7.5] & (7.5, 10] & (10, 25] & (25, 100] & $>$100",
        # 오수락이 없는 구간·질의 종류는 기록에 키가 없으므로 0으로 쓴다.
        block(lambda _, f: [f"{f.get('positive_returned_bin', {}).get(b, 0):,.1f} / {f.get('negative_returned_bin', {}).get(b, 0):,.1f}" for b in bins]),
        "llccccc", wide=True))
    # 같은 오수락 수 비교(사용자 결정 2026-11-12): 시험 46시퀀스에 임계값 하나를 쓰고 합산 오수락 수를 맞춘다.
    matched_path = Path("/mnt/d/NSD_datasets/nsd_results/pdv2_matched_false_accepts_20261112/result.json")
    matched = read(matched_path)
    assert matched["status"] == "passed" and matched["reconciliation"]["status"] == "passed"
    assert matched["ledger_check"]["all_sha256_match"] and matched["threshold_selected_on"] == "test"
    rows = []
    for k in (0, 5, 10, 30, 100):
        c = matched["criteria"][f"FA<={k}"]
        cells = [str(k)]
        for method in ("nsd", "bevplace"):
            m = c["methods"][method]["mean_sd"]
            assert abs(m["FP"]["mean"] - k) < 1e-9
            cells += [f"{100 * m['city_macro_recall']['mean']:.2f}" + r"\textsubscript{$\pm$" + f"{100 * m['city_macro_recall']['sd']:.2f}" + "}",
                      f"{100 * m['recall']['mean']:.2f}", f"{m['TP']['mean']:,.0f}"]
        d, ci = c["difference_nsd_minus_bevplace"], c["intervals_95_pp"]["reselect"]["city_macro_recall"]
        cells.append(f"${d['city_macro_recall_pp']:+.2f}$ $[{ci[0]:+.2f}, {ci[1]:+.2f}]$")
        rows.append(" & ".join(cells) + r" \\")
    (output / "appdx_matched_false_accepts.tex").write_text(table(
        r"\textbf{Matched false-accept budgets.} One acceptance threshold per method and run is applied to all 46 test sequences "
        r"and set to the lowest value whose pooled false accepts do not exceed the budget. Means over three runs; city recall $\pm$ SD. "
        r"The last column is the city-recall difference (NSD minus BEVPlace++, points) with a paired 95\% interval that reselects the threshold in each resample.",
        "tab:matched_false_accepts",
        r"\shortstack{FA\\budget} & \shortstack{NSD\\city recall} & \shortstack{NSD\\pooled recall} & \shortstack{NSD\\TP} & "
        r"\shortstack{BEVPlace++\\city recall} & \shortstack{BEVPlace++\\pooled recall} & \shortstack{BEVPlace++\\TP} & Difference",
        rows, "rccrccrc", wide=True))
    return list(paths.values())   # matched_path는 저장소 밖(D 드라이브)이라 원천 해시 목록에 넣지 않는다


def geography_tables(output: Path) -> list:
    """정확한 분할 목록과 분할 쌍 지리 감사 부록 표 2개를 쓴다.

    목록은 시험 명세의 학습8·검증9·시험46과 같아야 한다. 감사 표는 비방향 쌍을 측정(공통 좌표, MulRan 정합 포함)·
    하한(HeLiPR Bridge)·도시 수준으로 나눠 그대로 옮기며, 측정하지 않은 쌍을 0%로 채우지 않는다.
    """
    paths = {name: GEOGRAPHY / name for name in GEOGRAPHY_SHA256}
    paths.update({"review_" + name: GEO_REVIEW / name for name in GEO_REVIEW_SHA256})
    for name, path in paths.items():
        assert digest(path) == {**GEOGRAPHY_SHA256, **{"review_" + k: v for k, v in GEO_REVIEW_SHA256.items()}}[name], path
    assert read(paths["review_verification.json"])["publication_blockers"] == []
    # 방향 가정 없는 Bridge 하한(소수 둘째 자리 반올림 값). 반올림 여유 0.005 km를 빼고 0.1 km 단위로 내려 표시한다.
    bridge_free = read(paths["review_bridge_rotation_free_check.json"])["per_sequence"]
    roster, audit = read(paths["roster.json"]), read(paths["geo_audit.json"])
    protocol = read(CURRENT / "protocol.json")
    assert roster["checks"]["all_pass"] and audit["_meta"]["roster_sha256"] == GEOGRAPHY_SHA256["roster.json"]
    assert read(paths["verification.json"])["status"] == "passed"
    rows = roster["roster"]
    korean = {r["key"]: r["split"] for r in rows if r["country"] == "South Korea" and r["site"] != "Bridge"}
    assert set(bridge_free) == set(korean)
    for split in ("train", "val", "test"):
        assert sorted(r["key"] for r in rows if r["split"] == split) == sorted(protocol[split])
    assert sum(r["ledger_queries"] for r in rows if r["split"] == "test") == 195726
    groups = {}
    for r in rows:
        groups.setdefault((r["split"], r["dataset"], r["site"]), []).append(r)
    names = {"train": "Training", "val": "Validation", "test": "Test"}
    lines, last = [], None
    for (split, dataset, site), members in groups.items():
        label = dataset if dataset in ("KITTI", "KITTI-360", "NCLT") else f"{dataset} ({site})"
        queries = f"{sum(m['ledger_queries'] for m in members):,}" if split == "test" else "--"
        lines.append((r"\midrule " if last and last != split else "") + f"{names[split]} & {label} & {members[0]['city']} & " + r"\raggedright "
                     + ", ".join(m["sequence"].replace("_", r"\_") for m in members)
                     + f" & {sum(m['cache_keyframes'] for m in members):,} & {queries}" + r" \\")
        last = split
    (output / "appdx_split_roster.tex").write_text(table(
        r"Exact sequence roster of the 8/9/46 split.", "tab:split_roster",
        r"Split & Dataset & City & Sequences & Keyframes & Queries", lines, r"lllp{0.43\textwidth}rr", wide=True, compact=True))
    lines = []
    for pair, label in (("train-val", "Training--validation"), ("train-test", "Training--test"), ("val-test", "Validation--test")):
        s = audit["summary"][pair]
        assert s["rule_pass"] and s["unverified_pairs"] == 0
        measured = sum(v for k, v in s["classes"].items() if k.startswith("measured"))
        registered = s["classes"].get("measured_registered", 0)
        bounded, city = s["classes"].get("bounded", 0), s["classes"].get("geographic_exclusion", 0)
        assert measured + bounded + city == s["unordered_pairs"]
        other = pair.split("-")[1]
        free = min((v["lb_any_route_point_km"] for k, v in bridge_free.items() if korean[k] == other), default=None)
        bound = f"{((free - 0.005) * 10 // 1) / 10:.1f}" if bounded else "--"
        lines.append(f"{label} & {s['unordered_pairs']} & {measured} ({registered}) & {100 * s['measured_max_frac_50m']:.2f} & "
                     f"{s['measured_min_dist_m'] / 1000:.2f} & {bounded} & {bound} & {city} & {int(s['geographic_min_lower_bound_km'] // 1):,}" + r" \\")
    (output / "appdx_split_geo.tex").write_text(table(
        r"Cross-split geographic audit over unordered sequence pairs. Measured: trajectory distances in a common frame (reg.: after registration "
        r"onto MulRan). Bounded and city level: conditional lower bounds, not trajectory measurements.", "tab:split_geo",
        r"& & \multicolumn{3}{c}{Measured} & \multicolumn{2}{c}{Bounded} & \multicolumn{2}{c}{City level} \\ "
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}"
        r"Split pair & Pairs & $n$ (reg.) & \shortstack{Within\\50\,m (\%)} & \shortstack{Min.\\dist. (km)} & $n$ & \shortstack{Bound\\(km)} & $n$ & \shortstack{Bound\\(km)}",
        lines, "lrrrrrrrr", wide=True))
    return list(paths.values())


def direction_tables(output: Path, summary: dict) -> list:
    """현재 모델의 방향별 성능·원시 회전·원인 대조·재표집 민감도 부록 표를 만든다.

    원천은 `results/pdv2_review_direction_yaw_20261105/`의 검증 완료 JSON이다(2026-11-06 원고 반영).
    """
    paths = {"direction": DIRECTION / "direction_summary.json",
             "sequence": DIRECTION / "direction_learning/sequence_summary.json",
             "block": DIRECTION / "direction_learning/summary.json",
             "six": DIRECTION / "direction_learning/fixed_city_block_summary.json",
             "yaw": DIRECTION / "expanded46/yaw_summary.json",
             "cause": DIRECTION / "expanded46/causes/summary.json",
             "complete": DIRECTION / "expanded46/completion_validation.json"}
    src = {k: read(p) for k, p in paths.items()}
    assert src["complete"]["status"] == src["yaw"]["status"] == src["cause"]["status"] == "passed"
    assert src["direction"]["original_totals_reproduced"]
    counts = src["direction"]["counts"]
    assert counts["queries"] == 195726 and src["yaw"]["counts"]["queries"] == 2944
    # 방향 전체 묶음의 학습 차이가 표4·표S1과 같은 원장임을 확인한다.
    for k in METRICS:
        delta = 100 * (summary["mean"]["convs_only"][k] - summary["mean"]["fixed"][k])
        assert abs(src["sequence"]["comparisons"]["convs_only-fixed"]["all"]["overall"][k]["delta_pp"] - delta) < 1e-9

    rows = []
    for group, label in (("forward_only", "Forward-only"), ("reverse_only", "Reverse-only"), ("mixed", "Mixed")):
        for method, name in (("diff", "NSD"), ("bev_native", "BEVPlace++")):
            m = src["direction"]["methods"][method]["mean_std"][group]
            rows.append(f"{label} & {name} & " + " & ".join(value_cell(m[k]["mean"]) for k in METRICS) + r" \\")
    (output / "appdx_direction.tex").write_text(table(
        rf"Final performance by traversal direction (\%). Forward-only, reverse-only, and mixed groups contain "
        rf"{counts['forward_only']:,}, {counts['reverse_only']:,}, and {counts['mixed']:,} of the {counts['positive']:,} positive queries. "
        r"Each group keeps the common negatives and excludes other positives. Means follow the Table~\ref{tab:results} aggregation; "
        r"the mixed group spans five cities.",
        "tab:direction", r"Group & Method & AP & Max F1 & R@100P & \shortstack{Revisit\\R@1}", rows, "llcccc", compact=True))

    rows = []
    procedures = (("Sequences", "sequence"), ("Seq.\\ + blocks", "block"), ("Blocks, 6 cities", "six"))
    for group, label in (("all", "All"), ("forward_only", "Forward-only"), ("reverse_only", "Reverse-only")):
        for name, key in procedures:
            r = src[key]["comparisons"]["convs_only-fixed"][group]["overall"]
            rows.append(f"{label} & DiffAttnConv learning & {name} & " + " & ".join(interval_cell(r[k]) for k in METRICS) + r" \\")
        r = src["sequence"]["comparisons"]["diff-fixed"][group]["overall"]
        rows.append(f"{label} & NSD & Sequences & " + " & ".join(interval_cell(r[k]) for k in METRICS) + r" \\")
    kept = [src["six"]["comparisons"]["convs_only-fixed"][g]["complete_city_replicates"] for g in ("all", "forward_only", "reverse_only")]
    (output / "appdx_direction_intervals.tex").write_text(table(
        r"Differences from uniform subtraction by traversal direction, with conditional 95\% intervals (percentage points). Sequence intervals use 10,000 resamples of "
        r"sequences with positives within cities, as in Table~\ref{tab:paired_current}. Block intervals use 2,000 resamples of all sequences in a city "
        r"and circular blocks of 100 consecutive queries, which also changes the sampled population and the handling of zero-positive strata. "
        rf"Six-city rows keep only replicates with positives in all six cities ({kept[0]:,}, {kept[1]:,}, and {kept[2]:,} for all, forward-only, "
        r"and reverse-only queries). Point estimates are the original six-city means.",
        "tab:direction_intervals", r"Group & Model & Resampling & AP & Max F1 & R@100P & Revisit R@1", rows, "lllcccc", wide=True, compact=True))

    rows = []
    for angle, result in src["yaw"]["final_recognition"].items():
        p = result["pooled_query_seed_pairs"]
        rows.append(f"${float(angle):g}^\\circ$ & " + " & ".join(value_cell(result["mean_std"][k]["mean"]) for k in METRICS)
                    + f" & {100 * p['winner_agreement'] / p['queries']:.2f}" + r" \\")
    (output / "appdx_yaw.tex").write_text(table(
        rf"Raw-point rotation of {src['yaw']['counts']['queries']:,} fixed queries (64 per sequence) with three starts (\%). "
        r"Gallery features and candidate-side temporal context stay fixed, so rotation changes coordinates rather than creating a new "
        r"reverse-view observation. Same returned place is measured over 8,832 query--start pairs per angle. The sample does not replace full-query accuracy.",
        "tab:yaw", r"Yaw & AP & Max F1 & R@100P & \shortstack{Revisit\\R@1} & \shortstack{Same\\returned\\place}", rows, "rccccc", compact=True))

    def magnitude(v: float) -> str:
        if v >= .001:
            return f"{v:.3f}"
        mantissa, exponent = f"{v:.2e}".split("e")
        return mantissa + r"\,$\times$\,10\textsuperscript{$-$" + str(abs(int(exponent))) + "}"
    arms = (("actual", "Actual raw rotation"), ("ideal_grid", "Ideal rasterized features"),
            ("ideal_transport", "Covariant transport phase"), ("ideal_both", "Both replacements"),
            ("ideal_both_exact_geometry", "Both + original edge geometry"), ("direct_output_phase", "Direct output-phase rotation"))
    rows = []
    for arm, label in arms:
        values = [src["cause"]["results"][a]["arms"][arm] for a in ("6.0", "90.0", "180.0")]
        largest = max(v["measurements"]["fixed_winner_confidence_change"]["max"] for v in values)
        rows.append(f"{label} & " + " & ".join(str(v["winner_changes"]) for v in values) + f" & {magnitude(largest)}" + r" \\")
    (output / "appdx_yaw_causes.tex").write_text(table(
        r"Mechanism controls on 736 queries (every fourth query of the rotation sample). Counts are returned-place changes out of 2,208 "
        r"query--start pairs per angle; the last column is the largest score change of the original $0^\circ$ returned place over the three angles. "
        r"Replacements isolate mechanisms and are not deployment variants.",
        "tab:yaw_causes", r"Condition & $6^\circ$ & $90^\circ$ & $180^\circ$ & Max $|\Delta S|$", rows, "lrrrr"))
    return list(paths.values())


def generate(output: Path) -> dict:
    """검산 뒤 주표와 보조표를 쓰고 원천 목록을 반환한다."""
    output.mkdir(parents=True, exist_ok=True)
    summary = read(CURRENT / "summary.json")
    report = read(CURRENT / "integrated_report.json")
    validation = read(CURRENT / "integrated_validation.json")
    protocol = read(CURRENT / "protocol.json")
    assert validation["status"] == "passed"
    assert validation["protocol_sha256"] == digest(CURRENT / "protocol.json")
    assert validation["queries"] == summary["queries_per_case"] == 195726
    assert protocol["selected_policy"] == "matching_only"
    for case in protocol["cases"].values():
        if case["checkpoint"]:
            assert digest(ROOT / case["checkpoint"]) == case["checkpoint_sha256"]
    for kind in VARIANTS:
        cases = ([kind] if kind in ("identity", "fixed")
                 else [f"{kind}_s{i}" for i in (1, 2, 3)])
        for metric in METRICS:
            vals = []
            for case in cases:
                item = summary["per_case"][case]
                macro = statistics.mean(x[metric] for x in item["groups"].values())
                assert abs(macro - item["macro"][metric]) < 1e-12
                vals.append(macro)
            assert abs(statistics.mean(vals) - summary["mean"][kind][metric]) < 1e-12
            if len(vals) > 1:
                assert abs(statistics.stdev(vals) - summary["std"][kind][metric]) < 1e-12

    seqot_sources = seqot_adoption()
    seqot_costs, seqot_cost_sources = seqot_cost_adoption()
    baseline, ringpp_costs, ringpp_sources = ringpp_adoption()
    baseline, fresco_costs, fresco_sources = fresco_adoption(baseline)
    # 표2·표S4·표S17의 RING++는 I2 인식·비용이다. 정확max I1(ringpp_costs)은 위에서 검산만 하고 표에 쓰지 않는다.
    baseline, ringpp_i2_costs, ringpp_i2_sources = ringpp_i2_adoption(baseline)
    # 표2·표S4의 DeepRING은 초기배율 인식 원장이다. 비용은 같은 구조·추론 연산의 기존 공통 실측(EFFICIENCY)을 그대로 쓴다.
    baseline, deepring_sources = deepring_adoption(baseline)
    rowspec = []
    for name, label in NAMES.items():
        item = baseline["methods"][name]
        means = {k: item["mean_std"][k]["mean"] for k in METRICS}
        stds = {k: item["mean_std"][k]["std"] for k in METRICS}
        rowspec.append((label, means, stds))
    for metric in METRICS:
        assert abs(baseline["methods"]["bevplace"]["mean_std"][metric]["mean"]
                   - summary["mean"]["bev_native"][metric]) < 1e-12
    rowspec.append((r"\textbf{NSD}", summary["mean"]["diff"], summary["std"]["diff"]))
    rankings = {k: sorted({x[1][k] for x in rowspec}, reverse=True) for k in METRICS}
    # 표2(주표)는 인식 성능과 비용을 한 표에 둔다(2026-11-06 표2·3 통합). 인식 칸은 여기서, 비용 칸은 아래 비용 블록에서 채운다.
    accuracy = {name: " & ".join(value_cell(mean[k], sd[k], rankings[k].index(mean[k])) for k in METRICS)
                for name, (label, mean, sd) in zip((*NAMES, "nsd"), rowspec)}
    labels = {name: label for name, (label, _, _) in zip((*NAMES, "nsd"), rowspec)}

    rows = []
    for kind, label in VARIANTS.items():
        sd = summary["std"][kind] if kind not in ("fixed", "identity") else dict.fromkeys(METRICS)
        rows.append(label + " & " + " & ".join(value_cell(summary["mean"][kind][k], sd[k])
                    for k in METRICS) + r" \\")
    (output / "appdx_stage2_controls.tex").write_text(table(
        r"Stage-2 controls. Every row keeps the indexes, candidate union, fusion weights, and scoring rule, and recomputes alignment, the returned place, and acceptance. Learned rows start from the same pretrained weights and identical initial outputs; $^\dagger$the pointwise gate starts from a separately pretrained model without message passing. Values in \%; trained variants report mean $\pm$ SD across three starts.",
        "tab:stage2_controls", r"Variant & AP & Max F1 & R@100P & \shortstack{Revisit\\R@1}", rows, compact=True))

    adopted_costs = {"ringpp": ringpp_i2_costs, "seqot": seqot_costs}
    measurements = [adopted_costs[m["method"]][10000] if m["method"] in adopted_costs else m
                    for m in read(EFFICIENCY)["measurements"] if m["n_map"] == 10000 and m["method"] in NAMES]
    # FreSCo는 공통 비용 원천에 없고, 같은 규약의 채택 측정만 쓴다.
    assert all(m["method"] != "fresco" for m in read(EFFICIENCY)["measurements"])
    measurements.append(fresco_costs[10000])
    assert len(measurements) == len(NAMES) == 10
    measured = read(CURRENT_EFFICIENCY)
    assert measured["selected"] == "diff/matching_only" and measured["n_map"] == 10000
    assert measured["verification"]["final_ledger"]["winners_exact"]
    assert measured["verification"]["final_ledger"]["queries"] == 288
    assert measured["variant"] == ADOPTED_COST["variant"] and digest(CURRENT_EFFICIENCY) == ADOPTED_COST["sha256"]
    assert measured["verification"]["published_measurement"]["winners_equal"]
    cost_protocol = CURRENT_EFFICIENCY.parent / f"protocol_{ADOPTED_COST['variant']}.json"
    assert digest(cost_protocol) == measured["method_protocol_sha256"]
    assert read(cost_protocol)["checkpoint"]["checkpoint_sha256"] == protocol["cases"]["diff_s1"]["checkpoint_sha256"]
    for path, expected in read(cost_protocol)["sources"].items():
        assert digest(ROOT / path) == expected, path
    measurements.append(measured)
    fields = {}
    for m in measurements:
        assert m["status"] == "passed" and (m["publish_ready"] or m is measured or m is ringpp_i2_costs[10000]
                                            or m is seqot_costs[10000] or m is fresco_costs[10000])
        fields[m["method"]] = [m["resident_bytes"]["total"] / (10095 * 1000),
                                 m["timings_ms"]["update"]["mean"],
                                 m["timings_ms"]["query"]["mean"],
                                 m["timings_ms"]["processing"]["mean"]]
        # 도착당 전체 처리 시간(processing)은 갱신과 질의 평균의 합과 같다.
        assert abs(fields[m["method"]][3] - fields[m["method"]][1] - fields[m["method"]][2]) < 1e-9
    # 재측정 전인 방법은 칸을 비우고 순위에서 뺀다(현재 없음).
    for name in COST_PENDING:
        fields.pop(name)
    ranking = [sorted({v[i] for v in fields.values()}) for i in range(4)]
    rows = []
    for name in (*NAMES, "nsd"):
        values = [accuracy[name]]
        if name in COST_PENDING:
            rows.append(labels[name] + " & " + " & ".join(values + ["--"] * 4) + r" \\")
            continue
        for i, v in enumerate(fields[name]):
            cell = f"{v:.2f}"
            rank = ranking[i].index(v)
            if rank < 2:
                cell = (r"\textbf{" if rank == 0 else r"\underline{") + cell + "}"
            values.append(cell)
        rows.append(labels[name] + " & " + " & ".join(values) + r" \\")
    # 부록: 지도 크기별 비용. 비교 방법은 같은 규약의 N=1,000·5,000·10,000 실측, NSD는 같은 구현(window)을 한 세션에서 잰 값.
    scaled = {m["method"]: {} for m in read(EFFICIENCY)["measurements"] if m["method"] in NAMES}
    for m in read(EFFICIENCY)["measurements"]:
        if m["method"] in NAMES:
            assert m["status"] == "passed" and m["publish_ready"]
            scaled[m["method"]][m["n_map"]] = m
    scaled["ringpp"] = ringpp_i2_costs
    scaled["seqot"] = seqot_costs
    scaled["fresco"] = fresco_costs
    scaled["nsd"] = {}
    for n in (1000, 5000, 10000):
        m = read(MAPSIZE / f"nsd_n{n}_window.json")
        assert m["status"] == "passed" and m["n_map"] == n and m["verification"]["final_ledger"]["winners_exact"]
        scaled["nsd"][n] = m
    assert scaled["nsd"][10000]["resident_bytes"]["total"] == measured["resident_bytes"]["total"]
    scale_rows = []
    for name in (*NAMES, "nsd"):
        if name in COST_PENDING:
            scale_rows.append(labels[name].split("~")[0] + " & " + " & ".join(["--"] * 7) + r" \\")
            continue
        runs = [scaled[name][n] for n in (1000, 5000, 10000)]
        update = [r["timings_ms"]["update"]["mean"] for r in runs]
        cells = [f"{sum(update) / 3:.2f} [{min(update):.2f}, {max(update):.2f}]"]
        cells += [f"{r['timings_ms']['query']['mean']:.2f}" for r in runs]
        cells += [f"{r['resident_bytes']['total'] / 1e6:.2f}" for r in runs]
        scale_rows.append(labels[name].split("~")[0] + " & " + " & ".join(cells) + r" \\")
    (output / "appdx_cost_scaling.tex").write_text(table(
        r"Cost versus initial gallery size $N$ on Sejong01 (96 arrivals, three repetitions). Update time is summarized as the mean [min, max] across the three workloads. "
        r"NSD values come from one session; its $N{=}10{,}000$ run differs from Table~\ref{tab:efficiency} by "
        + f"{100 * (scaled['nsd'][10000]['timings_ms']['processing']['mean'] / measured['timings_ms']['processing']['mean'] - 1):.1f}"
        + r"\% in processing time with identical resident memory.",
        "tab:cost_scaling",
        r"& & \multicolumn{3}{c}{Query (ms)} & \multicolumn{3}{c}{Resident memory (MB)} \\ \cmidrule(lr){3-5}\cmidrule(l){6-8}"
        r"Method & Update (ms) & 1k & 5k & 10k & 1k & 5k & 10k",
        scale_rows, "lccccccc", wide=True))

    stale = output / "table3_efficiency.tex"
    if stale.exists():
        stale.unlink()
    (output / "table2_results.tex").write_text(table(
        r"\textbf{Recognition performance and cost.} Recognition uses six-city macro means over the 46 test sequences; learned models report mean $\pm$ SD across three runs. "
        r"Cost is measured on Sejong01 with 10,000 initial eligible candidates. Bold marks the best values; underlining marks the second best.",
        r"tab:results}\label{tab:efficiency",
        r"& \multicolumn{4}{c}{Recognition (\%)} & \multicolumn{4}{c}{Cost} \\ \cmidrule(lr){2-5}\cmidrule(l){6-9}"
        r"Method & AP & Max F1 & R@100P & \shortstack{Revisit\\R@1} & \shortstack{Resident memory\\(KB/frame)} & \shortstack{Update\\(ms)} & \shortstack{Query\\(ms)} & \shortstack{Total\\(ms)}",
        rows, "lcccccccc", wide=True))

    rows = []
    # 2단계 제거(diff−identity)는 같은 절차로 계산된 방향별 학습 원천의 전체 묶음에서 가져온다.
    stage2 = read(DIRECTION / "direction_learning/sequence_summary.json")["comparisons"]
    for k in METRICS:
        # 같은 재표집 절차임을 학습 대조 구간의 일치(합산 순서 차이 1e-14 수준)로 확인한다.
        assert max(abs(x - y) for x, y in zip(stage2["convs_only-fixed"]["all"]["overall"][k]["ci95_pp"], report["comparisons"]["convs_only-fixed"]["conditional_95_pp"][k])) < 1e-9
        assert abs(stage2["diff-identity"]["all"]["overall"][k]["delta_pp"] - 100 * (summary["mean"]["diff"][k] - summary["mean"]["identity"][k])) < 1e-9
    rows.append("NSD minus no temporal spectrum refining & " + " & ".join(
        interval_cell(stage2["diff-identity"]["all"]["overall"][k]) for k in METRICS) + r" \\")
    labels = {"diff-fixed": "NSD minus uniform", "convs_only-fixed": "DiffAttnConv learning minus uniform",
              "diff-other_only": "NSD minus projection, norm, and gate learning", "diff-abs_kv": "NSD minus absolute K/V",
              "diff-point_gate": "NSD minus pointwise gate", "diff-bev_native": "NSD minus BEVPlace++"}
    for case, label in labels.items():
        a, b = case.split("-")
        cells = []
        for k in METRICS:
            delta = 100 * (summary["mean"][a][k] - summary["mean"][b][k])
            lo, hi = report["comparisons"][case]["conditional_95_pp"][k]
            cells.append(f"{signed(delta)} [{signed(lo)}, {signed(hi)}]")
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    macro = read(MACRO / "summary.json")
    for case, label in (("diff-no_alignment", "NSD minus no alignment re-ranking"),
                        ("diff-no_temporal", "NSD minus no temporal evidence")):
        comp = macro["comparisons"][case]
        rows.append(label + " & " + " & ".join(
            f"{signed(comp['delta_pp'][k])} [{signed(comp['conditional_95_pp'][k][0])}, {signed(comp['conditional_95_pp'][k][1])}]"
            for k in METRICS) + r" \\")
    # 두 자릿수 차이(정렬 재점수 제거)로 숫자 열이 넓어져, 열 간격 2pt와 이름 열 90pt(긴 이름은 두 줄)로 축소를 피한다.
    rows = [r"\raggedright " + r for r in rows]
    (output / "appdx_intervals.tex").write_text(table(
        "Paired differences and conditional 95\\% intervals (percentage points).",
        "tab:paired_current", "Comparison & AP & Max F1 & R@100P & Revisit R@1", rows, "p{90pt}cccc", wide=True, compact=True))

    rows = []
    cities = {"karlsruhe": "Karlsruhe", "seoul": "Seoul", "daejeon": "Daejeon",
              "ann_arbor": "Ann Arbor", "stockholm": "Stockholm", "hamburg": "Hamburg"}
    for city, label in cities.items():
        for kind, model in [("diff", "NSD"), ("bev_native", "BEVPlace++")]:
            means = [statistics.mean(summary["per_case"][f"{kind}_s{i}"]["groups"][city][k]
                                     for i in (1, 2, 3)) for k in METRICS]
            rows.append(label + " & " + model + " & " + " & ".join(f"{100 * x:.2f}" for x in means) + r" \\")
    (output / "appdx_city.tex").write_text(table(
        "City-level final performance (\\%), averaged across three runs.", "tab:city_current",
        "City & Method & AP & Max F1 & R@100P & Revisit R@1", rows, "llcccc"))

    rows = [r"\multicolumn{5}{@{}l}{\emph{DiffAttnConv learning minus uniform subtraction}} \\"]
    for city, label in cities.items():
        vals = report["comparisons"]["convs_only-fixed"]["city_delta_pp"][city]
        rows.append(label + " & " + " & ".join(signed(vals[k]) for k in METRICS) + r" \\")
    rows.append(r"\multicolumn{5}{@{}l}{\emph{NSD minus no temporal spectrum refining}} \\")
    for city, label in cities.items():
        vals = {k: 100 * statistics.mean(summary["per_case"][f"diff_s{i}"]["groups"][city][k] - summary["per_case"]["identity"]["groups"][city][k]
                                         for i in (1, 2, 3)) for k in METRICS}
        rows.append(label + " & " + " & ".join(signed(vals[k]) for k in METRICS) + r" \\")
    (output / "appdx_learning_city.tex").write_text(table(
        "Paired differences by city (percentage points): DiffAttnConv learning minus uniform subtraction, and NSD minus no temporal spectrum refining.",
        "tab:learning_city", "City & AP & Max F1 & R@100P & Revisit R@1", rows))

    dataset_sources = dataset_table(output, summary, baseline)
    variant_sources = baseline_variant_table(output, baseline)
    threshold_sources = threshold_tables(output, summary)
    geography_sources = geography_tables(output)

    index = read(INDEX_BUDGET / "summary.json")
    index_check = read(INDEX_BUDGET / "integrated_validation.json")
    assert index["status"] == index_check["status"] == "passed"
    assert index_check["summary_sha256"] == digest(INDEX_BUDGET / "summary.json")
    assert index_check["protocol_sha256"] == digest(INDEX_BUDGET / "protocol.json")
    assert index_check["queries"] == 195726 and index_check["ledgers"] == 690
    assert index_check["exact_matched_counts"] and index_check["original_winners_exact"]
    index_labels = {
        "key_200": "Spectral key, top-200",
        "aux_200": "BEV magnitude, top-200",
        "dual_100_100": "Dual index, top-100 + top-100 (NSD)",
        "key_matched": r"Spectral key, top-$K_q$",
        "aux_matched": r"BEV magnitude, top-$K_q$",
    }
    rows = []
    for arm, label in index_labels.items():
        for metric in (*METRICS, "candidate_recall"):
            vals = [index["per_case"][f"{arm}_s{i}"]["macro"][metric] for i in (1, 2, 3)]
            assert abs(statistics.mean(vals) - index["mean"][arm][metric]) < 1e-12
            assert abs(statistics.stdev(vals) - index["std"][arm][metric]) < 1e-12
        cells = [f'{index["mean_candidates"][arm]:.1f}',
                 value_cell(index["mean"][arm]["candidate_recall"])]
        cells += [value_cell(index["mean"][arm][k], index["std"][arm][k]) for k in METRICS]
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    (output / "appdx_index_budget.tex").write_text(table(
        r"Candidate-retrieval ablation. Candidate recall and final metrics are in \%; final metrics use the Table~\ref{tab:results} aggregation and three starts.",
        "tab:index_budget", r"Proposal source & Mean $|\mathcal C_q|$ & Candidate recall & AP & Max F1 & R@100P & Revisit R@1",
        rows, "lrrrrrr", wide=True))

    # 표4: 단계별 거시 대조. 체크포인트·융합 가중치·점수 척도는 고정하고 한 단계씩 바꾼다(2026-11-06).
    assert macro["status"] == "passed" and macro["queries_per_case"] == 195726
    for k in METRICS:
        assert abs(macro["mean"]["diff"][k] - summary["mean"]["diff"][k]) < 1e-12
        # 색인 대조 실행의 두 색인 행은 같은 승자이며 AP가 점수 재계산 오차(2.2e-9)만큼 다르다.
        assert abs(index["mean"]["dual_100_100"][k] - summary["mean"]["diff"][k]) < 1e-6
    # 과거 전용 행은 질의 이후 키프레임을 쓰지 않는 무지연 시스템이다(사용자 결정 2026-11-12): 층마다 앞선 10키프레임(이웃 수는 NSD와 같음)으로
    # 정제하고 수락 증거도 앞선 10키프레임만 쓰며, 사전학습과 추가 학습을 모두 이 조건으로 한다.
    zero_lag = json.loads(Path("/mnt/d/NSD_datasets/nsd_results/pdv2_zero_lag_pretrain_20261112/p10/result.json").read_text())
    assert not zero_lag["test_used_for_selection"] and zero_lag["causality"]["winners_equal"]
    assert zero_lag["causality"]["max_plan_frame_minus_query"] == 0 and zero_lag["consistency_with_p1"]["max_gap_pp"] < 1e-9
    percent = zero_lag["groups"]["all"]["percent"]
    for k in METRICS:  # 같은 분석의 대칭 NSD는 표의 NSD 행과 같아야 한다
        assert abs(percent["symmetric_nsd"][k]["mean"] / 100 - summary["mean"]["diff"][k]) < 1e-9
    past_only = {stat: {k: percent["P10_past10_pretrain_zero_lag"][k][stat] / 100 for k in METRICS} for stat in ("mean", "sd")}
    stage_rows = [
        (r"\textbf{NSD}", summary["mean"]["diff"], summary["std"]["diff"]),
        (r"\emph{Place indexing}", None, None),
        ("No BEV magnitude index", index["mean"]["key_200"], index["std"]["key_200"]),
        ("No spectral key index", index["mean"]["aux_200"], index["std"]["aux_200"]),
        (r"\emph{Temporal spectrum refining}", None, None),
        ("No temporal spectrum refining", summary["mean"]["identity"], dict.fromkeys(METRICS)),
        ("Uniform subtraction", summary["mean"]["fixed"], dict.fromkeys(METRICS)),
        # DiffAttnConv만 학습한 대조는 학습 범위 대조이며, 캡션에 그 역할을 적어 2단계에 둔다(사용자 결정 2026-11-06).
        ("DiffAttnConv learning", summary["mean"]["convs_only"], summary["std"]["convs_only"]),
        (r"\emph{Matching and acceptance}", None, None),
        ("No alignment re-ranking", macro["mean"]["no_alignment"], macro["std"]["no_alignment"]),
        ("No temporal evidence", macro["mean"]["no_temporal"], macro["std"]["no_temporal"]),
        (r"\emph{No keyframes after the query}", None, None),
        ("Past-only NSD", past_only["mean"], past_only["sd"]),
    ]
    # 이름 열을 83.5pt로 고정해 가장 긴 이름(101pt)만 두 줄로 접고 숫자는 축소 없이 8pt를 유지한다(나머지 이름 최대 83.2pt).
    # 단계 묶음 사이에 가로선을 둔다(사용자 지시 2026-11-06).
    rows = [(rf"\midrule\multicolumn{{5}}{{@{{}}l}}{{{label}}} \\" if mean is None else
             r"\raggedright " + label + " & " + " & ".join(value_cell(mean[k], sd[k]) for k in METRICS) + r" \\")
            for label, mean, sd in stage_rows]
    (output / "table4_components.tex").write_text(table(
        r"\textbf{Component ablation.} Rows change one component of the trained NSD; DiffAttnConv learning and Past-only NSD are trained separately. Values are in \%; trained variants report mean $\pm$ SD across three starts.",
        "tab:online_components", r"Variant & AP & Max F1 & R@100P & \shortstack{Revisit\\R@1}", rows, "p{83.5pt}cccc", compact=True))

    # 부록: 크기 점수만 쓸 때의 반환 오류 분해(사용자 지시 2026-11-10). 표4 '정렬 재점수 제거'와 표2 NSD 원장에서 직접 계산한다.
    from _pdv2_aliasing_decomposition import city_rates, transitions
    aliasing = city_rates(transitions())
    assert abs(aliasing["macro"]["magnitude_error"] - 100 * (1 - macro["mean"]["no_alignment"]["R@1"])) < 1e-9
    assert abs(aliasing["macro"]["nsd_error"] - 100 * (1 - summary["mean"]["diff"]["R@1"])) < 1e-9
    alias_keys = ("magnitude_error", "magnitude_distant", "nsd_error", "nsd_distant")
    alias_cities = {"karlsruhe": "Karlsruhe", "seoul": "Seoul", "daejeon": "Daejeon",
                    "ann_arbor": "Ann Arbor", "stockholm": "Stockholm", "hamburg": "Hamburg"}
    rows = [label + " & " + " & ".join(f"{aliasing[c][k]:.2f}" for k in alias_keys) + r" \\"
            for c, label in alias_cities.items()]
    rows.append(r"\midrule City macro & " + " & ".join(f"{aliasing['macro'][k]:.2f}" for k in alias_keys) + r" \\")
    (output / "appdx_aliasing.tex").write_text(table(
        r"Returned-place errors among revisit queries (\%). \emph{Magnitude scores only} returns the place with the two index scores, as in the no-alignment row of Table~\ref{tab:online_components}; NSD adds alignment re-ranking. An error returns a place more than 5\,m from the query, and a distant error one at least 25\,m away. Rates average the revisit-bearing sequences of each city, then the three starts.",
        "tab:aliasing", r"& \multicolumn{2}{c}{Magnitude scores only} & \multicolumn{2}{c}{NSD} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5} City & Error & Distant error & Error & Distant error",
        rows, "lcccc"))

    direction_sources = direction_tables(output, summary)

    sources = [CURRENT / (n + ".json") for n in
               ("protocol", "summary", "integrated_report", "integrated_validation")]
    sources += [BASELINES, EFFICIENCY, CURRENT_EFFICIENCY, cost_protocol, Path(__file__)] + seqot_sources + seqot_cost_sources + ringpp_sources
    sources += fresco_sources + ringpp_i2_sources + deepring_sources
    sources += [INDEX_BUDGET / (n + ".json") for n in
                ("protocol", "summary", "integrated_validation")]
    sources += direction_sources + dataset_sources + variant_sources + threshold_sources + geography_sources
    sources += [MACRO / "summary.json", MACRO / "protocol.json"]
    sources += [ROOT / "scripts/_pdv2_aliasing_decomposition.py"]
    sources += [MAPSIZE / f"nsd_n{n}_window.json" for n in (1000, 5000, 10000)] + [MAPSIZE / "protocol_window.json"]
    manifest = {"model": "diff", "policy": "matching_only", "queries": 195726,
                "aggregation": summary["aggregation"], "table3_nsd": "measured_diff_s1_window", "table3_nsd_adoption": ADOPTED_COST,
                "table2_ringpp": "exact_max_i2_normalization_recognition_and_cost", "table2_ringpp_adoption": ADOPTED_RINGPP_I2,
                "ringpp_official_i1_retained": ADOPTED_RINGPP,
                "table2_deepring": "eq9_w0_scale_initialization_recognition_existing_same_implementation_cost",
                "table2_deepring_adoption": ADOPTED_DEEPRING,
                "table2_seqot": "paper_recipe_stable_gem_recognition_and_cost", "table2_seqot_adoption": ADOPTED_SEQOT,
                "table2_seqot_cost_adoption": ADOPTED_SEQOT_COST,
                "table2_fresco": "official_untrained_single_run_kitti_groundfix_recognition_and_existing_cost", "table2_fresco_adoption": ADOPTED_FRESCO,
                "cost_pending": list(COST_PENDING),
                "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "outputs": {p.name: digest(p) for p in output.glob("*.tex")}}
    (output / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    """출력 폴더를 받아 검산한 표를 생성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/paper/body_v2.3/tables_v3")
    args = parser.parse_args()
    result = generate(args.output_dir)
    print(json.dumps({"tables": len(result["outputs"]), "queries": result["queries"], "status": "passed"}))


if __name__ == "__main__":
    main()
