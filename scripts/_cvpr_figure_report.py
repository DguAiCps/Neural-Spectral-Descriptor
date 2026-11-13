#!/usr/bin/env python3
"""원문 그림을 직접 확인한 주석으로 CVPR 참고 자료 HTML과 CSV를 만든다."""
from __future__ import annotations

import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/related works/CVPR_2024-2026_NSD"

# 번호, PDF 쪽(1부터), 폭, 주된 역할, 직접 확인한 내용, NSD에 참고할 점.
ANNOTATIONS = {
    "BoQ": {
        "task": "영상 장소 인식", "relevance": "attention 설명 방식",
        "figures": [
            (3, 8, "1col", "attention", "세 입력 장면과 네 학습 질의의 cross-attention을 격자로 보여준다. 질의에 따라 작은 구조와 넓은 영역을 다르게 본다.", "문맥 attention이 어떤 이웃에 반응하는지 대응시켜 설명할 수 있다."),
        ],
        "other": "Fig. 1(p.1)은 여러 데이터셋의 R@1 요약인 도입부 그림이다. 구조도는 Fig. 2다.",
    },
    "CricaVPR": {
        "task": "영상 장소 인식", "relevance": "문맥·상관 학습과 정성 비교",
        "figures": [
            (5, 6, "2col", "검색 사례", "질의와 여러 방법의 검색 결과를 배열한다. 시점·조명 변화와 비슷하게 생긴 다른 장소에서 정답과 오답을 색으로 구별한다.", "같은 질의를 사용하는 NSD·비교 방법의 성공과 실패 사례에 적합하다."),
            (6, 7, "1col", "attention", "사전 학습 DINOv2와 적응 후 특징 지도를 비교한다. 정적인 식별 구조에 반응하는지 보여준다.", "개선 전후 활성 지도를 동일한 입력과 색 범위로 비교한다."),
        ],
        "other": "Fig. 1(p.1)의 R@1–디스크립터 차원 그래프는 도입부 요약이다. 실험 결과 그림 수에 넣지 않았다.",
    },
    "DiffLoc": {
        "task": "LiDAR 절대 위치·자세 추정", "relevance": "궤적 오차와 불확실성 설명",
        "figures": [
            (4, 6, "2col", "궤적", "Oxford에서 네 방법의 예측 궤적을 정답 궤적과 겹쳐 비교한다.", "검색 오답의 공간적 위치를 지도 위에 표시하는 방식에 참고한다."),
            (5, 7, "2col", "궤적", "NCLT에서 네 방법의 예측 궤적과 정답을 비교한다.", "데이터셋별 오류 패턴이 다르면 같은 형식의 패널로 비교한다."),
            (6, 7, "1col", "신뢰도", "프레임 순서에 따른 위치 오차와 예측 분산을 두 세로축으로 표시한다.", "오답 구간에서 최종 신뢰도가 실제로 낮아지는지 보여주는 데 적합하다."),
        ],
        "other": "Fig. 3(p.5)은 불확실성 산출 절차도다. 검색 성능과 절대 자세 추정 오차는 다른 평가 대상이다.",
    },
    "LiSA": {
        "task": "의미 정보를 사용하는 LiDAR 위치 추정", "relevance": "특징 활성과 구성요소 효과 설명",
        "figures": [
            (3, 6, "2col", "궤적", "Oxford에서 여덟 방법의 예측 궤적을 정답과 비교한다.", "오류가 집중되는 장소를 비교하는 배치를 참고한다."),
            (4, 7, "2col", "궤적", "NCLT에서 여덟 방법의 예측 궤적을 비교한다.", "다른 환경에서도 같은 시각화 규칙을 적용한다."),
            (5, 8, "2col", "attention", "의미 정보 유무에 따른 점군 활성값을 비교하고 줄기·건물·주차 차량·보행자를 확대한다.", "문맥 전후 특징 차이를 실제 구조에 연결해 보여주는 예시다."),
            (6, 8, "1col", "attention", "의미 정보 유무에 따른 활성값 분포와 위치 추정 사례를 함께 보여준다.", "활성값 변화와 최종 결과를 연결해야 한다는 점을 참고한다."),
            (7, 8, "1col", "attention", "SPVNAS와 SphereFormer의 의미 분할 결과를 비교한다.", "보조 모듈의 출력 품질을 따로 검증하는 형식이다."),
        ],
        "other": "Fig. 1(p.1)은 도입부 궤적 예시다. 공식 부록은 ZIP이며 여기서는 링크만 보존한다.",
    },
    "SALAD": {
        "task": "영상 장소 인식", "relevance": "특징 집계의 작동 원인과 실패 사례",
        "figures": [
            (3, 7, "2col", "attention", "입력 영상과 국소 특징의 중요도 지도를 비교한다. 버려지는 특징과 집계에 사용되는 특징을 살펴본다.", "어떤 공간 정보가 최종 표현에 남는지 시각화하는 데 적합하다."),
            (4, 8, "2col", "특징 대응", "두 시점의 대응 패치와 클러스터 할당 히스토그램을 비교한다.", "같은 장소의 다른 시점에서 표현이 일관적인지 점검한다."),
            (5, 8, "2col", "검색 사례", "질의별 상위 세 검색 결과를 정답·오답 색으로 구분한다. 마지막 행에는 실패 사례를 포함한다.", "성공뿐 아니라 현재 남아 있는 실패 유형도 함께 제시한다."),
        ],
        "other": "실험의 introspective results 부분에 해당한다. Fig. 1–2는 개념·구조 설명이다.",
    },
    "SUE": {
        "task": "영상 검색 결과의 불확실성 추정", "relevance": "최종 승자 고정 후 신뢰도 보정과 가장 가까운 설명 방식",
        "figures": [
            (3, 7, "1col", "신뢰도", "L2·STUN·SUE별로 가장 확신하는 질의와 가장 불확실한 질의 및 검색 결과를 비교한다.", "높은 점수의 오답을 보정이 얼마나 낮추는지 사례로 보여준다."),
            (4, 7, "1col", "신뢰도", "두 질의에서 주변 검색 후보를 함께 보여주며 정답과 오답의 신뢰도 판단 근거를 설명한다.", "단일 쌍 점수 외에 주변 후보 분포가 주는 정보를 설명한다."),
            (5, 8, "2col", "신뢰도", "기하 검증 불확실성과 각 방법의 불확실성을 두 축에 놓고 정답·오답 및 결합 분류 경계를 표시한다.", "원 점수와 보정 점수의 상호 보완성을 산점도로 설명하기 좋다."),
            (6, 8, "1col", "신뢰도", "기하 검증은 낮은 신뢰도를 주지만 SUE는 높은 신뢰도를 주는 올바른 검색 사례를 보여준다.", "보정이 유용한 조건과 예외를 분리해 보여준다."),
            (7, 8, "1col", "민감도", "K와 λ를 바꾸며 여러 데이터셋의 AUC-PR 변화를 두 패널로 비교한다.", "새 설정 선택이 필요하다면 검증셋에서 안정적인 구간을 보여준다."),
        ],
        "other": "Fig. 1(p.1)의 PR 곡선은 도입부 요약이다. NSD와 모델·센서·평가 규약이 같다는 뜻은 아니다.",
    },
    "TransLoc4D": {
        "task": "4D 레이더 장소 인식", "relevance": "시간 이웃과 동적 이상점 처리 설명",
        "figures": [
            (5, 8, "2col", "검색 사례", "야간·비 조건의 질의와 주간 참조 영상, 원래 레이더 점군과 정제 점군, 시간 이웃을 함께 보여준다.", "환경 변화와 시간 이웃의 역할을 한 사례 안에서 설명할 수 있다."),
            (6, 8, "1col", "정제 결과", "원시 점군·절대 속도·필터링 결과를 비교해 동적 이상점 제거를 보여준다.", "전처리 효과를 최종 검색 효과와 구분해 시각화한다."),
        ],
        "other": "Fig. 4(p.6)는 데이터셋·장비 설명이다. 레이더 입력은 LiDAR 입력과 다르다.",
    },
    "AGPlace": {
        "task": "항공–지상 다중 모달 장소 인식", "relevance": "시간 순서에 따른 표현의 연속성",
        "figures": [
            (6, 8, "1col", "표현 분포", "연속 궤적의 디스크립터를 t-SNE로 나타내고 프레임 순서를 색으로 표시한다. ODE 유무를 비교한다.", "시간 문맥이 표현을 어떻게 바꾸는지 탐색하는 보조 그림에 적합하다."),
            (7, 8, "1col", "attention", "항공·지상 입력의 중요도 지도로 구조적 랜드마크에 대한 반응을 보여준다.", "서로 다른 시점에서 공통 구조를 보는지 확인한다."),
        ],
        "other": "Fig. 3–4는 방법 설명의 거리·궤적 시각화이며 Fig. 5는 데이터 예시다. t-SNE만으로 성능 기여를 입증할 수 없다.",
    },
    "ForestLPR": {
        "task": "숲 환경의 LiDAR 장소 인식", "relevance": "NSD와 직접 가까운 검색 과제",
        "figures": [
            (5, 8, "1col", "조건별 성능", "ANYmal과 Karawatha04에서 정답 판정 거리 임계값에 따른 R@1을 두 패널로 비교한다.", "거리 허용 오차에 따른 민감도를 참고한다. 이 논문은 임계값마다 양성 질의 집합도 달라 단조 증가가 보장되지 않는다."),
            (6, 8, "1col", "attention", "높이별 BEV 패치 attention을 점군 정면도에 투영한다.", "BEV의 추상적인 가중치를 실제 3D 구조에 연결하는 방식이다."),
        ],
        "other": "Fig. 4(p.5)는 데이터셋·장비 설명이고 Fig. 1(p.1)은 attention 도입부 그림이다. 공식 부록 ZIP은 링크만 보존한다.",
    },
    "HOTFormerLoc": {
        "task": "지상·항공 LiDAR 장소 인식", "relevance": "NSD와 직접 가까운 검색 과제",
        "figures": [
            (5, 7, "1col", "검색 곡선", "CS-Wild-Places의 Baseline과 Unseen 조건에서 검색 후보 수 N에 따른 Recall@N을 비교한다.", "후보 검색 능력을 보여준다. NSD 최종 수락의 AP·R@100P를 대체하지 않는다."),
        ],
        "other": "Fig. 4(p.6)는 항공 지도와 지상 경로·서브맵 설명이다. Fig. 1(p.1)의 성능 요약은 도입부에 있다. 다른 정량 비교는 주로 표다.",
    },
    "LightLoc": {
        "task": "빠른 학습을 위한 LiDAR 위치 추정", "relevance": "최종 시스템에서 신뢰도 활용",
        "figures": [
            (3, 7, "1col", "신뢰도", "프레임별 위치 오차와 신뢰도, LOAM·융합 결과·정답의 궤적을 두 패널에 보여준다.", "점수 품질이 후단 시스템의 실패 억제로 연결되는지를 설명한다."),
        ],
        "other": "Fig. 1(p.1)은 위치 오차–학습 시간 요약이다. 추론 지연 그래프로 해석하면 안 된다.",
    },
    "BEV-SLD": {
        "task": "LiDAR BEV 랜드마크 기반 전역 위치 추정", "relevance": "환경별 실패 지도와 거리 조건 분석",
        "figures": [
            (5, 6, "2col", "성공 위치 지도", "참조·시험 경로 위에서 여러 방법이 올바르게 위치를 추정한 지점을 표시한다.", "NSD와 BEVPlace++의 오수락·성공 위치 차이를 지도에서 비교한다."),
            (6, 7, "1col", "attention", "보지 못한 장면에서 BEV 입력·랜드마크 열지도·국소 최댓값을 보여준다.", "중간 출력의 전이와 전체 시스템의 새 장소 일반화를 구분해야 한다."),
            (7, 8, "1col", "조건별 성능", "MCD·NCLT·Wild-Places에서 최근접 참조 자세까지의 거리를 3m 구간으로 나누고 성공률과 표본 수를 함께 표시한다.", "지도 경로에서 떨어진 정도별 난이도를 분석한다. 정답 판정 거리 임계값 sweep과 다르다."),
        ],
        "other": "추가로 Fig. 3(p.4)의 초기·학습 후 랜드마크 비교는 방법 절에 있지만 §4.4가 분석에 다시 사용한다. 주목록 43개와 별도 표시한다. 이 방법의 장면별 학습을 NSD의 장소 분리 일반화와 동일시하지 않는다.",
        "extra": [(3, 4, "방법 절의 진단 그림을 실험에서 다시 사용: 초기·학습 후 랜드마크")],
    },
    "C-LaV": {
        "task": "악천후 LiDAR 장소 인식", "relevance": "NSD와 직접 가까운 검색 과제",
        "figures": [
            (6, 7, "1col", "성능·표현 분포", "잠재 특징 정제 전후의 PR 곡선과 noisy·denoised·database 특징의 공동 t-SNE를 함께 보여준다.", "최종 성능 변화와 표현 변화를 같은 그림에서 연결하되 투영 그림만으로 원인을 단정하지 않는다."),
        ],
        "other": "Fig. 4(p.6)는 경로 구성, Fig. 5(p.6)는 맑은 날·악천후 BEV 채널 예시다. Fig. 1(p.1)의 성능 레이더 차트는 도입부 요약이다.",
    },
    "EfficientVPR": {
        "task": "효율적인 영상 장소 인식", "relevance": "정확도와 연산비용을 함께 설명하는 구성",
        "figures": [
            (5, 5, "1col", "검색 사례", "시점·개보수·조명·가림 등 어려운 질의에서 여러 방법의 검색 결과를 비교한다.", "대표적인 혼동 유형을 패널별로 분리해 설명한다."),
            (6, 8, "1col", "attention", "여러 미세 조정 방식의 특징 열지도와 실제 검색 결과를 짝지어 보여준다.", "attention 그림 옆에 최종 결과를 붙여 모듈 설명과 성능을 연결한다."),
            (7, 8, "1col", "비용 분해", "모듈별 파라미터 수와 지연을 두 누적 가로 막대로 표시한다. 같은 하드웨어에서 다섯 번 반복하며 지연 축 단위는 0.1ms다.", "NSD·BEVPlace++의 인코딩·갱신·최종 판정 시간을 같은 측정 범위로 나누는 데 적합하다."),
        ],
        "other": "Fig. 1(p.1)의 평균 R@1–차원 그래프는 도입부다. Fig. 5는 배치상 이른 페이지에 있지만 정성 검색 결과이므로 주목록에 포함했다.",
    },
    "HypeVPR": {
        "task": "원근 영상–전방위 영상 장소 인식", "relevance": "구조화된 표현과 검색 사례 설명",
        "figures": [
            (3, 7, "1col", "검색 사례", "두 방법의 상위 세 파노라마 결과에서 정답·오답과 대응하는 국소 창을 표시한다.", "검색 결과뿐 아니라 어디가 일치하는지 함께 보여준다."),
            (4, 8, "1col", "표현 분포", "1,000개 계층적 표현을 Poincaré ball 위에 나타내고 계층별 색을 다르게 한다.", "표현 공간의 의미를 보이는 그림이며 t-SNE 그림이 아니다."),
        ],
        "other": "Fig. 1(d)(p.1)는 도입부 효율 요약이다. 비용 표의 CPU 매칭 시간은 인코딩을 제외하므로 NSD 전체 처리 시간과 바로 비교할 수 없다.",
    },
    "LEADER": {
        "task": "신뢰할 수 있는 대응점 기반 LiDAR 재위치 추정", "relevance": "신뢰도와 실제 오류의 관계",
        "figures": [
            (4, 5, "2col", "궤적", "Oxford에서 방법별 예측 궤적과 정답 궤적을 비교한다.", "공간적인 실패 구간을 보여주는 배치를 참고한다."),
            (5, 6, "2col", "궤적", "NCLT에서 예측 궤적과 정답 궤적을 비교한다.", "데이터셋이 달라도 같은 축·색 규칙을 유지한다."),
            (7, 7, "1col", "오차 분포", "여러 방법의 위치 오차 누적분포(CDF)를 비교한다.", "평균 수치가 감추는 큰 오류의 꼬리를 보여준다."),
            (8, 7, "1col", "오차 분포", "TRR 유무에 따른 대응점·장면점 오차 분포를 로그 축과 작은 오차 확대 창으로 비교한다.", "어떤 오류 범위를 보정이 줄이는지 비교하는 형식이다."),
            (9, 7, "1col", "신뢰도", "신뢰도 순위 구간별 평균 대응 오차를 표시한다. 낮은 신뢰도 쪽으로 갈수록 오차가 커지는지 확인한다.", "NSD 최종 점수 구간별 실제 오답률을 확인하는 데 직접 참고할 수 있다."),
        ],
        "other": "Fig. 6(p.6)는 학습·시험 궤적 구성이다. Fig. 1(p.1)은 평균 위치 오차의 도입부 요약이다.",
    },
    "TACO": {
        "task": "LiDAR 위치 추정과 3D 객체 검출의 공동 학습", "relevance": "궤적 오차와 다중 과제의 정성 평가",
        "figures": [
            (5, 7, "2col", "궤적", "네 방법의 예측 궤적을 비교하고 오류가 큰 부분을 확대한다.", "실패가 집중되는 구간은 확대 창으로 보여준다."),
            (6, 7, "1col", "다른 과제", "기준 방법과 제안 방법의 3D 객체 검출 상자를 비교한다.", "객체 검출을 하지 않는 NSD에는 직접 대응하지 않는다."),
        ],
        "other": "Fig. 2–3(p.4)은 데이터 분포와 라벨 예시다. 이름이 같은 CVPR 2024의 다른 TACO 논문과 구별한다.",
    },
}

RECOMMENDATIONS = [
    ("우선 1 · 최종 점수 보정이 왜 도움이 되는가", "SUE Fig. 5, LEADER Fig. 9, C-LaV Fig. 6",
     "1col 두 패널을 권한다. (a) 원 점수와 보정 점수의 산점도에 정답·오답을 표시한다. (b) 최종 혼합 전후의 고정밀 PR 곡선 또는 점수 순위 구간별 오답률을 표시한다. 상단에 전체 질의·시퀀스 범위와 집계 방식을 적는다.",
     "최종 승자를 고정한 점수 보정이면 검색 후보가 바뀌었다고 설명하지 않는다. 목적은 AP·R@100P 변화의 근거다. 이 그림만으로 DiffAttnConv 자체의 기여가 입증되지는 않는다."),
    ("우선 2 · 혼동 사례와 남아 있는 실패", "SALAD Fig. 5, CricaVPR Fig. 5, TransLoc4D Fig. 5",
     "질의·정답 장소·비슷하게 생긴 오답 장소를 BEV 또는 거리 영상으로 나란히 놓고, 짧은 시간 이웃과 보정 전후 점수를 붙인다. 개선 사례와 여전히 실패하는 사례를 모두 담는다. 여러 사례라면 2col이 읽기 쉽다.",
     "대표 사례의 선택 기준을 명시한다. 같은 질의에서 NSD와 BEVPlace++를 비교하며 그림에 들어갈 사례를 골랐다는 사실을 숨기지 않는다. 그림이 전체 평균을 대신하지 않는다."),
    ("선택 · 처리 단계별 시간 분해", "EfficientVPR Fig. 7",
     "현재 효율성 Fig. 4가 정확도와 비용의 관계를 이미 보여주므로 추가한다면 인코딩·상태 갱신·최종 검색 및 판정의 누적 막대가 덜 중복된다. Table 3의 동일 하드웨어·지도 크기·질의 범위를 사용한다.",
     "메모리의 특징 크기·상주 수치 상태·프로세스 전체 사용량을 섞지 않는다. 전체 속도 우위와 검색 단계만의 속도 우위를 구분한다."),
    ("문맥 기여를 직접 주장할 때 · attention과 최종 결과 연결", "BoQ Fig. 3, ForestLPR Fig. 6, EfficientVPR Fig. 6",
     "질의별 시간 이웃 가중치와 최종 매칭 결과를 짝지어 제시한다. 같은 시간 이웃과 학습 조건에서 문맥 없음·표준 attention·DiffAttnConv를 비교하는 최종 성능 대조가 먼저 필요하다.",
     "예쁜 열지도나 t-SNE만으로 문맥의 성능 기여·인과성을 주장하지 않는다. 새 설정은 학습·검증에서만 선택하고, 고정된 시험 출력은 분석·보고에 사용한다."),
]


def escape(value: object) -> str:
    """HTML 본문과 속성에 쓸 문자열을 이스케이프한다."""
    return html.escape(str(value), quote=True)


def main() -> None:
    """시각 확인 주석을 출처와 결합하고 링크·파일 무결성을 검사한다."""
    manifest = json.loads((OUT / "manifest.json").read_text())
    inventory = json.loads((OUT / "figure_inventory.json").read_text())
    lookup = {p["paper"]: {f["figure"]: f for f in p["figures"]} for p in inventory}
    records, rows = [], []
    for paper in manifest["papers"]:
        key = f"{paper['year']}_{paper['short']}"
        annotation = ANNOTATIONS[paper["short"]]
        figures = []
        for number, page, width, category, description, lesson in annotation["figures"]:
            original = lookup[key][number]
            assert original["pdf_page"] == page, (key, number, page, original["pdf_page"])
            figure = dict(figure=number, pdf_page=page, width=width, category=category,
                          description_ko=description, nsd_lesson_ko=lesson,
                          preview=original["preview"], visually_checked=True)
            figures.append(figure)
            rows.append(dict(year=paper["year"], paper=paper["short"], title=paper["title"],
                             task_ko=annotation["task"], **figure, official_url=paper["url"],
                             local_pdf=f"papers/{key}/main.pdf"))
        records.append(dict(paper=key, short=paper["short"], year=paper["year"],
                            task_ko=annotation["task"], relevance_ko=annotation["relevance"],
                            other_figures_ko=annotation["other"], figures=figures,
                            extra_method_figures=annotation.get("extra", [])))
    assert len(rows) == 43 and len(records) == 17
    analysis = dict(date="2026-10-30", language="ko", papers=records,
                    scope_ko="CVPR 2024–2026 본학회에서 NSD와 관련된 17편을 선별했다. 전수 체계적 문헌고찰은 아니다.",
                    classification_ko="본문 실험·분석 목적의 그림 43개. 배치 위치보다 용도를 기준으로 했다. 도입부 요약·구조도·데이터 구성은 별도 설명한다. BEV-SLD Fig. 3은 방법 절에서 제시한 진단 그림이므로 추가 항목으로 구분한다.",
                    page_numbering="본문 PDF의 첫 장을 1로 세며 학회 페이지 번호와 다름",
                    supplementary_analysis="PDF 부록은 보관했지만 그림 목록은 본문 기준이다. ZIP 부록 두 개는 링크만 보존한다.",
                    recommendations=[dict(title=t, references=r, proposal_ko=p, limit_ko=l)
                                     for t, r, p, l in RECOMMENDATIONS])
    (OUT / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    with (OUT / "experiment_figures.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "references.bib").write_text("\n\n".join(p["bibtex"] for p in manifest["papers"]) + "\n")

    parts = ['''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CVPR 2024–2026 · NSD 관련 논문의 실험 그림</title>
<style>
body{font:16px/1.7 system-ui,sans-serif;color:#172638;background:#f5f7fa;margin:0}
main{max-width:1160px;margin:auto;padding:36px 24px 64px}h1{font-size:30px;line-height:1.35}
h2{margin-top:2em;font-size:23px}h3{margin:0;font-size:20px}a{color:#155a98}
.lead,.paper,.proposal{background:white;padding:24px;border:1px solid #dce3eb;border-radius:10px;margin:18px 0}
.muted{color:#526273;font-size:14px}.tag{display:inline-block;background:#e8f0f8;padding:2px 9px;border-radius:5px;margin:5px 5px 5px 0;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dce3eb;padding:10px;text-align:left;vertical-align:top}th{background:#f2f5f9}
.scroll{overflow-x:auto}.gallery{display:flex;flex-wrap:wrap;gap:18px;margin-top:12px}.gallery figure{width:210px;margin:0}.gallery img{width:210px;box-shadow:0 1px 6px #bcc7d3}.gallery figcaption{font-size:13px}
summary{cursor:pointer;color:#155a98}input,select{padding:10px;border:1px solid #bcc7d3;border-radius:5px;font:inherit;margin:4px 8px 4px 0}
input{min-width:260px}.note{border-left:4px solid #4078a5;padding-left:14px}.paper[hidden]{display:none}
@media(max-width:600px){main{padding:16px 10px}.paper,.lead,.proposal{padding:14px}h1{font-size:25px}td,th{padding:7px}input{min-width:0;width:90%}}
@media print{body{background:white}.paper{break-inside:avoid}input,select,.filter{display:none}.gallery img{max-width:100%}}
</style><main><h1>CVPR 2024–2026<br>NSD 관련 논문의 실험 그림 조사</h1>
<p class="muted">확인일: 2026-10-30 · 공식 CVF 공개본 · 본학회 17편 · 한국어 분석</p>
<div class="lead"><strong>본문 PDF 17개와 PDF 부록 15개를 저장하고, 본문의 실험·분석 그림 43개를 원문 이미지로 확인했다.</strong>
<p>LiDAR 장소 인식·재위치 추정을 중심으로, 문맥·신뢰도·검색 효율을 참고할 수 있는 영상 및 레이더 장소 인식을 포함했다. 모든 관련 논문을 빠짐없이 수집한 체계적 문헌고찰은 아니다. 센서와 과제·데이터·학습·평가 규약이 달라 성능 수치를 직접 비교하지 않는다.</p>
<p>표의 쪽수는 <strong>본문 PDF의 첫 장부터 센 쪽수</strong>다. 1col·2col은 원문에서 확인한 한 단·두 단 배치다. 구조도, 데이터 소개, 첫 장 요약 그림은 별도 설명한다. 부록 그림 전체를 분석한 목록은 아니다. LiSA와 ForestLPR의 ZIP 부록은 공식 링크를 보존했다.</p>
<p><a href="experiment_figures.csv">43개 그림 목록 CSV</a> · <a href="analysis.json">한국어 분석 JSON</a> · <a href="manifest.json">PDF 출처·해시</a> · <a href="references.bib">17편 BibTeX</a> · <a href="completion_validation.json">검증 결과</a></p></div>
<h2>무엇을 참고할 수 있는가</h2>
<p>이 표본에서는 큰 정량 비교와 ablation을 표로 정리하고, 그림은 <strong>검색 성공·실패, 주목하는 구조, 신뢰도와 오류의 관계, 궤적에서 오류가 생기는 위치</strong>를 설명하는 경우가 반복된다. 이것이 CVPR의 필수 형식이라는 뜻은 아니다. 많은 논문이 정확도–차원 또는 시간 관계를 첫 장에 두므로, 그 그림을 실험 절 그림으로 세면 구성이 왜곡된다.</p>
<p class="note">NSD의 현재 Fig. 4가 효율성 비교를 맡는다면 다음 그림은 <strong>최종 점수 보정의 효과</strong>와 <strong>구체적인 혼동 사례</strong>를 우선하는 구성이 좋다. 아래 제안은 문헌에서 관찰한 사실과 구분한 우리 원고에 대한 판단이다.</p>''']
    for title, reference, proposal, limit in RECOMMENDATIONS:
        parts.append(f'<section class="proposal"><h3>{escape(title)}</h3><p class="muted">참고: {escape(reference)}</p><p>{escape(proposal)}</p><p class="note">{escape(limit)}</p></section>')
    parts.append('<h2>논문별 빠른 목록</h2><div class="scroll"><table><thead><tr><th>연도·논문</th><th>과제</th><th>실험·분석 그림</th></tr></thead><tbody>')
    for record in records:
        nums = ", ".join(f"{f['figure']} (p.{f['pdf_page']})" for f in record["figures"])
        parts.append(f'<tr><td><a href="#{escape(record["paper"])}">{record["year"]} {escape(record["short"])}</a></td><td>{escape(record["task_ko"])}</td><td>Fig. {nums}</td></tr>')
    parts.append('''</tbody></table></div><h2>원문과 상세 분석</h2><div class="filter"><input id="search" aria-label="논문 또는 그림 내용 검색" placeholder="논문명, 신뢰도, attention, 비용…"><select id="year" aria-label="연도"><option value="">모든 연도</option><option>2024</option><option>2025</option><option>2026</option></select><span id="count" class="muted">17편</span></div>''')
    for paper, record in zip(manifest["papers"], records):
        key = record["paper"]
        links = [f'<a href="{escape(paper["url"])}">CVF 공식 페이지</a>']
        for file in paper["files"]:
            label = "본문 PDF" if file["kind"] == "main" else "부록 PDF"
            links.append(f'<a href="{escape(file["path"])}">{label} ({file["pages"]}쪽)</a>')
        for external in paper.get("other_supplements", []):
            links.append(f'<a href="{escape(external["url"])}">공식 ZIP 부록 (미저장)</a>')
        parts.append(f'<article class="paper" id="{escape(key)}" data-year="{paper["year"]}"><h3>{paper["year"]} · {escape(paper["short"])}</h3><p>{escape(paper["title"])}</p><span class="tag">{escape(record["task_ko"])}</span><span class="tag">{escape(record["relevance_ko"])}</span><p>{" · ".join(links)}</p><div class="scroll"><table><thead><tr><th>그림·쪽·폭</th><th>무엇을 보여주는가</th><th>NSD에 참고할 점</th></tr></thead><tbody>')
        for figure in record["figures"]:
            parts.append(f'<tr><td><a href="papers/{escape(key)}/main.pdf#page={figure["pdf_page"]}">Fig. {figure["figure"]}<br>p.{figure["pdf_page"]} · {figure["width"]}</a><br><span class="tag">{escape(figure["category"])}</span></td><td>{escape(figure["description_ko"])}</td><td>{escape(figure["nsd_lesson_ko"])}</td></tr>')
        parts.append(f'</tbody></table></div><p class="muted">구분·주의: {escape(record["other_figures_ko"])}</p><details><summary>원문 페이지 미리보기 — 클릭하면 원본 크기로 열기</summary><div class="gallery">')
        pages = {f["pdf_page"]: f["preview"] for f in record["figures"]}
        for number, page, _ in record["extra_method_figures"]:
            pages[page] = lookup[key][number]["preview"]
        for page, preview in sorted(pages.items()):
            parts.append(f'<figure><a href="{escape(preview)}"><img loading="lazy" src="{escape(preview)}" alt="{escape(paper["short"])} 원문 PDF {page}쪽"></a><figcaption>PDF p.{page} · 원문 페이지</figcaption></figure>')
        parts.append('</div></details></article>')
    parts.append('''<h2>보관과 재현</h2><p>각 논문 폴더의 main.pdf·supplement.pdf는 공식 공개본이며 official.html과 manifest.json에 출처를 남겼다. 그림 미리보기는 원문 페이지 렌더링이고, 설명은 이 조사에서 작성한 한국어 요약이다. 저작권과 저자 표시는 원문에 따른다. 원고에 원문 그림을 그대로 재사용한다는 제안은 아니다.</p>
<p>sources/에 2024–2026 공식 목차를 보존했다. scripts/_cvpr_related_figure_survey.py가 수집, scripts/_cvpr_figure_catalog.py가 캡션·페이지 미리보기 추출, scripts/_cvpr_figure_report.py가 이 보고서와 검증을 담당한다. 그림 목록의 해석은 자동 캡션 분류만으로 만들지 않고 원문 이미지를 확인했다.</p>
<p class="muted">부록 PDF는 보관 대상이다. 그림 번호 분석은 본문 기준이며, 보고서의 추가 제안은 실행 완료한 새 실험을 뜻하지 않는다.</p></main>
<script>
const s=document.querySelector('#search'),y=document.querySelector('#year');
function filter(){let n=0;document.querySelectorAll('article.paper').forEach(p=>{p.hidden=!p.textContent.toLowerCase().includes(s.value.toLowerCase())||(y.value&&p.dataset.year!==y.value);if(!p.hidden)n++;});document.querySelector('#count').textContent=n+'편';}
s.addEventListener('input',filter);y.addEventListener('change',filter);
</script></html>''')
    report = "\n".join(parts)
    (OUT / "index.html").write_text(report)

    # 분석 결과를 반환하기 전에 저장 파일·쪽수·해시·로컬 링크를 검증한다.
    checked_files, byte_count = [], 0
    for paper in manifest["papers"]:
        for file in paper["files"]:
            data = (OUT / file["path"]).read_bytes()
            assert data.startswith(b"%PDF"), file["path"]
            assert len(data) == file["bytes"], file["path"]
            assert hashlib.sha256(data).hexdigest() == file["sha256"], file["path"]
            byte_count += len(data)
            checked_files.append(file["path"])
    soup = BeautifulSoup(report, "html.parser")
    ids = {node["id"] for node in soup.find_all(id=True)}
    local_links = 0
    for node in soup.find_all(["a", "img"]):
        url = node.get("href", node.get("src", ""))
        if url.startswith("https://"):
            continue
        path, _, fragment = url.partition("#")
        if path and path != "completion_validation.json":
            assert (OUT / path).is_file(), path
            local_links += 1
        elif not path:
            assert fragment in ids, fragment
    validation = dict(status="passed", papers=len(records), pdfs=len(checked_files),
                      pdf_bytes=byte_count, selected_figures=len(rows),
                      extra_method_figures=1, main_figure_inventory=sum(len(p["figures"]) for p in inventory),
                      selected_figure_pages_visually_checked=sorted({f"{row['year']}_{row['paper']}:p{row['pdf_page']}" for row in rows}),
                      years=dict(Counter(p["year"] for p in manifest["papers"])),
                      figure_categories=dict(Counter(row["category"] for row in rows)),
                      local_links_checked=local_links, pdf_files=checked_files,
                      supplements_not_downloaded=[x for p in manifest["papers"] for x in p.get("other_supplements", [])])
    (OUT / "completion_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: validation[k] for k in ("status", "papers", "pdfs", "selected_figures", "pdf_bytes", "local_links_checked")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
