# Causal PR 캠페인 트랙 A(선택 동결) + 트랙 B(NSD test)

- 날짜: 2026-08-10 ~ 08-11
- 사전 등록: docs/CAUSAL_PR_PREREGISTRATION.md (rev.2+A1)
- 선행 일지: 2026-08-11_causal_retrain.md

## 왜

Online loop-closure 결정 지표(PR/AP/maxF1/R@100P)를 사전 등록 프로토콜로
산출 — 전 선택(tuple·confidence 통계·τ*)을 val 4시퀀스에서 동결한 뒤
Strict-v2 37시퀀스 test를 1회 실행.

## 트랙 A — 선택 동결 (전부 val 전용)

- Causal edge classifier 재학습 (양 트랙): holdout P@|pos| 0.780(pd_causal) /
  0.771(frozen). 특징은 전부 prefix 계산(전역 N×N 금지), `/n` 정규화 →
  `min(Δ,5000)/5000` 교체, reciprocal rank는 A1의 삽입-시점 정의.
- Tuple 재선택 (기존 `_pd_tuple_select` 규칙): pd_causal pbev **(2,1,2,2)** /
  hspec **(2,1,2,0.5)** — bidirectional과 동일 재선택(가중 구조의 안정성);
  Frozen-GNN은 (0.5,1,2,1) 양쪽.
- Confidence 통계 (사전 규칙: val macro AP, tie<0.005→absolute):
  **absolute 압승** — pd_causal pbev 0.518 vs margin 0.168 (3배).
  pool-상대 min–max margin은 threshold 통계로 부적격임이 실증됨.
- τ* (시드별 val macro F1): pd_causal ~0.907–0.909 / frozen ~0.812–0.815.
- 절차 결함 1건: confidence 선택이 seed 1 시점에 조기 동결(러너 freeze
  규율 × 체인 시드별 호출 충돌) → 전 시드 재집계로 교정(사전 규칙 "존재
  전 시드"의 올바른 이행), 교정 전후 결론 동일.

## 검증 (사전 등록 §4, 전 항목 통과)

1. **실데이터 전체 스택 prefix-invariance**: kitti/05 T=1200 절단 →
   1,170 쿼리 원장 7배열 bit-일치 (graph→GNN→classifier→주입→pass-2→
   prefix 풀→fusion 전 구간, CPU 결정적 실행).
2. 공식 RING PR 구현 교차검증 (4 tests): 혼동행렬 셀은 동일, 차이는 recall
   분모뿐(공식 가변·부풀림, 우리 고정) — **R@100P는 정의상 완전 일치**
   (P=1 ⇔ FP=0 ⇔ 분모 동일). 공식 소스 sha 가드 포함.
3. ω provenance: fitter의 `_test_` 미제외 잠재 누수 발견·수정 후 docker
   재실행 bit-일치 (선행 일지 참조).
4. Workflow adversarial 검증: causal 평가 경로 2종 + 베이스라인 평가기
   5종 전부 PASS (planted 비인과 정답 반증 시도 포함).

## 트랙 B 결과 — NSD (37시퀀스, 6그룹 macro, `pr_summary.json`)

| | AP | maxF1 | R@100P | R@99P | R@95P | 운영점 @τ* (P/R/F1) |
|---|---|---|---|---|---|---|
| NSD causal-재학습 | **0.561±0.004** | 0.564 | **0.222±0.007** | 0.263 | 0.302 | 0.550/0.560/0.459 |
| NSD-H causal-재학습 | 0.561±0.005 | 0.560 | 0.222±0.012 | 0.261 | 0.299 | 0.532/0.565/0.459 |
| NSD Frozen-GNN | 0.431±0.018 | 0.475 | 0.058±0.008 | 0.070 | 0.102 | 0.325/0.666/0.409 |
| NSD-H Frozen-GNN | 0.446±0.015 | 0.486 | 0.067±0.012 | 0.078 | 0.110 | 0.338/0.668/0.420 |

판독: ① **재학습 효과 결정적** — AP +0.13(+30%), R@100P 4배(0.06→0.22).
학습·추론 그래프 일치가 결정 지표에서 특히 중요. ② NSD/NSD-H 두 operating
point는 causal PR에서 동등(±0.001). ③ val-고정 τ*가 test에서 P≈R 균형점
으로 이전(붕괴 없음) — 배치 운영점 주장 가능. ④ Frozen-GNN은 τ*에서
recall 쏠림(P 0.33/R 0.67) — 점수 캘리브레이션 자체가 재학습으로 개선됨.

## 다음

트랙 C 베이스라인 6종 체인 실행 중 (SC++→BEVPlace→SeqOT(trailing)→
RING#→RING/RING++; RING 블록이 지배 ~1일) → 최종 비교표 + 부록 반영.
