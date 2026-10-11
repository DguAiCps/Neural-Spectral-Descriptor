# Causal PR 캠페인 최종 — 트랙 C 베이스라인 6종 + 8-method 비교표

- 날짜: 2026-08-11 ~ 08-14
- 사전 등록: docs/CAUSAL_PR_PREREGISTRATION.md (rev.2+A1)
- 선행 일지: 2026-08-11_causal_retrain.md, 2026-08-11_causal_pr_trackAB.md

## 왜

사전 등록 캠페인의 마지막 트랙: 베이스라인 6종을 동일 causal 프로토콜
(전-프레임 쿼리, past-only 후보, native score)로 덤프해 8-method 결정 지표
비교표를 완성.

## 설정

- 평가기 5파일 신규 (`_causal_{scpp,bevplace,seqot,ringsharp,ring}_pr.py`),
  workflow adversarial 검증 5/5 PASS (planted 비인과 정답 반증 포함).
- Native score: SC++ cyclic distance(부호 반전), BEVPlace/SeqOT cosine,
  RING/RING++ 1−상관(부호 반전), RING# translation-branch error(주)+rot 병기.
- SeqOT는 **trailing window**로 서술자 재계산 (중심 창의 미래 ±10프레임
  누수 제거, meta 공시). RING# 가중치 = pd 재학습 선택 ckpt (sha 검증).
- **RING 평가기 10배 최적화**: 공식 circorr2가 쌍마다 InstanceNorm+FFT를
  재계산(O(N²)회) → 프레임당 1회로 호이스팅 + 쿼리당 융합 커널.
  수학 동일(계산 순서 재배치), kitti/05 양 변형에서 slow 경로와
  **maxdiff 0.00e+00·j* 불일치 0/2051 bit-일치 검증 후 투입**.
  100h 예상 → 실제 잔여 ~15h.
- 장애 2건: ① Windows 재부팅(8/12 새벽)으로 중단 → 체인 재개(무손실),
  ② 재부팅 후 WSL2 GPU wedge로 DCC04 ringpp가 24h 정체(CPU 100%/GPU 17%,
  CUDA 호출 미반환) → 컨테이너 재시작 + 워커 재기동으로 해소(이후 무장애).

## 결과 (37seq 6그룹 macro, `results/pd2026_pr/pr_summary.json`)

| Method | AP | maxF1 | R@100P | R@95P |
|---|---|---|---|---|
| BEVPlace++ | **0.769** | **0.788** | 0.312 | 0.466 |
| RING | 0.747 | 0.762 | 0.298 | 0.498 |
| SC++ | 0.688 | 0.699 | **0.417** | **0.523** |
| RING++ | 0.625 | 0.642 | 0.237 | 0.363 |
| **NSD causal-재학습** | 0.561±0.004 | 0.564 | 0.222±0.007 | 0.302 |
| NSD-H causal-재학습 | 0.561±0.005 | 0.560 | 0.222±0.012 | 0.299 |
| NSD-H Frozen-GNN | 0.446±0.015 | 0.486 | 0.067 | 0.110 |
| NSD Frozen-GNN | 0.431±0.018 | 0.475 | 0.058 | 0.102 |
| RING#-L (pd) | 0.359 | 0.430 | 0.095 | 0.234 |
| SeqOT (trailing) | 0.339 | 0.414 | 0.177 | 0.253 |

운영점(@τ*, val 고정→test): NSD causal P 0.550/R 0.560, NSD-H 0.532/0.565.

## 결론

1. **Causal 결정 지표에서는 per-frame 방법이 구조적으로 우위** — BEVPlace++/
   RING/SC++가 상위권. 인과 전환에서 잃는 것이 없고(프레임 단독 서술자)
   native 거리의 cross-query 캘리브레이션이 좋음. 특히 SC++는 R@100P
   0.417로 무오탐 운영의 전통적 강자임이 재확인됨.
2. **NSD causal-재학습은 중위권(AP 5위)** — SeqOT·RING#과 자기 Frozen
   변형은 크게 앞서지만, 문맥 기반 방법의 구조적 비용(causal에서 그래프
   절반 상실)과 416D key cosine confidence의 분리력 한계가 드러남.
   재학습의 효과는 결정적(AP +30%, R@100P 4배)이며 저장 비용 축(NSD
   3.1KB vs BEVPlace 32KB, RING 28.8KB)은 유지됨.
3. 사전 등록의 배치 규칙(결과 무관 부록 + 전 결과 보고)이 그대로 적용됨 —
   본문 주장은 offline symmetric retrieval(NSD-H 1위, 기존 확정)이고,
   causal PR은 부록의 한계 분석으로 공시. 선택적 보고 여지 없음.
4. Offline(R@1)과 causal(PR)의 순위 역전 자체가 논문의 정직한 기여:
   "표를 잘 받는 방법"과 "배치 결정에 좋은 방법"이 다르며, NSD는 전자를
   주 무대로 규정하고 후자의 현재 위치와 개선 방향(confidence 캘리브레이션)
   을 공시.

## 다음

부록 LaTeX 반영(표+회계 특성화 문단), 캠페인 자산 커밋(사용자 승인 후).
