# Causal(full) 재학습 3시드 — 사전 등록 PR 캠페인 트랙 B의 학습 단계

- 날짜: 2026-08-10 ~ 08-11
- 관련 브랜치: feat/pbev-stored-spectrum
- 사전 등록: docs/CAUSAL_PR_PREREGISTRATION.md (rev.2 + A1, 커밋 3d9426f/c45fa8e)

## 왜 (진행 이유)

Online loop-closure 결정 지표(PR/AP/maxF1/R@100P) 평가의 주 행은 causal
재학습 모델이어야 함 — frozen 가중치의 causal 평가는 학습·추론 그래프
불일치로 성능을 왜곡하고(구 증거 0.39 vs 재학습 0.47), 서열 보장도 없어
"하한"이라 부를 수 없음(prereg §0). 구 causal_retrain(2026-07)은 temporal
엣지만 제한한 부분 개입이었으므로, 이번이 최초의 **full causal**
(similarity 후보 생성까지 prefix) 재학습.

## 설정 (파라미터)

- Config: `configs/training_place_disjoint_causal.yaml` — 기반 pd config 대비
  정확히 3개: temporal_neighbors 10→20 (past half-window 10; 총 in-degree
  ~10 유지, 최대 lag 5→10 공시), temporal_edge_mode past_to_current,
  similarity_causal true. 캐시 키 무영향 확인(선택 파라미터 불변).
- 신규 causal 경로 (`graph_manager._causal_prefix_candidates` + causal 분기):
  후보를 same-sequence prefix(j ≤ i−gap−1, gap=half_window)에서만 gemv로
  생성(전역 top-k 후 필터 금지 — 미래가 top-k 슬롯을 점유하는 누수 차단),
  엣지 (j→i) 방출, density는 노드별 자기 prefix에서 k_eff=min(50,|prefix|),
  prior 보정은 train-split 상수(offline 학습 그래프 전용 — prereg §1.1 대안).
  **gemv 선택 이유**: blocked GEMM은 BLAS 누적 순서가 행렬 형태 의존이라
  bit-수준 절단 불변성이 깨짐(테스트로 검출) — 독립 row-dot인 gemv로 교체.
- 검증(학습 전): 절단 불변 bit-일치·brute-force 일치·시퀀스 경계·전 엣지
  src<dst·verified-loop 차단 가드 전부 통과. 실데이터 스모크(kitti_07 +
  실제 similarity_dist) 정상.
- 학습: 공식 레시피 pd와 동일(no_interdiff, gated context α=0.0625), 시드
  1/2/3, `scripts/_run_causal_retrain.sh`.

## 결과

- Causal 학습 그래프: 134,830 노드, 1,770,314 엣지 (temporal 1,347,420 =
  노드당 ~10 설계대로; similarity 422,894 = bidirectional 595,829 대비
  −29%). 구축 52–56 s (gemv prefix 스윕이 Sejong01 18k에서 1.7 s — 비용
  우려 해소). 시드 간 엣지 수 완전 동일(결정적 구축).
- 3시드 학습 완료 (시드당 ~50분): best val(1-pass) recall@1 =
  **0.533 / 0.529 / 0.540** — 시드 간 정합, 이상 수렴 없음.
- 체크포인트: `checkpoints/pd_causal_s{1,2,3}/best_model.pth`.

## 부수 발견 (ω provenance 검증 중)

`_pd_fit_key_remetrize.py`가 `_test_` 캐시를 제외하지 않아 — 8/4 적합
시점엔 test 캐시가 없어 원본 아티팩트는 청정하지만 — **현 시점 재실행은
test 42캐시를 조용히 수집**(재적합 상대차 39%로 검출). `_test_` 스킵 수정
후 docker 재실행이 커밋 아티팩트와 **bit-일치** (sha 77c4f4d2…) →
ω 재사용의 provenance 증명 + 향후 test 누수 통로 폐쇄.

## 다음

트랙 A 체인 (`_causal_track_a_chain.sh`, 실행 중): causal edge classifier
2트랙(pd_causal/pd-frozen) 재학습 → val tuple 그리드 → tuple 동결 →
confidence(absolute vs margin, val macro AP)·τ*(val macro F1) 선택 →
트랙 A 동결 → 37-seq test 덤프(양 트랙) + 베이스라인 6종 causal 덤프.
