# Causal(온라인) PR 평가 — 사전 등록 실험 계획 (재학습 포함), rev.2

작성 2026-08-08, **rev.2 2026-08-11** (동결 전 검토 반영: 후보 생성의 선발-후-제거
금지, classifier prefix 불변성, frozen 명칭 교정, 버퍼 회계 분리, confidence
수식 고정, temporal fan-in 서술 교정, 조건부 배치 규칙 제거).
**이 문서의 모든 선택 규칙은 test 접촉 전에 동결한다.** 동결 시점 = 이 문서의
git 커밋; 이후 수정은 Amendments에 사유와 함께 추가.

## 0. 목적과 주장 범위

- 목적: 온라인 loop-closure 결정 품질을 표준 지표(PR 곡선, AP, max F1,
  R@100P)로 측정. causal past-only 조건 + 정답 없는 프레임 포함.
- 주장 범위: "causal 조건의 검색·결정 품질"까지만. 증분 계산의 시스템 비용·
  실시간성·SLAM 통합(ATE)은 범위 외.
- **배치 결정 (결과 무관, 지금 고정): 본 실험 전체는 부록(supplement)에
  수록한다.** 논문의 주 주장은 offline symmetric retrieval이고 causal PR은
  보조 deployment 분석이기 때문이다. 모든 방법·모든 지표를 결과와 무관하게
  전부 보고한다. "comparable" 여부의 해석은 3-seed mean AP 차이와
  저장공간·latency Pareto 관계로 서술하며, best seed가 아닌 mean±std를
  보고한다.
- 두 NSD 트랙을 모두 보고한다:
  (a) **Frozen-GNN causal adaptation** — GNN 가중치와 ω는 동결, causal
  classifier와 fusion 선택만 train/val에서 재구축. ("하한"이라 부르지
  않는다: causal 재학습보다 낮다는 보장이 없고, 전체 모델이 frozen인 것도
  아니다. 역할 = 재학습 없이 도달 가능한 지점의 진단.)
  (b) **Causal-retrained** — causal 그래프로 재학습한 정식 수치 (**주 행**).

## 1. 프로토콜 동결 (사전 고정 규칙)

### 1.1 Causality 정의 — full causal (후보 생성 단계 포함)

전 엣지 src<dst는 필요조건일 뿐이다. **후보 생성 자체가 prefix 내부에서
일어나야 한다** — 전역 top-k를 구한 뒤 미래를 제거하는 방식은 미래 프레임이
top-k 슬롯을 점유해 과거 후보 구성을 바꾸므로 금지한다.

- Temporal 엣지: past→current만 (`past_to_current`).
- Similarity 엣지: **시퀀스별 incremental index로 생성** —
  `query i against index(frames ≤ i − skip)` → 과거 후보 선택 → 엣지
  (j → i) 방출 → frame i 삽입. skip은 기존 temporal-exclusion 규약과 동일
  값 사용.
- Density·adaptive prior·top-k 선정: **prefix 내부에서만 계산**을 기본으로
  한다. 초반 프레임은 `k_eff = min(k, |prefix|)` (온라인 현실과 일치).
  대안으로 보정 상수(사전확률·밀도 스케일)를 **train split에서 고정**하는
  것은 causal적으로 합법(평가 시퀀스의 미래 정보가 아님)이며, bayesian
  edge_method의 null 분포 통계가 prefix에서 불안정할 경우 이 대안을
  채택한다. **어느 쪽을 쓰는지는 val 단계 종료 전에 확정·기록한다.**
- Verified loop-closure 엣지([graph_manager.py:1357](../src/keyframe/graph_manager.py#L1357)
  의 명시적 양방향 방출): causal 모드에서 **past→current로 제한하거나
  비활성화** (기본: 비활성화; pd 경로에서 미사용임을 확인 후 기록).
- Edge classifier 후보·특징: §2.1의 prefix 규칙.
- 검색 후보: `j ≤ t − 30` (recent-frame exclusion, 기존 규약).
- 검증: prefix-invariance test (§4.1)가 후보 생성 단계까지 포함해 성립해야
  한다 — incremental 구현에서는 구성상 자동으로 성립하며, 테스트는 그 증명.

### 1.2 쿼리 집합 — 전 프레임 (TN/FN 포함)
- 쿼리 = 프레임 인덱스 t ≥ 30인 모든 프레임.
- 쿼리 분류 (과거 DB(≤ t−30) 최근접 GT 거리 d_min 기준):
  - **positive query**: d_min ≤ 5 m
  - **negative query**: d_min > 25 m
  - **ambiguous query**: 5 m < d_min ≤ 25 m → **PR 집합에서 제외** (공시)

### 1.3 판정 규칙 (임계값 τ 스윕) — 쿼리 분류와 매치 판정의 분리

버퍼 존은 **쿼리의 분류에만** 사용한다. 수락된 매치의 정오 판정에는
사용하지 않는다 (잘못 반환된 10 m 매치를 면책하면 false acceptance가
precision에서 사라져 낙관 편향).

- 예측(수락) = top-1 결정 점수 s(t) ≥ τ.
- **수락된 매치의 정답 조건: 반환된 top-1이 쿼리로부터 5 m 이내.**
  5 m를 벗어난 수락 매치는 거리와 무관하게 **FP** (positive query에서든
  negative query에서든).
- TP: positive query ∧ 수락 ∧ top-1 ≤ 5 m.
- FN: positive query ∧ 비수락. TN: negative query ∧ 비수락.
- **Recall 분모 = 전체 positive query 수 (τ와 무관하게 고정).**
  Precision = TP/(TP+FP).
- 공식 RING 구현(plot_PR_curve.py)은 positive query의 오수락을 FP에만 넣고
  FN에 포함하지 않아 recall 분모가 τ에 따라 변할 수 있다 — 교차검증은
  "차이를 탐지하고 설명하는 테스트"로만 사용하고(§4.3), 주 평가는 위의
  고정 분모 정의를 쓴다.

### 1.4 지표
- PR 곡선 (전 임계값 스윕), **AP** (step-wise interpolated), **max F1**,
  **R@100P** + R@99P/R@95P (R@100P 취약성 방어).
- 구분 명시: AP·max F1·R@100P·R@9xP는 **test 점수를 스윕해 얻는 평가
  통계**이고, τ*는 **실제 배치 가능한 고정 운영점**이다.
- **운영점 행**: τ* = 각 시드의 validation macro F1 최대점 (방법별·시드별,
  val에서 고정) → test에 그대로 적용해 P/R@τ* 보고 (시드 mean±std).

### 1.5 결정 점수 (방법별, 사전 고정)

베이스라인 (native score; 전부 기존 루프에서 계산됨 — 덤프만 추가):

| 방법 | score |
|---|---|
| SC++ | cyclic column-cosine distance top-1 (부호 반전) |
| RING/RING++ | 1 − 정규화 circular correlation (공식 `dist = 1 − score`) |
| RING#-L | 병기 덤프: translation-branch `errs.min()`(주) + rot_dist(비교용) |
| BEVPlace++ | 8192D cosine top-1 |
| SeqOT | 256D cosine top-1 (trailing window; §3 주의) |

NSD/NSD-H — **retrieval과 confidence의 분리를 수식으로 고정**:
1. 최종 retrieval 매치는 항상 `j* = argmax_j S_fusion(q, j)` (동결 tuple의
   fused ranking; confidence 후보가 매치 선택을 바꾸지 않는다).
2. Confidence 후보는 다음 2종만 validation에서 비교:
   - **absolute**: `cos(z_q, z_{j*})` — 정규화 전 raw key cosine을 fused
     argmax 위치에서 평가 (cross-query 비교 가능)
   - **margin**: 최종 fusion ranking의 `S₁ − S₂`
3. 선택 기준: **validation sequence-macro AP 최대.**
   Tie-break: 차이 < 0.005면 **absolute** 선택 (더 단순·비교 가능).
4. 선택 결과는 **세 시드 공통으로 고정** (시드별로 다른 통계 금지).
5. fused top-1 값 자체는 per-query min–max라 임계값 통계로 사용 금지.

### 1.6 집계
- per-sequence 스윕 → 데이터셋 6그룹 macro (Strict-v2 그룹) → 전체 =
  6그룹 macro. 전역 단일-τ 풀링은 점수 스케일 이질성으로 채택하지 않음(공시).
- 로스터: Strict-v2 37시퀀스. NSD 3시드 mean±std (macro), 베이스라인 단일.

## 2. 재학습 스펙 (causal-retrained NSD)

### 2.1 코드 변경 범위 (4곳 — "graph_manager 10줄"이 아님)

1. **Graph builder** (`src/keyframe/graph_manager.py`):
   similarity 후보 생성을 시퀀스별 incremental index로 재구현 (§1.1) —
   prefix 검색, (j → i) 방출, density/prior prefix 계산(또는 train-split
   상수), verified-loop 엣지 처리, provenance 스탬프. temporal은 기존
   `past_to_current` 재사용.
2. **Edge classifier feature builder**
   (`build_and_train_edge_classifier.py` + `_pd_build_edge_classifier.py`):
   - 후보를 current i의 과거에서만 생성 (전역 N×N 행렬 금지)
   - margin·density·rank·reciprocal rank를 **시점 i의 prefix에서 계산**
   - `abs(src−dst)/n` (n=시퀀스 길이 — prefix 비불변) →
     **고정 정규화 `min(Δframe, D_max)/D_max`, D_max = 5000**으로 교체
   - μ/σ 표준화는 causal 특징 분포에서 재산출
3. **Two-pass 평가 경로** (`_pd_branch_fusion.py`, `_pd_b3_rerank.py`):
   pass-1·pass-2 forward 모두 causal 그래프, classifier-selected 엣지
   **past→current로만 삽입** (현재 src2→dst2 그대로 삽입 — 수정 필요).
4. **Evaluator/score-dump**: 전-프레임 쿼리 집합(§1.2), causal 후보 마스크,
   per-query score 원장.
- Trainer·모델·체크포인트 포맷 변경 없음 (감사 확인).
- 훈련 그래프 구축 비용: incremental index로 시퀀스당 O(N·prefix) — GPU/
  블록 numpy로 시퀀스당 수 분, 전체 1–2 h 추가 예상.

### 2.2 학습 구성
- Config: `configs/training_place_disjoint.yaml` + causal 플래그.
  **temporal_neighbors: 20** — 정확한 서술: half_window = M//2이므로 legacy
  M=10은 과거 5+미래 5, causal M=20은 **과거 10**. 즉 "과거 밴드 유지"가
  아니라 **총 incoming temporal degree ~10을 유지하기 위해 과거 범위를
  10으로 확장**하는 것이며, **최대 temporal lag가 5→10으로 늘어남을
  공시한다.**
- 나머지 하이퍼파라미터·인코더 프리셋(no_interdiff)·gated context(α=0.0625)·
  시드(1/2/3) 전부 pd와 동일. pdnn 구성은 범위 외.
- 산출: `checkpoints/pd_causal_s{1,2,3}`. Best-epoch 규칙: pd와 동일 (val).

### 2.3 선택 스택 재구축
| 단계 | 조치 | 근거 |
|---|---|---|
| ω 재계량 | 재사용 (checkpoint-free); provenance 위해 재실행·bit-일치 확인 | train 캐시 288D+pose만 사용 |
| 레이아웃/spec 캐시 | 재사용 | 인코더 레벨 |
| Edge classifier | **causal 특징으로 재학습** → `edge_classifier_pd2026_causal.pt` | GNN 임베딩 의존 + §2.1의 prefix 특징 |
| B3 / branch fusion (val) | 재실행 | 체크포인트 의존 |
| Fusion tuple | val 4시퀀스에서 재선택 (기존 grid·규칙) | 체크포인트 의존 |
- Frozen-GNN causal adaptation 트랙: 기존 pd 체크포인트 + ω 동결, causal
  classifier·tuple만 재구축 (동일 §2.1 코드 사용; 명칭·정의 §0).
  완전 zero-shot 진단(구 classifier 그대로)은 분포 불일치로 실용 의미가
  낮아 채택하지 않음 (공시).

## 3. 실행 매트릭스

| 트랙 | 대상 | 산출 |
|---|---|---|
| A. val | confidence 통계 선택(§1.5), τ*(시드별), tuple 재선택, density/prior 방식 확정 | val 4시퀀스에서 종료 후 동결 |
| B. NSD test | Frozen-GNN adaptation(3시드) + causal-retrained(3시드) × 37seq × {NSD, NSD-H}, 전-프레임 dump | PR 원장 JSON |
| C. 베이스라인 test | 6종 × 37seq, causal 후보 마스크 + native score dump. per-frame 방법은 서술자 재사용(검색만 재실행); RING/RING++ 상관 재실행(과거 후보 ~절반); SeqOT는 **trailing window로 서술자 재계산** (중심 창은 미래 누수) | PR 원장 JSON |
| D. 집계 | PR lib → 곡선 + AP/maxF1/R@100P/R@99/95P + 운영점, 6그룹 macro | 표·그림 (부록) |

## 4. 검증 계획

1. **Prefix-invariance test** (인과성의 결정적 증명): 표본 쿼리 t에 대해
   전체 시퀀스 처리 결과의 s(t)·선택 엣지·후보 집합 == 시퀀스를 t에서
   절단하고 처리한 결과 (후보 생성 단계 포함; NSD 전 스택 + 베이스라인
   1종, 시퀀스당 ≥20 쿼리).
2. Bidirectional 회귀: causal 플래그 해제 시 기존 pd R@1 수치 재현.
3. 공식 PR 구현 교차검증 — **차이-탐지 테스트**: RING 1개 시퀀스에서 동일
   점수 행렬로 우리 PR lib vs 공식 plot_PR_curve.py를 비교하고, 회계 차이
   (오수락의 FN 미포함 → 가변 recall 분모)로 설명되는 차이만 허용.
4. PR lib 단위 테스트 (toy 케이스: 쿼리 3분류, FP 판정, 고정 분모, AP 공식).
5. 엣지 감사: 학습·추론 그래프 전 엣지 src<dst assert + 후보 생성 로그의
   prefix-only assert; classifier 특징의 prefix 불변성 spot-check
   (절단 전후 특징 벡터 일치).
6. ω bit-일치; confidence/τ*/tuple 선택 로그가 전부 val임을 스크립트 검사.

## 5. 일정·비용 (GPU 1대, 직렬)

| 일차 | 내용 |
|---|---|
| 1–2 | §2.1 코드 4곳 (incremental builder가 최대 항목) + PR lib + dump 변형 + 검증 1·2·4·5 |
| 2 야간 | causal 재학습 3시드 (~5 h + 그래프 구축 1–2 h) |
| 3 | 선택 스택 재구축 (classifier per-seq 체인 → B3 → branch fusion val) → 트랙 A 동결 |
| 4 | 트랙 B (양 트랙 NSD test dump) ∥ SC++/BEVPlace/SeqOT 재검색 |
| 5 | RING/RING++/RING# 상관 재실행 (지배 항목) + SeqOT trailing 재계산 |
| 6 | 집계·표·그림·일지·부록 반영 |
- 총 **~6일** (rev.1의 ~5일에서 상향: incremental builder + classifier
  prefix 재작성 반영).

## 6. 위험 (배치·보고는 §0에서 결과 무관으로 고정됨)

1. Frozen-GNN adaptation과 causal-retrained의 서열은 사전에 가정하지
   않는다 (구 증거에서 시퀀스별 역전 존재). 두 트랙 모두 보고.
2. R@100P의 GT 노이즈 취약성 → ambiguous 쿼리 제외 + R@99/95P 병기.
3. temporal lag 5→10 변경의 효과는 causal 조건의 일부로 공시 (bidirectional
   과의 분리 비교는 하지 않는다 — 조건이 다름을 명시).
4. Causal classifier의 성능 미지수 — fallback(temporal-only 두-pass) 채택
   여부는 **val AP 기준, 트랙 A 동결 전에** 결정·기록.
5. Incremental builder의 수치 재현성: prefix 검색은 결정적(FAISS exact/
   blocked numpy)으로 구현 — 근사 인덱스 금지.

## Amendments

- A1 (2026-08-11, 구현 착수 시 스펙 구체화; val/test 접촉 전):
  ① reciprocal rank(classifier 특징 [4])의 causal 정의 확정 — "sim(i,j)가
  후보 노드 j의 **삽입 시점 prefix top-M 유사도 목록**에서 차지하는 삽입
  위치(strictly-greater 개수, M 상한)". j가 미래의 i를 순위화하지 않으며
  O(1)/쌍·prefix-불변. ② 특징 [13]의 고정 정규화 상수 D_max = 5000.
  ③ Frozen-GNN 트랙의 causal 그래프도 재학습 트랙과 동일 조건
  (temporal_neighbors 20 past band) — 두 트랙의 조건 통일.
- rev.2 (2026-08-11, 동결 전): 검토 7건 반영 — ① 후보 생성 prefix화
  (선발-후-제거 금지) + verified-loop 엣지 처리, ② classifier prefix
  불변성 + 고정 정규화, ③ "frozen-causal 하한" → "Frozen-GNN causal
  adaptation", ④ 버퍼 회계 분리 (쿼리 3분류 / 매치 판정 5 m 단일 기준 /
  고정 recall 분모), ⑤ confidence 수식·tie-break·시드 공통·τ* 구분,
  ⑥ temporal_neighbors=20 서술 교정 (총 degree 유지·최대 lag 5→10),
  ⑦ 조건부 본문/부록 규칙 삭제 → 결과 무관 부록 고정 + 전 결과 보고.
