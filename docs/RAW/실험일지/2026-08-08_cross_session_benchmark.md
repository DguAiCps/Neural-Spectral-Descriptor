# RING-style cross-session 벤치마크 (7쌍 × 8 method, 공통 keyframe roster)

- 날짜: 2026-08-07 ~ 08-08
- 관련 브랜치: feat/pbev-stored-spectrum

## 왜 (진행 이유)

교수님 지적: intra-session 표만으로는 "표준 관행" 보완이 부족 — map/query 세션 분리
프로토콜이 가장 중요한 추가. 기존 cross-session 수치는 key-only 진단치뿐이라
RING 계열과 정면 비교 불가였음. fusion+alignment의 cross-session 확장이 목표.

## 설정 (파라미터)

- 명칭: **"RING-style cross-session protocol under a common keyframe roster"**
  (pd2026 keyframe cache에서 20 m/5 m 재샘플링 — official RING roster 아님, 호명 금지).
- 7쌍: NCLT 02-04→03-17, MulRan DCC/KAIST/Riverside 01→02·01→03.
  map 20 m / query 5 m spacing, **roster = map 20 m 내 query 1회 고정** (전
  threshold 동일 분모, 재필터 금지), planar, R@1% k=max(round(n_db/100),1)=2~3.
- 후보 생성 **DB-only** (`db@q` top-k, 매 query `assert cand < n_db`);
  [db;q] concat은 alignment 인덱싱 전용. NSD fusion은 동결 tuple
  (pd/pbev (2,1,2,2), pd/hspec12 (2,1,2,0.5)), pool_k=100 union 의미 보존.
- Pose 단일 소스 = `keys_pd_s1_*_gp.npz` (전역 GT; 캐시 pose는 세션 re-center 금지).
- 신규 파일만 사용 (`_xs_*` 계열; `eval_utils.py`/`_pdb_*` byte-identical).
  lifted fusion 헬퍼는 docker 원본과 bit-identity 검증 (`_xs_verify_lift.py` 전 항목 PASS).
  4개 베이스라인 드라이버는 workflow 작성 + adversarial 검증 4/4 PASS.
- 베이스라인: SC++/RING/RING++ training-free, BEVPlace++/SeqOT는 pd 재학습 ckpt.
  RING/RING++은 기존 BEV 캐시 read-only 재사용 (`compute_bevs` 클로버 차단).
  SeqOT는 전체 native 스택에서 서술자 계산 후 슬라이스 (윈도우 오염 방지).
- 검증: roster sha256 method 불변 게이트(집계기 hard-fail), NSD key-only
  NCLT 20 m 0.359가 기존 진단치 0.364와 정합, 분모 고정 invariant JSON 기록.

## 결과 (site-balanced macro; results/pd2026_xsession/xs_summary.json)

| Method | R@1@5m | R@1@20m | R@1%@20m |
|---|---|---|---|
| SC++ | 0.356 | 0.605 | 0.678 |
| BEVPlace++ | **0.384** | **0.798** | **0.858** |
| SeqOT | 0.329 | 0.734 | 0.794 |
| RING | 0.354 | 0.755 | 0.816 |
| RING++ | 0.342 | 0.727 | 0.779 |
| NSD (3-seed) | 0.383±0.001 | 0.695±0.002 | 0.755±0.002 |
| NSD-H (3-seed) | 0.383±0.000 | 0.703±0.001 | 0.760±0.002 |
| NSD-H pdnn (보조) | 0.380±0.001 | 0.688±0.002 | 0.749±0.002 |
| NSD key-only (부록) | 0.265±0.002 | 0.533±0.005 | 0.657±0.001 |

Coverage 상한 (전 method 동일): 5 m 0.49–0.60 (20 m 간격 기하), riverside_01_03만
0.013 (반대차선) — R@1@5m 단독 해석 금지, 20 m 병기 필수.

추가 (08-08): RING# pd-재학습 ckpt의 cross-session도 완료 — macro
0.176/0.418/0.540 (5m/20m/1%@20m)로 최하위. 본문 Table 2는 교수님 구성대로
RING# 제외, 부록 per-pair 표에 편입.

## 결론 / 다음

1. **주 지표 R@1@5m: NSD/NSD-H가 BEVPlace++와 동률 1위** (0.383 vs 0.384;
   NCLT는 NSD-H 0.439 단독 1위). 정밀(5 m) 지표에서 최강 베이스라인과 대등.
2. **R@1@20m: BEVPlace++ 1위 (0.798), NSD-H 0.703** — 단 BEVPlace++는
   32.8 KB/frame vs NSD-H 4 KB (8배). 저장 비용 축(효율성 표)이 차별점.
   저장 비교군(SC++ 4.9 KB) 대비로는 0.605 vs 0.703으로 NSD-H 우위.
3. **fusion+alignment의 cross-session 확장 효과 입증**: key-only 0.533 →
   fused 0.703 (@20m macro), NCLT는 0.359→0.764 (2배+). 기존 open item 해소.
4. pdnn(NCLT 미학습)도 0.688로 pd 0.703에 근접 — fusion이 key-only에서 봤던
   pd/pdnn 격차(0.36/0.22)를 대부분 흡수.
5. 다음: RING# pd-val 선택 → 42-seq test → strict rev3; 효율성 N=1000 벤치;
   본문 Table 2 통합 (R@1@5m·R@1@20m·R@1%@20m 3열 + coverage 캡션 공시).
