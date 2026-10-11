# 현재 실험 결과 종합 보고서

- 기준일: 2026-08-08
- 범위: 실험 설계, 결과, 해석 및 주장 가능 범위
- 제외 범위: 논문 원고의 문장·구성 검토, online/causal 평가
- 핵심 평가 대상: NSD, NSD-H

## 1. 한눈에 보는 결론

현재 실험에서 가장 강하게 뒷받침되는 결론은 다음과 같다.

> **NSD 계열은 작은 검색 상태로 높은 장소 인식 성능을 제공한다.**
> NSD-H는 엄격한 place-disjoint offline 평가에서 가장 높은 성능을 기록했고,
> cross-session 5 m 평가에서는 대형 learned baseline인 BEVPlace++와 0.11%p
> 이내의 성능을 보였다. 20 m cross-session 성능은 BEVPlace++보다 낮지만,
> 전체 검색 상태가 8배 작고 query가 약 9.8배 빠르다.

세 실험의 역할은 서로 다르다.

1. **Strict-v2 place-disjoint 평가**는 NSD-H의 일반화 성능이 가장 높다는 것을
   보여주는 주 성능 실험이다.
2. **Cross-session 평가**는 서로 다른 주행 세션 사이에서도 compact
   representation이 경쟁력 있게 작동한다는 것을 보여준다.
3. **동일 조건 효율성 평가**는 정확도뿐 아니라 실제 저장공간과 검색시간에서도
   NSD 계열이 유리하다는 것을 보여준다.

따라서 현재 결과는 “모든 조건에서 최고 정확도”보다는 다음 주장에 가장 잘
맞는다.

> **Place-disjoint 환경에서는 선두 성능을, cross-session 환경에서는 훨씬 작은
> 전체 검색 상태로 비교 가능한 성능을 제공하는 compact offline retrieval
> system.**

---

## 2. 실험 구성 요약

### 2.1 Strict-v2 place-disjoint offline retrieval

- 평가 대상: 37개 test sequence
- 집계 단위: 6개 평가 그룹
- 지표: sequence-macro Recall@1 at 5 m
- 최종 요약: six-group-balanced macro
- 시간 제외: query와 candidate 사이 최소 30 frame
- 검색 방식: 과거·미래 keyframe을 모두 허용하는 offline symmetric retrieval
- NSD/NSD-H: 3개 seed 평균
- RING#-L: place-disjoint split에서 clean retrain한 단일 seed

초기 동결 roster는 42개 sequence였으나 GPS 기반 장소 중복 감사 후 5개
sequence가 strict 결과에서 제외되었다.

- KITTI 00: train KITTI 07과 경로 중복
- KITTI-360 0004: validation KITTI 09와 동일 경로
- HeLiPR Bridge04: training bridge와 동일 장소
- NCLT 2012-02-04, 2012-03-17: training NCLT와 동일 캠퍼스

즉, 제외된 것은 **5개 sequence이며 제외 사유는 4종**이다. Strict-v2는 최초
42개 roster에서 이들을 제외한 37개 place-disjoint subset이다.

### 2.2 RING-style cross-session retrieval

- 총 7개 map-query pair
- NCLT: 2012-02-04 → 2012-03-17
- MulRan: DCC/KAIST/Riverside의 01 → 02, 01 → 03
- map spacing: 20 m
- query spacing: 5 m
- query roster: 20 m 안에 map pose가 있는 query를 한 번만 선정
- 모든 거리 threshold에서 동일한 query 분모 사용
- candidate는 DB에서만 검색
- 집계: NCLT, DCC, KAIST, Riverside의 site-balanced macro
- 주요 지표: R@1@5 m, R@1@20 m, R@1%@20 m

이 평가는 공식 RING roster를 그대로 재현한 것이 아니라, 공통 keyframe
roster에서 모든 방법을 다시 평가한 RING-style 비교다.

### 2.3 동일 조건 효율성 평가

- sequence: KITTI 00
- DB: keyframe 0–999, 총 1,000개
- query: keyframe 1000–1199, 총 200개
- batch size: 1
- point cloud는 RAM에 미리 적재하여 disk I/O 제외
- GPU 구간은 CUDA synchronize 적용
- warm-up: 10회
- storage: retrieval과 reranking에 실제 필요한 전체 배열의 byte 합
- 측정값: build mean, query mean/p50/p95

이 평가는 단순 global descriptor dimension이 아니라, 실제 검색을 수행하기 위해
저장해야 하는 전체 상태를 비교한다.

---

## 3. Strict-v2 결과와 평가

### 3.1 최종 결과

| Method | KITTI 05/08 | Town 01–03 | HeLiPR Daejeon | MulRan Daejeon | KITTI-360 | MCD | 6-group macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| RING | 0.897 | **0.913** | 0.869 | 0.823 | 0.892 | 0.809 | 0.867 |
| RING++ | 0.893 | 0.769 | 0.835 | 0.821 | 0.831 | 0.799 | 0.825 |
| RING#-L | 0.607 | 0.470 | 0.669 | 0.708 | 0.486 | 0.338 | 0.546 |
| SC++ | 0.892 | 0.620 | 0.732 | **0.863** | 0.867 | 0.819 | 0.799 |
| SeqOT | 0.594 | 0.495 | 0.625 | 0.797 | 0.550 | 0.468 | 0.588 |
| BEVPlace++ | 0.943 | 0.818 | 0.847 | 0.835 | 0.953 | 0.861 | 0.876 |
| **NSD** | 0.886 | 0.828 | **0.915** | 0.852 | 0.895 | **0.940** | **0.886 ± 0.002** |
| **NSD-H** | **0.968** | 0.848 | 0.903 | 0.861 | **0.974** | 0.934 | **0.915 ± 0.0005** |

### 3.2 핵심 해석

NSD-H가 전체 macro 0.9146으로 1위다.

- BEVPlace++ 대비: **+3.83%p**
- RING 대비: **+4.73%p**
- NSD 대비: **+2.86%p**

NSD-H의 강점은 한 그룹의 큰 성능 향상만으로 만들어진 결과가 아니다.

- KITTI와 KITTI-360에서 1위
- 나머지 네 그룹에서 모두 2위
- 가장 낮은 그룹도 Town의 0.848

즉, NSD-H는 특정 환경 특화형이라기보다 그룹 전반에서 성능 하한이 높은
운영점이다.

NSD는 전체 0.8860으로 2위이며 BEVPlace++보다 0.97%p 높다. 특히 HeLiPR
Daejeon과 MCD에서 1위다. 다만 0.97%p 차이는 NSD의 seed 표준편차만 보고
통계적 우위라고 해석할 수는 없다. 현재 결과가 직접 보여주는 것은 순위와
관측된 평균 차이다.

### 3.3 NSD와 NSD-H의 역할 차이

- **NSD-H**: KITTI 및 KITTI-360처럼 height-polar 정보가 효과적인 환경에서
  강하고, 전체 macro 및 worst-group 관점에서 우수하다.
- **NSD**: HeLiPR Daejeon과 MCD에서 NSD-H보다 높아, BEV spectrum이 더 잘
  맞는 환경이 존재한다.

따라서 NSD-H가 모든 환경에서 NSD를 대체하는 관계는 아니다. 두 방법은 같은
invariant key를 사용하면서 auxiliary projection의 저장량과 환경 적합성이 다른
두 개의 고정 operating point로 보는 것이 정확하다.

### 3.4 RING# clean retrain의 의미

RING#-L은 place-disjoint training split으로 재학습한 후 validation macro가 가장
높았던 epoch 1을 사전 고정 규칙으로 선택했다.

- epoch 1 validation macro: 0.4638
- epoch 2: 0.4503
- epoch 3: 0.4261
- epoch 4: 0.4400
- epoch 5: 0.4326
- 최종 strict macro: 0.5464

이 실험의 가장 큰 성과는 RING#의 성능 자체가 아니라 비교 조건의 정리다. 기존
RING#은 학습 중복 때문에 27개 clean sequence만 집계했지만, 현재는 다른 방법과
동일한 37개 sequence에서 비교할 수 있다.

단, epoch 1 이후 validation 성능이 지속적으로 악화됐고 단일 seed 결과이므로,
이 수치를 RING# 방법론 전체의 일반적 성능으로 확대하면 안 된다. 동일 split과
공식 trainer를 이용한 현재 구현의 clean-retrain 결과로 한정해야 한다.

### 3.5 Strict-v2 결과의 신뢰도

강점:

- train/validation/test 역할이 분리됨
- fusion tuple, 재계량, edge classifier, checkpoint가 test 전에 동결됨
- NSD 계열은 3개 seed 평가
- 모든 baseline이 동일한 37-sequence roster와 공통 metric으로 평가됨
- post-hoc exclusion 이후 성능은 오히려 감소했고 순위는 유지됨

주의점:

- 42-sequence 결과는 7-group macro, strict 결과는 6-group macro이므로 두 값의
  차이는 순수한 sequence 제거 효과와 그룹 재가중 효과가 섞여 있다.
- GPS 감사의 place-overlap 판정 기준은 거리와 overlap 비율을 포함한 명시적인
  규칙으로 고정해 두는 것이 좋다.
- BEVPlace++는 official KITTI pretrained checkpoint에서 시작했으며, NSD 계열은
  external pretraining을 사용하지 않았다.
- baseline은 대부분 단일 학습 또는 단일 실행 결과이므로, NSD의 seed 표준편차와
  baseline 간 차이를 곧바로 통계적 유의성으로 해석할 수 없다.

### 3.6 Strict-v2에 대한 최종 판정

현재 실험 중 가장 강하고 직접적인 결과다. 다음 주장이 충분히 성립한다.

> NSD-H는 37-sequence place-disjoint offline retrieval에서 가장 높은
> six-group-balanced Recall@1을 기록하며, 모든 평가 그룹에서 1위 또는 2위를
> 유지한다.

---

## 4. Cross-session 결과와 평가

### 4.1 최종 결과

| Method | R@1@5 m | R@1@20 m | R@1%@20 m |
|---|---:|---:|---:|
| SC++ | 0.356 | 0.605 | 0.678 |
| **BEVPlace++** | **0.384** | **0.798** | **0.858** |
| SeqOT | 0.329 | 0.734 | 0.794 |
| RING | 0.354 | 0.755 | 0.816 |
| RING++ | 0.342 | 0.727 | 0.779 |
| RING#-L | 0.176 | 0.404 | 0.514 |
| **NSD** | **0.383 ± 0.001** | 0.695 ± 0.002 | 0.755 ± 0.002 |
| **NSD-H** | **0.383 ± 0.0004** | 0.703 ± 0.001 | 0.760 ± 0.002 |
| NSD-H, no NCLT training | 0.380 ± 0.001 | 0.688 ± 0.002 | 0.749 ± 0.002 |
| NSD key-only | 0.265 ± 0.002 | 0.533 ± 0.005 | 0.657 ± 0.001 |

### 4.2 5 m 결과: 수치적으로 비교 가능한 수준

5 m macro에서:

- BEVPlace++: 0.38374
- NSD-H: 0.38267
- NSD: 0.38251

NSD-H와 BEVPlace++의 차이는 **0.11%p**, NSD와의 차이는 **0.12%p**다. 실용적인
크기로 보면 매우 작은 차이이며, “5 m 지표에서 comparable performance”라는
표현이 적절하다.

다만 다음 표현은 구분해야 한다.

- 가능: “within 0.11 percentage points”, “numerically comparable”
- 현재 증거만으로 불가: “statistically equivalent”, “statistically tied”

통계적 동등성을 주장하려면 동일 query에 대한 paired success 기록과 equivalence
margin을 사전에 정한 paired bootstrap 또는 동등성 검정이 필요하다.

### 4.3 5 m coverage 상한

20 m 간격의 map에서 5 m 정답을 요구하므로, descriptor가 완벽해도 일부 query는
5 m 안에 map pose 자체가 없다.

| Pair | Coverage@5 m |
|---|---:|
| NCLT | 0.487 |
| DCC 01→02 | 0.536 |
| DCC 01→03 | 0.492 |
| KAIST 01→02 | 0.561 |
| KAIST 01→03 | 0.603 |
| Riverside 01→02 | 0.568 |
| Riverside 01→03 | 0.013 |

따라서 5 m 결과는 정밀 위치 일치 능력을 보여주지만, 전체 place retrieval 능력을
단독으로 나타내지는 않는다. 특히 Riverside 01→03은 반대 차선 재방문 때문에
5 m 결과가 사실상 구조적으로 제한된다. Cross-session 성능은 반드시 20 m 결과와
함께 해석해야 한다.

### 4.4 20 m 결과: 정확도와 저장량의 trade-off

20 m에서는 BEVPlace++가 뚜렷하게 앞선다.

- BEVPlace++: 0.7980
- RING: 0.7553
- SeqOT: 0.7337
- RING++: 0.7269
- NSD-H: 0.7032
- NSD: 0.6946

NSD-H는 BEVPlace++보다 **9.47%p**, RING보다 **5.20%p** 낮다. 따라서
cross-session 전체에서 최고 정확도 또는 SOTA라고 평가할 수 없다.

대신 NSD-H는 4 KB/frame, BEVPlace++는 32 KB/frame을 사용한다. 즉 20 m에서
9.47%p의 성능을 양보하는 대신 저장 상태를 8배 줄이는 운영점이다. 이것이 현재
cross-session 결과의 정확한 의미다.

### 4.5 Retained spectrum의 효과

NSD key-only와 full NSD-H의 차이는 크다.

- 전체 macro @20 m: 0.533 → 0.703, **+17.0%p**
- NCLT @20 m: 0.359 → 0.764, **+40.5%p**

이는 416D retrieval key만 사용할 때 놓치는 cross-session match를 보조 magnitude
index와 cyclic alignment가 상당 부분 복구한다는 것을 보여준다.

다만 현재 ablation은 보조 index와 alignment를 함께 추가한 결과다. 따라서
“retained complex spectrum 전체 경로의 기여”는 주장할 수 있지만, 별도 분해
실험 없이 “alignment 단독으로 +17%p”라고 말해서는 안 된다.

### 4.6 NCLT 미학습 결과

NCLT를 training에서 완전히 제외한 NSD-H도 다음 성능을 유지한다.

- R@1@5 m: 0.3805
- R@1@20 m: 0.6882
- R@1%@20 m: 0.7486

기본 NSD-H와 비교하면 20 m macro 차이는 1.51%p다. 학습 중 NCLT를 본 효과는
존재하지만, 전체 cross-session 성능 대부분은 NCLT 학습 없이도 유지된다. 이는
특정 캠퍼스나 한 sensor family에만 맞춘 결과가 아니라는 보조 증거다.

### 4.7 Cross-session 결과의 신뢰도

강점:

- 모든 방법이 동일 map/query roster 사용
- threshold별 query 재필터링 없이 동일 분모 유지
- DB-only candidate 검색을 명시적으로 보장
- 모든 방법에 동일한 global pose source 사용
- method마다 roster hash가 같은지 자동 검증
- NSD 계열은 3개 seed 결과 제공
- 5/10/15/20 m 및 per-pair 결과가 모두 보존됨

주의점:

- 총 7개 pair, 4개 site이므로 보편적인 cross-session 일반화 전체를 대표하지는
  않는다.
- NCLT site는 한 pair, 나머지 site는 두 pair이지만 site-balanced macro에서 동일한
  site 가중치를 받는다.
- 5 m 성능 차이에 대한 paired 통계 검정은 아직 없다.
- RING# cross-session canonical 결과는 최종 JSON 기준
  **0.176/0.404/0.514**다. 일부 실험일지의 0.176/0.418/0.540은 이전 집계값으로
  최종 수치와 일치하지 않는다.

### 4.8 Cross-session에 대한 최종 판정

다음 해석이 타당하다.

> NSD와 NSD-H는 5 m 정밀 cross-session 지표에서 BEVPlace++와 0.12%p 이내의
> 수치적 성능을 보인다. 20 m place-level 정확도는 더 낮지만, 훨씬 작은 전체
> retrieval state에서 경쟁력 있는 성능을 유지한다.

---

## 5. 동일 조건 효율성 결과와 평가

### 5.1 최종 결과

| Method | 실제 저장 상태/frame | Build mean | Query mean | Query p95 |
|---|---:|---:|---:|---:|
| SC++ | 4,880 B | 3.96 ms | 33.33 ms | 35.44 ms |
| BEVPlace++ | 32,768 B | 19.38 ms | 18.95 ms | 19.01 ms |
| SeqOT | 1,024 B | 10.70 ms | 0.57 ms | 0.60 ms |
| RING | 28,800 B | 8.13 ms | 2.62 ms | 2.68 ms |
| RING++ | 172,800 B | 2,389.35 ms | 30.13 ms | 30.31 ms |
| RING#-L | 103,040 B | 27.66 ms | 24.85 ms | 27.51 ms |
| **NSD** | **3,200 B** | **13.66 ms** | **0.98 ms** | **1.25 ms** |
| **NSD-H** | **4,096 B** | **20.72 ms** | **1.94 ms** | **2.23 ms** |

### 5.2 저장 효율

BEVPlace++ 대비:

- NSD: 32,768 / 3,200 = **10.24배 작음**
- NSD-H: 32,768 / 4,096 = **8배 작음**

중요한 점은 NSD의 저장량을 416D retrieval key만으로 계산하지 않았다는 것이다.

- NSD: 416D key + cylindrical spectrum + BEV spectrum = 800 floats, 3,200 B
- NSD-H: 416D key + cylindrical spectrum + height spectrum = 1,024 floats,
  4,096 B

따라서 정확한 표현은 “416D descriptor만 저장”이 아니라 **전체 retrieval state가
800/1,024 floats**라는 것이다.

RING#도 같은 원칙을 적용하면 nominal 160D global key 외에 rotation spectrum과
translation feature가 필요하여 총 103,040 B/frame이 된다. 이 결과는 nominal
descriptor dimension보다 실제 배포 상태를 비교해야 하는 이유를 보여준다.

### 5.3 검색시간

BEVPlace++ 대비 query mean:

- NSD: 18.949 / 0.983 = **약 19.3배 빠름**
- NSD-H: 18.949 / 1.941 = **약 9.8배 빠름**

NSD-H의 build time은 20.72 ms로 BEVPlace++의 19.38 ms보다 약 6.9% 느리다.
따라서 NSD-H가 build, storage, query 모든 항목에서 BEVPlace++보다 우월한 것은
아니다. 강점은 storage와 query latency이고, build는 비슷하거나 소폭 느리다.

SeqOT는 1 KB/frame과 0.57 ms query로 가장 작고 빠르다. 하지만 strict macro가
0.588, cross-session 5 m가 0.329로 정확도가 크게 낮다. 그러므로 NSD가 모든
방법보다 가장 작다고 주장할 수는 없지만, 높은 정확도를 유지하는 방법 가운데
강한 storage–accuracy 절충을 제공한다.

### 5.4 Pareto 관점

Strict-v2 기준:

- SeqOT: 1 KB, 0.588
- NSD: 3.2 KB, 0.886
- NSD-H: 4 KB, 0.915
- BEVPlace++: 32 KB, 0.876

BEVPlace++는 NSD-H보다 저장량이 크면서 strict 정확도도 낮다. NSD와 NSD-H는
서로 다른 storage–accuracy 운영점을 제공하고, SeqOT는 더 작지만 정확도가 크게
낮다. 따라서 두 NSD 운영점이 storage–accuracy Pareto front에 있다는 해석은
타당하다.

Cross-session 20 m에서는 BEVPlace++가 정확도는 높지만 저장량이 8–10배 크다.
이 경우 NSD 계열은 정확도 우위가 아니라 compact trade-off 운영점이다.

### 5.5 효율성 실험의 적용 범위

현재 결과가 직접 보장하는 범위는 다음과 같다.

- KITTI 00
- RTX 5080
- DB 크기 1,000
- query 200개
- batch 1
- disk I/O 제외
- completed-session graph의 batch 비용을 keyframe당 상각한 offline 설정

따라서 결과를 임의의 DB 크기, 다른 GPU, online per-scan latency 또는 end-to-end
센서 입력 지연으로 일반화하면 안 된다. 현재 실험은 동일한 N=1000 조건에서의
matched implementation benchmark로 해석해야 한다.

### 5.6 효율성 실험에 대한 최종 판정

현재 연구의 compactness 주장을 가장 직접적으로 뒷받침하는 실험이다.

> NSD/NSD-H는 모든 retrieval 및 reranking 상태를 포함해 각각 3.2/4.0 KB만
> 저장하며, 높은 정확도를 유지하면서 BEVPlace++보다 query가 약 19.3/9.8배
> 빠르다.

---

## 6. 실험 전체에서 주장 가능한 것과 피해야 할 것

### 6.1 강하게 주장 가능한 내용

1. NSD-H는 Strict-v2 37-sequence place-disjoint offline 평가에서 가장 높은
   six-group-balanced Recall@1을 기록한다.
2. NSD-H는 모든 Strict-v2 그룹에서 1위 또는 2위로, 그룹 간 성능 하한이 높다.
3. NSD와 NSD-H의 전체 retrieval state는 각각 3.2 KB와 4.0 KB/frame이다.
4. NSD-H는 BEVPlace++보다 8배 작은 상태로 Strict-v2에서는 더 높은 정확도를
   기록한다.
5. Cross-session 5 m에서는 NSD 계열과 BEVPlace++의 관측 차이가 0.12%p
   이내다.
6. Cross-session 20 m에서는 NSD-H가 정확도를 일부 양보하지만, 8배 작은 저장
   상태와 약 9.8배 빠른 query를 제공한다.
7. 보조 spectrum retrieval과 alignment를 포함한 full pipeline은 key-only보다
   cross-session 20 m macro를 17.0%p 향상한다.
8. NCLT를 학습에서 제외해도 cross-session 성능 대부분이 유지된다.

### 6.2 제한해서 표현해야 하는 내용

- “descriptor dimension이 작다”보다는 “total deployed retrieval state가 작다”가
  정확하다.
- NSD는 416D key만 저장하는 것이 아니라 총 800 floats를 저장한다.
- NSD-H는 총 1,024 floats를 저장한다.
- Cross-session 5 m는 “statistical tie”가 아니라 “numerically comparable”이다.
- RING# 결과는 현재 split과 구현에서의 clean retrain 결과다.
- 효율성 수치는 N=1000, KITTI 00, RTX 5080 조건에 한정된다.

### 6.3 현재 결과로 주장하면 안 되는 내용

- 모든 cross-session 지표에서 최고 성능
- Cross-session SOTA
- 통계적으로 BEVPlace++와 동등함
- 416D만으로 최종 성능 달성
- 모든 baseline보다 저장량이 작음
- online 또는 real-time loop closure 성능
- 임의 DB 크기에서 10–19배 빠름
- RING# 방법론 자체가 본질적으로 낮은 성능임

---

## 7. 현재 실험의 종합 평가

| 실험 | 평가 | 핵심 이유 |
|---|---|---|
| Strict-v2 place-disjoint | **매우 강함** | 37 sequences, 6-group macro, NSD-H 1위, 전 그룹 top-2 |
| Cross-session | **강한 보완 실험** | 5 m 수치적 동급, 20 m 정확도 열세, compactness 차별화 |
| Matched efficiency | **매우 강함** | 동일 N·sequence·timing 규약, 전체 retrieval state 계상 |
| RING# clean retrain | **공정성 개선에 중요** | 37-sequence 동일 비교 가능; 성능 자체 해석은 단일 seed 한계 |
| NCLT-free ablation | **유용한 일반화 증거** | NCLT 미학습에도 full 성능과 작은 차이 |

전체적으로 현재 실험 패키지는 다음 세 요소를 모두 갖추었다.

- 정확도: place-disjoint에서 선두
- 일반화: cross-session 및 unseen-dataset 결과
- 효율성: 작고 빠른 실제 retrieval state

가장 설득력 있는 최종 메시지는 다음과 같다.

> **NSD는 nominal descriptor dimension만 작은 방법이 아니라, 보조 검색과
> alignment에 필요한 상태까지 모두 포함해도 3.2–4.0 KB/frame에 불과한 compact
> retrieval system이다. NSD-H는 이 크기로 place-disjoint 평가에서 선두 성능을
> 기록하고, cross-session 5 m에서는 32 KB/frame의 BEVPlace++와 비교 가능한
> 성능을 유지한다.**

---

## 8. 결과 출처

- Strict-v2 revision 3:
  `results/pd2026_baselines/summary_strict_v2_revision3.json`
- NSD/NSD-H strict seed 통계:
  `results/pd2026_baselines/strict_seed_stats.json`
- Cross-session 최종 집계:
  `results/pd2026_xsession/xs_summary.json`
- 동일 조건 효율성:
  `results/matched_bench_n1000.json`
- RING# place-disjoint 재학습 기록:
  `experiment_logs/2026-08-08_ringsharp_pd_retrain.md`
- Cross-session 실행 기록:
  `experiment_logs/2026-08-08_cross_session_benchmark.md`
- 효율성 실행 기록:
  `experiment_logs/2026-08-08_matched_efficiency_bench.md`
