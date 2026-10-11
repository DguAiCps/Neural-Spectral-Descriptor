# 동일 조건 효율성 벤치마크 (N=1000, batch 1, disk I/O 제외, CUDA sync)

- 날짜: 2026-08-08
- 관련 브랜치: feat/pbev-stored-spectrum

## 왜 (진행 이유)

기존 레이턴시 벤치 3종은 방법별 샘플링이 달라(RING 60/30 vs NSD 150/all)
동일 표 비교가 부적절 (교수님 지적). codec 성격이 핵심 기여이므로 동일
sequence·동일 DB 크기에서 일괄 재측정. storage 정의도 "global descriptor만"이
아닌 **검색에 실제 저장하는 전체 상태**로 통일.

## 설정 (파라미터)

- `scripts/_bench_matched_n1000.py` (신규; 기존 벤치 3종의 method별 측정
  코드를 import/이식, provenance 주석). kitti/00, DB = keyframe 0–999,
  query = 1000–1199 (DB만 검색), batch 1, 전체 포인트클라우드 RAM 프리로드
  (disk I/O 제외), warmup 10, GPU 구간 CUDA sync, mean/p50/p95.
- Storage = 실측 (각 method가 검색을 위해 보유하는 배열의 실제 바이트 합 / N):
  SC++ ring key+SC matrix; RING/RING++ 상관용 스펙트럼(f16); RING# global(f32)
  + rotation용 spec(f16) + translation용 trans feature(f16) — rerank가 실제
  인덱싱하는 상태만; NSD 800 float, NSD-H 1,024 float.
- NSD graph/edge 단계는 batch 계산을 keyframe당 상각 (기존 벤치와 동일 규약,
  breakdown JSON 기록). RING# 가중치는 timing 무관이라 paper-era 기본값.

## 결과 (`results/matched_bench_n1000.json`)

| Method | descriptor dim | storage/frame | build ms | query ms (p95) |
|---|---|---|---|---|
| SC++ | 20 (+20×60 SC) | 4,880 B | 3.96 | 33.33 (35.4) |
| BEVPlace++ | 8,192 | 32,768 B | 19.38 | 18.95 (19.0) |
| SeqOT | 256 | 1,024 B | 10.70 | 0.57 (0.6) |
| RING | (spec 120×120) | 28,800 B | 8.13 | 2.62 (2.7) |
| RING++ | (spec 6ch) | 172,800 B | 2,389 | 30.12 (30.3) |
| RING# | 160 | 103,040 B | 27.66 | 24.85 (27.5) |
| **NSD** | 416 | **3,200 B** | 13.66 | **0.98** (1.3) |
| **NSD-H** | 416 | **4,096 B** | 20.71 | 1.94 (2.2) |

## 결론 / 다음

1. **NSD 계열이 storage-정확도 파레토 우위**: cross-session 주 지표 동률
   1위인 BEVPlace++ 대비 storage 1/8–1/10, query 10–19배 빠름. RING++는
   173 KB/frame + build 2.4 s/frame (CPU LPD), RING#은 전체 상태 회계 시
   103 KB/frame — "160D 서술자"라는 명목 치수와 실제 발자국의 괴리가 드러남.
2. SeqOT(1 KB)만 storage에서 NSD보다 작으나 정확도가 크게 낮음 (strict 0.588,
   xs 5m 0.329).
3. 기존 이종-sampling 벤치 JSON들은 부록으로 강등 + sampling 차이 명시 예정.
4. 다음: Main Table 3 통합 (descriptor dim | storage | build | query 4열).
