# Place-disjoint 동결 test 1회 실행 (intra-session 42시퀀스 + cross-session 5프로토콜)

- 날짜: 2026-08-04
- 관련 브랜치: feat/pbev-stored-spectrum

## 왜 (진행 이유)

선택 스택 동결 후 유일한 test 접촉. 목표: ① 독립 테스트 세트에서의 성능 입증,
② 장소 일반화(미학습 사이트/도시/데이터셋) 정량화, ③ NCLT 미학습(pdnn)
ablation으로 센서 일반화 입증.

## 설정 (파라미터)

- 동결 스펙: pd/pdnn × 시드 1–3 체크포인트, ω/classifier `_pd2026`/`_pdnn2026`,
  tuple pd/pbev=(2,1,2,2)·pd/hspec12=(2,1,2,0.5)·pdnn 양쪽=(2,1,2,1) — 재선택
  없음 (`_pd_apply_frozen.py`, val Roundabout03에서 bit-단위 재현 검증 후 실행)
- Intra-session: 42 test 시퀀스 × 2구성 × 3시드 × 2 operating point = 504런,
  DTH 5 m / SKIP 30. 캐시는 `_pd_build_test_caches.py` (시퀀스당 프로세스)
- Cross-session (416D key cosine, coarse 단계만): MulRan 3사이트(DB=01+02→03),
  NCLT(02-04@20m→03-17@5m, RING 관행), Town(01+02→03). **주의: 캐시 pose가
  세션별 re-center라 MulRan/NCLT는 전역 GT로 pose 교체(_gp) 후 평가**
  (`_pd_fix_xsession_poses.py`; 교체 전 KAIST가 0.000이었던 원인)

## 결과

Intra-session (3시드 seq-macro R@1, `results/pd2026/test_summary_intra.json`):

| 그룹 | pd/pbev | pd/hspec12 | pdnn/pbev | pdnn/hspec12 |
|---|---|---|---|---|
| KITTI 00/05/08 | 0.915 | 0.975 | 0.917 | 0.976 |
| HeLiPR Town+B4 (미학습 사이트) | 0.820 | 0.834 | 0.825 | 0.831 |
| HeLiPR 대전 04–06 (겨울·야간) | 0.915 | 0.903 | 0.912 | 0.905 |
| MulRan 대전 (미학습 도시) | 0.852 | 0.861 | 0.854 | 0.860 |
| NCLT 페어 (미접촉 세션) | 0.975 | 0.970 | 0.961 | 0.960 |
| KITTI-360 (zero-shot) | 0.881 | 0.966 | 0.869 | 0.971 |
| MCD (zero-shot 새 도시+FOV) | 0.940 | 0.934 | 0.943 | 0.944 |
| **전체 매크로** | 0.900 | **0.920** | 0.897 | **0.921** |

Cross-session (key-only, 3시드, `results/pd2026/test_summary_xsession.json`):
KAIST 0.97/0.96(5m) · DCC 0.67(5m)/0.87(20m) · Riverside 0.35(5m, 매칭가능
551/5765 — 반대차선 재방문 특성) · NCLT 0.36(pd)/0.22(pdnn)@20m ·
Town 0.65(로컬 프레임 공유 가정 caveat).

## 결론 / 다음

1. **val→test 순위 역전**: val에서 pbev>hspec12였으나 test에서는 hspec12가
   전 구성 우세 (KITTI 9쿼리 val의 HDL-64E 대표성 부족). 독립 test의 필요성을
   본문에서 그대로 논증 가능.
2. **pdnn≈pd + NCLT zero-shot 0.96**: HDL-32E를 전혀 학습하지 않고도 0.96 —
   캘리브레이티드 풀링의 센서 일반화 최강 근거. 단 cross-session 하드 케이스
   (13개월 간격)에서는 pd 0.356 vs pdnn 0.220으로 격차 존재.
3. Cross-session은 coarse(키 단독) 진단치 — RING 계열과 정면 비교하려면
   spectrum fusion+alignment의 cross-session 확장이 다음 단계.
4. Zero-shot: KITTI-360 0.97(hspec12), MCD 0.93–0.94 — "미학습 도시" 주장 성립.
   가장 어려운 그룹은 미학습 사이트 Town 0.82–0.83.
