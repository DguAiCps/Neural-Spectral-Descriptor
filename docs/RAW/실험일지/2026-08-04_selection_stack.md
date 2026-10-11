# Place-disjoint 선택 스택 재유도 (ω → edge classifier → B3 rerank → fusion tuple)

- 날짜: 2026-08-04
- 관련 브랜치: feat/pbev-stored-spectrum

## 왜 (진행 이유)

독립 테스트 주장이 성립하려면 다운스트림 선택 요소 전부(키 재계량 ω, edge
classifier, fusion tuple, 체크포인트 선택)를 **새 val에서만** 재유도하고 test
접촉 전에 동결해야 함. paper-main 스택을 pd 프로토콜용으로 적응.

## 설정 (파라미터)

- ω: `scripts/_pd_fit_key_remetrize.py` — train 시퀀스 캐시만 사용 (pd 31,388
  재방문쌍 / pdnn 25,354쌍), closed-form (DTH 5 m, SKIP 30)
- Classifier: `scripts/_pd_build_edge_classifier.py` — train 전용, holdout도
  train 내부 분할(kitti/07, nclt/2013-02-23, helipr/Bridge01, mulran/Sejong02).
  **시퀀스당 프로세스 격리 필수** (`--only`): 단일 프로세스 실행 시 메모리
  단편화 누적으로 ~28 GB cgroup OOM (exit 137, 1회 발생 후 수정)
- Rerank/fusion: `scripts/_pd_b3_rerank.py`(시드별), `_pd_branch_fusion.py`,
  tuple 선택 `_pd_tuple_select.py` (val 4시퀀스 전용)
- 체인: `scripts/_pd_run_stack.sh`

## 결과 (val = 선택 세트 수치, test 아님)

- ω: 동결 paper-main ω와 cos 0.999 (인코더의 안정 속성; max|diff| 0.61)
- Classifier holdout AUC: pd 0.837 / pdnn 0.920
- 선택된 tuple과 val 3시드 R̄q:
  | 구성/브랜치 | tuple | R̄q | R_min |
  |---|---|---|---|
  | pd/pbev (NSD) | (2,1,2,2) | 0.951 ± 0.0004 | 0.939 |
  | pd/hspec12 (NSD-H) | (2,1,2,0.5) | 0.927 | 0.878 |
  | pdnn/pbev | (2,1,2,1) | 0.945 | 0.936 |
  | pdnn/hspec12 | (2,1,2,1) | 0.910 | 0.861 |
- Provenance: `results/pd2026/tuple_selection.json` (selection_pool에 val 4개만
  기록됨을 확인)

## 결론 / 다음

동결 스펙 확정. 주의 신호 2건: KITTI 09는 루프 쿼리 9개뿐(val 기여 미미),
새 val에서 pbev > hspec12 (paper-main R̄s 순위와 반대 — test에서 재확인).
다음: 동결 test 1회 실행 (캐시 40건 빌드 → apply 252런 → cross-session 5개
프로토콜).
