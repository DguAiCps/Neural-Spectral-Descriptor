# Place-disjoint 3-way split 재학습 (pd / pdnn × 3시드)

- 날짜: 2026-08-03 ~ 08-04
- 관련 브랜치: feat/pbev-stored-spectrum

## 왜 (진행 이유)

AAAI-27 리뷰 리스크 대응 (reject 대비 보강): ① 독립 테스트 세트 부재(모든
하이퍼파라미터가 val=보고 세트에서 선택됨), ② 학습·평가에 같은 장소가 반복되어
generalization claim 범위 제한. 장소 단위로 train/val/test를 분리한 3-way
split로 재학습해 두 문제를 동시에 해소.

## 설정 (파라미터)

- Config: `configs/training_place_disjoint.yaml`(pd, NCLT 학습 포함) /
  `configs/training_place_disjoint_no_nclt.yaml`(pdnn, NCLT 완전 제외 ablation)
- Split: train = KITTI 01/02/06/07 · HeLiPR Bridge01–03, Roundabout01–02 ·
  MulRan Sejong01/02 (· NCLT 5세션); val = KITTI 09 · Roundabout03 · Sejong03
  (· NCLT 2012-01-08); test = 동결 42개 시퀀스 (KITTI 00/05/08, HeLiPR Town
  전체+대전 사이트, MulRan 대전 3사이트 전체, NCLT 02-04/03-17, KITTI-360,
  MCD)
- CLI: `--encoder-preset no_interdiff --use-gated-context
  --gate-initial-alpha 0.0625 --seed {1,2,3}`
- 캐시: `data/preprocessed_pd2026` 신규 빌드 (구 캐시는 raw 불일치 확인됨)
- 실행: docker `ring_baseline`, `scripts/_run_place_disjoint_retrain.sh`
  (순차 6런, 재개 가능)

## 결과

- 6/6 런 정상 수렴 (early stopping, 50–80 epoch). 로그: `logs/retrain_*.log`
- Best val coarse R@1 (rerank 전): pd_s1 0.596 / pd_s2 0.609 / pd_s3 0.599 /
  pdnn_s1 0.607 / pdnn_s2 0.610 / pdnn_s3 0.613
- 체크포인트: `checkpoints/{pd,pdnn}_s{1,2,3}/best_model.pth`
- 소요: 캐시 빌드 포함 총 ~7.5h (1런 1.3–2.2h)

## 결론 / 다음

시드·구성 간 일관된 수렴. pdnn이 소폭 높은 것은 val에서 가장 어려운 NCLT가
빠진 효과로 해석. 다음: 선택 스택(ω→classifier→tuple)을 새 val에서만 재유도
(→ 2026-08-04_selection_stack.md).
