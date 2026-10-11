# ImLPR pd 재학습 (clean, from-scratch) — 3프로토콜 재평가

- 날짜: 2026-08-21 ~ 08-26
- 관련 브랜치: feat/pbev-stored-spectrum
- 선행 일지: 2026-08-20_helios_imlpr_baselines.md

## 왜

Zero-shot ImLPR의 helipr_daejeon 오염(† 공시)을 해소한 clean 비교 확보
(사용자 지시). RING# 전례의 규율 재사용: 사전 고정 선택 규칙 + sha 동결.

## 설정

- **Clean 초기화**: 공식 config 기본(`load_weights=ImLPR_default.pth`)은
  오염 ckpt 이어학습 → **비활성화, DINOv2 사전학습 백본 + 새 어댑터/SALAD
  from-scratch** (그들의 논문 절차; "previous weights not found" 로그로 확인).
- 데이터: pd 16 train 시퀀스, 공식 3.0 m 거리 서브샘플(0.8 m 키프레임 →
  ~40k 프레임), 공식 피클 형식(스키마를 그들의 생성기 출력과 필드 단위
  대조 검증; positives 10 m/non-neg 30 m, RING#-prepare 그룹핑: NCLT 캠퍼스
  단일, 나머지 시퀀스별). RIV는 동결된 멀티센서 recipe 재사용.
- 레시피: 공식 그대로 (batch 2048, TruncatedSmoothAP, lr 5e-5, 100 epochs).
  16GB 메모리 노브만 조정·공시: batch_split 64→32(유효 배치·loss 불변),
  num_workers 0(shm 산수: 3.3GB×워커×prefetch>8GB), sensor_range=120(우리
  RIV range 스케일 정합).
- 시행착오 4건(전부 환경): shm 부족 / 공식 loss의 np.NINF(NumPy 2.0 제거
  별칭 → 런처 심으로 복원, upstream 무수정) / 16GB OOM / WSL2에서
  expandable_segments가 "device not ready" 유발(4크래시) → 기본 할당자.
  +공식 트레이너 자체 버그(epoch 5 eval 호출 signature 불일치) → 런처 스텁.
- 학습: ~3일(epoch당 ~45분, num_workers=0 비용), loss 1.14→0.21, 크래시 0.
- **선택(사전 고정)**: 저장 epoch {10..100} × pd-val 4시퀀스 macro R@1 →
  **e100 선택**(0.662→0.861 단조 상승; kitti/09 1.000), sha `f9637f81…` 동결
  (`imlpr_pd_ckpt_selection.json`).

## 결과 (3프로토콜, "ImLPR (pd retrain)" 행 — clean, 오염 마크 없음)

| 프로토콜 | zero-shot† | **pd retrain** | 판독 |
|---|---|---|---|
| Table 1 macro (rev5) | 0.677†(0.701) | **0.627** | 재학습이 낮음 — VLP-16 도메인은 개선(town 0.28→0.30, daejeon 0.56→0.64†→clean 0.64) but 대량 데이터 일반화 상실(mcd 0.87→0.59, kitti360 0.83→0.74) |
| Table 2 @5m/@20m | 0.337/0.671 | **0.343/0.730** | 재학습 우위 — NCLT 0.55→**0.82**(@20m 0.70→0.92): HDL-32E 도메인 해소 |
| Table 13 AP/R@100P | 0.560/0.328 | 0.513/0.293 | zero-shot 우위 유지 |

## 결론

1. **"재학습 = 항상 개선"이 아님을 정직하게 보여주는 사례**: 우리 규모
   (40k 프레임, 4센서)의 from-scratch는 저자들의 대규모 단일센서 학습이
   주는 일반화를 회복하지 못함. 학습 데이터에 있는 센서 도메인(VLP-16,
   HDL-32E)은 뚜렷이 개선.
2. 세 표 모두 두 행 병기(zero-shot† + clean retrain)로 공시 — 어느 행
   기준으로도 순위 결론 불변 (NSD-H 1위, ImLPR 중하위권).
3. Strict-v2 **revision5** + xs/PR 재집계 + 논문 3표 반영(각주 ∗ 규약),
   빌드 클린 29p.
