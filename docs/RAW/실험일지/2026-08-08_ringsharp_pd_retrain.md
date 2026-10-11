# RING#-L pd2026 place-disjoint 재학습 + clean 42/37-seq 재평가

- 날짜: 2026-08-07 ~ 08-08
- 관련 브랜치: feat/pbev-stored-spectrum

## 왜 (진행 이유)

기존 RING# 수치는 paper-era 24-seq 학습 가중치의 zero-shot이라 pd test 10개
시퀀스가 train_overlap으로 매크로에서 제외됐고(27-seq만 clean), 다른 방법은
37-seq라 동일 매크로 비교가 불성립. 교수님 지시: pd 재학습으로 전 로스터
clean 편입 (가장 깨끗한 해결).

## 설정 (파라미터)

- 게이트: 재학습 전 현행 가중치로 kitti/05 재평가 → 기존 수치 **델타 0.0** 재현.
- Prepare: `scripts/_pdb_ringsharp_prepare.py` (신규; `_ringsharp_prepare.py`
  불변) — pd 16 train 시퀀스, **134,830 tuples**, 스테이징
  `_external/ringsharp_data_pd2026/` 분리. NCLT 5세션 단일 그룹, 나머지
  per-seq 그룹, KITTI z-up 켤레변환 등 기존 스크립트 규약 그대로.
  `nsc_dataset.py`의 HeLiPR 구경로 하드코딩(`seq/seq`)을 평탄화 대응
  flat-first/nested-fallback으로 수정 (미수정 시 학습 즉사).
- 학습: 공식 trainer, 기존 레시피 계승 (5 epoch, batch 16, NSC_MICRO_BATCH=2,
  lr 1e-4), **단일 seed**, per-epoch ckpt. 총 ~11 h (8,427 iter/epoch,
  1.05 s/it). Loss 81.3→69.8→61.3→58.8→56.0→55.8.
- **Checkpoint 선택 (사전 고정 규칙)**: "epoch 1–5 중 pd-validation
  sequence-macro 5 m R@1 최고, 동률 시 이른 epoch" (val = kitti/09,
  nclt/2012-01-08, helipr/Roundabout03, mulran/Sejong03; test 접촉 전 규칙 동결).
- 결과: e1 0.4638 > e2 0.4503 > e4 0.4400 > e5 0.4326 > e3 0.4261 →
  **epoch 1 선택** (이후 단조 열화 — 축소 스플릿 과적합), sha256
  `3f3481ed…49d1da75` 동결 (`ringsharp_pd_ckpt_selection.json`).
  참고: kitti/09는 전 epoch 0.000 (9쿼리, RING#가 전패) — 순위에 영향 없음.
- Test: 선택 ckpt로만 42-seq 전체 (`_pdb_ringsharp_pd_eval.py` wrapper;
  `_pdb_ringsharp_eval.py`는 byte-identical 유지), ~4.5 h.

## 결과

Strict-v2 **revision 3** (`summary_strict_v2_revision3.json`; 프로토콜 불변,
RING# 열만 교체; scpp/ring 재집계 검증으로 그룹핑 revision 2와 일치 확인):

| method | strict_macro(37) |
|---|---|
| NSD-H | **0.9146** |
| NSD | 0.8860 |
| BEVPlace++ | 0.8763 |
| RING | 0.8673 |
| RING++ | 0.8248 |
| SC++ | 0.7988 |
| SeqOT | 0.5881 |
| **RING# (pd clean)** | **0.5464** (구 zero-shot·27-seq 0.5901) |

그룹별 (clean vs 구): helipr_daejeon 0.669↑(0.638) / town 0.470↓(0.559) /
kitti360 0.486↓(0.593) / kitti 0.607↓(0.633) / mcd 0.338↓(0.385) /
mulran 0.709↓(0.733).

## 결론 / 다음

1. RING#은 clean 조건에서도 최하위권 유지 — 표 순위 결론 불변, 이제
   train_overlap 각주 없이 **전 방법 동일 37-seq 매크로**로 직접 비교 가능.
2. 선택 규칙 사전 고정 + sha 동결로 test-based selection 의혹 차단
   (교수님 요구사항 충족).
3. 다음: (옵션) cross-session RING# 행 추가, 본문 Table 1의 RING# 열 교체.
